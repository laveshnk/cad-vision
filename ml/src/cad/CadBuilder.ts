/**
 * CadBuilder: builds and owns the solids in the playground.
 *
 * Two ways in, and they meet in the same scene:
 *
 *   Hands   pinch with both hands to size a shape at the origin, release the
 *           upper pinch and drag the lower one to set its height.
 *   Voice   the agent calls `addPrimitive` / `removeLast` / `setColor` /
 *           `clear` with already-validated numbers.
 *
 * The builder consumes only plain data — device-space deltas from the gesture
 * layer, plain numbers from the agent — so it depends on neither `src/vision`
 * nor `src/voice`. `src/main.ts` is the only place the two are joined.
 *
 * Hand-building flow:
 *   onExtrude(dual-hand)   both hands pinch: a preview appears at the world
 *                          origin, sized from the gap between the pinches.
 *                          Box -> square, cuboid -> rectangle from the
 *                          horizontal and vertical gaps, cylinder / sphere ->
 *                          diameter.
 *   onExtrude(single-hand) the upper pinch released: the base freezes and the
 *                          remaining hand's vertical travel drives the height
 *                          (or the radius, for a sphere).
 *   commit({ flat })       called when the last pinch releases. `flat` (both
 *                          released together, or the lower one first) commits
 *                          a thin plate instead.
 */

import * as THREE from 'three';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import type { SceneObjectSummary, ShapeId } from '../../shared/agentTools';
import type { CadScene } from './CadScene';

/** Same four primitives the voice agent knows about. */
export type CadTool = ShapeId;

/**
 * Extrusion delta; structurally compatible with the vision layer's extrude
 * event payload so the orchestrator can forward it directly.
 */
export interface CadExtrudeInput {
  mode: 'dual-hand' | 'single-hand';
  /**
   * Horizontal / vertical gap between the two pinch centers (units of video
   * width, dual-hand).
   */
  spanX?: number;
  spanY?: number;
  /** Cumulative vertical travel in device units (single-hand, +Y up). */
  cumulativeHeight?: number;
}

/** A shape the agent asked for, in world units. */
export interface CadPrimitiveSpec {
  shape: CadTool;
  width: number;
  depth: number;
  height: number;
  x: number;
  z: number;
  color?: string | null;
}

export interface CadBuilderOptions {
  /** Minimum footprint (world units) for a build to survive a pinch release. */
  minFootprint?: number;
  /** Minimum committed height / radius (world units). */
  minSize?: number;
  /** Height used when committing a build that was never extruded. */
  defaultHeight?: number;
  /** Height of a flat commit (two-hand build without a height step). */
  flatHeight?: number;
  /** World units of base size per unit (video width) of two-hand pinch gap. */
  baseSizeScale?: number;
  /** World units of extrusion per device-space unit of vertical drag. */
  extrudeScale?: number;
  /** Preview wireframe / fill color. */
  previewColor?: number;
  /** Default body color for hand-built solids. */
  bodyColor?: string;
  /** Committed mesh edge-line color. */
  edgeColor?: number;
}

/** base: two-hand base sizing. height: one pinch released, the other drives height. */
type BuildPhase = 'base' | 'height';

/** Wireframe preview handles (translucent fill + wireframe). */
interface Preview {
  group: THREE.Group;
  fill: THREE.Mesh;
  wire: THREE.Mesh;
}

interface Build {
  tool: CadTool;
  /** Ground-plane position the shape is centered on. */
  position: THREE.Vector3;
  width: number;
  depth: number;
  /** Base radius (cylinder / sphere). */
  radius: number;
  height: number;
  phase: BuildPhase;
  preview: Preview;
  /** Extrusion rebase snapshots for continuity across mode switches. */
  extrudeMode: 'dual-hand' | 'single-hand' | null;
  baseHeight: number;
  baseRadius: number;
}

/** A solid that is finished and in the scene. */
interface CommittedObject extends SceneObjectSummary {
  mesh: THREE.Mesh;
}

/** Shared unit geometries — previews only rescale, never rebuild. */
const UNIT_BOX = new THREE.BoxGeometry(1, 1, 1);
const UNIT_CYLINDER = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
const UNIT_SPHERE = new THREE.SphereGeometry(0.5, 32, 16);

/**
 * EdgesGeometry crease thresholds per tool: boxes show every edge,
 * cylinders only the top/bottom rims, spheres stay smooth.
 */
const EDGE_THRESHOLD: Record<CadTool, number> = { box: 1, cuboid: 1, cylinder: 15, sphere: 25 };

export class CadBuilder {
  private readonly scene: CadScene;
  private readonly options: Required<CadBuilderOptions>;
  /** Persistent hierarchy of committed meshes. */
  private readonly root: THREE.Group;
  private readonly committed: CommittedObject[] = [];

  private tool: CadTool = 'box';
  private build: Build | null = null;
  private nextId = 1;

  constructor(scene: CadScene, options: CadBuilderOptions = {}) {
    this.scene = scene;
    this.options = {
      minFootprint: options.minFootprint ?? 0.15,
      minSize: options.minSize ?? 0.05,
      defaultHeight: options.defaultHeight ?? 0.5,
      flatHeight: options.flatHeight ?? 0.05,
      baseSizeScale: options.baseSizeScale ?? 10,
      extrudeScale: options.extrudeScale ?? 4.5,
      previewColor: options.previewColor ?? 0x0284c7,
      bodyColor: options.bodyColor ?? '#3f3f46',
      edgeColor: options.edgeColor ?? 0xc9d2de,
    };
    this.root = new THREE.Group();
    this.root.name = 'cad-committed';
    scene.scene.add(this.root);
  }

  get activeTool(): CadTool {
    return this.tool;
  }

  get objectCount(): number {
    return this.committed.length;
  }

  get hasActiveBuild(): boolean {
    return this.build !== null;
  }

  /** Select the primitive the next build will use. */
  setTool(tool: CadTool): void {
    if (this.tool === tool) return;
    this.tool = tool;
    const build = this.build;
    if (build && build.phase === 'base') {
      // Swap the preview geometry while keeping the size drawn so far.
      this.disposePreview(build);
      build.tool = tool;
      build.preview = this.createPreview();
      this.scene.scene.add(build.preview.group);
      this.refreshPreview();
    }
  }

  /* ------------------------------------------------------------------ */
  /* Hand input                                                          */
  /* ------------------------------------------------------------------ */

  /**
   * Extrusion input. Dual-hand: size the base of the build centered on the
   * origin from the pinch gap. Single-hand: vertical drag (device units
   * scaled into world units) adds height; spheres grow their radius.
   */
  onExtrude(input: CadExtrudeInput): void {
    if (input.mode === 'dual-hand') {
      this.sizeBase(input.spanX, input.spanY);
      return;
    }
    const build = this.build;
    if (!build) return;
    if (build.extrudeMode !== 'single-hand') {
      // First single-hand frame: rebase so the height grows from where the
      // base phase left it.
      build.extrudeMode = 'single-hand';
      build.phase = 'height';
      build.baseHeight = build.height;
      build.baseRadius = build.radius;
    }
    const growth = (input.cumulativeHeight ?? 0) * this.options.extrudeScale;
    build.height = Math.max(this.options.minSize, build.baseHeight + growth);
    if (build.tool === 'sphere') {
      build.radius = Math.max(this.options.minSize, build.baseRadius + growth);
    }
    this.refreshPreview();
  }

  /**
   * Commit the active build: replace the wireframe preview with a solid matte
   * mesh (+ crisp edge outlines) and keep it in the scene. `flat` commits a
   * thin plate instead (spheres are unaffected).
   * @returns true if a mesh was committed.
   */
  commit({ flat = false }: { flat?: boolean } = {}): boolean {
    const build = this.build;
    if (!build) return false;
    const footprint = Math.max(build.width, build.depth, build.radius);
    if (footprint < this.options.minFootprint) {
      this.cancel();
      return false;
    }
    const height = flat
      ? this.options.flatHeight
      : build.phase === 'height'
        ? build.height
        : Math.max(build.height, this.options.defaultHeight);

    this.addObject({
      shape: build.tool,
      width: Math.max(build.width, this.options.minSize),
      depth: Math.max(build.depth, this.options.minSize),
      height,
      radius: Math.max(build.radius, this.options.minSize),
      x: build.position.x,
      z: build.position.z,
      color: this.options.bodyColor,
    });
    this.disposePreview(build);
    this.build = null;
    return true;
  }

  /** Discard the active preview without committing. */
  cancel(): void {
    if (!this.build) return;
    this.disposePreview(this.build);
    this.build = null;
  }

  /* ------------------------------------------------------------------ */
  /* Voice-agent input                                                   */
  /* ------------------------------------------------------------------ */

  /**
   * Build a solid outright, from numbers the agent already validated. Any
   * in-progress hand build is committed first so the two input paths cannot
   * fight over the same preview.
   */
  addPrimitive(spec: CadPrimitiveSpec): SceneObjectSummary {
    this.commit();
    const radius = Math.max(spec.width, this.options.minSize) / 2;
    return this.addObject({
      shape: spec.shape,
      width: Math.max(spec.width, this.options.minSize),
      depth: Math.max(spec.depth, this.options.minSize),
      height: Math.max(spec.height, this.options.minSize),
      radius,
      x: spec.x,
      z: spec.z,
      color: spec.color ?? this.options.bodyColor,
    });
  }

  /** Delete the most recent solid. Returns what was removed, if anything. */
  removeLast(): SceneObjectSummary | null {
    const object = this.committed.pop();
    if (!object) return null;
    this.root.remove(object.mesh);
    disposeMesh(object.mesh);
    return toSummary(object);
  }

  /** Recolor the newest solid, or every solid. Returns how many changed. */
  setColor(color: string, target: 'last' | 'all'): number {
    const objects =
      target === 'all' ? this.committed : this.committed.slice(-1);
    for (const object of objects) {
      object.color = color;
      (object.mesh.material as THREE.MeshStandardMaterial).color.set(color);
    }
    return objects.length;
  }

  /** Snapshot of everything in the scene, for the agent to reason over. */
  describe(): { objects: SceneObjectSummary[]; activeShape: CadTool } {
    return { objects: this.committed.map(toSummary), activeShape: this.tool };
  }

  /* ------------------------------------------------------------------ */
  /* Scene utilities                                                     */
  /* ------------------------------------------------------------------ */

  /** Remove every committed solid from the scene. */
  clear(): void {
    this.cancel();
    for (const object of this.committed) {
      this.root.remove(object.mesh);
      disposeMesh(object.mesh);
    }
    this.committed.length = 0;
  }

  /** Download everything as a binary `model.stl`, ready for a slicer. */
  exportStl(): boolean {
    if (this.committed.length === 0) return false;
    this.root.updateMatrixWorld(true);
    const exporter = new STLExporter();
    const view = exporter.parse(this.root, { binary: true }) as DataView;
    // Copy into a plain Uint8Array so the Blob part is a fresh ArrayBuffer.
    const bytes = new Uint8Array(view.byteLength);
    bytes.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
    const blob = new Blob([bytes], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'model.stl';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
    return true;
  }

  /** Detach from the scene and release all resources. */
  dispose(): void {
    this.clear();
    this.scene.scene.remove(this.root);
  }

  /* ------------------------------------------------------------------ */
  /* Internals                                                           */
  /* ------------------------------------------------------------------ */

  /** Build the mesh for a finished solid and record it. */
  private addObject(spec: {
    shape: CadTool;
    width: number;
    depth: number;
    height: number;
    radius: number;
    x: number;
    z: number;
    color: string;
  }): SceneObjectSummary {
    const geometry = this.buildGeometry(spec.shape, spec.width, spec.height, spec.depth, spec.radius);
    const mesh = new THREE.Mesh(geometry, this.bodyMaterial(spec.color));
    mesh.position.set(spec.x, spec.shape === 'sphere' ? spec.radius : spec.height / 2, spec.z);
    // Cast + receive soft shadows so solids read as physically grounded.
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.add(
      new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry, EDGE_THRESHOLD[spec.shape]),
        new THREE.LineBasicMaterial({
          color: this.options.edgeColor,
          transparent: true,
          opacity: 0.9,
        })
      )
    );
    this.root.add(mesh);

    const object: CommittedObject = {
      id: this.nextId++,
      shape: spec.shape,
      // Round shapes report their diameter as width, which is what the agent
      // and the spoken descriptions talk in.
      width: spec.shape === 'cylinder' || spec.shape === 'sphere' ? spec.radius * 2 : spec.width,
      depth: spec.shape === 'cylinder' || spec.shape === 'sphere' ? spec.radius * 2 : spec.depth,
      height: spec.shape === 'sphere' ? spec.radius * 2 : spec.height,
      x: spec.x,
      z: spec.z,
      color: spec.color,
      mesh,
    };
    this.committed.push(object);
    return toSummary(object);
  }

  /** Spawn a new build (preview + state) centered on `position`. */
  private startBuild(position: THREE.Vector3): Build {
    const preview = this.createPreview();
    const build: Build = {
      tool: this.tool,
      position,
      width: 0,
      depth: 0,
      radius: 0,
      height: this.options.minSize,
      phase: 'base',
      preview,
      extrudeMode: null,
      baseHeight: 0,
      baseRadius: 0,
    };
    this.build = build;
    this.scene.scene.add(preview.group);
    this.refreshPreview();
    return build;
  }

  /**
   * Two-hand base sizing around (0, 0, 0) from the pinch spans (units of
   * video width): cuboid -> width x depth rectangle from the horizontal /
   * vertical gaps; box -> square and cylinder / sphere -> diameter from the
   * straight-line gap.
   */
  private sizeBase(spanX: number | undefined, spanY: number | undefined): void {
    if (spanX === undefined || spanY === undefined) return;
    const build = this.build ?? this.startBuild(new THREE.Vector3(0, 0, 0));
    build.phase = 'base';
    build.extrudeMode = 'dual-hand';
    const { minSize, baseSizeScale } = this.options;
    const size = Math.max(minSize, Math.hypot(spanX, spanY) * baseSizeScale);
    if (build.tool === 'cuboid') {
      build.width = Math.max(minSize, spanX * baseSizeScale);
      build.depth = Math.max(minSize, spanY * baseSizeScale);
    } else {
      build.width = size;
      build.depth = size;
    }
    build.radius = size / 2;
    this.refreshPreview();
  }

  /** Rescale / reposition the preview meshes from the build state. */
  private refreshPreview(): void {
    const build = this.build;
    if (!build) return;
    const { fill, wire, group } = build.preview;
    const min = this.options.minSize;
    const { x, z } = build.position;
    let scaleX = min;
    let scaleY = min;
    let scaleZ = min;
    switch (build.tool) {
      case 'box':
      case 'cuboid': {
        const height = Math.max(build.height, min);
        scaleX = Math.max(build.width, min);
        scaleY = height;
        scaleZ = Math.max(build.depth, min);
        group.position.set(x, height / 2, z);
        break;
      }
      case 'cylinder': {
        const height = Math.max(build.height, min);
        const radius = Math.max(build.radius, min);
        scaleX = radius * 2;
        scaleY = height;
        scaleZ = radius * 2;
        group.position.set(x, height / 2, z);
        break;
      }
      case 'sphere': {
        const radius = Math.max(build.radius, min);
        scaleX = radius * 2;
        scaleY = radius * 2;
        scaleZ = radius * 2;
        group.position.set(x, radius, z);
        break;
      }
    }
    fill.scale.set(scaleX, scaleY, scaleZ);
    wire.scale.set(scaleX, scaleY, scaleZ);
  }

  /** Build the final (fresh, per-mesh) geometry for a committed solid. */
  private buildGeometry(
    tool: CadTool,
    width: number,
    height: number,
    depth: number,
    radius: number
  ): THREE.BufferGeometry {
    switch (tool) {
      case 'box':
      case 'cuboid':
        return new THREE.BoxGeometry(width, height, depth);
      case 'cylinder':
        return new THREE.CylinderGeometry(radius, radius, height, 48);
      case 'sphere':
        return new THREE.SphereGeometry(radius, 48, 24);
    }
  }

  /** Shared preview geometry per tool (never disposed — module singletons). */
  private unitGeometry(tool: CadTool): THREE.BufferGeometry {
    switch (tool) {
      case 'box':
      case 'cuboid':
        return UNIT_BOX;
      case 'cylinder':
        return UNIT_CYLINDER;
      case 'sphere':
        return UNIT_SPHERE;
    }
  }

  private createPreview(): Preview {
    const group = new THREE.Group();
    group.name = 'cad-preview';
    const fill = new THREE.Mesh(
      this.unitGeometry(this.tool),
      new THREE.MeshBasicMaterial({
        color: this.options.previewColor,
        transparent: true,
        opacity: 0.16,
        depthWrite: false,
      })
    );
    const wire = new THREE.Mesh(
      this.unitGeometry(this.tool),
      new THREE.MeshBasicMaterial({
        color: this.options.previewColor,
        wireframe: true,
        transparent: true,
        opacity: 0.55,
      })
    );
    group.add(fill, wire);
    return { group, fill, wire };
  }

  private bodyMaterial(color: string): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      color: new THREE.Color(color),
      roughness: 0.85,
      metalness: 0.08,
    });
  }

  private disposePreview(build: Build): void {
    const { group, fill, wire } = build.preview;
    this.scene.scene.remove(group);
    (fill.material as THREE.Material).dispose();
    (wire.material as THREE.Material).dispose();
    // UNIT_* geometries are shared module singletons: not disposed.
  }
}

/** Strip the Three.js handle so only plain data crosses the boundary. */
function toSummary(object: CommittedObject): SceneObjectSummary {
  return {
    id: object.id,
    shape: object.shape,
    width: object.width,
    depth: object.depth,
    height: object.height,
    x: object.x,
    z: object.z,
    color: object.color,
  };
}

/** Dispose a committed mesh's geometry, material and edge overlays. */
function disposeMesh(mesh: THREE.Mesh): void {
  mesh.geometry.dispose();
  (mesh.material as THREE.Material).dispose();
  for (const child of mesh.children) {
    if (child instanceof THREE.LineSegments) {
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
  }
}
