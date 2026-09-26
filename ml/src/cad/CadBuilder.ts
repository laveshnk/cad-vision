/**
 * CadBuilder: gesture-driven primitive construction for the CAD viewport.
 *
 * The builder consumes only device-space coordinates (x / y in [-1, 1],
 * +Y up), pinch lifecycle callbacks and extrusion deltas — it has no
 * dependency on the vision layer (src/vision) or MediaPipe. The app
 * orchestrator (src/main.ts) is the single place where gesture events are
 * translated into the calls below.
 *
 * Construction flow per primitive:
 *   onPinchStart(x1, y1) → raycast Point A onto the ground plane (y = 0)
 *                          and spawn a wireframe preview for the active tool.
 *   onPinchDrag(x2, y2)  → raycast Point B and update the footprint / base
 *                          dimensions of the preview (box: corner-to-corner
 *                          rectangle; cylinder / sphere: center + radius).
 *   onPinchEnd()         → freeze the footprint and await height / extrusion.
 *   onExtrude(input)     → vertical drag (single-hand) or pull-scale
 *                          (dual-hand) drives the height (box / cylinder)
 *                          or radius (sphere).
 *   commit()             → replace the wireframe with a solid matte mesh +
 *                          EdgesGeometry outlines and add it to the scene.
 *                          The orchestrator triggers this on extrude
 *                          release or fist / open-palm transitions.
 */

import * as THREE from 'three';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import type { CadScene } from './CadScene';

export type CadTool = 'box' | 'cylinder' | 'sphere';

/**
 * Extrusion delta; structurally compatible with the vision layer's extrude
 * event payload so the orchestrator can forward it directly.
 */
export interface CadExtrudeInput {
  mode: 'dual-hand' | 'single-hand';
  /** Cumulative pull scale since extrusion start (dual-hand). */
  scaleFactor?: number;
  /** Cumulative vertical travel in device units (single-hand, +Y up). */
  cumulativeHeight?: number;
}

export interface CadBuilderOptions {
  /** Minimum footprint (world units) for a build to survive a pinch release. */
  minFootprint?: number;
  /** Minimum committed height / radius (world units). */
  minSize?: number;
  /** Height used when committing a build that was never extruded. */
  defaultHeight?: number;
  /** Baseline height that the dual-hand scale factor multiplies. */
  extrudeBaseHeight?: number;
  /** World units of extrusion per device-space unit of vertical drag. */
  extrudeScale?: number;
  /** Device-space drag distance that re-purposes a pinch as a new drawing. */
  redrawDistance?: number;
  /** Workspace radius on the ground plane for raycast clamping. */
  groundRadius?: number;
  /** Preview wireframe / fill color. */
  previewColor?: number;
  /** Committed mesh body color (dark gray matte). */
  bodyColor?: number;
  /** Committed mesh edge-line color. */
  edgeColor?: number;
}

type BuildPhase = 'footprint' | 'awaiting-height' | 'height';

/** Wireframe preview handles (fill + wireframe + Point-A marker). */
interface Preview {
  group: THREE.Group;
  fill: THREE.Mesh;
  wire: THREE.Mesh;
  marker: THREE.Mesh;
}

interface Build {
  tool: CadTool;
  /** Point A (pinch start) on the ground plane. */
  origin: THREE.Vector3;
  /** Latest Point B (pinch drag) on the ground plane. */
  point: THREE.Vector3;
  /** Footprint extents (box: corner-to-corner). */
  width: number;
  depth: number;
  /** Base radius (cylinder / sphere: center-to-edge). */
  radius: number;
  /** Current height (box / cylinder). */
  height: number;
  phase: BuildPhase;
  preview: Preview;
  /** Extrusion rebase snapshots for continuity across mode switches. */
  extrudeMode: 'dual-hand' | 'single-hand' | null;
  baseHeight: number;
  baseRadius: number;
}

/** Shared unit geometries — previews only rescale, never rebuild. */
const UNIT_BOX = new THREE.BoxGeometry(1, 1, 1);
const UNIT_CYLINDER = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
const UNIT_SPHERE = new THREE.SphereGeometry(0.5, 32, 16);
const MARKER = new THREE.SphereGeometry(0.07, 12, 8);

/**
 * EdgesGeometry crease thresholds per tool: boxes show every edge,
 * cylinders only the top/bottom rims, spheres stay smooth.
 */
const EDGE_THRESHOLD: Record<CadTool, number> = { box: 1, cylinder: 15, sphere: 25 };

export class CadBuilder {
  private readonly scene: CadScene;
  private readonly options: Required<CadBuilderOptions>;
  /** Persistent hierarchy of committed meshes. */
  private readonly root: THREE.Group;
  private readonly committed: THREE.Mesh[] = [];
  private readonly raycaster = new THREE.Raycaster();
  private readonly groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private readonly ndc = new THREE.Vector2();

  private tool: CadTool = 'box';
  private build: Build | null = null;

  constructor(scene: CadScene, options: CadBuilderOptions = {}) {
    this.scene = scene;
    this.options = {
      minFootprint: options.minFootprint ?? 0.15,
      minSize: options.minSize ?? 0.05,
      defaultHeight: options.defaultHeight ?? 0.5,
      extrudeBaseHeight: options.extrudeBaseHeight ?? 0.5,
      extrudeScale: options.extrudeScale ?? 4.5,
      redrawDistance: options.redrawDistance ?? 0.35,
      groundRadius: options.groundRadius ?? 16,
      previewColor: options.previewColor ?? 0x0284c7,
      bodyColor: options.bodyColor ?? 0x3f3f46,
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

  /** Select the primitive tool for the next (or in-progress) build. */
  setTool(tool: CadTool): void {
    if (this.tool === tool) return;
    this.tool = tool;
    const build = this.build;
    if (build && build.phase === 'footprint') {
      // Swap the preview geometry while keeping the drawn footprint.
      this.disposePreview(build);
      build.tool = tool;
      build.preview = this.createPreview();
      this.scene.scene.add(build.preview.group);
      this.refreshPreview();
    }
  }

  /* ------------------------------------------------------------------ */
  /* Gesture input (device space)                                       */
  /* ------------------------------------------------------------------ */

  /** Pinch engaged at device coords (x, y): lock Point A, spawn a preview. */
  onPinchStart(x: number, y: number): void {
    if (this.build) return; // pending build: this pinch is likely extrude prep
    const origin = this.groundPoint(x, y);
    const preview = this.createPreview();
    this.build = {
      tool: this.tool,
      origin,
      point: origin.clone(),
      width: 0,
      depth: 0,
      radius: 0,
      height: this.options.minSize,
      phase: 'footprint',
      preview,
      extrudeMode: null,
      baseHeight: 0,
      baseRadius: 0,
    };
    this.scene.scene.add(preview.group);
    this.refreshPreview();
  }

  /**
   * Pinch moved to device coords (x, y): update the footprint from Point B.
   * (startX / startY are the device coords where this pinch began; a far
   * travel while awaiting height starts a new primitive instead.)
   */
  onPinchDrag(x: number, y: number, startX = x, startY = y): void {
    const build = this.build;
    if (!build) return;
    if (build.phase === 'footprint') {
      this.updateFootprint(this.groundPoint(x, y));
    } else if (build.phase === 'awaiting-height') {
      const travel = Math.hypot(x - startX, y - startY);
      if (travel > this.options.redrawDistance) {
        // A long drag re-purposes this pinch as a new drawing: finalize the
        // pending build with its default height and start over.
        this.commit();
        this.onPinchStart(startX, startY);
        this.updateFootprint(this.groundPoint(x, y));
      }
    }
  }

  /** Pinch released: freeze the footprint and await height / extrusion. */
  onPinchEnd(): void {
    const build = this.build;
    if (!build || build.phase !== 'footprint') return;
    const footprint = Math.max(build.width, build.depth, build.radius);
    if (footprint < this.options.minFootprint) {
      this.cancel(); // accidental tap: nothing meaningful was drawn
      return;
    }
    build.phase = 'awaiting-height';
  }

  /**
   * Extrusion delta: single-hand vertical drag adds height (device units
   * scaled into world units); dual-hand pull scales the baseline height.
   * Spheres grow their radius instead of a height.
   */
  onExtrude(input: CadExtrudeInput): void {
    const build = this.build;
    if (!build) return;
    if (build.phase === 'footprint') {
      // Extrusion can begin straight from a drag (e.g. the second hand
      // pinches while the first is still drawing): freeze the footprint.
      build.phase = 'awaiting-height';
    }
    if (build.phase === 'awaiting-height') {
      build.phase = 'height';
      build.extrudeMode = null;
    }
    if (build.extrudeMode !== input.mode) {
      // First extrusion frame or a mode switch: rebase so dimensions stay
      // continuous across the transition.
      const first = build.extrudeMode === null;
      build.extrudeMode = input.mode;
      build.baseHeight = first
        ? Math.max(build.height, this.options.extrudeBaseHeight)
        : build.height;
      build.baseRadius = first
        ? Math.max(build.radius, this.options.minSize)
        : build.radius;
    }
    if (input.mode === 'dual-hand') {
      const scale = Math.max(input.scaleFactor ?? 1, 0.05);
      build.height = Math.max(this.options.minSize, build.baseHeight * scale);
      build.radius = Math.max(this.options.minSize, build.baseRadius * scale);
    } else {
      const growth = (input.cumulativeHeight ?? 0) * this.options.extrudeScale;
      build.height = Math.max(this.options.minSize, build.baseHeight + growth);
      build.radius = Math.max(this.options.minSize, build.baseRadius + growth);
    }
    this.refreshPreview();
  }

  /**
   * Commit the active build: replace the wireframe preview with a solid
   * matte mesh (+ crisp edge overlays) and keep it in the scene.
   * @returns true if a mesh was committed.
   */
  commit(): boolean {
    const build = this.build;
    if (!build || (build.phase !== 'awaiting-height' && build.phase !== 'height')) {
      return false;
    }
    const footprint = Math.max(build.width, build.depth, build.radius);
    if (footprint < this.options.minFootprint) {
      this.cancel();
      return false;
    }
    const height =
      build.phase === 'height'
        ? build.height
        : Math.max(build.height, this.options.defaultHeight);
    const radius = Math.max(build.radius, this.options.minSize);
    const geometry = this.buildGeometry(
      build.tool,
      Math.max(build.width, this.options.minSize),
      height,
      Math.max(build.depth, this.options.minSize),
      radius
    );
    const mesh = new THREE.Mesh(geometry, this.bodyMaterial());
    mesh.position.copy(this.buildCenter(build));
    mesh.position.y = build.tool === 'sphere' ? radius : height / 2;
    // Cast + receive soft shadows so solids read as physically grounded.
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, EDGE_THRESHOLD[build.tool]),
      new THREE.LineBasicMaterial({
        color: this.options.edgeColor,
        transparent: true,
        opacity: 0.9,
      })
    );
    mesh.add(edges);
    this.root.add(mesh);
    this.committed.push(mesh);
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

  /** Remove every committed mesh from the scene. */
  clear(): void {
    this.cancel();
    for (const mesh of this.committed) {
      this.root.remove(mesh);
      disposeMesh(mesh);
    }
    this.committed.length = 0;
  }

  /** Download all committed meshes as a binary `model.stl`. */
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

  /**
   * Raycast from device coords ([-1, 1], +Y up) through the camera onto the
   * ground plane (y = 0). Rays pointing at/above the horizon are clamped
   * slightly downward so a ground hit always exists; the result is clamped
   * to the workspace radius.
   */
  private groundPoint(x: number, y: number): THREE.Vector3 {
    this.raycaster.setFromCamera(this.ndc.set(x, y), this.scene.camera);
    const ray = this.raycaster.ray;
    if (ray.direction.y > -0.05) {
      ray.direction.y = -0.05;
      ray.direction.normalize();
    }
    const point = new THREE.Vector3();
    ray.intersectPlane(this.groundPlane, point);
    const radius = Math.hypot(point.x, point.z);
    const max = this.options.groundRadius;
    if (radius > max) {
      point.x = (point.x / radius) * max;
      point.z = (point.z / radius) * max;
    }
    return point;
  }

  /** Update footprint dimensions from a ground-plane Point B. */
  private updateFootprint(point: THREE.Vector3): void {
    const build = this.build;
    if (!build || build.phase !== 'footprint') return;
    build.point = point;
    build.width = Math.abs(point.x - build.origin.x);
    build.depth = Math.abs(point.z - build.origin.z);
    build.radius = Math.hypot(point.x - build.origin.x, point.z - build.origin.z);
    this.refreshPreview();
  }

  /** Rescale / reposition the preview meshes from the build state. */
  private refreshPreview(): void {
    const build = this.build;
    if (!build) return;
    const { fill, wire, marker, group } = build.preview;
    const min = this.options.minSize;
    const center = this.buildCenter(build);
    let scaleX = min;
    let scaleY = min;
    let scaleZ = min;
    switch (build.tool) {
      case 'box': {
        const height = Math.max(build.height, min);
        scaleX = Math.max(build.width, min);
        scaleY = height;
        scaleZ = Math.max(build.depth, min);
        group.position.set(center.x, height / 2, center.z);
        break;
      }
      case 'cylinder': {
        const height = Math.max(build.height, min);
        const radius = Math.max(build.radius, min);
        scaleX = radius * 2;
        scaleY = height;
        scaleZ = radius * 2;
        group.position.set(center.x, height / 2, center.z);
        break;
      }
      case 'sphere': {
        const radius = Math.max(build.radius, min);
        scaleX = radius * 2;
        scaleY = radius * 2;
        scaleZ = radius * 2;
        group.position.set(center.x, radius, center.z);
        break;
      }
    }
    fill.scale.set(scaleX, scaleY, scaleZ);
    wire.scale.set(scaleX, scaleY, scaleZ);
    marker.position.set(build.origin.x, 0, build.origin.z);
  }

  /**
   * Footprint anchor: boxes span corner-to-corner (midpoint center);
   * cylinders / spheres are centered on Point A with radius to Point B.
   */
  private buildCenter(build: Build): THREE.Vector3 {
    if (build.tool === 'box') {
      return new THREE.Vector3(
        (build.origin.x + build.point.x) / 2,
        0,
        (build.origin.z + build.point.z) / 2
      );
    }
    return new THREE.Vector3(build.origin.x, 0, build.origin.z);
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
    const marker = new THREE.Mesh(
      MARKER,
      new THREE.MeshBasicMaterial({ color: this.options.previewColor })
    );
    group.add(fill, wire, marker);
    return { group, fill, wire, marker };
  }

  private bodyMaterial(): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      color: this.options.bodyColor,
      roughness: 0.85,
      metalness: 0.08,
    });
  }

  private disposePreview(build: Build): void {
    const { group, fill, wire, marker } = build.preview;
    this.scene.scene.remove(group);
    (fill.material as THREE.Material).dispose();
    (wire.material as THREE.Material).dispose();
    (marker.material as THREE.Material).dispose();
    // UNIT_* geometries and MARKER are shared module singletons: not disposed.
  }
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
