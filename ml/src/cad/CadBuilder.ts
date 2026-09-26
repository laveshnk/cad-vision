/**
 * CadBuilder: gesture-driven primitive construction for the CAD viewport.
 *
 * The builder consumes only device-space coordinates (x / y in [-1, 1],
 * +Y up), pinch lifecycle callbacks and extrusion deltas — it has no
 * dependency on the vision layer (src/vision) or MediaPipe. The app
 * orchestrator (src/main.ts) is the single place where gesture events are
 * translated into the calls below.
 *
 * Two-hand construction (primary flow):
 *   onExtrude(dual-hand) → both hands pinch: spawn a preview centered on the
 *                          world origin (0, 0, 0) and size its base from the
 *                          two pinches: box → square (side = pinch gap);
 *                          cylinder → circle (diameter = pinch gap); cuboid →
 *                          rectangle (horizontal gap → width, vertical gap →
 *                          depth); sphere → diameter. Any drawn single-hand
 *                          footprint is replaced (committed if it was already
 *                          released, discarded otherwise).
 *   onExtrude(single-hand) → one pinch released (the upper hand): the base
 *                          freezes and the still-pinched hand's vertical
 *                          travel drives the height (box / cylinder) or
 *                          radius (sphere). Re-pinching the second hand
 *                          returns to base sizing, keeping the height.
 *   commit({ flat })     → the orchestrator calls this when the last pinch
 *                          releases (or on fist / zoom transitions). `flat`
 *                          (both pinches released together, or the lower one
 *                          first) commits a flat plate of `flatHeight`.
 *
 * Single-hand footprint drawing (legacy flow):
 *   onPinchStart(x1, y1) → raycast Point A onto the ground plane (y = 0)
 *                          and spawn a wireframe preview for the active tool.
 *   onPinchDrag(x2, y2)  → raycast Point B and update the footprint / base
 *                          dimensions of the preview (box: square, cuboid:
 *                          corner-to-corner rectangle; cylinder / sphere:
 *                          center + radius).
 *   onPinchEnd()         → freeze the footprint; it is committed with the
 *                          default height on the next build / camera gesture.
 *
 * commit() replaces the wireframe with a solid matte mesh + EdgesGeometry
 * outlines and adds it to the scene.
 *
 * Selection (SELECT mode):
 *   pickAt(x, y)         → raycast a device-space pinch position against the
 *                          committed meshes: a hit selects that mesh
 *                          (highlighted) and anchors a ground-plane drag; a
 *                          miss clears the selection.
 *   dragTo(x, y)         → move the selected mesh so the grabbed point
 *                          follows the pinch, constrained by the active
 *                          drag constraint (`setDragConstraint`): 'xz' slides
 *                          across the ground plane with the elevation
 *                          locked at its current height; 'y' maps vertical
 *                          hand travel to a world-Y lift / lower (horizontal
 *                          drift ignored, never below the floor). Switching
 *                          the constraint mid-drag re-anchors, so the mesh
 *                          never jerks or resets.
 *   rotateSelection(d)   → secondary-hand open-palm rotation: yaw the selected
 *                          mesh to an anchored angle + d, with a compass
 *                          ring (circle + yaw needle) rendered around it in
 *                          the 3D viewport and the AR mirror.
 *   setSelectedColor(h)   → live-repaint the selected mesh's body material
 *                          from a `#rgb` / `#rrggbb` hex string (SELECT-mode
 *                          color wheel; the selection highlight is kept).
 *   selectedProjection()  → the selected mesh's center projected onto a
 *                          width × height canvas (CSS px) so the host can
 *                          anchor selection-adjacent UI (the color wheel).
 *   endDrag() / deselect() → finish the drag / clear the selection.
 */

import * as THREE from 'three';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import type { CadScene } from './CadScene';

export type CadTool = 'box' | 'cuboid' | 'cylinder' | 'sphere';

/**
 * SELECT-mode drag constraint: `'xz'` (default) slides the grabbed mesh
 * across the ground plane with its elevation locked; `'y'` maps vertical
 * hand travel to a world-Y lift / lower while horizontal drift is ignored.
 * Structurally compatible with the vision overlay's toggle type — the
 * orchestrator bridges the two without a shared import.
 */
export type DragConstraint = 'xz' | 'y';

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
  /** Device-space drag distance that re-purposes a pinch as a new drawing. */
  redrawDistance?: number;
  /** Workspace radius on the ground plane for raycast clamping. */
  groundRadius?: number;
  /** World units of lift per device-space unit of vertical drag (SELECT mode). */
  dragElevationScale?: number;
  /** Highest allowed mesh center height while drag-lifting (world units). */
  dragMaxHeight?: number;
  /** Preview wireframe / fill color. */
  previewColor?: number;
  /** Committed mesh body color (dark gray matte). */
  bodyColor?: number;
  /** Committed mesh edge-line color. */
  edgeColor?: number;
}

/**
 * footprint / awaiting-height: single-hand drawing. base: two-hand base
 * sizing. height: one pinch released, the other drives the height.
 */
type BuildPhase = 'footprint' | 'awaiting-height' | 'base' | 'height';

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
  /** Two-hand build: centered on the world origin, sized by the pinch gap. */
  centered: boolean;
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
const EDGE_THRESHOLD: Record<CadTool, number> = { box: 1, cuboid: 1, cylinder: 15, sphere: 25 };

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
  /** Currently selected (highlighted) committed mesh, or null. */
  private selected: THREE.Mesh | null = null;
  /**
   * Active selection drag: offset from the grabbed ground point to the mesh,
   * plus the vertical (lift) anchor — mesh height and device-space pinch
   * height at grab time, with the clamp range that keeps the mesh resting
   * on or above the ground plane. `lastX` / `lastY` remember the newest
   * pinch position so a mid-drag constraint toggle can re-anchor cleanly.
   */
  private selectionDrag: {
    offsetX: number;
    offsetZ: number;
    baseY: number;
    startDeviceY: number;
    minY: number;
    maxY: number;
    lastX: number;
    lastY: number;
  } | null = null;
  /** Active SELECT-mode drag constraint: ground plane ('xz') or Y axis ('y'). */
  private constraint: DragConstraint = 'xz';
  /** Yaw (radians) the open-palm rotation anchored at, or null when idle. */
  private selectionRotationAnchor: number | null = null;
  /** Whether an open-palm rotation gesture is currently running. */
  private rotating = false;
  /** Compass ring around the selection (built lazily, reused across gestures). */
  private rotationRing: THREE.Group | null = null;
  /** Yaw needle child of the compass ring. */
  private rotationNeedle: THREE.Line | null = null;
  /** Compass-ring radius in world units (refreshed from the selection's extents). */
  private rotationRingRadius = 1;

  constructor(scene: CadScene, options: CadBuilderOptions = {}) {
    this.scene = scene;
    this.options = {
      minFootprint: options.minFootprint ?? 0.15,
      minSize: options.minSize ?? 0.05,
      defaultHeight: options.defaultHeight ?? 0.5,
      flatHeight: options.flatHeight ?? 0.05,
      baseSizeScale: options.baseSizeScale ?? 10,
      extrudeScale: options.extrudeScale ?? 4.5,
      redrawDistance: options.redrawDistance ?? 0.35,
      groundRadius: options.groundRadius ?? 16,
      dragElevationScale: options.dragElevationScale ?? 3,
      dragMaxHeight: options.dragMaxHeight ?? 5,
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

  /** Number of currently selected meshes (0 or 1). */
  get selectedCount(): number {
    return this.selected ? 1 : 0;
  }

  /** Committed meshes (read-only view) — consumed by the AR mirror. */
  get committedMeshes(): readonly THREE.Mesh[] {
    return this.committed;
  }

  /** The currently selected committed mesh (AR highlight), or null. */
  get selectedMesh(): THREE.Mesh | null {
    return this.selected;
  }

  /** Select the primitive tool for the next (or in-progress) build. */
  setTool(tool: CadTool): void {
    if (this.tool === tool) return;
    this.tool = tool;
    const build = this.build;
    if (build && (build.phase === 'footprint' || build.phase === 'base')) {
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
    this.startBuild(this.groundPoint(x, y), 'footprint', false);
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
   * Extrusion input. Dual-hand: size the base of a two-hand build centered
   * on the origin from the pinch gap. Single-hand: vertical drag (device
   * units scaled into world units) adds height; spheres grow their radius.
   */
  onExtrude(input: CadExtrudeInput): void {
    if (input.mode === 'dual-hand') {
      this.sizeBase(input.spanX, input.spanY);
      return;
    }
    const build = this.build;
    if (!build || !build.centered) return;
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
   * Commit the active build: replace the wireframe preview with a solid
   * matte mesh (+ crisp edge overlays) and keep it in the scene. `flat`
   * commits a plate of `flatHeight` instead (spheres are unaffected).
   * @returns true if a mesh was committed.
   */
  commit({ flat = false }: { flat?: boolean } = {}): boolean {
    const build = this.build;
    if (!build || build.phase === 'footprint') return false;
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
    const radius = Math.max(build.radius, this.options.minSize);
    const geometry = this.buildGeometry(
      build.tool,
      Math.max(build.width, this.options.minSize),
      height,
      Math.max(build.depth, this.options.minSize),
      radius
    );
    // Selection drag-lift needs the local Y extents to clamp the mesh to the
    // ground plane; the AR mirror reuses the bounding box implicitly.
    geometry.computeBoundingBox();
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

  /** Remove every committed mesh from the scene (and any selection). */
  clear(): void {
    this.cancel();
    for (const mesh of this.committed) {
      this.root.remove(mesh);
      disposeMesh(mesh);
    }
    this.committed.length = 0;
    this.selected = null;
    this.selectionDrag = null;
    this.endRotateSelection();
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
    if (this.rotationRing) {
      this.scene.scene.remove(this.rotationRing);
      this.rotationRing.traverse((obj) => {
        if (obj instanceof THREE.Line) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
      this.rotationRing = null;
      this.rotationNeedle = null;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Selection (SELECT mode)                                            */
  /* ------------------------------------------------------------------ */

  /**
   * Raycast a device-space point against the committed meshes (SELECT mode).
   * A hit selects that mesh and anchors a ground-plane drag at the pinch
   * position; a miss clears the selection.
   * @returns true if a mesh is now selected.
   */
  pickAt(x: number, y: number): boolean {
    this.raycaster.setFromCamera(this.ndc.set(x, y), this.scene.camera);
    const hits = this.raycaster.intersectObjects(this.committed, false);
    const mesh = hits.length > 0 ? (hits[0].object as THREE.Mesh) : null;
    if (!mesh) {
      this.deselect();
      return false;
    }
    this.select(mesh);
    const grab = this.groundPoint(x, y);
    // Lock the pinch offset relative to the mesh origin, plus the vertical
    // (lift) anchor: the mesh may never sink below the ground plane.
    const bottom = mesh.geometry.boundingBox?.min.y ?? 0;
    const minY = -bottom;
    this.selectionDrag = {
      offsetX: mesh.position.x - grab.x,
      offsetZ: mesh.position.z - grab.z,
      baseY: mesh.position.y,
      startDeviceY: y,
      minY,
      maxY: Math.max(minY, this.options.dragMaxHeight),
      lastX: x,
      lastY: y,
    };
    return true;
  }

  /**
   * Drag the selected mesh so its grabbed ground point follows the given
   * device-space position (SELECT mode), constrained by the active drag
   * constraint:
   * - `'xz'`: X / Z track the ground-plane raycast in real time; the
   *   elevation stays locked at whatever height the mesh currently has
   *   (Y untouched).
   * - `'y'`: vertical hand travel lifts / lowers the mesh (scaled, clamped
   *   so it never sinks below the ground plane); horizontal drift is
   *   ignored (X / Z untouched).
   */
  dragTo(x: number, y: number): void {
    const mesh = this.selected;
    const drag = this.selectionDrag;
    if (!mesh || !drag) return;
    drag.lastX = x;
    drag.lastY = y;
    if (this.constraint === 'xz') {
      const target = this.groundPoint(x, y);
      mesh.position.x = target.x + drag.offsetX;
      mesh.position.z = target.z + drag.offsetZ;
    } else {
      mesh.position.y = THREE.MathUtils.clamp(
        drag.baseY + (y - drag.startDeviceY) * this.options.dragElevationScale,
        drag.minY,
        drag.maxY
      );
    }
    if (this.rotating) this.updateRotationRing();
  }

  /** The active SELECT-mode drag constraint ('xz' | 'y'). */
  get dragConstraint(): DragConstraint {
    return this.constraint;
  }

  /**
   * Switch the SELECT-mode drag constraint. A switch in the middle of a drag
   * re-anchors the offsets at the mesh's live position and the newest pinch
   * position, so toggling between the ground plane and the Y axis never
   * resets the object's position or causes a jerk.
   */
  setDragConstraint(constraint: DragConstraint): void {
    if (this.constraint === constraint) return;
    this.constraint = constraint;
    const mesh = this.selected;
    const drag = this.selectionDrag;
    if (!mesh || !drag) return;
    const grab = this.groundPoint(drag.lastX, drag.lastY);
    drag.offsetX = mesh.position.x - grab.x;
    drag.offsetZ = mesh.position.z - grab.z;
    drag.baseY = mesh.position.y;
    drag.startDeviceY = drag.lastY;
  }

  /**
   * Begin rotating the selected mesh around the world Y axis (SELECT mode,
   * open-palm secondary hand): anchors at the mesh's current yaw and shows
   * the compass ring (circle + yaw needle) around it.
   */
  beginRotateSelection(): void {
    const mesh = this.selected;
    if (!mesh) return;
    this.selectionRotationAnchor = mesh.rotation.y;
    this.rotating = true;
    this.ensureRotationRing().visible = true;
    this.updateRotationRing();
  }

  /**
   * Rotate the selected mesh to `anchor + deltaRotation` (SELECT mode): the
   * vision layer reports the cumulative open-palm tilt since its gesture
   * began, and the first call after a gesture starts (or after
   * `endRotateSelection`) anchors at the mesh's current yaw — so
   * `selectedMesh.rotation.y = initialRotation + deltaRotation`.
   */
  rotateSelection(deltaRotation: number): void {
    const mesh = this.selected;
    if (!mesh || !Number.isFinite(deltaRotation)) return;
    if (this.selectionRotationAnchor === null) this.beginRotateSelection();
    const anchor = this.selectionRotationAnchor;
    if (anchor === null) return; // no selection to anchor at
    mesh.rotation.y = anchor + deltaRotation;
    if (this.rotating) this.updateRotationRing();
  }

  /** End the open-palm rotation (hides the compass ring; the yaw persists). */
  endRotateSelection(): void {
    this.selectionRotationAnchor = null;
    this.rotating = false;
    if (this.rotationRing) this.rotationRing.visible = false;
  }

  /** Whether an open-palm rotation gesture is currently running. */
  get isRotating(): boolean {
    return this.rotating;
  }

  /**
   * World-space compass ring around the selection while rotating (consumed
   * by the AR mirror): center + radius + current yaw; `null` when idle.
   */
  get selectionRingWorld(): { center: THREE.Vector3; radius: number; angle: number } | null {
    if (!this.rotating || !this.selected) return null;
    return {
      center: this.selected.position,
      radius: this.rotationRingRadius,
      angle: this.selected.rotation.y,
    };
  }

  /**
   * Live-repaint the active selection's body material from a CSS hex string
   * (`#rgb` or `#rrggbb`, case-insensitive — the SELECT-mode color wheel's
   * format). The selection highlight (emissive tint + accent edges) is kept,
   * so the mesh stays visibly selected in its new color.
   * @returns true when the color was applied; false when nothing is selected
   * or the hex is malformed (the scene is left untouched).
   */
  setSelectedColor(hexColor: string): boolean {
    const mesh = this.selected;
    const hex = parseHexColor(hexColor);
    if (!mesh || hex === null) return false;
    (mesh.material as THREE.MeshStandardMaterial).color.setHex(hex);
    return true;
  }

  /**
   * The selected mesh's center projected through the live scene camera onto
   * a `canvasWidth × canvasHeight` canvas (CSS px, top-left origin — the
   * same mapping the AR mirror uses). Lets the orchestrator anchor
   * selection-adjacent UI (the SELECT-mode color wheel) in lockstep with the
   * 3D viewport without leaking Three.js types.
   * @returns null when nothing is selected or the center is behind the camera.
   */
  selectedProjection(
    canvasWidth: number,
    canvasHeight: number
  ): { x: number; y: number } | null {
    const mesh = this.selected;
    if (!mesh) return null;
    const projected = this.scene.projectToCanvas(mesh.position, canvasWidth, canvasHeight);
    if (projected.z > 1) return null; // behind the camera
    return { x: projected.x, y: projected.y };
  }

  /** End the active selection drag (the selection itself persists). */
  endDrag(): void {
    this.selectionDrag = null;
    this.endRotateSelection();
  }

  /** Clear the selection and its highlight. */
  deselect(): void {
    if (this.selected) this.setSelectionStyle(this.selected, false);
    this.selected = null;
    this.selectionDrag = null;
    this.endRotateSelection();
  }

  private select(mesh: THREE.Mesh): void {
    if (this.selected === mesh) return;
    this.deselect();
    this.selected = mesh;
    this.setSelectionStyle(mesh, true);
  }

  /** Highlight / restore a committed mesh: emissive tint + accent edges. */
  private setSelectionStyle(mesh: THREE.Mesh, selected: boolean): void {
    const body = mesh.material as THREE.MeshStandardMaterial;
    body.emissive.setHex(selected ? 0x0284c7 : 0x000000);
    body.emissiveIntensity = selected ? 0.35 : 1;
    for (const child of mesh.children) {
      if (child instanceof THREE.LineSegments) {
        (child.material as THREE.LineBasicMaterial).color.setHex(
          selected ? 0x0284c7 : this.options.edgeColor
        );
      }
    }
  }

  /**
   * Compass ring for the open-palm rotation: a world-fixed horizontal circle
   * around the selected mesh plus a yaw needle (the object's local +X
   * direction) that sweeps as it rotates. Built once and reused across
   * gestures — only position / scale / needle yaw are refreshed per frame.
   * Lives in the scene (not the committed root) so it never shows up in the
   * STL export, the raycast pick or the AR ghost list.
   */
  private ensureRotationRing(): THREE.Group {
    if (this.rotationRing) return this.rotationRing;
    const segments = 72;
    const circle: THREE.Vector3[] = [];
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      circle.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
    }
    const ring = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(circle),
      new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.85 })
    );
    const needle = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(1, 0, 0),
      ]),
      new THREE.LineBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.95 })
    );
    const group = new THREE.Group();
    group.name = 'cad-rotation-ring';
    group.add(ring, needle);
    group.visible = false;
    this.rotationRing = group;
    this.rotationNeedle = needle;
    this.scene.scene.add(group);
    return group;
  }

  /** Refresh the ring's position / scale and the needle's yaw from the selection. */
  private updateRotationRing(): void {
    const mesh = this.selected;
    const ring = this.rotationRing;
    if (!mesh || !ring) return;
    const box = mesh.geometry.boundingBox;
    const halfWidth = box ? (box.max.x - box.min.x) / 2 : 0.5;
    const halfDepth = box ? (box.max.z - box.min.z) / 2 : 0.5;
    this.rotationRingRadius = Math.max(halfWidth, halfDepth) * 1.35 + 0.2;
    ring.position.copy(mesh.position);
    ring.scale.set(this.rotationRingRadius, 1, this.rotationRingRadius);
    if (this.rotationNeedle) this.rotationNeedle.rotation.y = mesh.rotation.y;
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

  /** Spawn a new build (preview + state) anchored at `origin`. */
  private startBuild(origin: THREE.Vector3, phase: BuildPhase, centered: boolean): Build {
    const preview = this.createPreview();
    const build: Build = {
      tool: this.tool,
      origin,
      point: origin.clone(),
      width: 0,
      depth: 0,
      radius: 0,
      height: this.options.minSize,
      phase,
      preview,
      centered,
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
   * video width): cuboid → width × depth rectangle from the horizontal /
   * vertical gaps; box → square and cylinder / sphere → diameter from the
   * straight-line gap. Starts the centered build on the first dual-hand
   * frame; a single-hand footprint in progress is dropped (it was the
   * lead-in to this pinch) and a released one is committed first.
   */
  private sizeBase(spanX: number | undefined, spanY: number | undefined): void {
    if (spanX === undefined || spanY === undefined) return;
    let build = this.build;
    if (build && !build.centered) {
      if (build.phase === 'footprint') this.cancel();
      else this.commit();
      build = null;
    }
    if (!build) build = this.startBuild(new THREE.Vector3(0, 0, 0), 'base', true);
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

  /** Update footprint dimensions from a ground-plane Point B. */
  private updateFootprint(point: THREE.Vector3): void {
    const build = this.build;
    if (!build || build.phase !== 'footprint') return;
    const dx = point.x - build.origin.x;
    const dz = point.z - build.origin.z;
    build.radius = Math.hypot(dx, dz);
    if (build.tool === 'box') {
      // Square: the larger extent, extended from Point A toward Point B.
      const side = Math.max(Math.abs(dx), Math.abs(dz));
      build.width = side;
      build.depth = side;
      build.point = new THREE.Vector3(
        build.origin.x + (dx < 0 ? -side : side),
        0,
        build.origin.z + (dz < 0 ? -side : side)
      );
    } else {
      build.point = point;
      build.width = Math.abs(dx);
      build.depth = Math.abs(dz);
    }
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
      case 'box':
      case 'cuboid': {
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
   * Footprint anchor: boxes / cuboids span corner-to-corner (midpoint center);
   * cylinders / spheres are centered on Point A with radius to Point B.
   */
  private buildCenter(build: Build): THREE.Vector3 {
    if (build.tool === 'box' || build.tool === 'cuboid') {
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

/**
 * Parse a strict CSS hex color (`#rgb` or `#rrggbb`, case-insensitive) into
 * a `0xrrggbb` integer; null when malformed. Never throws, so a bad value
 * leaves the scene untouched.
 */
function parseHexColor(hexColor: string): number | null {
  const match = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hexColor);
  if (!match) return null;
  const digits = match[1];
  if (digits.length === 3) {
    // #rgb expands by digit duplication (#f00 → #ff0000).
    return (
      (parseInt(digits[0], 16) * 17) << 16 |
      (parseInt(digits[1], 16) * 17) << 8 |
      parseInt(digits[2], 16) * 17
    );
  }
  return parseInt(digits, 16);
}
