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
 *   pickAt(x, y, t)      → raycast a device-space pinch position against the
 *                          committed meshes: a hit selects that mesh (outline
 *                          highlight) and arms a drag; a miss clears the
 *                          selection. A quick pinch (tap) therefore just
 *                          selects — the selection persists after release.
 *   dragTo(x, y, t)      → once the pinch has been held for `dragHoldMs`,
 *                          move the selected mesh so the grabbed point
 *                          stays exactly under the pinch, constrained by the
 *                          active drag constraint (`setDragConstraint`):
 *                          'xz' slides it on a horizontal plane at the grab
 *                          point's height (elevation locked), 'y' lifts /
 *                          lowers it along a camera-facing vertical plane
 *                          through the grab point (horizontal drift ignored,
 *                          never below the floor). Switching
 *                          the constraint mid-drag re-anchors, so the mesh
 *                          never jerks or resets.
 *   rotateSelection(d)   → secondary-hand open-palm rotation: yaw the selected
 *                          mesh to an anchored angle + d, with a compass
 *                          ring (circle + yaw needle) rendered around it in
 *                          the 3D viewport and the AR mirror.
 *   setSelectedColor(h)   → live-repaint the selected mesh's body material
 *                          from a `#rgb` / `#rrggbb` hex string (SELECT-mode
 *                          color wheel; the selection outline is kept).
 *   selectedProjection()  → the selected mesh's center projected onto a
 *                          width × height canvas (CSS px) so the host can
 *                          anchor selection-adjacent UI (the color wheel).
 *   deleteSelectedMesh()  → remove the selected mesh from the scene and the
 *                          committed list, disposing its geometry /
 *                          material / edge overlays (the host gates this
 *                          behind a confirmation dialog).
 *   endDrag() / deselect() → finish the drag / clear the selection.
 *
 * CSG Booleans (SELECT mode, see booleanOps.ts):
 *   setBooleanTool(op)   → arm / disarm the Boolean tool ('subtract' |
 *                          'union'), mirrored on the vision overlay's
 *                          toggles; arming alone never mutates the scene.
 *   (clash tracking)     → as the selection moves (drag / rotate), the
 *                          builder tests its world AABB against every other
 *                          committed mesh (`box3.intersectsBox`) and keeps a
 *                          translucent amber clash indicator on the overlap
 *                          region while two solids intersect.
 *   applyBoolean([op])    → execute the armed (or explicit) operation on the
 *                          live clash: subtract carves the selected cutter
 *                          out of the intersected base (the base is replaced
 *                          by the CSG result with fresh EdgesGeometry, the
 *                          cutter is removed); union fuses both solids into
 *                          one continuous body. The result stays selected.
 */

import * as THREE from 'three';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import type { CadScene } from './CadScene';
import { AxisLock } from './axisLock';
import { boxesClash, evaluateBoolean, overlapBox, worldBounds, type BooleanOperation } from './booleanOps';

/** World up: the fist-rotation yaw axis. */
const WORLD_Y = new THREE.Vector3(0, 1, 0);

/** Lowest world-space Y over a mesh's actual vertices (exact, unlike an AABB of a rotated box). */
function worldBottom(mesh: THREE.Mesh): number {
  const position = mesh.geometry.getAttribute('position');
  if (!position) return mesh.position.y;
  const vertex = new THREE.Vector3();
  let bottom = Infinity;
  for (let i = 0; i < position.count; i++) {
    vertex.fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld);
    if (vertex.y < bottom) bottom = vertex.y;
  }
  return bottom;
}

export type CadTool = 'box' | 'cuboid' | 'cylinder' | 'sphere';

/**
 * SELECT-mode CSG Boolean operation, structurally compatible with the
 * vision overlay's `OverlayBooleanTool` (the orchestrator bridges the two
 * without a shared import): `'subtract'` carves the selected cutter out of
 * the intersected base mesh, `'union'` merges both solids into one.
 */
export type { BooleanOperation };

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
  /**
   * Allow the legacy single-hand footprint drawing (one pinch raycast onto
   * the floor starts a build). Off by default: a single stray / misdetected
   * pinch would drop objects in random spots — building takes the
   * deliberate two-hand gesture. Default false.
   */
  singleHandFootprint?: boolean;
  /**
   * EDIT mode fist rotation: radians of object rotation per device-space
   * unit of fist travel (left / right → vertical axis, up / down → the
   * camera's horizontal axis). Default 2.5 (a fist sweep across a third of
   * the view ≈ 95°).
   */
  fistRotateSpeed?: number;
  /** Per-axis fist delta (device units) below which rotation ignores jitter. */
  fistRotateDeadzone?: number;
  /**
   * SELECT mode: a pinch must be held this long (ms) before it starts moving
   * the picked mesh, so a quick pinch only selects it. Default 300.
   */
  dragHoldMs?: number;
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

/**
 * Selection outline accent (bright sky on the matte dark-gray bodies) and the
 * world-units gap the outline shell keeps around the selected mesh — the
 * silhouette border stays equally visible on small and large solids alike.
 */
const SELECTION_OUTLINE_COLOR = 0x0ea5e9;
const SELECTION_OUTLINE_OFFSET = 0.05;

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
    /** Pinch timestamp (ms) at grab time; moving starts `dragHoldMs` later. */
    grabbedAt: number;
    /** False while the pinch is still a possible tap (mesh stays put). */
    moving: boolean;
    /**
     * Height of the grabbed surface point above the mesh origin: drags run
     * on planes through that point, so it stays under the fingertip.
     */
    grabHeight: number;
    /** Lift-plane hit height at the last anchor ('y' constraint), or null. */
    liftStartY: number | null;
    /** The mesh was already selected at pinch start: a tap deselects it. */
    toggleOnTap: boolean;
  } | null = null;
  /** One-axis-at-a-time filter for fist rotation (see `AxisLock`). */
  private readonly fistAxisLock = new AxisLock();
  /** Scratch plane for grab-height / lift raycasts. */
  private readonly dragPlane = new THREE.Plane();
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
  /**
   * Selection outline: an inverted-hull shell (the selected mesh's own
   * geometry with a back-face accent material, scaled a hair larger) drawn
   * around the selected mesh. Built lazily and reused across selections —
   * only its geometry / transform are refreshed.
   */
  private selectionOutline: THREE.Mesh | null = null;
  /**
   * CSG Boolean tooling: the armed operation (`setBooleanTool`), the
   * committed mesh the selection currently clashes with (world-AABB
   * intersection, recomputed as the selection moves), and the translucent
   * amber clash indicator volume (the overlap region of the two AABBs —
   * lazily built, hidden when no clash).
   */
  private booleanTool: BooleanOperation | null = null;
  private clashTarget: THREE.Mesh | null = null;
  private clashBox: THREE.Group | null = null;
  /** Scratch boxes for clash detection (reused every frame). */
  private readonly scratchBoxA = new THREE.Box3();
  private readonly scratchBoxB = new THREE.Box3();

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
      singleHandFootprint: options.singleHandFootprint ?? false,
      fistRotateSpeed: options.fistRotateSpeed ?? 2.5,
      fistRotateDeadzone: options.fistRotateDeadzone ?? 0.0015,
      dragHoldMs: options.dragHoldMs ?? 300,
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
    if (!this.options.singleHandFootprint) return; // two-hand builds only
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
    this.hideSelectionOutline();
    this.endRotateSelection();
    this.clashTarget = null;
    this.syncClashIndicator(null); // no meshes left to clash
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
    if (this.selectionOutline) {
      this.scene.scene.remove(this.selectionOutline);
      (this.selectionOutline.material as THREE.Material).dispose();
      this.selectionOutline = null;
    }
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
    if (this.clashBox) {
      this.scene.scene.remove(this.clashBox);
      this.clashBox.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
      this.clashBox = null;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Selection (SELECT mode)                                            */
  /* ------------------------------------------------------------------ */

  /**
   * Raycast a device-space point against the committed meshes (SELECT mode).
   * A hit selects that mesh and arms a drag anchored at the pinch position
   * (it starts moving only after `dragHoldMs`, see `dragTo`); a miss clears
   * the selection.
   * @param timestamp pinch start time (ms); omit to allow moving immediately.
   * @returns true if a mesh is now selected.
   */
  pickAt(x: number, y: number, timestamp = -Infinity): boolean {
    // Hand coords are the webcam frame's device space = interaction-camera NDC.
    this.raycaster.setFromCamera(this.ndc.set(x, y), this.scene.interactionCamera);
    const hits = this.raycaster.intersectObjects(this.committed, false);
    const mesh = hits.length > 0 ? (hits[0].object as THREE.Mesh) : null;
    if (!mesh) {
      this.deselect();
      return false;
    }
    const wasSelected = this.selected === mesh;
    this.select(mesh);
    // Lock the pinch offset relative to the mesh origin at the grabbed
    // surface point's height, plus the vertical (lift) anchor: the mesh may
    // never sink below the ground plane.
    // Floor clamp from the *world* bounds, so a tilted (fist-rotated) mesh
    // is clamped by its real lowest point.
    mesh.updateMatrixWorld(true);
    const minY = mesh.position.y - worldBottom(mesh);
    const drag = {
      offsetX: 0,
      offsetZ: 0,
      baseY: mesh.position.y,
      startDeviceY: y,
      minY,
      maxY: Math.max(minY, this.options.dragMaxHeight),
      lastX: x,
      lastY: y,
      grabbedAt: timestamp,
      moving: false,
      grabHeight: hits[0].point.y - mesh.position.y,
      liftStartY: null,
      toggleOnTap: wasSelected,
    };
    this.selectionDrag = drag;
    this.reanchorDrag(mesh, drag);
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
   *
   * Hold-to-move: until the pinch has been held for `dragHoldMs` the mesh
   * stays put (a quick pinch only selects); then the grabbed point follows
   * the fingertip exactly (it catches up with any drift during the hold).
   * @param timestamp current pinch time (ms); omit to skip the hold gate.
   */
  dragTo(x: number, y: number, timestamp = Infinity): void {
    const mesh = this.selected;
    const drag = this.selectionDrag;
    if (!mesh || !drag) return;
    drag.lastX = x;
    drag.lastY = y;
    if (!drag.moving) {
      // Still a possible tap: the mesh stays put. The grab anchor is kept
      // where the pinch landed, so once moving starts the grabbed point sits
      // exactly under the fingertip again.
      if (timestamp - drag.grabbedAt < this.options.dragHoldMs) return;
      drag.moving = true;
    }
    if (this.constraint === 'xz') {
      const target = this.planePoint(x, y, mesh.position.y + drag.grabHeight);
      mesh.position.x = target.x + drag.offsetX;
      mesh.position.z = target.z + drag.offsetZ;
    } else {
      // The grabbed point follows the fingertip's height on the lift plane;
      // fall back to scaled hand travel if the ray misses the plane.
      const hitY = this.liftHitY(x, y, mesh, drag);
      const lift =
        hitY !== null && drag.liftStartY !== null
          ? hitY - drag.liftStartY
          : (y - drag.startDeviceY) * this.options.dragElevationScale;
      mesh.position.y = THREE.MathUtils.clamp(drag.baseY + lift, drag.minY, drag.maxY);
    }
    this.syncSelectionOutline(); // the outline follows the dragged mesh exactly
    if (this.rotating) this.updateRotationRing();
    this.updateClash(); // the clash indicator tracks the dragged mesh live
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
    this.reanchorDrag(mesh, drag);
  }

  /**
   * Re-anchor the drag offsets at the mesh's live position and the newest
   * pinch position, so the next `dragTo` continues without a jump.
   */
  private reanchorDrag(
    mesh: THREE.Mesh,
    drag: NonNullable<CadBuilder['selectionDrag']>
  ): void {
    const grab = this.planePoint(drag.lastX, drag.lastY, mesh.position.y + drag.grabHeight);
    drag.offsetX = mesh.position.x - grab.x;
    drag.offsetZ = mesh.position.z - grab.z;
    drag.baseY = mesh.position.y;
    drag.startDeviceY = drag.lastY;
    drag.liftStartY = this.liftHitY(drag.lastX, drag.lastY, mesh, drag);
  }

  /**
   * Height where the pinch ray crosses the lift plane: the vertical plane
   * through the grabbed point, facing the camera (horizontally). `null`
   * when the ray runs parallel to it.
   */
  private liftHitY(
    x: number,
    y: number,
    mesh: THREE.Mesh,
    drag: NonNullable<CadBuilder['selectionDrag']>
  ): number | null {
    const camera = this.scene.interactionCamera;
    this.raycaster.setFromCamera(this.ndc.set(x, y), camera);
    const normal = camera.getWorldDirection(new THREE.Vector3()).setY(0);
    if (normal.lengthSq() < 1e-8) return null;
    normal.normalize();
    const through = new THREE.Vector3(
      mesh.position.x - drag.offsetX,
      mesh.position.y + drag.grabHeight,
      mesh.position.z - drag.offsetZ
    );
    this.dragPlane.setFromNormalAndCoplanarPoint(normal, through);
    const hit = this.raycaster.ray.intersectPlane(this.dragPlane, new THREE.Vector3());
    return hit ? hit.y : null;
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
    this.syncSelectionOutline(); // the outline follows the spun mesh exactly
    if (this.rotating) this.updateRotationRing();
    this.updateClash(); // a spun mesh may swing into (or out of) a clash
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

  /* ------------------------------------------------------------------ */
  /* CSG Boolean operations (SELECT mode)                               */
  /* ------------------------------------------------------------------ */

  /** The armed Boolean operation, or null when none is active. */
  get armedBooleanTool(): BooleanOperation | null {
    return this.booleanTool;
  }

  /**
   * Live Boolean state for the vision overlay (plain data): the armed tool
   * and whether the selection currently clashes with another committed
   * mesh — the armed overlay button glows amber while a clash is live.
   */
  get booleanState(): { tool: BooleanOperation | null; clash: boolean } {
    return { tool: this.booleanTool, clash: this.clashTarget !== null };
  }

  /**
   * Arm / disarm the CSG Boolean tool (mutually exclusive; null disarms).
   * Arming alone never mutates the scene — `applyBoolean` executes.
   */
  setBooleanTool(tool: BooleanOperation | null): void {
    this.booleanTool = tool;
  }

  /**
   * Recompute the selection's clash: the first committed mesh whose
   * world-space AABB strictly overlaps the selection's (the spec's
   * `box3.intersectsBox` test, with a drawable overlap volume). Refreshed
   * as the selection moves (drag / rotate) or the selection changes; the
   * translucent amber indicator tracks the overlap region while a clash
   * is live and hides otherwise.
   */
  private updateClash(): void {
    const selection = this.selected;
    if (!selection) {
      this.clashTarget = null;
      this.syncClashIndicator(null);
      return;
    }
    selection.updateMatrixWorld(true);
    worldBounds(selection, this.scratchBoxA);
    for (const mesh of this.committed) {
      if (mesh === selection) continue;
      mesh.updateMatrixWorld(true);
      worldBounds(mesh, this.scratchBoxB);
      if (boxesClash(this.scratchBoxA, this.scratchBoxB)) {
        this.clashTarget = mesh;
        this.syncClashIndicator(overlapBox(this.scratchBoxA, this.scratchBoxB));
        return;
      }
    }
    this.clashTarget = null;
    this.syncClashIndicator(null);
  }

  /**
   * Place / hide the clash indicator: a translucent amber box (+ crisp
   * outline) filling the two solids' world-AABB overlap. Lazily built and
   * reused — only its transform is refreshed per frame.
   */
  private syncClashIndicator(overlap: THREE.Box3 | null): void {
    const group = this.clashBox ?? this.createClashBox();
    if (!overlap) {
      group.visible = false;
      return;
    }
    const size = overlap.getSize(new THREE.Vector3());
    const center = overlap.getCenter(new THREE.Vector3());
    group.visible = true;
    group.position.copy(center);
    group.scale.set(Math.max(size.x, 1e-4), Math.max(size.y, 1e-4), Math.max(size.z, 1e-4));
  }

  /** Build the clash indicator: unit box + edge outline, hidden by default. */
  private createClashBox(): THREE.Group {
    const fill = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({
        color: 0xf59e0b, // amber clash
        transparent: true,
        opacity: 0.3,
        depthWrite: false,
      })
    );
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.9 })
    );
    const group = new THREE.Group();
    group.name = 'cad-clash-indicator';
    group.add(fill, outline);
    group.visible = false;
    this.clashBox = group;
    this.scene.scene.add(group);
    return group;
  }

  /**
   * Execute the armed (or explicit) Boolean operation on the current clash:
   *
   * - **subtract**: the *selected* mesh is the cutter — it is subtracted
   *   from the intersected base (`Evaluator.evaluate(base, cutter,
   *   SUBTRACTION)`), carving the cavity; the base is replaced by the result
   *   (fresh geometry + crisp `EdgesGeometry`) and the cutter is removed;
   * - **union**: both solids fuse (`ADDITION`) into one continuous body
   *   wearing the *selected* mesh's color; both are replaced by the result.
   *
   * The result keeps the base's geometry frame + transform (a drop-in
   * replacement in the committed scene graph), stays selected, and the
   * clash indicator refreshes. A no-op (false) without a clash, without a
   * selection, or when the CSG fails — the scene is never half-edited.
   *
   * @param operation override for the armed tool (the overlay's X-cross
   *        trigger always subtracts); defaults to the armed tool.
   * @returns true when an operation ran.
   */
  applyBoolean(operation?: BooleanOperation): boolean {
    const op = operation ?? this.booleanTool;
    if (!op) return false;
    const selection = this.selected;
    const target = this.clashTarget;
    if (!selection || !target || selection === target) {
      this.updateClash();
      return false;
    }
    // subtract: base = the intersected mesh, cutter = the selection.
    // union: base = the selection, tool = the other (each keeps its colors).
    const base = op === 'subtract' ? target : selection;
    const tool = op === 'subtract' ? selection : target;
    const material = op === 'subtract' ? firstMaterial(base).clone() : undefined;
    this.endDrag(); // the drag anchor died with the old meshes
    const result = evaluateBoolean(base, tool, op, material);
    if (!result) return false;
    if (op === 'union') {
      // Remember the operands (and where they sit relative to the union) so
      // the union can be ungrouped later — wherever it has moved / turned.
      result.updateMatrixWorld(true);
      base.updateMatrixWorld(true);
      tool.updateMatrixWorld(true);
      const toUnion = result.matrixWorld.clone().invert();
      const parts: UnionPart[] = [base, tool].map((mesh) => ({
        mesh,
        local: toUnion.clone().multiply(mesh.matrixWorld),
      }));
      result.userData.parts = parts;
    }

    // Swap the result in at the base's slot; both operands go. (Replace the
    // base first, then drop the tool — splicing first would shift the base's
    // index and leave the disposed base behind as an invisible, pickable,
    // exportable ghost.)
    this.root.remove(base, tool);
    const slot = this.committed.indexOf(base);
    if (slot >= 0) this.committed[slot] = result;
    else this.committed.push(result);
    const toolIndex = this.committed.indexOf(tool);
    if (toolIndex >= 0) this.committed.splice(toolIndex, 1);
    this.root.add(result);
    if (op === 'subtract') {
      disposeMesh(base);
      disposeMesh(tool);
    } // a union keeps its operands alive (detached) for ungrouping

    // The result is the new selection (outline + clash refresh in select()).
    this.selected = null;
    this.select(result);
    return true;
  }

  /** Whether the selection is a union that can be split back into its parts. */
  get canUngroup(): boolean {
    return partsOf(this.selected).length > 0;
  }

  /**
   * Ungroup the selected union: its original parts come back — each with
   * its own geometry and colors — placed where they now belong after any
   * move / rotation of the union (stored part-in-union transforms), and
   * the union mesh is removed. Nested unions split one level per call. The
   * first part becomes the selection.
   * @returns true when a union was ungrouped.
   */
  ungroupSelection(): boolean {
    const group = this.selected;
    const parts = partsOf(group);
    if (!group || parts.length === 0) return false;
    group.updateMatrixWorld(true);
    this.deselect();
    const restored = parts.map(({ mesh, local }) => {
      group.matrixWorld.clone().multiply(local).decompose(mesh.position, mesh.quaternion, mesh.scale);
      mesh.updateMatrixWorld(true);
      this.root.add(mesh);
      return mesh;
    });
    this.root.remove(group);
    const slot = this.committed.indexOf(group);
    if (slot >= 0) this.committed.splice(slot, 1, ...restored);
    else this.committed.push(...restored);
    group.userData.parts = []; // the parts live on — dispose only the union
    disposeMesh(group);
    this.select(restored[0]);
    return true;
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
   * format). The selection outline highlight is a separate shell, so the
   * repainted mesh stays visibly selected in its new color.
   * @returns true when the color was applied; false when nothing is selected
   * or the hex is malformed (the scene is left untouched).
   */
  setSelectedColor(hexColor: string): boolean {
    const mesh = this.selected;
    const hex = parseHexColor(hexColor);
    if (!mesh || hex === null) return false;
    paintMesh(mesh, hex);
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

  /**
   * Delete the active selection: remove the mesh from the committed scene
   * graph, dispose its geometry, material and edge overlays (GPU memory is
   * reclaimed), and clear every selection reference — drag anchor, rotation
   * gesture and highlight. A no-op when nothing is selected. The orchestrator
   * only reaches this after its confirmation dialog (destructive action).
   */
  deleteSelectedMesh(): void {
    const mesh = this.selected;
    if (!mesh) return;
    this.endDrag(); // drop any active drag anchor + running rotation
    this.deselect(); // restore the highlight state + clear selection refs
    this.root.remove(mesh);
    const index = this.committed.indexOf(mesh);
    if (index >= 0) this.committed.splice(index, 1);
    disposeMesh(mesh);
  }

  /** End the active selection drag (the selection itself persists). */
  endDrag(): void {
    this.selectionDrag = null;
    this.endRotateSelection();
  }

  /**
   * The picking pinch was released (EDIT mode). A quick pinch (a tap — it
   * never started moving) on a mesh that was *already* selected toggles the
   * selection off; any other release just ends the drag and keeps the
   * selection (a first tap selects, a hold-and-drag moves).
   */
  releasePick(): void {
    const drag = this.selectionDrag;
    if (drag && !drag.moving && drag.toggleOnTap) this.deselect();
    else this.endDrag();
  }

  /**
   * Fist rotation of the selection (EDIT mode), one axis at a time
   * (`AxisLock`): each fist gesture picks its axis from its dominant
   * direction and sticks to it — no free tumbling with every wobble:
   * - mostly **left / right** → spin about the vertical (world Y) axis — the
   *   side facing you follows the fist;
   * - mostly **up / down** → tip about the camera's horizontal (screen-X)
   *   axis — the side facing you tips up / down with the fist, from any
   *   camera angle.
   * The lock clears when the fist holds still briefly or the gesture ends
   * (`endFistRotation`).
   * Deltas are device units (`fistRotateSpeed` radians per unit; jitter
   * below `fistRotateDeadzone` per axis is dropped). The mesh turns about
   * its own center (world AABB center), and its lowest point keeps its
   * height (measured on the actual vertices), so it neither sinks into nor
   * creeps up off the floor as you turn it back and forth.
   */
  rotateSelectionBy(deltaX: number, deltaY: number): void {
    const mesh = this.selected;
    if (!mesh || !Number.isFinite(deltaX) || !Number.isFinite(deltaY)) return;
    const { fistRotateSpeed: speed, fistRotateDeadzone: deadzone } = this.options;
    const locked = this.fistAxisLock.update(deltaX, deltaY);
    const yaw = Math.abs(locked.deltaX) < deadzone ? 0 : locked.deltaX * speed;
    const pitch = Math.abs(locked.deltaY) < deadzone ? 0 : -locked.deltaY * speed;
    if (yaw === 0 && pitch === 0) return;

    mesh.updateMatrixWorld(true);
    const bottomBefore = worldBottom(mesh);
    const pivot = worldBounds(mesh, this.scratchBoxA).getCenter(new THREE.Vector3());
    // Screen-horizontal axis: the camera's right vector (horizontal, since
    // the orbit camera always keeps world-up).
    const camera = this.scene.camera;
    camera.updateMatrixWorld();
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).setY(0);
    if (right.lengthSq() < 1e-8) right.set(1, 0, 0);
    right.normalize();

    const turn = new THREE.Quaternion();
    if (yaw !== 0) turn.multiply(new THREE.Quaternion().setFromAxisAngle(WORLD_Y, yaw));
    if (pitch !== 0) turn.premultiply(new THREE.Quaternion().setFromAxisAngle(right, pitch));
    // Rotate about the pivot: orientation and position together.
    mesh.position.sub(pivot).applyQuaternion(turn).add(pivot);
    mesh.quaternion.premultiply(turn);
    mesh.updateMatrixWorld(true);

    // Keep the lowest point where it was (never below the floor).
    const lift = Math.max(0, bottomBefore) - worldBottom(mesh);
    if (lift !== 0) {
      mesh.position.y += lift;
      mesh.updateMatrixWorld(true);
    }
    this.syncSelectionOutline();
    if (this.rotating) this.updateRotationRing();
    this.updateClash();
  }

  /** The fist gesture ended: the next fist picks its rotation axis afresh. */
  endFistRotation(): void {
    this.fistAxisLock.reset();
  }

  /** Clear the selection and remove its outline highlight. */
  deselect(): void {
    this.fistAxisLock.reset(); // a new selection starts a fresh rotation axis
    this.hideSelectionOutline();
    this.selected = null;
    this.selectionDrag = null;
    this.endRotateSelection();
    this.updateClash(); // no selection: any clash indicator goes with it
  }

  private select(mesh: THREE.Mesh): void {
    if (this.selected === mesh) return;
    this.deselect();
    this.selected = mesh;
    this.showSelectionOutline(mesh);
    this.updateClash(); // a freshly picked mesh may already intersect
  }

  /**
   * Outline highlight: draw a distinct silhouette border around the selected
   * mesh instead of touching its base material or edge overlays (a color
   * picked on the wheel therefore repaints a *clean* mesh). The shell shares
   * the mesh's geometry with a `BackSide` accent material and is uniformly
   * scaled so its silhouette floats `SELECTION_OUTLINE_OFFSET` world units
   * beyond the mesh — equally visible on boxes, cylinders and spheres (an
   * `EdgesGeometry` outline would vanish on smooth spheres).
   *
   * Like the rotation ring, the shell lives in the scene (not the committed
   * root), so it never shows up in the STL export, the raycast pick or the
   * AR ghost list.
   */
  private showSelectionOutline(mesh: THREE.Mesh): void {
    const outline = this.ensureSelectionOutline();
    outline.geometry = mesh.geometry;
    outline.scale.setScalar(this.outlineScale(mesh));
    this.syncSelectionOutline();
    outline.visible = true;
  }

  /** Hide the outline shell (kept for reuse; the geometry is re-bound next select). */
  private hideSelectionOutline(): void {
    if (this.selectionOutline) this.selectionOutline.visible = false;
  }

  /** Copy the selected mesh's position / rotation onto the outline shell. */
  private syncSelectionOutline(): void {
    const mesh = this.selected;
    const outline = this.selectionOutline;
    if (!mesh || !outline) return;
    outline.position.copy(mesh.position);
    outline.quaternion.copy(mesh.quaternion);
  }

  /** Uniform scale that keeps `SELECTION_OUTLINE_OFFSET` world units of shell. */
  private outlineScale(mesh: THREE.Mesh): number {
    if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
    const radius = mesh.geometry.boundingSphere?.radius ?? 0.5;
    return radius > 1e-6 ? 1 + SELECTION_OUTLINE_OFFSET / radius : 1;
  }

  /**
   * Build the reusable outline shell once: an inverted hull (back faces only)
   * in the selection accent. Its geometry is re-bound per selection, so only
   * the material is owned here.
   */
  private ensureSelectionOutline(): THREE.Mesh {
    if (this.selectionOutline) return this.selectionOutline;
    const outline = new THREE.Mesh(
      new THREE.BufferGeometry(), // placeholder — re-bound on every select
      new THREE.MeshBasicMaterial({
        color: SELECTION_OUTLINE_COLOR,
        side: THREE.BackSide,
        fog: false,
      })
    );
    outline.name = 'cad-selection-outline';
    outline.visible = false;
    outline.frustumCulled = false; // mirrors the (possibly dragged) mesh exactly
    this.selectionOutline = outline;
    this.scene.scene.add(outline);
    return outline;
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
  /**
   * Raycast device coords onto the horizontal plane at `height` (the grabbed
   * point's height), so a dragged object's grabbed point stays under the
   * fingertip in depth as well as sideways. Clamped below the camera (a ray
   * must be able to reach it) and to the workspace radius.
   */
  private planePoint(x: number, y: number, height: number): THREE.Vector3 {
    if (height <= 1e-4) return this.groundPoint(x, y);
    const camera = this.scene.interactionCamera;
    this.raycaster.setFromCamera(this.ndc.set(x, y), camera);
    const ray = this.raycaster.ray;
    if (ray.direction.y > -0.05) {
      ray.direction.y = -0.05;
      ray.direction.normalize();
    }
    const planeHeight = Math.min(height, camera.position.y - 0.1);
    this.dragPlane.set(new THREE.Vector3(0, 1, 0), -planeHeight);
    const point = new THREE.Vector3();
    if (!ray.intersectPlane(this.dragPlane, point)) return this.groundPoint(x, y);
    const radius = Math.hypot(point.x, point.z);
    const max = this.options.groundRadius;
    if (radius > max) {
      point.x = (point.x / radius) * max;
      point.z = (point.z / radius) * max;
    }
    return point;
  }

  private groundPoint(x: number, y: number): THREE.Vector3 {
    // Hand coords are the webcam frame's device space = interaction-camera NDC.
    this.raycaster.setFromCamera(this.ndc.set(x, y), this.scene.interactionCamera);
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
  for (const material of materialsOf(mesh)) material.dispose();
  // A union's detached operands go with it.
  for (const part of partsOf(mesh)) disposeMesh(part.mesh);
  for (const child of mesh.children) {
    if (child instanceof THREE.LineSegments) {
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
  }
}

/** A union's operand, with its transform relative to the union mesh. */
interface UnionPart {
  mesh: THREE.Mesh;
  local: THREE.Matrix4;
}

/** The stored operands of a union mesh (empty for any other mesh). */
function partsOf(mesh: THREE.Mesh | null): UnionPart[] {
  const parts = mesh?.userData.parts as UnionPart[] | undefined;
  return Array.isArray(parts) ? parts : [];
}

/** A mesh's material(s) as a list (unions carry one per operand). */
function materialsOf(mesh: THREE.Mesh): THREE.Material[] {
  return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

/** The first (or only) material of a mesh. */
function firstMaterial(mesh: THREE.Mesh): THREE.Material {
  return materialsOf(mesh)[0];
}

/**
 * Repaint every material of a mesh — and, for a union, its stored parts
 * too, so a later ungroup gives the parts the color the whole was painted.
 */
function paintMesh(mesh: THREE.Mesh, hex: number): void {
  for (const material of materialsOf(mesh)) {
    (material as THREE.MeshStandardMaterial).color?.setHex(hex);
  }
  for (const part of partsOf(mesh)) paintMesh(part.mesh, hex);
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
