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
 */

import * as THREE from 'three';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import type { SceneObjectSummary, ShapeId } from '../../shared/agentTools';
import type { CadScene } from './CadScene';

/** Same four primitives the voice agent knows about. */
export type CadTool = ShapeId;

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
  /** Device-space drag distance that re-purposes a pinch as a new drawing. */
  redrawDistance?: number;
  /** Workspace radius on the ground plane for raycast clamping. */
  groundRadius?: number;
  /** World units of lift per device-space unit of vertical drag (SELECT mode). */
  dragElevationScale?: number;
  /** Highest allowed mesh center height while drag-lifting (world units). */
  dragMaxHeight?: number;
  /**
   * SELECT mode: a pinch must be held this long (ms) before it starts moving
   * the picked mesh, so a quick pinch only selects it. Default 300.
   */
  dragHoldMs?: number;
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
  private readonly committed: CommittedObject[] = [];

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
  } | null = null;
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
      dragHoldMs: options.dragHoldMs ?? 300,
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
    for (const object of this.committed) {
      this.root.remove(object.mesh);
      disposeMesh(object.mesh);
    }
    this.committed.length = 0;
    this.selected = null;
    this.selectionDrag = null;
    this.hideSelectionOutline();
    this.endRotateSelection();
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
    this.select(mesh);
    // Lock the pinch offset relative to the mesh origin at the grabbed
    // surface point's height, plus the vertical (lift) anchor: the mesh may
    // never sink below the ground plane.
    const bottom = mesh.geometry.boundingBox?.min.y ?? 0;
    const minY = -bottom;
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
   * format). The selection outline highlight is a separate shell, so the
   * repainted mesh stays visibly selected in its new color.
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

  /** Clear the selection and remove its outline highlight. */
  deselect(): void {
    this.hideSelectionOutline();
    this.selected = null;
    this.selectionDrag = null;
    this.endRotateSelection();
  }

  private select(mesh: THREE.Mesh): void {
    if (this.selected === mesh) return;
    this.deselect();
    this.selected = mesh;
    this.showSelectionOutline(mesh);
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
