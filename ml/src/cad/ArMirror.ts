/**
 * ArMirror: plain-data AR projection of the CAD scene onto the 2D vision
 * overlay canvas (SELECT-mode spatial mirror).
 *
 * `buildArSceneFrame` projects the ground grid (Y = 0) and every committed
 * solid through the live Three.js camera's webcam-aspect twin
 * (`CadScene.interactionCamera` + `projectToCanvas`) into 2D canvas space —
 * the canvas passed in is the webcam image's on-screen rect, so proportions
 * are true (a sphere stays round) and ghosts align with the hands:
 *
 * - the grid becomes semi-transparent cyan polylines matching the 3D
 *   perspective — a 16 × 16 world-unit floor window centered on the origin
 *   by default, drawn as faint 2-unit minor lines plus stronger major
 *   divisions (every `gridMajorStep` units) so the extended floor stays
 *   readable on the video thumbnail;
 * - every committed mesh becomes a "ghost": its projected silhouette (convex
 *   hull of the projected vertices — exact for the convex primitives the
 *   builder emits) plus its crisp `EdgesGeometry` segments;
 * - the active selection is flagged so the overlay can render it with an
 *   energetic highlight, and while an open-palm rotation gesture is running
 *   the builder's compass ring (circle + yaw needle) is projected too.
 *
 * Culling keeps the per-frame cost flat at 60 FPS: points behind the camera
 * are rejected cheaply in view space (positive view-space z) *before* the
 * NDC projection and canvas mapping run, projected geometry past NDC depth
 * `z > 1` is dropped, and segments are additionally trivial-rejected
 * (`isSegmentOnCanvas`) when both endpoints fall into the same outside
 * half-plane of the canvas — no wasted `lineTo` operations for off-screen
 * grid lines.
 *
 * The returned payload is plain data (no Three.js types), structurally
 * compatible with `DebugOverlay`'s `arScene` provider types — the same
 * pattern `CadExtrudeInput` uses — so `main.ts` can forward it directly
 * without breaking the vision ↔ CAD decoupling. Runs inside the overlay's
 * rAF-driven per-frame render, so it stays allocation-light: scratch
 * vectors are reused (safe — `projectToCanvas` clones its input) and only
 * the plain payload objects are allocated.
 */

import * as THREE from 'three';
import type { CadBuilder } from './CadBuilder';
import type { CadScene } from './CadScene';
import { convexHull2D, isBehindCamera, isSegmentOnCanvas, type Point2D } from './arProjection';

/** A projected 2D canvas point (CSS px). */
export type ArPoint = Point2D;

/** One projected mesh edge segment (endpoints in 2D canvas space). */
export type ArSegment = [ArPoint, ArPoint];

/** One projected ground-grid line (straight 3D lines stay straight). */
export interface ArGridLine {
  points: ArPoint[];
  /** Major division (every `gridMajorStep` units) — rendered stronger. */
  major: boolean;
}

/** Translucent ghost of one committed mesh, projected to canvas space. */
export interface ArMeshGhost {
  /** Projected silhouette: convex hull of the mesh's projected vertices. */
  hull: ArPoint[];
  /** Projected `EdgesGeometry` segments (crisp accent lines). */
  edges: ArSegment[];
  /** Whether this mesh is the active selection (energetic highlight). */
  selected: boolean;
}

/**
 * Rotational compass ring around the selected object while an open-palm
 * rotation gesture is running (SELECT mode).
 */
export interface ArRotationRing {
  /** Projected circle segments (both endpoints visible). */
  circle: ArSegment[];
  /** Projected yaw indicator: object center → ring edge (null when culled). */
  needle: ArSegment | null;
}

/** One SELECT-mode AR frame: projected ground grid + mesh ghosts. */
export interface ArSceneFrame {
  /** Projected ground-grid lines, minor and major divisions. */
  grid: ArGridLine[];
  /** Ghosts of the committed meshes, in scene order. */
  meshes: ArMeshGhost[];
  /** Compass ring around the selection while rotating, or null. */
  rotationRing: ArRotationRing | null;
}

export interface ArMirrorOptions {
  /**
   * Half extent of the mirrored ground grid. Clamped to ≤ 8 → at most a
   * 16 × 16 world-unit floor: the grid is the AR mirror's biggest per-frame
   * projection cost, so its extent is capped on integrated GPUs' behalf.
   */
  gridHalfExtent?: number;
  /**
   * Minor ground-grid line spacing (world units). Clamped to ≥ 2 — 1-unit
   * lines quadruple the projected segment count for no extra readability.
   */
  gridStep?: number;
  /** Major (stronger) grid division every this many world units. */
  gridMajorStep?: number;
}

/** Scratch segment endpoints — reused across lines, meshes and frames. */
const segmentFrom = new THREE.Vector3();
const segmentTo = new THREE.Vector3();
/** Scratch vertices for silhouette / edge endpoints. */
const vertexA = new THREE.Vector3();
const vertexB = new THREE.Vector3();
/** Scratch point for the view-space behind-camera pre-check. */
const viewSpacePoint = new THREE.Vector3();

/** Grid extent cap: at most a 16 × 16 world-unit floor window. */
const MAX_GRID_HALF_EXTENT = 8;
/** Grid density floor: at least 2 world units between minor grid lines. */
const MIN_GRID_STEP = 2;

/**
 * Build one SELECT-mode AR frame: the ground grid + every committed mesh,
 * projected through the live 3D camera onto a `canvasWidth × canvasHeight`
 * (CSS px) canvas. Called once per rendered overlay frame.
 */
export function buildArSceneFrame(
  scene: CadScene,
  builder: CadBuilder,
  canvasWidth: number,
  canvasHeight: number,
  options: ArMirrorOptions = {}
): ArSceneFrame {
  // Project through the webcam-aspect interaction camera (synced once per
  // frame): ghosts keep their true proportions on the camera view and line
  // up with the hands, whatever the wide viewport's aspect is.
  const camera = scene.interactionCamera;
  // Grid density is clamped for integrated-GPU performance (see
  // `ArMirrorOptions`): at most a 16 × 16 world-unit floor with 2-unit minor
  // lines — roughly a quarter of the previous per-frame segment count.
  const halfExtent = Math.min(
    options.gridHalfExtent ?? MAX_GRID_HALF_EXTENT,
    MAX_GRID_HALF_EXTENT
  );
  const step = Math.max(options.gridStep ?? MIN_GRID_STEP, MIN_GRID_STEP);
  const majorStep = options.gridMajorStep ?? 4;
  const grid: ArGridLine[] = [];
  const cells = Math.max(1, Math.round(halfExtent / step));
  // A major division every `majorEvery` minor lines (index math keeps this
  // exact for negative line indices too: -10 % 5 === -0 === 0).
  const majorEvery = Math.max(1, Math.round(majorStep / step));
  for (let i = -cells; i <= cells; i++) {
    const at = i * step;
    const major = i % majorEvery === 0;
    // Lines along X (fixed Z) and along Z (fixed X) on the Y = 0 plane,
    // dropped wholesale when either endpoint falls behind the camera or the
    // whole segment lies outside the canvas bounds.
    const alongX = projectSegment(
      scene,
      camera,
      segmentFrom.set(-halfExtent, 0, at),
      segmentTo.set(halfExtent, 0, at),
      canvasWidth,
      canvasHeight
    );
    if (alongX) grid.push({ points: alongX, major });
    const alongZ = projectSegment(
      scene,
      camera,
      segmentFrom.set(at, 0, -halfExtent),
      segmentTo.set(at, 0, halfExtent),
      canvasWidth,
      canvasHeight
    );
    if (alongZ) grid.push({ points: alongZ, major });
  }

  const selected = builder.selectedMesh;
  const meshes: ArMeshGhost[] = builder.committedMeshes.map((mesh) =>
    meshGhost(scene, camera, mesh, mesh === selected, canvasWidth, canvasHeight)
  );
  const rotationRing = projectRotationRing(scene, camera, builder, canvasWidth, canvasHeight);
  return { grid, meshes, rotationRing };
}

/**
 * Cheap behind-camera pre-check in view space: the camera looks down its
 * local -Z, so a point with positive view-space z lies behind it. Testing
 * this first lets culling skip the full projection (vector clone + view and
 * projection matrix multiplies + canvas mapping) for points the NDC `z > 1`
 * test would reject anyway; that test still runs after projecting and keeps
 * catching the rarer beyond-far-plane case, so culling behavior is
 * unchanged. The camera's world matrix must be synced —
 * `CadScene.interactionCamera` guarantees that once per frame.
 */
function isPointBehindCamera(
  worldPoint: THREE.Vector3,
  camera: THREE.PerspectiveCamera
): boolean {
  return viewSpacePoint.copy(worldPoint).applyMatrix4(camera.matrixWorldInverse).z > 0;
}

/**
 * Project a straight world-space segment; `null` when culled — either
 * endpoint behind the camera, past NDC depth 1, or both endpoints in the
 * same outside half-plane of the canvas (trivial reject).
 */
function projectSegment(
  scene: CadScene,
  camera: THREE.PerspectiveCamera,
  a: THREE.Vector3,
  b: THREE.Vector3,
  canvasWidth: number,
  canvasHeight: number
): ArSegment | null {
  // View-space pre-check: either endpoint behind the camera culls the whole
  // segment, so skip both projections entirely.
  if (isPointBehindCamera(a, camera) || isPointBehindCamera(b, camera)) return null;
  const pa = scene.projectToCanvas(a, canvasWidth, canvasHeight, camera);
  const pb = scene.projectToCanvas(b, canvasWidth, canvasHeight, camera);
  if (isBehindCamera(pa.z) || isBehindCamera(pb.z)) return null;
  const from = { x: pa.x, y: pa.y };
  const to = { x: pb.x, y: pb.y };
  if (!isSegmentOnCanvas(from, to, canvasWidth, canvasHeight)) return null;
  return [from, to];
}

/**
 * Project the selection's rotational compass ring (null unless an open-palm
 * rotation gesture is running): a 64-segment circle around the object plus
 * the yaw needle along the object's local +X direction (three.js maps a
 * `rotation.y = θ` local +X axis to the world direction `(cos θ, 0, -sin θ)`).
 */
function projectRotationRing(
  scene: CadScene,
  camera: THREE.PerspectiveCamera,
  builder: CadBuilder,
  canvasWidth: number,
  canvasHeight: number
): ArRotationRing | null {
  const ring = builder.selectionRingWorld;
  if (!ring) return null;
  const segments = 64;
  const circle: ArSegment[] = [];
  for (let i = 0; i < segments; i++) {
    const a0 = (i / segments) * Math.PI * 2;
    const a1 = ((i + 1) / segments) * Math.PI * 2;
    const segment = projectSegment(
      scene,
      camera,
      segmentFrom.set(
        ring.center.x + Math.cos(a0) * ring.radius,
        ring.center.y,
        ring.center.z + Math.sin(a0) * ring.radius
      ),
      segmentTo.set(
        ring.center.x + Math.cos(a1) * ring.radius,
        ring.center.y,
        ring.center.z + Math.sin(a1) * ring.radius
      ),
      canvasWidth,
      canvasHeight
    );
    if (segment) circle.push(segment);
  }
  const needle = projectSegment(
    scene,
    camera,
    segmentFrom.set(ring.center.x, ring.center.y, ring.center.z),
    segmentTo.set(
      ring.center.x + Math.cos(ring.angle) * ring.radius,
      ring.center.y,
      ring.center.z - Math.sin(ring.angle) * ring.radius
    ),
    canvasWidth,
    canvasHeight
  );
  return { circle, needle };
}

/**
 * Project one committed mesh into a ghost: silhouette hull + edge segments.
 * `updateMatrixWorld(true)` guarantees a just-dragged position is reflected
 * immediately (the WebGL render for this frame may not have run yet — the
 * AR ghost and the pinch drag stay in lockstep with zero latency).
 */
function meshGhost(
  scene: CadScene,
  camera: THREE.PerspectiveCamera,
  mesh: THREE.Mesh,
  selected: boolean,
  canvasWidth: number,
  canvasHeight: number
): ArMeshGhost {
  mesh.updateMatrixWorld(true);

  // Silhouette: project every vertex, cull the ones behind the camera, then
  // take the convex hull of the survivors (the projected 2D footprint).
  const silhouette: ArPoint[] = [];
  const position = mesh.geometry.getAttribute('position');
  if (position) {
    for (let i = 0; i < position.count; i++) {
      vertexA.fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld);
      // View-space pre-check: skip the projection for vertices behind the
      // camera — they would be culled right after projecting anyway.
      if (isPointBehindCamera(vertexA, camera)) continue;
      const projected = scene.projectToCanvas(vertexA, canvasWidth, canvasHeight, camera);
      if (isBehindCamera(projected.z)) continue;
      silhouette.push({ x: projected.x, y: projected.y });
    }
  }

  // Crisp accent edges from the committed EdgesGeometry overlays
  // (LineSegments children, positions in pairs), culled per segment.
  const edges: ArSegment[] = [];
  for (const child of mesh.children) {
    if (!(child instanceof THREE.LineSegments)) continue;
    const edgePosition = child.geometry.getAttribute('position');
    if (!edgePosition) continue;
    for (let i = 0; i + 1 < edgePosition.count; i += 2) {
      vertexA.fromBufferAttribute(edgePosition, i).applyMatrix4(child.matrixWorld);
      vertexB.fromBufferAttribute(edgePosition, i + 1).applyMatrix4(child.matrixWorld);
      const segment = projectSegment(scene, camera, vertexA, vertexB, canvasWidth, canvasHeight);
      if (segment) edges.push(segment);
    }
  }

  return { hull: convexHull2D(silhouette), edges, selected };
}

