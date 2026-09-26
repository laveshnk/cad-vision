/**
 * arProjection: pure 2D projection math for the SELECT-mode AR mirror.
 *
 * Shared by `CadScene.projectToCanvas` and the AR mirror builder
 * (`ArMirror.ts`). Deliberately free of Three.js imports so the mapping,
 * culling and hull logic stay unit-testable in isolation (pure logic only —
 * see `__tests__/arProjection.test.ts`).
 */

/** A plain 2D point in canvas space (CSS px, +Y down). */
export interface Point2D {
  x: number;
  y: number;
}

/** A world point projected through a camera into 2D canvas space. */
export interface ProjectedPoint {
  /** Canvas-space X (CSS px, 0 = left edge). */
  x: number;
  /** Canvas-space Y (CSS px, 0 = top edge, +Y down). */
  y: number;
  /** NDC depth: `> 1` means the point lies behind the camera — cull it. */
  z: number;
}

/**
 * Map normalized device coordinates [-1, 1] (three.js NDC convention, +Y up)
 * onto a 2D canvas of the given size (CSS px, +Y down):
 *
 *   screenX = (ndcX * 0.5 + 0.5) * canvasWidth
 *   screenY = (-ndcY * 0.5 + 0.5) * canvasHeight
 *
 * The NDC depth passes through untouched for behind-camera culling
 * (`isBehindCamera`).
 */
export function ndcToCanvas(
  ndcX: number,
  ndcY: number,
  ndcZ: number,
  canvasWidth: number,
  canvasHeight: number
): ProjectedPoint {
  return {
    x: (ndcX * 0.5 + 0.5) * canvasWidth,
    y: (-ndcY * 0.5 + 0.5) * canvasHeight,
    z: ndcZ,
  };
}

/**
 * Map a 2D canvas point (CSS px, +Y down) back onto normalized device
 * coordinates [-1, 1] (+Y up) — the exact inverse of `ndcToCanvas`:
 *
 *   ndcX = screenX / canvasWidth * 2 - 1
 *   ndcY = (1 - screenY / canvasHeight) * 2 - 1
 *
 * Used to inverse-map viewport-local UI geometry (the SELECT-mode color wheel
 * and Delete HUD) back into device space, so the camera thumbnail can mirror
 * exactly where a fingertip must point.
 */
export function canvasToNdc(
  screenX: number,
  screenY: number,
  canvasWidth: number,
  canvasHeight: number
): Point2D {
  if (canvasWidth <= 0 || canvasHeight <= 0) return { x: 0, y: 0 };
  return {
    x: (screenX / canvasWidth) * 2 - 1,
    y: (1 - screenY / canvasHeight) * 2 - 1,
  };
}

/**
 * Re-express an NDC x coordinate from one perspective frustum in another that
 * shares the same camera pose and vertical field of view but has a different
 * aspect (width / height). With the vertical extent fixed, horizontal NDC
 * scales by the aspect ratio: `ndcX_to = ndcX_from * fromAspect / toAspect`
 * (NDC y is unchanged). Used to map the webcam-frame interaction frustum onto
 * the wider 3D viewport.
 */
export function remapNdcX(ndcX: number, fromAspect: number, toAspect: number): number {
  return toAspect > 0 ? (ndcX * fromAspect) / toAspect : ndcX;
}

/**
 * Behind-camera cull. After the perspective divide, points behind the camera
 * project to an NDC depth beyond the far bound (`z > 1`); drawing them would
 * smear geometry across the canvas, so such points / segments are skipped.
 */
export function isBehindCamera(ndcZ: number): boolean {
  return ndcZ > 1;
}

/**
 * Canvas-bounds cull for a projected segment (Cohen–Sutherland trivial
 * reject): a segment can only be invisible when *both* endpoints lie in the
 * same outside half-plane — both left of, right of, above or below the
 * canvas. A segment with one endpoint inside, or a long line that merely
 * crosses the canvas with both endpoints outside, stays visible, so culling
 * never punches holes into projected geometry (e.g. ground-grid lines
 * spanning the whole floor). Endpoints exactly on the border count as
 * inside.
 */
export function isSegmentOnCanvas(
  a: Point2D,
  b: Point2D,
  canvasWidth: number,
  canvasHeight: number
): boolean {
  const shared = outcode(a, canvasWidth, canvasHeight) & outcode(b, canvasWidth, canvasHeight);
  return shared === 0;
}

/** Cohen–Sutherland outcode: bit flags for the outside half-planes of a point. */
function outcode(p: Point2D, canvasWidth: number, canvasHeight: number): number {
  let code = 0;
  if (p.x < 0) code |= 1; // left
  else if (p.x > canvasWidth) code |= 2; // right
  if (p.y < 0) code |= 4; // above
  else if (p.y > canvasHeight) code |= 8; // below
  return code;
}

/**
 * Convex hull of a set of 2D points (Andrew's monotone chain), returned in
 * counter-clockwise order. Used as the projected silhouette ("2D footprint")
 * of a solid: every primitive the builder commits is convex, so
 * hull(projected vertices) === projected outline. Collinear and duplicate
 * points are dropped; fewer than 3 input points return a defensive copy
 * (degenerate — nothing to fill).
 */
export function convexHull2D(points: readonly Point2D[]): Point2D[] {
  if (points.length < 3) return [...points];
  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o: Point2D, a: Point2D, b: Point2D): number =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point2D[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point2D[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}
