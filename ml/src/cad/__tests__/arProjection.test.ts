import { describe, expect, it } from 'vitest';
import { convexHull2D, isBehindCamera, isSegmentOnCanvas, ndcToCanvas, remapNdcX } from '../arProjection';

describe('ndcToCanvas', () => {
  it('maps NDC corners to the canvas corners (Y axis flipped)', () => {
    expect(ndcToCanvas(-1, -1, 0, 300, 200)).toEqual({ x: 0, y: 200, z: 0 });
    expect(ndcToCanvas(1, 1, 0, 300, 200)).toEqual({ x: 300, y: 0, z: 0 });
  });

  it('maps the NDC origin to the canvas center and passes depth through', () => {
    expect(ndcToCanvas(0, 0, 0.5, 400, 240)).toEqual({ x: 200, y: 120, z: 0.5 });
    expect(ndcToCanvas(0, 0, 1.4, 400, 240).z).toBe(1.4);
  });

  it('scales linearly across the canvas', () => {
    expect(ndcToCanvas(0.5, -0.25, 0, 200, 100)).toEqual({ x: 150, y: 62.5, z: 0 });
  });
});

describe('isBehindCamera', () => {
  it('culls exactly the region beyond NDC depth 1 (behind the camera)', () => {
    expect(isBehindCamera(1.0001)).toBe(true);
    expect(isBehindCamera(2)).toBe(true);
    expect(isBehindCamera(1)).toBe(false);
    expect(isBehindCamera(0.37)).toBe(false);
    expect(isBehindCamera(-1)).toBe(false);
  });
});

describe('isSegmentOnCanvas', () => {
  const w = 100;
  const h = 80;

  it('keeps segments with both endpoints inside (borders count as inside)', () => {
    expect(isSegmentOnCanvas({ x: 10, y: 10 }, { x: 90, y: 70 }, w, h)).toBe(true);
    expect(isSegmentOnCanvas({ x: 0, y: 0 }, { x: 100, y: 80 }, w, h)).toBe(true);
  });

  it('keeps a segment with one endpoint outside (it may cross the canvas)', () => {
    expect(isSegmentOnCanvas({ x: -20, y: 40 }, { x: 50, y: 40 }, w, h)).toBe(true);
    expect(isSegmentOnCanvas({ x: 50, y: -10 }, { x: 50, y: 90 }, w, h)).toBe(true);
  });

  it('keeps a long line crossing the canvas with both endpoints outside', () => {
    expect(isSegmentOnCanvas({ x: -50, y: 40 }, { x: 150, y: 40 }, w, h)).toBe(true);
    expect(isSegmentOnCanvas({ x: -50, y: -50 }, { x: 150, y: 130 }, w, h)).toBe(true);
  });

  it('trivially rejects segments fully outside one half-plane', () => {
    expect(isSegmentOnCanvas({ x: -30, y: 10 }, { x: -5, y: 70 }, w, h)).toBe(false); // both left
    expect(isSegmentOnCanvas({ x: 120, y: 10 }, { x: 105, y: 70 }, w, h)).toBe(false); // both right
    expect(isSegmentOnCanvas({ x: 10, y: -30 }, { x: 90, y: -5 }, w, h)).toBe(false); // both above
    expect(isSegmentOnCanvas({ x: 10, y: 90 }, { x: 90, y: 200 }, w, h)).toBe(false); // both below
  });
});

describe('convexHull2D', () => {
  it('wraps a square counter-clockwise, dropping interior + collinear points', () => {
    const hull = convexHull2D([
      { x: 0, y: 0 },
      { x: 4, y: 0 },
      { x: 4, y: 4 },
      { x: 0, y: 4 },
      { x: 2, y: 2 }, // interior
      { x: 2, y: 0 }, // collinear on the bottom edge
    ]);
    expect(hull).toEqual([
      { x: 0, y: 0 },
      { x: 4, y: 0 },
      { x: 4, y: 4 },
      { x: 0, y: 4 },
    ]);
  });

  it('collapses duplicate points', () => {
    const hull = convexHull2D([
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 3, y: 0 },
      { x: 3, y: 3 },
      { x: 0, y: 3 },
    ]);
    expect(hull).toHaveLength(4);
  });

  it('returns degenerate inputs (< 3 points) as a defensive copy', () => {
    const points = [
      { x: 1, y: 2 },
      { x: 3, y: 4 },
    ];
    expect(convexHull2D(points)).toEqual(points);
    expect(convexHull2D(points)).not.toBe(points);
    expect(convexHull2D([])).toEqual([]);
  });

  it('degrades a collinear cloud to its endpoints (nothing to fill)', () => {
    const hull = convexHull2D([
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
      { x: 3, y: 3 },
    ]);
    expect(hull).toEqual([
      { x: 0, y: 0 },
      { x: 3, y: 3 },
    ]);
  });
});

describe('remapNdcX (webcam-frame frustum → wider viewport frustum)', () => {
  it('keeps the center and scales horizontal NDC by the aspect ratio', () => {
    // 4:3 webcam frame shown on a 2:1 viewport: its edges land at ±(4/3)/2.
    expect(remapNdcX(0, 4 / 3, 2)).toBe(0);
    expect(remapNdcX(1, 4 / 3, 2)).toBeCloseTo(2 / 3);
    expect(remapNdcX(-0.5, 4 / 3, 2)).toBeCloseTo(-1 / 3);
  });

  it('is the identity for equal aspects and round-trips', () => {
    expect(remapNdcX(0.37, 1.5, 1.5)).toBeCloseTo(0.37);
    expect(remapNdcX(remapNdcX(0.37, 4 / 3, 2.1), 2.1, 4 / 3)).toBeCloseTo(0.37);
  });

  it('leaves the value untouched for a degenerate target aspect', () => {
    expect(remapNdcX(0.5, 4 / 3, 0)).toBe(0.5);
  });
});
