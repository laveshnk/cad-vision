import { describe, expect, it } from 'vitest';
import {
  buildLandmark,
  distance3,
  midpoint3,
  rawToDeviceX,
  rawToDeviceY,
  subtract3,
  toDeviceSpace,
} from '../coordinates';

describe('rawToDeviceX', () => {
  it('mirrors X into [-1, 1]', () => {
    expect(rawToDeviceX(0)).toBe(1); // left edge of raw frame -> right of screen
    expect(rawToDeviceX(1)).toBe(-1); // right edge of raw frame -> left of screen
    expect(rawToDeviceX(0.5)).toBe(0);
    expect(rawToDeviceX(0.75)).toBeCloseTo(-0.5);
  });
});

describe('rawToDeviceY', () => {
  it('maps Y up into [-1, 1]', () => {
    expect(rawToDeviceY(0)).toBe(1); // top of image -> +1
    expect(rawToDeviceY(1)).toBe(-1); // bottom of image -> -1
    expect(rawToDeviceY(0.5)).toBeCloseTo(0);
    expect(rawToDeviceY(0.25)).toBeCloseTo(0.5);
  });
});

describe('toDeviceSpace', () => {
  it('applies the mirrored + Y-up mapping', () => {
    const p = toDeviceSpace({ x: 0.25, y: 0.25, z: 0.3 });
    expect(p.x).toBeCloseTo(0.5);
    expect(p.y).toBeCloseTo(0.5);
    expect(p.z).toBe(0.3);
  });
});

describe('buildLandmark', () => {
  it('retains normalized, pixel and device representations', () => {
    const lm = buildLandmark({ x: 0.25, y: 0.75, z: 0 }, 1000, 500);
    expect(lm.normalized).toEqual({ x: 0.25, y: 0.75, z: 0 });
    expect(lm.pixel).toEqual({ x: 250, y: 375 });
    expect(lm.device.x).toBeCloseTo(0.5);
    expect(lm.device.y).toBeCloseTo(-0.5);
  });
});

describe('vec3 math', () => {
  it('distance3 is Euclidean', () => {
    expect(distance3({ x: 0, y: 0, z: 0 }, { x: 3, y: 4, z: 0 })).toBe(5);
    expect(distance3({ x: 1, y: 1, z: 1 }, { x: 1, y: 1, z: 1 })).toBe(0);
  });

  it('midpoint3 averages components', () => {
    expect(midpoint3({ x: 0, y: 2, z: 4 }, { x: 2, y: 4, z: 6 })).toEqual({ x: 1, y: 3, z: 5 });
  });

  it('subtract3 computes a - b', () => {
    expect(subtract3({ x: 5, y: 5, z: 5 }, { x: 2, y: 3, z: 1 })).toEqual({ x: 3, y: 2, z: 4 });
  });
});
