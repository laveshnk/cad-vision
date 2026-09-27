import { describe, expect, it } from 'vitest';
import { CM_PER_UNIT, formatDimension, solidDimensions } from '../dimensions';

describe('solidDimensions', () => {
  it('reports L / W / H for boxes (X, Z, Y) in centimetres', () => {
    expect(solidDimensions({ kind: 'box', width: 1.2, height: 0.5, depth: 0.8 })).toEqual([
      { label: 'L', cm: 1.2 * CM_PER_UNIT },
      { label: 'W', cm: 0.8 * CM_PER_UNIT },
      { label: 'H', cm: 0.5 * CM_PER_UNIT },
    ]);
  });

  it('reports radius + height for cylinders and radius only for spheres', () => {
    expect(solidDimensions({ kind: 'cylinder', radius: 0.4, height: 2 }, 10)).toEqual([
      { label: 'R', cm: 4 },
      { label: 'H', cm: 20 },
    ]);
    expect(solidDimensions({ kind: 'sphere', radius: 0.75 }, 10)).toEqual([{ label: 'R', cm: 7.5 }]);
  });

  it('never reports negative sizes', () => {
    const [length] = solidDimensions({ kind: 'box', width: -2, height: 1, depth: 1 }, 10);
    expect(length.cm).toBe(20);
  });
});

describe('formatDimension', () => {
  it('prints one decimal with the cm unit', () => {
    expect(formatDimension({ label: 'H', cm: 12.345 })).toBe('H 12.3 cm');
    expect(formatDimension({ label: 'R', cm: 5 })).toBe('R 5.0 cm');
  });
});
