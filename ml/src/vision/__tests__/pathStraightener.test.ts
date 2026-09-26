import { describe, expect, it } from 'vitest';
import { PathStraightener } from '../PathStraightener';
import type { Vec2 } from '../types';

/**
 * Shaky hand path from `from` to `to` in `steps` frames: straight progress
 * plus a sideways sine wiggle of `wiggle` amplitude (deterministic).
 */
function shakyLeg(from: Vec2, to: Vec2, steps: number, wiggle: number, phase = 0): Vec2[] {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  const nx = -dy / length;
  const ny = dx / length;
  const points: Vec2[] = [];
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    // Wiggle fades out at the leg's end so the hand actually arrives at `to`.
    const w = wiggle * Math.sin(i * 1.7 + phase) * Math.sin(Math.PI * t);
    points.push({ x: from.x + dx * t + nx * w, y: from.y + dy * t + ny * w });
  }
  return points;
}

/** Distance from `p` to the infinite line through `a` and `b`. */
function distanceToLine(p: Vec2, a: Vec2, b: Vec2): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.abs((p.x - a.x) * dy - (p.y - a.y) * dx) / Math.hypot(dx, dy);
}

function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

describe('PathStraightener', () => {
  it('turns a shaky A → B → C (back along the same line) into a straight path', () => {
    // The (5, 7) → (10, 12) → (8, 10) example, scaled to hand units.
    const A = { x: 0.1, y: 0.14 };
    const B = { x: 0.2, y: 0.24 };
    const C = { x: 0.16, y: 0.2 };
    const straightener = new PathStraightener();
    straightener.reset(A);

    const outAB = shakyLeg(A, B, 20, 0.012).map((p) => straightener.update(p));
    const outBC = shakyLeg(B, C, 10, 0.012, 1).map((p) => straightener.update(p));

    // The raw hand wiggles ~0.012 off the line; the output stays on it.
    for (const p of [...outAB, ...outBC]) expect(distanceToLine(p, A, B)).toBeLessThan(0.002);
    expect(distance(outAB[outAB.length - 1], B)).toBeLessThan(0.01);
    expect(distance(outBC[outBC.length - 1], C)).toBeLessThan(0.01);
  });

  it('makes a clean corner: A → B then a real turn toward D', () => {
    const A = { x: 0, y: 0 };
    const B = { x: 0.3, y: 0 };
    const D = { x: 0.3, y: 0.3 };
    const straightener = new PathStraightener();
    straightener.reset(A);

    const outAB = shakyLeg(A, B, 20, 0.01).map((p) => straightener.update(p));
    const outBD = shakyLeg(B, D, 20, 0.01, 2).map((p) => straightener.update(p));

    // First leg: straight along the X axis.
    for (const p of outAB) expect(Math.abs(p.y)).toBeLessThan(0.002);
    // Second leg, once the corner is detected and the heading settled
    // (beyond cornerDeviation + settleDistance): a straight line heading up.
    const leg = outBD.filter((q) => q.y > 0.15);
    const first = leg[0];
    const last = leg[leg.length - 1];
    for (const p of leg) expect(distanceToLine(p, first, last)).toBeLessThan(0.002);
    expect(Math.abs(last.x - first.x)).toBeLessThan(0.02); // ~vertical
    expect(distance(last, D)).toBeLessThan(0.03);
  });

  it('holds still while a resting hand trembles', () => {
    const straightener = new PathStraightener();
    straightener.reset({ x: 0, y: 0 });
    for (let i = 0; i < 30; i++) {
      const p = straightener.update({ x: 0.008 * Math.sin(i * 2.3), y: 0.008 * Math.cos(i * 1.9) });
      expect(p).toEqual({ x: 0, y: 0 });
    }
  });

  it('ignores tremor along the line after stopping (deadband)', () => {
    const straightener = new PathStraightener();
    straightener.reset({ x: 0, y: 0 });
    for (let i = 1; i <= 10; i++) straightener.update({ x: 0.02 * i, y: 0 });
    // Tremor of ±0.003 along the line (±0.004 sideways) around x = 0.2: the
    // output may take up the deadband slack once, then stays put.
    const tremor = (i: number) => ({ x: 0.2 + 0.003 * Math.sin(i * 2.1), y: 0.004 * Math.cos(i) });
    for (let i = 0; i < 10; i++) straightener.update(tremor(i));
    const settled = straightener.position;
    expect(distance(settled, { x: 0.2, y: 0 })).toBeLessThan(0.005);
    for (let i = 10; i < 40; i++) {
      expect(distance(straightener.update(tremor(i)), settled)).toBeLessThan(1e-9);
    }
  });
});
