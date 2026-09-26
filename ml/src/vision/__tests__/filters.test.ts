import { describe, expect, it } from 'vitest';
import { EmaScalar, HandSmootherBank, LandmarkSmoother, OneEuroScalar, Vec3Smoother } from '../filters';
import type { RawHand } from '../types';

function rawHand(handedness: 'Left' | 'Right', x: number, y: number): RawHand {
  const landmarks = Array.from({ length: 21 }, (_, i) => ({
    x: x + i * 0.001,
    y: y + i * 0.001,
    z: 0,
  }));
  return { handedness, landmarks, score: 0.9 };
}

describe('EmaScalar', () => {
  it('passes through the first value', () => {
    const ema = new EmaScalar(0.35);
    expect(ema.filter(10)).toBe(10);
  });

  it('converges toward the input', () => {
    const ema = new EmaScalar(0.5);
    ema.filter(0);
    let value = ema.filter(100);
    value = ema.filter(100);
    value = ema.filter(100);
    expect(value).toBeGreaterThan(50);
    expect(value).toBeLessThan(100);
  });

  it('alpha 1 is a pass-through', () => {
    const ema = new EmaScalar(1);
    ema.filter(1);
    expect(ema.filter(42)).toBe(42);
  });
});

describe('OneEuroScalar', () => {
  it('starts at the first value and tracks changes', () => {
    const filter = new OneEuroScalar({ minCutoff: 1.0, beta: 0.5 });
    expect(filter.filter(0, 1 / 30)).toBe(0);
    const out = filter.filter(10, 1 / 30);
    expect(out).toBeGreaterThan(0);
    expect(out).toBeLessThanOrEqual(10);
  });
});

describe('Vec3Smoother', () => {
  it('smooths each axis independently (first frame is identity)', () => {
    const smoother = new Vec3Smoother({ strategy: 'ema', alpha: 0.5 });
    const first = smoother.filter({ x: 1, y: 2, z: 3 });
    expect(first).toEqual({ x: 1, y: 2, z: 3 });
    const second = smoother.filter({ x: 2, y: 2, z: 2 });
    expect(second).toEqual({ x: 1.5, y: 2, z: 2.5 });
  });
});

describe('LandmarkSmoother', () => {
  it('smooths 21 landmarks per index', () => {
    const smoother = new LandmarkSmoother({ strategy: 'ema', alpha: 0.5 });
    const hand = rawHand('Right', 0.5, 0.5).landmarks;
    const out1 = smoother.filter(hand);
    expect(out1).toHaveLength(21);
    expect(out1[3].x).toBeCloseTo(hand[3].x);

    const moved = hand.map((lm) => ({ ...lm, x: lm.x + 0.1 }));
    const out2 = smoother.filter(moved);
    expect(out2[3].x).toBeCloseTo(hand[3].x + 0.05);
  });
});

describe('HandSmootherBank', () => {
  it('returns hands in deterministic Left-first order', () => {
    const bank = new HandSmootherBank({ strategy: 'ema', alpha: 0.35 });
    const out = bank.process([rawHand('Right', 0.7, 0.5), rawHand('Left', 0.3, 0.5)]);
    expect(out.map((h) => h.handedness)).toEqual(['Left', 'Right']);
  });

  it('keeps one slot per hand identity even with duplicate labels', () => {
    const bank = new HandSmootherBank();
    const out = bank.process([rawHand('Right', 0.7, 0.5), rawHand('Right', 0.3, 0.5)]);
    // Two detections always occupy two distinct slots (unique handedness keys).
    expect(out).toHaveLength(2);
    expect(new Set(out.map((h) => h.handedness)).size).toBe(2);
  });

  it('associates swapped labels with the nearest wrist history', () => {
    const bank = new HandSmootherBank({ strategy: 'ema', alpha: 0.35 });
    bank.process([rawHand('Left', 0.3, 0.5), rawHand('Right', 0.7, 0.5)]);
    // MediaPipe swaps the labels; nearest-wrist fallback should keep identities.
    const out = bank.process([rawHand('Right', 0.31, 0.5), rawHand('Left', 0.71, 0.5)]);
    expect(out).toHaveLength(2);
    // The slot previously tracking x≈0.3 stays near x≈0.31 (EMA with 0.35).
    const nearLeft = out.find((h) => Math.abs(h.landmarks[0].x - 0.3) < 0.05);
    expect(nearLeft).toBeDefined();
  });

  it('resets stale history after prolonged absence', () => {
    const bank = new HandSmootherBank({ strategy: 'ema', alpha: 0.35 });
    bank.process([rawHand('Left', 0.3, 0.5)]);
    for (let i = 0; i < 20; i++) bank.process([]);
    // Hand re-appears far away: should not ease in from the stale position.
    const out = bank.process([rawHand('Left', 0.8, 0.5)]);
    expect(out[0].landmarks[0].x).toBeCloseTo(0.8);
  });
});
