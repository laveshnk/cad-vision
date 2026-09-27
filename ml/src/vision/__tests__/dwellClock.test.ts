import { describe, expect, it } from 'vitest';
import { DwellClock } from '../DwellClock';

describe('DwellClock', () => {
  it('fires once the same target is held past the threshold, then resets', () => {
    const clock = new DwellClock();
    expect(clock.update(1, 16, 100)).toBe(-1); // first frame only selects the target
    for (let i = 0; i < 6; i++) expect(clock.update(1, 16, 100)).toBe(-1);
    expect(clock.update(1, 16, 100)).toBe(1);
    expect(clock.target).toBe(-1);
    expect(clock.elapsed).toBe(0);
  });

  it('restarts the clock when the target changes', () => {
    const clock = new DwellClock();
    clock.update(0, 16, 100);
    clock.update(0, 80, 100);
    expect(clock.update(2, 80, 100)).toBe(-1);
    expect(clock.elapsed).toBe(0);
    expect(clock.update(2, 99, 100)).toBe(-1);
    expect(clock.update(2, 1, 100)).toBe(2);
  });

  it('never accumulates without a target', () => {
    const clock = new DwellClock();
    for (let i = 0; i < 20; i++) expect(clock.update(-1, 50, 100)).toBe(-1);
    expect(clock.elapsed).toBe(0);
  });

  it('reports progress only for the dwelt-on button', () => {
    const clock = new DwellClock();
    clock.update(1, 0, 200);
    clock.update(1, 50, 200);
    expect(clock.progress(1, 200)).toBeCloseTo(0.25);
    expect(clock.progress(0, 200)).toBeNull();
    expect(clock.progress(1, 0)).toBe(0);
  });
});
