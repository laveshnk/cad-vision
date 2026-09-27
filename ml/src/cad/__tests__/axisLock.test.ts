import { describe, expect, it } from 'vitest';
import { AxisLock } from '../axisLock';

describe('AxisLock', () => {
  it('waits for decideDistance, then releases the travel on the dominant axis', () => {
    const lock = new AxisLock({ decideDistance: 0.03 });
    expect(lock.update(0.01, 0.002)).toEqual({ deltaX: 0, deltaY: 0 });
    expect(lock.update(0.01, 0.002)).toEqual({ deltaX: 0, deltaY: 0 });
    const out = lock.update(0.012, 0.001); // total ≈ (0.032, 0.005): horizontal
    expect(lock.lockedAxis).toBe('y');
    expect(out.deltaX).toBeCloseTo(0.032);
    expect(out.deltaY).toBe(0);
  });

  it('mostly vertical motion locks the x axis and drops horizontal wobble', () => {
    const lock = new AxisLock({ decideDistance: 0.03 });
    lock.update(0.004, 0.02);
    lock.update(-0.003, 0.02); // total (0.001, 0.04): vertical
    expect(lock.lockedAxis).toBe('x');
    expect(lock.update(0.015, 0.01)).toEqual({ deltaX: 0, deltaY: 0.01 });
    expect(lock.update(-0.02, -0.005)).toEqual({ deltaX: 0, deltaY: -0.005 });
  });

  it('ignores brief wobble onto the other axis (no switch)', () => {
    const lock = new AxisLock({ decideDistance: 0.03, switchDistance: 0.03 });
    lock.update(0.04, 0);
    expect(lock.lockedAxis).toBe('y');
    // Alternating: a vertical-dominated frame, then a horizontal one — the
    // pending switch resets each time the locked axis dominates again.
    for (let i = 0; i < 6; i++) {
      const out = i % 2 ? lock.update(0.02, 0.005) : lock.update(0.004, 0.02);
      expect(out.deltaY).toBe(0);
    }
    expect(lock.lockedAxis).toBe('y');
  });

  it('switches axis mid-gesture: right for a while, then up', () => {
    const lock = new AxisLock({ decideDistance: 0.03, switchDistance: 0.03 });
    let yaw = 0;
    let pitch = 0;
    for (let i = 0; i < 5; i++) {
      const out = lock.update(0.02, 0.001); // moving right
      yaw += out.deltaX;
      pitch += out.deltaY;
    }
    expect(lock.lockedAxis).toBe('y');
    expect(yaw).toBeCloseTo(0.1);
    expect(pitch).toBe(0);
    for (let i = 0; i < 5; i++) {
      const out = lock.update(0.001, 0.02); // now moving up, fist still closed
      yaw += out.deltaX;
      pitch += out.deltaY;
    }
    expect(lock.lockedAxis).toBe('x');
    // All the upward travel arrives as pitch (released on the switch).
    expect(pitch).toBeCloseTo(0.1);
    // Only the tiny horizontal drift before the switch leaked into yaw.
    expect(yaw).toBeCloseTo(0.1 + 0.001, 3);
  });

  it('releases the lock after the fist holds still, so a new axis can be chosen', () => {
    const lock = new AxisLock({ decideDistance: 0.03, stillDistance: 0.002, stillFrames: 3 });
    lock.update(0.04, 0);
    expect(lock.lockedAxis).toBe('y');
    for (let i = 0; i < 3; i++) lock.update(0.0005, 0.0005);
    expect(lock.lockedAxis).toBeNull();
    lock.update(0, 0.04);
    expect(lock.lockedAxis).toBe('x');
  });

  it('reset() clears the lock and any pending travel', () => {
    const lock = new AxisLock({ decideDistance: 0.03 });
    lock.update(0.02, 0);
    lock.reset();
    expect(lock.update(0.02, 0)).toEqual({ deltaX: 0, deltaY: 0 });
    expect(lock.lockedAxis).toBeNull();
  });
});
