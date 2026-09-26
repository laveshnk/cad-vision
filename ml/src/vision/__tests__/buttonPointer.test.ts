import { describe, expect, it } from 'vitest';
import { ButtonPointer, type ButtonPointerSample } from '../ButtonPointer';

const hover = (target: string | null, pinching = false): ButtonPointerSample => ({
  target,
  pinching,
});

/** Feed the same samples every `step` ms for `frames` frames; collect activations. */
function run(
  pointer: ButtonPointer,
  samples: ButtonPointerSample[],
  frames: number,
  start: number,
  step = 100
): (string | null)[] {
  const out: (string | null)[] = [];
  for (let i = 0; i < frames; i++) out.push(pointer.update(samples, start + i * step));
  return out;
}

describe('ButtonPointer — dwell', () => {
  it('activates after the index tip dwells for dwellMs', () => {
    const pointer = new ButtonPointer({ dwellMs: 500 });
    const fired = run(pointer, [hover('shape:cuboid')], 6, 0);
    // t = 0 starts the dwell; 500 ms later (6th frame) it fires once.
    expect(fired).toEqual([null, null, null, null, null, 'shape:cuboid']);
  });

  it('does not repeat while the finger stays on the same button', () => {
    const pointer = new ButtonPointer({ dwellMs: 300 });
    const fired = run(pointer, [hover('shape:sphere')], 20, 0);
    expect(fired.filter((f) => f !== null)).toEqual(['shape:sphere']);
    expect(pointer.progress).toBe(0);
  });

  it('re-arms after the finger leaves and comes back', () => {
    const pointer = new ButtonPointer({ dwellMs: 200 });
    run(pointer, [hover('mode:create')], 4, 0);
    pointer.update([hover(null)], 400);
    const fired = run(pointer, [hover('mode:create')], 4, 500);
    expect(fired.filter((f) => f !== null)).toEqual(['mode:create']);
  });

  it('restarts the dwell when the finger moves to another button', () => {
    const pointer = new ButtonPointer({ dwellMs: 500 });
    run(pointer, [hover('shape:box')], 4, 0); // 300 ms on box
    const fired = run(pointer, [hover('shape:cylinder')], 5, 400); // 400 ms on cylinder
    expect(fired.every((f) => f === null)).toBe(true);
    expect(pointer.hovered).toBe('shape:cylinder');
    expect(pointer.progress).toBeCloseTo(0.8);
  });

  it('caps a stalled frame gap so a dwell cannot complete in one jump', () => {
    const pointer = new ButtonPointer({ dwellMs: 500, maxFrameGapMs: 100 });
    pointer.update([hover('shape:box')], 0);
    expect(pointer.update([hover('shape:box')], 5_000)).toBeNull();
  });
});

describe('ButtonPointer — pinch', () => {
  it('activates immediately on a pinch over a button, once per pinch', () => {
    const pointer = new ButtonPointer({ dwellMs: 10_000 });
    const fired = run(pointer, [hover('shape:cuboid', true)], 5, 0);
    expect(fired).toEqual(['shape:cuboid', null, null, null, null]);
  });

  it('a held pinch cannot scrub across buttons', () => {
    const pointer = new ButtonPointer({ dwellMs: 10_000 });
    expect(pointer.update([hover('shape:box', true)], 0)).toBe('shape:box');
    expect(pointer.update([hover('shape:sphere', true)], 100)).toBeNull();
    // Leaving the buttons releases the latch.
    pointer.update([hover(null, true)], 200);
    expect(pointer.update([hover('shape:sphere', true)], 300)).toBe('shape:sphere');
  });

  it('a pinch press counts as the dwell activation (no double fire)', () => {
    const pointer = new ButtonPointer({ dwellMs: 200 });
    expect(pointer.update([hover('shape:cylinder', true)], 0)).toBe('shape:cylinder');
    const fired = run(pointer, [hover('shape:cylinder')], 5, 100);
    expect(fired.every((f) => f === null)).toBe(true);
  });

  it('uses the first hand over a button; the other hand can pinch elsewhere', () => {
    const pointer = new ButtonPointer({ dwellMs: 10_000 });
    const fired = pointer.update([hover(null, true), hover('mode:view', true)], 0);
    expect(fired).toBe('mode:view');
  });
});
