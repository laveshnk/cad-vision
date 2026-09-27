import { describe, expect, it } from 'vitest';
import { resolveThumbWidth } from '../ThumbResizer';

describe('resolveThumbWidth', () => {
  it('applies the pointer drag delta 1:1 while inside the bounds', () => {
    expect(resolveThumbWidth(300, 0, 180, 720)).toBe(300);
    expect(resolveThumbWidth(300, 120, 180, 720)).toBe(420);
    expect(resolveThumbWidth(300, -60, 180, 720)).toBe(240);
  });

  it('clamps shrink drags at the minimum width', () => {
    expect(resolveThumbWidth(200, -400, 180, 720)).toBe(180);
    expect(resolveThumbWidth(180, -1, 180, 720)).toBe(180); // already minimal
  });

  it('clamps grow drags at the maximum width (e.g. the window clamp)', () => {
    expect(resolveThumbWidth(600, 400, 180, 720)).toBe(720);
    expect(resolveThumbWidth(720, 50, 180, 720)).toBe(720); // already maximal
  });

  it('never returns a width outside [minWidth, maxWidth]', () => {
    expect(resolveThumbWidth(0, 0, 180, 720)).toBe(180);
    expect(resolveThumbWidth(5000, -10, 180, 720)).toBe(720);
  });
});
