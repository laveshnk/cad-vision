import { describe, expect, it } from 'vitest';
import { hslToHex, wheelColorAt } from '../ColorWheel';

describe('hslToHex', () => {
  it('maps the primary + secondary hues at full saturation / mid lightness', () => {
    expect(hslToHex(0, 1, 0.5)).toBe('#ff0000');
    expect(hslToHex(60, 1, 0.5)).toBe('#ffff00');
    expect(hslToHex(120, 1, 0.5)).toBe('#00ff00');
    expect(hslToHex(180, 1, 0.5)).toBe('#00ffff');
    expect(hslToHex(240, 1, 0.5)).toBe('#0000ff');
    expect(hslToHex(300, 1, 0.5)).toBe('#ff00ff');
  });

  it('collapses to grayscale at zero saturation and the lightness extremes', () => {
    expect(hslToHex(0, 0, 0)).toBe('#000000');
    expect(hslToHex(0, 0, 1)).toBe('#ffffff');
    expect(hslToHex(210, 0, 0.5)).toBe('#808080');
  });

  it('wraps out-of-range hues back into the wheel (negative + > 360)', () => {
    expect(hslToHex(-300, 1, 0.5)).toBe(hslToHex(60, 1, 0.5));
    expect(hslToHex(420, 1, 0.5)).toBe(hslToHex(60, 1, 0.5));
    expect(hslToHex(-300, 1, 0.5)).toBe('#ffff00');
  });
});

describe('wheelColorAt', () => {
  const radius = 100;

  it('the wheel center is fully desaturated (mid gray at lightness 0.5)', () => {
    expect(wheelColorAt(0, 0, radius)).toBe('#808080');
  });

  it('hue follows the angle; the rim is fully saturated', () => {
    expect(wheelColorAt(radius, 0, radius)).toBe('#ff0000'); // 0° = red
    expect(wheelColorAt(0, radius, radius)).toBe('#80ff00'); // 90° = chartreuse
    expect(wheelColorAt(-radius, 0, radius)).toBe('#00ffff'); // 180° = cyan
    expect(wheelColorAt(0, -radius, radius)).toBe('#8000ff'); // 270° = violet
  });

  it('saturation scales with the distance from the center', () => {
    expect(wheelColorAt(radius / 2, 0, radius)).toBe('#bf4040'); // hsl(0, 50%, 50%)
  });

  it('returns null outside the disc and for a degenerate radius', () => {
    expect(wheelColorAt(radius + 1, 0, radius)).toBeNull();
    expect(wheelColorAt(0, radius + 0.5, radius)).toBeNull();
    expect(wheelColorAt(5, 5, 0)).toBeNull();
    expect(wheelColorAt(-5, -5, -1)).toBeNull();
  });

  it('honours the lightness parameter', () => {
    expect(wheelColorAt(radius, 0, radius, 1)).toBe('#ffffff');
    expect(wheelColorAt(0, 0, radius, 0)).toBe('#000000');
  });
});
