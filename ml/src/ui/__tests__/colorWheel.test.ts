import { describe, expect, it } from 'vitest';
import { DwellTracker, hslToHex, wheelColorAt, wheelSliceAt } from '../ColorWheel';

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

describe('wheelSliceAt', () => {
  const radius = 100;

  /** Slice key at a hue angle (degrees) on the rim. */
  const sliceAt = (degrees: number): string | null =>
    wheelSliceAt(
      Math.cos((degrees * Math.PI) / 180) * radius,
      Math.sin((degrees * Math.PI) / 180) * radius,
      radius
    );

  it('collapses the desaturated center into a single neutral slice', () => {
    expect(wheelSliceAt(0, 0, radius)).toBe('neutral');
    expect(wheelSliceAt(3, 4, radius)).toBe('neutral'); // saturation 0.05
    expect(wheelSliceAt(3, 4, radius)).toBe(wheelSliceAt(0, 0, radius));
  });

  it('quantizes the hue into 15° sectors (stable within, distinct across)', () => {
    expect(sliceAt(5)).toBe(sliceAt(10)); // both inside the 0–15° sector
    expect(sliceAt(5)).not.toBe(sliceAt(20)); // 20° crosses into the next sector
    expect(sliceAt(350)).toBe(sliceAt(-10)); // hue wraps: 350° ≡ -10°
  });

  it('returns null outside the disc', () => {
    expect(wheelSliceAt(radius + 1, 0, radius)).toBeNull();
    expect(wheelSliceAt(0, 0, 0)).toBeNull();
  });
});

describe('DwellTracker', () => {
  it('does not commit before the dwell duration elapses', () => {
    const tracker = new DwellTracker({ durationMs: 2000 });
    expect(tracker.advance('red', 500)).toBe(false);
    expect(tracker.advance('red', 500)).toBe(false); // 1000 ms
    expect(tracker.advance('red', 499)).toBe(false); // 1499 ms
    expect(tracker.progress).toBeCloseTo(0.7495, 3);
  });

  it('commits exactly once when the dwell completes, then resets', () => {
    const tracker = new DwellTracker({ durationMs: 2000 });
    expect(tracker.advance('red', 500)).toBe(false); // 500 ms
    expect(tracker.advance('red', 500)).toBe(false); // 1000 ms
    expect(tracker.advance('red', 500)).toBe(false); // 1500 ms
    expect(tracker.advance('red', 500)).toBe(true); // 2000 ms → commit
    expect(tracker.key).toBeNull(); // fired and auto-reset
    expect(tracker.progress).toBe(0);
    expect(tracker.advance('red', 100)).toBe(false); // a fresh clock started
    expect(tracker.progress).toBeCloseTo(0.05, 5);
  });

  it('resets when the hovered key changes or the hover is lost', () => {
    const tracker = new DwellTracker({ durationMs: 2000 });
    expect(tracker.advance('red', 500)).toBe(false);
    expect(tracker.advance('blue', 500)).toBe(false); // key change restarted the clock
    expect(tracker.advance(null, 100)).toBe(false); // hover lost → full reset
    expect(tracker.key).toBeNull();
    expect(tracker.advance('blue', 500)).toBe(false); // only 500 ms since the loss
    expect(tracker.progress).toBeCloseTo(0.25, 5);
  });

  it('clamps huge time steps so a stalled feed cannot complete a dwell in one jump', () => {
    const tracker = new DwellTracker({ durationMs: 2000, maxStepMs: 500 });
    expect(tracker.advance('red', 100_000)).toBe(false); // only 500 ms counted
    expect(tracker.progress).toBeCloseTo(0.25, 5);
    expect(tracker.advance('red', 500)).toBe(false); // 1000 ms
    expect(tracker.advance('red', 500)).toBe(false); // 1500 ms
    expect(tracker.advance('red', 500)).toBe(true); // 2000 ms
  });

  it('ignores negative time steps', () => {
    const tracker = new DwellTracker({ durationMs: 2000 });
    expect(tracker.advance('red', -1000)).toBe(false);
    expect(tracker.progress).toBe(0);
  });

  it('defaults to a 1.2 second lock at a realistic frame cadence', () => {
    const tracker = new DwellTracker();
    for (let i = 0; i < 39; i++) {
      expect(tracker.advance('red', 30)).toBe(false); // 39 × 30 = 1170 ms
    }
    expect(tracker.advance('red', 30)).toBe(true); // 1200 ms ≥ 1200
  });
});
