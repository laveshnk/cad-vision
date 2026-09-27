/**
 * Dwell clock for a group of hover-to-press buttons: tracks which button
 * (index) a pointing fingertip rests on and for how long. Changing target
 * restarts the clock; reaching the threshold fires once and resets.
 */
export class DwellClock {
  /** Button being dwelt on (-1 = none). */
  target = -1;
  /** Time (ms) spent on `target` so far. */
  elapsed = 0;

  reset(): void {
    this.target = -1;
    this.elapsed = 0;
  }

  /**
   * Advance by `dt` ms with the currently hovered button `target` (-1 = none).
   * Returns the index that completed its `thresholdMs` dwell (the clock is
   * reset), or -1.
   */
  update(target: number, dt: number, thresholdMs: number): number {
    if (target !== this.target) {
      this.target = target;
      this.elapsed = 0;
    } else if (target >= 0 && dt > 0) {
      this.elapsed += dt;
    }
    if (this.target < 0 || this.elapsed < thresholdMs) return -1;
    const fired = this.target;
    this.reset();
    return fired;
  }

  /** Dwell completion [0, 1] for button `index`, or null when it is not being dwelt on. */
  progress(index: number, thresholdMs: number): number | null {
    if (this.target !== index) return null;
    return thresholdMs > 0 ? Math.min(1, this.elapsed / thresholdMs) : 0;
  }
}
