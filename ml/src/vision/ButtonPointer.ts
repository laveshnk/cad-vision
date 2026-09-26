/**
 * ButtonPointer: finger "press" logic for on-overlay buttons.
 *
 * Pure, DOM-free state machine fed once per frame with where each hand's
 * index tip is (which button key it hovers, if any) and whether that hand is
 * pinching. A button activates when:
 *   - the index tip dwells over it for `dwellMs`, or
 *   - a hand pinches while its index tip is over it (immediate).
 * Each activation is latched: the same button cannot re-fire until the
 * finger leaves it (dwell) or the pinch leaves the buttons (pinch), so a held
 * finger or pinch never repeats or scrubs across buttons.
 */

/** One hand's index tip this frame. */
export interface ButtonPointerSample {
  /** Key of the button under the index tip, or `null`. */
  target: string | null;
  pinching: boolean;
}

export interface ButtonPointerOptions {
  /** Index-tip dwell time (ms) before a hovered button activates. Default 500. */
  dwellMs?: number;
  /** Largest frame gap (ms) counted toward a dwell, so a stalled feed can't jump it. Default 500. */
  maxFrameGapMs?: number;
}

export class ButtonPointer {
  private readonly dwellMs: number;
  private readonly maxFrameGapMs: number;
  private dwellKey: string | null = null;
  private dwellElapsed = 0;
  /** Button that already fired from the current dwell (no repeat until left). */
  private dwellFired: string | null = null;
  private pinchLatched = false;
  private lastTimestamp: number | null = null;

  constructor(options: ButtonPointerOptions = {}) {
    this.dwellMs = options.dwellMs ?? 500;
    this.maxFrameGapMs = options.maxFrameGapMs ?? 500;
  }

  /** Button currently being dwelled on (for hover feedback), or `null`. */
  get hovered(): string | null {
    return this.dwellKey;
  }

  /** Dwell progress on the hovered button in [0, 1] (0 once it has fired). */
  get progress(): number {
    if (this.dwellKey === null || this.dwellKey === this.dwellFired || this.dwellMs <= 0) return 0;
    return Math.min(1, this.dwellElapsed / this.dwellMs);
  }

  /**
   * Advance one frame.
   * @returns the key of the button activated this frame, or `null`.
   */
  update(samples: readonly ButtonPointerSample[], timestamp: number): string | null {
    let dwellKey: string | null = null;
    let pinchKey: string | null = null;
    for (const sample of samples) {
      if (sample.target === null) continue;
      dwellKey ??= sample.target;
      if (sample.pinching) pinchKey ??= sample.target;
    }

    const dt =
      this.lastTimestamp === null
        ? 0
        : Math.max(0, Math.min(timestamp - this.lastTimestamp, this.maxFrameGapMs));
    this.lastTimestamp = timestamp;

    if (dwellKey !== this.dwellKey) {
      this.dwellKey = dwellKey;
      this.dwellElapsed = 0;
      this.dwellFired = null;
    } else if (dwellKey !== null) {
      this.dwellElapsed += dt;
    }

    let activated: string | null = null;
    if (pinchKey !== null && !this.pinchLatched) {
      activated = pinchKey;
      this.pinchLatched = true;
    }
    if (pinchKey === null) this.pinchLatched = false;

    if (
      this.dwellKey !== null &&
      this.dwellFired !== this.dwellKey &&
      this.dwellElapsed >= this.dwellMs
    ) {
      activated ??= this.dwellKey;
      this.dwellFired = this.dwellKey;
    } else if (activated !== null && activated === this.dwellKey) {
      // A pinch press also counts as this dwell's activation.
      this.dwellFired = this.dwellKey;
    }
    return activated;
  }

  reset(): void {
    this.dwellKey = null;
    this.dwellElapsed = 0;
    this.dwellFired = null;
    this.pinchLatched = false;
    this.lastTimestamp = null;
  }
}
