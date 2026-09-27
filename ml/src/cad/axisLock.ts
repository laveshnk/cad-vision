/**
 * AxisLock: single-axis filter for fist rotation of the selected object.
 *
 * A fist gesture turns the object about **one** axis at a time instead of
 * tumbling it freely with every wobble of the hand:
 *   - motion is accumulated until it travels `decideDistance`; the dominant
 *     direction then picks the axis — mostly horizontal → `'y'` (spin about
 *     the vertical axis), mostly vertical → `'x'` (tip about the horizontal
 *     axis) — and the accumulated travel along it is released at once, so
 *     no motion is lost;
 *   - while locked, only that axis's component passes; the other is dropped;
 *   - the lock clears when the fist stays still for `stillFrames` frames
 *     (below `stillDistance` per frame) or the host calls `reset()` (the
 *     fist gesture ended), so the next motion can choose again.
 *
 * Pure math on device-space deltas — no Three.js, unit-testable.
 */

export type RotationAxis = 'x' | 'y';

export interface AxisLockOptions {
  /** Travel (device units) needed before an axis is chosen. Default 0.03. */
  decideDistance?: number;
  /** Per-frame travel below which the fist counts as still. Default 0.002. */
  stillDistance?: number;
  /** Consecutive still frames that release the lock. Default 8 (~0.25 s). */
  stillFrames?: number;
}

export class AxisLock {
  private readonly decideDistance: number;
  private readonly stillDistance: number;
  private readonly stillFrames: number;
  private axis: RotationAxis | null = null;
  private pendingX = 0;
  private pendingY = 0;
  private still = 0;

  constructor(options: AxisLockOptions = {}) {
    this.decideDistance = options.decideDistance ?? 0.03;
    this.stillDistance = options.stillDistance ?? 0.002;
    this.stillFrames = options.stillFrames ?? 8;
  }

  /** The locked axis, or null while undecided. */
  get lockedAxis(): RotationAxis | null {
    return this.axis;
  }

  /**
   * Filter one frame of fist motion (device units, +X right, +Y up).
   * @returns the motion to apply: only the locked axis's component (the
   * other is 0); nothing until an axis has been chosen.
   */
  update(deltaX: number, deltaY: number): { deltaX: number; deltaY: number } {
    if (Math.hypot(deltaX, deltaY) < this.stillDistance) {
      this.still++;
      if (this.still >= this.stillFrames) this.reset();
      return { deltaX: 0, deltaY: 0 };
    }
    this.still = 0;

    if (this.axis === null) {
      this.pendingX += deltaX;
      this.pendingY += deltaY;
      if (Math.hypot(this.pendingX, this.pendingY) < this.decideDistance) {
        return { deltaX: 0, deltaY: 0 };
      }
      // Decide, and release the accumulated travel along the chosen axis.
      this.axis = Math.abs(this.pendingX) >= Math.abs(this.pendingY) ? 'y' : 'x';
      const released =
        this.axis === 'y'
          ? { deltaX: this.pendingX, deltaY: 0 }
          : { deltaX: 0, deltaY: this.pendingY };
      this.pendingX = 0;
      this.pendingY = 0;
      return released;
    }
    return this.axis === 'y' ? { deltaX, deltaY: 0 } : { deltaX: 0, deltaY };
  }

  /** Clear the lock (e.g. the fist opened / the gesture ended). */
  reset(): void {
    this.axis = null;
    this.pendingX = 0;
    this.pendingY = 0;
    this.still = 0;
  }
}
