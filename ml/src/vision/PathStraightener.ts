/**
 * PathStraightener: turns a wiggly 2D hand path into straight segments.
 *
 * Camera motion that tracks every hand tremor feels unsteady (and can make
 * people dizzy), so camera gestures run through this filter while building
 * gestures stay live. A path A → B → C drawn with a shaky hand comes out as
 * the straight segments A → B and B → C:
 *
 *   - At rest nothing moves until the hand travels `startDistance`; that
 *     displacement fixes the segment's direction.
 *   - While moving, the output is the hand's progress *along* the segment's
 *     line — sideways wiggle is discarded. Moving back along the same line
 *     (A → B → back toward A) stays on it.
 *   - For the first `settleDistance` of a segment the direction is still
 *     refined from the *average* displacement of every sample so far, so
 *     sideways wiggle cancels out and it locks onto the intended heading
 *     rather than the first few noisy frames.
 *   - When the hand drifts more than `cornerDeviation` off the line, it has
 *     really changed direction: a new segment starts at the corner.
 *   - Along-line changes within `deadband` are held (tremor at rest).
 *
 * Pure and deterministic: positions in, positions out, no timing.
 */

import type { Vec2 } from './types';

export interface PathStraightenerOptions {
  /** Distance to travel from rest before a segment's direction is chosen. */
  startDistance?: number;
  /** Sideways drift off the current line that starts a new segment (a corner). */
  cornerDeviation?: number;
  /** Progress over which a new segment's direction is still refined. */
  settleDistance?: number;
  /** Along-line changes smaller than this are held (tremor / backlash). */
  deadband?: number;
}

export class PathStraightener {
  private readonly options: Required<PathStraightenerOptions>;

  /** Raw (hand) point where the current segment starts. */
  private segmentRaw: Vec2 = { x: 0, y: 0 };
  /** Output point where the current segment starts. */
  private segmentOut: Vec2 = { x: 0, y: 0 };
  /** Unit direction of the current segment; `null` until chosen. */
  private direction: Vec2 | null = null;
  /** Output progress along `direction` (deadband-filtered). */
  private progress = 0;
  private settled = false;
  /** Sum of displacements since the segment started (direction averaging). */
  private sumX = 0;
  private sumY = 0;

  constructor(options: PathStraightenerOptions = {}) {
    this.options = {
      startDistance: options.startDistance ?? 0.02,
      cornerDeviation: options.cornerDeviation ?? 0.05,
      settleDistance: options.settleDistance ?? 0.08,
      deadband: options.deadband ?? 0.004,
    };
  }

  /** Start a fresh path at `start` (output starts there too). */
  reset(start: Vec2 = { x: 0, y: 0 }): void {
    this.segmentRaw = { ...start };
    this.segmentOut = { ...start };
    this.direction = null;
    this.progress = 0;
    this.settled = false;
    this.sumX = 0;
    this.sumY = 0;
  }

  /** Current straightened output position. */
  get position(): Vec2 {
    const d = this.direction;
    if (!d) return { ...this.segmentOut };
    return {
      x: this.segmentOut.x + d.x * this.progress,
      y: this.segmentOut.y + d.y * this.progress,
    };
  }

  /** Feed the latest raw hand position; returns the straightened position. */
  update(raw: Vec2): Vec2 {
    const { startDistance, cornerDeviation, settleDistance } = this.options;
    const vx = raw.x - this.segmentRaw.x;
    const vy = raw.y - this.segmentRaw.y;
    const travel = Math.hypot(vx, vy);

    if (!this.settled) {
      // Early in a segment: the heading is the average displacement so far.
      this.sumX += vx;
      this.sumY += vy;
      if (!this.direction && travel < startDistance) return this.position; // at rest
      const sumLength = Math.hypot(this.sumX, this.sumY);
      if (sumLength > 1e-9) this.direction = { x: this.sumX / sumLength, y: this.sumY / sumLength };
      if (travel >= settleDistance) this.settled = true;
    }
    if (!this.direction) return this.position;

    let d = this.direction;
    let along = vx * d.x + vy * d.y;
    const sideways = Math.abs(vx * d.y - vy * d.x);

    if (this.settled && sideways > cornerDeviation) {
      // Corner: start a new segment from the current output point, anchored
      // at the foot of the hand's position on the old line.
      const out = this.position;
      const foot = { x: this.segmentRaw.x + d.x * along, y: this.segmentRaw.y + d.y * along };
      const nx = raw.x - foot.x;
      const ny = raw.y - foot.y;
      const length = Math.hypot(nx, ny);
      this.segmentRaw = foot;
      this.segmentOut = out;
      this.direction = { x: nx / length, y: ny / length };
      this.progress = 0;
      this.settled = false;
      this.sumX = nx;
      this.sumY = ny;
      d = this.direction;
      along = length;
    }

    // Deadband (backlash): output progress trails the hand by at most
    // `deadband`, so sub-deadband tremor never moves the camera.
    const gap = along - this.progress;
    if (Math.abs(gap) > this.options.deadband) {
      this.progress = along - Math.sign(gap) * this.options.deadband;
    }
    return this.position;
  }
}
