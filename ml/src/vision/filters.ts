/**
 * Signal-conditioning utilities: EMA and One-Euro smoothing for scalars,
 * 3D points, full 21-landmark hands, and a per-hand smoother bank that keeps
 * filter history stable across frames despite arbitrary MediaPipe output order.
 */

import type { Handedness, RawHand, RawLandmark, Vec3 } from './types';

/* -------------------------------------------------------------------------- */
/* Scalar filters                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Exponential moving average: `s = alpha * x + (1 - alpha) * s_prev`.
 * `alpha = 1` is a pass-through; `alpha = 0` freezes the signal.
 */
export class EmaScalar {
  private smoothed: number | null = null;

  constructor(public readonly alpha: number) {}

  filter(value: number, _dt?: number): number {
    this.smoothed =
      this.smoothed === null ? value : this.alpha * value + (1 - this.alpha) * this.smoothed;
    return this.smoothed;
  }

  get value(): number | null {
    return this.smoothed;
  }

  reset(): void {
    this.smoothed = null;
  }
}

export interface OneEuroOptions {
  /** Base cutoff frequency in Hz. Lower = smoother at rest. Default 1.0. */
  minCutoff?: number;
  /** Speed coefficient: cutoff grows with signal speed to reduce lag. Default 0.007. */
  beta?: number;
  /** Cutoff for the derivative estimate, Hz. Default 1.0. */
  dCutoff?: number;
}

/**
 * One-Euro filter (adaptive low-pass) for a scalar signal sampled per frame.
 * Well-suited to hand tracking: heavy smoothing when the hand is still,
 * low latency when it moves fast.
 */
export class OneEuroScalar {
  private prev: number | null = null;
  private prevFiltered = 0;
  private prevDeriv = 0;

  private readonly minCutoff: number;
  private readonly beta: number;
  private readonly dCutoff: number;

  constructor(options: OneEuroOptions = {}) {
    this.minCutoff = options.minCutoff ?? 1.0;
    this.beta = options.beta ?? 0.007;
    this.dCutoff = options.dCutoff ?? 1.0;
  }

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value: number, dt = 1 / 30): number {
    if (this.prev === null) {
      this.prev = value;
      this.prevFiltered = value;
      return value;
    }
    const deriv = (value - this.prev) / dt;
    const aD = OneEuroScalar.alpha(this.dCutoff, dt);
    const derivHat = aD * deriv + (1 - aD) * this.prevDeriv;
    const cutoff = this.minCutoff + this.beta * Math.abs(derivHat);
    const a = OneEuroScalar.alpha(cutoff, dt);
    const filtered = a * value + (1 - a) * this.prevFiltered;
    this.prev = value;
    this.prevFiltered = filtered;
    this.prevDeriv = derivHat;
    return filtered;
  }

  reset(): void {
    this.prev = null;
    this.prevFiltered = 0;
    this.prevDeriv = 0;
  }
}

/* -------------------------------------------------------------------------- */
/* Point / landmark smoothing                                                   */
/* -------------------------------------------------------------------------- */

export type SmoothingStrategy = 'ema' | 'one-euro';

export interface Vec3SmootherOptions {
  strategy?: SmoothingStrategy;
  /** EMA alpha (default 0.35). Ignored for the 'one-euro' strategy. */
  alpha?: number;
  oneEuro?: OneEuroOptions;
}

/** Smooths a single 3D point across frames using the selected strategy. */
export class Vec3Smoother {
  private readonly fx: EmaScalar | OneEuroScalar;
  private readonly fy: EmaScalar | OneEuroScalar;
  private readonly fz: EmaScalar | OneEuroScalar;

  constructor(options: Vec3SmootherOptions = {}) {
    const make = () =>
      options.strategy === 'one-euro'
        ? new OneEuroScalar(options.oneEuro)
        : new EmaScalar(options.alpha ?? 0.35);
    this.fx = make();
    this.fy = make();
    this.fz = make();
  }

  filter(p: Vec3, dt = 1 / 30): Vec3 {
    return { x: this.fx.filter(p.x, dt), y: this.fy.filter(p.y, dt), z: this.fz.filter(p.z, dt) };
  }

  reset(): void {
    this.fx.reset();
    this.fy.reset();
    this.fz.reset();
  }
}

/** Smooths all 21 landmarks of a single hand across frames. */
export class LandmarkSmoother {
  private readonly smoothers: Vec3Smoother[] = [];
  private lastWrist: Vec3 | null = null;

  constructor(options: Vec3SmootherOptions = {}) {
    for (let i = 0; i < 21; i++) {
      this.smoothers.push(new Vec3Smoother(options));
    }
  }

  filter(landmarks: RawLandmark[], dt = 1 / 30): RawLandmark[] {
    const out: RawLandmark[] = [];
    for (let i = 0; i < 21; i++) {
      const raw = landmarks[i];
      if (!raw) {
        out.push({ x: 0, y: 0, z: 0 });
        continue;
      }
      const s = this.smoothers[i].filter(raw, dt);
      out.push({ x: s.x, y: s.y, z: s.z, visibility: raw.visibility });
    }
    this.lastWrist = out[0] ? { x: out[0].x, y: out[0].y, z: out[0].z } : null;
    return out;
  }

  get lastWristPosition(): Vec3 | null {
    return this.lastWrist;
  }

  reset(): void {
    for (const s of this.smoothers) s.reset();
    this.lastWrist = null;
  }
}

/* -------------------------------------------------------------------------- */
/* Per-hand smoother bank (identity association)                               */
/* -------------------------------------------------------------------------- */

export interface HandSmootherBankOptions extends Vec3SmootherOptions {
  /**
   * Frames a hand may be absent before its smoothing history is discarded, so a
   * re-detected hand starts fresh instead of easing in from a stale position.
   */
  resetAfterMissedFrames?: number;
  /**
   * Wrist travel (normalized units) above which the hand is considered to have
   * teleported and its smoothing history is reset.
   */
  teleportThreshold?: number;
}

interface HandSlot {
  handedness: Handedness;
  smoother: LandmarkSmoother;
  missedFrames: number;
}

/**
 * Maintains one `LandmarkSmoother` per hand (Left / Right) and associates each
 * incoming MediaPipe detection with the correct filter history. MediaPipe emits
 * hands in arbitrary order and occasionally mislabels them, so association is
 * handedness-first with a nearest-wrist fallback. Output is keyed by slot
 * identity, guaranteeing at most one Left and one Right hand per frame.
 */
export class HandSmootherBank {
  private readonly strategy: SmoothingStrategy;
  private readonly alpha: number;
  private readonly oneEuro: OneEuroOptions | undefined;
  private readonly resetAfterMissedFrames: number;
  private readonly teleportThreshold: number;
  private readonly slots = new Map<Handedness, HandSlot>();

  constructor(options: HandSmootherBankOptions = {}) {
    this.strategy = options.strategy ?? 'ema';
    this.alpha = options.alpha ?? 0.35;
    this.oneEuro = options.oneEuro;
    this.resetAfterMissedFrames = options.resetAfterMissedFrames ?? 8;
    this.teleportThreshold = options.teleportThreshold ?? 0.45;
  }

  private getSlot(handedness: Handedness): HandSlot {
    let slot = this.slots.get(handedness);
    if (!slot) {
      slot = {
        handedness,
        smoother: new LandmarkSmoother({
          strategy: this.strategy,
          alpha: this.alpha,
          oneEuro: this.oneEuro,
        }),
        missedFrames: 0,
      };
      this.slots.set(handedness, slot);
    }
    return slot;
  }

  /**
   * Smooth a new frame's raw hands.
   *
   * Association strategy: slots are claimed by minimal total wrist-travel
   * (with a tiny label-agreement tie-break), so swapped or duplicated
   * MediaPipe handedness labels cannot swap filter histories. When no slot has
   * history yet, hands are assigned by their reported label. A single visible
   * hand is assigned to the slot matching its (stabilized) reported label
   * unless that would imply a teleport, in which case nearest-wrist
   * continuity wins.
   *
   * @returns the hands present this frame, keyed by slot identity (unique
   *          handedness), in deterministic [Left, Right] order.
   */
  process(hands: RawHand[], dt = 1 / 30): RawHand[] {
    const valid = hands.filter((h) => h.landmarks && h.landmarks.length >= 21);
    const slotL = this.getSlot('Left');
    const slotR = this.getSlot('Right');

    const travel = (slot: HandSlot, hand: RawHand): number => {
      const last = slot.smoother.lastWristPosition;
      if (last === null) return Number.POSITIVE_INFINITY;
      const wrist = hand.landmarks[0];
      return Math.hypot(last.x - wrist.x, last.y - wrist.y, last.z - wrist.z);
    };

    const pairs: Array<[HandSlot, RawHand]> = [];
    if (valid.length === 1) {
      const hand = valid[0];
      const dL = travel(slotL, hand);
      const dR = travel(slotR, hand);
      // Trust the (stabilized) reported label for the slot choice unless it
      // implies a teleport: the label's slot is preferred whenever it has no
      // history (a genuinely new hand) or the hand is where that slot expects
      // it. Otherwise fall back to nearest-wrist continuity.
      const labelSlot = this.getSlot(hand.handedness);
      const dLabel = travel(labelSlot, hand);
      const slot =
        dLabel === Number.POSITIVE_INFINITY || dLabel <= this.teleportThreshold
          ? labelSlot
          : dL <= dR
            ? slotL
            : slotR;
      pairs.push([slot, hand]);
    } else if (valid.length >= 2) {
      const [h0, h1] = valid;
      const hasHistory =
        slotL.smoother.lastWristPosition !== null || slotR.smoother.lastWristPosition !== null;
      if (!hasHistory) {
        // First frame: trust the reported labels, then fall back to slot order.
        const first = this.getSlot(h0.handedness);
        pairs.push([first, h0]);
        pairs.push([first === slotL ? slotR : slotL, h1]);
      } else {
        const labelBonus = (slot: HandSlot, hand: RawHand) =>
          slot.handedness === hand.handedness ? 1e-9 : 0;
        const direct =
          travel(slotL, h0) + travel(slotR, h1) - labelBonus(slotL, h0) - labelBonus(slotR, h1);
        const swapped =
          travel(slotR, h0) + travel(slotL, h1) - labelBonus(slotR, h0) - labelBonus(slotL, h1);
        if (direct <= swapped) {
          pairs.push([slotL, h0], [slotR, h1]);
        } else {
          pairs.push([slotR, h0], [slotL, h1]);
        }
      }
    }

    // Filter through each slot's history; track absence for stale-history resets.
    const assignments = new Map<HandSlot, RawHand>(pairs);
    const out: RawHand[] = [];
    for (const [slot, hand] of assignments) {
      const wrist = hand.landmarks[0];
      const last = slot.smoother.lastWristPosition;
      const teleported =
        last !== null &&
        Math.hypot(last.x - wrist.x, last.y - wrist.y, last.z - wrist.z) >
          this.teleportThreshold;
      if (teleported) slot.smoother.reset();
      const smoothed = slot.smoother.filter(hand.landmarks, dt);
      slot.missedFrames = 0;
      out.push({ handedness: slot.handedness, landmarks: smoothed, score: hand.score });
    }

    for (const slot of this.slots.values()) {
      if (assignments.has(slot)) continue;
      slot.missedFrames++;
      if (slot.missedFrames > this.resetAfterMissedFrames) {
        slot.smoother.reset();
        slot.missedFrames = 0;
      }
    }

    out.sort((a, b) => (a.handedness === b.handedness ? 0 : a.handedness === 'Left' ? -1 : 1));
    return out;
  }

  reset(): void {
    for (const slot of this.slots.values()) slot.smoother.reset();
    this.slots.clear();
  }
}

