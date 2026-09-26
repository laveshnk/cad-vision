/**
 * HandednessStabilizer: temporal + geometric stabilizer for MediaPipe
 * handedness labels, inserted between the tracker and the smoother bank.
 *
 * MediaPipe's Left/Right labels are re-classified every frame and suffer from
 * three failure modes this module addresses:
 *
 *  1. Flicker — transient label flips at rotational ambiguity (palm away,
 *     edge-on hands, partial occlusion).
 *  2. Collisions — both hands occasionally receive the same label, which
 *     downstream slot logic would otherwise resolve with a coin flip.
 *  3. Wrong seeds — a mislabeled first detection locks the wrong identity in
 *     for the whole session.
 *
 * Strategy:
 *  - Hands are matched to tracks frame-to-frame by wrist continuity; each
 *    track accumulates a rolling vote (MediaPipe label weighted by score,
 *    plus a geometric chirality vote once calibrated).
 *  - The geometric vote uses the rotation-invariant triple product
 *    (indexMCP − wrist) × (pinkyMCP − wrist) · (thumbMCP − wrist); its sign
 *    distinguishes left from right hands regardless of palm orientation. The
 *    sign → label mapping is self-calibrated from high-confidence MediaPipe
 *    reports, so it adapts to any mirroring / z convention.
 *  - A track only changes label when the vote decision disagrees for
 *    `flipFrames` consecutive frames (hysteresis kills flicker).
 *  - Two simultaneously visible tracks can never share a label: the weaker
 *    vote is forced to the opposite label.
 *  - Tracks absent longer than `reentryFrames` are forgotten, so a re-entering
 *    hand re-seeds its label from fresh evidence.
 */

import type { Handedness, RawHand, RawLandmark } from './types';

export interface HandednessStabilizerOptions {
  /** Rolling vote window length, in frames (default 5). */
  voteWindow?: number;
  /** Consecutive frames a contrary decision must hold before a track relabels (default 3). */
  flipFrames?: number;
  /**
   * MediaPipe score at/above which a report is treated as ground truth and
   * used to calibrate the geometric vote (default 0.9).
   */
  highConfidence?: number;
  /**
   * Weight of the calibrated geometric chirality vote (default 0.75). Kept
   * above any sub-`highConfidence` MediaPipe score so geometry outvotes
   * unreliable reports while high-confidence reports still win outright.
   */
  geometricWeight?: number;
  /** Frames a track may be absent before it is forgotten (default 10). */
  reentryFrames?: number;
  /** Wrist travel (normalized units) above which a hand cannot match a track (default 0.45). */
  maxWristTravel?: number;
  /** Minimum |chirality| for the geometric vote to be trusted (default 1e-6). */
  chiralityEpsilon?: number;
}

interface Vote {
  label: Handedness;
  weight: number;
}

interface Track {
  id: number;
  label: Handedness;
  votes: Vote[];
  lastWrist: { x: number; y: number; z: number };
  missed: number;
  flipCandidate: Handedness | null;
  flipStreak: number;
  lastConfidence: number;
}

type Chirality = -1 | 0 | 1;

/** Landmark indices used for the chirality triple product. */
const WRIST = 0;
const THUMB_MCP = 2;
const INDEX_MCP = 5;
const PINKY_MCP = 17;

function opposite(label: Handedness): Handedness {
  return label === 'Left' ? 'Right' : 'Left';
}

export class HandednessStabilizer {
  private readonly voteWindow: number;
  private readonly flipFrames: number;
  private readonly highConfidence: number;
  private readonly geometricWeight: number;
  private readonly reentryFrames: number;
  private readonly maxWristTravel: number;
  private readonly chiralityEpsilon: number;

  private readonly tracks: Track[] = [];
  private nextTrackId = 1;

  /** Per-label chirality sign tallies, fed by high-confidence reports. */
  private readonly signTallies: Record<Handedness, { positive: number; negative: number }>;

  constructor(options: HandednessStabilizerOptions = {}) {
    this.voteWindow = Math.max(1, Math.round(options.voteWindow ?? 5));
    this.flipFrames = Math.max(1, Math.round(options.flipFrames ?? 3));
    this.highConfidence = options.highConfidence ?? 0.9;
    this.geometricWeight = options.geometricWeight ?? 0.75;
    this.reentryFrames = Math.max(1, Math.round(options.reentryFrames ?? 10));
    this.maxWristTravel = options.maxWristTravel ?? 0.45;
    this.chiralityEpsilon = options.chiralityEpsilon ?? 1e-6;
    this.signTallies = {
      Left: { positive: 0, negative: 0 },
      Right: { positive: 0, negative: 0 },
    };
  }

  /**
   * Stabilize the handedness labels of one frame's raw hands.
   *
   * @returns new hand objects (landmarks shared) with corrected labels, in
   *          deterministic [Left, Right] order.
   */
  process(hands: RawHand[]): RawHand[] {
    const valid = hands.filter((hand) => hand.landmarks && hand.landmarks.length >= 21);
    if (valid.length === 0) {
      this.ageTracks(new Set());
      return [];
    }

    // 1. Greedy nearest-wrist matching of hands onto existing tracks.
    const candidates: Array<{ track: Track; hand: RawHand; distance: number }> = [];
    for (const track of this.tracks) {
      for (const hand of valid) {
        const distance = wristDistance(track.lastWrist, hand.landmarks[WRIST]);
        if (distance <= this.maxWristTravel) candidates.push({ track, hand, distance });
      }
    }
    candidates.sort((a, b) => a.distance - b.distance);
    const matchedTracks = new Set<Track>();
    const matchedHands = new Set<RawHand>();
    const pairs = new Map<Track, RawHand>();
    for (const candidate of candidates) {
      if (matchedTracks.has(candidate.track) || matchedHands.has(candidate.hand)) continue;
      matchedTracks.add(candidate.track);
      matchedHands.add(candidate.hand);
      pairs.set(candidate.track, candidate.hand);
    }

    const active: Array<{ hand: RawHand; track: Track }> = [];

    // 2. Update matched tracks (votes, hysteresis, wrist memory).
    for (const [track, hand] of pairs) {
      this.calibrate(hand);
      this.pushVotes(track, this.votesFor(hand));
      track.lastWrist = wristOf(hand);
      track.missed = 0;
      this.applyDecision(track, this.decision(track, hand));
      active.push({ hand, track });
    }

    // 3. Spawn tracks for unmatched hands (label seeded from this frame's votes).
    for (const hand of valid) {
      if (matchedHands.has(hand)) continue;
      active.push({ hand, track: this.spawn(hand) });
    }

    // 4. Age unmatched tracks; forget those absent too long.
    this.ageTracks(new Set(active.map((entry) => entry.track)));

    // 5. Mutual exclusivity: two visible tracks can never share a label.
    if (active.length === 2) {
      const [a, b] = active;
      if (a.track.label === b.track.label) {
        const keeper = a.track.lastConfidence >= b.track.lastConfidence ? a : b;
        const loser = keeper === a ? b : a;
        loser.track.label = opposite(keeper.track.label);
        loser.track.votes.length = 0;
        loser.track.flipCandidate = null;
        loser.track.flipStreak = 0;
      }
    }

    // 6. Emit stabilized hands in deterministic [Left, Right] order.
    const out: RawHand[] = active.map(({ hand, track }) => ({ ...hand, handedness: track.label }));
    out.sort((a, b) => (a.handedness === b.handedness ? 0 : a.handedness === 'Left' ? -1 : 1));
    return out;
  }

  /** Forget all tracks and calibration (e.g. when tracking stops). */
  reset(): void {
    this.tracks.length = 0;
    this.nextTrackId = 1;
    this.signTallies.Left.positive = 0;
    this.signTallies.Left.negative = 0;
    this.signTallies.Right.positive = 0;
    this.signTallies.Right.negative = 0;
  }

  /* ---------------------------------------------------------------------- */
  /* Internals                                                                */
  /* ---------------------------------------------------------------------- */

  /** Append votes to a track and trim to the rolling window. */
  private pushVotes(track: Track, votes: Vote[]): void {
    track.votes.push(...votes);
    if (track.votes.length > this.voteWindow) {
      track.votes.splice(0, track.votes.length - this.voteWindow);
    }
  }

  /** Build this frame's votes for a hand: MediaPipe label + geometric chirality. */
  private votesFor(hand: RawHand): Vote[] {
    const votes: Vote[] = [];
    if (hand.score > 0) {
      votes.push({
        label: hand.handedness,
        // High-confidence reports outweigh the geometric vote outright.
        weight:
          hand.score >= this.highConfidence
            ? this.highConfidence + this.geometricWeight
            : hand.score,
      });
    }
    const geometric = this.geometricVote(hand.landmarks);
    if (geometric) votes.push(geometric);
    return votes;
  }

  /** Vote decision for a track, falling back to the hand's label on a tie. */
  private decision(track: Track, fallbackHand: RawHand): { label: Handedness; confidence: number } {
    const { left, right } = tally(track.votes);
    const total = left + right;
    if (total <= 0) return { label: fallbackHand.handedness, confidence: 0 };
    const label = right > left ? 'Right' : left > right ? 'Left' : fallbackHand.handedness;
    return { label, confidence: Math.abs(right - left) / total };
  }

  /** Apply hysteresis: relabel only after `flipFrames` consecutive contrary decisions. */
  private applyDecision(track: Track, decision: { label: Handedness; confidence: number }): void {
    track.lastConfidence = decision.confidence;
    if (decision.label === track.label) {
      track.flipCandidate = null;
      track.flipStreak = 0;
      return;
    }
    if (track.flipCandidate === decision.label) {
      track.flipStreak++;
    } else {
      track.flipCandidate = decision.label;
      track.flipStreak = 1;
    }
    if (track.flipStreak >= this.flipFrames) {
      track.label = decision.label;
      track.flipCandidate = null;
      track.flipStreak = 0;
      track.votes.length = 0; // avoid an immediate flip-back
    }
  }

  /** Create a track for a newly seen hand, seeded from this frame's votes. */
  private spawn(hand: RawHand): Track {
    this.calibrate(hand);
    const votes = this.votesFor(hand);
    const { left, right } = tally(votes);
    const total = left + right;
    const track: Track = {
      id: this.nextTrackId++,
      label: right > left ? 'Right' : left > right ? 'Left' : hand.handedness,
      votes,
      lastWrist: wristOf(hand),
      missed: 0,
      flipCandidate: null,
      flipStreak: 0,
      lastConfidence: total > 0 ? Math.abs(right - left) / total : 0,
    };
    this.tracks.push(track);
    return track;
  }

  /** Record the chirality sign of a high-confidence report for calibration. */
  private calibrate(hand: RawHand): void {
    if (hand.score < this.highConfidence) return;
    const sign = chiralitySign(hand.landmarks, this.chiralityEpsilon);
    if (sign === 0) return;
    const counts = this.signTallies[hand.handedness];
    if (sign > 0) counts.positive++;
    else counts.negative++;
  }

  /** Geometric chirality vote, or null when uncalibrated / degenerate geometry. */
  private geometricVote(landmarks: RawLandmark[]): Vote | null {
    const rightSign = dominantSign(this.signTallies.Right);
    const leftSign = dominantSign(this.signTallies.Left);
    if (rightSign === 0 || leftSign === 0 || rightSign === leftSign) return null;
    const sign = chiralitySign(landmarks, this.chiralityEpsilon);
    if (sign === 0) return null;
    return { label: sign === rightSign ? 'Right' : 'Left', weight: this.geometricWeight };
  }

  /** Age absent tracks and forget those missing longer than `reentryFrames`. */
  private ageTracks(active: Set<Track>): void {
    for (let i = this.tracks.length - 1; i >= 0; i--) {
      const track = this.tracks[i];
      if (active.has(track)) continue;
      track.missed++;
      if (track.missed > this.reentryFrames) this.tracks.splice(i, 1);
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

function wristOf(hand: RawHand): { x: number; y: number; z: number } {
  const wrist = hand.landmarks[WRIST];
  return { x: wrist.x, y: wrist.y, z: wrist.z };
}

function wristDistance(a: { x: number; y: number; z: number }, b: RawLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function tally(votes: Vote[]): { left: number; right: number } {
  let left = 0;
  let right = 0;
  for (const vote of votes) {
    if (vote.label === 'Left') left += vote.weight;
    else right += vote.weight;
  }
  return { left, right };
}

/** Dominant chirality sign for a label's tallies; 0 while ambiguous. */
function dominantSign(counts: { positive: number; negative: number }): Chirality {
  const total = counts.positive + counts.negative;
  if (total < 2) return 0;
  if (counts.positive === counts.negative) return 0;
  return counts.positive > counts.negative ? 1 : -1;
}

/**
 * Rotation-invariant handedness chirality: the sign of
 * (indexMCP − wrist) × (pinkyMCP − wrist) · (thumbMCP − wrist).
 * Unlike a 2D cross product it does not flip between palm-facing and
 * back-facing views. Returns 0 when the geometry is degenerate (e.g. no z).
 */
export function chiralitySign(landmarks: RawLandmark[], epsilon = 1e-6): Chirality {
  const wrist = landmarks[WRIST];
  const thumb = landmarks[THUMB_MCP];
  const index = landmarks[INDEX_MCP];
  const pinky = landmarks[PINKY_MCP];
  if (!wrist || !thumb || !index || !pinky) return 0;

  const v1 = { x: index.x - wrist.x, y: index.y - wrist.y, z: index.z - wrist.z };
  const v2 = { x: pinky.x - wrist.x, y: pinky.y - wrist.y, z: pinky.z - wrist.z };
  const v3 = { x: thumb.x - wrist.x, y: thumb.y - wrist.y, z: thumb.z - wrist.z };

  // cross(v1, v2) · v3
  const crossX = v1.y * v2.z - v1.z * v2.y;
  const crossY = v1.z * v2.x - v1.x * v2.z;
  const crossZ = v1.x * v2.y - v1.y * v2.x;
  const triple = crossX * v3.x + crossY * v3.y + crossZ * v3.z;

  if (Math.abs(triple) < epsilon) return 0;
  return triple > 0 ? 1 : -1;
}
