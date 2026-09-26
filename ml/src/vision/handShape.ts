/**
 * Hand-shape geometry used to tell a real closed fist apart from fingers that
 * are merely curled (a claw / hook, a loose half-curl, a thumbs-up).
 *
 * Every measure is a ratio of distances within the same hand, so it is
 * independent of hand size, camera distance and in-plane rotation. Points are
 * first made isotropic (x / y / z in the same units, pixel-scaled) so the
 * video aspect ratio does not distort distances in different directions.
 */

import { distance3 } from './coordinates';
import type { HandFrame, Vec3 } from './types';

const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
/** [fingertip, MCP] pairs for index / middle / ring / pinky. */
const FINGER_TIP_MCP: ReadonlyArray<readonly [number, number]> = [
  [8, 5],
  [12, 9],
  [16, 13],
  [20, 17],
];
/** Index / middle / ring knuckles (MCP + PIP) the thumb rests on in a fist. */
const THUMB_TUCK_TARGETS = [5, 6, 9, 10, 13, 14];

export interface HandShape {
  /**
   * Per finger (index, middle, ring, pinky): `dist(tip, wrist) / dist(mcp, wrist)`.
   * About 0.6–0.8 when the fingertip is folded into the palm (real fist),
   * about 1.0–1.3 for a claw / hook curl, and ~2 for an open finger.
   */
  foldRatios: number[];
  /**
   * Thumb-tip distance to the nearest index / middle / ring knuckle, divided
   * by palm size (wrist → middle MCP). Small when the thumb is wrapped over
   * or alongside the fingers; large for a thumbs-up or a splayed thumb.
   */
  thumbTuckRatio: number;
  /** Thumb-tip <-> index-tip distance in normalized units (pinch metric). */
  thumbIndexDistance: number;
}

/**
 * Landmarks in isotropic units: normalized x / y rescaled by the video pixel
 * size (MediaPipe's z uses the same scale as x). Falls back to normalized
 * coordinates when no pixel size is available.
 */
function isotropicPoints(hand: HandFrame): Vec3[] {
  let sx = 0;
  let sy = 0;
  for (const lm of hand.landmarks) {
    if (sx === 0 && lm.normalized.x > 0.05) sx = lm.pixel.x / lm.normalized.x;
    if (sy === 0 && lm.normalized.y > 0.05) sy = lm.pixel.y / lm.normalized.y;
    if (sx > 0 && sy > 0) break;
  }
  if (!(sx > 0 && sy > 0)) return hand.landmarks.map((lm) => lm.normalized);
  return hand.landmarks.map((lm) => ({
    x: lm.normalized.x * sx,
    y: lm.normalized.y * sy,
    z: lm.normalized.z * sx,
  }));
}

/** Measure the fold / thumb-tuck geometry of one conditioned hand. */
export function measureHandShape(hand: HandFrame): HandShape {
  const p = isotropicPoints(hand);
  const wrist = p[WRIST];
  const palmSize = Math.max(distance3(wrist, p[MIDDLE_MCP]), 1e-6);

  const foldRatios = FINGER_TIP_MCP.map(
    ([tip, mcp]) => distance3(p[tip], wrist) / Math.max(distance3(p[mcp], wrist), 1e-6)
  );
  const thumbTuck = Math.min(...THUMB_TUCK_TARGETS.map((i) => distance3(p[THUMB_TIP], p[i])));

  return {
    foldRatios,
    thumbTuckRatio: thumbTuck / palmSize,
    thumbIndexDistance: distance3(
      hand.landmarks[THUMB_TIP].normalized,
      hand.landmarks[INDEX_TIP].normalized
    ),
  };
}
