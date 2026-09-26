/**
 * Hand-shape geometry used to tell a real closed fist apart from fingers that
 * are merely curled (a claw / hook, a loose half-curl, a thumbs-up), and to
 * measure wrist roll (twisting the fist like a doorknob).
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
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;
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
export function isotropicPoints(hand: HandFrame): Vec3[] {
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

/**
 * Wrist roll angle (radians) of a hand: the rotation of the knuckle line
 * (index MCP → pinky MCP) around the hand's own long axis (wrist → middle
 * MCP), like turning a doorknob. Measured in the mirrored, Y-up view the user
 * sees (X right, Y up, Z toward the camera), so a positive change is a
 * counter-clockwise twist about the wrist → knuckles axis as seen on screen.
 *
 * Only differences between frames are meaningful. Returns `null` when the
 * hand points straight at the camera (the reference direction degenerates).
 */
export function measureWristRoll(hand: HandFrame): number | null {
  // Mirror X, flip Y up, and Z toward the camera (MediaPipe z < 0 = closer).
  const p = isotropicPoints(hand).map((q) => ({ x: -q.x, y: -q.y, z: -q.z }));
  const axis = normalize(sub(p[MIDDLE_MCP], p[WRIST]));
  if (!axis) return null;

  // Knuckle line, projected onto the plane perpendicular to the hand axis.
  const knuckles = sub(p[PINKY_MCP], p[INDEX_MCP]);
  const k = sub(knuckles, scale(axis, dot(knuckles, axis)));

  // Reference basis in that plane: r1 = view direction (toward the camera)
  // projected off the axis, r2 = axis × r1.
  const toCamera = { x: 0, y: 0, z: 1 };
  const r1 = normalize(sub(toCamera, scale(axis, dot(toCamera, axis))), 0.2);
  if (!r1) return null;
  const r2 = cross(axis, r1);
  return Math.atan2(dot(k, r2), dot(k, r1));
}

/** Wrap an angle difference into (-π, π]. */
export function wrapAngle(angle: number): number {
  let a = angle % (2 * Math.PI);
  if (a > Math.PI) a -= 2 * Math.PI;
  if (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function scale(a: Vec3, s: number): Vec3 {
  return { x: a.x * s, y: a.y * s, z: a.z * s };
}

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

/** Unit vector, or `null` when shorter than `minLength` (degenerate). */
function normalize(a: Vec3, minLength = 1e-9): Vec3 | null {
  const length = Math.sqrt(dot(a, a));
  return length < minLength ? null : scale(a, 1 / length);
}
