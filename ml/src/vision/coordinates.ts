/**
 * Coordinate-space conversion utilities.
 *
 * The camera feed is displayed mirrored (selfie view). MediaPipe reports
 * un-mirrored, image-normalized coordinates, so the device-space mapping below
 * both mirrors X and maps to [-1, 1]:
 *
 *   x_device = (1 - raw_x) * 2 - 1
 *   y_device = -(raw_y * 2 - 1)   (+Y up)
 */

import type { Landmark, RawLandmark, Vec2, Vec3 } from './types';

/** Mirrored + range-mapped X: (1 - x) * 2 - 1. */
export function rawToDeviceX(x: number): number {
  return (1 - x) * 2 - 1;
}

/** Range-mapped, Y-up: -(y * 2 - 1). */
export function rawToDeviceY(y: number): number {
  return -(y * 2 - 1);
}

/** Convert a raw (image-normalized) landmark to device space. */
export function toDeviceSpace(p: RawLandmark): Vec3 {
  return { x: rawToDeviceX(p.x), y: rawToDeviceY(p.y), z: p.z };
}

/** Convert a raw (image-normalized) landmark to pixel space. */
export function toPixelSpace(p: RawLandmark, width: number, height: number): Vec2 {
  return { x: p.x * width, y: p.y * height };
}

/** Build a fully conditioned landmark (all coordinate spaces) from a raw one. */
export function buildLandmark(raw: RawLandmark, width: number, height: number): Landmark {
  return {
    normalized: { x: raw.x, y: raw.y, z: raw.z },
    pixel: toPixelSpace(raw, width, height),
    device: toDeviceSpace(raw),
  };
}

/** Euclidean distance between two 3D points. */
export function distance3(a: Vec3, b: Vec3): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Midpoint of two 3D points. */
export function midpoint3(a: Vec3, b: Vec3): Vec3 {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

/** Component-wise difference `a - b`. */
export function subtract3(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

/** Arithmetic mean of a list of 3D points. */
export function centroid(points: Vec3[]): Vec3 {
  if (points.length === 0) return { x: 0, y: 0, z: 0 };
  let x = 0;
  let y = 0;
  let z = 0;
  for (const p of points) {
    x += p.x;
    y += p.y;
    z += p.z;
  }
  return { x: x / points.length, y: y / points.length, z: z / points.length };
}
