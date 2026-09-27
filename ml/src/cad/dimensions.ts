/**
 * Solid dimensions in centimetres for the viewport's measurement labels.
 *
 * Pure math over plain descriptors (no Three.js objects), so it stays
 * unit-testable: CadBuilder maps each committed mesh / live preview onto a
 * `SolidShape` and the host renders the returned entries.
 *
 * Scale: one world unit (one ground-grid cell) is `CM_PER_UNIT` cm.
 * - box / cuboid → L (along X), W (along Z), H (along Y);
 * - cylinder     → R (radius), H (height);
 * - sphere       → R (radius);
 * - any other solid (CSG results) → L / W / H of its local bounding box.
 */

/** Centimetres per world unit (one ground-grid cell = 10 cm). */
export const CM_PER_UNIT = 10;

export type SolidShape =
  | { kind: 'box'; width: number; height: number; depth: number }
  | { kind: 'cylinder'; radius: number; height: number }
  | { kind: 'sphere'; radius: number };

export interface DimensionEntry {
  /** Short axis label: L, W, H or R. */
  label: string;
  /** Measured size in centimetres. */
  cm: number;
}

/** Centimetre dimensions of a solid, in display order. */
export function solidDimensions(shape: SolidShape, cmPerUnit = CM_PER_UNIT): DimensionEntry[] {
  const cm = (units: number) => Math.abs(units) * cmPerUnit;
  switch (shape.kind) {
    case 'box':
      return [
        { label: 'L', cm: cm(shape.width) },
        { label: 'W', cm: cm(shape.depth) },
        { label: 'H', cm: cm(shape.height) },
      ];
    case 'cylinder':
      return [
        { label: 'R', cm: cm(shape.radius) },
        { label: 'H', cm: cm(shape.height) },
      ];
    case 'sphere':
      return [{ label: 'R', cm: cm(shape.radius) }];
  }
}

/** One entry as display text, e.g. `L 12.5 cm` (one decimal). */
export function formatDimension(entry: DimensionEntry): string {
  return `${entry.label} ${entry.cm.toFixed(1)} cm`;
}
