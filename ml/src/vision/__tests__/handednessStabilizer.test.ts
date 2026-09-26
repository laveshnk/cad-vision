import { describe, expect, it } from 'vitest';
import { chiralitySign, HandednessStabilizer } from '../HandednessStabilizer';
import type { RawHand, RawLandmark } from '../types';

/**
 * Synthetic hand with a well-defined chirality: the thumb MCP sits toward the
 * camera (negative z) and the pinky MCP is offset along ±x. Mirroring x flips
 * the triple-product sign, giving the two hands distinct chirality.
 */
function syntheticHand(
  handedness: 'Left' | 'Right',
  x = 0.5,
  options: { mirror?: boolean; score?: number; flatZ?: boolean } = {}
): RawHand {
  const { mirror = false, score = 0.95, flatZ = false } = options;
  const s = mirror ? -1 : 1;
  const landmarks: RawLandmark[] = Array.from({ length: 21 }, () => ({ x, y: 0.5, z: 0 }));
  landmarks[0] = { x, y: 0.5, z: 0 }; // wrist
  landmarks[2] = { x: x + 0.02 * s, y: 0.48, z: flatZ ? 0 : -0.05 }; // thumb MCP
  landmarks[5] = { x, y: 0.4, z: 0 }; // index MCP
  landmarks[17] = { x: x + 0.08 * s, y: 0.4, z: 0 }; // pinky MCP
  return { handedness, landmarks, score };
}

describe('chiralitySign', () => {
  it('distinguishes mirrored hands', () => {
    const right = syntheticHand('Right');
    const left = syntheticHand('Left', 0.5, { mirror: true });
    expect(chiralitySign(right.landmarks)).not.toBe(0);
    expect(chiralitySign(left.landmarks)).not.toBe(0);
    expect(chiralitySign(right.landmarks)).toBe(-chiralitySign(left.landmarks));
  });

  it('returns 0 for degenerate (flat z) geometry', () => {
    const flat = syntheticHand('Right', 0.5, { flatZ: true });
    expect(chiralitySign(flat.landmarks)).toBe(0);
  });
});

describe('HandednessStabilizer', () => {
  it('splits duplicate labels into distinct, stable labels', () => {
    const stabilizer = new HandednessStabilizer();
    let anchorLabel: string | null = null;
    for (let frame = 0; frame < 5; frame++) {
      // Both hands reported 'Right' (flat z: geometry cannot calibrate).
      const out = stabilizer.process([
        syntheticHand('Right', 0.3, { flatZ: true }),
        syntheticHand('Right', 0.7, { flatZ: true }),
      ]);
      expect(out).toHaveLength(2);
      const labels = out.map((hand) => hand.handedness);
      expect(new Set(labels).size).toBe(2);
      // Assignment is stable: the hand at x=0.3 keeps the same label every frame.
      const label = out.find((hand) => hand.landmarks[0].x === 0.3)?.handedness;
      expect(label).toBeDefined();
      if (anchorLabel === null) anchorLabel = label!;
      else expect(label).toBe(anchorLabel);
    }
  });

  it('ignores a single-frame label flicker', () => {
    const stabilizer = new HandednessStabilizer();
    for (let i = 0; i < 5; i++) {
      expect(stabilizer.process([syntheticHand('Right', 0.5, { flatZ: true })])[0].handedness).toBe(
        'Right'
      );
    }
    // One mislabeled frame at low confidence…
    expect(stabilizer.process([syntheticHand('Left', 0.5, { flatZ: true, score: 0.6 })])[0].handedness).toBe(
      'Right'
    );
    // …and tracking continues unchanged.
    expect(stabilizer.process([syntheticHand('Right', 0.5, { flatZ: true })])[0].handedness).toBe(
      'Right'
    );
  });

  it('accepts a persistent relabel only after the hysteresis window', () => {
    const stabilizer = new HandednessStabilizer();
    for (let i = 0; i < 5; i++) stabilizer.process([syntheticHand('Right', 0.5, { flatZ: true })]);
    const outputs: string[] = [];
    for (let i = 0; i < 6; i++) {
      outputs.push(stabilizer.process([syntheticHand('Left', 0.5, { flatZ: true })])[0].handedness);
    }
    // Window (5) + flip hysteresis (3): flips on the 5th contrary frame.
    expect(outputs.slice(0, 4)).toEqual(['Right', 'Right', 'Right', 'Right']);
    expect(outputs.slice(4)).toEqual(['Left', 'Left']);
  });

  /** Runs one high-confidence right hand + one mirrored left hand so the
   *  geometric vote is calibrated; tight wrist matching isolates new spawns. */
  function calibratedStabilizer(): HandednessStabilizer {
    const stabilizer = new HandednessStabilizer({ maxWristTravel: 0.1 });
    for (let i = 0; i < 3; i++) {
      stabilizer.process([syntheticHand('Right', 0.2)]); // non-mirrored chirality
    }
    for (let i = 0; i < 3; i++) {
      stabilizer.process([syntheticHand('Left', 0.8, { mirror: true })]);
    }
    return stabilizer;
  }

  it('corrects a mislabeled new hand via geometric chirality', () => {
    const stabilizer = calibratedStabilizer();
    // MediaPipe calls this hand 'Left', but its geometry matches the
    // calibrated *right*-hand chirality, at low confidence.
    const out = stabilizer.process([syntheticHand('Left', 0.5, { score: 0.6 })]);
    expect(out[0].handedness).toBe('Right');
  });

  it('keeps a correctly labeled hand when geometry agrees', () => {
    const stabilizer = calibratedStabilizer();
    const out = stabilizer.process([
      syntheticHand('Left', 0.5, { mirror: true, score: 0.6 }),
    ]);
    expect(out[0].handedness).toBe('Left');
  });

  it('re-seeds from a fresh report after prolonged absence', () => {
    const stabilizer = new HandednessStabilizer();
    for (let i = 0; i < 3; i++) stabilizer.process([syntheticHand('Right', 0.5, { flatZ: true })]);
    // Absent longer than the re-entry window: the track is forgotten.
    for (let i = 0; i < 11; i++) stabilizer.process([]);
    const out = stabilizer.process([syntheticHand('Left', 0.5, { flatZ: true })]);
    expect(out[0].handedness).toBe('Left');
  });

  it('retains identity across a brief absence', () => {
    const stabilizer = new HandednessStabilizer();
    for (let i = 0; i < 3; i++) stabilizer.process([syntheticHand('Right', 0.5, { flatZ: true })]);
    for (let i = 0; i < 3; i++) stabilizer.process([]);
    // The vote window still holds the old identity: a one-frame contrary
    // label cannot immediately re-seed a tracked hand.
    const out = stabilizer.process([syntheticHand('Left', 0.5, { flatZ: true })]);
    expect(out[0].handedness).toBe('Right');
  });

  it('emits hands in deterministic [Left, Right] order', () => {
    const stabilizer = new HandednessStabilizer({ maxWristTravel: 0.1 });
    const out = stabilizer.process([
      syntheticHand('Left', 0.7, { flatZ: true }),
      syntheticHand('Right', 0.3, { flatZ: true }),
    ]);
    expect(out.map((hand) => hand.handedness)).toEqual(['Left', 'Right']);
  });

  it('reset() clears all state', () => {
    const stabilizer = calibratedStabilizer();
    stabilizer.reset();
    const out = stabilizer.process([syntheticHand('Left', 0.5, { score: 0.6 })]);
    // No calibration survives: the low-confidence report is trusted.
    expect(out[0].handedness).toBe('Left');
  });
});
