import { describe, expect, it } from 'vitest';
import { buildLandmark } from '../coordinates';
import { GestureClassifier } from '../GestureClassifier';
import type { GestureSignalEvent, HandFrame, RawLandmark } from '../types';

/** Fast, deterministic classifier options for tests. */
const TEST_OPTIONS = {
  pinchStartThreshold: 0.045,
  pinchReleaseThreshold: 0.065,
  pinchDistanceSmoothing: null,
  fistEnterFrames: 1,
  fistExitFrames: 1,
  orbitOpenPalmGraceFrames: 3,
  handLossGraceFrames: 3,
};

interface HandSpec {
  /** Thumb-tip <-> index-tip distance (normalized units). */
  pinchDist?: number;
  fist?: boolean;
  /** MCP indices (5 / 9 / 13 / 17) forced open in a fist pose (partial fist). */
  extend?: number[];
  /** Raw-frame offset applied to the whole hand. */
  dx?: number;
  dy?: number;
}

/**
 * Synthetic MediaPipe-style hand: a vertical hand anchored at the wrist with
 * four fingers + thumb. `pinchDist` controls the thumb-tip (4) / index-tip (8)
 * distance; `fist` curls all fingertips toward the wrist (unless listed in
 * `extend`, which keeps individual fingers open for partial-fist poses).
 */
function makeHand(handedness: 'Left' | 'Right', spec: HandSpec = {}): HandFrame {
  const { pinchDist = 0.2, fist = false, extend = [], dx = 0, dy = 0 } = spec;
  const wx = 0.5 + dx;
  const wy = 0.62 + dy;
  const lm: RawLandmark[] = Array.from({ length: 21 }, () => ({ x: wx, y: wy, z: 0 }));
  lm[0] = { x: wx, y: wy, z: 0 }; // wrist
  const mcpXs = [-0.03, -0.01, 0.01, 0.03];
  [5, 9, 13, 17].forEach((mcp, i) => {
    const curled = fist && !extend.includes(mcp);
    lm[mcp] = { x: wx + mcpXs[i], y: wy - 0.08, z: 0 };
    lm[mcp + 1] = { x: wx + mcpXs[i], y: wy - 0.12, z: 0 }; // PIP
    lm[mcp + 2] = { x: wx + mcpXs[i], y: wy - 0.14, z: 0 }; // DIP
    const tipY = curled ? wy - 0.05 : wy - 0.18;
    lm[mcp + 3] = { x: wx + mcpXs[i], y: tipY, z: 0 }; // fingertip
  });
  lm[1] = { x: wx - 0.03, y: wy - 0.02, z: 0 };
  lm[2] = { x: wx - 0.05, y: wy - 0.05, z: 0 };
  lm[3] = { x: wx - 0.06, y: wy - 0.08, z: 0 };
  if (fist) {
    // Thumb wrapped over the knuckles, clear of the index tip so the pose is
    // a fist and not a pinch (dist(4, 8) ≈ 0.102 > fistPinchGuardDistance).
    lm[4] = { x: wx - 0.09, y: wy - 0.03, z: 0 };
    if (!extend.includes(5)) lm[8] = { x: wx + 0.01, y: wy - 0.05, z: 0 };
  } else {
    // Thumb tip / index tip symmetric around (wx, wy - 0.10).
    lm[4] = { x: wx - pinchDist / 2, y: wy - 0.1, z: 0 };
    lm[8] = { x: wx + pinchDist / 2, y: wy - 0.1, z: 0 };
  }
  return {
    handedness,
    score: 0.9,
    landmarks: lm.map((p) => buildLandmark(p, 1000, 1000)),
  };
}

function typesOf(events: GestureSignalEvent[]): string[] {
  return events.map((e) => e.type);
}

describe('GestureClassifier — pinch / draw', () => {
  it('starts a pinch below the trigger threshold and enters DRAWING_BASE', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const result = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(typesOf(result.events)).toEqual(['pinch_start', 'state_change', 'pinch_drag']);
    expect(classifier.currentState).toBe('DRAWING_BASE');
    const start = result.events.find((e) => e.type === 'pinch_start');
    expect(start).toMatchObject({ hand: 'Right' });
    expect(start?.type === 'pinch_start' ? start.distance : 0).toBeCloseTo(0.04);
  });

  it('maps pinch positions into mirrored, Y-up device space', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    // Pinch center raw (0.5, 0.52) -> device (0, -0.04).
    const result = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    const start = result.events.find((e) => e.type === 'pinch_start');
    expect(start).toBeDefined();
    if (start?.type === 'pinch_start') {
      expect(start.position.x).toBeCloseTo(0);
      expect(start.position.y).toBeCloseTo(-0.04);
    }
  });

  it('emits drag deltas in device space (raw -x/-y movement -> device +x/+y)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04, dx: -0.1, dy: -0.1 })],
      (t += 100)
    );
    const drag = result.events.find((e) => e.type === 'pinch_drag');
    expect(drag).toBeDefined();
    if (drag?.type === 'pinch_drag') {
      expect(drag.delta.x).toBeCloseTo(0.2);
      expect(drag.delta.y).toBeCloseTo(0.2);
      expect(drag.currentPos.x).toBeCloseTo(0.2);
      expect(drag.currentPos.y).toBeCloseTo(0.16);
    }
  });

  it('applies hysteresis between the trigger and release thresholds', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    // Between thresholds: no engagement.
    let result = classifier.process([makeHand('Right', { pinchDist: 0.05 })], (t += 100));
    expect(typesOf(result.events)).toEqual([]);
    // Below trigger: engage.
    result = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(typesOf(result.events)).toContain('pinch_start');
    // Back between thresholds: still engaged (no release).
    result = classifier.process([makeHand('Right', { pinchDist: 0.05 })], (t += 100));
    expect(typesOf(result.events)).not.toContain('pinch_end');
    expect(typesOf(result.events)).toContain('pinch_drag');
    // Above release threshold: release and return to IDLE.
    result = classifier.process([makeHand('Right', { pinchDist: 0.08 })], (t += 100));
    expect(typesOf(result.events)).toEqual(['pinch_end', 'state_change']);
    expect(classifier.currentState).toBe('IDLE');
  });
});

describe('GestureClassifier — fist / orbit', () => {
  it('enters ORBITING on a closed fist and emits device-space deltas', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    let result = classifier.process([makeHand('Right', { fist: true })], (t += 100));
    expect(typesOf(result.events)).toEqual(['state_change', 'orbit_start']);
    expect(classifier.currentState).toBe('ORBITING');

    // Move the fist down in the raw image (dy +0.05) -> device deltaY -0.1.
    result = classifier.process([makeHand('Right', { fist: true, dy: 0.05 })], (t += 100));
    const orbit = result.events.find((e) => e.type === 'orbit');
    expect(orbit).toBeDefined();
    if (orbit?.type === 'orbit') {
      expect(orbit.deltaX).toBeCloseTo(0);
      expect(orbit.deltaY).toBeCloseTo(-0.1);
    }
  });

  it('ends the orbit after the open-palm grace window', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Left', { fist: true })], (t += 100));
    expect(classifier.currentState).toBe('ORBITING');
    // Open palm for more than orbitOpenPalmGraceFrames (3).
    for (let i = 0; i < 3; i++) {
      classifier.process([makeHand('Left', {})], (t += 100));
    }
    expect(classifier.currentState).toBe('ORBITING');
    const result = classifier.process([makeHand('Left', {})], (t += 100));
    expect(typesOf(result.events)).toContain('orbit_end');
    expect(classifier.currentState).toBe('IDLE');
  });
});

describe('GestureClassifier — robust fist detection', () => {
  it('detects a fist with one finger still extended (3 of 4 curled)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const result = classifier.process(
      [makeHand('Right', { fist: true, extend: [13] })],
      (t += 100)
    );
    expect(typesOf(result.events)).toEqual(['state_change', 'orbit_start']);
    expect(classifier.currentState).toBe('ORBITING');
  });

  it('does not classify a pinch as a fist (pinch guard)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const result = classifier.process([makeHand('Right', { pinchDist: 0.03 })], (t += 100));
    expect(classifier.currentState).toBe('DRAWING_BASE');
    expect(typesOf(result.events)).not.toContain('orbit_start');
    expect(result.snapshots[0].fistActive).toBe(false);
  });

  it('debounces fist engagement to 2 consecutive frames by default', () => {
    const classifier = new GestureClassifier({
      pinchStartThreshold: 0.045,
      pinchReleaseThreshold: 0.065,
      pinchDistanceSmoothing: null,
    });
    let t = 0;
    let result = classifier.process([makeHand('Right', { fist: true })], (t += 100));
    expect(typesOf(result.events)).toEqual([]);
    result = classifier.process([makeHand('Right', { fist: true })], (t += 100));
    expect(typesOf(result.events)).toEqual(['state_change', 'orbit_start']);
    expect(classifier.currentState).toBe('ORBITING');
  });
});

describe('GestureClassifier — extrusion', () => {
  it('two-hand pinch pull enters EXTRUDING and reports scale/deltas', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    // Pinch centers: Left (0.3, 0.52), Right (0.7, 0.52) -> D = 0.4.
    let result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2 }),
        makeHand('Right', { pinchDist: 0.04, dx: 0.2 }),
      ],
      (t += 100)
    );
    expect(typesOf(result.events)).toContain('extrude_start');
    expect(classifier.currentState).toBe('EXTRUDING');

    // Pull the left hand further away: D = 0.5.
    result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.3 }),
        makeHand('Right', { pinchDist: 0.04, dx: 0.2 }),
      ],
      (t += 100)
    );
    const extrude = result.events.find((e) => e.type === 'extrude');
    expect(extrude).toBeDefined();
    if (extrude?.type === 'extrude') {
      expect(extrude.mode).toBe('dual-hand');
      expect(extrude.distance).toBeCloseTo(0.5);
      expect(extrude.scaleFactor).toBeCloseTo(0.5 / 0.4);
      expect(extrude.deltaDistance).toBeCloseTo(0.1);
    }
  });

  it('falls back to single-hand vertical extrusion when one hand releases', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2 }),
        makeHand('Right', { pinchDist: 0.04, dx: 0.2 }),
      ],
      (t += 100)
    );
    // Right hand releases.
    let result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2 }),
        makeHand('Right', { pinchDist: 0.2, dx: 0.2 }),
      ],
      (t += 100)
    );
    expect(typesOf(result.events)).toContain('pinch_end');
    expect(classifier.currentState).toBe('EXTRUDING');

    // Left hand moves up in the image (dy -0.05) -> device deltaHeight +0.1.
    result = classifier.process([makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: -0.05 })], (t += 100));
    const extrude = result.events.find((e) => e.type === 'extrude');
    expect(extrude).toBeDefined();
    if (extrude?.type === 'extrude') {
      expect(extrude.mode).toBe('single-hand');
      expect(extrude.hand).toBe('Left');
      expect(extrude.deltaHeight).toBeCloseTo(0.1);
      expect(extrude.cumulativeHeight).toBeCloseTo(0.1);
    }
  });

  it('returns to IDLE when the last pinch releases', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    const result = classifier.process([makeHand('Right', { pinchDist: 0.2 })], (t += 100));
    expect(typesOf(result.events)).toEqual(['pinch_end', 'state_change']);
    expect(classifier.currentState).toBe('IDLE');
  });
});

describe('GestureClassifier — hand loss', () => {
  it('finalizes gestures after the grace window with synthetic events', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(classifier.currentState).toBe('DRAWING_BASE');

    // Absent within the grace window: state holds, no events.
    for (let i = 0; i < 3; i++) {
      const held = classifier.process([], (t += 100));
      expect(held.events).toHaveLength(0);
    }
    // Beyond the grace window (missedFrames > 3): synthetic pinch_end + IDLE.
    const result = classifier.process([], (t += 100));
    expect(typesOf(result.events)).toEqual(['pinch_end', 'state_change']);
    expect(classifier.currentState).toBe('IDLE');
  });

  it('reports per-hand metrics in snapshots', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const result = classifier.process([makeHand('Left', { pinchDist: 0.03 })], (t += 100));
    expect(result.snapshots).toHaveLength(1);
    expect(result.snapshots[0].pinchActive).toBe(true);
    expect(result.snapshots[0].pinchDistance).toBeCloseTo(0.03);
    expect(result.metrics.pinchDistances[0].distance).toBeCloseTo(0.03);

    const idle = classifier.process([], (t += 100));
    expect(idle.snapshots).toHaveLength(0);
  });
});
