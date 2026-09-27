import { describe, expect, it } from 'vitest';
import { buildLandmark } from '../coordinates';
import { GestureClassifier } from '../GestureClassifier';
import type { GestureSignalEvent, HandFrame, RawLandmark } from '../types';

/** Fast, deterministic classifier options for tests. */
const TEST_OPTIONS = {
  pinchStartThreshold: 0.045,
  pinchReleaseThreshold: 0.065,
  pinchDistanceSmoothing: null,
  pinchEnterFrames: 1,
  fistEnterFrames: 1,
  fistExitFrames: 1,
  orbitOpenPalmGraceFrames: 3,
  handLossGraceFrames: 3,
};

interface HandSpec {
  /** Thumb-tip <-> index-tip distance (normalized units). */
  pinchDist?: number;
  /** Real closed fist: fingertips folded into the palm, thumb wrapped over. */
  fist?: boolean;
  /** Raw-frame offset applied to the whole hand. */
  dx?: number;
  dy?: number;
}

/**
 * Synthetic MediaPipe-style hand: a vertical hand anchored at the wrist with
 * four fingers + thumb (palm size wrist -> middle MCP = 0.08). `pinchDist`
 * controls the thumb-tip (4) / index-tip (8) distance; `fist` folds each
 * finger toward the camera (-z) so the tip rests on the palm below its
 * knuckle (compact version of the fixture in gestureClassifier.test.ts).
 */
function makeHand(handedness: 'Left' | 'Right', spec: HandSpec = {}): HandFrame {
  const { pinchDist = 0.2, fist = false, dx = 0, dy = 0 } = spec;
  const wx = 0.5 + dx;
  const wy = 0.62 + dy;
  const lm: RawLandmark[] = Array.from({ length: 21 }, () => ({ x: wx, y: wy, z: 0 }));
  lm[0] = { x: wx, y: wy, z: 0 }; // wrist
  const mcpXs = [-0.03, -0.01, 0.01, 0.03];
  [5, 9, 13, 17].forEach((mcp, i) => {
    const x = wx + mcpXs[i];
    lm[mcp] = { x, y: wy - 0.08, z: 0 };
    if (fist) {
      lm[mcp + 1] = { x, y: wy - 0.1, z: -0.03 }; // PIP (knuckle bent toward camera)
      lm[mcp + 2] = { x, y: wy - 0.07, z: -0.045 }; // DIP
      lm[mcp + 3] = { x, y: wy - 0.045, z: -0.03 }; // tip on the palm
    } else {
      lm[mcp + 1] = { x, y: wy - 0.12, z: 0 }; // PIP
      lm[mcp + 2] = { x, y: wy - 0.14, z: 0 }; // DIP
      lm[mcp + 3] = { x, y: wy - 0.18, z: 0 }; // fingertip
    }
  });
  lm[1] = { x: wx - 0.03, y: wy - 0.02, z: 0 };
  lm[2] = { x: wx - 0.05, y: wy - 0.05, z: 0 };
  lm[3] = { x: wx - 0.06, y: wy - 0.08, z: 0 };
  if (fist) {
    lm[4] = { x: wx - 0.01, y: wy - 0.085, z: -0.06 }; // thumb wrapped over the knuckles
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

describe('GestureClassifier — interaction mode state machine', () => {
  it('defaults to create mode (legacy gesture set)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    expect(classifier.currentMode).toBe('create');
    let t = 0;
    const result = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(classifier.currentState).toBe('DRAWING_BASE');
    expect(typesOf(result.events)).toContain('pinch_start');
  });

  it('setMode emits a single mode_change and is a no-op for the same mode', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    const events = classifier.setMode('view', 1000);
    expect(typesOf(events)).toEqual(['mode_change']);
    const change = events[0];
    if (change.type === 'mode_change') {
      expect(change.from).toBe('create');
      expect(change.to).toBe('view');
    }
    expect(classifier.currentMode).toBe('view');
    expect(classifier.setMode('view', 1100)).toEqual([]);
  });

  it('VIEW mode keeps pinches inert (no pinch events, no draw state)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    classifier.setMode('view', 100);
    let t = 100;
    const engage = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(typesOf(engage.events)).toEqual([]);
    expect(classifier.currentState).toBe('IDLE');
    const release = classifier.process([makeHand('Right', { pinchDist: 0.2 })], (t += 100));
    expect(typesOf(release.events)).toEqual([]);
    expect(classifier.currentState).toBe('IDLE');
  });

  it('VIEW mode keeps camera gestures live (fist orbits despite a pinching hand)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    classifier.setMode('view', 100);
    let t = 100;
    const frames = [makeHand('Left', { pinchDist: 0.04 }), makeHand('Right', { fist: true })];
    const result = classifier.process(frames, (t += 100));
    expect(typesOf(result.events)).toContain('orbit_start');
    expect(classifier.currentState).toBe('ORBITING');
  });

  it('CREATE mode blocks orbit while a pinch is active (pinch wins)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const frames = [makeHand('Left', { pinchDist: 0.04 }), makeHand('Right', { fist: true })];
    const result = classifier.process(frames, (t += 100));
    expect(typesOf(result.events)).not.toContain('orbit_start');
    expect(classifier.currentState).toBe('DRAWING_BASE');
  });

  it('SELECT mode enters SELECTING on a pinch and emits pinch events', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    classifier.setMode('select', 100);
    let t = 100;
    const engage = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(classifier.currentState).toBe('SELECTING');
    expect(typesOf(engage.events)).toEqual(['pinch_start', 'state_change', 'pinch_drag']);
    const drag = classifier.process(
      [makeHand('Right', { pinchDist: 0.04, dy: -0.05 })],
      (t += 100)
    );
    expect(typesOf(drag.events)).toEqual(['pinch_drag']);
    const release = classifier.process([makeHand('Right', { pinchDist: 0.2 })], (t += 100));
    expect(typesOf(release.events)).toEqual(['pinch_end', 'state_change']);
    expect(classifier.currentState).toBe('IDLE');
  });

  it('SELECT mode never extrudes with two pinches', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    classifier.setMode('select', 100);
    let t = 100;
    const result = classifier.process(
      [makeHand('Left', { pinchDist: 0.04 }), makeHand('Right', { pinchDist: 0.04 })],
      (t += 100)
    );
    expect(classifier.currentState).toBe('SELECTING');
    const types = typesOf(result.events);
    expect(types).not.toContain('extrude_start');
    expect(types).not.toContain('extrude');
    expect(types.filter((type) => type === 'pinch_start')).toHaveLength(2);
    expect(types.filter((type) => type === 'pinch_drag')).toHaveLength(2);
  });

  it('SELECT mode keeps camera gestures live', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    classifier.setMode('select', 100);
    let t = 100;
    const result = classifier.process([makeHand('Right', { fist: true })], (t += 100));
    expect(typesOf(result.events)).toContain('orbit_start');
    expect(classifier.currentState).toBe('ORBITING');
  });

  it('switching modes mid-pinch force-releases the pinch and returns to IDLE', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(classifier.currentState).toBe('DRAWING_BASE');
    const events = classifier.setMode('view', (t += 100));
    expect(typesOf(events)).toEqual(['pinch_end', 'state_change', 'mode_change']);
    expect(classifier.currentState).toBe('IDLE');
    // The dropped pinch cannot re-engage (VIEW pinches are inert).
    const followUp = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(typesOf(followUp.events)).toEqual([]);
  });

  it('VIEW -> SELECT mid-pinch emits no synthetic pinch_end (no matching start)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    classifier.setMode('view', 100);
    let t = 100;
    classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    const events = classifier.setMode('select', (t += 100));
    expect(typesOf(events)).toEqual(['mode_change']);
  });
});
