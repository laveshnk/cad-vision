import { describe, expect, it } from 'vitest';
import { buildLandmark } from '../coordinates';
import { GestureClassifier } from '../GestureClassifier';
import { measureWristRoll, wrapAngle } from '../handShape';
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
  /** Claw / hook: fingers curled at PIP + DIP, knuckles straight (not a fist). */
  hook?: boolean;
  /** In a fist pose, stick the thumb up (thumbs-up — not a fist). */
  thumbUp?: boolean;
  /** MCP indices (5 / 9 / 13 / 17) forced open in a fist pose (partial fist). */
  extend?: number[];
  /**
   * Wrist roll (radians): rigid rotation about the vertical axis through the
   * wrist, like twisting a doorknob.
   */
  twist?: number;
  /**
   * In-plane tilt (radians): rigid rotation of the whole hand about the
   * wrist in the image plane — drives the open-palm rotation gesture.
   */
  tilt?: number;
  /** Raw-frame offset applied to the whole hand. */
  dx?: number;
  dy?: number;
}

/**
 * Synthetic MediaPipe-style hand: a vertical hand anchored at the wrist with
 * four fingers + thumb (palm size wrist -> middle MCP = 0.08). `pinchDist`
 * controls the thumb-tip (4) / index-tip (8) distance. `fist` folds each
 * finger toward the camera (-z) and back so the tip rests on the palm below
 * its knuckle (unless listed in `extend`); `hook` curls only the PIP / DIP
 * joints so the tips stay out at knuckle height.
 */
function makeHand(handedness: 'Left' | 'Right', spec: HandSpec = {}): HandFrame {
  const {
    pinchDist = 0.2,
    fist = false,
    hook = false,
    thumbUp = false,
    extend = [],
    twist = 0,
    tilt = 0,
    dx = 0,
    dy = 0,
  } = spec;
  const wx = 0.5 + dx;
  const wy = 0.62 + dy;
  const lm: RawLandmark[] = Array.from({ length: 21 }, () => ({ x: wx, y: wy, z: 0 }));
  lm[0] = { x: wx, y: wy, z: 0 }; // wrist
  const mcpXs = [-0.03, -0.01, 0.01, 0.03];
  [5, 9, 13, 17].forEach((mcp, i) => {
    const x = wx + mcpXs[i];
    lm[mcp] = { x, y: wy - 0.08, z: 0 };
    if (fist && !extend.includes(mcp)) {
      lm[mcp + 1] = { x, y: wy - 0.1, z: -0.03 }; // PIP (knuckle bent toward camera)
      lm[mcp + 2] = { x, y: wy - 0.07, z: -0.045 }; // DIP
      lm[mcp + 3] = { x, y: wy - 0.045, z: -0.03 }; // tip on the palm
    } else if (hook) {
      lm[mcp + 1] = { x, y: wy - 0.12, z: 0 }; // PIP (knuckle straight)
      lm[mcp + 2] = { x, y: wy - 0.12, z: -0.025 }; // DIP
      lm[mcp + 3] = { x, y: wy - 0.105, z: -0.04 }; // tip out at knuckle height
    } else {
      lm[mcp + 1] = { x, y: wy - 0.12, z: 0 }; // PIP
      lm[mcp + 2] = { x, y: wy - 0.14, z: 0 }; // DIP
      lm[mcp + 3] = { x, y: wy - 0.18, z: 0 }; // fingertip
    }
  });
  lm[1] = { x: wx - 0.03, y: wy - 0.02, z: 0 };
  lm[2] = { x: wx - 0.05, y: wy - 0.05, z: 0 };
  lm[3] = { x: wx - 0.06, y: wy - 0.08, z: 0 };
  if (hook) {
    lm[4] = { x: wx - 0.08, y: wy - 0.09, z: 0 }; // relaxed thumb out to the side
  } else if (fist) {
    lm[4] = thumbUp
      ? { x: wx - 0.06, y: wy - 0.2, z: 0 } // thumbs-up
      : { x: wx - 0.01, y: wy - 0.085, z: -0.06 }; // wrapped over the knuckles
  } else {
    // Thumb tip / index tip symmetric around (wx, wy - 0.10).
    lm[4] = { x: wx - pinchDist / 2, y: wy - 0.1, z: 0 };
    lm[8] = { x: wx + pinchDist / 2, y: wy - 0.1, z: 0 };
  }
  const c = Math.cos(twist);
  const s = Math.sin(twist);
  const twisted = lm.map((p) => ({
    x: wx + (p.x - wx) * c + p.z * s,
    y: p.y,
    z: -(p.x - wx) * s + p.z * c,
  }));
  const tc = Math.cos(tilt);
  const ts = Math.sin(tilt);
  const tilted = twisted.map((p) => ({
    x: wx + (p.x - wx) * tc - (p.y - wy) * ts,
    y: wy + (p.x - wx) * ts + (p.y - wy) * tc,
    z: p.z,
  }));
  return {
    handedness,
    score: 0.9,
    landmarks: tilted.map((p) => buildLandmark(p, 1000, 1000)),
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

describe('GestureClassifier — accidental pinch guards', () => {
  const GUARDED = { ...TEST_OPTIONS, pinchEnterFrames: 2, pinchEdgeMargin: 0.03 };

  it('ignores a one-frame pinch blip; engages after pinchEnterFrames', () => {
    const classifier = new GestureClassifier(GUARDED);
    let t = 0;
    // A single pinched frame (flicker / misdetection) never starts a pinch.
    expect(typesOf(classifier.process([makeHand('Right', { pinchDist: 0.03 })], (t += 100)).events)).toEqual([]);
    expect(typesOf(classifier.process([makeHand('Right')], (t += 100)).events)).toEqual([]);
    expect(classifier.currentState).toBe('IDLE');
    // Two consecutive pinched frames do.
    classifier.process([makeHand('Right', { pinchDist: 0.03 })], (t += 100));
    const result = classifier.process([makeHand('Right', { pinchDist: 0.03 })], (t += 100));
    expect(typesOf(result.events)).toContain('pinch_start');
  });

  it('never starts a pinch at the frame border (hand half out of view)', () => {
    const classifier = new GestureClassifier(GUARDED);
    let t = 0;
    // Pinch center at x ≈ 0.99 (right border): held for many frames, no start.
    for (let i = 0; i < 5; i++) {
      const events = classifier.process([makeHand('Right', { pinchDist: 0.03, dx: 0.49 })], (t += 100)).events;
      expect(typesOf(events)).not.toContain('pinch_start');
    }
    // Moving the same pinch into the frame engages it.
    classifier.process([makeHand('Right', { pinchDist: 0.03, dx: 0.3 })], (t += 100));
    const result = classifier.process([makeHand('Right', { pinchDist: 0.03, dx: 0.3 })], (t += 100));
    expect(typesOf(result.events)).toContain('pinch_start');
  });

  it('keeps an engaged pinch when it drifts to the border (only the start is guarded)', () => {
    const classifier = new GestureClassifier(GUARDED);
    let t = 0;
    classifier.process([makeHand('Right', { pinchDist: 0.03 })], (t += 100));
    classifier.process([makeHand('Right', { pinchDist: 0.03 })], (t += 100));
    const result = classifier.process([makeHand('Right', { pinchDist: 0.03, dx: 0.49 })], (t += 100));
    expect(typesOf(result.events)).not.toContain('pinch_end');
    expect(typesOf(result.events)).toContain('pinch_drag');
  });

  it('reports how long a two-hand build lasted on extrude_end', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    const both = (pinch: number) => [
      makeHand('Left', { pinchDist: pinch, dx: -0.2 }),
      makeHand('Right', { pinchDist: pinch, dx: 0.2 }),
    ];
    classifier.process(both(0.03), 1000);
    classifier.process(both(0.03), 1100);
    const result = classifier.process(both(0.2), 1400);
    expect(result.events.find((e) => e.type === 'extrude_end')).toMatchObject({ durationMs: 400 });
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

describe('GestureClassifier — single-fist camera move', () => {
  it('reports rightward fist motion as a positive device deltaX', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { fist: true })], (t += 100));
    // Raw -x is device +x (mirrored selfie view): the hand moved right on screen.
    const result = classifier.process([makeHand('Right', { fist: true, dx: -0.05 })], (t += 100));
    const move = result.events.find((e) => e.type === 'orbit');
    expect(move?.type === 'orbit' && move.deltaX).toBeCloseTo(0.1);
  });
});

describe('GestureClassifier — wrist roll (one fist)', () => {
  it('stays silent below rollEngageAngle, then releases the accumulated twist', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const rollOf = (twist: number): number => {
      const result = classifier.process([makeHand('Right', { fist: true, twist })], (t += 100));
      const orbit = result.events.find((e) => e.type === 'orbit');
      return orbit?.type === 'orbit' ? orbit.deltaRoll : NaN;
    };
    classifier.process([makeHand('Right', { fist: true })], (t += 100));
    expect(classifier.currentState).toBe('ORBITING');
    expect(rollOf(0.1)).toBe(0); // 0.1 rad < 0.15 engage angle
    expect(rollOf(0.3)).toBeCloseTo(0.3); // engaged: whole twist released
    expect(rollOf(0.4)).toBeCloseTo(0.1); // then per-frame deltas
    expect(rollOf(0.2)).toBeCloseTo(-0.2); // and back the other way
    expect(classifier.process([makeHand('Right', { fist: true, twist: 0.2 })], (t += 100))
      .metrics.orbitRoll).toBeCloseTo(0.2);
  });

  it('does not report roll when the fist only moves (no twist)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { fist: true })], (t += 100));
    for (let i = 1; i <= 5; i++) {
      const result = classifier.process(
        [makeHand('Right', { fist: true, dx: 0.03 * i, dy: -0.02 * i })],
        (t += 100)
      );
      const orbit = result.events.find((e) => e.type === 'orbit');
      expect(orbit?.type === 'orbit' && orbit.deltaRoll).toBe(0);
    }
  });

  it('re-arms the engage threshold for each new fist', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { fist: true })], (t += 100));
    classifier.process([makeHand('Right', { fist: true, twist: 0.5 })], (t += 100));
    // Open past the grace window, then close a new fist and twist a little.
    for (let i = 0; i < 4; i++) classifier.process([makeHand('Right', {})], (t += 100));
    expect(classifier.currentState).toBe('IDLE');
    classifier.process([makeHand('Right', { fist: true })], (t += 100));
    const result = classifier.process([makeHand('Right', { fist: true, twist: 0.1 })], (t += 100));
    const orbit = result.events.find((e) => e.type === 'orbit');
    expect(orbit?.type === 'orbit' && orbit.deltaRoll).toBe(0);
  });
});

describe('measureWristRoll', () => {
  it('measures a doorknob twist as a change in roll angle, sign included', () => {
    const base = measureWristRoll(makeHand('Right', { fist: true }));
    const ccw = measureWristRoll(makeHand('Right', { fist: true, twist: 0.4 }));
    const cw = measureWristRoll(makeHand('Right', { fist: true, twist: -0.4 }));
    expect(base).not.toBeNull();
    expect(wrapAngle((ccw ?? 0) - (base ?? 0))).toBeCloseTo(0.4);
    expect(wrapAngle((cw ?? 0) - (base ?? 0))).toBeCloseTo(-0.4);
  });

  it('is unaffected by moving the hand around the frame', () => {
    const a = measureWristRoll(makeHand('Left', { fist: true }));
    const b = measureWristRoll(makeHand('Left', { fist: true, dx: 0.2, dy: -0.1 }));
    expect(wrapAngle((b ?? 0) - (a ?? 0))).toBeCloseTo(0);
  });

  it('wraps angle differences across ±π', () => {
    expect(wrapAngle(3.1 - -3.1)).toBeCloseTo(6.2 - 2 * Math.PI);
    expect(wrapAngle(-3.1 - 3.1)).toBeCloseTo(2 * Math.PI - 6.2);
  });
});

describe('GestureClassifier — two-fist zoom', () => {
  it('enters ZOOMING with two fists; zooms only when both fists move', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const frame = (leftDx: number, rightDx: number) =>
      classifier.process(
        [makeHand('Left', { fist: true, dx: leftDx }), makeHand('Right', { fist: true, dx: rightDx })],
        (t += 100)
      );
    let result = frame(-0.2, 0.2);
    expect(typesOf(result.events)).toEqual(['state_change', 'zoom_start']);
    expect(classifier.currentState).toBe('ZOOMING');

    // Both fists move toward each other: 0.4 -> 0.2 apart.
    result = frame(-0.1, 0.1);
    const closer = result.events.find((e) => e.type === 'zoom');
    expect(closer?.type).toBe('zoom');
    if (closer?.type === 'zoom') {
      expect(closer.distance).toBeCloseTo(0.2);
      expect(closer.deltaScale).toBeCloseTo(0.5);
      expect(closer.scaleFactor).toBeCloseTo(0.5);
      expect(closer.deltaAngle).toBe(0);
    }
    expect(result.metrics.zoomScaleFactor).toBeCloseTo(0.5);

    // …and both back out: 0.2 -> 0.3 apart.
    result = frame(-0.15, 0.15);
    const apart = result.events.find((e) => e.type === 'zoom');
    if (apart?.type === 'zoom') expect(apart.deltaScale).toBeCloseTo(1.5);
    expect(typesOf(result.events)).not.toContain('orbit');
  });

  it('one fist moving while the other stays still moves the camera — no zoom', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, cameraPath: null });
    let t = 0;
    const frame = (rightDx: number) =>
      classifier.process(
        [makeHand('Left', { fist: true, dx: -0.2 }), makeHand('Right', { fist: true, dx: rightDx })],
        (t += 100)
      );
    frame(0.2);
    // Right fist moves toward the still Left fist (the old anchor-zoom case).
    let result = frame(0.1);
    expect(typesOf(result.events)).not.toContain('zoom');
    const orbit = result.events.find((e) => e.type === 'orbit');
    expect(orbit).toMatchObject({ hand: 'Right', deltaRoll: 0 });
    // Raw -x image motion is +x in mirrored device space: 0.1 * 2 = 0.2.
    if (orbit?.type === 'orbit') expect(orbit.deltaX).toBeCloseTo(0.2);
    expect(result.metrics.zoomAnchor).toBe('Left');
    expect(classifier.currentState).toBe('ZOOMING');

    // Keeps moving the camera, never zooming, while only one fist moves.
    for (const dx of [0.05, 0, -0.05]) {
      result = frame(dx);
      expect(typesOf(result.events)).not.toContain('zoom');
      expect(typesOf(result.events)).toContain('orbit');
    }
  });

  it('does nothing while both fists are still', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const still = () => [
      makeHand('Left', { fist: true, dx: -0.2 }),
      makeHand('Right', { fist: true, dx: 0.2 }),
    ];
    classifier.process(still(), (t += 100));
    for (let i = 0; i < 4; i++) {
      const types = typesOf(classifier.process(still(), (t += 100)).events);
      expect(types).not.toContain('zoom');
      expect(types).not.toContain('orbit');
    }
  });

  it('respects zoomMoveSpeed: a slow drift of one fist is ignored', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, zoomMoveSpeed: 0.05 });
    let t = 0;
    let dx = 0.2;
    classifier.process(
      [makeHand('Left', { fist: true, dx: -0.2 }), makeHand('Right', { fist: true, dx })],
      (t += 100)
    );
    for (let i = 0; i < 4; i++) {
      dx -= 0.01; // 0.01 / frame is well under the 0.05 threshold
      const result = classifier.process(
        [makeHand('Left', { fist: true, dx: -0.2 }), makeHand('Right', { fist: true, dx })],
        (t += 100)
      );
      expect(typesOf(result.events)).not.toContain('zoom');
      expect(typesOf(result.events)).not.toContain('orbit');
    }
  });

  it('picks the steadier fist as the anchor, whichever hand it is', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const frame = (leftDx: number) =>
      classifier.process(
        [makeHand('Left', { fist: true, dx: leftDx }), makeHand('Right', { fist: true, dx: 0.2 })],
        (t += 100)
      );
    frame(-0.2);
    // Left moves while Right stays still: Right is the (reported) anchor and
    // Left moves the camera.
    const result = frame(-0.3);
    expect(result.metrics.zoomAnchor).toBe('Right');
    expect(result.events.find((e) => e.type === 'orbit')).toMatchObject({ hand: 'Left' });
  });

  it('switches the anchor when the other fist becomes clearly steadier', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const frame = (leftDx: number, rightDx: number) =>
      classifier.process(
        [
          makeHand('Left', { fist: true, dx: leftDx }),
          makeHand('Right', { fist: true, dx: rightDx }),
        ],
        (t += 100)
      );
    frame(-0.2, 0.2);
    let left = -0.2;
    for (let i = 0; i < 3; i++) frame((left -= 0.02), 0.2); // Left moves
    expect(frame((left -= 0.02), 0.2).metrics.zoomAnchor).toBe('Right');
    let right = 0.2;
    let result = frame(left, (right += 0.02)); // now Right moves, Left still
    for (let i = 0; i < 4; i++) result = frame(left, (right += 0.02));
    expect(result.metrics.zoomAnchor).toBe('Left');
  });

  it('turns when both fists rotate around each other (after the engage angle)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    // Steering wheel: both fists on a circle of radius 0.3 around (0.5, 0.5).
    const at = (phi: number) =>
      classifier.process(
        [
          makeHand('Left', { fist: true, dx: -0.3 * Math.cos(phi), dy: -0.3 * Math.sin(phi) }),
          makeHand('Right', { fist: true, dx: 0.3 * Math.cos(phi), dy: 0.3 * Math.sin(phi) }),
        ],
        (t += 100)
      );
    const zoomOf = (phi: number) => at(phi).events.find((e) => e.type === 'zoom');
    const angleOf = (phi: number): number => {
      const zoom = zoomOf(phi);
      return zoom?.type === 'zoom' ? zoom.deltaAngle : NaN;
    };
    at(0);
    expect(angleOf(0.05)).toBe(0); // 0.05 rad < 0.12 engage angle
    expect(angleOf(0.2)).toBeCloseTo(0.2); // engaged: whole sweep released
    expect(angleOf(0.3)).toBeCloseTo(0.1); // then per-frame deltas
    const back = zoomOf(0.25);
    if (back?.type === 'zoom') {
      expect(back.deltaAngle).toBeCloseTo(-0.05); // and back the other way
      expect(back.deltaScale).toBeCloseTo(1); // turning does not zoom
    }
  });

  it('upgrades a single-fist move to a zoom when the second fist closes', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process(
      [makeHand('Left', { dx: -0.2 }), makeHand('Right', { fist: true, dx: 0.2 })],
      (t += 100)
    );
    expect(classifier.currentState).toBe('ORBITING');
    const result = classifier.process(
      [makeHand('Left', { fist: true, dx: -0.2 }), makeHand('Right', { fist: true, dx: 0.2 })],
      (t += 100)
    );
    const types = typesOf(result.events);
    expect(types).toContain('orbit_end');
    expect(types).toContain('zoom_start');
    expect(types.indexOf('orbit_end')).toBeLessThan(types.indexOf('zoom_start'));
    expect(classifier.currentState).toBe('ZOOMING');
  });

  it('falls back to a single-fist move when one hand opens past the grace window', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const bothFists = () => [
      makeHand('Left', { fist: true, dx: -0.2 }),
      makeHand('Right', { fist: true, dx: 0.2 }),
    ];
    const leftOpen = () => [makeHand('Left', { dx: -0.2 }), makeHand('Right', { fist: true, dx: 0.2 })];
    classifier.process(bothFists(), (t += 100));
    for (let i = 0; i < 3; i++) classifier.process(leftOpen(), (t += 100));
    expect(classifier.currentState).toBe('ZOOMING');
    const result = classifier.process(leftOpen(), (t += 100));
    expect(typesOf(result.events)).toEqual([
      'zoom_end',
      'state_change',
      'state_change',
      'orbit_start',
    ]);
    expect(classifier.currentState).toBe('ORBITING');
  });

  it('ends the zoom when a zooming hand is lost', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process(
      [makeHand('Left', { fist: true, dx: -0.2 }), makeHand('Right', { fist: true, dx: 0.2 })],
      (t += 100)
    );
    // Right hand absent within grace: hold, no zoom deltas.
    for (let i = 0; i < 3; i++) {
      const held = classifier.process([makeHand('Left', { fist: true, dx: -0.2 })], (t += 100));
      expect(typesOf(held.events)).not.toContain('zoom');
      expect(classifier.currentState).toBe('ZOOMING');
    }
    const result = classifier.process([makeHand('Left', { fist: true, dx: -0.2 })], (t += 100));
    expect(typesOf(result.events)).toContain('zoom_end');
    expect(classifier.currentState).toBe('ORBITING');
  });

  it('lets a pinch end the zoom', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process(
      [makeHand('Left', { fist: true, dx: -0.2 }), makeHand('Right', { fist: true, dx: 0.2 })],
      (t += 100)
    );
    const pinching = () => [
      makeHand('Left', { fist: true, dx: -0.2 }),
      makeHand('Right', { pinchDist: 0.03, dx: 0.2 }),
    ];
    // First open frame releases the fist (a pinch never engages from a fist)…
    let result = classifier.process(pinching(), (t += 100));
    expect(typesOf(result.events)).not.toContain('pinch_start');
    // …then the pinch engages and wins over the zoom.
    result = classifier.process(pinching(), (t += 100));
    expect(typesOf(result.events)).toContain('zoom_end');
    expect(classifier.currentState).toBe('DRAWING_BASE');
  });
});

/** Pointing pose: index finger up, middle / ring / pinky folded, thumb tucked. */
const POINT: HandSpec = { fist: true, extend: [5] };

function pointingOf(classifier: GestureClassifier, hand: HandFrame, t: number): boolean {
  const result = classifier.process([hand], t);
  return result.snapshots.find((s) => s.handedness === hand.handedness)?.pointing ?? false;
}

describe('GestureClassifier — pointing pose (presses overlay buttons)', () => {
  it('detects index-up / others-curled after the debounce frames', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, pointingDebounceFrames: 2 });
    expect(pointingOf(classifier, makeHand('Right', POINT), 100)).toBe(false);
    expect(pointingOf(classifier, makeHand('Right', POINT), 200)).toBe(true);
    expect(classifier.currentState).toBe('IDLE'); // pointing is not a fist / pinch
  });

  it('is not an open palm, a fist or a pinch', () => {
    const cases: HandSpec[] = [{}, { fist: true }, { pinchDist: 0.03 }, { hook: true }];
    for (const spec of cases) {
      const classifier = new GestureClassifier({ ...TEST_OPTIONS, pointingDebounceFrames: 1 });
      let t = 0;
      let pointing = false;
      for (let i = 0; i < 3; i++) pointing ||= pointingOf(classifier, makeHand('Right', spec), (t += 100));
      expect(pointing, JSON.stringify(spec)).toBe(false);
    }
  });

  it('rejects two raised fingers (index + middle)', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, pointingDebounceFrames: 1 });
    expect(pointingOf(classifier, makeHand('Right', { fist: true, extend: [5, 9] }), 100)).toBe(false);
  });

  it('debounces a single-frame dropout while held', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, pointingDebounceFrames: 2 });
    let t = 0;
    pointingOf(classifier, makeHand('Right', POINT), (t += 100));
    pointingOf(classifier, makeHand('Right', POINT), (t += 100));
    // One open-palm frame does not drop the pose; two do.
    expect(pointingOf(classifier, makeHand('Right'), (t += 100))).toBe(true);
    expect(pointingOf(classifier, makeHand('Right', POINT), (t += 100))).toBe(true);
    pointingOf(classifier, makeHand('Right'), (t += 100));
    expect(pointingOf(classifier, makeHand('Right'), (t += 100))).toBe(false);
  });

  it('never emits pinch events while pointing', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, pointingDebounceFrames: 1 });
    let t = 0;
    const events: string[] = [];
    for (let i = 0; i < 4; i++) {
      events.push(...typesOf(classifier.process([makeHand('Right', POINT)], (t += 100)).events));
    }
    expect(events.filter((e) => e.startsWith('pinch'))).toEqual([]);
  });
});

describe('GestureClassifier — snapshot poses (open palm / OK / X cross)', () => {
  /** Run one frame and return the given hand's snapshot. */
  function snapshotOf(
    classifier: GestureClassifier,
    spec: HandSpec,
    handedness: 'Left' | 'Right' = 'Right'
  ) {
    return classifier.process([makeHand(handedness, spec)], 100).snapshots[0];
  }

  it('open palm: all fingers extended + thumb out', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    expect(snapshotOf(classifier, {}).openPalm).toBe(true);
  });

  it('open palm is not a fist, a pinch or the OK gesture', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    for (const spec of [{ fist: true }, { pinchDist: 0.03 }, POINT]) {
      expect(snapshotOf(classifier, spec).openPalm, JSON.stringify(spec)).toBe(false);
    }
  });

  it('OK gesture: thumb + index loop with the other three fingers extended', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    const snapshot = snapshotOf(classifier, { pinchDist: 0.03 });
    expect(snapshot.okGesture).toBe(true);
    expect(snapshot.pinchActive).toBe(true); // the loop is pinch-shaped
  });

  it('OK gesture rejects an open palm (thumb far from the index)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    expect(snapshotOf(classifier, {}).okGesture).toBe(false);
  });

  it('OK gesture rejects pointing and a fist (other fingers curled)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    expect(snapshotOf(classifier, POINT).okGesture).toBe(false);
    expect(snapshotOf(classifier, { fist: true }).okGesture).toBe(false);
  });

  it('X cross: index + pinky extended, middle + ring folded', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    expect(snapshotOf(classifier, { fist: true, extend: [5, 17] }).xCross).toBe(true);
  });

  it('X cross rejects an open palm, pointing and a fist', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    expect(snapshotOf(classifier, {}).xCross).toBe(false); // middle extended
    expect(snapshotOf(classifier, POINT).xCross).toBe(false); // pinky folded
    expect(snapshotOf(classifier, { fist: true }).xCross).toBe(false); // index folded
  });
});

describe('GestureClassifier — robust fist detection', () => {
  it('detects a real fist: all fingertips folded into the palm, thumb wrapped', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    const result = classifier.process([makeHand('Right', { fist: true })], 100);
    expect(typesOf(result.events)).toEqual(['state_change', 'orbit_start']);
    expect(result.snapshots[0].fistActive).toBe(true);
  });

  it('rejects a claw / hook curl (tips stay out at knuckle height)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    for (let i = 0; i < 5; i++) {
      const result = classifier.process([makeHand('Right', { hook: true })], (t += 100));
      expect(result.snapshots[0].fistActive).toBe(false);
    }
    expect(classifier.currentState).toBe('IDLE');
  });

  it('rejects a thumbs-up (fingers folded, thumb not tucked)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    for (let i = 0; i < 5; i++) {
      const result = classifier.process(
        [makeHand('Right', { fist: true, thumbUp: true })],
        (t += 100)
      );
      expect(result.snapshots[0].fistActive).toBe(false);
    }
    expect(classifier.currentState).toBe('IDLE');
  });

  it('does not enter a fist with one finger still extended (3 of 4 folded)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    for (let i = 0; i < 5; i++) {
      classifier.process([makeHand('Right', { fist: true, extend: [13] })], (t += 100));
    }
    expect(classifier.currentState).toBe('IDLE');
  });

  it('holds an engaged fist when one finger loosens (hold hysteresis)', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process([makeHand('Right', { fist: true })], (t += 100));
    expect(classifier.currentState).toBe('ORBITING');
    for (let i = 0; i < 5; i++) {
      const result = classifier.process(
        [makeHand('Right', { fist: true, extend: [13] })],
        (t += 100)
      );
      expect(result.snapshots[0].fistActive).toBe(true);
    }
    expect(classifier.currentState).toBe('ORBITING');
  });

  it('never starts a pinch from a closed fist (thumb resting on the knuckles)', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, pinchStartThreshold: 0.2 });
    const result = classifier.process([makeHand('Right', { fist: true })], 100);
    expect(typesOf(result.events)).not.toContain('pinch_start');
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
      pinchEnterFrames: 1,
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

  it('two-hand build: upper hand releases, lower hand drives height, last release ends', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    // Right hand higher in the image (smaller raw y) than the left.
    classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: 0.1 }),
        makeHand('Right', { pinchDist: 0.04, dx: 0.2, dy: -0.1 }),
      ],
      (t += 100)
    );
    // Upper (Right) hand relaxes its pinch: still EXTRUDING, no extrude_end.
    let result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: 0.1 }),
        makeHand('Right', { pinchDist: 0.2, dx: 0.2, dy: -0.1 }),
      ],
      (t += 100)
    );
    expect(typesOf(result.events)).toContain('pinch_end');
    expect(typesOf(result.events)).not.toContain('extrude_end');
    expect(classifier.currentState).toBe('EXTRUDING');

    // Lower (Left) hand drags up twice: cumulative height accumulates.
    classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: 0.05 }),
        makeHand('Right', { pinchDist: 0.2, dx: 0.2, dy: -0.1 }),
      ],
      (t += 100)
    );
    result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: 0 }),
        makeHand('Right', { pinchDist: 0.2, dx: 0.2, dy: -0.1 }),
      ],
      (t += 100)
    );
    const extrude = result.events.find((e) => e.type === 'extrude');
    expect(extrude).toMatchObject({ mode: 'single-hand', hand: 'Left' });
    if (extrude?.type === 'extrude') expect(extrude.cumulativeHeight).toBeCloseTo(0.2);

    // Lower hand releases: extrusion ends (the app commits the solid).
    result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.2, dx: -0.2, dy: 0 }),
        makeHand('Right', { pinchDist: 0.2, dx: 0.2, dy: -0.1 }),
      ],
      (t += 100)
    );
    expect(result.events.find((e) => e.type === 'extrude_end')).toMatchObject({ heightSet: true });
    expect(classifier.currentState).toBe('IDLE');
  });

  it('reports aspect-corrected horizontal / vertical pinch spans', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    // Pinch centers: Left (0.3, 0.62), Right (0.7, 0.42) in a square frame.
    const result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: 0.1 }),
        makeHand('Right', { pinchDist: 0.04, dx: 0.2, dy: -0.1 }),
      ],
      100
    );
    const extrude = result.events.find((e) => e.type === 'extrude');
    expect(extrude).toBeDefined();
    if (extrude?.type === 'extrude') {
      expect(extrude.spanX).toBeCloseTo(0.4);
      expect(extrude.spanY).toBeCloseTo(0.2);
    }
  });

  it('ends flat when the lower pinch releases first; the upper pinch is consumed', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    const upper = { pinchDist: 0.04, dx: 0.2, dy: -0.1 };
    classifier.process(
      [makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: 0.1 }), makeHand('Right', upper)],
      (t += 100)
    );
    // Lower (Left) hand releases while the upper (Right) keeps pinching.
    let result = classifier.process(
      [makeHand('Left', { pinchDist: 0.2, dx: -0.2, dy: 0.1 }), makeHand('Right', upper)],
      (t += 100)
    );
    expect(result.events.find((e) => e.type === 'extrude_end')).toMatchObject({ heightSet: false });
    expect(classifier.currentState).toBe('IDLE');

    // The still-held upper pinch neither extrudes nor starts a drawing.
    result = classifier.process(
      [makeHand('Left', { pinchDist: 0.2, dx: -0.2, dy: 0.1 }), makeHand('Right', { ...upper, dy: -0.2 })],
      (t += 100)
    );
    expect(result.events).toEqual([]);
    expect(classifier.currentState).toBe('IDLE');

    // Once released, the upper hand can pinch-draw again.
    classifier.process([makeHand('Right', { ...upper, pinchDist: 0.2 })], (t += 100));
    result = classifier.process([makeHand('Right', upper)], (t += 100));
    expect(typesOf(result.events)).toContain('pinch_start');
    expect(classifier.currentState).toBe('DRAWING_BASE');
  });

  it('ends flat when both pinches release on the same frame', () => {
    const classifier = new GestureClassifier(TEST_OPTIONS);
    let t = 0;
    classifier.process(
      [
        makeHand('Left', { pinchDist: 0.04, dx: -0.2, dy: 0.1 }),
        makeHand('Right', { pinchDist: 0.04, dx: 0.2, dy: -0.1 }),
      ],
      (t += 100)
    );
    const result = classifier.process(
      [
        makeHand('Left', { pinchDist: 0.2, dx: -0.2, dy: 0.1 }),
        makeHand('Right', { pinchDist: 0.2, dx: 0.2, dy: -0.1 }),
      ],
      (t += 100)
    );
    expect(result.events.find((e) => e.type === 'extrude_end')).toMatchObject({ heightSet: false });
    expect(classifier.currentState).toBe('IDLE');
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

describe('GestureClassifier — open-palm rotation (SELECT mode)', () => {
  it('engages when one hand pinches and the other shows an open palm', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', {})],
      (t += 100)
    );
    const rotate = result.events.find((e) => e.type === 'select_rotate');
    expect(rotate).toBeDefined();
    if (rotate?.type === 'select_rotate') {
      expect(rotate.hand).toBe('Right');
      expect(rotate.palmHand).toBe('Left');
      expect(rotate.deltaRotation).toBeCloseTo(0);
    }
    expect(classifier.currentState).toBe('SELECTING');
  });

  it('accumulates palm tilt as the anchored rotation delta', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', {})],
      (t += 100)
    );
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { tilt: 0.3 })],
      (t += 100)
    );
    const rotate = result.events.find((e) => e.type === 'select_rotate');
    expect(rotate).toBeDefined();
    if (rotate?.type === 'select_rotate') {
      expect(rotate.deltaRotation).toBeCloseTo(0.3, 5);
    }
    // Same tilt again: the cumulative delta is stable (no drift).
    const again = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { tilt: 0.3 })],
      (t += 100)
    );
    const repeat = again.events.find((e) => e.type === 'select_rotate');
    if (repeat?.type === 'select_rotate') {
      expect(repeat.deltaRotation).toBeCloseTo(0.3, 5);
    }
  });

  it('wraps the palm tilt across ±π without a jump', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { tilt: 3.0 })],
      (t += 100)
    );
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { tilt: -3.0 })],
      (t += 100)
    );
    const rotate = result.events.find((e) => e.type === 'select_rotate');
    expect(rotate).toBeDefined();
    if (rotate?.type === 'select_rotate') {
      // -6 rad of tilt, unwrapped through +2π.
      expect(rotate.deltaRotation).toBeCloseTo(-6 + 2 * Math.PI, 5);
    }
  });

  it('ends when the secondary hand closes into a pinch', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', {})],
      (t += 100)
    );
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { pinchDist: 0.04 })],
      (t += 100)
    );
    const end = result.events.find((e) => e.type === 'select_rotate_end');
    expect(end).toBeDefined();
    if (end?.type === 'select_rotate_end') {
      expect(end.palmHand).toBe('Left');
      expect(end.reason).toBe('palm closed');
    }
    expect(result.events.find((e) => e.type === 'select_rotate')).toBeUndefined();
  });

  it('ends when the pinch releases and re-anchors on the next open palm', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { tilt: 0.3 })],
      (t += 100)
    );
    const release = classifier.process(
      [makeHand('Right', { pinchDist: 0.2 }), makeHand('Left', { tilt: 0.3 })],
      (t += 100)
    );
    const end = release.events.find((e) => e.type === 'select_rotate_end');
    expect(end).toBeDefined();
    if (end?.type === 'select_rotate_end') {
      expect(end.reason).toBe('pinch released');
    }
    // Re-pinching with the palm at its new tilt restarts the anchor at 0
    // (the object keeps its yaw — the delta is always gesture-relative).
    const reEngage = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { tilt: 0.3 })],
      (t += 100)
    );
    const rotate = reEngage.events.find((e) => e.type === 'select_rotate');
    expect(rotate).toBeDefined();
    if (rotate?.type === 'select_rotate') {
      expect(rotate.deltaRotation).toBeCloseTo(0, 5);
    }
  });

  it('requires a real open palm — a fist secondary hand does not rotate', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { fist: true })],
      (t += 100)
    );
    expect(result.events.find((e) => e.type === 'select_rotate')).toBeUndefined();
    expect(result.events.find((e) => e.type === 'select_rotate_end')).toBeUndefined();
  });

  it('does not rotate outside SELECT mode', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'create' });
    let t = 0;
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', {})],
      (t += 100)
    );
    expect(result.events.find((e) => e.type === 'select_rotate')).toBeUndefined();
    expect(classifier.currentState).toBe('DRAWING_BASE');
  });

  it('ends with a mode switch while rotating', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', {})],
      (t += 100)
    );
    const events = classifier.setMode('view', (t += 100));
    const end = events.find((e) => e.type === 'select_rotate_end');
    expect(end).toBeDefined();
    if (end?.type === 'select_rotate_end') {
      expect(end.reason).toBe('mode switch');
    }
  });

  it('reports the live rotation delta in metrics', () => {
    const classifier = new GestureClassifier({ ...TEST_OPTIONS, initialMode: 'select' });
    let t = 0;
    classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', {})],
      (t += 100)
    );
    const result = classifier.process(
      [makeHand('Right', { pinchDist: 0.04 }), makeHand('Left', { tilt: -0.5 })],
      (t += 100)
    );
    expect(result.metrics.selectRotation).toBeCloseTo(-0.5, 5);
    const idle = classifier.process([makeHand('Right', { pinchDist: 0.04 })], (t += 100));
    expect(idle.metrics.selectRotation).toBeNull();
  });
});
