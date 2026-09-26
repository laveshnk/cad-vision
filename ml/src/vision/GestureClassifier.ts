/**
 * GestureClassifier: deterministic gesture classification + finite state machine.
 *
 * Pipeline per frame (input = smoothed, normalized `HandFrame`s):
 *   1. Per-hand metric updates: pinch distance (EMA + hysteresis), fist
 *      detection (per-finger curl test + pinch guard), palm-center tracking.
 *   2. Lost-hand finalization (grace frames, then synthetic release events).
 *   3. State machine: IDLE <-> DRAWING_BASE <-> EXTRUDING <-> ORBITING.
 *
 * The classifier is a pure function of its inputs: `process()` returns the
 * events to emit and never touches the DOM, which keeps it unit-testable.
 */

import { centroid, distance3, midpoint3, subtract3 } from './coordinates';
import { EmaScalar } from './filters';
import type {
  ExtrudeMode,
  GestureMetrics,
  GestureSignalEvent,
  GestureState,
  HandFrame,
  HandSnapshot,
  Handedness,
  Vec2,
  Vec3,
} from './types';

export interface GestureClassifierOptions {
  /** Thumb-tip <-> index-tip distance below which a pinch engages. Normalized units. */
  pinchStartThreshold?: number;
  /** Pinch distance above which a pinch releases (hysteresis). Normalized units. */
  pinchReleaseThreshold?: number;
  /**
   * EMA alpha applied to the pinch-distance signal before thresholding
   * (reduces flicker near the thresholds); `null` disables it.
   */
  pinchDistanceSmoothing?: number | null;
  /**
   * Fingertip-to-MCP proximity below which a finger counts as curled even
   * when the tip-vs-PIP wrist comparison is inconclusive. Normalized units.
   */
  fistCurlMcpDistance?: number;
  /**
   * Thumb-tip <-> index-tip distance below which the hand counts as
   * pinching, which vetoes fist classification. Normalized units.
   */
  fistPinchGuardDistance?: number;
  /** Consecutive fist-positive frames required to engage the FIST state. */
  fistEnterFrames?: number;
  /** Consecutive non-fist frames required to leave the FIST state. */
  fistExitFrames?: number;
  /** Frames an orbiting hand may open (palm drag) before the orbit ends. */
  orbitOpenPalmGraceFrames?: number;
  /** Frames a hand may vanish before its gesture state is finalized. */
  handLossGraceFrames?: number;
}

/** Per-hand temporal state carried across frames. */
interface HandTrack {
  handedness: Handedness;
  missedFrames: number;
  /** Pinch (hysteresis + smoothed signal). */
  pinchActive: boolean;
  pinchDistance: number;
  pinchStartPos: Vec3 | null;
  lastPinchCenter: Vec3 | null;
  pinchEma: EmaScalar | null;
  /** Fist detection (debounced). */
  fistActive: boolean;
  fistFrames: number;
  openFrames: number;
  /** Palm center (device space) used for orbit deltas. */
  palmCenter: Vec3;
  prevPalmCenter: Vec3;
  /** Consecutive non-fist frames while orbiting. */
  orbitOpenFrames: number;
}

/** Result of processing one frame. */
export interface ClassifierFrameResult {
  events: GestureSignalEvent[];
  snapshots: HandSnapshot[];
  metrics: GestureMetrics;
}

const THUMB_TIP = 4;
const INDEX_TIP = 8;
const WRIST = 0;
/** Palm center landmarks: wrist + the four MCP joints. */
const PALM_INDICES = [0, 5, 9, 13, 17];
/** [fingertip, PIP, MCP] triples (index / middle / ring / pinky) for the fist test. */
const FINGERS: ReadonlyArray<readonly [number, number, number]> = [
  [8, 6, 5],
  [12, 10, 9],
  [16, 14, 13],
  [20, 18, 17],
];

export class GestureClassifier {
  private readonly options: Required<
    Pick<
      GestureClassifierOptions,
      | 'pinchStartThreshold'
      | 'pinchReleaseThreshold'
      | 'fistCurlMcpDistance'
      | 'fistPinchGuardDistance'
      | 'fistEnterFrames'
      | 'fistExitFrames'
      | 'orbitOpenPalmGraceFrames'
      | 'handLossGraceFrames'
    >
  > & { pinchDistanceSmoothing: number | null };

  private state: GestureState = 'IDLE';
  private readonly tracks = new Map<Handedness, HandTrack>();
  private orbitHand: Handedness | null = null;

  private extrudeMode: ExtrudeMode | null = null;
  private extrudeReferenceDistance: number | null = null;
  private extrudePrevDistance: number | null = null;
  private extrudeSingleHand: Handedness | null = null;
  private extrudeHeight = 0;
  private extrudePrevY: number | null = null;

  private lastExtrudeDistance: number | null = null;
  private lastExtrudeScale: number | null = null;
  private lastExtrudeDelta: number | null = null;
  private lastOrbitDelta: Vec2 | null = null;

  constructor(options: GestureClassifierOptions = {}) {
    this.options = {
      pinchStartThreshold: options.pinchStartThreshold ?? 0.045,
      pinchReleaseThreshold: options.pinchReleaseThreshold ?? 0.065,
      // `null` explicitly disables smoothing; `??` would swallow it.
      pinchDistanceSmoothing:
        options.pinchDistanceSmoothing !== undefined ? options.pinchDistanceSmoothing : 0.5,
      fistCurlMcpDistance: options.fistCurlMcpDistance ?? 0.08,
      fistPinchGuardDistance: options.fistPinchGuardDistance ?? 0.07,
      fistEnterFrames: options.fistEnterFrames ?? 2,
      fistExitFrames: options.fistExitFrames ?? 2,
      orbitOpenPalmGraceFrames: options.orbitOpenPalmGraceFrames ?? 10,
      handLossGraceFrames: options.handLossGraceFrames ?? 3,
    };
  }

  get currentState(): GestureState {
    return this.state;
  }

  reset(): void {
    this.state = 'IDLE';
    this.tracks.clear();
    this.orbitHand = null;
    this.extrudeMode = null;
    this.extrudeReferenceDistance = null;
    this.extrudePrevDistance = null;
    this.extrudeSingleHand = null;
    this.extrudeHeight = 0;
    this.extrudePrevY = null;
    this.lastExtrudeDistance = null;
    this.lastExtrudeScale = null;
    this.lastExtrudeDelta = null;
    this.lastOrbitDelta = null;
  }

  private getOrCreateTrack(hand: HandFrame): HandTrack {
    let track = this.tracks.get(hand.handedness);
    if (!track) {
      const palmCenter = centroid(PALM_INDICES.map((i) => hand.landmarks[i].device));
      track = {
        handedness: hand.handedness,
        missedFrames: 0,
        pinchActive: false,
        pinchDistance: Infinity,
        pinchStartPos: null,
        lastPinchCenter: null,
        pinchEma:
          this.options.pinchDistanceSmoothing !== null
            ? new EmaScalar(this.options.pinchDistanceSmoothing)
            : null,
        fistActive: false,
        fistFrames: 0,
        openFrames: 0,
        palmCenter,
        prevPalmCenter: palmCenter,
        orbitOpenFrames: 0,
      };
      this.tracks.set(hand.handedness, track);
    }
    return track;
  }

  /**
   * Process one frame of conditioned hands.
   * @returns events to emit plus per-frame snapshots/metrics for the HUD.
   */
  process(hands: HandFrame[], timestamp: number): ClassifierFrameResult {
    const events: GestureSignalEvent[] = [];

    // De-duplicate by handedness (the smoother bank already guarantees this).
    const presentHands = new Map<Handedness, HandFrame>();
    for (const hand of hands) {
      if (!hand.landmarks || hand.landmarks.length < 21) continue;
      if (presentHands.has(hand.handedness)) continue;
      presentHands.set(hand.handedness, hand);
    }

    // Phase 1: update present hands.
    for (const hand of presentHands.values()) {
      const track = this.getOrCreateTrack(hand);
      track.missedFrames = 0;
      this.updatePinch(track, hand, timestamp, events);
      this.updateFist(track, hand);
      this.updatePalmCenter(track, hand);
    }

    // Phase 2: finalize hands lost beyond the grace window.
    for (const track of [...this.tracks.values()]) {
      if (presentHands.has(track.handedness)) continue;
      track.missedFrames++;
      if (track.missedFrames > this.options.handLossGraceFrames) {
        this.finalizeTrack(track, timestamp, events);
        this.tracks.delete(track.handedness);
      }
    }

    // Phase 3: state machine.
    this.updateState(presentHands, timestamp, events);

    return {
      events,
      snapshots: this.buildSnapshots(presentHands),
      metrics: this.buildMetrics(presentHands),
    };
  }

  /* ------------------------------------------------------------------------ */
  /* Phase 1: per-hand updates                                                 */
  /* ------------------------------------------------------------------------ */

  /**
   * Pinch hysteresis on the smoothed thumb-tip <-> index-tip distance:
   * engage below `pinchStartThreshold`, release above `pinchReleaseThreshold`.
   */
  private updatePinch(
    track: HandTrack,
    hand: HandFrame,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    const thumb = hand.landmarks[THUMB_TIP];
    const index = hand.landmarks[INDEX_TIP];
    const rawDistance = distance3(thumb.normalized, index.normalized);
    const dist = track.pinchEma ? track.pinchEma.filter(rawDistance) : rawDistance;
    track.pinchDistance = dist;

    const center = midpoint3(thumb.device, index.device);

    if (!track.pinchActive) {
      if (dist < this.options.pinchStartThreshold) {
        track.pinchActive = true;
        track.pinchStartPos = center;
        track.lastPinchCenter = center;
        events.push({
          type: 'pinch_start',
          timestamp,
          hand: track.handedness,
          position: center,
          distance: dist,
        });
      }
    } else {
      track.lastPinchCenter = center;
      if (dist > this.options.pinchReleaseThreshold) {
        track.pinchActive = false;
        const startPos = track.pinchStartPos ?? center;
        const endPos = center;
        events.push({
          type: 'pinch_end',
          timestamp,
          hand: track.handedness,
          startPos,
          endPos,
          delta: subtract3(endPos, startPos),
        });
        track.pinchStartPos = null;
        track.lastPinchCenter = null;
      }
    }
  }

  /**
   * Fist detection: robust, angle-invariant per-finger curl test.
   *
   * A non-thumb finger is *curled* when either
   *   1. `dist(tip, wrist) < dist(pip, wrist)` — the tip folds back toward
   *      the wrist past its own PIP joint (a distance-ratio test that holds
   *      regardless of camera angle or hand tilt), or
   *   2. `dist(tip, mcp) < fistCurlMcpDistance` — the tip is physically
   *      near its MCP joint (fully closed joint).
   *
   * A fist is confirmed when at least 3 of the 4 non-thumb fingers are
   * curled AND the hand is not pinching (thumb-tip to index-tip distance
   * above `fistPinchGuardDistance`). Temporal debouncing prevents
   * IDLE/ORBITING flicker.
   */
  private updateFist(track: HandTrack, hand: HandFrame): void {
    const wrist = hand.landmarks[WRIST].normalized;
    let curledFingers = 0;
    for (const [tip, pip, mcp] of FINGERS) {
      const tipToWrist = distance3(hand.landmarks[tip].normalized, wrist);
      const pipToWrist = distance3(hand.landmarks[pip].normalized, wrist);
      const tipToMcp = distance3(
        hand.landmarks[tip].normalized,
        hand.landmarks[mcp].normalized
      );
      if (tipToWrist < pipToWrist || tipToMcp < this.options.fistCurlMcpDistance) {
        curledFingers++;
      }
    }
    const thumbIndexDistance = distance3(
      hand.landmarks[THUMB_TIP].normalized,
      hand.landmarks[INDEX_TIP].normalized
    );
    const isFist =
      curledFingers >= 3 && thumbIndexDistance > this.options.fistPinchGuardDistance;

    if (isFist) {
      track.fistFrames++;
      track.openFrames = 0;
    } else {
      track.openFrames++;
      track.fistFrames = 0;
    }

    if (!track.fistActive && track.fistFrames >= this.options.fistEnterFrames) {
      track.fistActive = true;
    } else if (track.fistActive && track.openFrames >= this.options.fistExitFrames) {
      track.fistActive = false;
    }
  }

  /** Track the palm center (device space) used to derive orbit deltas. */
  private updatePalmCenter(track: HandTrack, hand: HandFrame): void {
    const center = centroid(PALM_INDICES.map((i) => hand.landmarks[i].device));
    // No jump delta on (re)appearance after an absence.
    track.prevPalmCenter = track.missedFrames > 0 ? center : track.palmCenter;
    track.palmCenter = center;
  }

  /** Emit synthetic release events when a hand disappears mid-gesture. */
  private finalizeTrack(track: HandTrack, timestamp: number, events: GestureSignalEvent[]): void {
    if (track.pinchActive) {
      const startPos = track.pinchStartPos ?? track.lastPinchCenter;
      if (startPos) {
        const endPos = track.lastPinchCenter ?? startPos;
        events.push({
          type: 'pinch_end',
          timestamp,
          hand: track.handedness,
          startPos,
          endPos,
          delta: subtract3(endPos, startPos),
        });
      }
      track.pinchActive = false;
      track.pinchStartPos = null;
    }
    if (this.orbitHand === track.handedness) {
      events.push({ type: 'orbit_end', timestamp, hand: track.handedness });
      this.orbitHand = null;
      this.setState('IDLE', 'orbit hand lost', timestamp, events);
    }
  }

  /* ------------------------------------------------------------------------ */
  /* Phase 3: finite state machine                                             */
  /* ------------------------------------------------------------------------ */

  private updateState(
    presentHands: Map<Handedness, HandFrame>,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    // Active pinch tracks, including hands absent within the loss-grace window.
    const activePinches = [...this.tracks.values()].filter((t) => t.pinchActive);

    // --- ORBITING: maintenance, deltas and teardown (pinch always wins). ---
    if (this.state === 'ORBITING' && this.orbitHand) {
      const track = this.tracks.get(this.orbitHand);
      const hand = presentHands.get(this.orbitHand);
      const endReason = !track
        ? 'orbit hand lost'
        : track.pinchActive
          ? 'pinch started while orbiting'
          : null;

      if (endReason) {
        events.push({ type: 'orbit_end', timestamp, hand: this.orbitHand });
        this.setState('IDLE', endReason, timestamp, events);
        this.orbitHand = null;
      } else if (hand && track) {
        // Continue while FIST is held; tolerate a brief open palm ("palm drag").
        if (!track.fistActive) {
          track.orbitOpenFrames++;
          if (track.orbitOpenFrames > this.options.orbitOpenPalmGraceFrames) {
            events.push({ type: 'orbit_end', timestamp, hand: this.orbitHand });
            this.setState('IDLE', 'orbit released (open palm)', timestamp, events);
            this.orbitHand = null;
          }
        } else {
          track.orbitOpenFrames = 0;
        }
        if (this.orbitHand) {
          const delta = subtract3(track.palmCenter, track.prevPalmCenter);
          this.lastOrbitDelta = { x: delta.x, y: delta.y };
          events.push({
            type: 'orbit',
            timestamp,
            hand: this.orbitHand,
            deltaX: delta.x,
            deltaY: delta.y,
          });
        }
      }
      // Hand absent within the grace window: hold state, emit nothing.
    }

    // --- Pinch-driven states: DRAWING_BASE / EXTRUDING. ---
    if (activePinches.length >= 2) {
      this.handleDualHandExtrude(activePinches, presentHands, timestamp, events);
    } else if (activePinches.length === 1) {
      this.handleSinglePinch(activePinches[0], presentHands, timestamp, events);
    } else if (this.state === 'DRAWING_BASE') {
      this.setState('IDLE', 'pinch released', timestamp, events);
    } else if (this.state === 'EXTRUDING') {
      events.push({
        type: 'extrude_end',
        timestamp,
        mode: this.extrudeMode ?? 'dual-hand',
      });
      this.clearExtrudeReference();
      this.setState('IDLE', 'extrusion released', timestamp, events);
    }

    // --- ORBITING engagement: only from IDLE with no active pinch. ---
    if (this.state === 'IDLE' && activePinches.length === 0) {
      const fistHand = [...presentHands.values()].find(
        (h) => this.tracks.get(h.handedness)?.fistActive
      );
      if (fistHand) {
        this.orbitHand = fistHand.handedness;
        this.setState('ORBITING', 'closed fist detected', timestamp, events);
        events.push({ type: 'orbit_start', timestamp, hand: this.orbitHand });
      }
    }
  }

  /**
   * Two-hand pinch pull: distance D between the two pinch centers
   * (`center = (P_thumb + P_index) / 2`) drives `scaleFactor` / `deltaDistance`.
   */
  private handleDualHandExtrude(
    activePinches: HandTrack[],
    presentHands: Map<Handedness, HandFrame>,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    if (this.state !== 'EXTRUDING') {
      this.setState('EXTRUDING', 'dual-hand pinch engaged', timestamp, events);
      events.push({ type: 'extrude_start', timestamp, mode: 'dual-hand' });
      this.extrudeMode = 'dual-hand';
      this.extrudeSingleHand = null;
      this.extrudeHeight = 0;
      this.extrudePrevY = null;
      this.clearExtrudeReference();
    } else if (this.extrudeMode !== 'dual-hand') {
      // Re-entering dual-hand mode (e.g. second hand re-pinched): re-reference.
      this.extrudeMode = 'dual-hand';
      this.extrudeSingleHand = null;
      this.extrudeHeight = 0;
      this.extrudePrevY = null;
      this.clearExtrudeReference();
    }

    const present = activePinches.filter((t) => presentHands.has(t.handedness));
    if (present.length < 2) return; // one hand absent within grace: hold state

    const [a, b] = present;
    const handA = presentHands.get(a.handedness);
    const handB = presentHands.get(b.handedness);
    if (!handA || !handB) return;

    const centerA = midpoint3(
      handA.landmarks[THUMB_TIP].normalized,
      handA.landmarks[INDEX_TIP].normalized
    );
    const centerB = midpoint3(
      handB.landmarks[THUMB_TIP].normalized,
      handB.landmarks[INDEX_TIP].normalized
    );
    const distance = distance3(centerA, centerB);

    if (this.extrudeReferenceDistance === null) {
      this.extrudeReferenceDistance = distance;
      this.extrudePrevDistance = distance;
    }
    const scaleFactor = distance / this.extrudeReferenceDistance;
    const deltaDistance = distance - (this.extrudePrevDistance ?? distance);
    this.extrudePrevDistance = distance;
    this.lastExtrudeDistance = distance;
    this.lastExtrudeScale = scaleFactor;
    this.lastExtrudeDelta = deltaDistance;

    events.push({
      type: 'extrude',
      timestamp,
      mode: 'dual-hand',
      distance,
      scaleFactor,
      deltaDistance,
    });
  }

  /** One active pinch: DRAWING_BASE drag, or single-hand extrusion (Condition A). */
  private handleSinglePinch(
    track: HandTrack,
    presentHands: Map<Handedness, HandFrame>,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    const hand = presentHands.get(track.handedness);
    if (!hand) return; // absent within grace: hold state, emit nothing

    if (this.state === 'EXTRUDING') {
      // Condition A: vertical delta of landmark 8 drives extrusion height.
      if (this.extrudeMode !== 'single-hand' || this.extrudeSingleHand !== track.handedness) {
        this.extrudeMode = 'single-hand';
        this.extrudeSingleHand = track.handedness;
        this.extrudeHeight = 0;
        this.extrudePrevY = hand.landmarks[INDEX_TIP].device.y;
      } else {
        const y = hand.landmarks[INDEX_TIP].device.y;
        const deltaHeight = y - (this.extrudePrevY ?? y);
        this.extrudePrevY = y;
        this.extrudeHeight += deltaHeight;
        this.lastExtrudeDistance = null;
        this.lastExtrudeScale = null;
        this.lastExtrudeDelta = null;
        events.push({
          type: 'extrude',
          timestamp,
          mode: 'single-hand',
          hand: track.handedness,
          deltaHeight,
          cumulativeHeight: this.extrudeHeight,
        });
      }
      return;
    }

    // DRAWING_BASE: emit pinch drag deltas.
    if (this.state !== 'DRAWING_BASE') {
      this.setState('DRAWING_BASE', 'index pinch engaged', timestamp, events);
    }
    const startPos = track.pinchStartPos ?? track.lastPinchCenter;
    const currentPos = track.lastPinchCenter;
    if (startPos && currentPos) {
      events.push({
        type: 'pinch_drag',
        timestamp,
        hand: track.handedness,
        currentPos,
        startPos,
        delta: subtract3(currentPos, startPos),
        distance: track.pinchDistance,
      });
    }
  }

  private setState(
    to: GestureState,
    reason: string,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    if (this.state === to) return;
    if (to !== 'EXTRUDING') {
      this.lastExtrudeDistance = null;
      this.lastExtrudeScale = null;
      this.lastExtrudeDelta = null;
    }
    if (to !== 'ORBITING') this.lastOrbitDelta = null;
    events.push({ type: 'state_change', timestamp, from: this.state, to, reason });
    this.state = to;
  }

  private clearExtrudeReference(): void {
    this.extrudeReferenceDistance = null;
    this.extrudePrevDistance = null;
    this.lastExtrudeDistance = null;
    this.lastExtrudeScale = null;
    this.lastExtrudeDelta = null;
  }

  private buildSnapshots(presentHands: Map<Handedness, HandFrame>): HandSnapshot[] {
    const snapshots: HandSnapshot[] = [];
    for (const [handedness, hand] of presentHands) {
      const track = this.tracks.get(handedness);
      snapshots.push({
        handedness,
        score: hand.score,
        pinchDistance: track?.pinchDistance ?? Number.POSITIVE_INFINITY,
        pinchActive: track?.pinchActive ?? false,
        fistActive: track?.fistActive ?? false,
        landmarks: hand.landmarks,
      });
    }
    return snapshots;
  }

  private buildMetrics(presentHands: Map<Handedness, HandFrame>): GestureMetrics {
    const pinchDistances = [...presentHands.values()].map((hand) => ({
      hand: hand.handedness,
      distance: this.tracks.get(hand.handedness)?.pinchDistance ?? Number.POSITIVE_INFINITY,
    }));
    return {
      pinchDistances,
      extrusionDistance: this.lastExtrudeDistance,
      extrusionScaleFactor: this.lastExtrudeScale,
      extrusionDeltaDistance: this.lastExtrudeDelta,
      extrusionHeight: this.state === 'EXTRUDING' ? this.extrudeHeight : null,
      orbitDelta: this.lastOrbitDelta,
    };
  }
}

