/**
 * GestureClassifier: deterministic gesture classification + finite state machine.
 *
 * Pipeline per frame (input = smoothed, normalized `HandFrame`s):
 *   1. Per-hand metric updates: hand shape (fold / thumb-tuck ratios), pinch
 *      distance (EMA + hysteresis), fist detection (all fingers folded into
 *      the palm + thumb tucked, with enter/hold hysteresis), palm center.
 *   2. Lost-hand finalization (grace frames, then synthetic release events).
 *   3. State machine: IDLE <-> DRAWING_BASE <-> EXTRUDING <-> ORBITING <-> ZOOMING.
 *      One closed fist = ORBITING (camera navigation deltas); two closed
 *      fists = ZOOMING (palm-center distance drives the camera zoom).
 *
 * The classifier is a pure function of its inputs: `process()` returns the
 * events to emit and never touches the DOM, which keeps it unit-testable.
 */

import { centroid, distance3, midpoint3, subtract3 } from './coordinates';
import { EmaScalar } from './filters';
import { measureHandShape, type HandShape } from './handShape';
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
   * A finger counts as folded into the palm when
   * `dist(tip, wrist) < fistFoldRatio * dist(mcp, wrist)` — the fingertip
   * rests on the palm, below its own knuckle. A claw / hook curl keeps the
   * tip at or beyond the knuckle and fails this. Unitless ratio.
   */
  fistFoldRatio?: number;
  /** Folded fingers (of index / middle / ring / pinky) required to enter a fist. */
  fistMinFoldedFingers?: number;
  /**
   * The thumb must be tucked: thumb-tip distance to the nearest index /
   * middle / ring knuckle below `fistThumbTuckRatio * palmSize` (wrist →
   * middle MCP). Rejects thumbs-up / splayed thumbs. `Infinity` disables it.
   */
  fistThumbTuckRatio?: number;
  /**
   * Hysteresis while a fist is held: ratio thresholds are relaxed by this
   * amount and one finger may loosen, so a held fist does not flicker.
   */
  fistHoldSlack?: number;
  /**
   * Thumb-tip <-> index-tip distance below which the hand counts as
   * pinching (with the index finger not folded), which vetoes a fist.
   * Normalized units.
   */
  fistPinchGuardDistance?: number;
  /** Consecutive fist-positive frames required to engage the FIST state. */
  fistEnterFrames?: number;
  /** Consecutive non-fist frames required to leave the FIST state. */
  fistExitFrames?: number;
  /** Frames an orbiting / zooming hand may open (palm drag) before the gesture ends. */
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
/** Palm center landmarks: wrist + the four MCP joints. */
const PALM_INDICES = [0, 5, 9, 13, 17];

export class GestureClassifier {
  private readonly options: Required<
    Pick<
      GestureClassifierOptions,
      | 'pinchStartThreshold'
      | 'pinchReleaseThreshold'
      | 'fistFoldRatio'
      | 'fistMinFoldedFingers'
      | 'fistThumbTuckRatio'
      | 'fistHoldSlack'
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

  private zoomReferenceDistance: number | null = null;
  private zoomPrevDistance: number | null = null;
  private lastZoomDistance: number | null = null;
  private lastZoomScale: number | null = null;

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
      fistFoldRatio: options.fistFoldRatio ?? 0.9,
      fistMinFoldedFingers: options.fistMinFoldedFingers ?? 4,
      fistThumbTuckRatio: options.fistThumbTuckRatio ?? 0.75,
      fistHoldSlack: options.fistHoldSlack ?? 0.15,
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
    this.clearZoomReference();
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
      const shape = measureHandShape(hand);
      this.updatePinch(track, hand, shape, timestamp, events);
      this.updateFist(track, shape);
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
   * A pinch never engages while the hand is (becoming) a fist or the index
   * finger is folded into the palm — a thumb resting on a closed fist is not
   * a pinch.
   */
  private updatePinch(
    track: HandTrack,
    hand: HandFrame,
    shape: HandShape,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    const thumb = hand.landmarks[THUMB_TIP];
    const index = hand.landmarks[INDEX_TIP];
    const rawDistance = distance3(thumb.normalized, index.normalized);
    const dist = track.pinchEma ? track.pinchEma.filter(rawDistance) : rawDistance;
    track.pinchDistance = dist;

    const center = midpoint3(thumb.device, index.device);

    const fistLike =
      track.fistActive ||
      track.fistFrames > 0 ||
      shape.foldRatios[0] < this.options.fistFoldRatio;

    if (!track.pinchActive) {
      if (dist < this.options.pinchStartThreshold && !fistLike) {
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
   * Fist detection: a *real* closed fist, not just curled fingers.
   *
   * Entering requires, on the same frame:
   *   1. at least `fistMinFoldedFingers` (default all 4) fingers folded into
   *      the palm: `dist(tip, wrist) < fistFoldRatio * dist(mcp, wrist)`.
   *      Claw / hook curls and half-curls keep the tips at or beyond the
   *      knuckles and fail this;
   *   2. the thumb tucked over / alongside the fingers
   *      (`thumbTuckRatio < fistThumbTuckRatio`) — rejects a thumbs-up;
   *   3. not a pinch (thumb-index guard, only while the index is unfolded).
   *
   * All tests are ratios within the hand (size / distance / rotation
   * invariant). Once engaged, thresholds relax by `fistHoldSlack` and one
   * finger may loosen, and frame debouncing (`fistEnterFrames` /
   * `fistExitFrames`) prevents IDLE/ORBITING flicker.
   */
  private updateFist(track: HandTrack, shape: HandShape): void {
    const slack = track.fistActive ? this.options.fistHoldSlack : 0;
    const foldLimit = this.options.fistFoldRatio + slack;
    const minFolded = track.fistActive
      ? Math.max(1, this.options.fistMinFoldedFingers - 1)
      : this.options.fistMinFoldedFingers;
    const folded = shape.foldRatios.filter((r) => r < foldLimit).length;
    const thumbTucked = shape.thumbTuckRatio < this.options.fistThumbTuckRatio + slack;
    const indexFolded = shape.foldRatios[0] < foldLimit;
    const pinching =
      !indexFolded && shape.thumbIndexDistance < this.options.fistPinchGuardDistance;
    const isFist = folded >= minFolded && thumbTucked && !pinching;

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
    if (this.state === 'ZOOMING') {
      this.endZoom('zoom hand lost', timestamp, events);
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

    // --- ZOOMING: two fists; palm-center distance drives the zoom. ---
    if (this.state === 'ZOOMING') {
      this.handleZoom(presentHands, timestamp, events);
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

    // --- Fist engagement: only with no active pinch. Two fists zoom (also
    // upgrading a running single-fist orbit); one fist from IDLE orbits. ---
    if (activePinches.length === 0 && (this.state === 'IDLE' || this.state === 'ORBITING')) {
      const fistHands = [...presentHands.values()].filter(
        (h) => this.tracks.get(h.handedness)?.fistActive
      );
      if (fistHands.length >= 2) {
        if (this.state === 'ORBITING' && this.orbitHand) {
          events.push({ type: 'orbit_end', timestamp, hand: this.orbitHand });
          this.orbitHand = null;
        }
        this.clearZoomReference();
        this.setState('ZOOMING', 'two closed fists detected', timestamp, events);
        events.push({ type: 'zoom_start', timestamp });
        for (const track of this.tracks.values()) track.orbitOpenFrames = 0;
        this.handleZoom(presentHands, timestamp, events);
      } else if (fistHands.length === 1 && this.state === 'IDLE') {
        this.orbitHand = fistHands[0].handedness;
        this.setState('ORBITING', 'closed fist detected', timestamp, events);
        events.push({ type: 'orbit_start', timestamp, hand: this.orbitHand });
      }
    }
  }

  /**
   * Two-fist zoom: distance between the two palm centers (normalized space).
   * Ends when a pinch starts, a hand is lost, or one hand stays open longer
   * than `orbitOpenPalmGraceFrames`. The first frame (and the first frame
   * after a hand reappears) only records the reference, so there is no jump.
   */
  private handleZoom(
    presentHands: Map<Handedness, HandFrame>,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    const tracks = [...this.tracks.values()];
    if (tracks.some((t) => t.pinchActive)) {
      this.endZoom('pinch started while zooming', timestamp, events);
      return;
    }
    if (tracks.length < 2) {
      this.endZoom('zoom hand lost', timestamp, events);
      return;
    }
    for (const track of tracks) {
      if (!presentHands.has(track.handedness)) continue;
      track.orbitOpenFrames = track.fistActive ? 0 : track.orbitOpenFrames + 1;
      if (track.orbitOpenFrames > this.options.orbitOpenPalmGraceFrames) {
        this.endZoom('zoom released (open palm)', timestamp, events);
        return;
      }
    }

    const present = [...presentHands.values()];
    if (present.length < 2) {
      // One hand absent within grace: hold state, re-reference on return.
      this.zoomPrevDistance = null;
      return;
    }

    const [a, b] = present;
    const distance = distance3(
      centroid(PALM_INDICES.map((i) => a.landmarks[i].normalized)),
      centroid(PALM_INDICES.map((i) => b.landmarks[i].normalized))
    );
    if (distance <= 1e-6) return;
    if (this.zoomReferenceDistance === null) this.zoomReferenceDistance = distance;
    if (this.zoomPrevDistance === null) {
      this.zoomPrevDistance = distance;
      this.lastZoomDistance = distance;
      this.lastZoomScale = distance / this.zoomReferenceDistance;
      return;
    }

    const deltaScale = distance / this.zoomPrevDistance;
    const scaleFactor = distance / this.zoomReferenceDistance;
    this.zoomPrevDistance = distance;
    this.lastZoomDistance = distance;
    this.lastZoomScale = scaleFactor;
    events.push({ type: 'zoom', timestamp, distance, scaleFactor, deltaScale });
  }

  private endZoom(reason: string, timestamp: number, events: GestureSignalEvent[]): void {
    events.push({ type: 'zoom_end', timestamp });
    this.clearZoomReference();
    this.setState('IDLE', reason, timestamp, events);
  }

  private clearZoomReference(): void {
    this.zoomReferenceDistance = null;
    this.zoomPrevDistance = null;
    this.lastZoomDistance = null;
    this.lastZoomScale = null;
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
      zoomDistance: this.state === 'ZOOMING' ? this.lastZoomDistance : null,
      zoomScaleFactor: this.state === 'ZOOMING' ? this.lastZoomScale : null,
    };
  }
}

