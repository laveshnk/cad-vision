/**
 * GestureClassifier: deterministic gesture classification + finite state machine.
 *
 * Pipeline per frame (input = smoothed, normalized `HandFrame`s):
 *   1. Per-hand metric updates: hand shape (fold / thumb-tuck ratios), pinch
 *      distance (EMA + hysteresis), fist detection (all fingers folded into
 *      the palm + thumb tucked, with enter/hold hysteresis), palm center.
 *   2. Lost-hand finalization (grace frames, then synthetic release events).
 *   3. State machine: IDLE <-> DRAWING_BASE <-> EXTRUDING <-> ORBITING <-> ZOOMING
 *      (+ SELECTING in select mode). One closed fist = ORBITING (camera
 *      navigation deltas); two closed fists = ZOOMING: the steadier fist is
 *      the anchor (pivot) and the other fist's motion relative to it drives
 *      zoom (distance) and turn (angle).
 *
 * The classifier is mode-partitioned (`setMode`, VIEW / SELECT / CREATE):
 * VIEW keeps pinches inert (pure navigation), SELECT maps a held pinch to the
 * SELECTING state (mesh picking, no drawing / extrusion) — and while that
 * pinch is held, an open palm on the other hand emits `select_rotate` yaw
 * deltas for the selected object — and CREATE keeps the full build gesture
 * set. Camera gestures (fist orbit / two-fist zoom) stay live in every
 * mode.
 *
 * The classifier is a pure function of its inputs: `process()` returns the
 * events to emit and never touches the DOM, which keeps it unit-testable.
 */

import { centroid, distance3, midpoint3, subtract3 } from './coordinates';
import { EmaScalar } from './filters';
import { PathStraightener, type PathStraightenerOptions } from './PathStraightener';
import {
  frameAspect,
  measureHandShape,
  measureWristRoll,
  palmAzimuth,
  palmCenter2D,
  wrapAngle,
  type HandShape,
} from './handShape';
import type {
  ExtrudeMode,
  GestureMetrics,
  GestureSignalEvent,
  GestureState,
  HandFrame,
  HandSnapshot,
  Handedness,
  InteractionMode,
  Vec2,
  Vec3,
} from './types';

export interface GestureClassifierOptions {
  /** Thumb-tip <-> index-tip distance below which a pinch engages. Normalized units. */
  pinchStartThreshold?: number;
  /** Pinch distance above which a pinch releases (hysteresis). Normalized units. */
  pinchReleaseThreshold?: number;
  /**
   * Consecutive frames the pinch must hold (below `pinchStartThreshold`)
   * before it engages — a one-frame thumb/index touch from a flickering or
   * misdetected hand never becomes a pinch. Default 2.
   */
  pinchEnterFrames?: number;
  /**
   * A pinch cannot *start* with its center within this margin of the frame
   * border (normalized units): hands half out of view / spurious detections
   * at the edges are the main source of accidental pinches. Default 0.03.
   */
  pinchEdgeMargin?: number;
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
  /**
   * Cumulative wrist roll (radians) a one-fist gesture must twist before roll
   * deltas are reported — keeps plain fist movement from rotating the view.
   */
  rollEngageAngle?: number;
  /** Per-frame roll changes smaller than this (radians) are treated as jitter. */
  rollDeadzone?: number;
  /**
   * Straightening for one-fist camera moves (see `PathStraightener`): hand
   * wiggle is removed so the camera travels in straight segments. `null`
   * passes raw palm deltas through.
   */
  cameraPath?: PathStraightenerOptions | null;
  /**
   * Two fists: the anchor (pivot) switches to the other hand only when that
   * hand's smoothed speed drops below this fraction of the anchor's speed.
   */
  zoomAnchorSwitchRatio?: number;
  /**
   * Two fists: cumulative angle (radians) the two fists must turn around
   * each other (steering-wheel) before turn deltas are reported — keeps a
   * straight pull (zoom) from also rotating the view.
   */
  zoomTurnEngageAngle?: number;
  /**
   * Two fists: a fist counts as moving when its smoothed palm speed exceeds
   * this (units of video width per frame). Both moving → zoom / turn; only
   * one moving → that fist moves the camera like a single fist; neither →
   * nothing.
   */
  zoomMoveSpeed?: number;
  /** Frames an orbiting / zooming hand may open (palm drag) before the gesture ends. */
  orbitOpenPalmGraceFrames?: number;
  /** Frames a hand may vanish before its gesture state is finalized. */
  handLossGraceFrames?: number;
  /**
   * A fingertip counts as extended when `dist(tip, wrist) >
   * openPalmExtensionRatio * dist(pip, wrist)` — the fingertip beyond its
   * own PIP joint (open-palm detection for the SELECT-mode rotation
   * gesture). Unitless ratio.
   */
  openPalmExtensionRatio?: number;
  /** Extended fingertips (of index / middle / ring / pinky) required for an open palm. */
  openPalmMinExtendedFingers?: number;
  /**
   * The open-palm hand must hold the thumb out: thumb-tip distance to the
   * nearest index / middle / ring knuckle above `openPalmThumbTuckRatio * palmSize`
   * (wrist → middle MCP) — the inverse of `fistThumbTuckRatio`.
   */
  openPalmThumbTuckRatio?: number;
  /**
   * Pointing hand (index finger up, others curled) — the only pose that
   * presses overlay UI buttons. The index counts as up when
   * `dist(tip, wrist) > pointingIndexRatio * dist(mcp, wrist)`.
   */
  pointingIndexRatio?: number;
  /**
   * Middle / ring / pinky count as curled when
   * `dist(tip, wrist) < pointingCurlRatio * dist(mcp, wrist)`.
   */
  pointingCurlRatio?: number;
  /** Threshold relaxation while a pointing pose is held (hysteresis). */
  pointingHoldSlack?: number;
  /** Consecutive frames (pointing / not pointing) required to enter / leave the pose. */
  pointingDebounceFrames?: number;
  /**
   * Initial interaction mode (VIEW / SELECT / CREATE). Defaults to
   * `'create'` — the full legacy gesture set; switch at runtime with
   * `setMode()`.
   */
  initialMode?: InteractionMode;
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
  /** Consecutive frames the pinch-start conditions have held (debounce). */
  pinchCandidateFrames: number;
  /**
   * The pinch already finished a two-hand build (the lower hand released
   * first); it is ignored until released so it cannot start a new drawing.
   */
  pinchConsumed: boolean;
  /** Fist detection (debounced). */
  fistActive: boolean;
  fistFrames: number;
  openFrames: number;
  /** Pointing pose (index up, others curled; debounced) — presses overlay UI. */
  pointingActive: boolean;
  /** Consecutive frames disagreeing with `pointingActive` (debounce counter). */
  pointingFlipFrames: number;
  /** Palm center (device space) used for orbit deltas. */
  palmCenter: Vec3;
  prevPalmCenter: Vec3;
  /** Aspect-corrected, mirrored Y-up palm center (two-fist anchor math). */
  palm2D: Vec2;
  prevPalm2D: Vec2;
  /** Smoothed palm speed (units of video width per frame) — picks the anchor. */
  speedEma: number;
  /** Wrist roll angle (radians) this / previous frame; `null` if undefined. */
  roll: number | null;
  prevRoll: number | null;
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
      | 'pinchEnterFrames'
      | 'pinchEdgeMargin'
      | 'fistFoldRatio'
      | 'fistMinFoldedFingers'
      | 'fistThumbTuckRatio'
      | 'fistHoldSlack'
      | 'fistPinchGuardDistance'
      | 'rollEngageAngle'
      | 'rollDeadzone'
      | 'zoomAnchorSwitchRatio'
      | 'zoomTurnEngageAngle'
      | 'zoomMoveSpeed'
      | 'fistEnterFrames'
      | 'fistExitFrames'
      | 'orbitOpenPalmGraceFrames'
      | 'handLossGraceFrames'
      | 'openPalmExtensionRatio'
      | 'openPalmMinExtendedFingers'
      | 'openPalmThumbTuckRatio'
      | 'pointingIndexRatio'
      | 'pointingCurlRatio'
      | 'pointingHoldSlack'
      | 'pointingDebounceFrames'
    >
  > & { pinchDistanceSmoothing: number | null };

  private state: GestureState = 'IDLE';
  /** Active interaction mode (VIEW / SELECT / CREATE). */
  private mode: InteractionMode = 'create';
  /** Straightens the one-fist camera path; `null` = raw deltas. */
  private readonly cameraPath: PathStraightener | null;
  /** Raw cumulative palm path since the one-fist gesture started. */
  private rawCameraPath: Vec2 = { x: 0, y: 0 };
  private lastCameraOut: Vec2 = { x: 0, y: 0 };
  private readonly tracks = new Map<Handedness, HandTrack>();
  private orbitHand: Handedness | null = null;

  private zoomReferenceDistance: number | null = null;
  /** False until the first two-fist frame has recorded its reference. */
  private zoomPrimed = false;
  /** The steadier fist: pivot for the moving fist. */
  private zoomAnchor: Handedness | null = null;
  private zoomAngleTotal = 0;
  private zoomTurnEngaged = false;
  /** Two-fist sub-gesture last frame: both fists zooming, or one moving the camera. */
  private zoomSubMode: 'zoom' | 'move' | null = null;
  /**
   * Fists drive the host's selected object instead of the camera (EDIT mode
   * with a selection; set by the host via `setObjectRotation`).
   */
  private objectRotation = false;
  /** Which fist drove the last two-fist object-rotation frame. */
  private rotationFist: Handedness | null = null;
  private lastZoomDistance: number | null = null;
  private lastZoomScale: number | null = null;

  private extrudeMode: ExtrudeMode | null = null;
  private extrudeReferenceDistance: number | null = null;
  private extrudePrevDistance: number | null = null;
  private extrudeSingleHand: Handedness | null = null;
  private extrudeHeight = 0;
  private extrudePrevY: number | null = null;
  /** Upper pinch (higher on screen) during the last dual-hand frame; null on a tie. */
  private extrudeUpperHand: Handedness | null = null;
  /** Timestamp of the current two-hand build's `extrude_start`. */
  private extrudeStartedAt = 0;

  private lastExtrudeDistance: number | null = null;
  private lastExtrudeScale: number | null = null;
  private lastExtrudeDelta: number | null = null;
  private lastOrbitDelta: Vec2 | null = null;
  /** Cumulative wrist roll since the current one-fist gesture started. */
  private orbitRollTotal = 0;
  private rollEngaged = false;

  /**
   * Live open-palm rotation gesture (SELECT mode): while exactly one hand
   * holds a pinch, the other hand's open-palm tilt drives the selected
   * object's yaw. `delta` is the cumulative unwrapped palm rotation since
   * the gesture began (anchored, so it always restarts at 0).
   */
  private rotation: {
    pinchHand: Handedness;
    palmHand: Handedness;
    delta: number;
    /** Previous raw palm azimuth, for frame-to-frame unwrapping. */
    prevAngle: number;
  } | null = null;

  constructor(options: GestureClassifierOptions = {}) {
    this.options = {
      pinchStartThreshold: options.pinchStartThreshold ?? 0.045,
      pinchReleaseThreshold: options.pinchReleaseThreshold ?? 0.065,
      pinchEnterFrames: options.pinchEnterFrames ?? 2,
      pinchEdgeMargin: options.pinchEdgeMargin ?? 0.03,
      // `null` explicitly disables smoothing; `??` would swallow it.
      pinchDistanceSmoothing:
        options.pinchDistanceSmoothing !== undefined ? options.pinchDistanceSmoothing : 0.5,
      fistFoldRatio: options.fistFoldRatio ?? 0.9,
      fistMinFoldedFingers: options.fistMinFoldedFingers ?? 4,
      fistThumbTuckRatio: options.fistThumbTuckRatio ?? 0.75,
      fistHoldSlack: options.fistHoldSlack ?? 0.15,
      fistPinchGuardDistance: options.fistPinchGuardDistance ?? 0.07,
      rollEngageAngle: options.rollEngageAngle ?? 0.15,
      rollDeadzone: options.rollDeadzone ?? 0.003,
      zoomAnchorSwitchRatio: options.zoomAnchorSwitchRatio ?? 0.5,
      zoomTurnEngageAngle: options.zoomTurnEngageAngle ?? 0.12,
      zoomMoveSpeed: options.zoomMoveSpeed ?? 0.004,
      fistEnterFrames: options.fistEnterFrames ?? 2,
      fistExitFrames: options.fistExitFrames ?? 2,
      orbitOpenPalmGraceFrames: options.orbitOpenPalmGraceFrames ?? 10,
      handLossGraceFrames: options.handLossGraceFrames ?? 3,
      openPalmExtensionRatio: options.openPalmExtensionRatio ?? 1.0,
      openPalmMinExtendedFingers: options.openPalmMinExtendedFingers ?? 4,
      openPalmThumbTuckRatio: options.openPalmThumbTuckRatio ?? 0.6,
      pointingIndexRatio: options.pointingIndexRatio ?? 1.5,
      pointingCurlRatio: options.pointingCurlRatio ?? 1.2,
      pointingHoldSlack: options.pointingHoldSlack ?? 0.15,
      pointingDebounceFrames: options.pointingDebounceFrames ?? 2,
    };
    this.cameraPath =
      options.cameraPath === null ? null : new PathStraightener(options.cameraPath);
    this.mode = options.initialMode ?? 'create';
  }

  get currentState(): GestureState {
    return this.state;
  }

  /**
   * Route fist gestures to the host's selected object (true) or the camera
   * (false, default). While on: one fist emits `orbit` movement deltas with
   * no wrist-roll and no path straightening (raw, live palm motion — the
   * host turns the object: up / down → screen-horizontal axis, left / right
   * → vertical axis); two fists never zoom — the faster-moving fist drives
   * the rotation, and when both move about equally fast the right hand wins.
   */
  setObjectRotation(enabled: boolean): void {
    if (this.objectRotation === enabled) return;
    this.objectRotation = enabled;
    this.rotationFist = null;
  }

  /** Active interaction mode (VIEW / SELECT / CREATE). */
  get currentMode(): InteractionMode {
    return this.mode;
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
    this.extrudeUpperHand = null;
    this.lastExtrudeDistance = null;
    this.lastExtrudeScale = null;
    this.lastExtrudeDelta = null;
    this.lastOrbitDelta = null;
    this.orbitRollTotal = 0;
    this.rollEngaged = false;
    this.rotation = null;
    this.resetCameraPath();
  }

  /**
   * Switch the interaction mode (VIEW / SELECT / CREATE). Any in-flight pinch
   * is force-released (synthetic `pinch_end`) and pinch-driven states fall
   * back to IDLE, so a mode switch never leaves a build gesture running —
   * camera gestures (orbit / zoom) are unaffected. Switching to the current
   * mode is a no-op. The mode survives `reset()`.
   * @returns the events to emit (the caller dispatches them).
   */
  setMode(mode: InteractionMode, timestamp: number = performance.now()): GestureSignalEvent[] {
    if (mode === this.mode) return [];
    const events: GestureSignalEvent[] = [];
    this.abortPinches(timestamp, events);
    this.endRotation('mode switch', timestamp, events);
    if (
      this.state === 'DRAWING_BASE' ||
      this.state === 'EXTRUDING' ||
      this.state === 'SELECTING'
    ) {
      if (this.state === 'EXTRUDING') this.clearExtrudeReference();
      this.setState('IDLE', 'mode switched', timestamp, events);
    }
    events.push({ type: 'mode_change', timestamp, from: this.mode, to: mode });
    this.mode = mode;
    return events;
  }

  /**
   * Force-release every active pinch (mode switches). Synthetic `pinch_end`s
   * are emitted only when the outgoing mode emitted their `pinch_start`.
   */
  private abortPinches(timestamp: number, events: GestureSignalEvent[]): void {
    for (const track of this.tracks.values()) {
      if (!track.pinchActive) continue;
      const startPos = track.pinchStartPos ?? track.lastPinchCenter;
      if (startPos && this.mode !== 'view') {
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
      track.pinchConsumed = false;
      track.pinchStartPos = null;
      track.lastPinchCenter = null;
    }
  }

  private resetCameraPath(): void {
    this.rawCameraPath = { x: 0, y: 0 };
    this.lastCameraOut = { x: 0, y: 0 };
    this.cameraPath?.reset();
  }

  /**
   * Straighten one frame of raw palm motion: accumulate the raw path, run it
   * through the straightener and return the change in straightened position.
   */
  private straightenCameraDelta(raw: Vec2): Vec2 {
    // Rotating a selected object needs the live, unfiltered fist motion:
    // straightening delays and re-aims it (fine for a camera glide, wrong
    // for a direct "grab and turn").
    if (!this.cameraPath || this.objectRotation) return raw;
    this.rawCameraPath = { x: this.rawCameraPath.x + raw.x, y: this.rawCameraPath.y + raw.y };
    const out = this.cameraPath.update(this.rawCameraPath);
    const delta = { x: out.x - this.lastCameraOut.x, y: out.y - this.lastCameraOut.y };
    this.lastCameraOut = out;
    return delta;
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
        pinchConsumed: false,
        pinchCandidateFrames: 0,
        fistActive: false,
        fistFrames: 0,
        openFrames: 0,
        pointingActive: false,
        pointingFlipFrames: 0,
        palmCenter,
        prevPalmCenter: palmCenter,
        palm2D: palmCenter2D(hand),
        prevPalm2D: palmCenter2D(hand),
        speedEma: 0,
        roll: null,
        prevRoll: null,
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
      const shape = measureHandShape(hand);
      this.updatePinch(track, hand, shape, timestamp, events);
      this.updateFist(track, shape);
      this.updatePointing(track, shape);
      // Reads `missedFrames` to avoid jump deltas on reappearance; reset after.
      this.updatePalmCenter(track, hand);
      track.missedFrames = 0;
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
   *
   * Pinch tracking always runs (metrics / the overlay need it), but
   * in VIEW mode no `pinch_start` / `pinch_end` events are emitted — pinches
   * are inert for pure navigation.
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
      // Debounced, in-frame start: the pinch must hold for pinchEnterFrames
      // and its center must sit inside the frame (not at the border).
      const normalizedCenter = midpoint3(thumb.normalized, index.normalized);
      const margin = this.options.pinchEdgeMargin;
      const inFrame =
        normalizedCenter.x > margin &&
        normalizedCenter.x < 1 - margin &&
        normalizedCenter.y > margin &&
        normalizedCenter.y < 1 - margin;
      if (dist < this.options.pinchStartThreshold && !fistLike && inFrame) {
        track.pinchCandidateFrames++;
      } else {
        track.pinchCandidateFrames = 0;
      }
      if (track.pinchCandidateFrames >= this.options.pinchEnterFrames) {
        track.pinchCandidateFrames = 0;
        track.pinchActive = true;
        track.pinchStartPos = center;
        track.lastPinchCenter = center;
        if (this.mode !== 'view') {
          events.push({
            type: 'pinch_start',
            timestamp,
            hand: track.handedness,
            position: center,
            distance: dist,
          });
        }
      }
    } else {
      track.lastPinchCenter = center;
      if (dist > this.options.pinchReleaseThreshold) {
        track.pinchActive = false;
        track.pinchConsumed = false;
        const startPos = track.pinchStartPos ?? center;
        const endPos = center;
        if (this.mode !== 'view') {
          events.push({
            type: 'pinch_end',
            timestamp,
            hand: track.handedness,
            startPos,
            endPos,
            delta: subtract3(endPos, startPos),
          });
        }
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

  /**
   * Pointing pose: the index finger up (tip well beyond its knuckle), the
   * middle / ring / pinky curled toward the palm, and no pinch (thumb away
   * from the index tip). Ratio-based like the fist test, so size / distance /
   * rotation invariant; thresholds relax by `pointingHoldSlack` while held,
   * and `pointingDebounceFrames` agreeing frames are needed to flip.
   */
  private updatePointing(track: HandTrack, shape: HandShape): void {
    const slack = track.pointingActive ? this.options.pointingHoldSlack : 0;
    const [index, middle, ring, pinky] = shape.foldRatios;
    const indexUp = index > this.options.pointingIndexRatio - slack;
    const curlLimit = this.options.pointingCurlRatio + slack;
    const othersCurled = middle < curlLimit && ring < curlLimit && pinky < curlLimit;
    const pointing =
      indexUp &&
      othersCurled &&
      !track.pinchActive &&
      shape.thumbIndexDistance > this.options.pinchReleaseThreshold;

    if (pointing === track.pointingActive) {
      track.pointingFlipFrames = 0;
      return;
    }
    track.pointingFlipFrames++;
    if (track.pointingFlipFrames >= this.options.pointingDebounceFrames) {
      track.pointingActive = pointing;
      track.pointingFlipFrames = 0;
    }
  }

  /** Track the palm center (device space) and wrist roll used for orbit deltas. */
  private updatePalmCenter(track: HandTrack, hand: HandFrame): void {
    const center = centroid(PALM_INDICES.map((i) => hand.landmarks[i].device));
    const roll = measureWristRoll(hand);
    const palm2D = palmCenter2D(hand);
    // No jump delta on (re)appearance after an absence.
    const reappeared = track.missedFrames > 0;
    track.prevPalmCenter = reappeared ? center : track.palmCenter;
    track.palmCenter = center;
    track.prevPalm2D = reappeared ? palm2D : track.palm2D;
    track.palm2D = palm2D;
    const speed = Math.hypot(palm2D.x - track.prevPalm2D.x, palm2D.y - track.prevPalm2D.y);
    track.speedEma = track.speedEma * 0.6 + speed * 0.4;
    track.prevRoll = reappeared ? roll : track.roll;
    track.roll = roll;
  }

  /** Emit synthetic release events when a hand disappears mid-gesture. */
  private finalizeTrack(track: HandTrack, timestamp: number, events: GestureSignalEvent[]): void {
    if (track.pinchActive) {
      const startPos = track.pinchStartPos ?? track.lastPinchCenter;
      if (startPos && this.mode !== 'view') {
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
      track.pinchConsumed = false;
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
    // VIEW mode: pinches are inert — treated as absent, so navigation never
    // enters a build state and a pinching hand cannot veto fist gestures.
    let activePinches =
      this.mode === 'view'
        ? []
        : [...this.tracks.values()].filter((t) => t.pinchActive && !t.pinchConsumed);

    // Two-hand build: the lower pinch released while the upper one is still
    // held ends the extrusion as a flat build; the upper pinch is consumed.
    if (
      activePinches.length === 1 &&
      this.state === 'EXTRUDING' &&
      this.extrudeMode === 'dual-hand' &&
      this.extrudeUpperHand === activePinches[0].handedness
    ) {
      activePinches[0].pinchConsumed = true;
      activePinches = [];
      events.push({
        type: 'extrude_end',
        timestamp,
        mode: 'dual-hand',
        heightSet: false,
        durationMs: timestamp - this.extrudeStartedAt,
      });
      this.clearExtrudeReference();
      this.setState('IDLE', 'lower pinch released first (flat)', timestamp, events);
    }

    // --- ORBITING: maintenance, deltas and teardown (pinch always wins). ---
    if (this.state === 'ORBITING' && this.orbitHand) {
      const track = this.tracks.get(this.orbitHand);
      const hand = presentHands.get(this.orbitHand);
      const endReason = !track
        ? 'orbit hand lost'
        : this.mode !== 'view' && track.pinchActive
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
          const raw = subtract3(track.palmCenter, track.prevPalmCenter);
          const delta = this.straightenCameraDelta({ x: raw.x, y: raw.y });
          // Wrist roll turns the camera; it is ignored while the fist
          // rotates a selected object (only up / down / left / right count).
          const deltaRoll = this.objectRotation ? 0 : this.rollDelta(track);
          this.lastOrbitDelta = { x: delta.x, y: delta.y };
          events.push({
            type: 'orbit',
            timestamp,
            hand: this.orbitHand,
            deltaX: delta.x,
            deltaY: delta.y,
            deltaRoll,
          });
        }
      }
      // Hand absent within the grace window: hold state, emit nothing.
    }

    // --- ZOOMING: two fists; palm-center distance drives the zoom. ---
    if (this.state === 'ZOOMING') {
      this.handleZoom(presentHands, timestamp, events);
    }

    // --- Pinch-driven states, partitioned by the interaction mode. ---
    if (this.mode === 'select') {
      // SELECT: pinches pick / drag meshes — a dedicated SELECTING state,
      // never drawing or extruding. Each active pinch keeps reporting drag
      // deltas so the app can follow the grabbed mesh.
      if (activePinches.length >= 1) {
        if (this.state !== 'SELECTING') {
          this.setState('SELECTING', 'pinch engaged (select mode)', timestamp, events);
        }
        for (const track of activePinches) {
          if (presentHands.has(track.handedness)) {
            this.emitPinchDrag(track, timestamp, events);
          }
        }
      } else if (this.state === 'SELECTING') {
        this.setState('IDLE', 'pinch released', timestamp, events);
      }
      // Secondary-hand open-palm rotation of the pinched selection.
      this.updateOpenPalmRotation(presentHands, activePinches, timestamp, events);
    } else if (activePinches.length >= 2) {
      this.handleDualHandExtrude(activePinches, presentHands, timestamp, events);
    } else if (activePinches.length === 1) {
      this.handleSinglePinch(activePinches[0], presentHands, timestamp, events);
    } else if (this.state === 'DRAWING_BASE' || this.state === 'SELECTING') {
      this.setState('IDLE', 'pinch released', timestamp, events);
    } else if (this.state === 'EXTRUDING') {
      events.push({
        type: 'extrude_end',
        timestamp,
        mode: this.extrudeMode ?? 'dual-hand',
        heightSet: this.extrudeMode === 'single-hand',
        durationMs: timestamp - this.extrudeStartedAt,
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
        this.orbitRollTotal = 0;
        this.rollEngaged = false;
        this.resetCameraPath();
        this.setState('ORBITING', 'closed fist detected', timestamp, events);
        events.push({ type: 'orbit_start', timestamp, hand: this.orbitHand });
      }
    }
  }

  /**
   * Wrist-roll delta for this frame (radians, + = counter-clockwise twist about
   * the wrist → knuckles axis as seen on screen). Nothing is reported until the
   * cumulative twist since the fist closed exceeds `rollEngageAngle` (then the
   * whole accumulated twist is released at once); after that, per-frame
   * changes below `rollDeadzone` are dropped as jitter.
   */
  private rollDelta(track: HandTrack): number {
    if (track.roll === null || track.prevRoll === null) return 0;
    const raw = wrapAngle(track.roll - track.prevRoll);
    this.orbitRollTotal += raw;
    if (!this.rollEngaged) {
      if (Math.abs(this.orbitRollTotal) < this.options.rollEngageAngle) return 0;
      this.rollEngaged = true;
      return this.orbitRollTotal;
    }
    return Math.abs(raw) < this.options.rollDeadzone ? 0 : raw;
  }

  /**
   * Two-fist navigation. Each fist counts as moving when its smoothed palm
   * speed exceeds `zoomMoveSpeed`:
   *   - **both moving** → zoom / turn from the two fists relative to each
   *     other: the change in the gap between them → `deltaScale`, and their
   *     rotation around each other (steering-wheel, + = counter-clockwise on
   *     screen) → `deltaAngle`, after `zoomTurnEngageAngle` has accumulated;
   *   - **only one moving** → that fist moves the camera exactly like a single
   *     fist (straightened `orbit` deltas); the still fist does nothing, so
   *     moving one hand never zooms;
   *   - **neither moving** → nothing.
   * The steadier fist is still reported as the `anchor` (with
   * `zoomAnchorSwitchRatio` hysteresis) for the HUD. Ends when a pinch
   * starts, a hand is lost, or one hand stays open longer than
   * `orbitOpenPalmGraceFrames`. The first frame (and the first frame after
   * a hand reappears) only records the reference, so there is no jump.
   */
  private handleZoom(
    presentHands: Map<Handedness, HandFrame>,
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    const tracks = [...this.tracks.values()];
    if (this.mode !== 'view' && tracks.some((t) => t.pinchActive)) {
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

    const present = tracks.filter((t) => presentHands.has(t.handedness));
    if (present.length < 2) {
      // One hand absent within grace: hold state, re-reference on return.
      this.zoomPrimed = false;
      return;
    }

    const [a, b] = present;
    const anchor = this.pickAnchor(a, b);
    // Gap vector a → b now and last frame (aspect-corrected, mirrored, Y up).
    const nowX = b.palm2D.x - a.palm2D.x;
    const nowY = b.palm2D.y - a.palm2D.y;
    const prevX = b.prevPalm2D.x - a.prevPalm2D.x;
    const prevY = b.prevPalm2D.y - a.prevPalm2D.y;
    const distance = Math.hypot(nowX, nowY);
    const prevDistance = Math.hypot(prevX, prevY);
    if (distance <= 1e-6 || prevDistance <= 1e-6) return;

    if (!this.zoomPrimed) {
      this.zoomPrimed = true;
      if (this.zoomReferenceDistance === null) this.zoomReferenceDistance = distance;
      this.lastZoomDistance = distance;
      this.lastZoomScale = distance / this.zoomReferenceDistance;
      return;
    }

    const aMoving = a.speedEma > this.options.zoomMoveSpeed;
    const bMoving = b.speedEma > this.options.zoomMoveSpeed;

    if (this.objectRotation && aMoving && bMoving) {
      // Object rotation: never zoom — the faster fist drives; when both move
      // about equally fast (within 25 %), the right hand wins.
      const faster =
        a.speedEma > b.speedEma * 1.25 ? a : b.speedEma > a.speedEma * 1.25 ? b : null;
      const driver = faster ?? (a.handedness === 'Right' ? a : b);
      if (this.zoomSubMode !== 'move' || this.rotationFist !== driver.handedness) {
        this.resetCameraPath();
      }
      this.zoomSubMode = 'move';
      this.rotationFist = driver.handedness;
      const raw = subtract3(driver.palmCenter, driver.prevPalmCenter);
      const delta = this.straightenCameraDelta({ x: raw.x, y: raw.y });
      this.lastOrbitDelta = { x: delta.x, y: delta.y };
      events.push({
        type: 'orbit',
        timestamp,
        hand: driver.handedness,
        deltaX: delta.x,
        deltaY: delta.y,
        deltaRoll: 0,
      });
      this.lastZoomDistance = distance;
      return;
    }
    if (aMoving !== bMoving) {
      // Only one fist moving: it moves the camera (or, with object rotation
      // on, turns the selected object) like a single fist.
      if (this.zoomSubMode !== 'move') this.resetCameraPath();
      this.zoomSubMode = 'move';
      const moving = aMoving ? a : b;
      const raw = subtract3(moving.palmCenter, moving.prevPalmCenter);
      const delta = this.straightenCameraDelta({ x: raw.x, y: raw.y });
      this.lastOrbitDelta = { x: delta.x, y: delta.y };
      events.push({
        type: 'orbit',
        timestamp,
        hand: moving.handedness,
        deltaX: delta.x,
        deltaY: delta.y,
        deltaRoll: 0,
      });
      // Keep the zoom reference current so a later two-fist zoom has no jump.
      this.lastZoomDistance = distance;
      return;
    }
    if (!aMoving) {
      this.zoomSubMode = null; // both still: nothing to do
      this.lastZoomDistance = distance;
      return;
    }

    // Both fists moving: zoom by their gap, turn by their rotation.
    this.zoomSubMode = 'zoom';
    const deltaScale = distance / prevDistance;
    const scaleFactor = distance / (this.zoomReferenceDistance ?? distance);
    const rawAngle = wrapAngle(Math.atan2(nowY, nowX) - Math.atan2(prevY, prevX));
    this.zoomAngleTotal += rawAngle;
    let deltaAngle = 0;
    if (this.zoomTurnEngaged) {
      deltaAngle = rawAngle;
    } else if (Math.abs(this.zoomAngleTotal) >= this.options.zoomTurnEngageAngle) {
      this.zoomTurnEngaged = true;
      deltaAngle = this.zoomAngleTotal;
    }
    this.lastZoomDistance = distance;
    this.lastZoomScale = scaleFactor;
    events.push({
      type: 'zoom',
      timestamp,
      anchor: anchor.handedness,
      distance,
      scaleFactor,
      deltaScale,
      deltaAngle,
    });
  }

  /** Keep the current anchor unless the other fist is clearly steadier. */
  private pickAnchor(a: HandTrack, b: HandTrack): HandTrack {
    const current = this.zoomAnchor === a.handedness ? a : this.zoomAnchor === b.handedness ? b : null;
    if (!current) {
      const steadier = a.speedEma <= b.speedEma ? a : b;
      this.zoomAnchor = steadier.handedness;
      return steadier;
    }
    const other = current === a ? b : a;
    if (other.speedEma < current.speedEma * this.options.zoomAnchorSwitchRatio) {
      this.zoomAnchor = other.handedness;
      return other;
    }
    return current;
  }

  private endZoom(reason: string, timestamp: number, events: GestureSignalEvent[]): void {
    events.push({ type: 'zoom_end', timestamp });
    this.clearZoomReference();
    this.setState('IDLE', reason, timestamp, events);
  }

  private clearZoomReference(): void {
    this.zoomReferenceDistance = null;
    this.zoomPrimed = false;
    this.zoomAnchor = null;
    this.zoomAngleTotal = 0;
    this.zoomTurnEngaged = false;
    this.zoomSubMode = null;
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
      this.extrudeStartedAt = timestamp;
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
    const spanX = Math.abs(centerA.x - centerB.x);
    const spanY = Math.abs(centerA.y - centerB.y) * frameAspect(handA);
    // Upper pinch: larger device y (+Y up); a tie leaves it undecided.
    const yA = a.lastPinchCenter?.y ?? 0;
    const yB = b.lastPinchCenter?.y ?? 0;
    this.extrudeUpperHand = yA > yB ? a.handedness : yB > yA ? b.handedness : null;

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
      spanX,
      spanY,
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
    this.emitPinchDrag(track, timestamp, events);
  }

  /** Emit a `pinch_drag` for an active pinch (drawing or selecting). */
  private emitPinchDrag(track: HandTrack, timestamp: number, events: GestureSignalEvent[]): void {
    const startPos = track.pinchStartPos ?? track.lastPinchCenter;
    const currentPos = track.lastPinchCenter;
    if (!startPos || !currentPos) return;
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

  /**
   * Secondary-hand open-palm rotation (SELECT mode): while exactly one hand
   * holds a pinch (the dominant hand picking / dragging an object) and the
   * other hand shows an open palm, the palm's tilt — the wrist → middle-MCP
   * azimuth — drives the selected object's yaw. Emits `select_rotate` with
   * the cumulative unwrapped angle since the gesture began (anchored, so
   * the host applies `rotation.y = initialRotation + deltaRotation`);
   * `select_rotate_end` fires when the pinch releases, the palm closes /
   * starts pinching, the hand is lost or the mode switches. Requires SELECT
   * mode: elsewhere the same poses route to build / navigation gestures.
   */
  private updateOpenPalmRotation(
    presentHands: Map<Handedness, HandFrame>,
    activePinches: HandTrack[],
    timestamp: number,
    events: GestureSignalEvent[]
  ): void {
    const pinchHand =
      activePinches.length === 1 && presentHands.has(activePinches[0].handedness)
        ? activePinches[0].handedness
        : null;
    let palmHand: HandFrame | null = null;
    if (pinchHand) {
      for (const hand of presentHands.values()) {
        if (hand.handedness !== pinchHand && this.isOpenPalm(hand)) {
          palmHand = hand;
          break;
        }
      }
    }

    if (!pinchHand || !palmHand) {
      const reason = activePinches.length === 0 ? 'pinch released' : 'palm closed';
      this.endRotation(reason, timestamp, events);
      return;
    }

    const angle = palmAzimuth(palmHand);
    if (!this.rotation || this.rotation.palmHand !== palmHand.handedness) {
      // Fresh gesture (or the palm moved to the other hand): end the old
      // one, then anchor the new one at the current tilt so the object
      // never jumps.
      this.endRotation('palm hand lost', timestamp, events);
      this.rotation = {
        pinchHand,
        palmHand: palmHand.handedness,
        delta: 0,
        prevAngle: angle,
      };
    } else {
      // Unwrap frame-to-frame so crossing ±π keeps rotating smoothly.
      this.rotation.delta += wrapAngle(angle - this.rotation.prevAngle);
      this.rotation.prevAngle = angle;
    }
    events.push({
      type: 'select_rotate',
      timestamp,
      hand: pinchHand,
      palmHand: palmHand.handedness,
      deltaRotation: this.rotation.delta,
    });
  }

  /** End the open-palm rotation (no-op when none is running). */
  private endRotation(reason: string, timestamp: number, events: GestureSignalEvent[]): void {
    if (!this.rotation) return;
    events.push({
      type: 'select_rotate_end',
      timestamp,
      palmHand: this.rotation.palmHand,
      reason,
    });
    this.rotation = null;
  }

  /**
   * Open palm: every fingertip (index / middle / ring / pinky) extended
   * past its own PIP joint plus the thumb held out — the deliberate "show
   * palm" pose. A fist, claw, pinch or loose curl fails at least one test.
   */
  private isOpenPalm(hand: HandFrame): boolean {
    const shape = measureHandShape(hand);
    const extended = shape.extensionRatios.filter(
      (r) => r > this.options.openPalmExtensionRatio
    ).length;
    return (
      extended >= this.options.openPalmMinExtendedFingers &&
      shape.thumbTuckRatio > this.options.openPalmThumbTuckRatio
    );
  }

  /**
   * "OK" gesture: thumb and index tips touching (a pinch-shaped loop) while
   * middle / ring / pinky stay extended past their own PIP joints — the
   * deliberate "hold to confirm" pose for in-vision dialogs.
   */
  private isOkGesture(shape: HandShape): boolean {
    const [, middle, ring, pinky] = shape.extensionRatios;
    return (
      shape.thumbIndexDistance < this.options.pinchReleaseThreshold &&
      middle > this.options.openPalmExtensionRatio &&
      ring > this.options.openPalmExtensionRatio &&
      pinky > this.options.openPalmExtensionRatio
    );
  }

  /**
   * "X" cross pose: index and pinky extended past their own PIP joints while
   * middle and ring are folded into the palm (a closed-fist fold test) —
   * the secondary-hand trigger for CSG boolean operations. A pointing hand
   * (only the index up) or an open palm fails at least one test.
   */
  private isXCross(shape: HandShape): boolean {
    const [indexExt, , , pinkyExt] = shape.extensionRatios;
    const [, middleFold, ringFold] = shape.foldRatios;
    const foldLimit = this.options.fistFoldRatio;
    return (
      indexExt > this.options.openPalmExtensionRatio &&
      pinkyExt > this.options.openPalmExtensionRatio &&
      middleFold < foldLimit &&
      ringFold < foldLimit
    );
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
    this.extrudeUpperHand = null;
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
      // Stateless per-frame poses (dialog answers + boolean trigger): all
      // ratio tests, so they are size / distance / rotation invariant.
      const shape = measureHandShape(hand);
      snapshots.push({
        handedness,
        score: hand.score,
        pinchDistance: track?.pinchDistance ?? Number.POSITIVE_INFINITY,
        pinchActive: track?.pinchActive ?? false,
        fistActive: track?.fistActive ?? false,
        pointing: track?.pointingActive ?? false,
        openPalm: this.isOpenPalm(hand),
        okGesture: this.isOkGesture(shape),
        xCross: this.isXCross(shape),
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
      orbitRoll: this.state === 'ORBITING' ? this.orbitRollTotal : null,
      zoomDistance: this.state === 'ZOOMING' ? this.lastZoomDistance : null,
      zoomScaleFactor: this.state === 'ZOOMING' ? this.lastZoomScale : null,
      zoomAnchor: this.state === 'ZOOMING' ? this.zoomAnchor : null,
      zoomAngle: this.state === 'ZOOMING' ? this.zoomAngleTotal : null,
      selectRotation: this.rotation ? this.rotation.delta : null,
    };
  }
}

