/**
 * GestureEngine: facade wiring camera tracking, landmark conditioning and
 * gesture classification into a single event-emitting pipeline.
 *
 * Pipeline per frame:
 *   HandTracker (raw MediaPipe hands)
 *     -> HandednessStabilizer (label flicker / collision / mislabel correction)
 *     -> HandSmootherBank (EMA over all 21 landmarks, per-hand history)
 *     -> coordinate conditioning (normalized / pixel / device spaces)
 *     -> GestureClassifier (FSM: pinch/draw, extrude, orbit, zoom)
 *     -> listeners (typed `on(...)` + convenience subscriptions)
 *
 * Device space is mirrored X in [-1, 1] with +Y up — ready for CAD viewports.
 */

import { buildLandmark } from './coordinates';
import { GestureClassifier, type GestureClassifierOptions } from './GestureClassifier';
import { HandSmootherBank, type HandSmootherBankOptions } from './filters';
import { HandednessStabilizer, type HandednessStabilizerOptions } from './HandednessStabilizer';
import { HandTracker, type HandTrackerOptions } from './HandTracker';
import type {
  ExtrudeEvent,
  GestureEvent,
  GestureEventName,
  GestureSignalEvent,
  GestureState,
  HandFrame,
  OrbitEvent,
  PinchDragEvent,
  PinchEndEvent,
  PinchStartEvent,
  RawHand,
  ZoomEvent,
  Unsubscribe,
} from './types';

export interface GestureEngineOptions {
  /** MediaPipe / camera options. */
  tracker?: HandTrackerOptions;
  /** Landmark smoothing options (EMA alpha defaults to 0.35). */
  smoothing?: HandSmootherBankOptions;
  /** Handedness stabilization (label flicker / collision / mislabel correction). */
  handedness?: HandednessStabilizerOptions;
  /** Gesture classification / FSM options (pinch hysteresis, fist, etc.). */
  classifier?: GestureClassifierOptions;
}

export type GestureListener = (event: GestureEvent) => void;
export type SignalListener<T extends GestureSignalEvent> = (event: T) => void;

/** All subscribable event names (gesture signals + the per-frame debug event). */
export type SubscribableEventName = GestureEventName | 'frame';

/** All subscribable event names (gesture signals, per-frame debug, and the
 *  catch-all `'gesture'` stream). */
export type ListenerName = SubscribableEventName | 'gesture';

export class GestureEngine {
  private readonly tracker: HandTracker;
  private readonly smootherBank: HandSmootherBank;
  private readonly handednessStabilizer: HandednessStabilizer;
  private readonly classifier: GestureClassifier;

  private readonly listeners = new Map<ListenerName, Set<GestureListener>>();
  private lastFrameTime = 0;
  private fpsEma = 0;

  constructor(options: GestureEngineOptions = {}) {
    this.tracker = new HandTracker(options.tracker);
    this.handednessStabilizer = new HandednessStabilizer(options.handedness);
    this.smootherBank = new HandSmootherBank(options.smoothing);
    this.classifier = new GestureClassifier(options.classifier);
    this.tracker.onResults((hands, timestamp) => this.handleRawHands(hands, timestamp));
  }

  /** Current high-level gesture state. */
  get state(): GestureState {
    return this.classifier.currentState;
  }

  get isRunning(): boolean {
    return this.tracker.isRunning;
  }

  /** Start the camera + processing loop. Must be called from a user gesture. */
  async start(video: HTMLVideoElement): Promise<void> {
    await this.tracker.start(video);
  }

  /** Stop processing and release the camera. */
  stop(): void {
    this.tracker.stop();
    this.handednessStabilizer.reset();
    this.smootherBank.reset();
    this.classifier.reset();
    this.lastFrameTime = 0;
    this.fpsEma = 0;
  }

  /**
   * Subscribe to an event by name, or to `'gesture'` for every gesture signal
   * event (everything except `'frame'`).
   */
  on(name: ListenerName, listener: GestureListener): Unsubscribe {
    let set = this.listeners.get(name);
    if (!set) {
      set = new Set();
      this.listeners.set(name, set);
    }
    set.add(listener);
    return () => {
      set.delete(listener);
      if (set.size === 0) this.listeners.delete(name);
    };
  }

  /* ------------------------------------------------------------------------ */
  /* Convenience subscriptions                                                 */
  /* ------------------------------------------------------------------------ */

  onPinchStart(listener: SignalListener<PinchStartEvent>): Unsubscribe {
    return this.on('pinch_start', listener as GestureListener);
  }

  onPinchDrag(listener: SignalListener<PinchDragEvent>): Unsubscribe {
    return this.on('pinch_drag', listener as GestureListener);
  }

  onPinchEnd(listener: SignalListener<PinchEndEvent>): Unsubscribe {
    return this.on('pinch_end', listener as GestureListener);
  }

  /** Extrusion events (start / pull / end are all delivered as `extrude*`). */
  onExtrude(listener: SignalListener<ExtrudeEvent>): Unsubscribe {
    return this.on('extrude', listener as GestureListener);
  }

  onOrbit(listener: SignalListener<OrbitEvent>): Unsubscribe {
    return this.on('orbit', listener as GestureListener);
  }

  /** Two-fist zoom deltas (`deltaScale > 1` = fists apart, `< 1` = fists closer). */
  onZoom(listener: SignalListener<ZoomEvent>): Unsubscribe {
    return this.on('zoom', listener as GestureListener);
  }

  /* ------------------------------------------------------------------------ */
  /* Frame pipeline                                                            */
  /* ------------------------------------------------------------------------ */

  private handleRawHands(hands: RawHand[], timestamp: number): void {
    const now = timestamp || performance.now();
    const dtSeconds = this.lastFrameTime > 0 ? Math.max((now - this.lastFrameTime) / 1000, 1e-4) : 1 / 30;
    this.lastFrameTime = now;
    const instantFps = 1 / dtSeconds;
    this.fpsEma = this.fpsEma > 0 ? this.fpsEma * 0.9 + instantFps * 0.1 : instantFps;

    // 1. Stabilize handedness labels (temporal votes + geometric chirality).
    const stabilized = this.handednessStabilizer.process(hands);

    // 2. Smooth all 21 landmarks per hand (stable identity association).
    const smoothed = this.smootherBank.process(stabilized, dtSeconds);

    // 3. Condition into normalized / pixel / device coordinate spaces.
    const { width, height } = this.tracker.videoSize;
    const frames: HandFrame[] = smoothed.map((hand) => ({
      handedness: hand.handedness,
      score: hand.score,
      landmarks: hand.landmarks.map((lm) => buildLandmark(lm, width, height)),
    }));

    // 4. Classify gestures and emit signal events.
    const result = this.classifier.process(frames, now);
    for (const event of result.events) {
      this.emit(event);
    }

    // 5. Per-frame debug event for the overlay / HUD.
    this.emit({
      type: 'frame',
      timestamp: now,
      state: this.classifier.currentState,
      hands: result.snapshots,
      metrics: result.metrics,
      fps: this.fpsEma,
      video: { width, height },
    });
  }

  private emit(event: GestureEvent): void {
    // Type-specific listeners.
    const set = this.listeners.get(event.type);
    if (set) {
      for (const listener of [...set]) listener(event);
    }
    // 'gesture' listeners receive every signal event (not the high-rate frames).
    if (event.type !== 'frame') {
      const gestureSet = this.listeners.get('gesture');
      if (gestureSet) {
        for (const listener of [...gestureSet]) listener(event);
      }
    }
  }
}
