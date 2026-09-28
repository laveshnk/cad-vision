/**
 * HandTracker: owns the webcam feed and the MediaPipe `HandLandmarker`.
 *
 * NOTE ON RUNNING MODE: the `@mediapipe/tasks-vision` web package implements
 * LIVE_STREAM-style processing for `HandLandmarker` through running mode
 * `VIDEO` + a requestAnimationFrame loop calling `detectForVideo()` with
 * monotonically increasing timestamps (the JS package exposes the async
 * LIVE_STREAM callback overloads only for other tasks). This is the official
 * MediaPipe web pattern for live webcam hand tracking and is functionally
 * identical to LIVE_STREAM: every camera frame is processed as it arrives.
 */

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { Handedness, RawHand } from './types';

/** Default CDN fallbacks used when local assets are missing. */
const DEFAULT_WASM_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10/wasm';
const DEFAULT_MODEL_CDN =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

/**
 * Default camera constraints, hard-clamped to 640 × 480 @ 30 fps: enough
 * pixels for reliable hand tracking at a fraction of the inference cost —
 * a 1280 × 720 stream carries 3× the pixels and is the usual cause of a
 * hot, throttling machine on integrated GPUs. The `max` bounds keep a
 * picky webcam driver from silently negotiating a bigger stream. User
 * constraints are merged on top of these in `start`, so explicit
 * overrides still win.
 */
const DEFAULT_CAMERA_CONSTRAINTS: MediaTrackConstraints = {
  width: { ideal: 640, max: 640 },
  height: { ideal: 480, max: 480 },
  frameRate: { ideal: 30, max: 30 },
  facingMode: 'user',
};

export interface HandTrackerOptions {
  /** Local WASM runtime location (see `npm run setup:assets`). */
  wasmBasePath?: string;
  /** Local hand_landmarker.task model location. */
  modelAssetPath?: string;
  /** CDN fallbacks used when the local assets cannot be loaded. */
  fallbackWasmBasePath?: string;
  fallbackModelAssetPath?: string;
  /** Inference delegate; GPU is preferred, CPU is the fallback. */
  delegate?: 'GPU' | 'CPU';
  numHands?: number;
  minHandDetectionConfidence?: number;
  minHandPresenceConfidence?: number;
  minTrackingConfidence?: number;
  /**
   * MediaPipe reports handedness assuming a mirrored (selfie) image, while the
   * raw getUserMedia frames fed to it are un-mirrored. With the default `true`
   * the labels are swapped so they match the user's physical hands on the
   * mirrored on-screen preview.
   */
  swapHandedness?: boolean;
  /**
   * Camera constraints, merged on top of the 640 × 480 @ 30 fps defaults
   * (see `DEFAULT_CAMERA_CONSTRAINTS`): entries here override the clamp, so
   * pass e.g. `{ facingMode: 'environment' }` or an explicit `width` to
   * deliberately request a different stream.
   */
  camera?: MediaTrackConstraints;
}

export type HandResultHandler = (hands: RawHand[], timestamp: number) => void;

interface NormalizedHandResult {
  landmarks?: { x: number; y: number; z: number; visibility: number }[][];
  handedness?: { categoryName?: string; score?: number }[][];
  handednesses?: { categoryName?: string; score?: number }[][];
}

export class HandTracker {
  private readonly options: Required<HandTrackerOptions>;

  private landmarker: HandLandmarker | null = null;
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private running = false;
  private rafId = 0;
  private lastVideoTime = -1;
  private lastTimestamp = 0;
  private resultHandler: HandResultHandler | null = null;

  constructor(options: HandTrackerOptions = {}) {
    this.options = {
      wasmBasePath: options.wasmBasePath ?? '/mediapipe/wasm',
      modelAssetPath: options.modelAssetPath ?? '/models/hand_landmarker.task',
      fallbackWasmBasePath: options.fallbackWasmBasePath ?? DEFAULT_WASM_CDN,
      fallbackModelAssetPath: options.fallbackModelAssetPath ?? DEFAULT_MODEL_CDN,
      delegate: options.delegate ?? 'GPU',
      numHands: options.numHands ?? 2,
      minHandDetectionConfidence: options.minHandDetectionConfidence ?? 0.5,
      minHandPresenceConfidence: options.minHandPresenceConfidence ?? 0.5,
      minTrackingConfidence: options.minTrackingConfidence ?? 0.5,
      swapHandedness: options.swapHandedness ?? true,
      camera: options.camera ?? DEFAULT_CAMERA_CONSTRAINTS,
    };
  }

  /** Register the per-frame callback receiving raw (un-smoothed) hands. */
  onResults(handler: HandResultHandler): void {
    this.resultHandler = handler;
  }

  get isRunning(): boolean {
    return this.running;
  }

  get videoSize(): { width: number; height: number } {
    return { width: this.video?.videoWidth ?? 0, height: this.video?.videoHeight ?? 0 };
  }

  /** Acquire the camera, load the model and start the per-frame loop. */
  async start(video: HTMLVideoElement): Promise<void> {
    if (this.running) return;
    this.video = video;

    // Clamp the camera input: 640 × 480 @ 30 fps keeps the webcam from
    // defaulting to a 720p/1080p stream (3× the pixels → 3× the per-frame
    // inference work). Explicit user constraints are spread last, so they
    // still win over the clamp.
    this.stream = await navigator.mediaDevices.getUserMedia({
      video:
        typeof this.options.camera === 'object'
          ? { ...DEFAULT_CAMERA_CONSTRAINTS, ...this.options.camera }
          : DEFAULT_CAMERA_CONSTRAINTS,
      audio: false,
    });
    video.srcObject = this.stream;
    await new Promise<void>((resolve) => {
      if (video.readyState >= HTMLMediaElement.HAVE_METADATA) return resolve();
      video.onloadedmetadata = () => resolve();
    });
    await video.play();

    // Confirm the negotiated stream in the console: a camera driver that
    // ignores the `max` bounds shows up here as 1280 × 720.
    const cameraTrack = this.stream?.getVideoTracks()[0];
    const cameraSettings = cameraTrack?.getSettings();
    console.log(
      '[HandTracker] Camera stream:',
      `${cameraSettings?.width ?? video.videoWidth}×${cameraSettings?.height ?? video.videoHeight}`,
      '@',
      cameraSettings?.frameRate ?? 'unknown',
      'fps'
    );

    this.landmarker = await this.createLandmarker();
    this.running = true;
    this.lastVideoTime = -1;
    this.lastTimestamp = 0;
    this.rafId = requestAnimationFrame(this.loop);
  }

  /** Stop the loop, release the camera and close the landmarker. */
  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    this.landmarker?.close();
    this.landmarker = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    // The result handler is kept: it is registered once (GestureEngine's
    // constructor), so clearing it would silence every later restart.
  }

  private async createLandmarker(): Promise<HandLandmarker> {
    const build = async (wasmPath: string, modelPath: string, delegate: 'GPU' | 'CPU') => {
      // Make each attempt visible in the browser console: the delegate that
      // actually initializes is the last of these lines before tracking starts.
      console.log('[HandTracker] Initializing with delegate:', delegate, wasmPath);
      const fileset = await FilesetResolver.forVisionTasks(wasmPath);
      return HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelPath, delegate },
        runningMode: 'VIDEO',
        numHands: this.options.numHands,
        minHandDetectionConfidence: this.options.minHandDetectionConfidence,
        minHandPresenceConfidence: this.options.minHandPresenceConfidence,
        minTrackingConfidence: this.options.minTrackingConfidence,
      });
    };

    // Prefer local, self-hosted assets; fall back to CDN, then CPU delegate.
    const attempts: Array<() => Promise<HandLandmarker>> = [
      () => build(this.options.wasmBasePath, this.options.modelAssetPath, this.options.delegate),
      () =>
        build(
          this.options.fallbackWasmBasePath,
          this.options.fallbackModelAssetPath,
          this.options.delegate
        ),
    ];
    if (this.options.delegate === 'GPU') {
      attempts.push(async () => {
        // Reached only after both GPU attempts threw: CPU inference is
        // several times slower and the usual cause of a hot, throttling
        // machine — warn loudly instead of degrading silently.
        console.warn('[HandTracker] Failed GPU delegate, falling back to CPU!');
        return build(
          this.options.fallbackWasmBasePath,
          this.options.fallbackModelAssetPath,
          'CPU'
        );
      });
    }

    let lastError: unknown = null;
    for (const attempt of attempts) {
      try {
        return await attempt();
      } catch (err) {
        lastError = err;
      }
    }
    throw new Error(
      `HandTracker failed to initialize HandLandmarker: ${
        lastError instanceof Error ? lastError.message : String(lastError)
      }`
    );
  }

  private readonly loop = (): void => {
    if (!this.running || !this.landmarker || !this.video) return;
    const video = this.video;

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      // Only run inference when a fresh camera frame is available.
      if (video.currentTime !== this.lastVideoTime) {
        this.lastVideoTime = video.currentTime;
        // Monotonically increasing timestamp in ms.
        const timestamp = Math.max(performance.now(), this.lastTimestamp + 1);
        this.lastTimestamp = timestamp;
        try {
          const result = this.landmarker.detectForVideo(video, timestamp);
          this.handleResult(result, timestamp);
        } catch (err) {
          console.warn('[HandTracker] detectForVideo failed:', err);
        }
      }
    }
    this.rafId = requestAnimationFrame(this.loop);
  };

  private handleResult(result: NormalizedHandResult, timestamp: number): void {
    const hands: RawHand[] = [];
    const landmarks = result.landmarks ?? [];
    // `handedness` is the current field name; older versions used `handednesses`.
    const handedness = result.handedness ?? result.handednesses ?? [];

    for (let i = 0; i < landmarks.length; i++) {
      const category = handedness[i]?.[0];
      if (!category?.categoryName) continue;
      let label: Handedness = category.categoryName === 'Left' ? 'Left' : 'Right';
      if (this.options.swapHandedness) label = label === 'Left' ? 'Right' : 'Left';
      hands.push({
        handedness: label,
        landmarks: landmarks[i].map((p) => ({
          x: p.x,
          y: p.y,
          z: p.z,
          visibility: p.visibility,
        })),
        score: category.score ?? 0,
      });
    }

    this.resultHandler?.(hands, timestamp);
  }
}
