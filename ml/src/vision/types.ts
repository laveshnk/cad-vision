/**
 * Shared type definitions for the gesture-driven CAD vision module.
 *
 * Coordinate conventions used throughout:
 * - `normalized`: MediaPipe image space, x/y in [0, 1] (un-mirrored source frame).
 * - `pixel`:      Source-frame pixel coordinates.
 * - `device`:     Normalized device space [-1, 1]; X mirrored to match the on-screen
 *                 selfie view, +Y pointing up. This is the space exposed to CAD consumers.
 */

/** Handedness label (after mirroring correction, i.e. the user's physical hand). */
export type Handedness = 'Left' | 'Right';

/**
 * Application interaction mode. Partitions gesture routing:
 * - `view`:   camera navigation only — pinches are inert;
 * - `select`: pinches pick / drag committed meshes;
 * - `create`: pinches draw footprints and extrude new primitives.
 * Camera gestures (fist orbit / two-fist zoom) stay live in every mode.
 */
export type InteractionMode = 'view' | 'select' | 'create';

export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** A raw MediaPipe hand landmark (image-normalized; z is relative depth in x-scale units). */
export interface RawLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/** A raw detected hand, prior to smoothing / normalization. */
export interface RawHand {
  handedness: Handedness;
  landmarks: RawLandmark[];
  /** MediaPipe handedness classification score (0..1). */
  score: number;
}

/** A fully conditioned landmark available in several coordinate spaces. */
export interface Landmark {
  /** Smoothed, image-normalized coordinates [0, 1], un-mirrored (MediaPipe convention). */
  normalized: Vec3;
  /** Pixel coordinates within the source video frame. */
  pixel: Vec2;
  /** Normalized device space [-1, 1]; X mirrored (matches on-screen view), +Y up. */
  device: Vec3;
}

/** A conditioned hand, ready for gesture classification. */
export interface HandFrame {
  handedness: Handedness;
  score: number;
  landmarks: Landmark[];
}

/**
 * High-level CAD gesture state driven by the classifier's finite state machine.
 * `SELECTING` is the select-mode analogue of `DRAWING_BASE`: a held pinch picks
 * / drags an existing mesh instead of drawing a new one.
 */
export type GestureState =
  | 'IDLE'
  | 'DRAWING_BASE'
  | 'SELECTING'
  | 'EXTRUDING'
  | 'ORBITING'
  | 'ZOOMING';

export interface PinchStartEvent {
  type: 'pinch_start';
  timestamp: number;
  hand: Handedness;
  /** Pinch center (midpoint of thumb & index tips) in device space. */
  position: Vec3;
  /** Pinch distance in normalized units at trigger time. */
  distance: number;
}

export interface PinchDragEvent {
  type: 'pinch_drag';
  timestamp: number;
  hand: Handedness;
  /** Current pinch center in device space. */
  currentPos: Vec3;
  /** Pinch center in device space when the pinch started. */
  startPos: Vec3;
  /** `currentPos - startPos` in device space. */
  delta: Vec3;
  /** Current (hysteresis-gated) pinch distance in normalized units. */
  distance: number;
}

export interface PinchEndEvent {
  type: 'pinch_end';
  timestamp: number;
  hand: Handedness;
  startPos: Vec3;
  /** Last stable pinch center in device space before release. */
  endPos: Vec3;
  /** `endPos - startPos` in device space. */
  delta: Vec3;
}

/**
 * SELECT-mode open-palm rotation: the dominant hand holds a pinch on the
 * selected mesh while the *other* hand shows an open palm (all fingertips
 * extended + thumb out); that palm's tilt (wrist → middle-MCP azimuth)
 * drives the mesh's yaw — `selectedMesh.rotation.y = initialRotation + deltaRotation`.
 */
export interface SelectRotateEvent {
  type: 'select_rotate';
  timestamp: number;
  /** The hand holding the pinch on the selected object. */
  hand: Handedness;
  /** The open-palm hand driving the rotation. */
  palmHand: Handedness;
  /**
   * Cumulative palm rotation since the gesture began (radians, unwrapped;
   * + = counter-clockwise tilt on the mirrored screen).
   */
  deltaRotation: number;
}

export interface SelectRotateEndEvent {
  type: 'select_rotate_end';
  timestamp: number;
  /** The open-palm hand that was driving the rotation. */
  palmHand: Handedness;
  /** Why the rotation ended ('palm closed' | 'pinch released' | 'palm hand lost' | 'mode switch'). */
  reason: string;
}

export type ExtrudeMode = 'dual-hand' | 'single-hand';

export interface ExtrudeStartEvent {
  type: 'extrude_start';
  timestamp: number;
  mode: ExtrudeMode;
}

/**
 * Extrusion / scale event.
 * - `dual-hand`: both hands pinching; `scaleFactor` / `deltaDistance` / `distance`
 *   describe the pull between the two pinch centers.
 * - `single-hand`: vertical drag of the pinching hand's index tip while in the
 *   EXTRUDING state; `deltaHeight` (device units, +Y up) drives extrusion height.
 */
export interface ExtrudeEvent {
  type: 'extrude';
  timestamp: number;
  mode: ExtrudeMode;
  /** Current pull distance between pinch centers (normalized units). dual-hand only. */
  distance?: number;
  /** `distance / distance_at_extrude_start` — cumulative scale. dual-hand only. */
  scaleFactor?: number;
  /** Per-frame pull delta `D - D_prev` (normalized units). dual-hand only. */
  deltaDistance?: number;
  /**
   * Horizontal / vertical gap between the two pinch centers (absolute,
   * aspect-corrected units of video width). dual-hand only.
   */
  spanX?: number;
  spanY?: number;
  /** Per-frame vertical delta of landmark 8 in device units. single-hand only. */
  deltaHeight?: number;
  /** Cumulative vertical delta since single-hand mode began. single-hand only. */
  cumulativeHeight?: number;
  /** Which hand drives the single-hand extrusion. */
  hand?: Handedness;
}

export interface ExtrudeEndEvent {
  type: 'extrude_end';
  timestamp: number;
  mode: ExtrudeMode;
  /**
   * True when the extrusion ended from single-hand height mode (the upper
   * pinch released first, then the lower hand set the height). False when
   * both pinches released together or the lower pinch released first while
   * the upper one was still held — consumers treat that as a flat build.
   */
  heightSet: boolean;
  /**
   * How long the build lasted (ms since its `extrude_start`). Consumers can
   * drop implausibly short builds — a stray / misdetected hand that pinched
   * for an instant, rather than a deliberate two-hand build.
   */
  durationMs: number;
}

export interface OrbitStartEvent {
  type: 'orbit_start';
  timestamp: number;
  hand: Handedness;
}

/** Single-fist camera navigation delta, in device space units. */
export interface OrbitEvent {
  type: 'orbit';
  timestamp: number;
  hand: Handedness;
  deltaX: number;
  deltaY: number;
  /**
   * Wrist-roll change this frame (radians): twisting the fist like a
   * doorknob. + = counter-clockwise about the wrist → knuckles axis as seen
   * on the mirrored screen. 0 until the twist passes `rollEngageAngle`.
   */
  deltaRoll: number;
}

export interface OrbitEndEvent {
  type: 'orbit_end';
  timestamp: number;
  hand: Handedness;
}

export interface ZoomStartEvent {
  type: 'zoom_start';
  timestamp: number;
}

/**
 * Two-fist camera navigation around an anchor. The steadier fist is the
 * `anchor` (pivot); only the other fist's motion relative to it counts.
 * Distances are palm-center gaps in aspect-corrected units of video width;
 * the app zooms in as the fists move apart, out as they close, and turns the
 * scene as the moving fist circles the anchor.
 */
export interface ZoomEvent {
  type: 'zoom';
  timestamp: number;
  /** The steadier fist acting as the pivot this frame. */
  anchor: Handedness;
  /** Current anchor ↔ moving-fist palm distance. */
  distance: number;
  /** `distance / distance_at_zoom_start` — cumulative ratio. */
  scaleFactor: number;
  /** `distance / distance_prev_frame` — per-frame ratio (< 1 = fists closer, > 1 = fists apart). */
  deltaScale: number;
  /**
   * Angle (radians) the moving fist swept around the anchor this frame,
   * + = counter-clockwise on the mirrored screen. 0 until the sweep passes
   * `zoomTurnEngageAngle`.
   */
  deltaAngle: number;
}

export interface ZoomEndEvent {
  type: 'zoom_end';
  timestamp: number;
}

export interface StateChangeEvent {
  type: 'state_change';
  timestamp: number;
  from: GestureState;
  to: GestureState;
  reason: string;
}

/** The active interaction mode changed (VIEW / SELECT / CREATE). */
export interface ModeChangeEvent {
  type: 'mode_change';
  timestamp: number;
  from: InteractionMode;
  to: InteractionMode;
}

/** Per-hand snapshot for the debug overlay / HUD. */
export interface HandSnapshot {
  handedness: Handedness;
  score: number;
  /** Current (smoothed) thumb-tip <-> index-tip distance, normalized units. */
  pinchDistance: number;
  pinchActive: boolean;
  fistActive: boolean;
  /**
   * Pointing pose (index finger up, the other fingers curled, no pinch;
   * debounced) — the only pose that presses overlay UI buttons.
   */
  pointing: boolean;
  /**
   * Open palm: every fingertip extended past its own PIP joint plus the
   * thumb held out — the deliberate "show palm" pose (drives the SELECT-mode
   * rotation gesture and answers spatial dialogs).
   */
  openPalm: boolean;
  /**
   * "OK" gesture: thumb and index tips touching while middle / ring / pinky
   * stay extended — holding it answers an in-vision confirmation dialog.
   */
  okGesture: boolean;
  /**
   * "X" cross pose: index and pinky extended, middle and ring folded —
   * the secondary-hand trigger for CSG boolean operations.
   */
  xCross: boolean;
  landmarks: Landmark[];
}

/** Real-time gesture metrics for the HUD. */
export interface GestureMetrics {
  pinchDistances: { hand: Handedness; distance: number }[];
  extrusionDistance: number | null;
  extrusionScaleFactor: number | null;
  extrusionDeltaDistance: number | null;
  extrusionHeight: number | null;
  orbitDelta: Vec2 | null;
  /** Cumulative wrist roll (radians) since the one-fist gesture started. */
  orbitRoll: number | null;
  /** Anchor ↔ moving-fist palm distance while ZOOMING (units of video width). */
  zoomDistance: number | null;
  /** Cumulative zoom ratio (`distance / distance_at_zoom_start`) while ZOOMING. */
  zoomScaleFactor: number | null;
  /** The anchor (steadier) fist while ZOOMING. */
  zoomAnchor: Handedness | null;
  /** Cumulative angle (radians) the moving fist swept around the anchor. */
  zoomAngle: number | null;
  /** Cumulative open-palm rotation delta (radians) while rotating (SELECT mode). */
  selectRotation: number | null;
}

/** Emitted once per processed camera frame (used by the 2D debug overlay). */
export interface FrameEvent {
  type: 'frame';
  timestamp: number;
  state: GestureState;
  /** Active interaction mode (drives the overlay's mode buttons). */
  mode: InteractionMode;
  hands: HandSnapshot[];
  metrics: GestureMetrics;
  fps: number;
  video: { width: number; height: number };
}

/** All gesture signal events (everything except high-rate `frame` events). */
export type GestureSignalEvent =
  | PinchStartEvent
  | PinchDragEvent
  | PinchEndEvent
  | SelectRotateEvent
  | SelectRotateEndEvent
  | ExtrudeStartEvent
  | ExtrudeEvent
  | ExtrudeEndEvent
  | OrbitStartEvent
  | OrbitEvent
  | OrbitEndEvent
  | ZoomStartEvent
  | ZoomEvent
  | ZoomEndEvent
  | StateChangeEvent
  | ModeChangeEvent;

/** Every event the engine can emit. */
export type GestureEvent = GestureSignalEvent | FrameEvent;

export type GestureEventName = GestureSignalEvent['type'];

/** Callback that removes a subscription. */
export type Unsubscribe = () => void;
