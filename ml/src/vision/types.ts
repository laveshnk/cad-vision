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

/** High-level CAD gesture state driven by the classifier's finite state machine. */
export type GestureState = 'IDLE' | 'DRAWING_BASE' | 'EXTRUDING' | 'ORBITING';

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
}

export interface OrbitStartEvent {
  type: 'orbit_start';
  timestamp: number;
  hand: Handedness;
}

/** Camera orbit/navigation delta, in device space units. */
export interface OrbitEvent {
  type: 'orbit';
  timestamp: number;
  hand: Handedness;
  deltaX: number;
  deltaY: number;
}

export interface OrbitEndEvent {
  type: 'orbit_end';
  timestamp: number;
  hand: Handedness;
}

export interface StateChangeEvent {
  type: 'state_change';
  timestamp: number;
  from: GestureState;
  to: GestureState;
  reason: string;
}

/** Per-hand snapshot for the debug overlay / HUD. */
export interface HandSnapshot {
  handedness: Handedness;
  score: number;
  /** Current (smoothed) thumb-tip <-> index-tip distance, normalized units. */
  pinchDistance: number;
  pinchActive: boolean;
  fistActive: boolean;
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
}

/** Emitted once per processed camera frame (used by the 2D debug overlay). */
export interface FrameEvent {
  type: 'frame';
  timestamp: number;
  state: GestureState;
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
  | ExtrudeStartEvent
  | ExtrudeEvent
  | ExtrudeEndEvent
  | OrbitStartEvent
  | OrbitEvent
  | OrbitEndEvent
  | StateChangeEvent;

/** Every event the engine can emit. */
export type GestureEvent = GestureSignalEvent | FrameEvent;

export type GestureEventName = GestureSignalEvent['type'];

/** Callback that removes a subscription. */
export type Unsubscribe = () => void;
