/** Public API of the CAD gesture-recognition engine. */
export { GestureEngine } from './GestureEngine';
export type { GestureEngineOptions, GestureListener, SignalListener } from './GestureEngine';
export type { SubscribableEventName, ListenerName } from './GestureEngine';
export { GestureClassifier } from './GestureClassifier';
export { PathStraightener } from './PathStraightener';
export type { PathStraightenerOptions } from './PathStraightener';
export type { GestureClassifierOptions, ClassifierFrameResult } from './GestureClassifier';
export { HandTracker, DEFAULT_WASM_CDN, DEFAULT_MODEL_CDN } from './HandTracker';
export type { HandTrackerOptions, HandResultHandler } from './HandTracker';
export { HandednessStabilizer, chiralitySign } from './HandednessStabilizer';
export type { HandednessStabilizerOptions } from './HandednessStabilizer';
export { DebugOverlay, MODE_LABELS, STATE_COLORS } from './DebugOverlay';
export type { DebugOverlayOptions, OverlayShape, OverlayShapeIcon } from './DebugOverlay';
export type {
  ArPoint,
  ArSegment,
  ArGridLine,
  ArMeshGhost,
  ArRotationRing,
  ArSceneFrame,
  ArSceneProvider,
  DragConstraint,
  OverlayBooleanTool,
  OverlayBooleanState,
  OverlayConfirmIntent,
  OverlayHudDisc,
  OverlayHudRect,
  OverlaySelectionHud,
  SelectionHudProvider,
} from './DebugOverlay';
export {
  measureHandShape,
  measureWristRoll,
  wrapAngle,
  isotropicPoints,
  palmCenter2D,
  palmAzimuth,
} from './handShape';
export type { HandShape } from './handShape';
export { HandSmootherBank, LandmarkSmoother, Vec3Smoother, EmaScalar, OneEuroScalar } from './filters';
export type {
  HandSmootherBankOptions,
  Vec3SmootherOptions,
  OneEuroOptions,
  SmoothingStrategy,
} from './filters';
export * from './coordinates';
export * from './types';
