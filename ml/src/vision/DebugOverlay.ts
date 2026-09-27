/**
 * DebugOverlay: 2D-canvas debug renderer for the gesture engine.
 *
 * Draws the mirrored webcam landmarks (21 per hand) with the MediaPipe skeleton
 * and pinch indicators. Pure Canvas 2D — no WebGL.
 *
 * Along the top edge it renders the interaction-mode switcher ([ VIEW ],
 * [ EDIT ], [ CREATE ] — the `select` mode is labeled EDIT): boxy, mutually
 * exclusive toggle buttons reported through the `onModeRequest` callback.
 * Each mode's options sit under its own button. In VIEW mode the trash bin
 * sits below [ VIEW ] (`onTrashRequest`). In EDIT mode a stack of mutually
 * exclusive toggles, [ XZ PLANE ] (default) and [ Y AXIS ], sits vertically
 * below [ EDIT ] (`onDragConstraintRequest`), followed by the CSG Boolean
 * tool toggles `[ SUBTRACT ]` / `[ UNION ]` (`onBooleanToolRequest`). In
 * CREATE mode a column of shape icon buttons (cube / cuboid / cylinder /
 * sphere, from the `shapes` option) is stacked down the right edge, under
 * [ CREATE ] (`onShapeRequest`).
 *
 * Every button responds to a mouse click or to a **pointing** hand (index
 * finger up, other fingers curled — `HandSnapshot.pointing`) holding its
 * index tip over the button for `dwellMs`; a ring marks a pointing
 * fingertip. Pinches, fists and open palms never press a toggle button, so
 * moving, editing or building objects can't switch modes / shapes by
 * accident — including the VIEW-mode trash bin (`onTrashRequest`), which
 * clears the scene through the confirmation dialog below.
 *
 * Destructive actions never open browser popups or DOM modals: the trash bin
 * (a scene clear) and the selection's Delete button open an in-vision
 * spatial dialog on this canvas — two spatial targets,
 * `[ CONFIRM (Hold) ]` and `[ CANCEL (Hold) ]`. Confirming is a pointing
 * fingertip held on the confirm target (or the "OK" gesture — thumb + index
 * loop, other fingers extended — held anywhere) for `confirmHoldMs`, the
 * same wait as the color wheel lock and the Delete button, or a mouse
 * click; a pinch never confirms instantly. Canceling is a pointing hold on
 * the cancel target, an open palm held briefly, or moving every hand away. While the dialog is open every other overlay interaction
 * is frozen and the host freezes the scene (see `confirmActive`).
 *
 * The live stats (MODE / STATE / FPS / HANDS) are *not* drawn here — the host
 * mounts them as a DOM metrics bar below the camera view (see
 * `src/ui/MetricsBar.ts`); this canvas stays dedicated to the camera image.
 *
 * Alignment: the <video> is CSS-mirrored (`scaleX(-1)`) and displayed with
 * `object-fit: cover`, which center-crops it into its container (the floating
 * camera thumbnail, or that thumbnail expanded). The canvas backing store
 * therefore matches the *container* (CSS px × devicePixelRatio, re-measured
 * every frame), and landmarks are mapped through the same mirrored cover
 * transform so they land exactly on the webcam image — while the UI buttons
 * stay anchored to the visible container edges (never cropped). Typography
 * scales with the container width (see `fontScale`).
 *
 * In SELECT mode the overlay additionally renders a live AR spatial mirror
 * of the 3D CAD scene (translucent ground grid + mesh ghosts) through the
 * `arScene` provider — plain projected 2D data supplied by the host, so this
 * module stays free of Three.js. It also mirrors the selection's floating
 * controls (color-wheel disc outline + Delete button, device-space data via
 * the `selectionHud` provider) so users can visually align their hands with
 * the viewport-anchored UI.
 */

import { HandLandmarker } from '@mediapipe/tasks-vision';
import type {
  FrameEvent,
  GestureState,
  HandSnapshot,
  InteractionMode,
} from './types';
import { DwellClock } from './DwellClock';

const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;

/** A skeleton connection between two landmark indices. */
export interface SkeletonConnection {
  start: number;
  end: number;
}

interface HandStyle {
  skeleton: string;
  joint: string;
  jointFill: string;
  label: string;
}

function styleFor(hand: HandSnapshot, state: GestureState): HandStyle {
  if (state === 'SELECTING') {
    return { skeleton: '#06b6d4', joint: '#67e8f9', jointFill: '#a5f3fc', label: '#06b6d4' };
  }
  if (hand.pinchActive || state === 'DRAWING_BASE' || state === 'EXTRUDING') {
    return { skeleton: '#22c55e', joint: '#86efac', jointFill: '#bbf7d0', label: '#22c55e' };
  }
  if (state === 'ZOOMING') {
    return { skeleton: '#a855f7', joint: '#d8b4fe', jointFill: '#e9d5ff', label: '#a855f7' };
  }
  if (hand.fistActive || state === 'ORBITING') {
    return { skeleton: '#3b82f6', joint: '#93c5fd', jointFill: '#bfdbfe', label: '#3b82f6' };
  }
  return { skeleton: '#facc15', joint: '#fde047', jointFill: '#fef08a', label: '#facc15' };
}

/** Mirrored `object-fit: cover` mapping from video-bitmap coords to panel CSS px. */
interface ViewTransform {
  /** Horizontal offset of the cropped video inside the panel. */
  ox: number;
  /** Vertical offset of the cropped video inside the panel. */
  oy: number;
  /** Cover-scaled video width in CSS pixels. */
  dispW: number;
  /** Cover-scaled video height in CSS pixels. */
  dispH: number;
}

/* -------------------------------------------------------------------- */
/* AR spatial mirror (SELECT mode)                                      */
/* -------------------------------------------------------------------- */

/**
 * SELECT-mode drag constraint, mirrored structurally from the CAD builder's
 * `DragConstraint` (no shared import — the orchestrator bridges the two):
 * `'xz'` slides the grabbed mesh across the ground plane, `'y'` maps
 * vertical hand travel to a world-Y lift / lower.
 */
export type DragConstraint = 'xz' | 'y';

/**
 * CSG Boolean tool, mirrored structurally from the CAD builder's
 * `BooleanOperation` (no shared import — the orchestrator bridges the two):
 * `'subtract'` carves the selected cutter out of the intersected base mesh,
 * `'union'` merges both solids into one continuous body.
 */
export type OverlayBooleanTool = 'subtract' | 'union';

/** The Boolean tool toggles, top to bottom (below the constraint stack). */
const BOOLEAN_BUTTONS: readonly OverlayBooleanTool[] = ['subtract', 'union'];

/**
 * Live CSG Boolean state supplied by the host per frame (plain data): which
 * tool is armed and whether the selection currently clashes with another
 * mesh (the operation could run). Null skips the layer.
 */
export interface OverlayBooleanState {
  /** Armed tool, or null when no Boolean tool is active. */
  tool: OverlayBooleanTool | null;
  /** Whether the selected mesh intersects another committed mesh. */
  clash: boolean;
  /** Whether the selection is a union that can be ungrouped. */
  canUngroup?: boolean;
}

/**
 * What the in-vision spatial confirmation dialog is asking about. The host
 * maps the intent onto its destructive action (clear the scene, delete the
 * selected object); the overlay only renders it and reports the answer.
 */
export type OverlayConfirmIntent = 'clear-scene' | 'delete-selection';

/** Dialog titles per intent (drawn above the two spatial targets). */
const CONFIRM_TITLES: Record<OverlayConfirmIntent, string> = {
  'clear-scene': 'CLEAR SCENE?',
  'delete-selection': 'DELETE OBJECT?',
};


/**
 * Plain-data AR mirror payloads, structurally compatible with
 * `src/cad/ArMirror.ts`'s `ArSceneFrame` (the same pattern as the CAD side's
 * `CadExtrudeInput`) — `main.ts` forwards the CAD-side projection directly
 * into the `arScene` provider, keeping the vision layer free of Three.js.
 */

/** A projected 2D canvas point (CSS px, +Y down). */
export interface ArPoint {
  x: number;
  y: number;
}

/** One projected mesh edge segment (endpoints in 2D canvas space). */
export type ArSegment = [ArPoint, ArPoint];

/** One projected ground-grid line (straight 3D lines stay straight). */
export interface ArGridLine {
  points: ArPoint[];
  /** Major division — rendered stronger than the 1-unit minor lines. */
  major: boolean;
}

/** Translucent ghost of one committed mesh, projected to canvas space. */
export interface ArMeshGhost {
  /** Projected silhouette polygon (convex hull of the projected vertices). */
  hull: ArPoint[];
  /** Projected `EdgesGeometry` segments (crisp accent lines). */
  edges: ArSegment[];
  /** The active selection gets an energetic highlight. */
  selected: boolean;
}

/**
 * Rotational compass ring around the selected object while an open-palm
 * rotation gesture is running (SELECT mode).
 */
export interface ArRotationRing {
  /** Projected circle segments (both endpoints visible). */
  circle: ArSegment[];
  /** Projected yaw indicator: object center → ring edge (null when culled). */
  needle: ArSegment | null;
}

/** One AR mirror frame: projected ground grid + mesh ghosts. */
export interface ArSceneFrame {
  /** Projected ground-grid lines, minor and major divisions. */
  grid: ArGridLine[];
  /** Ghosts of the committed meshes, in scene order. */
  meshes: ArMeshGhost[];
  /** Compass ring around the selection while rotating, or null. */
  rotationRing: ArRotationRing | null;
}

/**
 * Projects the CAD scene onto the overlay — called once per rendered frame
 * (only in SELECT mode) with the size (CSS px) of the webcam image's
 * on-screen rect (the cover-scaled video, which may overhang the canvas);
 * the overlay offsets the result into place. The projection must use the
 * webcam frame's aspect so proportions stay true. Returns `null` to skip
 * the AR layer entirely.
 */
export type ArSceneProvider = (width: number, height: number) => ArSceneFrame | null;

/* -------------------------------------------------------------------- */
/* SELECT-mode selection-HUD mirror (color wheel + Delete button)       */
/* -------------------------------------------------------------------- */

/**
 * A disc in device space ([-1, 1], +X right in the mirrored view, +Y up):
 * center plus per-axis radii — an ellipse, since device units map onto the
 * video rect's width / height separately. Mirrors the color wheel.
 */
export interface OverlayHudDisc {
  x: number;
  y: number;
  radiusX: number;
  radiusY: number;
}

/**
 * A rectangle in device space: (x, y) is its minimum corner (left edge,
 * bottom edge — +Y up), with width / height extending toward +X / +Y.
 * Mirrors the selection HUD's Delete button.
 */
export interface OverlayHudRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * One frame of the SELECT-mode selection HUD, expressed in device space so
 * the drawn controls sit exactly where a tracked fingertip must point to hit
 * the real (viewport-anchored) controls.
 */
export interface OverlaySelectionHud {
  /** Color-wheel disc outline, or null while the wheel is dismissed. */
  wheel: OverlayHudDisc | null;
  /** Delete-button bounds, or null while the selection HUD is hidden. */
  deleteButton: OverlayHudRect | null;
  /** Color-lock dwell completion [0, 1] (fills the wheel outline ring). */
  wheelProgress: number;
  /** Delete-button dwell completion [0, 1] (fills the delete HUD). */
  deleteProgress: number;
}

/**
 * Supplies the selection HUD per rendered frame; null skips the layer (no
 * selection, outside SELECT mode, or while the delete confirmation is open).
 * Plain data from the host — the overlay stays free of UI dependencies.
 */
export type SelectionHudProvider = () => OverlaySelectionHud | null;

/** AR mirror paint: minor ground-grid strokes (1-unit lines). */
const AR_GRID_STROKE = 'rgba(0, 150, 255, 0.18)';
/** AR mirror paint: major ground-grid divisions (stronger, every 5 units). */
const AR_GRID_MAJOR_STROKE = 'rgba(0, 150, 255, 0.4)';
/** AR mirror paint: ghost fill for unselected meshes. */
const AR_GHOST_FILL = 'rgba(40, 120, 200, 0.35)';
/** AR mirror paint: ghost fill for the active selection (higher opacity). */
const AR_SELECTED_FILL = 'rgba(40, 120, 200, 0.5)';
/** AR mirror paint: crisp edge accents. */
const AR_EDGE_STROKE = 'rgba(0, 200, 255, 0.8)';
/** AR mirror paint: edge accents for the active selection. */
const AR_SELECTED_EDGE_STROKE = 'rgba(0, 200, 255, 0.95)';
/** AR mirror paint: amber outline of the active selection. */
const AR_SELECTED_OUTLINE = 'rgba(251, 191, 36, 0.9)';
/** AR mirror paint: cyan halo behind the selection outline. */
const AR_SELECTED_GLOW = 'rgba(0, 200, 255, 0.8)';
/** AR mirror paint: dashed rotation compass ring. */
const AR_RING_STROKE = 'rgba(0, 200, 255, 0.9)';
/** AR mirror paint: amber yaw needle inside the compass ring. */
const AR_RING_NEEDLE_STROKE = 'rgba(251, 191, 36, 0.95)';

/** Selection-HUD paint: color-wheel disc tint + outline. */
const HUD_WHEEL_FILL = 'rgba(56, 189, 248, 0.10)';
const HUD_WHEEL_STROKE = 'rgba(255, 255, 255, 0.9)';
/** Selection-HUD paint: color-lock dwell arc filling the wheel outline. */
const HUD_WHEEL_PROGRESS_STROKE = '#38bdf8';
/** Selection-HUD paint: Delete button tint + outline (danger red). */
const HUD_DELETE_FILL = 'rgba(248, 113, 113, 0.14)';
const HUD_DELETE_STROKE = 'rgba(248, 113, 113, 0.9)';
/** Selection-HUD paint: Delete label + its dwell progress fill. */
const HUD_DELETE_LABEL = '#fecaca';
const HUD_DELETE_PROGRESS_FILL = '#f87171';

/** Trash-bin paint: translucent danger tint, outline + dwell progress. */
const TRASH_STROKE = 'rgba(248, 113, 113, 0.9)';
const TRASH_INK = '#fecaca';
const TRASH_PROGRESS_FILL = '#f87171';

/** Spatial dialog paint: scrim, card, title + the two targets. */
const DIALOG_SCRIM = 'rgba(3, 7, 18, 0.55)';
const DIALOG_CARD_FILL = 'rgba(15, 23, 42, 0.88)';
const DIALOG_CARD_STROKE = 'rgba(148, 163, 184, 0.4)';
const DIALOG_TITLE = '#f8fafc';
const DIALOG_HINT = '#94a3b8';
/** Confirm target: danger red — pinching it (or holding OK) answers yes. */
const DIALOG_CONFIRM_FILL = 'rgba(248, 113, 113, 0.16)';
const DIALOG_CONFIRM_STROKE = 'rgba(248, 113, 113, 0.95)';
const DIALOG_CONFIRM_INK = '#fecaca';
/** Cancel target: calm blue — an open palm (or leaving) answers no. */
const DIALOG_CANCEL_FILL = 'rgba(56, 189, 248, 0.14)';
const DIALOG_CANCEL_STROKE = 'rgba(56, 189, 248, 0.9)';
const DIALOG_CANCEL_INK = '#bae6fd';
/** OK-gesture hold progress filling the confirm target's bottom edge. */
const DIALOG_OK_PROGRESS = '#fbbf24';

/** Boolean toggle paint: armed "ready to run" accent (amber clash cue). */
const BOOLEAN_READY_STROKE = 'rgba(251, 191, 36, 0.95)';
const BOOLEAN_READY_INK = '#fde68a';

/** Options for the debug overlay. */
/** A CREATE-mode shape button: `id` is the host's opaque shape key. */
export interface OverlayShape<S extends string = string> {
  id: S;
  /** Name of the shape (drawn as text only when no `icon` is given). */
  label: string;
  /** Line icon drawn on the button instead of the label. */
  icon?: OverlayShapeIcon;
}

/** Built-in line icons for the CREATE-mode shape buttons. */
export type OverlayShapeIcon = 'cube' | 'cuboid' | 'cylinder' | 'sphere';

/**
 * Options for the debug overlay. `S` is the host's shape id type — shape ids
 * are opaque strings here, so the overlay has no CAD dependency.
 */
export interface DebugOverlayOptions<S extends string = string> {
  /** Hand skeleton connections (defaults to MediaPipe HAND_CONNECTIONS). */
  connections?: ReadonlyArray<SkeletonConnection>;
  /**
   * Called when a mode button is activated — via mouse click or an index-tip
   * dwell of a pointing hand (>= `dwellMs` over the button). The host
   * decides what to do with the request (typically `engine.setMode`).
   */
  onModeRequest?: (mode: InteractionMode) => void;
  /**
   * Called when a SELECT-mode drag-constraint toggle is activated — via
   * mouse click or a pointing index-tip dwell. The host routes
   * it to the CAD builder's `setDragConstraint`; the overlay keeps the
   * visual state (both default to 'xz').
   */
  onDragConstraintRequest?: (constraint: DragConstraint) => void;
  /** Index-tip dwell time (ms) before a hovered mode button activates. Default 500. */
  dwellMs?: number;
  /** CREATE-mode shape buttons, left to right below the mode bar (none by default). */
  shapes?: ReadonlyArray<OverlayShape<S>>;
  /** Initially highlighted shape (defaults to the first of `shapes`). */
  activeShape?: S;
  /**
   * Called when a CREATE-mode shape button is activated — via mouse click or
   * a pointing index-tip dwell. The host routes it to
   * the CAD builder's tool (typically `builder.setTool`).
   */
  onShapeRequest?: (shape: S) => void;
  /**
   * SELECT-mode AR mirror: projects the live CAD scene (ground grid +
   * committed meshes) through the shared 3D camera onto this canvas, once
   * per rendered frame. Implemented on the CAD side (`buildArSceneFrame`)
   * and injected here by the orchestrator — the overlay stays Three.js-free.
   */
  arScene?: ArSceneProvider;
  /**
   * SELECT-mode selection HUD: mirrors the color wheel (as its full live
   * hue / saturation disc — pointing at a color here picks that color) and
   * the Delete button (device-space plain data + dwell progress) onto this
   * canvas, in sync with the viewport's controls. Supplied by the orchestrator; null skips the layer.
   */
  selectionHud?: SelectionHudProvider;
  /**
   * In-vision trash bin (bottom-right corner of the camera view): called
   * when it is activated — a pointing index-tip dwell of `trashDwellMs`,
   * a fresh pinch that closes over the icon, or a mouse click. The host
   * routes it through the spatial confirmation dialog before clearing.
   */
  onTrashRequest?: () => void;
  /** Pointing index-tip dwell (ms) that activates the trash bin. Default 600. */
  trashDwellMs?: number;
  /**
   * Spatial confirmation dialog: the dialog is open / answered through
   * `openConfirm` / `closeConfirm` (host-driven); these callbacks report
   * the gesture / mouse answer. The host maps the intent onto its
   * destructive action — nothing is confirmed here.
   */
  onConfirmRequest?: (intent: OverlayConfirmIntent) => void;
  /** The dialog was dismissed without confirming (open palm / hand away). */
  onCancelRequest?: (intent: OverlayConfirmIntent) => void;
  /**
   * Hold time (ms) that answers the dialog: a pointing fingertip on the
   * confirm / cancel target, or the "OK" gesture (thumb + index loop, other
   * fingers extended) held anywhere, confirms after this long. Match it to
   * the host's other hold-to-act timings. Default 1200.
   */
  confirmHoldMs?: number;
  /**
   * An open palm must stay up for this long (ms) to cancel the dialog —
   * guards against the natural pinch-release "open hand" the instant after
   * the dialog opens. Default 350.
   */
  confirmPalmCancelMs?: number;
  /**
   * Frames with no visible hands after which the dialog auto-cancels
   * ("move hand away"). Default 12 (~0.4 s at 30 FPS).
   */
  confirmHandLossFrames?: number;
  /**
   * Called when a SELECT-mode Boolean tool toggle is activated — via mouse
   * click or a pointing index-tip dwell. `null` disarms the current tool;
   * the overlay keeps the visual state. The host typically arms the CAD
   * builder's Boolean tool and fires it if meshes already intersect.
   */
  onBooleanToolRequest?: (tool: OverlayBooleanTool | null) => void;
  /**
   * Called when the secondary-hand "X" cross pose (index + pinky extended,
   * middle + ring folded) has been held long enough to fire — the host
   * triggers its armed Boolean operation (SUBTRACT by default).
   */
  onBooleanTrigger?: () => void;
  /**
   * Frames the X-cross pose must be held before firing (debounce so a
   * transition into the pose never mis-fires). Default 6 (~0.2 s).
   */
  booleanTriggerFrames?: number;
  /**
   * SELECT-mode Boolean state (armed tool + live clash availability),
   * supplied by the orchestrator per frame; null skips the highlight.
   */
  booleanState?: () => OverlayBooleanState | null;
  /**
   * Called when the EDIT-mode UNGROUP button is activated (pointing dwell or
   * mouse click) while `booleanState().canUngroup` is true — the host splits
   * the selected union back into its parts.
   */
  onUngroupRequest?: () => void;
}

/** A hit-testable rectangle in CSS pixels. */
interface ButtonRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * User-facing mode names. The `select` mode is presented as **EDIT** (pick,
 * move, recolor, delete, Boolean) — the id stays `select` in the API.
 */
export const MODE_LABELS: Record<InteractionMode, string> = {
  view: 'VIEW',
  select: 'EDIT',
  create: 'CREATE',
};

/** Mode buttons, left to right, along the top edge of the overlay. */
const MODE_BUTTONS: readonly InteractionMode[] = ['view', 'select', 'create'];

/** SELECT-mode drag-constraint toggles, top to bottom (below [ EDIT ]). */
const CONSTRAINT_BUTTONS: readonly DragConstraint[] = ['xz', 'y'];

export class DebugOverlay<S extends string = string> {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly connections: ReadonlyArray<SkeletonConnection>;
  private readonly onModeRequest: ((mode: InteractionMode) => void) | null;
  private readonly onDragConstraintRequest: ((constraint: DragConstraint) => void) | null;
  private readonly dwellMs: number;
  private readonly arScene: ArSceneProvider | null;
  private readonly selectionHud: SelectionHudProvider | null;
  /** Last rendered button rects (CSS px) — hit targets for mouse + finger. */
  private buttonRects: ButtonRect[] = [];
  private readonly modeDwell = new DwellClock();
  private lastFrameTimestamp: number | null = null;
  /** Last rendered constraint-button rects (CSS px); empty outside SELECT mode. */
  private constraintRects: ButtonRect[] = [];
  private readonly constraintDwell = new DwellClock();
  /** Visual + authoritative overlay state of the drag constraint (host mirrors it). */
  private dragConstraint: DragConstraint = 'xz';
  private readonly shapes: ReadonlyArray<OverlayShape<S>>;
  private readonly onShapeRequest: ((shape: S) => void) | null;
  /** Highlighted CREATE-mode shape (host mirrors it). */
  private activeShapeId: S | null;
  /** Last rendered shape-button rects (CSS px); empty outside CREATE mode. */
  private shapeRects: ButtonRect[] = [];
  private readonly shapeDwell = new DwellClock();
  /**
   * In-vision trash bin (bottom-right): last rendered rect (null = hidden,
   * e.g. while the confirmation dialog is open) + its dwell clock.
   */
  private trashRect: ButtonRect | null = null;
  private readonly trashDwell = new DwellClock();
  /** Trash dwell threshold (ms) — slower than a mode button (destructive). */
  private readonly trashDwellMs: number;
  private readonly onTrashRequest: (() => void) | null;
  /**
   * Spatial confirmation dialog: the open intent (null = closed), the last
   * rendered target rects, and its interaction clocks (OK hold, open-palm
   * cancel, hands-away frames).
   */
  private openIntent: OverlayConfirmIntent | null = null;
  private confirmRects: { confirm: ButtonRect; cancel: ButtonRect } | null = null;
  private confirmOkElapsed = 0;
  /** Pointing-hold time on the cancel target. */
  private confirmCancelElapsed = 0;
  private confirmPalmElapsed = 0;
  private confirmNoHandFrames = 0;
  private confirmOpenedAgo = 0;
  private readonly confirmHoldMs: number;
  private readonly confirmPalmCancelMs: number;
  private readonly confirmHandLossFrames: number;
  private readonly onConfirmRequest: ((intent: OverlayConfirmIntent) => void) | null;
  private readonly onCancelRequest: ((intent: OverlayConfirmIntent) => void) | null;
  /**
   * SELECT-mode Boolean tool toggles: last rendered rects, the dwell clock,
   * the armed tool (host mirrors it) and the X-cross trigger debounce.
   */
  private booleanRects: ButtonRect[] = [];
  private readonly booleanDwell = new DwellClock();
  private armedBooleanTool: OverlayBooleanTool | null = null;
  private xCrossFrames = 0;
  private xCrossFired = false;
  private readonly booleanTriggerFrames: number;
  private readonly onBooleanToolRequest: ((tool: OverlayBooleanTool | null) => void) | null;
  private readonly onBooleanTrigger: (() => void) | null;
  private readonly booleanState: (() => OverlayBooleanState | null) | null;
  /** EDIT-mode UNGROUP action button (one-shot; below the Boolean toggles). */
  private ungroupRect: ButtonRect | null = null;
  private readonly ungroupDwell = new DwellClock();
  private readonly onUngroupRequest: (() => void) | null;
  /** Last rendered mirrored cover transform (device-space UI hit tests). */
  private lastView: ViewTransform | null = null;

  constructor(canvas: HTMLCanvasElement, options: DebugOverlayOptions<S> = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('DebugOverlay: 2D canvas context unavailable');
    this.ctx = ctx;
    this.connections = options.connections ?? HandLandmarker.HAND_CONNECTIONS;
    this.onModeRequest = options.onModeRequest ?? null;
    this.onDragConstraintRequest = options.onDragConstraintRequest ?? null;
    this.dwellMs = options.dwellMs ?? 500;
    this.arScene = options.arScene ?? null;
    this.selectionHud = options.selectionHud ?? null;
    this.shapes = options.shapes ?? [];
    this.onShapeRequest = options.onShapeRequest ?? null;
    this.activeShapeId = options.activeShape ?? this.shapes[0]?.id ?? null;
    this.onTrashRequest = options.onTrashRequest ?? null;
    this.trashDwellMs = options.trashDwellMs ?? 600;
    this.onConfirmRequest = options.onConfirmRequest ?? null;
    this.onCancelRequest = options.onCancelRequest ?? null;
    this.confirmHoldMs = options.confirmHoldMs ?? 1200;
    this.confirmPalmCancelMs = options.confirmPalmCancelMs ?? 350;
    this.confirmHandLossFrames = options.confirmHandLossFrames ?? 12;
    this.onBooleanToolRequest = options.onBooleanToolRequest ?? null;
    this.onBooleanTrigger = options.onBooleanTrigger ?? null;
    this.booleanTriggerFrames = options.booleanTriggerFrames ?? 6;
    this.booleanState = options.booleanState ?? null;
    this.onUngroupRequest = options.onUngroupRequest ?? null;
    canvas.addEventListener('click', this.onCanvasClick);
    canvas.addEventListener('mousemove', this.onCanvasMouseMove);
  }

  /** Detach the mode-button mouse listeners (the canvas stays with the host). */
  dispose(): void {
    this.canvas.removeEventListener('click', this.onCanvasClick);
    this.canvas.removeEventListener('mousemove', this.onCanvasMouseMove);
  }

  /** Highlighted CREATE-mode shape. */
  get activeShape(): S | null {
    return this.activeShapeId;
  }

  /** Whether the in-vision spatial confirmation dialog is currently open. */
  get confirmActive(): boolean {
    return this.openIntent !== null;
  }

  /** The open confirmation intent, or null while the dialog is closed. */
  get confirmIntent(): OverlayConfirmIntent | null {
    return this.openIntent;
  }

  /**
   * Open the in-vision spatial confirmation dialog for a host-defined
   * intent (e.g. clearing the scene, deleting the selection). Every other
   * overlay interaction freezes while it is open; the answer arrives via
   * `onConfirmRequest` / `onCancelRequest`. No-op when already open — one
   * destructive question at a time.
   */
  openConfirm(intent: OverlayConfirmIntent): void {
    if (this.openIntent !== null) return;
    this.openIntent = intent;
    this.confirmOkElapsed = 0;
    this.confirmCancelElapsed = 0;
    this.confirmPalmElapsed = 0;
    this.confirmNoHandFrames = 0;
    this.confirmOpenedAgo = 0;
    // The fresh-pinch memory is deliberately kept: a pinch held while the
    // dialog opens (e.g. the pinch that triggered the trash) must stay
    // "engaged", so it can never count as a fresh confirm later — the user
    // has to release and pinch again to answer.
  }

  /** Close the dialog without answering (host-side state change). */
  closeConfirm(): void {
    this.openIntent = null;
    this.confirmRects = null;
    this.confirmOkElapsed = 0;
    this.confirmCancelElapsed = 0;
    this.confirmPalmElapsed = 0;
    this.confirmNoHandFrames = 0;
    this.confirmOpenedAgo = 0;
  }

  /** The armed Boolean tool the overlay currently displays (host mirrors it). */
  get booleanTool(): OverlayBooleanTool | null {
    return this.armedBooleanTool;
  }

  /** Click anywhere inside a rendered mode / constraint / shape / Boolean
   *  button, the trash bin, or a confirmation-dialog target activates it. */
  private readonly onCanvasClick = (event: MouseEvent): void => {
    // The spatial dialog is modal: a click may only answer it.
    if (this.openIntent !== null) {
      const rects = this.confirmRects;
      if (!rects) return;
      if (this.inRect(rects.confirm, event.offsetX, event.offsetY)) this.answerConfirm();
      else if (this.inRect(rects.cancel, event.offsetX, event.offsetY)) this.answerCancel();
      return;
    }
    const trash = this.trashRect;
    if (trash && this.inRect(trash, event.offsetX, event.offsetY)) {
      this.requestTrash();
      return;
    }
    const boolean = this.hitRects(this.booleanRects, event.offsetX, event.offsetY);
    if (boolean >= 0) {
      this.requestBoolean(boolean);
      return;
    }
    const ungroup = this.ungroupRect;
    if (ungroup && this.inRect(ungroup, event.offsetX, event.offsetY)) {
      this.requestUngroup();
      return;
    }
    const shape = this.hitRects(this.shapeRects, event.offsetX, event.offsetY);
    if (shape >= 0) {
      this.requestShape(shape);
      return;
    }
    const constraint = this.hitRects(this.constraintRects, event.offsetX, event.offsetY);
    if (constraint >= 0) {
      this.requestConstraint(constraint);
      return;
    }
    const index = this.hitRects(this.buttonRects, event.offsetX, event.offsetY);
    if (index >= 0) this.requestMode(index);
  };

  /** Pointer feedback while hovering any interactive overlay surface. */
  private readonly onCanvasMouseMove = (event: MouseEvent): void => {
    if (this.openIntent !== null) {
      const rects = this.confirmRects;
      this.canvas.style.cursor =
        rects && (this.inRect(rects.confirm, event.offsetX, event.offsetY) ||
          this.inRect(rects.cancel, event.offsetX, event.offsetY))
          ? 'pointer'
          : 'default';
      return;
    }
    const trash = this.trashRect;
    const hovering =
      (trash !== null && this.inRect(trash, event.offsetX, event.offsetY)) ||
      this.hitRects(this.booleanRects, event.offsetX, event.offsetY) >= 0 ||
      (this.ungroupRect !== null && this.canUngroupNow() &&
        this.inRect(this.ungroupRect, event.offsetX, event.offsetY)) ||
      this.hitRects(this.buttonRects, event.offsetX, event.offsetY) >= 0 ||
      this.hitRects(this.constraintRects, event.offsetX, event.offsetY) >= 0 ||
      this.hitRects(this.shapeRects, event.offsetX, event.offsetY) >= 0;
    this.canvas.style.cursor = hovering ? 'pointer' : 'default';
  };

  /** Render one frame of landmarks + in-vision UI (stats live in the DOM bar). */
  render(frame: FrameEvent): void {
    const { cssWidth, cssHeight } = this.ensureSize(frame.video.width, frame.video.height);
    this.ctx.clearRect(0, 0, cssWidth, cssHeight);
    const view = this.coverTransform(cssWidth, cssHeight, frame.video.width, frame.video.height);
    this.lastView = view; // device-space UI hit tests between frames
    const dialogOpen = this.openIntent !== null;
    // SELECT mode: live AR mirror of the 3D scene, drawn beneath the hands.
    if (frame.mode === 'select') this.drawArMirror(view, cssWidth);
    for (const hand of frame.hands) {
      this.drawHand(hand, frame.state, view, cssWidth);
    }
    this.drawDualHandsLink(frame, view, cssWidth);
    this.drawZoomAnchor(frame, view);
    // SELECT mode: mirror the selection's color wheel + Delete button as a
    // HUD (drawn over the hands' skeletons so the alignment guide reads).
    // Hidden while the confirmation dialog is open — the scene is frozen.
    if (frame.mode === 'select' && !dialogOpen) this.drawSelectionHud(view, cssWidth);
    this.drawModeButtons(frame, cssWidth);
    if (frame.mode === 'select' && !dialogOpen) {
      this.drawConstraintButtons(cssWidth);
      this.drawBooleanButtons(cssWidth);
      this.drawUngroupButton(cssWidth);
    } else {
      this.constraintRects = [];
      this.booleanRects = [];
      this.ungroupRect = null;
    }
    if (frame.mode === 'create' && !dialogOpen) this.drawShapeButtons(cssWidth);
    else this.shapeRects = [];
    // The trash bin is a VIEW-mode control; it also hides while the dialog
    // is open (one question at a time).
    if (frame.mode === 'view' && !dialogOpen) this.drawTrashButton(cssWidth);
    else this.trashRect = null;
    const dt = this.frameDt(frame);
    if (dialogOpen) {
      this.drawConfirmDialog(cssWidth, cssHeight);
      this.updateConfirmInteraction(frame, view, dt);
    } else {
      this.updateModeInteraction(frame, view, dt);
      this.updateConstraintInteraction(frame, view, dt);
      this.updateShapeInteraction(frame, view, dt);
      this.updateBooleanInteraction(frame, view, dt);
      this.updateUngroupInteraction(frame, view, dt);
      this.updateTrashInteraction(frame, view, dt);
    }
    this.drawPointerCursors(frame, view);
  }

  /**
   * Dwell clock tick from frame timestamps; capped so a stalled camera feed
   * cannot complete a dwell in one jump.
   */
  private frameDt(frame: FrameEvent): number {
    const dt =
      this.lastFrameTimestamp === null
        ? 0
        : Math.max(0, Math.min(frame.timestamp - this.lastFrameTimestamp, 500));
    this.lastFrameTimestamp = frame.timestamp;
    return dt;
  }

  /**
   * Ring on each pointing hand's index fingertip: the visible cursor for the
   * overlay buttons (only a pointing hand can press them).
   */
  private drawPointerCursors(frame: FrameEvent, view: ViewTransform): void {
    const ctx = this.ctx;
    for (const hand of frame.hands) {
      if (!hand.pointing) continue;
      const tip = this.toCanvas(hand, INDEX_TIP, view);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 11, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
      ctx.fill();
    }
  }

  /**
   * The part of the webcam frame actually visible in the camera view, in
   * device space ([-1, 1], +Y up). The video is cover-cropped into the view
   * (a 16:9 webcam in a 4:3 view loses its left / right edges), so hosts
   * placing controls that must show up in the camera view anchor them
   * inside this rect. The whole frame until the first render.
   */
  visibleDeviceRect(): { minX: number; maxX: number; minY: number; maxY: number } {
    const view = this.lastView;
    const rect = this.canvas.getBoundingClientRect();
    if (!view || view.dispW <= 0 || view.dispH <= 0 || rect.width <= 0) {
      return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    }
    // Canvas CSS px → device (the overlay's own mirrored cover mapping).
    const toX = (px: number) => ((px - view.ox) / view.dispW) * 2 - 1;
    const toY = (py: number) => 1 - ((py - view.oy) / view.dispH) * 2;
    return {
      minX: Math.max(-1, toX(0)),
      maxX: Math.min(1, toX(rect.width)),
      minY: Math.max(-1, toY(rect.height)),
      maxY: Math.min(1, toY(0)),
    };
  }

  /**
   * Device-space hit test against the last rendered UI buttons (mode bar +
   * SELECT-mode constraint stack + CREATE-mode shape row), e.g. to tell
   * whether a device-space point sits under the overlay's buttons. Runs
   * through the same mirrored cover transform as the landmarks, using the
   * previous frame's metrics (the buttons do not move between frames).
   */
  isUiAtDevice(x: number, y: number): boolean {
    const view = this.lastView;
    if (!view) return false;
    // Device [-1, 1] (mirrored, +Y up) -> normalized [0, 1] -> canvas CSS px.
    const nx = (1 - x) / 2;
    const ny = (1 - y) / 2;
    const px = view.ox + (1 - nx) * view.dispW;
    const py = view.oy + ny * view.dispH;
    const trash = this.trashRect;
    const rects = this.confirmRects;
    return (
      this.hitRects(this.buttonRects, px, py) >= 0 ||
      this.hitRects(this.constraintRects, px, py) >= 0 ||
      this.hitRects(this.shapeRects, px, py) >= 0 ||
      this.hitRects(this.booleanRects, px, py) >= 0 ||
      (this.ungroupRect !== null && this.inRect(this.ungroupRect, px, py)) ||
      (trash !== null && this.inRect(trash, px, py)) ||
      (rects !== null &&
        (this.inRect(rects.confirm, px, py) || this.inRect(rects.cancel, px, py)))
    );
  }

  /** Whether a CSS-pixel point lies inside a button rectangle. */
  private inRect(rect: ButtonRect, px: number, py: number): boolean {
    return px >= rect.x && px <= rect.x + rect.width && py >= rect.y && py <= rect.y + rect.height;
  }

  /** Index of the first rect containing the CSS-pixel point, or -1. */
  private hitRects(rects: ReadonlyArray<ButtonRect | null>, px: number, py: number): number {
    return rects.findIndex((rect) => rect !== null && this.inRect(rect, px, py));
  }

  /**
   * The button a *pointing* hand's index tip (landmark 8) rests on — the
   * first hit among the frame's hands — or -1. Pinches never press buttons.
   */
  private pointedButton(
    frame: FrameEvent,
    view: ViewTransform,
    rects: ReadonlyArray<ButtonRect | null>
  ): number {
    for (const hand of frame.hands) {
      if (!hand.pointing) continue;
      const tip = this.toCanvas(hand, INDEX_TIP, view);
      const index = this.hitRects(rects, tip.x, tip.y);
      if (index >= 0) return index;
    }
    return -1;
  }

  /**
   * Chrome of a non-active overlay button: dark glass fill + outline, and a
   * dwell progress bar along the bottom edge while `progress` is non-null.
   */
  private drawIdleButton(
    rect: ButtonRect,
    progress: number | null,
    idleLineWidth: number,
    idleStroke = 'rgba(148, 163, 184, 0.55)',
    progressFill = '#38bdf8'
  ): void {
    const ctx = this.ctx;
    const { x, y, width, height } = rect;
    const dwelling = progress !== null;
    ctx.fillStyle = dwelling ? 'rgba(15, 23, 42, 0.85)' : 'rgba(15, 23, 42, 0.6)';
    ctx.fillRect(x, y, width, height);
    ctx.lineWidth = dwelling ? 2 : idleLineWidth;
    ctx.strokeStyle = dwelling ? '#e2e8f0' : idleStroke;
    ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
    if (dwelling) {
      ctx.fillStyle = progressFill;
      ctx.fillRect(x + 2, y + height - 5, (width - 4) * progress, 3);
    }
  }

  /**
   * Chrome of the active (inverted) toggle: solid light fill, dark outline
   * and a high-contrast indicator bar `barInset` px above the bottom edge.
   */
  private drawActiveButton(
    rect: ButtonRect,
    barInset: number,
    barThickness: number,
    stroke = '#0f172a',
    bar = '#0284c7',
    lineWidth = 2
  ): void {
    const ctx = this.ctx;
    const { x, y, width, height } = rect;
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(x, y, width, height);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = stroke;
    ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
    ctx.fillStyle = bar;
    ctx.fillRect(x + 3, y + height - barInset, width - 6, barThickness);
  }

  /**
   * Typography scale: the camera renders as a small floating thumbnail
   * (~220–340 CSS px wide) or an expanded card (420–640 px), so fonts track
   * the canvas width instead of staying fixed.
   */
  private fontScale(cssWidth: number): number {
    return Math.min(1, Math.max(0.7, cssWidth / 480));
  }

  /**
   * Match the canvas backing store to the *panel* (CSS pixels × DPR) so text
   * renders crisp and the HUD is never cropped, regardless of video size.
   */
  private ensureSize(
    videoWidth: number,
    videoHeight: number
  ): { cssWidth: number; cssHeight: number } {
    const rect = this.canvas.getBoundingClientRect();
    const cssWidth = rect.width > 0 ? rect.width : videoWidth || 640;
    const cssHeight = rect.height > 0 ? rect.height : videoHeight || 480;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const bitmapWidth = Math.max(1, Math.round(cssWidth * dpr));
    const bitmapHeight = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== bitmapWidth || this.canvas.height !== bitmapHeight) {
      this.canvas.width = bitmapWidth;
      this.canvas.height = bitmapHeight;
    }
    // Draw in CSS pixels regardless of the device pixel ratio.
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { cssWidth, cssHeight };
  }

  /**
   * Replicates the CSS `object-fit: cover` + center-crop of the mirrored
   * <video>, so landmark pixels line up with the webcam image.
   */
  private coverTransform(
    cssWidth: number,
    cssHeight: number,
    videoWidth: number,
    videoHeight: number
  ): ViewTransform {
    const vw = videoWidth || cssWidth;
    const vh = videoHeight || cssHeight;
    const scale = Math.max(cssWidth / vw, cssHeight / vh);
    const dispW = vw * scale;
    const dispH = vh * scale;
    return { ox: (cssWidth - dispW) / 2, oy: (cssHeight - dispH) / 2, dispW, dispH };
  }

  /** Mirror-converted panel-space position of a normalized landmark. */
  private toCanvas(hand: HandSnapshot, index: number, view: ViewTransform) {
    const lm = hand.landmarks[index];
    return {
      x: view.ox + (1 - lm.normalized.x) * view.dispW,
      y: view.oy + lm.normalized.y * view.dispH,
    };
  }

  /** Begin a rounded panel path; the caller fills and/or strokes it. */
  private panelPath(x: number, y: number, width: number, height: number, radius: number): void {
    this.ctx.beginPath();
    if (typeof this.ctx.roundRect === 'function') {
      this.ctx.roundRect(x, y, width, height, radius);
    } else {
      this.ctx.rect(x, y, width, height);
    }
  }

  private drawHand(
    hand: HandSnapshot,
    state: GestureState,
    view: ViewTransform,
    cssWidth: number
  ): void {
    const style = styleFor(hand, state);
    const ctx = this.ctx;

    // Skeleton.
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = style.skeleton;
    ctx.beginPath();
    for (const connection of this.connections) {
      const { start: a, end: b } = connection;
      if (a >= hand.landmarks.length || b >= hand.landmarks.length) continue;
      const pa = this.toCanvas(hand, a, view);
      const pb = this.toCanvas(hand, b, view);
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
    }
    ctx.stroke();

    // Joints (21 landmarks); slightly larger fingertips.
    for (let i = 0; i < hand.landmarks.length; i++) {
      const p = this.toCanvas(hand, i, view);
      const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
      ctx.beginPath();
      ctx.arc(p.x, p.y, isTip ? 5 : 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = isTip ? style.joint : style.jointFill;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = style.skeleton;
      ctx.stroke();
    }

    // Pinch indicator: thumb <-> index line + live distance.
    const thumb = this.toCanvas(hand, THUMB_TIP, view);
    const index = this.toCanvas(hand, INDEX_TIP, view);
    ctx.setLineDash(hand.pinchActive ? [] : [6, 6]);
    ctx.lineWidth = hand.pinchActive ? 3 : 1.5;
    ctx.strokeStyle = hand.pinchActive ? '#22c55e' : 'rgba(34, 197, 94, 0.5)';
    ctx.beginPath();
    ctx.moveTo(thumb.x, thumb.y);
    ctx.lineTo(index.x, index.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Handedness + metrics label near the wrist (rounded chip).
    const wrist = this.toCanvas(hand, 0, view);
    const label = `${hand.handedness} · d=${hand.pinchDistance.toFixed(3)}${hand.fistActive ? ' · FIST' : ''}`;
    ctx.font = `${Math.round(12 * this.fontScale(cssWidth))}px ui-monospace, monospace`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(3, 7, 18, 0.78)';
    this.panelPath(wrist.x - textWidth / 2 - 6, wrist.y + 10, textWidth + 12, 20, 5);
    ctx.fill();
    ctx.fillStyle = style.label;
    ctx.textAlign = 'center';
    ctx.fillText(label, wrist.x, wrist.y + 24);
    ctx.textAlign = 'left';
  }

  /** Dashed line between the two pinch centers while both hands pinch. */
  private drawDualHandsLink(frame: FrameEvent, view: ViewTransform, cssWidth: number): void {
    const pinching = frame.hands.filter((h) => h.pinchActive);
    if (pinching.length < 2) return;
    const a = this.toCanvas(pinching[0], THUMB_TIP, view);
    const b = this.toCanvas(pinching[1], THUMB_TIP, view);
    const ctx = this.ctx;
    ctx.setLineDash([10, 8]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(249, 115, 22, 0.9)';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);

    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const label = `D=${(frame.metrics.extrusionDistance ?? 0).toFixed(3)} ×${(frame.metrics.extrusionScaleFactor ?? 1).toFixed(2)}`;
    ctx.font = `bold ${Math.round(13 * this.fontScale(cssWidth))}px ui-monospace, monospace`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(3, 7, 18, 0.78)';
    this.panelPath(mid.x - textWidth / 2 - 6, mid.y - 24, textWidth + 12, 20, 5);
    ctx.fill();
    ctx.fillStyle = '#fdba74';
    ctx.textAlign = 'center';
    ctx.fillText(label, mid.x, mid.y - 10);
    ctx.textAlign = 'left';
  }

  /** Two-fist navigation: ring on the anchor (pivot) fist, line to the moving fist. */
  private drawZoomAnchor(frame: FrameEvent, view: ViewTransform): void {
    const anchorHand = frame.metrics.zoomAnchor;
    if (frame.state !== 'ZOOMING' || !anchorHand) return;
    const anchor = frame.hands.find((h) => h.handedness === anchorHand);
    const mover = frame.hands.find((h) => h.handedness !== anchorHand);
    if (!anchor) return;
    const pivot = this.toCanvas(anchor, MIDDLE_MCP, view);
    const ctx = this.ctx;
    if (mover) {
      const end = this.toCanvas(mover, MIDDLE_MCP, view);
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.85)';
      ctx.beginPath();
      ctx.moveTo(pivot.x, pivot.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#e9d5ff';
    ctx.beginPath();
    ctx.arc(pivot.x, pivot.y, 16, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#a855f7';
    ctx.beginPath();
    ctx.arc(pivot.x, pivot.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  /* ------------------------------------------------------------------ */
  /* AR spatial mirror (SELECT mode)                                     */
  /* ------------------------------------------------------------------ */

  /**
   * Translucent CAD mirror: the projected ground grid and ghost meshes are
   * drawn beneath the hand skeletons, turning the vision canvas into a live
   * AR spatial view of the 3D environment (in lockstep with pinch-driven
   * drags — the provider runs inside this same rAF-driven frame).
   */
  private drawArMirror(view: ViewTransform, cssWidth: number): void {
    if (!this.arScene) return;
    // Project into the webcam image's on-screen rect (the same cover
    // transform the landmarks use): true proportions, aligned with the hands.
    const frame = this.arScene(view.dispW, view.dispH);
    if (!frame) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(view.ox, view.oy);

    // Ground grid in two batched passes: faint 1-unit minor lines first,
    // then the stronger major divisions (every 5 units) — a 30 × 30 unit
    // floor window stays readable at thumbnail scale.
    ctx.lineWidth = 1;
    ctx.strokeStyle = AR_GRID_STROKE;
    ctx.beginPath();
    for (const line of frame.grid) {
      if (line.major || line.points.length < 2) continue;
      ctx.moveTo(line.points[0].x, line.points[0].y);
      for (let i = 1; i < line.points.length; i++) ctx.lineTo(line.points[i].x, line.points[i].y);
    }
    ctx.stroke();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = AR_GRID_MAJOR_STROKE;
    ctx.beginPath();
    for (const line of frame.grid) {
      if (!line.major || line.points.length < 2) continue;
      ctx.moveTo(line.points[0].x, line.points[0].y);
      for (let i = 1; i < line.points.length; i++) ctx.lineTo(line.points[i].x, line.points[i].y);
    }
    ctx.stroke();

    for (const mesh of frame.meshes) this.drawArMesh(mesh, cssWidth);
    if (frame.rotationRing) this.drawArRotationRing(frame.rotationRing, cssWidth);
    ctx.restore();
  }

  /** Rotational compass ring: dashed cyan circle + amber yaw needle. */
  private drawArRotationRing(ring: ArRotationRing, cssWidth: number): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.setLineDash([9, 7]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = AR_RING_STROKE;
    ctx.shadowColor = AR_SELECTED_GLOW;
    ctx.shadowBlur = Math.max(5, cssWidth * 0.02);
    ctx.beginPath();
    for (const [a, b] of ring.circle) {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
    if (ring.needle) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = AR_RING_NEEDLE_STROKE;
      ctx.beginPath();
      ctx.moveTo(ring.needle[0].x, ring.needle[0].y);
      ctx.lineTo(ring.needle[1].x, ring.needle[1].y);
      ctx.stroke();
    }
    ctx.restore();
  }

  /** One mesh ghost: translucent silhouette fill + crisp accent edges. */
  private drawArMesh(mesh: ArMeshGhost, cssWidth: number): void {
    const ctx = this.ctx;

    // Ghost fill over the projected silhouette (2D footprint) of the solid.
    if (mesh.hull.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(mesh.hull[0].x, mesh.hull[0].y);
      for (let i = 1; i < mesh.hull.length; i++) ctx.lineTo(mesh.hull[i].x, mesh.hull[i].y);
      ctx.closePath();
      ctx.fillStyle = mesh.selected ? AR_SELECTED_FILL : AR_GHOST_FILL;
      ctx.fill();

      if (mesh.selected) {
        // Energetic highlight: glowing cyan halo behind an amber outline.
        ctx.save();
        ctx.shadowColor = AR_SELECTED_GLOW;
        ctx.shadowBlur = Math.max(6, cssWidth * 0.025);
        ctx.lineWidth = 2;
        ctx.strokeStyle = AR_SELECTED_OUTLINE;
        ctx.stroke();
        ctx.restore();
      }
    }

    // Crisp accent edges (projected EdgesGeometry) on top of the fill.
    ctx.lineWidth = mesh.selected ? 2 : 1.25;
    ctx.strokeStyle = mesh.selected ? AR_SELECTED_EDGE_STROKE : AR_EDGE_STROKE;
    ctx.beginPath();
    for (const [a, b] of mesh.edges) {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
  }

  /* ------------------------------------------------------------------ */
  /* SELECT-mode selection HUD (color wheel + Delete mirror)            */
  /* ------------------------------------------------------------------ */

  /**
   * Mirror the selection's floating controls onto the camera thumbnail: the
   * color wheel's disc outline and the Delete button, expressed in device
   * space by the host (the inverse of the viewport mapping the real controls
   * live in), so the drawn controls sit exactly where a tracked fingertip
   * must point. Both dwell progress bars (color lock, delete) fill as live
   * feedback — the same model as the overlay's own buttons.
   */
  private drawSelectionHud(view: ViewTransform, cssWidth: number): void {
    if (!this.selectionHud) return;
    const hud = this.selectionHud();
    if (!hud) return;
    const ctx = this.ctx;
    const scale = this.fontScale(cssWidth);
    ctx.save();
    ctx.translate(view.ox, view.oy);

    if (hud.wheel) {
      const cx = ((hud.wheel.x + 1) / 2) * view.dispW;
      const cy = ((1 - hud.wheel.y) / 2) * view.dispH;
      const rx = Math.max((hud.wheel.radiusX / 2) * view.dispW, 1);
      const ry = Math.max((hud.wheel.radiusY / 2) * view.dispH, 1);
      this.drawHueDisc(cx, cy, rx, ry);
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = HUD_WHEEL_STROKE;
      ctx.stroke();
      // Color-lock dwell: an arc sweeps the rim as the timed hover fills.
      if (hud.wheelProgress > 0) {
        ctx.beginPath();
        ctx.ellipse(
          cx,
          cy,
          rx,
          ry,
          0,
          -Math.PI / 2,
          -Math.PI / 2 + Math.min(1, hud.wheelProgress) * Math.PI * 2
        );
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = HUD_WHEEL_PROGRESS_STROKE;
        ctx.stroke();
      }
    }

    if (hud.deleteButton) {
      // Device rect (min corner, +Y up) → canvas rect (top-left, +Y down).
      const left = ((hud.deleteButton.x + 1) / 2) * view.dispW;
      const right = ((hud.deleteButton.x + hud.deleteButton.width + 1) / 2) * view.dispW;
      const top = ((1 - (hud.deleteButton.y + hud.deleteButton.height)) / 2) * view.dispH;
      const bottom = ((1 - hud.deleteButton.y) / 2) * view.dispH;
      const x = Math.min(left, right);
      const y = Math.min(top, bottom);
      const width = Math.abs(right - left);
      const height = Math.abs(bottom - top);
      if (width > 2 && height > 2) {
        this.panelPath(x, y, width, height, 6 * scale);
        ctx.fillStyle = HUD_DELETE_FILL;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = HUD_DELETE_STROKE;
        ctx.stroke();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `bold ${Math.max(7, Math.round(9 * scale))}px ui-monospace, monospace`;
        ctx.fillStyle = HUD_DELETE_LABEL;
        ctx.shadowColor = 'rgba(15, 23, 42, 0.9)';
        ctx.shadowBlur = 3;
        ctx.fillText('DEL', x + width / 2, y + height / 2);
        ctx.shadowBlur = 0;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        // Pointing dwell on Delete fills along its bottom edge.
        if (hud.deleteProgress > 0) {
          ctx.fillStyle = HUD_DELETE_PROGRESS_FILL;
          ctx.fillRect(
            x + 1,
            y + height - 3.5,
            (width - 2) * Math.min(1, hud.deleteProgress),
            2.5
          );
        }
      }
    }

    ctx.restore();
  }

  /**
   * The color wheel's live hue / saturation disc, drawn into the camera
   * view exactly where (and how) the 3D viewport's wheel maps: hue by angle
   * (clockwise from +x on screen), saturation by radius, lightness 0.5 —
   * fading linearly to gray at the center — so a pointing fingertip on a
   * color here repaints the object with that same color. Falls back to a
   * flat translucent disc without conic-gradient support.
   */
  private drawHueDisc(cx: number, cy: number, rx: number, ry: number): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(1, ry / rx); // circle of radius rx → the device-space ellipse
    ctx.beginPath();
    ctx.arc(0, 0, rx, 0, Math.PI * 2);
    if (typeof ctx.createConicGradient === 'function') {
      const hues = ctx.createConicGradient(0, 0, 0);
      for (let i = 0; i <= 12; i++) hues.addColorStop(i / 12, `hsl(${i * 30}, 100%, 50%)`);
      ctx.fillStyle = hues;
      ctx.fill();
      const saturation = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
      saturation.addColorStop(0, 'rgb(128, 128, 128)');
      saturation.addColorStop(1, 'rgba(128, 128, 128, 0)');
      ctx.fillStyle = saturation;
      ctx.fill();
    } else {
      ctx.fillStyle = HUD_WHEEL_FILL;
      ctx.fill();
    }
    ctx.restore();
  }

  /* ------------------------------------------------------------------ */
  /* Mode switcher buttons (top edge of the overlay)                    */
  /* ------------------------------------------------------------------ */


  private requestMode(index: number): void {
    this.onModeRequest?.(MODE_BUTTONS[index]);
  }

  /** Shared layout metrics of the top-edge button bars (mode + constraint). */
  private barMetrics(cssWidth: number): { margin: number; gap: number; height: number } {
    const scale = this.fontScale(cssWidth);
    return {
      margin: Math.round(8 * scale),
      gap: Math.round(6 * scale),
      height: Math.max(22, Math.round(32 * scale)),
    };
  }

  /**
   * Finger interaction with the mode buttons: a *pointing* hand's index tip
   * (landmark 8) dwelling over a button for `dwellMs` activates it. Pinches
   * never press buttons, so a pinch-drag (moving / building an object) that
   * sweeps across the bar never switches modes.
   */
  private updateModeInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    const target = this.pointedButton(frame, view, this.buttonRects);
    const fired = this.modeDwell.update(target, dt, this.dwellMs);
    if (fired >= 0) this.requestMode(fired);
  }

  /**
   * Boxy mode switcher across the top edge: three sharp-cornered, mutually
   * exclusive toggle buttons. The active one is inverted — solid light fill,
   * dark text, high-contrast indicator bar — while a dwell fills a progress
   * bar along the bottom edge of the hovered button.
   */
  private drawModeButtons(frame: FrameEvent, cssWidth: number): void {
    const ctx = this.ctx;
    const scale = this.fontScale(cssWidth);
    const { margin, gap, height } = this.barMetrics(cssWidth);
    const width = (cssWidth - margin * 2 - gap * (MODE_BUTTONS.length - 1)) / MODE_BUTTONS.length;

    this.buttonRects = [];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    MODE_BUTTONS.forEach((mode, i) => {
      const x = margin + i * (width + gap);
      this.buttonRects.push({ x, y: margin, width, height });
      const label = `[ ${MODE_LABELS[mode]} ]`;
      // Shrink the label to fit the button.
      let fontSize = Math.max(9, Math.round(13 * scale));
      ctx.font = `bold ${fontSize}px ui-monospace, monospace`;
      while (fontSize > 8 && ctx.measureText(label).width > width - 8) {
        fontSize -= 1;
        ctx.font = `bold ${fontSize}px ui-monospace, monospace`;
      }

      const rect = { x, y: margin, width, height };
      if (frame.mode === mode) {
        this.drawActiveButton(rect, 7, 4);
        ctx.fillStyle = '#0f172a';
      } else {
        const progress = this.modeDwell.progress(i, this.dwellMs);
        this.drawIdleButton(rect, progress, 1.5);
        ctx.fillStyle = progress !== null ? '#f8fafc' : '#cbd5e1';
      }
      ctx.fillText(label, x + width / 2, margin + height / 2);
    });
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  /* ------------------------------------------------------------------ */
  /* Drag-constraint toggles (EDIT mode, below [ EDIT ])                 */
  /* ------------------------------------------------------------------ */


  private requestConstraint(index: number): void {
    const constraint = CONSTRAINT_BUTTONS[index];
    if (!constraint || this.dragConstraint === constraint) return;
    this.dragConstraint = constraint;
    this.onDragConstraintRequest?.(constraint);
  }

  /**
   * Finger interaction with the constraint stack: the same pointing-dwell
   * model as the mode bar, active only while the buttons are visible
   * (EDIT mode; the rects are empty otherwise).
   */
  private updateConstraintInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    const target = this.pointedButton(frame, view, this.constraintRects);
    const fired = this.constraintDwell.update(target, dt, this.dwellMs);
    if (fired >= 0) this.requestConstraint(fired);
  }

  /**
   * Shared layout of the EDIT-mode vertical stack (drag constraints, then the
   * CSG Boolean tools below them), in the column directly under [ EDIT ]:
   * left edge (`margin`) / gap / button size + the top of the first stack
   * (just under the mode bar).
   */
  private editStackMetrics(cssWidth: number): {
    margin: number;
    gap: number;
    height: number;
    width: number;
    top: number;
  } {
    const { margin, gap, height: barHeight } = this.barMetrics(cssWidth);
    // EDIT options: a column of square icon buttons directly below
    // [ VIEW ] (the trash bin's spot in VIEW mode — never shown together).
    const size = this.iconButtonSize(cssWidth);
    return {
      margin,
      gap,
      height: size,
      width: size,
      top: margin + barHeight + gap,
    };
  }

  /** Side length of the square icon buttons — trash bin, EDIT tools, shapes (CSS px). */
  private iconButtonSize(cssWidth: number): number {
    return Math.max(30, Math.round(42 * this.fontScale(cssWidth)));
  }

  /**
   * EDIT-mode drag-constraint toggles, stacked vertically directly below
   * [ EDIT ]: `[ XZ PLANE ]` (default active) and `[ Y AXIS ]` — mutually
   * exclusive, boxy industrial style
   * (crisp border, monospace, inverted background when active) with the
   * same dwell progress bar as the mode buttons.
   */
  private drawConstraintButtons(cssWidth: number): void {
    const { margin, gap, height, width, top } = this.editStackMetrics(cssWidth);

    this.constraintRects = [];
    CONSTRAINT_BUTTONS.forEach((constraint, i) => {
      const x = margin;
      const y = top + i * (height + gap);
      const rect = { x, y, width, height };
      this.constraintRects.push(rect);

      let ink: string;
      if (this.dragConstraint === constraint) {
        this.drawActiveButton(rect, 6, 3);
        ink = '#0f172a';
      } else {
        const progress = this.constraintDwell.progress(i, this.dwellMs);
        this.drawIdleButton(rect, progress, 1);
        ink = progress !== null ? '#f8fafc' : '#cbd5e1';
      }
      this.drawEditIcon(constraint, x + width / 2, y + height / 2 - 1, width * 0.3, ink);
    });
  }

  /* ------------------------------------------------------------------ */
  /* CSG Boolean tool toggles (SELECT mode, below the constraints)      */
  /* ------------------------------------------------------------------ */


  /**
   * Activate a Boolean toggle: mutually exclusive (arming the other tool
   * switches), re-pressing the armed one disarms it (`null` request). The
   * host arms its CAD-side tool and — per its own clash state — may fire
   * the operation immediately.
   */
  private requestBoolean(index: number): void {
    const tool = BOOLEAN_BUTTONS[index];
    if (!tool) return;
    const next = this.armedBooleanTool === tool ? null : tool;
    this.armedBooleanTool = next;
    this.onBooleanToolRequest?.(next);
  }

  /**
   * Boolean interaction: pointing index-tip dwell on the toggles (the same
   * model as the constraint stack) plus the secondary-hand "X" cross pose
   * (index + pinky extended, middle + ring folded) held for a few frames —
   * the trigger that fires the armed operation. SELECT mode only.
   */
  private updateBooleanInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    if (frame.mode !== 'select') {
      this.booleanDwell.reset();
      this.xCrossFrames = 0;
      this.xCrossFired = false;
      return;
    }
    // X-cross trigger: debounced pose hold, fires at most once per pose.
    if (frame.hands.some((hand) => hand.xCross)) {
      this.xCrossFrames++;
      if (!this.xCrossFired && this.xCrossFrames >= this.booleanTriggerFrames) {
        this.xCrossFired = true;
        this.onBooleanTrigger?.();
      }
    } else {
      this.xCrossFrames = 0;
      this.xCrossFired = false;
    }
    // Pointing dwell on the toggles.
    const target = this.pointedButton(frame, view, this.booleanRects);
    const fired = this.booleanDwell.update(target, dt, this.dwellMs);
    if (fired >= 0) this.requestBoolean(fired);
  }

  /**
   * SELECT-mode CSG Boolean tool toggles, stacked below the drag-constraint
   * toggles: `[ SUBTRACT ]` and `[ UNION ]` — mutually
   * exclusive, boxy, re-press disarms. While a tool is armed *and* the
   * selection intersects another mesh (live `booleanState` provider), the
   * armed button glows amber: the operation is ready to fire.
   */
  private drawBooleanButtons(cssWidth: number): void {
    const { margin, gap, height, width, top } = this.editStackMetrics(cssWidth);
    const booleanTop = top + CONSTRAINT_BUTTONS.length * (height + gap);
    const state = this.booleanState?.() ?? null;

    this.booleanRects = [];
    BOOLEAN_BUTTONS.forEach((tool, i) => {
      const x = margin;
      const y = booleanTop + i * (height + gap);
      const rect = { x, y, width, height };
      this.booleanRects.push(rect);

      let ink: string;
      if (this.armedBooleanTool === tool) {
        // Ready-to-run cue: amber outline + indicator bar (a clash is live).
        const ready = state !== null && state.clash;
        if (ready) this.drawActiveButton(rect, 6, 3, BOOLEAN_READY_STROKE, BOOLEAN_READY_STROKE, 2.5);
        else this.drawActiveButton(rect, 6, 3);
        ink = ready ? BOOLEAN_READY_INK : '#0f172a';
      } else {
        const progress = this.booleanDwell.progress(i, this.dwellMs);
        this.drawIdleButton(rect, progress, 1);
        ink = progress !== null ? '#f8fafc' : '#cbd5e1';
      }
      this.drawEditIcon(tool, x + width / 2, y + height / 2 - 1, width * 0.3, ink);
    });
  }

  /**
   * Line icons for the EDIT-mode tools, centered at (cx, cy), half-size r:
   * - `xz` — slide on the floor: a floor plane (parallelogram) with a
   *   four-way move arrow;
   * - `y` — lift / lower: a double-headed vertical arrow over the floor;
   * - `subtract` — cut a hole: a square, a dashed circle biting its corner,
   *   and a minus sign;
   * - `union` — merge: an overlapping square + circle and a plus sign;
   * - `ungroup` — split: a square and a circle pulled apart, with arrows
   *   pointing away from each other.
   */
  private drawEditIcon(
    icon: DragConstraint | OverlayBooleanTool | 'ungroup',
    cx: number,
    cy: number,
    r: number,
    ink: string
  ): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1.5, r * 0.13);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    const arrowHead = (x: number, y: number, dx: number, dy: number) => {
      const h = r * 0.28;
      ctx.moveTo(x, y);
      ctx.lineTo(x - dx * h + dy * h * 0.7, y - dy * h - dx * h * 0.7);
      ctx.moveTo(x, y);
      ctx.lineTo(x - dx * h - dy * h * 0.7, y - dy * h + dx * h * 0.7);
    };
    ctx.beginPath();
    if (icon === 'xz') {
      // Floor plane in perspective.
      ctx.moveTo(cx - r * 0.55, cy - r * 0.45);
      ctx.lineTo(cx + r, cy - r * 0.45);
      ctx.lineTo(cx + r * 0.55, cy + r * 0.45);
      ctx.lineTo(cx - r, cy + r * 0.45);
      ctx.closePath();
      ctx.stroke();
      // Four-way move arrow on the plane.
      ctx.beginPath();
      ctx.moveTo(cx - r * 0.55, cy);
      ctx.lineTo(cx + r * 0.55, cy);
      ctx.moveTo(cx + r * 0.12, cy - r * 0.3);
      ctx.lineTo(cx - r * 0.12, cy + r * 0.3);
      arrowHead(cx + r * 0.55, cy, 1, 0);
      arrowHead(cx - r * 0.55, cy, -1, 0);
      arrowHead(cx + r * 0.12, cy - r * 0.3, 0.37, -0.93);
      arrowHead(cx - r * 0.12, cy + r * 0.3, -0.37, 0.93);
    } else if (icon === 'ungroup') {
      // Two parts pulled apart + outward arrows.
      ctx.rect(cx - r, cy - r * 0.2, r * 0.8, r * 0.8);
      ctx.moveTo(cx + r * 0.95, cy + r * 0.2);
      ctx.arc(cx + r * 0.55, cy + r * 0.2, r * 0.4, 0, Math.PI * 2);
      ctx.moveTo(cx - r * 0.15, cy - r * 0.7);
      ctx.lineTo(cx - r * 0.85, cy - r * 0.7);
      arrowHead(cx - r * 0.85, cy - r * 0.7, -1, 0);
      ctx.moveTo(cx + r * 0.15, cy - r * 0.7);
      ctx.lineTo(cx + r * 0.85, cy - r * 0.7);
      arrowHead(cx + r * 0.85, cy - r * 0.7, 1, 0);
    } else if (icon === 'y') {
      // Floor line + vertical double arrow.
      ctx.moveTo(cx - r * 0.8, cy + r);
      ctx.lineTo(cx + r * 0.8, cy + r);
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx, cy + r * 0.62);
      arrowHead(cx, cy - r, 0, -1);
      arrowHead(cx, cy + r * 0.62, 0, 1);
    } else {
      // Square + circle overlapping its top-right corner, with a sign.
      ctx.rect(cx - r * 0.95, cy - r * 0.55, r * 1.35, r * 1.35);
      ctx.stroke();
      ctx.beginPath();
      if (icon === 'subtract') ctx.setLineDash([r * 0.2, r * 0.16]);
      ctx.arc(cx + r * 0.42, cy - r * 0.5, r * 0.55, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      // Sign in the lower-right, outside both shapes.
      const gx = cx + r * 0.72;
      const gy = cy + r * 0.62;
      const g = r * 0.28;
      ctx.moveTo(gx - g, gy);
      ctx.lineTo(gx + g, gy);
      if (icon === 'union') {
        ctx.moveTo(gx, gy - g);
        ctx.lineTo(gx, gy + g);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  /* ------------------------------------------------------------------ */
  /* UNGROUP action (EDIT mode, below the Boolean toggles)              */
  /* ------------------------------------------------------------------ */

  /** Whether the host reports an ungroupable (union) selection. */
  private canUngroupNow(): boolean {
    return this.booleanState?.()?.canUngroup === true;
  }

  private requestUngroup(): void {
    this.ungroupDwell.reset();
    if (this.canUngroupNow()) this.onUngroupRequest?.();
  }

  /** Pointing dwell on UNGROUP (only while the selection is a union). */
  private updateUngroupInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    const target = this.canUngroupNow() ? this.pointedButton(frame, view, [this.ungroupRect]) : -1;
    if (this.ungroupDwell.update(target, dt, this.dwellMs) >= 0) this.requestUngroup();
  }

  /**
   * UNGROUP: a one-shot icon button (two parts pulled apart) at the bottom
   * of the EDIT tool column. Dimmed and inert unless the selection is a
   * union; a pointing dwell fills its progress bar.
   */
  private drawUngroupButton(cssWidth: number): void {
    const { margin, gap, height, width, top } = this.editStackMetrics(cssWidth);
    const x = margin;
    const y = top + (CONSTRAINT_BUTTONS.length + BOOLEAN_BUTTONS.length) * (height + gap);
    const rect = { x, y, width, height };
    this.ungroupRect = rect;
    const enabled = this.canUngroupNow();
    const progress = enabled ? this.ungroupDwell.progress(0, this.dwellMs) : null;
    this.drawIdleButton(rect, progress, 1, enabled ? undefined : 'rgba(148, 163, 184, 0.2)');
    const ink =
      progress !== null ? '#f8fafc' : enabled ? '#cbd5e1' : 'rgba(148, 163, 184, 0.35)';
    this.drawEditIcon('ungroup', x + width / 2, y + height / 2 - 1, width * 0.3, ink);
  }

  /* ------------------------------------------------------------------ */
  /* In-vision trash bin (top-left, below [ VIEW ])                      */
  /* ------------------------------------------------------------------ */

  /** Fire the trash request (pointing dwell / mouse click). */
  private requestTrash(): void {
    this.trashDwell.reset();
    this.onTrashRequest?.();
  }

  /**
   * Pointing index-tip dwell on the trash bin: `trashDwellMs` (600 ms by
   * default — a touch slower than a mode button, since this one clears the
   * whole scene through the confirmation dialog).
   */
  private updateTrashInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    const target = this.pointedButton(frame, view, [this.trashRect]);
    if (this.trashDwell.update(target, dt, this.trashDwellMs) >= 0) this.requestTrash();
  }

  /**
   * In-vision trash bin (VIEW mode only): a boxy recycle-bin icon button in
   * the top-left corner, directly below the [ VIEW ] mode button (danger-tinted, matching
   * the Delete HUD). Only a pointing index fingertip held on it (or a mouse
   * click) activates it — a pointing dwell fills a progress bar along its
   * bottom edge; pinches never fire it. Hidden while the confirmation dialog
   * is open.
   */
  private drawTrashButton(cssWidth: number): void {
    const size = this.iconButtonSize(cssWidth);
    const { margin, gap, height: barHeight } = this.barMetrics(cssWidth);
    const x = margin;
    const y = margin + barHeight + gap;
    const rect = { x, y, width: size, height: size };
    this.trashRect = rect;

    const progress = this.trashDwell.progress(0, this.trashDwellMs);
    this.drawIdleButton(rect, progress, 1.5, TRASH_STROKE, TRASH_PROGRESS_FILL);
    this.drawTrashIcon(x + size / 2, y + size / 2, size * 0.3, progress !== null ? '#f8fafc' : TRASH_INK);
  }

  /** Recycle-bin line icon (lid + can + slats), centered at (cx, cy), half-size r. */
  private drawTrashIcon(cx: number, cy: number, r: number, ink: string): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1.5, r * 0.16);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    // Lid: a horizontal bar with a small lift handle.
    ctx.moveTo(cx - r, cy - r * 0.62);
    ctx.lineTo(cx + r, cy - r * 0.62);
    ctx.moveTo(cx - r * 0.32, cy - r * 0.62);
    ctx.lineTo(cx - r * 0.22, cy - r * 0.95);
    ctx.lineTo(cx + r * 0.22, cy - r * 0.95);
    ctx.lineTo(cx + r * 0.32, cy - r * 0.62);
    // Can body: slightly tapered, resting on the button's baseline.
    ctx.moveTo(cx - r * 0.72, cy - r * 0.38);
    ctx.lineTo(cx - r * 0.58, cy + r);
    ctx.lineTo(cx + r * 0.58, cy + r);
    ctx.lineTo(cx + r * 0.72, cy - r * 0.38);
    // Vertical slats.
    ctx.moveTo(cx - r * 0.18, cy - r * 0.16);
    ctx.lineTo(cx - r * 0.12, cy + r * 0.62);
    ctx.moveTo(cx + r * 0.18, cy - r * 0.16);
    ctx.lineTo(cx + r * 0.12, cy + r * 0.62);
    ctx.stroke();
    ctx.restore();
  }

  /* ------------------------------------------------------------------ */
  /* Spatial confirmation dialog (in-vision, gesture-answered)         */
  /* ------------------------------------------------------------------ */

  /** The dialog was answered "yes": report it and close. */
  private answerConfirm(): void {
    const intent = this.openIntent;
    if (intent === null) return;
    this.closeConfirm();
    this.onConfirmRequest?.(intent);
  }

  /** The dialog was dismissed: report it and close (nothing destructive). */
  private answerCancel(): void {
    const intent = this.openIntent;
    if (intent === null) return;
    this.closeConfirm();
    this.onCancelRequest?.(intent);
  }

  /**
   * Dialog gestures, per frame while open:
   * - **Confirm**: a pointing index fingertip held on the confirm target, or
   *   the "OK" gesture (thumb + index loop, middle / ring / pinky extended)
   *   held anywhere in view, for `confirmHoldMs`. Pinches never answer it.
   * - **Cancel**: a pointing hold on the cancel target (`confirmHoldMs`), or
   *   an open palm held for `confirmPalmCancelMs` — suppressed
   *   during the first moments after opening, because the pinch that
   *   triggered the dialog releases into an open hand — or simply moving
   *   every hand away for `confirmHandLossFrames` frames.
   */
  private updateConfirmInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    this.confirmOpenedAgo += dt;
    if (frame.hands.length === 0) {
      this.confirmNoHandFrames++;
      if (this.confirmNoHandFrames >= this.confirmHandLossFrames) {
        this.answerCancel();
        return;
      }
    } else {
      this.confirmNoHandFrames = 0;
    }
    // Pointing fingertips on the targets (same hold model as the color
    // wheel lock and the Delete button — nothing answers instantly).
    const rects = this.confirmRects;
    let onConfirm = false;
    let onCancel = false;
    if (rects) {
      for (const hand of frame.hands) {
        if (!hand.pointing) continue;
        const tip = this.toCanvas(hand, INDEX_TIP, view);
        if (this.inRect(rects.confirm, tip.x, tip.y)) onConfirm = true;
        else if (this.inRect(rects.cancel, tip.x, tip.y)) onCancel = true;
      }
    }
    // Confirm: hold on the confirm target, or hold the OK gesture.
    if (onConfirm || frame.hands.some((hand) => hand.okGesture)) {
      this.confirmOkElapsed += dt;
      if (this.confirmOkElapsed >= this.confirmHoldMs) {
        this.answerConfirm();
        return;
      }
    } else {
      this.confirmOkElapsed = 0;
    }
    // Cancel: hold on the cancel target.
    if (onCancel) {
      this.confirmCancelElapsed += dt;
      if (this.confirmCancelElapsed >= this.confirmHoldMs) {
        this.answerCancel();
        return;
      }
    } else {
      this.confirmCancelElapsed = 0;
    }
    // Cancel: an open palm held briefly (after the post-open grace).
    if (frame.hands.some((hand) => hand.openPalm) && this.confirmOpenedAgo >= 600) {
      this.confirmPalmElapsed += dt;
      if (this.confirmPalmElapsed >= this.confirmPalmCancelMs) {
        this.answerCancel();
      }
    } else {
      this.confirmPalmElapsed = 0;
    }
  }

  /**
   * The in-vision spatial confirmation dialog: a translucent scrim + card
   * centered on the camera view with two spatial targets —
   * `[ CONFIRM (Hold) ]` (danger red) and `[ CANCEL (Hold) ]` (calm blue).
   * The confirm hold progress fills the confirm target's bottom edge; the
   * cancel hold / open-palm progress fills the cancel target's. No DOM modals, no
   * browser popups — the hands stay visible underneath.
   */
  private drawConfirmDialog(cssWidth: number, cssHeight: number): void {
    const intent = this.openIntent;
    if (intent === null) {
      this.confirmRects = null;
      return;
    }
    const ctx = this.ctx;
    const scale = this.fontScale(cssWidth);
    const margin = Math.round(8 * scale);
    const pad = Math.round(12 * scale);
    const gap = Math.round(6 * scale);
    const cardWidth = Math.min(Math.round(250 * scale), cssWidth - margin * 2);
    const targetHeight = Math.max(24, Math.round(30 * scale));
    const titleSize = Math.max(10, Math.round(13 * scale));
    const hintSize = Math.max(8, Math.round(9 * scale));
    const title = CONFIRM_TITLES[intent];
    const hint = 'POINT & HOLD — PALM OR AWAY CANCELS';
    const confirmLabel = '[ CONFIRM (Hold) ]';
    const cancelLabel = '[ CANCEL (Hold) ]';

    // Lay out: title / hint / two side-by-side targets inside the card.
    const targetWidth = (cardWidth - pad * 2 - gap) / 2;
    let labelSize = Math.max(8, Math.round(10 * scale));
    ctx.font = `bold ${labelSize}px ui-monospace, monospace`;
    while (
      labelSize > 7 &&
      (ctx.measureText(confirmLabel).width > targetWidth - 6 ||
        ctx.measureText(cancelLabel).width > targetWidth - 6)
    ) {
      labelSize -= 1;
      ctx.font = `bold ${labelSize}px ui-monospace, monospace`;
    }
    const cardHeight = pad * 2 + titleSize + hintSize + targetHeight + gap * 3;
    const cardX = (cssWidth - cardWidth) / 2;
    const cardY = (cssHeight - cardHeight) / 2;
    const targetY = cardY + pad + titleSize + hintSize + gap * 2;
    const confirmRect: ButtonRect = {
      x: cardX + pad,
      y: targetY,
      width: targetWidth,
      height: targetHeight,
    };
    const cancelRect: ButtonRect = {
      x: cardX + pad + targetWidth + gap,
      y: targetY,
      width: targetWidth,
      height: targetHeight,
    };
    this.confirmRects = { confirm: confirmRect, cancel: cancelRect };

    // Scrim + card.
    ctx.fillStyle = DIALOG_SCRIM;
    ctx.fillRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = DIALOG_CARD_FILL;
    ctx.fillRect(cardX, cardY, cardWidth, cardHeight);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = DIALOG_CARD_STROKE;
    ctx.strokeRect(cardX + 1, cardY + 1, cardWidth - 2, cardHeight - 2);

    // Title + hint.
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = DIALOG_TITLE;
    ctx.font = `bold ${titleSize}px ui-monospace, monospace`;
    ctx.fillText(title, cssWidth / 2, cardY + pad + titleSize / 2);
    ctx.fillStyle = DIALOG_HINT;
    ctx.font = `${hintSize}px ui-monospace, monospace`;
    ctx.fillText(hint, cssWidth / 2, cardY + pad + titleSize + gap + hintSize / 2);

    // Confirm target (danger red) with the hold progress fill.
    ctx.fillStyle = DIALOG_CONFIRM_FILL;
    ctx.fillRect(confirmRect.x, confirmRect.y, confirmRect.width, confirmRect.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = DIALOG_CONFIRM_STROKE;
    ctx.strokeRect(
      confirmRect.x + 1,
      confirmRect.y + 1,
      confirmRect.width - 2,
      confirmRect.height - 2
    );
    ctx.fillStyle = DIALOG_CONFIRM_INK;
    ctx.font = `bold ${labelSize}px ui-monospace, monospace`;
    ctx.fillText(
      confirmLabel,
      confirmRect.x + confirmRect.width / 2,
      confirmRect.y + confirmRect.height / 2
    );
    if (this.confirmHoldMs > 0 && this.confirmOkElapsed > 0) {
      const progress = Math.min(1, this.confirmOkElapsed / this.confirmHoldMs);
      ctx.fillStyle = DIALOG_OK_PROGRESS;
      ctx.fillRect(
        confirmRect.x + 2,
        confirmRect.y + confirmRect.height - 4.5,
        (confirmRect.width - 4) * progress,
        3
      );
    }

    // Cancel target (calm blue) with the open-palm progress fill.
    ctx.fillStyle = DIALOG_CANCEL_FILL;
    ctx.fillRect(cancelRect.x, cancelRect.y, cancelRect.width, cancelRect.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = DIALOG_CANCEL_STROKE;
    ctx.strokeRect(
      cancelRect.x + 1,
      cancelRect.y + 1,
      cancelRect.width - 2,
      cancelRect.height - 2
    );
    ctx.fillStyle = DIALOG_CANCEL_INK;
    ctx.fillText(
      cancelLabel,
      cancelRect.x + cancelRect.width / 2,
      cancelRect.y + cancelRect.height / 2
    );
    const cancelProgress = Math.max(
      this.confirmPalmCancelMs > 0 ? this.confirmPalmElapsed / this.confirmPalmCancelMs : 0,
      this.confirmHoldMs > 0 ? this.confirmCancelElapsed / this.confirmHoldMs : 0
    );
    if (cancelProgress > 0) {
      const progress = Math.min(1, cancelProgress);
      ctx.fillStyle = DIALOG_CANCEL_STROKE;
      ctx.fillRect(
        cancelRect.x + 2,
        cancelRect.y + cancelRect.height - 4.5,
        (cancelRect.width - 4) * progress,
        3
      );
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  /* ------------------------------------------------------------------ */
  /* Shape picker (CREATE mode, row below the mode bar)                 */
  /* ------------------------------------------------------------------ */


  private requestShape(index: number): void {
    const shape = this.shapes[index];
    if (!shape || this.activeShapeId === shape.id) return;
    this.activeShapeId = shape.id;
    this.onShapeRequest?.(shape.id);
  }

  /**
   * Finger interaction with the shape row: the same pointing-dwell model as
   * the mode bar, active only while the row is visible (CREATE).
   */
  private updateShapeInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    const target = this.pointedButton(frame, view, this.shapeRects);
    const fired = this.shapeDwell.update(target, dt, this.dwellMs);
    if (fired >= 0) this.requestShape(fired);
  }

  /**
   * CREATE-mode shape picker: square icon buttons stacked vertically down
   * the right edge, below the mode bar — mutually exclusive boxy toggles
   * (inverted when active) with the same dwell progress bar as the mode
   * buttons. Shapes without an `icon` fall back to their text label.
   */
  private drawShapeButtons(cssWidth: number): void {
    this.shapeRects = [];
    if (this.shapes.length === 0) return;
    const ctx = this.ctx;
    const scale = this.fontScale(cssWidth);
    const { margin, gap, height: barHeight } = this.barMetrics(cssWidth);
    // Same square size as the EDIT tool / trash icon buttons.
    const size = this.iconButtonSize(cssWidth);
    const x = cssWidth - margin - size;
    const top = margin + barHeight + gap;

    this.shapes.forEach((shape, i) => {
      const y = top + i * (size + gap);
      const rect = { x, y, width: size, height: size };
      this.shapeRects.push(rect);
      let ink: string;
      if (this.activeShapeId === shape.id) {
        this.drawActiveButton(rect, 5, 3);
        ink = '#0f172a';
      } else {
        const progress = this.shapeDwell.progress(i, this.dwellMs);
        this.drawIdleButton(rect, progress, 1.5);
        ink = progress !== null ? '#f8fafc' : '#cbd5e1';
      }
      if (shape.icon) {
        this.drawShapeIcon(shape.icon, x + size / 2, y + size / 2 - 1, size * 0.3, ink);
      } else {
        ctx.fillStyle = ink;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `bold ${Math.max(7, Math.round(9 * scale))}px ui-monospace, monospace`;
        ctx.fillText(shape.label.slice(0, 4), x + size / 2, y + size / 2);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      }
    });
  }

  /** Line icon for a shape button, centered at (cx, cy) with half-size r. */
  private drawShapeIcon(icon: OverlayShapeIcon, cx: number, cy: number, r: number, ink: string): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1.5, r * 0.14);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    if (icon === 'cube' || icon === 'cuboid') {
      // Isometric box: outer hexagon + the three edges meeting at the near
      // top corner. The cuboid is stretched wide and squat.
      const sx = icon === 'cuboid' ? 1.3 : 1;
      const sy = icon === 'cuboid' ? 0.75 : 1;
      const hx = r * 0.87 * sx;
      const hy = r * 0.5 * sy;
      const v = r * sy; // vertical edge length
      const top = { x: cx, y: cy - hy - v / 2 };
      const near = { x: cx, y: cy + hy - v / 2 };
      const pts = [
        top,
        { x: cx + hx, y: cy - v / 2 },
        { x: cx + hx, y: cy + v / 2 },
        { x: cx, y: cy + hy + v / 2 },
        { x: cx - hx, y: cy + v / 2 },
        { x: cx - hx, y: cy - v / 2 },
      ];
      ctx.moveTo(pts[0].x, pts[0].y);
      for (const p of pts.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.closePath();
      ctx.moveTo(pts[5].x, pts[5].y);
      ctx.lineTo(near.x, near.y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.moveTo(near.x, near.y);
      ctx.lineTo(pts[3].x, pts[3].y);
    } else if (icon === 'cylinder') {
      const rx = r * 0.8;
      const ry = r * 0.3;
      const h = r * 1.3;
      ctx.ellipse(cx, cy - h / 2, rx, ry, 0, 0, Math.PI * 2);
      ctx.moveTo(cx - rx, cy - h / 2);
      ctx.lineTo(cx - rx, cy + h / 2);
      ctx.ellipse(cx, cy + h / 2, rx, ry, 0, Math.PI, 0, true);
      ctx.lineTo(cx + rx, cy - h / 2);
    } else {
      // Sphere: outline + equator and meridian ellipses.
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.moveTo(cx + r, cy);
      ctx.ellipse(cx, cy, r, r * 0.35, 0, 0, Math.PI * 2);
      ctx.moveTo(cx, cy - r);
      ctx.ellipse(cx, cy, r * 0.35, r, 0, -Math.PI / 2, Math.PI * 1.5);
    }
    ctx.stroke();
    ctx.restore();
  }

}
