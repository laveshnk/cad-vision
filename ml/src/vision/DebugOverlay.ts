/**
 * DebugOverlay: 2D-canvas debug renderer for the gesture engine.
 *
 * Draws the mirrored webcam landmarks (21 per hand) with the MediaPipe skeleton,
 * pinch indicators, and a HUD with live metrics. Pure Canvas 2D — no WebGL.
 *
 * Along the top edge it renders the interaction-mode switcher ([ VIEW ],
 * [ SELECT ], [ CREATE ]): boxy, mutually exclusive toggle buttons that
 * respond to mouse clicks, an index-tip dwell (>= `dwellMs`) or a pinch
 * that *begins* over a button — reported through the `onModeRequest`
 * callback. In SELECT mode a second stack of mutually exclusive toggles,
 * [ XZ PLANE ] (default) and [ Y AXIS (ELEVATE) ], sits vertically below
 * the mode bar in the top-left corner (`onDragConstraintRequest`); only a
 * fresh pinch activates a button, so object drags sweeping across the bars
 * never toggle anything. The stats HUD is anchored bottom-left so the
 * buttons own the top edge.
 *
 * Alignment: the <video> is CSS-mirrored (`scaleX(-1)`) and displayed with
 * `object-fit: cover`, which center-crops it into its container (the floating
 * camera thumbnail, or that thumbnail expanded). The canvas backing store
 * therefore matches the *container* (CSS px × devicePixelRatio, re-measured
 * every frame), and landmarks are mapped through the same mirrored cover
 * transform so they land exactly on the webcam image — while the HUD stays
 * anchored to the visible container edges (never cropped). Typography scales
 * with the container width (see `fontScale`).
 *
 * In SELECT mode the overlay additionally renders a live AR spatial mirror
 * of the 3D CAD scene (translucent ground grid + mesh ghosts) through the
 * `arScene` provider — plain projected 2D data supplied by the host, so this
 * module stays free of Three.js.
 */

import { HandLandmarker } from '@mediapipe/tasks-vision';
import type {
  FrameEvent,
  GestureState,
  HandSnapshot,
  Handedness,
  InteractionMode,
} from './types';

const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;

/** A skeleton connection between two landmark indices. */
export interface SkeletonConnection {
  start: number;
  end: number;
}

/**
 * State color-coding: green = pinch/draw, cyan = select, blue = orbit,
 * purple = zoom, yellow = idle.
 */
export const STATE_COLORS: Record<GestureState, string> = {
  IDLE: '#facc15', // yellow
  DRAWING_BASE: '#22c55e', // green
  SELECTING: '#06b6d4', // cyan
  EXTRUDING: '#f97316', // orange
  ORBITING: '#3b82f6', // blue
  ZOOMING: '#a855f7', // purple
};

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
 * Projects the CAD scene onto the overlay canvas — called once per rendered
 * frame (only in SELECT mode) with the current canvas size in CSS px.
 * Returns `null` to skip the AR layer entirely.
 */
export type ArSceneProvider = (cssWidth: number, cssHeight: number) => ArSceneFrame | null;

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

/** Options for the debug overlay. */
export interface DebugOverlayOptions {
  /** Hand skeleton connections (defaults to MediaPipe HAND_CONNECTIONS). */
  connections?: ReadonlyArray<SkeletonConnection>;
  /**
   * Called when a mode button is activated — via mouse click, index-tip
   * dwell (>= `dwellMs` over the button) or a pinch on the button. The host
   * decides what to do with the request (typically `engine.setMode`).
   */
  onModeRequest?: (mode: InteractionMode) => void;
  /**
   * Called when a SELECT-mode drag-constraint toggle is activated — via
   * mouse click, index-tip dwell or a pinch on the button. The host routes
   * it to the CAD builder's `setDragConstraint`; the overlay keeps the
   * visual state (both default to 'xz').
   */
  onDragConstraintRequest?: (constraint: DragConstraint) => void;
  /** Index-tip dwell time (ms) before a hovered mode button activates. Default 500. */
  dwellMs?: number;
  /**
   * SELECT-mode AR mirror: projects the live CAD scene (ground grid +
   * committed meshes) through the shared 3D camera onto this canvas, once
   * per rendered frame. Implemented on the CAD side (`buildArSceneFrame`)
   * and injected here by the orchestrator — the overlay stays Three.js-free.
   */
  arScene?: ArSceneProvider;
}

/** A hit-testable rectangle in CSS pixels. */
interface ButtonRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Mode buttons, left to right, along the top edge of the overlay. */
const MODE_BUTTONS: readonly InteractionMode[] = ['view', 'select', 'create'];

/** SELECT-mode drag-constraint toggles, top to bottom (top-left corner). */
const CONSTRAINT_BUTTONS: readonly DragConstraint[] = ['xz', 'y'];

/** Constraint button labels, in CONSTRAINT_BUTTONS order. */
const CONSTRAINT_LABELS: Record<DragConstraint, string> = {
  xz: '[ XZ PLANE ]',
  y: '[ Y AXIS (ELEVATE) ]',
};

export class DebugOverlay {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly connections: ReadonlyArray<SkeletonConnection>;
  private readonly onModeRequest: ((mode: InteractionMode) => void) | null;
  private readonly onDragConstraintRequest: ((constraint: DragConstraint) => void) | null;
  private readonly dwellMs: number;
  private readonly arScene: ArSceneProvider | null;
  /** Last rendered button rects (CSS px) — hit targets for mouse + finger. */
  private buttonRects: ButtonRect[] = [];
  /** Button the index tip is dwelling over (-1 = none). */
  private dwellTarget = -1;
  private dwellElapsed = 0;
  private lastFrameTimestamp: number | null = null;
  /** Latch: one pinch activates at most one button until it leaves the bar. */
  private pinchLatched = false;
  /** Last rendered constraint-button rects (CSS px); empty outside SELECT mode. */
  private constraintRects: ButtonRect[] = [];
  /** Constraint toggle the index tip is dwelling over (-1 = none). */
  private constraintDwellTarget = -1;
  private constraintDwellElapsed = 0;
  /** Latch: one pinch activates at most one constraint toggle. */
  private constraintPinchLatched = false;
  /** Visual + authoritative overlay state of the drag constraint (host mirrors it). */
  private dragConstraint: DragConstraint = 'xz';
  /**
   * Previous frame's pinch state per hand — only a *fresh* pinch (one that
   * just closed over a button) can activate a button, so a pinch-drag
   * sweeping across the button bars never toggles anything mid-gesture.
   */
  private readonly prevPinchStates = new Map<Handedness, boolean>();
  /** Last rendered mirrored cover transform (device-space UI hit tests). */
  private lastView: ViewTransform | null = null;

  constructor(canvas: HTMLCanvasElement, options: DebugOverlayOptions = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('DebugOverlay: 2D canvas context unavailable');
    this.ctx = ctx;
    this.connections = options.connections ?? HandLandmarker.HAND_CONNECTIONS;
    this.onModeRequest = options.onModeRequest ?? null;
    this.onDragConstraintRequest = options.onDragConstraintRequest ?? null;
    this.dwellMs = options.dwellMs ?? 500;
    this.arScene = options.arScene ?? null;
    canvas.addEventListener('click', this.onCanvasClick);
    canvas.addEventListener('mousemove', this.onCanvasMouseMove);
  }

  /** Detach the mode-button mouse listeners (the canvas stays with the host). */
  dispose(): void {
    this.canvas.removeEventListener('click', this.onCanvasClick);
    this.canvas.removeEventListener('mousemove', this.onCanvasMouseMove);
  }

  /** Click anywhere inside a rendered mode / constraint button activates it. */
  private readonly onCanvasClick = (event: MouseEvent): void => {
    const constraint = this.hitConstraintButton(event.offsetX, event.offsetY);
    if (constraint >= 0) {
      this.requestConstraint(constraint);
      return;
    }
    const index = this.hitButton(event.offsetX, event.offsetY);
    if (index >= 0) this.requestMode(index);
  };

  /** Pointer feedback while hovering the mode / constraint buttons. */
  private readonly onCanvasMouseMove = (event: MouseEvent): void => {
    const hovering =
      this.hitButton(event.offsetX, event.offsetY) >= 0 ||
      this.hitConstraintButton(event.offsetX, event.offsetY) >= 0;
    this.canvas.style.cursor = hovering ? 'pointer' : 'default';
  };

  /** Render one frame of landmarks + HUD. */
  render(frame: FrameEvent): void {
    const { cssWidth, cssHeight } = this.ensureSize(frame.video.width, frame.video.height);
    this.ctx.clearRect(0, 0, cssWidth, cssHeight);
    const view = this.coverTransform(cssWidth, cssHeight, frame.video.width, frame.video.height);
    this.lastView = view; // device-space UI hit tests between frames
    // SELECT mode: live AR mirror of the 3D scene, drawn beneath the hands.
    if (frame.mode === 'select') this.drawArMirror(cssWidth, cssHeight);
    for (const hand of frame.hands) {
      this.drawHand(hand, frame.state, view, cssWidth);
    }
    this.drawDualHandsLink(frame, view, cssWidth);
    this.drawZoomAnchor(frame, view);
    this.drawModeButtons(frame, cssWidth);
    if (frame.mode === 'select') this.drawConstraintButtons(cssWidth);
    else this.constraintRects = [];
    const dt = this.frameDt(frame);
    this.updateModeInteraction(frame, view, dt);
    this.updateConstraintInteraction(frame, view, dt);
    this.snapshotPinchStates(frame);
    this.drawHud(frame, cssWidth, cssHeight);
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
   * Remember each hand's pinch state for the next frame: only a *fresh*
   * pinch (one that just closed over a button) may activate a button, so
   * a pinch-drag sweeping across the button bars never toggles anything
   * mid-gesture.
   */
  private snapshotPinchStates(frame: FrameEvent): void {
    this.prevPinchStates.clear();
    for (const hand of frame.hands) {
      this.prevPinchStates.set(hand.handedness, hand.pinchActive);
    }
  }

  /**
   * Device-space hit test against the last rendered UI buttons (mode bar +
   * SELECT-mode constraint stack). The host uses it to keep a UI pinch
   * (toggling a button) from also picking / drawing in the 3D scene. Runs
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
    return this.hitButton(px, py) >= 0 || this.hitConstraintButton(px, py) >= 0;
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
  private drawArMirror(cssWidth: number, cssHeight: number): void {
    if (!this.arScene) return;
    const frame = this.arScene(cssWidth, cssHeight);
    if (!frame) return;
    const ctx = this.ctx;

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
  /* Mode switcher buttons (top edge of the overlay)                    */
  /* ------------------------------------------------------------------ */

  /** CSS-pixel hit test against the last rendered mode buttons. */
  private hitButton(px: number, py: number): number {
    for (let i = 0; i < this.buttonRects.length; i++) {
      const r = this.buttonRects[i];
      if (px >= r.x && px <= r.x + r.width && py >= r.y && py <= r.y + r.height) return i;
    }
    return -1;
  }

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
   * Finger interaction with the mode buttons: the index tip (landmark 8)
   * dwelling over a button for `dwellMs` activates it; a *fresh* pinch that
   * closes while the tip is over a button activates it immediately (latched
   * once per pinch so the same pinch cannot scrub across buttons — and a
   * pinch that began elsewhere, e.g. a mesh drag, never toggles a mode as
   * it sweeps across the bar).
   */
  private updateModeInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    let dwellTarget = -1;
    let pinchTarget = -1;
    for (const hand of frame.hands) {
      const tip = this.toCanvas(hand, INDEX_TIP, view);
      const index = this.hitButton(tip.x, tip.y);
      if (index < 0) continue;
      if (dwellTarget < 0) dwellTarget = index;
      const freshPinch = hand.pinchActive && !this.prevPinchStates.get(hand.handedness);
      if (pinchTarget < 0 && freshPinch) pinchTarget = index;
    }

    if (pinchTarget >= 0 && !this.pinchLatched) {
      this.requestMode(pinchTarget);
      this.pinchLatched = true;
    }
    if (pinchTarget < 0) this.pinchLatched = false;

    // Dwell clock from frame timestamps; capped so a stalled camera feed
    // cannot complete a dwell in one jump.
    if (dwellTarget !== this.dwellTarget) {
      this.dwellTarget = dwellTarget;
      this.dwellElapsed = 0;
    } else if (dwellTarget >= 0 && dt > 0) {
      this.dwellElapsed += dt;
    }
    if (this.dwellTarget >= 0 && this.dwellElapsed >= this.dwellMs) {
      this.requestMode(this.dwellTarget);
      this.dwellTarget = -1;
      this.dwellElapsed = 0;
    }
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
      const label = `[ ${mode.toUpperCase()} ]`;
      // Shrink the label to fit the button.
      let fontSize = Math.max(9, Math.round(13 * scale));
      ctx.font = `bold ${fontSize}px ui-monospace, monospace`;
      while (fontSize > 8 && ctx.measureText(label).width > width - 8) {
        fontSize -= 1;
        ctx.font = `bold ${fontSize}px ui-monospace, monospace`;
      }

      const active = frame.mode === mode;
      if (active) {
        ctx.fillStyle = '#f8fafc'; // solid inverted background
        ctx.fillRect(x, margin, width, height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#0f172a';
        ctx.strokeRect(x + 1, margin + 1, width - 2, height - 2);
        ctx.fillStyle = '#0284c7'; // high-contrast indicator bar
        ctx.fillRect(x + 3, margin + height - 7, width - 6, 4);
        ctx.fillStyle = '#0f172a';
      } else {
        const dwelling = this.dwellTarget === i;
        ctx.fillStyle = dwelling ? 'rgba(15, 23, 42, 0.85)' : 'rgba(15, 23, 42, 0.6)';
        ctx.fillRect(x, margin, width, height);
        ctx.lineWidth = dwelling ? 2 : 1.5;
        ctx.strokeStyle = dwelling ? '#e2e8f0' : 'rgba(148, 163, 184, 0.55)';
        ctx.strokeRect(x + 1, margin + 1, width - 2, height - 2);
        if (dwelling && this.dwellMs > 0) {
          const progress = Math.min(1, this.dwellElapsed / this.dwellMs);
          ctx.fillStyle = '#38bdf8';
          ctx.fillRect(x + 2, margin + height - 5, (width - 4) * progress, 3);
        }
        ctx.fillStyle = dwelling ? '#f8fafc' : '#cbd5e1';
      }
      ctx.fillText(label, x + width / 2, margin + height / 2);
    });
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  /* ------------------------------------------------------------------ */
  /* Drag-constraint toggles (SELECT mode, top-left corner)             */
  /* ------------------------------------------------------------------ */

  /** CSS-pixel hit test against the last rendered constraint buttons. */
  private hitConstraintButton(px: number, py: number): number {
    for (let i = 0; i < this.constraintRects.length; i++) {
      const r = this.constraintRects[i];
      if (px >= r.x && px <= r.x + r.width && py >= r.y && py <= r.y + r.height) return i;
    }
    return -1;
  }

  private requestConstraint(index: number): void {
    const constraint = CONSTRAINT_BUTTONS[index];
    if (!constraint || this.dragConstraint === constraint) return;
    this.dragConstraint = constraint;
    this.onDragConstraintRequest?.(constraint);
  }

  /**
   * Finger interaction with the constraint stack: the same dwell /
   * fresh-pinch model as the mode bar, active only while the buttons are
   * visible (SELECT mode).
   */
  private updateConstraintInteraction(frame: FrameEvent, view: ViewTransform, dt: number): void {
    if (frame.mode !== 'select') {
      this.constraintDwellTarget = -1;
      this.constraintDwellElapsed = 0;
      this.constraintPinchLatched = false;
      return;
    }
    let dwellTarget = -1;
    let pinchTarget = -1;
    for (const hand of frame.hands) {
      const tip = this.toCanvas(hand, INDEX_TIP, view);
      const index = this.hitConstraintButton(tip.x, tip.y);
      if (index < 0) continue;
      if (dwellTarget < 0) dwellTarget = index;
      const freshPinch = hand.pinchActive && !this.prevPinchStates.get(hand.handedness);
      if (pinchTarget < 0 && freshPinch) pinchTarget = index;
    }

    if (pinchTarget >= 0 && !this.constraintPinchLatched) {
      this.requestConstraint(pinchTarget);
      this.constraintPinchLatched = true;
    }
    if (pinchTarget < 0) this.constraintPinchLatched = false;

    if (dwellTarget !== this.constraintDwellTarget) {
      this.constraintDwellTarget = dwellTarget;
      this.constraintDwellElapsed = 0;
    } else if (dwellTarget >= 0 && dt > 0) {
      this.constraintDwellElapsed += dt;
    }
    if (this.constraintDwellTarget >= 0 && this.constraintDwellElapsed >= this.dwellMs) {
      this.requestConstraint(this.constraintDwellTarget);
      this.constraintDwellTarget = -1;
      this.constraintDwellElapsed = 0;
    }
  }

  /**
   * SELECT-mode drag-constraint toggles, stacked vertically in the top-left
   * corner below the mode bar: `[ XZ PLANE ]` (default active) and
   * `[ Y AXIS (ELEVATE) ]` — mutually exclusive, boxy industrial style
   * (crisp border, monospace, inverted background when active) with the
   * same dwell progress bar as the mode buttons.
   */
  private drawConstraintButtons(cssWidth: number): void {
    const ctx = this.ctx;
    const scale = this.fontScale(cssWidth);
    const { margin, gap, height: barHeight } = this.barMetrics(cssWidth);
    const height = Math.max(20, Math.round(26 * scale));
    const width = Math.min(Math.round((cssWidth - margin * 2) * 0.55), 190);
    const top = margin + barHeight + gap;

    this.constraintRects = [];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    CONSTRAINT_BUTTONS.forEach((constraint, i) => {
      const x = margin;
      const y = top + i * (height + gap);
      this.constraintRects.push({ x, y, width, height });
      const label = CONSTRAINT_LABELS[constraint];
      // Shrink the label to fit the button.
      let fontSize = Math.max(9, Math.round(12 * scale));
      ctx.font = `bold ${fontSize}px ui-monospace, monospace`;
      while (fontSize > 7 && ctx.measureText(label).width > width - 8) {
        fontSize -= 1;
        ctx.font = `bold ${fontSize}px ui-monospace, monospace`;
      }

      const active = this.dragConstraint === constraint;
      if (active) {
        ctx.fillStyle = '#f8fafc'; // solid inverted background
        ctx.fillRect(x, y, width, height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#0f172a';
        ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
        ctx.fillStyle = '#0284c7'; // high-contrast indicator bar
        ctx.fillRect(x + 3, y + height - 6, width - 6, 3);
        ctx.fillStyle = '#0f172a';
      } else {
        const dwelling = this.constraintDwellTarget === i;
        ctx.fillStyle = dwelling ? 'rgba(15, 23, 42, 0.85)' : 'rgba(15, 23, 42, 0.6)';
        ctx.fillRect(x, y, width, height);
        ctx.lineWidth = dwelling ? 2 : 1;
        ctx.strokeStyle = dwelling ? '#e2e8f0' : 'rgba(148, 163, 184, 0.55)';
        ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
        if (dwelling && this.dwellMs > 0) {
          const progress = Math.min(1, this.constraintDwellElapsed / this.dwellMs);
          ctx.fillStyle = '#38bdf8';
          ctx.fillRect(x + 2, y + height - 5, (width - 4) * progress, 3);
        }
        ctx.fillStyle = dwelling ? '#f8fafc' : '#cbd5e1';
      }
      ctx.fillText(label, x + width / 2, y + height / 2);
    });
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  /** Rounded HUD panel (bottom-left, always inside the visible panel) with mode, state, FPS, hands and live metrics. */
  private drawHud(frame: FrameEvent, cssWidth: number, cssHeight: number): void {
    const ctx = this.ctx;
    const lines: string[] = [];
    lines.push(`MODE: ${frame.mode.toUpperCase()}`);
    lines.push(`STATE: ${frame.state}`);
    lines.push(`FPS: ${frame.fps.toFixed(1)}`);
    lines.push(
      `HANDS: ${frame.hands.length}${
        frame.hands.length > 0 ? ` (${frame.hands.map((h) => h.handedness).join(', ')})` : ''
      }`
    );
    for (const p of frame.metrics.pinchDistances) {
      lines.push(`pinch[${p.hand}]: ${p.distance === Infinity ? '∞' : p.distance.toFixed(3)}`);
    }
    if (frame.metrics.extrusionDistance !== null) {
      lines.push(
        `extrude D: ${frame.metrics.extrusionDistance.toFixed(3)} (Δ ${frame.metrics.extrusionDeltaDistance?.toFixed(3) ?? '0.000'})`
      );
      lines.push(`scale: ×${frame.metrics.extrusionScaleFactor?.toFixed(2) ?? '1.00'}`);
    }
    if (frame.metrics.extrusionHeight !== null) {
      lines.push(`extrude h: ${frame.metrics.extrusionHeight.toFixed(3)}`);
    }
    if (frame.metrics.orbitDelta) {
      lines.push(
        `orbit Δ: (${frame.metrics.orbitDelta.x.toFixed(3)}, ${frame.metrics.orbitDelta.y.toFixed(3)})`
      );
    }
    if (frame.metrics.orbitRoll !== null) {
      lines.push(`roll: ${((frame.metrics.orbitRoll * 180) / Math.PI).toFixed(0)}°`);
    }
    if (frame.metrics.zoomDistance !== null) {
      lines.push(
        `zoom D: ${frame.metrics.zoomDistance.toFixed(3)} ×${(frame.metrics.zoomScaleFactor ?? 1).toFixed(2)}`
      );
    }
    if (frame.metrics.zoomAnchor !== null) {
      const turn = ((frame.metrics.zoomAngle ?? 0) * 180) / Math.PI;
      lines.push(`anchor: ${frame.metrics.zoomAnchor}  turn: ${turn.toFixed(0)}°`);
    }
    if (frame.metrics.selectRotation !== null) {
      const rotate = ((frame.metrics.selectRotation * 180) / Math.PI).toFixed(0);
      lines.push(`rotate: ${rotate}°`);
    }

    const scale = this.fontScale(cssWidth);
    const fontSize = Math.round(13 * scale);
    ctx.font = `${fontSize}px ui-monospace, monospace`;
    const padding = Math.round(12 * scale);
    const lineHeight = Math.max(12, Math.round(18 * scale));
    let maxTextWidth = 0;
    for (const line of lines) maxTextWidth = Math.max(maxTextWidth, ctx.measureText(line).width);
    const panelWidth = Math.min(maxTextWidth + padding * 2 + 6, cssWidth - 24);
    const panelHeight = lines.length * lineHeight + padding * 2;

    const x = 12;
    // The mode buttons own the top edge, so the stats panel is anchored to
    // the bottom-left corner (clamped to stay inside the visible panel).
    const y = Math.max(12, cssHeight - panelHeight - 12);
    ctx.fillStyle = 'rgba(3, 7, 18, 0.78)';
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.lineWidth = 1;
    this.panelPath(x, y, panelWidth, panelHeight, 8);
    ctx.fill();
    ctx.stroke();

    // State-colored accent bar down the left edge of the panel.
    ctx.fillStyle = STATE_COLORS[frame.state];
    ctx.fillRect(x + 5, y + 8, 3, panelHeight - 16);

    lines.forEach((line, i) =>
      ctx.fillText(line, x + padding + 6, y + padding + fontSize + i * lineHeight)
    );
  }
}
