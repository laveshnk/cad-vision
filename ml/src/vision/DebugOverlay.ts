/**
 * DebugOverlay: 2D-canvas debug renderer for the gesture engine.
 *
 * Draws the mirrored webcam landmarks (21 per hand) with the MediaPipe skeleton,
 * pinch indicators, and a HUD with live metrics. Pure Canvas 2D — no WebGL.
 *
 * Alignment: the <video> is CSS-mirrored (`scaleX(-1)`) and displayed with
 * `object-fit: cover`, which center-crops it into its container (the floating
 * camera thumbnail, or that thumbnail expanded). The canvas backing store
 * therefore matches the *container* (CSS px × devicePixelRatio, re-measured
 * every frame), and landmarks are mapped through the same mirrored cover
 * transform so they land exactly on the webcam image — while the HUD stays
 * anchored to the visible container edges (never cropped). Typography scales
 * with the container width (see `fontScale`).
 */

import { HandLandmarker } from '@mediapipe/tasks-vision';
import type { FrameEvent, GestureState, HandSnapshot } from './types';

const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;

/** A skeleton connection between two landmark indices. */
export interface SkeletonConnection {
  start: number;
  end: number;
}

/** State color-coding: green = pinch/draw, blue = orbit, purple = zoom, yellow = idle. */
export const STATE_COLORS: Record<GestureState, string> = {
  IDLE: '#facc15', // yellow
  DRAWING_BASE: '#22c55e', // green
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

export class DebugOverlay {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly connections: ReadonlyArray<SkeletonConnection>;

  constructor(canvas: HTMLCanvasElement, connections?: ReadonlyArray<SkeletonConnection>) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('DebugOverlay: 2D canvas context unavailable');
    this.ctx = ctx;
    this.connections = connections ?? HandLandmarker.HAND_CONNECTIONS;
  }

  /** Render one frame of landmarks + HUD. */
  render(frame: FrameEvent): void {
    const { cssWidth, cssHeight } = this.ensureSize(frame.video.width, frame.video.height);
    this.ctx.clearRect(0, 0, cssWidth, cssHeight);
    const view = this.coverTransform(cssWidth, cssHeight, frame.video.width, frame.video.height);
    for (const hand of frame.hands) {
      this.drawHand(hand, frame.state, view, cssWidth);
    }
    this.drawDualHandsLink(frame, view, cssWidth);
    this.drawZoomAnchor(frame, view);
    this.drawHud(frame, cssWidth);
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

  /** Rounded HUD panel (top-left, always inside the visible panel) with state, FPS, hands and live metrics. */
  private drawHud(frame: FrameEvent, cssWidth: number): void {
    const ctx = this.ctx;
    const lines: string[] = [];
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
    const y = 12;
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
