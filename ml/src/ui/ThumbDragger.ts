/**
 * ThumbDragger: repositions the floating camera thumbnail (the "camera
 * window") by dragging its **outer frame** — the header bar and the metrics
 * bar below the video. The video stage / overlay canvas is deliberately NOT
 * a handle, so gestures and clicks inside the camera view (mode buttons,
 * trash bin, hand tracking) never drag the window around.
 *
 * Pointer capture keeps the drag alive when the cursor leaves the card; the
 * position is clamped so the card always stays (mostly) inside its
 * workspace. Pure UI — no vision or CAD imports.
 */

export interface ThumbDraggerOptions {
  /** Workspace margin kept between the card edge and the window (px). */
  margin?: number;
  /** Notification on every move (the card's new left / top, px). */
  onMove?: (left: number, top: number) => void;
}

export class ThumbDragger {
  private readonly thumb: HTMLElement;
  private readonly margin: number;
  private readonly onMove: ((left: number, top: number) => void) | undefined;
  /** Pointer position + card offset at drag start (null = not dragging). */
  private dragStart: { pointerX: number; pointerY: number; left: number; top: number } | null =
    null;
  /** The frame elements that act as the drag handle. */
  private readonly handles: HTMLElement[] = [];

  constructor(thumb: HTMLElement, options: ThumbDraggerOptions = {}) {
    this.thumb = thumb;
    this.margin = options.margin ?? 16;
    this.onMove = options.onMove;

    // The handle is strictly the card's outer frame: header + metrics bar.
    for (const selector of ['.vision-thumb-header', '.metrics-bar']) {
      const handle = thumb.querySelector(selector);
      if (handle instanceof HTMLElement) {
        handle.addEventListener('pointerdown', this.onPointerDown);
        this.handles.push(handle);
      }
    }
  }

  /** Unmount the handle listeners. */
  dispose(): void {
    this.endDrag();
    for (const handle of this.handles) handle.removeEventListener('pointerdown', this.onPointerDown);
    this.handles.length = 0;
  }

  /**
   * Begin a reposition drag: primary button only, and never on (or inside)
   * an interactive child of the handle (the expand button). Capture rides
   * on the *card* so movement stays smooth even over the video stage.
   */
  private readonly onPointerDown = (event: PointerEvent): void => {
    if (this.dragStart !== null) return; // already dragging
    if (event.button !== 0) return; // primary button only
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest('button')) return; // the expand button stays a button
    event.preventDefault();
    const rect = this.thumb.getBoundingClientRect();
    this.dragStart = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      left: rect.left,
      top: rect.top,
    };
    this.thumb.classList.add('vision-thumb--dragging');
    this.thumb.setPointerCapture(event.pointerId);
    this.thumb.addEventListener('pointermove', this.onPointerMove);
    this.thumb.addEventListener('pointerup', this.onPointerUp);
    this.thumb.addEventListener('pointercancel', this.onPointerUp);
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    const start = this.dragStart;
    if (!start) return;
    const { left, top } = this.clamp(
      start.left + event.clientX - start.pointerX,
      start.top + event.clientY - start.pointerY
    );
    this.thumb.style.left = `${left}px`;
    this.thumb.style.top = `${top}px`;
    this.onMove?.(left, top);
  };

  /** End the drag: release the pointer + restore the cursor. */
  private readonly onPointerUp = (): void => {
    this.endDrag();
  };

  private endDrag(): void {
    this.dragStart = null;
    this.thumb.classList.remove('vision-thumb--dragging');
    this.thumb.removeEventListener('pointermove', this.onPointerMove);
    this.thumb.removeEventListener('pointerup', this.onPointerUp);
    this.thumb.removeEventListener('pointercancel', this.onPointerUp);
  }

  /** Keep the card inside the workspace (its parent), with a margin. */
  private clamp(left: number, top: number): { left: number; top: number } {
    const parent = this.thumb.parentElement;
    const width = this.thumb.getBoundingClientRect().width;
    const height = this.thumb.getBoundingClientRect().height;
    const maxX = (parent?.clientWidth ?? width) - width + this.margin;
    const maxY = (parent?.clientHeight ?? height) - height + this.margin;
    return {
      left: Math.min(Math.max(left, -this.margin), Math.max(maxX, -this.margin)),
      top: Math.min(Math.max(top, -this.margin), Math.max(maxY, -this.margin)),
    };
  }
}
