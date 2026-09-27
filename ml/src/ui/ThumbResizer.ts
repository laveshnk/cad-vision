/**
 * ThumbResizer: mouse-drag resizing for the floating camera thumbnail.
 *
 * Appends a corner grip (`.thumb-resize`, bottom-right of the video stage)
 * and turns pointer drags on it into width changes on the thumbnail: the
 * drag delta is clamped between `minWidth` and `maxWidth` (itself clamped to
 * the window so the card never runs off-screen), and applied as an inline
 * `style.width` — the 4:3 stage keeps the height following automatically, and
 * the vision overlay re-measures its canvas every frame, so no resize
 * listeners are needed downstream.
 *
 * An inline width beats the CSS presets (`.vision-thumb` / `.expanded`), so
 * the host should clear it (`` thumb.style.width = '' ``) when the user
 * toggles the preset sizes again. Pure UI — no vision or CAD imports.
 *
 * The width resolution math is exported as a pure function
 * (`resolveThumbWidth`) so it stays unit-testable without a DOM (see
 * `__tests__/thumbResizer.test.ts`).
 */

export interface ThumbResizerOptions {
  /** Smallest allowed thumbnail width (CSS px). Default 180. */
  minWidth?: number;
  /** Largest allowed thumbnail width (CSS px, before the window clamp). Default 720. */
  maxWidth?: number;
  /** Margin kept between the thumbnail's right edge and the window (CSS px). Default 16. */
  windowMargin?: number;
  /** Called after each applied width change (receives the new width in px). */
  onResize?: (width: number) => void;
}

/**
 * Resolve one drag step: the pointer moved `pointerDeltaX` px right of the
 * drag start; the new width is `startWidth + pointerDeltaX` clamped to
 * [minWidth, maxWidth]. Pure — identical pointer travel always yields the
 * identical width, so it is trivially unit-testable.
 */
export function resolveThumbWidth(
  startWidth: number,
  pointerDeltaX: number,
  minWidth: number,
  maxWidth: number
): number {
  return Math.min(maxWidth, Math.max(minWidth, startWidth + pointerDeltaX));
}

export class ThumbResizer {
  private readonly thumb: HTMLElement;
  private readonly handle: HTMLDivElement;
  private readonly minWidth: number;
  private readonly maxWidth: number;
  private readonly windowMargin: number;
  private readonly onResize: ((width: number) => void) | undefined;
  /** Width (px) when the current drag started; null while not dragging. */
  private dragStartWidth: number | null = null;
  /** Pointer X (px) when the current drag started; null while not dragging. */
  private dragStartX: number | null = null;

  constructor(thumb: HTMLElement, options: ThumbResizerOptions = {}) {
    this.thumb = thumb;
    this.minWidth = options.minWidth ?? 180;
    this.maxWidth = options.maxWidth ?? 720;
    this.windowMargin = options.windowMargin ?? 16;
    this.onResize = options.onResize;

    this.handle = document.createElement('div');
    this.handle.className = 'thumb-resize';
    this.handle.title = 'Drag to resize the camera thumbnail';
    this.handle.setAttribute('role', 'separator');
    this.handle.setAttribute('aria-label', 'Drag to resize the camera thumbnail');
    this.handle.addEventListener('pointerdown', this.onPointerDown);
    // The grip sits inside the video stage (bottom-right corner), above the
    // overlay canvas; fallback: the thumbnail itself hosts it.
    const stage = thumb.querySelector('.stage');
    (stage instanceof HTMLElement ? stage : thumb).appendChild(this.handle);
  }

  /** Unmount the grip and drop its listeners. */
  dispose(): void {
    this.endDrag();
    this.handle.removeEventListener('pointerdown', this.onPointerDown);
    this.handle.remove();
  }

  /**
   * Begin a resize drag: capture the pointer so the grip keeps receiving
   * moves even when the cursor leaves the (fast-shrinking) thumbnail, and
   * freeze the CSS width transition so the card tracks the cursor 1:1.
   */
  private readonly onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0) return; // primary button only
    event.preventDefault();
    this.dragStartWidth = this.thumb.getBoundingClientRect().width;
    this.dragStartX = event.clientX;
    this.thumb.classList.add('vision-thumb--resizing');
    this.handle.setPointerCapture(event.pointerId);
    this.handle.addEventListener('pointermove', this.onPointerMove);
    this.handle.addEventListener('pointerup', this.onPointerUp);
    this.handle.addEventListener('pointercancel', this.onPointerUp);
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (this.dragStartWidth === null || this.dragStartX === null) return;
    const available =
      window.innerWidth - this.thumb.getBoundingClientRect().left - this.windowMargin;
    const maxWidth = Math.max(this.minWidth, Math.min(this.maxWidth, available));
    const width = resolveThumbWidth(
      this.dragStartWidth,
      event.clientX - this.dragStartX,
      this.minWidth,
      maxWidth
    );
    this.thumb.style.width = `${width}px`;
    this.onResize?.(width);
  };

  /** End the drag: release the pointer and restore the width transition. */
  private readonly onPointerUp = (): void => {
    this.endDrag();
  };

  private endDrag(): void {
    this.dragStartWidth = null;
    this.dragStartX = null;
    this.thumb.classList.remove('vision-thumb--resizing');
    this.handle.removeEventListener('pointermove', this.onPointerMove);
    this.handle.removeEventListener('pointerup', this.onPointerUp);
    this.handle.removeEventListener('pointercancel', this.onPointerUp);
  }
}
