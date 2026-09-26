/**
 * SelectionMenu: floating action HUD for the active SELECT-mode selection.
 *
 * A small glass pill anchored below the selected mesh's screen projection
 * (the same anchor the color wheel uses): currently one destructive action,
 * **Delete**. Pure UI — no vision or CAD imports. The host (src/main.ts):
 *   - re-anchors it every frame via `show(x, y)` / hides it via `hide()`
 *     (root-local CSS px, the color wheel's coordinate space);
 *   - triggers deletion through `onDeleteRequest` (mouse click), a pinch
 *     landing on the button (`hitDelete` hit-test), or a *pointing* index
 *     fingertip dwelling on the button (`advanceDeleteDwell` — the same
 *     timed-hover model the vision overlay's buttons use);
 *   - mirrors its live placement into the camera-thumbnail HUD via
 *     `deleteRect` / `deleteDwellProgress`.
 *
 * Deletion never happens here: the host routes every trigger through the
 * confirmation dialog before touching the scene.
 */

import { DwellTracker } from './ColorWheel';
import { hitTestElement } from './hitTest';

export interface SelectionMenuOptions {
  /** Request deletion of the active selection (mouse click on Delete). */
  onDeleteRequest?: () => void;
  /** Vertical distance below the anchor point (CSS px). */
  offsetPx?: number;
  /**
   * Continuous pointing-finger hover on the Delete button required to
   * trigger it (ms). Default 800 — a touch slower than the overlay's mode
   * buttons, since this one opens a destructive confirmation.
   */
  dwellMs?: number;
}

/** Lucide-style trash icon (currentColor stroke), matching the Toolbar's. */
const TRASH_ICON =
  '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M10 11v6"/><path d="M14 11v6"/>';

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export class SelectionMenu {
  private readonly root: HTMLElement;
  private readonly element: HTMLDivElement;
  private readonly deleteButton: HTMLButtonElement;
  private readonly onDeleteRequest: (() => void) | undefined;
  private readonly offsetPx: number;
  /** Pointing-finger dwell clock on the Delete button (timed hover trigger). */
  private readonly dwell: DwellTracker;
  private shown = false;

  constructor(root: HTMLElement, options: SelectionMenuOptions = {}) {
    this.root = root;
    this.onDeleteRequest = options.onDeleteRequest;
    this.offsetPx = options.offsetPx ?? 100;
    this.dwell = new DwellTracker({ durationMs: options.dwellMs ?? 800 });

    this.element = document.createElement('div');
    this.element.className = 'selection-menu';
    this.element.setAttribute('role', 'toolbar');
    this.element.setAttribute('aria-label', 'Selection actions');

    this.deleteButton = document.createElement('button');
    this.deleteButton.type = 'button';
    this.deleteButton.className = 'selection-menu-delete';
    this.deleteButton.title = 'Delete the selected object';
    this.deleteButton.setAttribute('aria-label', 'Delete the selected object');
    this.deleteButton.innerHTML =
      `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" ` +
      `fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ` +
      `stroke-linejoin="round" aria-hidden="true">${TRASH_ICON}</svg><span>Delete</span>`;
    this.deleteButton.addEventListener('click', () => this.onDeleteRequest?.());
    this.element.appendChild(this.deleteButton);
    root.appendChild(this.element);
  }

  /** Whether the menu is currently shown. */
  get visible(): boolean {
    return this.shown;
  }

  /**
   * Anchor the menu below a point in root-local CSS px (the selected mesh's
   * screen projection): horizontally centered on it, offset down by
   * `offsetPx`, clamped so the pill always stays fully inside the root.
   */
  show(screenX: number, screenY: number): void {
    this.shown = true;
    const rect = this.element.getBoundingClientRect();
    const width = Math.max(rect.width, 1);
    const height = Math.max(rect.height, 1);
    const rootWidth = Math.max(this.root.clientWidth, width);
    const rootHeight = Math.max(this.root.clientHeight, height);
    const x = clamp(screenX, width / 2, rootWidth - width / 2);
    const y = clamp(screenY + this.offsetPx, height / 2, rootHeight - height / 2);
    this.element.style.left = `${x}px`;
    this.element.style.top = `${y}px`;
    this.element.classList.add('selection-menu--visible');
  }

  /** Hide the menu (fades out via CSS). */
  hide(): void {
    this.shown = false;
    this.element.classList.remove('selection-menu--visible');
    this.dwell.reset();
  }

  /**
   * Gesture hit-test: whether a root-local point (CSS px) lands on the
   * Delete button — used by the host to route a pinch on the HUD into a
   * (confirmation-gated) delete request instead of a scene pick.
   */
  hitDelete(screenX: number, screenY: number): boolean {
    return this.shown && hitTestElement(this.deleteButton, this.root, screenX, screenY);
  }

  /**
   * Pointing-finger dwell on the Delete button: advance the timed-hover
   * clock by `dtMs` while the given root-local point (CSS px — typically a
   * pointing hand's index fingertip) stays on the button. Returns true
   * exactly once when the dwell completes (the host then triggers the
   * confirmation-gated delete); a null point or a miss resets the clock.
   */
  advanceDeleteDwell(
    point: { x: number; y: number } | null,
    dtMs: number
  ): boolean {
    const hovering = point !== null && this.hitDelete(point.x, point.y);
    return this.dwell.advance(hovering ? 'delete' : null, dtMs);
  }

  /** Delete-button dwell completion [0, 1] (for the HUD progress fill). */
  get deleteDwellProgress(): number {
    return this.dwell.progress;
  }

  /**
   * Live Delete-button bounds in root-local CSS px (the space `hitDelete`
   * uses), or null while hidden — lets the host mirror the button's
   * placement into the camera-thumbnail HUD.
   */
  get deleteRect(): { x: number; y: number; width: number; height: number } | null {
    if (!this.shown) return null;
    const buttonRect = this.deleteButton.getBoundingClientRect();
    const rootRect = this.root.getBoundingClientRect();
    return {
      x: buttonRect.left - rootRect.left,
      y: buttonRect.top - rootRect.top,
      width: buttonRect.width,
      height: buttonRect.height,
    };
  }

  /** Unmount the menu from its host element. */
  dispose(): void {
    this.element.remove();
  }
}
