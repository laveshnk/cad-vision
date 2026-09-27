/**
 * ConfirmDialog: modal safety confirmation for destructive actions
 * (SELECT-mode object deletion).
 *
 * Centers over its host element (`#viewport`) with a dimmed backdrop: one
 * question plus [Confirm] [Cancel] actions. Pure UI — no vision or CAD
 * imports. The host opens / closes it (`open()` / `close()` / `isOpen`) and
 * receives the answer through `onConfirm` / `onCancel` (mouse click on the
 * buttons or the backdrop). Gesture and keyboard routing (pinch hit-tests
 * via `hitConfirm` / `hitCancel`, Enter / Escape / Delete keys)
 * intentionally live in the host, so this component stays a dumb view.
 *
 * Callbacks fire *before* the dialog closes itself, so a host handler can
 * still consult `isOpen` if it needs to.
 */

import { hitTestElement } from './hitTest';

export interface ConfirmDialogOptions {
  /** Question shown above the actions. */
  message?: string;
  /** Answer "yes" — the destructive action may proceed. */
  onConfirm?: () => void;
  /** Answer "no" — the action is abandoned, nothing changes. */
  onCancel?: () => void;
}

const DEFAULT_MESSAGE = 'Are you sure you want to delete this object?';

export class ConfirmDialog {
  private readonly root: HTMLElement;
  private readonly element: HTMLDivElement;
  private readonly confirmButton: HTMLButtonElement;
  private readonly cancelButton: HTMLButtonElement;
  private readonly onConfirm: (() => void) | undefined;
  private readonly onCancel: (() => void) | undefined;
  private opened = false;

  constructor(root: HTMLElement, options: ConfirmDialogOptions = {}) {
    this.root = root;
    this.onConfirm = options.onConfirm;
    this.onCancel = options.onCancel;

    this.element = document.createElement('div');
    this.element.className = 'confirm-dialog';
    this.element.setAttribute('role', 'dialog');
    this.element.setAttribute('aria-modal', 'true');
    this.element.setAttribute('aria-label', options.message ?? DEFAULT_MESSAGE);

    const backdrop = document.createElement('div');
    backdrop.className = 'confirm-dialog-backdrop';
    // Clicking outside the card dismisses — the safe answer for a
    // destructive confirmation.
    backdrop.addEventListener('click', () => this.answer(this.onCancel));

    const card = document.createElement('div');
    card.className = 'confirm-dialog-card';

    const message = document.createElement('p');
    message.className = 'confirm-dialog-message';
    message.textContent = options.message ?? DEFAULT_MESSAGE;

    const actions = document.createElement('div');
    actions.className = 'confirm-dialog-actions';
    this.cancelButton = this.createAction('Cancel', 'confirm-dialog-cancel', this.onCancel);
    this.confirmButton = this.createAction('Confirm', 'confirm-dialog-confirm', this.onConfirm);
    actions.append(this.cancelButton, this.confirmButton);

    card.append(message, actions);
    this.element.append(backdrop, card);
    root.appendChild(this.element);
  }

  /** Whether the dialog is currently open (modal). */
  get isOpen(): boolean {
    return this.opened;
  }

  /** Show the dialog (focuses Confirm for immediate keyboard access). */
  open(): void {
    if (this.opened) return;
    this.opened = true;
    this.element.classList.add('confirm-dialog--open');
    this.confirmButton.focus({ preventScroll: true });
  }

  /** Hide the dialog without answering (used after a host-side answer). */
  close(): void {
    if (!this.opened) return;
    this.opened = false;
    this.element.classList.remove('confirm-dialog--open');
    this.confirmButton.blur();
  }

  /** Gesture hit-test: does a root-local point land on [Confirm]? */
  hitConfirm(screenX: number, screenY: number): boolean {
    return this.opened && hitTestElement(this.confirmButton, this.root, screenX, screenY);
  }

  /** Gesture hit-test: does a root-local point land on [Cancel]? */
  hitCancel(screenX: number, screenY: number): boolean {
    return this.opened && hitTestElement(this.cancelButton, this.root, screenX, screenY);
  }

  /** Unmount the dialog from its host element. */
  dispose(): void {
    this.element.remove();
  }

  /** Wire one action button: answers the dialog, then closes it. */
  private createAction(
    label: string,
    className: string,
    handler: (() => void) | undefined
  ): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = className;
    button.textContent = label;
    button.addEventListener('click', () => this.answer(handler));
    return button;
  }

  /** Run the answer callback (while still open), then close. */
  private answer(handler: (() => void) | undefined): void {
    if (!this.opened) return;
    handler?.();
    this.close();
  }
}
