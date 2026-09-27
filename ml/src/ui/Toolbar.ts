/**
 * Toolbar: CAD-styled glass header with camera control and scene utilities.
 * The camera control is a single toggle ("Start camera" ↔ "Stop") whose state
 * is driven by `setCameraRunning`; STL export is exposed through
 * `onExportStl`, camera lifecycle through `onCameraStart` / `onCameraStop`,
 * and the dimension-label toggle through `onDimensionsToggle` (its pressed
 * state mirrors `setDimensionsVisible`).
 * Scene clearing lives in-vision (the overlay's trash bin + spatial
 * confirmation), so no mouse-only destructive button is mounted here.
 * UI-only module — no vision or CAD imports.
 */

export interface ToolbarCallbacks {
  onExportStl?: () => void;
  /** Request webcam + hand tracking to start. */
  onCameraStart?: () => void;
  /** Stop tracking and release the webcam. */
  onCameraStop?: () => void;
  /** Flip the viewport's dimension labels (cm) on / off. */
  onDimensionsToggle?: () => void;
}

/** Lucide-style stroke icons (currentColor). */
const ICONS = {
  logo: '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
  export:
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
  camera:
    '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3Z"/><circle cx="12" cy="13" r="3"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  ruler:
    '<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/>',
} as const;

function icon(id: keyof typeof ICONS): string {
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" ` +
    `fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ` +
    `stroke-linejoin="round" aria-hidden="true">${ICONS[id]}</svg>`
  );
}

export class Toolbar {
  private readonly listeners: Array<() => void> = [];
  private cameraButton: HTMLButtonElement | null = null;
  private dimensionsButton: HTMLButtonElement | null = null;
  private cameraRunning = false;

  constructor(
    private readonly root: HTMLElement,
    private readonly callbacks: ToolbarCallbacks = {}
  ) {
    this.mount();
  }

  /**
   * Reflect camera / tracking state: the single camera button flips between
   * "Start camera" (idle) and "Stop" (running), so the visible action is
   * always the one currently available.
   */
  setCameraRunning(running: boolean): void {
    this.cameraRunning = running;
    const button = this.cameraButton;
    if (!button) return;
    if (running) {
      button.innerHTML = `${icon('stop')}<span>Stop</span>`;
      button.title = 'Stop tracking and release the camera';
      button.setAttribute('aria-label', button.title);
      button.classList.remove('accent');
      button.classList.add('danger');
    } else {
      button.innerHTML = `${icon('camera')}<span>Start camera</span>`;
      button.title = 'Start webcam + hand tracking';
      button.setAttribute('aria-label', button.title);
      button.classList.remove('danger');
      button.classList.add('accent');
    }
  }

  /** Reflect whether the dimension labels are shown (pressed toggle). */
  setDimensionsVisible(visible: boolean): void {
    this.dimensionsButton?.setAttribute('aria-pressed', String(visible));
  }

  private mount(): void {
    this.root.classList.add('toolbar');
    this.root.setAttribute('role', 'toolbar');
    this.root.setAttribute('aria-label', 'CAD controls');

    const brand = document.createElement('span');
    brand.className = 'toolbar-brand';
    brand.innerHTML = `${icon('logo')}<span>CAD&nbsp;Vision</span>`;
    this.root.appendChild(brand);

    const cameraGroup = document.createElement('div');
    cameraGroup.className = 'toolbar-group';
    cameraGroup.setAttribute('role', 'group');
    cameraGroup.setAttribute('aria-label', 'Camera');
    cameraGroup.appendChild(this.createCameraToggle());
    this.root.appendChild(cameraGroup);

    const spacer = document.createElement('span');
    spacer.className = 'toolbar-spacer';
    this.root.appendChild(spacer);

    const utilityGroup = document.createElement('div');
    utilityGroup.className = 'toolbar-group';
    utilityGroup.setAttribute('role', 'group');
    utilityGroup.setAttribute('aria-label', 'Scene utilities');
    this.dimensionsButton = this.createButton(
      'ruler',
      'Dimensions',
      'Show / hide object dimensions (cm)',
      this.callbacks.onDimensionsToggle,
      'accent'
    );
    this.dimensionsButton.setAttribute('aria-pressed', 'false');
    utilityGroup.appendChild(this.dimensionsButton);
    utilityGroup.appendChild(
      this.createButton('export', 'Export STL', 'Export STL', this.callbacks.onExportStl, 'accent')
    );
    this.root.appendChild(utilityGroup);

    this.setCameraRunning(false);
  }

  /** Camera toggle: dispatches start / stop according to the current state. */
  private createCameraToggle(): HTMLButtonElement {
    const button = this.createButton(
      'camera',
      'Start camera',
      'Start webcam + hand tracking',
      () => {
        if (this.cameraRunning) this.callbacks.onCameraStop?.();
        else this.callbacks.onCameraStart?.();
      },
      'accent'
    );
    this.cameraButton = button;
    return button;
  }

  private createButton(
    iconId: keyof typeof ICONS,
    label: string,
    title: string,
    handler: (() => void) | undefined,
    variant: 'danger' | 'accent'
  ): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `toolbar-button utility-button ${variant}`;
    button.title = title;
    button.setAttribute('aria-label', title);
    button.innerHTML = `${icon(iconId)}<span>${label}</span>`;
    if (handler) {
      const onClick = () => handler();
      button.addEventListener('click', onClick);
      this.listeners.push(() => button.removeEventListener('click', onClick));
    }
    return button;
  }

  /** Unmount all children and detach listeners. */
  dispose(): void {
    for (const detach of this.listeners) detach();
    this.listeners.length = 0;
    this.cameraButton = null;
    this.dimensionsButton = null;
    this.root.classList.remove('toolbar');
    this.root.removeAttribute('role');
    this.root.removeAttribute('aria-label');
    this.root.replaceChildren();
  }
}
