/**
 * Toolbar: CAD-styled glass header with camera control and scene utilities.
 * The camera control is a single toggle ("Start camera" ↔ "Stop") whose state
 * is driven by `setCameraRunning`; scene commands are exposed through
 * `onClearScene` / `onExportStl`, camera lifecycle through `onCameraStart` /
 * `onCameraStop`. UI-only module — no vision or CAD imports.
 */

export type ToolId = 'box' | 'cuboid' | 'cylinder' | 'sphere';

export interface ToolbarCallbacks {
  onClearScene?: () => void;
  onExportStl?: () => void;
  /** Request webcam + hand tracking to start. */
  onCameraStart?: () => void;
  /** Stop tracking and release the webcam. */
  onCameraStop?: () => void;
}

interface ToolSpec {
  id: ToolId;
  label: string;
  title: string;
}

const TOOLS: readonly ToolSpec[] = [
  { id: 'box', label: 'Box', title: 'Box — two-hand pinch sizes a square base' },
  { id: 'cuboid', label: 'Cuboid', title: 'Cuboid — two-hand pinch spans a rectangle base' },
  { id: 'cylinder', label: 'Cylinder', title: 'Cylinder — two-hand pinch sizes a base circle' },
  { id: 'sphere', label: 'Sphere', title: 'Sphere — pinch-drag a base circle' },
];

/** Lucide-style stroke icons (currentColor). */
const ICONS = {
  logo: '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
  box: '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
  cuboid:
    '<path d="M2 9l5-4h15v10l-5 4H2Z"/><path d="M2 9h15v10"/><path d="M17 9l5-4"/>',
  cylinder: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/>',
  sphere:
    '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
  clear:
    '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M10 11v6"/><path d="M14 11v6"/>',
  export:
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
  camera:
    '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3Z"/><circle cx="12" cy="13" r="3"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
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
    utilityGroup.appendChild(
      this.createButton('clear', 'Clear scene', 'Clear scene', this.callbacks.onClearScene, 'danger')
    );
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
    this.root.classList.remove('toolbar');
    this.root.removeAttribute('role');
    this.root.removeAttribute('aria-label');
    this.root.replaceChildren();
  }
}
