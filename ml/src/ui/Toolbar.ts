/**
 * Toolbar: CAD-styled glass header with camera controls, primitive tool
 * selection and scene utilities. Tools can be selected with the mouse or
 * programmatically via `setActiveTool` (e.g. from gesture shortcuts);
 * selection state is exposed through `onToolSelect`, scene commands through
 * `onClearScene` / `onExportStl`, camera lifecycle through `onCameraStart` /
 * `onCameraStop` (+ `setCameraRunning` to reflect state). UI-only module — no
 * vision or CAD imports.
 */

export type ToolId = 'box' | 'cylinder' | 'sphere';

export interface ToolbarCallbacks {
  onToolSelect?: (tool: ToolId) => void;
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
  { id: 'box', label: 'Box', title: 'Box — pinch-drag a rectangle footprint' },
  { id: 'cylinder', label: 'Cylinder', title: 'Cylinder — pinch-drag a base circle' },
  { id: 'sphere', label: 'Sphere', title: 'Sphere — pinch-drag a base circle' },
];

/** Lucide-style stroke icons (currentColor). */
const ICONS = {
  logo: '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
  box: '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
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
  private readonly toolButtons = new Map<ToolId, HTMLButtonElement>();
  private readonly listeners: Array<() => void> = [];
  private readonly cameraButtons: Array<HTMLButtonElement> = [];
  private tool: ToolId = 'box';

  constructor(
    private readonly root: HTMLElement,
    private readonly callbacks: ToolbarCallbacks = {}
  ) {
    this.mount();
  }

  get activeTool(): ToolId {
    return this.tool;
  }

  /**
   * Reflect camera / tracking state: while running, only "Stop" is enabled;
   * when stopped, only "Start camera" is enabled.
   */
  setCameraRunning(running: boolean): void {
    for (const button of this.cameraButtons) {
      button.disabled = button.dataset.camera === 'stop' ? !running : running;
    }
  }

  /** Programmatically select a tool (mirrors button state). */
  setActiveTool(tool: ToolId): void {
    if (this.tool === tool) return;
    this.select(tool);
  }

  private mount(): void {
    this.root.classList.add('toolbar');
    this.root.setAttribute('role', 'toolbar');
    this.root.setAttribute('aria-label', 'CAD tools');

    const brand = document.createElement('span');
    brand.className = 'toolbar-brand';
    brand.innerHTML = `${icon('logo')}<span>CAD&nbsp;Vision</span>`;
    this.root.appendChild(brand);

    const cameraGroup = document.createElement('div');
    cameraGroup.className = 'toolbar-group';
    cameraGroup.setAttribute('role', 'group');
    cameraGroup.setAttribute('aria-label', 'Camera');
    cameraGroup.appendChild(
      this.createCameraButton('camera', 'Start camera', this.callbacks.onCameraStart, 'accent')
    );
    cameraGroup.appendChild(
      this.createCameraButton('stop', 'Stop', this.callbacks.onCameraStop, 'danger')
    );
    this.root.appendChild(cameraGroup);

    this.root.appendChild(this.createSeparator());

    const toolGroup = document.createElement('div');
    toolGroup.className = 'toolbar-group';
    toolGroup.setAttribute('role', 'group');
    toolGroup.setAttribute('aria-label', 'Primitive tools');
    for (const spec of TOOLS) {
      toolGroup.appendChild(this.createToolButton(spec));
    }
    this.root.appendChild(toolGroup);

    this.root.appendChild(this.createSeparator());

    const utilityGroup = document.createElement('div');
    utilityGroup.className = 'toolbar-group';
    utilityGroup.setAttribute('role', 'group');
    utilityGroup.setAttribute('aria-label', 'Scene utilities');
    utilityGroup.appendChild(
      this.createUtilityButton('clear', 'Clear scene', this.callbacks.onClearScene, 'danger')
    );
    utilityGroup.appendChild(
      this.createUtilityButton('export', 'Export STL', this.callbacks.onExportStl, 'accent')
    );
    this.root.appendChild(utilityGroup);

    this.select('box', false);
    this.setCameraRunning(false);
  }

  private createToolButton(spec: ToolSpec): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'toolbar-button tool-button';
    button.dataset.tool = spec.id;
    button.title = spec.title;
    button.innerHTML = `${icon(spec.id)}<span>${spec.label}</span>`;
    button.setAttribute('aria-pressed', 'false');
    const onClick = () => this.select(spec.id);
    button.addEventListener('click', onClick);
    this.listeners.push(() => button.removeEventListener('click', onClick));
    this.toolButtons.set(spec.id, button);
    return button;
  }

  private createUtilityButton(
    iconId: 'clear' | 'export',
    label: string,
    handler: (() => void) | undefined,
    variant: 'danger' | 'accent'
  ): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `toolbar-button utility-button ${variant}`;
    button.title = label;
    button.innerHTML = `${icon(iconId)}<span>${label}</span>`;
    if (handler) {
      const onClick = () => handler();
      button.addEventListener('click', onClick);
      this.listeners.push(() => button.removeEventListener('click', onClick));
    }
    return button;
  }

  private createCameraButton(
    iconId: 'camera' | 'stop',
    label: string,
    handler: (() => void) | undefined,
    variant: 'danger' | 'accent'
  ): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `toolbar-button utility-button ${variant}`;
    button.dataset.camera = iconId;
    button.title = label;
    button.innerHTML = `${icon(iconId)}<span>${label}</span>`;
    if (handler) {
      const onClick = () => handler();
      button.addEventListener('click', onClick);
      this.listeners.push(() => button.removeEventListener('click', onClick));
    }
    this.cameraButtons.push(button);
    return button;
  }

  private createSeparator(): HTMLElement {
    const separator = document.createElement('span');
    separator.className = 'toolbar-separator';
    return separator;
  }

  /** Apply selection state and notify `onToolSelect` (unless silent). */
  private select(tool: ToolId, notify = true): void {
    this.tool = tool;
    for (const [id, button] of this.toolButtons) {
      const active = id === tool;
      button.setAttribute('aria-pressed', String(active));
      button.classList.toggle('active', active);
    }
    if (notify) this.callbacks.onToolSelect?.(tool);
  }

  /** Unmount all children and detach listeners. */
  dispose(): void {
    for (const detach of this.listeners) detach();
    this.listeners.length = 0;
    this.toolButtons.clear();
    this.cameraButtons.length = 0;
    this.root.classList.remove('toolbar');
    this.root.removeAttribute('role');
    this.root.removeAttribute('aria-label');
    this.root.replaceChildren();
  }
}
