/**
 * Toolbar: CAD-styled glass header with camera / voice control and scene
 * utilities.
 *
 * Both inputs are single toggles whose labels always name the action that is
 * currently available: "Start camera" ↔ "Stop", "Voice" ↔ "Voice off", driven
 * by `setCameraRunning` / `setVoiceState`. A third button, "Talk", stays
 * hidden unless `setManualListenVisible(true)` — it only appears when
 * hands-free mouth detection is unavailable and the user has to say so
 * themselves. UI-only module — no vision, voice or CAD imports.
 */

import type { VoiceHudState } from './VoiceHud';

export type ToolId = 'box' | 'cuboid' | 'cylinder' | 'sphere';

export interface ToolbarCallbacks {
  onClearScene?: () => void;
  onExportStl?: () => void;
  /** Request webcam + hand tracking to start. */
  onCameraStart?: () => void;
  /** Stop tracking and release the webcam. */
  onCameraStop?: () => void;
  /** Turn the voice agent on or off (state comes back via `setVoiceState`). */
  onVoiceToggle?: () => void;
  /** Manual push-to-talk fallback when hands-free detection is unavailable. */
  onVoiceListen?: () => void;
}

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
  mic:
    '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/>',
  micOff:
    '<path d="m2 2 20 20"/><path d="M18.89 13.23A7.12 7.12 0 0 0 19 12v-2"/><path d="M5 10v2a7 7 0 0 0 12 5"/><path d="M15 9.34V5a3 3 0 0 0-5.68-1.33"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12"/><path d="M12 19v3"/>',
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
  private voiceButton: HTMLButtonElement | null = null;
  private listenButton: HTMLButtonElement | null = null;
  private cameraRunning = false;
  private voiceState: VoiceHudState = 'off';
  private manualListenWanted = false;

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

  /**
   * Reflect voice state. The label flips between "Voice" and "Voice off";
   * `data-voice` carries the live state so CSS can tint the button while the
   * agent is listening, thinking or speaking.
   */
  setVoiceState(state: VoiceHudState): void {
    this.voiceState = state;
    const button = this.voiceButton;
    if (!button) return;
    const on = state !== 'off';
    button.dataset.voice = state;
    button.innerHTML = `${icon(on ? 'micOff' : 'mic')}<span>${on ? 'Voice off' : 'Voice'}</span>`;
    button.title = on ? 'Turn the voice agent off' : 'Talk to the agent while you build';
    button.setAttribute('aria-label', button.title);
    button.setAttribute('aria-pressed', String(on));
    button.classList.toggle('danger', on);
    button.classList.toggle('accent', !on);
    this.applyManualListenVisibility();
  }

  /**
   * Whether the manual "Talk" button is wanted — true once hands-free mouth
   * detection is known to be unavailable. The intent is remembered, because
   * the button is also hidden whenever voice is off and has to come back by
   * itself when voice is switched on again.
   */
  setManualListenVisible(visible: boolean): void {
    this.manualListenWanted = visible;
    this.applyManualListenVisibility();
  }

  /**
   * Mark the manual button as actively listening. It is a toggle, not a
   * push-to-talk, so the label has to say what the next press will do —
   * otherwise it is far too easy to speak into a mic that never seals the turn.
   */
  setManualListening(listening: boolean): void {
    const button = this.listenButton;
    if (!button) return;
    button.setAttribute('aria-pressed', String(listening));
    button.classList.toggle('active', listening);
    button.innerHTML = `${icon('mic')}<span>${listening ? 'Send' : 'Talk'}</span>`;
    button.title = listening
      ? 'Stop listening and send what you just said'
      : 'Open the microphone (hands-free detection is unavailable)';
    button.setAttribute('aria-label', button.title);
  }

  private applyManualListenVisibility(): void {
    const button = this.listenButton;
    if (!button) return;
    button.hidden = !this.manualListenWanted || this.voiceState === 'off';
  }

  private mount(): void {
    this.root.classList.add('toolbar');
    this.root.setAttribute('role', 'toolbar');
    this.root.setAttribute('aria-label', 'CAD controls');

    const brand = document.createElement('span');
    brand.className = 'toolbar-brand';
    brand.innerHTML = `${icon('logo')}<span>CAD&nbsp;Vision</span>`;
    this.root.appendChild(brand);

    const inputGroup = document.createElement('div');
    inputGroup.className = 'toolbar-group';
    inputGroup.setAttribute('role', 'group');
    inputGroup.setAttribute('aria-label', 'Camera and voice');
    inputGroup.appendChild(this.createCameraToggle());
    inputGroup.appendChild(this.createVoiceToggle());
    inputGroup.appendChild(this.createListenButton());
    this.root.appendChild(inputGroup);

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
      this.createButton(
        'export',
        'Export for printing',
        'Download model.stl — drop it into a slicer or send it to a printer',
        this.callbacks.onExportStl,
        'accent'
      )
    );
    this.root.appendChild(utilityGroup);

    this.setCameraRunning(false);
    this.setVoiceState('off');
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

  /** Voice toggle: the agent's on/off switch. */
  private createVoiceToggle(): HTMLButtonElement {
    const button = this.createButton(
      'mic',
      'Voice',
      'Talk to the agent while you build',
      () => this.callbacks.onVoiceToggle?.(),
      'accent'
    );
    this.voiceButton = button;
    return button;
  }

  /** Manual push-to-talk; hidden until hands-free detection gives up. */
  private createListenButton(): HTMLButtonElement {
    const button = this.createButton(
      'mic',
      'Talk',
      'Open the microphone (hands-free detection is unavailable)',
      () => this.callbacks.onVoiceListen?.(),
      'accent'
    );
    button.classList.add('listen-button');
    button.setAttribute('aria-pressed', 'false');
    button.hidden = true;
    this.listenButton = button;
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
    this.voiceButton = null;
    this.listenButton = null;
    this.root.classList.remove('toolbar');
    this.root.removeAttribute('role');
    this.root.removeAttribute('aria-label');
    this.root.replaceChildren();
  }
}
