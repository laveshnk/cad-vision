/**
 * MetricsBar: sleek boxy monospace stats bar mounted directly underneath
 * the camera view (inside the floating thumbnail card, below its header).
 *
 * Replaces the old yellow-on-black canvas HUD that overlay the camera feed:
 * the metrics now live in the DOM below the video, so they never cover the
 * webcam image or the in-vision UI. Pure UI — no vision or CAD imports; the
 * host feeds it plain values extracted from the per-frame event.
 *
 * `update()` is called every processed frame; it only touches the DOM when
 * a value actually changed, so a 60 Hz feed costs nothing when idle.
 */

/** One snapshot of the live stats (plain data — bridge-friendly). */
export interface MetricsSnapshot {
  /** Active interaction mode (VIEW / SELECT / CREATE). */
  mode: string;
  /** Gesture FSM state (IDLE, SELECTING, …). */
  state: string;
  /** Tracking frames per second. */
  fps: number;
  /** Number of tracked hands. */
  hands: number;
  /** Tracked handedness labels, comma-joined (empty when handless). */
  handedness: string;
}

interface MetricsCell {
  /** The mounted cell element (label + value). */
  cell: HTMLElement;
  value: HTMLElement;
  lastText: string | null;
}

export class MetricsBar {
  private readonly element: HTMLElement;
  private readonly cells: Record<'mode' | 'state' | 'fps' | 'hands', MetricsCell>;

  constructor(root: HTMLElement) {
    // The host (`#metrics`) is the bar itself — the cells mount into it.
    this.element = root;
    this.element.classList.add('metrics-bar');
    this.element.setAttribute('role', 'status');
    this.element.setAttribute('aria-label', 'Tracking metrics');

    this.cells = {
      mode: MetricsBar.createCell('MODE', ''),
      state: MetricsBar.createCell('STATE', ''),
      fps: MetricsBar.createCell('FPS', ''),
      hands: MetricsBar.createCell('HANDS', ''),
    };
    this.element.append(
      this.cells.mode.cell,
      this.cells.state.cell,
      this.cells.fps.cell,
      this.cells.hands.cell
    );
    this.update({
      mode: '—',
      state: '—',
      fps: 0,
      hands: 0,
      handedness: '',
    });
  }

  /** Update the displayed values (writes only what changed). */
  update(snapshot: MetricsSnapshot): void {
    this.write(this.cells.mode, snapshot.mode || '—');
    this.write(this.cells.state, snapshot.state || '—');
    this.write(this.cells.fps, snapshot.fps > 0 ? snapshot.fps.toFixed(1) : '—');
    const hands =
      snapshot.hands > 0
        ? `${snapshot.hands}${snapshot.handedness ? ` (${snapshot.handedness})` : ''}`
        : '0';
    this.write(this.cells.hands, hands);
  }

  /** Unmount the cells from the host element. */
  dispose(): void {
    this.element.replaceChildren();
    this.element.removeAttribute('role');
    this.element.removeAttribute('aria-label');
  }

  /** Write a cell's text only when it changed (per-frame cheapness). */
  private write(cell: MetricsCell, text: string): void {
    if (cell.lastText === text) return;
    cell.lastText = text;
    cell.value.textContent = text;
  }

  private static createCell(label: string, initialValue: string): MetricsCell {
    const cell = document.createElement('span');
    cell.className = 'metrics-bar__cell';
    const labelElement = document.createElement('span');
    labelElement.className = 'metrics-bar__label';
    labelElement.textContent = label;
    const value = document.createElement('span');
    value.className =
      label === 'FPS' ? 'metrics-bar__value metrics-bar__value--fps' : 'metrics-bar__value';
    value.textContent = initialValue;
    cell.append(labelElement, value);
    return { cell, value, lastText: null };
  }
}
