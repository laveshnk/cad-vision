/**
 * ColorWheel: lightweight floating HSL color wheel for the CAD viewport
 * (SELECT-mode selection repaint).
 *
 * A canvas-drawn disc — hue sweeps around the angle, saturation grows with
 * the radius, lightness is fixed (0.5 by default) — mounted as a
 * `pointer-events: none` overlay, so it never blocks the 3D canvas. The
 * wheel is pure UI: it knows nothing about gestures or vision. The app
 * orchestrator (src/main.ts) feeds it viewport-local CSS pixel positions
 * (the space CadScene's NDC frustum maps onto) and applies the picked hex
 * to the selected mesh through CadBuilder.
 *
 * Interaction contract (driven entirely by the orchestrator):
 *   showAt(x, y)       → pin the disc center at a point (clamped inside the
 *                        root), e.g. a fixed corner of the reachable area.
 *   show(x, y)         → float the wheel next to an anchor point (the
 *                        selected mesh's screen projection); it flips left
 *                        and clamps so the disc always stays on screen.
 *   hide()             → fade out; `pickColorAt` then returns null.
 *   pickColorAt(x, y)  → `#rrggbb` under a viewport-local point, or null
 *                        when the point misses the disc; also moves the
 *                        on-disc hover cursor for immediate feedback.
 *   advanceDwell(c,dt) → timed hover lock: while the hover stays on one
 *                        color slice, the cursor's countdown ring fills;
 *                        `dwellMs` (default 1.2 s) of continuous hover
 *                        commits — returns the locked hex exactly once,
 *                        then the clock resets. A null color resets it.
 *   resetDwell()       → empty the countdown ring and restart the clock.
 *
 * The HSL math, the color-slice quantization and the dwell tracker are
 * exported as pure functions / classes so they stay unit-testable without
 * a DOM (see __tests__/colorWheel.test.ts).
 */

export interface ColorWheelOptions {
  /** Disc diameter in CSS pixels. */
  size?: number;
  /**
   * Headless: keep the wheel's geometry, color picking and timed hover lock
   * but never mount a visible disc (the host draws its own — e.g. in the
   * camera view). Default false.
   */
  headless?: boolean;
  /** Gap between the anchor point and the wheel rim (CSS px). */
  gap?: number;
  /** Disc lightness (hue = angle, saturation = radius); HSL L in [0, 1]. */
  lightness?: number;
  /** Continuous hover time required to lock a color (ms). Default 1200. */
  dwellMs?: number;
}

/** An RGB color with channels already scaled to 0–255. */
export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Hue (degrees) + saturation ([0, 1]) at a wheel offset: hue follows the
 * angle, saturation grows with `distance / radius`. Null when the offset
 * lies outside the disc or the radius degenerates. This is the same mapping
 * that answers `pickColorAt`, so the pickable disc and the visible disc
 * never diverge.
 */
function wheelHueSaturation(
  dx: number,
  dy: number,
  radius: number
): { hue: number; saturation: number } | null {
  if (radius <= 0) return null;
  const distance = Math.hypot(dx, dy);
  if (distance > radius) return null;
  return { hue: (Math.atan2(dy, dx) * 180) / Math.PI, saturation: distance / radius };
}

/**
 * HSL → RGB (channels 0–255). `h` in degrees (wrapped into [0, 360)),
 * `s` / `l` clamped into [0, 1]. Never throws.
 */
export function hslToRgb(h: number, s: number, l: number): RgbColor {
  const hue = (((h % 360) + 360) % 360) / 60;
  const light = clamp(l, 0, 1);
  const c = (1 - Math.abs(2 * light - 1)) * clamp(s, 0, 1);
  const x = c * (1 - Math.abs((hue % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (hue < 1) {
    r = c;
    g = x;
  } else if (hue < 2) {
    r = x;
    g = c;
  } else if (hue < 3) {
    g = c;
    b = x;
  } else if (hue < 4) {
    g = x;
    b = c;
  } else if (hue < 5) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }
  const m = light - c / 2;
  const channel = (value: number): number => Math.round(clamp(value + m, 0, 1) * 255);
  return { r: channel(r), g: channel(g), b: channel(b) };
}

/** HSL → `#rrggbb` hex string (lowercase, always 7 characters). */
export function hslToHex(h: number, s: number, l: number): string {
  const { r, g, b } = hslToRgb(h, s, l);
  const channel = (value: number): string => value.toString(16).padStart(2, '0');
  return `#${channel(r)}${channel(g)}${channel(b)}`;
}

/**
 * Hex color at a wheel offset (`dx`, `dy` in px from the disc center):
 * `#rrggbb`, or null when the offset lies outside the disc. `lightness`
 * defaults to 0.5.
 */
export function wheelColorAt(
  dx: number,
  dy: number,
  radius: number,
  lightness = 0.5
): string | null {
  const wheel = wheelHueSaturation(dx, dy, radius);
  if (!wheel) return null;
  return hslToHex(wheel.hue, wheel.saturation, lightness);
}

/** Zoom a freshly shown wheel starts from before easing up (pop-in). */
const APPEAR_ZOOM = 0.3;

/** Hue sector width (degrees) — the tolerance band of one "color slice". */
const SLICE_DEGREES = 15;
/** Saturation below this the wheel reads as the neutral (gray) center slice. */
const NEUTRAL_SATURATION = 0.08;

/**
 * Quantized "color slice" key at a wheel offset: hue sectors of 15° (the
 * stability tolerance for a hand-held hover), with the desaturated center
 * collapsing into a single neutral slice. Null outside the disc. The timed
 * hover lock dwells on slice keys, not exact hexes, so micro-jitter of the
 * fingertip never restarts the clock.
 */
export function wheelSliceAt(dx: number, dy: number, radius: number): string | null {
  const wheel = wheelHueSaturation(dx, dy, radius);
  if (!wheel) return null;
  if (wheel.saturation < NEUTRAL_SATURATION) return 'neutral';
  const hue = ((wheel.hue % 360) + 360) % 360;
  return `hue-${Math.floor(hue / SLICE_DEGREES)}`;
}

export interface DwellTrackerOptions {
  /** Continuous hover time required to commit (ms). Default 1200. */
  durationMs?: number;
  /**
   * Largest single time step counted (ms): a stalled / resumed feed cannot
   * complete a dwell in one giant jump (mirrors the overlay's frame clamp).
   */
  maxStepMs?: number;
}

/**
 * DwellTracker: the timed hover lock's clock. Accumulates time while the
 * hovered key stays constant; any key change or hover loss resets it.
 * Fires exactly once when the accumulated time reaches `durationMs`, then
 * resets. Pure logic — no DOM, no timers; the host advances it per frame.
 */
export class DwellTracker {
  private readonly durationMs: number;
  private readonly maxStepMs: number;
  private accumulatedMs = 0;
  private currentKey: string | null = null;

  constructor(options: DwellTrackerOptions = {}) {
    this.durationMs = options.durationMs ?? 1200;
    this.maxStepMs = options.maxStepMs ?? 500;
  }

  /** The key currently being dwelled on (null when not hovering). */
  get key(): string | null {
    return this.currentKey;
  }

  /** Current dwell completion ratio [0, 1]. */
  get progress(): number {
    if (this.currentKey === null || this.durationMs <= 0) return 0;
    return Math.min(1, this.accumulatedMs / this.durationMs);
  }

  /**
   * Advance the clock by `dtMs` while hovering `key` (null = hover lost:
   * full reset). Returns true exactly once when the dwell completes.
   */
  advance(key: string | null, dtMs: number): boolean {
    if (key === null) {
      this.reset();
      return false;
    }
    if (key !== this.currentKey) {
      this.currentKey = key;
      this.accumulatedMs = 0;
    }
    this.accumulatedMs += Math.min(Math.max(dtMs, 0), this.maxStepMs);
    if (this.accumulatedMs >= this.durationMs) {
      this.reset();
      return true;
    }
    return false;
  }

  /** Stop dwelling: drop the key and empty the clock. */
  reset(): void {
    this.currentKey = null;
    this.accumulatedMs = 0;
  }
}

export class ColorWheel {
  private readonly root: HTMLElement;
  private readonly options: Required<ColorWheelOptions>;
  private readonly element: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly cursor: HTMLDivElement;
  private readonly dot: HTMLSpanElement;
  private readonly ringFill: SVGCircleElement;
  private readonly dwell: DwellTracker;
  /** Disc center in root-local CSS px — the origin `pickColorAt` measures from. */
  private centerX = 0;
  private centerY = 0;
  /** Color slice under the last `pickColorAt` (null when it missed the disc). */
  private hoverSlice: string | null = null;
  private shown = false;
  /** Radius multiplier (e.g. enlarged while a fingertip hovers the disc). */
  private zoom = 1;
  /** Zoom the disc is easing toward (`animateZoomTo`). */
  private zoomTarget = 1;

  constructor(root: HTMLElement, options: ColorWheelOptions = {}) {
    this.root = root;
    this.options = {
      size: options.size ?? 132,
      gap: options.gap ?? 22,
      lightness: options.lightness ?? 0.5,
      dwellMs: options.dwellMs ?? 1200,
      headless: options.headless ?? false,
    };
    this.dwell = new DwellTracker({ durationMs: this.options.dwellMs });
    this.element = document.createElement('div');
    this.element.className = 'color-wheel';
    this.element.setAttribute('role', 'img');
    this.element.setAttribute('aria-label', 'Selection color wheel');
    this.canvas = document.createElement('canvas');
    this.cursor = document.createElement('div');
    this.cursor.className = 'color-wheel-cursor';
    // Hover cursor = color dot wrapped in the timed hover lock's countdown
    // ring (the ring fills as the dwell approaches its commit).
    const ring = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    ring.setAttribute('class', 'color-wheel-ring');
    ring.setAttribute('viewBox', '0 0 28 28');
    ring.setAttribute('aria-hidden', 'true');
    const track = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    track.setAttribute('class', 'color-wheel-ring-track');
    track.setAttribute('cx', '14');
    track.setAttribute('cy', '14');
    track.setAttribute('r', '12');
    this.ringFill = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    this.ringFill.setAttribute('class', 'color-wheel-ring-fill');
    this.ringFill.setAttribute('cx', '14');
    this.ringFill.setAttribute('cy', '14');
    this.ringFill.setAttribute('r', '12');
    this.ringFill.setAttribute('pathLength', '100');
    ring.append(track, this.ringFill);
    this.dot = document.createElement('span');
    this.dot.className = 'color-wheel-dot';
    this.cursor.append(ring, this.dot);
    this.element.append(this.canvas, this.cursor);
    if (!this.options.headless) root.appendChild(this.element);
    this.renderWheel();
  }

  /** Whether the wheel is currently shown. */
  get visible(): boolean {
    return this.shown;
  }

  /**
   * Disc center in root-local CSS px (the same space `pickColorAt` measures
   * from), or null while hidden — lets the host mirror the wheel's live
   * placement into the camera-thumbnail HUD.
   */
  get center(): { x: number; y: number } | null {
    return this.shown ? { x: this.centerX, y: this.centerY } : null;
  }

  /** Disc radius in CSS px: half of `size`, times the current zoom. */
  get radius(): number {
    return (this.options.size / 2) * this.zoom;
  }

  /**
   * Enlarge (or restore) the active disc: picking, clamping and `radius`
   * all follow the zoom, so a host-drawn disc of `radius` stays exact.
   */
  /** Current radius multiplier (1 = the configured `size`). */
  get zoomFactor(): number {
    return this.zoom;
  }

  setZoom(factor: number): void {
    this.zoom = Number.isFinite(factor) && factor > 0 ? factor : 1;
    this.zoomTarget = this.zoom;
  }

  /**
   * Ease the zoom toward `target` over one frame of `dtMs` (exponential
   * approach, time constant `timeConstantMs`): smooth, interruptible grow /
   * shrink animations — e.g. hover enlarge and release. Picking and
   * `radius` follow the animated value, so a host-drawn disc stays exact.
   */
  animateZoomTo(target: number, dtMs: number, timeConstantMs = 80): void {
    if (!Number.isFinite(target) || target <= 0) return;
    this.zoomTarget = target;
    const blend = dtMs > 0 ? 1 - Math.exp(-dtMs / Math.max(1, timeConstantMs)) : 0;
    this.zoom += (this.zoomTarget - this.zoom) * blend;
    if (Math.abs(this.zoomTarget - this.zoom) < 0.002) this.zoom = this.zoomTarget;
  }

  /** Timed hover lock completion [0, 1] (the countdown ring's fill ratio). */
  get dwellProgress(): number {
    return this.dwell.progress;
  }

  /**
   * Float the wheel next to an anchor point in root-local CSS px (the same
   * space `pickColorAt` uses). The disc sits a gap to the anchor's right,
   * flips to the left when it would overflow the root's right edge, and is
   * clamped both ways so it always stays fully inside the root.
   */
  show(screenX: number, screenY: number): void {
    this.shown = true;
    const radius = this.radius;
    const width = Math.max(this.root.clientWidth, radius * 2);
    const height = Math.max(this.root.clientHeight, radius * 2);
    const reach = radius + this.options.gap;
    let x = screenX + reach; // prefer the anchor's right side
    if (x + radius > width) x = screenX - reach; // flip when it would overflow
    this.centerX = clamp(x, radius, width - radius);
    this.centerY = clamp(screenY, radius, height - radius);
    this.element.style.left = `${this.centerX}px`;
    this.element.style.top = `${this.centerY}px`;
    this.element.classList.add('color-wheel--visible');
  }

  /**
   * Pin the disc center at a root-local point (CSS px), clamped so the disc
   * stays fully inside the root — for a fixed placement such as a corner.
   */
  showAt(centerX: number, centerY: number): void {
    // Pop-in: a freshly shown wheel starts small and eases up to its zoom
    // (via `animateZoomTo`) instead of snapping into view.
    if (!this.shown) this.zoom = Math.min(this.zoom, APPEAR_ZOOM);
    this.shown = true;
    const radius = this.radius;
    const width = Math.max(this.root.clientWidth, radius * 2);
    const height = Math.max(this.root.clientHeight, radius * 2);
    this.centerX = clamp(centerX, radius, width - radius);
    this.centerY = clamp(centerY, radius, height - radius);
    this.element.style.left = `${this.centerX}px`;
    this.element.style.top = `${this.centerY}px`;
    this.element.classList.add('color-wheel--visible');
  }

  /** Hide the wheel (fades out via CSS); `pickColorAt` returns null until the next `show`. */
  hide(): void {
    this.shown = false;
    this.element.classList.remove('color-wheel--visible');
    this.setHover(null);
    this.resetDwell();
  }

  /**
   * Color under a root-local point (CSS px): the `#rrggbb` hex beneath it,
   * or null when the wheel is hidden or the point misses the disc. Also
   * moves the hover cursor onto the point and remembers its color slice
   * (the timed hover lock's dwell key).
   */
  pickColorAt(screenX: number, screenY: number): string | null {
    if (!this.shown) {
      this.hoverSlice = null;
      return null;
    }
    const dx = screenX - this.centerX;
    const dy = screenY - this.centerY;
    const radius = this.radius;
    const color = wheelColorAt(dx, dy, radius, this.options.lightness);
    this.hoverSlice = wheelSliceAt(dx, dy, radius);
    this.setHover(color === null ? null : { x: screenX, y: screenY, color });
    return color;
  }

  /**
   * Advance the timed hover lock by `dtMs` (a frame delta from the host):
   * while the hover stays on one color slice, the cursor's countdown ring
   * fills; `dwellMs` of continuous hover commits — the locked hex is
   * returned exactly once and the clock resets. A null `color` (hover
   * lost or the wheel hidden) resets the clock.
   */
  advanceDwell(color: string | null, dtMs: number): string | null {
    if (color === null) {
      this.dwell.reset();
      this.setDwellProgress(0);
      return null;
    }
    const committed = this.dwell.advance(this.hoverSlice, dtMs);
    this.setDwellProgress(this.dwell.progress);
    return committed ? color : null;
  }

  /** Reset the timed hover lock (the countdown ring empties). */
  resetDwell(): void {
    this.dwell.reset();
    this.setDwellProgress(0);
  }

  /** Unmount the wheel from its host element. */
  dispose(): void {
    this.element.remove();
  }

  /** Show / move the hover cursor to a root-local point, or hide it. */
  private setHover(hover: { x: number; y: number; color: string } | null): void {
    if (!hover) {
      this.cursor.classList.remove('color-wheel-cursor--visible');
      return;
    }
    const size = this.options.size;
    this.cursor.style.left = `${clamp(hover.x - this.centerX + size / 2, 0, size)}px`;
    this.cursor.style.top = `${clamp(hover.y - this.centerY + size / 2, 0, size)}px`;
    this.dot.style.background = hover.color;
    this.cursor.classList.add('color-wheel-cursor--visible');
  }

  /** Fill the cursor's countdown ring (0 = empty, 1 = complete). */
  private setDwellProgress(ratio: number): void {
    this.ringFill.style.strokeDashoffset = `${100 - clamp(ratio, 0, 1) * 100}`;
  }

  /**
   * Paint the static HSL disc once at the display's pixel ratio: hue by
   * angle, saturation by radius. The corners stay transparent and the rim
   * gets a one-pixel anti-aliased fade (the element itself does not clip,
   * so the hover cursor's countdown ring may overhang the disc edge); the
   * visible disc and the pickable disc are the exact same circle.
   */
  private renderWheel(): void {
    const size = this.options.size;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const bitmap = Math.max(1, Math.round(size * dpr));
    this.canvas.width = bitmap;
    this.canvas.height = bitmap;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    const radius = bitmap / 2;
    const image = ctx.createImageData(bitmap, bitmap);
    const data = image.data;
    for (let py = 0; py < bitmap; py++) {
      for (let px = 0; px < bitmap; px++) {
        const dx = px + 0.5 - radius;
        const dy = py + 0.5 - radius;
        const distance = Math.hypot(dx, dy);
        if (distance > radius) continue; // transparent corners
        const hue = (Math.atan2(dy, dx) * 180) / Math.PI;
        const saturation = Math.min(1, distance / radius);
        const { r, g, b } = hslToRgb(hue, saturation, this.options.lightness);
        const offset = (py * bitmap + px) * 4;
        data[offset] = r;
        data[offset + 1] = g;
        data[offset + 2] = b;
        data[offset + 3] = distance > radius - 1 ? Math.round(255 * (radius - distance)) : 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }
}
