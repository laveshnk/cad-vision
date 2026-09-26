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
 *   show(x, y)         → float the wheel next to an anchor point (the
 *                        selected mesh's screen projection); it flips left
 *                        and clamps so the disc always stays on screen.
 *   hide()             → fade out; `pickColorAt` then returns null.
 *   pickColorAt(x, y)  → `#rrggbb` under a viewport-local point, or null
 *                        when the point misses the disc; also moves the
 *                        on-disc hover cursor for immediate feedback.
 *
 * The HSL math is exported as pure functions so it stays unit-testable
 * without a DOM (see __tests__/colorWheel.test.ts).
 */

export interface ColorWheelOptions {
  /** Disc diameter in CSS pixels. */
  size?: number;
  /** Gap between the anchor point and the wheel rim (CSS px). */
  gap?: number;
  /** Disc lightness (hue = angle, saturation = radius); HSL L in [0, 1]. */
  lightness?: number;
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

export class ColorWheel {
  private readonly root: HTMLElement;
  private readonly options: Required<ColorWheelOptions>;
  private readonly element: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly cursor: HTMLDivElement;
  /** Disc center in root-local CSS px — the origin `pickColorAt` measures from. */
  private centerX = 0;
  private centerY = 0;
  private shown = false;

  constructor(root: HTMLElement, options: ColorWheelOptions = {}) {
    this.root = root;
    this.options = {
      size: options.size ?? 132,
      gap: options.gap ?? 22,
      lightness: options.lightness ?? 0.5,
    };
    this.element = document.createElement('div');
    this.element.className = 'color-wheel';
    this.element.setAttribute('role', 'img');
    this.element.setAttribute('aria-label', 'Selection color wheel');
    this.canvas = document.createElement('canvas');
    this.cursor = document.createElement('div');
    this.cursor.className = 'color-wheel-cursor';
    this.element.append(this.canvas, this.cursor);
    root.appendChild(this.element);
    this.renderWheel();
  }

  /** Whether the wheel is currently shown. */
  get visible(): boolean {
    return this.shown;
  }

  /**
   * Float the wheel next to an anchor point in root-local CSS px (the same
   * space `pickColorAt` uses). The disc sits a gap to the anchor's right,
   * flips to the left when it would overflow the root's right edge, and is
   * clamped both ways so it always stays fully inside the root.
   */
  show(screenX: number, screenY: number): void {
    this.shown = true;
    const radius = this.options.size / 2;
    const width = Math.max(this.root.clientWidth, this.options.size);
    const height = Math.max(this.root.clientHeight, this.options.size);
    const reach = radius + this.options.gap;
    let x = screenX + reach; // prefer the anchor's right side
    if (x + radius > width) x = screenX - reach; // flip when it would overflow
    this.centerX = clamp(x, radius, width - radius);
    this.centerY = clamp(screenY, radius, height - radius);
    this.element.style.left = `${this.centerX}px`;
    this.element.style.top = `${this.centerY}px`;
    this.element.classList.add('color-wheel--visible');
  }

  /** Hide the wheel (fades out via CSS); `pickColorAt` returns null until the next `show`. */
  hide(): void {
    this.shown = false;
    this.element.classList.remove('color-wheel--visible');
    this.setHover(null);
  }

  /**
   * Color under a root-local point (CSS px): the `#rrggbb` hex beneath it,
   * or null when the wheel is hidden or the point misses the disc. Also
   * moves the hover cursor onto the point for on-wheel feedback.
   */
  pickColorAt(screenX: number, screenY: number): string | null {
    if (!this.shown) return null;
    const color = wheelColorAt(
      screenX - this.centerX,
      screenY - this.centerY,
      this.options.size / 2,
      this.options.lightness
    );
    this.setHover(color === null ? null : { x: screenX, y: screenY, color });
    return color;
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
    this.cursor.style.background = hover.color;
    this.cursor.classList.add('color-wheel-cursor--visible');
  }

  /**
   * Paint the static HSL disc once at the display's pixel ratio: hue by
   * angle, saturation by radius (clamped past the rim — the square canvas
   * corners are clipped away by the element's circular `border-radius`, so
   * the visible disc and the pickable disc are the exact same circle).
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
        const hue = (Math.atan2(dy, dx) * 180) / Math.PI;
        const saturation = Math.min(1, Math.hypot(dx, dy) / radius);
        const { r, g, b } = hslToRgb(hue, saturation, this.options.lightness);
        const offset = (py * bitmap + px) * 4;
        data[offset] = r;
        data[offset + 1] = g;
        data[offset + 2] = b;
        data[offset + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }
}
