/**
 * DimensionLabels: floating measurement chips over the 3D viewport, one per
 * solid (e.g. `L 12.0 cm · W 8.0 cm · H 5.0 cm`), pinned just above each
 * solid's top. Pure UI: the host projects anchors to root-local CSS px and
 * passes plain text lines each frame; chips are pooled and re-used, so a
 * per-frame `render` costs no DOM churn while the object count is stable.
 * `pointer-events: none` — the labels never block the viewport.
 */

export interface DimensionLabel {
  /** Root-local CSS px of the anchor (the chip's bottom-center sits here). */
  x: number;
  y: number;
  /** One text entry per dimension, e.g. `['R 4.0 cm', 'H 20.0 cm']`. */
  entries: string[];
  /** Style as a live (still-being-sized) build preview. */
  preview?: boolean;
}

export class DimensionLabels {
  private readonly layer: HTMLDivElement;
  private readonly chips: HTMLDivElement[] = [];

  constructor(root: HTMLElement) {
    this.layer = document.createElement('div');
    this.layer.className = 'dimension-layer';
    this.layer.setAttribute('aria-hidden', 'true');
    root.appendChild(this.layer);
  }

  /** Show exactly `labels` (an empty list hides every chip). */
  render(labels: readonly DimensionLabel[]): void {
    while (this.chips.length < labels.length) {
      const chip = document.createElement('div');
      chip.className = 'dimension-chip';
      this.layer.appendChild(chip);
      this.chips.push(chip);
    }
    this.chips.forEach((chip, i) => {
      const label = labels[i];
      if (!label) {
        chip.hidden = true;
        return;
      }
      chip.hidden = false;
      chip.style.transform = `translate(${label.x}px, ${label.y}px) translate(-50%, -100%)`;
      chip.classList.toggle('dimension-chip--preview', label.preview === true);
      const text = label.entries.join(' · ');
      if (chip.textContent !== text) chip.textContent = text;
    });
  }

  /** Unmount the layer. */
  dispose(): void {
    this.layer.remove();
    this.chips.length = 0;
  }
}
