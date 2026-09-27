/**
 * hitTest: shared DOM hit-testing for the floating UI overlays (selection
 * menu, confirm dialog).
 *
 * The overlays are mounted inside a host element (`#viewport`) and driven
 * with *root-local* CSS pixel coordinates — the same space the gesture
 * bridge derives from device space. This helper answers "is a root-local
 * point on this element?" by measuring the element's live bounding rect
 * (both rects via `getBoundingClientRect`, so centering transforms,
 * clamping and any future layout changes are accounted for automatically).
 */

/** Whether the root-local point (x, y) lies inside `element`'s live rect. */
export function hitTestElement(
  element: Element,
  root: Element,
  x: number,
  y: number
): boolean {
  const elementRect = element.getBoundingClientRect();
  const rootRect = root.getBoundingClientRect();
  return (
    x >= elementRect.left - rootRect.left &&
    x <= elementRect.right - rootRect.left &&
    y >= elementRect.top - rootRect.top &&
    y <= elementRect.bottom - rootRect.top
  );
}
