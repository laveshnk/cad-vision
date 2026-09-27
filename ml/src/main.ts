/**
 * App orchestrator: gesture-driven CAD workbench.
 *
 * Strictly decoupled module wiring:
 *
 *   Vision (GestureEngine)  --typed events-->  App (this file)
 *        --> CAD (CadScene + CadBuilder)  +  UI (Toolbar / MetricsBar / …)
 *
 * The vision layer emits device-space coordinates (x / y in [-1, 1], +Y up),
 * deltas and state events; the CAD layer consumes only those — nothing in
 * src/cad or src/ui imports from src/vision.
 *
 * Interaction modes (VIEW / SELECT / CREATE) are picked on the vision
 * overlay's button bar and gate the gesture routing below: pinches are inert
 * in VIEW, pick / drag / recolor / delete meshes in SELECT (a Delete HUD
 * follows the selection and a color wheel sits mid-right; a pointing
 * fingertip sweeps the wheel and a 1.2 s hover locks the color; deletion is
 * confirmation-gated),
 * and build primitives in CREATE. SELECT also carries the CSG Boolean tools
 * (`[ SUBTRACT ]` / `[ UNION ]` overlay toggles + the secondary-hand X-cross
 * trigger; a live world-AABB clash indicator marks intersecting solids).
 * Camera gestures (fist orbit / two-fist zoom) stay live in every mode.
 *
 * Destructive actions are confirmed in-vision only: the overlay's trash bin
 * (scene clear) and every delete trigger open a spatial confirmation dialog
 * drawn on the camera canvas — `[ CONFIRM (Hold) ]` / `[ CANCEL (Hold) ]` —
 * answered by a pointing hold on a target (or a held OK gesture) for
 * `HOLD_TO_ACT_MS`, the same wait as the color wheel lock and the Delete
 * button, or cancelled by an open palm / moving the hands away. No window.confirm(), no DOM modals. While
 * it is open (`overlay.confirmActive`) every scene gesture is frozen.
 *
 * The live stats (mode / state / FPS / hands) render in the DOM metrics bar
 * mounted directly underneath the camera view (`MetricsBar`); the camera
 * card itself is repositioned by dragging its outer frame (`ThumbDragger`)
 * and resized via its corner grip (`ThumbResizer`).
 */

import { DebugOverlay, GestureEngine, MODE_LABELS } from './vision';
import type {
  FrameEvent,
  GestureSignalEvent,
  Handedness,
  OverlayConfirmIntent,
  OverlayHudDisc,
  OverlayHudRect,
  OverlaySelectionHud,
} from './vision';
import { CadScene } from './cad/CadScene';
import { CadBuilder, type CadTool } from './cad/CadBuilder';
import { buildArSceneFrame } from './cad/ArMirror';
import { formatDimension } from './cad/dimensions';
import { Toolbar } from './ui/Toolbar';
import { ColorWheel } from './ui/ColorWheel';
import { SelectionMenu } from './ui/SelectionMenu';
import { MetricsBar } from './ui/MetricsBar';
import { ThumbDragger } from './ui/ThumbDragger';
import { ThumbResizer } from './ui/ThumbResizer';
import { DimensionLabels } from './ui/DimensionLabels';

const video = document.querySelector<HTMLVideoElement>('#video');
const canvas = document.querySelector<HTMLCanvasElement>('#overlay');
const statusText = document.querySelector<HTMLElement>('#status');
const toolbarRoot = document.querySelector<HTMLElement>('#toolbar');
const viewport = document.querySelector<HTMLElement>('#viewport');
const visionThumb = document.querySelector<HTMLElement>('#vision-thumb');
const thumbExpand = document.querySelector<HTMLButtonElement>('#thumb-expand');
const metricsRoot = document.querySelector<HTMLElement>('#metrics');

if (
  !video ||
  !canvas ||
  !statusText ||
  !toolbarRoot ||
  !viewport ||
  !visionThumb ||
  !thumbExpand ||
  !metricsRoot
) {
  throw new Error('cad-vision: required DOM elements are missing');
}

const statusOutput: HTMLElement = statusText;
// Non-null locals: hoisted function declarations cannot rely on the
// module-level null-check narrowing above.
const videoElement: HTMLVideoElement = video;
const overlayCanvas: HTMLCanvasElement = canvas;
const viewportElement: HTMLElement = viewport;

/* ---- Vision ---- */
const engine = new GestureEngine({
  smoothing: { strategy: 'ema', alpha: 0.35 },
  classifier: {
    pinchStartThreshold: 0.045,
    pinchReleaseThreshold: 0.065,
    // Accidental-pinch guards: a pinch must hold 2 frames and cannot start
    // at the frame border (stray / half-visible hands in the corners).
    pinchEnterFrames: 2,
    pinchEdgeMargin: 0.03,
    pinchDistanceSmoothing: 0.5,
    fistFoldRatio: 0.9,
    fistMinFoldedFingers: 4,
    fistThumbTuckRatio: 0.75,
    fistHoldSlack: 0.15,
    fistPinchGuardDistance: 0.07,
    fistEnterFrames: 3,
    rollEngageAngle: 0.15,
    rollDeadzone: 0.003,
    zoomAnchorSwitchRatio: 0.5,
    zoomTurnEngageAngle: 0.12,
    // Camera moves travel in straight segments (hand wiggle removed); pinch /
    // extrude events are not filtered, so building stays live.
    cameraPath: { startDistance: 0.02, cornerDeviation: 0.05, settleDistance: 0.08, deadband: 0.004 },
    fistExitFrames: 2,
    orbitOpenPalmGraceFrames: 10,
    handLossGraceFrames: 3,
  },
});

/* ---- CAD ---- */
const cadScene = new CadScene(viewport);
const builder = new CadBuilder(cadScene);

/* ---- Vision overlay ---- */
/**
 * One hold-to-act wait (ms) shared by the SELECT-mode touch UI: locking a
 * color on the wheel, the Delete button, and answering the delete / clear
 * confirmation dialog all take the same pointing hold.
 */
const HOLD_TO_ACT_MS = 1200;

/** Shortest two-hand build (ms) that is committed; quicker ones are discarded. */
const MIN_BUILD_MS = 400;

/** Whether the viewport shows every solid's dimensions (cm). */
let dimensionsVisible = false;

const overlay = new DebugOverlay<CadTool>(canvas, {
  confirmHoldMs: HOLD_TO_ACT_MS,
  // Dimensions (ruler) toggle at the right end of the mode bar — in every
  // mode; mirrored by the toolbar's Dimensions button.
  onDimensionsToggle: (visible) => setDimensionsVisible(visible),
  // Mode switcher on the vision overlay: the button bar (mouse click, finger
  // dwell or pinch) requests engine mode changes; the engine feeds the
  // active mode back through the per-frame event, which renders the button.
  onModeRequest: (mode) => engine.setMode(mode),
  // CREATE mode shows shape icon buttons down the right edge of the camera
  // view; the picked shape is the builder's tool for the next build
  // (default: cube).
  shapes: [
    { id: 'box', label: 'CUBE', icon: 'cube' },
    { id: 'cuboid', label: 'CUBOID', icon: 'cuboid' },
    { id: 'cylinder', label: 'CYLINDER', icon: 'cylinder' },
    { id: 'sphere', label: 'SPHERE', icon: 'sphere' },
  ],
  activeShape: builder.activeTool,
  onShapeRequest: (shape) => builder.setTool(shape),
  // SELECT-mode drag constraint toggles ([ XZ PLANE ] / [ Y AXIS ]) on the
  // vision overlay: the request routes straight into the builder, which
  // enforces it inside dragTo (the overlay renders the active state).
  onDragConstraintRequest: (constraint) => builder.setDragConstraint(constraint),
  // In-vision trash bin (bottom-right of the camera view): pointing dwell,
  // a fresh pinch over the icon or a mouse click opens the spatial
  // confirmation dialog — the scene only clears on Confirm.
  onTrashRequest: () => requestSceneClear(),
  // Spatial confirmation dialog: every destructive action (scene clear,
  // object deletion) is answered in-vision — no browser popups, no DOM
  // modals. The overlay reports the gesture / mouse answer back here.
  onConfirmRequest: (intent) => confirmDestructive(intent),
  onCancelRequest: () => cancelDestructive(),
  // SELECT-mode CSG Boolean tool toggles ([ SUBTRACT ] / [ UNION ]): the
  // request arms the builder's tool; when meshes already intersect, the
  // operation fires immediately (arm + run).
  onBooleanToolRequest: (tool) => {
    builder.setBooleanTool(tool);
    if (tool !== null) builder.applyBoolean();
  },
  // Secondary-hand "X" cross (index + pinky up, middle + ring folded) held
  // briefly: fire a SUBTRACT on the live clash.
  onBooleanTrigger: () => builder.applyBoolean('subtract'),
  // Live Boolean state (armed tool + clash availability) feeds the overlay's
  // ready-to-run glow on the armed toggle.
  booleanState: () => ({ ...builder.booleanState, canUngroup: builder.canUngroup }),
  // EDIT-mode UNGROUP: split the selected union back into its parts (each
  // with its own colors, placed where the union has moved / turned to).
  onUngroupRequest: () => {
    if (builder.ungroupSelection()) setStatus('Ungrouped');
  },
  // SELECT-mode AR mirror: the ground grid + every committed mesh are
  // projected through the shared 3D camera onto the vision canvas, turning
  // it into a translucent live spatial mirror of the 3D viewport — in
  // lockstep with pinch-driven drags (the engine emits pinch events before
  // the per-frame event, so ghost and viewport never diverge).
  arScene: (width, height) => buildArSceneFrame(cadScene, builder, width, height),
  // SELECT-mode selection HUD: the color wheel's disc outline and the Delete
  // button are mirrored into the camera thumbnail (device space, so the
  // drawn controls sit exactly where a fingertip must point), with both
  // dwell progress bars as live feedback.
  selectionHud: () => selectionHudFrame(),
});

/* ---- UI ---- */
const toolbar = new Toolbar(toolbarRoot, {
  onExportStl: () => builder.exportStl(),
  onCameraStart: () => startCamera(),
  onCameraStop: () => stopCamera(),
  onDimensionsToggle: () => setDimensionsVisible(!dimensionsVisible),
});

// Dimension labels (cm) over the 3D view: one chip per solid (and the live
// build preview), re-projected after every render so they track the damped
// camera and dragged / rotated solids exactly.
const dimensionLabels = new DimensionLabels(viewport);
cadScene.onFrame(() => {
  if (!dimensionsVisible) {
    dimensionLabels.render([]);
    return;
  }
  const width = viewportElement.clientWidth;
  const height = viewportElement.clientHeight;
  const labels = [];
  for (const annotation of builder.dimensionAnnotations()) {
    const point = cadScene.projectToCanvas(annotation.anchor, width, height);
    if (point.z > 1) continue; // behind the camera
    labels.push({
      x: point.x,
      y: point.y,
      entries: annotation.entries.map(formatDimension),
      preview: annotation.preview,
    });
  }
  dimensionLabels.render(labels);
});

/** Show / hide the dimension labels, keeping both toggles in sync. */
function setDimensionsVisible(visible: boolean): void {
  dimensionsVisible = visible;
  overlay.setDimensionsVisible(visible);
  toolbar.setDimensionsVisible(visible);
}

// HSL color wheel (EDIT mode), headless: nothing is drawn in the 3D view —
// the wheel lives in the camera view (the overlay draws it from this
// model's center / radius). The model keeps the geometry, hue picking and
// timed hover lock; the bridge below feeds it viewport-local pixels (the
// same space the camera-view disc maps to) and applies the picked hex
// through CadBuilder.setSelectedColor.
const colorWheel = new ColorWheel(viewport, { dwellMs: HOLD_TO_ACT_MS });

/** Color-wheel zoom while a pointing fingertip hovers it (camera view). */
const WHEEL_HOVER_ZOOM = 2;

// Floating selection HUD (SELECT mode): a Delete action anchored below the
// selection's screen projection. Deletion is destructive, so every trigger
// routes through the in-vision spatial confirmation dialog below before
// the mesh is removed.
const selectionMenu = new SelectionMenu(viewport, {
  dwellMs: HOLD_TO_ACT_MS,
  onDeleteRequest: () => requestDeleteSelection(),
});

// Live stats bar, mounted directly underneath the camera view (inside the
// floating thumbnail card): mode / state / FPS / hands per processed frame.
const metricsBar = new MetricsBar(metricsRoot);

// Camera-window dragging: the card's outer frame (header + metrics bar)
// repositions the thumbnail; the video stage never drags it.
const thumbDragger = new ThumbDragger(visionThumb);

// Mouse-drag resizing of the floating camera thumbnail: a corner grip on
// the card's outer frame (bottom-right, outside the video stage) scales
// the card (the overlay re-measures its canvas every frame, so nothing
// else needs a resize listener).
const thumbResizer = new ThumbResizer(visionThumb);

function setStatus(text: string): void {
  statusOutput.textContent = text;
}

/** Forward gesture events to the console as JSON (downstream consumer demo). */
function logGestureEvent(event: GestureSignalEvent): void {
  console.log(JSON.stringify({ source: 'cad-vision-gesture', event }));
}

/* ---- Vision -> CAD bridge (device-space coordinates only) ---- */

/** MediaPipe index fingertip — the pointing hand's cursor on the color wheel. */
const INDEX_TIP = 8;

/**
 * The hand currently holding the SELECT-mode drag on the selected mesh (a
 * quick tap-select leaves it null). The pointing hand driving the color
 * cursor is any hand other than this one.
 */
let selectHand: Handedness | null = null;

/**
 * The color wheel was dismissed by a completed timed hover lock: it stays
 * hidden until the pointing gesture ends or a fresh object is grabbed
 * (the interaction resets after a color is locked in).
 */
let colorPicked = false;

/** Timestamp of the previous frame processed by the selection UI (dwell clock). */
let lastSelectionStamp: number | null = null;

/**
 * Device space (the webcam frame; [-1, 1], +Y up) → viewport-local CSS px,
 * through the webcam-aspect interaction camera (see `CadScene.deviceToCanvas`).
 */
function deviceToViewport(x: number, y: number): { x: number; y: number } {
  return cadScene.deviceToCanvas(x, y, viewportElement.clientWidth, viewportElement.clientHeight);
}

// Mode-routed pinches (the classifier never emits pinches in VIEW mode):
// - SELECT: a quick pinch on a committed mesh selects it (the selection stays
//   after release, so its color can be changed); pinch and *hold* to move it
//   (ground plane or vertical lift, per the active constraint toggle) and
//   release to drop it. A pinch on empty ground deselects. The wheel is a
//   UI surface — a pinch over the disc never re-picks the 3D scene behind it
//   (color selection is a timed hover, not a pinch) — and a pinch on the
//   selection HUD's Delete button requests deletion through the
//   confirmation dialog, never an instant erase. Only the hand that grabbed
//   the mesh drags it — the other hand stays free to pick colors.
// Overlay buttons are pressed only by a pointing hand (index finger up), never
// by a pinch, so pinches always act on the scene.
// - CREATE: draw a footprint (pinch start/drag raycast onto the ground plane).
engine.onPinchStart((e) => {
  // The spatial confirmation dialog is modal: while it is open, a pinch may
  // only answer it — and the overlay handles pinch-on-target itself — so
  // the scene below stays frozen.
  if (overlay.confirmActive) return;
  // A pinch that closes over an overlay button (mode / constraint / Boolean
  // toggles, the trash bin, a dialog target) belongs to the UI, never the
  // scene: it neither picks / builds nor arms a drag.
  if (overlay.isUiAtDevice(e.position.x, e.position.y)) return;
  const point = deviceToViewport(e.position.x, e.position.y);
  if (engine.mode === 'select') {
    if (colorWheel.pickColorAt(point.x, point.y) !== null) return;
    if (selectionMenu.hitDelete(point.x, point.y)) {
      requestDeleteSelection();
      return;
    }
    const picked = builder.pickAt(e.position.x, e.position.y, e.timestamp);
    selectHand = picked ? e.hand : null;
    if (picked) colorPicked = false; // fresh grab re-arms the color wheel
  } else {
    builder.onPinchStart(e.position.x, e.position.y);
  }
});
engine.onPinchDrag((e) => {
  if (overlay.confirmActive) return; // modal: the scene is frozen while confirming
  if (engine.mode === 'select') {
    // Only the grabbing hand drags the mesh; hand 2 hovering (or confirming
    // a color on the wheel) never fights the drag.
    if (selectHand === null || e.hand === selectHand) {
      builder.dragTo(e.currentPos.x, e.currentPos.y, e.timestamp);
    }
  } else {
    builder.onPinchDrag(e.currentPos.x, e.currentPos.y, e.startPos.x, e.startPos.y);
  }
});
engine.onPinchEnd((e) => {
  if (engine.mode === 'select') {
    // Only the grabbing hand's release ends the drag — hand 2 releasing a
    // color-confirming pinch must not drop the held object.
    if (selectHand === null || e.hand === selectHand) {
      // A tap on the already-selected object deselects it; a first tap
      // selects (the selection persists); a hold-and-drag just drops it.
      builder.releasePick();
      selectHand = null;
    }
  } else {
    builder.onPinchEnd();
  }
});

// SELECT-mode secondary-hand rotation: while one hand holds a pinch on a
// mesh, the other hand's open palm tilts to spin the selection around the
// world Y axis (anchored deltas; a compass ring renders around the object
// in both the 3D viewport and the AR mirror while rotating).
engine.onSelectRotate((e) => builder.rotateSelection(e.deltaRotation));
engine.onSelectRotateEnd(() => builder.endRotateSelection());

// Two-hand build: pinch both hands and pull apart to size the base (spawned
// at the origin); release the upper pinch, then drag the lower one vertically
// to set the height.
engine.onExtrude((e) => builder.onExtrude(e));

// Releasing the last pinch (or a fist / zoom transition) commits the pending
// build as a solid mesh.
// Flat when both pinches let go together or the lower one let go first.
// A two-hand build shorter than MIN_BUILD_MS is a stray / misdetected hand
// pinching for an instant, not a deliberate build: discard it.
engine.on('extrude_end', (event) => {
  if (event.type !== 'extrude_end') return;
  if (event.durationMs < MIN_BUILD_MS) builder.cancel();
  else builder.commit({ flat: !event.heightSet });
});
engine.on('orbit_start', (event) => {
  if (event.type === 'orbit_start') builder.commit();
});
// A fist gesture ended: the next fist rotation chooses its axis afresh.
engine.on('orbit_end', () => builder.endFistRotation());
engine.on('zoom_end', () => builder.endFistRotation());
engine.on('zoom_start', (event) => {
  if (event.type === 'zoom_start') builder.commit();
});

// Fist move. With an object selected in EDIT mode the fist turns the object
// (up / down → world X axis, left / right → world Y axis; with two fists the
// faster one drives, right hand on a tie — see setObjectRotation below).
// Otherwise one fist orbits the view around the locked origin (the camera
// never pans) — the scene follows the hand, and rolling the wrist (twisting
// the fist like a doorknob) turns the scene with the twist.
engine.onOrbit((e) => {
  if (fistRotatesObject()) {
    builder.rotateSelectionBy(e.deltaX, e.deltaY);
    return;
  }
  cadScene.onOrbit({ deltaX: e.deltaX, deltaY: e.deltaY });
  if (e.deltaRoll !== 0) cadScene.onRotate({ deltaAngle: e.deltaRoll });
});

// Two fists: the steadier fist is the anchor (pivot). Moving the other fist
// away from it zooms in, toward it zooms out; circling it turns the scene.
engine.onZoom((e) => {
  cadScene.onZoom({ deltaScale: e.deltaScale });
  if (e.deltaAngle !== 0) cadScene.onRotate({ deltaAngle: e.deltaAngle });
});

/* ---- Misc wiring ---- */

engine.on('gesture', (event) => logGestureEvent(event as GestureSignalEvent));

engine.on('state_change', (event) => {
  const change = event as { to: string; reason: string };
  statusOutput.dataset.state = change.to;
  statusOutput.title = `${change.to} — ${change.reason}`;
  setStatus(`Mode: ${MODE_LABELS[engine.mode]} — State: ${change.to} (${change.reason})`);
});

// A mode switch aborts any in-flight build / selection drag so the new mode
// starts from a clean slate (committed meshes are untouched). Leaving SELECT
// also clears the selection, restoring the mesh's normal look (its highlight
// tint / outline go; a color picked on the wheel is kept).
engine.on('mode_change', (event) => {
  if (event.type !== 'mode_change') return;
  builder.cancel();
  builder.endDrag();
  if (event.to !== 'select') builder.deselect();
  selectHand = null; // the next frame hides the wheel outside SELECT mode
  colorPicked = false;
  lastSelectionStamp = null;
  statusOutput.dataset.mode = event.to;
  setStatus(`Mode: ${MODE_LABELS[event.to]} — State: ${engine.state}`);
});

// Debug overlay: re-render on every processed frame; the SELECT-mode
// selection UI rides along — the Delete HUD re-anchored beside the
// selection's live screen projection, the color wheel pinned mid-right of
// the reachable area, with the pointing index fingertip
// driving the timed hover color lock. The selection UI updates *before* the
// overlay renders so the thumbnail HUD (wheel outline + Delete mirror) is
// drawn from the same frame's geometry, never a frame stale. The stats bar
// below the camera view tracks the same frame (mode / state / FPS / hands).
/** Whether fists currently turn the selected object instead of the camera. */
function fistRotatesObject(): boolean {
  return engine.mode === 'select' && builder.selectedMesh !== null && !overlay.confirmActive;
}

engine.on('frame', (event) => {
  if (event.type !== 'frame') return;
  // Fists drive the selection (not the camera) while one is selected in EDIT.
  engine.setObjectRotation(fistRotatesObject());
  // Hand coords live in the webcam frame: keep the scene's interaction
  // camera at the webcam aspect (true AR proportions, aligned picking).
  if (event.video.height > 0) cadScene.setInteractionAspect(event.video.width / event.video.height);
  updateSelectionUi(event);
  metricsBar.update({
    mode: MODE_LABELS[event.mode],
    state: event.state,
    fps: event.fps,
    hands: event.hands.length,
    handedness: event.hands.map((hand) => hand.handedness).join(','),
  });
  overlay.render(event);
});

// Start in VIEW mode (pure camera navigation): nothing can be created or
// moved until the user picks another mode on the overlay's button bar.
engine.setMode('view');

/**
 * Delete flow (SELECT mode): every trigger — the selection HUD's Delete
 * button (mouse or pinch), or the Delete / Backspace keys — routes through
 * the in-vision spatial confirmation dialog; the mesh is only removed when
 * the dialog is answered with Confirm (a pointing hold on the target or an
 * OK-gesture hold, a mouse click or Enter).
 */
function requestDeleteSelection(): void {
  if (overlay.confirmActive) return;
  if (engine.mode !== 'select' || !builder.selectedMesh) return;
  overlay.openConfirm('delete-selection');
}

/**
 * Scene-clear flow: the in-vision trash bin (pointing dwell, fresh pinch or
 * mouse click) opens the same spatial confirmation dialog; the scene only
 * clears on Confirm. Nothing to ask when the scene is already empty.
 */
function requestSceneClear(): void {
  if (overlay.confirmActive) return;
  if (builder.committedMeshes.length === 0) {
    setStatus('Nothing to clear'); // no meshes: nothing to confirm
    return;
  }
  overlay.openConfirm('clear-scene');
}

/**
 * The spatial dialog was answered with Confirm: run the destructive intent
 * (the overlay has already closed itself). No browser popups, no DOM
 * modals — the answer arrived as a gesture (or mouse / keyboard).
 */
function confirmDestructive(intent: OverlayConfirmIntent): void {
  if (intent === 'delete-selection') {
    if (engine.mode !== 'select' || !builder.selectedMesh) return;
    builder.deleteSelectedMesh();
    selectHand = null; // the drag died with the object
    colorPicked = false;
    lastSelectionStamp = null;
    colorWheel.hide();
    selectionMenu.hide();
    setStatus('Object deleted');
    return;
  }
  builder.clear();
  selectHand = null;
  colorPicked = false;
  lastSelectionStamp = null;
  colorWheel.hide(); // no selection left to anchor the wheel
  selectionMenu.hide(); // …nor the Delete HUD
  setStatus('Scene cleared');
}

/** The spatial dialog was dismissed (open palm / hand away / Escape). */
function cancelDestructive(): void {
  setStatus('Canceled');
}

/**
 * SELECT-mode selection UI: the Delete HUD floats beside the selected
 * mesh's live screen projection (it tracks drags and camera orbits); the
 * color wheel is pinned at mid-height on the right edge of the reachable,
 * visible area (`showColorWheelAtSide`). A *pointing* index fingertip is the live color cursor — hues
 * repaint the mesh as it sweeps — and dwelling on one color slice for 1.2 s
 * locks the color in: it is applied and saved to the object, the wheel
 * disappears, and the interaction resets (the wheel re-arms once the
 * pointing gesture ends or a fresh object is grabbed). The same fingertip
 * dwelling on the Delete button for ~0.8 s requests deletion through the
 * confirmation dialog — finger-interactive alongside mouse, pinch and
 * keyboard. Hidden while the delete confirmation is open (modal), outside
 * SELECT mode, or when nothing is selected.
 */
function updateSelectionUi(frame: FrameEvent): void {
  const dtMs =
    lastSelectionStamp === null ? 0 : Math.max(0, frame.timestamp - lastSelectionStamp);
  lastSelectionStamp = frame.timestamp;
  const anchor =
    !overlay.confirmActive && engine.mode === 'select'
      ? builder.selectedProjection(viewportElement.clientWidth, viewportElement.clientHeight)
      : null;
  if (!anchor) {
    colorWheel.hide();
    selectionMenu.hide();
    return;
  }
  // Cursor: a pointing hand that is not holding the selection pinch.
  const cursor = frame.hands.find((hand) => hand.pointing && hand.handedness !== selectHand);
  // The color-picking interaction resets once the pointing gesture ends.
  if (colorPicked && cursor === undefined) colorPicked = false;
  const fingertip = cursor?.landmarks[INDEX_TIP];
  const point = fingertip ? deviceToViewport(fingertip.device.x, fingertip.device.y) : null;
  // Finger-interactive delete: a pointing fingertip dwelling on the Delete
  // button triggers the (confirmation-gated) delete request.
  if (selectionMenu.advanceDeleteDwell(point, dtMs)) {
    requestDeleteSelection();
    return;
  }
  if (colorPicked) {
    colorWheel.hide(); // dismissed after a completed timed hover lock
    selectionMenu.show(anchor.x, anchor.y);
    return;
  }
  showColorWheelAtSide();
  selectionMenu.show(anchor.x, anchor.y);
  // Enlarge the (camera-view) wheel while the fingertip hovers it, easing
  // in and out: picking follows the animated disc, and the center stays put
  // (room is reserved), so the hue under the finger doesn't jump.
  // Hysteresis comes for free — once enlarged, the finger has the bigger
  // disc to stay inside.
  const center = colorWheel.center;
  const hovering =
    point !== null &&
    center !== null &&
    Math.hypot(point.x - center.x, point.y - center.y) <= colorWheel.radius;
  // Smooth, interruptible grow / shrink (and pop-in on first show).
  colorWheel.animateZoomTo(hovering ? WHEEL_HOVER_ZOOM : 1, dtMs);
  if (!point) {
    colorWheel.advanceDwell(null, dtMs); // hover lost: the dwell clock resets
    return;
  }
  const hex = colorWheel.pickColorAt(point.x, point.y);
  const committed = colorWheel.advanceDwell(hex, dtMs);
  if (committed !== null) {
    // Timed hover lock complete: apply + save, dismiss the wheel, reset.
    builder.setSelectedColor(committed);
    colorPicked = true;
    colorWheel.hide();
    setStatus('Color locked');
    return;
  }
  if (hex) builder.setSelectedColor(hex); // live preview while sweeping
}

/**
 * Pin the color wheel at mid-height on the right edge of the part of the
 * webcam view that is both hand-reachable and visible in the camera view
 * (`overlay.visibleDeviceRect()`, mapped into the viewport). Mid-height
 * matters: a pointing fingertip low in the frame drags the wrist and palm
 * out of view and the hand tracker loses the hand — a bottom-corner wheel
 * could never be pointed at. The camera view draws the same wheel (small,
 * live-synced) at the same spot, and pointing at a color there picks it.
 */
function showColorWheelAtSide(): void {
  const width = viewportElement.clientWidth;
  const height = viewportElement.clientHeight;
  const visible = overlay.visibleDeviceRect();
  const edge = cadScene.deviceToCanvas(visible.maxX, (visible.minY + visible.maxY) / 2, width, height);
  // Room for the hover-enlarged disc, so the center never moves on zoom.
  const baseRadius = colorWheel.radius / colorWheel.zoomFactor;
  const inset = baseRadius * WHEEL_HOVER_ZOOM + 16;
  colorWheel.showAt(Math.min(width, edge.x) - inset, edge.y);
}

/**
 * One frame of the camera-thumbnail selection HUD: the *real* color wheel
 * and Delete button placements (viewport-local px) inverse-mapped into
 * device space via `CadScene.canvasToDevice` — the exact inverse of the
 * mapping the pointing fingertip travels — so the thumbnail's drawn
 * controls sit precisely where the finger has to point. Null hides the HUD
 * (no selection, outside SELECT mode, or while the delete confirmation
 * dialog freezes the scene).
 */
function selectionHudFrame(): OverlaySelectionHud | null {
  if (engine.mode !== 'select' || !builder.selectedMesh || overlay.confirmActive) return null;
  const width = viewportElement.clientWidth;
  const height = viewportElement.clientHeight;
  if (width <= 0 || height <= 0) return null;
  const toDevice = (px: number, py: number) => cadScene.canvasToDevice(px, py, width, height);

  let wheel: OverlayHudDisc | null = null;
  const wheelCenter = colorWheel.center;
  if (wheelCenter) {
    const center = toDevice(wheelCenter.x, wheelCenter.y);
    // The disc's px radius maps to different device extents on each axis.
    const rimX = toDevice(wheelCenter.x + colorWheel.radius, wheelCenter.y);
    const rimY = toDevice(wheelCenter.x, wheelCenter.y + colorWheel.radius);
    wheel = {
      x: center.x,
      y: center.y,
      radiusX: Math.abs(rimX.x - center.x),
      radiusY: Math.abs(rimY.y - center.y),
    };
  }

  let deleteButton: OverlayHudRect | null = null;
  const rect = selectionMenu.deleteRect;
  if (rect) {
    const topLeft = toDevice(rect.x, rect.y);
    const bottomRight = toDevice(rect.x + rect.width, rect.y + rect.height);
    deleteButton = {
      x: Math.min(topLeft.x, bottomRight.x),
      y: Math.min(topLeft.y, bottomRight.y),
      width: Math.abs(bottomRight.x - topLeft.x),
      height: Math.abs(bottomRight.y - topLeft.y),
    };
  }

  return {
    wheel,
    deleteButton,
    wheelProgress: colorWheel.dwellProgress,
    deleteProgress: selectionMenu.deleteDwellProgress,
  };
}

/** Start webcam + hand tracking (invoked from the toolbar). */
function startCamera(): void {
  toolbar.setCameraRunning(true); // disable Start while the request is pending
  setStatus('Requesting camera…');
  engine
    .start(videoElement)
    .then(() => {
      statusOutput.dataset.state = 'IDLE';
      setStatus('Tracking — state: IDLE');
    })
    .catch((err: unknown) => {
      toolbar.setCameraRunning(false);
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
      console.error('[cad-vision] failed to start:', err);
    });
}

/** Stop tracking and release the webcam (invoked from the toolbar). */
function stopCamera(): void {
  engine.stop();
  builder.cancel(); // drop any pending preview so the scene stays clean
  builder.deselect(); // no lingering selection highlight once tracking stops
  selectHand = null;
  colorPicked = false;
  lastSelectionStamp = null;
  colorWheel.hide(); // tracking stopped: the floating wheel must not linger
  selectionMenu.hide(); // …nor the selection HUD
  overlay.closeConfirm(); // …nor a half-asked destructive question
  toolbar.setCameraRunning(false);
  statusOutput.dataset.state = 'IDLE';
  statusOutput.title = '';
  setStatus('Stopped');
  const ctx = overlayCanvas.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Floating camera thumbnail: toggle between PiP and expanded states. A
// manual drag-resize leaves an inline width behind — clear it so the preset
// CSS sizes take over again.
thumbExpand.addEventListener('click', () => {
  visionThumb.style.width = '';
  const expanded = visionThumb.classList.toggle('expanded');
  thumbExpand.textContent = expanded ? '⤡' : '⤢';
  thumbExpand.title = expanded ? 'Collapse camera thumbnail' : 'Expand camera thumbnail';
  thumbExpand.setAttribute('aria-label', thumbExpand.title);
});

// Expose for experimentation from the browser console.
Object.assign(window, {
  cadVision: {
    engine,
    overlay,
    cadScene,
    builder,
    toolbar,
    colorWheel,
    selectionMenu,
    metricsBar,
    thumbDragger,
    thumbResizer,
  },
});

// Keyboard shortcuts: Delete / Backspace delete the selected object through
// the same spatial confirmation gate as the HUD button; while the
// confirmation is open, Enter (or Delete again) confirms and Escape cancels.
document.addEventListener('keydown', (event) => {
  if (event.repeat) return; // a held key never double-answers
  const key = event.key;
  if (overlay.confirmActive) {
    if (key === 'Escape') {
      event.preventDefault();
      overlay.closeConfirm(); // keyboard dismiss = the safe answer
      cancelDestructive();
    } else if (key === 'Enter' || key === 'Delete' || key === 'Backspace') {
      event.preventDefault();
      const intent = overlay.confirmIntent;
      if (intent !== null) confirmDestructive(intent);
    }
    return;
  }
  if ((key === 'Delete' || key === 'Backspace') && engine.mode === 'select') {
    if (!builder.selectedMesh) return;
    event.preventDefault();
    requestDeleteSelection();
  }
});

