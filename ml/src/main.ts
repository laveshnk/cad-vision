/**
 * App orchestrator: gesture-driven CAD workbench.
 *
 * Strictly decoupled module wiring:
 *
 *   Vision (GestureEngine)  --typed events-->  App (this file)
 *        --> CAD (CadScene + CadBuilder)  +  UI (Toolbar)
 *
 * The vision layer emits device-space coordinates (x / y in [-1, 1], +Y up),
 * deltas and state events; the CAD layer consumes only those — nothing in
 * src/cad or src/ui imports from src/vision.
 *
 * Interaction modes (VIEW / SELECT / CREATE) are picked on the vision
 * overlay's button bar and gate the gesture routing below: pinches are inert
 * in VIEW, pick / drag / recolor meshes in SELECT (a floating color wheel
 * follows the selection; hand 2 sweeps it while hand 1 holds the pinch), and
 * build primitives in CREATE. Camera gestures (fist orbit / two-fist zoom)
 * stay live in every mode.
 */

import { GestureEngine } from './vision/GestureEngine';
import { DebugOverlay } from './vision/DebugOverlay';
import type { FrameEvent, GestureSignalEvent, Handedness } from './vision/types';
import { CadScene } from './cad/CadScene';
import { CadBuilder, type CadTool } from './cad/CadBuilder';
import { buildArSceneFrame } from './cad/ArMirror';
import { Toolbar } from './ui/Toolbar';
import { ColorWheel } from './ui/ColorWheel';

const video = document.querySelector<HTMLVideoElement>('#video');
const canvas = document.querySelector<HTMLCanvasElement>('#overlay');
const statusText = document.querySelector<HTMLElement>('#status');
const toolbarRoot = document.querySelector<HTMLElement>('#toolbar');
const viewport = document.querySelector<HTMLElement>('#viewport');
const visionThumb = document.querySelector<HTMLElement>('#vision-thumb');
const thumbExpand = document.querySelector<HTMLButtonElement>('#thumb-expand');

if (
  !video ||
  !canvas ||
  !statusText ||
  !toolbarRoot ||
  !viewport ||
  !visionThumb ||
  !thumbExpand
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
const overlay = new DebugOverlay<CadTool>(canvas, {
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
  // SELECT-mode AR mirror: the ground grid + every committed mesh are
  // projected through the shared 3D camera onto the vision canvas, turning
  // it into a translucent live spatial mirror of the 3D viewport — in
  // lockstep with pinch-driven drags (the engine emits pinch events before
  // the per-frame event, so ghost and viewport never diverge).
  arScene: (width, height) => buildArSceneFrame(cadScene, builder, width, height),
});

/* ---- UI ---- */
const toolbar = new Toolbar(toolbarRoot, {
  onClearScene: () => {
    builder.clear();
    colorWheel.hide(); // no selection left to anchor the wheel
  },
  onExportStl: () => builder.exportStl(),
  onCameraStart: () => startCamera(),
  onCameraStop: () => stopCamera(),
});

// Floating HSL color wheel (SELECT mode): pure UI mounted into the CAD
// viewport — the bridge below feeds it viewport-local pixels and applies
// the picked hex through CadBuilder.setSelectedColor.
const colorWheel = new ColorWheel(viewport);

function setStatus(text: string): void {
  statusOutput.textContent = text;
}

/** Forward gesture events to the console as JSON (downstream consumer demo). */
function logGestureEvent(event: GestureSignalEvent): void {
  console.log(JSON.stringify({ source: 'cad-vision-gesture', event }));
}

/* ---- Vision -> CAD bridge (device-space coordinates only) ---- */

/** MediaPipe index fingertip — hand 2's cursor on the color wheel. */
const INDEX_TIP = 8;

/**
 * The hand currently holding the SELECT-mode pinch on the selected mesh.
 * While it is set, the *other* hand's index fingertip drives the color
 * wheel, and that hand's pinch (confirming a hue) never steals the drag.
 */
let selectHand: Handedness | null = null;

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
//   release to drop it. A pinch on empty ground deselects. A pinch landing
//   on the floating color wheel repaints the selection with the hue beneath
//   it instead of re-picking the 3D scene behind the disc, and only the hand
//   that grabbed the mesh drags it — the other hand stays free to pick colors.
// Overlay buttons are pressed only by a pointing hand (index finger up), never
// by a pinch, so pinches always act on the scene.
// - CREATE: draw a footprint (pinch start/drag raycast onto the ground plane).
engine.onPinchStart((e) => {
  if (engine.mode === 'select') {
    const point = deviceToViewport(e.position.x, e.position.y);
    const wheelColor = colorWheel.pickColorAt(point.x, point.y);
    if (wheelColor) {
      builder.setSelectedColor(wheelColor);
      return;
    }
    selectHand = builder.pickAt(e.position.x, e.position.y, e.timestamp) ? e.hand : null;
  } else {
    builder.onPinchStart(e.position.x, e.position.y);
  }
});
engine.onPinchDrag((e) => {
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
      builder.endDrag();
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
engine.on('extrude_end', (event) => {
  if (event.type === 'extrude_end') builder.commit({ flat: !event.heightSet });
});
engine.on('orbit_start', (event) => {
  if (event.type === 'orbit_start') builder.commit();
});
engine.on('zoom_start', (event) => {
  if (event.type === 'zoom_start') builder.commit();
});

// Camera move: one fist orbits the view around the locked origin (the camera
// never pans) — the scene follows the hand, and rolling the wrist (twisting
// the fist like a doorknob) turns the scene with the twist.
engine.onOrbit((e) => {
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
  setStatus(`Mode: ${engine.mode.toUpperCase()} — State: ${change.to} (${change.reason})`);
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
  statusOutput.dataset.mode = event.to;
  setStatus(`Mode: ${event.to.toUpperCase()} — State: ${engine.state}`);
});

// Debug overlay: re-render on every processed frame; the SELECT-mode color
// wheel rides along — re-anchored beside the selection's live screen
// projection and repainting the mesh under the other hand's index fingertip.
engine.on('frame', (event) => {
  if (event.type !== 'frame') return;
  // Hand coords live in the webcam frame: keep the scene's interaction
  // camera at the webcam aspect (true AR proportions, aligned picking).
  if (event.video.height > 0) cadScene.setInteractionAspect(event.video.width / event.video.height);
  overlay.render(event);
  updateColorWheel(event);
});

// Start in VIEW mode (pure camera navigation): nothing can be created or
// moved until the user picks another mode on the overlay's button bar.
engine.setMode('view');

/**
 * SELECT-mode color wheel: anchored adjacent to the selected mesh's live
 * screen projection (it tracks drags and camera orbits). A *pointing* index
 * fingertip is the live color cursor — every hue it sweeps over repaints the
 * mesh instantly — whether the selection was made with a quick pinch and
 * released, or is still held by the other hand. Hidden outside SELECT mode
 * or when nothing is selected.
 */
function updateColorWheel(frame: FrameEvent): void {
  const anchor =
    engine.mode === 'select'
      ? builder.selectedProjection(viewportElement.clientWidth, viewportElement.clientHeight)
      : null;
  if (!anchor) {
    colorWheel.hide();
    return;
  }
  colorWheel.show(anchor.x, anchor.y);
  // Cursor: a pointing hand that is not holding the selection pinch.
  const cursor = frame.hands.find((hand) => hand.pointing && hand.handedness !== selectHand);
  const fingertip = cursor?.landmarks[INDEX_TIP];
  if (!fingertip) return;
  const point = deviceToViewport(fingertip.device.x, fingertip.device.y);
  const hex = colorWheel.pickColorAt(point.x, point.y);
  if (hex) builder.setSelectedColor(hex);
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
  colorWheel.hide(); // tracking stopped: the floating wheel must not linger
  toolbar.setCameraRunning(false);
  statusOutput.dataset.state = 'IDLE';
  statusOutput.title = '';
  setStatus('Stopped');
  const ctx = overlayCanvas.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Floating camera thumbnail: toggle between PiP and expanded states.
thumbExpand.addEventListener('click', () => {
  const expanded = visionThumb.classList.toggle('expanded');
  thumbExpand.textContent = expanded ? '⤡' : '⤢';
  thumbExpand.title = expanded ? 'Collapse camera thumbnail' : 'Expand camera thumbnail';
  thumbExpand.setAttribute('aria-label', thumbExpand.title);
});

// Expose for experimentation from the browser console.
Object.assign(window, {
  cadVision: { engine, overlay, cadScene, builder, toolbar, colorWheel },
});

