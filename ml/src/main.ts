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
 * in VIEW, pick / drag meshes in SELECT, and build primitives in CREATE.
 * Camera gestures (fist orbit / two-fist zoom) stay live in every mode.
 */

import { GestureEngine } from './vision/GestureEngine';
import { DebugOverlay } from './vision/DebugOverlay';
import type { GestureSignalEvent } from './vision/types';
import { CadScene } from './cad/CadScene';
import { CadBuilder, type CadTool } from './cad/CadBuilder';
import { Toolbar } from './ui/Toolbar';

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

const overlay = new DebugOverlay<CadTool>(canvas, {
  // Mode switcher on the vision overlay: the button bar (mouse click, finger
  // dwell or pinch) requests engine mode changes; the engine feeds the
  // active mode back through the per-frame event, which renders the button.
  onModeRequest: (mode) => engine.setMode(mode),
  // CREATE mode pops up a shape row under the mode bar; the picked shape is
  // the builder's tool for the next build (default: cube).
  shapes: [
    { id: 'box', label: 'CUBE' },
    { id: 'cuboid', label: 'CUBOID' },
    { id: 'cylinder', label: 'CYLINDER' },
    { id: 'sphere', label: 'SPHERE' },
  ],
  activeShape: builder.activeTool,
  onShapeRequest: (shape) => builder.setTool(shape),
});

/* ---- UI ---- */
const toolbar = new Toolbar(toolbarRoot, {
  onClearScene: () => builder.clear(),
  onExportStl: () => builder.exportStl(),
  onCameraStart: () => startCamera(),
  onCameraStop: () => stopCamera(),
});

function setStatus(text: string): void {
  statusOutput.textContent = text;
}

/** Forward gesture events to the console as JSON (downstream consumer demo). */
function logGestureEvent(event: GestureSignalEvent): void {
  console.log(JSON.stringify({ source: 'cad-vision-gesture', event }));
}

/* ---- Vision -> CAD bridge (device-space coordinates only) ---- */

// Mode-routed pinches (the classifier never emits pinches in VIEW mode):
// - SELECT: pick a committed mesh / drag it on the ground plane; a pinch on
//   empty ground deselects.
// - CREATE: draw a footprint (pinch start/drag raycast onto the ground plane).
engine.onPinchStart((e) => {
  if (engine.mode === 'select') builder.pickAt(e.position.x, e.position.y);
  else builder.onPinchStart(e.position.x, e.position.y);
});
engine.onPinchDrag((e) => {
  if (engine.mode === 'select') builder.dragTo(e.currentPos.x, e.currentPos.y);
  else builder.onPinchDrag(e.currentPos.x, e.currentPos.y, e.startPos.x, e.startPos.y);
});
engine.onPinchEnd(() => {
  if (engine.mode === 'select') builder.endDrag();
  else builder.onPinchEnd();
});

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
// starts from a clean slate (committed meshes are untouched).
engine.on('mode_change', (event) => {
  if (event.type !== 'mode_change') return;
  builder.cancel();
  builder.endDrag();
  statusOutput.dataset.mode = event.to;
  setStatus(`Mode: ${event.to.toUpperCase()} — State: ${engine.state}`);
});

// Debug overlay: re-render on every processed frame.
engine.on('frame', (event) => {
  if (event.type === 'frame') overlay.render(event);
});

// Start in VIEW mode (pure camera navigation): nothing can be created or
// moved until the user picks another mode on the overlay's button bar.
engine.setMode('view');

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
Object.assign(window, { cadVision: { engine, overlay, cadScene, builder, toolbar } });

