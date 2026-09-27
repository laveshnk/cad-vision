/**
 * App orchestrator: a hands-and-voice 3D playground.
 *
 * Strictly decoupled module wiring — this file is the only place the layers
 * meet:
 *
 *   Vision (GestureEngine) --typed events--> App --> CAD (CadScene + CadBuilder)
 *   Voice  (VoiceAgent)    --validated CadCommands--> App --> CAD
 *                          <--SceneSummary + viewport snapshot-- App
 *
 * The vision layer emits device-space coordinates (x / y in [-1, 1], +Y up),
 * deltas and state events; the CAD layer consumes only those — nothing in
 * src/cad or src/ui imports from src/vision.
 *
 * Interaction modes (VIEW / SELECT / CREATE) are picked on the vision
 * overlay's button bar and gate the gesture routing below: pinches are inert
 * in VIEW, pick / drag / recolor / delete meshes in SELECT (a floating color
 * wheel + Delete HUD follow the selection; a pointing fingertip sweeps the
 * wheel and a 1.2 s hover locks the color; the Delete button answers to
 * mouse, pinch, keyboard and a pointing dwell — deletion is
 * confirmation-gated), and build primitives in CREATE. Camera gestures
 * (fist orbit / two-fist zoom) stay live in every mode.
 */

import { GestureEngine } from './vision/GestureEngine';
import { DebugOverlay } from './vision/DebugOverlay';
import type {
  FrameEvent,
  GestureSignalEvent,
  Handedness,
} from './vision/types';
import type {
  OverlayHudDisc,
  OverlayHudRect,
  OverlaySelectionHud,
} from './vision/DebugOverlay';
import { CadScene } from './cad/CadScene';
import { CadBuilder, type CadTool } from './cad/CadBuilder';
import { buildArSceneFrame } from './cad/ArMirror';
import { Toolbar } from './ui/Toolbar';
import { ColorWheel } from './ui/ColorWheel';
import { SelectionMenu } from './ui/SelectionMenu';
import { ConfirmDialog } from './ui/ConfirmDialog';
import { ThumbResizer } from './ui/ThumbResizer';

const video = document.querySelector<HTMLVideoElement>('#video');
const canvas = document.querySelector<HTMLCanvasElement>('#overlay');
const statusText = document.querySelector<HTMLElement>('#status');
const toolbarRoot = document.querySelector<HTMLElement>('#toolbar');
const viewport = document.querySelector<HTMLElement>('#viewport');
const visionThumb = document.querySelector<HTMLElement>('#vision-thumb');
const thumbExpand = document.querySelector<HTMLButtonElement>('#thumb-expand');
const voicePill = document.querySelector<HTMLElement>('#voice-pill');
const voiceCaptions = document.querySelector<HTMLElement>('#voice-captions');

if (
  !video ||
  !canvas ||
  !statusText ||
  !toolbarRoot ||
  !viewport ||
  !visionThumb ||
  !thumbExpand ||
  !voicePill ||
  !voiceCaptions
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
  // SELECT-mode selection HUD: the color wheel's disc outline and the Delete
  // button are mirrored into the camera thumbnail (device space, so the
  // drawn controls sit exactly where a fingertip must point), with both
  // dwell progress bars as live feedback.
  selectionHud: () => selectionHudFrame(),
});

/* ---- UI ---- */
const voiceHud = new VoiceHud({ pill: voicePill, captions: voiceCaptions });

const toolbar = new Toolbar(toolbarRoot, {
  onClearScene: () => {
    builder.clear();
    colorWheel.hide(); // no selection left to anchor the wheel
    selectionMenu.hide(); // …nor the Delete HUD
  },
  onExportStl: () => builder.exportStl(),
  onCameraStart: () => startCamera(),
  onCameraStop: () => stopCamera(),
  onVoiceToggle: () => void toggleVoice(),
  onVoiceListen: () => voiceAgent.toggleListening(),
});

// Floating HSL color wheel (SELECT mode): pure UI mounted into the CAD
// viewport — the bridge below feeds it viewport-local pixels and applies
// the picked hex through CadBuilder.setSelectedColor.
const colorWheel = new ColorWheel(viewport);

// Floating selection HUD (SELECT mode): a Delete action anchored below the
// selection's screen projection. Deletion is destructive, so every trigger
// routes through the confirmation dialog below before the mesh is removed.
const selectionMenu = new SelectionMenu(viewport, {
  onDeleteRequest: () => requestDeleteSelection(),
});

// Modal safety confirmation ("Are you sure you want to delete this object?
// [Confirm] [Cancel]") — nothing is deleted until it is answered.
const confirmDialog = new ConfirmDialog(viewport, {
  onConfirm: () => confirmDeleteSelection(),
  onCancel: () => cancelDeleteSelection(),
});

// Mouse-drag resizing of the floating camera thumbnail: a corner grip inside
// the video stage scales the card (the overlay re-measures its canvas every
// frame, so nothing else needs a resize listener).
const thumbResizer = new ThumbResizer(visionThumb);

function setStatus(text: string): void {
  statusOutput.textContent = text;
}

/* ---- Voice -> CAD bridge (validated commands only) ---- */

/** Current scene, assembled fresh for every agent turn. */
function readScene(): SceneSummary {
  return { ...builder.describe(), cameraRunning };
}

/**
 * Carry out a command the agent asked for. The returned string is fed back to
 * the model as the tool result, so it is phrased the way the agent should
 * repeat it — plain and spoken, not a status code.
 */
function executeCommand(command: CadCommand): string {
  switch (command.name) {
    case 'set_shape':
      builder.setTool(command.shape);
      return `Next hand-built shape is a ${command.shape}.`;
    case 'add_shape': {
      const object = builder.addPrimitive(command);
      return `Built a ${object.shape} ${object.width} wide and ${object.height} tall at ${object.x}, ${object.z}.`;
    }
    case 'remove_last': {
      const removed = builder.removeLast();
      return removed ? `Removed the ${removed.shape}.` : 'There was nothing left to remove.';
    }
    case 'clear_scene':
      builder.clear();
      return 'Cleared the scene.';
    case 'set_color': {
      const count = builder.setColor(command.color, command.target);
      if (count === 0) return 'There is nothing to paint yet.';
      return `Painted ${command.target === 'all' ? `all ${count} objects` : 'it'} ${command.color}.`;
    }
    case 'describe_scene':
      return describeScene(readScene());
    case 'export_for_printing':
      return builder.exportStl()
        ? 'Downloaded model.stl — ready for a slicer or a printer.'
        : 'Nothing to export yet; the scene is empty.';
    case 'start_camera':
      if (cameraRunning) return 'The camera is already on.';
      startCamera();
      return 'Camera on — hands are live.';
    case 'stop_camera':
      if (!cameraRunning) return 'The camera is already off.';
      stopCamera();
      return 'Camera off.';
  }
}

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
  const point = deviceToViewport(e.position.x, e.position.y);
  if (confirmDialog.isOpen) {
    // Modal: while the delete confirmation is open, a pinch may only answer
    // it — the scene below stays frozen.
    if (confirmDialog.hitConfirm(point.x, point.y)) confirmDeleteSelection();
    else if (confirmDialog.hitCancel(point.x, point.y)) cancelDeleteSelection();
    return;
  }
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
  if (confirmDialog.isOpen) return; // modal: the scene is frozen while confirming
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

/** Toggle the voice agent, greeting the user the first time it comes up. */
async function toggleVoice(): Promise<void> {
  if (voiceAgent.isRunning) {
    voiceAgent.stop();
    return;
  }
  // Presage watches the same <video> the tracker uses, so hands-free mouth
  // detection is only possible once the camera is live.
  await voiceAgent.start(cameraRunning ? videoElement : null);
  if (voiceAgent.isRunning) {
    void voiceAgent.say('Voice is on. Tell me what you want to build.');
  }
}

/* ---- Vision -> CAD bridge (device-space coordinates only) ---- */

// Two-hand build: pinch both hands and pull apart to size the shape (spawned
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

if (debugMode) {
  engine.on('gesture', (event) => {
    console.log(JSON.stringify({ source: 'cad-vision-gesture', event: event as GestureSignalEvent }));
  });
}

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
  colorPicked = false;
  lastSelectionStamp = null;
  statusOutput.dataset.mode = event.to;
  setStatus(`Mode: ${event.to.toUpperCase()} — State: ${engine.state}`);
});

// Debug overlay: re-render on every processed frame; the SELECT-mode
// selection UI rides along — color wheel + Delete HUD re-anchored beside the
// selection's live screen projection, with the pointing index fingertip
// driving the timed hover color lock. The selection UI updates *before* the
// overlay renders so the thumbnail HUD (wheel outline + Delete mirror) is
// drawn from the same frame's geometry, never a frame stale.
engine.on('frame', (event) => {
  if (event.type !== 'frame') return;
  // Hand coords live in the webcam frame: keep the scene's interaction
  // camera at the webcam aspect (true AR proportions, aligned picking).
  if (event.video.height > 0) cadScene.setInteractionAspect(event.video.width / event.video.height);
  updateSelectionUi(event);
  overlay.render(event);
});

// Start in VIEW mode (pure camera navigation): nothing can be created or
// moved until the user picks another mode on the overlay's button bar.
engine.setMode('view');

/**
 * Delete flow (SELECT mode): every trigger — the selection HUD's Delete
 * button (mouse or pinch), or the Delete / Backspace keys — routes through
 * this confirmation gate; the mesh is only removed on Confirm.
 */
function requestDeleteSelection(): void {
  if (confirmDialog.isOpen) return;
  if (engine.mode !== 'select' || !builder.selectedMesh) return;
  confirmDialog.open();
}

/** Confirmation answered with Confirm: remove the selected object. */
function confirmDeleteSelection(): void {
  if (!confirmDialog.isOpen) return;
  confirmDialog.close();
  builder.deleteSelectedMesh();
  selectHand = null; // the drag died with the object
  colorPicked = false;
  lastSelectionStamp = null;
  colorWheel.hide();
  selectionMenu.hide();
  setStatus('Object deleted');
}

/** Confirmation answered with Cancel: the object stays, nothing changes. */
function cancelDeleteSelection(): void {
  if (!confirmDialog.isOpen) return;
  confirmDialog.close();
}

/**
 * SELECT-mode selection UI: the color wheel + Delete HUD float beside the
 * selected mesh's live screen projection (they track drags and camera
 * orbits). A *pointing* index fingertip is the live color cursor — hues
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
    !confirmDialog.isOpen && engine.mode === 'select'
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
  colorWheel.show(anchor.x, anchor.y);
  selectionMenu.show(anchor.x, anchor.y);
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
 * One frame of the camera-thumbnail selection HUD: the *real* color wheel
 * and Delete button placements (viewport-local px) inverse-mapped into
 * device space via `CadScene.canvasToDevice` — the exact inverse of the
 * mapping the pointing fingertip travels — so the thumbnail's drawn
 * controls sit precisely where the finger has to point. Null hides the HUD
 * (no selection, outside SELECT mode, or while the delete confirmation
 * dialog freezes the scene).
 */
function selectionHudFrame(): OverlaySelectionHud | null {
  if (engine.mode !== 'select' || !builder.selectedMesh || confirmDialog.isOpen) return null;
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
  cameraRunning = true;
  toolbar.setCameraRunning(true); // disable Start while the request is pending
  setStatus('Starting…');
  engine
    .start(videoElement)
    .then(() => {
      statusOutput.dataset.state = 'IDLE';
      setStatus('IDLE');
      // Hands-free mouth detection needs this feed; if voice is already on it
      // takes over from the manual button here.
      voiceAgent.attachVideo(videoElement);
    })
    .catch((err: unknown) => {
      cameraRunning = false;
      toolbar.setCameraRunning(false);
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
      console.error('[cad-vision] failed to start:', err);
    });
}

/** Stop tracking and release the webcam (invoked from the toolbar or by voice). */
function stopCamera(): void {
  cameraRunning = false;
  engine.stop();
  voiceAgent.detachVideo(); // back to the manual trigger while the camera is off
  builder.cancel(); // drop any pending preview so the scene stays clean
  builder.deselect(); // no lingering selection highlight once tracking stops
  selectHand = null;
  colorPicked = false;
  lastSelectionStamp = null;
  colorWheel.hide(); // tracking stopped: the floating wheel must not linger
  selectionMenu.hide(); // …nor the selection HUD
  toolbar.setCameraRunning(false);
  statusOutput.dataset.state = 'IDLE';
  statusOutput.title = '';
  setStatus('Idle');
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
  cadVision: { engine, overlay, cadScene, builder, toolbar, colorWheel, selectionMenu, confirmDialog, thumbResizer },
});

// Keyboard shortcuts: Delete / Backspace delete the selected object through
// the same confirmation gate as the HUD button; while the confirmation is
// open, Enter (or Delete again) confirms and Escape cancels.
document.addEventListener('keydown', (event) => {
  if (event.repeat) return; // a held key never double-answers
  const key = event.key;
  if (confirmDialog.isOpen) {
    if (key === 'Escape') {
      event.preventDefault();
      cancelDeleteSelection();
    } else if (key === 'Enter' || key === 'Delete' || key === 'Backspace') {
      event.preventDefault();
      confirmDeleteSelection();
    }
    return;
  }
  if ((key === 'Delete' || key === 'Backspace') && engine.mode === 'select') {
    if (!builder.selectedMesh) return;
    event.preventDefault();
    requestDeleteSelection();
  }
});

if (debugMode) {
  Object.assign(window, {
    cadVision: { engine, overlay, cadScene, builder, toolbar, voiceAgent, voiceHud },
  });
}
