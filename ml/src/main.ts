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
 * Hands own continuous motion (sizing, extruding, orbiting); the voice agent
 * owns the discrete verbs (build this, undo that, paint it red). Neither
 * knows the other exists: the vision layer emits device-space coordinates,
 * the voice layer speaks only in `CadCommand`s, and nothing in `src/cad` or
 * `src/ui` imports either of them.
 */

import { GestureEngine } from './vision/GestureEngine';
import { DebugOverlay } from './vision/DebugOverlay';
import type { GestureSignalEvent } from './vision/types';
import { CadScene } from './cad/CadScene';
import { CadBuilder } from './cad/CadBuilder';
import { Toolbar } from './ui/Toolbar';
import { VoiceHud } from './ui/VoiceHud';
import { VoiceAgent } from './voice/VoiceAgent';
import { describeScene } from '../shared/agentTools';
import type { CadCommand, SceneSummary } from '../shared/agentTools';

/** `?debug` exposes the module graph on `window` and logs gesture events. */
const debugMode = new URLSearchParams(window.location.search).has('debug');

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

const overlay = new DebugOverlay(canvas);

/* ---- CAD ---- */
const cadScene = new CadScene(viewport);
const builder = new CadBuilder(cadScene);

/** Tracked here rather than read off the engine so the agent and the toolbar
 *  always agree on whether the camera is live. */
let cameraRunning = false;

/* ---- UI ---- */
const voiceHud = new VoiceHud({ pill: voicePill, captions: voiceCaptions });

const toolbar = new Toolbar(toolbarRoot, {
  onClearScene: () => builder.clear(),
  onExportStl: () => builder.exportStl(),
  onCameraStart: () => startCamera(),
  onCameraStop: () => stopCamera(),
  onVoiceToggle: () => void toggleVoice(),
  onVoiceListen: () => voiceAgent.toggleListening(),
});

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

const voiceAgent = new VoiceAgent(
  {
    executor: { execute: executeCommand },
    getScene: readScene,
    // The agent sees the viewport, so it can judge sizes and free space.
    getSnapshot: () => cadScene.snapshotJpeg(),
  },
  {
    onState: (state) => {
      voiceHud.setState(state);
      toolbar.setVoiceState(state);
      toolbar.setManualListening(state === 'listening');
    },
    onPartial: (text) => voiceHud.showHeard(text, { partial: true }),
    onTranscript: (text) => voiceHud.showHeard(text),
    onReply: (text) => voiceHud.showReply(text),
    onError: (message) => voiceHud.showError(message),
    onPresageStatus: (status) => {
      // Any state other than `ready` means the face is not opening the mic, so
      // the user needs a button — "still connecting" strands them just as
      // badly as "unavailable".
      toolbar.setManualListenVisible(status !== 'ready');
    },
  }
);

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
  setStatus(change.to);
});

// Debug overlay: re-render on every processed frame.
engine.on('frame', (event) => {
  if (event.type === 'frame') overlay.render(event);
});

/** Start webcam + hand tracking (invoked from the toolbar or by voice). */
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
  toolbar.setCameraRunning(false);
  statusOutput.dataset.state = 'IDLE';
  statusOutput.title = '';
  setStatus('Idle');
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

// Release the microphone and the agent session on the way out.
window.addEventListener('pagehide', () => voiceAgent.stop());

if (debugMode) {
  Object.assign(window, {
    cadVision: { engine, overlay, cadScene, builder, toolbar, voiceAgent, voiceHud },
  });
}
