# CAD Vision — a hands-and-voice 3D playground (`ml/`)

Self-contained TypeScript application that turns a webcam into a place to
build things. Your **hands** do the continuous work: a MediaPipe gesture engine
emits normalized events (pinch, extrude, camera move, zoom) that a decoupled
Three.js module turns into solids. Your **voice** does the discrete work: an
agent hears what you ask for, builds it, and answers out loud — and it knows
you started talking because the camera watched your mouth move, so there is no
button to press. The camera renders as a floating video-call-style thumbnail
(landmarks, skeleton, HUD overlay) over the full-bleed 3D viewport.

Anything you build exports as `model.stl`, so it can go to a slicer or a
printer — but that is a choice at the end, not the point.

## Quick start

```bash
cd ml
npm install        # also copies the MediaPipe WASM runtime + model (see below)
npm run dev        # http://localhost:5173 — click "Start camera"
```

`npm install` runs `scripts/fetch-assets.mjs`, which copies the MediaPipe WASM
runtime from `node_modules` into `public/mediapipe/wasm` and downloads the
`hand_landmarker.task` model (float16) into `public/models/`. If asset setup
fails (e.g. offline), the app falls back to CDN URLs at runtime. You can
re-run it any time with `npm run setup:assets`.

Camera access requires a secure context: `localhost` works for development;
serving over the network needs HTTPS.

### Voice (optional)

Hands work with `npm run dev` alone. Voice needs the small Node server that
holds the API keys:

```bash
cp .env.example .env   # fill in Gemini + ElevenLabs (+ Presage) keys
npm run dev:server     # http://localhost:8787 — Vite proxies /api and /ws to it
```

Then click **Voice** in the toolbar. Each service degrades on its own: without
`PRESAGE_API_KEY` a manual **Talk** button appears instead of hands-free mouth
detection, and without the ElevenLabs or Gemini keys the voice button reports
the failure and the hand-building side keeps working.

> ElevenLabs free accounts can only use *premade* voices. A Voice Library ID in
> `ELEVENLABS_VOICE_ID` returns `402 paid_plan_required`; the default in
> `.env.example` works on any plan.

## Architecture

```
shared/
└── agentTools.ts              # the voice contract: tool schemas, validators, SceneSummary
server/                        # Node 24 (type-stripped .ts); holds every API key
├── index.ts                   # http + ws: /api/voice/*, /api/agent/*, /ws/presage
├── agent.ts                   # Gemini turn loop + per-session history
├── elevenlabs.ts              # Scribe token minting + TTS streaming
└── presage.ts                 # SmartSpectra frames → "is the mouth moving"
src/
├── main.ts                    # app orchestrator: vision + voice events → CAD + UI
├── styles.css                 # light theme; full-bleed viewport, camera thumbnail, glass toolbar
├── cad/
│   ├── CadScene.ts            # Three.js viewport: camera rig, lights, grid, orbit, snapshots
│   └── CadBuilder.ts          # hand- and voice-built primitives + STL export
├── ui/
│   ├── Toolbar.ts             # camera + voice toggles, Clear scene, Export for printing
│   └── VoiceHud.ts            # voice state badge + caption strip
├── voice/
│   ├── PresageFrames.ts       # streams webcam stills to the server over a WebSocket
│   ├── TalkGate.ts            # hysteresis: when to open/close the mic, and barge-in
│   ├── Transcriber.ts         # ElevenLabs Scribe realtime connection
│   ├── Speaker.ts             # interruptible playback of the spoken reply
│   ├── VoiceAgent.ts          # facade: gate → transcript → agent → tools → speech
│   └── index.ts               # public API barrel
└── vision/
    ├── types.ts               # shared types + event payloads
    ├── coordinates.ts         # device-space mapping, vec3 math
    ├── filters.ts             # EMA / One-Euro, LandmarkSmoother, HandSmootherBank
    ├── HandTracker.ts         # webcam + MediaPipe HandLandmarker
    ├── HandednessStabilizer.ts # temporal + geometric Left/Right label stabilization
    ├── handShape.ts           # fold / thumb-tuck ratios (real fist) + wrist roll
    ├── PathStraightener.ts    # straight-segment filter for camera moves
    ├── GestureClassifier.ts   # pinch/fist/orbit/zoom detection + finite state machine
    ├── GestureEngine.ts       # facade: pipeline + event emitter
    ├── DebugOverlay.ts        # 2D canvas renderer (landmarks, skeleton, HUD)
    └── index.ts               # public API barrel
```

Per-frame pipeline:

```
HandTracker (raw MediaPipe hands)
  → HandednessStabilizer  Left/Right label votes + chirality, flicker/collision-proof
  → HandSmootherBank     EMA over all 21 landmarks, per-hand identity
  → coordinates          normalized / pixel / device spaces
  → GestureClassifier    FSM: IDLE | DRAWING_BASE | EXTRUDING | ORBITING | ZOOMING
  → listeners            typed events + per-frame debug event
```

Voice pipeline:

```
PresageFrames   webcam stills → the server → SmartSpectra → is the mouth moving
  → TalkGate      hysteresis: open after 150 ms of talking, close after 700 ms
                  of silence, barge-in after 300 ms of talking over playback
  → Transcriber   ElevenLabs Scribe realtime; emits one sealed utterance
  → /api/agent/turn  Gemini, with the scene summary and a viewport snapshot
  → ToolExecutor  validated CadCommands run against CadBuilder
  → Speaker       ElevenLabs TTS, interruptible
```

The gate is driven by the **face**, not the microphone, so the agent's own
speech can never re-trigger it — and talking over the reply cuts it off.

App wiring (strictly decoupled — `src/cad` and `src/ui` import nothing from
`src/vision` or `src/voice`; only device-space coordinates, deltas, state
events and validated `CadCommand`s cross the boundaries):

```
GestureEngine  --typed events-------->  main.ts (orchestrator)
VoiceAgent     --validated commands-->
               <--SceneSummary + viewport snapshot--
  ├── CadScene / CadBuilder      primitives, extrusion, orbit, STL export
  └── Toolbar / VoiceHud         camera + voice toggles, state, captions
```

API keys live only in `server/`. The browser gets a short-lived, single-use
Scribe token from `/api/voice/token`; nothing secret is ever bundled.

### Running mode note

The `@mediapipe/tasks-vision` web package implements live-stream processing for
`HandLandmarker` via running mode `VIDEO` + a `requestAnimationFrame` loop
calling `detectForVideo()` with monotonically increasing timestamps (the JS
package exposes the async `LIVE_STREAM` callback overloads only for other
tasks). This is the official MediaPipe web pattern for live webcam hand
tracking and is functionally equivalent: every camera frame is processed as it
arrives.

## Coordinate system

Landmarks are exposed in three spaces (`Landmark`):

| Space        | Range       | Notes                                         |
| ------------ | ----------- | --------------------------------------------- |
| `normalized` | `[0, 1]`    | MediaPipe image space, un-mirrored             |
| `pixel`      | video px    | raw pixel coordinates within the video frame   |
| `device`     | `[-1, 1]`   | X mirrored to the on-screen selfie view, +Y up |

Mapping: `x = (1 - raw_x) * 2 - 1`, `y = -(raw_y * 2 - 1)`. Raw pixel
coordinates are retained alongside the normalized ones. Handedness labels are
swapped by default (`swapHandedness: true`) so they match the user's physical
hands on the mirrored preview.

## Gestures

| Gesture | Detection | Events |
| ------- | --------- | ------ |
| **Pinch / draw** | 3D Euclidean distance between landmarks 4 (thumb tip) and 8 (index tip); trigger `< 0.045`, hysteresis release `> 0.065` (both configurable) | `pinch_start`, `pinch_drag` (position, delta in device space), `pinch_end` |
| **Extrude (dual-hand)** | both hands pinch; pull distance `D` between pinch centers `center = (P_thumb + P_index) / 2` | `extrude_start`, `extrude` (`distance`, `scaleFactor = D / D₀`, `deltaDistance`, `spanX` / `spanY` = horizontal / vertical pinch gap in aspect-corrected units of video width), `extrude_end` (`heightSet`: `false` when both pinches release together or the lower pinch releases first — that upper pinch is then consumed until released) |
| **Extrude (single-hand)** | while EXTRUDING with one pinch: vertical drag of landmark 8 | `extrude` (`deltaHeight`, `cumulativeHeight`, +Y up) |
| **Camera move (one fist)** | a *real* closed fist, not just curled fingers (`handShape.ts`): all 4 fingertips folded into the palm — `dist(tip, wrist) < 0.9 · dist(MCP, wrist)`, which rejects claw / hook curls and half-curls — and the thumb tucked over the fingers (thumb tip within 0.75 × palm size of an index / middle / ring knuckle, rejecting a thumbs-up); all ratios, so size / distance / rotation invariant. Needs 3 consecutive frames (app default); once held, thresholds relax by `fistHoldSlack` and one finger may loosen. Only engaged from IDLE with no active pinch, and a pinch never engages from a fist. Palm-center motion is reported as deltas, **straightened** by `PathStraightener` so the camera travels in straight segments instead of copying hand wiggle (see below); the app orbits the camera around the locked origin so the scene follows the fist (fist right → camera swings left, fist up → camera swings lower). **Wrist roll** — twisting the fist like a doorknob, measured as the rotation of the knuckle line (index MCP → pinky MCP) around the wrist → middle-MCP axis — turns the scene around the vertical axis with the twist; reported only after the twist exceeds `rollEngageAngle` (0.15 rad ≈ 9°) per fist, per-frame jitter under `rollDeadzone` dropped | `orbit_start`, `orbit` (`deltaX`, `deltaY` device units, `deltaRoll` radians), `orbit_end` |
| **Zoom / turn (two fists)** | both hands closed fists (same test). The steadier fist (lower smoothed palm speed; switches only when the other is below `zoomAnchorSwitchRatio` 0.5× its speed) is the **anchor** — the pivot — and only the other fist's motion relative to it counts, so the anchor's own jitter is ignored. Distance `D` from the anchor (aspect-corrected palm centers): moving fist farther away → zoom in, closer → zoom out. Circling the anchor → `deltaAngle` (counter-clockwise on screen = +) turns the scene, after `zoomTurnEngageAngle` (0.12 rad ≈ 7°) of sweep so a straight pull doesn't rotate. Engaged from IDLE, or upgraded from a one-fist move when the second fist closes; ends on a pinch, a lost hand, or one hand open longer than `orbitOpenPalmGraceFrames` (then falls back to a one-fist move if the other fist is still closed) | `zoom_start`, `zoom` (`anchor`, `distance`, `scaleFactor = D / D₀`, `deltaScale = D / D_prev`, `deltaAngle`), `zoom_end` |

**Straight camera paths.** Building gestures (pinch, extrude) are live and
unfiltered, but a camera that copies every hand tremor feels unsteady and can
cause motion sickness. One-fist camera moves therefore go through
`PathStraightener` (`classifier.cameraPath`): nothing moves until the fist
travels `startDistance` (0.02); the segment's heading is the *average*
displacement over its first `settleDistance` (0.08), so wiggle cancels out;
after that only progress along the line is passed on (moving back along the
same line stays on it), and sideways drift beyond `cornerDeviation` (0.05)
starts a new straight segment at the corner. Along-line tremor under
`deadband` (0.004) is held. A shaky A → B → C therefore plays back as the
straight segments A → B and B → C. `CadScene` then eases the camera toward
its target with gentle damping (`damping` 5). Set `cameraPath: null` for raw
deltas.

Finite state machine: `IDLE ↔ DRAWING_BASE ↔ EXTRUDING ↔ ORBITING ↔ ZOOMING`.
Pinch always wins over fist gestures; a lost hand finalizes its gesture after a
small grace window (synthetic `pinch_end` / `orbit_end` / `zoom_end`) so
states never get stuck.

Smoothing: EMA over all 21 landmarks (`alpha = 0.35` by default, or the
One-Euro filter via `strategy: 'one-euro'`). The `HandSmootherBank` keeps one
filter history per hand identity, associated by minimal wrist travel so
swapped MediaPipe handedness labels cannot cross-contaminate histories.

Handedness stabilization: MediaPipe re-classifies Left/Right every frame and
can flicker at rotational ambiguity, hand both hands the same label, or seed a
mislabel that sticks. The `HandednessStabilizer` (between tracker and
smoother) matches hands to tracks by wrist continuity and decides labels via a
rolling vote — the MediaPipe report weighted by score, plus a geometric
chirality vote using the rotation-invariant triple product
`(indexMCP − wrist) × (pinkyMCP − wrist) · (thumbMCP − wrist)` (self-calibrated
from high-confidence reports). Label flips need ~3 consecutive contrary frames
(hysteresis), two visible hands can never share a label (mutual exclusivity),
and hands absent > 10 frames re-seed from fresh evidence. Configurable via
`GestureEngine`'s `handedness` options (`voteWindow`, `flipFrames`,
`highConfidence`, `geometricWeight`, `reentryFrames`, …).

## CAD application workflow

The full-bleed viewport is a Three.js scene (floor grid, fog, damped orbit
camera, starting straight on so the grid is square to the screen) driven entirely by the gesture events above, with the camera rendered
as a floating thumbnail (top-left, click ⤢ to expand/collapse) over it:

1. **Two-hand build** — pinch with both hands: a translucent wireframe
   preview of the selected tool spawns centered on the origin `(0, 0, 0)`,
   its base sized by the two pinches (`baseSizeScale` world units per unit
   of video width):
   - **Box** — square, side = straight-line pinch gap;
   - **Cuboid** — rectangle, horizontal gap → width, vertical gap → depth;
   - **Cylinder** — circle, diameter = pinch gap;
   - **Sphere** — diameter = pinch gap.
2. **Set the height** — relax the **upper** pinch; the base freezes and the
   still-pinched lower hand's vertical drag sets the height (spheres grow
   their radius instead). Re-pinch the other hand to resize the base again.
   Releasing the last pinch commits a solid matte mesh with crisp
   `EdgesGeometry` outlines (a fist / zoom also commits). Releasing the
   **lower** pinch first, or both at once, commits a **flat** plate
   (`flatHeight`, spheres unaffected).
3. **Orbit the camera** — make a fist and move it: the camera orbits the
   world origin and the scene follows your hand — fist right swings the
   camera left, fist up swings it lower — with damping (`CadScene.onOrbit`);
   open palm stops. The camera focus is **locked to `(0, 0, 0)`**: the view
   only rotates and zooms, it never pans or translates (`enablePan` is
   permanently `false`).
   **Rotate** by rolling your wrist while holding the fist (twist it like a
   doorknob): the scene turns around the vertical axis with your twist
   (`CadScene.onRotate`, `rotateSpeed` 1.5×). Open and re-close the fist to
   ratchet further.
4. **Zoom / turn** — make fists with both hands and hold one still: it
   becomes the anchor (ringed in the camera thumbnail). Move the other fist
   away from it to zoom in, toward it to zoom out (`CadScene.onZoom`, clamped
   between `minDistance` and `maxDistance`), or circle it around the anchor
   to turn the scene (`CadScene.onRotate`).

Toolbar (mouse or programmatic): a **Start camera / Stop** toggle (webcam +
tracking lifecycle), a **Voice** toggle, **Clear scene**, and **Export for
printing** (binary `model.stl` download via
`three/examples/jsm/exporters/STLExporter`). New builds use the box
primitive; cylinder / sphere remain selectable by voice or via
`CadBuilder.setTool()`.

## Talking to it

Turn on **Voice** and start talking — the badge on the camera thumbnail turns
green when the mic is actually open, and captions along the bottom of the
viewport show what was heard and what the agent said back. Interrupt the reply
by talking over it.

The agent can call these, and nothing else (`shared/agentTools.ts`):

| Tool | Say something like |
| ---- | ------------------ |
| `add_shape` | "give me a tall red tube", "put a small ball next to it" |
| `set_shape` | "switch to cylinder" (so your *hands* build cylinders next) |
| `set_color` | "make it blue", "paint everything white" |
| `remove_last` | "undo that", "nope, take it back" |
| `clear_scene` | "start over" |
| `describe_scene` | "what have I got?" |
| `export_for_printing` | "save this for printing" |
| `start_camera` / `stop_camera` | "turn the camera on" |

Every call is re-validated in the browser before it reaches `CadBuilder`:
sizes and positions are clamped rather than rejected ("make it huge" should
still build something), but unknown tools, shapes and colors are refused and
reported back so the agent corrects itself out loud. Spoken synonyms map to
the four primitives — cube/block → box, tube → cylinder, ball/orb → sphere.

## Event API

```ts
import { GestureEngine } from './vision'; // or './vision/GestureEngine'

const engine = new GestureEngine({
  smoothing: { strategy: 'ema', alpha: 0.35 },
  classifier: { pinchStartThreshold: 0.045, pinchReleaseThreshold: 0.065 },
});

// Catch-all stream (everything except per-frame debug events):
engine.on('gesture', (event) => console.log(JSON.stringify(event)));

// Or per-type:
engine.onPinchStart((e) => { /* e.position (device), e.distance */ });
engine.onPinchDrag((e) => { /* e.currentPos, e.delta, e.startPos */ });
engine.onPinchEnd((e) => { /* e.endPos, e.delta */ });
engine.onExtrude((e) => { /* e.mode, e.scaleFactor | e.deltaHeight */ });
engine.onOrbit((e) => { /* e.deltaX, e.deltaY, e.deltaRoll (one fist) */ });
engine.onZoom((e) => { /* e.anchor, e.deltaScale, e.deltaAngle, e.scaleFactor (two fists) */ });

// Debug overlay / HUD data (fires once per processed frame):
engine.on('frame', (frame) => { /* state, hands, metrics, fps */ });

await engine.start(videoElement); // from a user gesture (camera permission)
engine.stop();
```

Event payloads use device space: `[-1, 1]`, X mirrored (matches the on-screen
view), +Y up — ready to map into a CAD viewport.

Append `?debug` to the URL to log every gesture event to the console as JSON
and expose the live module graph on `window.cadVision` (`engine`, `overlay`,
`cadScene`, `builder`, `toolbar`, `voiceAgent`, `voiceHud`).

## Configuration

`new GestureEngine({ tracker, handedness, smoothing, classifier })`:

- `tracker` — `HandTrackerOptions`: asset paths, delegate (`GPU`/`CPU`),
  confidence thresholds, camera constraints.
- `handedness` — `HandednessStabilizerOptions`: `voteWindow` (5),
  `flipFrames` (3), `highConfidence` (0.9), `geometricWeight` (0.75),
  `reentryFrames` (10) for Left/Right label stabilization.
- `smoothing` — `HandSmootherBankOptions`: `strategy: 'ema' | 'one-euro'`,
  `alpha` (default 0.35), teleport/stale-history thresholds.
- `classifier` — `GestureClassifierOptions`: pinch thresholds, fist shape
  (`fistFoldRatio` 0.9, `fistMinFoldedFingers` 4, `fistThumbTuckRatio` 0.75,
  `fistHoldSlack` 0.15, `fistPinchGuardDistance`) & debounce frames, wrist
  roll (`rollEngageAngle` 0.15 rad, `rollDeadzone` 0.003 rad), two-fist anchor
  (`zoomAnchorSwitchRatio` 0.5, `zoomTurnEngageAngle` 0.12 rad), camera path
  straightening (`cameraPath`: `startDistance`, `cornerDeviation`,
  `settleDistance`, `deadband`, or `null`), orbit
  open-palm grace, hand-loss grace frames.

## Debug overlay

The `<canvas id="overlay">` (pure Canvas 2D) mirrors the feed and draws:

- all 21 landmarks per hand + MediaPipe skeleton connections,
- state color-coding: **green** = pinch/draw, **blue** = one-fist move,
  **purple** = two-fist zoom / turn (ring = anchor fist, dashed line to the
  moving fist), **yellow** = idle
  (orange for EXTRUDING),
- thumb↔index pinch line with live distance, dual-hand extrusion link with
  `D` and scale factor,
- HUD: FPS, hand count + handedness, gesture state, live pinch/extrude/orbit
  metrics.

## Tests

```bash
npm test          # vitest — coordinates, filters, handedness stabilizer, path
                  # straightener, classifier/FSM, talk gate, tool validation
npm run typecheck # tsc --noEmit (browser) + tsc -p tsconfig.server.json (server)
npm run build     # typecheck + vite production build
```

## Troubleshooting

- **Camera blocked** — grant permission; `localhost` or HTTPS is required.
- **`HandLandmarker` init fails** — run `npm run setup:assets`; the engine also
  falls back to the jsDelivr/GCS CDN copies automatically.
- **Jittery landmarks** — lower `smoothing.alpha` (e.g. 0.25) or switch to
  `strategy: 'one-euro'`.
- **Pinch feels off** — tune `pinchStartThreshold` / `pinchReleaseThreshold`
  to hand size and camera distance.
- **Voice button errors immediately** — start the API server
  (`npm run dev:server`) and check `GET /healthz`, which reports which keys
  it found.
- **Spoken replies fail with 402** — `ELEVENLABS_VOICE_ID` is a Voice Library
  voice; free accounts can only use premade ones. See `.env.example`.
- **A "Talk" button appeared** — hands-free mouth detection is not running
  (no `PRESAGE_API_KEY`, the key was rejected, or the camera is off). It is a
  toggle, not a push-to-talk: press **Talk**, say your piece, then press
  **Send**. Starting the camera hands control back to your face automatically.
- **Server log says "not in a valid state" or `POST /device/pair` 401** — the
  Physiology API key in `PRESAGE_API_KEY` was rejected. The pipeline dies on
  that, and every later frame repeats the same error. Replace the key from
  [physiology.presagetech.com](https://physiology.presagetech.com/auth/login)
  and restart `npm run dev:server`. Voice still works through the **Talk**
  button in the meantime.
- **It hears you but nothing happens** — the turn is never being sealed. That
  means the gate never closed: either you are in manual mode and did not press
  **Send**, or mouth detection is stuck reporting speech (the gate gives up
  after `maxOpenMs`, 20 s, and commits anyway).
- **The mic opens when you are not talking** — raise `openDelayMs` in the
  `TalkGate` options; lower it if the first word keeps getting clipped.

