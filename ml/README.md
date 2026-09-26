# CAD Vision — Gesture-Driven CAD Workbench (`ml/`)

Self-contained TypeScript application that turns webcam hand tracking into an
interactive CAD tool. The MediaPipe-based gesture engine emits **normalized CAD
gesture events** (pinch/draw, extrude, orbit); a decoupled Three.js module
(orbit rig, ground-plane building, STL export) and a glassmorphism toolbar
consume them. The camera renders as a floating video-call-style thumbnail
(landmarks, skeleton, HUD overlay) over the full-bleed 3D viewport.

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

## Architecture

```
src/
├── main.ts                    # app orchestrator: vision events → CAD + UI wiring
├── styles.css                 # full-bleed viewport, floating camera thumbnail, glass toolbar
├── cad/
│   ├── CadScene.ts            # Three.js viewport: camera rig, lights, grid, orbit
│   └── CadBuilder.ts          # gesture-driven primitives + STL export
├── ui/
│   └── Toolbar.ts             # CAD toolbar: camera start/stop, Box/Cylinder/Sphere, Clear, Export STL
└── vision/
    ├── types.ts               # shared types + event payloads
    ├── coordinates.ts         # device-space mapping, vec3 math
    ├── filters.ts             # EMA / One-Euro, LandmarkSmoother, HandSmootherBank
    ├── HandTracker.ts         # webcam + MediaPipe HandLandmarker
    ├── HandednessStabilizer.ts # temporal + geometric Left/Right label stabilization
    ├── GestureClassifier.ts   # pinch/fist/orbit detection + finite state machine
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
  → GestureClassifier    FSM: IDLE | DRAWING_BASE | EXTRUDING | ORBITING
  → listeners            typed events + per-frame debug event
```

App wiring (strictly decoupled — `src/cad` and `src/ui` import nothing from
`src/vision`; only device-space coordinates, deltas and state events cross
the boundary):

```
GestureEngine  --typed events-->  main.ts (orchestrator)
  ├── CadScene / CadBuilder      primitives, extrusion, orbit, STL export
  └── Toolbar                    tool selection + scene utilities
```

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
| **Extrude (dual-hand)** | both hands pinch; pull distance `D` between pinch centers `center = (P_thumb + P_index) / 2` | `extrude_start`, `extrude` (`distance`, `scaleFactor = D / D₀`, `deltaDistance`), `extrude_end` |
| **Extrude (single-hand)** | while EXTRUDING with one pinch: vertical drag of landmark 8 | `extrude` (`deltaHeight`, `cumulativeHeight`, +Y up) |
| **Orbit** | closed fist — per-finger curl test: fingertip closer to the wrist than its own PIP joint, or tip within 0.08 of its MCP (≥ 3 of 4 non-thumb fingers curled) and not pinching (thumb-index distance > 0.07); needs 2 consecutive frames, only engaged from IDLE with no active pinch | `orbit_start`, `orbit` (`deltaX`, `deltaY` device units), `orbit_end` |

Finite state machine: `IDLE ↔ DRAWING_BASE ↔ EXTRUDING ↔ ORBITING`. Pinch
always wins over orbit; a lost hand finalizes its gesture after a small grace
window (synthetic `pinch_end` / `orbit_end`) so states never get stuck.

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
camera) driven entirely by the gesture events above, with the camera rendered
as a floating thumbnail (top-left, click ⤢ to expand/collapse) over it:

1. **Draw the footprint** — pinch and drag: the pinch start raycasts onto the
   ground plane (`y = 0`), spawning a translucent wireframe preview of the
   selected tool; dragging sets the footprint (box: corner-to-corner
   rectangle, cylinder/sphere: center + radius). Releasing freezes the base.
2. **Extrude** — pinch with both hands and pull apart to scale, or keep one
   pinch and drag vertically. Spheres grow their radius instead of height.
3. **Commit** — releasing the extrude gesture (or engaging a fist / open
   palm) freezes the preview into a solid matte mesh with crisp
   `EdgesGeometry` outlines.
4. **Orbit** — closed fist and move: the camera rig orbits with damping;
   open palm stops.

Toolbar (mouse or programmatic): **Start camera / Stop** (webcam + tracking
lifecycle), **Box / Cylinder / Sphere** tool selection (swaps the in-progress
preview too), **Clear scene**, and **Export STL** (binary `model.stl` download
via `three/examples/jsm/exporters/STLExporter`).

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
engine.onOrbit((e) => { /* e.deltaX, e.deltaY */ });

// Debug overlay / HUD data (fires once per processed frame):
engine.on('frame', (frame) => { /* state, hands, metrics, fps */ });

await engine.start(videoElement); // from a user gesture (camera permission)
engine.stop();
```

`main.ts` logs every gesture event to the console as JSON for downstream CAD
consumers. Event payloads use device space: `[-1, 1]`, X mirrored (matches the
on-screen view), +Y up — ready to map into a CAD viewport.

## Configuration

`new GestureEngine({ tracker, handedness, smoothing, classifier })`:

- `tracker` — `HandTrackerOptions`: asset paths, delegate (`GPU`/`CPU`),
  confidence thresholds, camera constraints.
- `handedness` — `HandednessStabilizerOptions`: `voteWindow` (5),
  `flipFrames` (3), `highConfidence` (0.9), `geometricWeight` (0.75),
  `reentryFrames` (10) for Left/Right label stabilization.
- `smoothing` — `HandSmootherBankOptions`: `strategy: 'ema' | 'one-euro'`,
  `alpha` (default 0.35), teleport/stale-history thresholds.
- `classifier` — `GestureClassifierOptions`: pinch thresholds, fist curl/guard distances
  & debounce frames, orbit open-palm grace, hand-loss grace frames.

## Debug overlay

The `<canvas id="overlay">` (pure Canvas 2D) mirrors the feed and draws:

- all 21 landmarks per hand + MediaPipe skeleton connections,
- state color-coding: **green** = pinch/draw, **blue** = orbit, **yellow** = idle
  (orange for EXTRUDING),
- thumb↔index pinch line with live distance, dual-hand extrusion link with
  `D` and scale factor,
- HUD: FPS, hand count + handedness, gesture state, live pinch/extrude/orbit
  metrics.

## Tests

```bash
npm test        # vitest — coordinates, filters, handedness stabilizer, classifier/FSM unit tests
npm run build   # tsc --noEmit + vite production build
```

## Troubleshooting

- **Camera blocked** — grant permission; `localhost` or HTTPS is required.
- **`HandLandmarker` init fails** — run `npm run setup:assets`; the engine also
  falls back to the jsDelivr/GCS CDN copies automatically.
- **Jittery landmarks** — lower `smoothing.alpha` (e.g. 0.25) or switch to
  `strategy: 'one-euro'`.
- **Pinch feels off** — tune `pinchStartThreshold` / `pinchReleaseThreshold`
  to hand size and camera distance.

