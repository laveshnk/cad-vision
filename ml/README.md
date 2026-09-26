# CAD Vision — Gesture-Driven CAD Workbench (`ml/`)

Self-contained TypeScript application that turns webcam hand tracking into an
interactive CAD tool. The MediaPipe-based gesture engine emits **normalized CAD
gesture events** (pinch/draw, extrude, camera move, zoom); a decoupled Three.js module
(orbit rig, ground-plane building, STL export) and a light glass toolbar
consume them. The camera renders as a floating video-call-style thumbnail
(landmarks, skeleton, HUD overlay) over the full-bleed 3D viewport.

Three interaction modes — **VIEW** (camera navigation only), **SELECT**
(pick / drag / recolor meshes) and **CREATE** (build primitives) — are
switched with a
button bar on the camera overlay (mouse click, or a **pointing** hand — index
finger up, other fingers curled — holding its fingertip on a button)
and strictly partition which gestures can act on the scene.

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
├── styles.css                 # light theme; full-bleed viewport, camera thumbnail, glass toolbar
├── cad/
│   ├── CadScene.ts            # Three.js viewport: camera rig, lights, grid, orbit
│   ├── CadBuilder.ts          # gesture-driven primitives + STL export
│   ├── arProjection.ts        # pure NDC→canvas math + 2D convex hull (AR mirror)
│   └── ArMirror.ts            # plain-data AR projection of grid + meshes for the overlay
├── ui/
│   ├── Toolbar.ts             # CAD toolbar: camera toggle, Clear scene, Export STL
│   └── ColorWheel.ts          # floating HSL color wheel (SELECT-mode live repaint)
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
  → GestureClassifier    mode-partitioned FSM: IDLE | DRAWING_BASE | SELECTING |
                         EXTRUDING | ORBITING | ZOOMING (VIEW / SELECT / CREATE)
  → listeners            typed events + per-frame debug event
```

App wiring (strictly decoupled — `src/cad` and `src/ui` import nothing from
`src/vision`; only device-space coordinates, deltas and state events cross
the boundary):

```
GestureEngine  --typed events-->  main.ts (orchestrator)
  ├── CadScene / CadBuilder      primitives, extrusion, selection, orbit, STL export
  └── Toolbar                    camera toggle + scene utilities
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

### Interaction modes

`GestureEngine.setMode('view' | 'select' | 'create')` partitions gesture
routing (classifier default `'create'` = the full legacy set; the app itself
starts in `'view'`):

| Mode | Pinches | Camera gestures (fist orbit / two-fist zoom) |
| ---- | ------- | -------------------------------------------- |
| **VIEW** | inert — no `pinch_*` / `extrude_*` events, never a build state, cannot veto orbit / zoom | live |
| **SELECT** | `pinch_start` raycasts against committed meshes (pick + highlight, empty ground deselects); `pinch_drag` drags the selected mesh under the active constraint toggle — `[ XZ PLANE ]` slides it across the ground (elevation locked) and `[ Y AXIS ]` maps vertical hand travel to a lift / lower; with the pinch held, an **open palm** on the other hand emits `select_rotate` yaw deltas (compass ring around the object); the vision overlay mirrors the 3D scene as a translucent AR layer; FSM state `SELECTING` | live |
| **CREATE** | full build gesture set (below) | live |

A mode switch force-releases in-flight pinches (synthetic `pinch_end`) and
returns pinch-driven states to IDLE; it emits a `mode_change` event and is
reflected on the overlay's button bar and in the per-frame event's `mode`.

| Gesture | Detection | Events |
| ------- | --------- | ------ |
| **Pinch / draw** | 3D Euclidean distance between landmarks 4 (thumb tip) and 8 (index tip); trigger `< 0.045`, hysteresis release `> 0.065` (both configurable) | `pinch_start`, `pinch_drag` (position, delta in device space), `pinch_end` |
| **Extrude (dual-hand)** | both hands pinch; pull distance `D` between pinch centers `center = (P_thumb + P_index) / 2` | `extrude_start`, `extrude` (`distance`, `scaleFactor = D / D₀`, `deltaDistance`, `spanX` / `spanY` = horizontal / vertical pinch gap in aspect-corrected units of video width), `extrude_end` (`heightSet`: `false` when both pinches release together or the lower pinch releases first — that upper pinch is then consumed until released) |
| **Extrude (single-hand)** | while EXTRUDING with one pinch: vertical drag of landmark 8 | `extrude` (`deltaHeight`, `cumulativeHeight`, +Y up) |
| **Camera move (one fist)** | a *real* closed fist, not just curled fingers (`handShape.ts`): all 4 fingertips folded into the palm — `dist(tip, wrist) < 0.9 · dist(MCP, wrist)`, which rejects claw / hook curls and half-curls — and the thumb tucked over the fingers (thumb tip within 0.75 × palm size of an index / middle / ring knuckle, rejecting a thumbs-up); all ratios, so size / distance / rotation invariant. Needs 3 consecutive frames (app default); once held, thresholds relax by `fistHoldSlack` and one finger may loosen. Only engaged from IDLE with no active pinch, and a pinch never engages from a fist. Palm-center motion is reported as deltas, **straightened** by `PathStraightener` so the camera travels in straight segments instead of copying hand wiggle (see below); the app orbits the camera around the locked origin so the scene follows the fist (fist right → camera swings left, fist up → camera swings lower). **Wrist roll** — twisting the fist like a doorknob, measured as the rotation of the knuckle line (index MCP → pinky MCP) around the wrist → middle-MCP axis — turns the scene around the vertical axis with the twist; reported only after the twist exceeds `rollEngageAngle` (0.15 rad ≈ 9°) per fist, per-frame jitter under `rollDeadzone` dropped | `orbit_start`, `orbit` (`deltaX`, `deltaY` device units, `deltaRoll` radians), `orbit_end` |
| **Zoom / turn (two fists)** | both hands closed fists (same test). Each fist counts as **moving** when its smoothed palm speed exceeds `zoomMoveSpeed` (0.004 video widths / frame). **Both moving** → zoom by the gap between them (aspect-corrected palm centers): apart → zoom in, together → zoom out; rotating them around each other (steering wheel) → `deltaAngle` (counter-clockwise on screen = +) turns the scene, after `zoomTurnEngageAngle` (0.12 rad ≈ 7°) so a straight pull doesn't rotate. **Only one moving** → that fist moves the camera exactly like a single fist (`orbit` deltas, straightened) and never zooms. **Neither moving** → nothing. The steadier fist is reported as the `anchor` (switches only when the other is below `zoomAnchorSwitchRatio` 0.5× its speed). Engaged from IDLE, or upgraded from a one-fist move when the second fist closes; ends on a pinch, a lost hand, or one hand open longer than `orbitOpenPalmGraceFrames` (then falls back to a one-fist move if the other fist is still closed) | `zoom_start`, `zoom` (`anchor`, `distance`, `scaleFactor = D / D₀`, `deltaScale = D / D_prev`, `deltaAngle`), `orbit` (one fist moving), `zoom_end` |
| **Open-palm rotation (SELECT)** | while exactly one hand holds a pinch, the *other* hand shows an open palm: all four fingertips extended past their own PIP joints (`dist(tip, wrist) > 1.0 · dist(PIP, wrist)` per finger) and the thumb held out (thumb-tip farther than `openPalmThumbTuckRatio` 0.6 × palm size from the nearest knuckle — the inverse of the fist thumb-tuck test). The palm's tilt (angle of the wrist → middle-MCP vector in the mirrored, aspect-corrected view) is tracked frame-to-frame, unwrapped across ±π and anchored at gesture start, so the app applies `selectedMesh.rotation.y = initialRotation + deltaRotation` and a re-opened palm never jumps the object. Ends — `select_rotate_end` — when the palm closes / starts pinching, the pinch releases, a hand is lost or the mode switches. Only in SELECT mode; elsewhere the same poses route to build / navigation gestures | `select_rotate` (`hand`, `palmHand`, cumulative `deltaRotation` radians), `select_rotate_end` (`palmHand`, `reason`) |

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

Finite state machine: `IDLE ↔ DRAWING_BASE ↔ SELECTING ↔ EXTRUDING ↔ ORBITING
↔ ZOOMING` (`SELECTING` only in select mode; no pinch-driven state is ever
entered in view mode). Outside view mode pinch always wins over fist
gestures; a lost hand finalizes its gesture after a
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
as a floating thumbnail (top-left, click ⤢ to expand/collapse) over it.
Interaction modes are switched with the button bar along the top of the
camera overlay — mouse click, or **point** (index finger up, other fingers
curled) and hold the fingertip on a button for 500 ms (a ring marks the
pointing fingertip, a progress bar fills). Pinches, fists and open palms
never press buttons, so moving, editing or building can't switch modes or
shapes by accident. The app starts in **VIEW** and the active
mode is shown inverted with an indicator bar (and in the bottom-left HUD).
In **CREATE** mode a column of shape icon buttons appears down the right edge
of the camera view — **cube** (default), **cuboid**, **cylinder**, **sphere** —
picked the same way (point-and-hold / click); it sets the shape for the next
build:

1. **Two-hand build** — pinch with both hands: a translucent wireframe
   preview of the selected shape spawns centered on the origin `(0, 0, 0)`,
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
3. **Single-hand footprint (legacy)** — pinch and drag with one hand: the
   pinch start raycasts onto the ground plane (`y = 0`) and dragging sets the
   footprint (box: square, cuboid: corner-to-corner rectangle,
   cylinder/sphere: center + radius). It is committed with the default height on the next build or
   camera gesture; pinching the second hand before releasing replaces it
   with a two-hand build.
4. **Select, move & recolor (SELECT mode)** — a quick pinch on a committed
   mesh **selects** it (highlighted; it stays selected after you let go).
   To **move** it, pinch and *hold* (the mesh starts following after
   `dragHoldMs`, 300 ms, so a quick pinch never nudges it), drag, and
   release to drop it. Dragging moves it under the active
   **constraint toggle** (top-left of the vision overlay): `[ XZ PLANE ]`
   (default) slides it on a horizontal plane at the height of the point you
   grabbed, with its elevation locked, while `[ Y AXIS (ELEVATE) ]` ignores
   horizontal drift and lifts / lowers it along a camera-facing vertical
   plane through the grab point (clamped so it never sinks below the floor).
   Either way the exact point you grabbed stays under your fingertip — in
   depth as well as sideways. Switching between the toggles mid-drag
   re-anchors, so the mesh never jerks or resets. Pinching empty ground
   deselects (so does leaving SELECT mode), restoring the mesh's normal look
   — the highlight tint / outline go, a picked color stays. While in SELECT mode the camera thumbnail doubles as a live
   **AR spatial mirror**: the ground grid (30 × 30 world units, 1-unit minor
   + stronger 5-unit major lines, off-canvas geometry culled) and every
   committed mesh are projected through the shared 3D camera and drawn as
   translucent ghosts (the active selection glows amber/cyan) in lockstep
   with the viewport. Keep the pinch held and show an **open palm** with
   your other hand: tilting it spins the selection around the vertical
   axis, with a compass ring (dashed circle + yaw needle) rendered around
   the object in both the 3D viewport and the AR mirror. A floating
   **HSL color wheel** appears beside the selected mesh (tracking it
   through drags and camera moves): **point** at the disc (index finger
   up) — every hue under the fingertip repaints the mesh live (angle =
   hue, radius = saturation, wheel center = gray). This works after a
   quick select-pinch, or with your other hand while one hand holds the
   mesh. Pinching directly on the wheel
   confirms the color under the pinch instead of re-picking the scene
   behind it, and only the grabbing hand's release ends the drag. The
   wheel hides when the selection is cleared or the mode changes. Pinches
   never draw or extrude in this mode.
5. **Orbit the camera** — make a fist and move it: the camera orbits the
   world origin and the scene follows your hand — fist right swings the
   camera left, fist up swings it lower — with damping (`CadScene.onOrbit`);
   open palm stops. The camera focus is **locked to `(0, 0, 0)`**: the view
   only rotates and zooms, it never pans or translates (`enablePan` is
   permanently `false`).
   **Rotate** by rolling your wrist while holding the fist (twist it like a
   doorknob): the scene turns around the vertical axis with your twist
   (`CadScene.onRotate`, `rotateSpeed` 1.5×). Open and re-close the fist to
   ratchet further.
6. **Zoom / turn** — make fists with both hands and move **both**: pull them
   apart to zoom in, bring them together to zoom out (`CadScene.onZoom`,
   clamped between `minDistance` and `maxDistance`), or turn them around
   each other like a steering wheel to turn the scene (`CadScene.onRotate`).
   If only one fist moves (the other held still), it just moves the camera
   like a single fist — it never zooms.

Toolbar (mouse or programmatic): a single **Start camera / Stop** toggle
(webcam + tracking lifecycle), **Clear scene**, and **Export STL** (binary
`model.stl` download via `three/examples/jsm/exporters/STLExporter`). New
builds use the cube by default; the shape is picked with the CREATE-mode
icon buttons on the camera view (or programmatically via
`CadBuilder.setTool()`).

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
engine.onSelectRotate((e) => { /* e.hand, e.palmHand, e.deltaRotation (SELECT, open palm) */ });
engine.onSelectRotateEnd((e) => { /* e.palmHand, e.reason */ });

// Debug overlay / HUD data (fires once per processed frame):
engine.on('frame', (frame) => { /* state, mode, hands, metrics, fps */ });

// Interaction modes (VIEW / SELECT / CREATE):
engine.setMode('select');
engine.on('mode_change', (event) => { /* event.from, event.to */ });

await engine.start(videoElement); // from a user gesture (camera permission)
engine.stop();
```

`main.ts` logs every gesture event to the console as JSON for downstream CAD
consumers. Event payloads use device space: `[-1, 1]`, X mirrored (matches the
on-screen view), +Y up — ready to map into a CAD viewport.

## Configuration

`new GestureEngine({ tracker, handedness, smoothing, classifier, initialMode })`:

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
  (`zoomAnchorSwitchRatio` 0.5, `zoomTurnEngageAngle` 0.12 rad,
  `zoomMoveSpeed` 0.004), camera path
  straightening (`cameraPath`: `startDistance`, `cornerDeviation`,
  `settleDistance`, `deadband`, or `null`), orbit
  open-palm grace, hand-loss grace frames, and open-palm detection for the
  SELECT-mode rotation (`openPalmExtensionRatio` 1.0,
  `openPalmMinExtendedFingers` 4, `openPalmThumbTuckRatio` 0.6), and the
  pointing pose that presses overlay buttons (`pointingIndexRatio` 1.5,
  `pointingCurlRatio` 1.2, `pointingHoldSlack` 0.15,
  `pointingDebounceFrames` 2).
- `initialMode` — `InteractionMode`: starting interaction mode. Defaults to
  `'create'` (full legacy gesture set); the app itself starts in `'view'`
  and switches via `setMode()` at runtime.

## Debug overlay

The `<canvas id="overlay">` (pure Canvas 2D) mirrors the feed and draws:

- the **mode switcher** along the top edge: three boxy, mutually exclusive
  toggle buttons (`[ VIEW ] [ SELECT ] [ CREATE ]`) — the active one is
  inverted (solid light fill + high-contrast indicator bar). Activated by
  mouse click or a **pointing** hand (`HandSnapshot.pointing`: index up,
  middle / ring / pinky curled, no pinch; debounced) holding its index tip
  (landmark 8) on the button for 500 ms (ring cursor + progress bar), via
  `DebugOverlay`'s `onModeRequest` callback. Pinches never press buttons,
  so an object drag sweeping across the bar never switches modes;
- in **SELECT mode**, a stack of two **drag-constraint toggles** below the
  mode bar in the top-left corner: `[ XZ PLANE ]` (default) and
  `[ Y AXIS (ELEVATE) ]` — same boxy style and pointing activation
  (`onDragConstraintRequest`);
- in **CREATE mode**, a column of square **shape icon buttons** down the
  right edge (cube (default), cuboid, cylinder, sphere — from the `shapes`
  option, each with an `icon`; picks reported through `onShapeRequest` →
  `CadBuilder.setTool`) — same boxy style and pointing activation;
- in **SELECT mode**, a live **AR spatial mirror**: the 3D ground grid and
  every committed mesh are projected through the scene camera's
  webcam-aspect twin (`CadScene.interactionCamera` — same pose and vertical
  FOV, the webcam frame's aspect — + `ArMirror`) into the webcam image's
  on-screen rect, so ghosts keep their true proportions (a sphere stays
  round) and line up with the hands; hand raycasts use the same camera, so
  pinching a ghost picks that object. They are drawn as translucent cyan
  ghosts beneath the hands — the active selection gets an energetic
  amber/cyan glowing outline, and while an open-palm rotation runs a dashed
  compass ring + amber yaw needle circles the selection;
- all 21 landmarks per hand + MediaPipe skeleton connections,
- state color-coding: **green** = pinch/draw, **cyan** = selecting, **blue**
  = one-fist move, **purple** = two-fist zoom / turn (ring = anchor fist,
  dashed line to the
  moving fist), **yellow** = idle
  (orange for EXTRUDING),
- thumb↔index pinch line with live distance, dual-hand extrusion link with
  `D` and scale factor,
- HUD (bottom-left): interaction mode, FPS, hand count + handedness, gesture
  state, live pinch/extrude/orbit metrics.

## Tests

```bash
npm test        # vitest — coordinates, filters, handedness stabilizer, path straightener, classifier/FSM + color-wheel math unit tests
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

