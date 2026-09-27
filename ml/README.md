# CAD Vision — Gesture-Driven CAD Workbench (`ml/`)

Self-contained TypeScript application that turns webcam hand tracking into an
interactive CAD tool. The MediaPipe-based gesture engine emits **normalized CAD
gesture events** (pinch/draw, extrude, camera move, zoom); a decoupled Three.js module
(orbit rig, ground-plane building, **CSG Boolean operations** via
`three-bvh-csg`, STL export) and a light glass toolbar
consume them. The camera renders as a floating video-call-style thumbnail
(landmarks, skeleton and the in-vision UI drawn on its Canvas 2D overlay) over
the full-bleed 3D viewport, with a live stats bar mounted directly underneath
the camera view.

Three interaction modes — **VIEW** (camera navigation only), **EDIT**
(pick / drag / recolor meshes; mode id `select` in the API) and **CREATE** (build primitives) — are
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
│   ├── CadBuilder.ts          # gesture-driven primitives, selection, CSG booleans, STL export
│   ├── booleanOps.ts          # three-bvh-csg wrapper: subtract / union + AABB clash helpers
│   ├── arProjection.ts        # pure NDC→canvas math + 2D convex hull (AR mirror)
│   └── ArMirror.ts            # plain-data AR projection of grid + meshes for the overlay
├── ui/
│   ├── Toolbar.ts             # CAD toolbar: camera toggle, Export STL
│   ├── ColorWheel.ts          # floating HSL color wheel + timed hover lock (dwell tracker)
│   ├── SelectionMenu.ts       # floating selection HUD (Delete action, EDIT mode)
│   ├── MetricsBar.ts          # boxy monospace stats bar under the camera view
│   ├── hitTest.ts             # root-local DOM hit-tests for the floating overlays
│   ├── ThumbDragger.ts        # camera-window drag handle (outer frame only)
│   └── ThumbResizer.ts        # corner resize grip on the camera card's frame
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
    ├── DebugOverlay.ts        # 2D canvas renderer (landmarks, skeleton, in-vision UI)
    └── index.ts               # public API barrel
```

Per-frame pipeline:

```
HandTracker (raw MediaPipe hands)
  → HandednessStabilizer  Left/Right label votes + chirality, flicker/collision-proof
  → HandSmootherBank     EMA over all 21 landmarks, per-hand identity
  → coordinates          normalized / pixel / device spaces
  → GestureClassifier    mode-partitioned FSM: IDLE | DRAWING_BASE | SELECTING |
                         EXTRUDING | ORBITING | ZOOMING (VIEW / EDIT / CREATE)
  → listeners            typed events + per-frame debug event
```

App wiring (strictly decoupled — `src/cad` and `src/ui` import nothing from
`src/vision`; only device-space coordinates, deltas and state events cross
the boundary):

```
GestureEngine  --typed events-->  main.ts (orchestrator)
  ├── CadScene / CadBuilder      primitives, extrusion, selection, CSG booleans, orbit, STL export
  └── Toolbar / MetricsBar       camera toggle + scene utilities, live stats
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
| **EDIT** | `pinch_start` raycasts against committed meshes (pick + highlight, empty ground deselects); `pinch_drag` drags the selected mesh under the active constraint toggle — `[ XZ PLANE ]` slides it across the ground (elevation locked) and `[ Y AXIS ]` maps vertical hand travel to a lift / lower; with the pinch held, an **open palm** on the other hand emits `select_rotate` yaw deltas (compass ring around the object); dragging one mesh into another shows the **clash indicator** and the `[ SUBTRACT ]` / `[ UNION ]` Boolean tools can cut a hole or merge the solids; the vision overlay mirrors the 3D scene as a translucent AR layer; FSM state `SELECTING` | live |
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
| **Open-palm rotation (EDIT)** | while exactly one hand holds a pinch, the *other* hand shows an open palm: all four fingertips extended past their own PIP joints (`dist(tip, wrist) > 1.0 · dist(PIP, wrist)` per finger) and the thumb held out (thumb-tip farther than `openPalmThumbTuckRatio` 0.6 × palm size from the nearest knuckle — the inverse of the fist thumb-tuck test). The palm's tilt (angle of the wrist → middle-MCP vector in the mirrored, aspect-corrected view) is tracked frame-to-frame, unwrapped across ±π and anchored at gesture start, so the app applies `selectedMesh.rotation.y = initialRotation + deltaRotation` and a re-opened palm never jumps the object. Ends — `select_rotate_end` — when the palm closes / starts pinching, the pinch releases, a hand is lost or the mode switches. Only in EDIT mode; elsewhere the same poses route to build / navigation gestures | `select_rotate` (`hand`, `palmHand`, cumulative `deltaRotation` radians), `select_rotate_end` (`palmHand`, `reason`) |

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
as a floating thumbnail (top-left, click ⤢ to expand/collapse, drag the card's
**outer frame** — header bar or stats bar — to reposition it, drag the grip on
its bottom-right corner — outside the video — to resize it) over it. A boxy
monospace **stats bar** (mode / state / FPS / hands) is mounted directly
underneath the camera view inside the card.
Interaction modes are switched with the button bar along the top of the
camera overlay — mouse click, or **point** (index finger up, other fingers
curled) and hold the fingertip on a button for 500 ms (a ring marks the
pointing fingertip, a progress bar fills). Pinches, fists and open palms
never press toggle buttons, so moving, editing or building can't switch modes or
shapes by accident — the trash bin and the confirmation dialog's targets
included (pointing only). The app starts in **VIEW** and the active
mode is shown inverted with an indicator bar (and in the stats bar under the
camera).
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
3. **No accidental builds** — only the deliberate two-hand gesture creates
   an object. A single pinch in CREATE mode does nothing (the legacy
   single-hand floor footprint is opt-in via `CadBuilder`'s
   `singleHandFootprint`), a pinch must hold for 2 frames
   (`pinchEnterFrames`) and cannot start within 3 % of the frame border
   (`pinchEdgeMargin` — stray / half-visible hands in the corners), and a
   two-hand build shorter than 0.4 s (`MIN_BUILD_MS`, from the
   `extrude_end` event's `durationMs`) is discarded instead of committed.
4. **Select, move, recolor & delete (EDIT mode)** — a quick pinch on a committed
   mesh **selects** it (a bright outline shell is drawn around it — the
   mesh's own material is never tinted; it stays selected after you let go).
   To **move** it, pinch and *hold* (the mesh starts following after
   `dragHoldMs`, 300 ms, so a quick pinch never nudges it), drag, and
   release to drop it. Dragging moves it under the active
   **constraint toggle** (camera view, directly below `[ EDIT ]`): `[ XZ PLANE ]`
   (default) slides it on a horizontal plane at the height of the point you
   grabbed, with its elevation locked, while `[ Y AXIS ]` ignores
   horizontal drift and lifts / lowers it along a camera-facing vertical
   plane through the grab point (clamped so it never sinks below the floor).
   Either way the exact point you grabbed stays under your fingertip — in
   depth as well as sideways. Switching between the toggles mid-drag
   re-anchors, so the mesh never jerks or resets. Pinching empty ground
   deselects (so does leaving EDIT mode), restoring the mesh's normal look
   — the selection outline goes, a picked color stays. While in EDIT mode the camera thumbnail doubles as a live
   **AR spatial mirror**: the ground grid (30 × 30 world units, 1-unit minor
   + stronger 5-unit major lines, off-canvas geometry culled) and every
   committed mesh are projected through the shared 3D camera and drawn as
   translucent ghosts (the active selection glows amber/cyan) in lockstep
   with the viewport. Keep the pinch held and show an **open palm** with
   your other hand: tilting it spins the selection around the vertical
   axis, with a compass ring (dashed circle + yaw needle) rendered around
   the object in both the 3D viewport and the AR mirror. A floating
   **HSL color wheel** appears pinned in the bottom-left corner of the
   hand-reachable area (the webcam frame's bottom-left corner mapped into
   the viewport — the literal viewport corner lies outside the finger's
   reach). The camera view shows the **same live color disc** in its own
   bottom-left corner — hue by angle, saturation by radius, exactly
   aligned with the viewport wheel — so you can pick colors right there:
   **point** at the disc (index finger
   up) — every hue under the fingertip repaints the mesh live (angle =
   hue, radius = saturation, wheel center = gray). This works after a
   quick select-pinch, or with your other hand while one hand holds the
   mesh. The camera thumbnail mirrors the wheel's disc outline and the
    Delete button as a HUD, drawn exactly where your fingertip has to point
    (with live dwell progress), so you can align your hand with the controls
    while looking at the camera feed. **Hold the fingertip on one color slice
    for 1.2 seconds** and a
   countdown ring fills around the cursor; when it completes, the color
   is locked in and saved to the object, the wheel disappears, and the
   interaction resets (point again — or re-grab the mesh — to bring it
   back). The dwell is keyed to 15° hue slices (the desaturated center
   is one neutral slice), so micro-jitter never restarts the clock,
   while any slice change or hover loss does. The wheel also hides when
   the selection is cleared or the mode changes; pinches over the disc
   never re-pick the 3D scene behind it, and only the grabbing hand's
   release ends the drag. Pinches never draw or extrude in this mode.
   **Delete**: the selection HUD below the object carries a **Delete**
   button — click it, pinch it, point at it (index-tip dwell of ~0.8 s with a progress
    fill), or press `Delete` / `Backspace`. Every trigger opens the
   in-vision **spatial confirmation dialog** (see below) — never a browser
   popup or a DOM modal. Only **Confirm** removes the mesh from the scene
   and frees its geometry / material / edge overlays; Cancel leaves
   everything untouched. While the dialog is open the scene is frozen —
   gestures only answer the dialog. The same dialog guards the in-vision
   **trash bin** (VIEW mode: top-left of the camera view, directly below
   `[ VIEW ]`),
   which clears the whole scene on Confirm: point your index fingertip at
   the bin and hold it there for 600 ms, or click it — then point at
   `[ CONFIRM ]` or `[ CANCEL ]` and hold. Pinches never fire the bin or
   answer the dialog.
5. **CSG Booleans — cut a hole or merge (EDIT mode)** — with two committed
   meshes overlapping, drag the selected one into the other: while their
   world-space bounding boxes intersect (`box3.intersectsBox`), a
   **translucent amber clash indicator** fills the overlap volume and the
   armed Boolean button glows amber (ready to fire). The two mutually
   exclusive tool toggles sit in the left stack below the drag-constraint
   toggles: `[ SUBTRACT ]` and `[ UNION ]` — point at
   one and hold (or click) to arm it; if a clash is already live the
   operation fires immediately. To fire while dragging (one hand holds the
   mesh), show an **X cross** with your free hand — index and pinky
   extended, middle and ring folded — and hold it briefly (~0.2 s).
   - **Subtract** carves the *selected* cutter out of the intersected
     base (`Evaluator.evaluate(base, cutter, SUBTRACTION)` from
     `three-bvh-csg`): the base is replaced by the result mesh (fresh
     geometry with crisp regenerated `EdgesGeometry` edge lines, the
     base's frame and transform), and the cutter is removed
     from the scene. The cutter is grown 0.1 % first: solids all rest on
     the floor, so the operands' bottom faces are coplanar, which makes
     BSP CSG leave slivers and stray fragments. Result edges come from
     `creaseEdges` — it skips the T-junction seams CSG leaves inside flat
     faces (collinear partner edges matched with a float-tolerant 1e-4 ×
     size) and ignores degenerate triangles (collapsed edges / zero area,
     whose normals are noise) so smooth or flat seams never show as fake
     creases; tests audit several cuts (through-hole, sphere corner, side
     bite, repeated cuts) for zero stray edges — and keep the base's edge color and the evaluator's normals, and
     the result renders double-sided so T-junction pixel cracks never show
     the floor through the solid.
   - **Union** fuses both solids into **one continuous body**
     (`ADDITION`), wearing the selected mesh's color, with unified edge
     lines replacing the two separate entities in the scene hierarchy.
   The result stays selected, so you can keep dragging it into more
   shapes; the clash indicator refreshes live. A failed / empty CSG is a
   safe no-op (the scene is never half-edited). Committed results export
   with the rest of the scene as `model.stl`.
6. **Orbit the camera** — make a fist and move it: the camera orbits the
   world origin and the scene follows your hand — fist right swings the
   camera left, fist up swings it lower — with damping (`CadScene.onOrbit`);
   open palm stops. The camera focus is **locked to `(0, 0, 0)`**: the view
   only rotates and zooms, it never pans or translates (`enablePan` is
   permanently `false`).
   **Rotate** by rolling your wrist while holding the fist (twist it like a
   doorknob): the scene turns around the vertical axis with your twist
   (`CadScene.onRotate`, `rotateSpeed` 1.5×). Open and re-close the fist to
   ratchet further.
7. **Zoom / turn** — make fists with both hands and move **both**: pull them
   apart to zoom in, bring them together to zoom out (`CadScene.onZoom`,
   clamped between `minDistance` and `maxDistance`), or turn them around
   each other like a steering wheel to turn the scene (`CadScene.onRotate`).
   If only one fist moves (the other held still), it just moves the camera
   like a single fist — it never zooms.

**Spatial confirmation dialog (no mouse popups).** Every destructive action
— deleting the selected object (Delete button, keyboard) or clearing the
whole scene (in-vision trash bin) — is confirmed *in vision*: a translucent
scrim + centered card drawn on the camera canvas with two spatial targets,
`[ CONFIRM (Hold) ]` (danger red) and `[ CANCEL (Hold) ]` (calm blue).
Confirm by **pointing** at the confirm target and holding for 1.2 s — the
same wait as locking a color on the wheel and pressing the Delete button
(one shared `HOLD_TO_ACT_MS` in `main.ts`) — or by holding the **OK
gesture** (thumb and index tips touching, middle / ring / pinky extended)
anywhere in view for the same time, by clicking the target, or by pressing
`Enter`; an amber progress bar fills the target. A pinch never confirms
instantly. Cancel by a pointing hold on the cancel target, an **open palm**
held briefly, moving every hand out of view, `Escape`, or clicking the
cancel target. The pinch that triggered the dialog stays
"engaged", so releasing it into an open hand can never instantly cancel —
and while the dialog is open the scene below is frozen. There is no
`window.confirm()` and no DOM modal anywhere in the delete / clear flows.

Toolbar (mouse or programmatic): a single **Start camera / Stop** toggle
(webcam + tracking lifecycle) and **Export STL** (binary
`model.stl` download via `three/examples/jsm/exporters/STLExporter`) —
scene clearing lives in vision (the trash bin + spatial confirmation), so
no mouse-only destructive button is mounted in the header. New
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
engine.onSelectRotate((e) => { /* e.hand, e.palmHand, e.deltaRotation (EDIT, open palm) */ });
engine.onSelectRotateEnd((e) => { /* e.palmHand, e.reason */ });

// Debug overlay / HUD data (fires once per processed frame):
engine.on('frame', (frame) => { /* state, mode, hands, metrics, fps */ });

// Interaction modes (VIEW / EDIT / CREATE):
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
  EDIT-mode rotation (`openPalmExtensionRatio` 1.0,
  `openPalmMinExtendedFingers` 4, `openPalmThumbTuckRatio` 0.6), and the
  pointing pose that presses overlay buttons (`pointingIndexRatio` 1.5,
  `pointingCurlRatio` 1.2, `pointingHoldSlack` 0.15,
  `pointingDebounceFrames` 2). The per-frame hand snapshots additionally
  report three stateless poses (reusing those thresholds — no new options):
  `openPalm` (dialog cancel), `okGesture` (thumb + index loop, other
  fingers extended — confirm hold) and `xCross` (index + pinky up,
  middle + ring folded — the Boolean trigger).
- `initialMode` — `InteractionMode`: starting interaction mode. Defaults to
  `'create'` (full legacy gesture set); the app itself starts in `'view'`
  and switches via `setMode()` at runtime.

## Debug overlay

The `<canvas id="overlay">` (pure Canvas 2D) mirrors the feed and draws:

- the **mode switcher** along the top edge: three boxy, mutually exclusive
  toggle buttons (`[ VIEW ] [ EDIT ] [ CREATE ]`) — the active one is
  inverted (solid light fill + high-contrast indicator bar). Activated by
  mouse click or a **pointing** hand (`HandSnapshot.pointing`: index up,
  middle / ring / pinky curled, no pinch; debounced) holding its index tip
  (landmark 8) on the button for 500 ms (ring cursor + progress bar), via
  `DebugOverlay`'s `onModeRequest` callback. Pinches never press buttons,
  so an object drag sweeping across the bar never switches modes;
- each mode's options sit under its own button: the VIEW-mode trash bin
  under `[ VIEW ]`, the EDIT-mode stack under `[ EDIT ]`, the CREATE-mode
  shape icons down the right edge under `[ CREATE ]`;
- in **EDIT mode**, a stack of two **drag-constraint toggles** directly
  below `[ EDIT ]`: `[ XZ PLANE ]` (default) and
  `[ Y AXIS ]` — same boxy style and pointing activation
  (`onDragConstraintRequest`);
- in **EDIT mode**, the two **CSG Boolean tool toggles** continue the
  same stack: `[ SUBTRACT ]` and `[ UNION ]` — mutually
  exclusive, re-press disarms (`onBooleanToolRequest` → arm, and fire when
  a clash is already live). While a tool is armed and the selection
  intersects another mesh (live `booleanState` provider), the armed button
  glows amber — ready to fire. The secondary-hand **X cross** (index +
  pinky extended, middle / ring folded, held ~0.2 s) fires SUBTRACT
  without touching the buttons (`onBooleanTrigger`);
- in **CREATE mode**, a column of square **shape icon buttons** down the
  right edge (cube (default), cuboid, cylinder, sphere — from the `shapes`
  option, each with an `icon`; picks reported through `onShapeRequest` →
  `CadBuilder.setTool`) — same boxy style and pointing activation;
- in **EDIT mode**, a live **AR spatial mirror**: the 3D ground grid and
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
- the **trash bin** (VIEW mode only, top-left, directly below `[ VIEW ]`):
  a boxy recycle-bin icon button that clears
  the scene through the spatial confirmation (`onTrashRequest`) — activated
  only by a pointing index-tip dwell of 600 ms (`trashDwellMs`, progress
  bar fills along its bottom edge) or a mouse click; pinches never fire
  it. Hidden while the confirmation dialog is open;
- the **spatial confirmation dialog**: scrim + centered card with the
  `[ CONFIRM (Hold) ]` / `[ CANCEL (Hold) ]` targets (see the
  workflow section). While it is open (`overlay.confirmActive` /
  `confirmIntent`), every other overlay interaction freezes; the host
  receives the answer through `onConfirmRequest` / `onCancelRequest`
  (`openConfirm('clear-scene' | 'delete-selection')` / `closeConfirm()`);
- the live stats (mode / state / FPS / hands) are **not** drawn on the
  canvas — they render in the DOM **metrics bar** (`src/ui/MetricsBar.ts`)
  mounted directly underneath the camera view inside the thumbnail card.

## Tests

```bash
npm test        # vitest — coordinates, filters, handedness stabilizer, path straightener, classifier/FSM (+ pose snapshots), CSG booleanOps, color-wheel math / dwell tracker unit tests
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

