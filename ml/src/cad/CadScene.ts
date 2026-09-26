/**
 * CadScene: self-contained Three.js CAD viewport.
 *
 * Owns the WebGLRenderer, a PerspectiveCamera on a damped spherical orbit
 * rig, soft studio lighting (hemisphere + shadow-casting key light), a
 * shadow-catching floor and an "infinite" grid faded into the background by
 * fog. External control inputs are plain device-space data —
 * `onOrbit({ deltaX, deltaY })` (orbit around the origin), `onRotate({ deltaAngle })`
 * (turn the scene around the vertical axis) and `onZoom({ deltaScale })` (dolly
 * in / out) — keeping this module fully decoupled from the vision layer.
 * `projectToCanvas` shares the camera's live perspective with 2D consumers.
 *
 * Hand input arrives in the webcam frame's device space, whose aspect (e.g.
 * 4:3) differs from the wide viewport's. `interactionCamera` is the view
 * camera's twin — same pose and vertical field of view — with the webcam
 * aspect (`setInteractionAspect`): the AR mirror projects through it (so
 * ghosts keep true proportions and line up with the hands on the camera
 * view) and hand raycasts use it (so pinching a ghost picks that object).
 *
 * The camera focus is locked to the world origin (0, 0, 0): the view can only
 * rotate and zoom, never pan / translate. The camera always sits on a sphere
 * centered on the origin (radius = zoom distance), recomputed every frame.
 */

import * as THREE from 'three';
import { ndcToCanvas, remapNdcX, type ProjectedPoint } from './arProjection';

export interface CadSceneOptions {
  /** Background & fog color. */
  background?: number;
  /** Camera distance from the orbit target (world units). */
  cameraDistance?: number;
  /** Camera azimuth around +Y (radians); 0 = straight on, grid square to the screen. */
  cameraAzimuth?: number;
  /** Camera polar angle from +Y (radians); clamped to stay above the floor. */
  cameraPolar?: number;
  /** Radians of orbit per device-space unit (hand travel). */
  orbitSpeed?: number;
  /** Scene rotation (radians) per radian of input turn (wrist roll / fist sweep). */
  rotateSpeed?: number;
  /** Exponent applied to zoom ratios (higher = more zoom per hand movement). */
  zoomSpeed?: number;
  /** Closest allowed camera distance (world units). */
  minDistance?: number;
  /** Farthest allowed camera distance (world units). */
  maxDistance?: number;
  /**
   * Exponential damping rate for camera motion (higher = snappier). Kept
   * gentle on purpose: a camera that follows the hand too tightly feels
   * jittery and can cause motion sickness; building stays live regardless.
   */
  damping?: number;
  /** Ground grid extent (world units). */
  gridSize?: number;
  /** Ground grid divisions. */
  gridDivisions?: number;
}

/** Fixed orbit pivot: the world origin. Frozen so nothing can move it. */
const ORBIT_TARGET: Readonly<THREE.Vector3> = Object.freeze(new THREE.Vector3(0, 0, 0));

/** Keep the camera above the floor and out of polar gimbal lock. */
const POLAR_MIN = 0.12;
const POLAR_MAX = 1.52;

/** Short ground-plane axis line (CAD orientation cue: X red, Z blue). */
function axisLine(from: THREE.Vector3, to: THREE.Vector3, color: number): THREE.Line {
  const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.65,
  });
  return new THREE.Line(geometry, material);
}

export class CadScene {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly grid: THREE.GridHelper;
  /** Invisible shadow-catching floor (a shade darker than the grid). */
  readonly ground: THREE.Mesh;
  /** Origin axis lines (X / Z) drawn just above the ground. */
  readonly axes: THREE.Group;
  /** Panning / translation is permanently disabled: the focus is locked to the origin. */
  readonly enablePan = false;
  /** Allow orbit / rotate input (`onOrbit`, `onRotate`). */
  enableRotate = true;
  /** Allow zoom / dolly input (`onZoom`). */
  enableZoom = true;

  private readonly options: Required<CadSceneOptions>;
  /** Current (smoothed) camera spherical coordinates. */
  private readonly spherical = new THREE.Spherical();
  /** Desired camera spherical coordinates (driven by `onOrbit`). */
  private readonly sphericalTarget = new THREE.Spherical();
  private readonly clock = new THREE.Clock();
  /** Webcam-frame twin of `camera` (see `interactionCamera`). */
  private readonly interactionCam = new THREE.PerspectiveCamera();
  /** Webcam frame aspect (width / height); null until known → viewport aspect. */
  private interactionAspect: number | null = null;
  private readonly resizeObserver: ResizeObserver;

  private frameId = 0;
  private disposed = false;

  constructor(container: HTMLElement, options: CadSceneOptions = {}) {
    this.options = {
      background: options.background ?? 0xf1f5f9,
      cameraDistance: options.cameraDistance ?? 10,
      cameraAzimuth: options.cameraAzimuth ?? 0,
      cameraPolar: options.cameraPolar ?? 1.05,
      orbitSpeed: options.orbitSpeed ?? 1.75,
      rotateSpeed: options.rotateSpeed ?? 1.5,
      zoomSpeed: options.zoomSpeed ?? 1.5,
      minDistance: options.minDistance ?? 2,
      maxDistance: options.maxDistance ?? 40,
      damping: options.damping ?? 5,
      gridSize: options.gridSize ?? 40,
      gridDivisions: options.gridDivisions ?? 40,
    };

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // Filmic tone mapping: softer highlight rolloff and richer material shading.
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    // Soft shadows give committed solids a physical ground contact.
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.options.background);
    // Fog fades the grid into the background so the floor reads as infinite.
    this.scene.fog = new THREE.Fog(
      this.options.background,
      this.options.gridSize * 0.45,
      this.options.gridSize * 0.95
    );

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
    this.spherical.set(
      this.options.cameraDistance,
      this.options.cameraPolar,
      this.options.cameraAzimuth
    );
    this.sphericalTarget.copy(this.spherical);
    this.updateCameraPosition();

    // Soft studio lighting: sky/ground hemisphere fill, shadow-casting key and
    // a cool rim light from the opposite side.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xd6dee9, 0.55));
    const key = new THREE.DirectionalLight(0xfff4e0, 1.5);
    key.position.set(7, 14, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 40;
    key.shadow.camera.left = -12;
    key.shadow.camera.right = 12;
    key.shadow.camera.top = 12;
    key.shadow.camera.bottom = -12;
    key.shadow.bias = -0.0004;
    key.shadow.radius = 4;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x93b4d8, 0.35);
    fill.position.set(-8, 6, -6);
    this.scene.add(fill);

    // Infinite floor grid, hovering just above the shadow-catching ground.
    this.grid = new THREE.GridHelper(
      this.options.gridSize,
      this.options.gridDivisions,
      0x94a3b8,
      0xd3dbe5
    );
    this.grid.position.y = 0.002;
    const gridMaterial = this.grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.85;
    this.scene.add(this.grid);

    // ShadowMaterial floor: invisible except where shadows land on it.
    this.ground = new THREE.Mesh(
      new THREE.PlaneGeometry(this.options.gridSize, this.options.gridSize),
      new THREE.ShadowMaterial({ opacity: 0.28 })
    );
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);

    // CAD orientation cue: X (red) and Z (blue) axes from the origin.
    this.axes = new THREE.Group();
    this.axes.add(axisLine(new THREE.Vector3(-0.9, 0.004, 0), new THREE.Vector3(2.2, 0.004, 0), 0xef4444));
    this.axes.add(axisLine(new THREE.Vector3(0, 0.004, -0.9), new THREE.Vector3(0, 0.004, 2.2), 0x60a5fa));
    this.scene.add(this.axes);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.frameId = requestAnimationFrame(this.tick);
  }

  /** The locked orbit target (always (0, 0, 0)); a copy, so callers cannot move it. */
  get target(): THREE.Vector3 {
    return ORBIT_TARGET.clone();
  }

  /** The renderer's canvas (mounted into the container). */
  get domElement(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  /**
   * Project a world-space point through the scene camera onto a 2D canvas of
   * the given size (CSS px). Shared projection used by the SELECT-mode AR
   * mirror on the vision overlay: NDC [-1, 1] maps onto the canvas rect
   * (`x` left → right, `y` top → down, +Y up flipped). The returned `z` is
   * the NDC depth — `> 1` means the point is behind the camera (cull it).
   *
   * The camera matrices are refreshed first (`updateMatrixWorld` also
   * recomputes `matrixWorldInverse`), so projections stay valid between
   * render ticks — the damped orbit rig moves the camera every frame.
   */
  projectToCanvas(
    vector3: THREE.Vector3,
    canvasWidth: number,
    canvasHeight: number,
    camera?: THREE.PerspectiveCamera
  ): ProjectedPoint {
    // A caller-supplied camera (e.g. `interactionCamera`) is already synced.
    if (!camera) this.camera.updateMatrixWorld();
    const projected = vector3.clone().project(camera ?? this.camera);
    return ndcToCanvas(projected.x, projected.y, projected.z, canvasWidth, canvasHeight);
  }

  /**
   * Set the aspect (width / height) of the webcam frame whose device-space
   * coordinates drive interaction. Ignored unless finite and positive.
   */
  setInteractionAspect(aspect: number): void {
    if (Number.isFinite(aspect) && aspect > 0) this.interactionAspect = aspect;
  }

  /**
   * The interaction camera, synced to the view camera's live pose: same
   * position, orientation, vertical FOV and clip planes, but the webcam
   * frame's aspect — so device-space hand coordinates are exactly its NDC.
   * Sync once per use (per raycast / per AR frame), then reuse.
   */
  get interactionCamera(): THREE.PerspectiveCamera {
    const view = this.camera;
    const cam = this.interactionCam;
    view.updateMatrixWorld();
    const aspect = this.interactionAspect ?? view.aspect;
    if (cam.aspect !== aspect || cam.fov !== view.fov || cam.near !== view.near || cam.far !== view.far) {
      cam.fov = view.fov;
      cam.near = view.near;
      cam.far = view.far;
      cam.aspect = aspect;
      cam.updateProjectionMatrix();
    }
    cam.position.copy(view.position);
    cam.quaternion.copy(view.quaternion);
    cam.updateMatrixWorld(); // also refreshes matrixWorldInverse
    return cam;
  }

  /**
   * Map a device-space point (the webcam frame; [-1, 1], +Y up) onto a canvas
   * showing the view camera (the 3D viewport, CSS px): the point that the
   * interaction camera sees at that position lands at the returned pixel.
   */
  deviceToCanvas(x: number, y: number, canvasWidth: number, canvasHeight: number): ProjectedPoint {
    const aspect = this.interactionAspect ?? this.camera.aspect;
    const viewAspect = canvasHeight > 0 ? canvasWidth / canvasHeight : aspect;
    return ndcToCanvas(remapNdcX(x, aspect, viewAspect), y, 0, canvasWidth, canvasHeight);
  }

  /**
   * Orbit the camera around the origin by a device-space delta (+X right,
   * +Y up). The scene follows the hand: hand right swings the camera left,
   * hand up swings it lower. Only the azimuth / polar angles change, so the
   * camera never leaves its origin-centered sphere. Damped every frame.
   */
  onOrbit(delta: { deltaX: number; deltaY: number }): void {
    if (!this.enableRotate) return;
    this.sphericalTarget.theta -= delta.deltaX * this.options.orbitSpeed;
    this.sphericalTarget.phi = THREE.MathUtils.clamp(
      this.sphericalTarget.phi + delta.deltaY * this.options.orbitSpeed,
      POLAR_MIN,
      POLAR_MAX
    );
  }

  /**
   * Turn the scene around the vertical axis by an input angle (radians,
   * + = counter-clockwise as seen on screen). The scene follows the turn, so
   * the camera orbits the opposite way. Damped every frame.
   */
  onRotate(delta: { deltaAngle: number }): void {
    if (!this.enableRotate) return;
    this.sphericalTarget.theta -= delta.deltaAngle * this.options.rotateSpeed;
  }

  /**
   * Dolly the camera by a hand-distance ratio: `deltaScale > 1` (hands moving
   * apart) moves closer (zoom in), `< 1` (hands closer) moves away (zoom out).
   * Clamped to [minDistance, maxDistance].
   */
  onZoom(delta: { deltaScale: number }): void {
    if (!this.enableZoom) return;
    if (!(delta.deltaScale > 0) || !Number.isFinite(delta.deltaScale)) return;
    this.sphericalTarget.radius = THREE.MathUtils.clamp(
      this.sphericalTarget.radius / Math.pow(delta.deltaScale, this.options.zoomSpeed),
      this.options.minDistance,
      this.options.maxDistance
    );
  }

  /** Match the drawing buffer to the container size. */
  resize(): void {
    const container = this.renderer.domElement.parentElement;
    const width = container?.clientWidth || 1;
    const height = container?.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /** Stop the render loop and release all GPU resources. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.frameId);
    this.resizeObserver.disconnect();
    this.grid.geometry.dispose();
    (this.grid.material as THREE.Material).dispose();
    this.ground.geometry.dispose();
    (this.ground.material as THREE.Material).dispose();
    for (const line of this.axes.children) {
      (line as THREE.Line).geometry.dispose();
      ((line as THREE.Line).material as THREE.Material).dispose();
    }
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private readonly tick = (): void => {
    if (this.disposed) return;
    this.frameId = requestAnimationFrame(this.tick);
    const dt = Math.min(this.clock.getDelta(), 0.1);
    const blend = 1 - Math.exp(-this.options.damping * dt);
    this.spherical.theta += (this.sphericalTarget.theta - this.spherical.theta) * blend;
    this.spherical.phi += (this.sphericalTarget.phi - this.spherical.phi) * blend;
    this.spherical.radius += (this.sphericalTarget.radius - this.spherical.radius) * blend;
    this.updateCameraPosition();
    this.renderer.render(this.scene, this.camera);
  };

  /**
   * Place the camera on its origin-centered sphere and aim at the origin.
   * Runs every frame, so any outside write to `camera.position` is undone.
   */
  private updateCameraPosition(): void {
    this.camera.position.setFromSpherical(this.spherical);
    this.camera.lookAt(ORBIT_TARGET);
  }
}
