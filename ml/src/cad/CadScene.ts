/**
 * CadScene: self-contained Three.js CAD viewport.
 *
 * Owns the WebGLRenderer, a PerspectiveCamera on a damped spherical orbit
 * rig, soft studio lighting (hemisphere + shadow-casting key light), a
 * shadow-catching floor and an "infinite" grid faded into the background by
 * fog. External control inputs are plain device-space data — `onPan({ deltaX, deltaY })`
 * (truck the camera sideways / vertically), `onZoom({ deltaScale })` (dolly in / out) and
 * `onOrbit({ deltaX, deltaY })` — keeping this module fully decoupled from the
 * vision layer.
 */

import * as THREE from 'three';

export interface CadSceneOptions {
  /** Background & fog color. */
  background?: number;
  /** Camera distance from the orbit target (world units). */
  cameraDistance?: number;
  /** Camera azimuth around +Y (radians). */
  cameraAzimuth?: number;
  /** Camera polar angle from +Y (radians); clamped to stay above the floor. */
  cameraPolar?: number;
  /** Orbit target height above the ground plane. */
  targetHeight?: number;
  /** Radians of orbit per device-space unit (hand travel). */
  orbitSpeed?: number;
  /** World units of sideways pan per device-space unit, per unit of camera distance. */
  panSpeed?: number;
  /** Exponent applied to zoom ratios (higher = more zoom per hand movement). */
  zoomSpeed?: number;
  /** Closest allowed camera distance (world units). */
  minDistance?: number;
  /** Farthest allowed camera distance (world units). */
  maxDistance?: number;
  /** Exponential damping rate for camera motion (higher = snappier). */
  damping?: number;
  /** Ground grid extent (world units). */
  gridSize?: number;
  /** Ground grid divisions. */
  gridDivisions?: number;
}

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

  private readonly options: Required<CadSceneOptions>;
  /** Current (smoothed) orbit target. */
  private readonly target = new THREE.Vector3();
  /** Desired orbit target (driven by `onPan`). */
  private readonly targetGoal = new THREE.Vector3();
  private readonly panRight = new THREE.Vector3();
  /** Current (smoothed) camera spherical coordinates. */
  private readonly spherical = new THREE.Spherical();
  /** Desired camera spherical coordinates (driven by `onOrbit`). */
  private readonly sphericalTarget = new THREE.Spherical();
  private readonly clock = new THREE.Clock();
  private readonly resizeObserver: ResizeObserver;

  private frameId = 0;
  private disposed = false;

  constructor(container: HTMLElement, options: CadSceneOptions = {}) {
    this.options = {
      background: options.background ?? 0x0b0e14,
      cameraDistance: options.cameraDistance ?? 10,
      cameraAzimuth: options.cameraAzimuth ?? 0.7,
      cameraPolar: options.cameraPolar ?? 1.05,
      targetHeight: options.targetHeight ?? 0.5,
      orbitSpeed: options.orbitSpeed ?? 1.75,
      panSpeed: options.panSpeed ?? 0.6,
      zoomSpeed: options.zoomSpeed ?? 1.5,
      minDistance: options.minDistance ?? 2,
      maxDistance: options.maxDistance ?? 40,
      damping: options.damping ?? 9,
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
    this.target.set(0, this.options.targetHeight, 0);
    this.targetGoal.copy(this.target);
    this.spherical.set(
      this.options.cameraDistance,
      this.options.cameraPolar,
      this.options.cameraAzimuth
    );
    this.sphericalTarget.copy(this.spherical);
    this.updateCameraPosition();

    // Soft studio lighting: sky/ground hemisphere fill, shadow-casting key and
    // a cool rim light from the opposite side.
    this.scene.add(new THREE.HemisphereLight(0x9db8dd, 0x0d1220, 0.5));
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
      0x5b6b85,
      0x232e3d
    );
    this.grid.position.y = 0.002;
    const gridMaterial = this.grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.85;
    this.scene.add(this.grid);

    // ShadowMaterial floor: invisible except where shadows land on it.
    this.ground = new THREE.Mesh(
      new THREE.PlaneGeometry(this.options.gridSize, this.options.gridSize),
      new THREE.ShadowMaterial({ opacity: 0.35 })
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

  /** The renderer's canvas (mounted into the container). */
  get domElement(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  /**
   * Apply an orbit delta in device-space units (+X right, +Y up) to the
   * camera rig. Motion is damped toward the target every frame.
   */
  onOrbit(delta: { deltaX: number; deltaY: number }): void {
    this.sphericalTarget.theta -= delta.deltaX * this.options.orbitSpeed;
    this.sphericalTarget.phi = THREE.MathUtils.clamp(
      this.sphericalTarget.phi + delta.deltaY * this.options.orbitSpeed,
      POLAR_MIN,
      POLAR_MAX
    );
  }

  /**
   * "Grab the scene" pan by a device-space delta (+X right, +Y up): the scene
   * follows the hand, so the camera trucks the opposite way (hand right =
   * camera left, hand up = camera down). Scales with camera distance so the
   * pan feels the same when zoomed in or out. Motion is damped every frame.
   */
  onPan(delta: { deltaX: number; deltaY: number }): void {
    const theta = this.sphericalTarget.theta;
    const scale = this.options.panSpeed * this.sphericalTarget.radius;
    // Camera right vector on the ground plane for azimuth `theta`.
    this.panRight.set(Math.cos(theta), 0, -Math.sin(theta));
    this.targetGoal.addScaledVector(this.panRight, -delta.deltaX * scale);
    this.targetGoal.y -= delta.deltaY * scale;
  }

  /**
   * Dolly the camera by a hand-distance ratio: `deltaScale > 1` (hands moving
   * apart) moves closer (zoom in), `< 1` (hands closer) moves away (zoom out).
   * Clamped to [minDistance, maxDistance].
   */
  onZoom(delta: { deltaScale: number }): void {
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
    this.target.lerp(this.targetGoal, blend);
    this.updateCameraPosition();
    this.renderer.render(this.scene, this.camera);
  };

  private updateCameraPosition(): void {
    this.camera.position.setFromSpherical(this.spherical).add(this.target);
    this.camera.lookAt(this.target);
  }
}
