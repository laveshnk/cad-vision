/**
 * Boolean CSG operations for the CAD builder (three-bvh-csg wrapper).
 *
 * `evaluateBoolean` carves or joins two committed solids with the
 * `three-bvh-csg` `Evaluator` (BVH-accelerated BSP CSG):
 *
 * - **subtract** (`SUBTRACTION`): the `cutter` is removed from the `base`,
 *   carving a cavity / hole — the base survives with the cutter's footprint
 *   hollowed out;
 * - **union** (`ADDITION`): both solids fuse into one continuous body.
 *
 * The evaluator's result geometry is expressed in the *base's local frame*
 * (the library assigns the base's world transform to the result brush), so
 * the returned mesh keeps the base's geometry frame, position, quaternion
 * and scale — a drop-in replacement for the base inside the committed
 * scene graph. Materials: the result wears the supplied `material` (the
 * caller picks the surviving look — the base's for a cut, either for a
 * merge).
 *
 * `overlapBox` / `worldBounds` are the clash-detection helpers: world-space
 * AABBs (`box3.intersectsBox`) plus their intersection region — the
 * translucent amber clash indicator volume. All pure math on plain
 * `three` types, so the module is unit-testable headless (no DOM / WebGL).
 */

import * as THREE from 'three';
import { ADDITION, Brush, Evaluator, SUBTRACTION } from 'three-bvh-csg';

/** Supported CSG operations (structurally mirrored by the vision overlay). */
export type BooleanOperation = 'subtract' | 'union';

/** EdgesGeometry crease threshold for CSG results (degrees, crisp seams). */
const CSG_EDGE_THRESHOLD = 12;

/** Shared, reuse-safe evaluator (single-threaded per-frame usage). */
const evaluator = new Evaluator();
// One material for the whole result (no per-source groups): committed
// meshes stay single-material — repaint / dispose keep working.
evaluator.useGroups = false;

/**
 * World-space AABB of a mesh: its (already computed) local bounding box
 * pushed through the world matrix. Cheaper than `Box3.setFromObject`
 * (no traversal) and identical for meshes without scaled children.
 */
export function worldBounds(mesh: THREE.Mesh, target = new THREE.Box3()): THREE.Box3 {
  const local = mesh.geometry.boundingBox;
  if (local) {
    target.copy(local).applyMatrix4(mesh.matrixWorld);
  } else {
    target.setFromObject(mesh);
  }
  return target;
}

/**
 * Intersection region of two world AABBs — the translucent clash indicator
 * volume — or `null` when the boxes are disjoint (or only touch / overlap
 * degenerately on an edge, which cannot be drawn).
 */
export function overlapBox(a: THREE.Box3, b: THREE.Box3, target = new THREE.Box3()): THREE.Box3 | null {
  const minX = Math.max(a.min.x, b.min.x);
  const minY = Math.max(a.min.y, b.min.y);
  const minZ = Math.max(a.min.z, b.min.z);
  const maxX = Math.min(a.max.x, b.max.x);
  const maxY = Math.min(a.max.y, b.max.y);
  const maxZ = Math.min(a.max.z, b.max.z);
  if (maxX - minX <= 1e-9 || maxY - minY <= 1e-9 || maxZ - minZ <= 1e-9) return null;
  return target.set(new THREE.Vector3(minX, minY, minZ), new THREE.Vector3(maxX, maxY, maxZ));
}

/** Whether two world AABBs clash (strictly overlapping volume). */
export function boxesClash(a: THREE.Box3, b: THREE.Box3): boolean {
  return overlapBox(a, b) !== null;
}

/**
 * Run a Boolean CSG operation between two committed meshes.
 *
 * @param base   the surviving solid — the result keeps its geometry frame
 *               and transform (for subtract: the mesh being carved; for
 *               union: one of the two merged solids).
 * @param tool   the other solid — for subtract, the cutter whose footprint
 *               is removed; for union, the merged-away partner.
 * @param operation `'subtract'` or `'union'`.
 * @param material the surviving material for the result mesh.
 * @returns a fresh mesh (new geometry + crisp `EdgesGeometry` overlay,
 *          base's transform, cast/receive shadows) — never added to a scene;
 *          `null` when the operation fails or produces an empty solid.
 */
export function evaluateBoolean(
  base: THREE.Mesh,
  tool: THREE.Mesh,
  operation: BooleanOperation,
  material: THREE.Material
): THREE.Mesh | null {
  // Detached brushes carrying the committed meshes' geometry + transform:
  // the evaluator reads `matrixWorld`, so update it after copying.
  const brushA = meshToBrush(base);
  const brushB = meshToBrush(tool);
  try {
    const op = operation === 'subtract' ? SUBTRACTION : ADDITION;
    const result = evaluator.evaluate(brushA, brushB, op);
    const geometry = result.geometry;
    const position = geometry.getAttribute('position');
    if (!position || position.count === 0) return null; // empty solid
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(base.position);
    mesh.quaternion.copy(base.quaternion);
    mesh.scale.copy(base.scale);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // Fresh crisp edge overlay (the caller owns the child's disposal).
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, CSG_EDGE_THRESHOLD),
      new THREE.LineBasicMaterial({
        color: 0x1e293b,
        transparent: true,
        opacity: 0.9,
      })
    );
    mesh.add(edges);
    return mesh;
  } catch (error) {
    // Degenerate input (coplanar / self-intersecting CSG is fussy): leave
    // the scene untouched — a failed Boolean is a no-op, never a crash.
    console.error('[cad-vision] boolean operation failed:', error);
    return null;
  }
}

/** A detached `Brush` mirroring a committed mesh's geometry + transform. */
function meshToBrush(mesh: THREE.Mesh): Brush {
  const brush = new Brush(mesh.geometry, mesh.material as THREE.Material);
  brush.position.copy(mesh.position);
  brush.quaternion.copy(mesh.quaternion);
  brush.scale.copy(mesh.scale);
  brush.updateMatrix();
  brush.updateMatrixWorld(true);
  return brush;
}
