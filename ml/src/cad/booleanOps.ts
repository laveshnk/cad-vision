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
 * The result keeps the evaluator's interpolated normals (flat on planar
 * faces, smooth on curved ones — recomputing them would average across
 * shared vertices and smear the shading over sharp corners), and its edge
 * overlay comes from `creaseEdges`, which skips the T-junction seams CSG
 * leaves inside flat faces (a plain `EdgesGeometry` draws every such
 * unmatched triangle edge, covering cut faces in stray lines).
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

/**
 * Subtract: the cutter is grown by this fraction about its own origin before
 * cutting. Every solid rests on the ground, so a cutter's bottom face is
 * normally coplanar with the base's — and coplanar faces make BSP CSG leave
 * slivers, stray fragments and zero-thickness skins where the cut should go
 * cleanly through. The tiny growth pushes shared faces just past each other
 * (0.1 % — invisible, but far above float noise). Union is left unscaled:
 * shrinking its partner instead introduced seams on visible faces.
 */
const CUTTER_GROWTH = 1e-3;

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
  const brushB = meshToBrush(tool, operation === 'subtract' ? 1 + CUTTER_GROWTH : 1);
  try {
    const op = operation === 'subtract' ? SUBTRACTION : ADDITION;
    const result = evaluator.evaluate(brushA, brushB, op);
    const geometry = result.geometry;
    const position = geometry.getAttribute('position');
    if (!position || position.count === 0) return null; // empty solid
    // Keep the evaluator's normals (flat faces stay flat, curves smooth).
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    // CSG output has T-junctions, which rasterize with pixel-wide cracks;
    // drawing back faces too makes a crack show the solid's own dark
    // interior instead of the bright floor behind it (no sparkles).
    material.side = THREE.DoubleSide;
    material.shadowSide = THREE.BackSide; // keep the usual acne-free shadows
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(base.position);
    mesh.quaternion.copy(base.quaternion);
    mesh.scale.copy(base.scale);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // Fresh crisp edge overlay (the caller owns the child's disposal).
    const edges = new THREE.LineSegments(
      creaseEdges(geometry, CSG_EDGE_THRESHOLD),
      new THREE.LineBasicMaterial({
        color: edgeColorOf(base),
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

/**
 * A detached `Brush` mirroring a committed mesh's geometry + transform,
 * optionally grown uniformly about the mesh origin by `growth`.
 */
function meshToBrush(mesh: THREE.Mesh, growth = 1): Brush {
  const brush = new Brush(mesh.geometry, mesh.material as THREE.Material);
  brush.position.copy(mesh.position);
  brush.quaternion.copy(mesh.quaternion);
  brush.scale.copy(mesh.scale).multiplyScalar(growth);
  brush.updateMatrix();
  brush.updateMatrixWorld(true);
  return brush;
}

/**
 * Crease edges of a (CSG result) triangle mesh as a `LineSegments`-ready
 * geometry: like `EdgesGeometry`, an edge shared by two faces is drawn when
 * their normals differ by more than `thresholdDeg`. Unlike it, an edge with
 * only one face is not automatically drawn: CSG splits faces with
 * T-junctions (one side's edge A–B meets the other side's A–M + M–B), and
 * such an edge is a seam *inside* a flat face wherever the faces along its
 * collinear, overlapping partner edges are coplanar with its own. Each
 * unmatched edge is split at its partners' endpoints and only the pieces
 * that are genuine creases (or true open boundaries with no partner at all)
 * are kept.
 */
export function creaseEdges(geometry: THREE.BufferGeometry, thresholdDeg = 12): THREE.BufferGeometry {
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  const triCount = Math.floor((index ? index.count : position.count) / 3);
  const cosThreshold = Math.cos(THREE.MathUtils.degToRad(thresholdDeg));
  geometry.computeBoundingBox();
  const size = geometry.boundingBox?.getSize(new THREE.Vector3()).length() ?? 1;
  const eps = Math.max(size, 1e-6) * 1e-5;
  // Drawn pieces shorter than this are float-noise gaps between partner
  // spans (they would render as specks on flat faces), not real creases.
  const minPiece = Math.max(size, 1e-6) * 2e-3;
  const quantum = eps * 10;

  const vertex = (i: number): THREE.Vector3 => {
    const k = index ? index.getX(i) : i;
    return new THREE.Vector3(position.getX(k), position.getY(k), position.getZ(k));
  };
  const keyOf = (v: THREE.Vector3): string =>
    `${Math.round(v.x / quantum)},${Math.round(v.y / quantum)},${Math.round(v.z / quantum)}`;

  interface Edge {
    a: THREE.Vector3;
    b: THREE.Vector3;
    faces: number[];
  }
  const normals: THREE.Vector3[] = [];
  const edges = new Map<string, Edge>();
  const triangle = new THREE.Triangle();
  for (let t = 0; t < triCount; t++) {
    const corners = [vertex(t * 3), vertex(t * 3 + 1), vertex(t * 3 + 2)];
    triangle.set(corners[0], corners[1], corners[2]);
    normals.push(triangle.getNormal(new THREE.Vector3()));
    if (triangle.getArea() <= eps * eps) continue; // degenerate sliver
    const keys = corners.map(keyOf);
    for (let e = 0; e < 3; e++) {
      const ka = keys[e];
      const kb = keys[(e + 1) % 3];
      if (ka === kb) continue;
      const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      const edge = edges.get(key);
      if (edge) edge.faces.push(t);
      else edges.set(key, { a: corners[e], b: corners[(e + 1) % 3], faces: [t] });
    }
  }

  const coplanar = (f: number, g: number) => normals[f].dot(normals[g]) >= cosThreshold;
  const out: number[] = [];
  const push = (edge: Edge) => out.push(edge.a.x, edge.a.y, edge.a.z, edge.b.x, edge.b.y, edge.b.z);
  const unmatched: Edge[] = [];
  for (const edge of edges.values()) {
    if (edge.faces.length === 1) unmatched.push(edge);
    else if (edge.faces.length !== 2 || !coplanar(edge.faces[0], edge.faces[1])) push(edge);
  }

  // T-junction seams: an unmatched edge is kept only if a collinear,
  // overlapping unmatched partner belongs to a non-coplanar face (a real
  // crease) or it has no partner at all (a genuine boundary).
  const direction = new THREE.Vector3();
  const offset = new THREE.Vector3();
  const lineDistance = (edge: Edge, p: THREE.Vector3, length: number): { along: number; off: number } => {
    offset.subVectors(p, edge.a);
    const along = offset.dot(direction);
    const off = offset.addScaledVector(direction, -along).length();
    return { along: along / length, off };
  };
  for (const edge of unmatched) {
    direction.subVectors(edge.b, edge.a);
    const length = direction.length();
    if (length <= eps) continue;
    direction.divideScalar(length);
    // Spans (edge parameter [0, 1]) covered by collinear partners, tagged
    // with whether the partner's face makes a crease with this edge's face.
    const spans: { lo: number; hi: number; crease: boolean }[] = [];
    for (const other of unmatched) {
      if (other === edge) continue;
      const pa = lineDistance(edge, other.a, length);
      const pb = lineDistance(edge, other.b, length);
      if (pa.off > eps || pb.off > eps) continue; // not collinear
      const lo = Math.max(0, Math.min(pa.along, pb.along));
      const hi = Math.min(1, Math.max(pa.along, pb.along));
      if (hi - lo <= eps / length) continue; // no overlapping span
      spans.push({ lo, hi, crease: !coplanar(edge.faces[0], other.faces[0]) });
    }
    // Split at every span boundary; draw a piece when a creasing partner
    // covers it, or no partner does (genuine boundary). Pieces covered only
    // by coplanar partners are T-junction seams inside a flat face.
    const cuts = [0, 1, ...spans.flatMap((span) => [span.lo, span.hi])].sort((x, y) => x - y);
    let runStart: number | null = null;
    const flush = (end: number) => {
      if (runStart === null) return;
      if ((end - runStart) * length > minPiece) {
        const from = edge.a.clone().addScaledVector(direction, runStart * length);
        const to = edge.a.clone().addScaledVector(direction, end * length);
        push({ a: from, b: to, faces: edge.faces });
      }
      runStart = null;
    };
    for (let i = 0; i + 1 < cuts.length; i++) {
      const lo = cuts[i];
      const hi = cuts[i + 1];
      if (hi - lo <= 1e-9) continue;
      const mid = (lo + hi) / 2;
      const covering = spans.filter((span) => span.lo <= mid && mid <= span.hi);
      const draw = covering.length === 0 || covering.some((span) => span.crease);
      if (draw) runStart ??= lo;
      else flush(lo);
    }
    flush(1);
  }

  const lines = new THREE.BufferGeometry();
  lines.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
  return lines;
}

/**
 * The base's edge-overlay color (its `LineSegments` child), so a CSG result
 * keeps the same crisp outline look as every other committed solid.
 */
function edgeColorOf(mesh: THREE.Mesh): number {
  for (const child of mesh.children) {
    if (child instanceof THREE.LineSegments) {
      const material = child.material as THREE.LineBasicMaterial;
      if (material.color) return material.color.getHex();
    }
  }
  return 0x1e293b;
}
