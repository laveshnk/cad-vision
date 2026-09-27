import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { boxesClash, creaseEdges, evaluateBoolean, overlapBox, worldBounds } from '../booleanOps';

/**
 * A committed-mesh stand-in: a cube with a computed bounding box, placed in
 * world space (the CadBuilder commits meshes exactly like this — geometry
 * centered on the local origin, world placement via `position`).
 */
function cube(size: number, position: [number, number, number]): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(size, size, size);
  geometry.computeBoundingBox();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial());
  mesh.position.set(...position);
  mesh.updateMatrix();
  mesh.updateMatrixWorld(true);
  return mesh;
}

describe('worldBounds', () => {
  it('returns the local bounding box pushed through the world matrix', () => {
    const mesh = cube(2, [1, 1, 1]);
    const bounds = worldBounds(mesh);
    expect(bounds.min.x).toBeCloseTo(0, 10);
    expect(bounds.min.y).toBeCloseTo(0, 10);
    expect(bounds.min.z).toBeCloseTo(0, 10);
    expect(bounds.max.x).toBeCloseTo(2, 10);
    expect(bounds.max.y).toBeCloseTo(2, 10);
    expect(bounds.max.z).toBeCloseTo(2, 10);
  });

  it('covers rotated placements with a world-space AABB', () => {
    const mesh = cube(2, [0, 0, 0]);
    mesh.rotation.y = Math.PI / 4; // 45°: the cube's footprint becomes √2 × √2
    mesh.updateMatrix();
    mesh.updateMatrixWorld(true);
    const bounds = worldBounds(mesh);
    expect(bounds.min.x).toBeCloseTo(-Math.SQRT2, 10);
    expect(bounds.max.x).toBeCloseTo(Math.SQRT2, 10);
  });
});

describe('overlapBox / boxesClash', () => {
  it('returns the intersection region of two overlapping boxes', () => {
    const a = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    const b = new THREE.Box3(new THREE.Vector3(0, -1, -1), new THREE.Vector3(2, 1, 1));
    const overlap = overlapBox(a, b);
    expect(overlap).not.toBeNull();
    expect(overlap?.min.x).toBeCloseTo(0, 10);
    expect(overlap?.max.x).toBeCloseTo(1, 10);
    expect(overlap?.min.y).toBeCloseTo(-1, 10);
    expect(overlap?.max.y).toBeCloseTo(1, 10);
  });

  it('reports a clash only for strictly overlapping volume', () => {
    const a = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(0, 0, 0));
    const overlapping = new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(1, 1, 1));
    const touching = new THREE.Box3(new THREE.Vector3(0, -1, -1), new THREE.Vector3(2, 1, 1));
    const disjoint = new THREE.Box3(new THREE.Vector3(5, 5, 5), new THREE.Vector3(6, 6, 6));
    expect(boxesClash(a, overlapping)).toBe(true);
    expect(boxesClash(a, touching)).toBe(false); // zero-width overlap: nothing to draw
    expect(boxesClash(a, disjoint)).toBe(false);
    expect(overlapBox(a, touching)).toBeNull();
    expect(overlapBox(a, disjoint)).toBeNull();
  });
});

describe('evaluateBoolean — subtract (cut hole)', () => {
  it('carves the cutter out of the base and keeps the base frame', () => {
    const base = cube(2, [0, 0, 0]); // spans [-1, 1]³
    const cutter = cube(2, [1, 0, 0]); // spans [0, 2] × [-1, 1]²
    const result = evaluateBoolean(base, cutter, 'subtract', base.material as THREE.Material);
    expect(result).not.toBeNull();
    expect(result).toBeInstanceOf(THREE.Mesh);
    // Fresh geometry (a copy, never the operands') with real content…
    expect(result?.geometry).not.toBe(base.geometry);
    expect(result?.geometry.getAttribute('position').count).toBeGreaterThan(0);
    // …expressed in the base's frame: the survivor is x ∈ [-1, 0].
    const bounds = result?.geometry.boundingBox;
    expect(bounds?.min.x).toBeCloseTo(-1, 3);
    // The cutter is grown 0.1 % (coplanar-face guard), so the cut face sits
    // a hair past x = 0.
    expect(bounds?.max.x).toBeCloseTo(0, 2);
    expect(bounds?.max.x).toBeLessThanOrEqual(0);
    expect(bounds?.min.y).toBeCloseTo(-1, 3);
    expect(bounds?.max.y).toBeCloseTo(1, 3);
    // Base transform inherited; crisp EdgesGeometry overlay regenerated.
    expect(result?.position.equals(base.position)).toBe(true);
    const edges = result?.children.find((child) => child instanceof THREE.LineSegments);
    expect(edges).toBeDefined();
  });

  it('returns null when the cutter consumes the whole base (empty result)', () => {
    const base = cube(2, [0, 0, 0]);
    const cutter = cube(4, [0, 0, 0]); // fully covers the base
    const result = evaluateBoolean(base, cutter, 'subtract', base.material as THREE.Material);
    expect(result).toBeNull();
  });

  it('leaves the base intact when the operands are disjoint', () => {
    const base = cube(2, [0, 0, 0]);
    const cutter = cube(2, [10, 0, 0]);
    const result = evaluateBoolean(base, cutter, 'subtract', base.material as THREE.Material);
    // No overlap: subtracting nothing still yields the base solid (the app
    // only reaches evaluateBoolean through a live clash, so this is a
    // defensive path — the base must survive unchanged).
    expect(result).not.toBeNull();
    const bounds = result?.geometry.boundingBox;
    expect(bounds?.min.x).toBeCloseTo(-1, 3);
    expect(bounds?.max.x).toBeCloseTo(1, 3);
  });
});

describe('evaluateBoolean — union (merge)', () => {
  it('fuses both solids into one continuous body spanning both', () => {
    const base = cube(2, [0, 0, 0]); // spans [-1, 1]³
    const partner = cube(2, [1, 0, 0]); // spans [0, 2] × [-1, 1]²
    const result = evaluateBoolean(base, partner, 'union', base.material as THREE.Material);
    expect(result).not.toBeNull();
    const bounds = result?.geometry.boundingBox;
    // The merged body spans the full union: x ∈ [-1, 2], y/z ∈ [-1, 1].
    expect(bounds?.min.x).toBeCloseTo(-1, 3);
    expect(bounds?.max.x).toBeCloseTo(2, 3);
    expect(bounds?.min.y).toBeCloseTo(-1, 3);
    expect(bounds?.max.y).toBeCloseTo(1, 3);
    // One single mesh replaces both entities (no groups / material arrays).
    expect(Array.isArray(result?.material)).toBe(false);
  });
});

/** Triangle soup (non-indexed) from [x, y, z] corner triples. */
function soup(triangles: [number, number, number][][]): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(triangles.flat(2), 3));
  return geometry;
}

/** Line segments of a creaseEdges result as [[ax, ay, az], [bx, by, bz]] pairs. */
function segmentsOf(lines: THREE.BufferGeometry): number[][][] {
  const p = lines.getAttribute('position');
  const out: number[][][] = [];
  for (let i = 0; i < p.count; i += 2) {
    out.push([
      [p.getX(i), p.getY(i), p.getZ(i)],
      [p.getX(i + 1), p.getY(i + 1), p.getZ(i + 1)],
    ]);
  }
  return out;
}

describe('creaseEdges', () => {
  it('finds exactly the 12 edges of a cube (face diagonals skipped)', () => {
    expect(segmentsOf(creaseEdges(new THREE.BoxGeometry(2, 2, 2)))).toHaveLength(12);
  });

  it('skips a T-junction seam inside a flat face', () => {
    // One triangle's edge (0,0)-(2,0) meets two triangles split at (1,0).
    const flat = soup([
      [[0, 0, 0], [2, 0, 0], [1, 1, 0]],
      [[0, 0, 0], [1, -1, 0], [1, 0, 0]],
      [[1, 0, 0], [1, -1, 0], [2, 0, 0]],
    ]);
    const segments = segmentsOf(creaseEdges(flat));
    // Only the 4 outer boundary edges — nothing along the y = 0 seam.
    expect(segments).toHaveLength(4);
    expect(segments.some(([a, b]) => Math.abs(a[1]) < 1e-6 && Math.abs(b[1]) < 1e-6)).toBe(false);
  });

  it('keeps a T-junction edge that is a real crease', () => {
    // Same layout, but the lower half folds down: y = 0 is now a crease.
    const folded = soup([
      [[0, 0, 0], [2, 0, 0], [1, 1, 0]],
      [[0, 0, 0], [1, -1, -1], [1, 0, 0]],
      [[1, 0, 0], [1, -1, -1], [2, 0, 0]],
    ]);
    const onSeam = segmentsOf(creaseEdges(folded)).filter(
      ([a, b]) => Math.abs(a[1]) < 1e-6 && Math.abs(b[1]) < 1e-6 && Math.abs(a[2]) < 1e-6
    );
    expect(onSeam.length).toBeGreaterThan(0);
  });

  it('draws only box edges and hole rims on a drilled cube (no seams on the cut faces)', () => {
    const base = cube(2, [0, 0, 0]);
    const drill = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 4, 48));
    drill.updateMatrixWorld(true);
    const result = evaluateBoolean(base, drill, 'subtract', new THREE.MeshStandardMaterial());
    expect(result).not.toBeNull();
    const edges = result?.children[0] as THREE.LineSegments;
    const segments = segmentsOf(edges.geometry);
    expect(segments.length).toBeGreaterThan(12);
    const onFeature = (p: number[]) => {
      const atFace = p.filter((c) => Math.abs(Math.abs(c) - 1) < 1e-3).length;
      const onRim = Math.abs(Math.hypot(p[0], p[2]) - 0.4) < 1e-2 && Math.abs(Math.abs(p[1]) - 1) < 1e-3;
      return atFace >= 2 || onRim;
    };
    for (const [a, b] of segments) {
      const mid = a.map((v, i) => (v + b[i]) / 2);
      expect(onFeature(a) && onFeature(b) && onFeature(mid), JSON.stringify([a, b])).toBe(true);
    }
  });

  it('keeps the evaluator normals (flat box faces stay axis-aligned)', () => {
    const base = cube(2, [0, 0, 0]);
    const drill = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 4, 48));
    drill.updateMatrixWorld(true);
    const result = evaluateBoolean(base, drill, 'subtract', new THREE.MeshStandardMaterial());
    const geometry = result?.geometry as THREE.BufferGeometry;
    const position = geometry.getAttribute('position');
    const normal = geometry.getAttribute('normal');
    // Every vertex on the top face (y = 1, off the hole rim) faces straight up.
    for (let i = 0; i < position.count; i++) {
      const onTop = Math.abs(position.getY(i) - 1) < 1e-4;
      const onRim = Math.abs(Math.hypot(position.getX(i), position.getZ(i)) - 0.4) < 1e-2;
      if (!onTop || onRim) continue;
      if (normal.getY(i) < 0.5) continue; // a side-face vertex sharing the top edge
      expect(normal.getY(i)).toBeCloseTo(1, 3);
    }
  });
});

/**
 * Stray-edge audit: a drawn edge is stray when every triangle touching its
 * midpoint is coplanar (within the 12° crease threshold) — i.e. a line lying
 * flat inside a face instead of on a real crease.
 */
function strayEdges(mesh: THREE.Mesh): number {
  const geometry = mesh.geometry;
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  const count = (index ? index.count : position.count) / 3;
  const corner = (i: number) => {
    const k = index ? index.getX(i) : i;
    return new THREE.Vector3(position.getX(k), position.getY(k), position.getZ(k));
  };
  const triangles: { triangle: THREE.Triangle; normal: THREE.Vector3 }[] = [];
  for (let t = 0; t < count; t++) {
    const triangle = new THREE.Triangle(corner(3 * t), corner(3 * t + 1), corner(3 * t + 2));
    if (triangle.getArea() < 1e-10) continue;
    triangles.push({ triangle, normal: triangle.getNormal(new THREE.Vector3()) });
  }
  const lines = (mesh.children[0] as THREE.LineSegments).geometry.getAttribute('position');
  const cosThreshold = Math.cos(THREE.MathUtils.degToRad(12));
  let stray = 0;
  for (let i = 0; i < lines.count; i += 2) {
    const mid = new THREE.Vector3(
      (lines.getX(i) + lines.getX(i + 1)) / 2,
      (lines.getY(i) + lines.getY(i + 1)) / 2,
      (lines.getZ(i) + lines.getZ(i + 1)) / 2
    );
    const touching = triangles.filter(
      (t) => t.triangle.closestPointToPoint(mid, new THREE.Vector3()).distanceTo(mid) < 1e-3
    );
    if (touching.length === 0) continue;
    const creased = touching.some((t) => t.normal.dot(touching[0].normal) < cosThreshold);
    if (!creased) stray++;
  }
  return stray;
}

/** A primitive mesh resting on the floor (y = 0), like a committed build. */
function solid(geometry: THREE.BufferGeometry, x: number, z: number): THREE.Mesh {
  geometry.computeBoundingBox();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial());
  const bottom = geometry.boundingBox?.min.y ?? 0;
  mesh.position.set(x, -bottom, z);
  mesh.updateMatrixWorld(true);
  return mesh;
}

describe('evaluateBoolean — no stray edges on cut faces', () => {
  const cut = (base: THREE.Mesh, cutter: THREE.Mesh) => {
    const result = evaluateBoolean(base, cutter, 'subtract', new THREE.MeshStandardMaterial());
    expect(result).not.toBeNull();
    result?.updateMatrixWorld(true);
    return result as THREE.Mesh;
  };

  it('cylinder through a box resting on the floor', () => {
    const base = solid(new THREE.BoxGeometry(3, 2, 3), 0, 0);
    const result = cut(base, solid(new THREE.CylinderGeometry(0.6, 0.6, 3, 48), 0.8, 0));
    expect(strayEdges(result)).toBe(0);
  });

  it('sphere biting a box corner', () => {
    const base = solid(new THREE.BoxGeometry(3, 2, 3), 0, 0);
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.8, 48, 24));
    sphere.position.set(1.4, 1.9, 1.4);
    sphere.updateMatrixWorld(true);
    expect(strayEdges(cut(base, sphere))).toBe(0);
  });

  it('two successive cuts (cutting a CSG result again)', () => {
    const base = solid(new THREE.BoxGeometry(3, 2, 3), 0, 0);
    const once = cut(base, solid(new THREE.CylinderGeometry(0.5, 0.5, 3, 48), 0.8, 0));
    const twice = cut(once, solid(new THREE.CylinderGeometry(0.4, 0.4, 3, 48), -0.7, 0.5));
    expect(strayEdges(twice)).toBe(0);
  });

  it('cylinder biting into a side face', () => {
    const base = solid(new THREE.BoxGeometry(3, 2, 3), 0, 0);
    const result = cut(base, solid(new THREE.CylinderGeometry(0.5, 0.5, 1.2, 48), 1.5, 0));
    expect(strayEdges(result)).toBe(0);
  });
});
