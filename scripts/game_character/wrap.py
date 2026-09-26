"""
wrap — fit a clean base topology onto the finished sculpt ("wrapping").

Studios keep one animation-tested base topology and wrap it onto every new
sculpt or scan (R3DS Wrap, shrink-wrap + relax in Maya/Blender) instead of
retopologizing each character from scratch: the loops around the eyes,
mouth and joints stay where deformation needs them, UV seams stay in the
same hidden places, and every character built this way shares one
topology (so blend shapes, weights and UV-space masks carry over).

The template is the MakeHuman base mesh (CC0, see reference_body): 13 378
quads, closed — eye sockets and the mouth bag included — and free of
self-intersections. The high-poly was sculpted on this mesh's limit
surface, so each template vertex has an exact anchor: its Catmull-Clark
limit position and normal on the nude body. Wrapping then moves every
vertex along that normal to the *outer* surface of what was sculpted on
top (shirt, jerkin, trousers, boots, hair):

  * ray from just inside the body along the limit normal; the first exit
    through the high-poly is the garment's outer surface;
  * a hit is trusted only if it is close (garment thickness, more on the
    scalp for hair) and faces the same way; eye-socket and mouth-bag
    vertices stay on the body (they sit behind the eyeballs / lips);
  * untrusted offsets are filled by harmonic interpolation from their
    neighbours, then the offset field is lightly smoothed, so the result
    keeps the template's edge flow and never folds.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

import bl_util as U
from landmarks import LM


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

def template_object(name, cache, body=None):
    """The base mesh (shaped by the spec's body) as a Blender object with its
    UVs and UV seams marked where the template's UVs are discontinuous."""
    import os
    import reference_body as RB
    ref = RB.build(cache, body)
    V, F = ref
    VT, FT = RB.load_body_uv(os.path.join(cache, "base.obj"))
    used = np.unique(np.concatenate([np.array(f) for f in F]))
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = [[int(remap[i]) for i in f] for f in F]
    me = bpy.data.meshes.new(name)
    me.from_pydata(V[used].astype(np.float64).tolist(), [], faces)
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    lt = np.array([t for f in FT for t in f])
    uv.data.foreach_set("uv", VT[lt].reshape(-1))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    ul = bm.loops.layers.uv.active
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        f1, f2 = e.link_faces
        a = {l.vert.index: tuple(l[ul].uv) for l in f1.loops}
        b = {l.vert.index: tuple(l[ul].uv) for l in f2.loops}
        e.seam = any((Vector(a[v.index]) - Vector(b[v.index])).length > 1e-6 for v in e.verts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    return ob


def face_islands(me):
    """UV-island id per face (flood fill across non-seam edges)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    isl = -np.ones(len(bm.faces), dtype=np.int64)
    k = 0
    for f0 in bm.faces:
        if isl[f0.index] >= 0:
            continue
        stack = [f0]
        isl[f0.index] = k
        while stack:
            f = stack.pop()
            for e in f.edges:
                if e.seam:
                    continue
                for g in e.link_faces:
                    if isl[g.index] < 0:
                        isl[g.index] = k
                        stack.append(g)
        k += 1
    bm.free()
    return isl


def interior_vertices(me):
    """Vertices of the eye-socket and mouth-bag islands (hidden behind the
    eyeballs and lips): the small islands centred on the eyes / stomion."""
    isl = face_islands(me)
    fc = np.array([p.center[:] for p in me.polygons])
    inner = set()
    for k in np.unique(isl):
        sel = isl == k
        # the head's own island is thousands of faces; the sockets and the
        # mouth bag are the small islands inside the head
        if sel.sum() < 1000 and fc[sel].mean(axis=0)[2] > LM["chin_z"]:
            inner.add(int(k))
    face_in = np.isin(isl, list(inner))
    out = np.ones(len(me.vertices), dtype=bool)
    for p in me.polygons:          # a vertex is interior only if all its faces are
        if not face_in[p.index]:
            out[list(p.vertices)] = False
    return out


def limit_frame(obj):
    """Catmull-Clark limit position + normal of every control vertex (the
    subdivided mesh keeps the control vertices first, at their limit
    positions when use_limit_surface is on)."""
    n = len(obj.data.vertices)
    m = obj.modifiers.new("Limit", "SUBSURF")
    m.levels = 1
    m.use_limit_surface = True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg).to_mesh()
    P = np.array([v.co[:] for v in ev.vertices[:n]])
    N = np.array([v.normal[:] for v in ev.vertices[:n]])
    obj.evaluated_get(dg).to_mesh_clear()
    obj.modifiers.remove(m)
    return P, N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def neighbours(me):
    nb = [[] for _ in me.vertices]
    for e in me.edges:
        a, b = e.vertices
        nb[a].append(b)
        nb[b].append(a)
    return nb


def harmonic_fill(values, known, nb):
    """Unknown entries = mean of their neighbours (graph Laplace equation
    with the known entries as boundary conditions)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    n = len(values)
    unk = np.nonzero(~known)[0]
    if len(unk) == 0:
        return values.copy()
    pos = -np.ones(n, dtype=np.int64)
    pos[unk] = np.arange(len(unk))
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(unk))
    for r, i in enumerate(unk):
        rows.append(r)
        cols.append(r)
        vals.append(float(len(nb[i])))
        for j in nb[i]:
            if known[j]:
                rhs[r] += values[j]
            else:
                rows.append(r)
                cols.append(pos[j])
                vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(unk), len(unk)))
    out = values.copy()
    out[unk] = spsolve(A, rhs)
    return out


def taubin(P, nb, w, iterations=60, lam=0.5, mu=-0.53):
    """Taubin (lambda|mu) smoothing, weighted per vertex: removes small
    detail (toes, grooves) without the shrinkage of plain Laplacian
    smoothing, so the smoothed base keeps the limb's volume."""
    A = _avg_matrix(nb)
    Q = np.array(P, dtype=np.float64)
    for _ in range(iterations):
        for f in (lam, mu):
            Q = Q + (f * w)[:, None] * (A @ Q - Q)
    return Q


def smooth(values, nb, free, iterations=2, factor=0.5):
    v = values.copy()
    for _ in range(iterations):
        avg = np.array([v[n].mean() if n else v[i] for i, n in enumerate(nb)])
        v[free] = v[free] * (1 - factor) + avg[free] * factor
    return v


# ---------------------------------------------------------------------------
# Booted variant of the base mesh
# ---------------------------------------------------------------------------

def _walk_loop(e0):
    """Closed edge loop through e0 across valence-4 vertices (None if the
    loop hits a pole or does not close)."""
    loop, v, e = [e0], e0.verts[1], e0
    for _ in range(600):
        if len(v.link_edges) != 4:
            return None
        fa = {f.index for f in e.link_faces}
        nxt = [x for x in v.link_edges if x is not e and not ({f.index for f in x.link_faces} & fa)]
        if len(nxt) != 1:
            return None
        e = nxt[0]
        v = e.other_vert(v)
        if e is e0:
            return loop
        loop.append(e)
    return None


def _ordered_ring(loop):
    """Vertices of a closed edge loop in walking order."""
    vs = [loop[0].verts[0], loop[0].verts[1]]
    for e in loop[1:-1]:
        vs.append(e.other_vert(vs[-1]))
    return vs


def forefoot_ring(bm, J, side=1):
    """The last closed loop around the forefoot before the toes split (the
    metatarsal-head ring): the most distal loop of >= 24 edges behind the
    ball joint."""
    B = np.array(J["ball_l"], dtype=np.float64)
    T = np.array(J["toe_l"], dtype=np.float64)
    if side < 0:
        B[0], T[0] = -B[0], -T[0]
    ax = T - B
    ax[2] = 0.0
    ax /= np.linalg.norm(ax)
    best, seen = None, set()
    for e in bm.edges:
        c = 0.5 * (np.array(e.verts[0].co) + np.array(e.verts[1].co))
        if c[0] * side < 0.05 or abs(np.dot(c - B, ax)) > 0.05 or c[2] > 0.09 or e.index in seen:
            continue
        dv = np.array(e.verts[1].co) - np.array(e.verts[0].co)
        if abs(np.dot(dv / np.linalg.norm(dv), ax)) > 0.35:
            continue
        L = _walk_loop(e)
        if not L:
            continue
        seen |= {x.index for x in L}
        off = float(np.mean([np.dot(np.array(v.co) - B, ax) for x in L for v in x.verts]))
        if len(L) >= 24 and off < 0 and (best is None or off > best[0]):
            best = (off, L)
    return (best[1] if best else None), B, ax


def quad_cap(bm, ring, a, b):
    """Quad grid over a closed ring of 2(a+b) vertices (Coons patch of the
    ring as its four sides). Returns the new interior vertices."""
    n = len(ring)
    assert n == 2 * (a + b), (n, a, b)

    def bnd(i, j):
        if j == 0:
            return ring[i]
        if i == a:
            return ring[a + j]
        if j == b:
            return ring[a + b + (a - i)]
        return ring[(2 * a + b + (b - j)) % n]
    X = lambda v: np.array(v.co, dtype=np.float64)
    grid, new = {}, []
    for i in range(a + 1):
        for j in range(b + 1):
            if i in (0, a) or j in (0, b):
                grid[i, j] = bnd(i, j)
                continue
            u, v = i / a, j / b
            p = ((1 - v) * X(bnd(i, 0)) + v * X(bnd(i, b)) + (1 - u) * X(bnd(0, j)) + u * X(bnd(a, j))
                 - ((1 - u) * (1 - v) * X(bnd(0, 0)) + u * (1 - v) * X(bnd(a, 0))
                    + (1 - u) * v * X(bnd(0, b)) + u * v * X(bnd(a, b))))
            grid[i, j] = bm.verts.new(p.tolist())
            new.append(grid[i, j])
    for i in range(a):
        for j in range(b):
            bm.faces.new([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
    return new


def boot_feet(obj, J, log=print):
    """Studios keep a 'shoe' variant of their base mesh: toes are wasted
    (and impossible to lay out without folds) inside a boot. Cut both feet
    at the metatarsal-head ring and close them with a domed quad grid; the
    wrap then stretches that toe box over the sculpted boot."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    P0 = np.array([v.co[:] for v in bm.verts])
    from scipy.spatial import cKDTree
    twin = cKDTree(P0).query(P0 * np.array([-1.0, 1.0, 1.0]))[1]
    loop, B, ax = forefoot_ring(bm, J, side=1)
    if loop is None:
        bm.free()
        log("wrap: no forefoot ring found, feet left as they are")
        return False
    ring_l = _ordered_ring(loop)
    # start at the ring's lowest medial vertex, go laterally along the sole
    k0 = int(np.argmin([v.co.z * 4.0 + v.co.x for v in ring_l]))
    ring_l = ring_l[k0:] + ring_l[:k0]
    if ring_l[1].co.x < ring_l[-1].co.x:
        ring_l = [ring_l[0]] + ring_l[1:][::-1]
    # the right ring: the left ring's mirror twins, in the same order, so the
    # two caps come out as exact mirror images
    rings = {1: ring_l, -1: [bm.verts[int(twin[v.index])] for v in ring_l]}
    ring_edges = {e.index for e in loop}
    ring_edges |= {bm.edges.get([bm.verts[int(twin[x.verts[0].index])], bm.verts[int(twin[x.verts[1].index])]]).index
                   for x in loop}
    # faces in front of the ring (the toes): flood fill from the toe tips
    front = set()
    for side in (1, -1):
        Bs = B * np.array([side, 1, 1]) if side < 0 else B
        axs = ax * np.array([side, 1, 1]) if side < 0 else ax
        seed = max((f for f in bm.faces if f.calc_center_median().x * side > 0.05 and f.calc_center_median().z < 0.1),
                   key=lambda f: np.dot(np.array(f.calc_center_median()) - Bs, axs))
        stack = [seed]
        front.add(seed.index)
        before = len(front)
        while stack:
            f = stack.pop()
            for e in f.edges:
                if e.index in ring_edges:
                    continue
                for g in e.link_faces:
                    if g.index not in front:
                        front.add(g.index)
                        stack.append(g)
        if len(front) - before > 0.1 * len(bm.faces):      # the ring did not close the toes off
            bm.free()
            log("wrap: forefoot ring does not separate the toes, feet left as they are")
            return False
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index in front], context="FACES")
    n = len(ring_l)
    a = int(round(n / 2 * 0.62))      # wide across the foot, shallow top to sole
    b = n // 2 - a
    for side in (1, -1):
        ring = rings[side]
        new = quad_cap(bm, ring, a, b)
        # dome the cap forward so its normals fan out over the toe box
        R = np.array([v.co[:] for v in ring])
        c = R.mean(axis=0)
        rad = np.linalg.norm(R - c, axis=1).mean()
        axs = ax * np.array([side, 1, 1])
        for v in new:
            p = np.array(v.co)
            rho = np.linalg.norm((p - c) - np.dot(p - c, axs) * axs)
            v.co = Vector(p + axs * 0.7 * rad * math.sqrt(max(0.0, 1.0 - (rho / rad) ** 2)))
    bm.verts.index_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()
    log(f"wrap: feet cut at the metatarsal ring ({n} verts), toe boxes capped {a}x{b}")
    return True


# ---------------------------------------------------------------------------
# Wrap
# ---------------------------------------------------------------------------

def max_offsets(P):
    """How far the sculpted surface may stand off the body, per vertex, from
    the same region masks that sculpted the costume: bare skin 0 (it is the
    template's own limit surface; a ray from a lip crease passes through the
    closed lips), garments up to ~2.5 cm, hair ~4 cm."""
    import costume
    if not costume.REGIONS:              # nothing sculpted on this body
        return np.full(len(P), 0.004)
    rid = costume.region_id(np.asarray(P, np.float32), LM["J"])
    kinds = np.array([costume.region_kind(r) for r in range(len(costume.region_names()))])
    k = kinds[rid]
    dm = np.full(len(P), 0.028)
    dm[k == "skin"] = 0.0          # bare skin *is* the template's limit surface: no ray
    dm[k == "hair"] = 0.045 * LM.get("s_head", 1.0)
    return dm


def outer_offsets(P, N, bvh, inner, dmax, start=0.004):
    """Offset along the limit normal to the outer high-poly surface, and
    whether it can be trusted (a hit within dmax that faces the same way)."""
    d = np.zeros(len(P))
    ok = np.zeros(len(P), dtype=bool)
    for i, (p, n) in enumerate(zip(P, N)):
        dmax_i = dmax[i]
        if inner[i] or dmax_i <= 0.0:
            ok[i] = True             # socket / mouth bag / bare skin: stays on the body
            continue
        hit, hn, _f, dist = bvh.ray_cast(Vector(p - n * start), Vector(n), dmax_i + start)
        if hit is None or Vector(n).dot(hn) < 0.2:
            continue
        d[i] = dist - start
        ok[i] = d[i] > -start + 1e-4
    return d, ok


def mirror_map(P, tol=0.002):
    """Index of each vertex's mirror twin across X (None if not symmetric)."""
    from scipy.spatial import cKDTree
    Q = P * np.array([-1.0, 1.0, 1.0])
    dist, idx = cKDTree(P).query(Q)
    return idx if float(np.max(dist)) < tol else None


def _avg_matrix(nb):
    import scipy.sparse as sp
    rows = np.concatenate([np.full(len(n), i) for i, n in enumerate(nb)])
    cols = np.concatenate([np.asarray(n, dtype=np.int64) for n in nb])
    deg = np.array([max(len(n), 1) for n in nb], dtype=np.float64)
    return sp.csr_matrix((1.0 / deg[rows], (rows, cols)), shape=(len(nb), len(nb)))


def _set_co(me, co):
    me.vertices.foreach_set("co", co.reshape(-1))
    me.update()


def _normals(me):
    n = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("normal", n)
    return n.reshape(-1, 3)


def relax(obj, bvh, w, iterations=12, step=0.6):
    """Tangential Laplacian relax + snap back onto the high-poly, weighted
    per vertex: evens out template detail that the sculpt covered up (toes
    inside a boot, the cleft bridged by trousers) instead of letting it
    crumple on the covering surface."""
    me = obj.data
    A = _avg_matrix(neighbours(me))
    co = np.array([v.co[:] for v in me.vertices])
    act = np.nonzero(w > 0)[0]
    for _ in range(iterations):
        n = _normals(me)
        lap = A @ co - co
        lap -= np.einsum("ij,ij->i", lap, n)[:, None] * n
        co = co + (step * w)[:, None] * lap
        for i in act:
            co[i] = bvh.find_nearest(Vector(co[i]))[0]
        _set_co(me, co)
    return co


def intersecting_verts(obj):
    """Vertices of faces that cut through faces of the same mesh piece."""
    from mathutils.bvhtree import BVHTree
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    t = BVHTree.FromBMesh(bm)
    out = set()
    n = 0
    for a, b in t.overlap(t):
        fa, fb = bm.faces[a], bm.faces[b]
        va, vb = {v.index for v in fa.verts}, {v.index for v in fb.verts}
        if a < b and not (va & vb):
            out |= va | vb
            n += 1
    bm.free()
    return np.array(sorted(out), dtype=np.int64), n


def dilate(mask, nb, rings=2):
    m = mask.copy()
    for _ in range(rings):
        m = m | np.array([m[n].any() if n else False for n in nb])
    return m


def symmetrize(obj, mi):
    me = obj.data
    co = np.array([v.co[:] for v in me.vertices])
    co = 0.5 * (co + co[mi] * np.array([-1.0, 1.0, 1.0]))
    _set_co(me, co)


def keep_sides(obj, P, eps=0.0006):
    """A vertex never crosses the mirror plane to the other side (where the
    sculpt bridges the midline — trousers over the gluteal cleft — offsets
    from both sides would otherwise meet and pass through each other). With
    the left side kept at x >= eps and the right at x <= -eps, the two
    halves of a symmetric mesh cannot intersect."""
    me = obj.data
    co = np.array([v.co[:] for v in me.vertices])
    side = np.sign(np.where(np.abs(P[:, 0]) > 1e-5, P[:, 0], 0.0))
    bad = (side != 0) & (co[:, 0] * side < eps)
    co[bad, 0] = side[bad] * eps
    _set_co(me, co)
    return int(bad.sum())


def wrap(obj, hp, log=print, repair_rounds=0):
    from mathutils.bvhtree import BVHTree
    me = obj.data
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(hp, dg)
    P, N = limit_frame(obj)
    inner = interior_vertices(me)
    nb = neighbours(me)
    dmax = max_offsets(P)
    d, ok = outer_offsets(P, N, bvh, inner, dmax)
    # lids and lips (two rings around the sockets and the mouth bag) stay on
    # the template's limit surface: rays from a lip crease pass through the
    # closed lips of the sculpt and would push one lip into the other
    keep = dilate(inner, nb, rings=2)
    d[keep] = 0.0
    ok[keep] = True
    log(f"wrap: {int(ok.sum())}/{len(ok)} offsets from rays, {int(inner.sum())} interior verts kept on the body")
    d = harmonic_fill(d, ok, nb)
    d = smooth(d, nb, ~inner, iterations=2, factor=0.5)
    # the template is mirror-symmetric: keep the game mesh symmetric too
    # (asymmetric sculpt noise lives in the normal map), which also lets the
    # LOD decimation work symmetrically
    mi = mirror_map(P)
    if mi is not None:
        d = 0.5 * (d + d[mi])
    else:
        log("wrap: template not mirror-symmetric, offsets left as is")
    # Where the sculpt covers template detail (boots over toes, trousers over
    # the cleft) rays from that detail fan out and land out of order on the
    # covering surface. There the template is smoothed first (toes merge into
    # a sock, the cleft fills in) and rays are cast from the smooth base.
    # Vertices next to the socket / mouth bags stay put (lid and lip lines).
    import costume
    w = np.clip((d - 0.003) / 0.004, 0.0, 1.0)
    if costume.REGIONS:
        rid = costume.region_id(np.asarray(P, np.float32), LM["J"])
        kinds = np.array([costume.region_kind(r) for r in range(len(costume.region_names()))])
        w[np.isin(kinds[rid], ("footwear", "sole"))] = 1.0      # toes inside the boots
    w = np.maximum(w, dilate(w > 0.5, nb, rings=3).astype(np.float64))
    w[keep] = 0.0
    Ps = taubin(P, nb, w, iterations=60)
    _set_co(me, Ps)
    Ns = _normals(me)
    cov = w > 0
    d2, ok2 = outer_offsets(Ps, Ns, bvh, inner, dmax + np.linalg.norm(Ps - P, axis=1) + 0.004)
    base, dirs = P.copy(), N.copy()
    base[cov], dirs[cov] = Ps[cov], Ns[cov]
    d[cov] = d2[cov]
    good = ~cov | ok2
    log(f"wrap: {int(cov.sum())} covered verts re-cast from a smoothed base, {int((~good).sum())} filled")
    d = harmonic_fill(d, good, nb)
    if mi is not None:
        d = 0.5 * (d + d[mi])
    co = base + dirs * d[:, None]
    _set_co(me, co)
    if mi is not None:
        symmetrize(obj, mi)
        n = keep_sides(obj, P)
        if n:
            log(f"wrap: {n} verts held on their side of the mirror plane")
    # optional: relax around any remaining fold (off by default: relaxing a
    # fold in place tends to oscillate rather than resolve it)
    for r in range(repair_rounds):
        vi, npairs = intersecting_verts(obj)
        log(f"wrap: round {r}: {npairs} self-intersecting face pairs")
        if npairs == 0:
            break
        m = np.zeros(len(P), dtype=bool)
        m[vi] = True
        wl = dilate(m, nb, rings=2 + r // 2).astype(np.float64)
        wl[keep & ~m] = 0.0
        relax(obj, bvh, wl, iterations=6)
        if mi is not None:
            symmetrize(obj, mi)
    return d


# ---------------------------------------------------------------------------
# Part labels for skinning (retopo.TRUNK/ARM/HAND/LEG/FINGER convention)
# ---------------------------------------------------------------------------

def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def part_labels(obj, J):
    """Body-part id per vertex from the nearest bone segment (arms and legs
    slightly penalized at their roots), cleaned by a few rounds of majority
    vote over the mesh so each part is one connected patch."""
    import humanoid
    from retopo import TRUNK, ARM, HAND, LEG, FINGER, FINGER_NAMES, lab
    me = obj.data
    co = np.array([v.co[:] for v in me.vertices])
    chains = humanoid.hand_chains(J)
    segs = []   # (label, a, b, bias)

    def m(p, side):
        q = np.array(p, dtype=np.float64)
        if side < 0:
            q[0] = -q[0]
        return q
    spine = ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head", "head_top"]
    for a, b in zip(spine[:-1], spine[1:]):
        segs.append((TRUNK, np.asarray(J[a], float), np.asarray(J[b], float), 1.0))
    for side in (1, -1):
        segs.append((TRUNK, m(J["clavicle_l"], side), m(J["upperarm_l"], side), 1.0))
        segs.append((lab(ARM, side), m(J["upperarm_l"], side), m(J["lowerarm_l"], side), 1.35))
        segs.append((lab(ARM, side), m(J["lowerarm_l"], side), m(J["hand_l"], side), 1.0))
        for f in ("index", "middle", "ring", "pinky"):
            segs.append((lab(HAND, side), m(J["hand_l"], side), m(chains[f][0][0], side), 1.0))
        for k, f in enumerate(FINGER_NAMES):
            pts = chains[f][0]
            for i in range(len(pts) - 1):
                if f == "thumb" and i == 0:
                    segs.append((lab(HAND, side), m(pts[0], side), m(pts[1], side), 1.0))
                    continue
                segs.append((lab(FINGER, side, k), m(pts[i], side), m(pts[i + 1], side), 1.0))
        segs.append((lab(LEG, side), m(J["thigh_l"], side), m(J["calf_l"], side), 1.25))
        segs.append((lab(LEG, side), m(J["calf_l"], side), m(J["foot_l"], side), 1.0))
        segs.append((lab(LEG, side), m(J["foot_l"], side), m(J["ball_l"], side), 1.0))
        segs.append((lab(LEG, side), m(J["ball_l"], side), m(J["toe_l"], side), 1.0))
    D = np.stack([_seg_dist(co, a, b) * bias for _l, a, b, bias in segs], axis=1)
    labels = np.array([segs[k][0] for k in np.argmin(D, axis=1)])
    # a vertex never belongs to the other side's limb
    side = np.where(co[:, 0] >= 0, 1, -1)
    kind = labels // 10 * 10
    wrong = (kind > 0) & (((labels % 10) < 5) != (side > 0)) & (np.abs(co[:, 0]) > 0.02)
    labels[wrong] = TRUNK
    nb = neighbours(me)
    for _ in range(4):
        new = labels.copy()
        for i, n in enumerate(nb):
            vals, cnt = np.unique(labels[n + [i]], return_counts=True)
            new[i] = vals[np.argmax(cnt)]
        labels = new
    attr = me.attributes.get("part") or me.attributes.new("part", "INT", "POINT")
    attr.data.foreach_set("value", labels.astype(np.int32))
    return labels


# ---------------------------------------------------------------------------
# LOD chain from the wrapped base mesh
# ---------------------------------------------------------------------------

def build_lods(J, hp_obj, cache, name="SK_Character", hp_pieces=None, log=print):
    """Wrapped template = LOD0 (quads, template UV seams, own ABF unwrap and
    atlas packing together with the separate pieces); LOD2 / LOD4 are
    collapse-decimated from it (UVs carried along). Pieces get their LODs
    from piece_lowpoly. Returns (None, lod4, lod2, lod0)."""
    import costume
    import lods
    import pieces as PC
    import piece_lowpoly
    import spec as SP
    import uvs
    lod0 = template_object(f"{name}_LOD0", cache, SP.SPEC.get("body") if SP.SPEC else None)
    if "footwear" in costume.GEAR:          # boots and shoes both hide the toes
        boot_feet(lod0, J, log=log)
    wrap(lod0, hp_obj, log=log)
    part_labels(lod0, J)
    states, extras = [], []
    for p in PC.PIECES:
        hp = (hp_pieces or {}).get(p["name"])
        if hp is None:
            continue
        o, st = piece_lowpoly.prepare(p, hp, name, log=log)
        states.append((p, hp, st))
        extras.append(o)
    uvs.unwrap(lod0, extras=extras)
    for p, hp, st in states:
        out = piece_lowpoly.finish(p, hp, name, st)
        log(f"piece {p['name']}: " + ", ".join(f"{k}={len(o.data.polygons)} faces" for k, o in sorted(out.items())))
    lod2 = lods.decimated(lod0, f"{name}_LOD2", 0.25)
    lod4 = lods.decimated(lod0, f"{name}_LOD4", 0.0625)
    for o in (lod0, lod2, lod4):
        o.data.shade_smooth()
    return None, lod4, lod2, lod0
