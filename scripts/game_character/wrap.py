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

def template_object(name, cache):
    """The base mesh as a Blender object with its UVs, UV seams marked where
    the template's UVs are discontinuous, and per-vertex 'island' ids."""
    import os
    import reference_body as RB
    ref = RB.build(cache)
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


def smooth(values, nb, free, iterations=2, factor=0.5):
    v = values.copy()
    for _ in range(iterations):
        avg = np.array([v[n].mean() if n else v[i] for i, n in enumerate(nb)])
        v[free] = v[free] * (1 - factor) + avg[free] * factor
    return v


# ---------------------------------------------------------------------------
# Wrap
# ---------------------------------------------------------------------------

def outer_offsets(P, N, bvh, inner, start=0.004, extra=None):
    """Offset along the limit normal to the outer high-poly surface, and
    whether it can be trusted. `extra`: per-vertex allowance on top of the
    garment thickness (how far a smoothed base sits inside the body)."""
    eye_z = float(LM["eye_l"][2])
    d = np.zeros(len(P))
    ok = np.zeros(len(P), dtype=bool)
    for i, (p, n) in enumerate(zip(P, N)):
        if inner[i]:
            ok[i] = True             # socket / mouth bag: stays on the body
            continue
        # garments are a few mm to ~2 cm thick; hair can stand ~4 cm off the scalp
        dmax = 0.045 if p[2] > eye_z - 0.03 else 0.028
        if extra is not None:
            dmax += extra[i]
        hit, hn, _f, dist = bvh.ray_cast(Vector(p - n * start), Vector(n), dmax + start)
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


def wrap(obj, hp, log=print):
    from mathutils.bvhtree import BVHTree
    me = obj.data
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(hp, dg)
    P, N = limit_frame(obj)
    inner = interior_vertices(me)
    nb = neighbours(me)
    d, ok = outer_offsets(P, N, bvh, inner)
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
    keep = dilate(inner, nb, rings=2)
    w = np.clip((d - 0.003) / 0.004, 0.0, 1.0)
    w = np.maximum(w, dilate(w > 0.5, nb, rings=3).astype(np.float64))
    w[keep] = 0.0
    A = _avg_matrix(nb)
    Ps = P.copy()
    for _ in range(40):
        Ps = Ps + (0.5 * w)[:, None] * (A @ Ps - Ps)
    _set_co(me, Ps)
    Ns = _normals(me)
    cov = w > 0
    d2, ok2 = outer_offsets(Ps, Ns, bvh, inner, extra=np.linalg.norm(Ps - P, axis=1) + 0.004)
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
    # then work out any remaining fold: relax just around it, a few rounds
    for r in range(8):
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

def build_lods(J, hp_obj, cache, name="SK_Character", hp_pouch=None, log=print):
    """Wrapped template = LOD0 (quads, template UV seams, own ABF unwrap and
    atlas packing together with the gear); LOD2 / LOD4 are collapse-decimated
    from it (UVs carried along). Returns (None, lod4, lod2, lod0) like
    retopo.build_lods."""
    import lods
    import retopo
    import uvs
    lod0 = template_object(f"{name}_LOD0", cache)
    wrap(lod0, hp_obj, log=log)
    part_labels(lod0, J)
    pouches = []
    if hp_pouch is not None:
        pc = retopo.pouch_cage(f"{name}_Pouch_LOD4")
        psurf = retopo.Surface(hp_pouch)
        retopo.project_nearest(pc, psurf, iters=1)
        pouches.append(pc)
    uvs.unwrap(lod0, extras=pouches)
    if pouches:
        prev = pouches[0]
        for level in ("LOD2", "LOD0"):
            o = prev.copy()
            o.data = prev.data.copy()
            o.name = o.data.name = f"{name}_Pouch_{level}"
            bpy.context.scene.collection.objects.link(o)
            retopo.subdivide(o, 1)
            retopo.project_nearest(o, psurf, iters=3)
            pouches.append(o)
            prev = o
    lod2 = lods.decimated(lod0, f"{name}_LOD2", 0.25)
    lod4 = lods.decimated(lod0, f"{name}_LOD4", 0.0625)
    for o in (lod0, lod2, lod4):
        o.data.shade_smooth()
    return None, lod4, lod2, lod0
