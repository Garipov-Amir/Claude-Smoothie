"""
reference_body — an anatomical reference figure to sculpt against.

Artists sculpt with reference (front/side photos, scans, anatomy figures).
This module builds a neutral anatomical reference from the MakeHuman base
mesh and morph targets — released as **CC0** by the MakeHuman project
(https://github.com/makehumancommunity/makehuman, "LICENSE.md", section C) —
and brings it into this pipeline's space (Blender axes, meters, same
stature), so the SDF sculpt can be compared with it: overlaid silhouettes,
cross-section differences, landmark checks.

The sculpt is built on its limit surface; wrap.py also uses its quad
topology as the base mesh that gets wrapped onto the finished sculpt (the
studio "base mesh" workflow). Sculpted clothing, UV layout and packing,
bakes, textures, rig and LODs stay this pipeline's own work.

Files are fetched once into a cache directory: base.obj, MakeHuman's
modifier list and whichever targets a body needs (macro targets for the
sex/age/muscle/weight/proportions/ethnicity blend, plus named modifiers).
"""

import os
import urllib.request

import numpy as np

RAW = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/"
FILES = {
    "base.obj": "3dobjs/base.obj",
    "modeling_modifiers.json": "modifiers/modeling_modifiers.json",
}


def _get(cache, rel, name=None):
    """Download data/<rel> into the cache once; returns the local path."""
    p = os.path.join(cache, name or rel)
    if not os.path.exists(p) or os.path.getsize(p) < 16:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".part"
        urllib.request.urlretrieve(RAW + rel, tmp)
        os.replace(tmp, p)
    return p


def fetch(cache):
    os.makedirs(cache, exist_ok=True)
    for name, rel in FILES.items():
        _get(cache, rel, name)
    return cache


def load_obj_groups(path):
    """All vertices + faces per group."""
    V, G, grp = [], {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("g "):
                grp = line.split()[1]
            elif line.startswith("f "):
                G.setdefault(grp, []).append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V, dtype=np.float64), G


def load_obj_body(path):
    """Vertices + faces of the 'body' group only (helpers/joint proxies skipped)."""
    V, F, grp = [], [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("g "):
                grp = line.split()[1]
            elif line.startswith("f ") and grp == "body":
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V, dtype=np.float64), F


def load_body_uv(path):
    """UV coordinates of the 'body' group: (vt array, per-face vt indices),
    faces in the same order as load_obj_body / load_obj_groups()['body']."""
    VT, FT, grp = [], [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("vt "):
                VT.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("g "):
                grp = line.split()[1]
            elif line.startswith("f ") and grp == "body":
                FT.append([int(t.split("/")[1]) - 1 for t in line.split()[1:]])
    return np.array(VT, dtype=np.float64), FT


def load_target(path):
    idx, d = [], []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            idx.append(int(p[0]))
            d.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(idx, dtype=int), np.array(d, dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# MakeHuman macro sliders -> weighted targets (same blend rules as MakeHuman)
# ---------------------------------------------------------------------------

def age_value(years):
    """MakeHuman's age slider from years: 1 y -> 0, 25 y -> 0.5, 90 y -> 1."""
    y = float(np.clip(years, 1.0, 90.0))
    return (y - 1.0) / 48.0 if y < 25.0 else 0.5 + (y - 25.0) / 130.0


def _three(v, names):
    """min/average/max weights of a 0..1 slider (0.5 = pure average)."""
    lo, avg, hi = names
    if v < 0.5:
        a = max(0.0, 1.0 - 2.0 * v)
        return {lo: a, avg: 1.0 - a}
    b = max(0.0, 2.0 * v - 1.0)
    return {avg: 1.0 - b, hi: b}


def _ages(a):
    if a < 0.5:
        young = max(0.0, (a - 0.1875) * 3.2)
        baby = max(0.0, 1.0 - a * 5.333)
        child = max(0.0, min(1.0, 5.333 * a) - young)
        return {"baby": baby, "child": child, "young": young}
    old = max(0.0, 2.0 * a - 1.0)
    return {"young": 1.0 - old, "old": old}


def macro_targets(body):
    """[(relative target path, weight)] for the macro sliders of a body spec."""
    g = float(body.get("sex", 1.0))
    genders = {"female": 1.0 - g, "male": g}
    ages = _ages(age_value(body.get("age", 25.0)))
    musc = _three(float(body.get("muscle", 0.5)), ("minmuscle", "averagemuscle", "maxmuscle"))
    wgt = _three(float(body.get("weight", 0.5)), ("minweight", "averageweight", "maxweight"))
    eth = dict(body.get("ethnicity") or {"caucasian": 1.0})
    tot = sum(eth.values()) or 1.0
    races = {k: v / tot for k, v in eth.items() if v > 0}
    p = float(body.get("proportions", 0.5))
    props = ({"uncommonproportions": 1.0 - 2.0 * p} if p < 0.5 else {"idealproportions": 2.0 * p - 1.0})
    out = []
    for gn, gw in genders.items():
        for an, aw in ages.items():
            ga = gw * aw
            if ga <= 1e-4:
                continue
            for rn, rw in races.items():
                out.append((f"targets/macrodetails/{rn}-{gn}-{an}.target", ga * rw))
            for mn, mw in musc.items():
                for wn, ww in wgt.items():
                    w = ga * mw * ww
                    if w <= 1e-4:
                        continue
                    out.append((f"targets/macrodetails/universal-{gn}-{an}-{mn}-{wn}.target", w))
                    for pn, pw in props.items():
                        if w * pw > 1e-4:
                            out.append((f"targets/macrodetails/proportions/{gn}-{an}-{mn}-{wn}-{pn}.target", w * pw))
    return [(r, w) for r, w in out if w > 1e-4]


def modifier_index(cache):
    """Target stem (side prefix removed) -> [relative target paths], from
    MakeHuman's modeling_modifiers.json: 'ear-shape-pointed' ->
    [targets/ears/l-ear-shape-pointed.target, targets/ears/r-ear-shape-pointed.target]."""
    import json
    fetch(cache)
    with open(os.path.join(cache, "modeling_modifiers.json")) as fh:
        groups = json.load(fh)
    idx = {}
    for g in groups:
        grp = g["group"]
        if grp.startswith("macrodetails") or grp in ("breast", "genitals"):
            continue
        for m in g["modifiers"]:
            t = m.get("target")
            if not t:
                continue
            stems = [f"{t}-{m['min']}", f"{t}-{m['max']}"] if "min" in m else [t]
            for st in stems:
                key = st[2:] if st[:2] in ("l-", "r-") else st
                idx.setdefault(key, set()).add(f"targets/{grp}/{st}.target")
    return {k: sorted(v) for k, v in idx.items()}


def modifier_targets(body, cache):
    idx = modifier_index(cache)
    out = []
    for key, v in (body.get("modifiers") or {}).items():
        if key not in idx:
            import difflib
            near = difflib.get_close_matches(key, list(idx), n=5)
            raise ValueError(f"unknown MakeHuman modifier {key!r}; close: {near}")
        out += [(rel, float(v)) for rel in idx[key]]
    return out


def _fetch_all(cache, rels):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda r: _get(cache, r), rels))


_BUILT = {}


def build(cache, body=None, full=False):
    """The reference figure for a body spec (spec.DEFAULT["body"] if None):
    MakeHuman base mesh + macro targets (sex, age, muscle, weight,
    proportions, ethnicity) + named modifiers, in Blender space (meters, Z
    up, facing -Y, feet on Z = 0), scaled to body["height_m"] (None keeps
    MakeHuman's own stature for that age/sex). full=True also returns the
    joint centers from MakeHuman's joint helpers and the eye radii."""
    import json
    if body is None:
        import spec as _spec
        body = _spec.DEFAULT["body"]
    key = json.dumps(body, sort_keys=True)
    if key not in _BUILT:
        _BUILT.clear()
        fetch(cache)
        V, G = load_obj_groups(os.path.join(cache, "base.obj"))
        parts = macro_targets(body) + modifier_targets(body, cache)
        paths = _fetch_all(cache, [r for r, _w in parts])
        for (rel, w), path in zip(parts, paths):
            i, d = load_target(path)
            if len(i):
                V[i] += w * d
        _BUILT[key] = (V, G)
    V, G = _BUILT[key]
    F = G["body"]
    used = np.unique(np.concatenate([np.array(f) for f in F]))
    # MakeHuman: Y up, +Z forward, decimeters -> Blender: Z up, -Y forward, meters
    B = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)
    lo = B[used, 2].min()
    hi = B[used, 2].max()
    stature = body.get("height_m")
    scale = float(stature) / (hi - lo) if stature else 0.1
    B = (B - np.array([0, 0, lo])) * scale
    B[:, 0] -= np.mean(B[used, 0])
    if not full:
        return B.astype(np.float32), F
    joints = {}
    for g, faces in G.items():
        if g.startswith("joint-"):
            idx = np.unique(np.concatenate([np.array(f) for f in faces]))
            joints[g[6:]] = B[idx].mean(axis=0)
    eye_r = {}
    for side in ("l", "r"):
        idx = np.unique(np.concatenate([np.array(f) for f in G[f"helper-{side}-eye"]]))
        c = joints[f"{side}-eye"]
        eye_r[side] = float(np.median(np.linalg.norm(B[idx] - c, axis=1)))
    import hashlib
    return {"verts": B.astype(np.float32), "faces": F, "joints": joints, "eye_radius": eye_r,
            "scale": scale, "key": hashlib.sha1(key.encode()).hexdigest()[:12]}


# ---------------------------------------------------------------------------
# Smooth limit surface samples + a signed distance primitive
# ---------------------------------------------------------------------------

def surface_samples(ref, cache, levels=3):
    """Catmull-Clark subdivide the reference (the base mesh is the control cage
    of a smooth body) and return (points, normals) of the dense result.
    Cached as npz per body shape — needs bpy the first time."""
    path = os.path.join(cache, "samples", f"ref_samples_l{levels}_{ref.get('key', 'default')}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        d = np.load(path)
        return d["p"], d["n"]
    import bpy
    # only the body's own vertices: the file also holds helper geometry
    # (joint cubes, tights, skirt, hair cap, teeth...) that would otherwise
    # come along as loose points with meaningless normals
    used = np.unique(np.concatenate([np.array(f) for f in ref["faces"]]))
    remap = -np.ones(len(ref["verts"]), dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = [[int(remap[i]) for i in f] for f in ref["faces"]]
    me = bpy.data.meshes.new("ref_tmp")
    me.from_pydata(ref["verts"][used].tolist(), [], faces)
    me.update()
    ob = bpy.data.objects.new("ref_tmp", me)
    bpy.context.scene.collection.objects.link(ob)
    m = ob.modifiers.new("s", "SUBSURF")
    m.levels = levels
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg).to_mesh()
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ev)
    # the base mesh's face winding is not consistently outward: fix it,
    # otherwise half the normals point in and the SDF sign is noise
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    p = np.array([v.co[:] for v in bm.verts], np.float32)
    nr = np.array([v.normal[:] for v in bm.verts], np.float32)
    bm.free()
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)
    np.savez_compressed(path, p=p, n=nr)
    return p, nr


class MeshSDF:
    """Signed distance to a dense, closed sample surface.

    Magnitude: distance to the nearest sample; sign: side of that sample's
    normal (pseudo-normal test). Within a few mm of the surface the
    point-to-plane distance is used instead, which makes the zero level smooth
    (no scalloping between samples)."""

    FAR = 0.05   # beyond this only the sign matters (edits/blends act within ~4 cm)

    def __init__(self, points, normals, near=0.004, coarse=0.006):
        from scipy.spatial import cKDTree
        self.p = np.asarray(points, np.float32)
        self.n = np.asarray(normals, np.float32)
        self.n /= np.linalg.norm(self.n, axis=1, keepdims=True) + 1e-12
        self.tree = cKDTree(self.p)
        self.near = near
        self.lo = self.p.min(0) - 0.002
        self.hi = self.p.max(0) + 0.002
        # coarse inside/outside grid: the sign for points far from the surface,
        # so the fine queries can stop at FAR (much faster KD searches).
        # Cells near the surface take the sign of their nearest sample; the
        # far cells connected to the grid border are outside, every other
        # far cell is enclosed by the body (a flood fill, no long KD searches)
        from scipy import ndimage
        self.c = coarse
        self.g0 = self.lo - 2 * coarse
        shape = tuple(np.ceil((self.hi - self.g0) / coarse).astype(int) + 3)
        axes = [self.g0[d] + coarse * np.arange(shape[d]) for d in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
        dist, i = self.tree.query(G, distance_upper_bound=2.0 * coarse, workers=-1)
        near = np.isfinite(dist)
        inside = np.zeros(len(G), dtype=bool)
        inside[near] = np.einsum("ij,ij->i", G[near] - self.p[i[near]], self.n[i[near]]) < 0
        far = (~near).reshape(shape)
        lab, _n = ndimage.label(far)
        border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                           lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        enclosed = far & ~np.isin(lab, border[border > 0])
        self.occ = inside.reshape(shape) | enclosed

    def bbox(self):
        return self.lo, self.hi

    def eval(self, P):
        shp = P.shape[:-1]
        Q = P.reshape(-1, 3)
        dist, idx = self.tree.query(Q, distance_upper_bound=self.FAR, workers=-1)
        far = ~np.isfinite(dist)
        d = np.empty(len(Q), np.float32)
        if far.any():
            gi = np.clip(np.round((Q[far] - self.g0) / self.c).astype(int), 0, np.array(self.occ.shape) - 1)
            d[far] = np.where(self.occ[gi[:, 0], gi[:, 1], gi[:, 2]], -self.FAR, self.FAR)
        nf = ~far
        if nf.any():
            ii = idx[nf]
            dvec = Q[nf] - self.p[ii]
            plane = np.einsum("ij,ij->i", dvec, self.n[ii])
            sgn = np.where(plane >= 0, 1.0, -1.0)
            d[nf] = np.where(dist[nf] < self.near, plane, sgn * dist[nf])
        return d.reshape(shp)

    def mirrored(self):
        raise NotImplementedError("the reference body is already two-sided")
