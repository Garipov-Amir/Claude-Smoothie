"""
reference_body — an anatomical reference figure to sculpt against.

Artists sculpt with reference (front/side photos, scans, anatomy figures).
This module builds a neutral anatomical reference from the MakeHuman base
mesh and morph targets — released as **CC0** by the MakeHuman project
(https://github.com/makehumancommunity/makehuman, "LICENSE.md", section C) —
and brings it into this pipeline's space (Blender axes, meters, same
stature), so the SDF sculpt can be compared with it: overlaid silhouettes,
cross-section differences, landmark checks.

It is used as *reference only*: the sculpt, topology, UVs, textures and rig
stay this pipeline's own work.

Files are fetched once into a cache directory:
    base.obj, caucasian-male-young.target,
    universal-male-young-maxmuscle-averageweight.target
"""

import os
import urllib.request

import numpy as np

RAW = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/"
FILES = {
    "base.obj": "3dobjs/base.obj",
    "caucasian-male-young.target": "targets/macrodetails/caucasian-male-young.target",
    "universal-male-young-maxmuscle-averageweight.target":
        "targets/macrodetails/universal-male-young-maxmuscle-averageweight.target",
}


def fetch(cache):
    os.makedirs(cache, exist_ok=True)
    for name, rel in FILES.items():
        p = os.path.join(cache, name)
        if not os.path.exists(p) or os.path.getsize(p) < 1000:
            urllib.request.urlretrieve(RAW + rel, p)
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


def load_target(path):
    idx, d = [], []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            idx.append(int(p[0]))
            d.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(idx, dtype=int), np.array(d, dtype=np.float64)


def build(cache, muscle=0.45, stature=1.80, full=False):
    """Young adult male, a bit above average muscle. Returns (verts, faces)
    in Blender space: meters, Z up, facing -Y, feet on Z = 0. With full=True
    returns a dict with the joint centers from MakeHuman's joint helpers too."""
    fetch(cache)
    V, G = load_obj_groups(os.path.join(cache, "base.obj"))
    F = G["body"]
    for name, w in (("caucasian-male-young.target", 1.0),
                    ("universal-male-young-maxmuscle-averageweight.target", muscle)):
        i, d = load_target(os.path.join(cache, name))
        V[i] += w * d
    used = np.unique(np.concatenate([np.array(f) for f in F]))
    # MakeHuman: Y up, +Z forward, decimeters -> Blender: Z up, -Y forward, meters
    B = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)
    lo = B[used, 2].min()
    hi = B[used, 2].max()
    scale = stature / (hi - lo)
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
    return {"verts": B.astype(np.float32), "faces": F, "joints": joints, "eye_radius": eye_r,
            "scale": scale}


# ---------------------------------------------------------------------------
# Smooth limit surface samples + a signed distance primitive
# ---------------------------------------------------------------------------

def surface_samples(ref, cache, levels=3):
    """Catmull-Clark subdivide the reference (the base mesh is the control cage
    of a smooth body) and return (points, normals) of the dense result.
    Cached as npz — needs bpy the first time."""
    path = os.path.join(cache, f"ref_samples_l{levels}.npz")
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

    def __init__(self, points, normals, near=0.004):
        from scipy.spatial import cKDTree
        self.p = np.asarray(points, np.float32)
        self.n = np.asarray(normals, np.float32)
        self.n /= np.linalg.norm(self.n, axis=1, keepdims=True) + 1e-12
        self.tree = cKDTree(self.p)
        self.near = near
        self.lo = self.p.min(0) - 0.002
        self.hi = self.p.max(0) + 0.002

    def bbox(self):
        return self.lo, self.hi

    def eval(self, P):
        shp = P.shape[:-1]
        Q = P.reshape(-1, 3)
        dist, idx = self.tree.query(Q, workers=-1)
        dvec = Q - self.p[idx]
        plane = np.einsum("ij,ij->i", dvec, self.n[idx])
        sgn = np.where(plane >= 0, 1.0, -1.0)
        d = np.where(dist < self.near, plane, sgn * dist)
        return d.reshape(shp).astype(np.float32)

    def mirrored(self):
        raise NotImplementedError("the reference body is already two-sided")
