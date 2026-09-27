"""
sdf — signed-distance-field "digital sculpting" in pure numpy.

This is the stand-in for a ZBrush/Blender-sculpt pass in a headless,
code-only pipeline: forms are built from analytic primitives (ellipsoids,
round cones, round boxes, tori) combined with *smooth* booleans, so muscle
masses, fat pads and cloth volumes flow into each other the way clay does,
instead of intersecting with hard creases. The field is polygonized with
marching cubes into the dense high-poly that normal/AO maps are baked from.

No bpy import here on purpose: the module runs (and can be unit-tested)
under plain Python + numpy + scikit-image.

Conventions: meters, Blender axes (Z up, character faces -Y, character's
left side is +X).
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------

def v3(x):
    return np.asarray(x, dtype=np.float32)


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def frame_from_axis(axis, up=(0.0, 0.0, 1.0)):
    """Rotation matrix whose local Z points along `axis` (columns = local X,Y,Z in world)."""
    z = normalize(axis)
    up = np.asarray(up, dtype=np.float64)
    if abs(np.dot(z, normalize(up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def euler_matrix(rx=0.0, ry=0.0, rz=0.0):
    """XYZ euler (degrees) -> rotation matrix, same order as Blender's default."""
    rx, ry, rz = (math.radians(a) for a in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (mz @ my @ mx).astype(np.float32)


def mirror_x(p):
    return (-p[0], p[1], p[2])


# ---------------------------------------------------------------------------
# Primitives. Each has .bbox() -> (min, max) and .eval(P[...,3]) -> distance.
# ---------------------------------------------------------------------------

class Prim:
    def mirrored(self):
        raise NotImplementedError


class Ellipsoid(Prim):
    """Oriented ellipsoid (Inigo Quilez's bound-corrected approximation —
    accurate near the surface, which is all smooth-min blending needs)."""

    def __init__(self, center, radii, rot=None):
        self.c = v3(center)
        self.r = v3(radii)
        self.R = np.eye(3, dtype=np.float32) if rot is None else np.asarray(rot, dtype=np.float32)

    def bbox(self):
        ext = np.abs(self.R) @ self.r
        return self.c - ext, self.c + ext

    def eval(self, P):
        q = (P - self.c) @ self.R  # world -> local
        k0 = np.linalg.norm(q / self.r, axis=-1)
        k1 = np.linalg.norm(q / (self.r * self.r), axis=-1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

    def mirrored(self):
        M = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        return Ellipsoid(mirror_x(self.c), self.r, M @ self.R @ M)


class RoundCone(Prim):
    """Capsule whose radius varies linearly from ra (at a) to rb (at b) — exact SDF."""

    def __init__(self, a, b, ra, rb=None):
        self.a, self.b = v3(a), v3(b)
        self.ra = float(ra)
        self.rb = float(ra if rb is None else rb)

    def bbox(self):
        r = max(self.ra, self.rb)
        return np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r

    def eval(self, P):
        a, b, r1, r2 = self.a, self.b, self.ra, self.rb
        ba = b - a
        l2 = float(ba @ ba)
        rr = r1 - r2
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        pa = P - a
        y = pa @ ba
        z = y - l2
        xv = pa * l2 - y[..., None] * ba
        x2 = np.einsum("...i,...i->...", xv, xv)
        y2 = y * y * l2
        z2 = z * z * l2
        k = math.copysign(1.0, rr) * rr * rr * x2
        d_mid = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
        d_b = np.sqrt(np.maximum(x2 + z2, 0.0)) * il2 - r2
        d_a = np.sqrt(np.maximum(x2 + y2, 0.0)) * il2 - r1
        cond_b = (np.sign(z) * a2 * z2) > k
        cond_a = (np.sign(y) * a2 * y2) < k
        return np.where(cond_b, d_b, np.where(cond_a, d_a, d_mid))

    def mirrored(self):
        return RoundCone(mirror_x(self.a), mirror_x(self.b), self.ra, self.rb)


def Capsule(a, b, r):
    return RoundCone(a, b, r, r)


class RoundBox(Prim):
    def __init__(self, center, half, radius=0.0, rot=None):
        self.c = v3(center)
        self.h = v3(half)
        self.rad = float(radius)
        self.R = np.eye(3, dtype=np.float32) if rot is None else np.asarray(rot, dtype=np.float32)

    def bbox(self):
        ext = np.abs(self.R) @ (self.h + self.rad)
        return self.c - ext, self.c + ext

    def eval(self, P):
        q = np.abs((P - self.c) @ self.R) - self.h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside - self.rad

    def mirrored(self):
        M = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        return RoundBox(mirror_x(self.c), self.h, self.rad, M @ self.R @ M)


class Torus(Prim):
    """Torus in its local XY plane (axis = local Z); `scale` squashes the tube cross-section."""

    def __init__(self, center, major, minor, rot=None, tube_scale=(1.0, 1.0)):
        self.c = v3(center)
        self.R_ = float(major)
        self.r_ = float(minor)
        self.R = np.eye(3, dtype=np.float32) if rot is None else np.asarray(rot, dtype=np.float32)
        self.ts = tube_scale

    def bbox(self):
        e = self.R_ + self.r_ * max(self.ts)
        ext = np.abs(self.R) @ v3([e, e, self.r_ * max(self.ts)])
        return self.c - ext, self.c + ext

    def eval(self, P):
        q = (P - self.c) @ self.R
        radial = np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2) - self.R_
        d = np.sqrt((radial / self.ts[0]) ** 2 + (q[..., 2] / self.ts[1]) ** 2)
        return (d - self.r_) * min(self.ts)

    def mirrored(self):
        M = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        return Torus(mirror_x(self.c), self.R_, self.r_, M @ self.R @ M, self.ts)


class Custom(Prim):
    """Any distance function with an explicit bounding box. `mirror_fn`
    builds the X-mirrored twin (needed for sym=True)."""

    def __init__(self, fn, lo, hi, mirror_fn=None):
        self.fn, self.lo, self.hi, self.mirror_fn = fn, v3(lo), v3(hi), mirror_fn

    def bbox(self):
        return self.lo, self.hi

    def eval(self, P):
        return self.fn(P)

    def mirrored(self):
        if self.mirror_fn is None:
            def fn(P, f=self.fn):
                Q = P.copy()
                Q[..., 0] = -Q[..., 0]
                return f(Q)
            lo, hi = self.lo.copy(), self.hi.copy()
            lo[0], hi[0] = -self.hi[0], -self.lo[0]
            return Custom(fn, lo, hi)
        return self.mirror_fn()


class Shell(Prim):
    """Wraps another primitive/field function as a hollow shell of given thickness."""

    def __init__(self, inner, thickness):
        self.inner = inner
        self.t = float(thickness)

    def bbox(self):
        lo, hi = self.inner.bbox()
        return lo - self.t, hi + self.t

    def eval(self, P):
        return np.abs(self.inner.eval(P)) - self.t


# ---------------------------------------------------------------------------
# Smooth booleans (polynomial smooth-min, Quilez)
# ---------------------------------------------------------------------------

def smin(a, b, k):
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


# ---------------------------------------------------------------------------
# The sculpt: a recorded list of operations, evaluable on any grid
# ---------------------------------------------------------------------------

# Field edits (cloth offsets, folds, displacement) only matter near the
# surface; evaluating them only inside this band keeps them cheap.
EDIT_BAND = 0.04


class SDFModel:
    """An ordered list of sculpt operations (smooth union / subtraction /
    displacement / arbitrary field edits).

    Recording ops instead of stamping them into one fixed grid is what makes
    multi-resolution sculpting possible: the same model is evaluated at 5 mm
    for fast previews, ~2.5 mm for the body and ~1 mm for the head and hands
    (where eyelids, lips and finger gaps would otherwise fuse together), and
    the pieces are stitched into one high-poly for baking.
    """

    def __init__(self):
        self.ops = []

    def add(self, prim, k=0.0, sym=False):
        """Smooth union. `sym=True` also adds the X-mirrored copy."""
        self.ops.append(("union", prim, k))
        if sym:
            self.ops.append(("union", prim.mirrored(), k))
        return self

    def sub(self, prim, k=0.0, sym=False):
        """Smooth subtraction (carve)."""
        self.ops.append(("sub", prim, k))
        if sym:
            self.ops.append(("sub", prim.mirrored(), k))
        return self

    def inter(self, prim, k=0.0):
        self.ops.append(("inter", prim, k))
        return self

    def edit(self, fn, bmin, bmax):
        """Arbitrary field edit inside a box: field = fn(P, field). Used for
        cloth layers (offset shells with hems), folds and surface noise."""
        self.ops.append(("edit", _BoxRegion(bmin, bmax), fn))
        return self

    def displace(self, fn, bmin, bmax):
        """Displacement: pushes the surface out by fn(P) (meters) near it."""
        return self.edit(lambda P, f: f - fn(P).astype(np.float32), bmin, bmax)


class Volume:
    """Dense float32 distance grid an SDFModel is evaluated into.

    Every op only touches the sub-block covering its bounding box grown by
    the blend radius — exact for smooth-min, because outside that block the
    primitive is farther than `k` from anything and smin == min.
    """

    FAR = 1.0

    def __init__(self, bmin, bmax, voxel=0.003):
        self.voxel = float(voxel)
        self.origin = v3(bmin)
        # sample coordinates are computed in float64 and rounded once: two
        # bricks sharing a sample layer then evaluate it at bit-identical
        # points (float32 origin + float32 steps differed by ~1e-7 m, enough
        # to open cracks along brick seams where the surface grazes the layer)
        self._o64 = np.asarray(bmin, dtype=np.float64)
        self.shape = tuple(int(math.ceil((bmax[i] - bmin[i]) / voxel)) + 1 for i in range(3))
        self.field = np.full(self.shape, self.FAR, dtype=np.float32)

    @property
    def bmax(self):
        return self.origin + self.voxel * (np.array(self.shape) - 1)

    def _block(self, prim, margin):
        lo, hi = prim.bbox()
        lo = (lo - margin - self.origin) / self.voxel
        hi = (hi + margin - self.origin) / self.voxel
        i0 = np.clip(np.floor(lo).astype(int), 0, np.array(self.shape) - 1)
        i1 = np.clip(np.ceil(hi).astype(int) + 1, 0, np.array(self.shape))
        if np.any(i1 <= i0):
            return None, None
        sl = tuple(slice(int(i0[d]), int(i1[d])) for d in range(3))
        axes = [(self._o64[d] + self.voxel * np.arange(i0[d], i1[d], dtype=np.float64)).astype(np.float32)
                for d in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return sl, P

    def evaluate(self, model):
        has_edits = any(op == "edit" for op, _p, _a in model.ops)
        for op, prim, arg in model.ops:
            margin = (arg if op != "edit" else 0.0) + 3 * self.voxel
            if has_edits and op != "edit":
                # later edits (cloth/hair offsets) move the surface up to
                # EDIT_BAND away from where the primitives put it, so the
                # distance must be valid that far out — otherwise an offset
                # surface gets clipped flat at the primitive's block boundary
                margin = max(margin, EDIT_BAND)
            sl, P = self._block(prim, margin)
            if sl is None:
                continue
            cur = self.field[sl]
            if op == "edit":
                near = np.abs(cur) < EDIT_BAND
                if near.any():
                    out = cur.copy()
                    out[near] = arg(P[near], cur[near]).astype(np.float32)
                    self.field[sl] = out
                continue
            d = prim.eval(P).astype(np.float32)
            if op == "union":
                self.field[sl] = smin(cur, d, arg)
            elif op == "sub":
                self.field[sl] = smax(cur, -d, arg)
            elif op == "inter":
                self.field[sl] = smax(cur, d, arg)
        return self

    def polygonize(self, step=1):
        """Marching cubes -> (verts[N,3] world meters, faces[M,3] int)."""
        from skimage.measure import marching_cubes
        f = self.field[::step, ::step, ::step] if step > 1 else self.field
        verts, faces, _n, _v = marching_cubes(f, level=0.0, spacing=(self.voxel * step,) * 3,
                                              allow_degenerate=False)
        verts = verts + self._o64
        # skimage's default ("descent") winding already gives outward normals
        # for a negative-inside SDF (checked: positive signed volume)
        return verts.astype(np.float32), faces.astype(np.int32)


_SPARSE = {}


def _brick_job(b):
    from skimage.measure import marching_cubes
    model, bmin, voxel, brick = _SPARSE["args"]
    lo = bmin + np.asarray(b) * voxel * brick
    # brick+1 samples per axis: exactly one shared sample layer with the
    # neighbor, so each cell is polygonized once and border verts coincide
    vol = Volume(lo, lo + voxel * (brick - 0.25), voxel=voxel).evaluate(model)
    f = vol.field
    if f.min() >= 0 or f.max() <= 0:
        return None
    v, fc, _nn, _vv = marching_cubes(f, level=0.0, spacing=(voxel,) * 3, allow_degenerate=False)
    return v + vol._o64, fc.astype(np.int64)


def sparse_polygonize(model, bmin, bmax, voxel=0.0012, brick=40, workers=None, log=None):
    """Narrow-band polygonization: a coarse pass finds the bricks the surface
    passes through, only those are evaluated at full resolution (in parallel
    worker processes), each brick is marching-cubed separately and the
    pieces are welded. Gives a uniform ~1 mm high-poly over a whole
    character in a few hundred MB instead of tens of GB for a dense grid.
    Returns (verts, faces)."""
    import os
    import multiprocessing as mp
    bmin = np.asarray(bmin, np.float64)
    bmax = np.asarray(bmax, np.float64)
    sub = 4
    coarse = voxel * brick / sub
    cv = Volume(bmin, bmax, voxel=coarse).evaluate(model)
    ci = np.argwhere(np.abs(cv.field) < coarse * 2.0)
    nb = np.ceil((bmax - bmin) / (voxel * brick)).astype(int)
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    bi = np.unique(np.floor((ci[:, None, :] + offs[None]) / sub).astype(int).reshape(-1, 3), axis=0)
    bi = bi[np.all((bi >= 0) & (bi < nb), axis=1)]
    if log:
        log(f"{len(bi)} bricks of {brick}^3 @ {voxel * 1000:.2f} mm")
    _SPARSE["args"] = (model, bmin, voxel, brick)
    workers = workers or os.cpu_count()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_brick_job, [tuple(b) for b in bi], chunksize=4)
    else:
        results = [_brick_job(tuple(b)) for b in bi]
    all_v, all_f, nv = [], [], 0
    for r in results:
        if r is None:
            continue
        all_v.append(r[0])
        all_f.append(r[1] + nv)
        nv += len(r[0])
    V = np.concatenate(all_v)
    Fc = np.concatenate(all_f)
    # weld the bricks: only vertices on shared brick faces are duplicated.
    # Both copies agree to ~1e-9 m but not bitwise (the field is evaluated in
    # differently shaped blocks), so rounding them to a key grid split pairs
    # that straddled a grid line and left cracks: match them by distance.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    rel = (V - bmin) / (voxel * brick)
    on = np.nonzero(np.any(np.abs(rel - np.round(rel)) < 1e-3, axis=1))[0]
    pairs = cKDTree(V[on]).query_pairs(voxel * 1e-3, output_type="ndarray")
    rep = np.arange(len(V))
    if len(pairs):
        n = len(on)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _nc, lab = connected_components(g, directed=False)
        first = np.full(lab.max() + 1, len(V))
        np.minimum.at(first, lab, on)
        rep[on] = first[lab]
    keep, inv = np.unique(rep, return_inverse=True)
    V = V[keep]
    Fc = inv[Fc]
    ok = (Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])
    return V.astype(np.float32), Fc[ok].astype(np.int32)


def sample(model, points, voxel_hint=0.002):
    """Evaluate the model's distance at arbitrary points (slow path, used for
    retopology ray-marching / projection). Points: (N,3)."""
    pts = np.asarray(points, dtype=np.float32)
    f = np.full(len(pts), Volume.FAR, dtype=np.float32)
    for op, prim, arg in model.ops:
        lo, hi = prim.bbox()
        m = (arg if op != "edit" else 0.0) + 3 * voxel_hint
        if op != "edit":
            m = max(m, EDIT_BAND)
        inside = np.all((pts >= lo - m) & (pts <= hi + m), axis=1)
        if not inside.any():
            continue
        P = pts[inside]
        cur = f[inside]
        if op == "edit":
            near = np.abs(cur) < EDIT_BAND
            out = cur.copy()
            if near.any():
                out[near] = arg(P[near], cur[near])
            f[inside] = out
            continue
        d = prim.eval(P).astype(np.float32)
        if op == "union":
            f[inside] = smin(cur, d, arg)
        elif op == "sub":
            f[inside] = smax(cur, -d, arg)
        elif op == "inter":
            f[inside] = smax(cur, d, arg)
    return f


class _BoxRegion(Prim):
    def __init__(self, lo, hi):
        self.lo, self.hi = v3(lo), v3(hi)

    def bbox(self):
        return self.lo, self.hi


# ---------------------------------------------------------------------------
# Procedural noise usable as displacement (value noise, numpy only)
# ---------------------------------------------------------------------------

def _hash3(ix, iy, iz, seed=0):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 144665) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def value_noise(P, scale, seed=0):
    q = P * scale
    i = np.floor(q).astype(np.int64)
    f = q - i
    u = f * f * (3.0 - 2.0 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1]) * \
                    (u[..., 2] if dz else 1 - u[..., 2])
                out = out + w * _hash3(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out * 2.0 - 1.0


def ridged(P, scale, octaves=3, seed=0, sharp=3.0):
    """Ridged multifractal-ish noise in [0,1]: thin ridges where the base
    noise crosses zero — reads as cloth folds / creases when stretched."""
    tot, norm, amp = 0.0, 0.0, 1.0
    for o in range(octaves):
        n = value_noise(P, scale * (2.1 ** o), seed + 17 * o)
        tot = tot + amp * (1.0 - np.abs(n)) ** sharp
        norm += amp
        amp *= 0.5
    return tot / norm


def fbm(P, scale, octaves=4, seed=0, gain=0.5):
    amp, tot, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * value_noise(P, scale * (2 ** o), seed + o)
        norm += amp
        amp *= gain
    return tot / norm
