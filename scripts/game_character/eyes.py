"""
eyes — eyeball meshes + a procedural eye texture.

Game eyes are separate meshes (so they can rotate with eye bones) sitting in
the eye holes of the skin mesh. The eyeball has a slight corneal bulge; its
UVs are an azimuthal projection around the view axis so the texture is a
simple disc: pupil, iris, limbal ring, sclera with faint veins.
"""
import math

import numpy as np

def _eye():
    from landmarks import LM
    return np.array(LM["eye_l"], dtype=np.float64), float(LM["eye_r_radius"]) - 0.0003


def make_eye(side, name, segments=24, rings=16):
    import bmesh
    import bpy
    from mathutils import Vector
    EYE_C, EYE_R = _eye()
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=EYE_R)
    uv = bm.loops.layers.uv.new("UVMap")
    for v in bm.verts:
        d = v.co.normalized()
        # view axis = -Y. theta: angle from the front pole
        th = math.acos(max(-1.0, min(1.0, -d.y)))
        if th < 0.55:  # cornea bulge
            v.co = v.co * (1.0 + 0.09 * (1 - th / 0.55) ** 2)
    for f in bm.faces:
        for l in f.loops:
            d = l.vert.co.normalized()
            th = math.acos(max(-1.0, min(1.0, -d.y)))
            ph = math.atan2(d.z, d.x)
            r = th / math.pi * 0.5
            l[uv].uv = (0.5 + r * math.cos(ph), 0.5 + r * math.sin(ph))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    c = EYE_C.copy()
    c[0] *= side
    o.location = Vector(c)
    return o


def eye_texture(res=512, iris_color=(0.28, 0.33, 0.20)):
    """Returns (H,W,3) sRGB albedo (row 0 = V 0) for the azimuthal eye UVs."""
    y, x = np.mgrid[0:res, 0:res].astype(np.float32)
    u = (x + 0.5) / res - 0.5
    v = (y + 0.5) / res - 0.5
    r = np.sqrt(u * u + v * v) * 2.0 * math.pi  # = theta
    ang = np.arctan2(v, u)
    rng = np.random.default_rng(7)
    sclera = np.array([0.86, 0.83, 0.79], np.float32)
    col = np.tile(sclera, (res, res, 1))
    # faint veins toward the back of the eye
    vein = np.zeros_like(r)
    for k in range(40):
        a0 = rng.uniform(-math.pi, math.pi)
        w = rng.uniform(0.004, 0.012)
        wig = 0.05 * np.sin(r * rng.uniform(6, 14) + rng.uniform(0, 6))
        dang = np.angle(np.exp(1j * (ang - a0 - wig)))
        vein = np.maximum(vein, np.exp(-(dang / w) ** 2) * (r > rng.uniform(0.8, 1.2)))
    col = col * (1 - 0.35 * vein[..., None]) + np.array([0.75, 0.25, 0.22]) * 0.35 * vein[..., None]
    iris_r, pupil_r = 0.52, 0.19
    fib = 0.5 + 0.5 * np.sin(ang * 90 + 3 * np.sin(ang * 7)) * np.exp(-((r - 0.35) / 0.2) ** 2)
    iris = np.array(iris_color, np.float32)
    inner = np.array([0.45, 0.33, 0.14], np.float32)  # warm collarette (hazel)
    t = np.clip((r - pupil_r) / (iris_r - pupil_r), 0, 1)[..., None]
    ic = inner * (1 - t) + iris * t
    ic = ic * (0.75 + 0.45 * fib[..., None])
    irm = (r < iris_r)[..., None]
    col = np.where(irm, ic, col)
    limb = np.exp(-((r - iris_r) / 0.035) ** 2)[..., None]
    col = col * (1 - 0.7 * limb)
    col = np.where((r < pupil_r)[..., None], np.array([0.02, 0.02, 0.02], np.float32), col)
    return np.clip(col, 0, 1).astype(np.float32)
