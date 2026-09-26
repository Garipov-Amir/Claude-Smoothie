"""
pieces — separate mesh pieces: gear and hair/horn parts that are their own
geometry in a game asset.

Wrapping one skin mesh over a hard box, a ponytail or a horn folds it into
self-intersections; production characters ship such parts as separate
pieces with their own high-poly, low-poly and LODs, baked against their own
high-poly and skinned rigidly. Each piece here is:

    name      mesh name suffix (SK_Character_<name>_LOD#, HighPoly_<name>)
    model     its own SDF model (never merged into the body sculpt)
    lo, hi    polygonization box
    bind      a bone name (rigid 100 %), or "skin": one weight set averaged
              from the body skin under the piece's anchor (pouch: its belt
              loop; pauldrons: the shoulder they rest on)
    lowpoly   "box" (a subdivided box cage fitted to `frame`) or "remesh"
              (QuadriFlow quad remesh to `faces` quads)
    material  region material (see texture.py)

Everything is placed from landmarks and body scales, and seated on the
sculpted surface (costume.surface_point), with clearance where a part hangs
along the body (ponytail, long beard) so it never cuts into it.
"""

import math

import numpy as np

from landmarks import LM
from sdf import SDFModel, Ellipsoid, RoundCone, RoundBox, Torus, Custom, euler_matrix, normalize, fbm, ridged

PIECES = []


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _add(name, ptype, model, lo, hi, bind, material, lowpoly="remesh", faces=400, frame=None, anchor=None):
    import costume
    names = {p["name"] for p in PIECES}
    base, k = name, 2
    while name in names:
        name, k = f"{base}{k}", k + 1
    PIECES.append({"name": name, "type": ptype, "model": model, "lo": np.asarray(lo, np.float32),
                   "hi": np.asarray(hi, np.float32), "bind": bind, "lowpoly": lowpoly, "faces": int(faces),
                   "frame": frame, "anchor": anchor, "material": material})
    costume.register(name.lower(), "piece", material, None)
    costume.GEAR.setdefault("pieces", []).append(name)


def _tube(model, pts, radii, k=0.004):
    for a, b, ra, rb in zip(pts[:-1], pts[1:], radii[:-1], radii[1:]):
        model.add(RoundCone(a, b, ra, rb), k=k)


def _bbox(pts, pad):
    P = np.asarray(pts, np.float32)
    return P.min(0) - pad, P.max(0) + pad


def _surface(m, origin, direction):
    import costume
    return costume.surface_point(m, origin, direction)


def _back_y(m, z, x=0.0):
    """y of the body's back surface at height z (marching in from behind)."""
    (_, _, _), (_, by1, _) = LM["bounds"]
    p = _surface(m, (x, by1 + 0.1, z), (0, -1, 0))
    return None if p is None else float(p[1])


def _front_y(m, z, x=0.0):
    (_, by0, _), _hi = LM["bounds"]
    p = _surface(m, (x, by0 - 0.1, z), (0, 1, 0))
    return None if p is None else float(p[1])


# ---------------------------------------------------------------------------

def pouch(m, J, p):
    """Leather belt pouch on the hip, hanging from a loop on the belt."""
    import costume
    s = LM["s_body"]
    side = -1.0 if p["side"] == "right" else 1.0
    z = LM["belt_z"] - 0.030 * s
    # march in from beside the hip, inside the hanging hands' reach
    q = _surface(m, (side * 0.35 * s, float(J["pelvis"][1]) - 0.03 * s, z), (-side, 0, 0))
    if q is None:
        return
    R = euler_matrix(0, 0, -18 * side)
    body_c = q + np.array([side * 0.018 * s, -0.004 * s, -0.018 * s], np.float32)
    half = np.array([0.014, 0.040, 0.045], np.float32) * s
    rad = 0.009 * s
    pm = SDFModel()
    pm.add(RoundBox(body_c, half, rad, R), k=0.0)
    pm.add(RoundBox(body_c + np.array([side * 0.012 * s, -0.002 * s, 0.030 * s], np.float32),
                    np.array([0.006, 0.043, 0.018], np.float32) * s, 0.004 * s, R), k=0.002)
    pm.add(RoundBox(q + np.array([side * 0.006 * s, -0.004 * s, 0.012 * s], np.float32),
                    np.array([0.004, 0.012, 0.030], np.float32) * s, 0.002 * s, R), k=0.002)  # belt loop

    def seam(P, f):  # stitched seam around the pouch's sides
        Q = (P - body_c) @ R
        band = smoothstep(0.0015, 0.0, np.abs(np.abs(Q[..., 1]) - half[1]))
        return f + 0.0006 * band
    pm.edit(seam, body_c - 0.08 * s, body_c + 0.08 * s)
    lo, hi = body_c - 0.09 * s, body_c + 0.09 * s
    _add("Pouch", "pouch", pm, lo, hi, "skin", p, lowpoly="box", frame=(body_c, R, half + rad), anchor="top")


def ponytail(m, J, p, hair_color):
    sh = LM["s_head"]
    c = np.asarray(LM["head_c"], np.float32)
    ez = float(LM["eye_l"][2])
    zb = ez + 0.035 * sh
    base = _surface(m, (0, c[1] + 0.25, zb), (0, -1, 0))
    if base is None:
        return
    L = float(p["length"]) * sh
    pts, radii = [base + np.array([0, -0.008, 0], np.float32)], [0.018 * sh]
    pts.append(base + np.array([0, 0.030, -0.030], np.float32) * sh)
    radii.append(0.024 * sh)
    n = 6
    for i in range(1, n + 1):
        z = float(pts[1][2]) - L * i / n
        r = 0.024 * sh * (1.0 - 0.75 * (i / n) ** 1.5) + 0.004 * sh
        yb = _back_y(m, z)
        y = max(float(pts[1][1]) + 0.004 * i * sh, (yb if yb is not None else pts[1][1]) + r + 0.014)
        pts.append(np.array([0, y, z], np.float32))
        radii.append(r)
    pm = SDFModel()
    _tube(pm, pts, radii, k=0.012 * sh)
    t0 = pts[1] + (pts[2] - pts[1]) * 0.15
    ax = normalize(pts[2] - pts[1])
    Rt = np.stack([np.cross(ax, [1, 0, 0]), [1, 0, 0], ax], 1).astype(np.float32)
    pm.add(Torus(t0, radii[1] * 0.92, 0.0045 * sh, rot=Rt), k=0.002)          # hair tie

    def strands(P, f):
        g = ridged(P * np.array([1.0, 1.0, 0.12], np.float32), 380.0 / sh, 2, seed=71, sharp=3.0)
        return f + 0.0010 * sh * g
    lo, hi = _bbox(pts, 0.05 * sh)
    pm.edit(strands, lo, hi)
    _add("Ponytail", "ponytail", pm, lo, hi, "head", {"material": "hair", "color": p["color"] or hair_color},
         faces=520)


def bun(m, J, p, hair_color):
    sh = LM["s_head"] * float(p["size"])
    c = np.asarray(LM["head_c"], np.float32)
    d = normalize(np.array([0, 0.75, 0.66], np.float32))
    q = _surface(m, c + d * 0.3, -d)
    if q is None:
        return
    cc = q + d * 0.016 * sh
    pm = SDFModel()
    pm.add(Ellipsoid(cc, np.array([0.040, 0.034, 0.032], np.float32) * sh), k=0.0)

    def swirl(P, f):
        Q = P - cc
        ang = np.arctan2(Q[..., 0], Q[..., 2])
        return f + 0.0012 * sh * np.sin(ang * 9 + np.linalg.norm(Q, axis=-1) * 180.0)
    lo, hi = cc - 0.06 * sh, cc + 0.06 * sh
    pm.edit(swirl, lo, hi)
    _add("Bun", "bun", pm, lo, hi, "head", {"material": "hair", "color": p["color"] or hair_color}, faces=260)


def beard_long(m, J, p, beard_color):
    sh = LM["s_head"]
    pog = np.asarray(LM["pogonion"], np.float32)
    chin_z = float(LM["chin_z"])
    L = float(p["length"]) * sh
    zs = [float(pog[2]) + 0.006 * sh, chin_z - 0.004 * sh, chin_z - 0.35 * L, chin_z - 0.7 * L, chin_z - L]
    rx = np.array([0.050, 0.054, 0.050, 0.040, 0.022]) * sh
    ry = np.array([0.022, 0.028, 0.026, 0.020, 0.012]) * sh
    pm = SDFModel()
    cs = []
    for i, z in enumerate(zs):
        yf = _front_y(m, z)
        y = float(pog[1]) + 0.012 * sh if i == 0 else min(float(pog[1]) + 0.012 * sh + 0.01 * i * sh,
                                                         (yf if yf is not None else float(pog[1])) - ry[i] - 0.012)
        cs.append(np.array([0, y, z], np.float32))
        pm.add(Ellipsoid(cs[-1], (rx[i], ry[i], max(0.025 * sh, 0.35 * L))), k=0.02 * sh)

    def strands(P, f):
        return f + 0.0012 * sh * ridged(P * np.array([1.0, 1.0, 0.12], np.float32), 300.0 / sh, 2, seed=73, sharp=3.0)
    lo, hi = _bbox(cs, 0.07 * sh)
    pm.edit(strands, lo, hi)
    _add("Beard", "beard_long", pm, lo, hi, "head", {"material": "hair", "color": p["color"] or beard_color}, faces=480)


def _curve(base, n, size, shape, side):
    """Horn centerline: from the base along the outward normal, bending
    up/back (curved), straight up-and-out, or curling back and down (ram)."""
    up, back, out = np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([side, 0, 0.0])
    pts = [base]
    steps = 10
    L = size
    p = np.array(base, np.float64)
    for i in range(1, steps + 1):
        u = i / steps
        if shape == "straight":
            d = normalize(0.55 * n + 0.8 * up + 0.2 * out)
        elif shape == "ram":
            a = u * 4.2
            d = normalize(0.4 * out + math.cos(a) * up + math.sin(a) * back + 0.3 * n * (1 - u))
        else:
            d = normalize(n * (1 - u) * 0.9 + (0.85 * up + 0.55 * back + 0.25 * out) * (0.4 + u))
        p = p + d * L / steps
        pts.append(p.astype(np.float32))
    return pts


def horns(m, J, p):
    sh = LM["s_head"]
    size = float(p["size"])
    c = np.asarray(LM["head_c"], np.float32)
    ez = float(LM["eye_l"][2])
    pm = SDFModel()
    allpts = []
    for side in (1.0, -1.0):
        o = np.array([side * 0.20, c[1] - 0.06, ez + 0.085 * sh], np.float32)
        q = _surface(m, o, normalize(np.array([c[0], c[1], ez + 0.07 * sh]) - o))
        if q is None:
            continue
        n = normalize(q - np.array([0, c[1], ez + 0.02 * sh], np.float32))
        L = (0.26 if p["shape"] == "ram" else 0.13) * sh * size
        pts = _curve(q - n * 0.010 * sh, n, L, p["shape"], side)
        r0 = 0.019 * sh * size
        radii = [r0 * (1.0 - 0.88 * (i / (len(pts) - 1)) ** 1.2) for i in range(len(pts))]
        _tube(pm, pts, radii, k=0.004 * sh)
        allpts += pts

    def rings(P, f):
        return f + 0.0009 * sh * size * (0.5 + 0.5 * np.sin(np.linalg.norm(P - c, axis=-1) * 2 * math.pi / (0.008 * sh)))
    lo, hi = _bbox(allpts, 0.03 * sh * size)
    pm.edit(rings, lo, hi)
    _add("Horns", "horns", pm, lo, hi, "head", {"material": "horn", "color": p["color"]}, faces=560)


def tusks(m, J, p):
    sh = LM["s_head"]
    size = float(p["size"])
    st = np.asarray(LM["stomion"], np.float32)
    pm = SDFModel()
    allpts = []
    for side in (1.0, -1.0):
        base = np.array([side * 0.018 * sh, st[1] + 0.006 * sh, st[2] - 0.013 * sh], np.float32)
        d0 = normalize(np.array([side * 0.30, -0.55, 1.0], np.float32))
        L = 0.036 * sh * size
        pts = [base]
        for i in range(1, 7):
            u = i / 6
            d = normalize(d0 + np.array([side * 0.15, 0.25, 0]) * u)
            pts.append((pts[-1] + d * L / 6).astype(np.float32))
        r0 = 0.0058 * sh * size
        _tube(pm, pts, [r0 * (1 - 0.8 * (i / 6) ** 1.3) for i in range(7)], k=0.002)
        allpts += pts
    lo, hi = _bbox(allpts, 0.012 * sh * size)
    _add("Tusks", "tusks", pm, lo, hi, "head", {"material": "horn", "color": p["color"]}, faces=240)


def pauldrons(m, J, p):
    """Shoulder guards: a domed shell over the deltoid (inner surface clears
    the sculpted shoulder by ~8 mm), a rolled rim, three rivets."""
    from sdf import sample
    s = LM["s_body"]
    sides = {"both": (1.0, -1.0), "left": (1.0,), "right": (-1.0,)}[p["side"]]
    mat = {"material": p["material"], "color": p["color"], "metal": p.get("metal", "iron")}
    for side in sides:
        S = np.asarray(J["upperarm_l"], np.float32) * np.array([side, 1, 1], np.float32)
        E = np.asarray(J["lowerarm_l"], np.float32) * np.array([side, 1, 1], np.float32)
        arm = normalize(E - S)
        # dome axis: perpendicular to the upper arm, outward-and-up (the
        # deltoid's outer face), tipped a little further up
        a = normalize(normalize(np.cross(arm, [0, side, 0])) + np.array([0, 0, 0.3]))
        cc = S + arm * 0.030 * s
        # cap directions within ~70 deg of the axis
        dirs = []
        for th in np.radians([0, 20, 40, 55]):
            for ph in np.radians(np.arange(0, 360, 30)):
                u = normalize(np.cross(a, [0, 1, 0]))
                v = np.cross(a, u)
                dirs.append(math.cos(th) * a + math.sin(th) * (math.cos(ph) * u + math.sin(ph) * v))
        dirs = np.array(dirs, np.float32)
        R_in = 0.04
        for R in np.arange(0.04, 0.20, 0.002):
            if (sample(m, (cc + dirs * R).astype(np.float32)) > 0.008).all():
                R_in = float(R)
                break
        t = 0.005
        rim_cos = math.cos(math.radians(58))

        def dome(P, cc=cc, a=a, R_in=R_in):
            d = P - cc
            r = np.linalg.norm(d, axis=-1)
            shell = np.abs(r - (R_in + t / 2)) - t / 2
            cut = rim_cos * r - (d @ a)                   # keep the cap: angle to the axis < 58 deg
            return np.maximum(shell, cut)
        pm = SDFModel()
        lo, hi = cc - (R_in + 0.03), cc + (R_in + 0.03)
        pm.add(Custom(dome, lo, hi), k=0.0)
        u = normalize(np.cross(a, [0, 1, 0]))
        rim_c = cc + a * (R_in + t / 2) * rim_cos
        rr = (R_in + t / 2) * math.sin(math.acos(rim_cos))
        Rt = np.stack([u, np.cross(a, u), a], 1).astype(np.float32)
        pm.add(Torus(rim_c, rr, 0.0035 * s, rot=Rt), k=0.002)
        for ang in (-0.6, 0.0, 0.6):
            dd = normalize(math.cos(0.5) * a + math.sin(0.5) * (math.cos(ang) * u + math.sin(ang) * np.cross(a, u)))
            pm.add(Ellipsoid(cc + dd * (R_in + t), (0.004, 0.004, 0.004)), k=0.001)
        _add("Pauldron_L" if side > 0 else "Pauldron_R", "pauldrons", pm, lo, hi, "skin", mat, lowpoly="dome",
             frame=(cc, a, R_in, t, math.acos(rim_cos)), anchor="all")


def build(m, J, S):
    """All pieces of the spec (the body model m is only read: pieces are
    seated on it, never merged into it)."""
    PIECES.clear()
    hair_color, beard_color = S["hair"]["color"], S["beard"]["color"]
    for p in S["pieces"]:
        t = p["type"]
        if t == "pouch":
            pouch(m, J, p)
        elif t == "ponytail":
            ponytail(m, J, p, hair_color)
        elif t == "bun":
            bun(m, J, p, hair_color)
        elif t == "beard_long":
            beard_long(m, J, p, beard_color)
        elif t == "horns":
            horns(m, J, p)
        elif t == "tusks":
            tusks(m, J, p)
        elif t == "pauldrons":
            pauldrons(m, J, p)
    return PIECES
