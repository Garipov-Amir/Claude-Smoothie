"""
costume — the garment library: clothing, footwear, hair and beards sculpted
onto any body, from the character spec (spec.py).

Cloth is sculpted the way it is in ZBrush/Marvelous-to-ZBrush workflows,
translated to distance fields:

* A garment is an *offset shell* of whatever is under it (body, or the
  previous garment): inside the garment's region the field is lowered by
  the cloth thickness + ease, so the surface grows outward.
* The region mask falls off over ~1-2 mm at hems, cuffs and necklines, which
  produces a real ledge — the thickness step a normal-map bake picks up as a
  crisp layered edge on the low-poly.
* Folds are displacement patterns in garment-local coordinates (distance
  along the limb, angle around it): spiral "pipe" folds on sleeves,
  compression folds at the inner elbow and knee, stacking above cuffs and
  boot tops, modulated with noise so no two folds repeat.

Every placement is relative to the body: heights come from joints and
landmarks (waistline, knee, ankle, neck base, chin), widths from the limb
radii measured on the mesh, sizes from the body/head/foot scales in
landmarks.scales(). Nothing assumes a 1.80 m adult.

Each garment registers a material *region* (the ID map): region_id() paints
the same masks that sculpted the cloth, so the ID map lines up with the hems
in the bake exactly.
"""

import math

import numpy as np

from landmarks import LM
from sdf import (Ellipsoid, RoundCone, Capsule, RoundBox, Torus, Custom, frame_from_axis,
                 normalize, fbm, ridged, sample)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def seg_param(P, a, b):
    ab = np.asarray(b - a, dtype=np.float32)
    t = ((P - a) @ ab) / float(ab @ ab)
    closest = a + t[..., None] * ab
    return t, np.linalg.norm(P - closest, axis=-1), closest


def arm_param(Q, J):
    """Parameter along the shoulder->elbow->wrist polyline (0..1 by length),
    distance to it, closest point and whether the forearm segment is the
    closer one — a straight shoulder->wrist line is wrong once the elbow bends."""
    S, E, W = (np.asarray(J[n], dtype=np.float32) for n in ("upperarm_l", "lowerarm_l", "hand_l"))
    L1, L2 = float(np.linalg.norm(E - S)), float(np.linalg.norm(W - E))
    t1, r1, c1 = seg_param(Q, S, E)
    t2, r2, c2 = seg_param(Q, E, W)
    t1u = np.where(t1 < 0, t1, np.clip(t1, 0, 1))
    use2 = r2 < r1
    t = np.where(use2, (L1 + np.clip(t2, 0, None) * L2) / (L1 + L2), t1u * L1 / (L1 + L2))
    r = np.where(use2, r2, r1)
    cl = np.where(use2[..., None], c2, c1)
    return t, r, cl, use2


def arm_length(J):
    return float(np.linalg.norm(J["lowerarm_l"] - J["upperarm_l"]) + np.linalg.norm(J["hand_l"] - J["lowerarm_l"]))


def around_angle(P, a, b, closest):
    """Angle of P around the segment a->b (0 = toward -Y/front)."""
    R = frame_from_axis(np.asarray(b - a), up=(0, -1, 0))
    d = P - closest
    return np.arctan2(d @ R[:, 1], d @ R[:, 0])


def abs_x(P):
    Q = P.copy()
    Q[..., 0] = np.abs(Q[..., 0])
    return Q


def tube_coords(s, th, radius, along=0.035, around=0.10):
    """Periodic noise coordinates on a limb surface (distance along the limb,
    angle around it as cos/sin so the pattern wraps seamlessly)."""
    return np.stack([s / along, np.cos(th) * radius / around, np.sin(th) * radius / around], axis=-1).astype(np.float32)


def not_hands(Q, J, k=1.0):
    """0 on the forearms/hands (they hang next to the hips and thighs in the
    A-pose; torso and leg garments must not grow onto them)."""
    t, r, _cl, _u = arm_param(Q, J)
    rf = LM["r_forearm"]
    return 1.0 - smoothstep(0.30, 0.45, t) * smoothstep(2.2 * rf * k, 1.5 * rf * k, r)


def box(lo, hi):
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return tuple(np.asarray(lo, float)), tuple(np.asarray(hi, float))


def body_box(zlo, zhi):
    (bx0, by0, _), (bx1, by1, _) = LM["bounds"]
    return box((bx0, by0, zlo), (bx1, by1, zhi))


# ---------------------------------------------------------------------------
# Regions (material ID map)
# ---------------------------------------------------------------------------

SKIN = 0
REGIONS = []      # [(name, kind, material dict, mask_fn)] — region id = index + 1
GEAR = {}         # hard-surface bits and shared info (buckle, footwear, pieces)


def register(name, kind, material, mask_fn):
    """kind: 'hair' | 'cloth' | 'footwear' | 'sole' | 'metal' | 'piece'."""
    REGIONS.append((name, kind, dict(material), mask_fn))
    return len(REGIONS)


def region_names():
    return ["skin"] + [r[0] for r in REGIONS]


def region_kind(rid):
    return "skin" if rid == SKIN else REGIONS[rid - 1][1]


def region_material(rid):
    return None if rid == SKIN else REGIONS[rid - 1][2]


def piece_region(k):
    """Region id of mesh piece k (1-based part index from the bake)."""
    ids = [i + 1 for i, r in enumerate(REGIONS) if r[1] == "piece"]
    return ids[k - 1] if 0 < k <= len(ids) else SKIN


def region_id(P, J, part=None):
    """Integer material region per surface point. `part` (per point: 0 body,
    k = the k-th mesh piece) decides the pieces exactly."""
    P = np.asarray(P, dtype=np.float32)
    rid = np.full(P.shape[:-1], SKIN, dtype=np.int32)
    for i, (_n, kind, _m, fn) in enumerate(REGIONS):
        if kind == "piece":
            continue
        rid[fn(P) > 0.5] = i + 1
    if part is not None:
        part = np.asarray(part).astype(int)
        for k in np.unique(part):
            if k > 0:
                rid[part == k] = piece_region(int(k))
    return rid


# ---------------------------------------------------------------------------
# Shirt
# ---------------------------------------------------------------------------

def _neck_masks(Q, neck, sb, v_depth=0.09):
    """1 below the neckline: crew (round, rising at the back), v, scoop."""
    x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
    ny, nb = LM["neck_axis_y"], LM["neck_base_z"]
    rn = np.sqrt(x ** 2 + (y - ny) ** 2)
    front = smoothstep(0.0, -0.03 * sb, y - ny)
    neck_z = nb - 0.003 + 0.030 * sb * smoothstep(-0.06 * sb, 0.05 * sb, y - ny)
    if neck == "scoop":
        neck_z = neck_z - 0.045 * sb * front * smoothstep(0.09 * sb, 0.0, x)
    m = 1.0 - smoothstep(0.095 * sb, 0.075 * sb, rn) * smoothstep(neck_z - 0.0015, neck_z + 0.0015, z)
    if neck == "v":
        zv = nb - v_depth * sb + (v_depth + 0.005) * sb * np.clip(x / (0.075 * sb), 0.0, 1.0)
        m = m * (1.0 - front * smoothstep(zv - 0.0015, zv + 0.0015, z))
    return m


def shirt_mask(P, J, g):
    """1 where the shirt is the outer-or-under layer."""
    Q = abs_x(P)
    z = Q[..., 2]
    sb = LM["s_body"]
    t, r, _cl, _u = arm_param(Q, J)
    ra = LM["r_upperarm"]
    near_arm = smoothstep(2.3 * ra, 1.7 * ra, r) * smoothstep(-0.15, 0.0, t)
    cuff_t = {"long": 0.952, "short": 0.30, "none": -0.02}[g["sleeves"]]
    m_cuff = 1.0 - near_arm * smoothstep(cuff_t - 0.003, cuff_t + 0.003, t)
    hem = LM["belt_z"] - 0.07 * sb if g["tuck"] == "in" else float(J["pelvis"][2]) - 0.06 * sb
    m_bot = smoothstep(hem, hem + 0.005, z)
    m_neck = _neck_masks(Q, g["neck"], sb, v_depth=0.07)
    cz = LM["chin_z"]
    m_head = 1.0 - smoothstep(cz - 0.054 * sb, cz - 0.048 * sb, z)  # never touch the jaw/chin
    return m_cuff * m_bot * m_neck * m_head * not_hands(Q, J, 0.8) ** (1 - near_arm)


def shirt(m, J, g):
    S, E, W = (np.asarray(J[n], dtype=np.float32) for n in ("upperarm_l", "lowerarm_l", "hand_l"))
    L_arm = arm_length(J)
    ease, sleeve_ease = g["thickness"], g["thickness"] * 0.94
    ra = LM["r_upperarm"]

    def fn(P, f):
        Q = abs_x(P)
        t, r, cl, lower = arm_param(Q, J)
        near_arm = smoothstep(2.3 * ra, 1.7 * ra, r) * smoothstep(-0.15, 0.0, t)
        mask = shirt_mask(P, J, g)
        # sleeve folds: ridged noise wrapped around the arm, concentrated at the
        # inner elbow (compression) and above the cuff (stacking)
        th = np.where(lower, around_angle(Q, E, W, cl), around_angle(Q, S, E, cl))
        s = t * L_arm
        seed = np.where(P[..., 0] < 0, 7, 3)
        tc = tube_coords(s, th, 0.045, along=0.030, around=0.07) + seed[..., None]
        rid = ridged(tc, 1.0, 2, seed=13, sharp=4.0) - 0.25
        drape = fbm(tube_coords(s, th, 0.045, along=0.09, around=0.12) + seed[..., None], 1.0, 2, seed=14)
        elbow = np.exp(-((t - 0.50) / 0.09) ** 2) * (0.35 + 0.65 * np.maximum(np.cos(th), 0.0))
        cuff = smoothstep(0.78, 0.92, t) if g["sleeves"] == "long" else 0.0
        amp = (0.0006 + 0.0024 * elbow + 0.0020 * cuff) * near_arm * smoothstep(0.10, 0.22, t)
        folds = amp * rid * 2.0 + 0.0012 * drape * near_arm
        torso = 1.0 - near_arm
        folds += 0.0008 * torso * fbm(P * np.array([1.0, 1.0, 0.35], np.float32), 30.0, 2, seed=5)
        off = ease * torso + sleeve_ease * near_arm * (1.0 - 0.35 * smoothstep(0.85, 0.95, t))
        return f - mask * (off + folds)

    hem = LM["belt_z"] - 0.1 * LM["s_body"]
    m.edit(fn, *body_box(min(hem, float(J["pelvis"][2]) - 0.1), LM["chin_z"] - 0.03))
    register("shirt", "cloth", g, lambda P: shirt_mask(P, J, g))


# ---------------------------------------------------------------------------
# Trousers
# ---------------------------------------------------------------------------

def trousers_hem(J, g):
    sb = LM["s_body"]
    hip, knee, ankle = (float(J[n][2]) for n in ("thigh_l", "calf_l", "foot_l"))
    return {"full": ankle + 0.035 * sb, "knee": knee - 0.06 * sb, "shorts": hip - 0.5 * (hip - knee)}[g["length"]]


def trousers_mask(P, J, g):
    Q = abs_x(P)
    z = Q[..., 2]
    sb = LM["s_body"]
    top = LM["belt_z"] + 0.013 * sb
    hem = trousers_hem(J, g)
    m_top = 1.0 - smoothstep(top - 0.0015, top + 0.0015, z)
    m_hem = smoothstep(hem - 0.0015, hem + 0.0015, z)
    cz = LM["crotch_z"]
    # no ease on the inner thigh right below the crotch, so the legs stay apart
    m_gap = 1.0 - smoothstep(0.010 * sb, 0.0, Q[..., 0]) * smoothstep(cz - 0.045 * sb, cz - 0.075 * sb, z)
    return m_top * m_hem * m_gap * not_hands(Q, J)


def trousers(m, J, g):
    H, A = (np.asarray(J[n], dtype=np.float32) for n in ("thigh_l", "foot_l"))
    L_leg = float(np.linalg.norm(A - H))
    ease = g["thickness"]
    cz = LM["crotch_z"]
    sb = LM["s_body"]

    def fn(P, f):
        Q = abs_x(P)
        z = Q[..., 2]
        t, r, cl = seg_param(Q, H, A)
        mask = trousers_mask(P, J, g)
        th = around_angle(Q, H, A, cl)
        s = t * L_leg
        seed = np.where(P[..., 0] < 0, 9, 4)
        rid = ridged(tube_coords(s, th, 0.07, along=0.040, around=0.10) + seed[..., None], 1.0, 2, seed=23, sharp=4.0) - 0.25
        drape = fbm(tube_coords(s, th, 0.07, along=0.12, around=0.15) + seed[..., None], 1.0, 2, seed=24)
        leg = smoothstep(0.02, 0.10, t)
        knee = np.exp(-((t - 0.50) / 0.08) ** 2) * (0.3 + 0.7 * np.maximum(-np.cos(th), 0.0))
        stack = smoothstep(0.62, 0.72, t) if g["length"] == "full" else 0.0
        folds = leg * (0.0006 + 0.0022 * knee + 0.0020 * stack) * rid * 2.0 + 0.0014 * drape * leg
        crotch = np.exp(-((z - (cz - 0.035 * sb)) / (0.05 * sb)) ** 2) * smoothstep(0.09 * sb, 0.03 * sb, Q[..., 0])
        folds += 0.0010 * crotch * ridged(P * np.array([30, 12, 60], np.float32), 1.0, 1, seed=25)
        return f - mask * (ease + folds)

    m.edit(fn, *body_box(trousers_hem(J, g) - 0.03, LM["belt_z"] + 0.04))
    register("trousers", "cloth", g, lambda P: trousers_mask(P, J, g))


# ---------------------------------------------------------------------------
# Vest / jerkin
# ---------------------------------------------------------------------------

def vest_mask(P, J, g):
    Q = abs_x(P)
    x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
    sb = LM["s_body"]
    t, r, _cl, _u = arm_param(Q, J)
    ra = LM["r_upperarm"]
    tl = t * arm_length(J)                         # meters along the arm from the shoulder joint
    arm = smoothstep(2.19 * ra, 1.67 * ra, r) * smoothstep(-0.077 * sb, -0.055 * sb, tl)
    hem = float(J["pelvis"][2]) - 0.049 * sb if g["length"] == "hip" else LM["belt_z"] - 0.012 * sb
    m_hem = smoothstep(hem - 0.0015, hem + 0.0015, z)
    nb, ny = LM["neck_base_z"], LM["neck_axis_y"]
    front = smoothstep(0.0, -0.03 * sb, y - ny)
    if g["neck"] == "v":
        zv = nb - 0.130 * sb + 0.135 * sb * np.clip(x / (0.075 * sb), 0.0, 1.0)
    else:
        zv = nb + 0.005 * sb + 0.0 * x
    m_v = 1.0 - front * smoothstep(zv - 0.0015, zv + 0.0015, z)
    rn = np.sqrt(x ** 2 + (y - ny) ** 2)
    m_back = 1.0 - smoothstep(0.100 * sb, 0.085 * sb, rn) * smoothstep(nb + 0.008 * sb, nb + 0.011 * sb, z)
    top = 1.0 - smoothstep(nb + 0.045 * sb, nb + 0.055 * sb, z)
    return (1.0 - arm) * m_hem * m_v * m_back * top * not_hands(Q, J)


def vest(m, J, g):
    """Sleeveless vest/jerkin, V or crew neck, waist or hip length, center seam."""
    thick = g["thickness"]
    sb = LM["s_body"]

    def fn(P, f):
        Q = abs_x(P)
        x = Q[..., 0]
        front = smoothstep(0.0, -0.03 * sb, Q[..., 1] - LM["neck_axis_y"])
        mask = vest_mask(P, J, g)
        seam = 0.0016 * np.exp(-(x / 0.0018) ** 2) * front
        wear = 0.0006 * fbm(P, 45.0, 2, seed=31)
        return f - mask * (thick + wear) + seam * mask

    m.edit(fn, *body_box(float(J["pelvis"][2]) - 0.08 * sb, LM["neck_base_z"] + 0.06 * sb))
    register("vest", "cloth", g, lambda P: vest_mask(P, J, g))


# ---------------------------------------------------------------------------
# Belt + buckle
# ---------------------------------------------------------------------------

def belt_mask(P, J, g):
    z = P[..., 2]
    c, w = LM["belt_z"], g["width"] * LM["s_body"]
    band = smoothstep(c - w / 2 - 0.0012, c - w / 2 + 0.0012, z) * (1.0 - smoothstep(c + w / 2 - 0.0012, c + w / 2 + 0.0012, z))
    return band * not_hands(abs_x(P), J)


def surface_point(m, origin, direction, max_dist=0.6, step=0.0015):
    """March from `origin` along `direction` until the field changes sign —
    seats hard-surface gear (buckle, pouch) onto the sculpted surface."""
    o = np.asarray(origin, np.float32)
    d = normalize(direction).astype(np.float32)
    ts = np.arange(0.0, max_dist, step, dtype=np.float32)
    pts = o + ts[:, None] * d
    vals = sample(m, pts)
    idx = np.nonzero(vals < 0)[0]
    if len(idx) == 0:
        return None
    return pts[idx[0]]


def belt(m, J, g):
    thick = g["thickness"]
    c, w = LM["belt_z"], g["width"] * LM["s_body"]
    m.edit(lambda P, f: f - belt_mask(P, J, g) * thick, *body_box(c - w, c + w))
    register("belt", "cloth", g, lambda P: belt_mask(P, J, g))
    if g["buckle"] != "none":
        s = LM["s_body"]
        p = surface_point(m, (0, LM["bounds"][0][1], c), (0, 1, 0))
        if p is None:
            return
        cc = p + np.array([0, -0.002, 0], np.float32)
        GEAR["buckle"] = RoundBox(cc, (0.024 * s, 0.0045, 0.021 * s), 0.0015)
        m.add(RoundBox(cc, (0.024 * s, 0.0022, 0.021 * s), 0.0015), k=0.0015)
        m.sub(RoundBox(cc + np.array([0.004 * s, -0.004, 0], np.float32), (0.015 * s, 0.006, 0.0135 * s), 0.001), k=0.001)
        m.add(Capsule(cc + np.array([-0.012 * s, -0.001, 0], np.float32), cc + np.array([0.012 * s, -0.003, 0], np.float32), 0.0016), k=0.001)
        buckle = GEAR["buckle"]
        register("buckle", "metal", {"material": "metal", "metal": g["buckle"]}, lambda P: (buckle.eval(P) < 0.0015).astype(np.float32))


# ---------------------------------------------------------------------------
# Footwear: boots (ankle / calf / knee) and shoes
# ---------------------------------------------------------------------------

def footwear_top(J, g):
    ankle, knee = float(J["foot_l"][2]), float(J["calf_l"][2])
    if g["type"] == "shoes":
        return ankle + 0.03 * LM["s_foot"]
    return ankle + {"ankle": 0.25, "calf": 0.61, "knee": 0.92}[g["height"]] * (knee - ankle)


def footwear_mask(P, J, g):
    top = footwear_top(J, g)
    Q = abs_x(P)
    z = Q[..., 2]
    return (z < top + 0.007).astype(np.float32) * not_hands(Q, J)


def footwear(m, J, g):
    """Boots/shoes: a stiff leather shaft (round cone around the leg, wide
    enough for the calf and the trousers), heel counter, instep and toe box
    scaled by the foot, a sole outlined from forefoot + heel ellipses, welt
    cuff and ankle creases."""
    from landmarks import leg_radius
    A, B = np.asarray(J["foot_l"], np.float32), np.asarray(J["ball_l"], np.float32)
    sf, sb = LM["s_foot"], LM["s_body"]
    boots = g["type"] == "boots"
    top = footwear_top(J, g)
    GEAR["footwear"] = {"top_z": top, "type": g["type"]}
    parts = []
    if boots:
        H, K = np.asarray(J["thigh_l"], np.float32), np.asarray(J["calf_l"], np.float32)
        axis = lambda z: np.array([np.interp(z, [A[2], K[2], H[2]], [A[i], K[i], H[i]]) for i in (0, 1)] + [z], np.float32)
        zs = np.linspace(A[2] + 0.03 * sf, top, 12)
        r_leg = max(leg_radius(LM, float(z)) for z in zs)
        shaft_top = axis(top) + np.array([-0.002 * sb, -0.012 * sb, 0], np.float32)
        r_top = r_leg + 0.0105
        r_bot = max(leg_radius(LM, float(A[2]) + 0.04 * sf) + 0.008, 0.80 * r_top)
        parts.append((RoundCone(shaft_top, A + np.array([0, 0.004 * sf, 0.01 * sf], np.float32), r_top, r_bot), 0.0))
    parts.append((Ellipsoid(A + np.array([0, 0.030, -0.030], np.float32) * sf, tuple(np.array([0.044, 0.052, 0.060]) * sf)), 0.03 * sf))
    parts.append((RoundCone(A + np.array([0, -0.010, -0.010], np.float32) * sf, B + np.array([0, 0, 0.022], np.float32) * sf,
                            0.044 * sf, 0.032 * sf), 0.03 * sf))
    parts.append((Ellipsoid(B + np.array([-0.002, -0.030, 0.018], np.float32) * sf, tuple(np.array([0.050, 0.085, 0.036]) * sf)), 0.03 * sf))
    for prim, k in parts:
        m.add(prim, k=k, sym=True)
    Ax, Ay, By = float(A[0]), float(A[1]), float(B[1])

    def sole_fn(P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        fore = np.sqrt(((x - Ax + 0.002 * sf) / (0.053 * sf)) ** 2 + ((y - (By - 0.030 * sf)) / (0.090 * sf)) ** 2) - 1.0
        heel = np.sqrt(((x - Ax) / (0.045 * sf)) ** 2 + ((y - (Ay + 0.030 * sf)) / (0.050 * sf)) ** 2) - 1.0
        mid = np.sqrt(((x - Ax + 0.004 * sf) / (0.040 * sf)) ** 2 + ((y - 0.5 * (Ay + By)) / (0.085 * sf)) ** 2) - 1.0
        d2 = np.minimum(np.minimum(fore * 0.05 * sf, heel * 0.045 * sf), mid * 0.04 * sf)
        tz = np.where(y > Ay - 0.01 * sf, 0.026 if boots else 0.020, 0.016) * sf  # heel block
        dz = np.maximum(-z, z - tz)
        return np.maximum(d2, dz) - 0.0015

    m.add(Custom(sole_fn, (Ax - 0.08 * sf, By - 0.16 * sf, -0.005), (Ax + 0.08 * sf, Ay + 0.12 * sf, 0.035 * sf)), k=0.003, sym=True)
    if boots:
        # folded cuff at the top of the shaft
        m.add(Torus(shaft_top + np.array([0, 0, -0.012 * sb], np.float32), r_top, 0.007 * sb, tube_scale=(0.9, 1.6)), k=0.003, sym=True)

        # ankle creases: horizontal wrinkles across the shaft front
        def fn(P, f):
            Q = abs_x(P)
            z = Q[..., 2]
            z0 = float(A[2])
            band = smoothstep(z0 + 0.026 * sf, z0 + 0.066 * sf, z) * (1.0 - smoothstep(top - 0.085 * sb, top - 0.045 * sb, z))
            front = smoothstep(0.0, -0.03, Q[..., 1] - A[1])
            nz = fbm(P, 25.0, 2, seed=41)
            return f - band * front * 0.0012 * np.sin(z * 2 * math.pi / (0.022 * sb) + 3.0 * nz)

        m.edit(fn, *body_box(float(A[2]), top))
    # flatten the soles on the ground plane
    (bx0, by0, _), (bx1, by1, _) = LM["bounds"]
    m.inter(Custom(lambda P: -P[..., 2], (bx0, by0, -0.04), (bx1, by1, 0.002)), k=0.0)
    register(g["type"], "footwear", g, lambda P: footwear_mask(P, J, g))
    sole = dict(g, color=g["sole"], material="leather")
    register("sole", "sole", sole, lambda P: ((P[..., 2] < 0.021 * sf) & (footwear_mask(P, J, g) > 0.5)).astype(np.float32))


# ---------------------------------------------------------------------------
# Gloves and bracers
# ---------------------------------------------------------------------------

def gloves_mask(P, J, g):
    Q = abs_x(P)
    t, r, _cl, _u = arm_param(Q, J)
    rf = LM["r_forearm"]
    m = smoothstep(0.975, 0.985, t) * smoothstep(4.0 * rf, 3.0 * rf, r)
    if g["fingers"] == "none":
        W = np.asarray(J["hand_l"], np.float32)
        ax = np.asarray(J["_hand_axis_l"], np.float32)
        ch = J["_chains_l"]
        mcp = float(np.mean([np.dot(ch[f][0][0] - W, ax) for f in ("index", "middle", "ring", "pinky")]))
        along = (Q - W) @ ax
        m = m * (1.0 - smoothstep(mcp + 0.010 * LM["s_body"], mcp + 0.014 * LM["s_body"], along))
        th = ch["thumb"][0]
        tt, tr, _c = seg_param(Q, th[1], th[3])
        m = m * (1.0 - smoothstep(0.35, 0.45, tt) * smoothstep(0.03, 0.02, tr))
    return m


def gloves(m, J, g):
    thick = g["thickness"]
    m.edit(lambda P, f: f - gloves_mask(P, J, g) * thick, *body_box(0.0, float(J["lowerarm_l"][2]) + 0.05))
    register("gloves", "cloth", g, lambda P: gloves_mask(P, J, g))


def bracers_mask(P, J, g):
    Q = abs_x(P)
    t, r, _cl, _u = arm_param(Q, J)
    rf = LM["r_forearm"]
    return smoothstep(0.615, 0.625, t) * (1.0 - smoothstep(0.925, 0.935, t)) * smoothstep(2.6 * rf, 1.9 * rf, r)


def bracers(m, J, g):
    thick = g["thickness"]

    def fn(P, f):
        Q = abs_x(P)
        t, _r, _cl, _u = arm_param(Q, J)
        lip = 0.0012 * (np.exp(-((t - 0.63) / 0.006) ** 2) + np.exp(-((t - 0.92) / 0.006) ** 2))   # rolled edges
        return f - bracers_mask(P, J, g) * (thick + lip)

    m.edit(fn, *body_box(0.0, float(J["lowerarm_l"][2]) + 0.05))
    register("bracers", "cloth", g, lambda P: bracers_mask(P, J, g))


# ---------------------------------------------------------------------------
# Hair: sculpted game-style hair cap + strand-clump relief
# ---------------------------------------------------------------------------

# hairline height around the head by azimuth (0 = front, pi = back), as
# offsets from eye height in Ranger-head units (scaled by the head size)
HAIRLINES = {
    "receding": [(0.00, 0.086), (0.35, 0.080), (0.70, 0.058), (0.95, 0.028), (1.10, -0.014), (1.22, -0.022),
                 (1.40, 0.018), (1.62, 0.016), (1.85, -0.029), (2.30, -0.059), (2.80, -0.072), (math.pi, -0.076)],
    "straight": [(0.00, 0.074), (0.35, 0.072), (0.70, 0.060), (0.95, 0.040), (1.10, 0.002), (1.22, -0.012),
                 (1.40, 0.016), (1.62, 0.014), (1.85, -0.030), (2.30, -0.062), (2.80, -0.076), (math.pi, -0.080)],
    "rounded": [(0.00, 0.070), (0.35, 0.066), (0.70, 0.052), (0.95, 0.028), (1.10, -0.008), (1.22, -0.020),
                (1.40, 0.014), (1.62, 0.010), (1.85, -0.036), (2.30, -0.072), (2.80, -0.090), (math.pi, -0.094)],
}
HAIR_STYLES = {        # top thickness, side thickness, clump depth, front lift
    "buzz": (0.0035, 0.0025, 0.0004, 0.0),
    "short": (0.0160, 0.0065, 0.0018, 0.004),
    "mohawk": (0.0240, 0.0012, 0.0022, 0.0),
}


def head_c():
    return np.asarray(LM["head_c"], np.float32)


def hairline_z(phi, kind="receding"):
    tab = HAIRLINES[kind]
    a = np.array([p[0] for p in tab])
    zz = np.array([p[1] for p in tab]) * LM["s_head"] + float(LM["eye_l"][2])
    return np.interp(np.abs(phi), a, zz)


def ear_mask(P):
    e = np.asarray(LM["ear_l"], np.float32)
    sh = LM["s_head"]
    ex = np.abs(P[..., 0])
    d = np.sqrt((ex - e[0] + 0.004 * sh) ** 2 + (P[..., 1] - e[1]) ** 2 + ((P[..., 2] - e[2]) / 1.6) ** 2) - 0.018 * sh
    return smoothstep(0.012 * sh, 0.004 * sh, d)


def hair_mask(P, h=None):
    h = h or HAIR_SPEC
    d = P - head_c()
    phi = np.arctan2(d[..., 0], -d[..., 1])
    zl = hairline_z(phi, h["hairline"])
    z = P[..., 2]
    sh = LM["s_head"]
    near = np.linalg.norm(d, axis=-1) < 0.16 * sh
    return smoothstep(zl - 0.002 * sh, zl + 0.024 * sh, z) ** 2.0 * (1.0 - ear_mask(P)) * near


HAIR_SPEC = {"hairline": "receding", "style": "short"}


def hair(m, J, h):
    HAIR_SPEC.clear()
    HAIR_SPEC.update(h)
    if h["style"] == "bald":
        return
    ez = float(LM["eye_l"][2])
    sh = LM["s_head"]
    top, side, clump, lift = HAIR_STYLES[h["style"]]
    top, side, clump, lift = top * sh, side * sh, clump * sh, lift * sh

    def fn(P, f):
        d = P - head_c()
        phi = np.arctan2(d[..., 0], -d[..., 1])
        z = P[..., 2]
        vert = smoothstep(ez - 0.004 * sh, ez + 0.106 * sh, z)
        thick = side + (top - side) * vert
        if h["style"] == "mohawk":
            strip = smoothstep(0.034 * sh, 0.020 * sh, np.abs(P[..., 0]))
            thick = side + (top - side) * strip * smoothstep(ez - 0.03 * sh, ez + 0.05 * sh, z)
        else:
            # a little extra lift at the front (short, textured, pushed back)
            thick = thick + lift * np.exp(-(phi / 0.6) ** 2) * smoothstep(ez + 0.056 * sh, ez + 0.096 * sh, z)
        # strand clumps stretched along the combing direction: front-to-back
        # on top, top-to-bottom on the sides
        top_c = fbm(P * np.array([1.0, 0.22, 1.0], np.float32), 160.0 / sh, 3, seed=51)
        side_c = fbm(P * np.array([1.0, 1.0, 0.25], np.float32), 160.0 / sh, 3, seed=53)
        clumps = clump * (vert * top_c + (1 - vert) * side_c) + 0.66 * clump * fbm(P, 45.0 / sh, 2, seed=52)
        return f - hair_mask(P, h) * (thick + clumps)

    c = head_c()
    m.edit(fn, (-0.13 * sh, c[1] - 0.17 * sh, ez - 0.12 * sh), (0.13 * sh, c[1] + 0.17 * sh, LM["head_top"] + 0.05 * sh))
    register("hair", "hair", {"material": "hair", "color": h["color"], "style": h["style"]}, lambda P: hair_mask(P, h))


# ---------------------------------------------------------------------------
# Beard (short, sculpted on the jaw; long beards are a separate piece)
# ---------------------------------------------------------------------------

def beard_mask(P):
    """Cheeks below the cheekbones, jaw, chin, under the chin and the
    moustache — the lips stay clear."""
    sh = LM["s_head"]
    E, ST, NT, EAR = (np.asarray(LM[k], np.float32) for k in ("eye_l", "stomion", "nose_tip", "ear_l"))
    c = head_c()
    d = P - c
    phi = np.abs(np.arctan2(d[..., 0], -d[..., 1]))            # 0 front .. pi/2 at the ears
    z = P[..., 2]
    nose_base = float(NT[2]) - 0.012 * sh
    ztop = np.interp(phi, [0.0, 0.30, 0.55, 0.95, 1.35, 1.6],
                     [nose_base, nose_base - 0.002 * sh, float(E[2]) - 0.040 * sh, float(E[2]) - 0.034 * sh,
                      float(EAR[2]) + 0.012 * sh, float(EAR[2]) + 0.012 * sh])
    zbot = np.interp(phi, [0.0, 0.5, 1.0, 1.35, 1.6],
                     [LM["chin_z"] - 0.030 * sh, LM["chin_z"] - 0.024 * sh, float(ST[2]) - 0.045 * sh,
                      float(EAR[2]) - 0.030 * sh, float(EAR[2]) - 0.020 * sh])
    band = smoothstep(zbot - 0.004 * sh, zbot + 0.006 * sh, z) * (1.0 - smoothstep(ztop - 0.006 * sh, ztop + 0.002 * sh, z))
    front = smoothstep(1.75, 1.55, phi)
    lips = np.exp(-(((P[..., 0]) / (0.028 * sh)) ** 2 + ((z - float(ST[2])) / (0.007 * sh)) ** 2))
    near = np.linalg.norm(d, axis=-1) < 0.15 * sh
    return band * front * (1.0 - smoothstep(0.35, 0.6, lips)) * near * (1.0 - ear_mask(P))


def beard(m, J, b):
    if b["style"] not in ("short", "long"):
        return
    sh = LM["s_head"]
    thick = 0.0075 * sh

    def fn(P, f):
        clumps = 0.0016 * sh * fbm(P * np.array([1.0, 1.0, 0.3], np.float32), 150.0 / sh, 3, seed=61)
        return f - beard_mask(P) * (thick + clumps)

    c = head_c()
    m.edit(fn, (-0.13 * sh, c[1] - 0.17 * sh, LM["chin_z"] - 0.06 * sh), (0.13 * sh, c[1] + 0.12 * sh, float(LM["eye_l"][2])))
    register("beard", "hair", {"material": "hair", "color": b["color"], "style": "beard"}, beard_mask)


# ---------------------------------------------------------------------------

# paint/sculpt order: under-layers first (a later layer wins where they overlap)
ORDER = ["shirt", "trousers", "vest", "bracers", "gloves", "belt", "boots", "shoes"]
BUILD = {"shirt": shirt, "trousers": trousers, "vest": vest, "bracers": bracers, "gloves": gloves,
         "belt": belt, "boots": footwear, "shoes": footwear}


def dress(m, J, S, hair_on=True):
    """Sculpt the spec's outfit, hair and beard onto the body model; pieces
    (pouch, long hair, horns, ...) are built by pieces.py."""
    GEAR.clear()
    REGIONS.clear()
    order = list(ORDER)
    shirt_g = next((g for g in S["outfit"] if g["type"] == "shirt"), None)
    if shirt_g and shirt_g["tuck"] == "out":           # an untucked shirt goes over the trousers
        order.remove("shirt")
        order.insert(order.index("trousers") + 1, "shirt")
    for t in order:
        for g in S["outfit"]:
            if g["type"] == t:
                BUILD[t](m, J, g)
    if hair_on:
        hair(m, J, S["hair"])
        beard(m, J, S["beard"])
    import pieces
    pieces.build(m, J, S)
