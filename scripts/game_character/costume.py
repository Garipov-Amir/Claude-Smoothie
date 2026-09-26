"""
costume — clothing, boots, belt gear and hair for the humanoid sculpt.

Cloth is sculpted the way it is in ZBrush/Marvelous-to-ZBrush workflows,
translated to distance fields:

* A garment is an *offset shell* of whatever is under it (body, or the
  previous garment): inside the garment's region the field is lowered by
  the cloth thickness + ease, so the surface grows outward.
* The region mask falls off over ~1-2 mm at hems, cuffs and necklines, which
  produces a real ledge — the thickness step a normal-map bake picks up as a
  crisp layered edge on the low-poly.
* Folds are displacement patterns in garment-local coordinates (distance
  along the limb, angle around it): spiral "pipe" folds on sleeves, compression
  folds at the inner elbow and knee, stacking above cuffs and boot tops. They
  are modulated with noise so no two folds repeat.

Hard gear (boots, buckle, pouch) is built from primitives and unioned on top.
"""

import math
import numpy as np

from landmarks import LM
from sdf import (SDFModel, Ellipsoid, RoundCone, Capsule, RoundBox, Torus, Custom, frame_from_axis,
                 euler_matrix, normalize, smin, smax, fbm, value_noise, ridged, sample)


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
    distance to it, closest point and the local segment ends — a straight
    shoulder->wrist line is wrong as soon as the elbow is bent."""
    S, E, W = (np.asarray(J[n], dtype=np.float32) for n in ("upperarm_l", "lowerarm_l", "hand_l"))
    L1, L2 = float(np.linalg.norm(E - S)), float(np.linalg.norm(W - E))
    t1, r1, c1 = seg_param(Q, S, E)
    t2, r2, c2 = seg_param(Q, E, W)
    # extend the ends so points past the shoulder/wrist still parametrize
    t1u = np.where(t1 < 0, t1, np.clip(t1, 0, 1))
    use2 = r2 < r1
    t = np.where(use2, (L1 + np.clip(t2, 0, None) * L2) / (L1 + L2), t1u * L1 / (L1 + L2))
    r = np.where(use2, r2, r1)
    cl = np.where(use2[..., None], c2, c1)
    return t, r, cl, use2


def around_angle(P, a, b, closest):
    """Angle of P around the segment a->b (0 = toward -Y/front)."""
    R = frame_from_axis(np.asarray(b - a), up=(0, -1, 0))
    d = P - closest
    return np.arctan2(d @ R[:, 1], d @ R[:, 0])


def abs_x(P):
    Q = P.copy()
    Q[..., 0] = np.abs(Q[..., 0])
    return Q


def side_sign(P):
    return np.where(P[..., 0] >= 0, 1.0, -1.0).astype(np.float32)


def tube_coords(s, th, radius, along=0.035, around=0.10):
    """Periodic 3-D noise coordinates on a limb surface: distance along the
    limb and angle around it (as cos/sin so the pattern wraps seamlessly).
    along << around stretches features around the limb — folds wrap."""
    return np.stack([s / along, np.cos(th) * radius / around, np.sin(th) * radius / around], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Garments
# ---------------------------------------------------------------------------

def shirt_mask(P, J):
    """1 where the shirt is the outer-or-under layer (torso above the tuck,
    sleeves up to the cuff, below the neckline)."""
    Q = abs_x(P)
    z = Q[..., 2]
    t, r, _cl, _u = arm_param(Q, J)
    near_arm = smoothstep(0.11, 0.08, r) * smoothstep(-0.15, 0.0, t)
    cuff_t = 0.952
    m_cuff = 1.0 - near_arm * smoothstep(cuff_t - 0.003, cuff_t + 0.003, t)
    m_bot = smoothstep(0.935, 0.940, z)
    ny, nb = LM["neck_axis_y"], LM["neck_base_z"]
    rn = np.sqrt(Q[..., 0] ** 2 + (Q[..., 1] - ny) ** 2)
    neck_z = nb - 0.003 + 0.030 * smoothstep(-0.06, 0.05, Q[..., 1] - ny)
    m_neck = 1.0 - smoothstep(0.095, 0.075, rn) * smoothstep(neck_z - 0.0015, neck_z + 0.0015, z)
    cz = LM["chin_z"]
    m_head = 1.0 - smoothstep(cz - 0.054, cz - 0.048, z)  # never touch the jaw/chin
    return m_cuff * m_bot * m_neck * m_head


def shirt(m, J, ease=0.0048, sleeve_ease=0.0045):
    S, E, W = (np.asarray(J[n], dtype=np.float32) for n in ("upperarm_l", "lowerarm_l", "hand_l"))
    L_arm = float(np.linalg.norm(W - S))

    L_arm = float(np.linalg.norm(E - S) + np.linalg.norm(W - E))

    def fn(P, f):
        Q = abs_x(P)
        z = Q[..., 2]
        t, r, cl, lower = arm_param(Q, J)
        near_arm = smoothstep(0.11, 0.08, r) * smoothstep(-0.15, 0.0, t)
        mask = shirt_mask(P, J)
        # sleeve folds: ridged noise wrapped around the arm, concentrated at the
        # inner elbow (compression) and above the cuff (stacking)
        th = np.where(lower, around_angle(Q, E, W, cl), around_angle(Q, S, E, cl))
        s = t * L_arm
        seed = np.where(P[..., 0] < 0, 7, 3)
        tc = tube_coords(s, th, 0.045, along=0.030, around=0.07) + seed[..., None]
        rid = ridged(tc, 1.0, 2, seed=13, sharp=4.0) - 0.25
        drape = fbm(tube_coords(s, th, 0.045, along=0.09, around=0.12) + seed[..., None], 1.0, 2, seed=14)
        elbow = np.exp(-((t - 0.50) / 0.09) ** 2) * (0.35 + 0.65 * np.maximum(np.cos(th), 0.0))
        cuff = smoothstep(0.78, 0.92, t)
        amp = (0.0006 + 0.0024 * elbow + 0.0020 * cuff) * near_arm * smoothstep(0.10, 0.22, t)
        folds = amp * rid * 2.0 + 0.0012 * drape * near_arm
        # torso: soft drag folds from the shoulder toward the chest
        torso = 1.0 - near_arm
        folds += 0.0008 * torso * fbm(P * np.array([1.0, 1.0, 0.35], np.float32), 30.0, 2, seed=5)
        off = ease * torso + sleeve_ease * near_arm * (1.0 - 0.35 * smoothstep(0.85, 0.95, t))
        return f - mask * (off + folds)

    m.edit(fn, (-0.8, -0.46, 0.90), (0.8, 0.25, LM["chin_z"] - 0.03))


def trousers_mask(P):
    Q = abs_x(P)
    z = Q[..., 2]
    m_top = 1.0 - smoothstep(1.018, 1.021, z)
    m_side = smoothstep(0.30, 0.26, Q[..., 0])  # keep hands out of the region
    # no ease on the inner thigh right below the crotch, so the legs stay apart
    m_gap = 1.0 - smoothstep(0.010, 0.0, Q[..., 0]) * smoothstep(0.83, 0.80, z)
    return m_top * m_side * m_gap * (1.0 - smoothstep(0.30, 0.20, z) * 0.0)


def trousers(m, J, ease=0.0055):
    H, K, A = (np.asarray(J[n], dtype=np.float32) for n in ("thigh_l", "calf_l", "foot_l"))
    L_leg = float(np.linalg.norm(A - H))

    def fn(P, f):
        Q = abs_x(P)
        z = Q[..., 2]
        t, r, cl = seg_param(Q, H, A)
        mask = trousers_mask(P)
        th = around_angle(Q, H, A, cl)
        s = t * L_leg
        seed = np.where(P[..., 0] < 0, 9, 4)
        rid = ridged(tube_coords(s, th, 0.07, along=0.040, around=0.10) + seed[..., None], 1.0, 2, seed=23, sharp=4.0) - 0.25
        drape = fbm(tube_coords(s, th, 0.07, along=0.12, around=0.15) + seed[..., None], 1.0, 2, seed=24)
        leg = smoothstep(0.02, 0.10, t)
        knee = np.exp(-((t - 0.50) / 0.08) ** 2) * (0.3 + 0.7 * np.maximum(-np.cos(th), 0.0))
        stack = smoothstep(0.62, 0.72, t)  # bunching above the boot shaft
        folds = leg * (0.0006 + 0.0022 * knee + 0.0020 * stack) * rid * 2.0 + 0.0014 * drape * leg
        crotch = np.exp(-((z - 0.84) / 0.05) ** 2) * smoothstep(0.09, 0.03, Q[..., 0])
        folds += 0.0010 * crotch * ridged(P * np.array([30, 12, 60], np.float32), 1.0, 1, seed=25)
        return f - mask * (ease + folds)

    m.edit(fn, (-0.4, -0.35, 0.15), (0.4, 0.25, 1.05))


def jerkin_mask(P, J):
    Q = abs_x(P)
    x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
    t, r, _cl, _u = arm_param(Q, J)
    L = float(np.linalg.norm(J["lowerarm_l"] - J["upperarm_l"]) + np.linalg.norm(J["hand_l"] - J["lowerarm_l"]))
    t = t * L / 0.55   # the thresholds below were tuned in meters-ish units of a 0.55 m arm
    arm = smoothstep(0.105, 0.080, r) * smoothstep(-0.14, -0.10, t)
    m_hem = smoothstep(0.918, 0.921, z)
    nb, ny = LM["neck_base_z"], LM["neck_axis_y"]
    zv = nb - 0.130 + 0.135 * np.clip(x / 0.075, 0.0, 1.0)
    front = smoothstep(0.0, -0.03, y - ny)
    m_v = 1.0 - front * smoothstep(zv - 0.0015, zv + 0.0015, z)
    rn = np.sqrt(x ** 2 + (y - ny) ** 2)
    m_back = 1.0 - smoothstep(0.100, 0.085, rn) * smoothstep(nb + 0.008, nb + 0.011, z)
    return (1.0 - arm) * m_hem * m_v * m_back * smoothstep(0.34, 0.30, x) * (1.0 - smoothstep(nb + 0.045, nb + 0.055, z))


def jerkin(m, J, thick=0.0060):
    """Sleeveless leather jerkin with a V neck, hip-length, center-front seam."""
    S, W = (np.asarray(J[n], dtype=np.float32) for n in ("upperarm_l", "hand_l"))

    def fn(P, f):
        Q = abs_x(P)
        x, y = Q[..., 0], Q[..., 1]
        front = smoothstep(0.0, -0.03, y - LM["neck_axis_y"])
        mask = jerkin_mask(P, J)
        seam = 0.0016 * np.exp(-(x / 0.0018) ** 2) * front
        wear = 0.0006 * fbm(P, 45.0, 2, seed=31)
        return f - mask * (thick + wear) + seam * mask

    m.edit(fn, (-0.36, -0.35, 0.90), (0.36, 0.25, LM["neck_base_z"] + 0.06))


def belt_mask(P):
    x, z = np.abs(P[..., 0]), P[..., 2]
    return smoothstep(0.9845, 0.9870, z) * (1.0 - smoothstep(1.0235, 1.0260, z)) * smoothstep(0.30, 0.26, x)


def belt(m, J, thick=0.0055):
    def fn(P, f):
        return f - belt_mask(P) * thick

    m.edit(fn, (-0.32, -0.35, 0.97), (0.32, 0.25, 1.04))


def surface_point(m, origin, direction, max_dist=0.4, step=0.0015):
    """March from `origin` along `direction` until the field changes sign —
    used to seat hard-surface gear (buckle, pouch) onto the sculpted surface."""
    o = np.asarray(origin, np.float32)
    d = normalize(direction).astype(np.float32)
    ts = np.arange(0.0, max_dist, step, dtype=np.float32)
    pts = o + ts[:, None] * d
    vals = sample(m, pts)
    idx = np.nonzero(vals < 0)[0]
    if len(idx) == 0:
        return None
    return pts[idx[0]]


GEAR = {}


def gear(m, J):
    # buckle: rectangular frame + prong, seated on the belt front
    p = surface_point(m, (0, -0.35, 1.005), (0, 1, 0))
    c = p + np.array([0, -0.002, 0], np.float32)
    GEAR["buckle"] = RoundBox(c, (0.024, 0.0045, 0.021), 0.0015)
    frame = RoundBox(c, (0.024, 0.0022, 0.021), 0.0015)
    m.add(frame, k=0.0015)
    m.sub(RoundBox(c + np.array([0.004, -0.004, 0], np.float32), (0.015, 0.006, 0.0135), 0.001), k=0.001)
    m.add(Capsule(c + np.array([-0.012, -0.001, 0], np.float32), c + np.array([0.012, -0.003, 0], np.float32), 0.0016), k=0.001)
    # belt pouch on the character's right hip: a *separate* model (and later
    # a separate mesh piece) — wrapping one skin mesh over a hard box folds
    # it into self-intersections; production gear is its own geometry
    q = surface_point(m, (-0.35, -0.03, 0.975), (1, 0, 0))
    R = euler_matrix(0, 0, 18)
    body_c = q + np.array([-0.018, -0.004, -0.018], np.float32)
    GEAR["pouch"] = RoundBox(body_c + np.array([-0.004, 0, 0.004], np.float32), (0.020, 0.046, 0.052), 0.009, R)
    GEAR["pouch_frame"] = (body_c, R, np.array([0.014 + 0.009, 0.040 + 0.009, 0.045 + 0.009]))
    pm = SDFModel()
    pm.add(RoundBox(body_c, (0.014, 0.040, 0.045), 0.009, R), k=0.0)
    pm.add(RoundBox(body_c + np.array([-0.012, -0.002, 0.030], np.float32), (0.006, 0.043, 0.018), 0.004, R), k=0.002)
    pm.add(RoundBox(q + np.array([-0.006, -0.004, 0.012], np.float32), (0.004, 0.012, 0.030), 0.002, R), k=0.002)  # loop

    def seam(P, f):  # stitched seam around the pouch's sides
        Q = (P - body_c) @ R
        band = smoothstep(0.0015, 0.0, np.abs(np.abs(Q[..., 1]) - 0.040))
        return f + 0.0006 * band
    pm.edit(seam, body_c - 0.08, body_c + 0.08)
    GEAR["pouch_model"] = pm


def boots(m, J):
    A, B = np.asarray(J["foot_l"], np.float32), np.asarray(J["ball_l"], np.float32)
    parts = []
    shaft_top = np.array([A[0] - 0.002, A[1] - 0.012, 0.345], np.float32)
    parts.append((RoundCone(shaft_top, A + np.array([0, 0.004, 0.01], np.float32), 0.056, 0.046), 0.0))
    parts.append((Ellipsoid(A + np.array([0, 0.030, -0.030], np.float32), (0.044, 0.052, 0.060)), 0.03))  # heel counter
    parts.append((RoundCone(A + np.array([0, -0.010, -0.010], np.float32), B + np.array([0, 0, 0.022], np.float32), 0.044, 0.032), 0.03))  # instep
    parts.append((Ellipsoid(B + np.array([-0.002, -0.030, 0.018], np.float32), (0.050, 0.085, 0.036)), 0.03))  # toe box
    for prim, k in parts:
        m.add(prim, k=k, sym=True)
    # sole: 2-D outline (forefoot + heel ellipses, joined at the waist of the foot)
    # extruded as a flat slab — follows the boot instead of a rectangular plate
    Ax, Ay, By = float(A[0]), float(A[1]), float(B[1])

    def sole_fn(P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        fore = np.sqrt(((x - Ax + 0.002) / 0.053) ** 2 + ((y - (By - 0.030)) / 0.090) ** 2) - 1.0
        heel = np.sqrt(((x - Ax) / 0.045) ** 2 + ((y - (Ay + 0.030)) / 0.050) ** 2) - 1.0
        mid = np.sqrt(((x - Ax + 0.004) / 0.040) ** 2 + ((y - 0.5 * (Ay + By)) / 0.085) ** 2) - 1.0
        d2 = np.minimum(np.minimum(fore * 0.05, heel * 0.045), mid * 0.04)
        top = np.where(y > Ay - 0.01, 0.026, 0.016)  # thicker heel block
        dz = np.maximum(-z, z - top)
        return np.maximum(d2, dz) - 0.0015

    m.add(Custom(sole_fn, (Ax - 0.07, By - 0.14, -0.005), (Ax + 0.07, Ay + 0.10, 0.03)), k=0.003, sym=True)
    # welt seam groove + folded cuff at the top of the shaft
    m.add(Torus(shaft_top + np.array([0, 0, -0.012], np.float32), 0.056, 0.007, tube_scale=(0.9, 1.6)), k=0.003, sym=True)

    # ankle wrinkles: horizontal creases across the shaft front
    def fn(P, f):
        Q = abs_x(P)
        z = Q[..., 2]
        band = smoothstep(0.10, 0.14, z) * (1.0 - smoothstep(0.26, 0.30, z))
        front = smoothstep(0.0, -0.03, Q[..., 1] - A[1])
        nz = fbm(P, 25.0, 2, seed=41)
        return f - band * front * 0.0012 * np.sin(z * 2 * math.pi / 0.022 + 3.0 * nz)

    m.edit(fn, (-0.34, -0.14, 0.05), (0.34, 0.12, 0.32))
    # flatten the soles on the ground plane
    m.inter(Custom(lambda P: -P[..., 2], (-0.40, -0.40, -0.04), (0.40, 0.25, 0.002)), k=0.0)


# ---------------------------------------------------------------------------
# Hair: sculpted short haircut (game-style "hair cap" + strand-clump relief)
# ---------------------------------------------------------------------------

# hairline height around the head by azimuth (0 = front, pi = back), radians,
# as offsets from eye height (a short masculine cut with the temples receding)
HAIRLINE = [(0.00, 0.086), (0.35, 0.080), (0.70, 0.058), (0.95, 0.028), (1.10, -0.014),
            (1.22, -0.022), (1.40, 0.018), (1.62, 0.016), (1.85, -0.029), (2.30, -0.059),
            (2.80, -0.072), (math.pi, -0.076)]


def head_c():
    return np.asarray(LM["head_c"], np.float32)


def hairline_z(phi):
    a = np.array([p[0] for p in HAIRLINE])
    zz = np.array([p[1] for p in HAIRLINE]) + float(LM["eye_l"][2])
    return np.interp(np.abs(phi), a, zz)


def ear_mask(P):
    e = np.asarray(LM["ear_l"], np.float32)
    ex = np.abs(P[..., 0])
    d = np.sqrt((ex - e[0] + 0.004) ** 2 + (P[..., 1] - e[1]) ** 2 + ((P[..., 2] - e[2]) / 1.6) ** 2) - 0.018
    return smoothstep(0.012, 0.004, d)


def hair(m, J, top=0.0160, side=0.0065):
    ez = float(LM["eye_l"][2])

    def fn(P, f):
        d = P - head_c()
        phi = np.arctan2(d[..., 0], -d[..., 1])
        zl = hairline_z(phi)
        z = P[..., 2]
        # feathered edge: thickness ramps up over ~2 cm above the hairline
        ramp = smoothstep(zl - 0.002, zl + 0.024, z) ** 2.0
        ear = ear_mask(P)
        vert = smoothstep(ez - 0.004, ez + 0.106, z)
        thick = side + (top - side) * vert
        # a little extra lift at the front (short, textured, pushed back)
        thick = thick + 0.004 * np.exp(-(phi / 0.6) ** 2) * smoothstep(ez + 0.056, ez + 0.096, z)
        # strand clumps: stretched along the combing direction — front-to-back
        # on top, top-to-bottom on the sides
        top_c = fbm(P * np.array([1.0, 0.22, 1.0], np.float32), 160.0, 3, seed=51)
        side_c = fbm(P * np.array([1.0, 1.0, 0.25], np.float32), 160.0, 3, seed=53)
        clumps = 0.0018 * (vert * top_c + (1 - vert) * side_c) + 0.0012 * fbm(P, 45.0, 2, seed=52)
        mask = ramp * (1.0 - ear)
        return f - mask * (thick + clumps)

    c = head_c()
    m.edit(fn, (-0.11, c[1] - 0.15, ez - 0.10), (0.11, c[1] + 0.15, LM["head_top"] + 0.04))


def dress(m, J, hair_on=True):
    shirt(m, J)
    trousers(m, J)
    jerkin(m, J)
    belt(m, J)
    gear(m, J)
    boots(m, J)
    if hair_on:
        hair(m, J)


# ---------------------------------------------------------------------------
# Material regions (the "ID map"): which garment/material owns a surface point
# ---------------------------------------------------------------------------

SKIN, HAIR, SHIRT, JERKIN, TROUSERS, BELT, METAL, POUCH, BOOTS, SOLE = range(10)
REGION_NAMES = ["skin", "hair", "shirt", "jerkin", "trousers", "belt", "metal", "pouch", "boots", "sole"]


def hair_mask(P):
    d = P - head_c()
    phi = np.arctan2(d[..., 0], -d[..., 1])
    zl = hairline_z(phi)
    z = P[..., 2]
    near = np.linalg.norm(d, axis=-1) < 0.16
    return smoothstep(zl - 0.002, zl + 0.024, z) ** 2.0 * (1.0 - ear_mask(P)) * near


def region_id(P, J):
    """Integer material region per surface point, evaluated with the same
    masks that sculpted the garments — so the ID map lines up with the hems
    in the bake exactly, like an ID map baked from high-poly vertex colors."""
    P = np.asarray(P, dtype=np.float32)
    z = P[..., 2]
    ax = np.abs(P[..., 0])
    rid = np.full(P.shape[:-1], SKIN, dtype=np.int32)
    rid[shirt_mask(P, J) > 0.5] = SHIRT
    rid[(trousers_mask(P) > 0.5) & (z < 1.02)] = TROUSERS
    rid[jerkin_mask(P, J) > 0.5] = JERKIN
    rid[belt_mask(P) > 0.5] = BELT
    if "pouch" in GEAR:
        rid[GEAR["pouch"].eval(P) < 0.003] = POUCH
    if "buckle" in GEAR:
        rid[GEAR["buckle"].eval(P) < 0.0015] = METAL
    boots = (z < 0.352) & (ax < 0.34)
    rid[boots] = BOOTS
    rid[boots & (z < 0.021)] = SOLE
    # low threshold: the whole feathered slope belongs to the hair material
    # (it blends to a dark scalp tone), otherwise skin color climbs the slope
    rid[(z > LM["chin_z"] - 0.01) & (hair_mask(P) > 0.05)] = HAIR
    return rid
