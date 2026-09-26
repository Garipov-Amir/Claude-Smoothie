"""
humanoid — anatomy "sculpt" of a realistic adult male in game A-pose,
expressed as smooth-blended SDF primitives (see sdf.py).

The same joint table drives the sculpt, the retopology cage and the
deform skeleton, so the three always agree: a bone's head sits exactly in
the joint the muscles were built around, and edge loops land on the
joints they have to deform.

Proportions follow the common 7.5-heads canon for a 1.80 m adult; the
A-pose (arms ~45 deg down, slight elbow/knee bend) is the modern default for
game characters because it keeps the shoulder mesh near the middle of its
deformation range.

Axes: Z up, character faces -Y, character's left is +X. Meters.
"""

import math
import os

import numpy as np

from sdf import (SDFModel, Ellipsoid, RoundCone, Capsule, RoundBox, Torus, Custom,
                 frame_from_axis, euler_matrix, normalize, smin, smax, fbm, value_noise)


# ---------------------------------------------------------------------------
# Skeleton (left side; right side mirrors X)
# ---------------------------------------------------------------------------

def _v(*a):
    return np.array(a, dtype=np.float64)


def skeleton(arm_angle=45.0):
    """Joint positions. Returns dict name -> np.array(3). Left-side names end in _l."""
    J = {}
    J["root"] = _v(0, 0, 0)
    J["pelvis"] = _v(0, 0.005, 0.965)
    J["spine_01"] = _v(0, 0.012, 1.045)
    J["spine_02"] = _v(0, 0.018, 1.165)
    J["spine_03"] = _v(0, 0.012, 1.300)
    J["neck_01"] = _v(0, 0.020, 1.485)
    J["head"] = _v(0, 0.010, 1.605)
    J["head_top"] = _v(0, 0.010, 1.800)

    J["clavicle_l"] = _v(0.022, -0.035, 1.455)
    J["upperarm_l"] = _v(0.188, 0.012, 1.446)
    a = math.radians(arm_angle)
    d_up = normalize(_v(math.cos(a), 0.015, -math.sin(a)))
    J["lowerarm_l"] = J["upperarm_l"] + d_up * 0.310
    d_lo = normalize(_v(math.cos(a) * 0.98, -0.12, -math.sin(a)))  # slight elbow bend forward
    J["hand_l"] = J["lowerarm_l"] + d_lo * 0.262

    J["thigh_l"] = _v(0.092, 0.000, 0.925)
    J["calf_l"] = _v(0.098, -0.012, 0.505)
    J["foot_l"] = _v(0.100, 0.020, 0.088)
    J["ball_l"] = _v(0.108, -0.125, 0.022)
    J["toe_l"] = _v(0.110, -0.205, 0.018)

    # hand frame: long axis, thumb side, palm normal
    hand_axis = normalize(d_lo + _v(0, 0, -0.08))
    thumb = normalize(_v(0, -1, 0) - hand_axis * np.dot(_v(0, -1, 0), hand_axis))
    palm_n = np.cross(thumb, hand_axis)  # points out of the palm
    if palm_n[0] > 0:  # palm faces the thigh (toward -X for the left hand)
        palm_n = -palm_n
    J["_hand_axis_l"] = hand_axis
    J["_thumb_dir_l"] = thumb
    J["_palm_n_l"] = palm_n
    return J


def mirror_name(n):
    return n[:-2] + "_r" if n.endswith("_l") else n


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Body sculpt
# ---------------------------------------------------------------------------

def build_body(m: SDFModel, J, detail=True):
    """Anatomy primitives. Order matters a little: big masses first, then
    muscles blended on top, then carves (spine groove, eye sockets)."""

    # ---------------- torso ----------------
    # ribcage: egg-shaped, tilted so the back is more upright than the chest
    m.add(Ellipsoid((0, 0.014, 1.300), (0.138, 0.105, 0.175), euler_matrix(-6, 0, 0)), k=0.0)
    # pectoral masses (fan from sternum to the armpit)
    # male pectorals: broad, flat, squarish plates (not rounded mounds)
    m.add(Ellipsoid((0.066, -0.070, 1.352), (0.084, 0.022, 0.052), euler_matrix(0, 8, -14)), k=0.06, sym=True)
    # chest front plane: fills the crease under the pecs so the chest reads as
    # one flat masculine plane (and the leather jerkin hangs from it)
    m.add(Ellipsoid((0, -0.010, 1.275), (0.132, 0.090, 0.150)), k=0.06)
    # lats: the V-taper from armpit to lower back
    m.add(Ellipsoid((0.108, 0.040, 1.285), (0.050, 0.088, 0.140), euler_matrix(0, -8, 6)), k=0.05, sym=True)
    # outer pec / serratus corner: squares off the chest section (a male
    # chest is closer to a rounded box than an ellipse in cross-section)
    m.add(Ellipsoid((0.105, -0.045, 1.300), (0.035, 0.040, 0.070)), k=0.045, sym=True)
    # serratus / side ribs
    # abdomen + rectus abdominis
    m.add(Ellipsoid((0, 0.004, 1.120), (0.122, 0.094, 0.135)), k=0.07)
    m.add(Ellipsoid((0, -0.058, 1.135), (0.068, 0.034, 0.125)), k=0.05)
    # lower-rib flank fill (male torso: straighter line from lats to hips)
    m.add(Ellipsoid((0.100, 0.004, 1.190), (0.036, 0.070, 0.075)), k=0.05, sym=True)
    # obliques over the iliac crest
    m.add(Ellipsoid((0.104, 0.004, 1.068), (0.040, 0.076, 0.066)), k=0.05, sym=True)
    # pelvis block
    m.add(Ellipsoid((0, 0.010, 0.965), (0.122, 0.094, 0.098)), k=0.07)
    # lower belly
    m.add(Ellipsoid((0, -0.050, 0.995), (0.085, 0.045, 0.070)), k=0.05)
    # glutes
    m.add(Ellipsoid((0.064, 0.064, 0.905), (0.074, 0.076, 0.100), euler_matrix(0, 10, 0)), k=0.045, sym=True)
    # erector spinae columns + spine groove
    m.add(Ellipsoid((0.030, 0.078, 1.150), (0.030, 0.030, 0.150)), k=0.04, sym=True)
    # trapezius: slope from neck to shoulder, plus the upper-back diamond
    m.add(Ellipsoid((0.080, 0.035, 1.463), (0.095, 0.042, 0.040), euler_matrix(0, 22, 0)), k=0.04, sym=True)
    m.add(Ellipsoid((0, 0.080, 1.380), (0.070, 0.030, 0.100)), k=0.05)
    # scapulae
    m.add(Ellipsoid((0.075, 0.095, 1.340), (0.045, 0.018, 0.065), euler_matrix(0, 0, -15)), k=0.03, sym=True)
    # clavicles
    m.add(Capsule((0.022, -0.052, 1.458), (0.150, -0.022, 1.474), 0.008), k=0.03, sym=True)

    # ---------------- neck ----------------
    m.add(RoundCone((0, 0.024, 1.455), (0, 0.016, 1.605), 0.064, 0.056), k=0.03)
    # sternocleidomastoid: behind the ear to the sternal notch
    m.add(Capsule((0.040, 0.012, 1.600), (0.014, -0.046, 1.480), 0.011), k=0.022, sym=True)
    m.add(Ellipsoid((0, -0.034, 1.548), (0.009, 0.008, 0.013)), k=0.014)  # larynx

    build_head(m, J, detail)

    # ---------------- arms ----------------
    for side in (1,):
        S, E, W = J["upperarm_l"], J["lowerarm_l"], J["hand_l"]
        au = normalize(E - S)
        al = normalize(W - E)
        Ru = frame_from_axis(au, up=(0, -1, 0))
        Rl = frame_from_axis(al, up=(0, -1, 0))
        fwd = _v(0, -1, 0)
        # deltoid caps the shoulder joint
        m.add(Ellipsoid(S + au * 0.030 + _v(0.008, 0, 0.004), (0.050, 0.055, 0.080), Ru), k=0.035, sym=True)
        m.add(RoundCone(S + au * 0.02, E, 0.043, 0.034), k=0.03, sym=True)
        # biceps (front) / triceps (back)
        m.add(Ellipsoid(lerp(S, E, 0.56) + fwd * 0.017, (0.034, 0.035, 0.090), Ru), k=0.025, sym=True)
        m.add(Ellipsoid(lerp(S, E, 0.42) - fwd * 0.019, (0.038, 0.036, 0.100), Ru), k=0.025, sym=True)
        # elbow + olecranon
        m.add(Ellipsoid(E - fwd * 0.012, (0.030, 0.030, 0.032), Rl), k=0.02, sym=True)
        # forearm: flexor/extensor mass near the elbow tapering to a flat wrist
        m.add(RoundCone(E + al * 0.01, W - al * 0.01, 0.036, 0.024), k=0.03, sym=True)
        m.add(Ellipsoid(lerp(E, W, 0.28) + fwd * 0.004, (0.040, 0.036, 0.085), Rl), k=0.03, sym=True)
        m.add(Ellipsoid(W, (0.027, 0.018, 0.022), frame_from_axis(al, up=J["_palm_n_l"])), k=0.015, sym=True)

        build_hand(m, J)

    # ---------------- legs ----------------
    H, K, A = J["thigh_l"], J["calf_l"], J["foot_l"]
    at = normalize(K - H)
    ac = normalize(A - K)
    Rt = frame_from_axis(at, up=(0, -1, 0))
    Rc = frame_from_axis(ac, up=(0, -1, 0))
    inward = _v(-1, 0, 0)
    fwd = _v(0, -1, 0)
    m.add(RoundCone(H + _v(-0.006, 0, 0.02), K, 0.080, 0.049), k=0.05, sym=True)
    # quadriceps (rectus femoris + vastus lateralis), teardrop vastus medialis
    m.add(Ellipsoid(lerp(H, K, 0.50) + fwd * 0.032, (0.054, 0.048, 0.175), Rt), k=0.04, sym=True)
    m.add(Ellipsoid(lerp(H, K, 0.48) - inward * 0.034 + fwd * 0.006, (0.042, 0.050, 0.170), Rt), k=0.04, sym=True)
    m.add(Ellipsoid(K + _v(-0.024, -0.030, 0.085), (0.033, 0.032, 0.060), Rt), k=0.03, sym=True)
    # hamstrings / adductors
    m.add(Ellipsoid(lerp(H, K, 0.45) - fwd * 0.034, (0.054, 0.045, 0.170), Rt), k=0.04, sym=True)
    m.add(Ellipsoid(lerp(H, K, 0.28) + inward * 0.026, (0.045, 0.060, 0.120), Rt), k=0.035, sym=True)
    # upper thigh mass (sartorius / rectus origin) fills the section below the fold
    m.add(Ellipsoid(lerp(H, K, 0.24) + fwd * 0.032, (0.056, 0.046, 0.115), Rt), k=0.04, sym=True)
    # upper hamstrings under the gluteal fold
    m.add(Ellipsoid(lerp(H, K, 0.30) - fwd * 0.036 - inward * 0.008, (0.050, 0.044, 0.110), Rt), k=0.04, sym=True)
    # knee
    m.add(Ellipsoid(K + _v(0, -0.038, 0.012), (0.024, 0.016, 0.028)), k=0.02, sym=True)
    m.add(Ellipsoid(K, (0.046, 0.044, 0.050)), k=0.03, sym=True)
    # lower leg: calf heads + tibia
    m.add(RoundCone(K, A + _v(0, 0, 0.02), 0.046, 0.034), k=0.04, sym=True)
    m.add(Ellipsoid(lerp(K, A, 0.27) - fwd * 0.034 - inward * 0.013, (0.037, 0.035, 0.095), Rc), k=0.03, sym=True)
    m.add(Ellipsoid(lerp(K, A, 0.25) - fwd * 0.032 + inward * 0.015, (0.039, 0.037, 0.090), Rc), k=0.03, sym=True)
    # Achilles tendon: the back of the ankle
    m.add(RoundCone(lerp(K, A, 0.62) - fwd * 0.030, A - fwd * 0.030 + _v(0, 0, -0.03), 0.011, 0.010), k=0.02, sym=True)
    m.add(Ellipsoid(lerp(K, A, 0.35) + fwd * 0.012, (0.024, 0.030, 0.140), Rc), k=0.03, sym=True)
    # thigh gap: keep the inner thighs apart below the crotch (they would
    # fuse once trousers add their ease, and retopo needs a clean crotch)
    m.sub(Capsule((0, 0.0, 0.25), (0, 0.0, 0.805), 0.010), k=0.012)
    # ankle bones (inner malleolus sits higher)
    m.add(Ellipsoid(A + _v(-0.026, 0.004, 0.006), (0.012, 0.014, 0.015)), k=0.012, sym=True)
    m.add(Ellipsoid(A + _v(0.027, 0.010, -0.004), (0.012, 0.014, 0.015)), k=0.012, sym=True)
    build_foot(m, J)


def eyelids(center, r_eye=0.0120, thick=0.0022, half_w=0.0156, up_h=0.0060, lo_h=0.0045,
            tilt=6.0, side=1.0):
    """Lids as a solid sphere slightly larger than the eyeball with an
    almond-shaped opening cut through it along the view axis. Union with the
    eyeball and the opening shows the eye, the rim of the cut is the lid
    margin (with real thickness, which is what catches light in a render)."""
    c = np.asarray(center, dtype=np.float32)
    ct, st = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))

    def fn(P):
        q = P - c
        d_sph = np.linalg.norm(q, axis=-1) - (r_eye + thick)
        x = q[..., 0] * side
        z = q[..., 2]
        xr = x * ct + z * st          # outer corner slightly higher
        zr = -x * st + z * ct
        hh = np.where(zr > 0, up_h, lo_h)
        # the upper lid arc peaks slightly toward the inner third
        hh = hh * (1.0 - 0.10 * np.clip(xr / half_w, -1, 1))
        e = np.sqrt((xr / half_w) ** 2 + (zr / hh) ** 2) - 1.0
        d_open = e * np.minimum(hh, half_w)
        front = q[..., 1] < 0.004
        d_open = np.where(front, d_open, 1.0)
        return smax(d_sph, -d_open, 0.0012)

    lo = c - (r_eye + thick + 0.002)
    hi = c + (r_eye + thick + 0.002)
    return Custom(fn, lo, hi)


# Head base as a loft of horizontal cross-sections driven by profile curves:
# front view half-width, side view front and back silhouettes, and how
# "boxy" the front/back of each section is (superellipse exponent). This is
# the planes-of-the-head approach — silhouettes first, features second —
# and it is far more controllable than stacking ellipsoids.
#            z      yf       yb      xw     n_front n_back
HEAD_PROFILE = [
    (1.5660, -0.0860, -0.0700, 0.0080, 2.2, 2.0),
    (1.5720, -0.0940, -0.0450, 0.0200, 2.2, 2.0),
    (1.5800, -0.0970, -0.0150, 0.0330, 2.2, 2.0),
    (1.5900, -0.0935, 0.0080, 0.0465, 2.2, 2.0),
    (1.6020, -0.0935, 0.0280, 0.0545, 2.2, 2.0),
    (1.6150, -0.0945, 0.0580, 0.0585, 2.2, 2.1),
    (1.6300, -0.0950, 0.0790, 0.0635, 2.4, 2.1),
    (1.6500, -0.0935, 0.0940, 0.0685, 2.7, 2.1),
    (1.6700, -0.0915, 0.1020, 0.0715, 2.8, 2.1),
    (1.6880, -0.0900, 0.1060, 0.0730, 2.6, 2.1),
    (1.7050, -0.0912, 0.1080, 0.0740, 2.4, 2.1),
    (1.7250, -0.0870, 0.1070, 0.0750, 2.3, 2.1),
    (1.7450, -0.0765, 0.1000, 0.0725, 2.2, 2.1),
    (1.7650, -0.0605, 0.0870, 0.0650, 2.1, 2.1),
    (1.7800, -0.0440, 0.0690, 0.0530, 2.0, 2.0),
    (1.7900, -0.0280, 0.0480, 0.0380, 2.0, 2.0),
    (1.7960, -0.0150, 0.0300, 0.0240, 2.0, 2.0),
    (1.7990, -0.0070, 0.0160, 0.0100, 2.0, 2.0),
]


def head_base(profile=HEAD_PROFILE):
    """Lofted head blank as a proper signed distance.

    The loft's implicit function F (superellipse sections) has the right
    zero set but a poor metric near the end caps (crown, under the chin):
    it collapses toward zero wherever sections shrink quickly, so any offset
    built on top (hair, a helmet, smooth blends) balloons there — seen as a
    flat "plate" and a cylinder sticking out of the hair. Fix: the magnitude
    is the true distance to densely sampled section rings + cap discs
    (KD-tree), the sign comes from F, and very close to the surface the
    gradient-normalized F takes over (it is accurate there and has none of
    the KD sampling ripple).
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial import cKDTree
    tab = np.array(profile, dtype=np.float64)
    zs = tab[:, 0]
    curves = [PchipInterpolator(zs, tab[:, i]) for i in range(1, 6)]
    z0, z1 = zs[0], zs[-1]

    def sections(z):
        yf, yb, xw, nf, nb = (c(z) for c in curves)
        return yf, yb, xw, nf, nb

    def F(P):
        z = np.clip(P[..., 2], z0, z1)
        yf, yb, xw, nf, nb = sections(z)
        yc = 0.5 * (yf + yb)
        ry = np.where(P[..., 1] < yc, yc - yf, yb - yc)
        n = np.where(P[..., 1] < yc, nf, nb)
        u = np.abs(P[..., 0]) / np.maximum(xw, 1e-4)
        v = np.abs(P[..., 1] - yc) / np.maximum(ry, 1e-4)
        r = (u ** n + v ** n) ** (1.0 / n)
        d = (r - 1.0) * np.minimum(xw, ry)
        return np.maximum(d, np.maximum(z0 - P[..., 2], P[..., 2] - z1))

    # dense surface samples: rings along z + filled end caps
    pts = []
    for z in np.linspace(z0, z1, 420):
        yf, yb, xw, nf, nb = (float(v) for v in sections(z))
        yc = 0.5 * (yf + yb)
        th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        c, s_ = np.cos(th), np.sin(th)
        front = s_ < 0
        n = np.where(front, nf, nb)
        ry = np.where(front, yc - yf, yb - yc)
        x = xw * np.sign(c) * np.abs(c) ** (2.0 / n)
        y = yc + ry * np.sign(s_) * np.abs(s_) ** (2.0 / n)
        pts.append(np.stack([x, y, np.full_like(x, z)], axis=1))
    for z in (z0, z1):
        yf, yb, xw, nf, nb = (float(v) for v in sections(z))
        g = np.linspace(-1, 1, 25)
        gx, gy = np.meshgrid(g, g)
        keep = gx ** 2 + gy ** 2 <= 1
        yc = 0.5 * (yf + yb)
        pts.append(np.stack([gx[keep] * xw, yc + gy[keep] * (yb - yf) / 2, np.full(keep.sum(), z)], axis=1))
    tree = cKDTree(np.concatenate(pts))

    def fn(P):
        shp = P.shape[:-1]
        Pf = P.reshape(-1, 3)
        f0 = F(Pf)
        h = 0.0008
        g = np.zeros(len(Pf))
        for ax in range(3):
            dP = np.zeros(3, dtype=np.float32)
            dP[ax] = h
            g += ((F(Pf + dP) - f0) / h) ** 2
        fnorm = f0 / np.maximum(np.sqrt(g), 0.2)
        out = fnorm.copy()
        # fnorm under-estimates distance, so |fnorm| >= 0.05 already means
        # "far" — only the band in between needs the exact KD distance
        band = (np.abs(fnorm) > 0.002) & (np.abs(fnorm) < 0.05)
        if band.any():
            dist, _ = tree.query(Pf[band], distance_upper_bound=0.08, workers=-1)
            dk = np.sign(f0[band]) * np.minimum(dist, 0.08)
            w = np.clip((np.abs(fnorm[band]) - 0.002) / 0.004, 0.0, 1.0)
            out[band] = fnorm[band] * (1 - w) + dk * w
        return out.reshape(shp).astype(np.float32)

    lo = (-0.08, -0.105, z0 - 0.002)
    hi = (0.08, 0.118, z1 + 0.002)
    return Custom(fn, lo, hi)


def build_head(m: SDFModel, J, detail=True):
    m.add(head_base(), k=0.0)
    # temples: slight hollow behind the orbit rim
    m.sub(Ellipsoid((0.078, -0.040, 1.702), (0.010, 0.024, 0.020)), k=0.015, sym=True)
    # mandible angle (gonion) + ramus give the jaw line its corner
    m.add(RoundCone((0.049, -0.010, 1.603), (0.020, -0.082, 1.577), 0.0060, 0.0075), k=0.016, sym=True)
    # chin: mental protuberance
    m.add(Ellipsoid((0, -0.0925, 1.579), (0.016, 0.009, 0.012)), k=0.010)
    # cheekbones
    m.add(Ellipsoid((0.050, -0.061, 1.667), (0.020, 0.018, 0.012), euler_matrix(0, 0, -25)), k=0.014, sym=True)
    # cheek fat pad (its front edge reads as the nasolabial fold)
    m.add(Ellipsoid((0.036, -0.078, 1.640), (0.018, 0.012, 0.020)), k=0.022, sym=True)
    # muzzle (dental arch) under the lips
    m.add(Ellipsoid((0, -0.089, 1.613), (0.024, 0.014, 0.020)), k=0.012)
    # brow ridge
    m.add(RoundCone((0.044, -0.083, 1.708), (0.012, -0.091, 1.710), 0.0045, 0.0050), k=0.014, sym=True)
    # eye sockets, eyeballs, lids
    eye = (0.0315, -0.0800, 1.6840)
    m.sub(Ellipsoid((0.031, -0.0960, 1.6840), (0.0185, 0.0130, 0.0125)), k=0.016, sym=True)
    m.add(Ellipsoid(eye, (0.0120, 0.0120, 0.0120)), k=0.0, sym=True)
    m.add(eyelids(eye), k=0.0030, sym=True)
    # upper-lid crease: a soft fold tucked under the brow
    m.add(Ellipsoid((0.0325, -0.0880, 1.6935), (0.0145, 0.0045, 0.0030), euler_matrix(-25, 0, -5)), k=0.004, sym=True)
    # nose: dorsum, tip, alae, columella, nostrils
    # straight male dorsum from the nasion (no notch), defined tip, tight alae
    m.add(RoundCone((0, -0.0930, 1.6880), (0, -0.1075, 1.6515), 0.0056, 0.0068), k=0.012)
    m.add(Ellipsoid((0, -0.0995, 1.6670), (0.0090, 0.0068, 0.0180)), k=0.010)
    m.add(Ellipsoid((0, -0.1090, 1.6455), (0.0080, 0.0068, 0.0066)), k=0.008)
    m.add(Ellipsoid((0.0100, -0.1010, 1.6375), (0.0064, 0.0070, 0.0052), euler_matrix(0, 0, 25)), k=0.007, sym=True)
    m.add(Ellipsoid((0, -0.1045, 1.6365), (0.0027, 0.0052, 0.0029)), k=0.004)
    m.sub(Ellipsoid((0.0058, -0.1015, 1.6335), (0.0019, 0.0029, 0.0012), euler_matrix(0, 0, 25)), k=0.0015, sym=True)
    # lips: upper (two halves = cupid's bow), lower, philtrum, parting line, corners
    m.add(Ellipsoid((0.0058, -0.1010, 1.6142), (0.0135, 0.0068, 0.0050), euler_matrix(-14, 0, -8)), k=0.004, sym=True)
    m.add(Ellipsoid((0, -0.0985, 1.6032), (0.0170, 0.0078, 0.0062), euler_matrix(14, 0, 0)), k=0.005)
    m.sub(Capsule((0.0024, -0.1058, 1.6290), (0.0024, -0.1055, 1.6180), 0.0011), k=0.002, sym=True)
    m.sub(RoundBox((0, -0.1080, 1.6090), (0.0170, 0.0100, 0.0003), 0.0003), k=0.0012)
    m.sub(Ellipsoid((0.0200, -0.0950, 1.6085), (0.0022, 0.0030, 0.0024)), k=0.003, sym=True)
    m.sub(Ellipsoid((0, -0.0985, 1.5935), (0.0115, 0.0035, 0.0030)), k=0.004)
    # ears: helix body, concha, lobe, tragus
    Re = euler_matrix(12, 0, 16)
    m.add(Ellipsoid((0.0755, 0.0120, 1.6700), (0.0095, 0.0195, 0.0300), Re), k=0.006, sym=True)
    m.sub(Ellipsoid((0.0825, 0.0080, 1.6670), (0.0050, 0.0110, 0.0160), Re), k=0.003, sym=True)
    m.add(Ellipsoid((0.0785, 0.0080, 1.6500), (0.0060, 0.0080, 0.0090), Re), k=0.004, sym=True)
    m.add(Ellipsoid((0.0725, -0.0035, 1.6640), (0.0040, 0.0040, 0.0050)), k=0.003, sym=True)


FINGERS = [  # (name, offset along the thumb axis, phalanx lengths, base radius)
    ("index", 0.0255, (0.041, 0.025, 0.020), 0.0090),
    ("middle", 0.0085, (0.045, 0.028, 0.021), 0.0092),
    ("ring", -0.0085, (0.043, 0.026, 0.020), 0.0086),
    ("pinky", -0.0250, (0.034, 0.020, 0.018), 0.0076),
]
THUMB = ((0.040, 0.030, 0.026), 0.0125)


def hand_chains(J):
    """Joint chains of the left hand: name -> (points[4], radii[4]).
    points[0] is the knuckle (MCP) / thumb CMC, points[3] the tip."""
    if "_chains_l" in J:
        return J["_chains_l"]
    W = J["hand_l"]
    ax, th, pn = J["_hand_axis_l"], J["_thumb_dir_l"], J["_palm_n_l"]
    chains = {}
    for name, off, lens, r0 in FINGERS:
        p = W + ax * 0.094 + th * off
        d = normalize(ax + th * off * 1.6)  # slight fan
        pts, rad, r = [p], [r0], r0
        for i, L in enumerate(lens):
            d = normalize(d + pn * (0.16 + 0.05 * i))  # relaxed curl toward the palm
            p = p + d * L
            r *= 0.88
            pts.append(p)
            rad.append(r)
        chains[name] = (np.array(pts), np.array(rad))
    lens, r = THUMB
    p = W + ax * 0.022 + th * 0.022 + pn * 0.006
    d = normalize(ax * 0.55 + th * 0.70 + pn * 0.35)
    pts, rad = [p], [r]
    for L in lens:
        p = p + d * L
        d = normalize(d + ax * 0.25 + pn * 0.12)
        r *= 0.88
        pts.append(p)
        rad.append(r)
    chains["thumb"] = (np.array(pts), np.array(rad))
    return chains


def build_hand(m: SDFModel, J):
    W = J["hand_l"]
    ax, th, pn = J["_hand_axis_l"], J["_thumb_dir_l"], J["_palm_n_l"]
    R = np.stack([th, pn, ax], axis=1).astype(np.float32)  # local x=thumb, y=palm normal, z=along hand
    # palm + heel pads
    m.add(RoundBox(W + ax * 0.052, (0.033, 0.008, 0.040), 0.010, R), k=0.012, sym=True)
    m.add(Ellipsoid(W + ax * 0.030 + th * 0.022 + pn * 0.008, (0.018, 0.012, 0.026), R), k=0.01, sym=True)  # thenar
    m.add(Ellipsoid(W + ax * 0.035 - th * 0.024 + pn * 0.006, (0.014, 0.010, 0.030), R), k=0.01, sym=True)  # hypothenar
    for name, (pts, rad) in hand_chains(J).items():
        for i in range(3):
            k = (0.008 if i == 0 else 0.003)
            m.add(RoundCone(pts[i], pts[i + 1], rad[i], rad[i + 1] / 0.88 * 0.9 if name != "thumb" else rad[i] * 0.88), k=k, sym=True)


def build_foot(m: SDFModel, J):
    A, B, T = J["foot_l"], J["ball_l"], J["toe_l"]
    m.add(Ellipsoid(A + _v(0, 0.030, -0.045), (0.030, 0.040, 0.038)), k=0.02, sym=True)  # heel
    m.add(RoundBox(lerp(A, B, 0.55) + _v(0.002, 0, -0.018), (0.034, 0.075, 0.020), 0.014,
                   euler_matrix(-12, 0, -4)), k=0.03, sym=True)
    m.add(Ellipsoid(B + _v(0.004, 0, 0.004), (0.046, 0.030, 0.021)), k=0.02, sym=True)
    # toes, big toe first
    for off, L, r in ((-0.030, 0.055, 0.0125), (-0.009, 0.050, 0.0085), (0.008, 0.045, 0.008),
                      (0.022, 0.040, 0.0075), (0.034, 0.033, 0.007)):
        p = B + _v(off, -0.010, 0.002)
        q = p + normalize(_v(off * 0.3, -1, -0.12)) * L
        m.add(RoundCone(p, q, r, r * 0.85), k=0.006, sym=True)
    # arch: carve the inner sole
    m.sub(Ellipsoid(lerp(A, B, 0.5) + _v(-0.040, 0.0, -0.040), (0.020, 0.050, 0.020)), k=0.015, sym=True)


# ---------------------------------------------------------------------------
# Build + regions
# ---------------------------------------------------------------------------

BOUNDS = ((-0.82, -0.46, -0.03), (0.82, 0.22, 1.84))

REF_CACHE = os.environ.get("GC_REF_CACHE", os.path.expanduser("~/.cache/game_character_ref"))


def build(arm_angle=45.0, clothing=True, hair=True, body="reference"):
    """body="reference": anatomy from the CC0 MakeHuman reference body (see
    reference_body.py) + this pipeline's eyes, clothing, gear and hair.
    body="procedural": the original primitive-sculpted body."""
    from landmarks import LM
    import landmarks
    LM.clear()
    m = SDFModel()
    if body == "reference":
        import reference_body as RB
        ref = RB.build(REF_CACHE, full=True)
        pts, nrm = RB.surface_samples(ref, REF_CACHE)
        J = landmarks.skeleton_from_reference(ref)
        m.add(RB.MeshSDF(pts, nrm))
        LM.update(landmarks.from_reference(ref, J))
        # eyeballs sit in the reference's eye pockets; record where the lids
        # cover them (the lid margin) before they join the surface
        r = LM["eye_r_radius"]
        body_only = SDFModel()
        body_only.ops = list(m.ops)
        LM["eye_margin_l"] = landmarks.eye_margin(body_only, LM["eye_l"], r)
        LM["eye_margin_r"] = landmarks.eye_margin(body_only, landmarks.mirror_x(LM["eye_l"]), r)
        for c in (LM["eye_l"], landmarks.mirror_x(LM["eye_l"])):
            m.add(Ellipsoid(tuple(c), (r, r, r)), k=0.0)
        # perineum height: first surface point going up the midline
        from sdf import sample
        zz = np.arange(0.5, 1.0, 0.001, dtype=np.float32)
        col = np.stack([np.zeros_like(zz), np.full_like(zz, float(J["pelvis"][1])), zz], 1)
        inside = np.nonzero(sample(body_only, col) < 0)[0]
        LM["crotch_z"] = float(zz[inside[0]]) if len(inside) else 0.84
    else:
        J = skeleton(arm_angle)
        build_body(m, J)
        LM.update(landmarks.procedural_defaults(J))
    LM["J"] = J
    LM["body"] = body
    if clothing:
        import costume
        costume.dress(m, J, hair_on=hair)
    return m, J
