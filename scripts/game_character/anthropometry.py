"""
anthropometry — measure the (nude) sculpt like a tailor / anthropometrist
and compare against reference body measurements.

Anatomy is judged objectively here instead of by eye: the SDF is sliced at
standard landmark levels, each cross-section's *convex hull perimeter* is the
circumference a tape measure would read, extents give breadths/depths, and
segment lengths come from the joints.

Targets: an athletic adult male of 1.80 m, derived from ANSUR II (US Army
anthropometric survey, 2012, male means, public domain) scaled to stature,
with waist/hip/belly values moved toward a fit build (ANSUR means include
many overweight subjects). Values are approximate reference numbers, meant
for +-5 % tolerance checks, not for a specific individual.

    python anthropometry.py            # prints the table for humanoid.build(clothing=False)
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

# name: (target value in meters, how it is measured)
TARGETS = {
    # heights (floor to landmark)
    "stature":              (1.800, "vertex height"),
    "eye_height":           (1.683, "pupil height"),
    "chin_height":          (1.568, "menton height (stature - head height)"),
    "acromion_height":      (1.470, "top of shoulder"),
    "crotch_height":        (0.840, "perineum"),
    "knee_height":          (0.505, "mid-patella"),
    # head & face
    "head_height":          (0.232, "vertex to menton"),
    "head_breadth":         (0.155, "max head width above the ears"),
    "head_length":          (0.197, "glabella to back of head"),
    "bizygomatic":          (0.140, "cheekbone width"),
    "bigonial":             (0.115, "jaw-angle width"),
    # circumferences
    "neck_circ":            (0.390, "mid neck"),
    "chest_circ":           (1.010, "at nipple level"),
    "waist_circ":           (0.840, "at navel"),
    "hip_circ":             (0.980, "at buttock max"),
    "thigh_circ":           (0.580, "below the gluteal fold"),
    "knee_circ":            (0.385, "mid-patella"),
    "calf_circ":            (0.380, "calf max"),
    "ankle_circ":           (0.230, "above the malleoli"),
    "biceps_circ":          (0.310, "mid upper arm, relaxed"),
    "forearm_circ":         (0.280, "forearm max"),
    "wrist_circ":           (0.175, "at the styloids"),
    # breadths / depths
    "biacromial":           (0.405, "shoulder-point width"),
    "chest_breadth":        (0.330, "at nipple level"),
    "chest_depth":          (0.245, "at nipple level"),
    "waist_breadth":        (0.290, "at navel"),
    "waist_depth":          (0.210, "at navel"),
    "hip_breadth":          (0.350, "max hip width"),
    # segment lengths
    "upper_arm_length":     (0.340, "acromion to elbow"),
    "forearm_length":       (0.265, "elbow to wrist"),
    "hand_length":          (0.193, "wrist crease to middle fingertip"),
    "thigh_length":         (0.435, "hip joint to knee"),
    "shank_length":         (0.425, "knee to ankle"),
}


def _slice(model, center, u, v, half=(0.30, 0.20), res=0.0015):
    """Sample the model on a plane through `center` spanned by unit vectors u, v.
    Returns (field[nv, nu], coords_u, coords_v)."""
    from sdf import sample
    cu = np.arange(-half[0], half[0], res)
    cv = np.arange(-half[1], half[1], res)
    U, V = np.meshgrid(cu, cv)
    P = center + U[..., None] * u + V[..., None] * v
    f = sample(model, P.reshape(-1, 3).astype(np.float32)).reshape(U.shape)
    return f, cu, cv


def _contour_at(f, cu, cv, pt_uv):
    """Contour (in plane coords) of the region containing pt_uv."""
    from skimage.measure import find_contours
    from matplotlib.path import Path
    res = cu[1] - cu[0]
    best = None
    for c in find_contours(f, 0.0):
        xy = np.stack([cu[0] + c[:, 1] * res, cv[0] + c[:, 0] * res], axis=1)
        if len(xy) < 8:
            continue
        if Path(xy).contains_point(pt_uv):
            area = abs(np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])) / 2
            if best is None or area < best[1]:
                best = (xy, area)
    return None if best is None else best[0]


def _hull_perimeter(xy):
    from scipy.spatial import ConvexHull
    h = ConvexHull(xy)
    p = xy[h.vertices]
    return float(np.sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)))


def section(model, center, normal, inside_pt=None, half=(0.30, 0.20)):
    """Circumference / breadth / depth of the cross-section through `center`
    perpendicular to `normal`, for the region containing `inside_pt`."""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    ref = np.array([0, 1.0, 0]) if abs(n[1]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c = np.asarray(center, float)
    f, cu, cv = _slice(model, c, u, v, half)
    p = np.asarray(inside_pt if inside_pt is not None else c, float) - c
    xy = _contour_at(f, cu, cv, (float(p @ u), float(p @ v)))
    if xy is None:
        return None
    return {"circ": _hull_perimeter(xy), "extent_u": float(np.ptp(xy[:, 0])), "extent_v": float(np.ptp(xy[:, 1])),
            "contour": xy, "u": u, "v": v}


def horizontal(model, z, inside_xy, half=(0.35, 0.22)):
    """Horizontal section at height z (u = world X, v = world Y)."""
    from sdf import sample
    res = 0.0015
    cu = np.arange(-half[0], half[0], res)
    cv = np.arange(-half[1] - 0.02, half[1], res)
    U, V = np.meshgrid(cu, cv)
    P = np.stack([U, V, np.full_like(U, z)], axis=-1)
    f = sample(model, P.reshape(-1, 3).astype(np.float32)).reshape(U.shape)
    xy = _contour_at(f, cu, cv, inside_xy)
    if xy is None:
        return None
    return {"circ": _hull_perimeter(xy), "breadth": float(np.ptp(xy[:, 0])), "depth": float(np.ptp(xy[:, 1])),
            "contour": xy}


def column_extent(model, x, y, z_lo, z_hi, step=0.001):
    """Lowest/highest surface z along a vertical line (e.g. vertex, perineum)."""
    from sdf import sample
    zs = np.arange(z_lo, z_hi, step)
    P = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], axis=1).astype(np.float32)
    f = sample(model, P)
    inside = np.nonzero(f < 0)[0]
    if len(inside) == 0:
        return None, None
    return float(zs[inside[0]]), float(zs[inside[-1]])


def measure(model, J):
    import humanoid
    M = {}
    spine_y = lambda z: float(np.interp(z, [0.9, 1.05, 1.2, 1.35, 1.5], [0.01, 0.012, 0.018, 0.014, 0.022]))
    # heights
    _lo, top = column_extent(model, 0.0, 0.012, 1.6, 1.9)
    M["stature"] = top
    M["eye_height"] = float(humanoid.EYE_CENTER[2]) if hasattr(humanoid, "EYE_CENTER") else 1.684
    chin_lo, _ = column_extent(model, 0.0, -0.085, 1.45, 1.70)
    M["chin_height"] = chin_lo
    M["head_height"] = top - chin_lo if (top and chin_lo) else None
    crotch, _ = column_extent(model, 0.0, 0.005, 0.5, 1.0)
    M["crotch_height"] = crotch
    M["knee_height"] = float(J["calf_l"][2])
    # shoulder top above the acromion
    S = J["upperarm_l"]
    _l, acr = column_extent(model, float(S[0]) - 0.01, float(S[1]), 1.3, 1.6)
    M["acromion_height"] = acr
    # head
    hs = horizontal(model, 1.735, (0.0, 0.01), half=(0.14, 0.16))
    M["head_breadth"] = hs["breadth"] if hs else None
    hl = horizontal(model, 1.705, (0.0, 0.01), half=(0.14, 0.16))
    M["head_length"] = hl["depth"] if hl else None
    zy = horizontal(model, 1.667, (0.0, -0.02), half=(0.14, 0.16))
    if zy:  # front of the face only: the ears sit in the same slice
        c = zy["contour"]
        M["bizygomatic"] = float(np.ptp(c[c[:, 1] < -0.015, 0]))
    go = horizontal(model, 1.600, (0.0, -0.03), half=(0.14, 0.16))
    M["bigonial"] = go["breadth"] if go else None
    # torso sections
    nk = horizontal(model, 1.530, (0.0, 0.02), half=(0.14, 0.14))
    M["neck_circ"] = nk["circ"] if nk else None
    ch = horizontal(model, 1.300, (0.0, spine_y(1.3)))
    if ch:
        M["chest_circ"], M["chest_breadth"], M["chest_depth"] = ch["circ"], ch["breadth"], ch["depth"]
    wa = horizontal(model, 1.075, (0.0, spine_y(1.075)))
    if wa:
        M["waist_circ"], M["waist_breadth"], M["waist_depth"] = wa["circ"], wa["breadth"], wa["depth"]
    # hips: max circumference 0.88..1.0
    best = None
    for z in np.arange(0.88, 1.0, 0.01):
        h = horizontal(model, z, (0.0, 0.02))
        if h and (best is None or h["circ"] > best["circ"]):
            best = h
    if best:
        M["hip_circ"], M["hip_breadth"] = best["circ"], best["breadth"]
    # shoulders
    # acromion (bony shoulder point) sits ~1.5 cm lateral/up of the joint
    # center; a surface slice would cut the A-posed arm, so use the landmark
    M["biacromial"] = 2.0 * (float(S[0]) + 0.015)
    # legs (horizontal slices through the left leg)
    H, K, A = J["thigh_l"], J["calf_l"], J["foot_l"]
    leg_xy = lambda z: (float(np.interp(z, [A[2], K[2], H[2]], [A[0], K[0], H[0]])),
                        float(np.interp(z, [A[2], K[2], H[2]], [A[1], K[1], H[1]])))
    th = horizontal(model, crotch - 0.03 if crotch else 0.80, leg_xy(0.80), half=(0.30, 0.2))
    M["thigh_circ"] = th["circ"] if th else None
    kn = horizontal(model, float(K[2]), leg_xy(float(K[2])), half=(0.30, 0.2))
    M["knee_circ"] = kn["circ"] if kn else None
    best = None
    for z in np.arange(0.30, 0.42, 0.01):
        h = horizontal(model, z, leg_xy(z), half=(0.30, 0.2))
        if h and (best is None or h["circ"] > best["circ"]):
            best = h
    M["calf_circ"] = best["circ"] if best else None
    an = horizontal(model, float(A[2]) + 0.04, leg_xy(float(A[2]) + 0.04), half=(0.30, 0.2))
    M["ankle_circ"] = an["circ"] if an else None
    # arms: sections perpendicular to the bones
    E, W = J["lowerarm_l"], J["hand_l"]
    b = section(model, S + (E - S) * 0.55, E - S, half=(0.12, 0.12))
    M["biceps_circ"] = b["circ"] if b else None
    best = None
    for t in (0.15, 0.2, 0.25, 0.3, 0.35):
        s = section(model, E + (W - E) * t, W - E, half=(0.12, 0.12))
        if s and (best is None or s["circ"] > best["circ"]):
            best = s
    M["forearm_circ"] = best["circ"] if best else None
    wr = section(model, E + (W - E) * 0.97, W - E, half=(0.12, 0.12))
    M["wrist_circ"] = wr["circ"] if wr else None
    # segment lengths
    M["upper_arm_length"] = float(np.linalg.norm(E - S)) + 0.03   # acromion sits ~3 cm lateral/up of the joint
    M["forearm_length"] = float(np.linalg.norm(W - E))
    ch_ = humanoid.hand_chains(J)["middle"][0]
    M["hand_length"] = float(np.linalg.norm(ch_[0] - W) + sum(np.linalg.norm(ch_[i + 1] - ch_[i]) for i in range(3))) + 0.008
    M["thigh_length"] = float(np.linalg.norm(K - H))
    M["shank_length"] = float(np.linalg.norm(A - K))
    return M


def report(M, tol=0.05):
    rows, n_ok = [], 0
    for k, (tgt, how) in TARGETS.items():
        v = M.get(k)
        if v is None:
            rows.append((k, tgt, None, None, "n/a", how))
            continue
        err = (v - tgt) / tgt
        ok = abs(err) <= tol
        n_ok += ok
        rows.append((k, tgt, v, err, "ok" if ok else ("LOW" if err < 0 else "HIGH"), how))
    return rows, n_ok


def print_report(rows, n_ok):
    print(f"{'measure':<18}{'target':>9}{'model':>9}{'err':>8}  status")
    for k, tgt, v, err, st, how in rows:
        vs = f"{v * 100:7.1f}cm" if v is not None else "      -  "
        es = f"{err * 100:+6.1f}%" if err is not None else "     - "
        print(f"{k:<18}{tgt * 100:7.1f}cm{vs}{es}  {st:<5} {how}")
    print(f"{n_ok}/{len(rows)} within tolerance")


if __name__ == "__main__":
    import humanoid
    m, J = humanoid.build(clothing=False, hair=False)
    rows, n_ok = report(measure(m, J))
    print_report(rows, n_ok)
