"""
landmarks — anatomical landmarks every stage keys off.

Everything downstream (garment masks, hairline, retopo rows and face loops,
eye sockets, texture color zones, eye bones) is expressed relative to these
points instead of hard-coded coordinates, so a different base body (another
reference, other proportions) flows through the whole pipeline.

`LM` is filled by humanoid.build(); modules read it at call time.
"""

import math

import numpy as np

LM = {}


def _v(*a):
    return np.array(a, dtype=np.float64)


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def skeleton_from_reference(ref):
    """Map MakeHuman joint helpers to this pipeline's joint names (left side
    + center; the right side mirrors X)."""
    R = ref["joints"]
    V = ref["verts"]
    J = {}
    J["root"] = _v(0, 0, 0)
    J["pelvis"] = R["pelvis"].astype(float)
    J["spine_01"] = R["spine-4"].astype(float)
    J["spine_02"] = R["spine-2"].astype(float)
    J["spine_03"] = R["spine-1"].astype(float)
    J["neck_01"] = R["neck"].astype(float)
    J["head"] = R["head"].astype(float)
    top = float(V[:, 2].max())
    J["head_top"] = _v(0, float(R["head"][1]), top)
    J["clavicle_l"] = R["l-clavicle"].astype(float)
    J["upperarm_l"] = R["l-shoulder"].astype(float)
    J["lowerarm_l"] = R["l-elbow"].astype(float)
    J["hand_l"] = R["l-hand"].astype(float)
    J["thigh_l"] = R["l-upper-leg"].astype(float)
    J["calf_l"] = R["l-knee"].astype(float)
    J["foot_l"] = R["l-ankle"].astype(float)
    J["ball_l"] = R["l-foot-1"].astype(float) + _v(0, 0, 0.012)
    J["toe_l"] = R["l-foot-2"].astype(float) + _v(0, 0, 0.010)
    # finger chains: MakeHuman finger-k-1..4 = MCP (CMC for the thumb), PIP, DIP, tip
    names = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}
    radii = {"thumb": 0.0115, "index": 0.0085, "middle": 0.0088, "ring": 0.0082, "pinky": 0.0072}
    chains = {}
    for k, nm in names.items():
        pts = np.array([R[f"l-finger-{k}-{i}"] for i in (1, 2, 3, 4)], dtype=np.float64)
        r0 = radii[nm]
        chains[nm] = (pts, np.array([r0, r0 * 0.9, r0 * 0.8, r0 * 0.72]))
    J["_chains_l"] = chains
    W = J["hand_l"]
    ax = normalize(chains["middle"][0][0] - W)
    th = chains["index"][0][0] - chains["pinky"][0][0]
    th = normalize(th - ax * np.dot(th, ax))
    pn = np.cross(th, ax)
    # palm normal points to where the fingers curl
    curl = chains["middle"][0][3] - chains["middle"][0][0]
    if np.dot(pn, curl - ax * np.dot(curl, ax)) < 0:
        pn = -pn
    J["_hand_axis_l"], J["_thumb_dir_l"], J["_palm_n_l"] = ax, th, normalize(pn)
    return J


def from_reference(ref, J):
    """Face / head / torso landmarks measured on the reference surface."""
    V = ref["verts"]
    used = np.unique(np.concatenate([np.array(f) for f in ref["faces"]]))
    P = V[used].astype(np.float64)
    R = ref["joints"]
    L = {}
    eye = R["l-eye"].astype(float)
    L["eye_l"] = eye
    L["eye_r_radius"] = float(ref["eye_radius"]["l"]) * 0.78
    L["head_top"] = float(P[:, 2].max())
    mid = P[np.abs(P[:, 0]) < 0.004]
    face = mid[(mid[:, 2] > eye[2] - 0.16) & (mid[:, 2] < eye[2] + 0.02)]
    # profile of the face midline: most anterior point per height
    zs = np.arange(eye[2] - 0.15, eye[2] + 0.015, 0.001)
    prof = []
    for z in zs:
        sl = face[np.abs(face[:, 2] - z) < 0.0015]
        prof.append(sl[:, 1].min() if len(sl) else np.nan)
    prof = np.array(prof)
    ok = ~np.isnan(prof)
    zs, prof = zs[ok], prof[ok]
    i_nose = int(np.argmin(np.where(zs < eye[2] - 0.02, prof, 9)))
    L["nose_tip"] = _v(0, prof[i_nose], zs[i_nose])
    # below the nose: upper lip max, stomion (dip), lower lip max, chin
    below = zs < zs[i_nose] - 0.012
    zb, pb = zs[below], prof[below]
    # lips = the two most anterior local maxima (min y) in the band 1-5 cm under the nose
    band = (zb > zs[i_nose] - 0.055)
    zl, pl = zb[band], pb[band]
    order = np.argsort(pl)
    up_i = order[0]
    lo_i = next(i for i in order[1:] if abs(zl[i] - zl[up_i]) > 0.008)
    a, b = sorted((up_i, lo_i), key=lambda i: zl[i])
    seg = slice(min(a, b), max(a, b) + 1)
    k = int(np.argmax(pl[seg])) + min(a, b)
    L["stomion"] = _v(0, pl[k], zl[k])
    L["upper_lip"] = _v(0, pl[max(a, b)], zl[max(a, b)])
    L["lower_lip"] = _v(0, pl[min(a, b)], zl[min(a, b)])
    # chin: the front of the jaw 2-10 cm under the mouth (bounded below and
    # behind — unbounded, "the lowest midline point in front" is the groin)
    st = L["stomion"]
    chin_band = mid[(mid[:, 2] < st[2] - 0.02) & (mid[:, 2] > st[2] - 0.10) & (mid[:, 1] < st[1] + 0.05)]
    L["chin"] = chin_band[np.argmin(chin_band[:, 2])]
    L["pogonion"] = chin_band[np.argmin(chin_band[:, 1])]
    drop = float(st[2] - L["chin"][2])
    assert 0.03 < drop < 0.095, f"chin {drop * 100:.1f} cm under the stomion: landmark search failed"
    # ears: lateral-most head point near eye height, behind the eyes
    head = P[(P[:, 2] > eye[2] - 0.05) & (P[:, 2] < eye[2] + 0.02) & (P[:, 1] > eye[1] + 0.04) & (P[:, 0] > 0)]
    e = head[np.argmax(head[:, 0])]
    L["ear_l"] = e
    # head center: middle of the skull between the ears, at brow height
    skull = P[(P[:, 2] > eye[2]) & (np.abs(P[:, 0]) < 0.12)]
    yc = 0.5 * (skull[:, 1].min() + skull[:, 1].max())
    L["head_c"] = _v(0, yc, eye[2] + 0.005)
    L["chin_z"] = float(L["chin"][2])
    sh = (L["head_top"] - L["chin_z"]) / 0.237          # head size relative to the Ranger's
    L["brow_z"] = eye[2] + 0.022 * sh
    # neck: base (sternal notch height, ~0.72 neck-joint-to-head lengths under
    # the neck joint) and axis
    L["neck_axis_y"] = float(J["neck_01"][1])
    L["neck_base_z"] = float(J["neck_01"][2]) - 0.72 * float(np.linalg.norm(J["head"] - J["neck_01"]))
    L["shoulder_z"] = float(J["upperarm_l"][2])
    # limb radii (median skin distance from the bone, mid-segment)
    for nm, a, b in (("r_upperarm", "upperarm_l", "lowerarm_l"), ("r_forearm", "lowerarm_l", "hand_l"),
                     ("r_thigh", "thigh_l", "calf_l"), ("r_calf", "calf_l", "foot_l")):
        A, B = np.asarray(J[a], float), np.asarray(J[b], float)
        ab = B - A
        t = ((P - A) @ ab) / float(ab @ ab)
        d = np.linalg.norm(P - (A + np.clip(t, 0, 1)[:, None] * ab), axis=1)
        sel = (t > 0.35) & (t < 0.65) & (d < 0.25 * np.linalg.norm(ab) + 0.06) & (P[:, 0] > 0.02)
        L[nm] = float(np.median(d[sel])) if sel.sum() > 20 else 0.05
    return L


def procedural_defaults(J):
    """Landmarks of the procedural (primitive-sculpted) body, for the old path."""
    return {
        "eye_l": _v(0.0315, -0.0800, 1.6840), "eye_r_radius": 0.0120, "head_top": 1.80,
        "nose_tip": _v(0, -0.117, 1.6455), "stomion": _v(0, -0.1085, 1.609),
        "upper_lip": _v(0, -0.108, 1.614), "lower_lip": _v(0, -0.107, 1.603),
        "chin": _v(0, -0.085, 1.566), "pogonion": _v(0, -0.101, 1.579), "ear_l": _v(0.083, 0.010, 1.668),
        "head_c": _v(0, 0.010, 1.690), "brow_z": 1.7065, "neck_axis_y": 0.02, "neck_base_z": 1.475,
        "chin_z": 1.566, "shoulder_z": float(J["upperarm_l"][2]),
    }


RANGER_RADII = {"r_upperarm": 0.048, "r_forearm": 0.034, "r_thigh": 0.091, "r_calf": 0.055}


def scales(L, J):
    """Body-relative sizes the costume and textures scale by (1.0 = the
    Ranger: 1.80 m, head 23.7 cm, foot 14.3 cm ankle-to-ball)."""
    stature = float(L["head_top"])
    pelvis, sp1 = float(J["pelvis"][2]), float(J["spine_01"][2])
    A, B = np.asarray(J["foot_l"], float), np.asarray(J["ball_l"], float)
    out = {
        "stature": stature,
        "s_body": stature / 1.80,
        "s_head": (float(L["head_top"]) - float(L["chin_z"])) / 0.237,
        "s_foot": float(np.linalg.norm(B - A)) / 0.1426,
        "belt_z": pelvis + 0.42 * (sp1 - pelvis),       # natural waistline, over the iliac crest
    }
    for k, v in RANGER_RADII.items():
        out.setdefault(k, L.get(k, v * out["s_body"]))
        out["s_" + k[2:]] = out[k] / v                   # limb thickness relative to the Ranger's
    return out


def eye_margin(model_body, eye_c, r_eye, n=48):
    """Lid margin around the eye: for each angle around the view axis, the
    point on the eyeball where the (body-only) surface starts covering it."""
    from sdf import sample
    view = _v(0, -1, 0)
    ex, ez = _v(1, 0, 0), _v(0, 0, 1)
    out = []
    thetas = np.radians(np.arange(2.0, 89.0, 0.5))
    for k in range(n):
        psi = 2 * math.pi * k / n
        side = math.cos(psi) * ex + math.sin(psi) * ez
        dirs = np.cos(thetas)[:, None] * view + np.sin(thetas)[:, None] * side
        pts = (eye_c + dirs * (r_eye + 0.0008)).astype(np.float32)
        f = sample(model_body, pts)
        covered = np.nonzero(f < 0)[0]
        i = covered[0] if len(covered) else len(thetas) - 1
        out.append((psi, eye_c + dirs[i] * (r_eye + 0.0006)))
    return out


def margin_point(margin, psi):
    """Interpolate the lid margin at angle psi (radians, around the view axis,
    0 = toward +X of that eye's local frame)."""
    ps = np.array([m[0] for m in margin])
    pts = np.array([m[1] for m in margin])
    psi = psi % (2 * math.pi)
    i = int(np.searchsorted(ps, psi)) % len(ps)
    j = (i - 1) % len(ps)
    a, b = ps[j], ps[i] if i else ps[i] + 2 * math.pi
    t = 0.0 if b == a else ((psi - a) % (2 * math.pi)) / ((b - a) % (2 * math.pi) or 1.0)
    return pts[j] * (1 - t) + pts[i] * t


def mirror_x(p):
    q = np.array(p, dtype=np.float64)
    q[0] = -q[0]
    return q
