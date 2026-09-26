"""
texture — PBR texture authoring in numpy, Substance-Painter style.

Every texel knows where it lives on the high-poly (the baked position map),
which way it faces (object-space normal), how occluded it is (AO) and how
convex/concave (curvature). With that, "smart materials" are just functions:

    region  = which garment owns the texel (same masks that sculpted it)
    albedo  = base color + color zones + procedural variation
              + cavity dirt (AO/concave) + edge wear (convex) + grime from the
              ground up + dust on up-facing surfaces
    rough   = per-material base, polished edges, oily T-zone, dirt is rough
    metal   = 1 only on bare metal, dropped where grime covers it
    height  = micro detail (pores, weave, leather grain, stitches, scratches)
              -> tangent-space detail normal, combined with the baked normal
                 using Reoriented Normal Mapping

Output: BaseColor (sRGB), Normal (OpenGL; DirectX variant = green flipped),
ORM (R = AO, G = roughness, B = metallic — the glTF / UE packing).

PBR sanity ranges followed (sRGB albedo): charcoal ~0.2, dark leather
0.15-0.30, skin 0.45-0.80 (never pure white or black), metals use their
measured reflectance as albedo with metallic = 1.
"""

import math

import numpy as np
from scipy import ndimage

import costume as C
from sdf import fbm, value_noise, ridged


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def srgb(*c):
    return np.array(c, dtype=np.float32)


def lerp(a, b, t):
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 1 and np.ndim(a) and np.shape(a)[-1] == 3:
        t = t[:, None]
    return a + (b - a) * t


def blob(P, c, r):
    """Soft spherical falloff (1 at c, 0 beyond r) — color-zone painting."""
    d = np.linalg.norm(P - np.asarray(c, np.float32), axis=-1)
    return smoothstep(r, 0.0, d)


def blob_sym(P, c, r):
    Q = P.copy()
    Q[:, 0] = np.abs(Q[:, 0])
    return blob(Q, c, r)


def cells(P, scale, seed=0):
    """Cheap cellular-ish pattern: distance to jittered lattice points in 3D
    (1 = cell center, 0 = cell border) — pores, leather grain, pebbling."""
    q = P * scale
    i = np.floor(q).astype(np.int64)
    best = np.full(len(P), 9.0, dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cc = i + np.array([dx, dy, dz])
                h = (cc[:, 0] * 73856093 ^ cc[:, 1] * 19349663 ^ cc[:, 2] * 83492791 ^ seed * 2654435761) & 0xFFFFFF
                jit = np.stack([(h & 0xFF) / 255.0, ((h >> 8) & 0xFF) / 255.0, ((h >> 16) & 0xFF) / 255.0], axis=1)
                d = np.linalg.norm(q - (cc + jit), axis=1)
                best = np.minimum(best, d)
    return np.clip(1.0 - best, 0.0, 1.0).astype(np.float32)


def clean_normal_bake(nts, cov, zmin=0.15, sigma=3.0, log=print):
    """Bake cleanup. Texels whose tangent normal points sideways/backwards
    (the ray caught the wrong surface: nostrils, under the pouch flap, finger
    webbing) are refilled from good neighbors by normalized convolution —
    the automated version of painting over bake errors."""
    n = nts * 2.0 - 1.0
    good = np.isfinite(n).all(-1) & (n[..., 2] >= zmin)
    bad = cov & ~good
    log(f"bake cleanup: {int(bad.sum())} bad normal texels")
    if not bad.any():
        return nts
    n = np.where(good[..., None], n, 0.0)
    w = good.astype(np.float32)
    filled = n.copy()
    todo = bad.copy()
    s = sigma
    for _ in range(6):
        num = np.stack([ndimage.gaussian_filter(n[..., c] * w, s) for c in range(3)], -1)
        den = ndimage.gaussian_filter(w, s)[..., None]
        est = num / np.maximum(den, 1e-6)
        ok = todo & (den[..., 0] > 1e-3)
        filled[ok] = est[ok]
        todo &= ~ok
        if not todo.any():
            break
        s *= 2
    filled[todo] = (0, 0, 1)
    filled /= np.linalg.norm(filled, axis=-1, keepdims=True) + 1e-9
    return filled * 0.5 + 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def author(maps, J, S=None, log=print):
    """maps: dict from bake.bake_maps (arrays with row 0 = V 0); S: the
    character spec (colors and materials; spec.SPEC if None).
    Returns dict of float32 images: basecolor (sRGB), normal_gl, normal_dx, orm,
    plus region map for debugging."""
    import spec as SP
    from landmarks import LM
    S = S or SP.SPEC or SP.resolve({})
    pos = maps["position"]
    Hh, Ww = pos.shape[:2]
    cov = maps["coverage"] > 0.5
    maps = dict(maps)
    maps["normal_ts"] = clean_normal_bake(maps["normal_ts"], ndimage.binary_dilation(cov, iterations=16), log=log)
    # work on the islands plus a generous gutter (positions were baked with
    # an EXTEND margin, so the gutter holds plausible data -> free padding)
    work = ndimage.binary_dilation(cov, iterations=12) & (np.abs(pos).sum(-1) > 1e-6)
    idx = np.nonzero(work)
    P = pos[idx].astype(np.float32)
    N = maps["normal_os"][idx] * 2.0 - 1.0
    ao = maps["ao"][idx][:, 0]
    curv_img = maps["curvature"][..., 0]
    curv_s = ndimage.gaussian_filter(curv_img, 1.0)
    curv_b = ndimage.gaussian_filter(curv_img, 4.0)
    cv = np.clip((curv_s[idx] - 0.5) * 6.0, -1, 1)       # fine edges/cavities
    cvb = np.clip((curv_b[idx] - 0.5) * 10.0, -1, 1)     # broad edges
    edge = smoothstep(0.08, 0.5, cv)
    cavity = smoothstep(-0.05, -0.45, cv)
    up = np.clip(N[:, 2], -1, 1)
    z = P[:, 2]
    n = len(P)
    log(f"texels to author: {n}")

    # ---- regions, anti-aliased as soft weights ----
    rid = C.region_id(P, J, part=maps["part"][idx] if "part" in maps else None)
    rid_img = np.full((Hh, Ww), -1, dtype=np.int32)
    rid_img[idx] = rid
    weights = {}
    for r in range(len(C.region_names())):
        m = np.zeros((Hh, Ww), dtype=np.float32)
        m[idx] = (rid == r)
        if not m.any():
            continue
        weights[r] = ndimage.gaussian_filter(m, 0.6)[idx]
    wsum = sum(weights.values()) + 1e-6
    for r in weights:
        weights[r] /= wsum

    albedo = np.zeros((n, 3), np.float32)
    rough = np.zeros(n, np.float32)
    metal = np.zeros(n, np.float32)
    height = np.zeros(n, np.float32)

    # UV-space coords (for woven patterns that must follow the cloth grain)
    uu_all = (idx[1].astype(np.float32) + 0.5) / Ww
    vv_all = (idx[0].astype(np.float32) + 0.5) / Hh
    # meters per texel, from the position map, to keep patterns physically sized
    dPdu = np.linalg.norm(np.gradient(pos, axis=1), axis=-1)
    dPdv = np.linalg.norm(np.gradient(pos, axis=0), axis=-1)
    mpt_u_all = np.clip(ndimage.median_filter(dPdu, 3)[idx], 1e-5, 0.01)
    mpt_v_all = np.clip(ndimage.median_filter(dPdv, 3)[idx], 1e-5, 0.01)
    ground = float(LM.get("s_body", 1.0))

    # snapshot of the full per-texel arrays: the material blocks below rebind
    # the short names (P, z, ao, ...) to their own subsets
    FULL = {"P": P, "z": z, "ao": ao, "cv": cv, "cvb": cvb, "edge": edge, "cavity": cavity,
            "up": up, "uu": uu_all, "vv": vv_all, "mpt_u": mpt_u_all, "mpt_v": mpt_v_all, "rid": rid}

    class Sub:
        """The texels one material touches (its soft region weight > 0) —
        expensive procedurals only run there."""

        def __init__(self, r):
            self.r = r
            self.w_full = weights.get(r)
            self.sel = np.nonzero(self.w_full > 1e-3)[0] if self.w_full is not None else None

        def __getattr__(self, k):
            if k not in FULL:
                raise AttributeError(k)
            full = FULL[k]
            v = full[self.sel]
            setattr(self, k, v)
            return v

        def noise(self):
            P_ = self.P
            self.n_lo = fbm(P_, 9.0, 3, seed=101) * 0.5 + 0.5
            self.n_mid = fbm(P_, 38.0, 3, seed=102) * 0.5 + 0.5
            self.n_hi = value_noise(P_, 400.0, seed=103) * 0.5 + 0.5
            self.grime = smoothstep(0.55 * ground, 0.02, self.z)     # dirt climbing up from the ground
            self.dust = smoothstep(0.35, 0.9, self.up) * 0.6
            self.n = len(self.sel)
            return self

    def paint(S_, a, ro, me, h):
        w = S_.w_full[S_.sel]
        albedo[S_.sel] += w[:, None] * a
        rough[S_.sel] += w * ro
        metal[S_.sel] += w * me
        height[S_.sel] += w * h

    def region(r):
        S_ = Sub(r)
        return None if S_.sel is None or len(S_.sel) == 0 else S_.noise()

    col = SP.color
    sh = float(LM.get("s_head", 1.0))

    # ================= SKIN =================
    def skin(R):
        P, z, n, cavity, edge, ao, n_mid, n_lo, n_hi = R.P, R.z, R.n, R.cavity, R.edge, R.ao, R.n_mid, R.n_lo, R.n_hi
        sk = S["skin"]
        E = np.asarray(LM["eye_l"], np.float32)
        NT = np.asarray(LM["nose_tip"], np.float32)
        ST = np.asarray(LM["stomion"], np.float32)
        EAR = np.asarray(LM["ear_l"], np.float32)
        HC = np.asarray(LM["head_c"], np.float32)
        cz = float(LM["chin_z"])
        ex, ey, ez = float(E[0]), float(E[1]), float(E[2])
        sy, sz = float(ST[1]), float(ST[2])
        base = col(sk["color"])
        a = np.repeat(base[None], n, 0)
        # the classic face color zones, relative to the base tone: yellowish
        # forehead, red middle third (nose, cheeks, ears), blue-grey lower
        # third (beard shadow, only where there is facial hair)
        red, yel, blu = base * srgb(1.0, 0.81, 0.86), base * srgb(1.057, 1.113, 1.023), base * srgb(0.714, 0.868, 1.023)
        stub_amt = float(sk["stubble"]) / 0.45
        z_fore = smoothstep(ez + 0.016 * sh, ez + 0.056 * sh, z) * blob(P, (0, ey + 0.020 * sh, ez + 0.056 * sh), 0.08 * sh)
        z_red = np.maximum.reduce([blob_sym(P, (ex + 0.0135 * sh, ey + 0.002 * sh, ez - 0.039 * sh), 0.030 * sh),
                                   blob(P, tuple(NT), 0.020 * sh), blob_sym(P, tuple(EAR), 0.034 * sh)])
        beard = (smoothstep(sz + 0.021 * sh, sz - 0.009 * sh, z) * smoothstep(cz - 0.036 * sh, cz + 0.004 * sh, z)
                 * smoothstep(HC[1] + 0.01 * sh, HC[1] - 0.04 * sh, P[:, 1]) * (1 - blob(P, (0, sy + 0.008 * sh, sz - 0.002 * sh), 0.016 * sh)))
        beard = np.maximum(beard, blob(P, (0, sy + 0.0035 * sh, sz + 0.013 * sh), 0.014 * sh) * 0.9)  # moustache area
        jaw = (float(EAR[0]) - 0.021 * sh, float(EAR[1]) - 0.040 * sh, ez - 0.049 * sh)
        beard = np.maximum(beard, smoothstep(ez - 0.024 * sh, ez - 0.054 * sh, z) * blob_sym(P, jaw, 0.03 * sh))
        a = lerp(a, yel, (z_fore * 0.30)[:, None])
        a = lerp(a, red, (z_red * 0.32)[:, None])
        a = lerp(a, blu, (beard * 0.30 * min(stub_amt, 1.5))[:, None])
        # stubble: at game texel density single follicles are sub-pixel, so it
        # is a soft shadow with a fine grain, strongest on chin/upper lip
        grain = value_noise(P, 1800.0, seed=5) * 0.5 + 0.5
        stub = beard * (0.55 + 0.45 * grain) * (0.8 + 0.4 * n_mid)
        stub_col = lerp(col(S["beard"]["color"]), base, 0.35)
        a = lerp(a, stub_col, (stub * 0.42 * stub_amt).clip(0, 0.8)[:, None])
        lips = blob(P, (0, sy + 0.0075 * sh, sz), 0.014 * sh) * smoothstep(0.004 * sh, 0.0, np.abs(P[:, 2] - sz) - 0.0045 * sh)
        a = lerp(a, col(sk["lips"]), (lips * 0.8)[:, None])
        # eyebrows: a tapered band along the brow ridge (thick at the head,
        # thin at the tail), hair streaks angled up-and-out, soft edges
        Pe = P.copy()
        Pe[:, 0] = np.abs(Pe[:, 0])
        tb = np.clip((Pe[:, 0] - (ex - 0.0205 * sh)) / (0.047 * sh), 0.0, 1.0)
        brow_c = ez + (0.0215 + 0.0055 * np.sin(np.minimum(tb * 1.25, 1.0) * math.pi * 0.85) - 0.006 * smoothstep(0.75, 1.0, tb)) * sh
        half = 0.0042 * sh * (1.0 - 0.55 * tb)
        dz = np.abs(z - brow_c)
        brow = (smoothstep(half + 0.0012 * sh, half - 0.0008 * sh, dz) * smoothstep(ex - 0.0225 * sh, ex - 0.0185 * sh, Pe[:, 0])
                * smoothstep(ex + 0.0285 * sh, ex + 0.0225 * sh, Pe[:, 0]) * smoothstep(ey + 0.014 * sh, ey + 0.004 * sh, Pe[:, 1]))
        ca, sa = math.cos(0.45), math.sin(0.45)
        q = np.stack([Pe[:, 0] * ca + P[:, 2] * sa, -Pe[:, 0] * sa + P[:, 2] * ca, P[:, 1]], axis=1).astype(np.float32)
        strokes = fbm(q * np.array([0.25, 1.0, 1.0], np.float32), 1400.0 / sh, 2, seed=7) * 0.5 + 0.5
        brow_m = brow * (0.55 + 0.45 * smoothstep(0.35, 0.7, strokes))
        a = lerp(a, col(S["hair"]["color"]) * 0.65, (brow_m * 0.92)[:, None])
        eyes = blob_sym(P, (ex, ey - 0.005 * sh, ez), 0.022 * sh)
        a = lerp(a, base * srgb(0.83, 0.83, 0.98), (eyes * 0.25)[:, None])
        a = a * (0.94 + 0.10 * n_mid[:, None])
        fr = smoothstep(0.80, 0.9, cells(P, 260.0, seed=9)) * (n_lo > 0.55)
        a = lerp(a, base * srgb(0.79, 0.70, 0.63), (fr * float(sk["freckles"]))[:, None])
        a = a * (1.0 - 0.18 * cavity)[:, None]
        pores = cells(P, 700.0, seed=11)
        ro = float(sk["roughness"]) - 0.12 * np.maximum(z_fore, blob(P, (0, float(NT[1]) + 0.011 * sh, float(NT[2]) + 0.019 * sh), 0.03 * sh)) - 0.10 * lips + 0.08 * beard * min(stub_amt, 1.0)
        ro = ro + 0.06 * (1 - pores)
        h = -0.00004 * smoothstep(0.55, 0.8, 1 - pores) + 0.00002 * n_hi + 0.00006 * brow_m * strokes
        h = h - 0.00006 * lips * smoothstep(0.6, 0.9, value_noise(P * np.array([1, 1, 6], np.float32), 900.0, 17) * 0.5 + 0.5)
        paint(R, a, ro, 0.0, h)

    # ================= HAIR (sculpted cap / beard / hair pieces) =================
    def hair(R, mat):
        P, z, n, n_mid = R.P, R.z, R.n, R.n_mid
        base = col(mat["color"])
        vertical = mat.get("style") in ("beard",) or mat.get("piece")
        strand = fbm(P * np.array([1.0, 0.18, 1.0], np.float32), 600.0, 3, seed=21) * 0.5 + 0.5
        strand_s = fbm(P * np.array([1.0, 1.0, 0.2], np.float32), 600.0, 3, seed=22) * 0.5 + 0.5
        if vertical:
            st = strand_s
        else:
            ez = float(LM["eye_l"][2])
            top = smoothstep(ez + 0.016 * sh, ez + 0.096 * sh, z)
            st = top * strand + (1 - top) * strand_s
        a = lerp(base * 0.44, np.minimum(base * 1.3 + 0.01, 1.0), (st * 0.8 + 0.2 * n_mid)[:, None])
        if mat.get("style") not in ("beard",) and not mat.get("piece"):
            # feathered hairline: scalp shows between strands — a darkened
            # skin/hair mix, never a bright band
            hm = C.hair_mask(P)
            scalp = lerp(col(S["skin"]["color"]) * 0.5, base, 0.3)
            a = lerp(scalp, a, smoothstep(0.02, 0.35, hm * (0.6 + 0.6 * st))[:, None])
        ro = 0.42 + 0.2 * (1 - st)
        paint(R, a, ro, 0.0, 0.00025 * (st - 0.5))

    # ================= WOVEN CLOTH (linen / fine cloth) =================
    def woven(R, mat, per=0.0024, stains=1.0, rough0=0.82):
        P, z, n, cavity, edge, ao, n_mid, uu, vv, mpt_u, mpt_v = R.P, R.z, R.n, R.cavity, R.edge, R.ao, R.n_mid, R.uu, R.vv, R.mpt_u, R.mpt_v
        gu = uu * Ww * mpt_u / per
        gv = vv * Hh * mpt_v / per
        weave = 0.5 + 0.25 * (np.sin(2 * math.pi * gu) * np.sign(np.sin(math.pi * gv)) +
                              np.sin(2 * math.pi * gv) * np.sign(np.sin(math.pi * gu)))
        slub = fbm(P * np.array([1, 1, 4], np.float32), 120.0, 2, seed=31) * 0.5 + 0.5
        base = col(mat["color"])
        a = base * (0.90 + 0.10 * weave + 0.08 * (slub - 0.5))[:, None]
        # sweat/dirt: armpits, collar, cuffs
        Sj = J["upperarm_l"]
        sb = float(LM.get("s_body", 1.0))
        pits = blob_sym(P, (float(Sj[0]) - 0.04 * sb, float(Sj[1]) + 0.01 * sb, float(Sj[2]) - 0.08 * sb), 0.07 * sb)
        collar = smoothstep(LM["neck_base_z"] - 0.035 * sb, LM["neck_base_z"] + 0.025 * sb, z)
        cuffs = blob_sym(P, tuple(J["hand_l"] - J["_hand_axis_l"] * 0.02), 0.05 * sb)
        stain = np.clip(pits * 0.5 + collar * 0.35 + cuffs * 0.4, 0, 1) * (0.5 + 0.5 * n_mid) * stains
        a = lerp(a, base * srgb(0.79, 0.73, 0.62), stain[:, None] * 0.6)
        a = a * (1.0 - 0.35 * cavity - 0.25 * (1 - ao))[:, None]
        a = lerp(a, a * 1.06, (edge * 0.5)[:, None])
        ro = rough0 + 0.06 * weave - 0.05 * stain
        paint(R, a, ro, 0.0, 0.00012 * (per / 0.0024) * (weave - 0.5) + 0.00005 * (slub - 0.5))

    # ================= WOOL (twill) =================
    def wool(R, mat):
        P, z, n, cavity, ao, n_mid, n_lo, n_hi, grime, uu, vv, mpt_u, mpt_v = R.P, R.z, R.n, R.cavity, R.ao, R.n_mid, R.n_lo, R.n_hi, R.grime, R.uu, R.vv, R.mpt_u, R.mpt_v
        per = 0.0028
        gu = uu * Ww * mpt_u / per
        gv = vv * Hh * mpt_v / per
        twill = 0.5 + 0.5 * np.sin(2 * math.pi * (gu + gv) * 0.7)
        base = col(mat["color"])
        a = base * (0.88 + 0.14 * twill[:, None] + 0.10 * (n_mid[:, None] - 0.5))
        Kj = J["calf_l"]
        knees = np.maximum(blob_sym(P, (float(Kj[0]), float(Kj[1]) - 0.05, float(Kj[2])), 0.07), 0)
        a = lerp(a, a * 1.35, (knees * 0.5 * (0.5 + n_hi))[:, None])
        mud = np.clip(grime * 0.8 * (0.4 + n_lo) + knees * 0.25 * n_lo, 0, 1)
        a = lerp(a, srgb(0.33, 0.28, 0.21), mud[:, None] * 0.7)
        a = a * (1.0 - 0.35 * cavity - 0.25 * (1 - ao))[:, None]
        ro = 0.86 + 0.04 * twill - 0.05 * knees
        paint(R, a, ro, 0.0, 0.00010 * (twill - 0.5) + 0.00004 * (n_hi - 0.5))

    # ================= LEATHER =================
    def leather(R, mat, grain_scale=1500.0, wear_amt=0.8, rough_base=0.55, grime_amt=0.0, stitch_dist=0.0045):
        r = R.r
        P, n, cavity, edge, cvb, ao, n_mid, n_lo, n_hi, grime, dust = (
            R.P, R.n, R.cavity, R.edge, R.cvb, R.ao, R.n_mid, R.n_lo, R.n_hi, R.grime, R.dust)
        base = col(mat["color"])
        grain = cells(P, grain_scale, seed=40 + r)
        wrinkle = ridged(P * np.array([1, 1, 1.6], np.float32), 90.0, 2, seed=50 + r, sharp=5.0)
        a = base * (0.85 + 0.25 * n_mid + 0.08 * grain)[:, None]
        # darker, more saturated in cavities; lighter & desaturated on worn edges
        a = a * (1.0 - 0.45 * cavity - 0.30 * (1 - ao))[:, None]
        worn = np.clip(edge * wear_amt * (0.6 + 0.8 * n_hi) + smoothstep(0.1, 0.6, cvb) * wear_amt * 0.4, 0, 1)
        a = lerp(a, np.minimum(base * 1.9 + 0.05, 1.0), worn[:, None] * 0.55)
        scr = smoothstep(0.93, 0.99, ridged(P * np.array([1.0, 3.0, 1.0], np.float32), 60.0, 1, seed=60 + r, sharp=8.0))
        a = lerp(a, np.minimum(base * 2.2, 1.0), (scr * 0.35)[:, None])
        # stitching along the garment border — measured in 3D (distance to
        # texels of *other* regions): in UV space every island seam would
        # also look like a border
        from scipy.spatial import cKDTree
        allP, allR = FULL["P"], FULL["rid"]
        lo, hi = P.min(0) - 0.02, P.max(0) + 0.02
        near = np.all((allP > lo) & (allP < hi), axis=1) & (allR != r)
        other = allP[near][::2]
        if len(other):
            dist_m, _ = cKDTree(other).query(P, distance_upper_bound=0.02, workers=-1)
            dist_m = np.minimum(dist_m, 0.02).astype(np.float32)
        else:
            dist_m = np.full(len(P), 0.02, np.float32)
        band = smoothstep(0.0012, 0.0004, np.abs(dist_m - stitch_dist))
        dash = smoothstep(0.1, 0.5, np.sin(2 * math.pi * (P[:, 0] + P[:, 1] + P[:, 2]) / 0.0055))
        stitch = band * dash
        a = lerp(a, lerp(base, srgb(0.78, 0.68, 0.52), 0.7), (stitch * 0.85)[:, None])
        mud = np.clip(grime * grime_amt * (0.5 + 0.8 * n_lo) + cavity * grime * grime_amt * 0.5, 0, 1)
        a = lerp(a, srgb(0.30, 0.25, 0.19), mud[:, None])
        a = lerp(a, srgb(0.50, 0.46, 0.40), (dust * 0.15 * n_lo)[:, None])
        ro = rough_base + 0.10 * (1 - grain) - 0.18 * worn + 0.35 * mud + 0.05 * wrinkle
        h = (0.00006 * (grain - 0.5) - 0.00010 * wrinkle * (0.5 + n_lo) + 0.00018 * stitch - 0.00006 * band
             - 0.00004 * scr)
        paint(R, a, np.clip(ro, 0.2, 0.95), 0.0, h)

    # ================= METAL =================
    METAL_RGB = {"brass": (0.82, 0.68, 0.40), "iron": (0.53, 0.51, 0.49), "steel": (0.63, 0.62, 0.64),
                 "silver": (0.95, 0.93, 0.88), "gold": (1.0, 0.78, 0.34), "bronze": (0.80, 0.50, 0.25)}

    def metal_mat(R, mat):
        P, cavity, edge, ao, n_mid, n_hi = R.P, R.cavity, R.edge, R.ao, R.n_mid, R.n_hi
        # a named metal sets the reflectance; an explicit color tints it (or
        # is the reflectance when no metal is named)
        kind = mat.get("metal")
        if kind:
            refl = srgb(*METAL_RGB.get(kind, METAL_RGB["iron"]))
            if mat.get("color") and mat.get("material") == "metal":
                refl = lerp(refl, col(mat["color"]), 0.5)
        else:
            kind = "iron"
            refl = col(mat["color"]) if mat.get("color") else srgb(*METAL_RGB["iron"])
        dark = refl * srgb(0.30, 0.29, 0.28) if kind in ("iron", "steel") else refl * srgb(0.30, 0.29, 0.22)
        tarnish = np.clip(cavity * 0.9 + (1 - ao) * 0.8 + 0.3 * n_mid, 0, 1)
        polish = np.clip(edge * 1.5, 0, 1)
        a = lerp(refl, dark, (tarnish * (1 - polish))[:, None])
        me = np.clip(1.0 - 0.8 * tarnish * (1 - polish), 0, 1)
        ro = 0.30 + 0.35 * tarnish - 0.15 * polish + 0.08 * n_hi
        scr = smoothstep(0.9, 0.99, ridged(P, 300.0, 1, seed=77, sharp=8.0))
        paint(R, a, ro, me, -0.00003 * scr)

    # ================= FUR =================
    def fur(R, mat):
        P, n, cavity, ao, n_mid = R.P, R.n, R.cavity, R.ao, R.n_mid
        base = col(mat["color"])
        strands = fbm(P * np.array([1.0, 1.0, 0.15], np.float32), 700.0, 3, seed=81) * 0.5 + 0.5
        tufts = fbm(P, 60.0, 2, seed=82) * 0.5 + 0.5
        a = lerp(base * 0.45, np.minimum(base * 1.45, 1.0), (0.7 * strands + 0.3 * tufts)[:, None])
        a = a * (1.0 - 0.3 * cavity - 0.3 * (1 - ao))[:, None]
        paint(R, a, 0.88 + 0.08 * strands, 0.0, 0.0003 * (strands - 0.5) + 0.0002 * (tufts - 0.5))

    # ================= HORN / BONE / TUSK =================
    def horn(R, mat):
        P, n, cavity, edge, ao, n_mid, n_hi = R.P, R.n, R.cavity, R.edge, R.ao, R.n_mid, R.n_hi
        base = col(mat["color"])
        hc = np.asarray(LM["head_c"], np.float32)
        dist = np.linalg.norm(P - hc, axis=1)
        bands = 0.5 + 0.5 * np.sin(dist * 2 * math.pi / (0.008 * sh))
        streak = fbm(P * np.array([1.0, 1.0, 0.3], np.float32), 250.0, 2, seed=91) * 0.5 + 0.5
        a = base * (0.80 + 0.18 * bands + 0.12 * (streak - 0.5))[:, None]
        a = a * (1.0 - 0.40 * cavity - 0.25 * (1 - ao))[:, None]
        a = lerp(a, np.minimum(base * 1.15, 1.0), (edge * 0.4)[:, None])
        paint(R, a, 0.45 + 0.15 * (1 - bands) + 0.1 * n_hi, 0.0, 0.00008 * (bands - 0.5))

    # ================= SOLE =================
    def sole(R, mat):
        P, n, n_mid, n_lo = R.P, R.n, R.n_mid, R.n_lo
        a = col(mat["color"]) * (0.9 + 0.2 * n_mid)[:, None]
        a = lerp(a, srgb(0.30, 0.25, 0.19), (0.6 * n_lo)[:, None])
        paint(R, a, np.full(n, 0.85, np.float32), 0.0, 0.00008 * (cells(P, 500.0, 3) - 0.5))

    # ---- dispatch every region to its material recipe ----
    LEATHER_USE = {  # garment -> (grain, wear, roughness, ground grime, stitch distance)
        "vest": (1500.0, 0.8, 0.55, 0.0, 0.0045), "belt": (1800.0, 1.0, 0.48, 0.0, 0.0030),
        "boots": (1300.0, 1.0, 0.45, 0.9, 0.006), "shoes": (1300.0, 1.0, 0.45, 0.9, 0.005),
        "gloves": (1900.0, 0.6, 0.50, 0.0, 0.0030), "bracers": (1600.0, 0.9, 0.50, 0.0, 0.0035),
        "trousers": (1500.0, 0.6, 0.55, 0.5, 0.0045), "shirt": (1600.0, 0.5, 0.55, 0.0, 0.0040),
    }
    for r, name in enumerate(C.region_names()):
        R = region(r)
        if R is None:
            continue
        kind = C.region_kind(r)
        mat = C.region_material(r)
        if r == C.SKIN:
            skin(R)
            continue
        mtype = mat.get("material", "cloth")
        if kind == "hair" or mtype == "hair":
            hair(R, dict(mat, piece=kind == "piece"))
        elif kind == "sole":
            sole(R, mat)
        elif kind == "metal" or mtype == "metal":
            metal_mat(R, mat)
        elif mtype == "leather":
            key = mat.get("type", "vest") if kind != "piece" else "belt"
            leather(R, mat, *LEATHER_USE.get(key, LEATHER_USE["vest"]))
        elif mtype == "wool":
            wool(R, mat)
        elif mtype == "linen":
            woven(R, mat)
        elif mtype == "cloth":
            woven(R, mat, per=0.0011, stains=0.4, rough0=0.72)
        elif mtype == "fur":
            fur(R, mat)
        elif mtype == "horn":
            horn(R, mat)
        else:
            woven(R, mat)

    n = len(idx[0])
    # ---- detail normal from height, combined with the baked normal (RNM) ----
    Himg = np.zeros((Hh, Ww), np.float32)
    Himg[idx] = height
    dHdx = np.gradient(Himg, axis=1)[idx] / mpt_u_all
    dHdy = np.gradient(Himg, axis=0)[idx] / mpt_v_all
    nd = np.stack([-dHdx, -dHdy, np.ones(n, np.float32)], axis=1)
    nd /= np.linalg.norm(nd, axis=1, keepdims=True)
    nb = maps["normal_ts"][idx] * 2.0 - 1.0
    nb /= np.linalg.norm(nb, axis=1, keepdims=True) + 1e-9
    t = nb + np.array([0, 0, 1.0], np.float32)
    u = nd * np.array([-1, -1, 1.0], np.float32)
    rnm = t * (np.sum(t * u, axis=1, keepdims=True) / np.maximum(t[:, 2:3], 1e-4)) - u
    ln = np.linalg.norm(rnm, axis=1, keepdims=True)
    ok = np.isfinite(rnm).all(1, keepdims=True) & (ln > 1e-6)
    rnm = np.where(ok, rnm / np.maximum(ln, 1e-6), nb)

    def to_img(vals, ch, fill):
        img = np.empty((Hh, Ww, ch), np.float32)
        img[:] = fill
        img[idx] = vals if vals.ndim == 2 else vals[:, None]
        return img

    out = {
        "basecolor": to_img(np.clip(albedo, 0.02, 0.95), 3, (0.5, 0.5, 0.5)),
        "normal_gl": to_img(rnm * 0.5 + 0.5, 3, (0.5, 0.5, 1.0)),
        "orm": to_img(np.stack([FULL["ao"], np.clip(rough, 0.05, 1.0), np.clip(metal, 0, 1)], axis=1), 3, (1.0, 0.8, 0.0)),
        "height": to_img(height, 1, 0.0),
        "region": rid_img,
    }
    ndx = out["normal_gl"].copy()
    ndx[..., 1] = 1.0 - ndx[..., 1]
    out["normal_dx"] = ndx
    return out


def save_png(img, path, flip=True):
    """img: float (H,W,C) row0 = V0. Saved top-down as 8-bit PNG."""
    from PIL import Image
    a = np.clip(img, 0, 1)
    if flip:
        a = a[::-1]
    a = (a * 255.0 + 0.5).astype(np.uint8)
    if a.shape[2] == 1:
        a = a[..., 0]
    Image.fromarray(a).save(path, optimize=True)
