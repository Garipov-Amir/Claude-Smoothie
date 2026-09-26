"""
retopo — code-driven *manual* retopology of the sculpt.

A game mesh's topology is designed, not computed: it has to deform. So the
cage here is laid out the way a character artist lays out a retopo in
TopoGun / Blender's poly-build, just with coordinates instead of clicks:

* Every limb is a tube of rings perpendicular to its bone, with extra rings
  bracketing each joint (elbow, knee, wrist, finger joints) — loops sit
  where the skin bends, and the rings keep their angular slots down the whole
  limb so edges flow along the muscles instead of spiralling.
* Junctions follow standard patterns: the torso ring splits into two leg
  rings plus a crotch vertex; the arm plugs into a 2x3-face hole in the
  torso side with its first ring running over the deltoid and through the
  armpit; the palm's 10-ring feeds four 4-sided finger tubes sharing their
  webbing edges; the thumb is extruded from the side of the palm.
* The face gets concentric loops around the eyes (two insets, the inner
  face removed for the eyeball) and around the mouth (one inset around the
  lip region, the lip line as its own loop), and a grid cap on the crown so
  there are no triangle fans.

Positions come from ray casts against the high-poly (from inside the body
along each ring's slot direction), so the cage already hugs the sculpt. It is
then subdivided once (Catmull-Clark) to LOD0 density and every vertex is
re-projected radially from its bone + relaxed tangentially a few times —
the same "relax + snap to surface" loop retopo tools run.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import humanoid
from landmarks import LM

# part labels (stored per vertex; decide which bone axis a vertex projects from)
TRUNK, ARM, HAND, LEG, FINGER = 0, 10, 20, 30, 40
FINGER_NAMES = ["index", "middle", "ring", "pinky", "thumb"]


def lab(kind, side=1, finger=0):
    return kind + (0 if side > 0 else 5) + (finger if kind == FINGER else 0)


def nrm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def mirror(p, side):
    p = np.array(p, dtype=np.float64)
    if side < 0:
        p[0] = -p[0]
    return p


def mirror_vec(v, side):
    return mirror(v, side)


# ---------------------------------------------------------------------------
# Surface queries against the high-poly
# ---------------------------------------------------------------------------

class Surface:
    def __init__(self, obj):
        dg = bpy.context.evaluated_depsgraph_get()
        self.bvh = BVHTree.FromObject(obj, dg)

    def ray(self, origin, direction, max_dist=0.35):
        loc, _n, _i, dist = self.bvh.ray_cast(Vector(origin), Vector(nrm(direction)), max_dist)
        return None if loc is None else np.array(loc)

    def nearest(self, p):
        loc, _n, _i, _d = self.bvh.find_nearest(Vector(p))
        return np.array(loc)

    def cast_ring(self, center, dirs, fallback_r=0.03, max_dist=0.3):
        pts = []
        for d in dirs:
            h = self.ray(center, d, max_dist)
            pts.append(h if h is not None else np.asarray(center) + nrm(d) * fallback_r)
        return pts


# ---------------------------------------------------------------------------
# Cage container
# ---------------------------------------------------------------------------

class Cage:
    def __init__(self):
        self.V, self.L, self.F = [], [], []
        self.seams = set()   # UV seams, as vertex-id pairs
        self.loops = {}      # named edge loops (e.g. hem lines) for later stages

    def seam_chain(self, ids, closed=False):
        n = len(ids)
        for i in range(n if closed else n - 1):
            a, b = ids[i], ids[(i + 1) % n]
            if a != b:
                self.seams.add((min(a, b), max(a, b)))

    def v(self, p, label):
        self.V.append(np.asarray(p, dtype=np.float64))
        self.L.append(label)
        return len(self.V) - 1

    def ring(self, pts, label):
        return [self.v(p, label) for p in pts]

    def bridge(self, a, b):
        n = len(a)
        assert n == len(b), (n, len(b))
        for i in range(n):
            j = (i + 1) % n
            self.F.append((a[i], a[j], b[j], b[i]))

    def grid_cap(self, ring, nu, nv, surf, center, label, push=None):
        """Close a ring of 2*(nu+nv) verts with an nu x nv quad grid (no
        triangle fans / n-gons). Interior verts: Coons-patch interpolation of
        the boundary, then pushed onto the surface by a ray from `center`."""
        assert len(ring) == 2 * (nu + nv)
        B = [self.V[i] for i in ring]
        # boundary walk: bottom (u 0->nu), right (v 0->nv), top (u nu->0), left (v nv->0)
        grid = {}
        k = 0
        for u in range(nu):
            grid[(u, 0)] = ring[k]; k += 1
        for v in range(nv):
            grid[(nu, v)] = ring[k]; k += 1
        for u in range(nu, 0, -1):
            grid[(u, nv)] = ring[k]; k += 1
        for v in range(nv, 0, -1):
            grid[(0, v)] = ring[k]; k += 1
        P = lambda u, v: self.V[grid[(u, v)]]
        for u in range(1, nu):
            for v in range(1, nv):
                s, t = u / nu, v / nv
                c = ((1 - t) * P(u, 0) + t * P(u, nv) + (1 - s) * P(0, v) + s * P(nu, v)
                     - ((1 - s) * (1 - t) * P(0, 0) + s * (1 - t) * P(nu, 0)
                        + (1 - s) * t * P(0, nv) + s * t * P(nu, nv)))
                h = surf.ray(center, c - np.asarray(center), 0.4)
                grid[(u, v)] = self.v(h if h is not None else c, label)
        for u in range(nu):
            for v in range(nv):
                self.F.append((grid[(u, v)], grid[(u + 1, v)], grid[(u + 1, v + 1)], grid[(u, v + 1)]))
        return grid


# ---------------------------------------------------------------------------
# Layout tables
# ---------------------------------------------------------------------------

# azimuths (deg from front toward the character's left) of the 16 trunk slots
TORSO_AZ = [0, 22, 44, 62, 80, 100, 118, 145, 180]
HEAD_AZ = [0, 8, 19, 30, 48, 75, 110, 145, 180]


def az16(half):
    return [math.radians(a) for a in half] + [math.radians(-a) for a in half[-2:0:-1]]


# trunk rows are placed on landmarks (see trunk_rows()); indices below refer
# to that list: crotch .. armpit (0-10), arm hole (11-13), neck (14-16),
# face (17-22), skull (23-25)
TORSO_FRACS = [0.0, 0.078, 0.148, 0.218, 0.280, 0.357, 0.439, 0.550, 0.680, 0.811, 0.931]


def trunk_rows():
    J = LM["J"]
    crotch = LM.get("crotch_z", 0.845)
    S_z = float(J["upperarm_l"][2])
    armpit = S_z - 0.084
    rows = [crotch + (armpit - crotch) * f for f in TORSO_FRACS]
    rows += [armpit, S_z - 0.028, S_z + 0.026]
    nb, cz = LM["neck_base_z"], LM["chin_z"]
    st = float(LM["stomion"][2])
    nose = float(LM["nose_tip"][2])
    ez = float(LM["eye_l"][2])
    r14 = max(nb + 0.03, S_z + 0.05)
    r16 = cz + 0.002
    rows += [r14, 0.5 * (r14 + r16), r16]
    rows += [0.5 * (cz + st) + 0.002, st, nose - 0.012]
    rows += [0.5 * (nose - 0.012 + ez - 0.012), ez - 0.012, ez + 0.014]
    rows += [ez + 0.038, ez + 0.064, ez + 0.088]
    return rows
ARM_HOLE_ROWS = (11, 12, 13)      # 1.362 / 1.418 / 1.472
ARM_HOLE_COLS = (3, 4, 5, 6)      # 62..118 deg
EYE_ROWS = (21, 22)               # 1.672 -> 1.698
EYE_COLS = (1, 2, 3)              # 8 -> 19 -> 30 deg
MOUTH_ROWS = (17, 18, 19)         # 1.589 -> 1.609 (lip line) -> 1.631
MOUTH_COLS = (14, 15, 0, 1, 2)    # -19 .. 19 deg


_CENTERS = {}


def trunk_center(z, surf=None):
    """Ray origin for a trunk ring: the centroid of the body's cross-section
    at that height (found by casting rays from a first guess on the spine
    axis — joint helpers sit near the back, not in the middle of the section)."""
    key = round(float(z), 5)
    if key in _CENTERS:
        return _CENTERS[key].copy()
    J = LM["J"]
    pts = [J["pelvis"], J["spine_01"], J["spine_02"], J["spine_03"], J["neck_01"], J["head"],
           np.asarray(LM["head_c"])]
    zs = np.array([p[2] for p in pts])
    ys = np.array([p[1] for p in pts])
    o = np.array([0.0, float(np.interp(z, zs, ys)), z])
    if surf is None:
        return o
    for _ in range(2):
        hits = []
        for k in range(24):
            a = 2 * math.pi * k / 24
            h = surf.ray(o, (math.sin(a), -math.cos(a), 0.0), 0.35)
            if h is not None:
                hits.append(h)
        if len(hits) >= 12:
            c = np.mean(hits, axis=0)
            o = np.array([0.0, c[1], z])
    _CENTERS[key] = o
    return o.copy()


def trunk_blend(z):
    rows = trunk_rows()
    return float(np.clip((z - rows[14]) / (rows[16] - rows[14]), 0.0, 1.0))


def trunk_dir(z, az):
    # slightly downward rays near the crotch so they reach the underside
    return np.array([math.sin(az), -math.cos(az), 0.0])


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_cage(J, surf):
    C = Cage()
    TRUNK_ROWS = trunk_rows()
    _CENTERS.clear()
    nrow, ncol = len(TRUNK_ROWS), 16
    G = [[None] * ncol for _ in range(nrow)]
    hole_interior = {(12, 4), (12, 5)}
    hole_boundary = {(r, c) for r in ARM_HOLE_ROWS for c in ARM_HOLE_COLS} - hole_interior

    # ---- trunk rings (arm-hole slots filled in later) ----
    for r, z in enumerate(TRUNK_ROWS):
        b = trunk_blend(z)
        tor, hd = az16(TORSO_AZ), az16(HEAD_AZ)
        ctr = trunk_center(z, surf)
        for c in range(ncol):
            if (r, min(c, 16 - c) if c else 0) in hole_interior | hole_boundary and c != 0 and c != 8:
                continue  # left (c) and right (16-c) hole slots come from the arm rings
            az = tor[c] * (1 - b) + hd[c] * b
            d = trunk_dir(z, az)
            o = ctr.copy()
            if r == 0:
                d = nrm(d + np.array([0, 0, -0.15]))  # slight downward aim onto the hip/pubic curve
            h = surf.ray(o, d, 0.35)
            G[r][c] = C.v(h if h is not None else o + d * 0.1, TRUNK)

    # ---- arm rings (both sides), their first ring IS the torso hole boundary ----
    arm_rings = {}
    for side in (1, -1):
        ring0 = arm_first_ring(J, surf, side)
        bnd_order = [(13, 3), (13, 4), (13, 5), (13, 6), (12, 6), (11, 6), (11, 5), (11, 4), (11, 3), (12, 3)]
        ids = []
        for k, (r, c) in enumerate(bnd_order):
            cc = c if side > 0 else (16 - c) % 16
            G[r][cc] = C.v(ring0[k], lab(ARM, side))
            ids.append(G[r][cc])
        arm_rings[side] = ids

    # ---- trunk faces (skip the arm holes) ----
    hole_faces = {(11, 3), (11, 4), (11, 5), (12, 3), (12, 4), (12, 5)}
    for r in range(nrow - 1):
        for c in range(ncol):
            c2 = (c + 1) % ncol
            if (r, c) in hole_faces or (r, (16 - c2) % 16) in hole_faces:
                continue
            C.F.append((G[r][c], G[r][c2], G[r + 1][c2], G[r + 1][c]))
    # crown cap (4x4 grid) from the last ring; ring walk must start at a grid corner
    top = G[nrow - 1]
    walk = top[14:] + top[:14]  # start 2 slots right of front so corners fall at +-45 deg
    cap = C.grid_cap(walk, 4, 4, surf, trunk_center(float(LM["eye_l"][2]) + 0.02, surf), TRUNK)

    # ---- UV seams on the trunk ----
    # head/neck island: cut around the neck base and up the back of the head
    # to the middle of the crown (walk index of slot 8 lands on grid (2,4))
    C.seam_chain(G[14], closed=True)
    C.seam_chain([G[r][8] for r in range(14, nrow)] + [cap[(2, 4)], cap[(2, 3)], cap[(2, 2)]])
    # torso: front / back split along the sides, from the crotch ring to the armpit
    for c in (5, 11):
        C.seam_chain([G[r][c] for r in range(0, 12)])
    for side in (1, -1):
        C.seam_chain(arm_rings[side], closed=True)

    # ---- legs from the crotch ring ----
    crotch_c = surf.ray(np.array([0, 0.005, 0.95]), (0, 0, -1), 0.3)
    crotch = C.v(crotch_c, TRUNK)
    for side in (1, -1):
        if side > 0:
            first = [G[0][c] for c in range(0, 9)] + [crotch]
        else:
            first = [G[0][0], crotch] + [G[0][c] for c in range(8, 16)]
        C.seam_chain(first, closed=True)
        build_leg(C, J, surf, side, first)

    # ---- arms + hands ----
    for side in (1, -1):
        build_arm(C, J, surf, side, arm_rings[side])

    return C, G


def arm_first_ring(J, surf, side):
    """The shoulder ring: a plane steeper than perpendicular-to-the-arm, so
    it runs over the top of the deltoid and down through the armpit."""
    S = mirror(J["upperarm_l"], side)
    E = mirror(J["lowerarm_l"], side)
    au = nrm(E - S)
    n = nrm(au + np.array([side * 1.0, 0, 0]))
    ctr = S - au * 0.030
    up = nrm(np.array([0, 0, 1.0]) - n * n[2])
    back = nrm(np.cross(n, up)) * (1 if side > 0 else -1)
    if back[1] < 0:
        back = -back
    pts = []
    for k in range(10):
        phi = math.radians(-54 + 36 * k)
        d = math.cos(phi) * up + math.sin(phi) * back
        h = surf.ray(ctr, d, 0.25)
        pts.append(h if h is not None else ctr + d * 0.05)
    return pts


def limb_frames(axis, up_hint, back_hint):
    up = nrm(up_hint - axis * np.dot(up_hint, axis))
    back = nrm(np.cross(axis, up))
    if np.dot(back, back_hint) < 0:
        back = -back
    return up, back


def build_arm(C, J, surf, side, ring0):
    S = mirror(J["upperarm_l"], side)
    E = mirror(J["lowerarm_l"], side)
    W = mirror(J["hand_l"], side)
    ax_h = mirror_vec(J["_hand_axis_l"], side)
    th = mirror_vec(J["_thumb_dir_l"], side)
    pn = mirror_vec(J["_palm_n_l"], side)
    L_up = np.linalg.norm(E - S)
    L_lo = np.linalg.norm(W - E)
    # rows: fractions of the upper arm (u) / forearm (l), bracketing the elbow
    fr = [("u", 0.12), ("u", 0.29), ("u", 0.48), ("u", 0.67), ("u", 0.83), ("u", 0.94), ("u", 1.0),
          ("l", 0.07), ("l", 0.19), ("l", 0.40), ("l", 0.61), ("l", 0.80), ("l", 0.94), ("l", 1.0)]
    rows = [t * L_up if seg == "u" else L_up + t * L_lo for seg, t in fr]
    prev = ring0
    label = lab(ARM, side)
    under = [ring0[7]]
    for s in rows:
        if s <= L_up:
            p = S + (E - S) * (s / L_up)
            axis = nrm(E - S) if s < L_up - 0.02 else nrm(W - S)
        else:
            p = E + (W - E) * ((s - L_up) / L_lo)
            axis = nrm(W - E)
        # frame: blends from "up" at the shoulder to "dorsal" at the wrist,
        # plus the 18-degree slot rotation the palm ring needs
        f = np.clip((s - L_up) / L_lo, 0.0, 1.0)
        up_hint = nrm((1 - f) * np.array([0, 0, 1.0]) + f * (-pn) * 1.0)
        up, back = limb_frames(axis, up_hint, np.array([0, 1.0, 0]))
        rot = -18.0 * f
        dirs = [math.cos(math.radians(-54 + 36 * k + rot)) * up + math.sin(math.radians(-54 + 36 * k + rot)) * back
                for k in range(10)]
        ring = C.ring(surf.cast_ring(p, dirs, 0.035), label)
        C.bridge(prev, ring)
        prev = ring
        under.append(ring[7])
    C.seam_chain(under)                 # along the underside of the arm
    C.seam_chain(prev, closed=True)     # wrist
    C.loops.setdefault("wrist", []).append(prev)
    build_hand(C, J, surf, side, prev)


def build_hand(C, J, surf, side, wrist):
    """wrist ring slots: d0..d4 (dorsal, thumb->pinky), p4..p0 (palmar, pinky->thumb)."""
    W = mirror(J["hand_l"], side)
    ax = mirror_vec(J["_hand_axis_l"], side)
    th = mirror_vec(J["_thumb_dir_l"], side)
    pn = mirror_vec(J["_palm_n_l"], side)
    chains = humanoid.hand_chains(J)
    hl = lab(HAND, side)
    b_i = mirror(chains["index"][0][0], side)
    b_p = mirror(chains["pinky"][0][0], side)
    half = 0.5 * abs(float(np.dot(b_i - b_p, th))) + chains["index"][1][0]
    widths = [half, half * 0.5, 0.0, -half * 0.5, -half * 0.97]

    def palm_ring(t, wscale, h):
        ctr = W + ax * t
        pts = []
        for i in range(5):
            tgt = ctr + th * widths[i] * wscale - pn * h
            pts.append(tgt)
        for i in reversed(range(5)):
            tgt = ctr + th * widths[i] * wscale + pn * h
            pts.append(tgt)
        out = []
        for q in pts:
            hit = surf.ray(ctr, q - ctr, 0.08)
            out.append(hit if hit is not None else q)
        return out

    palm_len = float(np.dot(mirror(chains["middle"][0][0], side) - W, ax))
    mid = C.ring(palm_ring(palm_len * 0.5, 1.0, 0.012), hl)
    # knuckle ring from the finger bases: separators between neighbors,
    # outer edges one finger-radius past index / pinky
    bases = [mirror(chains[nm][0][0], side) for nm in FINGER_NAMES[:4]]
    radii = [chains[nm][1][0] for nm in FINGER_NAMES[:4]]
    seps = [bases[0] + th * radii[0] * 1.05]
    seps += [(bases[i] + bases[i + 1]) / 2 for i in range(3)]
    seps += [bases[3] - th * radii[3] * 1.05]
    hk = [0.0095, 0.0105, 0.0105, 0.0100, 0.0085]
    knk_pts = [seps[i] - pn * hk[i] for i in range(5)] + [seps[i] + pn * hk[i] * 0.95 for i in reversed(range(5))]
    knk = C.ring(knk_pts, hl)
    # wrist->mid faces except the thumb-side one (the thumb grows from it)
    n = 10
    thumb_face = None
    for i in range(n):
        j = (i + 1) % n
        f = (wrist[i], wrist[j], mid[j], mid[i])
        if i == 9:  # edge p0 -> d0 is the thumb side (slot 9 -> slot 0)
            thumb_face = f
            continue
        C.F.append(f)
    C.bridge(mid, knk)
    # palm island: cut along the palm's two side edges
    C.seam_chain([wrist[5], mid[5], knk[5]])
    C.seam_chain([wrist[9], mid[9], knk[9]])
    d = knk[:5]
    p = list(reversed(knk[5:]))  # p0..p4
    # fingers: ring (d_i, d_i+1, p_i+1, p_i) with thumb side first
    for fi, name in enumerate(FINGER_NAMES[:4]):
        pts, rad = chains[name]
        pts = [mirror(q, side) for q in pts]
        base = [d[fi], d[fi + 1], p[fi + 1], p[fi]]
        rows = [(0, 0.5), (0, 1.0), (1, 1.0), (2, 0.72)]
        prev = base
        fl = lab(FINGER, side, fi)
        palm_a, palm_b = [base[2]], [base[3]]
        for seg, t in rows:
            c = pts[seg] + (pts[seg + 1] - pts[seg]) * t
            axis = nrm(pts[min(seg + 1, 3)] - pts[seg])
            dors, pinky = limb_frames(axis, -pn, -th)
            dirs = [nrm(dors - pinky), nrm(dors + pinky), nrm(-dors + pinky), nrm(-dors - pinky)]
            # base ring order is thumb-side first: (d_i thumb-side), (d_i+1 pinky-side)...
            dirs = [nrm(dors + (-1) * pinky), nrm(dors + pinky), nrm(-dors + pinky), nrm(-dors - pinky)]
            ring = C.ring(surf.cast_ring(c, dirs, rad[seg]), fl)
            C.bridge(prev, ring)
            prev = ring
            palm_a.append(ring[2])
            palm_b.append(ring[3])
        C.F.append(tuple(prev))  # fingertip cap
        # palmar strip of the finger joins the palm island
        C.seam_chain(palm_a + [palm_b[-1]] + palm_b[-2::-1])
    # thumb from the side face: (p0_w, d0_w, d0_m, p0_m) order -> ring
    pts, rad = chains["thumb"]
    pts = [mirror(q, side) for q in pts]
    w_p0, w_d0, m_d0, m_p0 = thumb_face[0], thumb_face[1], thumb_face[2], thumb_face[3]
    base = [w_d0, m_d0, m_p0, w_p0]
    prev = base
    tl = lab(FINGER, side, 4)
    tha, thb = [base[2]], [base[3]]
    for seg, t in [(0, 0.55), (1, 0.0), (1, 0.6), (2, 0.0), (2, 0.7)]:
        c = pts[seg] + (pts[seg + 1] - pts[seg]) * t
        axis = nrm(pts[seg + 1] - pts[seg])
        dors, toward_f = limb_frames(axis, -pn, ax)
        dirs = [nrm(dors - toward_f), nrm(dors + toward_f), nrm(-dors + toward_f), nrm(-dors - toward_f)]
        ring = C.ring(surf.cast_ring(c, dirs, rad[seg]), tl)
        C.bridge(prev, ring)
        prev = ring
        tha.append(ring[2])
        thb.append(ring[3])
    C.F.append(tuple(prev))
    C.seam_chain(tha + [thb[-1]] + thb[-2::-1])


def build_leg(C, J, surf, side, first):
    H = mirror(J["thigh_l"], side)
    K = mirror(J["calf_l"], side)
    A = mirror(J["foot_l"], side)
    B = mirror(J["ball_l"], side)
    label = lab(LEG, side)
    # slot directions from the first ring, measured around the thigh axis
    axis0 = nrm(K - H)
    ref_up, ref_back = limb_frames(axis0, np.array([0, -1.0, 0]), np.array([side * 1.0, 0, 0]))
    c0 = H + axis0 * ((H[2] - 0.845) / max(1e-6, -axis0[2] * 1.0)) * 1.0
    angs = []
    for i in first:
        v = C.V[i] - c0
        v = v - axis0 * np.dot(v, axis0)
        angs.append(math.atan2(np.dot(v, ref_back), np.dot(v, ref_up)))
    # rows down the leg (z) then the foot path
    # thigh rows between the crotch and the knee, 2 loops bracketing the knee,
    # shin, the boot top loop pair (hem of the boots), ankle
    crotch = LM.get("crotch_z", 0.845)
    kz, az = float(K[2]), float(A[2])
    top, kt = crotch - 0.045, kz + 0.043
    zrows = [top + (kt - top) * f for f in (0.0, 0.203, 0.439, 0.676, 0.851)]
    zrows += [kz + 0.007, kz - 0.027]
    zrows += [kz - 0.075, kz - 0.125, 0.346, 0.326]
    zrows += [0.326 + (az + 0.031 - 0.326) * f for f in (0.254, 0.550, 0.823, 1.0)]
    prev = first
    frame_up, frame_back = ref_up, ref_back
    inner_slot = 9 if side > 0 else 1   # the crotch vertex's slot = inner line of the leg
    inner = [first[inner_slot]]
    for z in zrows:
        if z >= K[2]:
            t = (H[2] - z) / (H[2] - K[2])
            p = H + (K - H) * t
            axis = nrm(K - H) if z > K[2] + 0.03 else nrm(A - H)
        else:
            t = (K[2] - z) / (K[2] - A[2])
            p = K + (A - K) * t
            axis = nrm(A - K)
        up, back = limb_frames(axis, frame_up, frame_back)
        dirs = [math.cos(a) * up + math.sin(a) * back for a in angs]
        ring = C.ring(surf.cast_ring(p, dirs, 0.05), label)
        C.bridge(prev, ring)
        prev = ring
        inner.append(ring[inner_slot])
        if abs(z - 0.346) < 1e-6:
            C.seam_chain(ring, closed=True)   # boot top
            C.loops.setdefault("boot_top", []).append(ring)
        frame_up, frame_back = up, back
    # foot: ring planes rotate from horizontal to facing forward
    foot = [
        (A + np.array([0, 0.004, -0.030]), nrm(np.array([0, -0.35, -1.0]))),
        (A + np.array([0, -0.030, -0.045]), nrm(np.array([0, -1.0, -0.9]))),
        (np.array([B[0], B[1] + 0.075, 0.045]), nrm(np.array([0, -1.0, -0.25]))),
        (np.array([B[0], B[1] + 0.015, 0.040]), np.array([0, -1.0, 0.0])),
        (np.array([B[0], B[1] - 0.045, 0.036]), nrm(np.array([0, -1.0, 0.05]))),
        (np.array([B[0], B[1] - 0.085, 0.032]), nrm(np.array([0, -1.0, 0.15]))),
    ]
    for p, axis in foot:
        up, back = limb_frames(axis, frame_up, frame_back)
        dirs = [math.cos(a) * up + math.sin(a) * back for a in angs]
        ring = C.ring(surf.cast_ring(p, dirs, 0.04), label)
        C.bridge(prev, ring)
        prev = ring
        inner.append(ring[inner_slot])
        frame_up, frame_back = up, back
    C.seam_chain(inner)
    # toe cap: 3 x 2 grid. slot walk must start at a corner: slot 0 is the
    # top-inner corner for the left foot
    # (mirrored slot order on the right leg puts the crotch vertex at slot 1)
    walk = prev if side > 0 else prev[2:] + prev[:2]
    C.grid_cap(walk, 3, 2, surf, np.array([B[0], B[1] - 0.03, 0.035]), label)


# ---------------------------------------------------------------------------
# To Blender, face loops, subdivision, projection + relax
# ---------------------------------------------------------------------------

def cage_to_bmesh(C, G):
    bm = bmesh.new()
    # create the layer first: adding a custom-data layer reallocates vertex
    # storage and invalidates every BMVert reference taken before it
    lay = bm.verts.layers.int.new("part")
    vs = [bm.verts.new(Vector(p)) for p in C.V]
    for v, l in zip(vs, C.L):
        v[lay] = l
    bm.verts.ensure_lookup_table()
    face_of = {}
    for f in C.F:
        try:
            bf = bm.faces.new([vs[i] for i in f])
        except ValueError:
            continue  # duplicate face
        face_of[tuple(sorted(f))] = bf
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for a, b in C.seams:
        e = bm.edges.get((vs[a], vs[b]))
        if e is not None:
            e.seam = True

    def trunk_face(r, c):
        c2 = (c + 1) % 16
        key = tuple(sorted((G[r][c], G[r][c2], G[r + 1][c2], G[r + 1][c])))
        return face_of.get(key)

    eyes, mouth = [], []
    for side in (1, -1):
        cols = EYE_COLS if side > 0 else [(16 - c) % 16 for c in EYE_COLS]
        cols = sorted(cols)
        f = [trunk_face(EYE_ROWS[0], c) for c in cols[:-1]]
        eyes.append([x for x in f if x])
    for r in MOUTH_ROWS[:-1]:
        for c in MOUTH_COLS[:-1]:
            fm = trunk_face(r, c)
            if fm:
                mouth.append(fm)
    return bm, eyes, mouth


def _boundary_loop(edges):
    """Order the verts of one closed boundary loop."""
    adj = {}
    for e in edges:
        a, b = e.verts
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = next(iter(adj))
    loop, prev, cur = [start], None, start
    while True:
        nxt = [v for v in adj[cur] if v is not prev]
        if not nxt or nxt[0] is start:
            break
        prev, cur = cur, nxt[0]
        loop.append(cur)
    return loop


def add_face_loops(bm, eyes, mouth):
    """Insets = concentric edge loops. Eyes get two; their inner faces are
    replaced by an eye-socket "bag": the lid margin loop is extruded twice
    into the head behind the eyeball and capped, so the skin mesh stays
    watertight (no see-through gap between lid and eyeball in an engine
    with backface culling). The mouth gets one ring around the lips."""
    depth_l = bm.verts.layers.float.new("eyedepth")
    side_l = bm.verts.layers.float.new("eyeside")
    for faces in eyes:
        side = 1.0 if np.mean([f.calc_center_median().x for f in faces]) > 0 else -1.0
        bmesh.ops.inset_region(bm, faces=faces, thickness=0.0045, depth=0.0, use_even_offset=True)
        bmesh.ops.inset_region(bm, faces=faces, thickness=0.0030, depth=0.0, use_even_offset=True)
        # "FACES" also removes the inner edge shared by the deleted faces;
        # FACES_ONLY left it as a wire edge -> loose verts after subdivision
        bmesh.ops.delete(bm, geom=faces, context="FACES")
        bnd = [e for e in bm.edges if e.is_boundary]
        # this eye's hole only (the other eye may already be closed or not)
        bnd = [e for e in bnd if (e.verts[0].co.x > 0) == (side > 0)]
        ring = _boundary_loop(bnd)
        for v in ring:
            v[side_l] = side
        for d in (1.0, 2.0):
            edges = [e for e in bm.edges if e.is_boundary and (e.verts[0].co.x > 0) == (side > 0)]
            ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
            for g in ret["geom"]:
                if isinstance(g, bmesh.types.BMVert):
                    g[depth_l] = d
                    g[side_l] = side
        edges = [e for e in bm.edges if e.is_boundary and (e.verts[0].co.x > 0) == (side > 0)]
        last = _boundary_loop(edges)
        n = len(last)
        if n % 2 == 0 and n >= 4:  # close with a strip of quads, no n-gon
            h = n // 2
            for i in range(h - 1):
                bm.faces.new([last[i], last[i + 1], last[n - 2 - i], last[n - 1 - i]])
        else:
            bm.faces.new(last)
    bmesh.ops.inset_region(bm, faces=mouth, thickness=0.0050, depth=0.0, use_even_offset=True)
    return bm


def subdivide(obj, levels=1):
    m = obj.modifiers.new("Sub", "SUBSURF")
    m.levels = levels
    m.render_levels = levels
    m.subdivision_type = "CATMULL_CLARK"
    m.uv_smooth = "PRESERVE_BOUNDARIES"
    m.boundary_smooth = "PRESERVE_CORNERS"
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    bpy.ops.object.modifier_apply(modifier=m.name)


def axis_polylines(J):
    """Per-label bone polylines used as projection anchors."""
    P = {}
    ez = float(LM["eye_l"][2])
    zs = [1.10, 1.30, LM["neck_base_z"] + 0.01, LM["chin_z"] + 0.03, ez + 0.01, ez + 0.05]
    P[TRUNK] = [np.array([0, trunk_center(0.95)[1], 0.95])] + [trunk_center(z) for z in zs]
    chains = humanoid.hand_chains(J)
    for side in (1, -1):
        S, E, W = (mirror(J[n], side) for n in ("upperarm_l", "lowerarm_l", "hand_l"))
        ax = mirror_vec(J["_hand_axis_l"], side)
        P[lab(ARM, side)] = [S, E, W]
        P[lab(HAND, side)] = [W, W + ax * 0.09]
        H, K, A, B, T = (mirror(J[n], side) for n in ("thigh_l", "calf_l", "foot_l", "ball_l", "toe_l"))
        P[lab(LEG, side)] = [H, K, A + np.array([0, 0, -0.02]), B + np.array([0, 0.02, 0.018]),
                             T + np.array([0, 0.03, 0.02])]
        for fi, name in enumerate(FINGER_NAMES):
            pts, _r = chains[name]
            P[lab(FINGER, side, fi)] = [mirror(q, side) for q in pts]
    return P


def closest_on_polyline(p, pts):
    best, bd = None, 1e9
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        q = a + ab * t
        d = np.linalg.norm(p - q)
        if d < bd:
            best, bd = q, d
    return best


def finger_surfaces(J):
    """label -> list of (a, b, ra, rb) round-cone segments of each finger.
    Fingers are the one place a mesh-based snap is unreliable (gaps of a few
    mm between neighbors), but their sculpted form *is* these round cones, so
    finger verts are projected onto them analytically."""
    chains = humanoid.hand_chains(J)
    out = {}
    for side in (1, -1):
        for fi, name in enumerate(FINGER_NAMES):
            pts, rad = chains[name]
            pts = [mirror(q, side) for q in pts]
            segs = []
            for i in range(3):
                rb = rad[i] * 0.88 if name == "thumb" else rad[i] * 0.9
                segs.append((pts[i], pts[i + 1], rad[i], rb))
            out[lab(FINGER, side, fi)] = segs
    return out


def project_round_cones(p, segs, bulge=0.0006):
    best, bd = None, 1e9
    for a, b, ra, rb in segs:
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        c = a + ab * t
        r = ra + (rb - ra) * t + bulge
        d = p - c
        n = np.linalg.norm(d)
        q = c + (d / n if n > 1e-9 else np.array([0, 0, 1.0])) * r
        dist = abs(n - r)
        if dist < bd:
            best, bd = q, dist
    return best


def project_and_relax(obj, surf, J, iters=6, pinned=None, relax=0.45):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    lay = bm.verts.layers.int.get("part")
    polys = axis_polylines(J)
    labels = np.array([v[lay] for v in bm.verts])
    nbrs = [[e.other_vert(v).index for e in v.link_edges] for v in bm.verts]
    # everything sharing a face (incl. quad diagonals, which become edges once
    # the mesh is triangulated)
    fnbrs = [sorted({u.index for f in v.link_faces for u in f.verts} - {v.index}) for v in bm.verts]
    boundary = np.array([v.is_boundary for v in bm.verts])
    margin, bag, _d, _s = eye_masks(obj)
    boundary = boundary | margin | bag
    pinned = boundary if pinned is None else (pinned | boundary)
    co = np.array([v.co[:] for v in bm.verts])
    anchors = np.array([closest_on_polyline(co[i], polys.get(labels[i], polys[TRUNK])) for i in range(len(co))])
    fsurf = finger_surfaces(J)
    kind = labels // 10 * 10
    # snap mode per vertex: 0 radial-from-bone, 1 nearest point, 2 analytic finger
    mode = np.zeros(len(co), dtype=int)
    mode[kind == FINGER] = 1
    mode[kind == HAND] = 1
    crotch = ((kind == TRUNK) & (co[:, 2] < 0.93)) | ((kind == LEG) & (co[:, 2] > 0.76))
    mode[crotch] = 1

    def project(co):
        out = co.copy()
        for i in range(len(co)):
            if pinned[i]:
                continue
            if mode[i] == 2:
                out[i] = project_round_cones(co[i], fsurf[labels[i]])
                continue
            if mode[i] == 1:
                out[i] = surf.nearest(co[i])
                continue
            a = anchors[i]
            d = co[i] - a
            h = surf.ray(a, d, 0.4) if np.linalg.norm(d) > 1e-4 else None
            if h is None or np.linalg.norm(h - co[i]) > 0.02:
                h = surf.nearest(co[i])
            out[i] = h
        return out

    co = project(co)
    for it in range(iters):
        # vertex normals from the current positions
        for v, p in zip(bm.verts, co):
            v.co = Vector(p)
        bm.normal_update()
        nor = np.array([v.normal[:] for v in bm.verts])
        avg = np.array([co[n].mean(axis=0) if n else co[i] for i, n in enumerate(nbrs)])
        delta = avg - co
        delta -= nor * np.sum(delta * nor, axis=1, keepdims=True)
        co = co + np.where(pinned[:, None], 0.0, relax * delta)
        co = project(co)
    # spike pass: a vertex that ended up far off its neighbors' surface
    # (a ray that dove into a nostril or ear canal) is pulled back to the
    # neighbor average — those cavities belong in the normal map, not the mesh
    for _ in range(2):
        for v, p in zip(bm.verts, co):
            v.co = Vector(p)
        bm.normal_update()
        nor = np.array([v.normal[:] for v in bm.verts])
        avg = np.array([co[n].mean(axis=0) if n else co[i] for i, n in enumerate(nbrs)])
        elen = np.array([np.mean(np.linalg.norm(co[n] - co[i], axis=1)) if n else 1.0 for i, n in enumerate(nbrs)])
        off = np.abs(np.sum((co - avg) * nor, axis=1))
        spike = (off > 0.45 * elen) & ~pinned
        # collapse guard: nearest-point snaps onto a sharp convex edge (the
        # pouch corners) can stack two verts on one point -> degenerate faces
        dmin = np.array([np.min(np.linalg.norm(co[n] - co[i], axis=1)) if n else 1.0 for i, n in enumerate(fnbrs)])
        stacked = (dmin < 0.15 * elen) & ~pinned
        spike |= stacked
        co[spike] = avg[spike]
    for v, p in zip(bm.verts, co):
        v.co = Vector(p)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return int(spike.sum())


def _float_attr(me, name):
    if name not in me.attributes:
        return np.zeros(len(me.vertices))
    a = np.zeros(len(me.vertices), dtype=np.float32)
    me.attributes[name].data.foreach_get("value", a)
    return a


def eye_masks(obj):
    """(margin verts, bag verts, side) from the subdivision-interpolated attributes."""
    me = obj.data
    depth = _float_attr(me, "eyedepth")
    side = _float_attr(me, "eyeside")
    eye = np.abs(side) > 0.99
    margin = eye & (depth < 0.01)
    bag = (depth > 0.01)
    return margin, bag, depth, side


def seat_eye_holes(obj):
    """Snap each lid-margin loop onto the lid margin measured on the sculpt
    (landmarks.eye_margin), evenly re-spaced so no two verts collapse, and
    lay the socket bag behind the eyeball: each bag ring shrinks toward the
    eye center and steps back into the head."""
    import landmarks
    me = obj.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    margin, bag, depth, _s = eye_masks(obj)
    for side in (1, -1):
        c = np.array(LM["eye_l"], dtype=np.float64)
        c[0] *= side
        mg = LM["eye_margin_l" if side > 0 else "eye_margin_r"]
        near = np.linalg.norm(co - c, axis=1) < 0.05
        idx = np.nonzero(margin & near & (np.sign(co[:, 0]) == side))[0]
        if len(idx):
            rel = co[idx] - c
            psis = np.arctan2(rel[:, 2], rel[:, 0])
            order = np.argsort(psis)
            n = len(idx)
            even = psis[order[0]] + np.arange(n) * 2 * math.pi / n
            shift = np.angle(np.mean(np.exp(1j * (psis[order] - even))))
            for k, oi in enumerate(order):
                co[idx[oi]] = landmarks.margin_point(mg, even[k] + shift)
        bidx = np.nonzero(bag & near & (np.sign(co[:, 0]) == side))[0]
        r = float(LM["eye_r_radius"])
        for i in bidx:
            rel = co[i] - c
            psi = math.atan2(rel[2], rel[0])
            m = landmarks.margin_point(mg, psi)
            d = float(depth[i])
            # behind the eye: shrink toward the axis, push back along +Y
            co[i] = c + (m - c) * max(0.15, 1.0 - 0.30 * d) + np.array([0, 1.0, 0]) * (0.35 * r * d)
    me.vertices.foreach_set("co", co.astype(np.float32).reshape(-1))
    me.update()


def relabel_from_cage(obj, C):
    """Subsurf interpolates integer attributes (a vertex between an arm label
    and a trunk label would get a meaningless in-between id), so labels are
    re-derived from the nearest cage vertex."""
    kd = KDTree(len(C.V))
    for i, p in enumerate(C.V):
        kd.insert(Vector(p), i)
    kd.balance()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    lay = bm.verts.layers.int.get("part") or bm.verts.layers.int.new("part")
    for v in bm.verts:
        _co, idx, _d = kd.find(v.co)
        v[lay] = C.L[idx]
    bm.to_mesh(obj.data)
    bm.free()


def pouch_cage(name="SK_Character_Pouch"):
    """Low-poly for the belt pouch: a subdivided box around its sculpted
    frame (flap and strap loop are normal-map detail)."""
    import costume
    c, R, half = costume.GEAR["pouch_frame"]
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
    M = np.asarray(R, dtype=np.float64)
    for v in bm.verts:
        p = np.array(v.co) * (np.asarray(half) * 0.98)
        v.co = Vector(M @ p + np.asarray(c, dtype=np.float64))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def project_nearest(obj, surf, iters=3, relax=0.4):
    """Plain snap + relax (for hard-surface pieces: nearest point is right)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    nbrs = [[e.other_vert(v).index for e in v.link_edges] for v in bm.verts]
    co = np.array([v.co[:] for v in bm.verts])
    co = np.array([surf.nearest(p) for p in co])
    for _ in range(iters):
        for v, p in zip(bm.verts, co):
            v.co = Vector(p)
        bm.normal_update()
        nor = np.array([v.normal[:] for v in bm.verts])
        avg = np.array([co[n].mean(axis=0) for n in nbrs])
        d = avg - co
        d -= nor * np.sum(d * nor, axis=1, keepdims=True)
        co = np.array([surf.nearest(p) for p in co + relax * d])
    for v, p in zip(bm.verts, co):
        v.co = Vector(p)
    bm.to_mesh(obj.data)
    bm.free()


def build_lods(J, hp_obj, name="SK_Character", iters=(6, 4), hp_pouch=None):
    """Cage -> UVs -> subdivision LODs.

    Returns (lod4_cage, lod2, lod0): the cage itself is the lowest clean LOD,
    one Catmull-Clark level (re-projected + relaxed) is LOD2, two levels is
    LOD0. All three share the cage's UV layout (subdivision interpolates UVs),
    so one texture set / one bake serves every LOD; LOD1 and LOD3 are
    decimated in-betweens made by the LOD stage.
    """
    import uvs
    surf = Surface(hp_obj)
    C, G = build_cage(J, surf)
    bm, eyes, mouth = cage_to_bmesh(C, G)
    add_face_loops(bm, eyes, mouth)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(f"{name}_LOD4")
    bm.to_mesh(me)
    bm.free()
    cage = bpy.data.objects.new(f"{name}_LOD4", me)
    bpy.context.scene.collection.objects.link(cage)
    seat_eye_holes(cage)
    pouches = []
    if hp_pouch is not None:
        pc = pouch_cage(f"{name}_Pouch_LOD4")
        psurf = Surface(hp_pouch)
        project_nearest(pc, psurf, iters=1)
        pouches.append(pc)
    uvs.unwrap(cage, extras=pouches)
    if pouches:
        prev = pouches[0]
        for level in ("LOD2", "LOD0"):
            o = prev.copy()
            o.data = prev.data.copy()
            o.name = o.data.name = f"{name}_Pouch_{level}"
            bpy.context.scene.collection.objects.link(o)
            subdivide(o, 1)
            project_nearest(o, psurf, iters=3)
            pouches.append(o)
            prev = o

    lods = [cage]
    prev = cage
    for level, it in zip(("LOD2", "LOD0"), iters):
        o = prev.copy()
        o.data = prev.data.copy()
        o.name = o.data.name = f"{name}_{level}"
        bpy.context.scene.collection.objects.link(o)
        subdivide(o, 1)
        relabel_from_cage(o, C)
        seat_eye_holes(o)
        project_and_relax(o, surf, J, iters=it)
        seat_eye_holes(o)
        lods.append(o)
        prev = o
    for o in lods:
        o.data.shade_smooth() if hasattr(o.data, "shade_smooth") else None
    return C, lods[0], lods[1], lods[2]
