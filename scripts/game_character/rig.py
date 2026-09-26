"""
rig — game deform skeleton, skinning and a test animation.

Skeleton: UE4/UE5-mannequin naming (root, pelvis, spine_01..03, neck_01, head,
clavicle/upperarm/lowerarm/hand, 3 phalanges x 5 fingers, thigh/calf/foot/ball,
upperarm/lowerarm/thigh/calf twist bones, eye bones). Matching those names
means Unreal's IK Retargeter and Unity's Humanoid avatar map it without
manual setup.

Bone rolls are chosen so every hinge joint flexes about its local X axis
(Z points "back"/dorsal): animators and procedural code can rely on
"X = bend" for elbows, knees, fingers, spine.

Skinning: Blender's bone-heat automatic weights on the main bones, then twist
bones take a linear share of their parent along the bone (the standard
candy-wrapper fix), then weights are cleaned up the way engines need them:
max 4 influences per vertex, tiny weights pruned, normalized to 1.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler

import humanoid

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _v(p):
    return Vector((float(p[0]), float(p[1]), float(p[2])))


def mirror(p, side):
    p = np.array(p, dtype=np.float64)
    if side < 0:
        p[0] = -p[0]
    return p


def bone_table(J):
    """name -> (head, tail, parent, z_axis_hint, deform)."""
    T = {}
    back = (0, 1, 0)
    T["root"] = ((0, 0, 0), (0, 0, 0.12), None, back, False)
    T["pelvis"] = (J["pelvis"], J["spine_01"], "root", back, True)
    T["spine_01"] = (J["spine_01"], J["spine_02"], "pelvis", back, True)
    T["spine_02"] = (J["spine_02"], J["spine_03"], "spine_01", back, True)
    T["spine_03"] = (J["spine_03"], J["neck_01"], "spine_02", back, True)
    T["neck_01"] = (J["neck_01"], J["head"], "spine_03", back, True)
    T["head"] = (J["head"], np.array([0, J["head"][1], J["head_top"][2] - 0.04]), "neck_01", back, True)
    chains = humanoid.hand_chains(J)
    for side, sfx in ((1, "_l"), (-1, "_r")):
        m = lambda n: mirror(J[n], side)
        pn = mirror(J["_palm_n_l"], side)
        T["clavicle" + sfx] = (m("clavicle_l"), m("upperarm_l"), "spine_03", back, True)
        T["upperarm" + sfx] = (m("upperarm_l"), m("lowerarm_l"), "clavicle" + sfx, back, True)
        T["lowerarm" + sfx] = (m("lowerarm_l"), m("hand_l"), "upperarm" + sfx, back, True)
        hand_tail = m("hand_l") + mirror(J["_hand_axis_l"], side) * 0.09
        T["hand" + sfx] = (m("hand_l"), hand_tail, "lowerarm" + sfx, tuple(-pn), True)
        S, E, W = m("upperarm_l"), m("lowerarm_l"), m("hand_l")
        T["upperarm_twist_01" + sfx] = (S + (E - S) * 0.5, S + (E - S) * 0.75, "upperarm" + sfx, back, True)
        T["lowerarm_twist_01" + sfx] = (E + (W - E) * 0.6, E + (W - E) * 0.85, "lowerarm" + sfx, back, True)
        for f in FINGERS:
            pts, _r = chains[f]
            pts = [mirror(p, side) for p in pts]
            parent = "hand" + sfx
            for i in range(3):
                nm = f"{f}_0{i + 1}{sfx}"
                T[nm] = (pts[i], pts[i + 1], parent, tuple(-pn), True)
                parent = nm
        H, K, A, B, Tt = m("thigh_l"), m("calf_l"), m("foot_l"), m("ball_l"), m("toe_l")
        T["thigh" + sfx] = (H, K, "pelvis", back, True)
        T["calf" + sfx] = (K, A, "thigh" + sfx, back, True)
        T["foot" + sfx] = (A, B, "calf" + sfx, (0, 0, 1), True)
        T["ball" + sfx] = (B, Tt, "foot" + sfx, (0, 0, 1), True)
        T["thigh_twist_01" + sfx] = (H + (K - H) * 0.5, H + (K - H) * 0.75, "thigh" + sfx, back, True)
        T["calf_twist_01" + sfx] = (K + (A - K) * 0.5, K + (A - K) * 0.75, "calf" + sfx, back, True)
        from landmarks import LM
        eye = mirror(LM["eye_l"], side)
        T["eye" + sfx] = (eye, eye + np.array([0, -0.02, 0]), "head", (0, 0, 1), True)
    return T


def build_armature(J, name="SK_Character_Skeleton"):
    arm = bpy.data.armatures.new(name)
    arm.display_type = "OCTAHEDRAL"
    obj = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    bpy.ops.object.mode_set(mode="EDIT")
    T = bone_table(J)
    for nm, (h, t, parent, zhint, deform) in T.items():
        eb = arm.edit_bones.new(nm)
        eb.head = _v(h)
        eb.tail = _v(t)
        eb.use_deform = deform
    for nm, (h, t, parent, zhint, deform) in T.items():
        eb = arm.edit_bones[nm]
        if parent:
            eb.parent = arm.edit_bones[parent]
            # connect when the child starts exactly at the parent's tail
            eb.use_connect = (eb.head - eb.parent.tail).length < 1e-5
        eb.align_roll(_v(zhint))
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


# ---------------------------------------------------------------------------
# Skinning
# ---------------------------------------------------------------------------

TWISTS = [("upperarm", "upperarm_twist_01", 0.25, 0.85), ("lowerarm", "lowerarm_twist_01", 0.35, 0.95),
          ("thigh", "thigh_twist_01", 0.25, 0.85), ("calf", "calf_twist_01", 0.30, 0.90)]


def candidate_bones(label):
    """Which bones may influence a vertex, from its retopo part label (see
    retopo.TRUNK/ARM/HAND/LEG/FINGER). This is what keeps heat from leaking
    between body parts that touch in space but not on the surface (inner
    thighs, arm vs. torso side) — the job Pinocchio's visibility test does."""
    kind, rest = label // 10 * 10, label % 10
    sfx = "_l" if rest < 5 else "_r"
    fi = rest % 5
    if kind == 0:
        return (["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head",
                 "clavicle_l", "clavicle_r", "thigh_l", "thigh_r", "upperarm_l", "upperarm_r"])
    if kind == 10:
        return ["spine_03", "clavicle" + sfx, "upperarm" + sfx, "lowerarm" + sfx, "hand" + sfx]
    if kind == 20:
        return ["lowerarm" + sfx, "hand" + sfx] + [f"{f}_01{sfx}" for f in FINGERS]
    if kind == 30:
        return ["pelvis", "thigh" + sfx, "calf" + sfx, "foot" + sfx, "ball" + sfx]
    if kind == 40:
        f = ["index", "middle", "ring", "pinky", "thumb"][fi]
        return ["hand" + sfx] + [f"{f}_0{i}{sfx}" for i in (1, 2, 3)]
    return ["pelvis"]


# junction bones that should only win when clearly closer (no visibility test)
BONE_BIAS = {"upperarm_l": 1.6, "upperarm_r": 1.6, "thigh_l": 1.25, "thigh_r": 1.25}


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def heat_weights(mesh_obj, arm_obj, heat_c=1.0):
    """Bone-heat skinning (Baran & Popovic 2007): solve
        (L + M H) w_b = M H p_b
    per bone, with L the cotangent Laplacian, M the vertex areas, H_ii = c/d_i^2
    the heat from the nearest candidate bone and p_b its indicator. One
    sparse factorization, one back-substitution per bone."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    me = mesh_obj.data
    V = np.array([v.co[:] for v in me.vertices])
    tris = np.array([p.vertices[:] for p in me.polygons if len(p.vertices) == 3])
    quads = [p.vertices[:] for p in me.polygons if len(p.vertices) == 4]
    if quads:
        q = np.array(quads)
        tris = np.concatenate([tris.reshape(-1, 3), q[:, [0, 1, 2]], q[:, [0, 2, 3]]]) if len(tris) else \
            np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    n = len(V)
    # cotangent Laplacian (negative cotangents clamped -> M-matrix, always solvable)
    I, Jx, W = [], [], []
    area = np.zeros(n)
    for k in range(3):
        i, j, o = tris[:, k], tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        u, v = V[i] - V[o], V[j] - V[o]
        cr = np.linalg.norm(np.cross(u, v), axis=1)
        cot = np.einsum("ij,ij->i", u, v) / np.maximum(cr, 1e-12)
        w = np.clip(0.5 * cot, 0.0, 50.0)
        I += [i, j]
        Jx += [j, i]
        W += [w, w]
    for k in range(3):
        a = V[tris[:, (k + 1) % 3]] - V[tris[:, k]]
        b = V[tris[:, (k + 2) % 3]] - V[tris[:, k]]
        np.add.at(area, tris[:, k], np.linalg.norm(np.cross(a, b), axis=1) / 6.0)
    I, Jx, W = np.concatenate(I), np.concatenate(Jx), np.concatenate(W)
    Wm = sp.coo_matrix((W, (I, Jx)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm
    # nearest candidate bone per vertex
    labels = np.zeros(n, dtype=int)
    if "part" in me.attributes:
        me.attributes["part"].data.foreach_get("value", labels)
    bones = {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in arm_obj.data.bones}
    names = sorted({nm for lab in np.unique(labels) for nm in candidate_bones(int(lab))})
    col = {nm: k for k, nm in enumerate(names)}
    D = np.full((n, len(names)), np.inf)
    for lab in np.unique(labels):
        sel = np.nonzero(labels == lab)[0]
        for nm in candidate_bones(int(lab)):
            h, t = bones[nm]
            D[sel, col[nm]] = seg_dist(V[sel], h, t) * BONE_BIAS.get(nm, 1.0)
    dmin = D.min(axis=1)
    nearest = D <= dmin[:, None] * 1.0001
    Hd = heat_c / np.maximum(dmin, 0.004) ** 2
    A = (L + sp.diags(area * Hd)).tocsc()
    lu = splu(A)
    Wt = np.zeros((n, len(names)))
    for nm, k in col.items():
        rhs = area * Hd * nearest[:, k]
        if rhs.any():
            Wt[:, k] = np.clip(lu.solve(rhs), 0.0, 1.0)
    Wt /= np.maximum(Wt.sum(axis=1, keepdims=True), 1e-9)
    # write vertex groups
    for g in list(mesh_obj.vertex_groups):
        mesh_obj.vertex_groups.remove(g)
    for nm, k in col.items():
        idx = np.nonzero(Wt[:, k] > 1e-4)[0]
        if len(idx) == 0:
            continue
        g = mesh_obj.vertex_groups.new(name=nm)
        for i in idx:
            g.add([int(i)], float(Wt[i, k]), "REPLACE")
    return Wt, names


def smooth_weights(mesh_obj, region_fn, iterations=6, factor=0.5):
    """Laplacian smoothing of all weights inside a region (shoulders, hips):
    spreads the falloff over more edge loops so the joint rotates skin
    instead of pinching it — what an artist does with the smooth brush."""
    me = mesh_obj.data
    n = len(me.vertices)
    co = np.array([v.co[:] for v in me.vertices])
    sel = region_fn(co)
    if not sel.any():
        return
    nb = [[] for _ in range(n)]
    for e in me.edges:
        a, b = e.vertices
        nb[a].append(b)
        nb[b].append(a)
    G = mesh_obj.vertex_groups
    Wm = np.zeros((n, len(G)))
    for v in me.vertices:
        for g in v.groups:
            Wm[v.index, g.group] = g.weight
    idx = np.nonzero(sel)[0]
    for _ in range(iterations):
        avg = np.array([Wm[nb[i]].mean(axis=0) if nb[i] else Wm[i] for i in idx])
        Wm[idx] = Wm[idx] * (1 - factor) + avg * factor
    Wm[idx] /= np.maximum(Wm[idx].sum(axis=1, keepdims=True), 1e-9)
    for gi, g in enumerate(G):
        for i in idx:
            w = float(Wm[i, gi])
            if w > 1e-4:
                g.add([int(i)], w, "REPLACE")
            else:
                g.remove([int(i)])


def skin(mesh_obj, arm_obj, max_influences=4, prune=0.01):
    # 1) heat weights for the main (non-twist, non-eye) bones
    arm = arm_obj.data
    heat_weights(mesh_obj, arm_obj)
    # shoulders and hips carry the widest rotations: soften their falloff
    S = np.array(arm.bones["upperarm_l"].head_local)
    H = np.array(arm.bones["thigh_l"].head_local)

    def joints(co):
        a = np.abs(co[:, 0:1])
        q = np.concatenate([a, co[:, 1:]], axis=1)
        return (np.linalg.norm(q - S, axis=1) < 0.11) | (np.linalg.norm(q - H, axis=1) < 0.10)
    smooth_weights(mesh_obj, joints, iterations=8, factor=0.5)
    mod = mesh_obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj
    mesh_obj.parent = arm_obj
    for b in arm.bones:
        b.use_deform = b.name != "root"
    # 2) twist split: the twist bone takes a growing share of its parent
    #    toward the far end of the segment
    me = mesh_obj.data
    co = np.array([v.co[:] for v in me.vertices])
    vg = mesh_obj.vertex_groups
    for sfx in ("_l", "_r"):
        for par, tw, t0, t1 in TWISTS:
            pb = arm.bones[par + sfx]
            h, t = np.array(pb.head_local), np.array(pb.tail_local)
            g_par = vg.get(par + sfx)
            if g_par is None:
                continue
            g_tw = vg.get(tw + sfx) or vg.new(name=tw + sfx)
            ab = t - h
            tt = np.clip(((co - h) @ ab) / (ab @ ab), 0, 1)
            share = np.clip((tt - t0) / (t1 - t0), 0, 1) * 0.6
            for v in me.vertices:
                w = 0.0
                for g in v.groups:
                    if g.group == g_par.index:
                        w = g.weight
                        break
                if w <= 0:
                    continue
                s = float(share[v.index])
                if s > 0:
                    g_par.add([v.index], w * (1 - s), "REPLACE")
                    g_tw.add([v.index], w * s, "REPLACE")
    cleanup_weights(mesh_obj, max_influences, prune)


def cleanup_weights(obj, max_influences=4, prune=0.01):
    """Engine-ready weights: <= max_influences per vertex, no slivers, sum = 1."""
    me = obj.data
    groups = obj.vertex_groups
    for v in me.vertices:
        ws = sorted(((g.weight, g.group) for g in v.groups), reverse=True)
        keep = [(w, gi) for w, gi in ws[:max_influences] if w >= prune]
        if not keep and ws:
            keep = [ws[0]]
        tot = sum(w for w, _ in keep) or 1.0
        keep_ids = {gi for _, gi in keep}
        for w, gi in ws:
            if gi not in keep_ids:
                groups[gi].remove([v.index])
        for w, gi in keep:
            groups[gi].add([v.index], w / tot, "REPLACE")


def weight_report(obj):
    me = obj.data
    counts = np.array([len(v.groups) for v in me.vertices])
    sums = np.array([sum(g.weight for g in v.groups) for v in me.vertices])
    return {"max_influences": int(counts.max()), "unweighted": int((counts == 0).sum()),
            "sum_min": float(sums.min()), "sum_max": float(sums.max())}


def bind_rigid(obj, arm_obj, bone):
    """Eyes etc.: 100% to one bone, as a skinned mesh (exports cleanly)."""
    g = obj.vertex_groups.new(name=bone)
    g.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    m = obj.modifiers.new("Armature", "ARMATURE")
    m.object = arm_obj
    obj.parent = arm_obj


def transfer_weights(src, dst, arm_obj):
    """LODs get their skin weights from LOD0 (Data Transfer, interpolated
    from the nearest face), then the same cleanup."""
    for g in src.vertex_groups:
        if dst.vertex_groups.get(g.name) is None:
            dst.vertex_groups.new(name=g.name)
    m = dst.modifiers.new("DT", "DATA_TRANSFER")
    m.object = src
    m.use_vert_data = True
    m.data_types_verts = {"VGROUP_WEIGHTS"}
    m.vert_mapping = "POLYINTERP_NEAREST"
    m.layers_vgroup_select_src = "ALL"
    m.layers_vgroup_select_dst = "NAME"
    for o in bpy.context.view_layer.objects:
        o.select_set(o == dst)
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.modifier_apply(modifier=m.name)
    cleanup_weights(dst)
    am = dst.modifiers.new("Armature", "ARMATURE")
    am.object = arm_obj
    dst.parent = arm_obj


# ---------------------------------------------------------------------------
# Test animation (walk cycle) + poses
# ---------------------------------------------------------------------------

def _rot(pb, x=0.0, y=0.0, z=0.0):
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = Euler((math.radians(x), math.radians(y), math.radians(z)), "XYZ")


def walk_cycle(arm_obj, frames=32, name="Walk"):
    """A readable in-place walk: legs/arms in counter-phase, knee flexion in
    swing, pelvis bob + yaw, spine counter-rotation, arms relaxed down from
    the A-pose."""
    arm_obj.animation_data_create()
    act = bpy.data.actions.new(name)
    arm_obj.animation_data.action = act
    P = arm_obj.pose.bones
    for f in range(frames + 1):
        ph = 2 * math.pi * f / frames
        for sfx, off in (("_l", 0.0), ("_r", math.pi)):
            s = math.sin(ph + off)
            c = math.cos(ph + off)
            sgn = 1 if sfx == "_l" else -1
            _rot(P["thigh" + sfx], x=-26 * s)
            _rot(P["calf" + sfx], x=6 + 48 * max(0.0, c) ** 1.5)
            _rot(P["foot" + sfx], x=-10 * s - 8 * max(0.0, c))
            _rot(P["ball" + sfx], x=12 * max(0.0, -s) * max(0.0, -c) * 2)
            _rot(P["upperarm" + sfx], x=20 * s, z=sgn * 38)  # +Z lowers the arm from the A-pose
            _rot(P["lowerarm" + sfx], x=-(14 + 10 * max(0.0, -s)))
            _rot(P["hand" + sfx], x=-6)
            for fg in FINGERS:
                for i in (1, 2, 3):
                    pb = P.get(f"{fg}_0{i}{sfx}")
                    if pb:
                        _rot(pb, x=-(12 if fg != "thumb" else 4))
        _rot(P["pelvis"], y=5 * math.sin(ph))
        P["pelvis"].location = (0, 0.012 * math.cos(2 * ph), 0)
        _rot(P["spine_01"], x=-3)
        _rot(P["spine_03"], y=-7 * math.sin(ph))
        _rot(P["neck_01"], y=3 * math.sin(ph))
        for pb in P:
            pb.keyframe_insert("rotation_euler", frame=f + 1)
        P["pelvis"].keyframe_insert("location", frame=f + 1)
    s = bpy.context.scene
    s.frame_start, s.frame_end = 1, frames
    s.render.fps = 30
    return act


def clear_pose(arm_obj):
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
