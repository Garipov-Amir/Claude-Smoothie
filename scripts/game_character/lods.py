"""
lods — LOD chain assembly + engine-readiness validation.

Chain (tris, ~halving each step):
  LOD0  2x subdivided cage (clean quads, triangulated for shipping)
  LOD1  LOD0 decimated 50%       (collapse, face/hands protected)
  LOD2  1x subdivided cage       (clean quads again)
  LOD3  LOD2 decimated 50%
  LOD4  the cage itself

The subdivision LODs are exact topological children of the cage, so they
share UVs by construction; the decimated in-betweens keep those UVs because
collapse decimation interpolates UV attributes and is steered away from the
face with a vertex-group factor (silhouette of the head + fingers is what
reads longest on screen). All LODs are skinned by transferring LOD0's weights.
"""

import math

import bmesh
import bpy
import numpy as np

import bl_util as U


def triangulate(obj):
    if any(len(p.vertices) != 3 for p in obj.data.polygons):
        m = obj.modifiers.new("Tri", "TRIANGULATE")
        m.quad_method = "BEAUTY"
        m.ngon_method = "BEAUTY"
        U.select_only([obj])
        bpy.ops.object.modifier_apply(modifier=m.name)


def protect_group(obj, name="LOD_protect"):
    """Weights that slow decimation down: head/face and hands."""
    g = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    co = U.verts_np(obj)
    w = np.zeros(len(co))
    w = np.maximum(w, np.clip((co[:, 2] - 1.53) / 0.05, 0, 1))                        # head
    w = np.maximum(w, np.clip((np.abs(co[:, 0]) - 0.55) / 0.05, 0, 1) * (co[:, 2] > 0.8))  # hands
    for i, wi in enumerate(w):
        if wi > 0:
            g.add([i], float(wi), "REPLACE")
    return g


def decimated(src, name, ratio):
    o = U.duplicate(src, name)
    for m in list(o.modifiers):
        o.modifiers.remove(m)
    for g in list(o.vertex_groups):
        o.vertex_groups.remove(g)
    protect_group(o)
    d = o.modifiers.new("Decimate", "DECIMATE")
    d.decimate_type = "COLLAPSE"
    d.ratio = ratio
    d.use_symmetry = True
    d.symmetry_axis = "X"
    d.vertex_group = "LOD_protect"
    d.vertex_group_factor = 4.0
    d.invert_vertex_group = True  # the group marks what to *keep*
    d.use_collapse_triangulate = True
    U.select_only([o])
    bpy.ops.object.modifier_apply(modifier=d.name)
    o.vertex_groups.remove(o.vertex_groups["LOD_protect"])
    return o


def low_eye(src, name, segments=12, rings=8):
    import eyes
    side = 1 if src.location.x > 0 else -1
    e = eyes.make_eye(side, name, segments, rings)
    e.data.materials.clear()
    for mat in src.data.materials:
        e.data.materials.append(mat)
    return e


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def mesh_report(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    sizes = np.array([len(f.verts) for f in bm.faces])
    nonman = [e for e in bm.edges if not e.is_manifold and not e.is_boundary]
    boundary = [e for e in bm.edges if e.is_boundary]
    degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-10)
    loose = sum(1 for v in bm.verts if not v.link_faces)
    poles = {}
    for v in bm.verts:
        if v.is_boundary:
            continue
        val = len(v.link_edges)
        if val != 4:
            poles[val] = poles.get(val, 0) + 1
    bm.free()
    r = {
        "verts": len(me.vertices), "faces": len(me.polygons), "tris": int((sizes - 2).sum()),
        "quads": int((sizes == 4).sum()), "ngons": int((sizes > 4).sum()), "triangles": int((sizes == 3).sum()),
        "non_manifold_edges": len(nonman), "boundary_edges": len(boundary),
        "degenerate_faces": degenerate, "loose_verts": loose,
        "poles_by_valence": {str(k): v for k, v in sorted(poles.items())},
        "materials": [m.name for m in me.materials if m],
        "scale_applied": tuple(round(s, 6) for s in obj.scale) == (1.0, 1.0, 1.0),
        "rotation_applied": all(abs(a) < 1e-6 for a in obj.rotation_euler),
    }
    if me.uv_layers:
        r.update(uv_report(obj))
    return r


def uv_report(obj, grid=1024):
    me = obj.data
    uv = me.uv_layers.active.data
    coords = np.array([d.uv[:] for d in uv])
    inside = bool(((coords >= -1e-6) & (coords <= 1 + 1e-6)).all())
    # overlap: rasterize triangles into a coverage-count grid
    from PIL import Image, ImageDraw
    counts = np.zeros((grid, grid), np.uint16)
    union = np.zeros((grid, grid), bool)
    for p in me.polygons:
        idx = list(p.loop_indices)
        for k in range(1, len(idx) - 1):
            tri = [coords[idx[0]], coords[idx[k]], coords[idx[k + 1]]]
            xs = [t[0] * grid for t in tri]
            ys = [t[1] * grid for t in tri]
            x0, x1 = int(max(0, min(xs) - 1)), int(min(grid, max(xs) + 2))
            y0, y1 = int(max(0, min(ys) - 1)), int(min(grid, max(ys) + 2))
            if x1 <= x0 or y1 <= y0:
                continue
            im = Image.new("L", (x1 - x0, y1 - y0), 0)
            ImageDraw.Draw(im).polygon([(x - x0, y - y0) for x, y in zip(xs, ys)], fill=1)
            a = np.asarray(im, dtype=np.uint16)
            union[y0:y1, x0:x1] |= a.astype(bool)
            # interior only (shared edges between neighbors are not overlap)
            from scipy import ndimage
            counts[y0:y1, x0:x1] += ndimage.binary_erosion(a).astype(np.uint16)
    covered = union.sum()
    overlap = (counts > 1).sum()
    return {"uv_inside_0_1": inside, "uv_coverage": float(covered / grid ** 2),
            "uv_overlap_fraction": float(overlap / max(covered, 1))}


def skin_report(obj, max_inf=4):
    me = obj.data
    counts = np.array([len(v.groups) for v in me.vertices])
    sums = np.array([sum(g.weight for g in v.groups) for v in me.vertices])
    return {"max_influences": int(counts.max()) if len(counts) else 0,
            "over_limit_verts": int((counts > max_inf).sum()),
            "unweighted_verts": int((counts == 0).sum()),
            "weights_normalized": bool(np.allclose(sums[counts > 0], 1.0, atol=1e-3))}


def skeleton_report(arm_obj):
    bones = arm_obj.data.bones
    names = [b.name for b in bones]
    required = ["root", "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head",
                "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l", "thigh_l", "calf_l", "foot_l", "ball_l",
                "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r", "thigh_r", "calf_r", "foot_r", "ball_r"]
    roots = [b.name for b in bones if b.parent is None]
    return {"bone_count": len(bones), "deform_bones": sum(1 for b in bones if b.use_deform),
            "single_root": roots == ["root"], "missing_humanoid_bones": [n for n in required if n not in names],
            "root_at_origin": tuple(round(c, 6) for c in bones["root"].head_local) == (0.0, 0.0, 0.0)}


def topology_report(obj):
    """Quad-mesh quality numbers for the *source* (untriangulated) meshes:
    what a lead looks at to judge a retopo."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    sizes = np.array([len(f.verts) for f in bm.faces])
    poles = {}
    for v in bm.verts:
        if v.is_boundary:
            continue
        val = len(v.link_edges)
        if val != 4:
            poles[str(val)] = poles.get(str(val), 0) + 1
    # quad shape quality: interior-angle deviation from 90 deg
    dev = []
    for f in bm.faces:
        if len(f.verts) != 4:
            continue
        for l in f.loops:
            dev.append(abs(math.degrees(l.calc_angle()) - 90.0))
    bm.free()
    dev = np.array(dev)
    return {"faces": int(len(sizes)), "quad_ratio": float((sizes == 4).mean()),
            "tris": int((sizes == 3).sum()), "ngons": int((sizes > 4).sum()),
            "poles_by_valence": poles,
            "quad_angle_dev_median_deg": float(np.median(dev)), "quad_angle_dev_p95_deg": float(np.percentile(dev, 95))}
