"""
piece_lowpoly — low-poly LODs for separate mesh pieces (see pieces.py).

    lowpoly="box"     a subdivided box cage fitted to the piece's frame
                      (LOD4 = 2x2 per side, LOD2/LOD0 = one/two Catmull-Clark
                      levels re-projected) — clean quads for boxy gear
    lowpoly="remesh"  QuadriFlow quad remesh of the piece's high-poly (LOD0,
                      all quads), then collapse-decimated LOD2 / LOD4 that
                      keep its UVs — for organic parts (hair, horns, beards)

Pieces share the body's UV atlas: prepare() returns the object to unwrap
together with the body; finish() then builds the remaining LODs from it.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Vector

import bl_util as U


def box_cage(name, frame):
    c, R, half = frame
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


def remesh(hp, name, faces, log=print):
    """Quad remesh of a high-poly piece to ~`faces` quads (QuadriFlow),
    snapped back onto the high-poly."""
    import retopo
    o = U.duplicate(hp, name)
    o.modifiers.clear()
    # QuadriFlow is slow on millions of triangles and only needs the shape
    if len(o.data.polygons) > 40000:
        d = o.modifiers.new("pre", "DECIMATE")
        d.ratio = 40000 / len(o.data.polygons)
        U.select_only([o])
        bpy.ops.object.modifier_apply(modifier=d.name)
    U.select_only([o])
    ok = False
    try:
        r = bpy.ops.object.quadriflow_remesh(target_faces=int(faces), use_mesh_symmetry=False, use_preserve_sharp=False,
                                             use_preserve_boundary=False, smooth_normals=False, mode="FACES", seed=0)
        ok = "FINISHED" in r and len(o.data.polygons) > 0 and all(len(p.vertices) == 4 for p in o.data.polygons)
    except RuntimeError as e:
        log(f"{name}: quadriflow failed ({e})")
    if not ok:     # fallback: a plain decimation to the same budget
        log(f"{name}: quadriflow unavailable, decimating instead")
        d = o.modifiers.new("dec", "DECIMATE")
        d.ratio = min(1.0, 2.0 * faces / max(len(o.data.polygons), 1))
        bpy.ops.object.modifier_apply(modifier=d.name)
    retopo.project_nearest(o, retopo.Surface(hp), iters=2)
    return o


def prepare(piece, hp, prefix, log=print):
    """-> (object to unwrap with the body, state for finish())."""
    import retopo
    name = piece["name"]
    if piece["lowpoly"] == "box":
        pc = box_cage(f"{prefix}_{name}_LOD4", piece["frame"])
        retopo.project_nearest(pc, retopo.Surface(hp), iters=1)
        return pc, ("box", pc)
    lod0 = remesh(hp, f"{prefix}_{name}_LOD0", piece["faces"], log=log)
    return lod0, ("remesh", lod0)


def finish(piece, hp, prefix, state):
    """All LODs {'LOD0', 'LOD2', 'LOD4'} of the piece, UVs shared."""
    import lods
    import retopo
    kind, o = state
    name = piece["name"]
    out = {}
    if kind == "box":
        out["LOD4"] = o
        prev = o
        surf = retopo.Surface(hp)
        for level in ("LOD2", "LOD0"):
            c = prev.copy()
            c.data = prev.data.copy()
            c.name = c.data.name = f"{prefix}_{name}_{level}"
            bpy.context.scene.collection.objects.link(c)
            retopo.subdivide(c, 1)
            retopo.project_nearest(c, surf, iters=3)
            out[level] = c
            prev = c
    else:
        out["LOD0"] = o
        n = len(o.data.polygons) * 2        # triangles
        out["LOD2"] = lods.decimated(o, f"{prefix}_{name}_LOD2", max(0.25, 48.0 / n))
        out["LOD4"] = lods.decimated(o, f"{prefix}_{name}_LOD4", max(0.08, 24.0 / n))
    for c in out.values():
        c.data.shade_smooth()
    return out


def load_highpolys(out_dir, names):
    """HighPoly_<name> objects from <out>/highpoly_<name>.npz."""
    import os
    res = {}
    for nm in names:
        p = os.path.join(out_dir, f"highpoly_{nm.lower()}.npz")
        if os.path.exists(p):
            d = np.load(p)
            o = U.mesh_from_arrays(f"HighPoly_{nm}", d["verts"], d["faces"])
            o.data.shade_smooth()
            o.hide_render = True
            res[nm] = o
    return res
