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


def dome_cage(name, frame, n_around=8, n_rings=2):
    """Coarse closed shell for a dome piece (pauldron): outer cap, rolled rim,
    inner cap — subdivided and projected like the box cage. QuadriFlow can't
    resolve two surfaces a few mm apart; a thin shell needs its topology
    laid out."""
    cc, a, r_in, t, cap = frame
    a = np.asarray(a, np.float64)
    u = np.cross(a, [0, 1.0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(a, u)
    bm = bmesh.new()

    def ring(radius, th):
        return [bm.verts.new((np.asarray(cc) + radius * (np.cos(th) * a + np.sin(th) * (np.cos(ph) * u + np.sin(ph) * v))).tolist())
                for ph in np.linspace(0, 2 * np.pi, n_around, endpoint=False)]
    ths = [cap * (k + 1) / n_rings for k in range(n_rings)]
    outer_pole = bm.verts.new((np.asarray(cc) + (r_in + t) * a).tolist())
    inner_pole = bm.verts.new((np.asarray(cc) + r_in * a).tolist())
    outer = [ring(r_in + t, th) for th in ths]
    inner = [ring(r_in, th) for th in ths]
    n = n_around
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new([outer_pole, outer[0][i], outer[0][j]])
        bm.faces.new([inner_pole, inner[0][j], inner[0][i]])
        for k in range(n_rings - 1):
            bm.faces.new([outer[k][i], outer[k + 1][i], outer[k + 1][j], outer[k][j]])
            bm.faces.new([inner[k][j], inner[k + 1][j], inner[k + 1][i], inner[k][i]])
        bm.faces.new([outer[-1][i], inner[-1][i], inner[-1][j], outer[-1][j]])    # rim
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.index_update()
    shell = np.zeros(len(bm.verts), np.float32)          # 1 outer surface, 0 inner
    for v in [outer_pole] + [x for r in outer for x in r]:
        shell[v.index] = 1.0
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.attributes.new("shell", "FLOAT", "POINT").data.foreach_set("value", shell)
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def _quadriflow(o, faces):
    """QuadriFlow at unit scale: its manifold check uses absolute tolerances
    and rejects a clean 3 cm tusk that it accepts once scaled to ~1 m."""
    me = o.data
    co = np.array([v.co[:] for v in me.vertices])
    c = co.mean(axis=0)
    k = 1.0 / max(float(np.ptp(co, axis=0).max()), 1e-6)
    me.vertices.foreach_set("co", ((co - c) * k).reshape(-1))
    me.update()
    U.select_only([o])
    try:
        r = bpy.ops.object.quadriflow_remesh(target_faces=max(int(faces), 24), use_mesh_symmetry=False,
                                             use_preserve_sharp=False, use_preserve_boundary=False,
                                             smooth_normals=False, mode="FACES", seed=0)
        ok = "FINISHED" in r and len(o.data.polygons) > 0 and all(len(p.vertices) == 4 for p in o.data.polygons)
    except RuntimeError:
        ok = False
    me = o.data
    co2 = np.array([v.co[:] for v in me.vertices])
    me.vertices.foreach_set("co", (co2 / k + c).reshape(-1))
    me.update()
    return ok


def remesh(hp, name, faces, log=print):
    """Quad remesh of a high-poly piece to ~`faces` quads (QuadriFlow, one
    connected part at a time — it fails on disjoint parts such as a pair of
    horns), snapped back onto the high-poly."""
    import retopo
    o = U.duplicate(hp, name)
    o.modifiers.clear()
    # QuadriFlow is slow on millions of triangles and only needs the shape
    if len(o.data.polygons) > 40000:
        d = o.modifiers.new("pre", "DECIMATE")
        d.ratio = 40000 / len(o.data.polygons)
        U.select_only([o])
        bpy.ops.object.modifier_apply(modifier=d.name)
    # QuadriFlow needs a clean manifold with consistent winding: marching
    # cubes leaves duplicate verts and slivers at thin tips
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bad = [e for e in bm.edges if not e.is_manifold]
    if bad:
        bmesh.ops.delete(bm, geom=list({f for e in bad for f in e.link_faces}), context="FACES")
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
        bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data)
    bm.free()
    U.select_only([o])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [p for p in bpy.context.selected_objects if p.type == "MESH"]
    areas = [sum(f.area for f in p.data.polygons) for p in parts]
    total = sum(areas) or 1.0
    done = []
    for p, a in zip(parts, areas):
        if len(p.data.polygons) < 8:          # marching-cubes crumbs
            bpy.data.objects.remove(p, do_unlink=True)
            continue
        if not _quadriflow(p, faces * a / total):
            log(f"{name}: quadriflow failed on a part, decimating it instead")
            d = p.modifiers.new("dec", "DECIMATE")
            d.ratio = min(1.0, 2.0 * faces * a / total / max(len(p.data.polygons), 1))
            U.select_only([p])
            bpy.ops.object.modifier_apply(modifier=d.name)
        done.append(p)
    U.select_only(done, done[0])
    if len(done) > 1:
        bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    o.name = o.data.name = name
    retopo.project_nearest(o, retopo.Surface(hp), iters=2)
    import lods
    _vi, n = lods._crossing_verts(o)
    if n:
        # snapping a coarse quad ring onto a sharp tip can fold it: fall back
        # to a clean collapse decimation of the high-poly for this piece
        log(f"{name}: remesh folded ({n} crossing pairs), using a decimated high-poly instead")
        bpy.data.objects.remove(o, do_unlink=True)
        src = U.duplicate(hp, name + "_src")
        src.modifiers.clear()
        o = lods.decimated(src, name, min(1.0, 2.0 * faces / max(len(src.data.polygons), 1)), log=log)
        bpy.data.objects.remove(src, do_unlink=True)
    return o


def prepare(piece, hp, prefix, log=print):
    """-> (object to unwrap with the body, state for finish())."""
    import retopo
    name = piece["name"]
    if piece["lowpoly"] == "dome":
        pc = dome_cage(f"{prefix}_{name}_LOD4", piece["frame"])
        return pc, ("dome", pc)
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
    if kind == "dome":
        # simple (non-smoothing) subdivision, then every vertex back onto its
        # own sphere: a thin shell stays two clean surfaces a few mm apart
        cc, a, r_in, t, cap = piece["frame"]
        cc, a = np.asarray(cc, np.float64), np.asarray(a, np.float64)
        out["LOD4"] = o
        prev = o
        for level in ("LOD2", "LOD0"):
            c = prev.copy()
            c.data = prev.data.copy()
            c.name = c.data.name = f"{prefix}_{name}_{level}"
            bpy.context.scene.collection.objects.link(c)
            m = c.modifiers.new("Sub", "SUBSURF")
            m.subdivision_type = "SIMPLE"
            m.levels = 1
            U.select_only([c])
            bpy.ops.object.modifier_apply(modifier=m.name)
            co = np.array([v.co[:] for v in c.data.vertices])
            sh = np.zeros(len(co), np.float32)
            c.data.attributes["shell"].data.foreach_get("value", sh)
            d = co - cc
            r = np.linalg.norm(d, axis=1)
            dh = d / np.maximum(r, 1e-9)[:, None]
            # the layer attribute (1 outer, 0 inner, in between on the rim)
            # says which sphere a vertex belongs to — a chord's sag can exceed
            # the shell thickness, so neither radius nor normal can tell
            rim = (sh > 0.02) & (sh < 0.98)
            side = dh[rim] - (dh[rim] @ a)[:, None] * a
            side /= np.maximum(np.linalg.norm(side, axis=1), 1e-9)[:, None]
            dh[rim] = np.cos(cap) * a + np.sin(cap) * side
            co = cc + dh * (r_in + t * sh)[:, None]
            c.data.vertices.foreach_set("co", co.reshape(-1))
            c.data.update()
            out[level] = c
            prev = c
    elif kind == "box":
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
