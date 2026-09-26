"""
uvs — unwrap, texel-density balancing and packing for the game mesh.

Seams are authored on the cage (see retopo: back of the head, torso sides,
arm undersides, inner legs, palmar side of hands/fingers, boot tops,
wrists, neck, shoulders and hips) — placed where a real character artist
hides them: under the arms, inside the legs, at garment boundaries.
The cage is unwrapped once (angle-based); every subdivision LOD inherits
those UVs, so a single texture set serves the whole LOD chain.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector


def _edit(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")


def islands(bm, uv):
    """Connected UV islands as lists of faces (flood fill across non-seam
    edges whose UVs are continuous)."""
    seen, out = set(), []
    for f0 in bm.faces:
        if f0.index in seen:
            continue
        stack, isl = [f0], []
        seen.add(f0.index)
        while stack:
            f = stack.pop()
            isl.append(f)
            for e in f.edges:
                if e.seam:
                    continue
                for g in e.link_faces:
                    if g.index not in seen:
                        seen.add(g.index)
                        stack.append(g)
        out.append(isl)
    return out


def unwrap(obj, head_scale=1.6, head_min_z=1.50, margin=0.0035, iterations=0):
    """ABF unwrap along the marked seams, equalize texel density, give the
    head islands `head_scale` x density (faces are what players look at),
    then pack into 0-1."""
    me = obj.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    _edit(obj)
    bpy.ops.uv.unwrap(method="ANGLE_BASED", fill_holes=True, correct_aspect=True, margin=0.001)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.object.mode_set(mode="OBJECT")

    # scale the head island(s) up around their own centers
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    uv = bm.loops.layers.uv.active
    for isl in islands(bm, uv):
        zc = np.mean([f.calc_center_median().z for f in isl])
        if zc < head_min_z + 0.08:
            continue
        loops = [l for f in isl for l in f.loops]
        c = sum((l[uv].uv for l in loops), Vector((0, 0))) / len(loops)
        for l in loops:
            l[uv].uv = c + (l[uv].uv - c) * head_scale
    bm.to_mesh(me)
    bm.free()

    _edit(obj)
    try:
        bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="CARDINAL",
                                scale=True, merge_overlap=False, margin_method="FRACTION",
                                margin=margin, shape_method="CONCAVE")
    except TypeError:
        bpy.ops.uv.pack_islands(rotate=True, margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_stats(obj):
    """Texel-density spread + UV-space coverage — the numbers a lead checks."""
    me = obj.data
    uv = me.uv_layers.active.data
    dens, area_uv = [], 0.0
    for p in me.polygons:
        idx = list(p.loop_indices)
        pts3 = [me.vertices[me.loops[i].vertex_index].co for i in idx]
        pts2 = [uv[i].uv for i in idx]
        a3 = 0.0
        a2 = 0.0
        for k in range(1, len(idx) - 1):
            a3 += (pts3[k] - pts3[0]).cross(pts3[k + 1] - pts3[0]).length / 2
            e1, e2 = pts2[k] - pts2[0], pts2[k + 1] - pts2[0]
            a2 += abs(e1.x * e2.y - e1.y * e2.x) / 2
        area_uv += a2
        if a3 > 1e-10:
            dens.append(math.sqrt(a2 / a3))
    d = np.array(dens)
    return {"uv_coverage": area_uv, "density_median": float(np.median(d)),
            "density_p5": float(np.percentile(d, 5)), "density_p95": float(np.percentile(d, 95))}
