"""
bake — Cycles high-poly -> low-poly map baking, results as numpy arrays.

Maps (all in the low-poly's UV space):
  normal_ts   tangent-space normal, OpenGL (+Y) convention, MikkTSpace
  normal_os   object-space normal of the high-poly (for top-down dust/dirt masks)
  ao          ambient occlusion of the high-poly
  curvature   Cycles "pointiness" of the high-poly (convex > 0.5 > concave)
  position    object-space position of the high-poly surface (float)
  coverage    1 where a texel belongs to a UV island (for dilation / masks)

Baking tips encoded here (the things that go wrong in real bakes):
* The low-poly is triangulated *before* baking and exported with the same
  triangulation, so the tangent basis the engine rebuilds matches the one
  the normal map was baked against (a "synced" workflow).
* A cage (ray-start offset) of ~1 cm and a short max ray distance avoid rays
  catching the wrong surface in tight places (armpits, between fingers).
* Every map is dilated ("margin") well past island borders so mips don't
  bleed background into the edges.
"""

import bpy
import numpy as np


def new_image(name, res, float_buffer=True, color_space="Non-Color"):
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, res, res, alpha=False, float_buffer=float_buffer)
    img.colorspace_settings.name = color_space
    return img


def image_to_np(img):
    w, h = img.size
    a = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(h, w, 4)  # row 0 = bottom (V=0)


def np_to_image(arr, name, color_space="sRGB", float_buffer=False):
    h, w = arr.shape[:2]
    img = bpy.data.images.get(name) or bpy.data.images.new(name, w, h, alpha=True, float_buffer=float_buffer)
    if tuple(img.size) != (w, h):
        img.scale(w, h)
    img.colorspace_settings.name = color_space
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :arr.shape[2] if arr.ndim == 3 else 1] = arr if arr.ndim == 3 else arr[..., None]
    img.pixels.foreach_set(rgba.reshape(-1))
    img.update()
    return img


def _target_material(low, img):
    mat = bpy.data.materials.get("BakeTarget") or bpy.data.materials.new("BakeTarget")
    mat.use_nodes = True
    nt = mat.node_tree
    node = nt.nodes.get("BakeImage") or nt.nodes.new("ShaderNodeTexImage")
    node.name = "BakeImage"
    node.image = img
    for n in nt.nodes:
        n.select = False
    node.select = True
    nt.nodes.active = node
    low.data.materials.clear()
    low.data.materials.append(mat)
    return mat


def _emit_material(hp, kind):
    """High-poly material emitting a data channel for EMIT bakes."""
    mat = bpy.data.materials.get(f"HPEmit_{kind}") or bpy.data.materials.new(f"HPEmit_{kind}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    if kind == "position":
        nt.links.new(geo.outputs["Position"], em.inputs["Color"])
    elif kind == "pointiness":
        nt.links.new(geo.outputs["Pointiness"], em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    hp.data.materials.clear()
    hp.data.materials.append(mat)


def _select(hp, low):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    hp.hide_set(False)
    hp.hide_render = False
    low.hide_render = False
    hp.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low


def bake_maps(hp, low, res=2048, extrusion=0.010, max_ray=0.025, margin=24, ao_distance=0.20,
              ao_samples=32, maps=("normal_ts", "normal_os", "ao", "curvature", "position"), log=print):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    b = s.render.bake
    b.use_selected_to_active = True
    b.use_cage = False
    b.cage_extrusion = extrusion
    b.max_ray_distance = max_ray
    b.margin = margin
    b.margin_type = "EXTEND"
    b.use_clear = True
    b.target = "IMAGE_TEXTURES"
    if s.world is None:
        s.world = bpy.data.worlds.new("World")
    s.world.light_settings.distance = ao_distance
    out = {}
    for name in maps:
        img = new_image(f"bake_{name}", res, float_buffer=True)
        _target_material(low, img)
        _select(hp, low)
        if name == "normal_ts":
            s.cycles.samples = 4
            b.normal_space = "TANGENT"
            b.normal_r, b.normal_g, b.normal_b = "POS_X", "POS_Y", "POS_Z"
            bpy.ops.object.bake(type="NORMAL")
        elif name == "normal_os":
            s.cycles.samples = 2
            b.normal_space = "OBJECT"
            b.normal_r, b.normal_g, b.normal_b = "POS_X", "POS_Y", "POS_Z"
            bpy.ops.object.bake(type="NORMAL")
        elif name == "ao":
            s.cycles.samples = ao_samples
            bpy.ops.object.bake(type="AO")
        elif name in ("curvature", "position"):
            s.cycles.samples = 2
            _emit_material(hp, "pointiness" if name == "curvature" else "position")
            bpy.ops.object.bake(type="EMIT")
        out[name] = image_to_np(img)[..., :3].copy()
        log(f"baked {name}")
    # coverage: texels inside UV islands (rasterized from the low-poly UVs)
    out["coverage"] = uv_coverage(low, res)
    return out


def uv_coverage(obj, res):
    from PIL import Image, ImageDraw
    im = Image.new("L", (res, res), 0)
    dr = ImageDraw.Draw(im)
    me = obj.data
    uv = me.uv_layers.active.data
    for p in me.polygons:
        pts = [(uv[i].uv.x * res, (1 - uv[i].uv.y) * res) for i in p.loop_indices]
        dr.polygon(pts, fill=255)
    a = np.asarray(im, dtype=np.float32)[::-1] / 255.0  # flip to row0 = V0
    return a


def composite(parts):
    """Merge per-piece bakes that share one UV atlas ("match by mesh name":
    each low-poly piece baked only against its own high-poly, so a body
    texel next to the pouch never catches the pouch's surface).

    parts: list of map dicts (each with "coverage"). Every texel goes to the
    piece whose islands are nearest, so each piece keeps its own dilated
    margin. Adds "part" (index of the owning piece, per texel)."""
    from scipy import ndimage
    dist = np.stack([ndimage.distance_transform_edt(p["coverage"] < 0.5) for p in parts])
    owner = np.argmin(dist, axis=0)
    out = {}
    for k in parts[0]:
        if k == "coverage":
            continue
        a = parts[0][k].copy()
        for i, p in enumerate(parts[1:], 1):
            a[owner == i] = p[k][owner == i]
        out[k] = a
    out["coverage"] = np.clip(sum(p["coverage"] for p in parts), 0, 1)
    out["part"] = owner.astype(np.float32)
    return out
