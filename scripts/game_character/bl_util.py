"""
bl_util — Blender-side helpers shared by every stage of the game-character
pipeline: fast numpy<->mesh transfer, scene reset, studio/look-dev
rendering (clay, wireframe-over-clay, textured), and small bmesh utilities.

Runs under Blender's bundled Python *or* the `bpy` pip module
(`pip install bpy`), which is how the pipeline runs on machines without a
Blender install (CI, cloud containers). EEVEE/Workbench need an OpenGL/EGL
context that headless containers usually lack, so everything here renders
with Cycles on the CPU.
"""

import math
import os

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def get_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


def select_only(objs, active=None):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (objs[0] if objs else None)


# ---------------------------------------------------------------------------
# Mesh <-> numpy
# ---------------------------------------------------------------------------

def mesh_from_arrays(name, verts, faces, collection=None):
    """Build a mesh object from (N,3) verts and (M,k) faces (fixed arity) or a
    list of index lists (mixed arity). foreach_set keeps million-triangle
    sculpts fast where from_pydata would take minutes."""
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, dtype=np.float32)
    if isinstance(faces, np.ndarray):
        flat = faces.reshape(-1).astype(np.int32)
        sizes = np.full(len(faces), faces.shape[1], dtype=np.int32)
    else:
        sizes = np.array([len(f) for f in faces], dtype=np.int32)
        flat = np.fromiter((i for f in faces for i in f), dtype=np.int32, count=int(sizes.sum()))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.reshape(-1))
    me.loops.add(len(flat))
    me.loops.foreach_set("vertex_index", flat)
    me.polygons.add(len(sizes))
    me.polygons.foreach_set("loop_start", starts)
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    obj = bpy.data.objects.new(name, me)
    link(obj, collection)
    return obj


def verts_np(obj, world=False):
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if world:
        M = np.array(obj.matrix_world)
        co = co @ M[:3, :3].T + M[:3, 3]
    return co


def set_verts_np(obj, co):
    obj.data.vertices.foreach_set("co", np.asarray(co, dtype=np.float32).reshape(-1))
    obj.data.update()


def apply_modifiers(obj):
    select_only([obj])
    for m in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def duplicate(obj, name, collection=None):
    new = obj.copy()
    new.data = obj.data.copy()
    new.name = name
    new.data.name = name
    for c in list(new.users_collection):
        c.objects.unlink(new)
    link(new, collection)
    return new


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


# ---------------------------------------------------------------------------
# Materials for look-dev renders
# ---------------------------------------------------------------------------

def clay_material(name="Clay", color=(0.62, 0.60, 0.58), roughness=0.55, sss=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if sss > 0:
        bsdf.inputs["Subsurface Weight"].default_value = sss
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        bsdf.inputs["Subsurface Scale"].default_value = 0.01
    return mat


def wire_material(name="ClayWire", color=(0.62, 0.60, 0.58), wire_color=(0.02, 0.02, 0.025),
                  thickness=0.0012):
    """Clay with the mesh's real edges drawn on top (Wireframe node in
    pixel-independent world size) — the way topology gets reviewed."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    wire = nt.nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = False
    wire.inputs["Size"].default_value = thickness
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*color, 1.0)
    mix.inputs["B"].default_value = (*wire_color, 1.0)
    nt.links.new(wire.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def set_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def world_bounds(objs):
    pts = []
    for o in objs:
        if o.type != "MESH":
            continue
        M = o.matrix_world
        pts.extend(M @ Vector(c) for c in o.bound_box)
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def setup_render(samples=32, resolution=(768, 768), view="AgX", look="None", transparent=False):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    try:
        s.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    s.cycles.max_bounces = 6
    s.render.resolution_x, s.render.resolution_y = resolution
    s.render.resolution_percentage = 100
    s.render.film_transparent = transparent
    s.render.image_settings.file_format = "PNG"
    s.view_settings.view_transform = view
    try:
        s.view_settings.look = look if look != "None" else "None"
    except TypeError:
        pass


def setup_studio(center, height, world_color=(0.045, 0.047, 0.05), world_strength=1.0, key=260.0,
                 name_prefix="Studio"):
    """Soft three-softbox studio sized to the subject (energies are for a
    ~1.8 m character under AgX)."""
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (*world_color, 1.0)
    bg.inputs["Strength"].default_value = world_strength
    for o in [o for o in bpy.data.objects if o.name.startswith(name_prefix)]:
        bpy.data.objects.remove(o, do_unlink=True)
    c = Vector(center)
    s = height / 1.8
    rigs = [  # (name, offset, energy, size, color)
        ("Key", Vector((-2.2, -2.6, 1.4)) * s, key, 1.6 * s, (1.0, 0.96, 0.9)),
        ("Fill", Vector((2.6, -1.8, 0.4)) * s, key * 0.35, 2.2 * s, (0.88, 0.93, 1.0)),
        ("Rim", Vector((1.2, 2.8, 1.6)) * s, key * 0.8, 1.2 * s, (0.9, 0.95, 1.0)),
        ("Bounce", Vector((0.0, -1.0, -2.2)) * s, key * 0.12, 3.0 * s, (1.0, 0.95, 0.9)),
    ]
    for nm, off, energy, size, col in rigs:
        ld = bpy.data.lights.new(f"{name_prefix}{nm}", "AREA")
        ld.energy = energy * s * s
        ld.size = size
        ld.color = col
        lo = bpy.data.objects.new(f"{name_prefix}{nm}", ld)
        link(lo)
        lo.location = c + off
        lo.rotation_euler = (c - lo.location).to_track_quat("-Z", "Y").to_euler()


def camera(name="Cam", ortho_scale=None, lens=85):
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        link(cam)
    cd = cam.data
    if ortho_scale:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho_scale
    else:
        cd.type = "PERSP"
        cd.lens = lens
    cd.clip_start = 0.01
    cd.clip_end = 100
    bpy.context.scene.camera = cam
    return cam


VIEW_DIRS = {  # direction the camera sits in, from the subject
    "front": Vector((0, -1, 0)),
    "back": Vector((0, 1, 0)),
    "left": Vector((1, 0, 0)),     # character's left side
    "right": Vector((-1, 0, 0)),
    "3q": Vector((-0.75, -1.0, 0.12)).normalized(),
    "3q_back": Vector((0.8, 0.9, 0.15)).normalized(),
}


def render_views(path_prefix, objs, views=("front", "left", "3q"), resolution=(640, 960), samples=24,
                 ortho=True, margin=1.12, focus=None, key=260.0, transparent=False):
    """Render one image per view. `focus=(center, size)` overrides auto-framing
    (use it for close-ups of the head/hands)."""
    if focus is None:
        lo, hi = world_bounds(objs)
        center = (lo + hi) / 2
        size = max(hi.z - lo.z, (hi.x - lo.x) * resolution[1] / resolution[0])
    else:
        center, size = Vector(focus[0]), focus[1]
    setup_render(samples, resolution, transparent=transparent)
    setup_studio(center, max(size, 0.3) if focus is None else 1.8, key=key)
    paths = []
    for v in views:
        d = VIEW_DIRS[v]
        dist = size * 4.0
        cam = camera(ortho_scale=size * margin if ortho else None, lens=85)
        cam.location = center + d * dist
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        if not ortho:
            fov = 2 * math.atan(36 / (2 * 85))
            cam.location = center + d * (size * margin / 2) / math.tan(fov / 2)
        p = f"{path_prefix}_{v}.png"
        bpy.context.scene.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
    return paths


def contact_sheet(paths, out_path, cols=None, bg=(24, 25, 27), labels=None):
    """Stitch rendered PNGs into one image with PIL (for quick review)."""
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in paths]
    cols = cols or len(ims)
    rows = (len(ims) + cols - 1) // cols
    w = max(i.width for i in ims)
    h = max(i.height for i in ims)
    sheet = Image.new("RGB", (w * cols, h * rows), bg)
    dr = ImageDraw.Draw(sheet)
    for n, im in enumerate(ims):
        x, y = (n % cols) * w, (n // cols) * h
        sheet.paste(im, (x, y))
        if labels:
            dr.text((x + 10, y + 10), labels[n], fill=(230, 230, 230))
    sheet.save(out_path)
    return out_path


def edge_overlay(obj, thickness=0.0007, color=(0.015, 0.015, 0.02), name_suffix="_Wire"):
    """Real-edge wireframe (quads stay quads) as a separate object: a
    Wireframe modifier on a copy — the Cycles Wireframe node would draw the
    render triangulation instead of the actual topology."""
    w = duplicate(obj, obj.name + name_suffix)
    w.modifiers.clear()
    m = w.modifiers.new("Wire", "WIREFRAME")
    m.thickness = thickness
    m.use_even_offset = True
    m.use_replace = True
    m.use_boundary = True
    mat = clay_material("WireInk", color=color, roughness=0.8)
    set_material(w, mat)
    # nudge outward so it never z-fights with the surface
    d = w.modifiers.new("Push", "DISPLACE")
    d.strength = thickness * 0.6
    d.mid_level = 0.0
    w.modifiers.move(1, 0)
    return w
