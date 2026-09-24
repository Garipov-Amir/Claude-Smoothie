"""
bpy_stylized_kit — helper library for building stylized 3D models in Blender.

Import this from any script run via `blender --background --python <script>`
(scripts/run_blender.sh adds this file's directory to sys.path automatically),
or from code executed live through the MCP bridge (mcp_server/blender_addon.py
also adds this directory to sys.path on register).

Design notes:
- Sticks to Principled BSDF sockets that are stable across Blender 3.x/4.x
  (Base Color, Metallic, Roughness) and avoids version-fragile socket names
  (e.g. "Specular" was renamed in 4.0) so generation scripts don't break
  across Blender updates.
- The default "hand-painted PBR" material fakes the classic stylized-game-art
  look non-destructively: baked-in ambient occlusion shading, warm/cool
  fresnel rim tinting, and noise-driven color/roughness micro-variation to
  read as painterly instead of flat-CG.
"""

import bpy
import math
import random
from mathutils import Vector


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def new_scene(clear=True):
    """Clear the default scene down to nothing so scripts start from a blank slate."""
    if clear:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        for block_collection in (bpy.data.meshes, bpy.data.materials,
                                  bpy.data.lights, bpy.data.cameras, bpy.data.images):
            for block in list(block_collection):
                if block.users == 0:
                    block_collection.remove(block)


def set_stylized_color_management():
    """AgX (Blender's default view transform) desaturates bright colors toward gray,
    which fights a flat/hand-painted look. Standard keeps colors punchy and close to
    the material's actual RGB values, closer to how game-art viewports render."""
    view_settings = bpy.context.scene.view_settings
    for transform in ("Standard", "Filmic", "Raw"):
        try:
            view_settings.view_transform = transform
            break
        except TypeError:
            continue


def set_world_background(color=(0.04, 0.045, 0.05), strength=1.0):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (*color, 1.0)
        bg.inputs["Strength"].default_value = strength


# ---------------------------------------------------------------------------
# Primitives & mesh helpers
# ---------------------------------------------------------------------------

_PRIMITIVE_OPS = {
    "cube": lambda **kw: bpy.ops.mesh.primitive_cube_add(**kw),
    "sphere": lambda **kw: bpy.ops.mesh.primitive_uv_sphere_add(**kw),
    "ico_sphere": lambda **kw: bpy.ops.mesh.primitive_ico_sphere_add(**kw),
    "cylinder": lambda **kw: bpy.ops.mesh.primitive_cylinder_add(**kw),
    "cone": lambda **kw: bpy.ops.mesh.primitive_cone_add(**kw),
    "torus": lambda **kw: bpy.ops.mesh.primitive_torus_add(**kw),
    "plane": lambda **kw: bpy.ops.mesh.primitive_plane_add(**kw),
}


def add_primitive(kind, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), name=None, **extra):
    """Add a primitive mesh. kind in cube/sphere/ico_sphere/cylinder/cone/torus/plane."""
    if kind not in _PRIMITIVE_OPS:
        raise ValueError(f"unknown primitive kind {kind!r}; choose from {list(_PRIMITIVE_OPS)}")
    _PRIMITIVE_OPS[kind](location=location, rotation=rotation, **extra)
    obj = bpy.context.active_object
    obj.scale = scale
    if name:
        obj.name = name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def shade_smooth(obj, auto_smooth_angle=None):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    if auto_smooth_angle is not None:
        try:  # Blender 4.1+: "Shade Auto Smooth" operator sets up the angle-based modifier itself
            bpy.ops.object.shade_auto_smooth(angle=math.radians(auto_smooth_angle))
        except (RuntimeError, AttributeError):
            bpy.ops.object.shade_smooth()
            try:
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = math.radians(auto_smooth_angle)
            except AttributeError:
                pass
    else:
        bpy.ops.object.shade_smooth()


def shade_flat(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_flat()


def add_subsurf(obj, levels=2, render_levels=2):
    mod = obj.modifiers.new("Subdivision", type="SUBSURF")
    mod.levels = levels
    mod.render_levels = render_levels
    return mod


def add_bevel(obj, width=0.02, segments=2, limit_method="ANGLE"):
    mod = obj.modifiers.new("Bevel", type="BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = limit_method
    return mod


def add_decimate(obj, ratio=0.5):
    """Useful for pushing an imported/reconstructed mesh toward a low-poly stylized look."""
    mod = obj.modifiers.new("Decimate", type="DECIMATE")
    mod.ratio = ratio
    return mod


def add_solidify_outline(obj, thickness=0.02, color=(0.02, 0.02, 0.02)):
    """Classic inverted-hull ink outline: a flipped-normal shell in a dark unlit color."""
    mod = obj.modifiers.new("OutlineShell", type="SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_flip_normals = True
    mat = bpy.data.materials.new(f"{obj.name}_outline")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
    mod.material_offset = len(obj.data.materials)
    obj.data.materials.append(mat)
    return mod


def join_objects(objects, name="Model"):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    return joined


def mirror_x(obj, obj_to_mirror_around=None):
    mod = obj.modifiers.new("MirrorX", type="MIRROR")
    mod.use_axis[0] = True
    return mod


# ---------------------------------------------------------------------------
# Polygon / loft-based mesh building
#
# Preferred over add_primitive()+join_objects() for organic character bodies:
# a stack of overlapping spheres/cones reads as a "bead necklace" of separate
# blobby shapes with hard seams. These build ONE continuous polygon surface
# by lofting a series of radius-tagged rings along a spine, matching how
# real characters are actually modeled. See reference/polygon_modeling.md.
# ---------------------------------------------------------------------------

def loft_profile(rings, segments=16, cap_start=True, cap_end=True, name="Loft"):
    """
    Build one continuous mesh by bridging a sequence of circular/elliptical
    rings into quad faces.

    rings: list of {"co": (x,y,z), "radius": r} (or radius=(rx,ry) for an
    elliptical cross-section), at least 2 entries, ordered along the part's
    main axis (e.g. hip -> chest -> neck -> head-top).

    Each ring's plane is perpendicular to the local spine tangent (average
    direction to its neighbors), so the loft follows curved spines, not just
    straight ones.
    """
    import bmesh

    if len(rings) < 2:
        raise ValueError("loft_profile needs at least 2 rings")

    coords = [Vector(r["co"]) for r in rings]
    bm = bmesh.new()
    ring_verts = []

    for i, ring in enumerate(rings):
        co = coords[i]
        if i == 0:
            tangent = (coords[1] - coords[0])
        elif i == len(rings) - 1:
            tangent = (coords[i] - coords[i - 1])
        else:
            tangent = (coords[i + 1] - coords[i - 1])
        if tangent.length < 1e-9:
            tangent = Vector((0, 0, 1))
        tangent.normalize()

        up_ref = Vector((0, 0, 1)) if abs(tangent.z) < 0.9 else Vector((1, 0, 0))
        basis1 = tangent.cross(up_ref)
        if basis1.length < 1e-6:
            up_ref = Vector((1, 0, 0)) if abs(tangent.x) < 0.9 else Vector((0, 1, 0))
            basis1 = tangent.cross(up_ref)
        basis1.normalize()
        basis2 = tangent.cross(basis1).normalized()

        radius = ring["radius"]
        rx, ry = radius if isinstance(radius, (tuple, list)) else (radius, radius)

        verts = []
        for k in range(segments):
            angle = 2 * math.pi * k / segments
            pt = co + math.cos(angle) * rx * basis1 + math.sin(angle) * ry * basis2
            verts.append(bm.verts.new(pt))
        ring_verts.append(verts)

    bm.verts.ensure_lookup_table()

    for i in range(len(ring_verts) - 1):
        a, b = ring_verts[i], ring_verts[i + 1]
        for k in range(segments):
            k2 = (k + 1) % segments
            bm.faces.new((a[k], a[k2], b[k2], b[k]))

    def _ring_radius(ring):
        r = ring["radius"]
        rx, ry = r if isinstance(r, (tuple, list)) else (r, r)
        return max(rx, ry)

    if cap_start and _ring_radius(rings[0]) > 1e-6:
        bm.faces.new(ring_verts[0])
    if cap_end and _ring_radius(rings[-1]) > 1e-6:
        bm.faces.new(ring_verts[-1])

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def build_profile_body(spine, segments=16, name="Body"):
    """
    Convenience wrapper over loft_profile: spine is a list of
    (position, radius) or (position, radius_x, radius_y) control points
    along a part's main axis. Direct replacement for stacking a sphere for
    the body and another sphere for the head — this makes one continuous
    tapered surface instead.
    """
    rings = []
    for point in spine:
        if len(point) == 2:
            co, r = point
            rings.append({"co": co, "radius": r})
        else:
            co, rx, ry = point
            rings.append({"co": co, "radius": (rx, ry)})
    return loft_profile(rings, segments=segments, cap_start=True, cap_end=True, name=name)


def rounded_cap_points(base_co, base_radius, tip_co, n=2):
    """
    n intermediate (position, radius) points between a full-radius ring and
    a tapering tip, following a quarter-circle falloff (radius * cos(t*90deg))
    instead of a straight linear taper. Insert into a build_profile_body
    spine to turn a sharp cone-like taper into a rounded dome cap — e.g. a
    head modeled as [..., (neck_co, neck_r), (chest_co, chest_r)] tapering
    straight to a near-zero tip reads as a POINTED cone; inserting these
    points between the last full-radius ring and the tip rounds it off:
        spine = [..., (head_co, head_r)] + k.rounded_cap_points(head_co, head_r, top_co) + [(top_co, 0.02)]
    """
    base_co, tip_co = Vector(base_co), Vector(tip_co)
    points = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        co = base_co.lerp(tip_co, t)
        radius = base_radius * math.cos(t * math.pi / 2)
        points.append((tuple(co), radius))
    return points


def add_tapered_limb(start, end, start_radius, end_radius=None, segments=10, name="Limb",
                      bend=None, bend_radius=None):
    """
    A straight or single-bend tapered limb (arm, leg, tail, a big quill) as
    one continuous watertight surface, capped at both ends — replaces a
    cylinder+sphere-glove combo with one smooth capsule-like shape.
    """
    if end_radius is None:
        end_radius = start_radius * 0.6
    if bend is not None:
        br = bend_radius if bend_radius is not None else (start_radius + end_radius) / 2
        spine = [(start, start_radius), (bend, br), (end, end_radius)]
    else:
        spine = [(start, start_radius), (end, end_radius)]
    return build_profile_body(spine, segments=segments, name=name)


def boolean_op(objects, operation="UNION", name="Merged"):
    """
    Combine objects via Blender's Boolean modifier (exact solver).
    operation: "UNION" (fuse into one watertight mesh, e.g. an arm into the
    torso), "INTERSECT" (keep only the overlap — this is how
    carve_from_silhouettes builds a visual hull from multiple views), or
    "DIFFERENCE" (subtract objects[1:] from objects[0]).
    Blender's boolean keeps each source face's material assignment,
    remapping into the combined material list — verified in
    scripts/_smoke_test_loft.py for UNION.
    """
    base = objects[0]
    for other in objects[1:]:
        bpy.ops.object.select_all(action="DESELECT")
        base.select_set(True)
        bpy.context.view_layer.objects.active = base
        mod = base.modifiers.new(f"{operation.title()}_{other.name}", type="BOOLEAN")
        mod.operation = operation
        mod.object = other
        try:
            mod.solver = "EXACT"
        except TypeError:
            pass
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(other, do_unlink=True)
    base.name = name
    return base


def boolean_union(objects, name="Merged"):
    """Fuse overlapping continuous parts into one watertight mesh. See boolean_op."""
    return boolean_op(objects, operation="UNION", name=name)


def assign_material_by_region(obj, mat, region_fn):
    """
    Paint material regions onto ONE continuous mesh by world-space face
    position, for when a lofted part needs multiple colors without being
    split into separate objects. region_fn(world_space_face_center) -> bool.
    """
    materials = list(obj.data.materials)
    if mat not in materials:
        obj.data.materials.append(mat)
        materials = list(obj.data.materials)
    mat_index = materials.index(mat)
    mw = obj.matrix_world
    for poly in obj.data.polygons:
        if region_fn(mw @ poly.center):
            poly.material_index = mat_index


# ---------------------------------------------------------------------------
# Hand-painted PBR material (default stylized look)
# ---------------------------------------------------------------------------

def build_hand_painted_material(
    name,
    base_color,
    variation=0.15,
    saturation_boost=1.15,
    cavity_strength=0.45,
    roughness=0.55,
    metallic=0.0,
    rim_color=(0.55, 0.70, 0.95),
    rim_strength=0.12,
):
    """
    Build a node-based material that fakes a hand-painted stylized-PBR look:
      - noise-driven base color variation (reads as brush strokes, not flat CG)
      - baked-in ambient-occlusion multiply for soft painted cavity shadows
      - a cool fresnel rim tint at grazing angles (warm/cool color separation)

    base_color: (r, g, b) in 0..1
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (650, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # --- painterly color variation ---
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 200)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-700, 200)
    noise.inputs["Scale"].default_value = 8.0
    noise.inputs["Detail"].default_value = 4.0
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-500, 200)
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.65
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    r, g, b = base_color
    dark = (max(r * (1 - variation), 0), max(g * (1 - variation), 0), max(b * (1 - variation), 0))
    light = (min(r * (1 + variation) * saturation_boost, 1), min(g * (1 + variation) * saturation_boost, 1), min(b * (1 + variation) * saturation_boost, 1))
    ramp.color_ramp.elements[0].color = (*dark, 1.0)
    ramp.color_ramp.elements[1].color = (*light, 1.0)

    # --- baked cavity/AO shading multiplied into albedo ---
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.location = (-500, -50)
    ao.samples = 8
    ao_ramp = nodes.new("ShaderNodeValToRGB")
    ao_ramp.location = (-300, -50)
    ao_ramp.color_ramp.elements[0].position = 0.0
    ao_ramp.color_ramp.elements[0].color = (1 - cavity_strength, 1 - cavity_strength, 1 - cavity_strength, 1)
    ao_ramp.color_ramp.elements[1].position = 1.0
    ao_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    links.new(ao.outputs["Color"], ao_ramp.inputs["Fac"])

    mix_cavity = nodes.new("ShaderNodeMixRGB")
    mix_cavity.location = (-100, 100)
    mix_cavity.blend_type = "MULTIPLY"
    mix_cavity.inputs["Fac"].default_value = 1.0
    links.new(ramp.outputs["Color"], mix_cavity.inputs["Color1"])
    links.new(ao_ramp.outputs["Color"], mix_cavity.inputs["Color2"])

    # --- cool fresnel rim tint ---
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.location = (-300, -300)
    fresnel.inputs["IOR"].default_value = 1.45
    rim_mix = nodes.new("ShaderNodeMixRGB")
    rim_mix.location = (150, 0)
    rim_mix.blend_type = "ADD"
    rim_mix.inputs["Color2"].default_value = (*rim_color, 1.0)
    links.new(mix_cavity.outputs["Color"], rim_mix.inputs["Color1"])
    fresnel_scale = nodes.new("ShaderNodeMath")
    fresnel_scale.location = (-100, -300)
    fresnel_scale.operation = "MULTIPLY"
    fresnel_scale.inputs[1].default_value = rim_strength
    links.new(fresnel.outputs["Fac"], fresnel_scale.inputs[0])
    links.new(fresnel_scale.outputs["Value"], rim_mix.inputs["Fac"])

    links.new(rim_mix.outputs["Color"], bsdf.inputs["Base Color"])

    # --- roughness micro-variation ---
    rough_ramp = nodes.new("ShaderNodeValToRGB")
    rough_ramp.location = (-500, 400)
    rough_ramp.color_ramp.elements[0].color = (roughness * 0.8,) * 3 + (1,)
    rough_ramp.color_ramp.elements[1].color = (min(roughness * 1.2, 1.0),) * 3 + (1,)
    links.new(noise.outputs["Fac"], rough_ramp.inputs["Fac"])
    links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def apply_material(obj, mat, slot_index=None):
    """
    Assign `mat` to `obj`. With slot_index=None (the common single-material-
    per-part case), `mat` is appended as a new slot AND every face is
    repointed to that slot — not just appended and left alone. Blender's
    Boolean modifier (EXACT solver) can leave a fresh, never-materialed
    mesh with an empty material slot 0 already present after
    modifier_apply (observed on a carve_from_silhouettes INTERSECT result
    built from two materialless volumes); appending without reassigning
    then puts the real material in slot 1 while every face still points at
    the empty slot 0, so the object renders flat white with the material
    silently doing nothing — found in loop-iteration testing by comparing
    `object.material_slots` against the render, not by looking at the
    node graph (which was fine). Reassigning every face here is safe for
    the toolkit's actual usage pattern (always exactly one apply_material
    call per part, before join_objects/assign_material_by_region) and
    fixes the bug regardless of why the phantom slot appeared.
    Pass an explicit slot_index (e.g. building up multiple regions ahead
    of assign_material_by_region) to append without touching face indices.
    """
    if slot_index is None:
        obj.data.materials.append(mat)
        idx = len(obj.data.materials) - 1
        for poly in obj.data.polygons:
            poly.material_index = idx
    else:
        while len(obj.data.materials) <= slot_index:
            obj.data.materials.append(None)
        obj.data.materials[slot_index] = mat
    return mat


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Image-driven geometry (no neural reconstruction — classic non-ML techniques)
# ---------------------------------------------------------------------------

def add_displace_from_heightmap(obj, image_path, strength=0.3, midlevel=0.5):
    """Bas-relief: push a subdivided plane/mesh's surface by a grayscale heightmap."""
    img = bpy.data.images.load(image_path, check_existing=True)
    tex = bpy.data.textures.new(f"{obj.name}_height", type="IMAGE")
    tex.image = img
    mod = obj.modifiers.new("Displace", type="DISPLACE")
    mod.texture = tex
    mod.strength = strength
    mod.mid_level = midlevel
    mod.texture_coords = "UV"
    return mod


def extrude_silhouette(mask_image_path, name="Cutout", depth=0.3, resolution=128):
    """
    'Cardboard cutout' 3D from a photo: the silhouette extruded to a flat
    slab of `depth` thickness. A stylized approximation, not a full
    reconstruction — see reference/polygon_modeling.md.

    Delegates to extrude_silhouette_volume(axis="top") for a clean,
    binary-sampled boundary. An earlier version built this via a Displace
    modifier (image-texture height source) + Solidify; that produced a
    jagged "fringe" of partial-height spikes all around the silhouette
    boundary REGARDLESS of resolution or a follow-up bevel (confirmed by
    directly inspecting vertex heights, not just the render — found in
    loop-iteration testing). Root cause: the Displace modifier bilinearly
    interpolates the mask texture, and a raster silhouette's curved/angled
    boundary essentially never aligns with the mesh's rectangular vertex
    grid, so a large fraction of boundary vertices sample a partial
    in-between value no matter how fine the grid or how big a bevel you
    add afterward. extrude_silhouette_volume avoids this by sampling the
    mask as a hard binary per-cell value (no texture interpolation at all)
    — the same technique already validated in carve_from_silhouettes.

    Note the resulting object is centered on the origin along its depth
    axis (spans -depth/2..+depth/2), not resting on the ground at z=0 like
    the old Displace-based version did — reposition it if you need that.
    """
    return extrude_silhouette_volume(mask_image_path, axis="top", size=2.0,
                                      depth=depth, resolution=resolution, name=name)


def _sample_mask_foreground(image_path, resolution):
    """Load a mask via Blender's own image loader (no PIL inside bpy's python)
    and return a resolution x resolution grid of foreground booleans, sampled
    at each cell's center. image.pixels is row-major from the BOTTOM row."""
    img = bpy.data.images.load(image_path, check_existing=True)
    img_w, img_h = img.size
    if img_w == 0 or img_h == 0:
        raise ValueError(f"could not read image size from {image_path!r}")
    channels = img.channels
    pixels = img.pixels[:]

    def sample(u, v_from_bottom):
        x = min(int(u * img_w), img_w - 1)
        y = min(int(v_from_bottom * img_h), img_h - 1)
        idx = (y * img_w + x) * channels
        return pixels[idx] > 0.5

    grid = [[sample((i + 0.5) / resolution, (j + 0.5) / resolution) for i in range(resolution)]
            for j in range(resolution)]
    return grid, img_w, img_h


_AXIS_BASES = {
    # (u_axis, v_axis, sweep_axis) — u/v span the cross-section (v is world-up),
    # sweep_axis is the direction the silhouette is extruded through.
    "front": (Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0))),
    "side": (Vector((0, 1, 0)), Vector((0, 0, 1)), Vector((1, 0, 0))),
    "top": (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))),
}


def extrude_silhouette_volume(mask_image_path, axis="front", size=2.0, depth=2.5, resolution=48, name="Volume"):
    """
    A solid prism whose cross-section matches a foreground silhouette mask,
    swept the full `depth` through `axis` — the classical shape-from-
    silhouette "generalized cone", approximated here as a straight prism
    (an orthographic simplification of the true perspective visual cone,
    appropriate for roughly-orthographic reference photos). This is what
    carve_from_silhouettes intersects across views to build a visual hull.

    axis: "front" (cross-section in the XZ plane, swept along Y),
          "side" (cross-section in YZ, swept along X),
          "top" (cross-section in XY, swept along Z).
    size: the cross-section's larger dimension, in world units, centered on
    the origin — pass the SAME size across all views of one carve so their
    volumes actually overlap where the subject overlaps.
    """
    import bmesh

    if axis not in _AXIS_BASES:
        raise ValueError(f"axis must be one of {list(_AXIS_BASES)}, got {axis!r}")
    u_axis, v_axis, sweep_axis = _AXIS_BASES[axis]

    grid, img_w, img_h = _sample_mask_foreground(mask_image_path, resolution)
    aspect = img_w / img_h if img_h else 1.0
    size_u, size_v = (size, size / aspect) if aspect >= 1.0 else (size * aspect, size)

    bm = bmesh.new()
    verts = [[None] * (resolution + 1) for _ in range(resolution + 1)]
    for j in range(resolution + 1):
        for i in range(resolution + 1):
            u, v = i / resolution, j / resolution
            co = (((u - 0.5) * size_u) * u_axis + ((v - 0.5) * size_v) * v_axis
                  - (depth / 2) * sweep_axis)
            verts[j][i] = bm.verts.new(co)
    bm.verts.ensure_lookup_table()

    faces = []
    for j in range(resolution):
        for i in range(resolution):
            if grid[j][i]:
                faces.append(bm.faces.new((verts[j][i], verts[j][i + 1], verts[j + 1][i + 1], verts[j + 1][i])))

    if not faces:
        bm.free()
        raise ValueError(f"{mask_image_path!r} has no foreground pixels above threshold — check the mask")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    extruded = bmesh.ops.extrude_face_region(bm, geom=faces)
    new_verts = [v for v in extruded["geom"] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=new_verts, vec=sweep_axis * depth)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def carve_from_silhouettes(views, size=2.0, depth_margin=1.3, resolution=48, name="Carved"):
    """
    Classical shape-from-silhouette / visual-hull reconstruction: intersect
    solid silhouette prisms from multiple calibrated views. A real volumetric
    result from multiple photos, not a flat relief — but still a coarse
    approximation (concavities invisible from every view can't be recovered;
    this is a documented limitation of visual-hull reconstruction itself, not
    just this implementation). See reference/polygon_modeling.md.

    views: list of {"mask": path, "axis": "front"|"side"|"top"}, >= 2 entries
    (ideally 3 — front+side+top — 2 views alone leaves a lot of "extruded
    blob" ambiguity along the missing axis). The source photos must be
    roughly orthographic and consistently framed (same subject scale/crop
    across views) — mismatched framing carves a wrong or empty shape.
    """
    if len(views) < 2:
        raise ValueError("carve_from_silhouettes needs at least 2 views")
    depth = size * depth_margin
    volumes = [
        extrude_silhouette_volume(v["mask"], axis=v["axis"], size=size, depth=depth,
                                   resolution=resolution, name=f"{name}_{v['axis']}")
        for v in views
    ]
    return boolean_op(volumes, operation="INTERSECT", name=name)


# ---------------------------------------------------------------------------
# Lighting, camera, render, export
# ---------------------------------------------------------------------------

def add_three_point_lighting(target=(0, 0, 0), radius=6.0, key_energy=120, fill_energy=40, rim_energy=70,
                              bounce_energy=18,
                              key_color=(1.0, 0.95, 0.85), fill_color=(0.75, 0.82, 1.0),
                              rim_color=(0.85, 0.9, 1.0), bounce_color=(0.7, 0.68, 0.65)):
    """
    Key/Fill/Rim + a dim underside Bounce light. Without the bounce light,
    every object's underside/bottom faces get essentially no direct light
    (all three of Key/Fill/Rim sit above the target) and render near-black
    regardless of the actual geometry there — found this in loop-iteration
    testing, where a symmetric carved shape's bottom rendered looking flat/
    cut off purely from lighting, even though the mesh itself was confirmed
    symmetric. `bounce_energy=0` restores the old lights-only-from-above look.
    """
    tx, ty, tz = target
    specs = [
        ("Key", (tx + radius * 0.6, ty - radius * 0.8, tz + radius * 0.9), key_energy, key_color),
        ("Fill", (tx - radius * 0.9, ty - radius * 0.3, tz + radius * 0.4), fill_energy, fill_color),
        ("Rim", (tx - radius * 0.2, ty + radius * 0.9, tz + radius * 0.7), rim_energy, rim_color),
        ("Bounce", (tx, ty - radius * 0.3, tz - radius * 0.6), bounce_energy, bounce_color),
    ]
    lights = []
    for lname, loc, energy, color in specs:
        light_data = bpy.data.lights.new(lname, type="AREA")
        light_data.energy = energy
        light_data.color = color
        light_data.size = radius * 0.5
        light_obj = bpy.data.objects.new(lname, light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc
        direction = Vector(target) - Vector(loc)
        light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        lights.append(light_obj)
    return lights


def setup_camera(location=(5, -5, 3.5), look_at=(0, 0, 0), lens=50):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = location
    direction = Vector(look_at) - Vector(location)
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam_obj
    return cam_obj


def world_bounds(objects):
    """
    True world-space bounding box center + size across objects, from each
    object's actual geometry (bound_box corners transformed by matrix_world)
    rather than its origin — an object's origin can sit anywhere relative to
    its mesh (e.g. loft_profile/build_profile_body place verts at absolute
    coordinates, leaving the object origin at world (0,0,0)), so origin
    averaging silently mis-centers framing.
    """
    mins = Vector((float("inf"),) * 3)
    maxs = Vector((float("-inf"),) * 3)
    for obj in objects:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            mins.x, mins.y, mins.z = min(mins.x, world_co.x), min(mins.y, world_co.y), min(mins.z, world_co.z)
            maxs.x, maxs.y, maxs.z = max(maxs.x, world_co.x), max(maxs.y, world_co.y), max(maxs.z, world_co.z)
    center = (mins + maxs) / 2
    size = max((maxs - mins).x, (maxs - mins).y, (maxs - mins).z)
    return center, size


def frame_camera_on_objects(cam_obj, objects, margin=1.3):
    """Nudge the camera back along its view axis until all objects fit in frame (rough heuristic)."""
    center, size = world_bounds(objects)
    direction = (cam_obj.location - center).normalized()
    cam_obj.location = center + direction * size * margin * 2.0


def render_still(filepath, resolution=(1024, 1024), samples=64, engine="CYCLES", transparent=False):
    scene = bpy.context.scene
    set_stylized_color_management()
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
        try:
            scene.cycles.device = "CPU"
        except Exception:
            pass
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.film_transparent = transparent
    scene.render.filepath = filepath
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    return filepath


def render_turntable(filepath_prefix, objects, frames=8, radius=None, height=None, resolution=(768, 768), samples=48):
    """Render N evenly-spaced angles around the object(s) for quick visual review.
    `height` is an offset above the object's actual geometric center (not world Z=0),
    so framing stays correct regardless of where each object's origin sits.

    Leave `radius`/`height` as None (the default) to auto-fit from the
    objects' actual world-space size — fixed defaults tuned for one kind of
    subject (e.g. a roughly human-sized character) silently crop or
    under-fill the frame for anything a different shape (a tall thin prop,
    a small trinket); this was a real bug found in loop-iteration testing
    (a sword's blade tip clipped out of frame at a fixed radius/height that
    worked fine for character-sized subjects). Pass explicit values only
    when you specifically want manual control over the shot."""
    center, size = world_bounds(objects)
    if radius is None:
        radius = size * 1.7
    if height is None:
        height = size * 0.15

    cam = setup_camera((center.x + radius, center.y, center.z + height), tuple(center))
    add_three_point_lighting(tuple(center), radius=radius)
    set_world_background()

    paths = []
    for i in range(frames):
        angle = (2 * math.pi / frames) * i
        cam.location = (center.x + radius * math.cos(angle), center.y + radius * math.sin(angle), center.z + height)
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        path = f"{filepath_prefix}_{i:02d}.png"
        render_still(path, resolution=resolution, samples=samples)
        paths.append(path)
    return paths


def export_glb(filepath, selected_only=False):
    bpy.ops.export_scene.gltf(filepath=filepath, export_format="GLB", use_selection=selected_only)
    return filepath


def save_blend(filepath):
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    return filepath
