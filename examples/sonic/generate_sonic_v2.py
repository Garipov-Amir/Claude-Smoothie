"""
Sonic the Hedgehog, v2 — continuous polygon modeling (Workflow A2) instead of
the primitive-assembly v1 (generate_sonic.py). Kept alongside v1 as a
before/after reference for the skill's loft/boolean technique. Same general
proportions/colors as v1, decided from scratch (no reference mesh data used).
"""
import sys
import os
import math

SCRIPT_DIR = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/scripts"
sys.path.insert(0, SCRIPT_DIR)
import bpy_stylized_kit as k
from mathutils import Vector

OUT = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/examples/sonic/out"
os.makedirs(OUT, exist_ok=True)

k.new_scene()

mat_blue = k.build_hand_painted_material("SonicBlue", k.hex_to_rgb("#1B63D8"), cavity_strength=0.5, roughness=0.45)
mat_peach = k.build_hand_painted_material("SonicPeach", k.hex_to_rgb("#FBD7A8"), cavity_strength=0.35, roughness=0.6)
mat_white = k.build_hand_painted_material("SonicWhite", k.hex_to_rgb("#F2F2EE"), cavity_strength=0.3, roughness=0.5)
mat_red = k.build_hand_painted_material("SonicRed", k.hex_to_rgb("#DE291E"), cavity_strength=0.4, roughness=0.4)
mat_dark = k.build_hand_painted_material("SonicDark", k.hex_to_rgb("#15171C"), cavity_strength=0.2, roughness=0.35)
mat_gold = k.build_hand_painted_material("SonicGold", k.hex_to_rgb("#E0A62C"), cavity_strength=0.3, roughness=0.35, metallic=0.5)

# --- torso + neck + head: ONE continuous lofted body instead of two spheres ---
body = k.build_profile_body([
    ((0, -0.05, 0.05), 0.06),     # tail-root taper
    ((0, 0.0, 0.35), 0.40),       # hips
    ((0, 0.02, 0.75), 0.52),      # chest (widest point)
    ((0, 0.06, 1.15), 0.34),      # neck-in
    ((0, 0.10, 1.42), 0.30),      # jaw/muzzle base
    ((0, 0.02, 1.72), 0.46),      # head (cranium, widest at top)
    ((0, -0.05, 2.00), 0.06),     # head-top taper
], segments=20, name="Body")
k.apply_material(body, mat_blue)

# peach belly patch + muzzle patch on the SAME continuous mesh
k.assign_material_by_region(body, mat_peach, lambda p: p.y > 0.20 and p.z < 1.55)
k.assign_material_by_region(body, mat_peach, lambda p: p.y > 0.30 and 1.40 < p.z < 1.62)

# silhouette-defining continuous parts: boolean-unioned into one watertight
# mesh so limbs/quills/ears/muzzle/shoes fuse seamlessly into the body.
# Small accessories (eyes, nose, gloves, straps, buckles) stay separate,
# simply overlapping — fusing tiny detail bits via boolean adds solver risk
# for no visible benefit (a hairline seam on a 3cm buckle isn't visible).
core_parts = [body]
accessories = []


def add_loft_part(fn, mat, name, **kw):
    obj = fn(name=name, **kw)
    k.apply_material(obj, mat)
    core_parts.append(obj)
    return obj


def add_accessory(obj, mat, name):
    obj.name = name
    k.apply_material(obj, mat)
    accessories.append(obj)
    return obj


# --- muzzle bump (small continuous taper forward of the face) ---
muzzle = add_loft_part(
    lambda name, **kw: k.build_profile_body([
        ((0, 0.40, 1.58), (0.10, 0.10)),
        ((0, 0.58, 1.53), (0.28, 0.24)),
        ((0, 0.66, 1.50), (0.10, 0.10)),
    ], segments=14, name=name),
    mat_peach, "Muzzle",
)
nose = k.add_primitive("ico_sphere", location=(0, 0.66, 1.51), scale=(0.08, 0.085, 0.07), name="Nose", subdivisions=2)
add_accessory(nose, mat_dark, "Nose")

# --- ears: small tapered lofts, not cones ---
for side in (1, -1):
    ear = add_loft_part(
        lambda name, **kw: k.build_profile_body([
            (Vector((0.30 * side, -0.02, 1.98)), 0.10),
            (Vector((0.42 * side, 0.02, 2.20)), 0.02),
        ], segments=10, name=name),
        mat_blue, f"Ear.{side}",
    )

# --- eyes: shared white almond band + close-set pupils (fixed in v1 already, kept) ---
eye_shared = k.add_primitive("ico_sphere", location=(0, 0.44, 1.76), scale=(0.32, 0.13, 0.18),
                              rotation=(math.radians(8), 0, 0), name="EyeShared", subdivisions=2)
add_accessory(eye_shared, mat_white, "EyeShared")
for side in (1, -1):
    pupil = k.add_primitive("ico_sphere", location=(0.12 * side, 0.55, 1.74), scale=(0.05, 0.045, 0.07),
                             rotation=(math.radians(8), 0, 0), name=f"Pupil.{side}", subdivisions=2)
    add_accessory(pupil, mat_dark, f"Pupil.{side}")

# --- quills: tapered lofts sweeping back, continuous surfaces instead of cones ---
quill_specs = [
    ((0.0, -0.10, 1.98), (0.0, -0.55, 2.55), 0.14),
    ((0.20, -0.15, 1.90), (0.55, -0.45, 2.30), 0.12),
    ((-0.20, -0.15, 1.90), (-0.55, -0.45, 2.30), 0.12),
    ((0.28, -0.15, 1.55), (0.75, -0.35, 1.65), 0.13),
    ((-0.28, -0.15, 1.55), (-0.75, -0.35, 1.65), 0.13),
    ((0.30, -0.10, 1.10), (0.80, -0.10, 1.05), 0.13),
    ((-0.30, -0.10, 1.10), (-0.80, -0.10, 1.05), 0.13),
]
for i, (start, end, r) in enumerate(quill_specs):
    quill = k.add_tapered_limb(start, end, r, 0.015, segments=8, name=f"Quill.{i}")
    k.apply_material(quill, mat_blue)
    core_parts.append(quill)

# --- tail: one tapered loft ---
tail = k.add_tapered_limb((0, -0.35, 0.35), (0, -0.75, 0.55), 0.14, 0.03, segments=10, name="Tail")
k.apply_material(tail, mat_blue)
core_parts.append(tail)

# --- arms: continuous tapered limbs with a bend at the elbow; glove is a separate accessory ---
for side in (1, -1):
    arm = k.add_tapered_limb((0.48 * side, 0.05, 1.05), (0.70 * side, 0.16, 0.68), 0.13, 0.10,
                              bend=(0.62 * side, 0.12, 0.88), segments=10, name=f"Arm.{side}")
    k.apply_material(arm, mat_blue)
    core_parts.append(arm)
    glove = k.add_primitive("ico_sphere", location=(0.70 * side, 0.17, 0.65), scale=(0.20, 0.20, 0.22),
                             name=f"Glove.{side}", subdivisions=2)
    add_accessory(glove, mat_white, f"Glove.{side}")

# --- legs + shoes: continuous tapered legs, shoes as lofted ovoid capsules (fused into core) ---
for side in (1, -1):
    leg = k.add_tapered_limb((0.24 * side, 0.0, 0.40), (0.26 * side, 0.05, 0.16), 0.13, 0.11,
                              segments=10, name=f"Leg.{side}")
    k.apply_material(leg, mat_blue)
    core_parts.append(leg)
    shoe = add_loft_part(
        lambda name, side=side, **kw: k.build_profile_body([
            (Vector((0.27 * side, -0.22, 0.15)), (0.19, 0.24)),
            (Vector((0.27 * side, 0.06, 0.17)), (0.24, 0.26)),
            (Vector((0.27 * side, 0.30, 0.11)), (0.16, 0.12)),
        ], segments=14, name=name),
        mat_red, f"Shoe.{side}",
    )
    strap = k.add_primitive("ico_sphere", location=(0.27 * side, 0.08, 0.27), scale=(0.22, 0.20, 0.09),
                             name=f"Strap.{side}", subdivisions=2)
    add_accessory(strap, mat_white, f"Strap.{side}")
    buckle = k.add_primitive("cylinder", location=(0.27 * side, -0.08, 0.27), scale=(0.045, 0.045, 0.03),
                              rotation=(math.radians(90), 0, 0), name=f"Buckle.{side}", vertices=12)
    add_accessory(buckle, mat_gold, f"Buckle.{side}")

print(f"Unioning {len(core_parts)} core parts...")
model = k.boolean_union(core_parts, name="Sonic")
k.shade_smooth(model, auto_smooth_angle=45)
for acc in accessories:
    k.shade_smooth(acc, auto_smooth_angle=45)
model = k.join_objects([model] + accessories, name="Sonic")

print("MODEL verts", len(model.data.vertices), "polys", len(model.data.polygons))
print("MATERIALS", [m.name if m else None for m in model.data.materials])

k.set_world_background()
paths = k.render_turntable(f"{OUT}/preview_v2", [model], frames=6, radius=4.2, height=1.5,
                            resolution=(900, 900), samples=64)
k.save_blend(f"{OUT}/sonic_v2.blend")
k.export_glb(f"{OUT}/sonic_v2.glb")
print("SONIC_V2_OK", paths)
