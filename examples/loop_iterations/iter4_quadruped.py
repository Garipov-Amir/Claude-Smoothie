"""Loop iteration 4: a quadruped creature (4-legged), exercising the loft
toolkit on a horizontal-spine body plan — Sonic and the iter1 humanoid were
both vertical-spine bipeds; this tests a horizontal torso with 4 symmetric
tapered legs and mirror-able proportions."""
import sys
import os
import math

SCRIPT_DIR = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/scripts"
sys.path.insert(0, SCRIPT_DIR)
import bpy_stylized_kit as k
from mathutils import Vector

OUT = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/examples/loop_iterations/out"
os.makedirs(OUT, exist_ok=True)

k.new_scene()

mat_fur = k.build_hand_painted_material("Fur", k.hex_to_rgb("#B8895C"), cavity_strength=0.45, roughness=0.65)
mat_belly = k.build_hand_painted_material("Belly", k.hex_to_rgb("#EAD9BE"), cavity_strength=0.3, roughness=0.6)
mat_dark = k.build_hand_painted_material("Dark", k.hex_to_rgb("#241C16"), cavity_strength=0.2, roughness=0.4)

# horizontal spine: tail -> hindquarters -> chest -> neck -> head, along Y
tail_co, snout_co = (0, -0.75, 0.55), (0, 0.95, 0.62)
body = k.build_profile_body([
    (tail_co, 0.05),
    ((0, -0.45, 0.50), 0.22),   # hindquarters
    ((0, -0.05, 0.48), 0.28),   # torso mid
    ((0, 0.35, 0.50), 0.24),    # chest/shoulders
    ((0, 0.62, 0.58), 0.16),    # neck
    ((0, 0.80, 0.62), 0.18),    # head
    *k.rounded_cap_points((0, 0.80, 0.62), 0.18, snout_co, n=1),
    (snout_co, 0.06),
], segments=16, name="Body")
k.apply_material(body, mat_fur)
k.assign_material_by_region(body, mat_belly, lambda p: p.z < 0.42)

core_parts = [body]
accessories = []


def add_core(fn, mat, name, **kw):
    obj = fn(name=name, **kw)
    k.apply_material(obj, mat)
    core_parts.append(obj)
    return obj


def add_accessory(obj, mat, name):
    obj.name = name
    k.apply_material(obj, mat)
    accessories.append(obj)
    return obj


# 4 legs: front pair near chest, back pair near hindquarters — straight
# tapered limbs, standard quadruped stance
leg_specs = [
    ("FL", (0.16, 0.30, 0.45), (0.17, 0.32, 0.0)),
    ("FR", (-0.16, 0.30, 0.45), (-0.17, 0.32, 0.0)),
    ("BL", (0.17, -0.35, 0.45), (0.18, -0.35, 0.0)),
    ("BR", (-0.17, -0.35, 0.45), (-0.18, -0.35, 0.0)),
]
for label, top, bottom in leg_specs:
    leg = k.add_tapered_limb(top, bottom, 0.10, 0.07, segments=8, name=f"Leg.{label}")
    k.apply_material(leg, mat_fur)
    core_parts.append(leg)

# ears: small tapered lofts on the head
for side in (1, -1):
    ear = add_core(
        lambda name, side=side, **kw: k.build_profile_body([
            (Vector((0.10 * side, 0.75, 0.76)), 0.06),
            (Vector((0.15 * side, 0.78, 0.92)), 0.01),
        ], segments=8, name=name),
        mat_fur, f"Ear.{side}",
    )

nose = k.add_primitive("ico_sphere", location=(0, 0.99, 0.60), scale=(0.045, 0.05, 0.04), name="Nose", subdivisions=2)
add_accessory(nose, mat_dark, "Nose")

print(f"Unioning {len(core_parts)} core parts...")
model = k.boolean_op(core_parts, operation="UNION", name="Quadruped")
k.shade_smooth(model, auto_smooth_angle=45)
for acc in accessories:
    k.shade_smooth(acc, auto_smooth_angle=45)
model = k.join_objects([model] + accessories, name="Quadruped")

print("MODEL verts", len(model.data.vertices), "polys", len(model.data.polygons))
print("MATERIALS", [m.name if m else None for m in model.data.materials])

k.set_world_background()
paths = k.render_turntable(f"{OUT}/iter4_quadruped", [model], frames=6, resolution=(800, 800), samples=48)
k.save_blend(f"{OUT}/iter4_quadruped.blend")
print("ITER4_OK", paths)
