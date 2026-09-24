"""Loop iteration 1: a simple stylized humanoid (robot), exercising the
loft/boolean toolkit on a standard biped body plan — Sonic exercised a
quadruped-ish hedgehog body, this tests arms/legs as PRIMARY limbs with a
more upright torso, different proportions."""
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

mat_body = k.build_hand_painted_material("RobotTeal", k.hex_to_rgb("#2E9E8F"), cavity_strength=0.5, roughness=0.4)
mat_accent = k.build_hand_painted_material("RobotOrange", k.hex_to_rgb("#E8862E"), cavity_strength=0.4, roughness=0.45)
mat_dark = k.build_hand_painted_material("RobotDark", k.hex_to_rgb("#20242A"), cavity_strength=0.25, roughness=0.35)

# torso + neck + head as one continuous spine — upright biped proportions,
# unlike Sonic's forward-leaning hedgehog blob
head_co, head_r, head_top_co = (0, 0.02, 1.42), 0.24, (0, 0.0, 1.62)
body = k.build_profile_body([
    ((0, 0, 0.02), 0.10),   # waist taper
    ((0, 0, 0.35), 0.30),   # hips
    ((0, 0.01, 0.75), 0.34),  # chest (broad, upright)
    ((0, 0.02, 1.05), 0.22),  # neck-in
    ((0, 0.03, 1.15), 0.20),  # neck
    (head_co, head_r),        # head
    *k.rounded_cap_points(head_co, head_r, head_top_co),  # rounded dome instead of a sharp cone tip
    (head_top_co, 0.03),
], segments=18, name="Body")
k.apply_material(body, mat_body)
k.assign_material_by_region(body, mat_accent, lambda p: 0.55 < p.z < 0.95 and abs(p.x) < 0.4)

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


# arms: shoulder -> elbow-bend -> hand, proper biped arm proportions (longer,
# more vertical drop than Sonic's stubby side-arms)
for side in (1, -1):
    arm = k.add_tapered_limb((0.34 * side, 0.0, 1.00), (0.42 * side, 0.05, 0.35), 0.09, 0.06,
                              bend=(0.40 * side, 0.02, 0.68), segments=10, name=f"Arm.{side}")
    k.apply_material(arm, mat_body)
    core_parts.append(arm)
    hand = k.add_primitive("ico_sphere", location=(0.42 * side, 0.05, 0.30), scale=(0.08, 0.08, 0.10),
                            name=f"Hand.{side}", subdivisions=2)
    add_accessory(hand, mat_dark, f"Hand.{side}")

# legs: hip -> knee-bend -> foot, primary weight-bearing limbs (unlike
# Sonic's mostly-hidden stubby legs)
for side in (1, -1):
    leg = k.add_tapered_limb((0.16 * side, 0.0, 0.30), (0.18 * side, 0.02, -0.55), 0.12, 0.08,
                              bend=(0.17 * side, 0.0, -0.12), segments=10, name=f"Leg.{side}")
    k.apply_material(leg, mat_body)
    core_parts.append(leg)
    foot = add_core(
        lambda name, side=side, **kw: k.build_profile_body([
            (Vector((0.18 * side, -0.10, -0.62)), (0.09, 0.10)),
            (Vector((0.18 * side, 0.05, -0.63)), (0.11, 0.11)),
            (Vector((0.18 * side, 0.20, -0.60)), (0.07, 0.06)),
        ], segments=12, name=name),
        mat_dark, f"Foot.{side}",
    )

print(f"Unioning {len(core_parts)} core parts...")
model = k.boolean_op(core_parts, operation="UNION", name="Robot")
k.shade_smooth(model, auto_smooth_angle=45)
for acc in accessories:
    k.shade_smooth(acc, auto_smooth_angle=45)
model = k.join_objects([model] + accessories, name="Robot")

print("MODEL verts", len(model.data.vertices), "polys", len(model.data.polygons))
print("MATERIALS", [m.name if m else None for m in model.data.materials])

k.set_world_background()
paths = k.render_turntable(f"{OUT}/iter1_humanoid", [model], frames=6, radius=3.2, height=0.9,
                            resolution=(800, 800), samples=48)
k.save_blend(f"{OUT}/iter1_humanoid.blend")
print("ITER1_OK", paths)
