"""Loop iteration 2: a stylized hard-surface prop (sword), exercising
add_bevel (hard edges) and mirror_x (symmetric crossguard) — neither has
been tested by anything so far this session."""
import sys
import os
import math

SCRIPT_DIR = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/scripts"
sys.path.insert(0, SCRIPT_DIR)
import bpy_stylized_kit as k

OUT = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/examples/loop_iterations/out"
os.makedirs(OUT, exist_ok=True)

k.new_scene()

mat_steel = k.build_hand_painted_material("Steel", k.hex_to_rgb("#C7CDD4"), cavity_strength=0.35,
                                           roughness=0.3, metallic=0.7)
mat_gold = k.build_hand_painted_material("Gold", k.hex_to_rgb("#D9A62C"), cavity_strength=0.3,
                                          roughness=0.35, metallic=0.6)
mat_leather = k.build_hand_painted_material("Leather", k.hex_to_rgb("#5A3825"), cavity_strength=0.5, roughness=0.7)

parts = []


def add(kind, loc, scale, mat, rot=(0, 0, 0), name="part", bevel=None, **extra):
    obj = k.add_primitive(kind, location=loc, rotation=rot, scale=scale, name=name, **extra)
    k.shade_smooth(obj, auto_smooth_angle=30)
    if bevel:
        k.add_bevel(obj, width=bevel, segments=3)
    k.apply_material(obj, mat)
    parts.append(obj)
    return obj


# blade: a tapered box (cube scaled thin+long, tip pinched via a second
# smaller cube stacked at the top for a simple point) with beveled edges
blade_body = add("cube", (0, 0, 0.75), (0.06, 0.02, 0.55), mat_steel, name="BladeBody", bevel=0.008)
blade_tip = add("cube", (0, 0, 1.34), (0.06, 0.02, 0.06), mat_steel, name="BladeTip",
                 rot=(0, 0, math.radians(45)), bevel=0.005)

# crossguard: model ONE half along +X, then mirror_x for the other half —
# exercises mirror_x for the first time this session
guard_half = k.add_primitive("cube", location=(0.14, 0, 0.18), scale=(0.16, 0.035, 0.025), name="GuardHalf")
k.shade_smooth(guard_half, auto_smooth_angle=30)
k.add_bevel(guard_half, width=0.006, segments=2)
k.mirror_x(guard_half)
k.apply_material(guard_half, mat_gold)
parts.append(guard_half)

# grip + pommel
grip = add("cylinder", (0, 0, 0.02), (0.035, 0.035, 0.16), mat_leather, name="Grip", vertices=12, bevel=0.005)
pommel = add("ico_sphere", (0, 0, -0.16), (0.055, 0.055, 0.055), mat_gold, name="Pommel", subdivisions=2)

model = k.join_objects(parts, name="Sword")
k.set_world_background()
paths = k.render_turntable(f"{OUT}/iter2_sword", [model], frames=4, resolution=(800, 800), samples=48)
k.save_blend(f"{OUT}/iter2_sword.blend")
print("MODEL verts", len(model.data.vertices), "polys", len(model.data.polygons))
print("ITER2_OK", paths)
