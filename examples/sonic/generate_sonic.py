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

# --- materials (one per color region, reused across parts) ---
mat_blue = k.build_hand_painted_material("SonicBlue", k.hex_to_rgb("#1B63D8"), cavity_strength=0.5, roughness=0.45)
mat_peach = k.build_hand_painted_material("SonicPeach", k.hex_to_rgb("#FBD7A8"), cavity_strength=0.35, roughness=0.6)
mat_white = k.build_hand_painted_material("SonicWhite", k.hex_to_rgb("#F2F2EE"), cavity_strength=0.3, roughness=0.5)
mat_red = k.build_hand_painted_material("SonicRed", k.hex_to_rgb("#DE291E"), cavity_strength=0.4, roughness=0.4)
mat_dark = k.build_hand_painted_material("SonicDark", k.hex_to_rgb("#15171C"), cavity_strength=0.2, roughness=0.35)
mat_gold = k.build_hand_painted_material("SonicGold", k.hex_to_rgb("#E0A62C"), cavity_strength=0.3, roughness=0.35, metallic=0.5)

parts = []


def add(kind, loc, scale, mat, rot=(0, 0, 0), name="part", smooth=True, **extra):
    obj = k.add_primitive(kind, location=loc, rotation=rot, scale=scale, name=name, **extra)
    if smooth:
        k.shade_smooth(obj, auto_smooth_angle=45)
    k.apply_material(obj, mat)
    parts.append(obj)
    return obj


# --- torso & head ---
body = add("ico_sphere", (0, 0, 0.95), (0.52, 0.48, 0.62), mat_blue, name="Body", subdivisions=3)
belly = add("ico_sphere", (0, 0.34, 0.85), (0.34, 0.24, 0.42), mat_peach, name="Belly", subdivisions=3)
head = add("ico_sphere", (0, 0.02, 1.62), (0.48, 0.46, 0.46), mat_blue, name="Head", subdivisions=3)
muzzle = add("ico_sphere", (0, 0.38, 1.52), (0.30, 0.36, 0.26), mat_peach, name="Muzzle", subdivisions=3)
nose = add("ico_sphere", (0, 0.63, 1.53), (0.085, 0.09, 0.075), mat_dark, name="Nose", subdivisions=2)

# --- ears ---
add("cone", (0.36, -0.02, 2.02), (0.11, 0.07, 0.26), mat_blue, rot=(math.radians(-25), 0, math.radians(20)), name="Ear.L", vertices=8)
add("cone", (-0.36, -0.02, 2.02), (0.11, 0.07, 0.26), mat_blue, rot=(math.radians(-25), 0, math.radians(-20)), name="Ear.R", vertices=8)

# --- eyes: one shared white almond shape spanning the face (classic Sonic look),
# with two close-set pupils, instead of two separate bulging eyeballs ---
eye_shared = add("ico_sphere", (0, 0.42, 1.72), (0.34, 0.14, 0.19), mat_white,
                  rot=(math.radians(8), 0, 0), name="EyeShared", subdivisions=2)
for side in (1, -1):
    pupil = add("ico_sphere", (0.13 * side, 0.53, 1.70), (0.055, 0.05, 0.075), mat_dark,
                rot=(math.radians(8), 0, 0), name=f"Pupil.{side}", subdivisions=2)

# --- quills: backswept spikes from head down the back, classic silhouette ---
quill_specs = [
    # (loc, scale(x,y,z-length), rot_x_deg)
    ((0.0, -0.18, 1.95), (0.16, 0.30, 0.55), -35),
    ((0.22, -0.22, 1.80), (0.14, 0.30, 0.55), -30),
    ((-0.22, -0.22, 1.80), (0.14, 0.30, 0.55), -30),
    ((0.30, -0.25, 1.45), (0.13, 0.34, 0.60), -15),
    ((-0.30, -0.25, 1.45), (0.13, 0.34, 0.60), -15),
    ((0.32, -0.20, 1.05), (0.12, 0.34, 0.58), -2),
    ((-0.32, -0.20, 1.05), (0.12, 0.34, 0.58), -2),
]
for i, (loc, scale, rx) in enumerate(quill_specs):
    add("cone", loc, scale, mat_blue, rot=(math.radians(90 + rx), 0, 0), name=f"Quill.{i}", vertices=8)

# --- tail ---
add("cone", (0, -0.55, 0.85), (0.13, 0.22, 0.20), mat_blue, rot=(math.radians(80), 0, 0), name="Tail", vertices=8)

# --- arms & gloves (shoulder at torso surface, glove at wrist end, no gap) ---
for side in (1, -1):
    shoulder, wrist = Vector((0.46 * side, 0.10, 1.05)), Vector((0.62 * side, 0.16, 0.68))
    mid = (shoulder + wrist) / 2
    direction = wrist - shoulder
    length = direction.length
    rot = direction.to_track_quat("Z", "Y").to_euler()
    arm = add("cylinder", tuple(mid), (0.115, 0.115, length / 2), mat_blue,
              rot=(rot.x, rot.y, rot.z), name=f"Arm.{side}", vertices=10)
    glove = add("ico_sphere", wrist, (0.21, 0.21, 0.23), mat_white, name=f"Glove.{side}", subdivisions=2)

# --- legs & shoes ---
for side in (1, -1):
    leg = add("cylinder", (0.26 * side, 0.02, 0.42), (0.13, 0.13, 0.16), mat_blue, name=f"Leg.{side}", vertices=10)
    shoe = add("ico_sphere", (0.27 * side, 0.14, 0.16), (0.24, 0.36, 0.20), mat_red, name=f"Shoe.{side}", subdivisions=2)
    strap = add("ico_sphere", (0.27 * side, 0.10, 0.28), (0.22, 0.22, 0.10), mat_white, name=f"Strap.{side}", subdivisions=2)
    buckle = add("cylinder", (0.27 * side, -0.06, 0.28), (0.045, 0.045, 0.03), mat_gold,
                 rot=(math.radians(90), 0, 0), name=f"Buckle.{side}", vertices=12, smooth=False)

model = k.join_objects(parts, name="Sonic")

k.set_world_background()
paths = k.render_turntable(f"{OUT}/preview", [model], frames=6, radius=4.2, height=1.5,
                            resolution=(900, 900), samples=64)
k.save_blend(f"{OUT}/sonic.blend")
k.export_glb(f"{OUT}/sonic.glb")
print("SONIC_OK", paths)
