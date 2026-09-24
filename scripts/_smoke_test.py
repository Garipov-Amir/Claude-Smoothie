"""Internal smoke test: exercises the core bpy_stylized_kit code paths and renders
one preview image. Not part of the public API — used to validate the toolkit
against the installed Blender version.

Run: scripts/run_blender.sh scripts/_smoke_test.py /tmp/smoke_out
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy_stylized_kit as k

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["/tmp/smoke_out"]
out_prefix = argv[0]
os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

k.new_scene()

body = k.add_primitive("ico_sphere", location=(0, 0, 1), scale=(1, 1, 1.2), name="Body", subdivisions=3)
k.shade_smooth(body, auto_smooth_angle=35)
k.add_subsurf(body, levels=1, render_levels=2)

ear1 = k.add_primitive("cone", location=(0.5, 0, 2.1), rotation=(0.3, 0, 0), scale=(0.25, 0.25, 0.6), name="Ear.L")
ear2 = k.add_primitive("cone", location=(-0.5, 0, 2.1), rotation=(-0.3, 0, 0), scale=(0.25, 0.25, 0.6), name="Ear.R")

mat = k.build_hand_painted_material("FoxOrange", k.hex_to_rgb("#E07A3E"))
for obj in (body, ear1, ear2):
    k.apply_material(obj, mat)

model = k.join_objects([body, ear1, ear2], name="Fox")
k.add_bevel(model, width=0.01, segments=2)

k.set_world_background()
paths = k.render_turntable(out_prefix, [model], frames=2, radius=5, height=2.2, resolution=(512, 512), samples=16)

print("SMOKE_TEST_OK", paths)
