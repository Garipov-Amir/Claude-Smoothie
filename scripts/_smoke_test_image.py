"""Internal smoke test for the image-driven pipeline.
Run: scripts/run_blender.sh scripts/_smoke_test_image.py <silhouette.png> <heightmap.png> <out_prefix>
"""
import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy_stylized_kit as k

argv = sys.argv[sys.argv.index("--") + 1:]
silhouette_path, heightmap_path, out_prefix = argv[0], argv[1], argv[2]
os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

k.new_scene()

cutout = k.extrude_silhouette(silhouette_path, name="PhotoCutout", depth=0.4, resolution=48)
mat = k.build_hand_painted_material("CutoutMat", k.hex_to_rgb("#E07A3E"))
k.apply_material(cutout, mat)
k.shade_smooth(cutout, auto_smooth_angle=40)

k.set_world_background()
paths = k.render_turntable(out_prefix, [cutout], frames=2, radius=4, height=1.5, resolution=(512, 512), samples=16)
print("SMOKE_TEST_IMAGE_OK", paths)
