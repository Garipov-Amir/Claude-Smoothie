"""Loop iteration 5: single-photo bas-relief (extrude_silhouette) at higher
resolution + add_bevel — testing polygon_modeling.md's own suggestion
("follow with add_bevel/add_subsurf for a softer look") which has never
actually been verified."""
import sys
import os

SCRIPT_DIR = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/scripts"
sys.path.insert(0, SCRIPT_DIR)
import bpy_stylized_kit as k

OUT = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/examples/loop_iterations/out"
MASK = "/tmp/claude-501-3dskills/imgprep_out/silhouette.png"
os.makedirs(OUT, exist_ok=True)

k.new_scene()

cutout = k.extrude_silhouette(MASK, name="Cutout", depth=0.35, resolution=96)
k.add_bevel(cutout, width=0.015, segments=3)
k.shade_smooth(cutout, auto_smooth_angle=50)

mat = k.build_hand_painted_material("CutoutMat", k.hex_to_rgb("#3E9E8F"))
k.apply_material(cutout, mat)

print("CUTOUT verts", len(cutout.data.vertices), "polys", len(cutout.data.polygons))

k.set_world_background()
paths = k.render_turntable(f"{OUT}/iter5_relief", [cutout], frames=3, resolution=(700, 700), samples=32)
print("ITER5_OK", paths)
