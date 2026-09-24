"""Loop iteration 3: carve_from_silhouettes with all 3 views (front+side+top)
— the "top" axis code path hasn't been exercised by anything so far."""
import sys
import os

SCRIPT_DIR = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/scripts"
sys.path.insert(0, SCRIPT_DIR)
import bpy_stylized_kit as k

OUT = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/examples/loop_iterations/out"
MASKS = "/tmp/claude-501-3dskills/carve3_test"
os.makedirs(OUT, exist_ok=True)

k.new_scene()

carved = k.carve_from_silhouettes([
    {"mask": os.path.join(MASKS, "front.png"), "axis": "front"},
    {"mask": os.path.join(MASKS, "side.png"), "axis": "side"},
    {"mask": os.path.join(MASKS, "top.png"), "axis": "top"},
], size=2.0, resolution=44, name="Carved3")
print("CARVED verts", len(carved.data.vertices), "polys", len(carved.data.polygons))
print("CARVED dimensions", tuple(carved.dimensions))

mat = k.build_hand_painted_material("CarvedMat", k.hex_to_rgb("#7B5EC9"))
k.apply_material(carved, mat)
k.shade_flat(carved)

k.set_world_background()
paths = k.render_turntable(f"{OUT}/iter3_carve3view", [carved], frames=4, resolution=(700, 700), samples=32)
print("ITER3_OK", paths)
