"""Internal smoke test for multi-view silhouette carving (carve_from_silhouettes).
Masks must already exist (generate them first with
_smoke_test_carve_make_masks.py under system python3, since Blender's bundled
python has no PIL).

Usage:
  python3 scripts/_smoke_test_carve_make_masks.py <outdir>
  scripts/run_blender.sh scripts/_smoke_test_carve.py <outdir>
"""
import sys
import os

import bpy  # noqa: F401 — confirms this only runs inside Blender

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import bpy_stylized_kit as k

argv = sys.argv[sys.argv.index("--") + 1:]
outdir = argv[0]

k.new_scene()

carved = k.carve_from_silhouettes(
    [
        {"mask": os.path.join(outdir, "front.png"), "axis": "front"},
        {"mask": os.path.join(outdir, "side.png"), "axis": "side"},
    ],
    size=2.0, resolution=40, name="Carved",
)
print("CARVED verts", len(carved.data.vertices), "polys", len(carved.data.polygons))
print("CARVED dimensions", tuple(carved.dimensions))

mat = k.build_hand_painted_material("CarvedMat", k.hex_to_rgb("#4E9BE0"))
k.apply_material(carved, mat)
k.shade_flat(carved)

# also smoke-test the DIFFERENCE path so boolean_op's generalization is exercised
cutter = k.add_primitive("cube", location=(0, 0, 0.9), scale=(0.3, 0.3, 0.3), name="Cutter")
diffed = k.boolean_op([carved, cutter], operation="DIFFERENCE", name="CarvedDiffed")
print("DIFFED verts", len(diffed.data.vertices), "polys", len(diffed.data.polygons))

k.set_world_background()
paths = k.render_turntable(os.path.join(outdir, "carve_preview"), [diffed], frames=4,
                            radius=3.5, height=0.5, resolution=(600, 600), samples=24)
print("SMOKE_TEST_CARVE_OK", paths)
