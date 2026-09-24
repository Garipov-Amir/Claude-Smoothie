"""Internal smoke test for the loft/polygon-modeling functions.
Run: scripts/run_blender.sh scripts/_smoke_test_loft.py <out_prefix>
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy_stylized_kit as k

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["/tmp/loft_smoke"]
out_prefix = argv[0]
os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

k.new_scene()

# a simple continuous "body" (tapered blob) as one loft, not a sphere stack
body = k.build_profile_body([
    ((0, 0, 0.0), 0.05),
    ((0, 0, 0.3), 0.45),
    ((0, 0, 0.8), 0.5),
    ((0, 0.05, 1.3), 0.35),
    ((0, 0.1, 1.65), 0.42),
    ((0, 0.15, 1.95), 0.05),
], segments=16, name="Body")

mat_blue = k.build_hand_painted_material("Blue", k.hex_to_rgb("#1B63D8"))
mat_peach = k.build_hand_painted_material("Peach", k.hex_to_rgb("#FBD7A8"))
k.apply_material(body, mat_blue)

# region-paint a peach "belly" patch on the lower-front half of the body, on
# the SAME continuous mesh (no separate object) — exercises assign_material_by_region
k.assign_material_by_region(body, mat_peach, lambda p: p.y > 0.03 and p.z < 1.5)

# a tapered limb as one continuous capsule, bending at an elbow
arm = k.add_tapered_limb((0.5, 0.05, 1.1), (0.75, 0.15, 0.55), 0.12, 0.09,
                          bend=(0.68, 0.1, 0.85), name="Arm")
k.apply_material(arm, mat_blue)

# fuse the arm into the body into one watertight mesh, verifying material
# assignment survives the boolean union
merged = k.boolean_union([body, arm], name="Merged")
k.shade_smooth(merged, auto_smooth_angle=40)

mat_names = [m.name if m else None for m in merged.data.materials]
mat_indices_used = sorted({p.material_index for p in merged.data.polygons})
print("MATERIALS_AFTER_UNION", mat_names, "indices_used", mat_indices_used)

k.set_world_background()
paths = k.render_turntable(out_prefix, [merged], frames=2, radius=4.5, height=1.5,
                            resolution=(512, 512), samples=16)
print("SMOKE_TEST_LOFT_OK", paths)
