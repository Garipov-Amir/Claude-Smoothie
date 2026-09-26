"""
Fast look at a character spec before the full build: body + garments +
hair + pieces sculpted at a coarse voxel (3 mm) and rendered in clay —
front, 3/4, side, back and two face close-ups. ~1-2 minutes.

    python preview_spec.py <spec.json | preset> <out.png> [voxel_m]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(spec_src, out_png, voxel=0.003):
    t = time.time()
    import spec as SP
    import humanoid
    import pieces
    import sdf
    S = SP.resolve(spec_src)
    m, J = humanoid.build(S)
    print(f"[{time.time() - t:5.1f}s] {S['name']}: pieces {[p['name'] for p in pieces.PIECES]}", flush=True)
    v, f = sdf.sparse_polygonize(m, *humanoid.bounds(), voxel=voxel, brick=32)
    import bpy  # noqa: F401  (bl_util needs it)
    import bl_util as U
    U.reset_scene()
    objs = [U.mesh_from_arrays("Sculpt", v, f)]
    for p in pieces.PIECES:
        pv, pf = sdf.sparse_polygonize(p["model"], p["lo"], p["hi"], voxel=voxel * 0.6, brick=24, workers=1)
        if len(pf):
            objs.append(U.mesh_from_arrays(p["name"], pv, pf))
    clay = U.clay_material(color=(0.58, 0.56, 0.53))
    for o in objs:
        o.data.shade_smooth()
        U.set_material(o, clay)
    from landmarks import LM
    tmp = os.path.splitext(out_png)[0] + "_parts"
    os.makedirs(tmp, exist_ok=True)
    head = (0.0, float(LM["eye_l"][1]) + 0.07 * LM["s_head"], float(LM["eye_l"][2]))
    paths = U.render_views(f"{tmp}/body", objs, views=("front", "3q", "left", "3q_back"), resolution=(500, 800),
                           samples=12, ortho=False)
    paths += U.render_views(f"{tmp}/head", objs, views=("front", "3q"), resolution=(500, 800), samples=12,
                            focus=(head, 0.34 * LM["s_head"]), ortho=False)
    U.contact_sheet(paths, out_png)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.time() - t:5.1f}s] wrote {out_png}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0], args[1], float(args[2]) if len(args) > 2 else 0.003)
