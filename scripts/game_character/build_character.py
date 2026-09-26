"""
Full pipeline: sculpt -> retopo/UV/LODs -> bake -> textures -> look-dev ->
rig/skin/anim -> LODs/export/validation.

    python build_character.py <out_dir> [--res 2048] [--voxel 0.0012] [--from STAGE]

Stages (each reads the previous stage's file, so any stage can be re-run):
    highpoly  -> highpoly.npz
    lowpoly   -> lowpoly.blend      (cage + LOD4/LOD2/LOD0 with UVs)
    bake      -> bakes_<res>.npz, baked.blend
    textures  -> textures/*.png
    lookdev   -> lookdev.blend      (game material + eyes)
    rig       -> rigged.blend       (skeleton, weights, walk cycle)
    export    -> export/*.fbx|glb, report.json
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGES = ["highpoly", "lowpoly", "bake", "textures", "lookdev", "rig", "export"]


def run(out, res=2048, voxel=0.0012, start="highpoly", stop="export"):
    t = time.time()
    todo = STAGES[STAGES.index(start):STAGES.index(stop) + 1]
    for st in todo:
        print(f"==== {st} ({time.time() - t:.0f}s)", flush=True)
        if st == "highpoly":
            import build_highpoly
            build_highpoly.main(out, voxel)
        elif st == "lowpoly":
            import build_lowpoly
            build_lowpoly.main(out)
        elif st == "bake":
            import build_bake
            build_bake.main(out, res)
        elif st == "textures":
            import build_textures
            build_textures.main(out, res)
        elif st == "lookdev":
            import build_lookdev
            build_lookdev.main(out)
        elif st == "rig":
            import build_rig
            build_rig.main(out)
        elif st == "export":
            import build_export
            build_export.main(out)
    print(f"==== done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--voxel", type=float, default=0.0012)
    ap.add_argument("--from", dest="start", default="highpoly", choices=STAGES)
    ap.add_argument("--to", dest="stop", default="export", choices=STAGES)
    a = ap.parse_args(argv)
    run(a.out, a.res, a.voxel, a.start, a.stop)
