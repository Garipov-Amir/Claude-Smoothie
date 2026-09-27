"""
Stage 2 — retopology: <out>/highpoly.npz -> <out>/lowpoly.blend
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bpy

import bl_util as U
import humanoid
import retopo


def load_highpoly(out_dir):
    d = np.load(os.path.join(out_dir, "highpoly.npz"), allow_pickle=True)
    hp = U.mesh_from_arrays("HighPoly", d["verts"], d["faces"])
    hp.data.shade_smooth()
    return hp


def main(out_dir, render=True):
    t = time.time()
    log = lambda s: print(f"[{time.time() - t:.1f}s] {s}", flush=True)
    U.reset_scene()
    import spec as SP
    import pieces
    import piece_lowpoly
    S = SP.from_out(out_dir)
    _m, J = humanoid.build(S)          # landmarks, garment regions, pieces
    hp = load_highpoly(out_dir)
    hp_pieces = piece_lowpoly.load_highpolys(out_dir, [p["name"] for p in pieces.PIECES])
    log("highpoly loaded")
    from landmarks import LM
    if LM.get("body") == "reference":
        # the sculpt sits on the reference's limit surface: wrap its base topology
        import wrap
        C, lod4, lod2, lod0 = wrap.build_lods(J, hp, humanoid.REF_CACHE, hp_pieces=hp_pieces, log=log)
    else:
        C, lod4, lod2, lod0 = retopo.build_lods(J, hp, hp_pieces=hp_pieces, log=log)
    for o in (lod4, lod2, lod0):
        log(f"{o.name}: {len(o.data.polygons)} faces, {U.tri_count(o)} tris")
    import uvs
    print("uv stats", uvs.uv_stats(lod0))
    hp.hide_render = True
    for o in hp_pieces.values():
        o.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "lowpoly.blend"))
    return lod4, lod2, lod0, hp


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0])
