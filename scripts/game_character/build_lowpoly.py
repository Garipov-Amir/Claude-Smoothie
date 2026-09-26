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
    U.reset_scene()
    J = humanoid.skeleton()
    hp = load_highpoly(out_dir)
    print(f"[{time.time() - t:.1f}s] highpoly loaded", flush=True)
    C, lod4, lod2, lod0 = retopo.build_lods(J, hp)
    for o in (lod4, lod2, lod0):
        print(f"[{time.time() - t:.1f}s] {o.name}: {len(o.data.polygons)} faces, {U.tri_count(o)} tris", flush=True)
    import uvs
    print("uv stats", uvs.uv_stats(lod0))
    hp.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "lowpoly.blend"))
    return lod4, lod2, lod0, hp


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0])
