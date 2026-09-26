"""
Stage 1 — high-poly sculpt -> <out>/highpoly.npz + highpoly_<piece>.npz
(pure numpy, no bpy needed). The character comes from <out>/spec.json.

    python build_highpoly.py <out_dir> [voxel_m]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import humanoid
import sdf
import spec as SP


def main(out_dir, voxel=0.0012):
    os.makedirs(out_dir, exist_ok=True)
    t = time.time()
    log = lambda s: print(f"[{time.time() - t:6.1f}s] {s}", flush=True)
    S = SP.from_out(out_dir)
    SP.save(S, out_dir)
    model, J = humanoid.build(S)
    import pieces
    log(f"{S['name']}: {len(S['outfit'])} garment layers, pieces {[p['name'] for p in pieces.PIECES]}")
    v, f = sdf.sparse_polygonize(model, *humanoid.bounds(), voxel=voxel, log=log)
    for p in pieces.PIECES:
        pv, pf = sdf.sparse_polygonize(p["model"], p["lo"], p["hi"], voxel=voxel * 0.8, brick=24, workers=1)
        np.savez_compressed(os.path.join(out_dir, f"highpoly_{p['name'].lower()}.npz"), verts=pv, faces=pf)
        log(f"{p['name']} highpoly: {len(pf)} tris")
    np.savez_compressed(os.path.join(out_dir, "highpoly.npz"), verts=v, faces=f,
                        joints=np.array([(k, *np.asarray(val, float)) for k, val in J.items()
                                         if not k.startswith("_") and np.shape(val) == (3,)], dtype=object))
    log(f"highpoly: {len(v)} verts, {len(f)} tris")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0], float(args[1]) if len(args) > 1 else 0.0012)
