"""
Stage 1 — high-poly sculpt -> <out>/highpoly.npz (pure numpy, no bpy needed).

    python build_highpoly.py <out_dir> [voxel_m]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import humanoid
import sdf


def main(out_dir, voxel=0.0012):
    os.makedirs(out_dir, exist_ok=True)
    t = time.time()
    model, J = humanoid.build()
    v, f = sdf.sparse_polygonize(model, *humanoid.BOUNDS, voxel=voxel,
                                 log=lambda s: print(f"[{time.time() - t:6.1f}s] {s}", flush=True))
    import costume
    pm = costume.GEAR.get("pouch_model")
    if pm is not None:
        c = np.asarray(costume.GEAR["pouch_frame"][0])
        pv, pf = sdf.sparse_polygonize(pm, c - 0.09, c + 0.09, voxel=voxel * 0.8, brick=24, workers=1)
        np.savez_compressed(os.path.join(out_dir, "highpoly_pouch.npz"), verts=pv, faces=pf)
        print(f"[{time.time() - t:6.1f}s] pouch highpoly: {len(pf)} tris", flush=True)
    np.savez_compressed(os.path.join(out_dir, "highpoly.npz"), verts=v, faces=f,
                        joints=np.array([(k, *np.asarray(val, float)) for k, val in J.items()
                                         if not k.startswith("_") and np.shape(val) == (3,)], dtype=object))
    print(f"[{time.time() - t:6.1f}s] highpoly: {len(v)} verts, {len(f)} tris", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0], float(args[1]) if len(args) > 1 else 0.0012)
