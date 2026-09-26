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
    np.savez_compressed(os.path.join(out_dir, "highpoly.npz"), verts=v, faces=f,
                        joints=np.array([(k, *np.asarray(val, float)) for k, val in J.items()], dtype=object))
    print(f"[{time.time() - t:6.1f}s] highpoly: {len(v)} verts, {len(f)} tris", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0], float(args[1]) if len(args) > 1 else 0.0012)
