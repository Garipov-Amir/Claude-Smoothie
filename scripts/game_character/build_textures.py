"""
Stage 4 — author PBR textures from the bakes: <out>/bakes_<res>.npz -> <out>/textures/*.png
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import humanoid
import texture


def main(out_dir, res=2048, name="T_Character"):
    t = time.time()
    import spec as SP
    maps = dict(np.load(os.path.join(out_dir, f"bakes_{res}.npz")))
    S = SP.from_out(out_dir)
    model, J = humanoid.build(S)   # also seats garments and gear, which region_id needs
    tex = texture.author(maps, J, S, log=lambda s: print(f"[{time.time() - t:6.1f}s] {s}", flush=True))
    td = os.path.join(out_dir, "textures")
    os.makedirs(td, exist_ok=True)
    texture.save_png(tex["basecolor"], os.path.join(td, f"{name}_BaseColor.png"))
    texture.save_png(tex["normal_gl"], os.path.join(td, f"{name}_Normal_OpenGL.png"))
    texture.save_png(tex["normal_dx"], os.path.join(td, f"{name}_Normal_DirectX.png"))
    texture.save_png(tex["orm"], os.path.join(td, f"{name}_ORM.png"))
    h = tex["height"]
    texture.save_png(np.clip(h / 0.0006 + 0.5, 0, 1), os.path.join(td, f"{name}_Height.png"))
    np.save(os.path.join(out_dir, "region.npy"), tex["region"])
    import costume
    sss = (tex["region"] == costume.SKIN).astype(np.float32)[..., None]
    texture.save_png(sss, os.path.join(td, f"{name}_SkinMask.png"))
    # eyes
    import eyes
    texture.save_png(eyes.eye_texture(512, iris_color=tuple(SP.color(S["eyes"]["iris"]))), os.path.join(td, "T_Eye_BaseColor.png"))
    print(f"[{time.time() - t:6.1f}s] textures written to {td}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0], int(args[1]) if len(args) > 1 else 2048)
