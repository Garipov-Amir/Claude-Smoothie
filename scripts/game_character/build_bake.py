"""
Stage 3 — bake high -> low: <out>/lowpoly.blend -> <out>/bakes_<res>.npz (+ previews)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bpy

import bl_util as U
import bake


def triangulate(obj):
    m = obj.modifiers.new("Tri", "TRIANGULATE")
    m.quad_method = "BEAUTY"
    m.ngon_method = "BEAUTY"
    m.keep_custom_normals = True
    U.apply_modifiers(obj)


def main(out_dir, res=2048):
    t = time.time()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(out_dir, "lowpoly.blend"))
    log = lambda s: print(f"[{time.time() - t:6.1f}s] {s}", flush=True)
    # (high-poly, low-poly) per mesh piece: each piece bakes only from its
    # own high-poly, then the bakes are merged in the shared UV atlas
    pairs = [("HighPoly", "SK_Character_LOD0"), ("HighPoly_Pouch", "SK_Character_Pouch_LOD0")]
    pairs = [(bpy.data.objects[h], bpy.data.objects[l]) for h, l in pairs
             if h in bpy.data.objects and l in bpy.data.objects]
    his = [h for h, _ in pairs]
    for _h, low in pairs:
        # keep the quad source, bake and ship the triangulated game mesh
        if f"{low.name}_quads" not in bpy.data.objects:
            q = U.duplicate(low, f"{low.name}_quads")
            q.hide_render = True
            triangulate(low)
    parts = []
    for hp, low in pairs:
        # every high-poly stays render-visible, so AO gets contact shadows
        # between pieces (the pouch darkens the hip); only the pair is selected
        for o in bpy.data.objects:
            if o.type == "MESH":
                o.hide_render = o not in his and o is not low
        log(f"baking {hp.name} -> {low.name}")
        parts.append(bake.bake_maps(hp, low, res=res, log=log))
    maps = parts[0] if len(parts) == 1 else bake.composite(parts)
    np.savez_compressed(os.path.join(out_dir, f"bakes_{res}.npz"), **{k: v.astype(np.float32) for k, v in maps.items()})
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "baked.blend"))
    # quick-look PNGs
    from PIL import Image
    prev = os.path.join(out_dir, "bake_preview")
    os.makedirs(prev, exist_ok=True)
    for k, v in maps.items():
        a = v
        if k == "position":
            a = (v - v.min((0, 1))) / (v.max((0, 1)) - v.min((0, 1)) + 1e-9)
        if k == "curvature":
            a = np.clip((v - 0.5) * 4 + 0.5, 0, 1)
        if k == "part":
            a = v / max(1.0, float(v.max()))
        a = np.clip(a, 0, 1)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        Image.fromarray((a[::-1] * 255).astype(np.uint8)).save(os.path.join(prev, f"{k}.png"))
    print(f"[{time.time() - t:6.1f}s] done", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0], int(args[1]) if len(args) > 1 else 2048)
