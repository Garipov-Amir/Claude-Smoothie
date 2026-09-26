"""
Stage — look-dev: game material from the authored textures + eyeballs.
<out>/baked.blend + textures -> <out>/lookdev.blend
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy

import bl_util as U
import eyes
import humanoid
import lookdev


def main(out_dir, prefix="T_Character"):
    bpy.ops.wm.open_mainfile(filepath=os.path.join(out_dir, "baked.blend"))
    humanoid.build(clothing=False)   # landmarks: eye centers and size
    T = os.path.join(out_dir, "textures")
    mat = lookdev.game_material("M_Character", T, prefix, skin_sss_mask=os.path.join(T, f"{prefix}_SkinMask.png"))
    for nm in ("SK_Character_LOD0", "SK_Character_LOD0_quads", "SK_Character_LOD2", "SK_Character_LOD4"):
        for piece in (nm, nm.replace("SK_Character_", "SK_Character_Pouch_")):
            o = bpy.data.objects.get(piece)
            if o:
                U.set_material(o, mat)
    em = lookdev.eye_material("M_Eye", os.path.join(T, "T_Eye_BaseColor.png"))
    for side in (1, -1):
        nm = f"Eye_{'L' if side > 0 else 'R'}"
        if nm not in bpy.data.objects:
            U.set_material(eyes.make_eye(side, nm), em)
    for nm in ("HighPoly", "HighPoly_Pouch"):
        hp = bpy.data.objects.get(nm)
        if hp:
            hp.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "lookdev.blend"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0])
