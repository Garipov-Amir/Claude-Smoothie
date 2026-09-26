"""
Stage 5 — skeleton + skin + test animation: <out>/lookdev.blend -> <out>/rigged.blend
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy

import humanoid
import rig


def main(out_dir):
    t = time.time()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(out_dir, "lookdev.blend"))
    J = humanoid.skeleton()
    arm = rig.build_armature(J)
    lod0 = bpy.data.objects["SK_Character_LOD0"]
    rig.skin(lod0, arm)
    print(f"[{time.time() - t:.1f}s] LOD0 skinned:", rig.weight_report(lod0), flush=True)
    for nm, bone in (("Eye_L", "eye_l"), ("Eye_R", "eye_r")):
        o = bpy.data.objects.get(nm)
        if o:
            rig.bind_rigid(o, arm, bone)
    rig.walk_cycle(arm)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "rigged.blend"))
    print(f"[{time.time() - t:.1f}s] saved rigged.blend", flush=True)
    return arm, lod0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0])
