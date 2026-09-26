"""
Stage 5 — skeleton + skin + test animation: <out>/lookdev.blend -> <out>/rigged.blend
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy

import bl_util as U
import humanoid
import rig

NAME = "SK_Character"


def attach_pieces(arm, lod0):
    """Separate pieces (their own meshes through retopo and bake) join the
    body mesh of each LOD, so every LOD ships as one skinned mesh / one draw
    call. LOD0 pieces get their weights here (a bone rigidly, or the skin
    under the anchor); the other LODs inherit them on export."""
    import pieces
    for p in pieces.PIECES:
        for level in ("LOD0", "LOD0_quads", "LOD2", "LOD4"):
            gear = bpy.data.objects.get(f"{NAME}_{p['name']}_{level}")
            body = bpy.data.objects.get(f"{NAME}_{level}")
            if gear is None or body is None:
                continue
            if level == "LOD0":
                if p["bind"] == "skin":
                    w = rig.bind_gear(gear, lod0, arm, anchor_frac=0.2 if p.get("anchor") == "top" else 1.0)
                else:
                    rig.bind_rigid(gear, arm, p["bind"])
                    w = {p["bind"]: 1.0}
                print(f"  {p['name']} weights:", w, flush=True)
            U.select_only([body, gear], body)
            bpy.ops.object.join()


def main(out_dir):
    t = time.time()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(out_dir, "lookdev.blend"))
    import spec as SP
    _m, J = humanoid.build(SP.from_out(out_dir))    # the skeleton of the body that was sculpted, + pieces
    arm = rig.build_armature(J)
    lod0 = bpy.data.objects[f"{NAME}_LOD0"]
    rig.skin(lod0, arm)
    print(f"[{time.time() - t:.1f}s] LOD0 skinned:", rig.weight_report(lod0), flush=True)
    attach_pieces(arm, lod0)
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
