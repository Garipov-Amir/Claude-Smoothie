"""
QA: re-import the shipped files into an empty scene and check they stand on
their own (skeleton, skin, animation, materials/textures travel with them).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy

import bl_util as U


def check(path):
    U.reset_scene()
    if path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        bpy.ops.import_scene.fbx(filepath=path)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    info = {
        "file": os.path.basename(path),
        "armatures": len(arms),
        "bones": len(arms[0].data.bones) if arms else 0,
        "meshes": sorted(o.name for o in meshes),
        "skinned": all(any(m.type == "ARMATURE" for m in o.modifiers) for o in meshes),
        "actions": sorted(a.name for a in bpy.data.actions),
        "images": sorted(i.name for i in bpy.data.images if i.size[0] > 0),
        "tris": sum(U.tri_count(o) for o in meshes),
    }
    return info, arms, meshes


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    for p in args:
        info, _a, _m = check(p)
        print(info)
