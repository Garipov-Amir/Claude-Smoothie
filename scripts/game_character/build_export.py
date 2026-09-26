"""
Stage — LOD chain, engine exports and the validation report.
<out>/rigged.blend -> <out>/export/{SK_Character.fbx, SK_Character_LOD#.fbx, SK_Character.glb, report.json}
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

import bl_util as U
import humanoid
import lods
import rig

NAME = "SK_Character"


def join(objs, name):
    U.select_only(objs, objs[0])
    bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    o.name = o.data.name = name
    return o


def build_chain(arm):
    lod0 = bpy.data.objects[f"{NAME}_LOD0"]
    lod2 = bpy.data.objects[f"{NAME}_LOD2"]
    lod4 = bpy.data.objects[f"{NAME}_LOD4"]
    for o in (lod2, lod4):
        lods.triangulate(o)
        for m in list(o.modifiers):
            o.modifiers.remove(m)
    lod1 = lods.decimated(lod0, f"{NAME}_LOD1", 0.5)
    for o in (lod1, lod2):
        rig.transfer_weights(lod0, o, arm)
    lod3 = lods.decimated(lod2, f"{NAME}_LOD3", 0.5)
    for o in (lod3, lod4):
        rig.transfer_weights(lod0, o, arm)
    chain = [lod0, lod1, lod2, lod3, lod4]
    # eyes per LOD (joined L+R, rigid to eye bones); cheap spheres for LOD3/4
    eye_l, eye_r = bpy.data.objects["Eye_L"], bpy.data.objects["Eye_R"]
    for o in (eye_l, eye_r):
        for m in list(o.modifiers):
            o.modifiers.remove(m)
        o.parent = None
        U.select_only([o])
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    eyes0 = join([eye_l, eye_r], f"{NAME}_Eyes_LOD0")
    eye_sets = [eyes0]
    for k in (1, 2):
        e = U.duplicate(eyes0, f"{NAME}_Eyes_LOD{k}")
        eye_sets.append(e)
    for k in (3, 4):
        lo = [lods.low_eye(eyes0, f"tmp_eye_{k}_{s}") for s in (1, -1)]
        lo[1].location.x = -abs(lo[1].location.x)
        lo[0].location.x = abs(lo[0].location.x)
        for o in lo:
            U.select_only([o])
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        eye_sets.append(join(lo, f"{NAME}_Eyes_LOD{k}"))
    for e in eye_sets:
        for g in list(e.vertex_groups):
            e.vertex_groups.remove(g)
        co = U.verts_np(e)
        gl, gr = e.vertex_groups.new(name="eye_l"), e.vertex_groups.new(name="eye_r")
        left = np.nonzero(co[:, 0] > 0)[0].tolist()
        right = np.nonzero(co[:, 0] <= 0)[0].tolist()
        gl.add(left, 1.0, "REPLACE")
        gr.add(right, 1.0, "REPLACE")
        m = e.modifiers.new("Armature", "ARMATURE")
        m.object = arm
        e.parent = arm
    return chain, eye_sets


def export_fbx(path, objs, arm, anim=True):
    U.select_only([arm] + objs, arm)
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"ARMATURE", "MESH"},
        apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
        use_mesh_modifiers=False, mesh_smooth_type="FACE", use_tspace=True,
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        armature_nodetype="NULL", bake_anim=anim, bake_anim_use_all_actions=True,
        bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True, path_mode="STRIP")


def export_glb(path, objs, arm, tex_dir, web_res=2048):
    """GLB for web/engine preview: LOD0 + eyes + skeleton + walk, textures
    downsized to web_res JPEG (normal map kept PNG — JPEG blocks show up as
    shading noise in normals)."""
    from PIL import Image
    swap = {}
    tmp = os.path.join(os.path.dirname(path), "_glb_tex")
    os.makedirs(tmp, exist_ok=True)
    for img in bpy.data.images:
        if not img.filepath or not os.path.exists(bpy.path.abspath(img.filepath)):
            continue
        src = bpy.path.abspath(img.filepath)
        im = Image.open(src)
        if max(im.size) > web_res:
            im = im.resize((web_res, web_res), Image.LANCZOS)
        normal = "Normal" in os.path.basename(src)
        dst = os.path.join(tmp, os.path.splitext(os.path.basename(src))[0] + (".png" if normal else ".jpg"))
        (im.convert("RGB").save(dst, quality=92) if not normal else im.save(dst))
        swap[img.name] = img.filepath
        img.filepath = dst
        img.reload()
    U.select_only([arm] + objs, arm)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_skins=True,
                              export_animations=True, export_yup=True, export_apply=False,
                              export_texcoords=True, export_normals=True, export_tangents=True,
                              export_materials="EXPORT", export_image_format="AUTO")
    for nm, fp in swap.items():
        bpy.data.images[nm].filepath = fp
        bpy.data.images[nm].reload()
    shutil.rmtree(tmp, ignore_errors=True)


def main(out_dir):
    t = time.time()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(out_dir, "rigged.blend"))
    humanoid.build(clothing=False)   # landmarks: LOD protection (face, hands), low-LOD eyes
    arm = bpy.data.objects[f"{NAME}_Skeleton"]
    rig.clear_pose(arm)
    bpy.context.scene.frame_set(1)
    for nm in ("HighPoly", "HighPoly_Pouch", f"{NAME}_LOD0_quads"):
        o = bpy.data.objects.get(nm)
        if o:
            o.hide_render = True
    topo = {}
    for nm in (f"{NAME}_LOD0_quads", f"{NAME}_LOD2", f"{NAME}_LOD4"):
        o = bpy.data.objects.get(nm)
        if o and any(len(p.vertices) == 4 for p in o.data.polygons):
            topo[nm] = lods.topology_report(o)
    chain, eye_sets = build_chain(arm)
    print(f"[{time.time() - t:.1f}s] LOD chain: " + ", ".join(f"{o.name}={U.tri_count(o)}" for o in chain), flush=True)
    ex = os.path.join(out_dir, "export")
    os.makedirs(ex, exist_ok=True)
    export_fbx(os.path.join(ex, f"{NAME}.fbx"), chain + eye_sets, arm)            # Unity: auto LOD group by suffix
    for k, (o, e) in enumerate(zip(chain, eye_sets)):                                # Unreal: one file per LOD
        export_fbx(os.path.join(ex, f"{NAME}_LOD{k}.fbx"), [o, e], arm, anim=(k == 0))
    export_glb(os.path.join(ex, f"{NAME}.glb"), [chain[0], eye_sets[0]], arm, os.path.join(out_dir, "textures"))
    print(f"[{time.time() - t:.1f}s] exported", flush=True)

    # ---- validation report ----
    rep = {"character_height_m": None, "source_topology": topo, "lods": {},
           "skeleton": lods.skeleton_report(arm), "textures": {}}
    lo, hi = U.world_bounds([chain[0]])
    rep["character_height_m"] = round(hi.z - lo.z, 3)
    for k, o in enumerate(chain):
        r = lods.mesh_report(o)
        r.update(lods.skin_report(o))
        rep["lods"][f"LOD{k}"] = r
    rep["eyes"] = {e.name: U.tri_count(e) for e in eye_sets}
    from PIL import Image
    td = os.path.join(out_dir, "textures")
    for f in sorted(os.listdir(td)):
        im = Image.open(os.path.join(td, f))
        w, h = im.size
        rep["textures"][f] = {"size": [w, h], "power_of_two": (w & (w - 1) == 0) and (h & (h - 1) == 0),
                              "mode": im.mode}
    rep["files"] = {f: os.path.getsize(os.path.join(ex, f)) for f in sorted(os.listdir(ex))}
    with open(os.path.join(ex, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "final.blend"))
    print(f"[{time.time() - t:.1f}s] report written", flush=True)
    return rep


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0])
