"""
Presentation sheets for a finished build (what a portfolio / review needs):

  highpoly.png   clay turnaround of the sculpt
  beauty.png     textured turnaround (engine-style material, eyes)
  head.png       face close-ups
  topology.png   true-edge wireframe: body, head, hand
  lods.png       LOD0..LOD4 with triangle counts
  textures.png   BaseColor / Normal / ORM + UV layout
  walk.png       walk-cycle frames
  poses.png      deformation stress poses

    python render_showcase.py <build_dir> <out_dir>
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

import bl_util as U
import humanoid
import rig


def only(objs):
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = o not in objs


def labels_sheet(paths, out, labels, cols=None):
    return U.contact_sheet(paths, out, cols=cols, labels=labels)


def main(build, out, samples=64):
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    _m, J = humanoid.build(clothing=False)
    from landmarks import LM
    head = (0.0, float(LM["eye_l"][1]) + 0.07, float(LM["eye_l"][2]) - 0.005)   # between the eyes, mid-skull

    # ---- high-poly clay ----
    import numpy as _np
    d = _np.load(os.path.join(build, "highpoly.npz"), allow_pickle=True)
    U.reset_scene()
    hp = U.mesh_from_arrays("HighPoly", d["verts"], d["faces"])
    hps = [hp]
    pp = os.path.join(build, "highpoly_pouch.npz")
    if os.path.exists(pp):
        dp = _np.load(pp)
        hps.append(U.mesh_from_arrays("HighPoly_Pouch", dp["verts"], dp["faces"]))
    clay = U.clay_material(color=(0.55, 0.53, 0.50))
    for o in hps:
        o.data.shade_smooth()
        U.set_material(o, clay)
    p = U.render_views(f"{tmp}/hp", hps, views=("front", "3q", "left", "3q_back"), resolution=(600, 900),
                       samples=samples // 2, ortho=False)
    p += U.render_views(f"{tmp}/hph", hps, views=("front", "3q"), resolution=(600, 900), samples=samples // 2,
                        focus=(head, 0.30), ortho=False)
    U.contact_sheet(p, os.path.join(out, "highpoly.png"))

    # ---- final asset ----
    bpy.ops.wm.open_mainfile(filepath=os.path.join(build, "final.blend"))
    arm = bpy.data.objects["SK_Character_Skeleton"]
    rig.clear_pose(arm)
    arm.animation_data.action = None
    L = [bpy.data.objects[f"SK_Character_LOD{k}"] for k in range(5)]
    E = [bpy.data.objects[f"SK_Character_Eyes_LOD{k}"] for k in range(5)]
    only([L[0], E[0]])
    p = U.render_views(f"{tmp}/beauty", [L[0], E[0]], views=("front", "3q", "left", "3q_back"),
                       resolution=(600, 900), samples=samples, ortho=False)
    U.contact_sheet(p, os.path.join(out, "beauty.png"))
    p = U.render_views(f"{tmp}/head", [L[0], E[0]], views=("front", "3q", "left"), resolution=(700, 800),
                       samples=samples, focus=(head, 0.27), ortho=False)
    U.contact_sheet(p, os.path.join(out, "head.png"))

    # ---- topology (quad source + real edges) ----
    q = bpy.data.objects["SK_Character_LOD0_quads"]
    q.parent = None
    for m in list(q.modifiers):
        q.modifiers.remove(m)
    U.set_material(q, U.clay_material("TopoClay", color=(0.62, 0.60, 0.57)))
    w = U.edge_overlay(q, thickness=0.00055)
    only([q, w])
    c_hand = tuple(J["hand_l"] + J["_hand_axis_l"] * 0.07)
    p = U.render_views(f"{tmp}/topo_body", [q], views=("3q",), resolution=(600, 900), samples=16, ortho=False)
    p += U.render_views(f"{tmp}/topo_head", [q], views=("3q",), resolution=(600, 900), samples=16,
                        focus=(head, 0.30), ortho=False)
    p += U.render_views(f"{tmp}/topo_hand", [q], views=("left",), resolution=(600, 900), samples=16,
                        focus=(c_hand, 0.22), ortho=False)
    U.contact_sheet(p, os.path.join(out, "topology.png"))

    # ---- LODs ----
    p, lab = [], []
    for k in range(5):
        wk = U.edge_overlay(L[k], thickness=0.0006)
        for mm in list(wk.modifiers):
            if mm.type == "ARMATURE":
                wk.modifiers.remove(mm)
        only([L[k], E[k], wk])
        p += U.render_views(f"{tmp}/lod{k}", [L[k]], views=("3q",), resolution=(420, 800), samples=16,
                            focus=((0, 0, 1.0), 1.95), ortho=False)
        lab.append(f"LOD{k}  {U.tri_count(L[k]) + U.tri_count(E[k])} tris")
        bpy.data.objects.remove(wk, do_unlink=True)
    labels_sheet(p, os.path.join(out, "lods.png"), lab)

    # ---- walk + poses ----
    act = bpy.data.actions.get("Walk")
    only([L[0], E[0]])
    arm.animation_data.action = act
    p = []
    for f in (1, 5, 9, 13, 17, 21):
        bpy.context.scene.frame_set(f)
        p += U.render_views(f"{tmp}/walk{f:02d}", [L[0], E[0]], views=("3q",), resolution=(420, 800),
                            samples=samples // 2, focus=((0, 0, 0.95), 2.0), ortho=False)
    U.contact_sheet(p, os.path.join(out, "walk.png"))
    arm.animation_data.action = None
    P = arm.pose.bones
    poses = []

    def arms_up(P):
        for sfx, sg in (("_l", 1), ("_r", -1)):
            rig._rot(P["clavicle" + sfx], z=-sg * 12)
            rig._rot(P["upperarm" + sfx], z=-sg * 60)
            rig._rot(P["lowerarm" + sfx], x=-35)

    def crouch(P):
        for sfx in ("_l", "_r"):
            rig._rot(P["thigh" + sfx], x=-70)
            rig._rot(P["calf" + sfx], x=100)
            rig._rot(P["foot" + sfx], x=-25)
        P["pelvis"].location = (0, -0.30, -0.10)
        rig._rot(P["spine_01"], x=-25)
        for sfx, sg in (("_l", 1), ("_r", -1)):
            rig._rot(P["upperarm" + sfx], x=-50, z=sg * 30)
            rig._rot(P["lowerarm" + sfx], x=-70)
            for fg in rig.FINGERS:
                for i in (1, 2, 3):
                    rig._rot(P[f"{fg}_0{i}{sfx}"], x=-(75 if fg != "thumb" else 30))

    for fn, tag in ((arms_up, "armsup"), (crouch, "crouch")):
        rig.clear_pose(arm)
        fn(P)
        bpy.context.view_layer.update()
        poses += U.render_views(f"{tmp}/pose_{tag}", [L[0], E[0]], views=("front", "3q"), resolution=(500, 800),
                                samples=samples // 2, focus=((0, 0, 1.0), 2.1), ortho=False)
    U.contact_sheet(poses, os.path.join(out, "poses.png"))
    rig.clear_pose(arm)

    # ---- textures + UVs ----
    from PIL import Image, ImageDraw
    td = os.path.join(build, "textures")
    tiles = []
    for f in ("T_Character_BaseColor.png", "T_Character_Normal_OpenGL.png", "T_Character_ORM.png"):
        tiles.append(Image.open(os.path.join(td, f)).convert("RGB").resize((768, 768)))
    uvim = Image.new("RGB", (768, 768), (28, 29, 33))
    dr = ImageDraw.Draw(uvim)
    me = L[0].data
    uv = me.uv_layers.active.data
    for poly in me.polygons:
        pts = [(uv[i].uv.x * 768, (1 - uv[i].uv.y) * 768) for i in poly.loop_indices]
        dr.polygon(pts, outline=(170, 180, 200))
    tiles.append(uvim)
    sheet = Image.new("RGB", (768 * 4, 768))
    for i, t in enumerate(tiles):
        sheet.paste(t, (768 * i, 0))
    sheet.save(os.path.join(out, "textures.png"))

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    main(args[0], args[1], int(args[2]) if len(args) > 2 else 64)
