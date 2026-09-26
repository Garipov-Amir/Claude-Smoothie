"""
lookdev — the game material (what an engine sees: textures only) for
previews in Blender, plus a presentation-light setup.
"""
import os

import bpy


def load_img(path, color_space):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = color_space
    return img


def game_material(name, tex_dir, prefix="T_Character", skin_sss_mask=None):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bc = nt.nodes.new("ShaderNodeTexImage")
    bc.image = load_img(os.path.join(tex_dir, f"{prefix}_BaseColor.png"), "sRGB")
    orm = nt.nodes.new("ShaderNodeTexImage")
    orm.image = load_img(os.path.join(tex_dir, f"{prefix}_ORM.png"), "Non-Color")
    nm = nt.nodes.new("ShaderNodeTexImage")
    nm.image = load_img(os.path.join(tex_dir, f"{prefix}_Normal_OpenGL.png"), "Non-Color")
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.space = "TANGENT"
    nt.links.new(bc.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(orm.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    nt.links.new(nm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    if skin_sss_mask:
        sm = nt.nodes.new("ShaderNodeTexImage")
        sm.image = load_img(skin_sss_mask, "Non-Color")
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 0.35
        nt.links.new(sm.outputs["Color"], mul.inputs[0])
        nt.links.new(mul.outputs["Value"], bsdf.inputs["Subsurface Weight"])
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        bsdf.inputs["Subsurface Scale"].default_value = 0.004
    return mat


def eye_material(name, tex_path):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = load_img(tex_path, "sRGB")
    nt.links.new(tx.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.06
    bsdf.inputs["Coat Weight"].default_value = 0.6
    bsdf.inputs["Coat Roughness"].default_value = 0.02
    return mat
