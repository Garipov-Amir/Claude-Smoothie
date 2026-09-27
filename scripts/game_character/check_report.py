"""
Pass/fail review of a build's validation report (<out>/export/report.json),
the checklist of the humanoid-character skill in one command.

    python check_report.py <out_dir | report.json> [...]

Prints one line per check that fails and a summary per build; exit code 1
if any build fails. Plain python, no bpy needed.
"""
import json
import os
import sys

LOD_CHECKS = [
    ("watertight", lambda v: v is True),
    ("self_intersecting_face_pairs", lambda v: v == 0),
    ("non_manifold_edges", lambda v: v == 0),
    ("boundary_edges", lambda v: v == 0),
    ("degenerate_faces", lambda v: v == 0),
    ("loose_verts", lambda v: v == 0),
    ("ngons", lambda v: v == 0),
    ("uv_inside_0_1", lambda v: v is True),
    ("max_influences", lambda v: v <= 4),
    ("unweighted_verts", lambda v: v == 0),
    ("weights_normalized", lambda v: v is True),
    ("scale_applied", lambda v: v is True),
    ("rotation_applied", lambda v: v is True),
]
UV_OVERLAP_MAX = {"LOD0": 0.0}          # decimated LODs may collapse across a seam
UV_OVERLAP_DEFAULT = 0.005
FILES = ["SK_Character.fbx", "SK_Character.glb"] + [f"SK_Character_LOD{i}.fbx" for i in range(5)]


def report_path(arg):
    for p in (arg, os.path.join(arg, "export", "report.json"), os.path.join(arg, "report.json")):
        if os.path.isfile(p):
            return p
    return None


def check(arg):
    path = report_path(arg)
    if path is None:
        return [f"no report.json in {arg}"], {}
    r = json.load(open(path))
    bad = []
    tris = {}
    for lod, v in sorted(r.get("lods", {}).items()):
        tris[lod] = v.get("tris")
        for key, ok in LOD_CHECKS:
            if key not in v:
                bad.append(f"{lod}: {key} missing")
            elif not ok(v[key]):
                bad.append(f"{lod}: {key} = {v[key]}")
        lim = UV_OVERLAP_MAX.get(lod, UV_OVERLAP_DEFAULT)
        if v.get("uv_overlap_fraction", 1.0) > lim:
            bad.append(f"{lod}: uv_overlap_fraction = {v.get('uv_overlap_fraction')} (max {lim})")
    if len(tris) != 5:
        bad.append(f"LODs present: {sorted(tris)}")
    sk = r.get("skeleton", {})
    if sk.get("bone_count") != 63:
        bad.append(f"skeleton: bone_count = {sk.get('bone_count')}")
    for key in ("single_root", "root_at_origin"):
        if sk.get(key) is not True:
            bad.append(f"skeleton: {key} = {sk.get(key)}")
    if sk.get("missing_humanoid_bones"):
        bad.append(f"skeleton: missing {sk['missing_humanoid_bones']}")
    for name, t in r.get("textures", {}).items():
        size = t.get("size") if isinstance(t, dict) else t
        if isinstance(size, (list, tuple)) and any(s & (s - 1) for s in size):
            bad.append(f"texture {name}: size {size} not a power of two")
    files = r.get("files", {})
    for f in FILES:
        if not files.get(f):
            bad.append(f"file missing or empty: export/{f}")
    return bad, tris


def main(dirs):
    failed = 0
    for d in dirs:
        bad, tris = check(d)
        name = os.path.normpath(d)
        counts = " / ".join(f"{t:,}" for _, t in sorted(tris.items()) if t is not None)
        status = "PASS" if not bad else f"FAIL ({len(bad)})"
        print(f"{name}: {status}   tris LOD0..4: {counts}")
        for b in bad:
            print(f"    {b}")
        failed += bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(main(args))
