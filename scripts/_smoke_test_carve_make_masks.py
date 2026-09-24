#!/usr/bin/env python3
"""Generate the two synthetic test masks for _smoke_test_carve.py.
Run under SYSTEM python3 (needs Pillow) before running the Blender smoke test.
Usage: python3 scripts/_smoke_test_carve_make_masks.py <outdir>
"""
import os
import sys

from PIL import Image, ImageDraw

outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)

# front view: a triangle pointing UP (wide base, narrow top) — asymmetric
# top/bottom so an accidental vertical flip in extrude_silhouette_volume
# would be visually obvious in the render.
front = Image.new("L", (256, 256), 0)
ImageDraw.Draw(front).polygon([(30, 230), (226, 230), (128, 20)], fill=255)
front_path = os.path.join(outdir, "front.png")
front.save(front_path)

# side view: a narrower rectangle-ish silhouette, different shape from
# front — so the intersection is genuinely 3D, not just "extrude a circle
# from two angles and get a sphere back".
side = Image.new("L", (256, 256), 0)
ImageDraw.Draw(side).rectangle([90, 40, 166, 230], fill=255)
side_path = os.path.join(outdir, "side.png")
side.save(side_path)

print(front_path, side_path)
