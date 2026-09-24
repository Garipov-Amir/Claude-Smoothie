#!/usr/bin/env python3
"""
image_prep — turns a source photo into the inputs the Blender-side "image to 3D"
helpers need: a color palette (for materials), a grayscale heightmap (for
bas-relief displacement), and a silhouette mask (for cutout extrusion).

This runs under the SYSTEM python3 (needs Pillow), not inside Blender's bpy
python — Blender's bundled interpreter doesn't have Pillow by default.

Single-view usage (bas-relief / cutout extrusion workflows):
    python3 image_prep.py <image_path> --outdir <dir> [--colors 6]

Writes into <dir>:
    palette.json     — {"colors": ["#rrggbb", ...], "background": "#rrggbb"}
    heightmap.png     — grayscale luminance map, same aspect ratio
    silhouette.png    — black/white foreground mask (best-effort background removal)

Multi-view usage (for bpy_stylized_kit.carve_from_silhouettes — visual-hull
reconstruction from several consistently-framed photos of the same subject):
    python3 image_prep.py --outdir <dir> --view front=front.jpg --view side=side.jpg [--view top=top.jpg]

Writes into <dir>: silhouette_<axis>.png for each --view (front/side/top),
plus palette.json from the first view given.

Prints a one-line JSON summary of the written paths to stdout.
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat


def extract_palette(img, n_colors=6):
    quant = img.convert("RGB").quantize(colors=n_colors, method=Image.MEDIANCUT)
    palette = quant.getpalette()[: n_colors * 3]
    counts = sorted(quant.getcolors(), reverse=True)
    colors = []
    for count, idx in counts:
        r, g, b = palette[idx * 3: idx * 3 + 3]
        colors.append("#{:02x}{:02x}{:02x}".format(r, g, b))
    return colors


def make_heightmap(img, out_path):
    gray = ImageOps.grayscale(img)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1.5))
    gray = ImageOps.autocontrast(gray)
    gray.save(out_path)
    return out_path


def make_silhouette_mask(img, out_path, threshold=None):
    """
    Best-effort foreground mask with no ML dependency:
    - If the source has real alpha (already a cutout PNG), use it directly.
    - Otherwise assume a roughly flat/light background and threshold by
      difference from the estimated background color (sampled from the
      image corners). This is a heuristic, not real segmentation — for busy
      backgrounds, tell the user to supply a pre-cutout PNG for best results.
    """
    if img.mode == "RGBA":
        alpha = img.split()[-1]
        if alpha.getextrema() != (255, 255):
            alpha.save(out_path)
            return out_path, "used existing alpha channel"

    rgb = img.convert("RGB")
    w, h = rgb.size
    corner_samples = [
        rgb.getpixel((1, 1)),
        rgb.getpixel((w - 2, 1)),
        rgb.getpixel((1, h - 2)),
        rgb.getpixel((w - 2, h - 2)),
    ]
    bg = tuple(sum(c[i] for c in corner_samples) // 4 for i in range(3))

    bg_img = Image.new("RGB", rgb.size, bg)
    diff = ImageChops.difference(rgb, bg_img).convert("L")
    mean, stddev = ImageStat.Stat(diff).mean[0], ImageStat.Stat(diff).stddev[0]
    thresh = threshold if threshold is not None else max(25, mean + 0.5 * stddev)
    mask_img = diff.point(lambda p: 255 if p > thresh else 0, mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    mask_img.save(out_path)
    return out_path, f"heuristic threshold={thresh:.0f} vs. estimated background {bg}"


def _parse_view(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--view must be AXIS=PATH, got {spec!r}")
    axis, path = spec.split("=", 1)
    if axis not in ("front", "side", "top"):
        raise argparse.ArgumentTypeError(f"--view axis must be front/side/top, got {axis!r}")
    return axis, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", nargs="?",
                     help="single-view mode: source photo for palette/heightmap/silhouette")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--colors", type=int, default=6)
    ap.add_argument("--view", action="append", type=_parse_view, default=[],
                     help="multi-view mode: AXIS=PATH, repeatable (front/side/top)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.view:
        if args.image_path:
            ap.error("pass either a single image_path or one or more --view, not both")
        if len(args.view) < 2:
            ap.error("multi-view mode needs at least 2 --view entries (front+side, ideally +top)")
        result = {"views": {}}
        first_img = None
        for axis, path in args.view:
            img = Image.open(path)
            img.load()
            if first_img is None:
                first_img = img
            out_path = os.path.join(args.outdir, f"silhouette_{axis}.png")
            silhouette_path, note = make_silhouette_mask(img, out_path)
            result["views"][axis] = {"silhouette": silhouette_path, "note": note}
        colors = extract_palette(first_img, args.colors)
        result["colors"] = colors
        palette_path = os.path.join(args.outdir, "palette.json")
        with open(palette_path, "w") as f:
            json.dump({"colors": colors}, f, indent=2)
        result["palette"] = palette_path
        print(json.dumps(result))
        return

    if not args.image_path:
        ap.error("pass an image_path (single-view mode) or one or more --view (multi-view mode)")

    img = Image.open(args.image_path)
    img.load()

    colors = extract_palette(img, args.colors)
    height_path = make_heightmap(img, os.path.join(args.outdir, "heightmap.png"))
    silhouette_path, note = make_silhouette_mask(img, os.path.join(args.outdir, "silhouette.png"))

    palette_path = os.path.join(args.outdir, "palette.json")
    with open(palette_path, "w") as f:
        json.dump({"colors": colors, "silhouette_note": note}, f, indent=2)

    print(json.dumps({
        "palette": palette_path,
        "heightmap": height_path,
        "silhouette": silhouette_path,
        "colors": colors,
        "silhouette_note": note,
    }))


if __name__ == "__main__":
    main()
