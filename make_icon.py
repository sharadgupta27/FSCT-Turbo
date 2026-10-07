"""
Generate the FSCT application icon.

Run this only when the artwork changes:

    python make_icon.py

It writes ``icon.ico`` (the Windows icon used by fsct_desktop.py and any
shortcut) and ``icon.png`` (512 px, for the desktop app's logo and the README).

Design notes
------------
The icon has to survive being drawn at 16x16 in a taskbar, so the silhouette
does the work: a conifer, centred, high contrast against a solid rounded-square
background. The point-cloud texture on top of the silhouette is what makes it
specifically a *LiDAR* tool rather than a generic forestry one, and it degrades
gracefully - at small sizes the dots blur back into the silhouette they were
carved out of instead of turning into noise.

Everything is drawn once at 1024x1024 and downsampled with LANCZOS for each
icon size, which keeps the small variants sharp.
"""

import os
import random

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Windows uses 16/32/48 in shell surfaces and 256 for large tiles; the rest
# keep intermediate DPI scalings from being resampled from a bad neighbour.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

# Deep forest green through to a lit canopy green.
BG_TOP = (10, 58, 43)
BG_BOTTOM = (26, 122, 78)

SILHOUETTE = (222, 245, 228)   # tree body, drawn semi-transparent
POINT_BRIGHT = (186, 255, 214)  # scan returns inside the crown
POINT_DIM = (140, 214, 176)     # scattered returns just off the crown
TRUNK = (196, 232, 205)
GROUND = (150, 226, 186)


def _background():
    """Rounded square with a vertical gradient and a soft top-left highlight."""
    bg = Image.new("RGB", (SIZE, SIZE), BG_TOP)
    draw = ImageDraw.Draw(bg)
    for y in range(SIZE):
        t = y / (SIZE - 1)
        # Ease the ramp so the midtones sit lower, which keeps the top corners
        # dark enough for the light silhouette to read against them.
        t = t * t * (3 - 2 * t)
        colour = tuple(
            round(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)
        )
        draw.line([(0, y), (SIZE, y)], fill=colour)

    glow = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(glow).ellipse([-260, -420, 700, 420], fill=90)
    glow = glow.filter(ImageFilter.GaussianBlur(140))
    bg = Image.composite(Image.new("RGB", (SIZE, SIZE), (120, 220, 175)), bg, glow)

    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, SIZE - 1, SIZE - 1], radius=int(SIZE * 0.22), fill=255
    )

    icon = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    icon.paste(bg, (0, 0), mask)
    return icon, mask


def _tree_mask():
    """Filled mask of the conifer, used both to draw it and to seed the points."""
    mask = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask)

    cx = SIZE // 2
    # Three overlapping tiers, widening towards the base.
    tiers = [
        (185, 455, 170),   # apex y, base y, half-width
        (355, 630, 240),
        (525, 800, 310),
    ]
    for apex_y, base_y, half in tiers:
        draw.polygon(
            [(cx, apex_y), (cx - half, base_y), (cx + half, base_y)], fill=255
        )

    draw.rounded_rectangle([cx - 36, 770, cx + 36, 862], radius=14, fill=255)
    return mask


def _scan_points(tree_mask):
    """
    Jittered grid of returns: dense and bright inside the crown, sparse and dim
    just outside it, the way a real scan of a single tree looks.
    """
    rng = random.Random(20260814)  # fixed, so re-running produces the same icon
    halo = tree_mask.filter(ImageFilter.MaxFilter(9)).filter(
        ImageFilter.GaussianBlur(18)
    )

    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    step = 26
    for gy in range(150, 900, step):
        for gx in range(150, 880, step):
            x = gx + rng.randint(-8, 8)
            y = gy + rng.randint(-8, 8)
            if not (0 <= x < SIZE and 0 <= y < SIZE):
                continue

            inside = tree_mask.getpixel((x, y)) > 128
            if inside:
                radius = rng.choice((7, 8, 10))
                alpha = rng.randint(210, 255)
                colour = POINT_BRIGHT + (alpha,)
            else:
                # Only keep the near misses; a full field of dots would fight
                # the silhouette at small sizes.
                if halo.getpixel((x, y)) < 40 or rng.random() > 0.35:
                    continue
                radius = rng.choice((4, 5))
                colour = POINT_DIM + (rng.randint(90, 150),)

            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour)

    return layer


def build_icon():
    icon, clip = _background()
    tree = _tree_mask()

    # The silhouette sits underneath at partial opacity: enough to give a solid
    # shape at 16 px, faint enough that the points still read at 256 px.
    body = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    body.paste(SILHOUETTE + (150,), (0, 0), tree)

    trunk = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(trunk).rounded_rectangle(
        [SIZE // 2 - 36, 770, SIZE // 2 + 36, 862], radius=14, fill=TRUNK + (235,)
    )

    # Ground plane: FSCT fits a DTM before it measures anything, so the icon
    # says so with a scanned ground line under the trunk.
    ground = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(ground)
    gdraw.rounded_rectangle([214, 848, 810, 872], radius=12, fill=GROUND + (215,))
    rng = random.Random(7)
    for _ in range(46):
        x = rng.randint(190, 834)
        y = rng.randint(884, 918)
        r = rng.choice((5, 6, 8))
        gdraw.ellipse([x - r, y - r, x + r, y + r], fill=GROUND + (rng.randint(70, 140),))

    icon.alpha_composite(body)
    icon.alpha_composite(_scan_points(tree))
    icon.alpha_composite(trunk)
    icon.alpha_composite(ground)

    # Re-clip: the ground dots and halo points can spill past the rounded corners.
    icon.putalpha(Image.composite(icon.getchannel("A"), Image.new("L", (SIZE, SIZE), 0), clip))

    # Thin inner rim, so the icon keeps an edge on a white or dark background.
    rim = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(rim).rounded_rectangle(
        [4, 4, SIZE - 5, SIZE - 5],
        radius=int(SIZE * 0.22) - 3,
        outline=(255, 255, 255, 46),
        width=8,
    )
    icon.alpha_composite(rim)
    return icon


def main():
    icon = build_icon()

    png_path = os.path.join(OUT_DIR, "icon.png")
    icon.resize((512, 512), Image.LANCZOS).save(png_path)

    ico_path = os.path.join(OUT_DIR, "icon.ico")
    # Pillow's own multi-size .ico writer downsamples with NEAREST, which shreds
    # the 16 px variant. Resample each size properly and hand over the frames.
    frames = [icon.resize((s, s), Image.LANCZOS) for s in ICO_SIZES]
    frames[-1].save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=frames[:-1])

    print(f"wrote {png_path}")
    print(f"wrote {ico_path}  sizes={list(ICO_SIZES)}")


if __name__ == "__main__":
    main()
