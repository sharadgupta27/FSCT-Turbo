"""
Redraws the README figures that are not screenshots.

    python readme_images/make_figures.py [path/to/example_FSCT_output]

The point-cloud figure is drawn from a finished run on the bundled example plot
(`data/test/example.las`). Run FSCT-Turbo on it first; the output folder
defaults to `data/test/example_FSCT_output`. The two charts use the benchmark
figures embedded below, so they need no data.
"""

import os
import sys

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Palette: categorical slots in fixed order, a one-hue sequential ramp and
# neutral chrome, all on a light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
HEIGHT_RAMP = LinearSegmentedColormap.from_list(
    "height", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": HAIRLINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def read_las(path):
    las = laspy.read(path)
    return las, np.column_stack([las.x, las.y, las.z])


def side_view(ax, xyz, colour, origin, size=0.12, zorder=1):
    """Scatter an elevation view along +y, far points first."""
    order = np.argsort(-xyz[:, 1])
    c = colour[order] if isinstance(colour, np.ndarray) and colour.ndim > 0 and len(colour) == len(xyz) else colour
    ax.scatter(xyz[order, 0] - origin[0], xyz[order, 2] - origin[2], c=c, s=size,
               linewidths=0, rasterized=True, zorder=zorder)


def overview(out_dir, input_las, dst):
    _, raw = read_las(input_las)
    seg, seg_xyz = read_las(os.path.join(out_dir, "segmented_cleaned.las"))
    _, dtm = read_las(os.path.join(out_dir, "DTM.las"))
    _, cyl = read_las(os.path.join(out_dir, "cleaned_cyl_vis.las"))
    stems, stem_xyz = read_las(os.path.join(out_dir, "stem_points_sorted.las"))
    veg, veg_xyz = read_las(os.path.join(out_dir, "veg_points_sorted.las"))
    trees = pd.read_csv(os.path.join(out_dir, "tree_data.csv"))

    origin = np.array([raw[:, 0].min(), 0.0, dtm[:, 2].min()])
    label = np.asarray(seg["label"]).astype(int)
    width = raw[:, 0].max() - origin[0]
    top = raw[:, 2].max() - origin[2]

    fig, axes = plt.subplots(1, 4, figsize=(11, 6.4), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.17, wspace=0.08)

    # 1. Input, coloured by height.
    z = raw[:, 2]
    side_view(axes[0], raw, HEIGHT_RAMP((z - z.min()) / np.ptp(z)), origin)

    # 2. Semantic segmentation.
    classes = [(2, "Vegetation", AQUA), (1, "Terrain", MUTED), (3, "Coarse woody debris", BLUE),
               (4, "Stem", ORANGE)]
    for value, _, colour in classes:
        side_view(axes[1], seg_xyz[label == value], colour, origin)

    # 3. DTM and fitted stem circles over the cloud.
    side_view(axes[2], seg_xyz, HAIRLINE, origin)
    side_view(axes[2], cyl, ORANGE, origin, size=1.2, zorder=3)
    d = dtm[np.argsort(dtm[:, 0])]
    axes[2].scatter(d[:, 0] - origin[0], d[:, 2] - origin[2], s=9, c=INK_2, linewidths=0, zorder=4)

    # 4. Points assigned to each tree, labelled with its measurements.
    tree_colours = {1: BLUE, 2: ORANGE, 3: AQUA, 4: VIOLET}
    side_view(axes[3], seg_xyz[label == 1], HAIRLINE, origin)
    for las, xyz in ((veg, veg_xyz), (stems, stem_xyz)):
        tid = np.asarray(las["tree_id"]).astype(int)
        keep = np.isin(tid, list(tree_colours))
        side_view(axes[3], xyz[keep], np.array([tree_colours[t] for t in tid[keep]]), origin)
    # A numbered tag on each crown top, and the measurements as a small table
    # in the empty sky above the plot (tree_data.csv, rounded).
    for _, t in trees.iterrows():
        tid = int(t["TreeId"])
        axes[3].annotate(str(tid), (t["Crown_top_x"] - origin[0], t["Crown_top_z"] - origin[2]),
                         xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                         fontsize=8, fontweight="bold", color=INK, zorder=5,
                         bbox=dict(boxstyle="circle,pad=0.25", fc=SURFACE, ec=tree_colours[tid], lw=1.4))
    table = axes[3].inset_axes([0.40, 0.80, 0.60, 0.20])
    table.axis("off")
    columns = [(0.13, "Tree", "left"), (0.68, "DBH", "right"), (1.0, "Height", "right")]
    for x, head, ha in columns:
        table.text(x, 1.0, head, ha=ha, va="top", fontsize=8, color=INK_2, transform=table.transAxes)
    for r, (_, t) in enumerate(trees.iterrows()):
        y = 1.0 - (r + 1) * 0.2
        tid = int(t["TreeId"])
        table.plot(0.03, y - 0.06, "o", ms=6, mfc=tree_colours[tid], mec="none", transform=table.transAxes,
                   clip_on=False)
        for (x, _, ha), value in zip(columns, (str(tid), f"{t['DBH']:.2f} m", f"{t['Height']:.1f} m")):
            table.text(x, y, value, ha=ha, va="top", fontsize=8, color=INK, transform=table.transAxes)

    titles = ["Input point cloud", "Semantic segmentation", "Terrain and stem circles", "Trees and measurements"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(f"{i + 1}  {title}", loc="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, width + 0.5)
        ax.set_ylim(-1, top + 4.5)
        ax.set_xlabel("Distance (m)")
        ax.tick_params(length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Height (m)")

    handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=c, mec="none", label=n) for _, n, c in classes]
    handles.append(Line2D([], [], ls="", marker="o", ms=5, mfc=INK_2, mec="none", label="Terrain model (DTM)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.52, 0.035), handletextpad=0.3, columnspacing=1.6)
    fig.text(0.05, 0.005, f"Side view of data/test/example.las ({len(raw):,} points), "
             "processed by FSCT-Turbo with default settings.", fontsize=8, color=MUTED)
    fig.savefig(dst, dpi=180)
    plt.close(fig)


# Stage times on example.las, 673,517 points, 8 workers, batch size 2. Each
# figure is the mean of two interleaved rounds (original, Turbo fp32, Turbo
# fp16, repeated), so the ratios compare runs made minutes apart. Original FSCT
# is SKrisanski/FSCT @ 68e2f1e with library-compatibility edits only.
STAGES = ["Preprocessing", "Segmentation", "Post-processing", "Measurement"]
STAGE_TIMES = {
    "Original FSCT": [3.68, 21.17, 1.53, 304.07],
    "FSCT-Turbo, default (fp32)": [0.85, 18.36, 1.32, 13.81],
    "FSCT-Turbo, use_amp (fp16)": [0.88, 15.47, 1.24, 12.87],
}


def stage_times(dst):
    colours = [BLUE, ORANGE, AQUA, YELLOW]
    names = list(STAGE_TIMES)
    base = sum(STAGE_TIMES[names[0]])

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), gridspec_kw=dict(height_ratios=[3, 2], hspace=0.7))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.11)
    panels = [(axes[0], names, 345, "All runs"), (axes[1], names[1:], 42, "FSCT-Turbo runs, zoomed in")]
    for ax, rows, xmax, title in panels:
        for r, name in enumerate(rows):
            left = 0.0
            for t, c in zip(STAGE_TIMES[name], colours):
                ax.barh(r, t, left=left, height=0.62, color=c, edgecolor=SURFACE, linewidth=1.5)
                left += t
            note = f"{left:.0f} s" if name == names[0] else f"{left:.1f} s, {base / left:.1f}x faster"
            ax.text(left + xmax * 0.012, r, note, va="center", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(rows)), rows)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", color=HAIRLINE, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK_2)
    axes[1].set_xlabel("Wall-clock time (s)")

    handles = [Line2D([], [], ls="", marker="s", ms=9, mfc=c, mec="none", label=s) for s, c in zip(STAGES, colours)]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.012, 0.935), handletextpad=0.3, columnspacing=1.4)
    fig.text(0.015, 0.955, "End-to-end run time on the example plot", fontsize=12, fontweight="bold")
    fig.savefig(dst, dpi=180)
    plt.close(fig)


# Circle-fit ablation: 60 real stem slices (median 2,262 points) from the
# example plot, one fit at a time in a single process. Mean ms per fit.
ABLATION = [
    ("Original: scikit-image RANSAC,\n10,000 trials", 2005),
    ("Same, with trials batched", 762),
    ("+ stopping confidence 0.99", 740),
    ("+ 1,000 trial cap", 75.0),
    ("+ at most 1,500 points per fit\n(FSCT-Turbo default)", 29.1),
]


def circle_fit(dst):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.80, bottom=0.17)
    y = np.arange(len(ABLATION))
    ms = np.array([m for _, m in ABLATION])
    ax.hlines(y, 10, ms, color=HAIRLINE, lw=2, zorder=1)
    ax.scatter(ms, y, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, m in zip(y, ms):
        speed = "" if yi == 0 else f",  {ms[0] / m:.1f}x"
        ax.text(m * 1.18, yi, f"{m:,g} ms{speed}", va="center", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(10, 9000)
    ax.set_xticks([10, 30, 100, 300, 1000, 3000], ["10", "30", "100", "300", "1,000", "3,000"])
    ax.set_yticks(y, [n for n, _ in ABLATION])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", color=HAIRLINE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_xlabel("Mean time per circle fit (ms, log scale)")
    fig.text(0.015, 0.92, "Where the circle-fitting speed-up comes from", fontsize=12, fontweight="bold")
    fig.text(0.015, 0.845, "Each step adds to the one above, on 60 real stem slices from the example plot.",
             fontsize=9, color=INK_2)
    fig.savefig(dst, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "test", "example_FSCT_output")
    input_las = os.path.join(os.path.dirname(out_dir), os.path.basename(out_dir).replace("_FSCT_output", ".las"))
    overview(out_dir, input_las, os.path.join(HERE, "example_overview.png"))
    stage_times(os.path.join(HERE, "stage_times.png"))
    circle_fit(os.path.join(HERE, "circle_fit.png"))
    print("Figures written to", HERE)
