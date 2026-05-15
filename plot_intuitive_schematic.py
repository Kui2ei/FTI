"""
Intuitive scientific schematic for rotation-key co-design.

The figure is intentionally not a flowchart: it uses spatial mappings,
cache-like key hierarchy, and a Pareto inset to explain the design intuition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import numpy as np


OUT_DIR = Path("results")

PALETTE = {
    "bg": "#FFFFFF",
    "ink": "#25313B",
    "muted": "#66727C",
    "grid": "#D8E0E6",
    "blue": "#2F5F88",
    "blue_dark": "#17365D",
    "blue_light": "#DCECF7",
    "cyan": "#1C9CC8",
    "cyan_light": "#DFF5FA",
    "green": "#2E8B57",
    "green_light": "#DFF1E8",
    "orange": "#D48A2A",
    "orange_light": "#FFF0D6",
    "red": "#C94C4C",
    "red_light": "#F8DDDD",
    "grey_light": "#EEF2F5",
    "grey_mid": "#AAB6BF",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 420,
        }
    )


def rounded_label(ax, x, y, text, w=None, h=2.4, fc="white", ec=None, fs=7.4, bold=False):
    if w is None:
        w = max(9.0, len(text) * 0.23)
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.18,rounding_size=0.7",
        facecolor=fc,
        edgecolor=ec or PALETTE["grid"],
        linewidth=0.75,
        zorder=5,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=PALETTE["ink"],
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        zorder=6,
    )
    return box


def draw_panel_title(ax, x, y, letter, title, subtitle=None):
    ax.text(
        x,
        y,
        letter,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["blue_dark"],
    )
    ax.text(
        x + 2.3,
        y - 0.05,
        title,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    if subtitle:
        ax.text(
            x + 2.3,
            y - 2.0,
            subtitle,
            ha="left",
            va="top",
            fontsize=6.8,
            color=PALETTE["muted"],
        )


def polar_xy(cx, cy, radius, angle_deg):
    angle = np.deg2rad(angle_deg)
    return cx + radius * np.cos(angle), cy + radius * np.sin(angle)


def draw_index_ring(ax, cx, cy, radius, label, tick_count=32):
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            radius,
            facecolor="white",
            edgecolor=PALETTE["ink"],
            linewidth=0.9,
            zorder=1,
        )
    )
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            radius * 0.74,
            facecolor=PALETTE["grey_light"],
            edgecolor=PALETTE["grid"],
            linewidth=0.5,
            alpha=0.72,
            zorder=0,
        )
    )
    for i in range(tick_count):
        angle = i * 360 / tick_count
        x1, y1 = polar_xy(cx, cy, radius * 0.92, angle)
        x2, y2 = polar_xy(cx, cy, radius * 1.04, angle)
        ax.plot([x1, x2], [y1, y2], color=PALETTE["grid"], linewidth=0.42, zorder=2)
    ax.text(cx, cy + 0.45, label, ha="center", va="center", fontsize=6.7, color=PALETTE["ink"], fontweight="bold")
    ax.text(cx, cy - 1.15, r"$\mathrm{mod}\;N$", ha="center", va="center", fontsize=6.2, color=PALETTE["muted"])


def draw_ring_segment(ax, cx, cy, radius, start, extent, color, width=0.86, alpha=0.84, label=None, label_offset=1.6):
    if extent <= 0:
        return start
    end = start + extent
    ax.add_patch(
        patches.Wedge(
            (cx, cy),
            radius + width / 2,
            start,
            end,
            width=width,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=alpha,
            zorder=4,
        )
    )
    ex, ey = polar_xy(cx, cy, radius + width * 0.15, end)
    ax.scatter([ex], [ey], s=18, color=color, edgecolor="white", linewidth=0.45, zorder=6)
    if label:
        mx, my = polar_xy(cx, cy, radius + label_offset, start + extent / 2)
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.6, color=color, fontweight="bold", zorder=7)
    return end


def draw_ring_point(ax, cx, cy, radius, angle, color, marker="o", size=22, label=None):
    x, y = polar_xy(cx, cy, radius, angle)
    ax.scatter([x], [y], s=size, marker=marker, color=color, edgecolor="white", linewidth=0.5, zorder=7)
    if label:
        tx, ty = polar_xy(cx, cy, radius + 1.55, angle)
        ax.text(tx, ty, label, ha="center", va="center", fontsize=6.3, color=PALETTE["muted"], zorder=7)


def draw_matrix(ax, x0, y0, n=5, cell=0.92):
    groups = [
        [-3, 0, 1],
        [-1, 1, 3],
        [-4, -2, 0, 2],
        [-2, 1, 4],
        [-4, -1, 0, 3],
        [-3, 2, 4],
    ]
    gap_x, gap_y = 1.05, 1.35
    msize = n * cell

    def draw_one_matrix(mx, my, diags, group_idx):
        ax.add_patch(
            patches.Polygon(
                [(mx + 0.42, my + 0.42), (mx + msize + 0.42, my + 0.42), (mx + msize, my), (mx, my)],
                closed=True,
                facecolor=PALETTE["grey_light"],
                edgecolor="none",
                alpha=0.72,
                zorder=0,
            )
        )
        for r in range(n):
            for c in range(n):
                ax.add_patch(
                    patches.Rectangle(
                        (mx + c * cell, my + (n - 1 - r) * cell),
                        cell,
                        cell,
                        facecolor="white",
                        edgecolor=PALETTE["grid"],
                        linewidth=0.30,
                        zorder=1,
                    )
                )
        for idx, d in enumerate(diags):
            tone = [PALETTE["cyan_light"], PALETTE["blue_light"], PALETTE["orange_light"]][idx % 3]
            edge = [PALETTE["cyan"], PALETTE["blue"], PALETTE["orange"]][idx % 3]
            for r in range(n):
                c = r + d
                if 0 <= c < n:
                    ax.add_patch(
                        patches.Rectangle(
                            (mx + c * cell + 0.055, my + (n - 1 - r) * cell + 0.055),
                            cell - 0.11,
                            cell - 0.11,
                            facecolor=tone,
                            edgecolor=edge,
                            linewidth=0.28,
                            alpha=0.88,
                            zorder=2,
                        )
                    )
        ax.add_patch(
            patches.Rectangle((mx, my), msize, msize, facecolor="none", edgecolor=PALETTE["ink"], linewidth=0.62, zorder=3)
        )
        ax.text(
            mx + msize / 2,
            my - 0.62,
            rf"$D_{{{group_idx}}}$",
            ha="center",
            va="top",
            fontsize=5.8,
            color=PALETTE["muted"],
        )

    for i, diags in enumerate(groups):
        col = i % 3
        row = i // 3
        mx = x0 + col * (msize + gap_x)
        my = y0 + (1 - row) * (msize + gap_y)
        draw_one_matrix(mx, my, diags, i + 1)

    field_w = 3 * msize + 2 * gap_x
    ellipsis_color = PALETTE["grey_mid"]
    ax.text(x0 + field_w + 0.65, y0 + msize + gap_y + msize / 2, r"$\cdots$", ha="center", va="center", fontsize=12, color=ellipsis_color)
    ax.text(x0 + field_w + 0.65, y0 + msize / 2, r"$\cdots$", ha="center", va="center", fontsize=12, color=ellipsis_color)
    ax.text(x0 + field_w / 2, y0 + msize + 0.55, r"$\vdots$", ha="center", va="center", fontsize=11, color=ellipsis_color)
    ax.text(x0 + field_w / 2, y0 - 0.95, r"$\cdots$", ha="center", va="center", fontsize=12, color=ellipsis_color)
    ax.text(
        x0 + field_w / 2,
        y0 - 2.5,
        r"many diagonal groups  $\rightarrow$  aggregate demand profile",
        ha="center",
        va="center",
        fontsize=6.8,
        color=PALETTE["muted"],
    )

    # Rotation indices live on a modular ring; frequency appears as heat around the ring.
    ring_cx = x0 + field_w + 6.1
    ring_cy = y0 + 5.6
    radius = 4.1
    draw_index_ring(ax, ring_cx, ring_cy, radius, "index\ndemand", tick_count=32)
    vals = [0.10, 0.62, 0.30, 0.88, 0.18, 0.35, 0.71, 0.22, 0.45, 0.80, 0.26, 0.12, 0.66, 0.38, 0.20, 0.91]
    for i, val in enumerate(vals):
        angle = 90 - i * 360 / len(vals)
        color = PALETTE["cyan"] if val > 0.65 else PALETTE["blue"]
        x1, y1 = polar_xy(ring_cx, ring_cy, radius * 0.98, angle)
        x2, y2 = polar_xy(ring_cx, ring_cy, radius * (1.02 + 0.34 * val), angle)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.8 + 1.4 * val, alpha=0.82, zorder=5)
    for angle in [24, 112, 188, 272]:
        draw_ring_point(ax, ring_cx, ring_cy, radius * 1.32, angle, PALETTE["orange"], marker="s", size=14)
    ax.text(ring_cx, ring_cy + radius + 1.55, "hot indices are\nvisible on the ring", ha="center", va="bottom", fontsize=6.3, color=PALETTE["muted"])
    ax.text(
        ring_cx,
        ring_cy - radius - 1.75,
        r"$\mathcal{D}_{profile}=\{(r,f_r)\}$",
        ha="center",
        va="top",
        fontsize=6.6,
        color=PALETTE["muted"],
    )


def draw_bsgs_lattice(ax, x0, y0, cols=7, rows=6, cell=2.82):
    cx, cy = x0 + 11.0, y0 + 8.2
    radius = 7.0
    draw_index_ring(ax, cx, cy, radius, "rotation\nindex", tick_count=32)

    start = 205
    ax.text(*polar_xy(cx, cy, radius + 1.65, start), "$0$", ha="center", va="center", fontsize=6.6, color=PALETTE["muted"])
    after_giant = draw_ring_segment(
        ax,
        cx,
        cy,
        radius,
        start,
        100,
        PALETTE["cyan"],
        width=0.9,
        label=r"$g\cdot B$",
        label_offset=2.0,
    )
    end = draw_ring_segment(
        ax,
        cx,
        cy,
        radius,
        after_giant,
        34,
        PALETTE["blue"],
        width=0.9,
        label=r"$b$",
        label_offset=2.0,
    )
    draw_ring_point(ax, cx, cy, radius + 0.55, end, PALETTE["ink"], marker="*", size=48, label=r"$r$")

    for angle, state in [(160, "reuse"), (226, "reuse"), (252, "reuse"), (294, "new"), (336, "reuse"), (28, "new")]:
        draw_ring_point(
            ax,
            cx,
            cy,
            radius * 1.12,
            angle,
            PALETTE["green"] if state == "reuse" else PALETTE["red"],
            marker="o" if state == "reuse" else "D",
            size=20 if state == "reuse" else 24,
        )

    rounded_label(
        ax,
        x0 + 1.2,
        y0 + 15.0,
        r"$r = g\cdot B + b\;(\mathrm{mod}\;N)$",
        w=20.2,
        h=2.45,
        fc=PALETTE["cyan_light"],
        ec=PALETTE["cyan"],
        fs=7.4,
        bold=True,
    )
    ax.text(
        x0 + 11.0,
        y0 - 1.2,
        "2-D BSGS becomes two additive displacements on the same modular index ring.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=PALETTE["muted"],
    )
    return x0 + 22.0, y0 + 16.5


def draw_chip(ax, x, y, w, h, label, sublabel, fill, edge, marker_color):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.55",
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.85,
            zorder=2,
        )
    )
    for i in range(6):
        pin_y = y + 0.6 + i * (h - 1.2) / 5
        ax.plot([x - 0.45, x], [pin_y, pin_y], color=edge, linewidth=0.55, zorder=1)
        ax.plot([x + w, x + w + 0.45], [pin_y, pin_y], color=edge, linewidth=0.55, zorder=1)
    ax.text(x + w / 2, y + h - 1.15, label, ha="center", va="center", fontsize=8.0, fontweight="bold", color=PALETTE["ink"])
    ax.text(x + w / 2, y + h - 2.85, sublabel, ha="center", va="center", fontsize=6.6, color=PALETTE["muted"])
    rng = np.random.default_rng(7)
    for i in range(13):
        px = x + 1.2 + (i % 7) * (w - 2.4) / 6
        py = y + 0.9 + (i // 7) * 1.08 + rng.uniform(-0.08, 0.08)
        ax.add_patch(
            patches.Circle(
                (px, py),
                0.23,
                facecolor=marker_color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.96,
                zorder=4,
            )
        )


def draw_key_hierarchy(ax, x0, y0):
    draw_chip(
        ax,
        x0,
        y0 + 12.1,
        21.0,
        6.1,
        r"key pool  $\mathcal{K}_{pool}$",
        "cache-like reusable rotation keys",
        PALETTE["blue_light"],
        PALETTE["blue"],
        PALETTE["blue"],
    )
    draw_chip(
        ax,
        x0 + 1.0,
        y0 + 5.6,
        19.0,
        5.5,
        "reuse hits",
        r"conjugation folds $k$ and $-k$",
        PALETTE["green_light"],
        PALETTE["green"],
        PALETTE["green"],
    )
    draw_chip(
        ax,
        x0 + 2.1,
        y0 - 0.3,
        16.8,
        5.0,
        r"extension  $\mathcal{K}_{new}$",
        "keys generated only for misses",
        PALETTE["red_light"],
        PALETTE["red"],
        PALETTE["red"],
    )

    # Dotted, non-directional associations emphasize mapping rather than execution order.
    for x1, y1, x2, y2 in [
        (x0 + 10.5, y0 + 12.1, x0 + 10.5, y0 + 11.1),
        (x0 + 10.5, y0 + 5.6, x0 + 10.5, y0 + 4.7),
    ]:
        ax.plot([x1, x2], [y1, y2], color=PALETTE["grey_mid"], linewidth=0.75, linestyle=(0, (2, 2)))

    ax.text(x0 - 0.2, y0 - 2.0, "best low-key point:", ha="left", va="top", fontsize=6.6, color=PALETTE["muted"])
    ax.text(
        x0 + 8.2,
        y0 - 2.0,
        r"$16$ new keys, $24$ reused",
        ha="left",
        va="top",
        fontsize=7.1,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    ax.text(
        x0 + 8.2,
        y0 - 3.8,
        r"$4365$ online rotations",
        ha="left",
        va="top",
        fontsize=7.0,
        color=PALETTE["muted"],
    )


def draw_multilevel_decoder(ax, x0, y0):
    ax.text(
        x0,
        y0 + 20.5,
        "Multi-dimensional additive displacement view",
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        x0,
        y0 + 18.45,
        r"each level contributes one modular displacement; the composition operator between levels is only $+$",
        ha="left",
        va="top",
        fontsize=6.9,
        color=PALETTE["muted"],
    )

    levels = [
        (r"$\Delta_1$", 32, PALETTE["blue"], r"$S_1=\{0,\ldots,7\}$"),
        (r"$\Delta_2$", 70, PALETTE["cyan"], r"$S_2=\{0,8,\ldots,15\cdot8\}$"),
        (r"$\Delta_3$", 53, PALETTE["green"], r"$S_3=\{0,128,\ldots,15\cdot128\}$"),
        (r"$\Delta_4$", 38, PALETTE["orange"], r"$S_4=\{0,2048,\ldots,15\cdot2048\}$"),
    ]

    cx, cy = x0 + 23.0, y0 + 9.2
    radius = 7.1
    draw_index_ring(ax, cx, cy, radius, "same\nindex ring", tick_count=32)
    start = 145
    current = start
    ax.text(*polar_xy(cx, cy, radius + 1.5, start), "$0$", ha="center", va="center", fontsize=6.5, color=PALETTE["muted"])
    for i, (term, extent, color, _) in enumerate(levels):
        current = draw_ring_segment(
            ax,
            cx,
            cy,
            radius + i * 0.54,
            current,
            extent,
            color,
            width=0.55,
            alpha=0.78,
            label=term,
            label_offset=1.85 + i * 0.25,
        )
    draw_ring_point(ax, cx, cy, radius + 2.25, current, PALETTE["ink"], marker="*", size=56, label=r"$r$")

    box_x = x0 + 50.5
    top_y = y0 + 14.2
    for i, (term, _, color, label) in enumerate(levels):
        y = top_y - i * 3.2
        ax.add_patch(
            patches.FancyBboxPatch(
                (box_x, y),
                9.5,
                1.85,
                boxstyle="round,pad=0.12,rounding_size=0.35",
                facecolor=PALETTE["grey_light"],
                edgecolor=color,
                linewidth=0.72,
                zorder=2,
            )
        )
        ax.text(box_x + 4.75, y + 0.92, term, ha="center", va="center", fontsize=7.5, color=color, fontweight="bold")
        ax.text(box_x + 11.2, y + 0.92, label, ha="left", va="center", fontsize=6.3, color=PALETTE["muted"])
        if i < len(levels) - 1:
            ax.text(box_x + 4.75, y - 0.7, "+", ha="center", va="center", fontsize=12.0, color=PALETTE["ink"])

    sum_x = x0 + 78.0
    sum_y = y0 + 8.0
    ax.add_patch(
        patches.FancyBboxPatch(
            (sum_x, sum_y),
            15.0,
            4.0,
            boxstyle="round,pad=0.18,rounding_size=0.5",
            facecolor=PALETTE["cyan_light"],
            edgecolor=PALETTE["cyan"],
            linewidth=0.85,
            zorder=2,
        )
    )
    ax.text(sum_x + 7.5, sum_y + 2.55, r"$r=\Delta_1+\Delta_2+\Delta_3+\Delta_4$", ha="center", va="center", fontsize=7.4, color=PALETTE["ink"], fontweight="bold")
    ax.text(sum_x + 7.5, sum_y + 1.15, r"$(\mathrm{mod}\;N)$", ha="center", va="center", fontsize=6.8, color=PALETTE["muted"])

    ax.plot([box_x + 9.7, sum_x], [top_y - 4.0, sum_y + 2.0], color=PALETTE["grey_mid"], linewidth=0.65, linestyle=(0, (2, 2)))
    ax.plot([cx + radius + 2.7, sum_x], [cy, sum_y + 2.0], color=PALETTE["grey_mid"], linewidth=0.65, linestyle=(0, (2, 2)))

    ax.text(
        x0 + 47.0,
        y0 + 0.45,
        "High-dimensional search is visualized as layered modular jumps, not as an unreadable dense 4-D cube.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["muted"],
    )
    ax.text(
        x0 + 47.0,
        y0 - 1.15,
        r"example level sizes:  $8\times16\times16\times16$; composition constraint: additive terms only",
        ha="center",
        va="top",
        fontsize=7.1,
        color=PALETTE["ink"],
    )


def draw_association_curve(ax, start, end, color=PALETTE["grey_mid"]):
    sx, sy = start
    ex, ey = end
    verts = [
        (sx, sy),
        (sx + (ex - sx) * 0.35, sy + 4.0),
        (sx + (ex - sx) * 0.62, ey - 4.0),
        (ex, ey),
    ]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = patches.PathPatch(path, facecolor="none", edgecolor=color, linewidth=0.75, linestyle=(0, (2, 3)), zorder=1)
    ax.add_patch(patch)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=PALETTE["bg"], constrained_layout=False)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.90])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    fig.text(
        0.5,
        0.967,
        "Intuitive View of Rotation-Key Co-design",
        ha="center",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.5,
        0.936,
        "Diagonal demand is projected onto additive coordinate layers; cache hits reuse existing keys, misses define the minimal key extension.",
        ha="center",
        va="top",
        fontsize=8.0,
        color=PALETTE["muted"],
    )

    draw_panel_title(ax, 3.0, 56.0, "A", "Demand field", "diagonal rotations are a spatial profile")
    draw_matrix(ax, 4.5, 35.0)

    draw_panel_title(ax, 35.0, 56.0, "B", "2-step additive displacement", "BSGS as giant jump plus baby correction")
    lattice_end = draw_bsgs_lattice(ax, 38.0, 34.1)

    draw_panel_title(ax, 73.0, 56.0, "C", "Cache-like key hierarchy", "reuse hits vs. generated misses")
    draw_key_hierarchy(ax, 74.0, 32.0)

    draw_multilevel_decoder(ax, 5.0, 6.0)

    # Directionless associations show correspondence, not execution order.
    draw_association_curve(ax, (27.0, 44.0), (38.0, 44.2))
    draw_association_curve(ax, (lattice_end[0] + 1.0, 43.8), (74.0, 49.0))

    # Legend uses semantic markers rather than step numbers.
    legend_x, legend_y = 37.5, 30.1
    ax.scatter([legend_x], [legend_y], s=22, marker="o", color=PALETTE["green"], edgecolor="white", linewidth=0.5, zorder=5)
    ax.text(legend_x + 1.0, legend_y, "reuse hit", va="center", ha="left", fontsize=6.9, color=PALETTE["muted"])
    ax.scatter([legend_x + 10.0], [legend_y], s=26, marker="D", color=PALETTE["red"], edgecolor="white", linewidth=0.5, zorder=5)
    ax.text(legend_x + 11.0, legend_y, "new-key miss", va="center", ha="left", fontsize=6.9, color=PALETTE["muted"])

    out_svg = OUT_DIR / "codesign_intuitive_schematic.svg"
    out_pdf = OUT_DIR / "codesign_intuitive_schematic.pdf"
    out_png = OUT_DIR / "codesign_intuitive_schematic.png"
    for path in [out_svg, out_pdf, out_png]:
        fig.savefig(path, facecolor=PALETTE["bg"], edgecolor="none", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)
    print(f"[OK] saved {out_svg}")
    print(f"[OK] saved {out_pdf}")
    print(f"[OK] saved {out_png}")


if __name__ == "__main__":
    main()
