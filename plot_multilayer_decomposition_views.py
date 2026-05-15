"""
Four standalone views for explaining multi-layer additive decomposition.

The four figures correspond to:
2. cumulative sum trace on a modular ring
3. stacked transparent layers
4. mixed-radix digit view
5. 3D voxel/slice projection
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

LAYER_COLORS = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["orange"]]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 420,
        }
    )


def polar_xy(cx, cy, radius, angle_deg):
    angle = np.deg2rad(angle_deg)
    return cx + radius * np.cos(angle), cy + radius * np.sin(angle)


def new_canvas(title, subtitle):
    fig = plt.figure(figsize=(6.4, 4.0), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.035, 0.045, 0.93, 0.84])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12.0, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.5, 0.915, subtitle, ha="center", va="top", fontsize=7.4, color=PALETTE["muted"])
    return fig, ax


def save_all(fig, stem):
    OUT_DIR.mkdir(exist_ok=True)
    paths = [
        OUT_DIR / f"{stem}.svg",
        OUT_DIR / f"{stem}.pdf",
        OUT_DIR / f"{stem}.png",
    ]
    for path in paths:
        fig.savefig(path, facecolor=PALETTE["bg"], edgecolor="none", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)
    for path in paths:
        print(f"[OK] saved {path}")


def rounded_box(ax, x, y, w, h, text, fc, ec, fs=7.0, bold=False):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=1.2",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=PALETTE["ink"],
        fontweight="bold" if bold else "normal",
        zorder=4,
    )


def draw_mod_ring(ax, cx, cy, radius, label=None, tick_count=32, alpha=1.0):
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            radius,
            facecolor="white",
            edgecolor=PALETTE["ink"],
            linewidth=1.0,
            alpha=alpha,
            zorder=1,
        )
    )
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            radius * 0.72,
            facecolor=PALETTE["grey_light"],
            edgecolor=PALETTE["grid"],
            linewidth=0.45,
            alpha=0.68 * alpha,
            zorder=0,
        )
    )
    for i in range(tick_count):
        angle = i * 360 / tick_count
        x1, y1 = polar_xy(cx, cy, radius * 0.91, angle)
        x2, y2 = polar_xy(cx, cy, radius * 1.045, angle)
        ax.plot([x1, x2], [y1, y2], color=PALETTE["grid"], linewidth=0.43, alpha=alpha, zorder=2)
    if label:
        ax.text(cx, cy + 0.8, label, ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["ink"])
        ax.text(cx, cy - 1.1, r"$\mathrm{mod}\;N$", ha="center", va="center", fontsize=6.3, color=PALETTE["muted"])


def draw_arc(ax, cx, cy, radius, start, extent, color, width=1.4, label=None, label_radius=2.0, alpha=0.85):
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
            linewidth=0.7,
            alpha=alpha,
            zorder=5,
        )
    )
    ex, ey = polar_xy(cx, cy, radius + width * 0.1, end)
    ax.scatter([ex], [ey], s=26, color=color, edgecolor="white", linewidth=0.6, zorder=7)
    if label:
        lx, ly = polar_xy(cx, cy, radius + label_radius, start + extent / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.0, color=color, fontweight="bold", zorder=8)
    return end


def label_point(ax, cx, cy, radius, angle, label, color=PALETTE["ink"], marker="o", size=35, offset=2.6):
    x, y = polar_xy(cx, cy, radius, angle)
    ax.scatter([x], [y], s=size, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=8)
    lx, ly = polar_xy(cx, cy, radius + offset, angle)
    ax.text(lx, ly, label, ha="center", va="center", fontsize=6.8, color=color, fontweight="bold", zorder=8)


def draw_dashed_curve(ax, start, end):
    sx, sy = start
    ex, ey = end
    verts = [(sx, sy), (sx + 15, sy + 7), (ex - 15, ey - 6), (ex, ey)]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(
        patches.PathPatch(path, facecolor="none", edgecolor=PALETTE["grey_mid"], linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    )


def figure_2_cumulative_sum_trace():
    fig, ax = new_canvas(
        "2. Cumulative Sum Trace on a Modular Ring",
        r"Intermediate sums make the additive constraint explicit: $s_i=s_{i-1}+\Delta_i$.",
    )
    cx, cy, radius = 39, 31, 17.0
    draw_mod_ring(ax, cx, cy, radius, "index\nring")

    start = 215
    extents = [55, 78, 48]
    labels = [r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$"]
    states = [(start, r"$s_0=0$")]
    current = start
    for i, (extent, label, color) in enumerate(zip(extents, labels, LAYER_COLORS)):
        current = draw_arc(ax, cx, cy, radius, current, extent, color, width=1.5, label=label, label_radius=2.8)
        states.append((current, rf"$s_{i + 1}$" if i < 2 else r"$s_3=r$"))

    for i, (angle, label) in enumerate(states):
        marker = "*" if i == len(states) - 1 else "o"
        size = 70 if marker == "*" else 34
        color = PALETTE["red"] if marker == "*" else PALETTE["ink"]
        label_point(ax, cx, cy, radius + 0.65, angle, label, color=color, marker=marker, size=size, offset=2.8)

    rounded_box(
        ax,
        67,
        35.5,
        25.0,
        8.5,
        r"$s_1=\Delta_1$" + "\n" + r"$s_2=s_1+\Delta_2$" + "\n" + r"$r=s_3=s_2+\Delta_3$",
        PALETTE["cyan_light"],
        PALETTE["cyan"],
        fs=6.9,
        bold=False,
    )
    rounded_box(
        ax,
        67,
        23.0,
        25.0,
        6.8,
        r"$r=\Delta_1+\Delta_2+\Delta_3\;(\mathrm{mod}\;N)$",
        PALETTE["grey_light"],
        PALETTE["ink"],
        fs=7.2,
        bold=True,
    )
    ax.text(39, 7.5, "Each colored arc is one layer; the dots are cumulative sums, not stages in a workflow.", ha="center", fontsize=6.8, color=PALETTE["muted"])
    save_all(fig, "decomp_view_2_cumulative_sum_trace")


def figure_3_stacked_transparent_layers():
    fig, ax = new_canvas(
        "3. Stacked Transparent Layers",
        "Each layer is a transparent copy of the same modular ring; stacking aligns their offsets.",
    )
    layer_specs = [
        (20, 45, r"Layer 1", r"$\Delta_1$", 215, 50, LAYER_COLORS[0]),
        (20, 31, r"Layer 2", r"$\Delta_2$", 265, 75, LAYER_COLORS[1]),
        (20, 17, r"Layer 3", r"$\Delta_3$", 340, 48, LAYER_COLORS[2]),
    ]
    for cx, cy, layer, delta, start, extent, color in layer_specs:
        ax.add_patch(
            patches.FancyBboxPatch(
                (cx - 9.3, cy - 7.0),
                18.6,
                14.0,
                boxstyle="round,pad=0.15,rounding_size=1.0",
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.10,
                zorder=0,
            )
        )
        draw_mod_ring(ax, cx, cy, 5.4, None, tick_count=24, alpha=0.86)
        draw_arc(ax, cx, cy, 5.4, start, extent, color, width=0.9, label=delta, label_radius=1.7, alpha=0.82)
        ax.text(cx - 13.7, cy + 4.6, layer, ha="left", va="center", fontsize=7.0, color=PALETTE["ink"], fontweight="bold")
        if cy > 18:
            ax.text(cx, cy - 8.2, "+", ha="center", va="center", fontsize=13.5, color=PALETTE["ink"])

    overlay_cx, overlay_cy, overlay_r = 66, 31, 13.0
    draw_mod_ring(ax, overlay_cx, overlay_cy, overlay_r, "aligned\nring")
    current = 215
    for delta, extent, color, offset in [
        (r"$\Delta_1$", 50, LAYER_COLORS[0], 2.5),
        (r"$\Delta_2$", 75, LAYER_COLORS[1], 3.1),
        (r"$\Delta_3$", 48, LAYER_COLORS[2], 3.7),
    ]:
        current = draw_arc(ax, overlay_cx, overlay_cy, overlay_r, current, extent, color, width=1.0, label=delta, label_radius=offset, alpha=0.70)
    label_point(ax, overlay_cx, overlay_cy, overlay_r + 0.8, current, r"$r$", color=PALETTE["red"], marker="*", size=70, offset=2.3)

    for _, cy, _, _, _, _, _ in layer_specs:
        draw_dashed_curve(ax, (30, cy), (52, overlay_cy))
    rounded_box(
        ax,
        51.0,
        7.8,
        34.0,
        5.2,
        r"overlay result: $\Delta_1+\Delta_2+\Delta_3$",
        PALETTE["cyan_light"],
        PALETTE["cyan"],
        fs=6.8,
        bold=True,
    )
    ax.text(66, 51.8, "Same coordinate frame, different layer colors", ha="center", fontsize=6.9, color=PALETTE["muted"])
    save_all(fig, "decomp_view_3_stacked_transparent_layers")


def figure_4_mixed_radix_digit_view():
    fig, ax = new_canvas(
        "4. Mixed-Radix Digit View",
        r"Digits choose layer-local offsets; only the produced offsets are added across layers.",
    )

    rows = [
        (43, r"$d_1=5$", r"$\sigma_1=1$", r"$\Delta_1=d_1\sigma_1$", LAYER_COLORS[0]),
        (31, r"$d_2=3$", r"$\sigma_2=8$", r"$\Delta_2=d_2\sigma_2$", LAYER_COLORS[1]),
        (19, r"$d_3=2$", r"$\sigma_3=128$", r"$\Delta_3=d_3\sigma_3$", LAYER_COLORS[2]),
    ]
    for y, digit, stride, delta, color in rows:
        rounded_box(ax, 8, y - 3.0, 13.0, 6.0, digit, "white", color, fs=8.2, bold=True)
        ax.text(25, y, r"$\times$", ha="center", va="center", fontsize=12, color=PALETTE["ink"])
        rounded_box(ax, 29, y - 3.0, 15.0, 6.0, stride, PALETTE["grey_light"], PALETTE["grid"], fs=7.8, bold=True)
        ax.text(47, y, r"$=$", ha="center", va="center", fontsize=11, color=PALETTE["ink"])
        rounded_box(ax, 51, y - 3.0, 20.0, 6.0, delta, color + "22" if False else PALETTE["grey_light"], color, fs=7.2, bold=True)
        if y > 20:
            ax.text(61, y - 6.0, "+", ha="center", va="center", fontsize=13, color=PALETTE["ink"])

    rounded_box(
        ax,
        76,
        24.5,
        19.0,
        12.0,
        r"$r=\Delta_1+\Delta_2+\Delta_3$" + "\n" + r"$(\mathrm{mod}\;N)$",
        PALETTE["cyan_light"],
        PALETTE["cyan"],
        fs=7.6,
        bold=True,
    )
    for y, _, _, _, color in rows:
        ax.annotate(
            "",
            xy=(76, 30.5),
            xytext=(71, y),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.9, linestyle=(0, (2, 2))),
            zorder=1,
        )

    cx, cy, radius = 29, 8.5, 6.2
    draw_mod_ring(ax, cx, cy, radius, None, tick_count=20)
    start = 195
    for extent, color, label in [(30, LAYER_COLORS[0], r"$\Delta_1$"), (55, LAYER_COLORS[1], r"$\Delta_2$"), (38, LAYER_COLORS[2], r"$\Delta_3$")]:
        start = draw_arc(ax, cx, cy, radius, start, extent, color, width=0.8, label=label, label_radius=1.4)
    ax.text(54, 8.5, "The multiplication is internal to a layer;\nlayer composition is still additive.", ha="center", va="center", fontsize=6.9, color=PALETTE["muted"])
    save_all(fig, "decomp_view_4_mixed_radix_digits")


def project_iso(x, y, z, origin=(19, 18), sx=5.2, sy=3.1, sz=4.6):
    ox, oy = origin
    return ox + sx * (x - y), oy + sy * (x + y) + sz * z


def draw_iso_cube(ax, x, y, z, size=0.52, color=PALETTE["red"], alpha=0.9):
    pts = [
        project_iso(x, y, z),
        project_iso(x + size, y, z),
        project_iso(x + size, y + size, z),
        project_iso(x, y + size, z),
    ]
    top = [
        project_iso(x, y, z + size),
        project_iso(x + size, y, z + size),
        project_iso(x + size, y + size, z + size),
        project_iso(x, y + size, z + size),
    ]
    side1 = [pts[0], pts[1], top[1], top[0]]
    side2 = [pts[1], pts[2], top[2], top[1]]
    ax.add_patch(patches.Polygon(side1, facecolor=color, edgecolor="white", linewidth=0.5, alpha=alpha * 0.75, zorder=5))
    ax.add_patch(patches.Polygon(side2, facecolor=color, edgecolor="white", linewidth=0.5, alpha=alpha * 0.55, zorder=5))
    ax.add_patch(patches.Polygon(top, facecolor=color, edgecolor="white", linewidth=0.5, alpha=alpha, zorder=6))


def draw_iso_plane(ax, corners, color, alpha=0.16):
    pts = [project_iso(*p) for p in corners]
    ax.add_patch(patches.Polygon(pts, facecolor=color, edgecolor=color, linewidth=0.8, alpha=alpha, zorder=2))


def figure_5_voxel_slice_projection():
    fig, ax = new_canvas(
        "5. 3-D Voxel / Slice Projection",
        r"For three layers, a lattice point $(\Delta_1,\Delta_2,\Delta_3)$ projects to one modular index $r$.",
    )

    # Grid points.
    for x in range(5):
        for y in range(5):
            for z in range(4):
                px, py = project_iso(x, y, z)
                ax.scatter([px], [py], s=7, color=PALETTE["grid"], edgecolor="none", zorder=1)

    # Axes.
    for end, label, color, text_shift in [
        ((5.1, 0, 0), r"$\Delta_1$", LAYER_COLORS[0], (3, -1)),
        ((0, 5.1, 0), r"$\Delta_2$", LAYER_COLORS[1], (-5, 0)),
        ((0, 0, 4.3), r"$\Delta_3$", LAYER_COLORS[2], (-1, 4)),
    ]:
        sx, sy = project_iso(0, 0, 0)
        ex, ey = project_iso(*end)
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.0), zorder=3)
        ax.text(ex + text_shift[0], ey + text_shift[1], label, ha="center", va="center", fontsize=7.4, color=color, fontweight="bold")

    selected = (3, 2, 3)
    draw_iso_plane(ax, [(selected[0], 0, 0), (selected[0], 4.5, 0), (selected[0], 4.5, 4.0), (selected[0], 0, 4.0)], LAYER_COLORS[0], alpha=0.11)
    draw_iso_plane(ax, [(0, selected[1], 0), (4.5, selected[1], 0), (4.5, selected[1], 4.0), (0, selected[1], 4.0)], LAYER_COLORS[1], alpha=0.12)
    draw_iso_plane(ax, [(0, 0, selected[2]), (4.5, 0, selected[2]), (4.5, 4.5, selected[2]), (0, 4.5, selected[2])], LAYER_COLORS[2], alpha=0.13)
    draw_iso_cube(ax, *selected, size=0.7, color=PALETTE["red"], alpha=0.88)
    target_px, target_py = project_iso(selected[0] + 0.35, selected[1] + 0.35, selected[2] + 0.75)
    ax.text(target_px + 3, target_py + 3, "selected\nlattice point", ha="left", va="center", fontsize=6.8, color=PALETTE["ink"], fontweight="bold")

    ring_cx, ring_cy, ring_r = 74, 31, 11.2
    draw_mod_ring(ax, ring_cx, ring_cy, ring_r, "projected\nindex")
    current = 205
    for extent, color, label in [(42, LAYER_COLORS[0], r"$\Delta_1$"), (62, LAYER_COLORS[1], r"$\Delta_2$"), (46, LAYER_COLORS[2], r"$\Delta_3$")]:
        current = draw_arc(ax, ring_cx, ring_cy, ring_r, current, extent, color, width=1.0, label=label, label_radius=2.2)
    label_point(ax, ring_cx, ring_cy, ring_r + 0.7, current, r"$r$", color=PALETTE["red"], marker="*", size=68, offset=2.3)

    ax.annotate(
        "",
        xy=(ring_cx - ring_r - 3, ring_cy),
        xytext=(target_px + 10, target_py),
        arrowprops=dict(arrowstyle="->", color=PALETTE["grey_mid"], linewidth=1.0, linestyle=(0, (2, 2))),
        zorder=3,
    )
    rounded_box(
        ax,
        55.0,
        8.3,
        40.0,
        7.0,
        "projection rule:\n" + r"$r=\Delta_1+\Delta_2+\Delta_3\;(\mathrm{mod}\;N)$",
        PALETTE["cyan_light"],
        PALETTE["cyan"],
        fs=6.5,
        bold=True,
    )
    ax.text(21, 4.2, "Useful for exactly three layers; higher dimensions need ring/layer projections.", ha="left", fontsize=6.6, color=PALETTE["muted"])
    save_all(fig, "decomp_view_5_voxel_slice_projection")


def main() -> None:
    setup_style()
    figure_2_cumulative_sum_trace()
    figure_3_stacked_transparent_layers()
    figure_4_mixed_radix_digit_view()
    figure_5_voxel_slice_projection()


if __name__ == "__main__":
    main()
