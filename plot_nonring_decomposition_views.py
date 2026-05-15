"""
Seven non-ring views for explaining multi-layer additive decomposition.

Each figure explains the same constraint:
    r = Delta_1 + Delta_2 + Delta_3 (mod N)
without using a circular modular ring.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


OUT_DIR = Path("results")

PALETTE = {
    "bg": "#FFFFFF",
    "ink": "#25313B",
    "muted": "#66727C",
    "grid": "#D8E0E6",
    "blue": "#2F5F88",
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

LAYER_COLORS = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"]]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 420,
        }
    )


def new_canvas(title: str, subtitle: str):
    fig = plt.figure(figsize=(6.4, 4.0), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.82])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12.0, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.5, 0.915, subtitle, ha="center", va="top", fontsize=7.2, color=PALETTE["muted"])
    return fig, ax


def save_all(fig, stem: str) -> None:
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


def rounded_box(ax, x, y, w, h, text, fc, ec, fs=7.0, bold=False, lw=0.9):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=1.1",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
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


def arrow(ax, start, end, color=PALETTE["ink"], lw=1.2, style="-|>", dashed=False, z=5):
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            linestyle=(0, (2, 2)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def xmap(value, x0=12, x1=88, n=100):
    return x0 + (x1 - x0) * value / n


def draw_ruler_axis(ax, y, x0=12, x1=88, n=100, label_y=-2.1):
    ax.plot([x0, x1], [y, y], color=PALETTE["ink"], linewidth=0.9, zorder=1)
    for value in range(0, n + 1, 10):
        x = xmap(value, x0, x1, n)
        tick_h = 1.1 if value % 25 else 1.8
        ax.plot([x, x], [y - tick_h / 2, y + tick_h / 2], color=PALETTE["grid"], linewidth=0.7, zorder=2)
    for value, label in [(0, "$0$"), (25, "$N/4$"), (50, "$N/2$"), (75, "$3N/4$"), (100, "$N$")]:
        ax.text(xmap(value, x0, x1, n), y + label_y, label, ha="center", va="top", fontsize=6.2, color=PALETTE["muted"])


def figure_1_unwrapped_modular_ruler():
    fig, ax = new_canvas(
        "1. Unwrapped Modular Ruler",
        "Cut the modular space at 0 and show layer offsets as adjacent segments on a ruler.",
    )
    y = 31
    draw_ruler_axis(ax, y)

    n = 100
    deltas = [30, 45, 38]
    labels = [r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$"]
    current = 0
    path_y = y + 7
    ax.text(xmap(0), path_y + 4.2, "$s_0=0$", ha="center", fontsize=6.8, color=PALETTE["ink"])
    for delta, label, color in zip(deltas, labels, LAYER_COLORS):
        remaining = delta
        while remaining > 0:
            end = min(n, current + remaining)
            arrow(ax, (xmap(current), path_y), (xmap(end), path_y), color=color, lw=3.0)
            mid = (current + end) / 2
            ax.text(xmap(mid), path_y + 3.0, label, ha="center", va="bottom", fontsize=7.0, color=color, fontweight="bold")
            remaining -= end - current
            current = end
            if remaining > 0:
                ax.plot([xmap(n), xmap(n)], [path_y - 2.0, path_y + 2.0], color=PALETTE["red"], linewidth=1.0)
                ax.text(xmap(n), path_y - 4.0, "wrap", ha="center", va="top", fontsize=6.2, color=PALETTE["red"])
                arrow(ax, (xmap(n), path_y - 2.8), (xmap(0), path_y - 2.8), color=PALETTE["red"], lw=0.9, dashed=True)
                current = 0
        ax.scatter([xmap(current)], [path_y], s=28, color=color, edgecolor="white", linewidth=0.6, zorder=8)

    ax.scatter([xmap(current)], [path_y], s=88, marker="*", color=PALETTE["red"], edgecolor="white", linewidth=0.7, zorder=9)
    ax.text(xmap(current), path_y + 5.4, r"$r=(\Delta_1+\Delta_2+\Delta_3)\;(\mathrm{mod}\;N)$", ha="center", fontsize=7.0, color=PALETTE["red"], fontweight="bold")
    rounded_box(ax, 26, 10.5, 48, 7.5, r"Only adjacency matters: each layer appends one offset segment.", PALETTE["grey_light"], PALETTE["grid"], fs=7.0)
    save_all(fig, "nonring_view_1_unwrapped_modular_ruler")


def figure_2_stacked_number_line():
    fig, ax = new_canvas(
        "2. Stacked Number Line",
        "Layer-local choices are shown separately, then aggregated into one cumulative line.",
    )
    deltas = [22, 36, 18]
    ys = [47, 38, 29]
    for i, (delta, y, color) in enumerate(zip(deltas, ys, LAYER_COLORS), start=1):
        ax.text(8, y + 1.0, rf"$L_{i}$", ha="right", va="center", fontsize=7.4, color=color, fontweight="bold")
        draw_ruler_axis(ax, y, x0=12, x1=58, n=50, label_y=1.8)
        arrow(ax, (xmap(0, 12, 58, 50), y + 2.5), (xmap(delta, 12, 58, 50), y + 2.5), color=color, lw=2.4)
        ax.scatter([xmap(delta, 12, 58, 50)], [y + 2.5], s=26, color=color, edgecolor="white", linewidth=0.5, zorder=7)
        ax.text(xmap(delta, 12, 58, 50), y + 5.0, rf"$\Delta_{i}$", ha="center", fontsize=6.8, color=color, fontweight="bold")
        if i < 3:
            ax.text(35, y - 4.3, "+", ha="center", va="center", fontsize=13, color=PALETTE["ink"])

    base_x, base_y = 66, 18
    cumulative = 0
    scale = 0.33
    ax.text(base_x, base_y + 12.5, "aggregate line", ha="left", fontsize=7.2, fontweight="bold", color=PALETTE["ink"])
    ax.plot([base_x, base_x + 30], [base_y, base_y], color=PALETTE["ink"], linewidth=0.8)
    ax.text(base_x, base_y - 2.2, "$0$", ha="center", fontsize=6.2, color=PALETTE["muted"])
    for i, (delta, color) in enumerate(zip(deltas, LAYER_COLORS), start=1):
        x0 = base_x + cumulative * scale
        x1 = base_x + (cumulative + delta) * scale
        ax.add_patch(patches.Rectangle((x0, base_y - 1.7), x1 - x0, 3.4, facecolor=color, edgecolor="white", linewidth=0.7, alpha=0.82, zorder=4))
        ax.text((x0 + x1) / 2, base_y + 3.3, rf"$\Delta_{i}$", ha="center", fontsize=6.5, color=color, fontweight="bold")
        cumulative += delta
        ax.scatter([base_x + cumulative * scale], [base_y], s=24, color=color, edgecolor="white", linewidth=0.5, zorder=6)
    ax.text(base_x + cumulative * scale, base_y - 2.2, "$r$", ha="center", fontsize=6.8, color=PALETTE["red"], fontweight="bold")
    rounded_box(ax, 60, 40, 32, 8.0, r"$r=\Delta_1+\Delta_2+\Delta_3\;(\mathrm{mod}\;N)$", PALETTE["cyan_light"], PALETTE["cyan"], fs=7.0, bold=True)
    save_all(fig, "nonring_view_2_stacked_number_line")


def draw_chip(ax, x, y, w, h, label, color, sublabel=None):
    rounded_box(ax, x, y, w, h, label, "white", color, fs=7.2, bold=True, lw=1.0)
    ax.add_patch(patches.Rectangle((x + 1.2, y + 0.8), w - 2.4, 0.7, facecolor=color, edgecolor="none", alpha=0.8, zorder=5))
    if sublabel:
        ax.text(x + w / 2, y - 1.2, sublabel, ha="center", va="top", fontsize=5.8, color=PALETTE["muted"])


def figure_3_coin_denomination():
    fig, ax = new_canvas(
        "3. Offset Denomination Chips",
        "Each layer contributes one denomination-like offset chip; the tray sums the selected chips.",
    )
    chips = [
        (10, 38, 17, 8, r"$\Delta_1$", LAYER_COLORS[0], "small offset"),
        (10, 26, 22, 8, r"$\Delta_2$", LAYER_COLORS[1], "medium offset"),
        (10, 14, 27, 8, r"$\Delta_3$", LAYER_COLORS[2], "large offset"),
    ]
    for i, chip in enumerate(chips):
        draw_chip(ax, *chip)
        if i < 2:
            ax.text(43, chip[1] - 3.5, "+", ha="center", va="center", fontsize=13, color=PALETTE["ink"])

    tray_x, tray_y = 49, 20
    ax.add_patch(
        patches.FancyBboxPatch(
            (tray_x, tray_y),
            32,
            24,
            boxstyle="round,pad=0.25,rounding_size=2.2",
            facecolor=PALETTE["grey_light"],
            edgecolor=PALETTE["grid"],
            linewidth=1.0,
            zorder=1,
        )
    )
    ax.text(tray_x + 16, tray_y + 21, "sum tray", ha="center", fontsize=7.2, color=PALETTE["ink"], fontweight="bold")
    for j, (_, _, w, h, label, color, _) in enumerate(chips):
        draw_chip(ax, tray_x + 4 + j * 8.6, tray_y + 9, 7.2, 6.2, label, color)
    ax.text(tray_x + 16, tray_y + 6, r"$\Delta_1+\Delta_2+\Delta_3$", ha="center", fontsize=7.0, color=PALETTE["ink"])
    arrow(ax, (82, tray_y + 12), (92, tray_y + 12), color=PALETTE["ink"], lw=1.1)
    rounded_box(ax, 91.5, tray_y + 8, 7.5, 8, "$r$", PALETTE["red_light"], PALETTE["red"], fs=9.0, bold=True)
    ax.text(50, 9.2, "This is intuitive, but less formal; useful as an explanatory inset.", ha="center", fontsize=6.7, color=PALETTE["muted"])
    save_all(fig, "nonring_view_3_offset_denomination_chips")


def figure_4_vector_addition():
    fig, ax = new_canvas(
        "4. Head-to-Tail Vector Addition",
        "Conceptual displacement vectors show additive composition without implying a geometric algorithm.",
    )
    points = [(15, 18), (36, 18), (53, 34), (74, 27)]
    labels = [r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$"]
    for i, (p0, p1, color, label) in enumerate(zip(points[:-1], points[1:], LAYER_COLORS, labels)):
        arrow(ax, p0, p1, color=color, lw=2.6)
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 3.2, label, ha="center", fontsize=7.2, color=color, fontweight="bold")
        ax.scatter([p0[0]], [p0[1]], s=24, color=PALETTE["ink"], edgecolor="white", linewidth=0.5, zorder=7)
    ax.scatter([points[-1][0]], [points[-1][1]], s=72, marker="*", color=PALETTE["red"], edgecolor="white", linewidth=0.7, zorder=8)
    arrow(ax, points[0], points[-1], color=PALETTE["red"], lw=1.3, dashed=True)
    ax.text(45, 11.5, "resultant displacement", ha="center", fontsize=6.8, color=PALETTE["red"])
    rounded_box(ax, 13, 45, 74, 7.6, r"$r=\Delta_1+\Delta_2+\Delta_3\;(\mathrm{mod}\;N)$", PALETTE["cyan_light"], PALETTE["cyan"], fs=7.4, bold=True)
    ax.text(50, 5.5, "Use the geometry only as a metaphor: each arrow is an offset contribution.", ha="center", fontsize=6.7, color=PALETTE["muted"])
    save_all(fig, "nonring_view_4_vector_addition")


def figure_5_algebra_tiles():
    fig, ax = new_canvas(
        "5. Pipeline-Less Algebra Tiles",
        "The selected layer offsets are tiles in one algebraic expression, not boxes in a workflow.",
    )
    tile_y = 33
    tile_w, tile_h = 15, 9
    xs = [12, 34, 56]
    for i, (x, color) in enumerate(zip(xs, LAYER_COLORS), start=1):
        rounded_box(ax, x, tile_y, tile_w, tile_h, rf"$\Delta_{i}$", "white", color, fs=9.0, bold=True)
        ax.text(x + tile_w / 2, tile_y - 3.1, rf"selected from $S_{i}$", ha="center", fontsize=5.9, color=PALETTE["muted"])
        for k in range(4):
            cx = x + 2.2 + k * 3.3
            ax.add_patch(patches.Rectangle((cx, tile_y - 8.5), 2.4, 2.0, facecolor=color, edgecolor="white", alpha=0.35, zorder=1))
        if i < 3:
            ax.text(x + tile_w + 4.0, tile_y + tile_h / 2, "+", ha="center", va="center", fontsize=14, color=PALETTE["ink"])
    ax.text(76, tile_y + tile_h / 2, "=", ha="center", va="center", fontsize=13, color=PALETTE["ink"])
    rounded_box(ax, 81, tile_y, 13.5, tile_h, "$r$", PALETTE["red_light"], PALETTE["red"], fs=9.0, bold=True)
    rounded_box(ax, 20, 14.5, 60, 7.0, r"$[\Delta_1]+[\Delta_2]+[\Delta_3]=[r]\quad(\mathrm{mod}\;N)$", PALETTE["grey_light"], PALETTE["grid"], fs=7.3, bold=True)
    ax.text(50, 53, "Candidate sets remain local; only the chosen tiles enter the sum.", ha="center", fontsize=6.9, color=PALETTE["muted"])
    save_all(fig, "nonring_view_5_algebra_tiles")


def figure_6_hierarchical_decomposition_tree():
    fig, ax = new_canvas(
        "6. Reverse Additive Decomposition Tree",
        "Start from the target and peel off one layer contribution at a time.",
    )
    nodes = {
        "r": (50, 50),
        "d3": (31, 38),
        "q2": (62, 38),
        "d2": (48, 26),
        "q1": (75, 26),
        "d1": (64, 14),
        "zero": (86, 14),
    }
    labels = {
        "r": "$r$",
        "d3": r"$\Delta_3$",
        "q2": r"$q_2=r-\Delta_3$",
        "d2": r"$\Delta_2$",
        "q1": r"$q_1=q_2-\Delta_2$",
        "d1": r"$\Delta_1$",
        "zero": "$0$",
    }
    colors = {
        "r": PALETTE["red"],
        "d3": LAYER_COLORS[2],
        "q2": PALETTE["grid"],
        "d2": LAYER_COLORS[1],
        "q1": PALETTE["grid"],
        "d1": LAYER_COLORS[0],
        "zero": PALETTE["grid"],
    }
    edges = [("r", "d3"), ("r", "q2"), ("q2", "d2"), ("q2", "q1"), ("q1", "d1"), ("q1", "zero")]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1] - 2.4, nodes[b][1] + 2.4], color=PALETTE["grey_mid"], linewidth=0.9, zorder=1)
    for key, (x, y) in nodes.items():
        fc = PALETTE["red_light"] if key == "r" else "white"
        ec = colors[key] if colors[key] != PALETTE["grid"] else PALETTE["grid"]
        rounded_box(ax, x - 8, y - 3, 16, 6, labels[key], fc, ec, fs=6.5 if key.startswith("q") else 7.2, bold=key in {"r", "d1", "d2", "d3"})
    ax.text(50, 6.0, r"Equivalent forward identity: $r=\Delta_1+\Delta_2+\Delta_3\;(\mathrm{mod}\;N)$", ha="center", fontsize=7.2, color=PALETTE["ink"], fontweight="bold")
    ax.text(50, 57.2, "This view explains search/decomposition without drawing an execution pipeline.", ha="center", fontsize=6.8, color=PALETTE["muted"])
    save_all(fig, "nonring_view_6_decomposition_tree")


def figure_7_ledger_accounting():
    fig, ax = new_canvas(
        "7. Ledger / Accounting View",
        "A table gives the most precise explanation of layer-local offsets and cumulative sums.",
    )
    x0, y0 = 9, 14
    widths = [18, 27, 36]
    row_h = 7.3
    headers = ["layer", "selected offset", "cumulative sum"]
    rows = [
        ("$L_1$", r"$\Delta_1$", r"$s_1=\Delta_1$"),
        ("$L_2$", r"$\Delta_2$", r"$s_2=s_1+\Delta_2$"),
        ("$L_3$", r"$\Delta_3$", r"$r=s_3=s_2+\Delta_3$"),
    ]
    table_w = sum(widths)
    ax.add_patch(patches.Rectangle((x0, y0), table_w, row_h * 4, facecolor="white", edgecolor=PALETTE["ink"], linewidth=0.9, zorder=1))
    x = x0
    for width, header in zip(widths, headers):
        ax.add_patch(patches.Rectangle((x, y0 + row_h * 3), width, row_h, facecolor=PALETTE["grey_light"], edgecolor=PALETTE["grid"], linewidth=0.7, zorder=2))
        ax.text(x + width / 2, y0 + row_h * 3.5, header, ha="center", va="center", fontsize=6.8, color=PALETTE["ink"], fontweight="bold")
        x += width
    for r_idx, row in enumerate(rows):
        y = y0 + row_h * (2 - r_idx)
        x = x0
        color = LAYER_COLORS[r_idx]
        for c_idx, (width, text) in enumerate(zip(widths, row)):
            fc = "white" if c_idx != 1 else [PALETTE["blue_light"], PALETTE["cyan_light"], PALETTE["green_light"]][r_idx]
            ax.add_patch(patches.Rectangle((x, y), width, row_h, facecolor=fc, edgecolor=PALETTE["grid"], linewidth=0.7, zorder=2))
            ax.text(x + width / 2, y + row_h / 2, text, ha="center", va="center", fontsize=6.8, color=color if c_idx == 1 else PALETTE["ink"], fontweight="bold" if c_idx == 1 else "normal")
            x += width
    rounded_box(ax, 70, 36, 23, 8.0, r"invariant:" + "\n" + r"$s_i=s_{i-1}+\Delta_i$", PALETTE["cyan_light"], PALETTE["cyan"], fs=6.8, bold=True)
    rounded_box(ax, 70, 22, 23, 8.0, r"final:" + "\n" + r"$r=s_3\;(\mathrm{mod}\;N)$", PALETTE["red_light"], PALETTE["red"], fs=6.8, bold=True)
    ax.text(44, 7.2, "Best for method sections: compact, exact, and hard to misread as a flowchart.", ha="center", fontsize=6.8, color=PALETTE["muted"])
    save_all(fig, "nonring_view_7_ledger_accounting")


def main() -> None:
    setup_style()
    figure_1_unwrapped_modular_ruler()
    figure_2_stacked_number_line()
    figure_3_coin_denomination()
    figure_4_vector_addition()
    figure_5_algebra_tiles()
    figure_6_hierarchical_decomposition_tree()
    figure_7_ledger_accounting()


if __name__ == "__main__":
    main()
