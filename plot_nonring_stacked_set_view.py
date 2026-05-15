"""
Enhanced stacked-number-line view for multi-layer additive decomposition.

This figure expands the original nonring_view_2 idea:
- each layer is a set of offsets with the same style but different lengths;
- one example chooses one offset from each set and concatenates them like an
  unwrapped modular ruler;
- the Minkowski-sum style coverage explains how all choices cover diagonal
  rotation demand.
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


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.2,
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


def xmap(value, x0=0, x1=100, n=100):
    return x0 + (x1 - x0) * value / n


def arrow(ax, start, end, color=PALETTE["ink"], lw=1.1, style="-|>", dashed=False, z=5):
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=lw,
            color=color,
            linestyle=(0, (2, 2)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def rounded_box(ax, x, y, w, h, text, fc, ec, fs=7.0, bold=False, lw=0.9, align="center"):
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
        x + (w / 2 if align == "center" else 1.2),
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fs,
        color=PALETTE["ink"],
        fontweight="bold" if bold else "normal",
        zorder=4,
    )


def draw_ruler_axis(ax, y, x0, x1, n=100, label_top=False):
    ax.plot([x0, x1], [y, y], color=PALETTE["ink"], linewidth=0.8, zorder=1)
    for value in range(0, n + 1, 10):
        x = xmap(value, x0, x1, n)
        tick_h = 0.85 if value % 25 else 1.35
        ax.plot([x, x], [y - tick_h / 2, y + tick_h / 2], color=PALETTE["grid"], linewidth=0.65, zorder=2)
    label_y = y + 1.65 if label_top else y - 1.65
    va = "bottom" if label_top else "top"
    for value, label in [(0, "$0$"), (50, "$N/2$"), (100, "$N$")]:
        ax.text(xmap(value, x0, x1, n), label_y, label, ha="center", va=va, fontsize=5.9, color=PALETTE["muted"])


def draw_offset_set(ax, x0, y0, w, h, title, set_label, lengths, color, n=60, selected_idx=None):
    fill = {
        PALETTE["blue"]: PALETTE["blue_light"],
        PALETTE["cyan"]: PALETTE["cyan_light"],
        PALETTE["green"]: PALETTE["green_light"],
        PALETTE["orange"]: PALETTE["orange_light"],
    }.get(color, PALETTE["grey_light"])
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.20,rounding_size=1.2",
            facecolor=fill,
            edgecolor=color,
            linewidth=0.85,
            alpha=0.35,
            zorder=0,
        )
    )
    ax.text(x0 + 1.1, y0 + h - 1.5, title, ha="left", va="top", fontsize=7.2, color=PALETTE["ink"], fontweight="bold")
    ax.text(x0 + w - 1.0, y0 + h - 1.5, set_label, ha="right", va="top", fontsize=7.0, color=color, fontweight="bold")

    axis_x0, axis_x1 = x0 + 6.5, x0 + w - 5.0
    y_start = y0 + h - 5.1
    gap = 2.25
    for idx, length in enumerate(lengths):
        y = y_start - idx * gap
        ax.plot([axis_x0, axis_x1], [y, y], color=PALETTE["grid"], linewidth=0.55, zorder=1)
        x_len = xmap(length, axis_x0, axis_x1, n)
        lw = 2.9 if idx == selected_idx else 1.9
        alpha = 0.98 if idx == selected_idx else 0.66
        arrow(ax, (axis_x0, y), (x_len, y), color=color, lw=lw, z=4)
        ax.scatter([x_len], [y], s=26 if idx == selected_idx else 14, color=color, edgecolor="white", linewidth=0.45, alpha=alpha, zorder=5)
        ax.text(axis_x1 + 1.0, y, rf"${length}$", ha="left", va="center", fontsize=5.8, color=PALETTE["muted"])
    ax.text(axis_x0, y0 + 1.05, "same style, different lengths", ha="left", va="bottom", fontsize=5.7, color=PALETTE["muted"])


def draw_example_ruler(ax, x0, y, width, deltas, colors, n=100):
    draw_ruler_axis(ax, y, x0, x0 + width, n=n)
    current = 0
    path_y = y + 5.1
    ax.text(xmap(0, x0, x0 + width, n), path_y + 3.1, "$0$", ha="center", fontsize=6.1, color=PALETTE["muted"])
    for idx, (delta, color) in enumerate(zip(deltas, colors), start=1):
        start = current
        end = min(n, current + delta)
        arrow(ax, (xmap(start, x0, x0 + width, n), path_y), (xmap(end, x0, x0 + width, n), path_y), color=color, lw=3.0, z=5)
        ax.text(xmap((start + end) / 2, x0, x0 + width, n), path_y + 2.65, rf"$\Delta_{idx}$", ha="center", fontsize=6.6, color=color, fontweight="bold")
        current = (current + delta) % n
        ax.scatter([xmap(current, x0, x0 + width, n)], [path_y], s=25, color=color, edgecolor="white", linewidth=0.45, zorder=6)
    ax.scatter([xmap(current, x0, x0 + width, n)], [path_y], s=78, marker="*", color=PALETTE["red"], edgecolor="white", linewidth=0.6, zorder=8)
    ax.text(xmap(current, x0, x0 + width, n), path_y - 3.2, "$r$", ha="center", va="top", fontsize=7.0, color=PALETTE["red"], fontweight="bold")
    return current


def draw_combination_lattice(ax, x0, y0, w, h, short_set, mid_set, long_set):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.20,rounding_size=1.2",
            facecolor="white",
            edgecolor=PALETTE["grid"],
            linewidth=0.8,
            zorder=0,
        )
    )
    ax.text(x0 + 1.2, y0 + h - 1.25, "coverage by all combinations", ha="left", va="top", fontsize=7.2, color=PALETTE["ink"], fontweight="bold")

    values = sorted({(a + b + c) % 100 for a in short_set for b in mid_set for c in long_set})
    demand_idx = np.linspace(3, len(values) - 4, 14, dtype=int)
    demand = [values[i] for i in demand_idx]

    line_y = y0 + 3.2
    draw_ruler_axis(ax, line_y, x0 + 5.5, x0 + w - 5.5, n=100)
    for value in values:
        xx = xmap(value, x0 + 5.5, x0 + w - 5.5, 100)
        ax.add_patch(patches.Rectangle((xx - 0.16, line_y + 0.9), 0.32, 1.75, facecolor=PALETTE["grey_mid"], edgecolor="none", alpha=0.22, zorder=2))
    for d in demand:
        xx = xmap(d, x0 + 5.5, x0 + w - 5.5, 100)
        ax.scatter([xx], [line_y + 2.95], s=18, marker="o", color=PALETTE["green"], edgecolor="white", linewidth=0.4, zorder=5)

    ax.text(x0 + 5.5, y0 + h - 3.2, r"$S_{\mathrm{sum}}=(S_s\oplus S_m\oplus S_l)\;(\mathrm{mod}\;N)$", ha="left", va="center", fontsize=6.8, color=PALETTE["ink"], fontweight="bold")
    ax.text(x0 + 52.0, y0 + h - 3.2, r"diagonal demand coverage:  $\mathcal{D}\subseteq S_{\mathrm{sum}}$", ha="left", va="center", fontsize=6.3, color=PALETTE["muted"])
    ax.text(x0 + 5.5, y0 + h - 5.35, r"for each demanded $d$, choose $(a,b,c)$ such that $d\equiv a+b+c\;(\mathrm{mod}\;N)$", ha="left", va="center", fontsize=5.9, color=PALETTE["muted"])

    legend_y = y0 + 1.05
    ax.scatter([x0 + 6.0], [legend_y], s=18, color=PALETTE["green"], edgecolor="white", linewidth=0.4)
    ax.text(x0 + 8.0, legend_y, "covered diag demand", ha="left", va="center", fontsize=5.9, color=PALETTE["muted"])
    ax.add_patch(patches.Rectangle((x0 + 28.0, legend_y - 0.55), 1.1, 1.1, facecolor=PALETTE["grey_mid"], edgecolor="none", alpha=0.25))
    ax.text(x0 + 30.0, legend_y, "reachable combination", ha="left", va="center", fontsize=5.9, color=PALETTE["muted"])


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(exist_ok=True)

    short_set = [7, 10, 13, 16]
    mid_set = [21, 27, 33, 39]
    long_set = [44, 52, 60, 68]
    selected = [13, 33, 52]
    colors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"]]

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.035, 0.045, 0.93, 0.86])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    fig.text(0.5, 0.965, "Stacked Number-Line Sets for Additive Diagonal Generation", ha="center", va="top", fontsize=13.0, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.5, 0.925, "Each layer is a same-style offset set; selecting one offset from each set generates a diagonal rotation by linear combination.", ha="center", va="top", fontsize=8.0, color=PALETTE["muted"])

    draw_offset_set(ax, 4, 40.5, 28.0, 16.5, "short offsets", r"$S_s$", short_set, PALETTE["blue"], n=70, selected_idx=2)
    draw_offset_set(ax, 36, 40.5, 28.0, 16.5, "medium offsets", r"$S_m$", mid_set, PALETTE["cyan"], n=70, selected_idx=2)
    draw_offset_set(ax, 68, 40.5, 28.0, 16.5, "long offsets", r"$S_l$", long_set, PALETTE["green"], n=70, selected_idx=1)

    for x in [33.8, 65.8]:
        ax.text(x, 49, r"$+$", ha="center", va="center", fontsize=16, color=PALETTE["ink"])

    rounded_box(
        ax,
        9.5,
        33.0,
        81.0,
        4.4,
        r"Layer choices form a modular sum set:  $S_{\mathrm{sum}}=\{(a+b+c)\;(\mathrm{mod}\;N)\mid a\in S_s,\;b\in S_m,\;c\in S_l\}$",
        PALETTE["grey_light"],
        PALETTE["grid"],
        fs=7.0,
        bold=True,
    )

    ax.text(5, 27.3, "example selection and composition", ha="left", fontsize=8.2, color=PALETTE["ink"], fontweight="bold")
    ax.text(5, 25.2, r"choose one member from each set:  $13\in S_s,\;33\in S_m,\;52\in S_l$", ha="left", fontsize=6.7, color=PALETTE["muted"])
    r_value = draw_example_ruler(ax, 7, 16.0, 52, selected, colors, n=100)
    rounded_box(
        ax,
        63,
        16.8,
        30.0,
        7.5,
        rf"$r=(13+33+52)\;(\mathrm{{mod}}\;N)={r_value}$" + "\n" + r"one reachable diagonal index",
        PALETTE["red_light"],
        PALETTE["red"],
        fs=7.0,
        bold=True,
    )

    draw_combination_lattice(ax, 5.5, 2.5, 89.0, 10.5, short_set, mid_set, long_set)

    out_stem = "nonring_view_2_stacked_number_line_sets"
    paths = [
        OUT_DIR / f"{out_stem}.svg",
        OUT_DIR / f"{out_stem}.pdf",
        OUT_DIR / f"{out_stem}.png",
    ]
    for path in paths:
        fig.savefig(path, facecolor=PALETTE["bg"], edgecolor="none", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    for path in paths:
        print(f"[OK] saved {path}")


if __name__ == "__main__":
    main()
