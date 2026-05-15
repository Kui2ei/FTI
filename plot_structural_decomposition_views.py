"""
Seven structural, non-ring intuition views for multi-layer decomposition.

These figures explain what multi-layer decomposition buys structurally:
coarse-to-fine localization, reusable dictionaries, key coverage, factorized
storage, sparse coding, multiscale tiling, and cache-like reuse.
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

LAYER_COLORS = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["orange"]]
LAYER_FILLS = [PALETTE["blue_light"], PALETTE["cyan_light"], PALETTE["green_light"], PALETTE["orange_light"]]


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


def new_canvas(title: str, subtitle: str, width=7.0, height=4.25):
    fig = plt.figure(figsize=(width, height), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.04, 0.055, 0.92, 0.80])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12.0, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.5, 0.912, subtitle, ha="center", va="top", fontsize=7.2, color=PALETTE["muted"])
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


def rounded_box(ax, x, y, w, h, text, fc, ec, fs=7.0, bold=False, lw=0.9, align="center"):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=1.0",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=3,
        )
    )
    tx = x + w / 2 if align == "center" else x + 1.2
    ax.text(
        tx,
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fs,
        color=PALETTE["ink"],
        fontweight="bold" if bold else "normal",
        zorder=4,
    )


def arrow(ax, start, end, color=PALETTE["ink"], lw=1.0, style="-|>", dashed=False, z=5):
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


def xmap(value, x0=8, x1=92, n=100):
    return x0 + (x1 - x0) * value / n


def draw_index_strip(ax, y, x0=8, x1=92, n=100, label=None):
    ax.plot([x0, x1], [y, y], color=PALETTE["ink"], linewidth=0.85, zorder=1)
    for value in range(0, n + 1, 10):
        x = xmap(value, x0, x1, n)
        tick = 1.3 if value % 20 == 0 else 0.8
        ax.plot([x, x], [y - tick / 2, y + tick / 2], color=PALETTE["grid"], linewidth=0.65, zorder=2)
    for value, text in [(0, "0"), (50, "N/2"), (100, "N")]:
        ax.text(xmap(value, x0, x1, n), y - 2.0, text, ha="center", va="top", fontsize=6.0, color=PALETTE["muted"])
    if label:
        ax.text(x0, y + 2.4, label, ha="left", va="bottom", fontsize=6.8, color=PALETTE["ink"], fontweight="bold")


def draw_heat_strip(ax, x0, y, w, h, values, cmap_colors=(PALETTE["blue"], PALETTE["cyan"])):
    cell_w = w / len(values)
    for i, val in enumerate(values):
        color = cmap_colors[1] if val > 0.62 else cmap_colors[0]
        ax.add_patch(
            patches.Rectangle(
                (x0 + i * cell_w, y),
                cell_w - 0.12,
                h,
                facecolor=color,
                edgecolor="none",
                alpha=0.18 + 0.65 * val,
                zorder=2,
            )
        )
    ax.add_patch(patches.Rectangle((x0, y), w, h, facecolor="none", edgecolor=PALETTE["grid"], linewidth=0.6, zorder=3))


def figure_1_coarse_to_fine_localization():
    fig, ax = new_canvas(
        "1. Multi-Scale Map / Coarse-to-Fine Localization",
        "Multi-layer decomposition can be read as locating a rotation index at successive scales.",
    )
    draw_index_strip(ax, 47, label="full index space")
    sectors = [(8, 47, 21, 4), (29, 47, 21, 4), (50, 47, 21, 4), (71, 47, 21, 4)]
    for i, (x, y, w, h) in enumerate(sectors):
        fc = PALETTE["blue_light"] if i == 2 else "white"
        ec = PALETTE["blue"] if i == 2 else PALETTE["grid"]
        ax.add_patch(patches.Rectangle((x, y - 2), w, h, facecolor=fc, edgecolor=ec, linewidth=0.85, alpha=0.9, zorder=3))
        ax.text(x + w / 2, y + 3.6, f"sector {i}", ha="center", fontsize=5.8, color=PALETTE["muted"])

    zooms = [
        (16, 25, 68, 8, PALETTE["blue"], "Level 1: coarse sector"),
        (29, 16, 42, 7, PALETTE["cyan"], "Level 2: sub-sector"),
        (41, 8, 18, 6, PALETTE["green"], "Level 3: exact index"),
    ]
    selected_windows = [(50, 47), (43, 25), (48, 16)]
    for i, (x, y, w, h, color, label) in enumerate(zooms):
        rounded_box(ax, x, y, w, h, "", "white", color, lw=0.9)
        ax.text(x + 1.2, y + h - 1.2, label, ha="left", va="top", fontsize=6.7, color=color, fontweight="bold")
        count = 8 if i == 0 else 6 if i == 1 else 4
        for j in range(count):
            cx = x + 5 + j * (w - 10) / (count - 1)
            ax.plot([cx, cx], [y + 1.4, y + h - 3.0], color=PALETTE["grid"], linewidth=0.55)
        target_x = x + w * ([0.57, 0.62, 0.70][i])
        ax.add_patch(patches.Rectangle((target_x - w * 0.06, y + 1.25), w * 0.12, h - 4.0, facecolor=LAYER_FILLS[i], edgecolor=color, linewidth=0.8, zorder=5))
        if i == 2:
            ax.scatter([target_x], [y + 2.5], s=76, marker="*", color=PALETTE["red"], edgecolor="white", linewidth=0.6, zorder=7)
            ax.text(target_x + 3, y + 2.6, "$r$", ha="left", va="center", fontsize=7.4, color=PALETTE["red"], fontweight="bold")

    for (sx, sy), (tx, ty) in zip(selected_windows, [(50, 33), (50, 23), (50, 14)]):
        arrow(ax, (sx, sy - 4), (tx, ty), color=PALETTE["grey_mid"], lw=0.8, dashed=True)
    ax.text(50, 3.0, "The intuition is hierarchical localization: large scale first, then progressively finer choices.", ha="center", fontsize=6.8, color=PALETTE["muted"])
    save_all(fig, "structural_view_1_coarse_to_fine_localization")


def figure_2_dictionary_codebook():
    fig, ax = new_canvas(
        "2. Dictionary / Codebook View",
        "A small set of reusable layer entries can encode many demanded rotation indices.",
    )
    ax.text(8, 50, "demand indices", ha="left", fontsize=7.2, fontweight="bold", color=PALETTE["ink"])
    demand = [8, 15, 24, 33, 41, 52, 61, 74, 86]
    for i, value in enumerate(demand):
        x = 10 + (i % 3) * 10
        y = 43 - (i // 3) * 7
        rounded_box(ax, x, y, 7.2, 4.2, rf"$r_{{{i+1}}}$", "white", PALETTE["grid"], fs=6.8, bold=True)
        ax.text(x + 3.6, y - 1.2, str(value), ha="center", va="top", fontsize=5.8, color=PALETTE["muted"])

    codebooks = [
        ("Layer 1 codebook", r"$\mathcal{C}_1$", [0, 4, 8, 12], PALETTE["blue"], PALETTE["blue_light"]),
        ("Layer 2 codebook", r"$\mathcal{C}_2$", [0, 16, 32, 48], PALETTE["cyan"], PALETTE["cyan_light"]),
        ("Layer 3 codebook", r"$\mathcal{C}_3$", [0, 64, 128], PALETTE["green"], PALETTE["green_light"]),
    ]
    for row, (title, symbol, entries, color, fill) in enumerate(codebooks):
        y = 48 - row * 13
        rounded_box(ax, 48, y, 39, 8.2, "", fill, color, lw=0.9)
        ax.text(50, y + 6.5, title, ha="left", va="center", fontsize=6.8, color=PALETTE["ink"], fontweight="bold")
        ax.text(84.5, y + 6.5, symbol, ha="right", va="center", fontsize=7.2, color=color, fontweight="bold")
        for j, entry in enumerate(entries):
            ex = 51 + j * 8.2
            rounded_box(ax, ex, y + 1.3, 6.2, 3.4, str(entry), "white", color, fs=6.0)

    arrow(ax, (38, 34), (46, 34), color=PALETTE["grey_mid"], lw=0.9, dashed=True)
    rounded_box(
        ax,
        18,
        9,
        64,
        8.0,
        r"one demand index  $\approx$  one entry from each codebook; codebooks are reused across many demands",
        PALETTE["grey_light"],
        PALETTE["grid"],
        fs=6.8,
    )
    save_all(fig, "structural_view_2_dictionary_codebook")


def figure_3_coverage_map():
    fig, ax = new_canvas(
        "3. Coverage Map",
        "Decomposition changes the coverage mask over demanded indices: hits are reused, misses require new keys.",
    )
    values = [0.1, 0.4, 0.8, 0.2, 0.7, 0.15, 0.9, 0.55, 0.25, 0.8, 0.35, 0.12, 0.62, 0.42, 0.78, 0.3]
    x0, w = 12, 76
    ax.text(x0, 51, "demand heat", ha="left", fontsize=7.0, fontweight="bold", color=PALETTE["ink"])
    draw_heat_strip(ax, x0, 47, w, 4.5, values)

    rows = [
        ("existing bsKey", [1, 2, 6, 9, 12], PALETTE["blue"], PALETTE["blue_light"]),
        ("layer reuse", [0, 2, 4, 6, 7, 10, 14], PALETTE["green"], PALETTE["green_light"]),
        ("candidate new", [5, 11, 15], PALETTE["red"], PALETTE["red_light"]),
    ]
    cell_w = w / len(values)
    covered = set()
    for r, (label, idxs, color, fill) in enumerate(rows):
        y = 37 - r * 9
        ax.text(x0 - 1.2, y + 2.2, label, ha="right", va="center", fontsize=6.6, color=PALETTE["muted"])
        ax.add_patch(patches.Rectangle((x0, y), w, 4.4, facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.6))
        for idx in idxs:
            covered.add(idx)
            ax.add_patch(patches.Rectangle((x0 + idx * cell_w, y), cell_w - 0.15, 4.4, facecolor=fill, edgecolor=color, linewidth=0.7))

    ax.text(x0 - 1.2, 8.5, "aggregate", ha="right", va="center", fontsize=6.6, color=PALETTE["ink"], fontweight="bold")
    ax.add_patch(patches.Rectangle((x0, 6.4), w, 4.4, facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.6))
    for i, val in enumerate(values):
        if val < 0.33:
            color, fill = PALETTE["grid"], PALETTE["grey_light"]
        elif i in covered:
            color, fill = PALETTE["green"], PALETTE["green_light"]
        else:
            color, fill = PALETTE["red"], PALETTE["red_light"]
        ax.add_patch(patches.Rectangle((x0 + i * cell_w, 6.4), cell_w - 0.15, 4.4, facecolor=fill, edgecolor=color, linewidth=0.65))
    ax.text(50, 2.7, "Good co-design aligns demand peaks with existing or reusable coverage.", ha="center", fontsize=6.8, color=PALETTE["muted"])
    save_all(fig, "structural_view_3_coverage_map")


def figure_4_factorized_tables():
    fig, ax = new_canvas(
        "4. Compression / Factorization View",
        "A dense per-rotation key table is replaced by several smaller layer tables.",
    )
    ax.text(10, 51, "dense key table", ha="left", fontsize=7.2, fontweight="bold", color=PALETTE["ink"])
    gx, gy, cell = 10, 23, 3.1
    rng = np.random.default_rng(4)
    for r in range(8):
        for c in range(12):
            active = rng.random() > 0.22
            fc = PALETTE["blue_light"] if active else "white"
            ec = PALETTE["blue"] if active else PALETTE["grid"]
            ax.add_patch(patches.Rectangle((gx + c * cell, gy + r * cell), cell - 0.15, cell - 0.15, facecolor=fc, edgecolor=ec, linewidth=0.45, alpha=0.9))
    ax.text(gx + 18, gy - 3.0, r"many independent $K_r$", ha="center", fontsize=6.4, color=PALETTE["muted"])

    arrow(ax, (50, 35), (58, 35), color=PALETTE["grey_mid"], lw=1.0)
    ax.text(54, 38, "factorize", ha="center", fontsize=6.6, color=PALETTE["muted"])

    table_specs = [
        (63, 43, 24, 5.0, r"$\mathcal{T}_1$ coarse", PALETTE["blue"], PALETTE["blue_light"], 6),
        (63, 31, 24, 5.0, r"$\mathcal{T}_2$ medium", PALETTE["cyan"], PALETTE["cyan_light"], 5),
        (63, 19, 24, 5.0, r"$\mathcal{T}_3$ fine", PALETTE["green"], PALETTE["green_light"], 4),
    ]
    for x, y, w, h, label, color, fill, count in table_specs:
        rounded_box(ax, x, y, w, h, "", fill, color, lw=0.85)
        ax.text(x + 1.2, y + h / 2, label, ha="left", va="center", fontsize=6.7, color=PALETTE["ink"], fontweight="bold")
        for i in range(count):
            ax.add_patch(patches.Circle((x + w - 2.8 - i * 2.7, y + h / 2), 0.7, facecolor=color, edgecolor="white", linewidth=0.35, zorder=5))
    rounded_box(ax, 60, 7, 30, 6.6, r"storage intuition: $|\mathcal{T}_1|+|\mathcal{T}_2|+|\mathcal{T}_3| \ll |\{K_r\}|$", PALETTE["grey_light"], PALETTE["grid"], fs=6.4)
    save_all(fig, "structural_view_4_factorized_tables")


def figure_5_sparse_coding():
    fig, ax = new_canvas(
        "5. Sparse Coding View",
        "Each demanded rotation activates only a few reusable basis atoms from the layer dictionaries.",
    )
    ax.text(8, 51, "basis atoms", ha="left", fontsize=7.1, color=PALETTE["ink"], fontweight="bold")
    for row, color in enumerate(LAYER_COLORS):
        y = 44 - row * 9
        ax.text(8, y + 1.4, rf"$\mathcal{{B}}_{row+1}$", ha="left", va="center", fontsize=6.8, color=color, fontweight="bold")
        for i in range(7):
            x = 18 + i * 4.2
            height = 2.0 + ((i + row) % 3)
            ax.add_patch(patches.Rectangle((x, y), 2.6, height, facecolor=color, edgecolor="white", linewidth=0.45, alpha=0.35 + 0.08 * i))
        if row < 2:
            ax.text(32, y - 4.2, "+", ha="center", va="center", fontsize=12, color=PALETTE["ink"])

    ax.text(52, 51, "sparse activation matrix", ha="left", fontsize=7.1, color=PALETTE["ink"], fontweight="bold")
    mx, my, cw, ch = 52, 23, 4.2, 3.6
    active = {(0, 0), (2, 0), (5, 0), (1, 1), (3, 1), (6, 1), (0, 2), (4, 2), (7, 2), (2, 3), (5, 3)}
    for r in range(4):
        ax.text(mx - 2, my + r * ch + ch / 2, rf"$r_{r+1}$", ha="right", va="center", fontsize=6.2, color=PALETTE["muted"])
        for c in range(8):
            fc = "white"
            ec = PALETTE["grid"]
            if (c, r) in active:
                fc = LAYER_FILLS[c % 3]
                ec = LAYER_COLORS[c % 3]
            ax.add_patch(patches.Rectangle((mx + c * cw, my + r * ch), cw - 0.25, ch - 0.25, facecolor=fc, edgecolor=ec, linewidth=0.55))
    ax.text(mx + 16, my - 3.0, "few nonzeros per demand", ha="center", fontsize=6.4, color=PALETTE["muted"])

    arrow(ax, (45, 33), (50, 33), color=PALETTE["grey_mid"], lw=0.8, dashed=True)
    arrow(ax, (87, 33), (92, 33), color=PALETTE["grey_mid"], lw=0.8, dashed=True)
    ax.text(89.5, 41, "reconstruct\nrotation set", ha="center", fontsize=6.4, color=PALETTE["muted"])
    for i, val in enumerate([0.2, 0.8, 0.35, 0.6, 0.12, 0.9, 0.55]):
        ax.add_patch(patches.Rectangle((92, 19 + i * 3.7), 4.5 * val, 1.5, facecolor=PALETTE["cyan"], edgecolor="none", alpha=0.45 + 0.45 * val))
    save_all(fig, "structural_view_5_sparse_coding")


def figure_6_multiscale_tiling():
    fig, ax = new_canvas(
        "6. Tiling / Mosaic View",
        "Large tiles cover broad regular structure; fine tiles patch local hot demand regions.",
    )
    x0, y0, w = 10, 47, 80
    values = [0.10, 0.16, 0.75, 0.82, 0.72, 0.20, 0.18, 0.64, 0.88, 0.77, 0.22, 0.18, 0.45, 0.73, 0.85, 0.38]
    ax.text(x0, y0 + 6, "demand mosaic", ha="left", fontsize=7.0, color=PALETTE["ink"], fontweight="bold")
    draw_heat_strip(ax, x0, y0, w, 5.0, values, cmap_colors=(PALETTE["orange"], PALETTE["red"]))

    tile_rows = [
        ("coarse tiles", 37, [(10, 23), (33, 23), (56, 23)], PALETTE["blue"], PALETTE["blue_light"]),
        ("medium tiles", 27, [(21, 15), (47, 16), (70, 13)], PALETTE["cyan"], PALETTE["cyan_light"]),
        ("fine patches", 17, [(23, 7), (43, 7), (77, 7)], PALETTE["green"], PALETTE["green_light"]),
    ]
    for label, y, tiles, color, fill in tile_rows:
        ax.text(x0 - 1.0, y + 2.5, label, ha="right", va="center", fontsize=6.5, color=PALETTE["muted"])
        ax.plot([x0, x0 + w], [y + 2.5, y + 2.5], color=PALETTE["grid"], linewidth=0.6)
        for start, tw in tiles:
            ax.add_patch(patches.FancyBboxPatch((start, y), tw, 5.0, boxstyle="round,pad=0.12,rounding_size=0.6", facecolor=fill, edgecolor=color, linewidth=0.8, alpha=0.88))
    rounded_box(ax, 23, 6.5, 54, 6.5, "multi-layer keys act like multi-scale tiles over structured demand", PALETTE["grey_light"], PALETTE["grid"], fs=6.8)
    save_all(fig, "structural_view_6_multiscale_tiling")


def figure_7_cache_hierarchy():
    fig, ax = new_canvas(
        "7. Cache Hierarchy View",
        "Co-design makes frequent rotation demand hit warm reusable key sets before paying for new keys.",
    )
    ax.text(8, 51, "demand stream", ha="left", fontsize=7.1, color=PALETTE["ink"], fontweight="bold")
    stream = [8, 16, 32, 9, 64, 17, 33, 8, 96, 32, 18, 64]
    for i, val in enumerate(stream):
        x = 9 + i * 6.2
        rounded_box(ax, x, 44, 4.7, 4.0, str(val), "white", PALETTE["grid"], fs=5.9)
    arrow(ax, (9, 41.5), (80, 41.5), color=PALETTE["grey_mid"], lw=0.8, dashed=True)

    caches = [
        (18, 30, 65, 6.2, "warm existing key pool", r"$\mathcal{K}_{bs}$", PALETTE["blue"], PALETTE["blue_light"], [8, 16, 32, 64]),
        (22, 20, 57, 6.2, "decomposition reuse layer", r"$\mathcal{K}_{reuse}$", PALETTE["green"], PALETTE["green_light"], [9, 17, 33]),
        (30, 10, 41, 6.2, "new-key fill on miss", r"$\mathcal{K}_{new}$", PALETTE["red"], PALETTE["red_light"], [18, 96]),
    ]
    for x, y, w, h, title, symbol, color, fill, keys in caches:
        rounded_box(ax, x, y, w, h, "", fill, color, lw=0.9)
        ax.text(x + 1.3, y + h / 2, title, ha="left", va="center", fontsize=6.8, color=PALETTE["ink"], fontweight="bold")
        ax.text(x + w - 1.2, y + h / 2, symbol, ha="right", va="center", fontsize=7.2, color=color, fontweight="bold")
        for i, key in enumerate(keys):
            ax.add_patch(patches.Circle((x + 31 + i * 4.1, y + h / 2), 0.95, facecolor=color, edgecolor="white", linewidth=0.35, zorder=5))
            ax.text(x + 31 + i * 4.1, y + h / 2 - 2.1, str(key), ha="center", va="top", fontsize=5.2, color=PALETTE["muted"])

    for sx, sy, tx, ty, color, label in [
        (24, 44, 38, 36.2, PALETTE["blue"], "hit"),
        (45, 44, 50, 26.2, PALETTE["green"], "reuse"),
        (64, 44, 55, 16.2, PALETTE["red"], "fill"),
    ]:
        arrow(ax, (sx, sy), (tx, ty), color=color, lw=0.9, dashed=True)
        ax.text((sx + tx) / 2, (sy + ty) / 2 + 1.0, label, ha="center", fontsize=6.0, color=color, fontweight="bold")
    ax.text(50, 4.5, "The hierarchy view explains why reuse and new-key cost should be co-designed with demand.", ha="center", fontsize=6.8, color=PALETTE["muted"])
    save_all(fig, "structural_view_7_cache_hierarchy")


def main() -> None:
    setup_style()
    figure_1_coarse_to_fine_localization()
    figure_2_dictionary_codebook()
    figure_3_coverage_map()
    figure_4_factorized_tables()
    figure_5_sparse_coding()
    figure_6_multiscale_tiling()
    figure_7_cache_hierarchy()


if __name__ == "__main__":
    main()
