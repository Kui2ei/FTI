from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT)
CJK_FONT_NAME = font_manager.FontProperties(fname=CJK_FONT).get_name()
mpl.rcParams["font.family"] = CJK_FONT_NAME
mpl.rcParams["font.sans-serif"] = [CJK_FONT_NAME, "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


PALETTE = {
    "ink": "#23313d",
    "muted": "#65727f",
    "grid": "#d8dee6",
    "paper": "#f7f8fa",
    "blue": "#2f6fb0",
    "blue_light": "#d9e9f8",
    "green": "#208b6d",
    "green_light": "#d9f0e8",
    "orange": "#c46a1a",
    "orange_light": "#f6e3cf",
    "red": "#b84545",
    "red_light": "#f3dada",
    "purple": "#6a5aa8",
    "purple_light": "#e4e0f4",
    "gold": "#b68a1c",
    "gold_light": "#f3e8c6",
}


def add_round_box(ax, x, y, w, h, fc, ec, lw=1.0, radius=0.028, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, s, size=10, color=None, weight="regular", ha="left", va="center", **kwargs):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color or PALETTE["ink"],
        fontweight=weight,
        ha=ha,
        va=va,
        **kwargs,
    )


def draw_axis(ax, x0, x1, y, label, ticks, color=PALETTE["muted"]):
    ax.plot([x0, x1], [y, y], color=color, lw=1.3)
    for t, txt in ticks:
        x = x0 + (x1 - x0) * t
        ax.plot([x, x], [y - 0.008, y + 0.008], color=color, lw=1.1)
        add_text(ax, x, y - 0.026, txt, size=7.5, color=color, ha="center")
    add_text(ax, x0, y + 0.03, label, size=8.5, color=color)


def draw_layer_row(ax, y, name, scale, digit_range, chosen_digit, color, x0=0.31, x1=0.71):
    add_text(ax, x0 - 0.115, y, name, size=10.2, weight="bold", color=color, ha="left")
    add_text(ax, x0 - 0.024, y, f"× {scale}", size=9.2, color=PALETTE["muted"], ha="right")
    ax.plot([x0, x1], [y, y], color=PALETTE["grid"], lw=5.5, solid_capstyle="round", zorder=0)

    dmin, dmax = digit_range
    span = dmax - dmin
    for d in range(dmin, dmax + 1):
        x = x0 + (x1 - x0) * (d - dmin) / span
        lw = 1.1 if d != 0 else 1.7
        ax.plot([x, x], [y - 0.017, y + 0.017], color=color, lw=lw, zorder=2)
        if d in (dmin, 0, dmax):
            add_text(ax, x, y - 0.037, str(d), size=7.2, color=PALETTE["muted"], ha="center")

    chosen_x = x0 + (x1 - x0) * (chosen_digit - dmin) / span
    ax.scatter([chosen_x], [y], s=230, color=color, edgecolors="white", linewidths=1.8, zorder=4)
    add_text(ax, chosen_x, y + 0.042, f"{chosen_digit:+d}", size=10, color=color, weight="bold", ha="center")
    contribution = chosen_digit * scale
    add_text(
        ax,
        x1 + 0.016,
        y,
        f"{chosen_digit:+d} × {scale} = {contribution:+d}",
        size=9.1,
        color=PALETTE["ink"],
        ha="left",
    )


def draw_layer_row_worked(
    ax,
    y,
    name,
    scale,
    digit_range,
    chosen_digit,
    color,
    cumulative,
    x0=0.255,
    x1=0.69,
):
    add_text(ax, 0.07, y + 0.014, name, size=10.8, weight="bold", color=color)
    add_text(ax, 0.07, y - 0.016, f"scale = {scale}", size=8.6, color=PALETTE["muted"])

    dmin, dmax = digit_range
    span = dmax - dmin
    ax.plot([x0, x1], [y, y], color=PALETTE["grid"], lw=6.0, solid_capstyle="round", zorder=0)
    for d in range(dmin, dmax + 1):
        x = x0 + (x1 - x0) * (d - dmin) / span
        lw = 1.1
        height = 0.019
        if d == 0:
            lw = 1.8
            height = 0.025
        ax.plot([x, x], [y - height, y + height], color=color, lw=lw, zorder=2)
        if d in (dmin, 0, dmax):
            add_text(ax, x, y - 0.044, str(d), size=7.5, color=PALETTE["muted"], ha="center")

    chosen_x = x0 + (x1 - x0) * (chosen_digit - dmin) / span
    ax.scatter([chosen_x], [y], s=250, color=color, edgecolors="white", linewidths=1.8, zorder=4)
    add_text(ax, chosen_x, y + 0.045, f"选 {chosen_digit:+d}", size=9.5, color=color, weight="bold", ha="center")

    contribution = chosen_digit * scale
    add_text(ax, 0.725, y + 0.015, f"{chosen_digit:+d} × {scale} = {contribution:+d}", size=9.4)
    add_text(ax, 0.725, y - 0.017, f"累计 {cumulative}", size=8.8, color=PALETTE["muted"])


def draw_matrix_texture(ax, x, y, w, h):
    add_round_box(ax, x, y, w, h, "#ffffff", PALETTE["grid"], lw=1.2, radius=0.018)
    n = 11
    for i in range(n):
        for j in range(n):
            cx = x + 0.018 + (w - 0.036) * j / (n - 1)
            cy = y + 0.018 + (h - 0.036) * i / (n - 1)
            alpha = 0.22 + 0.45 * (((i * 7 + j * 11) % 9) / 8)
            if (i - j) % 3 == 0 or (i + 2 * j) % 5 == 0:
                ax.scatter([cx], [cy], s=10, color=PALETTE["blue"], alpha=alpha, linewidths=0)
    for offset, c in [(-3, PALETTE["green"]), (0, PALETTE["orange"]), (4, PALETTE["purple"])]:
        xs = []
        ys = []
        for i in range(n):
            j = i + offset
            if 0 <= j < n:
                xs.append(x + 0.018 + (w - 0.036) * j / (n - 1))
                ys.append(y + 0.018 + (h - 0.036) * i / (n - 1))
        ax.plot(xs, ys, color=c, lw=2.0, alpha=0.92)


def main():
    fig, ax = plt.subplots(figsize=(15.8, 9.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfcfd")

    add_text(ax, 0.045, 0.952, "FFT-like 分层如何覆盖全局 rotation indices", size=22, weight="bold")
    add_text(
        ax,
        0.047,
        0.914,
        "例子：slots = 2^16，levelBudget = 5。把一个全局 diagonal rotation 写成 5 个小 offset 的和。",
        size=11.5,
        color=PALETTE["muted"],
    )

    r = 31245

    # Top context: diagonal method and the global rotation ruler.
    draw_matrix_texture(ax, 0.055, 0.72, 0.17, 0.14)
    add_text(ax, 0.245, 0.836, "任意明文矩阵按 diagonal method 计算：", size=10.2, color=PALETTE["muted"])
    add_text(ax, 0.245, 0.800, "A·x = Σ diag_r(A) · Rot(x, r)", size=14.2, weight="bold", color=PALETTE["blue"])
    add_text(ax, 0.245, 0.764, "dense matrix 可能需要 0...65535 中任意一个 r。", size=10.2, color=PALETTE["muted"])

    draw_axis(
        ax,
        0.50,
        0.945,
        0.800,
        "global rotation ruler",
        [(0, "0"), (0.25, "16384"), (0.5, "32768"), (0.75, "49152"), (1, "65535")],
    )
    rx = 0.50 + (0.945 - 0.50) * r / 65535
    ax.scatter([rx], [0.800], s=110, color=PALETTE["red"], zorder=5)
    add_text(ax, rx, 0.854, "目标 r = 31245", size=10, color=PALETTE["red"], weight="bold", ha="center")

    # Main worked example: one target rotation as a layered mixed-radix sum.
    add_round_box(ax, 0.045, 0.225, 0.91, 0.445, "#ffffff", PALETTE["grid"], lw=1.2)
    add_text(ax, 0.07, 0.637, "同一个 r，在 FFT-like 中不是单独造一把 key，而是拆成 5 层小 offset", size=13.2, weight="bold")
    add_text(
        ax,
        0.07,
        0.603,
        "C2S 源码中的 scalingFactor：8192, 1024, 128, 16, 1。每层只需要覆盖一个小 digit 范围。",
        size=10,
        color=PALETTE["muted"],
    )

    layers = [
        ("L4 coarse", 8192, (-7, 7), 4, PALETTE["blue"]),
        ("L3", 1024, (-7, 7), -1, PALETTE["green"]),
        ("L2", 128, (-7, 7), -4, PALETTE["orange"]),
        ("L1", 16, (-7, 7), 1, PALETTE["purple"]),
        ("L0 fine", 1, (-15, 15), -3, PALETTE["red"]),
    ]
    cumulative = 0
    ys = [0.535, 0.472, 0.409, 0.346, 0.283]
    for y, layer in zip(ys, layers):
        cumulative += layer[3] * layer[1]
        draw_layer_row_worked(ax, y, *layer, cumulative=cumulative)

    add_round_box(ax, 0.777, 0.298, 0.145, 0.125, PALETTE["green_light"], "#9acdbc", lw=1.1, radius=0.018)
    add_text(ax, 0.795, 0.394, "31245 精确覆盖", size=12.5, weight="bold", color=PALETTE["green"])
    add_text(ax, 0.795, 0.358, "4·8192 - 1·1024\n- 4·128 + 1·16 - 3·1", size=9.2, color=PALETTE["ink"], linespacing=1.35)
    add_text(ax, 0.795, 0.317, "= 31245", size=11.2, weight="bold", color=PALETTE["red"])

    # Coverage intuition.
    add_round_box(ax, 0.045, 0.065, 0.91, 0.145, "#eef4fa", "#bfd2e4", lw=1.0, radius=0.018)
    add_text(ax, 0.07, 0.175, "为什么能覆盖全局 rot indices", size=12.5, weight="bold", color=PALETTE["blue"])
    add_text(
        ax,
        0.07,
        0.138,
        "粗层先把 r 拉到接近的位置，后面的层只修正剩余误差。相邻尺度之间的间隔由下一层的 digit 范围覆盖，"
        "最后一层 scale=1 直接覆盖整数余数，所以不会留下只能靠额外大 rotation key 才能补的空洞。",
        size=10.1,
        color=PALETTE["ink"],
    )
    add_text(
        ax,
        0.07,
        0.101,
        "结论：FFT-like 的分层 rot-key basis 可以覆盖和 bsslot 同大小的明文矩阵-密文向量乘法所需的全局 diagonal rotations；"
        "但每个 diagonal 的 plaintext coefficient mask 仍要由原矩阵决定。",
        size=9.4,
        color=PALETTE["muted"],
    )

    png = RESULTS / "fftlike_rot_coverage.png"
    svg = RESULTS / "fftlike_rot_coverage.svg"
    pdf = RESULTS / "fftlike_rot_coverage.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(png)
    print(svg)
    print(pdf)


if __name__ == "__main__":
    main()
