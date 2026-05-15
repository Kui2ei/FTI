#!/usr/bin/env python3
"""Manual G=16 BSGS coverage picture for collapsed block layers 3-5."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "result2" / "bsgs-layer3-5-manual-g16.png"
OUT_BSGS_ONLY = ROOT / "result2" / "3-bsgs.png"

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

PAPER = (247, 250, 252)
WHITE = (255, 255, 255)
INK = (27, 34, 47)
MUTED = (85, 97, 112)
GRID = (206, 219, 232)
BLUE = (0, 55, 235)
BLUE_DARK = (0, 38, 180)
BLUE_SOFT = (223, 233, 255)
ORANGE = (245, 92, 18)
ORANGE_SOFT = (255, 235, 220)
GREEN = (0, 132, 43)
GREEN_SOFT = (220, 244, 226)
GRAY = (115, 124, 136)
GRAY_SOFT = (240, 243, 246)

W, H = 1700, 1155
MOD = 32
TARGET = [-12, -8, -4, 0, 4, 8, 12, 16]
BABY = [-12, -8, -4, 0]
GIANT = [0, 16]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


F = {
    "title": font(FONT_BOLD, 34),
    "subtitle": font(FONT_REG, 18),
    "h": font(FONT_BOLD, 25),
    "body": font(FONT_REG, 18),
    "body_bold": font(FONT_BOLD, 18),
    "small": font(FONT_REG, 15),
    "small_bold": font(FONT_BOLD, 15),
    "tiny": font(FONT_REG, 13),
    "mono": font(FONT_MONO, 18),
    "mono_big": font(FONT_MONO, 24),
}


def signed32(value: int) -> int:
    residue = value % MOD
    return residue - MOD if residue > MOD // 2 else residue


def mod_label(value: int) -> str:
    signed = signed32(value)
    return str(signed)


def rounded(draw: ImageDraw.ImageDraw, box, radius=10, fill=WHITE, outline=GRID, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, s: str, fill=INK, ft=None, anchor=None):
    draw.text(xy, s, fill=fill, font=ft or F["body"], anchor=anchor)


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, fill, outline, color):
    w = draw.textbbox((0, 0), label, font=F["small_bold"])[2] + 30
    rounded(draw, (x, y, x + w, y + 34), radius=8, fill=fill, outline=outline, width=2)
    text(draw, (x + w / 2, y + 17), label, color, F["small_bold"], anchor="mm")
    return x + w + 14


def xmap(value: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (value - lo) * (x1 - x0) / (hi - lo)


def arrow(draw: ImageDraw.ImageDraw, p0, p1, fill=GRAY, width=2):
    draw.line((*p0, *p1), fill=fill, width=width)
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 10
    draw.polygon(
        [
            (x1, y1),
            (x1 - ux * size + px * size * 0.55, y1 - uy * size + py * size * 0.55),
            (x1 - ux * size - px * size * 0.55, y1 - uy * size - py * size * 0.55),
        ],
        fill=fill,
    )


def draw_bsgs_legend(draw: ImageDraw.ImageDraw, x1: int, y0: int):
    legend_right = x1 - 12
    legend_top = y0 + 8
    legend_left = legend_right - 474
    legend_bottom = legend_top + 88
    rounded(draw, (legend_left, legend_top, legend_right, legend_bottom), radius=8, fill=WHITE, outline=GRID, width=1)

    rows = [
        (legend_top + 24, BLUE, "dot", "baby stencil {-12,-8,-4,0}"),
        (legend_top + 50, ORANGE, "arrow", "同一 stencil +16 -> {4,8,12,16}"),
        (legend_top + 76, GREEN, "dot", "两行并集 = target"),
    ]
    for yy, color, marker, label in rows:
        marker_x = legend_left + 28
        if marker == "arrow":
            arrow(draw, (marker_x - 8, yy), (marker_x + 18, yy), color, width=2)
        else:
            draw.ellipse((marker_x - 7, yy - 7, marker_x + 7, yy + 7), fill=color, outline=WHITE, width=2)
        text(draw, (legend_left + 58, yy), label, color, F["small_bold"], anchor="lm")


def draw_axis(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    values: list[int],
    color,
    label: str,
    lo: int = -16,
    hi: int = 16,
    label_color=None,
    show_tick_labels: bool = True,
    show_value_labels: bool = True,
):
    label_color = label_color or color
    draw.line((x0, y, x1, y), fill=(53, 65, 80), width=2)
    for tick in range(lo, hi + 1, 4):
        xx = xmap(tick, lo, hi, x0, x1)
        draw.line((xx, y - 9, xx, y + 9), fill=(53, 65, 80), width=1)
        if show_tick_labels:
            text(draw, (xx, y + 33), str(tick), MUTED, F["small"], anchor="mm")
    text(draw, (x0 - 18, y - 4), label, label_color, F["body_bold"], anchor="rm")
    for value in values:
        xx = xmap(value, lo, hi, x0, x1)
        draw.ellipse((xx - 9, y - 9, xx + 9, y + 9), fill=color, outline=WHITE, width=2)
        if show_value_labels:
            text(draw, (xx, y - 27), str(value), color, F["small_bold"], anchor="mm")


def draw_header(draw: ImageDraw.ImageDraw):
    text(draw, (54, 48), "block layers 3-5：手选 Gstep=16 的 BSGS 覆盖解释", BLUE_DARK, F["title"])
    text(
        draw,
        (56, 82),
        "不用 RotInOPENFHE.py 的默认 BSGS；这里只对 collapse 后的 8 个 distinct 中心做 baby + giant 分解。",
        MUTED,
        F["subtitle"],
    )
    x = 56
    y = 114
    for label, fill, outline, color in [
        ("collapse block: layers 3-5", BLUE_SOFT, BLUE, BLUE_DARK),
        ("target signed = {-12,-8,-4,0,4,8,12,16}", GRAY_SOFT, GRID, INK),
        ("modulus = 32", BLUE_SOFT, BLUE, BLUE_DARK),
        ("baby = {-12,-8,-4,0}", BLUE_SOFT, BLUE, BLUE_DARK),
        ("giant = {0,16}", ORANGE_SOFT, ORANGE, ORANGE),
    ]:
        x = pill(draw, x, y, label, fill, outline, color)


def draw_collapse_panel(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 54, 170, 1646, 378
    rounded(draw, (x0, y0, x1, y1))
    text(draw, (x0 + 24, y0 + 40), "1. collapse 后，layer 3-5 只剩 8 个循环对角线中心", INK, F["h"])
    text(
        draw,
        (x0 + 24, y0 + 72),
        "原来的 15 个 signed shift 在 mod32 下合并；这里把 residue 20,24,28 写成 signed 的 -12,-8,-4。",
        MUTED,
        F["body"],
    )
    axis_x0, axis_x1, axis_y = x0 + 170, x1 - 115, y0 + 138
    draw_axis(draw, axis_x0, axis_x1, axis_y, TARGET, GREEN, "target")
    text(draw, (axis_x0, y1 - 22), "unsigned residue: {0,4,8,12,16,20,24,28}", MUTED, F["small_bold"])
    text(draw, (axis_x0 + 550, y1 - 22), "signed view: {-12,-8,-4,0,4,8,12,16}", GREEN, F["small_bold"])


def draw_bsgs_panel(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 54, 405, 1646, 750
    rounded(draw, (x0, y0, x1, y1))
    text(draw, (x0 + 24, y0 + 40), "2. BSGS 不是删点，而是把每个目标写成 b + g (mod 32)", INK, F["h"])
    text(draw, (x0 + 24, y0 + 72), "选择 Gstep=16 后，只需要两行：g=0 保留 baby stencil，g=16 把同一 stencil 平移到右半边。", MUTED, F["body"])

    axis_x0, axis_x1 = x0 + 190, x1 - 170
    y_baby = y0 + 130
    y_giant = y0 + 210
    y_cover = y0 + 285
    giant_row = [signed32(v + 16) for v in BABY]
    cover = sorted(set(BABY + giant_row))

    draw_axis(draw, axis_x0, axis_x1, y_baby, BABY, BLUE, "g=0", show_tick_labels=False)
    draw_axis(
        draw,
        axis_x0,
        axis_x1,
        y_giant,
        giant_row,
        ORANGE,
        "g=16",
        show_tick_labels=False,
        show_value_labels=False,
    )
    draw_axis(draw, axis_x0, axis_x1, y_cover, cover, GREEN, "union")

    for b, shifted in zip(BABY, giant_row):
        bx = xmap(b, -16, 16, axis_x0, axis_x1)
        sx = xmap(shifted, -16, 16, axis_x0, axis_x1)
        arrow(draw, (bx, y_baby + 18), (sx, y_giant - 18), ORANGE, width=2)
    draw_bsgs_legend(draw, x1, y0)


def draw_bsgs_only_panel(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 20, 20, 1572, 375
    rounded(draw, (x0, y0, x1, y1))

    axis_x0, axis_x1 = x0 + 190, x1 - 170
    y_baby = y0 + 138
    y_giant = y0 + 218
    y_cover = y0 + 293
    giant_row = [signed32(v + 16) for v in BABY]
    cover = sorted(set(BABY + giant_row))

    draw_axis(draw, axis_x0, axis_x1, y_baby, BABY, BLUE, "g=0", show_tick_labels=False)
    draw_axis(
        draw,
        axis_x0,
        axis_x1,
        y_giant,
        giant_row,
        ORANGE,
        "g=16",
        show_tick_labels=False,
        show_value_labels=False,
    )
    draw_axis(draw, axis_x0, axis_x1, y_cover, cover, GREEN, "union")

    for b, shifted in zip(BABY, giant_row):
        bx = xmap(b, -16, 16, axis_x0, axis_x1)
        sx = xmap(shifted, -16, 16, axis_x0, axis_x1)
        arrow(draw, (bx, y_baby + 18), (sx, y_giant - 18), ORANGE, width=2)
    draw_bsgs_legend(draw, x1, y0)


def draw_table_panel(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 54, 778, 1075, 1095
    rounded(draw, (x0, y0, x1, y1))
    text(draw, (x0 + 24, y0 + 40), "3. 每个目标都有一个显式表示", INK, F["h"])
    text(draw, (x0 + 24, y0 + 70), "列是 baby，行是 giant；格子里写的是 signed32(b+g)。", MUTED, F["body"])

    table_x = x0 + 140
    table_y = y0 + 128
    cell_w = 190
    cell_h = 58
    for c, b in enumerate(BABY):
        cx = table_x + c * cell_w
        text(draw, (cx + cell_w / 2, table_y - 18), f"b={b}", BLUE_DARK, F["small_bold"], anchor="mm")
    for r, g in enumerate(GIANT):
        ry = table_y + r * cell_h
        text(draw, (table_x - 28, ry + cell_h / 2), f"g={g}", ORANGE if g else BLUE, F["small_bold"], anchor="rm")
        for c, b in enumerate(BABY):
            cx = table_x + c * cell_w
            result = signed32(b + g)
            fill = BLUE_SOFT if g == 0 else ORANGE_SOFT
            draw.rectangle((cx, ry, cx + cell_w, ry + cell_h), fill=fill, outline=GRID, width=1)
            text(draw, (cx + cell_w / 2, ry + 24), f"{b} + {g} = {result}", GREEN, F["small_bold"], anchor="mm")
            text(draw, (cx + cell_w / 2, ry + 46), f"mod32 residue {(b + g) % MOD}", MUTED, F["tiny"], anchor="mm")

    text(draw, (x0 + 32, y1 - 55), "所以：", INK, F["body_bold"])
    text(draw, (x0 + 95, y1 - 55), "({-12,-8,-4,0} + {0,16}) mod 32", GREEN, F["mono"])
    text(draw, (x0 + 95, y1 - 24), "= {-12,-8,-4,0,4,8,12,16}", BLUE_DARK, F["mono_big"])


def draw_key_panel(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 1105, 778, 1646, 1095
    rounded(draw, (x0, y0, x1, y1), fill=GREEN_SOFT, outline=(150, 210, 166), width=2)
    text(draw, (x0 + 26, y0 + 42), "实际 rotation key 可以更少", GREEN, F["h"])
    text(draw, (x0 + 26, y0 + 78), "0 是 identity，不需要生成 key。", INK, F["body_bold"])

    text(draw, (x0 + 26, y0 + 125), "baby keys:", BLUE_DARK, F["body_bold"])
    text(draw, (x0 + 190, y0 + 126), "{-12, -8, -4}", BLUE_DARK, F["mono_big"])

    text(draw, (x0 + 26, y0 + 175), "giant keys:", ORANGE, F["body_bold"])
    text(draw, (x0 + 190, y0 + 176), "{16}", ORANGE, F["mono_big"])

    text(draw, (x0 + 26, y0 + 222), "等价也可选 baby={0,4,8,12}, giant={0,16}；", MUTED, F["small"])
    text(draw, (x0 + 26, y0 + 244), "只是把 baby stencil 换到了右半边。", MUTED, F["small"])


def main():
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    draw_header(draw)
    draw_collapse_panel(draw)
    draw_bsgs_panel(draw)
    draw_table_panel(draw)
    draw_key_panel(draw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(OUT)

    bsgs_only = Image.new("RGB", (1592, 395), PAPER)
    draw_bsgs_only_panel(ImageDraw.Draw(bsgs_only))
    bsgs_only.save(OUT_BSGS_ONLY)
    print(OUT_BSGS_ONLY)


if __name__ == "__main__":
    main()
