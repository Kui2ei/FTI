#!/usr/bin/env python3
"""Simplified explanation for BSGS coverage on collapsed block layers 3-5."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "result2" / "bsgs-layer3-5-cover-mod32-simple.png"

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

PAPER = (247, 250, 252)
WHITE = (255, 255, 255)
INK = (25, 31, 43)
MUTED = (84, 96, 111)
GRID = (210, 222, 234)
BLUE = (0, 55, 235)
BLUE_DARK = (0, 36, 175)
BLUE_SOFT = (222, 232, 255)
ORANGE = (246, 92, 18)
ORANGE_SOFT = (255, 235, 220)
GREEN = (0, 130, 42)
GREEN_SOFT = (220, 244, 226)
GRAY = (118, 126, 137)
GRAY_SOFT = (239, 242, 245)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


F = {
    "title": font(FONT_BOLD, 32),
    "subtitle": font(FONT_REG, 18),
    "h": font(FONT_BOLD, 24),
    "body": font(FONT_REG, 17),
    "body_bold": font(FONT_BOLD, 17),
    "small": font(FONT_REG, 14),
    "small_bold": font(FONT_BOLD, 14),
    "tiny": font(FONT_REG, 12),
    "mono": font(FONT_MONO, 18),
    "mono_big": font(FONT_MONO, 23),
}


def reduce_rotation(index: int, slots: int) -> int:
    n = slots.bit_length() - 1
    if index >= 0:
        return index - ((index >> n) << n)
    return index + slots + ((abs(index) >> n) << n)


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
    size = 9
    draw.polygon(
        [
            (x1, y1),
            (x1 - ux * size + px * size * 0.55, y1 - uy * size + py * size * 0.55),
            (x1 - ux * size - px * size * 0.55, y1 - uy * size - py * size * 0.55),
        ],
        fill=fill,
    )


def draw_header(draw: ImageDraw.ImageDraw):
    text(draw, (54, 48), "block layers 3-5：BSGS 后为什么仍覆盖 {0,4,8,12,16,20,24,28}", BLUE_DARK, F["title"])
    text(draw, (56, 82), "只看一个事实：BSGS 的 giant step 是 8·4 = 32，在 mod32 下等于 0。", MUTED, F["subtitle"])

    x = 56
    y = 112
    for label, fill, outline, color in [
        ("collapse: layers=3", BLUE_SOFT, BLUE, BLUE_DARK),
        ("q=-7..7", GRAY_SOFT, GRID, INK),
        ("sf=4", BLUE_SOFT, BLUE, BLUE_DARK),
        ("g=8, b=2", ORANGE_SOFT, ORANGE, ORANGE),
        ("q = q_b + 8i", GREEN_SOFT, (157, 211, 169), GREEN),
    ]:
        x = pill(draw, x, y, label, fill, outline, color)


def draw_collapse_panel(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 54, 170, 1546, 430
    rounded(draw, (x0, y0, x1, y1))
    text(draw, (x0 + 24, y0 + 38), "1. collapse 后：15 条 signed shift，但 mod32 只有 8 个中心", INK, F["h"])
    text(draw, (x0 + 24, y0 + 68), "signed offset = 4q，q 从 -7 到 7；正负两边在循环意义下重合。", MUTED, F["body"])

    axis_x0, axis_x1 = x0 + 100, x1 - 70
    axis_y = y0 + 165
    draw.line((axis_x0, axis_y, axis_x1, axis_y), fill=(58, 70, 84), width=2)
    zero_x = xmap(0, -28, 28, axis_x0, axis_x1)
    draw.line((zero_x, axis_y - 46, zero_x, axis_y + 70), fill=GRID, width=2)

    text(draw, (axis_x0 - 55, axis_y - 32), "q", MUTED, F["small_bold"])
    text(draw, (axis_x0 - 55, axis_y + 20), "4q", MUTED, F["small_bold"])
    text(draw, (axis_x0 - 55, axis_y + 69), "mod32", MUTED, F["small_bold"])

    for q in range(-7, 8):
        off = q * 4
        residue = reduce_rotation(off, 32)
        xx = xmap(off, -28, 28, axis_x0, axis_x1)
        color = BLUE if q <= 0 else ORANGE
        draw.line((xx, axis_y - 8, xx, axis_y + 8), fill=(58, 70, 84), width=1)
        draw.ellipse((xx - 8, axis_y - 8, xx + 8, axis_y + 8), fill=color, outline=WHITE, width=2)
        text(draw, (xx, axis_y - 28), str(q), color, F["small_bold"], anchor="mm")
        text(draw, (xx, axis_y + 29), str(off), MUTED, F["small"], anchor="mm")
        text(draw, (xx, axis_y + 73), str(residue), GREEN, F["small_bold"], anchor="mm")

    band_y = axis_y - 48
    draw.line((axis_x0, band_y, zero_x, band_y), fill=BLUE, width=4)
    draw.line((xmap(4, -28, 28, axis_x0, axis_x1), band_y, axis_x1, band_y), fill=ORANGE, width=4)
    text(draw, ((axis_x0 + zero_x) / 2, band_y - 18), "baby q_b=-7..0", BLUE, F["small_bold"], anchor="mm")
    text(draw, ((xmap(4, -28, 28, axis_x0, axis_x1) + axis_x1) / 2, band_y - 18), "giant row q=1..7", ORANGE, F["small_bold"], anchor="mm")


def draw_bsgs_panel(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 54, 456, 1546, 718
    rounded(draw, (x0, y0, x1, y1))
    text(draw, (x0 + 24, y0 + 38), "2. BSGS 后：2 行 × 8 列；第二行只是第一行整体 +32", INK, F["h"])
    text(draw, (x0 + 24, y0 + 68), "列由 baby q_b 决定；行由 giant i 决定。第 16 个格子 q=8 是 32→0 的重复中心。", MUTED, F["body"])

    table_x = x0 + 130
    table_y = y0 + 105
    cell_w = 154
    cell_h = 58
    qbs = list(range(-7, 1))

    text(draw, (table_x - 26, table_y + 20), "i=0", BLUE, F["small_bold"], anchor="rm")
    text(draw, (table_x - 26, table_y + cell_h + 20), "i=1", ORANGE, F["small_bold"], anchor="rm")

    for c, qb in enumerate(qbs):
        cx = table_x + c * cell_w
        text(draw, (cx + cell_w / 2, table_y - 16), f"q_b={qb}", BLUE_DARK, F["tiny"], anchor="mm")

        off0 = qb * 4
        res0 = reduce_rotation(off0, 32)
        draw.rectangle((cx, table_y, cx + cell_w, table_y + cell_h), fill=BLUE_SOFT, outline=GRID, width=1)
        text(draw, (cx + cell_w / 2, table_y + 24), f"{off0} → {res0}", GREEN, F["small_bold"], anchor="mm")
        text(draw, (cx + cell_w / 2, table_y + 45), f"q={qb}", MUTED, F["tiny"], anchor="mm")

        q1 = qb + 8
        off1 = q1 * 4
        res1 = reduce_rotation(off1, 32)
        draw.rectangle((cx, table_y + cell_h, cx + cell_w, table_y + 2 * cell_h), fill=ORANGE_SOFT, outline=GRID, width=1)
        label = "32 → 0" if q1 == 8 else f"{off1} → {res1}"
        sub = "repeat" if q1 == 8 else f"q={q1}"
        color = GRAY if q1 == 8 else GREEN
        text(draw, (cx + cell_w / 2, table_y + cell_h + 24), label, color, F["small_bold"], anchor="mm")
        text(draw, (cx + cell_w / 2, table_y + cell_h + 45), sub, MUTED, F["tiny"], anchor="mm")

    arrow(draw, (table_x + cell_w * 2.5, table_y + cell_h + 44), (table_x + cell_w * 2.5, table_y + 48), ORANGE, width=2)
    arrow(draw, (table_x + cell_w * 5.5, table_y + cell_h + 44), (table_x + cell_w * 5.5, table_y + 48), ORANGE, width=2)
    text(draw, (table_x + cell_w * 4, y1 - 28), "每个 i=1 的格子 = 同列 i=0 的格子 + 32，因此 mod32 residue 不变。", ORANGE, F["body_bold"], anchor="mm")


def draw_conclusion(draw: ImageDraw.ImageDraw):
    x0, y0, x1, y1 = 54, 746, 1546, 852
    rounded(draw, (x0, y0, x1, y1), fill=GREEN_SOFT, outline=(151, 209, 165), width=2)
    formula = "Reduce((q_b+8i)·4, 32) = Reduce(q_b·4 + i·32, 32) = Reduce(q_b·4, 32)"
    text(draw, (x0 + 35, y0 + 42), formula, GREEN, F["mono"])
    text(draw, (x0 + 35, y0 + 78), "所以 BSGS 不改变 block layer 3-5 的 mod32 覆盖集合：", INK, F["body_bold"])
    text(draw, (x0 + 606, y0 + 79), "{0,4,8,12,16,20,24,28}", BLUE_DARK, F["mono_big"])


def main():
    img = Image.new("RGB", (1600, 890), PAPER)
    draw = ImageDraw.Draw(img)
    draw_header(draw)
    draw_collapse_panel(draw)
    draw_bsgs_panel(draw)
    draw_conclusion(draw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
