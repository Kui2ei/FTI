#!/usr/bin/env python3
"""Redraw result2/2-collapse.png for the levelBudget=3 collapse case.

For n = 32, RotInOPENFHE.py gives logslots = 5. With levelBudget = 3,
selectLayers(5, 3) returns layersCollapse = 2 and remCollapse = 1, so the
collapsed factors are C1, C2-3, and C4-5.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw

from fix_cooleytukey_sparse_png import (
    AXIS,
    BLUE,
    BLUE_DARK,
    BLUE_SOFT,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    F,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    WHITE,
    draw_centered_segments,
    draw_collapsed_matrix,
    draw_dft_matrix_formula,
    rounded,
    sub,
)


ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "result2" / "2-collapse.png"
NEW_ORANGE = (255, 176, 70)


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, fill, outline, r: int = 6) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2)


def mod32_signed(values: set[int]) -> set[int]:
    out: set[int] = set()
    for value in values:
        residue = value % 32
        if residue > 16:
            residue -= 32
        out.add(residue)
    return out


def sums(scales: list[int]) -> set[int]:
    return {
        sum(coeff * scale for coeff, scale in zip(coeffs, scales))
        for coeffs in product((-1, 0, 1), repeat=len(scales))
    }


def clipped_window(values: set[int]) -> set[int]:
    return {value for value in values if -16 <= value <= 16}


def draw_top_panels(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, CANVAS_WIDTH, 444), fill=WHITE)
    draw.text(
        (836, 8),
        "n = 32 的 Cooley–Tukey sparse–diagonal 折叠示意（levelBudget = 3）",
        font=F["cjk26"],
        fill=BLUE_DARK,
        anchor="ma",
    )

    rounded(draw, (372, 60, 1306, 138), radius=8, fill=WHITE, outline=BLUE, width=2)
    draw.text(
        (839, 74),
        "DFT" + sub("32") + "ᴺᴿ = C" + sub("1") + " · C" + sub("2-3") + " · C" + sub("4-5"),
        font=F["serif26"],
        fill=INK,
        anchor="ma",
    )
    draw.text(
        (839, 112),
        "RotInOPENFHE: selectLayers(5,3) => layersCollapse = 2, remCollapse = 1，因此最后折叠为 3 个矩阵。",
        font=F["cjk16"],
        fill=BLUE,
        anchor="ma",
    )

    rounded(draw, (1465, 10, 1662, 118), radius=8, fill=WHITE, outline=INK, width=1)
    legend = [("主对角线 0", BLUE), ("正偏移 + offset", ORANGE), ("负偏移 - offset", GREEN)]
    for i, (label, color) in enumerate(legend):
        yy = 30 + 30 * i
        draw.rectangle((1484, yy - 10, 1504, yy + 10), fill=color, outline=color)
        draw.text((1522, yy - 12), label, font=F["cjk16"], fill=INK)

    size = 170
    matrix_y = 180
    dft_x = 66
    c1_x = 372
    c23_x = 617
    c45_x = 862
    operator_y = matrix_y + size / 2

    draw_dft_matrix_formula(draw, dft_x, matrix_y, size)
    draw.text(
        ((dft_x + size + c1_x) / 2, operator_y),
        "=",
        font=F["serif36"],
        fill=INK,
        anchor="mm",
        stroke_width=1,
        stroke_fill=INK,
    )
    draw_collapsed_matrix(
        draw,
        c1_x,
        matrix_y,
        size,
        size,
        "block: layer 1",
        [-1, 0, 1],
        "3 条 distinct 循环对角线",
        ["signed: {-1,0,1}", "mod32: {0,1,31}"],
    )
    draw.text(((c1_x + size + c23_x) / 2, operator_y), "×", font=F["serif36"], fill=INK, anchor="mm")
    draw_collapsed_matrix(
        draw,
        c23_x,
        matrix_y,
        size,
        size,
        "block: layers 2-3",
        [-6, -4, -2, 0, 2, 4, 6],
        "7 条 distinct 循环对角线",
        ["signed: {-6,-4,-2,0,2,4,6}", "mod32: {0,2,4,6,26,28,30}"],
    )
    draw.text(((c23_x + size + c45_x) / 2, operator_y), "×", font=F["serif36"], fill=INK, anchor="mm")
    draw_collapsed_matrix(
        draw,
        c45_x,
        matrix_y,
        size,
        size,
        "block: layers 4-5",
        [-24, -16, -8, 0, 8, 16, 24],
        "4 条 distinct 循环对角线",
        ["formal signed: {-24,-16,-8,0,8,16,24}", "mod32 distinct: {0,8,16,24}"],
    )

    draw.text((1218, 248), "折叠数从 2 块变成 3 块", font=F["cjk16"], fill=BLUE, anchor="ma")
    draw.text((1218, 278), "每块更稀疏，但多消耗一层 level", font=F["cjk16"], fill=MUTED, anchor="ma")


def draw_axis(
    draw: ImageDraw.ImageDraw,
    y: int,
    label: str,
    previous: set[int],
    current: set[int],
) -> None:
    left, right = 162, 970

    def xmap(value: int) -> float:
        return left + (value + 16) * (right - left) / 32

    rounded(draw, (22, y - 22, 136, y + 10), radius=5, fill=WHITE, outline=BLUE, width=2)
    draw.text((79, y - 14), label, font=F["cjk15"], fill=BLUE, anchor="ma")

    draw.line((left, y, right, y), fill=AXIS, width=1)
    for value in range(-16, 17):
        x = xmap(value)
        major = value % 4 == 0
        draw.line((x, y - (8 if major else 5), x, y + (8 if major else 5)), fill=AXIS, width=1)
        draw.text((x, y + 12), str(value), font=F["cjk13"], fill=INK, anchor="ma")

    for value in sorted(previous & current):
        draw_marker(draw, xmap(value), y - 13, BLUE_SOFT, BLUE, r=6)
    for value in sorted(current - previous):
        draw_marker(draw, xmap(value), y - 13, NEW_ORANGE, ORANGE, r=6)


def draw_lower_panels(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 440, CANVAS_WIDTH, CANVAS_HEIGHT), fill=WHITE)

    left_panel = (8, 445, 1010, 842)
    rounded(draw, left_panel, radius=8, fill=WHITE, outline=BLUE, width=2)
    draw.rounded_rectangle((8, 445, 640, 480), radius=8, fill=BLUE_DARK, outline=BLUE_DARK, width=1)
    draw.text((24, 452), "levelBudget = 3 时：三段残差逐步补齐 diagonal index", fill=WHITE, font=F["cjk19"])

    coarse = {-16, -8, 0, 8, 16}
    even_residue = {-6, -4, -2, 0, 2, 4, 6}
    fine_residue = {-1, 0, 1}
    even_covered = clipped_window({c + b for c in coarse for b in even_residue})
    all_covered = clipped_window({e + a for e in even_covered for a in fine_residue})

    y1, y2, y3 = 530, 635, 740
    draw_axis(draw, y1, "层 4-5", coarse, coarse)
    draw.text((162, y1 - 52), "C mod32 = {0,8,16,24}", font=F["cjk16"], fill=BLUE)
    draw.text((390, y1 - 52), "显示为 signed 8 的倍数中心", font=F["cjk16"], fill=MUTED)

    draw.line((79, y1 + 24, 79, y2 - 58), fill=BLUE_DARK, width=4)
    draw.polygon([(72, y2 - 58), (86, y2 - 58), (79, y2 - 44)], fill=BLUE_DARK)

    draw_axis(draw, y2, "加层 2-3", coarse, even_covered)
    draw.text((162, y2 - 52), "B signed = {-6,-4,-2,0,2,4,6}", font=F["cjk16"], fill=ORANGE)
    draw.text((470, y2 - 52), "先填满所有偶数位置", font=F["cjk16"], fill=BLUE)

    draw.line((79, y2 + 24, 79, y3 - 58), fill=BLUE_DARK, width=4)
    draw.polygon([(72, y3 - 58), (86, y3 - 58), (79, y3 - 44)], fill=BLUE_DARK)

    draw_axis(draw, y3, "加层 1", even_covered, all_covered)
    draw.text((162, y3 - 52), "A signed = {-1,0,1}", font=F["cjk16"], fill=ORANGE)
    draw.text((354, y3 - 52), "最后由 ±1 把偶数之间的空隙补齐", font=F["cjk16"], fill=BLUE)

    draw_marker(draw, 260, 815, BLUE_SOFT, BLUE, r=7)
    draw.text((280, 802), "上一阶段已有点", font=F["cjk18"], fill=BLUE, anchor="la")
    draw_marker(draw, 560, 815, NEW_ORANGE, ORANGE, r=7)
    draw.text((580, 802), "加入当前块后新增点", font=F["cjk18"], fill=INK, anchor="la")

    right_panel = (1027, 445, 1657, 842)
    rounded(draw, right_panel, radius=8, fill=WHITE, outline=BLUE, width=2)

    draw.text((1045, 482), "s = c + b + a  (mod 32)", font=F["serif24"], fill=INK)
    draw.text((1045, 526), "层 4-5：C = {0,8,16,24}  (mod32)", font=F["cjk18"], fill=BLUE)
    draw.text((1045, 554), "层 2-3：B = {-6,-4,-2,0,2,4,6}", font=F["cjk18"], fill=ORANGE)
    draw.text((1045, 582), "层 1：A = {-1,0,1}", font=F["cjk18"], fill=GREEN)

    examples = [
        ("13 = ", [("8", BLUE), (" + ", INK), ("4", ORANGE), (" + ", INK), ("1", GREEN)], "(c,b,a)=(8,4,1)"),
        ("-11 = ", [("-8", BLUE), (" - ", INK), ("2", ORANGE), (" - ", INK), ("1", GREEN)], "(c,b,a)=(-8,-2,-1)"),
    ]
    for y, (lhs, parts, coeffs) in zip([634, 748], examples):
        rounded(draw, (1045, y, 1640, y + 78), radius=6, fill=WHITE, outline=BLUE, width=2)
        draw.text((1062, y + 20), lhs, font=F["serif36"], fill=INK)
        x = 1062 + draw.textbbox((0, 0), lhs, font=F["serif36"])[2]
        for text, color in parts:
            draw.text((x, y + 20), text, font=F["serif36"], fill=color)
            x += draw.textbbox((0, 0), text, font=F["serif36"])[2]
        draw.text((1390, y + 29), coeffs, font=F["serif17"], fill=INK)


def main() -> None:
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)
    draw_top_panels(draw)
    draw_lower_panels(draw)
    img.save(IMAGE)
    print(f"rewrote {IMAGE}")


if __name__ == "__main__":
    main()
