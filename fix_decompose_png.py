#!/usr/bin/env python3
"""Patch the decomposition view in result2/1-decompose.png."""

from __future__ import annotations

from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw

from fix_cooleytukey_sparse_png import (
    AXIS,
    BLUE,
    BLUE_DARK,
    BLUE_SOFT,
    F,
    GREEN,
    GRID,
    INK,
    MUTED,
    ORANGE,
    WHITE,
    draw_centered_segments,
    rounded,
    sub,
    sup,
)


ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "result2" / "1-decompose.png"
DFT_SIZE = 190
FACTOR_SIZE = 176
NEW_ORANGE = (255, 176, 70)


def d_symbol(denom: int) -> str:
    return "D" + sub(str(denom)) + sup("(32)")


def draw_dft_matrix_formula_clean(draw: ImageDraw.ImageDraw) -> None:
    x, y = 66, 150
    w = h = DFT_SIZE

    draw.text((x + w / 2, y - 32), "DFT" + sub("32") + "ᴺᴿ", font=F["serif24"], fill=INK, anchor="ma")
    draw.rectangle((x, y, x + w, y + h), fill=(251, 253, 255), outline=(80, 85, 100), width=2)
    for i in range(1, 8):
        gx = x + i * w / 8
        gy = y + i * h / 8
        draw.line((gx, y, gx, y + h), fill=(236, 241, 248), width=1)
        draw.line((x, gy, x + w, gy), fill=(236, 241, 248), width=1)

    cols = [x + 28, x + 62, x + 96, x + 130, x + 160]
    rows = [y + 28, y + 62, y + 96, y + 130, y + 164]
    entries = [
        ["1", "1", "1", "···", "1"],
        ["1", "ω", "ω²", "···", "ω³¹"],
        ["1", "ω²", "ω⁴", "···", "ω⁶²"],
        ["...", "...", "...", "...", "..."],
        ["1", "ω³¹", "ω⁶²", "···", "ω⁹⁶¹"],
    ]
    for row_y, row in zip(rows, entries):
        for col_x, text in zip(cols, row):
            draw.text((col_x, row_y), text, font=F["serif20"], fill=BLUE_DARK, anchor="mm")

    draw.line((36, y, 36, y + h), fill=INK, width=2)
    draw.polygon([(36, y - 4), (31, y + 10), (41, y + 10)], fill=INK)
    draw.polygon([(36, y + h + 4), (31, y + h - 10), (41, y + h - 10)], fill=INK)

    draw.line((x, y + h + 18, x + w, y + h + 18), fill=INK, width=2)
    draw.polygon([(x - 4, y + h + 18), (x + 10, y + h + 13), (x + 10, y + h + 23)], fill=INK)
    draw.polygon([(x + w + 4, y + h + 18), (x + w - 10, y + h + 13), (x + w - 10, y + h + 23)], fill=INK)
    draw.text((x + w / 2, y + h + 17), "32", font=F["serif20"], fill=INK, anchor="ma")


def draw_clipped_diag(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    offset_cells: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw one visible segment of a diagonal on the finite matrix square."""
    cell = h / 32
    b = offset_cells * cell
    pts: list[tuple[float, float]] = []

    y_left = y0 + b
    if y0 <= y_left <= y0 + h:
        pts.append((x0, y_left))

    x_top = x0 - b * w / h
    if x0 <= x_top <= x0 + w:
        pts.append((x_top, y0))

    y_right = y0 + h + b
    if y0 <= y_right <= y0 + h:
        pts.append((x0 + w, y_right))

    x_bottom = x0 + (h - b) * w / h
    if x0 <= x_bottom <= x0 + w:
        pts.append((x_bottom, y0 + h))

    unique: list[tuple[float, float]] = []
    for px, py in pts:
        if not any(abs(px - ux) < 0.01 and abs(py - uy) < 0.01 for ux, uy in unique):
            unique.append((px, py))

    if len(unique) < 2:
        return

    p0, p1 = max(
        ((a, b) for i, a in enumerate(unique) for b in unique[i + 1 :]),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)


def draw_factor_matrix(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    denom: int,
    offset: int,
    *,
    half_turn: bool = False,
) -> None:
    w = h = FACTOR_SIZE
    draw.text((x0 + w / 2, y0 - 34), d_symbol(denom), font=F["serif24"], fill=INK, anchor="ma")
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(250, 250, 250), outline=(18, 18, 18), width=1)

    for i in range(1, 32):
        gx = x0 + i * w / 32
        gy = y0 + i * h / 32
        draw.line((gx, y0, gx, y0 + h), fill=GRID, width=1)
        draw.line((x0, gy, x0 + w, gy), fill=GRID, width=1)

    if half_turn:
        draw_clipped_diag(draw, x0, y0, w, h, -offset, ORANGE, 3)
        draw_clipped_diag(draw, x0, y0, w, h, offset, ORANGE, 3)
    else:
        draw_clipped_diag(draw, x0, y0, w, h, -offset, ORANGE, 2)
        draw_clipped_diag(draw, x0, y0, w, h, offset, GREEN, 2)
    draw_clipped_diag(draw, x0, y0, w, h, 0, BLUE, 4)

    caption = "2 条非零循环对角线" if half_turn else "3 条非零循环对角线"
    draw_centered_segments(draw, x0 + w / 2, y0 + h + 8, [(caption, F["cjk16"], BLUE)])

    if half_turn:
        draw_centered_segments(
            draw,
            x0 + w / 2,
            y0 + h + 34,
            [
                ("offsets = ", F["cjk14"], INK),
                ("0", F["cjk14"], BLUE),
                (", ", F["cjk14"], INK),
                ("16", F["cjk14"], ORANGE),
            ],
        )
        draw_centered_segments(
            draw,
            x0 + w / 2,
            y0 + h + 51,
            [
                ("+16 与 -16 在 mod 32 下为同一条", F["cjk12"], ORANGE),
            ],
        )
    else:
        draw_centered_segments(
            draw,
            x0 + w / 2,
            y0 + h + 34,
            [
                ("offsets = ", F["cjk15"], INK),
                ("0", F["cjk15"], BLUE),
                (", ", F["cjk15"], INK),
                (f"±{offset}", F["cjk15"], ORANGE),
            ],
        )


def draw_decompose_top(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, 1672, 444), fill=WHITE)
    draw.text((836, 8), "n = 32 的 Cooley–Tukey sparse–diagonal 分解示意", font=F["cjk26"], fill=BLUE_DARK, anchor="ma")

    rounded(draw, (407, 60, 1280, 138), radius=8, fill=WHITE, outline=BLUE, width=2)
    draw.text(
        (844, 74),
        "DFT" + sub("32") + "ᴺᴿ = D" + sub("32") + sup("(32)") + " · D" + sub("16") + sup("(32)") + " · D" + sub("8") + sup("(32)") + " · D" + sub("4") + sup("(32)") + " · D" + sub("2") + sup("(32)"),
        font=F["serif26"],
        fill=INK,
        anchor="ma",
    )
    draw.text(
        (844, 112),
        "Cooley–Tukey 适用于 32×32 矩阵分解成 5 个只有少量循环对角线的稀疏因子。",
        font=F["cjk16"],
        fill=BLUE,
        anchor="ma",
    )

    rounded(draw, (1465, 10, 1662, 118), radius=8, fill=WHITE, outline=INK, width=1)
    legend = [("主对角线 0", BLUE), ("正偏移 +offset", ORANGE), ("负偏移 -offset", GREEN)]
    for i, (label, color) in enumerate(legend):
        yy = 30 + 30 * i
        draw.rectangle((1484, yy - 10, 1504, yy + 10), fill=color, outline=color)
        draw.text((1522, yy - 12), label, font=F["cjk16"], fill=INK)

    draw_dft_matrix_formula_clean(draw)
    draw.text((304, 260), "=", font=F["serif36"], fill=INK, anchor="mm", stroke_width=1, stroke_fill=INK)

    y0 = 181
    x_positions = [357, 598, 839, 1080, 1321]
    factors = [(32, 1, False), (16, 2, False), (8, 4, False), (4, 8, False), (2, 16, True)]
    for x0, (denom, offset, half_turn) in zip(x_positions, factors):
        draw_factor_matrix(draw, x0, y0, denom, offset, half_turn=half_turn)

    for left, right in zip(x_positions, x_positions[1:]):
        draw.text(((left + FACTOR_SIZE + right) / 2, 270), "×", font=F["serif36"], fill=INK, anchor="mm")


def draw_segments(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    segments: list[tuple[str, object, tuple[int, int, int]]],
) -> None:
    for text, font_obj, color in segments:
        draw.text((x, y), text, font=font_obj, fill=color)
        x += draw.textbbox((0, 0), text, font=font_obj)[2]


def sums_for(scales: list[int]) -> set[int]:
    out: set[int] = set()
    for coeffs in product((-1, 0, 1), repeat=len(scales)):
        value = sum(c * scale for c, scale in zip(coeffs, scales))
        if -16 <= value <= 16:
            out.add(value)
    return out


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    r: int = 7,
) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2)


def draw_decompose_axis(
    draw: ImageDraw.ImageDraw,
    y: int,
    label: str,
    previous: set[int],
    current: set[int],
) -> None:
    left, right = 162, 970

    def xmap(value: int) -> float:
        return left + (value + 16) * (right - left) / 32

    rounded(draw, (22, y - 22, 128, y + 10), radius=5, fill=WHITE, outline=BLUE, width=2)
    draw.text((75, y - 14), label, font=F["cjk16"], fill=BLUE, anchor="ma")

    draw.line((left, y, right, y), fill=AXIS, width=1)
    for value in range(-16, 17):
        xx = xmap(value)
        tick = 12 if value % 4 == 0 else 8
        draw.line((xx, y - tick / 2, xx, y + tick / 2), fill=AXIS, width=1)
        draw.text((xx, y + 12), str(value), font=F["cjk13"], fill=INK, anchor="ma")

    for value in sorted(previous & current):
        draw_marker(draw, xmap(value), y - 13, BLUE_SOFT, BLUE, r=6)
    for value in sorted(current - previous):
        draw_marker(draw, xmap(value), y - 13, NEW_ORANGE, ORANGE, r=6)


def draw_example_row(
    draw: ImageDraw.ImageDraw,
    y: int,
    lhs: str,
    parts: list[tuple[str, tuple[int, int, int]]],
    coeffs: str,
) -> None:
    rounded(draw, (1045, y, 1640, y + 50), radius=6, fill=WHITE, outline=BLUE, width=2)
    draw.text((1060, y + 9), lhs, font=F["serif32"], fill=INK)
    x = 1060 + draw.textbbox((0, 0), lhs, font=F["serif32"])[2]
    for text, color in parts:
        draw.text((x, y + 9), text, font=F["serif32"], fill=color)
        x += draw.textbbox((0, 0), text, font=F["serif32"])[2]
    draw.text((1320, y + 17), coeffs, font=F["serif17"], fill=INK)


def draw_decompose_lower(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 444, 1672, 941), fill=WHITE)

    left_panel = (8, 445, 1010, 852)
    rounded(draw, left_panel, radius=8, fill=WHITE, outline=BLUE, width=2)
    draw.rounded_rectangle((8, 445, 620, 480), radius=8, fill=BLUE_DARK, outline=BLUE_DARK, width=1)
    draw.rectangle((8, 472, 620, 480), fill=BLUE_DARK)
    draw.text((24, 452), "offset简单相加即可覆盖全部diagonal index", fill=WHITE, font=F["cjk19"])

    row_ys = [505, 572, 639, 706, 773]
    labels = ["只用 ±16", "加入 ±8", "加入 ±4", "加入 ±2", "加入 ±1"]
    scales_so_far: list[int] = []
    previous: set[int] = set()
    for y, label, scale in zip(row_ys, labels, [16, 8, 4, 2, 1]):
        if scales_so_far:
            draw.line((76, y - 48, 76, y - 28), fill=BLUE_DARK, width=4)
        scales_so_far.append(scale)
        current = sums_for(scales_so_far)
        old = current if not previous else previous
        draw_decompose_axis(draw, y, label, old, current)
        previous = current

    draw_marker(draw, 260, 823, BLUE_SOFT, BLUE, r=7)
    draw.text((280, 810), "已覆盖的点（由前几级得到）", font=F["cjk18"], fill=BLUE, anchor="la")
    draw_marker(draw, 615, 823, NEW_ORANGE, ORANGE, r=7)
    draw.text((635, 810), "本级新增覆盖的点", font=F["cjk18"], fill=INK, anchor="la")

    right_panel = (1027, 460, 1657, 852)
    rounded(draw, right_panel, radius=8, fill=WHITE, outline=BLUE, width=2)
    draw.text(
        (1045, 482),
        "s = ε₁₆ · 16 + ε₈ · 8 + ε₄ · 4 + ε₂ · 2 + ε₁ · 1,    εi ∈ {-1,0,1}",
        font=F["serif20"],
        fill=INK,
    )
    draw.text((1045, 532), "每一步加入 ±m：把上一步可达点之间的空隙继续细分。", font=F["cjk18"], fill=BLUE)
    draw.text((1045, 560), "尺度顺序：16 → 8 → 4 → 2 → 1。", font=F["cjk18"], fill=BLUE)

    draw_example_row(
        draw,
        592,
        "13 = ",
        [("8", ORANGE), (" + ", INK), ("4", GREEN), (" + ", INK), ("1", BLUE)],
        "(ε₁₆, ε₈, ε₄, ε₂, ε₁) = (0,1,1,0,1)",
    )
    draw_example_row(
        draw,
        660,
        "-11 = ",
        [("-8", ORANGE), (" - ", INK), ("2", GREEN), (" - ", INK), ("1", BLUE)],
        "(ε₁₆, ε₈, ε₄, ε₂, ε₁) = (0,-1,0,-1,-1)",
    )

    conclusion = (1045, 728, 1640, 846)
    rounded(draw, conclusion, radius=7, fill=WHITE, outline=GREEN, width=2)
    draw.text((1064, 746), "结论：", font=F["cjk22"], fill=GREEN)
    draw.text((1140, 746), "offset = 各层偏移之和。", font=F["cjk22"], fill=INK)
    draw.text((1064, 786), "±16、±8、±4、±2、±1 每层选 -1/0/+1，", font=F["cjk20"], fill=BLUE)
    draw.text((1064, 818), "简单相加即可覆盖全部diagonal index。", font=F["cjk20"], fill=BLUE)

def main() -> None:
    img = Image.open(IMAGE).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw_decompose_top(draw)
    draw_decompose_lower(draw)
    img.save(IMAGE)
    print(f"rewrote {IMAGE}")


if __name__ == "__main__":
    main()
