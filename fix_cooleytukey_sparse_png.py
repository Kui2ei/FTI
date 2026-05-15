#!/usr/bin/env python3
"""Patch the generated Cooley-Tukey sparse diagonal figure.

The original figure source is not present in this workspace, so this script
keeps the existing top context and redraws the parts that need correction:

* the first decomposed sparse factor, whose offsets are 0, +/-1;
* the lower explanation panel, whose ruler points are computed from the
  ternary choices at each scale instead of placed by hand.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "result2" / "2-collapse.png"
CANVAS_WIDTH = 1672
CANVAS_HEIGHT = 860
MATRIX_SIDE = 195
MATRIX_TOP = 176
MATRIX_LEFT = 66
MATRIX_GAP = 205

BLUE = (0, 34, 255)
BLUE_DARK = (0, 24, 190)
BLUE_SOFT = (109, 166, 255)
ORANGE = (255, 99, 18)
GREEN = (0, 130, 28)
INK = (18, 20, 30)
MUTED = (60, 64, 78)
GRID = (230, 230, 230)
AXIS = (40, 40, 40)
WHITE = (255, 255, 255)

FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_SERIF_CJK = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


F = {
    "cjk12": font(FONT_CJK, 12),
    "cjk13": font(FONT_CJK, 13),
    "cjk14": font(FONT_CJK, 14),
    "cjk15": font(FONT_CJK, 15),
    "cjk16": font(FONT_CJK, 16),
    "cjk18": font(FONT_CJK, 18),
    "cjk19": font(FONT_CJK, 19),
    "cjk20": font(FONT_CJK, 20),
    "cjk21": font(FONT_CJK, 21),
    "cjk22": font(FONT_CJK, 22),
    "cjk24": font(FONT_CJK, 24),
    "cjk26": font(FONT_CJK, 26),
    "serif20": font(FONT_SERIF, 20),
    "serif17": font(FONT_SERIF, 17),
    "serif24": font(FONT_SERIF, 24),
    "serif26": font(FONT_SERIF, 26),
    "serif28": font(FONT_SERIF, 28),
    "serif32": font(FONT_SERIF, 32),
    "serif36": font(FONT_SERIF, 36),
    "serif40": font(FONT_SERIF, 40),
    "serif_bold30": font(FONT_SERIF_BOLD, 30),
    "serif_cjk24": font(FONT_SERIF_CJK, 24),
}


SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
SUP = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")


def sub(text: str) -> str:
    return text.translate(SUB)


def sup(text: str) -> str:
    return text.translate(SUP)


def bold_text(draw: ImageDraw.ImageDraw, xy, text, fill, font_obj, anchor=None, stroke=1):
    draw.text(xy, text, fill=fill, font=font_obj, anchor=anchor, stroke_width=stroke, stroke_fill=fill)


def rounded(draw: ImageDraw.ImageDraw, box, radius=7, fill=WHITE, outline=BLUE, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_centered_segments(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    segments: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]],
):
    widths = [draw.textbbox((0, 0), text, font=ft)[2] for text, ft, _ in segments]
    x = center_x - sum(widths) / 2
    for (text, ft, color), width in zip(segments, widths):
        draw.text((x, y), text, font=ft, fill=color)
        x += width


def draw_segments(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    segments: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]],
):
    for text, ft, color in segments:
        draw.text((x, y), text, font=ft, fill=color)
        x += draw.textbbox((0, 0), text, font=ft)[2]


def d_symbol(denom: int) -> str:
    return "D" + sub(str(denom)) + sup("(32)")


def draw_dft_matrix_formula(draw: ImageDraw.ImageDraw, x: int, y: int, size: int):
    w = h = size

    draw.text((x + w / 2, y - 32), "DFT" + sub("32") + "ᴺᴿ", font=F["serif24"], fill=INK, anchor="ma")
    draw.rectangle((x, y, x + w, y + h), fill=(251, 253, 255), outline=(80, 85, 100), width=2)
    for i in range(1, 8):
        gx = x + i * w / 8
        gy = y + i * h / 8
        draw.line((gx, y, gx, y + h), fill=(236, 241, 248), width=1)
        draw.line((x, gy, x + w, gy), fill=(236, 241, 248), width=1)

    cols = [x + w * p for p in (0.20, 0.39, 0.58, 0.77, 0.91)]
    rows = [y + h * p for p in (0.18, 0.37, 0.55, 0.74, 0.92)]
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


def draw_factor_matrix(draw: ImageDraw.ImageDraw, x0: int, y0: int, denom: int, offset: int):
    w, h = 172, 188
    draw.text((x0 + w / 2, y0 - 34), d_symbol(denom), font=F["serif24"], fill=INK, anchor="ma")
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(250, 250, 250), outline=(18, 18, 18), width=1)
    for i in range(1, 32):
        gx = x0 + i * w / 32
        gy = y0 + i * h / 32
        draw.line((gx, y0, gx, y0 + h), fill=GRID, width=1)
        draw.line((x0, gy, x0 + w, gy), fill=GRID, width=1)

    cell = h / 32

    def clipped_diag(offset_cells: float, color: tuple[int, int, int]):
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

        if len(unique) >= 2:
            p0, p1 = max(
                ((a, b) for i, a in enumerate(unique) for b in unique[i + 1 :]),
                key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
            )
            width = 3 if color == BLUE else 2
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)

    clipped_diag(-offset, ORANGE)
    clipped_diag(offset, GREEN)
    clipped_diag(0, BLUE)

    draw_centered_segments(
        draw,
        x0 + w / 2,
        y0 + h + 8,
        [("3 条非零循环对角线", F["cjk16"], BLUE)],
    )
    draw_centered_segments(
        draw,
        x0 + w / 2,
        y0 + h + 36,
        [
            ("offsets = ", F["cjk15"], INK),
            ("0", F["cjk15"], BLUE),
            (", ", F["cjk15"], INK),
            (f"±{offset}", F["cjk15"], ORANGE),
        ],
    )


def draw_collapsed_matrix(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    title: str,
    offsets: list[int],
    caption: str,
    set_label: str,
):
    draw.text((x0 + w / 2, y0 - 34), title, font=F["serif20"], fill=INK, anchor="ma")
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(250, 250, 250), outline=(18, 18, 18), width=1)
    for i in range(1, 32):
        gx = x0 + i * w / 32
        gy = y0 + i * h / 32
        draw.line((gx, y0, gx, y0 + h), fill=GRID, width=1)
        draw.line((x0, gy, x0 + w, gy), fill=GRID, width=1)

    cell = h / 32

    def clipped_diag(offset_cells: float, color: tuple[int, int, int], width: int):
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

    for off in offsets:
        if off < 0:
            clipped_diag(abs(off), GREEN, 2)
        elif off > 0:
            clipped_diag(-off, ORANGE, 2)
    clipped_diag(0, BLUE, 4)

    draw_centered_segments(draw, x0 + w / 2, y0 + h + 8, [(caption, F["cjk16"], BLUE)])
    if isinstance(set_label, (list, tuple)):
        for i, line_text in enumerate(set_label):
            draw_centered_segments(draw, x0 + w / 2, y0 + h + 32 + i * 17, [(line_text, F["cjk12"], INK)])
    else:
        draw_centered_segments(draw, x0 + w / 2, y0 + h + 34, [(set_label, F["cjk14"], INK)])


def draw_top_panels(draw: ImageDraw.ImageDraw):
    draw.rectangle((0, 0, CANVAS_WIDTH, 444), fill=WHITE)
    draw.text((836, 8), "n = 32 的 Cooley–Tukey sparse–diagonal 折叠示意", font=F["cjk26"], fill=BLUE_DARK, anchor="ma")

    rounded(draw, (407, 60, 1280, 138), radius=8, fill=WHITE, outline=BLUE, width=2)
    draw.text(
        (844, 74),
        "DFT" + sub("32") + "ᴺᴿ = C" + sub("1-2") + " · C" + sub("3-5"),
        font=F["serif26"],
        fill=INK,
        anchor="ma",
    )
    draw.text(
        (844, 112),
        "层 1–2 得到 7 条；层 3–5 形式上 15 个 shift，mod 32 合并后 distinct diagonal 为 8 条。",
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

    square_side = MATRIX_SIDE
    matrix_y = MATRIX_TOP
    dft_x = MATRIX_LEFT
    c12_x = dft_x + square_side + MATRIX_GAP
    c35_x = c12_x + square_side + MATRIX_GAP
    operator_y = matrix_y + square_side / 2

    draw_dft_matrix_formula(draw, dft_x, matrix_y, square_side)
    draw.text(
        ((dft_x + square_side + c12_x) / 2, operator_y),
        "=",
        font=F["serif36"],
        fill=INK,
        anchor="mm",
        stroke_width=1,
        stroke_fill=INK,
    )
    draw_collapsed_matrix(
        draw,
        c12_x,
        matrix_y,
        square_side,
        square_side,
        "block: layers 1-2",
        list(range(-3, 4)),
        "7 条 distinct 循环对角线",
        [
            "signed: {-3,-2,-1,0,1,2,3}",
            "mod32: {0,1,2,3,29,30,31}",
        ],
    )
    draw.text(((c12_x + square_side + c35_x) / 2, operator_y), "×", font=F["serif36"], fill=INK, anchor="mm")
    draw_collapsed_matrix(
        draw,
        c35_x,
        matrix_y,
        square_side,
        square_side,
        "block: layers 3-5",
        [-16, -12, -8, -4, 0, 4, 8, 12, 16],
        "8 条 distinct 循环对角线",
        [
            "shown signed: {-16,-12,...,12,16}",
            "mod32 distinct: {0,4,8,12,16,20,24,28}",
        ],
    )
    draw.text((1190, 252), "折叠减少乘法层数", font=F["cjk16"], fill=BLUE, anchor="ma")
    draw.text((1190, 282), "但每层更稠密", font=F["cjk16"], fill=MUTED, anchor="ma")


def draw_corrected_first_factor(draw: ImageDraw.ImageDraw):
    # Clear only the first factor block so the original formula and neighbors remain untouched.
    draw.rectangle((344, 145, 540, 430), fill=WHITE)

    x0, y0 = 357, 181
    w, h = 172, 188
    title = "D" + sub("32") + sup("(32)")
    draw.text((x0 + w / 2, 148), title, font=F["serif24"], fill=INK, anchor="ma")

    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(250, 250, 250), outline=(18, 18, 18), width=1)
    for i in range(1, 32):
        gx = x0 + i * w / 32
        gy = y0 + i * h / 32
        draw.line((gx, y0, gx, y0 + h), fill=GRID, width=1)
        draw.line((x0, gy, x0 + w, gy), fill=GRID, width=1)

    cell = h / 32

    def clipped_diag(offset_cells: float, color: tuple[int, int, int]):
        # offset_cells > 0 draws below the main diagonal, < 0 above.
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
        if len(pts) >= 2:
            draw.line((pts[0][0], pts[0][1], pts[1][0], pts[1][1]), fill=color, width=4)

    clipped_diag(-1, ORANGE)
    clipped_diag(0, BLUE)
    clipped_diag(1, GREEN)

    draw_centered_segments(
        draw,
        x0 + w / 2,
        378,
        [
            ("3 条非零循环对角线", F["cjk16"], BLUE),
        ],
    )
    draw_centered_segments(
        draw,
        x0 + w / 2,
        406,
        [
            ("offsets = ", F["cjk15"], INK),
            ("0", F["cjk15"], BLUE),
            (", ", F["cjk15"], INK),
            ("±1", F["cjk15"], ORANGE),
        ],
    )


def reachable(scales: list[int]) -> set[int]:
    values = set()
    for coeffs in product((-1, 0, 1), repeat=len(scales)):
        s = sum(c * scale for c, scale in zip(coeffs, scales))
        if -16 <= s <= 16:
            values.add(s)
    return values


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, fill, outline, r=6):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2)


def draw_number_line(draw: ImageDraw.ImageDraw, y: int, label: str, prev: set[int], current: set[int]):
    left, right = 162, 970

    def xm(v: int) -> float:
        return left + (v + 16) * (right - left) / 32

    rounded(draw, (22, y - 22, 128, y + 10), radius=5, fill=WHITE, outline=BLUE, width=2)
    draw.text((75, y - 14), label, font=F["cjk16"], fill=BLUE, anchor="ma")

    draw.line((left, y, right, y), fill=AXIS, width=1)
    for v in range(-16, 17):
        x = xm(v)
        tick = 12 if v % 4 == 0 else 9
        draw.line((x, y - tick / 2, x, y + tick / 2), fill=AXIS, width=1)
        draw.text((x, y + 12), str(v), font=F["cjk13"], fill=INK, anchor="ma")

    for v in sorted(prev & current):
        draw_marker(draw, xm(v), y - 13, BLUE_SOFT, BLUE, r=6)
    for v in sorted(current - prev):
        draw_marker(draw, xm(v), y - 13, (255, 176, 70), ORANGE, r=6)


def draw_lower_panels(draw: ImageDraw.ImageDraw):
    draw.rectangle((0, 440, CANVAS_WIDTH, CANVAS_HEIGHT), fill=WHITE)

    left_panel = (8, 445, 1010, 842)
    rounded(draw, left_panel, radius=8, fill=WHITE, outline=BLUE, width=2)
    draw.rounded_rectangle((8, 445, 610, 480), radius=8, fill=BLUE_DARK, outline=BLUE_DARK, width=1)
    draw.text((24, 452), "折叠后仍可覆盖全部 diagonal index", fill=WHITE, font=F["cjk19"])

    fine = set(range(-3, 4))
    # The modulo-32 set {0,4,8,12,16,20,24,28} is shown on a signed
    # [-16,16] ruler. The two endpoints -16 and 16 represent the same
    # modulo-32 diagonal, but drawing both makes the closed range readable.
    coarse = {-16, -12, -8, -4, 0, 4, 8, 12, 16}
    covered = {a + b for a in fine for b in coarse if -16 <= a + b <= 16}

    left, right = 162, 970

    def xmap(v: int) -> float:
        return left + (v + 16) * (right - left) / 32

    def axis(y: int):
        draw.line((left, y, right, y), fill=AXIS, width=1)
        for v in range(-16, 17):
            x = xmap(v)
            major = v % 4 == 0
            draw.line((x, y - (8 if major else 5), x, y + (8 if major else 5)), fill=AXIS, width=1)
            draw.text((x, y + 12), str(v), font=F["cjk13"], fill=INK, anchor="ma")

    def row(y: int, label: str, previous: set[int], current: set[int]):
        rounded(draw, (22, y - 22, 136, y + 10), radius=5, fill=WHITE, outline=BLUE, width=2)
        draw.text((79, y - 14), label, font=F["cjk15"], fill=BLUE, anchor="ma")
        axis(y)
        for v in sorted(previous & current):
            draw_marker(draw, xmap(v), y - 13, BLUE_SOFT, BLUE, r=6)
        for v in sorted(current - previous):
            draw_marker(draw, xmap(v), y - 13, (255, 176, 70), ORANGE, r=6)

    y1, y2 = 545, 685
    row(y1, "层 3-5", coarse, coarse)
    draw.text((left, y1 - 52), "B mod32 = {0,4,8,12,16,20,24,28}", font=F["cjk16"], fill=BLUE)
    draw.text((left + 470, y1 - 52), "显示为 signed 4 的倍数中心", font=F["cjk16"], fill=MUTED)

    draw.line((79, y1 + 24, 79, y2 - 60), fill=BLUE_DARK, width=4)
    draw.polygon([(72, y2 - 60), (86, y2 - 60), (79, y2 - 46)], fill=BLUE_DARK)

    row(y2, "加层 1-2", coarse, covered)
    draw.text((left, y2 - 52), "A signed = {-3,-2,-1,0,1,2,3}", font=F["cjk16"], fill=ORANGE)
    draw.text((left + 390, y2 - 52), "每个 B 中心加上 A，填满相邻中心之间的空隙", font=F["cjk16"], fill=BLUE)

    draw.text((left, 748), "合并后 offset = b+a；在 [-16,16] 的尺度上，所有整数都被层 1-2 的残差补齐。", font=F["cjk18"], fill=GREEN)

    draw_marker(draw, 260, 808, BLUE_SOFT, BLUE, r=7)
    draw.text((280, 795), "层 3-5 已有点", font=F["cjk18"], fill=BLUE, anchor="la")
    draw_marker(draw, 560, 808, (255, 176, 70), ORANGE, r=7)
    draw.text((580, 795), "加入层 1-2 后新增点", font=F["cjk18"], fill=INK, anchor="la")

    right_panel = (1027, 445, 1657, 842)
    rounded(draw, right_panel, radius=8, fill=WHITE, outline=BLUE, width=2)

    draw.text(
        (1045, 482),
        "s = b + a  (mod 32)",
        font=F["serif24"],
        fill=INK,
    )
    draw.text((1045, 526), "层 3-5：B = {0,4,8,12,16,20,24,28}  (mod32)", font=F["cjk18"], fill=BLUE)
    draw.text((1045, 554), "层 1-2：A = {-3,-2,-1,0,1,2,3}", font=F["cjk18"], fill=ORANGE)

    examples = [
        ("13 = ", [("12", BLUE), (" + ", INK), ("1", ORANGE)], "(b,a)=(12,1)"),
        ("-11 = ", [("-12", BLUE), (" + ", INK), ("1", ORANGE)], "(b,a)=(-12,1)"),
    ]
    box_y = [608, 724]
    for y, (lhs, parts, coeffs) in zip(box_y, examples):
        rounded(draw, (1045, y, 1640, y + 78), radius=6, fill=WHITE, outline=BLUE, width=2)
        draw.text((1062, y + 20), lhs, font=F["serif36"], fill=INK)
        x = 1062 + draw.textbbox((0, 0), lhs, font=F["serif36"])[2]
        for text, color in parts:
            draw.text((x, y + 20), text, font=F["serif36"], fill=color)
            x += draw.textbbox((0, 0), text, font=F["serif36"])[2]
        draw.text(
            (1320, y + 29),
            coeffs,
            font=F["serif17"],
            fill=INK,
        )


def main() -> None:
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)
    draw_top_panels(draw)
    draw_lower_panels(draw)
    img.save(IMAGE)
    print(f"rewrote {IMAGE}")


if __name__ == "__main__":
    main()
