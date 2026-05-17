#!/usr/bin/env python3
"""Redraw the sparse-diagonal flowcharts in a paper-style schematic.

The script keeps the original PNG files untouched and writes new high-resolution
PNG files under "Nochangedelete copy/paper_polished_outputs".
"""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Nochangedelete copy" / "paper_polished_outputs"

FONT_SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_SANS_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

WHITE = (255, 255, 255)
PANEL = (248, 250, 252)
INK = (30, 41, 59)
MUTED = (71, 85, 105)
FAINT = (226, 232, 240)
GRID = (229, 234, 241)
AXIS = (51, 65, 85)
BLUE = (37, 99, 235)
BLUE_DARK = (30, 64, 175)
BLUE_SOFT = (191, 219, 254)
CYAN = (8, 145, 178)
CYAN_SOFT = (207, 250, 254)
ORANGE = (234, 88, 12)
ORANGE_SOFT = (254, 215, 170)
GREEN = (21, 128, 61)
GREEN_SOFT = (187, 247, 208)
BORDER = (148, 163, 184)

SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
SUP = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")


def sub(text: str) -> str:
    return text.translate(SUB)


def sup(text: str) -> str:
    return text.translate(SUP)


def font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_SERIF_BOLD if serif and bold else FONT_SERIF if serif else FONT_SANS_BOLD if bold else FONT_SANS
    return ImageFont.truetype(path, size)


F = {
    "title": font(34, bold=True),
    "subtitle": font(20),
    "section": font(23, bold=True),
    "body": font(19),
    "small": font(16),
    "tiny": font(14),
    "formula": font(30, serif=True),
    "formula_lg": font(42, serif=True, bold=True),
    "matrix": font(24, serif=True),
    "operator": font(48, serif=True, bold=True),
}


def text_w(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=ft)
    return box[2] - box[0]


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    radius: int = 10,
    fill: tuple[int, int, int] = WHITE,
    outline: tuple[int, int, int] = BORDER,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_centered_segments(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    segments: Sequence[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]],
) -> None:
    total = sum(text_w(draw, text, ft) for text, ft, _ in segments)
    x = center_x - total / 2
    for text, ft, color in segments:
        draw.text((x, y), text, font=ft, fill=color)
        x += text_w(draw, text, ft)


def draw_segments(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    segments: Sequence[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]],
) -> float:
    for text, ft, color in segments:
        draw.text((x, y), text, font=ft, fill=color)
        x += text_w(draw, text, ft)
    return x


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int] = BLUE_DARK,
    width: int = 4,
    head: int = 14,
) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=fill, width=width)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (x1, y1)
    p2 = (x1 - head * ux + head * 0.55 * px, y1 - head * uy + head * 0.55 * py)
    p3 = (x1 - head * ux - head * 0.55 * px, y1 - head * uy - head * 0.55 * py)
    draw.polygon((p1, p2, p3), fill=fill)


def marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    r: int = 9,
) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=3)


def xmap(value: int, left: float, right: float) -> float:
    return left + (value + 16) * (right - left) / 32


def draw_axis(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    y: int,
    *,
    label: str,
    label_color: tuple[int, int, int] = BLUE_DARK,
    label_width: int = 156,
    label_x: int | None = None,
    label_ticks: bool = True,
) -> None:
    lx = left - label_width - 24 if label_x is None else label_x
    rounded(draw, (lx, y - 28, lx + label_width, y + 18), radius=8, fill=WHITE, outline=label_color, width=2)
    draw.text((lx + label_width / 2, y - 19), label, font=F["small"], fill=label_color, anchor="ma")
    draw.line((left, y, right, y), fill=AXIS, width=2)
    for value in range(-16, 17):
        x = xmap(value, left, right)
        major = value % 4 == 0
        tick = 20 if major else 11
        draw.line((x, y - tick / 2, x, y + tick / 2), fill=AXIS, width=1)
        if label_ticks and major:
            draw.text((x, y + 18), str(value), font=F["tiny"], fill=MUTED, anchor="ma")


def draw_coverage_row(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    y: int,
    *,
    label: str,
    previous: Iterable[int],
    current: Iterable[int],
    note: str = "",
    label_color: tuple[int, int, int] = BLUE_DARK,
) -> None:
    prev = set(previous)
    curr = set(current)
    draw_axis(draw, left, right, y, label=label, label_color=label_color)
    for value in sorted(curr):
        if value in prev:
            marker(draw, xmap(value, left, right), y - 19, fill=BLUE_SOFT, outline=BLUE, r=9)
        else:
            marker(draw, xmap(value, left, right), y - 19, fill=ORANGE_SOFT, outline=ORANGE, r=9)
    if note:
        draw.text((left, y - 64), note, font=F["small"], fill=MUTED)


def clipped(values: Iterable[int]) -> set[int]:
    return {v for v in values if -16 <= v <= 16}


def ternary_sums(scales: Sequence[int]) -> set[int]:
    return clipped(sum(c * s for c, s in zip(coeffs, scales)) for coeffs in product((-1, 0, 1), repeat=len(scales)))


def diag_line(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    offset_cells: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
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
    for point in pts:
        if not any(abs(point[0] - old[0]) < 0.1 and abs(point[1] - old[1]) < 0.1 for old in unique):
            unique.append(point)
    if len(unique) < 2:
        return

    a, b = max(
        ((p, q) for i, p in enumerate(unique) for q in unique[i + 1 :]),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    draw.line((a[0], a[1], b[0], b[1]), fill=color, width=width)


def draw_sparse_matrix(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    side: int,
    *,
    title: str,
    offsets: Sequence[int],
    caption: str,
    detail: str,
    detail2: str = "",
) -> None:
    title_font = F["formula"] if title.startswith("D") else F["body"]
    draw.text((x + side / 2, y - 42), title, font=title_font, fill=INK, anchor="ma")
    draw.rectangle((x, y, x + side, y + side), fill=(253, 254, 255), outline=AXIS, width=2)
    for i in range(1, 32):
        gx = x + i * side / 32
        gy = y + i * side / 32
        draw.line((gx, y, gx, y + side), fill=GRID, width=1)
        draw.line((x, gy, x + side, gy), fill=GRID, width=1)

    for off in offsets:
        if off < 0:
            diag_line(draw, x, y, side, side, abs(off), GREEN, 3)
        elif off > 0:
            diag_line(draw, x, y, side, side, -off, ORANGE, 3)
    diag_line(draw, x, y, side, side, 0, BLUE, 6)

    draw.text((x + side / 2, y + side + 16), caption, font=F["small"], fill=BLUE_DARK, anchor="ma")
    draw.text((x + side / 2, y + side + 43), detail, font=F["tiny"], fill=MUTED, anchor="ma")
    if detail2:
        draw.text((x + side / 2, y + side + 64), detail2, font=F["tiny"], fill=MUTED, anchor="ma")


def draw_dft_matrix(draw: ImageDraw.ImageDraw, x: int, y: int, side: int) -> None:
    draw.text((x + side / 2, y - 44), "DFT" + sub("32") + "ᴺᴿ", font=F["formula"], fill=INK, anchor="ma")
    draw.rectangle((x, y, x + side, y + side), fill=(253, 254, 255), outline=AXIS, width=2)
    for i in range(1, 8):
        gx = x + i * side / 8
        gy = y + i * side / 8
        draw.line((gx, y, gx, y + side), fill=GRID, width=1)
        draw.line((x, gy, x + side, gy), fill=GRID, width=1)
    rows = [0.17, 0.36, 0.55, 0.74, 0.91]
    cols = [0.20, 0.39, 0.58, 0.77, 0.91]
    entries = [
        ["1", "1", "1", "···", "1"],
        ["1", "ω", "ω²", "···", "ω³¹"],
        ["1", "ω²", "ω⁴", "···", "ω⁶²"],
        ["···", "···", "···", "···", "···"],
        ["1", "ω³¹", "ω⁶²", "···", "ω⁹⁶¹"],
    ]
    for ry, row in zip(rows, entries):
        for cx, text in zip(cols, row):
            draw.text((x + cx * side, y + ry * side), text, font=F["matrix"], fill=BLUE_DARK, anchor="mm")
    draw.line((x - 36, y, x - 36, y + side), fill=INK, width=3)
    draw.polygon(((x - 36, y - 9), (x - 46, y + 15), (x - 26, y + 15)), fill=INK)
    draw.polygon(((x - 36, y + side + 9), (x - 46, y + side - 15), (x - 26, y + side - 15)), fill=INK)
    draw.line((x, y + side + 28, x + side, y + side + 28), fill=INK, width=3)
    draw.polygon(((x - 9, y + side + 28), (x + 17, y + side + 18), (x + 17, y + side + 38)), fill=INK)
    draw.polygon(((x + side + 9, y + side + 28), (x + side - 17, y + side + 18), (x + side - 17, y + side + 38)), fill=INK)
    draw.text((x + side / 2, y + side + 28), "32", font=F["body"], fill=INK, anchor="ma")


def legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    rounded(draw, (x, y, x + 300, y + 134), radius=12, fill=WHITE, outline=BORDER, width=2)
    items = [
        ("main diagonal 0", BLUE),
        ("positive offset +s", ORANGE),
        ("negative offset -s", GREEN),
    ]
    for idx, (label, color) in enumerate(items):
        yy = y + 28 + idx * 38
        draw.rounded_rectangle((x + 24, yy - 11, x + 48, yy + 11), radius=5, fill=color, outline=color)
        draw.text((x + 68, yy - 13), label, font=F["small"], fill=INK)


def title_block(draw: ImageDraw.ImageDraw, width: int, title: str, subtitle: str) -> None:
    draw.text((width / 2, 36), title, font=F["title"], fill=INK, anchor="ma")
    draw.text((width / 2, 84), subtitle, font=F["subtitle"], fill=MUTED, anchor="ma")


def formula_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    formula: str,
    note: str,
) -> None:
    rounded(draw, box, radius=14, fill=WHITE, outline=BLUE, width=2)
    x0, y0, x1, _ = box
    draw.text(((x0 + x1) / 2, y0 + 18), formula, font=F["formula"], fill=INK, anchor="ma")
    draw.text(((x0 + x1) / 2, y0 + 65), note, font=F["small"], fill=BLUE_DARK, anchor="ma")


def panel_header(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, text: str) -> None:
    draw.rounded_rectangle((x, y, x + w, y + 48), radius=12, fill=BLUE_DARK, outline=BLUE_DARK)
    draw.text((x + 24, y + 11), text, font=F["body"], fill=WHITE)


def expression_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    segments: Sequence[tuple[str, tuple[int, int, int]]],
    tail: str,
) -> None:
    rounded(draw, box, radius=12, fill=WHITE, outline=BLUE, width=2)
    x0, y0, _, _ = box
    x = x0 + 26
    for text, color in segments:
        draw.text((x, y0 + 24), text, font=F["formula_lg"], fill=color)
        x += text_w(draw, text, F["formula_lg"])
    draw.text((x + 28, y0 + 39), tail, font=F["body"], fill=INK)


def create_canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), WHITE)
    return img, ImageDraw.Draw(img)


def draw_footer_note(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.rounded_rectangle((x, y, x + 22, y + 22), radius=5, fill=CYAN, outline=CYAN)
    draw.text((x + 34, y - 2), text, font=F["small"], fill=MUTED)


def decompose() -> Image.Image:
    w, h = 2400, 1350
    img, draw = create_canvas(w, h)
    title_block(
        draw,
        w,
        "Cooley-Tukey Sparse-Diagonal Decomposition (n = 32)",
        "Each factor contributes a ternary offset; their sums cover every diagonal index.",
    )
    formula_box(
        draw,
        (560, 125, 1840, 225),
        "DFT" + sub("32") + "ᴺᴿ = D" + sub("32") + sup("(32)") + " · D" + sub("16") + sup("(32)") + " · D" + sub("8") + sup("(32)") + " · D" + sub("4") + sup("(32)") + " · D" + sub("2") + sup("(32)"),
        "Five sparse factors encode offsets {±1, ±2, ±4, ±8, ±16}.",
    )
    legend(draw, 2060, 46)
    draw_dft_matrix(draw, 100, 310, 250)
    draw.text((435, 430), "=", font=F["operator"], fill=INK, anchor="mm")

    xs = [530, 825, 1120, 1415, 1710]
    factors = [
        ("D" + sub("32") + sup("(32)"), [-1, 0, 1], "3 nonzero cyclic diagonals", "offsets: 0, ±1"),
        ("D" + sub("16") + sup("(32)"), [-2, 0, 2], "3 nonzero cyclic diagonals", "offsets: 0, ±2"),
        ("D" + sub("8") + sup("(32)"), [-4, 0, 4], "3 nonzero cyclic diagonals", "offsets: 0, ±4"),
        ("D" + sub("4") + sup("(32)"), [-8, 0, 8], "3 nonzero cyclic diagonals", "offsets: 0, ±8"),
        ("D" + sub("2") + sup("(32)"), [0, 16], "2 nonzero cyclic diagonals", "offsets: 0, 16 (±16 coincide)"),
    ]
    for i, (x, (title, offsets, cap, detail)) in enumerate(zip(xs, factors)):
        draw_sparse_matrix(draw, x, 305, 235, title=title, offsets=offsets, caption=cap, detail=detail)
        if i < len(xs) - 1:
            draw.text((x + 265, 430), "×", font=F["operator"], fill=INK, anchor="mm")

    rounded(draw, (55, 760, 1435, 1290), radius=14, fill=WHITE, outline=BORDER, width=2)
    panel_header(draw, 55, 760, 850, "Offset coverage by simple addition")
    left, right = 300, 1360
    stages = [
        ("±16", [16], ""),
        ("+ ±8", [16, 8], ""),
        ("+ ±4", [16, 8, 4], ""),
        ("+ ±2", [16, 8, 4, 2], ""),
        ("+ ±1", [16, 8, 4, 2, 1], ""),
    ]
    prev: set[int] = set()
    ys = [870, 960, 1050, 1140, 1230]
    for (label, scales, note), y in zip(stages, ys):
        curr = ternary_sums(scales)
        draw_coverage_row(draw, left, right, y, label=label, previous=prev, current=curr, note=note)
        prev = curr
    marker(draw, 525, 1270, fill=BLUE_SOFT, outline=BLUE, r=9)
    draw.text((550, 1255), "covered from previous stage", font=F["small"], fill=MUTED)
    marker(draw, 900, 1270, fill=ORANGE_SOFT, outline=ORANGE, r=9)
    draw.text((925, 1255), "newly covered at current stage", font=F["small"], fill=MUTED)

    rounded(draw, (1485, 760, 2345, 1290), radius=14, fill=WHITE, outline=BORDER, width=2)
    panel_header(draw, 1485, 760, 500, "Index synthesis rule")
    draw.text((1525, 850), "s = ε" + sub("16") + "·16 + ε" + sub("8") + "·8 + ε" + sub("4") + "·4 + ε" + sub("2") + "·2 + ε" + sub("1"), font=F["formula"], fill=INK)
    draw.text((1525, 902), "εi ∈ {-1, 0, 1}; diagonal offsets are formed by summing layer choices.", font=F["body"], fill=MUTED)
    expression_card(
        draw,
        (1525, 965, 2300, 1065),
        [("13 = ", INK), ("8", ORANGE), (" + ", INK), ("4", GREEN), (" + ", INK), ("1", BLUE)],
        "(ε16, ε8, ε4, ε2, ε1) = (0, 1, 1, 0, 1)",
    )
    expression_card(
        draw,
        (1525, 1100, 2300, 1200),
        [("-11 = ", INK), ("-8", ORANGE), (" - ", INK), ("2", GREEN), (" - ", INK), ("1", BLUE)],
        "(ε16, ε8, ε4, ε2, ε1) = (0, -1, 0, -1, -1)",
    )
    draw_footer_note(draw, 1525, 1243, "Conclusion: the sparse factors jointly span all diagonal indices in [-16, 16].")
    return img


def draw_collapse_top(
    draw: ImageDraw.ImageDraw,
    *,
    title: str,
    subtitle: str,
    formula: str,
    note: str,
    mode: str,
) -> None:
    title_block(draw, 2400, title, subtitle)
    formula_box(draw, (560, 125, 1840, 225), formula, note)
    legend(draw, 2060, 46)
    draw_dft_matrix(draw, 115, 320, 250)
    draw.text((455, 440), "=", font=F["operator"], fill=INK, anchor="mm")
    if mode == "lb2":
        xs = [590, 1170]
        draw_sparse_matrix(
            draw,
            xs[0],
            325,
            250,
            title="block: layers 1-2",
            offsets=list(range(-3, 4)),
            caption="7 distinct cyclic diagonals",
            detail="signed: {-3, -2, -1, 0, 1, 2, 3}",
            detail2="mod32: {0, 1, 2, 3, 29, 30, 31}",
        )
        draw.text((1000, 450), "×", font=F["operator"], fill=INK, anchor="mm")
        draw_sparse_matrix(
            draw,
            xs[1],
            325,
            250,
            title="block: layers 3-5",
            offsets=[-16, -12, -8, -4, 0, 4, 8, 12, 16],
            caption="8 distinct cyclic diagonals",
            detail="shown signed: {-16, -12, ..., 12, 16}",
            detail2="mod32 distinct: {0, 4, 8, ..., 28}",
        )
        rounded(draw, (1650, 365, 1985, 520), radius=14, fill=PANEL, outline=FAINT, width=2)
        draw.text((1818, 397), "Fewer factors", font=F["section"], fill=BLUE_DARK, anchor="ma")
        draw.text((1818, 444), "but denser sparse blocks", font=F["body"], fill=MUTED, anchor="ma")
    elif mode == "lb3":
        xs = [540, 900, 1260]
        mats = [
            ("block: layer 1", [-1, 0, 1], "3 distinct cyclic diagonals", "signed: {-1, 0, 1}", "mod32: {0, 1, 31}"),
            ("block: layers 2-3", [-6, -4, -2, 0, 2, 4, 6], "7 distinct cyclic diagonals", "signed: {-6, -4, -2, 0, 2, 4, 6}", "mod32: {0, 2, 4, 6, 26, 28, 30}"),
            ("block: layers 4-5", [-24, -16, -8, 0, 8, 16, 24], "4 distinct cyclic diagonals", "formal: {-24, -16, -8, 0, 8, 16, 24}", "mod32 distinct: {0, 8, 16, 24}"),
        ]
        for i, (x, item) in enumerate(zip(xs, mats)):
            draw_sparse_matrix(draw, x, 325, 245, title=item[0], offsets=item[1], caption=item[2], detail=item[3], detail2=item[4])
            if i < 2:
                draw.text((x + 305, 450), "×", font=F["operator"], fill=INK, anchor="mm")
        rounded(draw, (1650, 365, 1985, 520), radius=14, fill=PANEL, outline=FAINT, width=2)
        draw.text((1818, 397), "Three factors", font=F["section"], fill=BLUE_DARK, anchor="ma")
        draw.text((1818, 444), "sparser per block, one extra level", font=F["body"], fill=MUTED, anchor="ma")


def draw_lb2_body(draw: ImageDraw.ImageDraw, *, include_bsgs: bool = False) -> None:
    rounded(draw, (55, 720, 1450, 1245), radius=14, fill=WHITE, outline=BORDER, width=2)
    panel_header(draw, 55, 720, 860, "Collapsed factors still cover every diagonal index")
    left, right = 300, 1370
    coarse = {-16, -12, -8, -4, 0, 4, 8, 12, 16}
    fine = set(range(-3, 4))
    covered = clipped(b + a for b in coarse for a in fine)
    draw_coverage_row(
        draw,
        left,
        right,
        855,
        label="layers 3-5",
        previous=coarse,
        current=coarse,
        note="B mod32 = {0, 4, 8, 12, 16, 20, 24, 28}; shown as signed multiples of 4.",
    )
    arrow(draw, (170, 900), (170, 990), fill=BLUE_DARK, width=5, head=18)
    draw_coverage_row(
        draw,
        left,
        right,
        1065,
        label="+ layers 1-2",
        previous=coarse,
        current=covered,
        note="A signed = {-3, -2, -1, 0, 1, 2, 3}; each coarse center fills the adjacent gap.",
    )
    draw.text((300, 1160), "Merged offset:  s = b + a  (mod 32).  On the signed ruler, every integer in [-16, 16] is reached.", font=F["body"], fill=GREEN)
    marker(draw, 500, 1215, fill=BLUE_SOFT, outline=BLUE, r=9)
    draw.text((526, 1200), "coarse set", font=F["small"], fill=MUTED)
    marker(draw, 735, 1215, fill=ORANGE_SOFT, outline=ORANGE, r=9)
    draw.text((761, 1200), "filled by residual set", font=F["small"], fill=MUTED)

    rounded(draw, (1500, 720, 2345, 1285), radius=14, fill=WHITE, outline=BORDER, width=2)
    panel_header(draw, 1500, 720, 455, "Modular index examples")
    draw.text((1545, 810), "s = b + a  (mod 32)", font=F["formula"], fill=INK)
    draw.text((1545, 865), "B = {0, 4, 8, 12, 16, 20, 24, 28}", font=F["body"], fill=BLUE_DARK)
    draw.text((1545, 905), "A = {-3, -2, -1, 0, 1, 2, 3}", font=F["body"], fill=ORANGE)
    expression_card(
        draw,
        (1545, 970, 2295, 1075),
        [("13 = ", INK), ("12", BLUE), (" + ", INK), ("1", ORANGE)],
        "(b, a) = (12, 1)",
    )
    expression_card(
        draw,
        (1545, 1110, 2295, 1215),
        [("-11 = ", INK), ("-12", BLUE), (" + ", INK), ("1", ORANGE)],
        "(b, a) = (-12, 1)",
    )

    if include_bsgs:
        draw_bsgs_panel(draw, (90, 1300, 2310, 1755), compact=True)


def draw_lb3_body(draw: ImageDraw.ImageDraw) -> None:
    rounded(draw, (55, 720, 1450, 1245), radius=14, fill=WHITE, outline=BORDER, width=2)
    panel_header(draw, 55, 720, 850, "levelBudget = 3: three residual stages fill the ruler")
    left, right = 300, 1370
    c = {-16, -8, 0, 8, 16}
    b = {-6, -4, -2, 0, 2, 4, 6}
    a = {-1, 0, 1}
    even = clipped(ci + bi for ci in c for bi in b)
    full = clipped(e + ai for e in even for ai in a)
    rows = [
        ("layers 4-5", c, c, "C mod32 = {0, 8, 16, 24}; shown as signed multiples of 8."),
        ("+ layers 2-3", c, even, "B signed = {-6, -4, -2, 0, 2, 4, 6}; even positions are filled first."),
        ("+ layer 1", even, full, "A signed = {-1, 0, 1}; odd gaps are filled at the final stage."),
    ]
    for y, row in zip([840, 990, 1140], rows):
        draw_coverage_row(draw, left, right, y, label=row[0], previous=row[1], current=row[2], note=row[3])
    arrow(draw, (170, 880), (170, 935), fill=BLUE_DARK, width=5, head=18)
    arrow(draw, (170, 1030), (170, 1085), fill=BLUE_DARK, width=5, head=18)

    rounded(draw, (1500, 720, 2345, 1285), radius=14, fill=WHITE, outline=BORDER, width=2)
    panel_header(draw, 1500, 720, 455, "Three-term synthesis")
    draw.text((1545, 810), "s = c + b + a  (mod 32)", font=F["formula"], fill=INK)
    draw.text((1545, 865), "C = {0, 8, 16, 24}", font=F["body"], fill=BLUE_DARK)
    draw.text((1545, 905), "B = {-6, -4, -2, 0, 2, 4, 6}", font=F["body"], fill=ORANGE)
    draw.text((1545, 945), "A = {-1, 0, 1}", font=F["body"], fill=GREEN)
    expression_card(
        draw,
        (1545, 1010, 2295, 1115),
        [("13 = ", INK), ("8", BLUE), (" + ", INK), ("4", ORANGE), (" + ", INK), ("1", GREEN)],
        "(c, b, a) = (8, 4, 1)",
    )
    expression_card(
        draw,
        (1545, 1150, 2295, 1255),
        [("-11 = ", INK), ("-8", BLUE), (" - ", INK), ("2", ORANGE), (" - ", INK), ("1", GREEN)],
        "(c, b, a) = (-8, -2, -1)",
    )


def collapse_lb2() -> Image.Image:
    w, h = 2400, 1320
    img, draw = create_canvas(w, h)
    draw_collapse_top(
        draw,
        title="Cooley-Tukey Sparse-Diagonal Collapse (n = 32)",
        subtitle="Layer folding reduces multiplicative depth while preserving diagonal-index coverage.",
        formula="DFT" + sub("32") + "ᴺᴿ = C" + sub("1-2") + " · C" + sub("3-5"),
        note="Two collapsed blocks: 7 residual shifts and 8 modulo-distinct coarse shifts.",
        mode="lb2",
    )
    draw_lb2_body(draw, include_bsgs=False)
    return img


def collapse_lb3() -> Image.Image:
    w, h = 2400, 1320
    img, draw = create_canvas(w, h)
    draw_collapse_top(
        draw,
        title="Cooley-Tukey Sparse-Diagonal Collapse (levelBudget = 3)",
        subtitle="A larger level budget splits the collapse into three sparser blocks.",
        formula="DFT" + sub("32") + "ᴺᴿ = C" + sub("1") + " · C" + sub("2-3") + " · C" + sub("4-5"),
        note="selectLayers(5, 3) yields one singleton layer and two collapsed layer groups.",
        mode="lb3",
    )
    draw_lb3_body(draw)
    return img


def draw_bsgs_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    compact: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    rounded(draw, box, radius=16, fill=PANEL, outline=BORDER, width=2)
    draw.text((x0 + 180, y0 + 50), "BSGS completion for layers 3-5", font=F["section"], fill=INK)
    draw.text((x0 + 180, y0 + 90), "A baby stencil and its +16 giant shift cover the original coarse diagonal set.", font=F["body"], fill=MUTED)
    left, right = x0 + 320, x1 - 300
    y_top = y0 + (175 if compact else 160)
    y_mid = y_top + 115
    y_bot = y_mid + 115

    rounded(draw, (x1 - 720, y0 + 42, x1 - 70, y0 + 160), radius=12, fill=WHITE, outline=FAINT, width=2)
    marker(draw, x1 - 680, y0 + 75, fill=BLUE, outline=BLUE, r=7)
    draw.text((x1 - 640, y0 + 58), "baby stencil {-12, -8, -4, 0}", font=F["small"], fill=BLUE_DARK)
    arrow(draw, (x1 - 692, y0 + 112), (x1 - 657, y0 + 112), fill=ORANGE, width=3, head=9)
    draw.text((x1 - 640, y0 + 95), "same stencil +16 -> {4, 8, 12, 16}", font=F["small"], fill=ORANGE)
    marker(draw, x1 - 680, y0 + 143, fill=GREEN, outline=GREEN, r=7)
    draw.text((x1 - 640, y0 + 126), "union = target coarse set", font=F["small"], fill=GREEN)

    def axis_row(y: int, label: str, color: tuple[int, int, int]) -> None:
        draw.text((x0 + 190, y - 19), label, font=F["body"], fill=color, anchor="ra")
        draw.line((left, y, right, y), fill=AXIS, width=2)
        for value in range(-16, 17, 4):
            x = xmap(value, left, right)
            draw.line((x, y - 15, x, y + 15), fill=AXIS, width=1)
            draw.text((x, y + 26), str(value), font=F["tiny"], fill=MUTED, anchor="ma")

    baby = {-12, -8, -4, 0}
    giant = {4, 8, 12, 16}
    target = baby | giant
    axis_row(y_top, "g = 0", BLUE_DARK)
    for value in sorted(baby):
        x = xmap(value, left, right)
        marker(draw, x, y_top, fill=BLUE, outline=BLUE, r=10)
        draw.text((x, y_top - 48), str(value), font=F["small"], fill=BLUE_DARK, anchor="ma")
    axis_row(y_mid, "g = 16", ORANGE)
    for value in sorted(giant):
        marker(draw, xmap(value, left, right), y_mid, fill=ORANGE, outline=ORANGE, r=10)
    for value in sorted(baby):
        arrow(draw, (xmap(value, left, right), y_top + 30), (xmap(value + 16, left, right), y_mid - 30), fill=ORANGE, width=2, head=12)
    axis_row(y_bot, "union", GREEN)
    for value in sorted(target):
        x = xmap(value, left, right)
        marker(draw, x, y_bot, fill=GREEN, outline=GREEN, r=10)
        draw.text((x, y_bot - 46), str(value), font=F["small"], fill=GREEN, anchor="ma")


def collapse_with_bsgs() -> Image.Image:
    w, h = 2400, 1820
    img, draw = create_canvas(w, h)
    draw_collapse_top(
        draw,
        title="Collapsed Sparse-Diagonal Factors with BSGS (n = 32)",
        subtitle="Layer collapse preserves coverage; BSGS implements the coarse layer group with shifted stencils.",
        formula="DFT" + sub("32") + "ᴺᴿ = C" + sub("1-2") + " · C" + sub("3-5"),
        note="C3-5 has eight modulo-distinct diagonals; BSGS realizes them as baby + giant shifts.",
        mode="lb2",
    )
    draw_lb2_body(draw, include_bsgs=True)
    return img


def bsgs_only() -> Image.Image:
    w, h = 2400, 720
    img, draw = create_canvas(w, h)
    title_block(
        draw,
        w,
        "Baby-Step Giant-Step Coverage for the Coarse Diagonal Set",
        "The same stencil is evaluated once at g = 0 and once after a +16 shift; their union recovers the target set.",
    )
    draw_bsgs_panel(draw, (40, 130, 2360, 680), compact=False)
    return img


def save_outputs() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("1-decompose-paper.png", decompose()),
        ("2-collapse_3-bsgs-paper.png", collapse_with_bsgs()),
        ("2-collapse-lb2-paper.png", collapse_lb2()),
        ("2-collapse-lb3-paper.png", collapse_lb3()),
        ("3-bsgs-paper.png", bsgs_only()),
    ]
    paths: list[Path] = []
    for name, img in outputs:
        path = OUT_DIR / name
        img.save(path, dpi=(600, 600), optimize=True)
        paths.append(path)
        print(f"wrote {path}  size={img.size}")
    return paths


if __name__ == "__main__":
    save_outputs()
