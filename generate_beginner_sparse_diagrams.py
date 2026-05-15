#!/usr/bin/env python3
"""Generate beginner-friendly versions of the sparse-diagonal diagrams.

The original PNGs in "Nochangedelete copy" are kept untouched. This script
redraws the same three-step story with clearer Chinese labels and also creates
one compressed all-in-one summary image.
"""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Nochangedelete copy" / "refined_outputs"

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

PAPER = (247, 250, 252)
WHITE = (255, 255, 255)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
GRID = (226, 232, 240)
BORDER = (186, 199, 220)
BLUE = (29, 78, 216)
BLUE_DARK = (30, 64, 175)
BLUE_SOFT = (219, 234, 254)
ORANGE = (234, 88, 12)
ORANGE_SOFT = (255, 237, 213)
GREEN = (5, 150, 105)
GREEN_SOFT = (209, 250, 229)
PURPLE = (109, 40, 217)
PURPLE_SOFT = (237, 233, 254)
GRAY_SOFT = (241, 245, 249)

W, H = 1920, 1080
MOD = 32


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(FONT_MONO, size)
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


F = {
    "title": font(38, True),
    "subtitle": font(22),
    "h1": font(30, True),
    "h2": font(24, True),
    "body": font(21),
    "body_bold": font(21, True),
    "small": font(17),
    "small_bold": font(17, True),
    "tiny": font(14),
    "mono": font(20, mono=True),
    "mono_big": font(30, mono=True),
    "mono_huge": font(38, mono=True),
}


def text_w(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=ft)
    return box[2] - box[0]


def text_h(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=ft)
    return box[3] - box[1]


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    radius: int = 14,
    fill: tuple[int, int, int] = WHITE,
    outline: tuple[int, int, int] = BORDER,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    ft: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        current = ""
        for ch in raw:
            candidate = current + ch
            if current and text_w(draw, candidate, ft) > max_width:
                lines.append(current.rstrip())
                current = ch.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
    return lines or [""]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    ft: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, ft, max_width):
        draw.text((x, y), line, font=ft, fill=fill)
        y += text_h(draw, line, ft) + line_gap
    return int(y)


def draw_segments(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    segments: Sequence[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]],
) -> float:
    for part, ft, color in segments:
        draw.text((x, y), part, font=ft, fill=color)
        x += text_w(draw, part, ft)
    return x


def centered_segments(
    draw: ImageDraw.ImageDraw,
    cx: float,
    y: float,
    segments: Sequence[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]],
) -> None:
    width = sum(text_w(draw, part, ft) for part, ft, _ in segments)
    draw_segments(draw, cx - width / 2, y, segments)


def new_canvas(width: int = W, height: int = H) -> Image.Image:
    return Image.new("RGB", (width, height), PAPER)


def title_bar(draw: ImageDraw.ImageDraw, step: str, title: str, subtitle: str) -> None:
    draw.text((54, 36), step, font=F["h2"], fill=BLUE_DARK)
    draw.text((145, 29), title, font=F["title"], fill=INK)
    draw.text((147, 82), subtitle, font=F["subtitle"], fill=MUTED)
    draw.line((54, 124, W - 54, 124), fill=BORDER, width=2)


def xmap(value: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (value - lo) * (x1 - x0) / (hi - lo)


def signed32(value: int) -> int:
    residue = value % MOD
    if residue > MOD // 2:
        return residue - MOD
    return residue


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    fill: tuple[int, int, int],
    width: int = 3,
    head: int = 12,
) -> None:
    draw.line((*p0, *p1), fill=fill, width=width)
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    draw.polygon(
        [
            (x1, y1),
            (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55),
            (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55),
        ],
        fill=fill,
    )


def draw_pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    color: tuple[int, int, int],
) -> float:
    ft = F["small_bold"]
    width = text_w(draw, label, ft) + 34
    rounded(draw, (x, y, x + width, y + 38), radius=9, fill=fill, outline=outline, width=2)
    draw.text((x + width / 2, y + 19), label, font=ft, fill=color, anchor="mm")
    return x + width + 14


def draw_dft_icon(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 218) -> None:
    draw.text((x + size / 2, y - 35), "DFT_32^NR", font=F["h2"], fill=INK, anchor="ma")
    draw.rectangle((x, y, x + size, y + size), fill=(251, 253, 255), outline=MUTED, width=2)
    for i in range(1, 8):
        gx = x + i * size / 8
        gy = y + i * size / 8
        draw.line((gx, y, gx, y + size), fill=GRID, width=1)
        draw.line((x, gy, x + size, gy), fill=GRID, width=1)

    rows = [
        ["1", "1", "1", "...", "1"],
        ["1", "w", "w^2", "...", "w^31"],
        ["1", "w^2", "w^4", "...", "w^62"],
        ["...", "...", "...", "...", "..."],
        ["1", "w^31", "w^62", "...", "w^961"],
    ]
    cols = [x + 45, x + 84, x + 123, x + 162, x + 198]
    ys = [y + 40, y + 78, y + 116, y + 154, y + 192]
    for yy, row in zip(ys, rows):
        for xx, item in zip(cols, row):
            draw.text((xx, yy), item, font=F["tiny"], fill=BLUE_DARK, anchor="mm")

    draw.text((x + size / 2, y + size + 28), "32 x 32 dense DFT", font=F["small_bold"], fill=BLUE_DARK, anchor="ma")
    draw.text((x + size / 2, y + size + 54), "目标：用少量循环对角线表示", font=F["small"], fill=MUTED, anchor="ma")


def diag_endpoints(x: int, y: int, size: int, offset_px: float) -> tuple[tuple[float, float], tuple[float, float]]:
    if offset_px >= 0:
        return (x + offset_px, y), (x + size, y + size - offset_px)
    d = -offset_px
    return (x, y + d), (x + size - d, y + size)


def draw_diag_square(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int,
    offsets: Sequence[int],
    label_top: str,
    label_mid: str,
    label_bottom: str,
    range_max: int = 16,
    dense: bool = False,
) -> None:
    draw.text((x + size / 2, y - 32), label_top, font=F["h2"], fill=INK, anchor="ma")
    draw.rectangle((x, y, x + size, y + size), fill=WHITE, outline=MUTED, width=1)
    grid_n = 24 if dense else 18
    for i in range(1, grid_n):
        gx = x + i * size / grid_n
        gy = y + i * size / grid_n
        draw.line((gx, y, gx, y + size), fill=GRID, width=1)
        draw.line((x, gy, x + size, gy), fill=GRID, width=1)
    for offset in offsets:
        if offset == 0:
            color = BLUE
            width = 5 if not dense else 4
        elif offset > 0:
            color = ORANGE
            width = 3
        else:
            color = GREEN
            width = 3
        delta = offset * size / (2 * range_max)
        p0, p1 = diag_endpoints(x, y, size, delta)
        draw.line((*p0, *p1), fill=color, width=width)

    draw.text((x + size / 2, y + size + 13), label_mid, font=F["small_bold"], fill=BLUE_DARK, anchor="ma")
    draw.text((x + size / 2, y + size + 42), label_bottom, font=F["small"], fill=MUTED, anchor="ma")


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    rounded(draw, (x, y, x + 285, y + 126), radius=12, fill=WHITE, outline=BORDER, width=1)
    rows = [
        (BLUE, "主对角线：offset = 0"),
        (ORANGE, "正偏移：+ offset"),
        (GREEN, "负偏移：- offset"),
    ]
    for i, (color, label) in enumerate(rows):
        yy = y + 28 + i * 35
        draw.rectangle((x + 25, yy - 11, x + 50, yy + 14), fill=color)
        draw.text((x + 70, yy), label, font=F["small"], fill=INK, anchor="lm")


def draw_axis(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    label: str,
    values: Iterable[int],
    color: tuple[int, int, int],
    old_values: Iterable[int] | None = None,
    new_color: tuple[int, int, int] = ORANGE,
    label_color: tuple[int, int, int] = INK,
    show_all_ticks: bool = False,
    value_labels: bool = False,
) -> None:
    draw.text((x0 - 26, y), label, font=F["small_bold"], fill=label_color, anchor="rm")
    draw.line((x0, y, x1, y), fill=(51, 65, 85), width=2)
    tick_step = 1 if show_all_ticks else 4
    for tick in range(-16, 17, tick_step):
        xx = xmap(tick, -16, 16, x0, x1)
        length = 11 if tick % 4 == 0 else 6
        draw.line((xx, y - length, xx, y + length), fill=(51, 65, 85), width=1)
        if tick % 4 == 0:
            draw.text((xx, y + 24), str(tick), font=F["tiny"], fill=MUTED, anchor="mm")

    old = set(old_values or [])
    vals = sorted(set(values))
    for value in vals:
        xx = xmap(value, -16, 16, x0, x1)
        fill = color if value in old or old_values is None else new_color
        draw.ellipse((xx - 8, y - 8, xx + 8, y + 8), fill=fill, outline=WHITE, width=2)
        if value_labels:
            draw.text((xx, y - 27), str(value), font=F["tiny"], fill=fill, anchor="mm")


def sums_for(scales: Sequence[int]) -> set[int]:
    out: set[int] = set()
    for coeffs in product((-1, 0, 1), repeat=len(scales)):
        value = sum(c * s for c, s in zip(coeffs, scales))
        if -16 <= value <= 16:
            out.add(value)
    return out


def draw_examples_box_decompose(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    rounded(draw, (x, y, x + w, y + h), radius=14, fill=WHITE, outline=BLUE, width=2)
    draw.text((x + 24, y + 26), "怎么读这个分解？", font=F["h2"], fill=INK)
    draw_wrapped(
        draw,
        (x + 24, y + 68),
        "每一层只能选择三种情况：负偏移、0、正偏移。五层相加后，就是最终要访问的 cyclic diagonal。",
        F["body"],
        MUTED,
        w - 48,
        7,
    )
    draw.text((x + 24, y + 142), "通用写法：", font=F["body_bold"], fill=INK)
    draw.text(
        (x + 24, y + 180),
        "s = e16*16 + e8*8 + e4*4",
        font=F["mono"],
        fill=BLUE_DARK,
    )
    draw.text(
        (x + 24, y + 212),
        "    + e2*2 + e1*1",
        font=F["mono"],
        fill=BLUE_DARK,
    )
    draw.text(
        (x + 24, y + 238),
        "每个 e 只能取 -1、0、+1",
        font=F["small"],
        fill=MUTED,
    )
    rows = [
        [("13", INK), (" = ", INK), ("8", ORANGE), (" + ", INK), ("4", GREEN), (" + ", INK), ("1", BLUE)],
        [("-11", INK), (" = ", INK), ("-8", ORANGE), (" - ", INK), ("2", GREEN), (" - ", INK), ("1", BLUE)],
        [("6", INK), (" = ", INK), ("4", GREEN), (" + ", INK), ("2", ORANGE)],
    ]
    yy = y + 252
    for row in rows:
        rounded(draw, (x + 24, yy, x + w - 24, yy + 44), radius=8, fill=(248, 251, 255), outline=BORDER, width=1)
        segs = [(part, F["mono_big"], color) for part, color in row]
        draw_segments(draw, x + 46, yy + 3, segs)
        yy += 52


def draw_coverage_panel_decompose(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    rounded(draw, (x, y, x + w, y + h), radius=14, fill=WHITE, outline=BLUE, width=2)
    draw.rectangle((x, y, x + w, y + 46), fill=BLUE_DARK)
    draw.text((x + 22, y + 23), "为什么这些尺度能覆盖全部 offset？", font=F["body_bold"], fill=WHITE, anchor="lm")
    axis_x0, axis_x1 = x + 170, x + w - 45
    draw.text((x + w - 300, y + 62), "蓝色：已覆盖    橙色：本层新增", font=F["small"], fill=MUTED)
    scales_so_far: list[int] = []
    old: set[int] = set()
    for idx, scale in enumerate([16, 8, 4, 2, 1]):
        scales_so_far.append(scale)
        current = sums_for(scales_so_far)
        yy = y + 94 + idx * 62
        label = f"加入 ±{scale}"
        draw_axis(draw, axis_x0, axis_x1, yy, label, current, BLUE, old, show_all_ticks=scale <= 2)
        old = current


def draw_decompose() -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    title_bar(draw, "1 / 3", "逐层分解：每层只负责一个尺度", "从大步到小步：±16、±8、±4、±2、±1 逐层相加，最后覆盖全部 signed offset。")

    draw_dft_icon(draw, 70, 210, 220)
    draw.text((330, 314), "=", font=F["mono_huge"], fill=INK)

    rounded(draw, (480, 148, 1440, 232), radius=12, fill=WHITE, outline=BLUE, width=2)
    centered_segments(
        draw,
        960,
        166,
        [
            ("DFT_32^NR = ", F["mono_big"], INK),
            ("D32", F["mono_big"], BLUE_DARK),
            (" · D16 · D8 · D4 · D2", F["mono_big"], INK),
        ],
    )
    draw.text((960, 207), "五个因子都很稀疏：每个因子只有主对角线和一对正负偏移对角线。", font=F["small_bold"], fill=BLUE_DARK, anchor="ma")
    draw_legend(draw, 1565, 145)

    factors = [
        ("D32", 16),
        ("D16", 8),
        ("D8", 4),
        ("D4", 2),
        ("D2", 1),
    ]
    start_x, step_x, y0 = 455, 265, 300
    for i, (name, offset) in enumerate(factors):
        xx = start_x + i * step_x
        draw_diag_square(
            draw,
            xx,
            y0,
            185,
            [-offset, 0, offset],
            name,
            f"只允许 0、±{offset}",
            "乘法时把偏移继续相加",
            range_max=16,
        )
        if i < len(factors) - 1:
            draw.text((xx + 214, y0 + 80), "×", font=F["mono_huge"], fill=INK, anchor="mm")

    draw_coverage_panel_decompose(draw, 54, 620, 1105, 400)
    draw_examples_box_decompose(draw, 1188, 620, 678, 400)
    rounded(draw, (315, 1032, 1605, 1070), radius=10, fill=WHITE, outline=GREEN, width=2)
    draw.text(
        (960, 1051),
        "结论：offset 等于各层偏移之和；这些二进制尺度让稀疏因子相乘后能覆盖整个范围。",
        font=F["body_bold"],
        fill=GREEN,
        anchor="mm",
    )
    return img


def draw_examples_box_collapse(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    rounded(draw, (x, y, x + w, y + h), radius=14, fill=WHITE, outline=BLUE, width=2)
    draw.text((x + 24, y + 28), "折叠后的读法", font=F["h2"], fill=INK)
    draw_segments(
        draw,
        x + 24,
        y + 78,
        [
            ("s ≡ b + a  (mod 32)", F["mono_big"], BLUE_DARK),
        ],
    )
    draw.text((x + 24, y + 128), "B 先定位到 4 的倍数中心；A 再补中心附近的 ±3 残差。", font=F["body"], fill=MUTED)
    draw.text((x + 24, y + 168), "A = {-3,-2,-1,0,1,2,3}", font=F["body_bold"], fill=ORANGE)
    draw.text((x + 24, y + 200), "B = {-12,-8,-4,0,4,8,12,16}", font=F["body_bold"], fill=BLUE_DARK)
    rows = [
        [("13", INK), (" = ", INK), ("12", BLUE_DARK), (" + ", INK), ("1", ORANGE)],
        [("-11", INK), (" = ", INK), ("-12", BLUE_DARK), (" + ", INK), ("1", ORANGE)],
        [("6", INK), (" = ", INK), ("4", BLUE_DARK), (" + ", INK), ("2", ORANGE)],
    ]
    yy = y + 226
    for row in rows:
        rounded(draw, (x + 24, yy, x + w - 24, yy + 48), radius=8, fill=(248, 251, 255), outline=BORDER, width=1)
        segs = [(part, F["mono_big"], color) for part, color in row]
        draw_segments(draw, x + 46, yy + 5, segs)
        yy += 62


def draw_coverage_panel_collapse(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    rounded(draw, (x, y, x + w, y + h), radius=14, fill=WHITE, outline=BLUE, width=2)
    draw.rectangle((x, y, x + w, y + 48), fill=BLUE_DARK)
    draw.text((x + 22, y + 24), "折叠后为什么仍能覆盖？", font=F["body_bold"], fill=WHITE, anchor="lm")
    axis_x0, axis_x1 = x + 175, x + w - 58
    b_centers = [-12, -8, -4, 0, 4, 8, 12, 16]
    all_offsets = set(range(-16, 17))
    draw_axis(draw, axis_x0, axis_x1, y + 112, "B 中心", b_centers, BLUE, value_labels=True)
    draw.text((axis_x0, y + 72), "C_{3-5}：每隔 4 一个中心", font=F["small_bold"], fill=BLUE_DARK)
    draw_axis(draw, axis_x0, axis_x1, y + 236, "B + A", all_offsets, GREEN, b_centers, show_all_ticks=True)
    draw.text((axis_x0, y + 192), "C_{1-2}：给每个中心补上 -3 到 +3 的残差", font=F["small_bold"], fill=ORANGE)
    draw_wrapped(
        draw,
        (x + 28, y + h - 76),
        "折叠的代价是每个块的 diagonal 更多；收益是乘法层数从 5 层减少到 2 个块。",
        F["body_bold"],
        GREEN,
        w - 56,
    )


def draw_collapse() -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    title_bar(draw, "2 / 3", "折叠：把五层合成两个块", "把相邻尺度先合并：小尺度块 A 负责细残差，大尺度块 B 负责粗中心。")

    draw_dft_icon(draw, 70, 206, 220)
    draw.text((330, 310), "=", font=F["mono_huge"], fill=INK)

    rounded(draw, (520, 148, 1400, 232), radius=12, fill=WHITE, outline=BLUE, width=2)
    centered_segments(
        draw,
        960,
        166,
        [
            ("DFT_32^NR = ", F["mono_big"], INK),
            ("C_{1-2}", F["mono_big"], ORANGE),
            (" · ", F["mono_big"], INK),
            ("C_{3-5}", F["mono_big"], BLUE_DARK),
        ],
    )
    draw.text((960, 207), "层 1-2 形成 7 个残差；层 3-5 形成 8 个 4 的倍数中心。", font=F["small_bold"], fill=BLUE_DARK, anchor="ma")

    draw_diag_square(
        draw,
        540,
        310,
        230,
        [-3, -2, -1, 0, 1, 2, 3],
        "C_{1-2}",
        "A：7 条 residual diagonal",
        "A = {-3,-2,-1,0,1,2,3}",
        range_max=6,
        dense=True,
    )
    draw.text((860, 420), "×", font=F["mono_huge"], fill=INK, anchor="mm")
    draw_diag_square(
        draw,
        965,
        310,
        230,
        [-12, -8, -4, 0, 4, 8, 12, 16],
        "C_{3-5}",
        "B：8 条 coarse diagonal",
        "B = {-12,-8,-4,0,4,8,12,16}",
        range_max=18,
        dense=True,
    )
    rounded(draw, (1265, 345, 1780, 505), radius=12, fill=WHITE, outline=BORDER, width=1)
    draw.text((1290, 378), "折叠减少乘法层数", font=F["body_bold"], fill=BLUE_DARK)
    draw.text((1290, 418), "但每个块会更稠密", font=F["body"], fill=MUTED)
    draw.text((1290, 458), "理解关键：", font=F["body_bold"], fill=INK)
    draw.text((1395, 458), "粗中心 + 细残差", font=F["body_bold"], fill=GREEN)

    draw_coverage_panel_collapse(draw, 54, 620, 1105, 400)
    draw_examples_box_collapse(draw, 1188, 620, 678, 400)
    rounded(draw, (330, 1032, 1590, 1070), radius=10, fill=WHITE, outline=GREEN, width=2)
    draw.text(
        (960, 1051),
        "结论：C_{3-5} 给 4 的倍数中心，C_{1-2} 给 ±3 残差；两者相加仍覆盖全部 offset。",
        font=F["body_bold"],
        fill=GREEN,
        anchor="mm",
    )
    return img


def draw_bsgs_axis(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    label: str,
    values: Sequence[int],
    color: tuple[int, int, int],
    label_color: tuple[int, int, int],
    show_labels: bool = True,
) -> None:
    draw.text((x0 - 28, y), label, font=F["body_bold"], fill=label_color, anchor="rm")
    draw.line((x0, y, x1, y), fill=(51, 65, 85), width=2)
    for tick in range(-16, 17, 4):
        xx = xmap(tick, -16, 16, x0, x1)
        draw.line((xx, y - 12, xx, y + 12), fill=(51, 65, 85), width=1)
        draw.text((xx, y + 32), str(tick), font=F["small"], fill=MUTED, anchor="mm")
    for value in values:
        xx = xmap(value, -16, 16, x0, x1)
        draw.ellipse((xx - 10, y - 10, xx + 10, y + 10), fill=color, outline=WHITE, width=2)
        if show_labels:
            draw.text((xx, y - 34), str(value), font=F["small_bold"], fill=color, anchor="mm")


def draw_bsgs_story_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    rounded(draw, (x, y, x + w, y + h), radius=14, fill=WHITE, outline=BORDER, width=1)
    draw.text((x + 28, y + 30), "BSGS 在这里做了什么？", font=F["h2"], fill=INK)
    items = [
        (BLUE, "Baby step", "先保存一小组 stencil：{-12,-8,-4,0}。"),
        (ORANGE, "Giant step", "把同一组整体平移 +16，得到 {4,8,12,16}。"),
        (GREEN, "Union", "两组并起来，正好得到折叠块 C_{3-5} 需要的 8 个中心。"),
    ]
    yy = y + 88
    for color, head, body in items:
        draw.ellipse((x + 30, yy + 5, x + 50, yy + 25), fill=color, outline=WHITE, width=2)
        end_x = draw_segments(draw, x + 66, yy - 2, [(head + "：", F["body_bold"], color)])
        draw_wrapped(draw, (end_x + 6, yy - 2), body, F["body"], MUTED, w - int(end_x - x) - 50, 4)
        yy += 62
    rounded(draw, (x + 28, y + h - 82, x + w - 28, y + h - 28), radius=8, fill=GRAY_SOFT, outline=BORDER, width=1)
    draw.text((x + 48, y + h - 55), "和第 2 张的关系：BSGS 不是改变目标集合，而是用更少的复用结构生成 B。", font=F["small_bold"], fill=INK, anchor="lm")


def draw_bsgs() -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    title_bar(draw, "3 / 3", "BSGS：复用同一个 baby stencil 生成大步集合", "第 2 张里的 B 集合可以拆成两行：一行原始 baby stencil，一行加上 giant shift = 16。")

    # Main BSGS card.
    rounded(draw, (54, 160, 1866, 625), radius=18, fill=WHITE, outline=BORDER, width=2)
    x0, x1 = 270, 1690
    y_baby, y_giant, y_union = 270, 410, 550
    baby = [-12, -8, -4, 0]
    shifted = [v + 16 for v in baby]
    union = sorted(set(baby + shifted))
    draw_bsgs_axis(draw, x0, x1, y_baby, "g = 0", baby, BLUE, BLUE_DARK)
    draw_bsgs_axis(draw, x0, x1, y_giant, "g = 16", shifted, ORANGE, ORANGE)
    draw_bsgs_axis(draw, x0, x1, y_union, "union", union, GREEN, GREEN)
    for src, dst in zip(baby, shifted):
        draw_arrow(
            draw,
            (xmap(src, -16, 16, x0, x1), y_baby + 28),
            (xmap(dst, -16, 16, x0, x1), y_giant - 22),
            ORANGE,
            width=2,
            head=10,
        )

    rounded(draw, (1180, 184, 1788, 318), radius=12, fill=(250, 253, 255), outline=BORDER, width=1)
    draw.text((1210, 222), "蓝：baby stencil {-12,-8,-4,0}", font=F["body_bold"], fill=BLUE_DARK)
    draw.text((1210, 260), "橙：同一 stencil +16 -> {4,8,12,16}", font=F["body_bold"], fill=ORANGE)
    draw.text((1210, 298), "绿：两行并集 = C_{3-5} 的 B", font=F["body_bold"], fill=GREEN)

    draw_bsgs_story_card(draw, 54, 675, 850, 345)

    rounded(draw, (955, 675, 1866, 345 + 675), radius=14, fill=WHITE, outline=BLUE, width=2)
    draw.text((985, 710), "三张图的关系", font=F["h2"], fill=INK)
    y = 775
    steps = [
        (BLUE_DARK, "1. 分解", "五个尺度因子：±16、±8、±4、±2、±1。"),
        (ORANGE, "2. 折叠", "把小尺度合成 A，把大尺度合成 B，仍然覆盖全部 offset。"),
        (GREEN, "3. BSGS", "对 B 做 baby + giant 复用，减少实际需要维护的结构。"),
    ]
    for color, head, body in steps:
        draw.ellipse((985, y + 5, 1010, y + 30), fill=color)
        draw.text((1028, y), head, font=F["body_bold"], fill=color)
        draw_wrapped(draw, (1142, y), body, F["body"], MUTED, 650, 6)
        y += 80
    rounded(draw, (985, 975, 1810, 1022), radius=8, fill=GREEN_SOFT, outline=GREEN, width=1)
    draw.text((1398, 999), "核心：先证明覆盖，再考虑如何折叠和复用。", font=F["body_bold"], fill=GREEN, anchor="mm")
    return img


def compact_diag_strip(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    for i, (label, offset) in enumerate([("D32", 16), ("D16", 8), ("D8", 4), ("D4", 2), ("D2", 1)]):
        xx = x + i * 170
        draw_diag_square(draw, xx, y, 120, [-offset, 0, offset], label, f"0, ±{offset}", "", range_max=16)
        if i < 4:
            draw.text((xx + 144, y + 48), "×", font=F["mono_big"], fill=INK, anchor="mm")


def draw_mini_axis_points(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    values: Sequence[int],
    color: tuple[int, int, int],
    label: str,
    show_labels: bool = False,
) -> None:
    draw.text((x0 - 22, y), label, font=F["small_bold"], fill=color, anchor="rm")
    draw.line((x0, y, x1, y), fill=(51, 65, 85), width=2)
    for tick in range(-16, 17, 4):
        xx = xmap(tick, -16, 16, x0, x1)
        draw.line((xx, y - 9, xx, y + 9), fill=(51, 65, 85), width=1)
        draw.text((xx, y + 24), str(tick), font=F["tiny"], fill=MUTED, anchor="mm")
    for value in values:
        xx = xmap(value, -16, 16, x0, x1)
        draw.ellipse((xx - 7, y - 7, xx + 7, y + 7), fill=color, outline=WHITE, width=2)
        if show_labels:
            draw.text((xx, y - 25), str(value), font=F["tiny"], fill=color, anchor="mm")


def draw_all_in_one() -> Image.Image:
    width, height = 1920, 2350
    img = new_canvas(width, height)
    draw = ImageDraw.Draw(img)
    draw.text((70, 42), "n = 32 的 sparse-diagonal：从分解到折叠再到 BSGS", font=F["title"], fill=INK)
    draw.text((72, 96), "一张图读法：先看 offset 如何覆盖，再看如何减少层数，最后看 BSGS 如何复用大步集合。", font=F["subtitle"], fill=MUTED)
    draw.line((70, 140, width - 70, 140), fill=BORDER, width=2)

    # Panel 1.
    p1 = (70, 180, width - 70, 775)
    rounded(draw, p1, radius=18, fill=WHITE, outline=BORDER, width=2)
    draw.text((100, 215), "1. 逐层分解：每层提供一个尺度", font=F["h1"], fill=BLUE_DARK)
    draw.text((100, 258), "DFT_32^NR = D32 · D16 · D8 · D4 · D2", font=F["mono_big"], fill=INK)
    compact_diag_strip(draw, 125, 335)
    draw_wrapped(
        draw,
        (1080, 320),
        "五层分别贡献 ±16、±8、±4、±2、±1。乘起来时，最终 offset 是各层选择的偏移之和。",
        F["body"],
        MUTED,
        650,
    )
    scales = [16, 8, 4, 2, 1]
    current = sums_for(scales)
    draw_mini_axis_points(draw, 250, 1660, 700, sorted(current), GREEN, "覆盖结果")
    draw.text((960, 745), "到 ±1 后，[-16,16] 内的整数位置都可由这些尺度相加得到。", font=F["small_bold"], fill=GREEN, anchor="mm")

    # Panel 2.
    p2 = (70, 825, width - 70, 1475)
    rounded(draw, p2, radius=18, fill=WHITE, outline=BORDER, width=2)
    draw.text((100, 860), "2. 折叠：五层合成两个块", font=F["h1"], fill=ORANGE)
    draw.text((100, 905), "DFT_32^NR = C_{1-2} · C_{3-5}", font=F["mono_big"], fill=INK)
    draw_diag_square(draw, 180, 990, 200, [-3, -2, -1, 0, 1, 2, 3], "C_{1-2}", "A = {-3..3}", "细残差", range_max=6, dense=True)
    draw.text((455, 1085), "×", font=F["mono_huge"], fill=INK, anchor="mm")
    draw_diag_square(draw, 560, 990, 200, [-12, -8, -4, 0, 4, 8, 12, 16], "C_{3-5}", "B = 4 的倍数中心", "粗中心", range_max=18, dense=True)
    draw_wrapped(
        draw,
        (850, 1000),
        "折叠后的规则更简单：s ≡ b + a (mod 32)。B 先选中心，A 再补中心两侧的残差。",
        F["body"],
        MUTED,
        800,
    )
    draw_mini_axis_points(draw, 320, 1630, 1355, [-12, -8, -4, 0, 4, 8, 12, 16], BLUE, "B")
    draw_mini_axis_points(draw, 320, 1630, 1430, list(range(-16, 17)), GREEN, "B + A")
    draw.text((960, 1459), "层数减少，但每个块更密；覆盖能力保持不变。", font=F["small_bold"], fill=GREEN, anchor="mm")

    # Panel 3.
    p3 = (70, 1525, width - 70, 2265)
    rounded(draw, p3, radius=18, fill=WHITE, outline=BORDER, width=2)
    draw.text((100, 1560), "3. BSGS：用 baby + giant 复用 B", font=F["h1"], fill=GREEN)
    draw_wrapped(
        draw,
        (100, 1605),
        "第 2 步里的 B 可以拆成两行：一行是 baby stencil，另一行是同一 stencil 整体加 16。",
        F["body"],
        MUTED,
        1650,
    )
    x0, x1 = 310, 1680
    y_b, y_g, y_u = 1740, 1890, 2040
    baby = [-12, -8, -4, 0]
    shifted = [4, 8, 12, 16]
    union = [-12, -8, -4, 0, 4, 8, 12, 16]
    draw_bsgs_axis(draw, x0, x1, y_b, "baby", baby, BLUE, BLUE_DARK)
    draw_bsgs_axis(draw, x0, x1, y_g, "+16", shifted, ORANGE, ORANGE)
    draw_bsgs_axis(draw, x0, x1, y_u, "union", union, GREEN, GREEN)
    for src, dst in zip(baby, shifted):
        draw_arrow(draw, (xmap(src, -16, 16, x0, x1), y_b + 28), (xmap(dst, -16, 16, x0, x1), y_g - 22), ORANGE, width=2, head=10)
    rounded(draw, (280, 2190, 1640, 2248), radius=10, fill=GREEN_SOFT, outline=GREEN, width=1)
    draw.text((960, 2220), "总关系：分解证明覆盖；折叠减少层数；BSGS 对折叠后的大步集合做结构复用。", font=F["body_bold"], fill=GREEN, anchor="mm")
    return img


def save_all() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (draw_decompose(), OUT_DIR / "1-decompose-refined.png"),
        (draw_collapse(), OUT_DIR / "2-collapse-refined.png"),
        (draw_bsgs(), OUT_DIR / "3-bsgs-refined.png"),
        (draw_all_in_one(), OUT_DIR / "fft_sparse_diagonal_all_in_one.png"),
    ]
    paths: list[Path] = []
    for img, path in outputs:
        img.save(path, "PNG", optimize=True)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for saved in save_all():
        print(saved)
