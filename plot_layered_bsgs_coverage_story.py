from html import escape
from pathlib import Path


RESULTS = Path("results")
FONT = "Noto Sans CJK SC, Noto Sans CJK, Source Han Sans SC, Microsoft YaHei, Inter, Arial, sans-serif"

INK = "#22313f"
MUTED = "#5d6b78"
GRID = "#d8e1e9"
PAPER = "#f7fafc"
CARD = "#ffffff"
BLUE = "#2e86ab"
BLUE_LIGHT = "#d9edf6"
TEAL = "#208b6d"
TEAL_LIGHT = "#d8f0e8"
ORANGE = "#d77a30"
ORANGE_LIGHT = "#f6dfc3"
PURPLE = "#6957a8"
PURPLE_LIGHT = "#e6e1f5"
GREEN = "#166534"
GREEN_LIGHT = "#dff3e7"
GRAY = "#6f6860"
GRAY_LIGHT = "#e5e1dc"
BROWN = "#8a4d1f"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def tag(name, attrs=None, content=None):
    attrs = attrs or {}
    body = " ".join(f'{k}="{escape(fmt(v), quote=True)}"' for k, v in attrs.items() if v is not None)
    if content is None:
        return f"<{name} {body}/>"
    return f"<{name} {body}>{content}</{name}>"


def rect(x, y, w, h, fill=CARD, stroke=GRID, sw=1.2, rx=8, opacity=1):
    return tag("rect", {"x": x, "y": y, "width": w, "height": h, "rx": rx, "fill": fill, "stroke": stroke, "stroke-width": sw, "opacity": opacity})


def line(x1, y1, x2, y2, stroke=MUTED, sw=1.2, dash=None, opacity=1):
    return tag("line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": stroke, "stroke-width": sw, "stroke-dasharray": dash, "opacity": opacity})


def path(d, stroke=MUTED, sw=1.2, fill="none", dash=None, opacity=1):
    return tag("path", {"d": d, "stroke": stroke, "stroke-width": sw, "fill": fill, "stroke-dasharray": dash, "opacity": opacity})


def circle(cx, cy, r, fill, stroke="#ffffff", sw=1.0, opacity=1):
    return tag("circle", {"cx": cx, "cy": cy, "r": r, "fill": fill, "stroke": stroke, "stroke-width": sw, "opacity": opacity})


def text(x, y, s, size=18, weight=400, fill=INK, anchor="start", style=None):
    attrs = {
        "x": x,
        "y": y,
        "font-family": FONT,
        "font-size": size,
        "font-weight": weight,
        "fill": fill,
        "text-anchor": anchor,
    }
    if style:
        attrs["style"] = style
    return tag("text", attrs, escape(str(s)))


def multiline(x, y, lines, size=18, fill=MUTED, weight=400, gap=26):
    parts = []
    for i, s in enumerate(lines):
        parts.append(text(x, y + i * gap, s, size=size, fill=fill, weight=weight))
    return parts


def arrow(parts, x1, y1, x2, y2, color=MUTED, sw=2.0, dash=None):
    parts.append(line(x1, y1, x2, y2, color, sw, dash=dash))
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 9
    p1 = (x2 - ux * size + px * size * 0.55, y2 - uy * size + py * size * 0.55)
    p2 = (x2 - ux * size - px * size * 0.55, y2 - uy * size - py * size * 0.55)
    parts.append(tag("polygon", {"points": f"{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}", "fill": color}))


def axis_x(value, lo, hi, x, w):
    return x + (value - lo) * w / (hi - lo)


def draw_bit_split(parts, x, y):
    parts.append(rect(x, y, 500, 210))
    parts.append(text(x + 24, y + 36, "1. 先按 levelBudget 分层", size=24, weight=850))
    parts += multiline(
        x + 24,
        y + 72,
        [
            "logSlots = 9，levelBudget = 3",
            "所以切成 3 层，每层处理 3 个 bit",
            "每层有 2^3 = 8 个本地 digit",
        ],
        size=18,
        fill=MUTED,
        gap=28,
    )
    bx, by = x + 42, y + 160
    groups = [("高 3 bit", "scale = 64", PURPLE, PURPLE_LIGHT), ("中 3 bit", "scale = 8", TEAL, TEAL_LIGHT), ("低 3 bit", "scale = 1", ORANGE, ORANGE_LIGHT)]
    for i, (title, scale, color, fill) in enumerate(groups):
        gx = bx + i * 140
        parts.append(rect(gx, by, 122, 34, fill, color, sw=1.6, rx=5))
        parts.append(text(gx + 61, by + 23, title, size=15, weight=800, fill=color, anchor="middle"))
        parts.append(text(gx + 61, by + 55, scale, size=14, fill=color, anchor="middle"))
        if i < 2:
            parts.append(text(gx + 131, by + 23, "+", size=20, weight=800, fill=INK, anchor="middle"))


def draw_layer_bsgs(parts, x, y):
    parts.append(rect(x, y, 1200, 405))
    parts.append(text(x + 24, y + 38, "2. 每层内部用 BSGS：一个 baby stencil + 一个 giant shift", size=25, weight=850))
    parts.append(text(x + 24, y + 70, "图中蓝色是 baby list；暗灰色是同一层由 giant shift 适配出来的 indices，不需要再列入 baby list。", size=18, fill=MUTED))

    rows = [
        ("L2 / C2S s=2 / S2C s=2", 64, "覆盖该 digit: 0,64,...,448", PURPLE),
        ("L1 / C2S s=1 / S2C s=1", 8, "覆盖该 digit: 0,8,...,56", TEAL),
        ("L0 / C2S s=0 / S2C s=0", 1, "覆盖该 digit: 0,1,...,7", ORANGE),
    ]
    axis_left = x + 365
    axis_w = 585
    for idx, (name, sf, note, color) in enumerate(rows):
        yy = y + 126 + idx * 86
        parts.append(text(x + 28, yy + 7, name, size=19, weight=850, fill=color))
        parts.append(text(x + 28, yy + 35, f"sf = {sf}，g = 8，b = 2", size=16, fill=MUTED))
        parts.append(line(axis_left, yy, axis_left + axis_w, yy, GRID, 9, opacity=0.85))
        for q in range(-7, 8):
            xx = axis_left + (q + 7) * axis_w / 14
            is_baby = q <= 0
            fill = BLUE if is_baby else GRAY
            opacity = 1.0 if is_baby else 0.48
            parts.append(circle(xx, yy, 7.5, fill, opacity=opacity))
            if q in (-7, -4, 0, 4, 7):
                parts.append(text(xx, yy + 32, str(q), size=14, fill=fill, anchor="middle"))
        parts.append(text(axis_left, yy - 24, "baby q=-7..0", size=15, weight=800, fill=BLUE))
        parts.append(text(axis_left + axis_w, yy - 24, "giant shift 后得到 q=1..7", size=15, weight=800, fill=GRAY, anchor="end"))
        parts.append(text(x + 980, yy + 7, note, size=16, weight=700, fill=color))
        parts.append(text(x + 980, yy + 35, f"digit set = {{0..7}} × {sf}", size=15, fill=MUTED))
    parts.append(text(x + 28, y + 374, "C2S 是从粗到细读这些层；S2C 是从细到粗读这些层。顺序变了，覆盖基底不变。", size=18, weight=750, fill=INK))


def draw_coverage(parts, x, y):
    parts.append(rect(x, y, 1728, 430))
    parts.append(text(x + 24, y + 42, "3. 三层组合后，正好覆盖 0..N/2-1 = 0..511", size=26, weight=850))
    parts.append(text(x + 24, y + 78, "把一个 rotation index 写成三层 digit 的和：", size=19, fill=MUTED))
    parts.append(text(x + 420, y + 78, "r = d2·64 + d1·8 + d0，  d0,d1,d2 ∈ {0,...,7}", size=22, weight=850, fill=INK))
    parts.append(text(x + 1040, y + 78, "8 × 8 × 8 = 512 个组合", size=22, weight=850, fill=GREEN))

    bar_x, bar_y, bar_w, bar_h = x + 64, y + 132, 1540, 72
    parts.append(text(bar_x, bar_y - 18, "全局 coverage ruler：每个大块由最高层 d2 选择，宽度 64", size=18, weight=800))
    for d2 in range(8):
        bx = bar_x + d2 * bar_w / 8
        fill = PURPLE_LIGHT if d2 % 2 == 0 else "#f0eef8"
        parts.append(rect(bx, bar_y, bar_w / 8, bar_h, fill, PURPLE, sw=1.0, rx=0, opacity=0.92))
        parts.append(text(bx + bar_w / 16, bar_y + 30, f"d2={d2}", size=16, weight=800, fill=PURPLE, anchor="middle"))
        parts.append(text(bx + bar_w / 16, bar_y + 56, f"{d2*64}-{d2*64+63}", size=14, fill=MUTED, anchor="middle"))
    for t in [0, 64, 128, 192, 256, 320, 384, 448, 511]:
        xx = axis_x(t, 0, 511, bar_x, bar_w)
        parts.append(line(xx, bar_y + bar_h, xx, bar_y + bar_h + 16, INK, 1.1))
        parts.append(text(xx, bar_y + bar_h + 36, str(t), size=14, fill=INK, anchor="middle"))

    # Zoom a chosen coarse block to show middle and fine layers.
    target = 427
    d2, d1, d0 = 6, 5, 3
    z1x, z1y, z1w, z1h = x + 110, y + 285, 650, 58
    z2x, z2y, z2w, z2h = x + 930, y + 285, 470, 58
    src_x1 = bar_x + d2 * bar_w / 8
    src_x2 = bar_x + (d2 + 1) * bar_w / 8
    arrow(parts, (src_x1 + src_x2) / 2, bar_y + bar_h + 54, z1x + z1w / 2, z1y - 20, PURPLE, sw=1.5, dash="4 4")

    parts.append(text(z1x, z1y - 28, "放大 d2=6 的 64 个 indices：中层 d1 把它切成 8 段", size=17, weight=800, fill=TEAL))
    for d in range(8):
        sx = z1x + d * z1w / 8
        parts.append(rect(sx, z1y, z1w / 8, z1h, TEAL_LIGHT if d != d1 else "#bce6d7", TEAL, sw=1.0, rx=0))
        parts.append(text(sx + z1w / 16, z1y + 24, f"d1={d}", size=14, weight=800, fill=TEAL, anchor="middle"))
        parts.append(text(sx + z1w / 16, z1y + 48, f"{384+d*8}-{384+d*8+7}", size=12, fill=MUTED, anchor="middle"))

    parts.append(text(z2x, z2y - 28, "再放大 d1=5 的 8 个 indices：最低层 d0 选一个点", size=17, weight=800, fill=ORANGE))
    for d in range(8):
        sx = z2x + d * z2w / 8
        fill = ORANGE_LIGHT if d != d0 else "#f2c078"
        parts.append(rect(sx, z2y, z2w / 8, z2h, fill, ORANGE, sw=1.0, rx=0))
        parts.append(text(sx + z2w / 16, z2y + 24, f"d0={d}", size=14, weight=800, fill=ORANGE, anchor="middle"))
        parts.append(text(sx + z2w / 16, z2y + 49, str(424 + d), size=15, weight=850 if d == d0 else 500, fill=INK, anchor="middle"))
    target_x = z2x + (d0 + 0.5) * z2w / 8
    parts.append(circle(target_x, z2y + z2h + 25, 13, GREEN, stroke="#ffffff", sw=2.0))
    parts.append(text(target_x, z2y + z2h + 62, "r=427", size=20, weight=850, fill=GREEN, anchor="middle"))
    parts.append(text(x + 1432, z2y + 20, "例子", size=18, weight=850, fill=INK))
    parts.append(text(x + 1432, z2y + 52, "427 = 6·64 + 5·8 + 3", size=18, weight=800, fill=GREEN))


def draw_footer(parts, x, y):
    parts.append(rect(x, y, 1728, 135, fill=GREEN_LIGHT, stroke="#a7d8bb", rx=8))
    parts.append(text(x + 24, y + 40, "直观结论", size=25, weight=850, fill=GREEN))
    parts += multiline(
        x + 24,
        y + 76,
        [
            "每层 BSGS 只负责一个 3-bit digit；baby list 是局部 stencil，暗色 indices 是由 giant shift 适配出来的同层覆盖。",
            "三层相加形成混合基坐标：最高层定位 64 宽的大块，中层定位块内 8 宽的小段，最低层定位具体 index。",
            "因此 0..511 中每个 rotation index 都有唯一的 (d2,d1,d0)，没有空洞，也不会重复覆盖。",
        ],
        size=18,
        fill=INK,
        gap=26,
    )


def render():
    width, height = 1800, 1320
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, fill=PAPER, stroke="none", sw=0, rx=0),
        text(40, 56, "分层 BSGS 如何完整覆盖 0..N/2 的 rotation indices", size=34, weight=850),
        text(40, 92, "例子：logN=10, logSlots=9, levelBudget=[3,3]，N/2 = slots = 512", size=20, fill=MUTED),
    ]
    draw_bit_split(parts, 40, 125)
    draw_layer_bsgs(parts, 560, 125)
    draw_coverage(parts, 40, 560)
    draw_footer(parts, 40, 1045)
    parts.append("</svg>")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "layered_bsgs_coverage_story_logN10_logslots9.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    render()
