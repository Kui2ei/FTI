from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


OUT = Path("results/openfhe_rot_indices_selection_cn.png")

SCALE = 2
W, H = 1850, 1280

BG = "#f5f7fb"
CARD = "#ffffff"
INK = "#203040"
MUTED = "#5f7182"
GRID = "#d8e1ea"
BLUE = "#1f84b7"
BLUE_LIGHT = "#dceff7"
ORANGE = "#d97721"
ORANGE_LIGHT = "#fae4cf"
GREEN = "#0b8a5b"
GREEN_LIGHT = "#dff4ea"
RED = "#c23b3b"
GRAY = "#8795a1"
GRAY_LIGHT = "#eef2f6"

FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_CJK_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_MONO = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"
FONT_MONO_BOLD = "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf"


def font(size, bold=False, mono=False):
    path = FONT_MONO_BOLD if mono and bold else FONT_MONO if mono else FONT_CJK_BOLD if bold else FONT_CJK
    return ImageFont.truetype(path, size * SCALE)


F = {
    "title": font(36, bold=True),
    "subtitle": font(20),
    "panel": font(25, bold=True),
    "body": font(18),
    "body_bold": font(18, bold=True),
    "small": font(15),
    "small_bold": font(15, bold=True),
    "mono": font(16, mono=True),
    "mono_bold": font(16, bold=True, mono=True),
    "mono_small": font(14, mono=True),
}


def xy(v):
    return tuple(int(round(a * SCALE)) for a in v)


def rect(v):
    return tuple(int(round(a * SCALE)) for a in v)


def text(draw, pos, s, f, fill=INK, anchor=None):
    draw.text(xy(pos), s, font=f, fill=fill, anchor=anchor)


def line(draw, points, fill=INK, width=1):
    draw.line([xy(p) for p in points], fill=fill, width=width * SCALE)


def card(draw, x, y, w, h, r=14):
    draw.rounded_rectangle(rect((x, y, x + w, y + h)), radius=r * SCALE, fill=CARD, outline=GRID, width=2 * SCALE)


def chip(draw, x, y, w, h, label, fill, outline, txt=INK, r=8, f=None):
    draw.rounded_rectangle(rect((x, y, x + w, y + h)), radius=r * SCALE, fill=fill, outline=outline, width=2 * SCALE)
    text(draw, (x + w / 2, y + h / 2 - 1), label, f or F["small_bold"], txt, anchor="mm")


def reduce_rotation(index, slots):
    if slots & (slots - 1) == 0:
        n = slots.bit_length() - 1
        if index >= 0:
            return index - ((index >> n) << n)
        return index + slots + ((abs(index) >> n) << n)
    return (slots + index % slots) % slots


def compact(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


def compact_middle(values, head=4, tail=3):
    if len(values) <= head + tail + 1:
        return compact(values)
    return "[" + ", ".join(str(v) for v in values[:head]) + ", ..., " + ", ".join(str(v) for v in values[-tail:]) + "]"


def draw_top(draw):
    text(draw, (54, 48), "OpenFHE bootstrapping 的 rotation indices 选择逻辑", F["title"])
    text(
        draw,
        (54, 91),
        "示例来自文件末尾：RotInOPENFHE(logN=7, logSlots=6)，默认 levelBudget=[2,2], dim1=[0,0]",
        F["subtitle"],
        MUTED,
    )
    chip(draw, 1518, 45, 276, 38, "示意图，不是流程图", GRAY_LIGHT, "#bfccd8", MUTED, r=12, f=F["small_bold"])


def draw_layer_card(draw):
    x, y, w, h = 54, 132, 760, 250
    card(draw, x, y, w, h)
    text(draw, (x + 26, y + 35), "1. 先把 FFT layers 折叠成 level", F["panel"])
    text(draw, (x + 26, y + 68), "logSlots=6 表示 6 层 FFT；levelBudget=2，所以每个 level 折叠 3 层。", F["small"], MUTED)

    lx, ly = x + 38, y + 112
    bw, bh, gap = 78, 42, 9
    for i in range(6):
        fill = GREEN_LIGHT if i < 3 else ORANGE_LIGHT
        outline = GREEN if i < 3 else ORANGE
        chip(draw, lx + i * (bw + gap), ly, bw, bh, f"L{i}", fill, outline)

    chip(draw, lx, ly + 76, 3 * bw + 2 * gap, 34, "sf = 2^0 = 1", GREEN_LIGHT, GREEN, GREEN, r=7, f=F["small_bold"])
    chip(
        draw,
        lx + 3 * (bw + gap),
        ly + 76,
        3 * bw + 2 * gap,
        34,
        "sf = 2^3 = 8",
        ORANGE_LIGHT,
        ORANGE,
        ORANGE,
        r=7,
        f=F["small_bold"],
    )

    text(draw, (x + 572, ly + 6), "selectLayers:", F["small_bold"])
    text(draw, (x + 572, ly + 36), "layersCollapse=3", F["mono"], INK)
    text(draw, (x + 572, ly + 67), "rows=2, rem=0", F["mono"], INK)
    text(draw, (x + 572, ly + 86), "C2S: sf=8 -> 1", F["small"], MUTED)
    text(draw, (x + 572, ly + 111), "S2C: sf=1 -> 8", F["small"], MUTED)


def draw_param_card(draw):
    x, y, w, h = 842, 132, 954, 250
    card(draw, x, y, w, h)
    text(draw, (x + 26, y + 35), "2. 每个 collapsed level 只需要覆盖一个本地 q 区间", F["panel"])
    text(draw, (x + 26, y + 68), "layersCollapse=3 时，代码令 numRotations=2^(3+1)-1=15。", F["small"], MUTED)

    base_y = y + 112
    text(draw, (x + 34, base_y), "本地目标 q:", F["body_bold"])
    ax, ay = x + 158, base_y + 13
    cell = 42
    for idx, q in enumerate(range(-7, 8)):
        cx = ax + idx * cell
        line(draw, [(cx, ay), (cx, ay + 13)], GRAY, 1)
        text(draw, (cx, ay + 36), str(q), F["mono_small"], INK, anchor="mm")
    line(draw, [(ax, ay), (ax + 14 * cell, ay)], GRAY, 1)
    chip(draw, x + 32, y + 178, 244, 38, "halfRots = 1 - 8 = -7", GRAY_LIGHT, "#c6d2de", MUTED, r=8, f=F["mono_small"])
    chip(draw, x + 304, y + 178, 230, 38, "q = -7 ... 7", BLUE_LIGHT, BLUE, BLUE, r=8, f=F["mono_small"])
    chip(draw, x + 562, y + 178, 292, 38, "接下来用 BSGS 覆盖它", ORANGE_LIGHT, ORANGE, ORANGE, r=8, f=F["small_bold"])


def draw_bsgs_card(draw):
    x, y, w, h = 54, 414, 1742, 392
    card(draw, x, y, w, h)
    text(draw, (x + 26, y + 35), "3. BSGS：用 baby 小步 + giant 大步覆盖 q，而不是保存每个 q 的 key", F["panel"])
    text(draw, (x + 26, y + 68), "代码保存两类 rotation key：baby list 和 giant list；运行时的组合 q = q_b + q_g 覆盖目标区间。", F["small"], MUTED)

    grid_x, grid_y = x + 94, y + 150
    cw, ch = 76, 52
    baby_q = list(range(-7, 1))
    giant_q = [0, 8]

    text(draw, (grid_x + 4 * cw, grid_y - 46), "baby q_b = [-7, -6, -5, -4, -3, -2, -1, 0]  (g=8)", F["mono"], BLUE, anchor="mm")
    for c, q_b in enumerate(baby_q):
        chip(draw, grid_x + c * cw, grid_y - 24, cw - 8, 32, str(q_b), BLUE_LIGHT, BLUE, BLUE, r=7, f=F["mono_small"])

    for r, q_g in enumerate(giant_q):
        row_y = grid_y + r * (ch + 24)
        chip(draw, grid_x - 86, row_y, 66, 40, str(q_g), ORANGE_LIGHT, ORANGE, ORANGE, r=7, f=F["mono_small"])
        for c, q_b in enumerate(baby_q):
            q = q_b + q_g
            fill = GREEN_LIGHT if -7 <= q <= 7 else GRAY_LIGHT
            outline = GREEN if -7 <= q <= 7 else "#c6d2de"
            txt = GREEN if -7 <= q <= 7 else MUTED
            chip(draw, grid_x + c * cw, row_y, cw - 8, 40, str(q), fill, outline, txt, r=7, f=F["mono_small"])

    text(draw, (grid_x - 78, grid_y - 48), "giant q_g", F["small_bold"], ORANGE)
    text(draw, (grid_x + 4 * cw, grid_y + 2 * (ch + 24) + 8), "绿色格子覆盖 q=-7..7；灰色 q=8 是 g*b=16 带来的 padding。", F["small"], MUTED, anchor="mm")

    rx = x + 860
    text(draw, (rx, y + 124), "参数计算", F["body_bold"])
    text(draw, (rx, y + 158), "numRotations = 15", F["mono"], INK)
    text(draw, (rx, y + 192), "g = 2^(floor(3/2)+1+1) = 8", F["mono"], INK)
    text(draw, (rx, y + 226), "b = (15+1)/8 = 2", F["mono"], INK)
    text(draw, (rx, y + 276), "每个 scale 原始保存：", F["body_bold"])
    text(draw, (rx + 210, y + 276), "8 baby keys + 2 giant keys", F["mono_bold"], BLUE)
    text(draw, (rx, y + 322), "直觉：key 集合像两把尺子；组合时能量到整个局部 q 区间。", F["small"], MUTED)


def draw_scaled_table(draw):
    x, y, w, h = 54, 836, 1022, 350
    card(draw, x, y, w, h)
    text(draw, (x + 26, y + 35), "4. 乘以 scale factor，再做 ReduceRotation", F["panel"])
    text(draw, (x + 26, y + 68), "本例 M/4=slots=64，所以 C2S 和 S2C 的最终集合相同，只是遍历顺序不同。", F["small"], MUTED)

    baby_q = list(range(-7, 1))
    giant_q = [0, 8]
    rows = []
    for sf in [1, 8]:
        raw_baby = [q * sf for q in baby_q]
        raw_giant = [q * sf for q in giant_q]
        red_baby = [reduce_rotation(v, 64) for v in raw_baby]
        red_giant = [reduce_rotation(v, 64) for v in raw_giant]
        rows.append((sf, raw_baby, red_baby, raw_giant, red_giant))

    headers = ["scale", "baby", "giant"]
    col_x = [x + 36, x + 150, x + 710]
    for cx, label in zip(col_x, headers):
        text(draw, (cx, y + 116), label, F["small_bold"], MUTED)
    line(draw, [(x + 30, y + 142), (x + w - 30, y + 142)], GRID, 1)

    yy = y + 168
    for sf, raw_baby, red_baby, raw_giant, red_giant in rows:
        chip(draw, col_x[0], yy - 19, 72, 34, f"sf={sf}", GREEN_LIGHT if sf == 1 else ORANGE_LIGHT, GREEN if sf == 1 else ORANGE, GREEN if sf == 1 else ORANGE, r=7, f=F["small_bold"])
        text(draw, (col_x[1], yy - 16), f"q_b*sf: {compact_middle(raw_baby)}", F["mono_small"], BLUE)
        text(draw, (col_x[1], yy + 16), f"mod64:  {compact_middle(red_baby)}", F["mono_small"], BLUE)
        text(draw, (col_x[2], yy - 16), f"q_g*sf: {compact(raw_giant)}", F["mono_small"], ORANGE)
        text(draw, (col_x[2], yy + 16), f"mod64:  {compact(red_giant)}", F["mono_small"], ORANGE)
        yy += 92

    text(draw, (x + 36, y + 314), "最后 union 去重，并删掉 0 和 M/4。", F["body_bold"], INK)


def draw_ring(draw):
    x, y, w, h = 1104, 836, 692, 350
    card(draw, x, y, w, h)
    text(draw, (x + 26, y + 35), "最终返回的 rot indices", F["panel"])
    text(draw, (x + 26, y + 68), "在 mod 64 的环上看，选中的 key 分成两块。", F["small"], MUTED)

    cx, cy, r = x + 212, y + 204, 104
    draw.ellipse(rect((cx - r, cy - r, cx + r, cy + r)), outline="#cbd7e2", width=4 * SCALE)
    selected_orange = [8, 16, 24, 32, 40, 48, 56]
    selected_blue = [57, 58, 59, 60, 61, 62, 63]
    selected = set(selected_orange + selected_blue)

    for idx in range(64):
        ang = -math.pi / 2 + 2 * math.pi * idx / 64
        outer = (cx + math.cos(ang) * (r + 4), cy + math.sin(ang) * (r + 4))
        inner_len = 16 if idx in selected or idx in [0, 8, 16, 24, 32, 40, 48, 56] else 7
        inner = (cx + math.cos(ang) * (r - inner_len), cy + math.sin(ang) * (r - inner_len))
        color = ORANGE if idx in selected_orange else BLUE if idx in selected_blue else "#cbd7e2"
        width = 3 if idx in selected else 1
        line(draw, [inner, outer], color, width)

    for idx in selected_orange + selected_blue + [0]:
        ang = -math.pi / 2 + 2 * math.pi * idx / 64
        label_r = r + 34 if idx not in [57, 58, 59, 60, 61, 62, 63] else r + 48
        px = cx + math.cos(ang) * label_r
        py = cy + math.sin(ang) * label_r
        color = ORANGE if idx in selected_orange else BLUE if idx in selected_blue else RED
        text(draw, (px, py), str(idx), F["mono_small"], color, anchor="mm")

    line(draw, [(cx - 14, cy - r - 44), (cx + 14, cy - r - 20)], RED, 2)
    line(draw, [(cx + 14, cy - r - 44), (cx - 14, cy - r - 20)], RED, 2)

    rx = x + 392
    final = selected_orange + selected_blue
    text(draw, (rx, y + 122), "return sorted(unique(keys) - {0, M/4})", F["mono_small"], INK)
    text(draw, (rx, y + 168), "0 被删掉；本例 M/4=64，", F["small"], MUTED)
    text(draw, (rx, y + 196), "不在 0..63 中额外出现。", F["small"], MUTED)
    text(draw, (rx, y + 238), "[8, 16, 24, 32, 40, 48, 56,", F["mono_bold"], GREEN)
    text(draw, (rx, y + 270), " 57, 58, 59, 60, 61, 62, 63]", F["mono_bold"], GREEN)
    text(draw, (rx, y + 314), "共 14 个 rotation indices", F["body_bold"], GREEN)


def render():
    img = Image.new("RGB", (W * SCALE, H * SCALE), BG)
    draw = ImageDraw.Draw(img)
    draw_top(draw)
    draw_layer_card(draw)
    draw_param_card(draw)
    draw_bsgs_card(draw)
    draw_scaled_table(draw)
    draw_ring(draw)
    text(
        draw,
        (54, 1230),
        "特殊情况：当 levelBudget=[1,1] 时，代码走 FindLinearTransformRotationIndices，用 sqrt(slots) 规模的 BSGS；上图解释的是默认 bootstrapping C2S/S2C 路径。",
        F["small"],
        MUTED,
    )
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    return OUT


if __name__ == "__main__":
    print(render())
