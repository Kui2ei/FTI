from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


OUT = Path("results/openfhe_rot_indices_bsgs_schematic.png")

W, H = 1900, 1420
BG = "#f6f8fb"
INK = "#203040"
MUTED = "#5c6f7e"
GRID = "#d7e0e8"
CARD = "#ffffff"
BLUE = "#2e86ab"
BLUE_LIGHT = "#d9eef6"
AMBER = "#d77a30"
AMBER_LIGHT = "#f8dfc7"
GREEN = "#0f8b57"
GREEN_LIGHT = "#dff3e8"
RED = "#c2413b"
GRAY = "#7a8792"

FONT_REG = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"


def font(size, bold=False, mono=False):
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    return ImageFont.truetype(path, size)


F = {
    "title": font(42, bold=True),
    "subtitle": font(22),
    "panel": font(27, bold=True),
    "body": font(20),
    "body_bold": font(20, bold=True),
    "small": font(17),
    "small_bold": font(17, bold=True),
    "mono": font(18, mono=True),
    "mono_small": font(16, mono=True),
    "chip": font(16, bold=True),
}


def select_layers(log_slots, level_budget):
    layers = math.ceil(log_slots / level_budget)
    rows = log_slots // layers
    rem = log_slots % layers
    dim = rows + (1 if rem > 0 else 0)
    if dim < level_budget:
        layers -= 1
        rows = log_slots // layers
        rem = log_slots - rows * layers
        dim = rows + (1 if rem > 0 else 0)
        if dim > level_budget:
            while dim != level_budget:
                rows -= 1
                rem = log_slots - rows * layers
                dim = rows + (1 if rem > 0 else 0)
    return layers, rows, rem


def reduce_rotation(index, slots):
    if slots & (slots - 1) == 0:
        n = slots.bit_length() - 1
        if index >= 0:
            return index - ((index >> n) << n)
        return index + slots + ((abs(index) >> n) << n)
    return (slots + index % slots) % slots


def params(log_slots, level_budget):
    layers, rows, rem = select_layers(log_slots, level_budget)
    num_rot = (1 << (layers + 1)) - 1
    shift_base = layers // 2 + 1 + (1 if num_rot > 7 else 0)
    g = 1 << shift_base
    b = (num_rot + 1) // g
    half = 1 - ((num_rot + 1) // 2)
    return {
        "layers": layers,
        "rows": rows,
        "rem": rem,
        "num_rot": num_rot,
        "half": half,
        "q_values": list(range(half, half + num_rot)),
        "g": g,
        "b": b,
        "baby_q": list(range(half, half + g)),
        "giant_q": [g * i for i in range(b)],
    }


def compact(values):
    if not values:
        return "[]"
    if len(values) <= 7:
        return "[" + ", ".join(str(v) for v in values) + "]"
    return "[" + ", ".join(str(v) for v in values[:3]) + ", ..., " + ", ".join(str(v) for v in values[-3:]) + "]"


def card(draw, xy, radius=14, fill=CARD, outline=GRID):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)


def text(draw, xy, s, f, fill=INK, anchor=None):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def chip(draw, x, y, w, h, label, fill, outline=None, txt_fill=INK, r=10, f=None):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=r, fill=fill, outline=outline or fill, width=2)
    text(draw, (x + w / 2, y + h / 2), label, f or F["chip"], fill=txt_fill, anchor="mm")


def draw_layer_panel(draw, p, log_slots, level_budget):
    x, y, w, h = 60, 140, 1120, 305
    card(draw, (x, y, x + w, y + h))
    text(draw, (x + 28, y + 42), "Collapsed FFT layers", F["panel"])
    text(
        draw,
        (x + 28, y + 74),
        f"Example: logSlots={log_slots}, levelBudget={level_budget} -> layersCollapse={p['layers']}, rows={p['rows']}, rem={p['rem']}",
        F["small"],
        MUTED,
    )

    lx, ly = x + 48, y + 126
    box_w, box_h, gap = 90, 46, 10
    for i in range(log_slots):
        fill = BLUE_LIGHT if i < p["layers"] else AMBER_LIGHT
        outline = BLUE if i < p["layers"] else AMBER
        chip(draw, lx + i * (box_w + gap), ly, box_w, box_h, f"L{i}", fill, outline, txt_fill=INK, r=8)

    group_y = ly + 78
    group_w = p["layers"] * box_w + (p["layers"] - 1) * gap
    chip(draw, lx, group_y, group_w, 38, "collapsed level with sf=1", GREEN_LIGHT, GREEN, txt_fill=GREEN, r=8, f=F["small_bold"])
    chip(
        draw,
        lx + p["layers"] * (box_w + gap),
        group_y,
        group_w,
        38,
        "collapsed level with sf=8",
        AMBER_LIGHT,
        AMBER,
        txt_fill=AMBER,
        r=8,
        f=F["small_bold"],
    )

    rx = x + 720
    text(draw, (rx, y + 128), "direction order", F["small_bold"])
    text(draw, (rx, y + 165), "C2S: high scale first", F["small"], MUTED)
    chip(draw, rx + 215, y + 146, 82, 34, "sf=8", AMBER_LIGHT, AMBER, txt_fill=AMBER, r=8, f=F["small_bold"])
    chip(draw, rx + 310, y + 146, 82, 34, "sf=1", GREEN_LIGHT, GREEN, txt_fill=GREEN, r=8, f=F["small_bold"])
    text(draw, (rx, y + 217), "S2C: low scale first", F["small"], MUTED)
    chip(draw, rx + 215, y + 198, 82, 34, "sf=1", GREEN_LIGHT, GREEN, txt_fill=GREEN, r=8, f=F["small_bold"])
    chip(draw, rx + 310, y + 198, 82, 34, "sf=8", AMBER_LIGHT, AMBER, txt_fill=AMBER, r=8, f=F["small_bold"])


def draw_param_panel(draw, p):
    x, y, w, h = 1210, 140, 630, 305
    card(draw, (x, y, x + w, y + h))
    text(draw, (x + 28, y + 42), "Per-level BSGS parameters", F["panel"])
    lines = [
        f"numRotations = 2^(layers+1)-1 = {p['num_rot']}",
        f"local signed q = {p['half']}..{p['half'] + p['num_rot'] - 1}",
        f"g = 2^(floor(layers/2)+1+[numRot>7]) = {p['g']}",
        f"b = (numRotations+1)/g = {p['b']}",
        "dim1 overrides g when 0 < dim1 <= numRotations",
    ]
    yy = y + 96
    for line in lines:
        text(draw, (x + 34, yy), line, F["mono"], INK if "dim1" not in line else MUTED)
        yy += 38


def draw_bsgs_panel(draw, p):
    x, y, w, h = 60, 475, 1780, 340
    card(draw, (x, y, x + w, y + h))
    text(draw, (x + 28, y + 42), "Local diagonal interval as BSGS tiles", F["panel"])
    text(
        draw,
        (x + 28, y + 74),
        "The code stores one baby list and one giant list for each scaled level; their sums cover the local q interval.",
        F["small"],
        MUTED,
    )

    axis_x, axis_y = x + 160, y + 132
    cell_w, cell_h = 65, 44
    q_min, q_max = p["half"], p["half"] + p["g"] * p["b"] - 1

    text(draw, (x + 44, axis_y + 9), "q ruler", F["small_bold"], MUTED)
    for q in range(q_min, q_max + 1):
        cx = axis_x + (q - q_min) * cell_w
        draw.line((cx + cell_w / 2, axis_y - 12, cx + cell_w / 2, axis_y - 2), fill=GRAY, width=2)
        text(draw, (cx + cell_w / 2, axis_y - 28), str(q), F["small"], MUTED, anchor="mm")
    draw.line((axis_x, axis_y, axis_x + (q_max - q_min + 1) * cell_w, axis_y), fill=GRAY, width=2)

    rows = [(0, "giant=0"), (p["g"], "giant=8")]
    for ridx, (giant, label) in enumerate(rows):
        row_y = axis_y + 24 + ridx * 72
        text(draw, (x + 44, row_y + cell_h / 2), label, F["small_bold"], AMBER if giant else BLUE, anchor="lm")
        for bq in p["baby_q"]:
            q = bq + giant
            cx = axis_x + (q - q_min) * cell_w
            fill = BLUE_LIGHT if giant == 0 else AMBER_LIGHT
            outline = BLUE if giant == 0 else AMBER
            chip(draw, cx + 4, row_y, cell_w - 8, cell_h, str(q), fill, outline, txt_fill=INK, r=8)

    rx = x + 1320
    text(draw, (rx, y + 132), f"baby q_b = {compact(p['baby_q'])}", F["mono"], BLUE)
    text(draw, (rx, y + 172), f"giant q_g = {compact(p['giant_q'])}", F["mono"], AMBER)
    text(draw, (rx, y + 212), "candidate q = q_b + q_g", F["mono"], INK)
    text(draw, (rx, y + 252), f"key lists per level: g+b = {p['g'] + p['b']} entries", F["mono"], GREEN)
    text(draw, (rx, y + 292), "q=8 is padding from g*b=16", F["mono_small"], MUTED)


def row_values(p, sf, direct_mod, giant_mod):
    baby = [reduce_rotation(q * sf, direct_mod) for q in p["baby_q"]]
    giant = [reduce_rotation(gq * sf, giant_mod) for gq in p["giant_q"]]
    return baby, giant


def draw_key_card(draw, x, y, w, title, subtitle, rows):
    card(draw, (x, y, x + w, y + 260), radius=12)
    text(draw, (x + 24, y + 36), title, F["body_bold"])
    text(draw, (x + 24, y + 66), subtitle, F["small"], MUTED)
    yy = y + 124
    for label, baby, giant in rows:
        chip(draw, x + 24, yy - 23, 86, 32, label, "#eef2f6", "#c9d4de", txt_fill=INK, r=8, f=F["small_bold"])
        text(draw, (x + 126, yy), f"baby:  {compact(baby)}", F["mono_small"], BLUE)
        text(draw, (x + 126, yy + 34), f"giant: {compact(giant)}", F["mono_small"], AMBER)
        yy += 82


def draw_key_panel(draw, p, slots, m):
    x, y, w, h = 60, 845, 1780, 475
    card(draw, (x, y, x + w, y + h))
    text(draw, (x + 28, y + 42), "Scale, ReduceRotation, union", F["panel"])
    text(
        draw,
        (x + 28, y + 74),
        "For each collapsed level, q is multiplied by sf and reduced into the modulus used by that direction.",
        F["small"],
        MUTED,
    )

    c2s_rows = []
    for sf in (8, 1):
        baby, giant = row_values(p, sf, slots, m // 4)
        c2s_rows.append((f"sf={sf}", baby, giant))
    draw_key_card(
        draw,
        x + 28,
        y + 100,
        840,
        "C2S",
        f"baby modulo slots={slots}; giant modulo M/4={m//4}",
        c2s_rows,
    )

    s2c_rows = []
    for sf in (1, 8):
        baby, giant = row_values(p, sf, m // 4, m // 4)
        s2c_rows.append((f"sf={sf}", baby, giant))
    draw_key_card(
        draw,
        x + 912,
        y + 100,
        840,
        "S2C",
        f"baby and giant both modulo M/4={m//4}",
        s2c_rows,
    )

    ratio = m // (4 * slots)
    extras = []
    j = 1
    while j < ratio:
        extras.append(j * slots)
        j <<= 1

    final = set(extras)
    for sf in (8, 1):
        baby, giant = row_values(p, sf, slots, m // 4)
        final.update(baby)
        final.update(giant)
    for sf in (1, 8):
        baby, giant = row_values(p, sf, m // 4, m // 4)
        final.update(baby)
        final.update(giant)
    final.discard(0)
    final.discard(m // 4)
    final = sorted(final)

    bottom_y = y + 388
    draw.rounded_rectangle((x + 28, bottom_y, x + w - 28, bottom_y + 44), radius=12, fill=GREEN_LIGHT, outline=GREEN, width=2)
    text(draw, (x + 48, bottom_y + 12), f"packing extension: M != 4*slots, add {extras}", F["mono_small"], GREEN)
    text(draw, (x + 592, bottom_y + 12), f"return sorted(unique(keys) - {{0, M/4}}): {compact(final)}", F["mono_small"], INK)


def render():
    log_n = 10
    log_slots = 6
    level_budget = 2
    slots = 1 << log_slots
    m = (1 << log_n) * 2
    p = params(log_slots, level_budget)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (60, 55), "OpenFHE rotation-index selection for bootstrapping", F["title"])
    text(
        draw,
        (60, 92),
        f"Toy numbers from the same code path: logN={log_n}, logSlots={log_slots}, slots={slots}, M={m}, levelBudget=[2,2]",
        F["subtitle"],
        MUTED,
    )
    chip(draw, W - 390, 50, 330, 42, "schematic, not a flowchart", "#edf2f6", "#c9d4de", txt_fill=MUTED, r=12, f=F["small_bold"])

    draw_layer_panel(draw, p, log_slots, level_budget)
    draw_param_panel(draw, p)
    draw_bsgs_panel(draw, p)
    draw_key_panel(draw, p, slots, m)

    text(draw, (60, H - 36), "Same rule scales to RotInOPENFHE(16,14): larger q interval, same baby/giant split, then dedup.", F["small"], MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    return OUT


if __name__ == "__main__":
    print(render())
