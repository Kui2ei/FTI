import math
from pathlib import Path


LOG_N = 16
LEVEL_BUDGET = [5, 5]
RESULTS = Path("results")


def reduce_rotation(index, slots):
    if slots & (slots - 1) == 0:
        n = slots.bit_length() - 1
        if index >= 0:
            return index - ((index >> n) << n)
        return index + slots + ((abs(index) >> n) << n)
    return (slots + index % slots) % slots


def select_layers(log_slots, budget):
    layers = math.ceil(log_slots / budget)
    rows = log_slots // layers
    rem = log_slots % layers
    dim = rows + (1 if rem else 0)
    if dim < budget:
        layers -= 1
        rows = log_slots // layers
        rem = log_slots - rows * layers
        dim = rows + (1 if rem else 0)
        if dim > budget:
            while dim != budget:
                rows -= 1
                rem = log_slots - rows * layers
                dim = rows + (1 if rem else 0)
    return layers, rows, rem


def params(log_slots, lb):
    layers, rows, rem = select_layers(log_slots, lb)
    num_rot = (1 << (layers + 1)) - 1
    num_rot_rem = (1 << (rem + 1)) - 1
    g = 1 << (layers // 2 + 1 + (1 if num_rot > 7 else 0))
    b = (num_rot + 1) // g
    if rem:
        g_rem = 1 << (rem // 2 + 1 + (1 if num_rot_rem > 7 else 0))
        b_rem = (num_rot_rem + 1) // g_rem
    else:
        g_rem = 0
        b_rem = 0
    return {
        "layers": layers,
        "rows": rows,
        "rem": rem,
        "num_rot": num_rot,
        "num_rot_rem": num_rot_rem,
        "g": g,
        "b": b,
        "g_rem": g_rem,
        "b_rem": b_rem,
    }


def make_rows(log_slots, direction):
    m = (1 << LOG_N) * 2
    slots = 1 << log_slots
    half_mod = m // 4
    lb = LEVEL_BUDGET[0 if direction == "C2S" else 1]
    p = params(log_slots, lb)
    flag_rem = 0 if p["rem"] == 0 else 1
    rows = []

    def add(name, sf, half, num_rot, g, b, direct_mod, giant_mod):
        qs = [half + u for u in range(num_rot)]
        signed = [q * sf for q in qs]
        keys = [reduce_rotation(v, direct_mod) for v in signed]
        rows.append(
            {
                "name": name,
                "sf": sf,
                "half": half,
                "num_rot": num_rot,
                "g": g,
                "b": b,
                "direct_mod": direct_mod,
                "giant_mod": giant_mod,
                "qs": qs,
                "signed": signed,
                "keys": keys,
                "direction": direction,
            }
        )

    if direction == "C2S":
        for s in range(lb - 1, flag_rem - 1, -1):
            sf = 1 << ((s - flag_rem) * p["layers"] + p["rem"])
            half = 1 - ((p["num_rot"] + 1) // 2)
            add(f"C2S s={s}", sf, half, p["num_rot"], p["g"], p["b"], slots, half_mod)
        if flag_rem:
            half = 1 - ((p["num_rot_rem"] + 1) // 2)
            add("C2S rem s=0", 1, half, p["num_rot_rem"], p["g_rem"], p["b_rem"], slots, half_mod)
    else:
        for s in range(0, lb - flag_rem):
            sf = 1 << (s * p["layers"])
            half = 1 - ((p["num_rot"] + 1) // 2)
            add(f"S2C s={s}", sf, half, p["num_rot"], p["g"], p["b"], half_mod, half_mod)
        if flag_rem:
            s = lb - flag_rem
            sf = 1 << (s * p["layers"])
            half = 1 - ((p["num_rot_rem"] + 1) // 2)
            add(f"S2C rem s={s}", sf, half, p["num_rot_rem"], p["g_rem"], p["b_rem"], half_mod, half_mod)
    return rows, p


def txt(x, y, s, size=11, weight=400, fill="#22313f", anchor="start", rotate=None):
    rot = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{rot}>{s}</text>'
    )


def line(x1, y1, x2, y2, stroke="#c7d3dd", width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def rect(x, y, w, h, fill, stroke="#d8e1e9", rx=0, opacity=1):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'


def circle(x, y, r, fill, stroke="#ffffff", width=1, opacity=1):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'


def path(d, stroke="#8798a8", width=1, fill="none", dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{width}" fill="{fill}"{da}/>'


def sx(v, lo, hi, x0, w):
    return x0 + (v - lo) * w / (hi - lo)


def compact(values):
    if len(values) <= 8:
        return "[" + ", ".join(str(v) for v in values) + "]"
    return "[" + ", ".join(str(v) for v in values[:3]) + ", ..., " + ", ".join(str(v) for v in values[-2:]) + "]"


def draw_global_scale(parts, x, y, w, lo, hi):
    parts.append(txt(x, y - 16, "global signed-diag scale", size=12, weight=800))
    parts.append(line(x, y, x + w, y, "#71808d", 1.4))
    ticks = [-65536, -57344, -32768, -16384, -8192, -4096, 0, 4096, 8192, 16384, 32768, 57344, 65536]
    for t in ticks:
        if lo <= t <= hi:
            xx = sx(t, lo, hi, x, w)
            major = t in (-65536, -32768, 0, 32768, 65536)
            parts.append(line(xx, y - (8 if major else 5), xx, y + (8 if major else 5), "#536575", 1.0))
            parts.append(txt(xx, y + (23 if major else 17), str(t), size=7.5, fill="#536575", anchor="middle"))


def draw_row(parts, row, y, global_lo, global_hi, context_x, context_w, zoom_x, zoom_w):
    signed = row["signed"]
    keys = row["keys"]
    lo = min(signed)
    hi = max(signed)
    if lo == hi:
        lo -= 1
        hi += 1
    pad = max(1, int((hi - lo) * 0.08))
    zlo = lo - pad
    zhi = hi + pad
    zoom_factor = (global_hi - global_lo) / (zhi - zlo)
    axis_y = y + 54
    zoom_y = y + 106

    parts.append(rect(24, y, 1728, 158, "#ffffff", "#d8e1e9", rx=8))
    parts.append(txt(42, y + 25, row["name"], size=13, weight=850))
    parts.append(txt(42, y + 45, f"sf={row['sf']}", size=10, fill="#596a78"))
    parts.append(txt(42, y + 62, f"diag={row['half']}..{row['half'] + row['num_rot'] - 1} * sf", size=9.5, fill="#6c7a86"))
    parts.append(txt(42, y + 79, f"zoom x{zoom_factor:.1f}", size=10, weight=750, fill="#8a4d1f"))

    parts.append(line(context_x, axis_y, context_x + context_w, axis_y, "#dce5ec", 1))
    win_x1 = sx(lo, global_lo, global_hi, context_x, context_w)
    win_x2 = sx(hi, global_lo, global_hi, context_x, context_w)
    parts.append(rect(win_x1, axis_y - 12, max(2, win_x2 - win_x1), 24, "#f2c078", "#d77a30", rx=3, opacity=0.35))
    for t in (-65536, -32768, 0, 32768, 65536):
        if global_lo <= t <= global_hi:
            xx = sx(t, global_lo, global_hi, context_x, context_w)
            parts.append(line(xx, axis_y - 17, xx, axis_y + 17, "#edf2f6" if t else "#334150", 1))
    for q, v, key in zip(row["qs"], signed, keys):
        xx = sx(v, global_lo, global_hi, context_x, context_w)
        u = q - row["half"]
        fill = "#2e86ab" if u // row["g"] == 0 else "#d77a30"
        if q == 0:
            fill = "#202a33"
        parts.append(circle(xx, axis_y, 3.1, fill, opacity=0.78))
    parts.append(txt(context_x, axis_y + 29, "context: shared global scale", size=8.5, fill="#7a8792"))

    parts.append(path(f"M {win_x1:.1f} {axis_y+16:.1f} C {win_x1:.1f} {axis_y+34:.1f}, {zoom_x:.1f} {zoom_y-30:.1f}, {zoom_x:.1f} {zoom_y-12:.1f}", "#b56a2b", 1.1, dash="4 4"))
    parts.append(path(f"M {win_x2:.1f} {axis_y+16:.1f} C {win_x2:.1f} {axis_y+34:.1f}, {zoom_x+zoom_w:.1f} {zoom_y-30:.1f}, {zoom_x+zoom_w:.1f} {zoom_y-12:.1f}", "#b56a2b", 1.1, dash="4 4"))

    parts.append(line(zoom_x, zoom_y, zoom_x + zoom_w, zoom_y, "#9aa9b6", 1.2))
    for t in [lo, 0, hi]:
        if zlo <= t <= zhi:
            xx = sx(t, zlo, zhi, zoom_x, zoom_w)
            parts.append(line(xx, zoom_y - 16, xx, zoom_y + 17, "#1f2d3a" if t == 0 else "#8a98a5", 1))
            parts.append(txt(xx, zoom_y + 33, str(t), size=8, fill="#536575", anchor="middle"))
    for q, v, key in zip(row["qs"], signed, keys):
        xx = sx(v, zlo, zhi, zoom_x, zoom_w)
        u = q - row["half"]
        fill = "#2e86ab" if u // row["g"] == 0 else "#d77a30"
        if q == 0:
            fill = "#202a33"
        stroke = "#1f2933" if key in (0, 32768) else "#ffffff"
        parts.append(circle(xx, zoom_y, 5.4, fill, stroke=stroke, width=1.2))
        parts.append(txt(xx, zoom_y - 13, str(v), size=7.2, fill="#3c4a56", anchor="middle", rotate=-45))
        parts.append(txt(xx, zoom_y + 25, str(key), size=7.2, fill="#166534", anchor="middle", rotate=-45))

    giant_signed = row["g"] * row["sf"]
    giant_keys = [reduce_rotation(row["g"] * i * row["sf"], row["giant_mod"]) for i in range(row["b"])]
    baby_keys = [reduce_rotation((row["half"] + j) * row["sf"], row["direct_mod"]) for j in range(row["g"])]
    rx = zoom_x + zoom_w + 35
    parts.append(txt(rx, y + 40, f"giant step: {row['g']}*sf = {giant_signed}", size=10, weight=800, fill="#8a4d1f"))
    parts.append(txt(rx, y + 58, f"giant keys: {compact(giant_keys)}", size=9, fill="#8a4d1f"))
    parts.append(txt(rx, y + 77, f"baby keys: {compact(baby_keys)}", size=9, fill="#2e86ab"))
    parts.append(txt(rx, y + 104, "zoom labels:", size=9, weight=800, fill="#5f6f7d"))
    parts.append(txt(rx + 80, y + 104, "top=signed diag, bottom=key index", size=9, fill="#5f6f7d"))


def render(log_slots):
    slots = 1 << log_slots
    m = (1 << LOG_N) * 2
    c2s, p1 = make_rows(log_slots, "C2S")
    s2c, p2 = make_rows(log_slots, "S2C")
    rows = c2s + s2c
    global_lo = min(min(r["signed"]) for r in rows)
    global_hi = max(max(r["signed"]) for r in rows)
    pad = int((global_hi - global_lo) * 0.04)
    global_lo -= pad
    global_hi += pad

    width = 1780
    row_h = 176
    height = 205 + len(rows) * row_h + 40
    context_x = 250
    context_w = 430
    zoom_x = 725
    zoom_w = 645

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f7fafc", "none"),
        txt(34, 38, "Zoom-lens view of scaled FFT diagonal indices", size=22, weight=850),
        txt(34, 64, f"logN={LOG_N}, logSlots={log_slots}, slots={slots}, M={m}, M/4={m//4}, levelBudget={LEVEL_BUDGET}", size=12, fill="#556575"),
        rect(34, 82, 1698, 62, "#ffffff", "#d8e1e9", rx=7),
        txt(56, 108, "Left: all rows on one global scale.", size=12, weight=800),
        txt(315, 108, "Orange window is cropped and enlarged on the right. This preserves global scale while keeping local indices readable.", size=12, fill="#2e3d4a"),
        txt(315, 130, "Zoomed dots show signed diag=q*sf above and the ReduceRotation key index below.", size=11, fill="#607080"),
    ]
    draw_global_scale(parts, context_x, 172, context_w, global_lo, global_hi)
    parts.append(txt(zoom_x, 156, "zoomed local axis for each layer", size=12, weight=800))
    parts.append(line(zoom_x, 172, zoom_x + zoom_w, 172, "#9aa9b6", 1.4))

    y = 205
    parts.append(txt(42, y - 12, f"C2S: layers={p1['layers']}, rem={p1['rem']}, g={p1['g']}, b={p1['b']}", size=11, weight=850))
    for row in c2s:
        draw_row(parts, row, y, global_lo, global_hi, context_x, context_w, zoom_x, zoom_w)
        y += row_h
    y += 10
    parts.append(txt(42, y - 12, f"S2C: layers={p2['layers']}, rem={p2['rem']}, g={p2['g']}, b={p2['b']}", size=11, weight=850))
    for row in s2c:
        draw_row(parts, row, y, global_lo, global_hi, context_x, context_w, zoom_x, zoom_w)
        y += row_h

    parts.append("</svg>")
    out = RESULTS / f"fft_bsgs_zoom_lens_logslots{log_slots}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main():
    for log_slots in (15, 16):
        print(render(log_slots))


if __name__ == "__main__":
    main()
