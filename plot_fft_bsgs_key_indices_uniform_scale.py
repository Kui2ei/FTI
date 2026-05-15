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


def params(log_slots, lb, dim1=0):
    layers, rows, rem = select_layers(log_slots, lb)
    num_rot = (1 << (layers + 1)) - 1
    num_rot_rem = (1 << (rem + 1)) - 1
    g = dim1 if dim1 and dim1 <= num_rot else 1 << (layers // 2 + 1 + (1 if num_rot > 7 else 0))
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


def row_defs(log_slots, direction):
    slots = 1 << log_slots
    m = (1 << LOG_N) * 2
    half_mod = m // 4
    lb = LEVEL_BUDGET[0 if direction == "C2S" else 1]
    p = params(log_slots, lb)
    flag_rem = 0 if p["rem"] == 0 else 1
    rows = []
    if direction == "C2S":
        for s in range(lb - 1, flag_rem - 1, -1):
            sf = 1 << ((s - flag_rem) * p["layers"] + p["rem"])
            half = 1 - ((p["num_rot"] + 1) // 2)
            rows.append((f"C2S s={s}", sf, half, p["num_rot"], p["g"], p["b"], slots, half_mod))
        if flag_rem:
            half = 1 - ((p["num_rot_rem"] + 1) // 2)
            rows.append(("C2S rem s=0", 1, half, p["num_rot_rem"], p["g_rem"], p["b_rem"], slots, half_mod))
    else:
        for s in range(0, lb - flag_rem):
            sf = 1 << (s * p["layers"])
            half = 1 - ((p["num_rot"] + 1) // 2)
            rows.append((f"S2C s={s}", sf, half, p["num_rot"], p["g"], p["b"], half_mod, half_mod))
        if flag_rem:
            s = lb - flag_rem
            sf = 1 << (s * p["layers"])
            half = 1 - ((p["num_rot_rem"] + 1) // 2)
            rows.append((f"S2C rem s={s}", sf, half, p["num_rot_rem"], p["g_rem"], p["b_rem"], half_mod, half_mod))
    return rows, p


def txt(x, y, s, size=11, weight=400, fill="#22313f", anchor="start", rotate=None):
    t = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{t}>{s}</text>'
    )


def line(x1, y1, x2, y2, stroke="#cbd6df", width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def rect(x, y, w, h, fill, stroke="#d8e1e9", rx=0, opacity=1):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'


def circle(x, y, r, fill, stroke="#ffffff", width=1):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def sx(value, lo, hi, x0, w):
    return x0 + (value - lo) * w / (hi - lo)


def fmt(v):
    if abs(v) >= 1000:
        return f"{v//1024}k" if v % 1024 == 0 else str(v)
    return str(v)


def compact(values):
    if len(values) <= 7:
        return "[" + ", ".join(str(v) for v in values) + "]"
    return "[" + ", ".join(str(v) for v in values[:3]) + ", ..., " + ", ".join(str(v) for v in values[-2:]) + "]"


def draw_scale(parts, x0, y, w, lo, hi, title):
    parts.append(txt(x0, y - 16, title, size=12, weight=800))
    parts.append(line(x0, y, x0 + w, y, "#7b8b9a", 1.4))
    ticks = [-65536, -57344, -32768, -16384, -8192, -4096, 0, 4096, 8192, 16384, 32768, 57344, 65536]
    for t in ticks:
        if lo <= t <= hi:
            x = sx(t, lo, hi, x0, w)
            major = t in (-65536, -32768, 0, 32768, 65536)
            parts.append(line(x, y - (9 if major else 5), x, y + (9 if major else 5), "#536575", 1 if major else 0.8))
            parts.append(txt(x, y + 24 if major else y + 18, fmt(t), size=8 if major else 7, fill="#536575", anchor="middle"))
    parts.append(txt(x0 + w, y - 16, "signed diag = q * scalingFactor", size=10, fill="#607080", anchor="end"))


def draw_row(parts, row, y, x0, w, lo, hi):
    name, sf, half, num_rot, g, b, direct_mod, giant_mod = row
    q_values = [half + u for u in range(num_rot)]
    signed = [q * sf for q in q_values]
    direct_keys = [reduce_rotation(v, direct_mod) for v in signed]
    axis_y = y + 37
    parts.append(rect(28, y, 1686, 84, "#ffffff", "#d8e1e9", rx=7))
    parts.append(txt(45, y + 24, name, size=12, weight=800))
    parts.append(txt(45, y + 43, f"sf={sf}", size=10, fill="#5c6b78"))
    parts.append(txt(45, y + 60, f"diag={half}..{half + num_rot - 1} * sf", size=9, fill="#71808d"))

    parts.append(line(x0, axis_y, x0 + w, axis_y, "#e0e7ee", 1))
    for t in (-65536, -32768, 0, 32768, 65536):
        if lo <= t <= hi:
            x = sx(t, lo, hi, x0, w)
            parts.append(line(x, axis_y - 26, x, axis_y + 26, "#e3e9ef" if t else "#2a3742", 1.1 if t else 1.4))

    for q, value, key in zip(q_values, signed, direct_keys):
        x = sx(value, lo, hi, x0, w)
        u = q - half
        block = u // g
        fill = "#2e86ab" if block == 0 else "#d77a30"
        if q == 0:
            fill = "#202a33"
        stroke = "#1f2933" if key in (0, 32768) else "#ffffff"
        parts.append(circle(x, axis_y, 5.6, fill, stroke, 1.3))
        parts.append(txt(x, axis_y - 12, str(value), size=7.2, fill="#3c4a56", anchor="middle", rotate=-45))
        parts.append(txt(x, axis_y + 22, str(key), size=7.2, fill="#166534", anchor="middle", rotate=-45))

    giant_signed = g * sf
    giant_key = reduce_rotation(giant_signed, giant_mod)
    baby_keys = [reduce_rotation((half + j) * sf, direct_mod) for j in range(g)]
    giant_keys = [reduce_rotation(g * i * sf, giant_mod) for i in range(b)]
    rx = x0 + w + 30
    parts.append(txt(rx, y + 23, f"giant step = {g}*sf = {giant_signed}", size=10, weight=800, fill="#8a4d1f"))
    parts.append(txt(rx, y + 40, f"giant key(s): {compact(giant_keys)}", size=9, fill="#8a4d1f"))
    parts.append(txt(rx, y + 58, f"baby keys: {compact(baby_keys)}", size=9, fill="#2e86ab"))
    parts.append(txt(rx, y + 75, f"direct key modulus={direct_mod}", size=8.5, fill="#607080"))


def render(log_slots):
    slots = 1 << log_slots
    m = (1 << LOG_N) * 2
    c2s, p1 = row_defs(log_slots, "C2S")
    s2c, p2 = row_defs(log_slots, "S2C")
    rows = c2s + s2c
    lo = min((half + u) * sf for _, sf, half, nr, *_ in rows for u in range(nr))
    hi = max((half + u) * sf for _, sf, half, nr, *_ in rows for u in range(nr))
    pad = int((hi - lo) * 0.04)
    lo -= pad
    hi += pad

    width = 1760
    row_h = 98
    height = 185 + row_h * len(rows) + 45
    axis_x = 260
    axis_w = 1040
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f7fafc", "none"),
        txt(34, 38, "Uniform-scale view of collapsed FFT diag indices", size=22, weight=850),
        txt(34, 64, f"logN={LOG_N}, logSlots={log_slots}, slots={slots}, M={m}, levelBudget={LEVEL_BUDGET}", size=12, fill="#556575"),
        rect(34, 82, 1680, 58, "#ffffff", "#d8e1e9", rx=7),
        txt(56, 108, "All rows share one signed-diag scale.", size=12, weight=800),
        txt(286, 108, "Top labels are signed diag=q*sf; green labels are ReduceRotation(...) key indices.", size=12, fill="#2e3d4a"),
        txt(286, 128, "Large scaling factors spread out; small layers collapse near 0, revealing the multiscale symmetry.", size=11, fill="#607080"),
    ]
    draw_scale(parts, axis_x, 166, axis_w, lo, hi, f"global scale [{lo}, {hi}]")

    y = 198
    parts.append(txt(44, y - 12, f"C2S: layers={p1['layers']}, rem={p1['rem']}, g={p1['g']}, b={p1['b']}", size=11, weight=800))
    for row in c2s:
        draw_row(parts, row, y, axis_x, axis_w, lo, hi)
        y += row_h
    y += 8
    parts.append(txt(44, y - 12, f"S2C: layers={p2['layers']}, rem={p2['rem']}, g={p2['g']}, b={p2['b']}", size=11, weight=800))
    for row in s2c:
        draw_row(parts, row, y, axis_x, axis_w, lo, hi)
        y += row_h

    parts.append("</svg>")
    out = RESULTS / f"fft_bsgs_key_indices_uniform_logslots{log_slots}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main():
    for log_slots in (15, 16):
        print(render(log_slots))


if __name__ == "__main__":
    main()
