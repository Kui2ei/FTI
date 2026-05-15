import math
from pathlib import Path


RESULTS = Path("results")
LOG_N = 16
LEVEL_BUDGET = [5, 5]


def reduce_rotation(index, slots):
    if slots <= 0:
        raise ValueError("slots must be positive")
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


def collapsed_params(log_slots, budget, dim1=0):
    layers, rows, rem = select_layers(log_slots, budget)
    num_rot = (1 << (layers + 1)) - 1
    num_rot_rem = (1 << (rem + 1)) - 1
    g_default = 1 << (layers // 2 + 1 + (1 if num_rot > 7 else 0))
    g = g_default if dim1 == 0 or dim1 > num_rot else dim1
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


def build_rows(log_slots, direction):
    slots = 1 << log_slots
    m = (1 << LOG_N) * 2
    mod_s2c = m // 4
    lb = LEVEL_BUDGET[0 if direction == "C2S" else 1]
    p = collapsed_params(log_slots, lb)
    flag_rem = 0 if p["rem"] == 0 else 1
    rows = []

    if direction == "C2S":
        for s in range(lb - 1, flag_rem - 1, -1):
            sf = 1 << ((s - flag_rem) * p["layers"] + p["rem"])
            rows.append(
                {
                    "name": f"C2S s={s}",
                    "kind": "normal",
                    "sf": sf,
                    "q_min": 1 - ((p["num_rot"] + 1) // 2),
                    "q_max": ((p["num_rot"] + 1) // 2) - 1,
                    "g": p["g"],
                    "b": p["b"],
                    "direct_mod": slots,
                    "giant_mod": mod_s2c,
                }
            )
        if flag_rem:
            rows.append(
                {
                    "name": "C2S rem s=0",
                    "kind": "rem",
                    "sf": 1,
                    "q_min": 1 - ((p["num_rot_rem"] + 1) // 2),
                    "q_max": ((p["num_rot_rem"] + 1) // 2) - 1,
                    "g": p["g_rem"],
                    "b": p["b_rem"],
                    "direct_mod": slots,
                    "giant_mod": mod_s2c,
                }
            )
    else:
        for s in range(0, lb - flag_rem):
            sf = 1 << (s * p["layers"])
            rows.append(
                {
                    "name": f"S2C s={s}",
                    "kind": "normal",
                    "sf": sf,
                    "q_min": 1 - ((p["num_rot"] + 1) // 2),
                    "q_max": ((p["num_rot"] + 1) // 2) - 1,
                    "g": p["g"],
                    "b": p["b"],
                    "direct_mod": mod_s2c,
                    "giant_mod": mod_s2c,
                }
            )
        if flag_rem:
            s = lb - flag_rem
            rows.append(
                {
                    "name": f"S2C rem s={s}",
                    "kind": "rem",
                    "sf": 1 << (s * p["layers"]),
                    "q_min": 1 - ((p["num_rot_rem"] + 1) // 2),
                    "q_max": ((p["num_rot_rem"] + 1) // 2) - 1,
                    "g": p["g_rem"],
                    "b": p["b_rem"],
                    "direct_mod": mod_s2c,
                    "giant_mod": mod_s2c,
                }
            )
    return rows, p


def txt(x, y, content, size=11, anchor="start", weight=400, fill="#22313f", rotate=None):
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{content}</text>'
    )


def line(x1, y1, x2, y2, stroke="#bdc9d4", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def rect(x, y, w, h, fill, stroke="#d6e0e8", rx=0, opacity=1):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'
    )


def circle(x, y, r, fill, stroke="#ffffff", width=1):
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def path(d, stroke="#6b7c8d", width=1.5, fill="none", marker=False):
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{width}" fill="{fill}"{marker_attr}/>'


def sx(q, q_min, q_max, x0, w):
    return x0 + (q - q_min) * w / (q_max - q_min)


def compact(values, limit=8):
    values = list(values)
    if len(values) <= limit:
        return "[" + ", ".join(str(v) for v in values) + "]"
    head = ", ".join(str(v) for v in values[:4])
    tail = ", ".join(str(v) for v in values[-3:])
    return f"[{head}, ..., {tail}]"


def draw_row(parts, row, y, x0, axis_w):
    q_min = row["q_min"]
    q_max = row["q_max"]
    sf = row["sf"]
    g = row["g"]
    direct_mod = row["direct_mod"]
    giant_mod = row["giant_mod"]
    axis_y = y + 45
    label_x = 34

    parts.append(rect(24, y, 1720, 124, "#ffffff", "#d9e2ea", rx=7))
    parts.append(txt(label_x, y + 23, row["name"], size=13, weight=800))
    parts.append(txt(label_x, y + 42, f"sf={sf}", size=10, fill="#5d6d7c"))
    parts.append(txt(label_x, y + 58, f"q={q_min}..{q_max}", size=10, fill="#5d6d7c"))
    parts.append(txt(label_x, y + 74, f"direct mod={direct_mod}", size=10, fill="#5d6d7c"))

    parts.append(line(x0, axis_y, x0 + axis_w, axis_y, "#d0dae3", 1.2))
    zero_x = sx(0, q_min, q_max, x0, axis_w)
    parts.append(line(zero_x, axis_y - 24, zero_x, axis_y + 23, "#1f2d3a", 1.3))
    parts.append(txt(zero_x, axis_y - 30, "q=0", size=8, anchor="middle", fill="#1f2d3a"))

    for q in range(q_min, q_max + 1):
        x = sx(q, q_min, q_max, x0, axis_w)
        signed_diag = q * sf
        direct_key = reduce_rotation(signed_diag, direct_mod)
        fill = "#2e86ab" if q <= 0 else "#d77a30"
        if q == 0:
            fill = "#202a33"
        if direct_key in (0, 32768):
            stroke = "#202a33"
            sw = 1.8
        else:
            stroke = "#ffffff"
            sw = 1
        parts.append(circle(x, axis_y, 5.0, fill, stroke, sw))
        parts.append(txt(x, axis_y - 12, str(signed_diag), size=6.7, anchor="middle", fill="#3b4a58", rotate=-58))
        parts.append(txt(x, axis_y + 25, str(direct_key), size=7.0, anchor="middle", fill="#166534", rotate=-58))

    left_x = sx(q_min, q_min, q_max, x0, axis_w)
    zero_x = sx(0, q_min, q_max, x0, axis_w)
    right_x = sx(q_max, q_min, q_max, x0, axis_w)
    parts.append(line(left_x, axis_y - 21, zero_x, axis_y - 21, "#2e86ab", 2.2))
    parts.append(line(sx(1, q_min, q_max, x0, axis_w), axis_y - 21, right_x, axis_y - 21, "#d77a30", 2.2))

    if q_max > 0:
        arrow_y = axis_y + 48
        start = sx(q_min + 1.5, q_min, q_max, x0, axis_w)
        end = sx(q_max - 1.5, q_min, q_max, x0, axis_w)
        parts.append(path(f"M {start:.1f} {arrow_y:.1f} C {(start+end)/2:.1f} {arrow_y+28:.1f}, {(start+end)/2:.1f} {arrow_y+28:.1f}, {end:.1f} {arrow_y:.1f}", "#9b5b25", 1.5, marker=True))
        giant_signed = g * sf
        giant_key = reduce_rotation(giant_signed, giant_mod)
        parts.append(txt((start + end) / 2, arrow_y + 31, f"giant: +{g}*sf={giant_signed} -> key {giant_key}", size=9, anchor="middle", fill="#8a4d1f"))

    baby_keys = [reduce_rotation(q * sf, direct_mod) for q in range(q_min, q_min + g)]
    giant_keys = [reduce_rotation(g * i * sf, giant_mod) for i in range(row["b"])]
    summary_x = x0 + axis_w + 35
    parts.append(txt(summary_x, y + 24, "actual BSGS keys", size=10, weight=800))
    parts.append(txt(summary_x, y + 43, "baby:", size=9, weight=700, fill="#2e86ab"))
    parts.append(txt(summary_x + 44, y + 43, compact(baby_keys), size=9, fill="#2e86ab"))
    parts.append(txt(summary_x, y + 61, "giant:", size=9, weight=700, fill="#d77a30"))
    parts.append(txt(summary_x + 44, y + 61, compact(giant_keys), size=9, fill="#d77a30"))
    parts.append(txt(summary_x, y + 84, "point label:", size=9, weight=700, fill="#5d6d7c"))
    parts.append(txt(summary_x + 72, y + 84, "top=signed q*sf, bottom=direct key", size=9, fill="#5d6d7c"))


def render(log_slots):
    slots = 1 << log_slots
    m = (1 << LOG_N) * 2
    c2s_rows, p_enc = build_rows(log_slots, "C2S")
    s2c_rows, p_dec = build_rows(log_slots, "S2C")
    rows = c2s_rows + s2c_rows

    width = 1780
    row_h = 136
    height = 210 + row_h * len(rows)
    axis_x = 255
    axis_w = 960

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#9b5b25"/>',
        "</marker>",
        "</defs>",
        rect(0, 0, width, height, "#f7fafc", "none"),
        txt(32, 40, "FFT collapsed diagonal indices after scaling", size=22, weight=850),
        txt(32, 66, f"logN={LOG_N}, logSlots={log_slots}, slots={slots}, M={m}, M/4={m//4}, levelBudget={LEVEL_BUDGET}", size=12, fill="#52616f"),
        rect(32, 84, 1698, 82, "#ffffff", "#d9e2ea", rx=7),
        txt(54, 112, "How to read a row", size=12, weight=800),
        txt(190, 112, "Each dot is one diagonal of the collapsed FFT matrix: signed diag = q * scalingFactor.", size=12, fill="#2d3d4b"),
        txt(190, 134, "The green label is ReduceRotation(signed diag, direct modulus). Blue/orange show the two BSGS giant blocks.", size=12, fill="#2d3d4b"),
        txt(190, 154, "The right column is the actual BSGS key set used by that layer: baby keys plus giant keys.", size=12, fill="#2d3d4b"),
    ]

    y = 188
    parts.append(txt(34, y - 18, f"C2S params: layers={p_enc['layers']}, rem={p_enc['rem']}, numRot={p_enc['num_rot']}, g={p_enc['g']}, b={p_enc['b']}", size=11, weight=700))
    for row in c2s_rows:
        draw_row(parts, row, y, axis_x, axis_w)
        y += row_h

    y += 8
    parts.append(txt(34, y - 18, f"S2C params: layers={p_dec['layers']}, rem={p_dec['rem']}, numRot={p_dec['num_rot']}, g={p_dec['g']}, b={p_dec['b']}", size=11, weight=700))
    for row in s2c_rows:
        draw_row(parts, row, y, axis_x, axis_w)
        y += row_h

    parts.append("</svg>")
    out = RESULTS / f"fft_bsgs_key_indices_logslots{log_slots}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main():
    for log_slots in (15, 16):
        print(render(log_slots))


if __name__ == "__main__":
    main()
