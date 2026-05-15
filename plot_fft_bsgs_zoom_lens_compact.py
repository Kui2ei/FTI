from pathlib import Path
import argparse
import math
import sys

import plot_fft_bsgs_zoom_lens as base
from plot_fft_bsgs_zoom_lens import (
    circle,
    line,
    path,
    rect,
    reduce_rotation,
    sx,
    txt,
)


RESULTS = Path("results")


def split_list(values):
    mid = (len(values) + 1) // 2
    first = "[" + ", ".join(str(v) for v in values[:mid]) + ("," if len(values) > mid else "]")
    second = " " + ", ".join(str(v) for v in values[mid:]) + "]" if len(values) > mid else ""
    return first, second


def full_list(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


def global_ticks(global_lo, global_hi, data_lo, data_hi):
    span = global_hi - global_lo
    raw_step = max(1, span / 8)
    step = 1 << max(0, round(math.log2(raw_step)))
    lo_tick = math.ceil(global_lo / step) * step
    hi_tick = math.floor(global_hi / step) * step
    ticks = {0, data_lo, data_hi}
    t = lo_tick
    while t <= hi_tick:
        ticks.add(int(t))
        t += step
    return sorted(t for t in ticks if global_lo <= t <= global_hi)


def draw_header(parts, width, log_n, log_slots, slots, m, level_budget, direction, global_lo, global_hi, data_lo, data_hi, graph_x, graph_w):
    suffix = f" ({direction})" if direction else ""
    parts.append(txt(34, 48, f"Zoom-lens view of scaled FFT diagonal indices{suffix}", size=36, weight=850))
    parts.append(
        txt(
            34,
            84,
            f"logN={log_n}, logSlots={log_slots}, slots={slots}, M={m}, M/4={m//4}, levelBudget={level_budget}",
            size=19,
            fill="#556575",
        )
    )
    parts.append(rect(34, 106, width - 68, 54, "#ffffff", "#d8e1e9", rx=7))
    parts.append(txt(56, 141, "Orange window on the shared number line is enlarged directly below each row.", size=20, weight=750))
    parts.append(txt(graph_x, 193, "global signed-diag scale", size=20, weight=800))
    axis_y = 216
    parts.append(line(graph_x, axis_y, graph_x + graph_w, axis_y, "#71808d", 2.0))
    for t in global_ticks(global_lo, global_hi, data_lo, data_hi):
        if global_lo <= t <= global_hi:
            xx = sx(t, global_lo, global_hi, graph_x, graph_w)
            major = t in (data_lo, 0, data_hi)
            parts.append(line(xx, axis_y - (14 if major else 8), xx, axis_y + (14 if major else 8), "#536575", 1.3))
            parts.append(txt(xx, axis_y + (38 if major else 30), str(t), size=14, fill="#536575", anchor="middle"))


def draw_row(parts, row, y, global_lo, global_hi, graph_x, graph_w, note_x, note_w):
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

    card_h = 190
    axis_y = y + 64
    zoom_y = y + 129

    parts.append(rect(24, y, 1728, card_h, "#ffffff", "#d8e1e9", rx=7))
    parts.append(txt(42, y + 36, row["name"], size=26, weight=850))
    parts.append(txt(42, y + 71, f"sf={row['sf']}", size=18, fill="#596a78"))
    parts.append(txt(42, y + 101, f"diag={row['half']}..{row['half'] + row['num_rot'] - 1} * sf", size=17, fill="#6c7a86"))
    parts.append(txt(42, y + 133, f"zoom x{zoom_factor:.1f}", size=18, weight=750, fill="#8a4d1f"))

    parts.append(line(graph_x, axis_y, graph_x + graph_w, axis_y, "#dce5ec", 1.2))
    win_x1 = sx(lo, global_lo, global_hi, graph_x, graph_w)
    win_x2 = sx(hi, global_lo, global_hi, graph_x, graph_w)
    parts.append(rect(win_x1, axis_y - 13, max(2, win_x2 - win_x1), 26, "#f2c078", "#d77a30", rx=3, opacity=0.38))
    for t in (-65536, -32768, 0, 32768, 65536):
        if global_lo <= t <= global_hi:
            xx = sx(t, global_lo, global_hi, graph_x, graph_w)
            parts.append(line(xx, axis_y - 18, xx, axis_y + 18, "#edf2f6" if t else "#334150", 1.1))
    for q, v, _key in zip(row["qs"], signed, keys):
        xx = sx(v, global_lo, global_hi, graph_x, graph_w)
        u = q - row["half"]
        is_baby_index = u < row["g"]
        fill = "#2e86ab" if is_baby_index else "#5f554d"
        opacity = 0.88 if is_baby_index else 0.42
        if q == 0:
            fill = "#202a33"
            opacity = 0.95
        parts.append(circle(xx, axis_y, 4.2, fill, opacity=opacity))

    parts.append(
        path(
            f"M {win_x1:.1f} {axis_y+15:.1f} C {win_x1:.1f} {axis_y+25:.1f}, {graph_x:.1f} {zoom_y-23:.1f}, {graph_x:.1f} {zoom_y-10:.1f}",
            "#b56a2b",
            1.25,
            dash="4 4",
        )
    )
    parts.append(
        path(
            f"M {win_x2:.1f} {axis_y+15:.1f} C {win_x2:.1f} {axis_y+25:.1f}, {graph_x+graph_w:.1f} {zoom_y-23:.1f}, {graph_x+graph_w:.1f} {zoom_y-10:.1f}",
            "#b56a2b",
            1.25,
            dash="4 4",
        )
    )

    parts.append(line(graph_x, zoom_y, graph_x + graph_w, zoom_y, "#9aa9b6", 1.35))
    for t in [lo, 0, hi]:
        if zlo <= t <= zhi:
            xx = sx(t, zlo, zhi, graph_x, graph_w)
            parts.append(line(xx, zoom_y - 17, xx, zoom_y + 18, "#1f2d3a" if t == 0 else "#8a98a5", 1.1))

    for q, v, key in zip(row["qs"], signed, keys):
        xx = sx(v, zlo, zhi, graph_x, graph_w)
        u = q - row["half"]
        is_baby_index = u < row["g"]
        fill = "#2e86ab" if is_baby_index else "#5f554d"
        label_fill = "#263746" if is_baby_index else "#6a625a"
        key_fill = "#006b35" if is_baby_index else "#5f554d"
        point_opacity = 1.0 if is_baby_index else 0.55
        if q == 0:
            fill = "#202a33"
            label_fill = "#263746"
            key_fill = "#006b35"
            point_opacity = 1.0
        stroke = "#1f2933" if key in (0, 32768) else "#ffffff"
        parts.append(circle(xx, zoom_y, 7.6, fill, stroke=stroke, width=1.6, opacity=point_opacity))
        parts.append(txt(xx, zoom_y - 24, str(v), size=16, weight=700, fill=label_fill, anchor="middle", rotate=-55))
        parts.append(txt(xx, zoom_y + 51, str(key), size=16, weight=700, fill=key_fill, anchor="middle", rotate=-55))

    giant_signed = row["g"] * row["sf"]
    giant_keys = [reduce_rotation(row["g"] * i * row["sf"], row["giant_mod"]) for i in range(row["b"])]
    baby_keys = [reduce_rotation((row["half"] + j) * row["sf"], row["direct_mod"]) for j in range(row["g"])]
    baby_line1, baby_line2 = split_list(baby_keys)
    parts.append(txt(note_x, y + 44, f"g={row['g']}, b={row['b']}; step={giant_signed}", size=17, weight=800, fill="#8a4d1f"))
    parts.append(txt(note_x, y + 78, f"giant: {full_list(giant_keys)}", size=15, fill="#8a4d1f"))
    parts.append(txt(note_x, y + 112, f"baby: {baby_line1}", size=15, fill="#2e86ab"))
    if baby_line2:
        parts.append(txt(note_x + 48, y + 140, baby_line2, size=15, fill="#2e86ab"))
        label_y = y + 168
    else:
        label_y = y + 146
    parts.append(txt(note_x, label_y, "dimmed = not in baby list", size=13, fill="#6b625a"))


def render(log_n, log_slots, level_budget, direction=None):
    base.LOG_N = log_n
    base.LEVEL_BUDGET = level_budget
    slots = 1 << log_slots
    m = (1 << log_n) * 2
    c2s, p1 = base.make_rows(log_slots, "C2S")
    s2c, p2 = base.make_rows(log_slots, "S2C")
    if direction == "C2S":
        sections = [("C2S", c2s, p1)]
    elif direction == "S2C":
        sections = [("S2C", s2c, p2)]
    else:
        sections = [("C2S", c2s, p1), ("S2C", s2c, p2)]
    rows = [row for _name, section_rows, _p in sections for row in section_rows]
    data_lo = min(min(r["signed"]) for r in rows)
    data_hi = max(max(r["signed"]) for r in rows)
    pad = max(1, int((data_hi - data_lo) * 0.04))
    global_lo = data_lo - pad
    global_hi = data_hi + pad

    width = 1780
    row_h = 210
    graph_x = 235
    graph_w = 1165
    note_x = 1425
    note_w = width - note_x - 36
    y = 305
    height = y + len(rows) * row_h + 68 + (10 if len(sections) > 1 else 0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f7fafc", "none"),
    ]
    draw_header(parts, width, log_n, log_slots, slots, m, level_budget, direction, global_lo, global_hi, data_lo, data_hi, graph_x, graph_w)

    for idx, (section_name, section_rows, p) in enumerate(sections):
        if idx:
            y += 10
        parts.append(txt(42, y - 15, f"{section_name}: layers={p['layers']}, rem={p['rem']}, g={p['g']}, b={p['b']}", size=17, weight=850))
        for row in section_rows:
            draw_row(parts, row, y, global_lo, global_hi, graph_x, graph_w, note_x, note_w)
            y += row_h

    parts.append("</svg>")
    direction_part = f"_{direction.lower()}" if direction else ""
    out = RESULTS / f"fft_bsgs_zoom_lens_logN{log_n}_logslots{log_slots}{direction_part}_compact.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-n", type=int, default=16)
    parser.add_argument("--log-slots", type=int, default=16)
    parser.add_argument("--level-budget", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--direction", choices=["C2S", "S2C", "both"], default="both")
    args = parser.parse_args(sys.argv[1:])
    directions = ["C2S", "S2C"] if args.direction == "both" else [args.direction]
    for value in directions:
        print(render(args.log_n, args.log_slots, args.level_budget, value))


if __name__ == "__main__":
    main()
