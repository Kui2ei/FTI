from pathlib import Path


OUT = Path("results/fft_bsgs_diag_symmetry.svg")


def normal_layer(name, sf):
    return {
        "name": name,
        "sf": sf,
        "q_min": -7,
        "q_max": 7,
        "g": 8,
        "b": 2,
        "kind": "normal",
    }


def rem_layer(name, sf):
    return {
        "name": name,
        "sf": sf,
        "q_min": -15,
        "q_max": 15,
        "g": 16,
        "b": 2,
        "kind": "rem",
    }


def layer_points(layer):
    points = []
    q_min = layer["q_min"]
    g = layer["g"]
    for q in range(layer["q_min"], layer["q_max"] + 1):
        u = q - q_min
        i = u // g
        j = u % g
        points.append({"q": q, "u": u, "i": i, "j": j, "diag": q * layer["sf"]})
    return points


def sx(q, q_min, q_max, x0, w):
    return x0 + (q - q_min) * w / (q_max - q_min)


def text(x, y, s, size=11, anchor="start", weight=400, fill="#22313f"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def line(x1, y1, x2, y2, stroke="#aab6c2", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def circle(x, y, r, fill, stroke="#ffffff", width=1):
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def rect(x, y, w, h, fill, stroke="none", rx=0, opacity=1.0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'
    )


def draw_panel(parts, title, layers, x0, y0, w):
    parts.append(text(x0, y0, title, size=15, weight=700))
    parts.append(text(x0, y0 + 20, "each row: diag = q * scalingFactor", size=10, fill="#5f6f7d"))

    row_gap = 58
    axis_x = x0 + 170
    axis_w = w - 230
    y = y0 + 58

    for layer in layers:
        q_min, q_max = layer["q_min"], layer["q_max"]
        zero_x = sx(0, q_min, q_max, axis_x, axis_w)
        parts.append(text(x0, y + 4, layer["name"], size=11, weight=700))
        parts.append(text(x0, y + 20, f"sf={layer['sf']}", size=10, fill="#5f6f7d"))
        parts.append(text(x0, y + 34, f"diag=[{q_min}..{q_max}]*sf", size=9, fill="#7a8792"))

        parts.append(line(axis_x, y, axis_x + axis_w, y, stroke="#d3dce4", width=1.2))
        parts.append(line(zero_x, y - 17, zero_x, y + 17, stroke="#303c48", width=1.2))
        parts.append(text(zero_x, y + 33, "0", size=9, anchor="middle", fill="#303c48"))
        parts.append(text(axis_x, y + 33, str(q_min), size=9, anchor="middle", fill="#71808d"))
        parts.append(text(axis_x + axis_w, y + 33, str(q_max), size=9, anchor="middle", fill="#71808d"))

        for point in layer_points(layer):
            x = sx(point["q"], q_min, q_max, axis_x, axis_w)
            fill = "#2e86ab" if point["i"] == 0 else "#d77a30"
            if point["q"] == 0:
                fill = "#202a33"
            parts.append(circle(x, y, 5.2, fill))

        left_end = sx(min(0, q_min), q_min, q_max, axis_x, axis_w)
        left_start = sx(q_min, q_min, q_max, axis_x, axis_w)
        right_start = sx(1, q_min, q_max, axis_x, axis_w)
        right_end = sx(q_max, q_min, q_max, axis_x, axis_w)
        parts.append(line(left_start, y - 14, left_end, y - 14, stroke="#2e86ab", width=2.4))
        parts.append(line(right_start, y - 14, right_end, y - 14, stroke="#d77a30", width=2.4))

        if layer["kind"] == "normal":
            parts.append(text(axis_x + axis_w + 14, y - 4, "i=0: q=-7..0", size=9, fill="#2e86ab"))
            parts.append(text(axis_x + axis_w + 14, y + 11, "i=1: q=1..7", size=9, fill="#d77a30"))
        else:
            parts.append(text(axis_x + axis_w + 14, y - 4, "i=0: q=-15..0", size=9, fill="#2e86ab"))
            parts.append(text(axis_x + axis_w + 14, y + 11, "i=1: q=1..15", size=9, fill="#d77a30"))

        y += row_gap

    return y


def main():
    width, height = 1180, 920
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f8fafc"),
        text(36, 38, "Collapsed FFT diagonal symmetry under BSGS", size=21, weight=800),
        text(
            36,
            62,
            "Parameters shown: logN=16, logSlots=16, levelBudget=[5,5]. Blue/orange are the two giant blocks.",
            size=11,
            fill="#5f6f7d",
        ),
        rect(36, 82, 1108, 54, "#ffffff", stroke="#d7e0e7", rx=6),
        text(58, 108, "BSGS identity:", size=12, weight=700),
        text(
            160,
            108,
            "u = g*i + j,   q = half + u,   diag = q*sf = baby(j) + giant(i)",
            size=12,
            fill="#31404e",
        ),
        text(
            160,
            126,
            "normal layer: q=-7..7, g=8, b=2    |    rem layer: q=-15..15, gRem=16, bRem=2",
            size=11,
            fill="#5f6f7d",
        ),
    ]

    c2s = [
        normal_layer("C2S s=4", 8192),
        normal_layer("C2S s=3", 1024),
        normal_layer("C2S s=2", 128),
        normal_layer("C2S s=1", 16),
        rem_layer("C2S rem s=0", 1),
    ]

    s2c = [
        normal_layer("S2C s=0", 1),
        normal_layer("S2C s=1", 8),
        normal_layer("S2C s=2", 64),
        normal_layer("S2C s=3", 512),
        rem_layer("S2C rem s=4", 4096),
    ]

    draw_panel(parts, "CoeffsToSlots collapsed matrices", c2s, 46, 170, 1080)
    parts.append(line(36, 475, 1144, 475, stroke="#d7e0e7", width=1))
    draw_panel(parts, "SlotsToCoeffs collapsed matrices", s2c, 46, 515, 1080)

    parts.append(text(58, 892, "Interpretation:", size=11, weight=700))
    parts.append(
        text(
            150,
            892,
            "Every row is center-symmetric around q=0 before modular reduction; BSGS reuses the same baby stencil and shifts it by one giant offset.",
            size=11,
            fill="#5f6f7d",
        )
    )
    parts.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
