"""
Standalone Pareto intuition plot for rotation-key co-design.

Reads results/codesign_results.csv and visualizes the tradeoff between
new rotation keys and online rotations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "codesign_results.csv"

PALETTE = {
    "bg": "#FFFFFF",
    "ink": "#25313B",
    "muted": "#66727C",
    "grid": "#D8E0E6",
    "blue": "#2F5F88",
    "cyan": "#1C9CC8",
    "green": "#2E8B57",
    "orange": "#D48A2A",
    "red": "#C94C4C",
    "purple": "#7D6A9A",
    "grey_mid": "#AAB6BF",
}


COLORS = {
    "multi-2": PALETTE["blue"],
    "multi-2-conj": PALETTE["cyan"],
    "multi-3": PALETTE["orange"],
    "multi-3-conj": PALETTE["green"],
    "multi-4": PALETTE["purple"],
    "multi-4-conj": PALETTE["red"],
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 420,
        }
    )


def fallback_points():
    return [
        {"label": "multi-4-conj", "steps": "8 x 16 x 16 x 16", "new": 16, "online": 4365, "beta": None, "kind": "candidate"},
        {"label": "multi-2-conj", "steps": "64 x 512", "new": 106, "online": 2703, "beta": None, "kind": "candidate"},
        {"label": "multi-2-conj", "steps": "16384 x 2", "new": 1382, "online": 1420, "beta": None, "kind": "candidate"},
    ]


def load_points():
    if not CSV_PATH.exists():
        return fallback_points(), []

    candidates = {}
    best = []
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row.get("experiment")
            if kind not in {"beta_sweep_candidate", "beta_sweep_best"}:
                continue
            if not row.get("new_count") or not row.get("online_rotations_total"):
                continue
            point = {
                "label": row["label"],
                "steps": row["steps"],
                "new": int(float(row["new_count"])),
                "online": int(float(row["online_rotations_total"])),
                "beta": float(row["beta"]) if row.get("beta") else None,
                "kind": kind,
            }
            if kind == "beta_sweep_candidate":
                key = (point["label"], point["steps"], point["new"], point["online"])
                candidates[key] = point
            else:
                best.append(point)
    return list(candidates.values()) or fallback_points(), best


def pareto_frontier(points):
    unique = sorted({(p["online"], p["new"], p["label"], p["steps"]) for p in points}, key=lambda t: (t[0], t[1]))
    frontier = []
    best_new = float("inf")
    for online, new, label, steps in unique:
        if new < best_new:
            frontier.append((online, new, label, steps))
            best_new = new
    return frontier


def unique_best_path(best):
    seen = set()
    path = []
    for p in sorted(best, key=lambda item: item["beta"] if item["beta"] is not None else -1):
        key = (p["online"], p["new"], p["label"], p["steps"])
        if key in seen:
            continue
        seen.add(key)
        path.append(p)
    return path


def annotate_anchor(ax, text, xy, xytext):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="offset points",
        fontsize=6.7,
        color=PALETTE["ink"],
        arrowprops=dict(arrowstyle="-", color=PALETTE["grey_mid"], linewidth=0.65),
        zorder=5,
    )


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(exist_ok=True)
    candidates, best = load_points()

    fig, ax = plt.subplots(figsize=(6.2, 4.0), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    for label in sorted({p["label"] for p in candidates}):
        subset = [p for p in candidates if p["label"] == label]
        ax.scatter(
            [p["online"] for p in subset],
            [p["new"] for p in subset],
            s=24,
            color=COLORS.get(label, PALETTE["muted"]),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.78,
            label=label,
            zorder=3,
        )

    frontier = pareto_frontier(candidates)
    if frontier:
        ax.plot(
            [p[0] for p in frontier],
            [p[1] for p in frontier],
            color=PALETTE["ink"],
            linewidth=1.15,
            zorder=2,
            label="Pareto frontier",
        )

    best_path = unique_best_path(best)
    if best_path:
        ax.plot(
            [p["online"] for p in best_path],
            [p["new"] for p in best_path],
            color=PALETTE["ink"],
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            zorder=4,
        )
        ax.scatter(
            [p["online"] for p in best_path],
            [p["new"] for p in best_path],
            marker="s",
            s=30,
            facecolor="white",
            edgecolor=PALETTE["ink"],
            linewidth=0.8,
            zorder=5,
            label=r"best under $\beta$",
        )

    anchors = [
        ("low-key\n16 new keys", (4365, 16), (-110, 20)),
        ("balanced\n106 new keys", (2703, 106), (10, -34)),
        ("low-online\n1420 rotations", (1420, 1382), (8, -6)),
    ]
    for text, xy, xytext in anchors:
        annotate_anchor(ax, text, xy, xytext)

    ax.annotate(
        r"increasing $\beta$ prioritizes online latency",
        xy=(1500, 900),
        xytext=(4200, 22),
        fontsize=7.0,
        color=PALETTE["muted"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], linewidth=0.8),
    )

    ax.set_title("Pareto Intuition: Rotation-Key Storage vs. Online Rotations", loc="left", pad=8, fontweight="bold")
    ax.set_xlabel("online rotations")
    ax.set_ylabel("new rotation keys")
    ax.set_yscale("log")
    ax.grid(True, which="both", color=PALETTE["grid"], linewidth=0.48, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grey_mid"])
    ax.spines["bottom"].set_color(PALETTE["grey_mid"])
    ax.tick_params(colors=PALETTE["muted"])
    ax.legend(loc="upper right", frameon=True, edgecolor=PALETTE["grid"], framealpha=0.95, ncol=1)

    out_svg = OUT_DIR / "codesign_pareto_intuition.svg"
    out_pdf = OUT_DIR / "codesign_pareto_intuition.pdf"
    out_png = OUT_DIR / "codesign_pareto_intuition.png"
    for path in [out_svg, out_pdf, out_png]:
        fig.savefig(path, facecolor=PALETTE["bg"], edgecolor="none", bbox_inches="tight", pad_inches=0.055)
    plt.close(fig)
    print(f"[OK] saved {out_svg}")
    print(f"[OK] saved {out_pdf}")
    print(f"[OK] saved {out_png}")


if __name__ == "__main__":
    main()
