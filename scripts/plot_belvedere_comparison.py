"""Create a compact two-fold Belvedere inference/benchmark comparison figure."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAIRS = ["2015_2020", "2020_2023", "2015_2023"]
PAIR_LABELS = ["2015 → 2020", "2020 → 2023", "2015 → 2023"]
FOLD_LABELS = ["Fold 0 — Gorner held out", "Fold 1 — Zmutt held out"]

METHODS = [
    ("control_image_only", "Image search", "CNN"),
    ("procedure_1_soft_rerank", "Soft terrain rerank", "CNN"),
    ("procedure_2_physics_loss", "Physics loss", "CNN"),
    ("zero_physics_control", "Zero-physics control", "CNN"),
    ("procedure_3_terrain_head", "Terrain head", "CNN"),
    ("control_image_only__local7", "Image search + local7", "Local context"),
    ("procedure_1_soft_rerank__local7", "Soft rerank + local7", "Local context"),
    ("procedure_2_physics_loss__local7", "Physics loss + local7", "Local context"),
    ("procedure_3_terrain_head__local7", "Terrain head + local7", "Local context"),
]
FROZEN = [
    ("ncc_gray", "Gray NCC"),
    ("ncc_gradient", "Gradient NCC"),
    ("sift", "SIFT"),
    ("disk_lightglue", "DISK-LightGlue"),
]
COLORS = {"CNN": "#3B6FB6", "Local context": "#2A9D70", "Frozen benchmark": "#E08B32"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    summary = args.root.resolve() / 'summary'
    model_rows = load_csv(summary / "all_method_metrics.csv")
    frozen_rows = load_csv(summary / "frozen_benchmarks.csv")
    fig, axes = plt.subplots(2, 3, figsize=(29, 18), sharex=True)
    fig.subplots_adjust(left=0.205, right=0.90, top=0.90, bottom=0.10, wspace=0.25, hspace=0.25)

    for fold in range(2):
        for col, (pair, pair_label) in enumerate(zip(PAIRS, PAIR_LABELS)):
            ax = axes[fold, col]
            entries: list[tuple[str, str, str, float, int, float]] = []
            for key, label, family in METHODS:
                match = next((r for r in model_rows if int(r["fold"]) == fold and r["pair"] == pair and r["method"] == key), None)
                if match:
                    n = int(float(match["n"]))
                    count = int(float(match["success_2m_count"]))
                    rate = count / n if n else 0.0
                    median = float(match["median_error_m"])
                    entries.append((label, family, f"{count}/{n}", rate, count, median))
            for key, label in FROZEN:
                match = next((r for r in frozen_rows if r["pair"] == f"belvedere_{pair}" and r["method"] == key), None)
                if match:
                    n = int(float(match["queries"]))
                    rate = float(match["success_at_2m_all_queries"])
                    count = round(rate * n)
                    median = float(match["median_error_m_accepted"] or "nan")
                    entries.append((label, "Frozen benchmark", f"{count}/{n}", rate, count, median))

            entries.reverse()
            y = np.arange(len(entries))
            rates = [e[3] for e in entries]
            ax.barh(y, rates, color=[COLORS[e[1]] for e in entries], height=0.72, edgecolor="white", linewidth=0.5)
            ax.set_yticks(y, [e[0] for e in entries], fontsize=11)
            ax.set_xlim(0, 1.19)
            ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)], fontsize=10)
            ax.grid(axis="x", color="#D9DEE7", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_title(pair_label, fontsize=16, weight="bold", pad=12)
            if col == 0:
                ax.set_ylabel(FOLD_LABELS[fold], fontsize=14, weight="bold", labelpad=15)
            for yy, entry in zip(y, entries):
                _, _, count_text, rate, _, median = entry
                ax.text(min(rate + 0.012, 1.015), yy, f"{count_text}  |  med {median:.1f} m", va="center", fontsize=9.5, color="#242A31")
            if fold == 1:
                ax.set_xlabel("Queries localized within 2 m (fraction of all queries)", fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in COLORS]
    fig.legend(handles, list(COLORS), loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=12)
    fig.suptitle("Belvedere inference methods and frozen benchmarks", fontsize=21, weight="bold", y=0.985)
    fig.text(0.5, 0.02, "Bars show success@2 m; labels show successes / total queries and median endpoint error (m). Frozen benchmark error is among accepted matches; coverage is reported in the source table.", ha="center", fontsize=10.5, color="#444")
    output = summary / "belvedere_methods_benchmarks_twofold.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
