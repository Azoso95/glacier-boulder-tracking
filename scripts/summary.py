"""Collect matched fold/Belvedere metrics and render comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
# ruff: isort: off
import matplotlib.pyplot as plt
# ruff: isort: on


FOLDS = {0: "gorner", 1: "zmutt"}
PAIRS = ("heldout_test", "2020_2023", "2015_2020", "2015_2023")
METHODS = (
    "control_image_only", "procedure_1_soft_rerank",
    "procedure_2_physics_loss", "zero_physics_control",
    "procedure_3_terrain_head",
    "control_image_only__local7", "procedure_1_soft_rerank__local7",
    "procedure_2_physics_loss__local7", "procedure_3_terrain_head__local7",
)
LABELS = {
    "control_image_only": "Search image",
    "procedure_1_soft_rerank": "Soft terrain",
    "procedure_2_physics_loss": "Physics loss",
    "zero_physics_control": "Zero-physics control",
    "procedure_3_terrain_head": "Terrain head",
    "control_image_only__local7": "Search+local7",
    "procedure_1_soft_rerank__local7": "Soft+local7",
    "procedure_2_physics_loss__local7": "Physics+local7",
    "procedure_3_terrain_head__local7": "Head+local7",
}


def pair_dir(study: Path, pair: str) -> Path:
    return study / ("test" if pair == "heldout_test" else f"belvedere_{pair}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Descriptive 95% binomial interval for one small evaluation set."""
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z / denominator * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for fold, heldout in FOLDS.items():
        study = root / "reports" / f"fold_{fold:02d}_{heldout}"
        for pair in PAIRS:
            folder = pair_dir(study, pair)
            result = read_json(folder / "complete.json")
            metrics = result["metrics"]
            metrics["zero_physics_control"] = read_json(folder / "zero_physics_metrics.json")["metrics"]
            for method in METHODS:
                if method not in metrics:
                    raise KeyError(f"missing {method}: {folder}")
                value = metrics[method]
                total = int(value.get("queries", value.get("n", 0)))
                if total < 1:
                    raise ValueError(f"missing query count in {folder}")
                successes = round(float(value["success_at_2m"]) * total)
                lower, upper = wilson_interval(successes, total)
                rows.append({
                    "fold": fold, "heldout": heldout,
                    "pair": pair, "method": method,
                    "n": total,
                    "radius_m": float(result["radius_m"]),
                    **{key: float(val) for key, val in value.items()
                       if isinstance(val, (int, float))},
                    "success_2m_count": successes,
                    "success_2m_wilson_lower": lower,
                    "success_2m_wilson_upper": upper,
                })
    frame = pd.DataFrame(rows)
    if frame.n.eq(0).any():
        raise ValueError("missing query count in result")
    return frame


def plot(frame: pd.DataFrame, metric: str, output: Path, ylabel: str,
         *, log: bool = False) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True,
                             sharex=True, sharey="row")
    colors = ["#0072B2", "#009E73", "#CC79A7", "#BBBBBB",
              "#D55E00", "#56B4E9", "#88CCAA", "#DD88BB", "#E69F00"]
    for fold, heldout in FOLDS.items():
        for col, pair in enumerate(PAIRS):
            axis = axes[fold, col]
            selected = frame.loc[(frame.fold == fold) & (frame.pair == pair)].set_index("method")
            values = np.array([selected.loc[method, metric] for method in METHODS], dtype=float)
            if metric.startswith("success_at_"):
                values *= 100
            axis.bar(np.arange(len(METHODS)), values, color=colors)
            axis.set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS],
                            rotation=60, ha="right", fontsize=8)
            axis.set_title(f"{heldout.title()} held out: " +
                           ("test" if pair == "heldout_test" else pair.replace("_", "–")))
            axis.grid(axis="y", alpha=0.2)
            if log:
                axis.set_yscale("log")
            if col == 0:
                axis.set_ylabel(ylabel)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    frame = collect(root)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_method_metrics.csv", index=False)
    (output / "all_method_metrics.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8"
    )
    plot(frame, "success_at_2m", output / "ablation_success_2m.png",
         "Queries within 2 m (%)")
    plot(frame, "median_error_m", output / "ablation_median_error.png",
         "Median endpoint error (m)", log=True)
    for fold, heldout in FOLDS.items():
        subset = frame.loc[(frame.fold == fold) & (frame.pair == "heldout_test")]
        print(heldout, subset[["method", "n", "success_at_2m", "median_error_m"]]
              .to_string(index=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
