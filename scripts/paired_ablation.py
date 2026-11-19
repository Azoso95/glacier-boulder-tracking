"""Count per-query endpoint gains/losses for prespecified method transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = {0: "gorner", 1: "zmutt"}
PAIRS = ("heldout_test", "2020_2023", "2015_2020", "2015_2023")
TRANSITIONS = (
    ("control_image_only", "procedure_1_soft_rerank"),
    ("control_image_only", "procedure_2_physics_loss"),
    ("zero_physics_control", "procedure_2_physics_loss"),
    ("control_image_only", "procedure_3_terrain_head"),
    ("control_image_only", "control_image_only__local7"),
    ("procedure_1_soft_rerank", "procedure_1_soft_rerank__local7"),
)


def prediction_path(folder: Path, method: str) -> Path:
    if method == "zero_physics_control":
        return folder / "zero_physics_predictions.json"
    return folder / "predictions" / f"{method}.json"


def errors(folder: Path, method: str, truth: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(prediction_path(folder, method).read_text(encoding="utf-8"))
    positions = {record["query_id"]: np.asarray(record["prediction_xy_m"], dtype=float)
                 for record in records}
    if set(positions) != set(truth.index):
        raise ValueError(f"prediction IDs differ for {method} in {folder}")
    predicted = np.stack([positions[query_id] for query_id in truth.index])
    target = truth[["target_x_m", "target_y_m"]].to_numpy(dtype=float)
    return np.linalg.norm(predicted - target, axis=1), predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--auxiliary-dataset", type=Path,
                        help="Location of the unchanged Belvedere query tables")
    args = parser.parse_args()
    root = args.root.resolve()
    auxiliary_dataset = (args.auxiliary_dataset or root / "dataset").resolve()
    rows = []
    for fold, heldout in FOLDS.items():
        study = root / "reports" / f"fold_{fold:02d}_{heldout}"
        for pair in PAIRS:
            folder = study / ("test" if pair == "heldout_test" else f"belvedere_{pair}")
            truth_path = (root / "folds" / f"fold_{fold:02d}_{heldout}"
                          / f"test__{heldout}.parquet") if pair == "heldout_test" else (
                auxiliary_dataset / "inference" / "belvedere_multitemporal_2015"
                / pair / "evaluation_queries.parquet"
            )
            truth = pd.read_parquet(truth_path).set_index("query_id")
            methods = set().union(*(transition for transition in TRANSITIONS))
            values = {method: errors(folder, method, truth) for method in methods}
            for before, after in TRANSITIONS:
                old_errors, old_xy = values[before]
                new_errors, new_xy = values[after]
                old_success = old_errors <= 2.0
                new_success = new_errors <= 2.0
                rows.append({
                    "fold": fold, "heldout": heldout, "pair": pair,
                    "before": before, "after": after, "n": len(truth),
                    "old_success_count": int(old_success.sum()),
                    "new_success_count": int(new_success.sum()),
                    "gained": int((~old_success & new_success).sum()),
                    "lost": int((old_success & ~new_success).sum()),
                    "both_success": int((old_success & new_success).sum()),
                    "both_fail": int((~old_success & ~new_success).sum()),
                    "identical_endpoint_count": int(
                        (np.linalg.norm(new_xy - old_xy, axis=1) < 1e-6).sum()
                    ),
                    "median_error_change_m": float(np.median(new_errors - old_errors)),
                })
    frame = pd.DataFrame(rows)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "paired_ablation_changes.csv", index=False)
    (output / "paired_ablation_changes.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8"
    )
    print(frame.loc[frame.pair == "heldout_test", [
        "heldout", "before", "after", "n", "gained", "lost", "median_error_change_m"
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
