"""Build the two manuscript folds from the six prepared survey pairs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from glaciertrack.data.splitting import _subset_hdf5


GLACIERS = ("miage", "ciardoney", "gorner", "zmutt")
HELDOUT = ("gorner", "zmutt")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def queries_from_prepared(prepared: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    observations = pd.read_parquet(prepared / "manifests/observations.parquet")
    surveys = pd.read_parquet(prepared / "manifests/surveys.parquet")
    pairs = pd.read_parquet(prepared / "manifests/survey_pairs.parquet")
    if set(surveys.glacier_id) != set(GLACIERS):
        raise ValueError(f"glacier coverage differs: {set(surveys.glacier_id)}")
    survey = surveys.set_index(["glacier_id", "survey_id"])
    frames = []
    pair_specs = []
    for pair in pairs.itertuples(index=False):
        glacier = str(pair.glacier_id)
        reference = str(pair.reference_survey_id)
        target = str(pair.target_survey_id)
        first = survey.loc[(glacier, reference)]
        second = survey.loc[(glacier, target)]
        source = observations.loc[
            (observations.glacier_id == glacier)
            & (observations.survey_id == reference)
            & (observations.status == "visible")
        ].set_index("track_id")
        destination = observations.loc[
            (observations.glacier_id == glacier)
            & (observations.survey_id == target)
            & (observations.status == "visible")
        ].set_index("track_id")
        if source.index.has_duplicates or destination.index.has_duplicates:
            raise ValueError(f"duplicate track in {pair.pair_id}")
        ids = sorted(set(source.index) & set(destination.index))
        if not ids:
            raise ValueError(f"no matched visible tracks for {pair.pair_id}")
        frame = pd.DataFrame({
            "pair_id": pair.pair_id,
            "track_id": ids,
            "query_id": [f"{pair.pair_id}:{identifier}" for identifier in ids],
            "glacier_id": glacier,
            "source_x_m": source.loc[ids].x_m.to_numpy(float),
            "source_y_m": source.loc[ids].y_m.to_numpy(float),
            "target_x_m": destination.loc[ids].x_m.to_numpy(float),
            "target_y_m": destination.loc[ids].y_m.to_numpy(float),
            "match_present": True, "source_valid": True,
            "source_crs": first.crs, "target_crs": second.crs,
            "reference_survey_id": reference,
            "source_survey_id": reference,
            "target_survey_id": target,
            "survey_pair": f"{reference}--{target}",
            "effective_gsd_m_per_px": max(float(first.native_gsd_x_m),
                                            float(second.native_gsd_x_m)),
        })
        frames.append(frame)
        pair_specs.append({
            "pair_id": pair.pair_id, "glacier_id": glacier,
            "reference_survey_id": reference, "target_survey_id": target,
            "source_raster": str(first.raster_uri),
            "target_raster": str(second.raster_uri),
            "target_aoi": str(second.aoi_uri),
            "query_count": len(ids),
        })
    return pd.concat(frames, ignore_index=True), pair_specs


def spatial_half(queries: pd.DataFrame) -> dict[str, str]:
    ordered = queries.drop_duplicates("track_id").sort_values("track_id")
    points = ordered[["source_x_m", "source_y_m"]].to_numpy(dtype=float, copy=True)
    points -= points.mean(axis=0)
    _, _, axes = np.linalg.svd(points, full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    order = np.argsort(points @ axis, kind="stable")
    labels = np.full(len(order), "test", dtype=object)
    labels[order[:len(order) // 2]] = "validation"
    return dict(zip(ordered.track_id, labels, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    build_folds(args.prepared.resolve(), args.output.resolve(), args.run_root.resolve(),
                args.sam2_checkpoint.resolve())


def build_folds(prepared: Path, output: Path, run_root: Path, sam2: Path) -> None:
    if output.exists():
        raise FileExistsError(f"new fold root required: {output}")
    queries, pairs = queries_from_prepared(prepared)
    samples = pd.read_parquet(prepared / "manifests/samples.parquet")
    if queries.query_id.duplicated().any() or samples.sample_id.duplicated().any():
        raise ValueError("query or sample IDs are not unique")
    expected_pairs = {
        "ciardoney:survey_2019--survey_2023",
        "ciardoney:survey_2019--survey_2026",
        "ciardoney:survey_2023--survey_2026",
        "miage:survey_2022--survey_2024",
        "gorner:survey_1999--survey_2005",
        "zmutt:survey_1999--survey_2005",
    }
    if set(queries.pair_id) != expected_pairs:
        raise ValueError(f"unexpected pairs: {sorted(set(queries.pair_id))}")
    configs = Path(__file__).resolve().parents[1] / "configs"
    templates = {name: yaml.safe_load((configs / f"{name}.yaml").read_text(encoding="utf-8"))
                 for name in ("patch", "search")}
    expected_digest = templates["patch"]["prompt"]["sam2"]["checkpoint_sha256"]
    if digest(sam2) != expected_digest:
        raise ValueError("SAM2 checkpoint does not match the study checkpoint")
    output.mkdir(parents=True)
    fold_reports = []
    for fold, heldout in enumerate(HELDOUT):
        heldout_queries = queries.loc[queries.glacier_id == heldout]
        assignment = spatial_half(heldout_queries)
        labels = np.where(
            queries.glacier_id.eq(heldout),
            queries.track_id.map(assignment).fillna("train"), "train",
        )
        assigned = queries.assign(
            split=labels,
            spatial_group=[f"{g}_{s}" for g, s in
                           zip(queries.glacier_id, labels, strict=True)],
        )
        if set(assigned.loc[assigned.glacier_id == heldout, "split"]) != {"validation", "test"}:
            raise ValueError("held-out glacier lacks validation or test")
        if (assigned.loc[assigned.glacier_id != heldout, "split"] != "train").any():
            raise ValueError("a training glacier was held out")
        fold_dir = output / f"fold_{fold:02d}_{heldout}"
        fold_dir.mkdir()
        assigned.to_parquet(fold_dir / "assignments.parquet", index=False)
        manifest = samples.drop(columns=["split", "spatial_group"], errors="ignore").merge(
            assigned[["pair_id", "track_id", "split", "spatial_group"]],
            on=["pair_id", "track_id"], how="left", validate="many_to_one",
        )
        if manifest.split.isna().any() or len(manifest) != len(samples):
            raise ValueError("some prepared samples have no query assignment")
        manifest.to_parquet(fold_dir / "samples.parquet", index=False)
        cache = fold_dir / "cache"
        cache.mkdir()
        for phase in ("train", "validation"):
            ids = manifest.loc[manifest.split == phase, "sample_id"].tolist()
            if not ids:
                raise ValueError(f"no {phase} samples for fold {fold}")
            _subset_hdf5(prepared / "cache/training.h5", cache / f"{phase}.h5", ids)
        for glacier in GLACIERS:
            for phase in ("train", "validation", "test"):
                selected = assigned.loc[
                    (assigned.glacier_id == glacier) & (assigned.split == phase)
                ]
                selected.to_parquet(fold_dir / f"{phase}__{glacier}.parquet", index=False)
                if phase == "test" and glacier == heldout:
                    selected.drop(columns=["target_x_m", "target_y_m", "match_present"]).to_parquet(
                        fold_dir / f"source__{glacier}.parquet", index=False
                    )
        search_pairs = []
        for pair in pairs:
            glacier = pair["glacier_id"]
            stem = f'{glacier}__{pair["reference_survey_id"]}--{pair["target_survey_id"]}'
            for phase in ("train", "validation", "test"):
                assigned.loc[
                    (assigned.pair_id == pair["pair_id"]) & (assigned.split == phase)
                ].to_parquet(fold_dir / f"{phase}__{stem}.parquet", index=False)
            search_pairs.append({
                "source_raster": pair["source_raster"],
                "target_raster": pair["target_raster"],
                "target_aoi": pair["target_aoi"],
                "train_queries": str(fold_dir / f"train__{stem}.parquet"),
                "validation_queries": str(fold_dir / f"validation__{stem}.parquet"),
            })
        project = {"name": "glacier-boulder-tracker", "seed": 2400 + fold,
                   "output_dir": str(run_root)}
        base = deepcopy(templates["patch"])
        search = deepcopy(templates["search"])
        for config in (base, search):
            config["project"] = project
            config["prompt"]["sam2"].update(
                checkpoint=str(sam2), cache_dir=str(run_root.parent / "mask_cache")
            )
        base["training"].update(
            run_name=f"fold_{fold:02d}_{heldout}_patch",
            train_hdf5_cache=str(cache / "train.h5"),
            validation_hdf5_cache=str(cache / "validation.h5"),
            sample_manifest=str(fold_dir / "samples.parquet"),
            validation_sample_manifest=str(fold_dir / "samples.parquet"),
        )
        search["search_training"].update(
            run_name=f"fold_{fold:02d}_{heldout}_search", pairs=search_pairs
        )
        for filename, config in (("base.yaml", base), ("search.yaml", search)):
            (fold_dir / filename).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        counts = assigned.groupby(["glacier_id", "survey_pair", "split"]).size()
        report = {
            "fold": fold, "heldout_glacier": heldout,
            "training_glaciers": [g for g in GLACIERS if g != heldout],
            "query_counts": {"|".join(key): int(count) for key, count in counts.items()},
            "sample_counts": {str(k): int(v) for k, v in manifest.split.value_counts().items()},
            "search_pair_count": len(search_pairs),
            "radius_m": 160.0,
        }
        (fold_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        fold_reports.append(report)
        print(json.dumps(report), flush=True)
    complete = {
        "schema_version": 1,
        "protocol": "two_anchored_folds_ciardoney_2019_2023_2026_all_chronological",
        "prepared_manifest_sha256": digest(prepared / "manifests/samples.parquet"),
        "pairs": pairs, "folds": fold_reports,
        "total_queries": len(queries),
    }
    (output / "complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "folds": len(fold_reports),
                      "total_queries": len(queries)}))


if __name__ == "__main__":
    main()
