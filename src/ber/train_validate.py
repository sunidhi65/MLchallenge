from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import joblib

from ber.blocking import candidate_mapping
from ber.keyed_blocking import build_keyed_union_candidates
from ber.config import DEFAULT_RESOURCE_DIR, BlockingConfig, Paths, TrainingConfig
from ber.decision import enforce_bipartite_exclusivity, predictions_from_scores, tune_decision_rule, tune_source_specific_thresholds
from ber.features import build_pair_features
from ber.io import SOURCE_COLUMNS, ensure_dirs, read_ground_truth, read_train_source1, truth_dict
from ber.metrics import blocking_audit, macro_fbeta
from ber.modeling import downsample_training_pairs, scored_pairs, train_match_model
from ber.normalize import normalize_frame


def _sample_ids(
    truth: pd.DataFrame,
    max_s1: int | None,
    random_seed: int,
) -> list[str]:
    ids = truth["source1_entity_id"].to_numpy()
    if max_s1 is None or max_s1 >= len(ids):
        return ids.tolist()
    rng = np.random.default_rng(random_seed)
    matched = truth[truth["n_matches"].gt(0)]["source1_entity_id"].to_numpy()
    singleton = truth[truth["n_matches"].eq(0)]["source1_entity_id"].to_numpy()
    singleton_share = min(0.20, len(singleton) / max_s1) if len(singleton) else 0.0
    n_singleton = min(len(singleton), int(max_s1 * singleton_share))
    n_matched = max_s1 - n_singleton
    chosen = []
    if n_matched:
        chosen.extend(rng.choice(matched, size=min(n_matched, len(matched)), replace=False).tolist())
    if n_singleton:
        chosen.extend(rng.choice(singleton, size=n_singleton, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def _three_way_split(
    source1_ids: list[str],
    truth: dict[str, set[str]],
    random_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split by Source 1 entity into fit, tune, and holdout sets."""

    rng = np.random.default_rng(random_seed)
    matched = np.array([sid for sid in source1_ids if truth.get(sid)])
    singleton = np.array([sid for sid in source1_ids if not truth.get(sid)])

    def split_group(values: np.ndarray) -> tuple[list[str], list[str], list[str]]:
        values = values.copy()
        rng.shuffle(values)
        n = len(values)
        n_holdout = max(1, int(n * 0.20)) if n else 0
        n_tune = max(1, int(n * 0.20)) if n - n_holdout > 1 else 0
        holdout = values[:n_holdout].tolist()
        tune = values[n_holdout : n_holdout + n_tune].tolist()
        fit = values[n_holdout + n_tune :].tolist()
        return fit, tune, holdout

    fit_m, tune_m, hold_m = split_group(matched)
    fit_s, tune_s, hold_s = split_group(singleton)
    fit = fit_m + fit_s
    tune = tune_m + tune_s
    holdout = hold_m + hold_s
    rng.shuffle(fit)
    rng.shuffle(tune)
    rng.shuffle(holdout)
    return fit, tune, holdout


def _target_source_from_filename(filename: str) -> str:
    if "source2" in filename:
        return "S2"
    if "source3" in filename:
        return "S3"
    raise ValueError(f"Unexpected target filename: {filename}")


def _id_suffix_mod(entity_ids: pd.Series, modulo: int) -> pd.Series:
    suffix = entity_ids.str.split("-", n=1).str[1]
    numeric = pd.to_numeric(suffix, errors="coerce").fillna(-1).astype("int64")
    return numeric.mod(modulo).eq(0)


def _load_validation_target_pool(
    train_dir: Path,
    positive_target_ids: set[str],
    sample_modulo: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Load a memory-bounded target pool for train-only validation.

    The full training target side has more than 10M rows. For sampled
    validation runs we keep every true positive target for selected S1 rows,
    plus a deterministic sample of negatives from S2/S3. This makes the
    pipeline runnable on a laptop while preserving the key validation checks:
    blocking recall on known positives and F0.5 behavior with distractors.
    """

    if sample_modulo <= 0:
        raise ValueError("--target-sample-modulo must be positive")

    parts: list[pd.DataFrame] = []
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        path = train_dir / filename
        target_source = _target_source_from_filename(filename)
        kept = 0
        chunks = pd.read_csv(
            path,
            sep="\t",
            keep_default_na=False,
            usecols=SOURCE_COLUMNS,
            chunksize=chunk_size,
        )
        for chunk in chunks:
            mask = chunk["entity_id"].isin(positive_target_ids) | _id_suffix_mod(
                chunk["entity_id"], sample_modulo
            )
            selected = chunk.loc[mask].copy()
            if selected.empty:
                continue
            selected["target_source"] = target_source
            kept += len(selected)
            parts.append(selected)
        print(f"  {filename}: kept {kept:,} target rows")

    if not parts:
        raise RuntimeError("No target rows were loaded for validation.")
    targets = pd.concat(parts, ignore_index=True, sort=False).drop_duplicates("entity_id")
    missing_positive = positive_target_ids - set(targets["entity_id"])
    if missing_positive:
        raise RuntimeError(f"Target pool missed {len(missing_positive):,} positive target IDs.")
    return targets


def _write_predictions(path: Path, predictions: dict[str, list[str]]) -> None:
    rows = [
        {
            "source1_entity_id": source1_id,
            "matched_entity_ids": ",".join(entity_ids),
        }
        for source1_id, entity_ids in predictions.items()
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _evaluate_split(
    split_name: str,
    split_ids: list[str],
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    truths: dict[str, set[str]],
    decision_rule: dict[str, object],
) -> tuple[dict[str, float], dict[str, list[str]], pd.DataFrame]:
    scored = scored_pairs(candidates, probabilities)
    mode = decision_rule.get("mode", "global")
    if mode == "source_specific":
        threshold = {"S2": decision_rule["threshold_s2"], "S3": decision_rule["threshold_s3"]}
    else:
        threshold = decision_rule["threshold"]

    predictions = predictions_from_scores(scored, split_ids, threshold, decision_rule["cap"])
    if decision_rule.get("bipartite"):
        predictions = enforce_bipartite_exclusivity(predictions, scored)

    metrics = {
        f"{split_name}_macro_f05": macro_fbeta(predictions, truths, split_ids, beta=0.5),
        f"{split_name}_predicted_nonempty_rate": sum(bool(predictions.get(sid)) for sid in split_ids)
        / max(len(split_ids), 1),
    }
    if mode == "source_specific":
        metrics[f"{split_name}_threshold_s2"] = float(decision_rule["threshold_s2"])
        metrics[f"{split_name}_threshold_s3"] = float(decision_rule["threshold_s3"])
    else:
        metrics[f"{split_name}_threshold"] = float(decision_rule["threshold"])
    metrics[f"{split_name}_cap"] = -1.0 if decision_rule["cap"] is None else float(decision_rule["cap"])
    metrics[f"{split_name}_bipartite"] = bool(decision_rule.get("bipartite"))
    return metrics, predictions, scored


def run_train_validation(
    paths: Paths,
    blocking_config: BlockingConfig,
    training_config: TrainingConfig,
) -> dict[str, object]:
    ensure_dirs(paths.artifact_dir, paths.output_dir, paths.model_dir, paths.cache_dir)

    print("Reading train-only data...")
    source1 = read_train_source1(paths.train_dir)
    truth_frame = read_ground_truth(paths.train_dir)
    truths = truth_dict(truth_frame)

    selected_ids = _sample_ids(truth_frame, training_config.max_train_s1, training_config.random_seed)
    fit_ids, tune_ids, holdout_ids = _three_way_split(selected_ids, truths, training_config.random_seed)
    positive_target_ids = {target_id for source1_id in selected_ids for target_id in truths[source1_id]}

    print(
        f"Selected {len(selected_ids):,} S1 entities "
        f"({len(fit_ids):,} fit / {len(tune_ids):,} tune / {len(holdout_ids):,} holdout)."
    )

    print("Loading a memory-bounded target pool for validation...")
    target_pool = _load_validation_target_pool(
        paths.train_dir,
        positive_target_ids=positive_target_ids,
        sample_modulo=training_config.target_sample_modulo,
        chunk_size=training_config.target_chunk_size,
    )
    print(
        f"Target pool has {len(target_pool):,} rows "
        f"({len(positive_target_ids):,} selected positive target IDs included)."
    )

    print("Normalizing source records...")
    source1_norm = normalize_frame(source1[source1["entity_id"].isin(selected_ids)].copy())
    targets_norm = normalize_frame(target_pool)

    split_frames = {
        "fit": source1_norm[source1_norm["entity_id"].isin(fit_ids)].copy(),
        "tune": source1_norm[source1_norm["entity_id"].isin(tune_ids)].copy(),
        "holdout": source1_norm[source1_norm["entity_id"].isin(holdout_ids)].copy(),
    }

    candidates: dict[str, pd.DataFrame] = {}
    audits: dict[str, dict[str, float]] = {}
    for name, frame in split_frames.items():
        print(f"Building union candidates for {name}...")
        cand = build_keyed_union_candidates(frame, targets_norm, blocking_config)
        candidates[name] = cand
        audits[name] = blocking_audit(candidate_mapping(cand), truths, frame["entity_id"].tolist())
        cand.to_parquet(paths.cache_dir / f"{name}_candidates.parquet", index=False)
        print(f"{name} blocking audit: {audits[name]}")

    print("Building fit features...")
    fit_features, fit_labels = build_pair_features(candidates["fit"], source1_norm, targets_norm, truths)
    if fit_labels is None or fit_features.empty:
        raise RuntimeError("No fit features were produced. Candidate generation returned no pairs.")
    fit_features, fit_labels = downsample_training_pairs(
        fit_features,
        fit_labels,
        ratio=training_config.negative_downsample_ratio,
        random_seed=training_config.random_seed,
    )

    print(f"Training matcher on {len(fit_features):,} pairs...")
    model = train_match_model(fit_features, fit_labels, training_config.random_seed)

    print("Scoring tune and holdout candidates...")
    tune_features, _ = build_pair_features(candidates["tune"], source1_norm, targets_norm, truths)
    holdout_features, _ = build_pair_features(candidates["holdout"], source1_norm, targets_norm, truths)
    tune_prob = model.predict_proba(tune_features)
    holdout_prob = model.predict_proba(holdout_features)
    tune_scored = scored_pairs(candidates["tune"], tune_prob)

    print("Tuning F0.5 decision rule on tune split...")

    global_params, _ = tune_decision_rule(
        tune_scored,
        truths,
        tune_ids,
        thresholds=training_config.decision_thresholds,
        caps=training_config.max_match_caps,
    )
    global_preds_raw = predictions_from_scores(tune_scored, tune_ids, global_params["threshold"], global_params["cap"])
    global_preds_bip = enforce_bipartite_exclusivity(global_preds_raw, tune_scored)
    global_bip_score = macro_fbeta(global_preds_bip, truths, tune_ids, beta=0.5)

    source_params, _ = tune_source_specific_thresholds(
        tune_scored, truths, tune_ids,
        thresholds=training_config.decision_thresholds,
        cap=global_params["cap"],
        use_bipartite=True
    )

    candidates_rules = [
        ("global", global_params["macro_f05"], {"mode": "global", "threshold": global_params["threshold"], "cap": global_params["cap"], "bipartite": False}),
        ("global_bipartite", global_bip_score, {"mode": "global", "threshold": global_params["threshold"], "cap": global_params["cap"], "bipartite": True}),
        ("source_specific_bipartite", source_params["macro_f05"], {"mode": "source_specific", "threshold_s2": source_params["threshold_s2"], "threshold_s3": source_params["threshold_s3"], "cap": source_params["cap"], "bipartite": True}),
    ]

    best_rule_name, best_score, decision_rule = max(candidates_rules, key=lambda x: x[1])
    print(f"Selected decision rule: {best_rule_name} (score: {best_score:.6f})")

    tune_metrics, _, tune_scored = _evaluate_split(
        "tune",
        tune_ids,
        candidates["tune"],
        tune_prob,
        truths,
        decision_rule,
    )
    holdout_metrics, holdout_predictions, holdout_scored = _evaluate_split(
        "holdout",
        holdout_ids,
        candidates["holdout"],
        holdout_prob,
        truths,
        decision_rule,
    )

    _write_predictions(paths.output_dir / "validation_predictions.tsv", holdout_predictions)
    tune_scored.to_parquet(paths.output_dir / "tune_scored_pairs.parquet", index=False)
    holdout_scored.to_parquet(paths.output_dir / "holdout_scored_pairs.parquet", index=False)

    joblib.dump(model, paths.model_dir / "match_model.joblib")
    (paths.model_dir / "decision_rule.json").write_text(json.dumps(decision_rule, indent=2), encoding="utf-8")
    print(f"Wrote model to {paths.model_dir / 'match_model.joblib'} and decision rule to {paths.model_dir / 'decision_rule.json'}")

    metrics: dict[str, object] = {
        "selected_s1": len(selected_ids),
        "fit_s1": len(fit_ids),
        "tune_s1": len(tune_ids),
        "holdout_s1": len(holdout_ids),
        "target_pool_rows": len(targets_norm),
        "target_sample_modulo": training_config.target_sample_modulo,
        "best_decision_rule": decision_rule,
        "blocking": audits,
        "tune": tune_metrics,
        "holdout": holdout_metrics,
        "feature_columns": model.feature_columns,
    }
    metrics_path = paths.output_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")
    print(
        "Validation summary: "
        f"tune_macro_f05={tune_metrics['tune_macro_f05']:.6f}, "
        f"holdout_macro_f05={holdout_metrics['holdout_macro_f05']:.6f}, "
        f"rule={best_rule_name}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only validation pipeline for business entity resolution.")
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-s1", type=int, default=10_000, help="Sample this many Source 1 train rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--name-top-k", type=int, default=35)
    parser.add_argument("--address-top-k", type=int, default=35)
    parser.add_argument("--combined-top-k", type=int, default=45)
    parser.add_argument("--token-top-k", type=int, default=30)
    parser.add_argument(
        "--target-sample-modulo",
        type=int,
        default=500,
        help=(
            "For train-only sampled validation, keep all selected positive targets plus "
            "target IDs whose numeric suffix is divisible by this value. Larger values "
            "use less memory."
        ),
    )
    parser.add_argument("--target-chunk-size", type=int, default=500_000)
    parser.add_argument("--negative-ratio", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(resource_dir=args.resource_dir, artifact_dir=args.artifact_dir)
    blocking_config = BlockingConfig(
        name_top_k=args.name_top_k,
        address_top_k=args.address_top_k,
        combined_top_k=args.combined_top_k,
        token_top_k=args.token_top_k,
        max_candidates_per_entity=args.max_candidates,
    )
    training_config = TrainingConfig(
        random_seed=args.seed,
        max_train_s1=args.max_s1,
        target_sample_modulo=args.target_sample_modulo,
        target_chunk_size=args.target_chunk_size,
        negative_downsample_ratio=args.negative_ratio,
    )
    run_train_validation(paths, blocking_config, training_config)


if __name__ == "__main__":
    main()
