#!/usr/bin/env python3
"""Retrain the matcher on REAL full-scale candidate distributions.

measure_recall.py proved that the existing model + decision rule, frozen at
train time on a modulo-sampled (artificially easy) target pool, does not
transfer well to the real full-scale candidate distribution: raising top_k
recovers oracle headroom but *lowers* the realized score, because more real
distractors reach a threshold tuned for a smaller, cleaner set. This script
retrains on real full-scale candidates instead, so the model and decision
rule actually see what they'll face at inference time.

Writes to a NEW artifact dir (artifacts_recall_v3/), never touching
artifacts_final/models/ (which backs the validated 0.602 submission).

Memory-bounded like measure_recall.py / make_final_submission.py: one
country's full target pool + index in memory at a time, freed before the
next; only the much smaller extracted feature/label frames persist across
countries.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ber.blocking import candidate_mapping, combine_candidate_frames
from ber.config import BlockingConfig, TrainingConfig
from ber.decision import tune_decision_rule
from ber.features import build_pair_features
from ber.io import SOURCE_COLUMNS, read_ground_truth, read_train_source1, truth_dict
from ber.keyed_blocking import build_country_index, score_left_batch
from ber.metrics import blocking_audit, macro_fbeta
from ber.modeling import downsample_training_pairs, scored_pairs, train_match_model
from ber.normalize import normalize_frame

RAW_COUNTRY = {"india": "India", "us": "US"}
MEMORY_CEILING_GB = 11.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resource-dir", type=Path, default=Path("student_resource"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts_recall_v3"))
    p.add_argument("--output-dir", type=Path, default=Path("output_recall_v3"))
    p.add_argument("--n-s1-per-country", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=5_000)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--candidate-budget", type=int, default=150)
    p.add_argument("--target-chunk-size", type=int, default=500_000)
    return p.parse_args()


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**3) if sys.platform == "darwin" else raw / (1024**2)


def make_logger(log_path: Path):
    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def log_mem(label: str, log) -> None:
    rss = peak_rss_gb()
    log(f"  [mem] {label}: peak RSS so far = {rss:.2f} GB")
    if rss > MEMORY_CEILING_GB:
        log(f"  [mem] ABORTING at {label}: {rss:.2f} GB > {MEMORY_CEILING_GB} GB ceiling")
        raise MemoryError(f"Peak RSS {rss:.2f} GB exceeded ceiling at {label}")


def three_way_split(source1_ids, truth, seed):
    rng = np.random.default_rng(seed)
    matched = np.array([sid for sid in source1_ids if truth.get(sid)])
    singleton = np.array([sid for sid in source1_ids if not truth.get(sid)])

    def split_group(values):
        values = values.copy()
        rng.shuffle(values)
        n = len(values)
        n_holdout = max(1, int(n * 0.20)) if n else 0
        n_tune = max(1, int(n * 0.20)) if n - n_holdout > 1 else 0
        holdout = values[:n_holdout].tolist()
        tune = values[n_holdout:n_holdout + n_tune].tolist()
        fit = values[n_holdout + n_tune:].tolist()
        return fit, tune, holdout

    fit_m, tune_m, hold_m = split_group(matched)
    fit_s, tune_s, hold_s = split_group(singleton)
    fit, tune, holdout = fit_m + fit_s, tune_m + tune_s, hold_m + hold_s
    rng.shuffle(fit); rng.shuffle(tune); rng.shuffle(holdout)
    return fit, tune, holdout


def load_full_country_targets(train_dir: Path, raw_country: str, chunk_size: int, log) -> pd.DataFrame:
    parts = []
    for filename, tag in (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")):
        kept = 0
        for chunk in pd.read_csv(train_dir / filename, sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS, chunksize=chunk_size):
            sel = chunk.loc[chunk["country"] == raw_country].copy()
            if not sel.empty:
                sel["target_source"] = tag
                kept += len(sel)
                parts.append(sel)
        log(f"  {filename}: kept {kept:,} rows for country={raw_country}")
    targets = pd.concat(parts, ignore_index=True)
    del parts
    out = normalize_frame(targets)
    del targets
    return out


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "models").mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "output").mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "RECALL_RUN_LOG.md"
    log = make_logger(log_path)

    log(f"=== Retrain-on-real-scale starting (n_s1_per_country={args.n_s1_per_country:,}, top_k={args.top_k}, budget={args.candidate_budget}) ===")
    t_all = time.time()

    truth_frame_full = read_ground_truth(args.resource_dir / "dataset" / "train")
    truths_all = truth_dict(truth_frame_full)
    del truth_frame_full

    fit_frames, tune_frames, holdout_frames = [], [], []
    fit_ids_all, tune_ids_all, holdout_ids_all = [], [], []
    source1_norm_parts = []
    targets_norm_by_country = {}  # only kept transiently per-country; deleted after each country's features are built

    for country, raw_country in RAW_COUNTRY.items():
        log(f"--- Country: {country} ---")
        t_c = time.time()
        source1 = read_train_source1(args.resource_dir / "dataset" / "train")
        source1 = source1.loc[source1["country"] == raw_country].reset_index(drop=True)
        rng = np.random.default_rng(args.seed)
        n = min(args.n_s1_per_country, len(source1))
        idx = rng.choice(len(source1), size=n, replace=False)
        sample = source1.iloc[idx].reset_index(drop=True)
        sample_norm = normalize_frame(sample)
        sample_ids = sample_norm["entity_id"].tolist()
        log(f"Sampled {len(sample_ids):,} S1 rows for {country}")

        fit_c, tune_c, holdout_c = three_way_split(sample_ids, truths_all, args.seed)
        log(f"Split: fit={len(fit_c):,} tune={len(tune_c):,} holdout={len(holdout_c):,}")

        targets_norm = load_full_country_targets(args.resource_dir / "dataset" / "train", raw_country, args.target_chunk_size, log)
        log(f"Full {country} target pool: {len(targets_norm):,} rows")
        log_mem(f"after loading {country} target pool", log)

        t0 = time.time()
        country_index = build_country_index(targets_norm)
        log(f"Index build time={time.time()-t0:.1f}s")
        log_mem(f"after building {country} index", log)

        t0 = time.time()
        raw_frames = []
        for i in range(0, len(sample_norm), args.batch_size):
            batch = sample_norm.iloc[i:i + args.batch_size]
            scored = score_left_batch(batch, country_index, top_k=args.top_k, candidate_budget=args.candidate_budget)
            if not scored.empty:
                raw_frames.append(scored)
        log(f"Scoring all {len(sample_norm):,} {country} rows took {time.time()-t0:.1f}s")
        candidates = combine_candidate_frames(raw_frames, BlockingConfig(max_candidates_per_entity=args.top_k + 20))
        del raw_frames
        log_mem(f"after combining {country} candidates", log)

        audit = blocking_audit(candidate_mapping(candidates), truths_all, sample_ids)
        log(f"{country} full-sample blocking audit (top_k={args.top_k}): {json.dumps(audit)}")

        cand_fit = candidates[candidates["source1_entity_id"].isin(set(fit_c))]
        cand_tune = candidates[candidates["source1_entity_id"].isin(set(tune_c))]
        cand_holdout = candidates[candidates["source1_entity_id"].isin(set(holdout_c))]

        feat_fit, lab_fit = build_pair_features(cand_fit, sample_norm, targets_norm, truth=truths_all)
        feat_tune, _ = build_pair_features(cand_tune, sample_norm, targets_norm, truth=None)
        feat_holdout, _ = build_pair_features(cand_holdout, sample_norm, targets_norm, truth=None)
        log_mem(f"after building {country} features", log)

        if feat_fit is not None and not feat_fit.empty:
            fit_frames.append((feat_fit, lab_fit))
        tune_frames.append((cand_tune.reset_index(drop=True), feat_tune))
        holdout_frames.append((cand_holdout.reset_index(drop=True), feat_holdout))
        fit_ids_all.extend(fit_c)
        tune_ids_all.extend(tune_c)
        holdout_ids_all.extend(holdout_c)

        del targets_norm, country_index, candidates, cand_fit, cand_tune, cand_holdout, sample, source1
        gc.collect()
        log_mem(f"after freeing {country} target pool/index", log)
        log(f"=== Country {country} feature-building done in {time.time()-t_c:.1f}s ===")

    log("--- Training on combined fit set (India + US) ---")
    fit_features = pd.concat([f for f, _ in fit_frames], ignore_index=True)
    fit_labels = pd.concat([l for _, l in fit_frames], ignore_index=True)
    del fit_frames
    training_config = TrainingConfig()
    fit_features, fit_labels = downsample_training_pairs(fit_features, fit_labels, ratio=training_config.negative_downsample_ratio, random_seed=args.seed)
    log(f"Training on {len(fit_features):,} pairs ({int(fit_labels.sum()):,} positive)...")
    t0 = time.time()
    model = train_match_model(fit_features, fit_labels, args.seed)
    log(f"Training done in {time.time()-t0:.1f}s")
    log_mem("after training", log)

    log("--- Tuning decision rule on combined tune set ---")
    tune_candidates = pd.concat([c for c, _ in tune_frames], ignore_index=True)
    tune_features = pd.concat([f for _, f in tune_frames], ignore_index=True)
    del tune_frames
    tune_prob = model.predict_proba(tune_features)
    tune_scored = scored_pairs(tune_candidates, tune_prob)
    best_params, _ = tune_decision_rule(
        tune_scored, truths_all, tune_ids_all,
        thresholds=training_config.decision_thresholds, caps=training_config.max_match_caps,
    )
    threshold, cap = float(best_params["threshold"]), best_params["cap"]
    log(f"Tuned decision rule: threshold={threshold}, cap={cap}, tune_macro_f05={best_params['macro_f05']:.6f}")

    log("--- Evaluating on combined holdout set (the real, honest number) ---")
    holdout_candidates = pd.concat([c for c, _ in holdout_frames], ignore_index=True)
    holdout_features = pd.concat([f for _, f in holdout_frames], ignore_index=True)
    del holdout_frames
    holdout_prob = model.predict_proba(holdout_features)
    holdout_scored = scored_pairs(holdout_candidates, holdout_prob)
    from ber.decision import predictions_from_scores
    holdout_predictions = predictions_from_scores(holdout_scored, holdout_ids_all, threshold, cap)
    holdout_macro_f05 = macro_fbeta(holdout_predictions, truths_all, holdout_ids_all, beta=0.5)
    log(f"*** NEW real-scale holdout macro F0.5 = {holdout_macro_f05:.6f} (n={len(holdout_ids_all):,}, India+US combined) ***")
    log(f"For comparison: original modulo-sampled holdout was 0.945; frozen-model realistic scores measured earlier were india=0.6433-0.6458, us=0.7411")

    joblib.dump(model, args.artifact_dir / "models" / "match_model.joblib")
    (args.artifact_dir / "models" / "decision_rule.json").write_text(
        json.dumps({"threshold": threshold, "cap": cap}, indent=2), encoding="utf-8"
    )
    metrics = {
        "n_s1_per_country": args.n_s1_per_country,
        "top_k": args.top_k,
        "candidate_budget": args.candidate_budget,
        "fit_pairs": len(fit_features),
        "tune_s1": len(tune_ids_all),
        "holdout_s1": len(holdout_ids_all),
        "tuned_threshold": threshold,
        "tuned_cap": cap,
        "tune_macro_f05": best_params["macro_f05"],
        "holdout_macro_f05": holdout_macro_f05,
        "total_time_s": time.time() - t_all,
        "peak_rss_gb": peak_rss_gb(),
    }
    (args.artifact_dir / "output" / "retrain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Wrote model+rule to {args.artifact_dir / 'models'}, metrics to {args.artifact_dir / 'output' / 'retrain_metrics.json'}")
    log(f"=== Retrain-on-real-scale done in {time.time()-t_all:.1f}s ===")


if __name__ == "__main__":
    main()
