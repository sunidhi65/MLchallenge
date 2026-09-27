#!/usr/bin/env python3
"""Retrain on real full-scale candidates with all V5 changes in place:
transliteration (ber.normalize), frequency-capped multi-key blocking
(ber.keyed_blocking), the expanded 53-feature set (ber.features), and
compares three decision-rule strategies on the tune set -- global threshold,
global threshold + bipartite exclusivity, and source-specific (S2/S3)
thresholds + bipartite exclusivity -- picking whichever scores best on tune
before reporting the final holdout number, mirroring the reference repo's
own incremental-ablation methodology but measured against this data.

Hard-negative mining note: fit-set negatives are already drawn exclusively
from the blocking output (build_pair_features(candidates, ...) followed by
downsample_training_pairs), never from random corpus pairs -- this was
already true of retrain_recall_v3.py and needed no change; noted here since
the user asked to verify it explicitly.

Writes to artifacts_v5/ -- never touches artifacts_final/models/ or
artifacts_recall_v3/models/.
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
from ber.decision import enforce_bipartite_exclusivity, predictions_from_scores, tune_decision_rule, tune_source_specific_thresholds
from ber.features import build_pair_features
from ber.io import SOURCE_COLUMNS, read_ground_truth, read_train_source1, truth_dict
from ber.keyed_blocking import build_country_index, score_left_batch
from ber.metrics import blocking_audit, macro_fbeta
from ber.modeling import downsample_training_pairs, scored_pairs, train_match_model
from ber.normalize import normalize_frame

RAW_COUNTRY = {"india": "India", "us": "US"}
MEMORY_CEILING_GB = 8.0  # tighter ceiling this pass per the user's explicit V5 instruction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resource-dir", type=Path, default=Path("student_resource"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts_v5"))
    p.add_argument("--output-dir", type=Path, default=Path("output_v5"))
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "V5_RUN_LOG.md"
    log = make_logger(log_path)

    log(f"=== V5 retrain starting (n_s1_per_country={args.n_s1_per_country:,}, top_k={args.top_k}, budget={args.candidate_budget}, mem_ceiling={MEMORY_CEILING_GB}GB) ===")
    t_all = time.time()

    truth_frame_full = read_ground_truth(args.resource_dir / "dataset" / "train")
    truths_all = truth_dict(truth_frame_full)
    del truth_frame_full

    fit_frames, tune_frames, holdout_frames = [], [], []
    fit_ids_all, tune_ids_all, holdout_ids_all = [], [], []

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
        log(f"{country} full-sample blocking audit (top_k={args.top_k}, V5 blocking+translit): {json.dumps(audit)}")

        cand_fit = candidates[candidates["source1_entity_id"].isin(set(fit_c))]
        cand_tune = candidates[candidates["source1_entity_id"].isin(set(tune_c))]
        cand_holdout = candidates[candidates["source1_entity_id"].isin(set(holdout_c))]

        feat_fit, lab_fit = build_pair_features(cand_fit, sample_norm, targets_norm, truth=truths_all)
        feat_tune, _ = build_pair_features(cand_tune, sample_norm, targets_norm, truth=None)
        feat_holdout, _ = build_pair_features(cand_holdout, sample_norm, targets_norm, truth=None)
        log_mem(f"after building {country} features ({len(feat_fit.columns) if feat_fit is not None and not feat_fit.empty else 0} feature columns)", log)

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

    log("--- Training on combined fit set (India + US), negatives are blocking-output hard negatives by construction ---")
    fit_features = pd.concat([f for f, _ in fit_frames], ignore_index=True)
    fit_labels = pd.concat([l for _, l in fit_frames], ignore_index=True)
    del fit_frames
    training_config = TrainingConfig()
    fit_features, fit_labels = downsample_training_pairs(fit_features, fit_labels, ratio=training_config.negative_downsample_ratio, random_seed=args.seed)
    log(f"Training on {len(fit_features):,} pairs ({int(fit_labels.sum()):,} positive), {len(fit_features.columns)} features...")
    t0 = time.time()
    model = train_match_model(fit_features, fit_labels, args.seed)
    log(f"Training done in {time.time()-t0:.1f}s")
    log_mem("after training", log)

    log("--- Comparing decision-rule strategies on combined tune set ---")
    tune_candidates = pd.concat([c for c, _ in tune_frames], ignore_index=True)
    tune_features = pd.concat([f for _, f in tune_frames], ignore_index=True)
    del tune_frames
    tune_prob = model.predict_proba(tune_features)
    tune_scored = scored_pairs(tune_candidates, tune_prob)

    global_params, global_preds = tune_decision_rule(
        tune_scored, truths_all, tune_ids_all,
        thresholds=training_config.decision_thresholds, caps=training_config.max_match_caps,
    )
    log(f"[A] global threshold={global_params['threshold']}, cap={global_params['cap']}: tune_macro_f05={global_params['macro_f05']:.6f}")

    global_bipartite_preds = enforce_bipartite_exclusivity(global_preds, tune_scored)
    global_bipartite_score = macro_fbeta(global_bipartite_preds, truths_all, tune_ids_all, beta=0.5)
    log(f"[B] global + bipartite exclusivity: tune_macro_f05={global_bipartite_score:.6f}")

    source_params, source_preds_raw = tune_source_specific_thresholds(
        tune_scored, truths_all, tune_ids_all,
        thresholds=training_config.decision_thresholds, cap=global_params["cap"], use_bipartite=True,
    )
    log(f"[C] source-specific S2={source_params['threshold_s2']}, S3={source_params['threshold_s3']}, cap={source_params['cap']}, +bipartite: tune_macro_f05={source_params['macro_f05']:.6f}")

    candidates_ranked = [
        ("global", global_params["macro_f05"], {"mode": "global", "threshold": global_params["threshold"], "cap": global_params["cap"], "bipartite": False}),
        ("global_bipartite", global_bipartite_score, {"mode": "global", "threshold": global_params["threshold"], "cap": global_params["cap"], "bipartite": True}),
        ("source_specific_bipartite", source_params["macro_f05"], {"mode": "source_specific", "threshold_s2": source_params["threshold_s2"], "threshold_s3": source_params["threshold_s3"], "cap": source_params["cap"], "bipartite": True}),
    ]
    winner_name, winner_score, winner_rule = max(candidates_ranked, key=lambda x: x[1])
    log(f"*** Winning decision-rule strategy on tune: {winner_name} (tune_macro_f05={winner_score:.6f}) ***")

    log("--- Evaluating winning strategy on combined holdout set (the real, honest number) ---")
    holdout_candidates = pd.concat([c for c, _ in holdout_frames], ignore_index=True)
    holdout_features = pd.concat([f for _, f in holdout_frames], ignore_index=True)
    del holdout_frames
    holdout_prob = model.predict_proba(holdout_features)
    holdout_scored = scored_pairs(holdout_candidates, holdout_prob)

    if winner_rule["mode"] == "global":
        holdout_predictions = predictions_from_scores(holdout_scored, holdout_ids_all, winner_rule["threshold"], winner_rule["cap"])
    else:
        holdout_predictions = predictions_from_scores(holdout_scored, holdout_ids_all, {"S2": winner_rule["threshold_s2"], "S3": winner_rule["threshold_s3"]}, winner_rule["cap"])
    if winner_rule["bipartite"]:
        holdout_predictions = enforce_bipartite_exclusivity(holdout_predictions, holdout_scored)
    holdout_macro_f05 = macro_fbeta(holdout_predictions, truths_all, holdout_ids_all, beta=0.5)
    log(f"*** V5 real-scale holdout macro F0.5 = {holdout_macro_f05:.6f} (n={len(holdout_ids_all):,}, India+US combined, strategy={winner_name}) ***")
    log(f"For comparison: V3 retrain holdout was 0.8286 (top_k=100, no translit/P2-blocking/expanded-features/bipartite/source-thresholds); original modulo-sampled holdout was 0.945 (not real-scale, treated as optimistic)")

    joblib.dump(model, args.artifact_dir / "models" / "match_model.joblib")
    (args.artifact_dir / "models" / "decision_rule.json").write_text(json.dumps(winner_rule, indent=2), encoding="utf-8")
    metrics = {
        "n_s1_per_country": args.n_s1_per_country,
        "top_k": args.top_k,
        "candidate_budget": args.candidate_budget,
        "n_features": len(fit_features.columns),
        "fit_pairs": len(fit_features),
        "tune_s1": len(tune_ids_all),
        "holdout_s1": len(holdout_ids_all),
        "decision_rule_strategies_compared": {name: score for name, score, _ in candidates_ranked},
        "winning_strategy": winner_name,
        "winning_rule": winner_rule,
        "holdout_macro_f05": holdout_macro_f05,
        "total_time_s": time.time() - t_all,
        "peak_rss_gb": peak_rss_gb(),
    }
    (args.artifact_dir / "output" / "retrain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Wrote model+rule to {args.artifact_dir / 'models'}, metrics to {args.artifact_dir / 'output' / 'retrain_metrics.json'}")
    log(f"=== V5 retrain done in {time.time()-t_all:.1f}s ===")


if __name__ == "__main__":
    main()
