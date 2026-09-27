#!/usr/bin/env python3
"""Measure blocking recall at TRUE full-corpus scale on real train data.

train_validate.py's validation pool is deliberately small (every positive
target plus a modulo-sampled slice of negatives) so the pipeline fits on a
laptop. That makes buckets smaller and cleaner than they really are at full
scale, which is almost certainly why the 0.945 holdout macro F0.5 doesn't
match the 0.602 real leaderboard score. This script instead loads a
country's REAL, FULL, unsampled target pool (all of train_source2.tsv +
train_source3.tsv for that country) and scores a large random sample of real
S1 rows against it with the exact same blocking code path production uses
(build_country_index + score_left_batch), so the recall numbers reflect what
actually happens at true scale.

Reports, for the sampled S1 entities:
  - pair_recall / full_entity_recall / oracle_macro_f05 (blocking ceiling)
  - realistic_macro_f05 (oracle candidates run through the ACTUAL trained
    model + decision rule, if a model is available) -- the gap between this
    and oracle_macro_f05 is classifier/decision-rule loss, not blocking loss
  - a breakdown of *why* matches are missed: never became a candidate at all
    (blocking's fault) vs became a candidate but the model/rule didn't keep
    it (everything else's fault)

Memory-bounded and single-process, same pattern as make_final_submission.py:
one country's full target pool in memory at a time, freed before the next.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.blocking import candidate_mapping, combine_candidate_frames
from ber.config import BlockingConfig
from ber.decision import predictions_from_scores
from ber.features import build_pair_features
from ber.io import SOURCE_COLUMNS, read_ground_truth, read_train_source1, truth_dict
from ber.keyed_blocking import build_country_index, score_left_batch
from ber.metrics import blocking_audit, macro_fbeta
from ber.modeling import scored_pairs
from ber.normalize import normalize_frame

RAW_COUNTRY = {"india": "India", "us": "US"}
MEMORY_CEILING_GB = 11.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure real full-scale blocking recall.")
    parser.add_argument("--resource-dir", type=Path, default=Path("student_resource"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts_final/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_recall_v3"))
    parser.add_argument("--country", required=True, choices=["india", "us"])
    parser.add_argument("--n-s1", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, required=True, help="Tag for this measurement, e.g. 'baseline' or 'keys_v2'.")
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--candidate-budget", type=int, default=150)
    parser.add_argument("--target-chunk-size", type=int, default=500_000)
    return parser.parse_args()


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
        log(f"  [mem] ABORTING: peak RSS {rss:.2f} GB exceeded the {MEMORY_CEILING_GB} GB safety ceiling at stage: {label}")
        raise MemoryError(f"Peak RSS {rss:.2f} GB exceeded safety ceiling at stage: {label}")


@contextlib.contextmanager
def exclusive_lock(lock_path: Path):
    if lock_path.exists():
        pid_text = lock_path.read_text().strip()
        raise SystemExit(
            f"Refusing to start: {lock_path} already exists (pid {pid_text}). "
            f"Another recall-workflow job may be in progress. If it's not, delete {lock_path} and retry."
        )
    import os
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


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
    if not parts:
        return pd.DataFrame()
    targets = pd.concat(parts, ignore_index=True)
    del parts
    targets_norm = normalize_frame(targets)
    del targets
    return targets_norm


def sample_country_s1(train_dir: Path, raw_country: str, n: int, seed: int) -> pd.DataFrame:
    source1 = read_train_source1(train_dir)
    source1 = source1.loc[source1["country"] == raw_country].reset_index(drop=True)
    if n >= len(source1):
        return source1
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(source1), size=n, replace=False)
    return source1.iloc[idx].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "measurement").mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "RECALL_RUN_LOG.md"
    log = make_logger(log_path)
    lock_path = args.output_dir / ".lock"

    with exclusive_lock(lock_path):
        train_dir = args.resource_dir / "dataset" / "train"
        raw_country = RAW_COUNTRY[args.country]
        log(f"=== Measurement '{args.label}' starting for country={args.country} (n_s1={args.n_s1:,}) ===")
        t_country = time.time()

        t0 = time.time()
        sample = sample_country_s1(train_dir, raw_country, args.n_s1, args.seed)
        sample_norm = normalize_frame(sample)
        sample_ids = sample_norm["entity_id"].tolist()
        log(f"Sampled {len(sample_ids):,} S1 rows for {args.country} in {time.time()-t0:.1f}s")

        t0 = time.time()
        truth_frame = read_ground_truth(train_dir)
        truth_frame = truth_frame[truth_frame["source1_entity_id"].isin(set(sample_ids))]
        truths = truth_dict(truth_frame)
        del truth_frame
        truths_sample = {sid: truths.get(sid, set()) for sid in sample_ids}
        n_with_truth = sum(1 for sid in sample_ids if truths_sample[sid])
        log(f"Loaded ground truth for sample in {time.time()-t0:.1f}s ({n_with_truth:,}/{len(sample_ids):,} have >=1 true match)")
        log_mem("after loading sample + truth", log)

        t0 = time.time()
        targets_norm = load_full_country_targets(train_dir, raw_country, args.target_chunk_size, log)
        log(f"Full {args.country} target pool: {len(targets_norm):,} rows, load time={time.time()-t0:.1f}s")
        log_mem("after loading full target pool", log)

        t0 = time.time()
        country_index = build_country_index(targets_norm)
        log(f"Index build time={time.time()-t0:.1f}s")
        log_mem("after building blocking index", log)

        t0 = time.time()
        raw_frames = []
        n_batches = (len(sample_norm) + args.batch_size - 1) // args.batch_size
        for i in range(0, len(sample_norm), args.batch_size):
            batch = sample_norm.iloc[i : i + args.batch_size]
            scored = score_left_batch(batch, country_index, top_k=args.top_k, candidate_budget=args.candidate_budget)
            if not scored.empty:
                raw_frames.append(scored)
            batch_num = i // args.batch_size + 1
            if batch_num == 1 or batch_num % 5 == 0 or batch_num == n_batches:
                log(f"  scoring batch {batch_num}/{n_batches} done (elapsed {time.time()-t0:.1f}s)")
                log_mem(f"after scoring batch {batch_num}/{n_batches}", log)
        scoring_time = time.time() - t0
        per_row_ms = 1000 * scoring_time / max(len(sample_norm), 1)
        log(f"Scoring done in {scoring_time:.1f}s ({per_row_ms:.2f} ms/row)")

        config = BlockingConfig()
        candidates = combine_candidate_frames(raw_frames, config)
        del raw_frames
        cand_map = candidate_mapping(candidates)
        log_mem("after combining candidates", log)

        audit = blocking_audit(cand_map, truths_sample, sample_ids)
        log(f"Blocking audit ({args.label}, {args.country}): {json.dumps(audit, indent=2)}")

        entities_with_truth = [sid for sid in sample_ids if truths_sample[sid]]
        zero_candidates_at_all = [sid for sid in entities_with_truth if not cand_map.get(sid)]
        has_cand_zero_overlap = [
            sid for sid in entities_with_truth
            if cand_map.get(sid) and not (set(cand_map[sid]) & truths_sample[sid])
        ]
        partial_overlap = [
            sid for sid in entities_with_truth
            if cand_map.get(sid) and 0 < len(set(cand_map[sid]) & truths_sample[sid]) < len(truths_sample[sid])
        ]
        breakdown = {
            "entities_with_truth": len(entities_with_truth),
            "zero_candidates_at_all": len(zero_candidates_at_all),
            "has_candidates_but_zero_true_overlap": len(has_cand_zero_overlap),
            "partial_true_overlap": len(partial_overlap),
            "full_true_overlap": audit["full_entity_recall"] and int(round(audit["full_entity_recall"] * len(entities_with_truth))),
        }
        log(f"Miss breakdown ({args.label}, {args.country}): {json.dumps(breakdown, indent=2)}")

        realistic_macro_f05 = None
        model_path = args.model_dir / "match_model.joblib"
        rule_path = args.model_dir / "decision_rule.json"
        if model_path.exists() and rule_path.exists() and not candidates.empty:
            import joblib
            t0 = time.time()
            model = joblib.load(model_path)
            decision = json.loads(rule_path.read_text())
            threshold, cap = float(decision["threshold"]), decision["cap"]
            features, _ = build_pair_features(candidates, sample_norm, targets_norm, truth=None)
            probs = model.predict_proba(features)
            scored_df = scored_pairs(candidates, probs)
            predictions = predictions_from_scores(scored_df, sample_ids, threshold, cap)
            realistic_macro_f05 = macro_fbeta(predictions, truths_sample, sample_ids, beta=0.5)
            log(
                f"Realistic end-to-end macro F0.5 ({args.label}, {args.country}, threshold={threshold}, cap={cap}): "
                f"{realistic_macro_f05:.6f}  (computed in {time.time()-t0:.1f}s)"
            )
            del features, probs, scored_df, predictions, model
        else:
            log(f"No trained model found at {model_path} -- skipping realistic end-to-end score, oracle only.")

        full_country_s1_counts = {"india": 883_188, "us": 1_323_633}
        extrapolated_hours = (per_row_ms / 1000) * full_country_s1_counts[args.country] / 3600
        log(
            f"Extrapolated full-{args.country} scoring time at this rate: {extrapolated_hours:.2f}h "
            f"(previous production run: india=3.64h, us=1.58h -- for reference, not identical code path if keys changed)"
        )

        result = {
            "label": args.label,
            "country": args.country,
            "n_s1_sampled": len(sample_ids),
            "audit": audit,
            "breakdown": breakdown,
            "realistic_macro_f05": realistic_macro_f05,
            "scoring_ms_per_row": per_row_ms,
            "extrapolated_full_country_hours": extrapolated_hours,
            "target_pool_rows": len(targets_norm),
            "peak_rss_gb": peak_rss_gb(),
            "total_time_s": time.time() - t_country,
        }
        result_path = args.output_dir / "measurement" / f"{args.label}_{args.country}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"Wrote {result_path}")
        log(f"=== Measurement '{args.label}' for {args.country} done in {time.time()-t_country:.1f}s ===\n")

        del candidates, cand_map, targets_norm, country_index, sample_norm, sample
        gc.collect()


if __name__ == "__main__":
    main()
