#!/usr/bin/env python3
"""Generate the real leaderboard submission for the test set.

Replaces the ad-hoc heuristic scripts (make_rule_submission.py,
make_fuzzy_submission.py) that produced the 0.51 leaderboard score with the
actual validated pipeline: keyed blocking (scalable to the real multi-
million-row test corpus) -> pairwise features -> the trained LightGBM
matcher -> the F0.5-tuned decision rule.

Memory-bounded and resumable by design (see the postmortem in
docs/memory_postmortem.md): a country's target pool and blocking index are
built once, then S1 rows are processed in small batches -- candidates,
features, scoring and decision all happen per-batch and are written to disk
immediately, with a checkpoint recorded per batch. Peak memory scales with
batch size, not with a country's total candidate-pair count (which was the
root cause of the OOM crash: build_pair_features's merge duplicates every
text column across every candidate pair, and at India's full scale that was
tens of millions of rows at once).

A lockfile prevents two copies of this script from running concurrently
against the same output directory (running two large countries' worth of
work at once was the second contributor to the crash).
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

import joblib
import pandas as pd

from ber.blocking import candidate_mapping, combine_candidate_frames
from ber.config import BlockingConfig
from ber.decision import enforce_bipartite_exclusivity, predictions_from_scores
from ber.features import build_pair_features
from ber.keyed_blocking import build_country_index, score_left_batch
from ber.modeling import scored_pairs
from ber.normalize import normalize_frame

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the final test submission.")
    parser.add_argument("--resource-dir", type=Path, default=Path("student_resource"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts_final/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_final"))
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--candidate-budget", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=12_000, help="S1 rows processed per batch. Bounds peak memory.")
    parser.add_argument("--limit-s1", type=int, default=None, help="Debug only: cap S1 rows per country for a quick smoke test.")
    parser.add_argument("--country", type=str, default=None, help="Process only this normalized country (e.g. 'france'), for a targeted rerun.")
    parser.add_argument("--threshold-override", type=float, default=None, help="Use this decision threshold instead of decision_rule.json's.")
    parser.add_argument("--cap-override", type=str, default=None, help="Use this decision cap instead of decision_rule.json's ('none' for no cap, or an int).")
    return parser.parse_args()


def peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, KB on Linux.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**3) if sys.platform == "darwin" else raw / (1024**2)


MEMORY_CEILING_GB = 11.0  # hard abort if peak RSS exceeds this -- fail cleanly rather than risk another OOM crash


def log_mem(label: str) -> None:
    rss = peak_rss_gb()
    print(f"    [mem] {label}: peak RSS so far = {rss:.2f} GB", flush=True)
    if rss > MEMORY_CEILING_GB:
        print(f"    [mem] ABORTING: peak RSS {rss:.2f} GB exceeded the {MEMORY_CEILING_GB} GB safety ceiling.", flush=True)
        raise MemoryError(f"Peak RSS {rss:.2f} GB exceeded safety ceiling {MEMORY_CEILING_GB} GB at stage: {label}")


def write_rows(path: Path, header_col: str, rows: list[tuple[str, str]]) -> None:
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        if is_new:
            f.write(f"source1_entity_id\t{header_col}\n")
        for source1_id, joined in rows:
            f.write(f"{source1_id}\t{joined}\n")


@contextlib.contextmanager
def exclusive_lock(lock_path: Path):
    """Refuse to run if another instance already holds this output directory's lock.

    Two copies of this script running concurrently (against different output
    dirs but overlapping CPU/memory budget) is exactly what turned a large-
    but-survivable job into an OOM crash. This only guards a single output
    dir's own lock; it deliberately does not try to detect a *different*
    output dir's job running at the same time -- that's a operator
    discipline call, not something a lockfile can enforce across dirs.
    """
    if lock_path.exists():
        pid_text = lock_path.read_text().strip()
        raise SystemExit(
            f"Refusing to start: {lock_path} already exists (pid {pid_text}). "
            "Another run against this output dir may be in progress. If it's not "
            f"(e.g. it crashed), delete {lock_path} and retry."
        )
    lock_path.write_text(str(__import__("os").getpid()), encoding="utf-8")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def load_country_targets(test_dir: Path, raw_country_values: set[str]) -> pd.DataFrame:
    parts = []
    for filename, tag in (("test_source2.tsv", "S2"), ("test_source3.tsv", "S3")):
        for chunk in pd.read_csv(test_dir / filename, sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS, chunksize=500_000):
            sel = chunk.loc[chunk["country"].isin(raw_country_values)].copy()
            if not sel.empty:
                sel["target_source"] = tag
                parts.append(sel)
    if not parts:
        return pd.DataFrame()
    targets = pd.concat(parts, ignore_index=True)
    del parts
    targets_norm = normalize_frame(targets)
    del targets
    return targets_norm


def parse_decision_rule(decision: dict) -> tuple[float | dict, object, bool]:
    """Old format: {"threshold": .., "cap": ..}. New (V5) format adds
    "mode" ("global" or "source_specific") and "bipartite"; old files have
    neither key and default to global/no-bipartite, so this reads both
    transparently."""
    mode = decision.get("mode", "global")
    cap = decision["cap"]
    bipartite = decision.get("bipartite", False)
    if mode == "source_specific":
        threshold: float | dict = {"S2": decision["threshold_s2"], "S3": decision["threshold_s3"]}
    else:
        threshold = decision["threshold"]
    return threshold, cap, bipartite


def process_country(
    country: str,
    left_all: pd.DataFrame,
    test_dir: Path,
    raw_by_norm: dict[str, list[str]],
    config: BlockingConfig,
    model,
    threshold: float | dict,
    cap,
    bipartite: bool,
    args: argparse.Namespace,
    matching_path: Path,
    candidate_path: Path,
    progress_path: Path,
    progress: dict,
) -> None:
    left = left_all[left_all["country_norm"] == country].reset_index(drop=True)
    if args.limit_s1:
        left = left.head(args.limit_s1).reset_index(drop=True)
    print(f"\n=== Country: {country} === ({len(left):,} S1 rows)", flush=True)

    print("Loading target pool (S2+S3) for this country...", flush=True)
    t0 = time.time()
    targets_norm = load_country_targets(test_dir, set(raw_by_norm[country]))
    print(f"Target pool: {len(targets_norm):,} rows, load time={time.time()-t0:.1f}s", flush=True)
    log_mem("after loading target pool")

    country_index = None
    if targets_norm.empty:
        batches = [left]
    else:
        print("Building blocking index for this country (once, reused across all batches)...", flush=True)
        t0 = time.time()
        country_index = build_country_index(targets_norm)
        print(f"Index build time={time.time()-t0:.1f}s", flush=True)
        log_mem("after building blocking index")

        batch_size = args.batch_size
        batches = [left.iloc[i : i + batch_size].reset_index(drop=True) for i in range(0, len(left), batch_size)]

    country_progress = progress.setdefault(country, {"completed_batches": 0, "total_batches": len(batches)})
    start_batch = country_progress["completed_batches"]
    if start_batch >= len(batches):
        print(f"Country {country} already fully processed ({start_batch}/{len(batches)} batches). Skipping.", flush=True)
        return
    if start_batch > 0:
        print(f"Resuming {country} from batch {start_batch}/{len(batches)} (earlier batches already on disk).", flush=True)

    t_country = time.time()
    for batch_idx in range(start_batch, len(batches)):
        batch = batches[batch_idx]
        batch_ids = batch["entity_id"].tolist()
        t_batch = time.time()

        if targets_norm.empty:
            write_rows(candidate_path, "candidate_entity_ids", [(sid, "") for sid in batch_ids])
            write_rows(matching_path, "matched_entity_ids", [(sid, "") for sid in batch_ids])
        else:
            raw_candidates = score_left_batch(batch, country_index, top_k=args.top_k, candidate_budget=args.candidate_budget)
            candidates = combine_candidate_frames([raw_candidates], config)
            del raw_candidates
            cand_map = candidate_mapping(candidates)
            write_rows(candidate_path, "candidate_entity_ids", [(sid, ",".join(cand_map.get(sid, []))) for sid in batch_ids])

            if candidates.empty:
                write_rows(matching_path, "matched_entity_ids", [(sid, "") for sid in batch_ids])
            else:
                features, _ = build_pair_features(candidates, batch, targets_norm, truth=None)
                probs = model.predict_proba(features)
                scored = scored_pairs(candidates, probs)
                predictions = predictions_from_scores(scored, batch_ids, threshold, cap)
                if bipartite:
                    predictions = enforce_bipartite_exclusivity(predictions, scored)
                write_rows(matching_path, "matched_entity_ids", [(sid, ",".join(predictions.get(sid, []))) for sid in batch_ids])
                del features, probs, scored, predictions
            del candidates, cand_map

        country_progress["completed_batches"] = batch_idx + 1
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        del batch
        gc.collect()
        print(
            f"  batch {batch_idx+1}/{len(batches)} done in {time.time()-t_batch:.1f}s "
            f"({len(batch_ids):,} S1 rows)",
            flush=True,
        )
        log_mem(f"after batch {batch_idx+1}/{len(batches)}")

    del targets_norm, country_index
    gc.collect()
    print(f"=== Country {country} done in {time.time()-t_country:.1f}s ===", flush=True)


def main() -> None:
    args = parse_args()
    test_dir = args.resource_dir / "dataset" / "test"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    progress_path = args.output_dir / "_progress.json"
    lock_path = args.output_dir / ".lock"

    with exclusive_lock(lock_path):
        progress: dict = json.loads(progress_path.read_text()) if progress_path.exists() else {}

        print("Loading trained model + decision rule...", flush=True)
        model = joblib.load(args.model_dir / "match_model.joblib")
        decision = json.loads((args.model_dir / "decision_rule.json").read_text())
        threshold, cap, bipartite = parse_decision_rule(decision)
        if args.threshold_override is not None:
            # A manual override always means a single simple global threshold
            # (this is how France's out-of-distribution safety override is
            # applied on top of whatever the tuned rule says) -- bipartite
            # exclusivity is left as the loaded rule set it, since it can
            # only remove predictions, never add one, so it's safe to keep
            # even under a conservative manual override.
            threshold = args.threshold_override
        if args.cap_override is not None:
            cap = None if args.cap_override.lower() == "none" else int(args.cap_override)
        print(f"threshold={threshold}, cap={cap}, bipartite={bipartite}, batch_size={args.batch_size}", flush=True)

        print("Loading test source1...", flush=True)
        source1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS)
        source1["entity_id"] = source1["entity_id"].str.strip()
        source1_norm = normalize_frame(source1)
        raw_by_norm: dict[str, list[str]] = (
            source1_norm.groupby("country_norm")["country"].agg(lambda s: sorted(set(s))).to_dict()
        )
        countries = sorted(raw_by_norm.keys())
        if args.country:
            countries = [c for c in countries if c == args.country]
        print(f"Countries to process: {countries} (raw values: {raw_by_norm})", flush=True)

        config = BlockingConfig()

        for country in countries:
            process_country(
                country, source1_norm, test_dir, raw_by_norm, config, model, threshold, cap, bipartite,
                args, matching_path, candidate_path, progress_path, progress,
            )

    print("\nAll requested countries complete.", flush=True)
    print(f"Wrote {matching_path}", flush=True)
    print(f"Wrote {candidate_path}", flush=True)


if __name__ == "__main__":
    main()
