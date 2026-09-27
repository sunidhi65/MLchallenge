#!/usr/bin/env python3
"""Build and cache pairwise features to disk, once, so that calibration /
hyperparameter / threshold / cross-country experiments become seconds-long
instead of re-running an hour of candidate generation + feature building
every time.

Deliberately does NOT split into fit/tune/holdout here -- it saves every
sampled S1 entity's candidate rows with their labels, plus the full list of
sampled S1 IDs per country (needed because macro F0.5 averages over ALL
entities, including ones blocking found zero candidates for, which have no
rows in the feature table). Experiments do their own splitting, which is
what makes the cross-country "unseen country" simulation possible (train on
India, evaluate on US as a stand-in for France).

Memory-bounded: one country's target pool + index at a time, and features
are flushed to a parquet part file after every batch rather than
accumulating a whole country's feature frame in RAM.
"""
from __future__ import annotations

import argparse
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
from ber.features import build_pair_features
from ber.io import SOURCE_COLUMNS, read_ground_truth, read_train_source1, truth_dict
from ber.keyed_blocking import build_country_index, score_left_batch
from ber.metrics import blocking_audit
from ber.normalize import normalize_frame

RAW_COUNTRY = {"india": "India", "us": "US"}
MEMORY_CEILING_GB = 8.0
META_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_rank"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resource-dir", type=Path, default=Path("student_resource"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts_v6/cache"))
    p.add_argument("--output-dir", type=Path, default=Path("output_v6"))
    p.add_argument("--n-s1-per-country", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=5_000)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--candidate-budget", type=int, default=150)
    p.add_argument("--target-chunk-size", type=int, default=500_000)
    p.add_argument("--country", type=str, default=None, help="Only build this country (for resuming).")
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


def build_country_cache(country: str, args: argparse.Namespace, truths_all: dict, log) -> dict:
    raw_country = RAW_COUNTRY[country]
    country_cache = args.cache_dir / country
    country_cache.mkdir(parents=True, exist_ok=True)
    for stale in country_cache.glob("part_*.parquet"):
        stale.unlink()

    t_c = time.time()
    source1 = read_train_source1(args.resource_dir / "dataset" / "train")
    source1 = source1.loc[source1["country"] == raw_country].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_s1_per_country, len(source1))
    idx = rng.choice(len(source1), size=n, replace=False)
    sample = source1.iloc[idx].reset_index(drop=True)
    sample_norm = normalize_frame(sample)
    sample_ids = sample_norm["entity_id"].tolist()
    del source1, sample
    log(f"Sampled {len(sample_ids):,} S1 rows for {country}")

    targets_norm = load_full_country_targets(args.resource_dir / "dataset" / "train", raw_country, args.target_chunk_size, log)
    log(f"Full {country} target pool: {len(targets_norm):,} rows")
    log_mem(f"after loading {country} target pool", log)

    t0 = time.time()
    country_index = build_country_index(targets_norm)
    log(f"{country} index build time={time.time()-t0:.1f}s")
    log_mem(f"after building {country} index", log)

    config = BlockingConfig(max_candidates_per_entity=args.top_k + 20)
    n_batches = (len(sample_norm) + args.batch_size - 1) // args.batch_size
    all_cand_map: dict[str, list[str]] = {}
    total_rows = 0
    t_feat = time.time()

    for bi in range(n_batches):
        batch = sample_norm.iloc[bi * args.batch_size : (bi + 1) * args.batch_size]
        raw = score_left_batch(batch, country_index, top_k=args.top_k, candidate_budget=args.candidate_budget)
        if raw.empty:
            continue
        candidates = combine_candidate_frames([raw], config)
        del raw
        all_cand_map.update(candidate_mapping(candidates))

        feats, labels = build_pair_features(candidates, batch, targets_norm, truth=truths_all)
        part = feats.copy()
        part["label"] = labels.to_numpy() if labels is not None else 0
        for col in META_COLS:
            part[col] = candidates[col].to_numpy()
        part.to_parquet(country_cache / f"part_{bi:04d}.parquet", index=False)
        total_rows += len(part)
        del feats, labels, part, candidates
        gc.collect()

        if bi == 0 or (bi + 1) % 5 == 0 or bi == n_batches - 1:
            log(f"  {country} batch {bi+1}/{n_batches}: {total_rows:,} feature rows cached (elapsed {time.time()-t_feat:.1f}s)")
            log_mem(f"{country} after batch {bi+1}", log)

    audit = blocking_audit(all_cand_map, truths_all, sample_ids)
    log(f"{country} blocking audit: {json.dumps(audit)}")

    meta = {
        "country": country,
        "n_s1_sampled": len(sample_ids),
        "sample_ids": sample_ids,
        "target_pool_rows": len(targets_norm),
        "feature_rows": total_rows,
        "top_k": args.top_k,
        "candidate_budget": args.candidate_budget,
        "seed": args.seed,
        "audit": audit,
        "build_time_s": time.time() - t_c,
    }
    (country_cache / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    del targets_norm, country_index, sample_norm, all_cand_map
    gc.collect()
    log_mem(f"after freeing {country}", log)
    log(f"=== {country} cache done in {time.time()-t_c:.1f}s, {total_rows:,} feature rows ===")
    return meta


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(args.output_dir / "V6_RUN_LOG.md")

    log(f"=== Feature cache build starting (n_s1_per_country={args.n_s1_per_country:,}, top_k={args.top_k}) ===")
    t_all = time.time()

    truth_frame = read_ground_truth(args.resource_dir / "dataset" / "train")
    truths_all = truth_dict(truth_frame)
    del truth_frame
    log_mem("after loading ground truth", log)

    countries = [args.country] if args.country else list(RAW_COUNTRY.keys())
    summary = {}
    for country in countries:
        summary[country] = {k: v for k, v in build_country_cache(country, args, truths_all, log).items() if k != "sample_ids"}

    (args.cache_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"=== Feature cache build done in {time.time()-t_all:.1f}s. Summary: {json.dumps(summary, indent=2)} ===")


if __name__ == "__main__":
    main()
