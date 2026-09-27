#!/usr/bin/env python3
"""Root-cause a handful of real blocking-recall failures: does the S1 row
share ANY key at all with its true target (a ranking/top_k truncation
problem, fixable with parameters) or ZERO keys (a key-generation/
normalization problem, needing different keys or normalization fixes)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ber.io import read_ground_truth, read_train_source1, truth_dict
from ber.keyed_blocking import _row_keys, build_country_index, score_left_batch
from ber.normalize import normalize_frame

RAW_COUNTRY = {"india": "India", "us": "US"}


def load_full_country_targets(train_dir, raw_country, chunk_size=500_000):
    SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
    parts = []
    for filename, tag in (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")):
        for chunk in pd.read_csv(train_dir / filename, sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS, chunksize=chunk_size):
            sel = chunk.loc[chunk["country"] == raw_country].copy()
            if not sel.empty:
                sel["target_source"] = tag
                parts.append(sel)
    targets = pd.concat(parts, ignore_index=True)
    return normalize_frame(targets)


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "india"
    n_s1 = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    train_dir = Path("student_resource/dataset/train")
    raw_country = RAW_COUNTRY[country]

    source1 = read_train_source1(train_dir)
    source1 = source1.loc[source1["country"] == raw_country].reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(source1), size=min(n_s1, len(source1)), replace=False)
    sample = source1.iloc[idx].reset_index(drop=True)
    sample_norm = normalize_frame(sample)
    sample_ids = sample_norm["entity_id"].tolist()

    truth_frame = read_ground_truth(train_dir)
    truth_frame = truth_frame[truth_frame["source1_entity_id"].isin(set(sample_ids))]
    truths = truth_dict(truth_frame)

    print(f"Loading full {country} target pool...", flush=True)
    targets_norm = load_full_country_targets(train_dir, raw_country)
    print(f"Target pool: {len(targets_norm):,} rows", flush=True)
    target_by_id = targets_norm.set_index("entity_id", drop=False)

    print("Building index + scoring...", flush=True)
    country_index = build_country_index(targets_norm)
    scored = score_left_batch(sample_norm, country_index, top_k=60, candidate_budget=150)
    cand_map = scored.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    sample_by_id = sample_norm.set_index("entity_id", drop=False)

    shared_key_but_missing = []
    zero_shared_key = []
    checked = 0
    for sid in sample_ids:
        true_set = truths.get(sid, set())
        if not true_set:
            continue
        cands = cand_map.get(sid, set())
        missing = true_set - cands
        if not missing:
            continue
        s1_row = sample_by_id.loc[sid]
        s1_keys = set(_row_keys(s1_row["name_norm"], s1_row["address_norm"], s1_row["address_numbers"], country_index.freq))

        for tid in missing:
            if tid not in target_by_id.index:
                continue  # true target not even in pool (shouldn't happen; full pool loaded)
            t_row = target_by_id.loc[tid]
            if isinstance(t_row, pd.DataFrame):
                t_row = t_row.iloc[0]
            t_keys = set(_row_keys(t_row["name_norm"], t_row["address_norm"], t_row["address_numbers"], country_index.freq))

            shared = s1_keys & t_keys
            record = {
                "sid": sid, "tid": tid,
                "s1_name": s1_row["name_norm"], "t_name": t_row["name_norm"],
                "s1_addr": s1_row["address_norm"], "t_addr": t_row["address_norm"],
                "shared_keys": shared,
            }
            if shared:
                shared_key_but_missing.append(record)
            else:
                zero_shared_key.append(record)
            checked += 1
        if checked >= 400:
            break

    print(f"\nChecked {checked} missing (S1, true-target) pairs.")
    print(f"  Shared >=1 key but still missing from final candidates (ranking/top_k truncation issue): {len(shared_key_but_missing)}")
    print(f"  Zero shared keys at all (key-generation/normalization gap): {len(zero_shared_key)}")

    print("\n--- Examples: shared key but truncated ---")
    for r in shared_key_but_missing[:8]:
        print(f"  S1={r['sid']} true={r['tid']}")
        print(f"    s1_name={r['s1_name']!r:60} t_name={r['t_name']!r}")
        print(f"    s1_addr={r['s1_addr']!r:60} t_addr={r['t_addr']!r}")
        print(f"    shared_keys={r['shared_keys']}")

    print("\n--- Examples: zero shared key ---")
    for r in zero_shared_key[:15]:
        print(f"  S1={r['sid']} true={r['tid']}")
        print(f"    s1_name={r['s1_name']!r:60} t_name={r['t_name']!r}")
        print(f"    s1_addr={r['s1_addr']!r:60} t_addr={r['t_addr']!r}")


if __name__ == "__main__":
    main()
