#!/usr/bin/env python3
"""Priority 1 + Priority 3 audits for the V5 push: lightweight, streaming,
column-subset reads only -- no full target-pool load, so this is safe to
run alongside the still-in-progress output_recall_v3 India job.

Priority 1: how much of each source's business_name/business_address is in
a non-Latin script (not just accented Latin -- strip_accents() from
ber.normalize already removes accents via NFKD; if characters survive that
with ord>127, they're a genuinely different script), broken down by file
and country.

Priority 3: two claimed dataset facts, verified directly against
train_ground_truth.tsv rather than trusted --
  (a) zero cross-country matches (S1's country always equals every matched
      target's country)
  (b) each target (S2/S3) entity_id appears in at most one S1's matched list
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from ber.normalize import strip_accents

RESOURCE_DIR = Path("student_resource")
CHUNK_SIZE = 500_000


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path = Path("output_v5/V5_RUN_LOG.md")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def has_non_latin(text: str) -> bool:
    if not text:
        return False
    stripped = strip_accents(text)
    return any(ord(ch) > 127 for ch in stripped)


def audit_non_latin_file(path: Path, label: str) -> dict:
    if not path.exists():
        log(f"  {label}: file not found, skipping ({path})")
        return {}
    counts_by_country: dict[str, dict[str, int]] = {}
    total = 0
    t0 = time.time()
    for chunk in pd.read_csv(
        path, sep="\t", keep_default_na=False,
        usecols=["entity_id", "business_name", "business_address", "country"],
        chunksize=CHUNK_SIZE,
    ):
        for country, group in chunk.groupby("country", sort=False):
            c = counts_by_country.setdefault(country, {"rows": 0, "name_non_latin": 0, "addr_non_latin": 0, "either_non_latin": 0})
            name_flags = group["business_name"].map(has_non_latin)
            addr_flags = group["business_address"].map(has_non_latin)
            c["rows"] += len(group)
            c["name_non_latin"] += int(name_flags.sum())
            c["addr_non_latin"] += int(addr_flags.sum())
            c["either_non_latin"] += int((name_flags | addr_flags).sum())
        total += len(chunk)
    log(f"  {label}: {total:,} rows scanned in {time.time()-t0:.1f}s -> {counts_by_country}")
    return counts_by_country


def priority1_audit() -> None:
    log("=== Priority 1: non-Latin script audit ===")
    files = [
        (RESOURCE_DIR / "dataset" / "train" / "train_source1.tsv", "train_source1"),
        (RESOURCE_DIR / "dataset" / "train" / "train_source2.tsv", "train_source2"),
        (RESOURCE_DIR / "dataset" / "train" / "train_source3.tsv", "train_source3"),
        (RESOURCE_DIR / "dataset" / "test" / "test_source1.tsv", "test_source1"),
        (RESOURCE_DIR / "dataset" / "test" / "test_source2.tsv", "test_source2"),
        (RESOURCE_DIR / "dataset" / "test" / "test_source3.tsv", "test_source3"),
    ]
    grand = {}
    for path, label in files:
        result = audit_non_latin_file(path, label)
        for country, c in result.items():
            g = grand.setdefault(country, {"rows": 0, "name_non_latin": 0, "addr_non_latin": 0, "either_non_latin": 0})
            for k in g:
                g[k] += c[k]
    log(f"Priority 1 TOTAL across all files, by country: {grand}")
    for country, c in grand.items():
        if c["rows"]:
            pct = 100 * c["either_non_latin"] / c["rows"]
            log(f"  {country}: {c['either_non_latin']:,}/{c['rows']:,} rows ({pct:.2f}%) have non-Latin-script name or address")


def priority3_audit() -> None:
    log("=== Priority 3: verifying dataset facts against train_ground_truth.tsv ===")
    train_dir = RESOURCE_DIR / "dataset" / "train"

    t0 = time.time()
    id_country: dict[str, str] = {}
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        for chunk in pd.read_csv(train_dir / filename, sep="\t", keep_default_na=False, usecols=["entity_id", "country"], chunksize=CHUNK_SIZE):
            id_country.update(zip(chunk["entity_id"], chunk["country"]))
    log(f"Built id->country map for {len(id_country):,} IDs (S1+S2+S3) in {time.time()-t0:.1f}s")

    t0 = time.time()
    cross_country_mismatches = 0
    cross_country_examples = []
    total_links = 0
    target_owner: dict[str, str] = {}
    violation_targets: list[tuple[str, str, str]] = []  # (target_id, first_owner, second_owner)

    gt_path = train_dir / "train_ground_truth.tsv"
    for chunk in pd.read_csv(gt_path, sep="\t", keep_default_na=False, usecols=["source1_entity_id", "matched_entity_ids"], chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            s1_id = row.source1_entity_id
            s1_country = id_country.get(s1_id)
            matched = [m for m in str(row.matched_entity_ids).split(",") if m]
            for target_id in matched:
                total_links += 1
                target_country = id_country.get(target_id)
                if s1_country is not None and target_country is not None and s1_country != target_country:
                    cross_country_mismatches += 1
                    if len(cross_country_examples) < 10:
                        cross_country_examples.append((s1_id, s1_country, target_id, target_country))
                prior = target_owner.get(target_id)
                if prior is None:
                    target_owner[target_id] = s1_id
                elif prior != s1_id and len(violation_targets) < 10:
                    violation_targets.append((target_id, prior, s1_id))

    log(f"Scanned {total_links:,} ground-truth (S1, target) links in {time.time()-t0:.1f}s")
    log(f"FACT (a) cross-country matches: {cross_country_mismatches:,} / {total_links:,} links cross a country boundary")
    if cross_country_examples:
        log(f"  examples (s1_id, s1_country, target_id, target_country): {cross_country_examples}")
    else:
        log("  CONFIRMED: zero cross-country matches found.")

    log(f"FACT (b) quick pass (single-scan, first-collision-only) found {len(violation_targets)} early examples of a target claimed twice: {violation_targets}")
    log("Computing exact FACT (b) violation count (targets claimed by >1 S1) in a dedicated full pass...")


def priority3b_exact() -> None:
    train_dir = RESOURCE_DIR / "dataset" / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    t0 = time.time()
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(gt_path, sep="\t", keep_default_na=False, usecols=["matched_entity_ids"], chunksize=CHUNK_SIZE):
        for value in chunk["matched_entity_ids"]:
            for target_id in str(value).split(","):
                if target_id:
                    counts[target_id] += 1
    multi = {tid: c for tid, c in counts.items() if c > 1}
    log(f"Counted {len(counts):,} distinct target IDs referenced in ground truth, in {time.time()-t0:.1f}s")
    log(f"FACT (b) targets claimed by more than one S1 entity: {len(multi):,} / {len(counts):,}")
    if multi:
        log(f"  examples (target_id -> claim_count): {dict(list(multi.items())[:10])}")
    else:
        log("  CONFIRMED: every target (S2/S3) record belongs to at most one S1 entity.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "p1"):
        priority1_audit()
    if mode in ("all", "p3"):
        priority3_audit()
        priority3b_exact()
    log("=== audit_v5.py done ===")
