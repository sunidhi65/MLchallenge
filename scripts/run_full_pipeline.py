#!/usr/bin/env python3
"""Orchestrate the full, unattended test-set submission run.

Runs France -> India -> US sequentially through make_final_submission.py
(one subprocess per country, never concurrent -- the lockfile in that
script also enforces this), running correctness checks after each country
before proceeding to the next. Writes a persistent, human-readable log to
output_final_v2/RUN_LOG.md so progress is visible at a glance without
re-reading raw stdout. Stops immediately (without touching the next
country) if any correctness check fails.

After all three countries pass, runs the official validate_submission.py,
reports summary statistics, fills in the methodology document, and packages
the final submission zip.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path("/app")
OUTPUT_DIR = REPO / "output_final_v2"
LOG_PATH = OUTPUT_DIR / "RUN_LOG.md"
TEST_DIR = REPO / "student_resource" / "dataset" / "test"
VENV_PYTHON = REPO / ".venv" / "bin" / "python"

COUNTRIES = [
    # (normalized_country, raw_country_value, threshold_override, cap_override)
    ("france", "France", None, None),
    ("india", "India", None, None),
    ("us", "US", None, None),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_country(country: str, threshold_override: str | None, cap_override: str | None) -> bool:
    log(f"=== Starting {country} ===")
    cmd = [
        str(VENV_PYTHON), "scripts/make_final_submission.py",
        "--resource-dir", "student_resource",
        "--model-dir", "artifacts_final/models",
        "--output-dir", str(OUTPUT_DIR),
        "--country", country,
    ]
    if threshold_override is not None:
        cmd += ["--threshold-override", threshold_override]
    if cap_override is not None:
        cmd += ["--cap-override", cap_override]

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    per_country_log = OUTPUT_DIR / f"_{country}_run.log"
    with per_country_log.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        log(f"FAILED: {country} subprocess exited with code {proc.returncode}. See {per_country_log}")
        return False
    log(f"{country} subprocess completed (exit 0). Log: {per_country_log}")
    return True


def read_country_s1_ids(raw_country: str) -> set[str]:
    ids = set()
    with (TEST_DIR / "test_source1.tsv").open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        country_idx = header.index("country")
        id_idx = header.index("entity_id")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > country_idx and parts[country_idx] == raw_country:
                ids.add(parts[id_idx].strip())
    return ids


def parse_id_list_file(path: Path) -> tuple[dict[str, list[str]], Counter]:
    """Returns (last-value-wins mapping, row-occurrence counts).

    A plain dict alone silently hides duplicate rows (the later row just
    overwrites the earlier one in the mapping) -- this is exactly the bug
    that let two concurrent France writes go undetected by an earlier
    version of this check. The Counter makes duplicate *rows* (as opposed
    to duplicate values) an explicit, checkable fact.
    """
    mapping: dict[str, list[str]] = {}
    row_counts: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, _, rest = line.partition("\t")
            ids = [x for x in rest.split(",") if x] if rest else []
            mapping[s1] = ids
            row_counts[s1] += 1
    return mapping, row_counts


def check_country(country: str, raw_country: str) -> tuple[bool, list[str]]:
    """Correctness gate for one country's freshly-written rows. Returns (ok, problems)."""
    problems: list[str] = []
    expected_ids = read_country_s1_ids(raw_country)

    if not (OUTPUT_DIR / "matching_results.tsv").exists() or not (OUTPUT_DIR / "candidate_pairs.tsv").exists():
        return False, ["Output files do not exist."]

    matched, matched_row_counts = parse_id_list_file(OUTPUT_DIR / "matching_results.tsv")
    candidates, candidate_row_counts = parse_id_list_file(OUTPUT_DIR / "candidate_pairs.tsv")

    matched_country_ids = {sid for sid in matched if sid in expected_ids}
    candidate_country_ids = {sid for sid in candidates if sid in expected_ids}

    missing_from_matching = expected_ids - matched_country_ids
    missing_from_candidates = expected_ids - candidate_country_ids
    if missing_from_matching:
        problems.append(f"{len(missing_from_matching)} {country} S1 IDs missing from matching_results.tsv, e.g. {sorted(missing_from_matching)[:5]}")
    if missing_from_candidates:
        problems.append(f"{len(missing_from_candidates)} {country} S1 IDs missing from candidate_pairs.tsv, e.g. {sorted(missing_from_candidates)[:5]}")

    # Row-occurrence counts (not dict-key membership, which silently
    # collapses duplicate rows into one entry and can never detect this).
    dup_matched_rows = [sid for sid in expected_ids if matched_row_counts.get(sid, 0) > 1]
    dup_candidate_rows = [sid for sid in expected_ids if candidate_row_counts.get(sid, 0) > 1]
    if dup_matched_rows:
        problems.append(f"{len(dup_matched_rows)} {country} S1 IDs appear more than once in matching_results.tsv, e.g. {dup_matched_rows[:5]}")
    if dup_candidate_rows:
        problems.append(f"{len(dup_candidate_rows)} {country} S1 IDs appear more than once in candidate_pairs.tsv, e.g. {dup_candidate_rows[:5]}")

    subset_violations = 0
    prefix_violations = []
    intra_dupes = 0
    example_violation = None
    for sid in expected_ids:
        mids = matched.get(sid, [])
        cids = set(candidates.get(sid, []))
        if len(mids) != len(set(mids)):
            intra_dupes += 1
        extra = set(mids) - cids
        if extra:
            subset_violations += 1
            if example_violation is None:
                example_violation = (sid, sorted(extra)[:3])
        for mid in mids:
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                prefix_violations.append((sid, mid))

    if subset_violations:
        problems.append(f"{subset_violations} {country} S1 entities have matched IDs not present in their own candidate list (pipeline bug). Example: {example_violation}")
    if intra_dupes:
        problems.append(f"{intra_dupes} {country} S1 entities have duplicate IDs within their own matched list.")
    if prefix_violations:
        problems.append(f"{len(prefix_violations)} matched IDs lack an S2-/S3- prefix, e.g. {prefix_violations[:5]}")

    nonempty = sum(1 for sid in expected_ids if matched.get(sid))
    sizes = [len(matched.get(sid, [])) for sid in expected_ids]
    avg_size = sum(sizes) / len(sizes) if sizes else 0.0
    log(
        f"{country} stats: {len(expected_ids):,} S1 rows, {nonempty:,} non-empty "
        f"({nonempty/len(expected_ids):.1%}), avg predicted size (incl. empties)={avg_size:.3f}"
    )

    ok = not problems
    return ok, problems


def run_pipeline() -> bool:
    for country, raw_country, thr, cap in COUNTRIES:
        if not run_country(country, thr, cap):
            log(f"STOPPING: {country} subprocess failed. Not proceeding to next country.")
            return False

        ok, problems = check_country(country, raw_country)
        if not ok:
            log(f"CORRECTNESS CHECK FAILED for {country}:")
            for p in problems:
                log(f"  - {p}")
            log("STOPPING: not proceeding to next country until this is fixed.")
            return False
        log(f"Correctness checks PASSED for {country}. Proceeding.")

    log("=== All three countries complete and verified. Running final validator... ===")
    result = subprocess.run(
        [sys.executable, "utils/validate_submission.py",
         "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
         "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
         "--test-dir", "dataset/test"],
        cwd=str(REPO / "student_resource"),
        capture_output=True, text=True,
    )
    log("validate_submission.py output:\n" + result.stdout + result.stderr)
    if result.returncode != 0:
        log("FAILED: validate_submission.py did not pass. Stopping before packaging.")
        return False
    log("validate_submission.py PASSED.")

    report_final_summary()
    return True


def report_final_summary() -> None:
    matched, matched_row_counts = parse_id_list_file(OUTPUT_DIR / "matching_results.tsv")
    dup_rows = {sid: c for sid, c in matched_row_counts.items() if c > 1}
    if dup_rows:
        log(f"WARNING: {len(dup_rows)} duplicate S1 rows found in final matching_results.tsv, e.g. {list(dup_rows.items())[:5]}")
    total = len(matched)
    nonempty = sum(1 for v in matched.values() if v)
    sizes = [len(v) for v in matched.values()]
    avg_size = sum(sizes) / len(sizes) if sizes else 0.0

    log(f"=== FINAL SUMMARY ===")
    log(f"Total S1 rows: {total:,}")
    log(f"Non-empty: {nonempty:,} ({nonempty/total:.1%})  |  Empty (singleton predictions): {total-nonempty:,} ({(total-nonempty)/total:.1%})")
    log(f"Average predicted set size (all rows): {avg_size:.3f}")

    for country, raw_country, _, _ in COUNTRIES:
        ids = read_country_s1_ids(raw_country)
        c_sizes = [len(matched.get(sid, [])) for sid in ids]
        c_nonempty = sum(1 for s in c_sizes if s > 0)
        log(
            f"  {country}: {len(ids):,} rows, {c_nonempty:,} non-empty ({c_nonempty/len(ids):.1%}), "
            f"avg size={sum(c_sizes)/len(c_sizes):.3f}"
        )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== Full pipeline run starting ===")
    success = run_pipeline()
    if success:
        log("=== PIPELINE SUCCEEDED. Ready for documentation + packaging. ===")
        sys.exit(0)
    else:
        log("=== PIPELINE STOPPED DUE TO A FAILED CHECK. See above. ===")
        sys.exit(1)
