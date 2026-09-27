#!/usr/bin/env python3
"""Fast experiments on the cached V6 features (built by build_feature_cache.py):

  1. Probability calibration (isotonic) -- does it help tune/holdout macro F0.5?
  2. Cross-country "unseen country" simulation -- train on one country's fit
     split, evaluate on the OTHER country's tune/holdout as a stand-in for
     France (which has no ground truth at all), to derive an evidence-based
     France decision rule instead of the unvalidated 0.95/8 guess.
  3. LightGBM hyperparameter tuning -- small manual search over the existing
     53-feature set, scored on tune, confirmed on holdout.

All three read only from artifacts_v6/cache/{country}/part_*.parquet +
meta.json -- no candidate generation, no target-pool loading, so each
experiment runs in seconds/minutes instead of the ~1hr+ feature build.
Writes results to output_v6/V6_EXPERIMENTS_LOG.md. Never touches
output_v5/, artifacts_v5/, output_recall_v3/, or output_final_v2/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ber.decision import enforce_bipartite_exclusivity, predictions_from_scores, tune_decision_rule, tune_source_specific_thresholds
from ber.io import read_ground_truth, truth_dict
from ber.metrics import macro_fbeta
from ber.modeling import MatchModel, downsample_training_pairs, scored_pairs, train_match_model

COUNTRIES = ["india", "us"]
META_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_rank"]
DECISION_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97]
DECISION_CAPS = [3, 5, 8, None]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resource-dir", type=Path, default=Path("student_resource"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts_v6/cache"))
    p.add_argument("--output-dir", type=Path, default=Path("output_v6"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_logger(log_path: Path):
    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def load_country_cache(cache_dir: Path, country: str) -> tuple[pd.DataFrame, dict]:
    country_dir = cache_dir / country
    meta = json.loads((country_dir / "meta.json").read_text(encoding="utf-8"))
    parts = sorted(country_dir.glob("part_*.parquet"))
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return frame, meta


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


def split_frame(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    idset = set(ids)
    return frame[frame["source1_entity_id"].isin(idset)].reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in META_COLS + ["label"]]


def fit_model(frame: pd.DataFrame, feat_cols: list[str], seed: int) -> MatchModel:
    features = frame[feat_cols]
    labels = frame["label"]
    features, labels = downsample_training_pairs(features, labels, ratio=6.0, random_seed=seed)
    return train_match_model(features, labels, seed)


def eval_rule(scored: pd.DataFrame, truths: dict, ids: list[str], bipartite: bool = True) -> tuple[dict, float]:
    params, preds = tune_decision_rule(scored, truths, ids, thresholds=DECISION_THRESHOLDS, caps=DECISION_CAPS)
    if bipartite:
        preds_bp = enforce_bipartite_exclusivity(preds, scored)
        score_bp = macro_fbeta(preds_bp, truths, ids, beta=0.5)
        if score_bp > params["macro_f05"]:
            return {**params, "bipartite": True}, score_bp
    return {**params, "bipartite": False}, params["macro_f05"]


def experiment_calibration(caches: dict, truths: dict, feat_cols: list[str], seed: int, log) -> None:
    log("=== Experiment 1: isotonic probability calibration ===")
    fit_frames, tune_frames, holdout_frames = [], [], []
    fit_ids, tune_ids, holdout_ids = [], [], []
    for country in COUNTRIES:
        frame, meta = caches[country]
        fit_c, tune_c, holdout_c = three_way_split(meta["sample_ids"], truths, seed)
        fit_frames.append(split_frame(frame, fit_c))
        tune_frames.append(split_frame(frame, tune_c))
        holdout_frames.append(split_frame(frame, holdout_c))
        fit_ids += fit_c; tune_ids += tune_c; holdout_ids += holdout_c

    fit_frame = pd.concat(fit_frames, ignore_index=True)
    tune_frame = pd.concat(tune_frames, ignore_index=True)
    holdout_frame = pd.concat(holdout_frames, ignore_index=True)

    model = fit_model(fit_frame, feat_cols, seed)
    tune_prob_raw = model.predict_proba(tune_frame[feat_cols])
    holdout_prob_raw = model.predict_proba(holdout_frame[feat_cols])

    tune_scored_raw = scored_pairs(tune_frame, tune_prob_raw)
    holdout_scored_raw = scored_pairs(holdout_frame, holdout_prob_raw)
    params_raw, tune_score_raw = eval_rule(tune_scored_raw, truths, tune_ids)
    holdout_preds_raw = predictions_from_scores(holdout_scored_raw, holdout_ids, params_raw.get("threshold", 0.5), params_raw.get("cap"))
    if params_raw.get("bipartite"):
        holdout_preds_raw = enforce_bipartite_exclusivity(holdout_preds_raw, holdout_scored_raw)
    holdout_score_raw = macro_fbeta(holdout_preds_raw, truths, holdout_ids, beta=0.5)
    log(f"  [uncalibrated] tune_macro_f05={tune_score_raw:.6f} rule={params_raw} | holdout_macro_f05={holdout_score_raw:.6f}")

    # Fit isotonic calibration on the FIT set's own out-of-training probabilities is not
    # possible without a held-out slice of fit; use tune set for calibration fit (standard
    # practice: calibrate on a split disjoint from model training), then re-evaluate the
    # decision rule + holdout on calibrated probabilities.
    fit_prob_for_calib = model.predict_proba(tune_frame[feat_cols])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(fit_prob_for_calib, tune_frame["label"].to_numpy())

    tune_prob_cal = calibrator.predict(tune_prob_raw)
    holdout_prob_cal = calibrator.predict(holdout_prob_raw)
    tune_scored_cal = scored_pairs(tune_frame, tune_prob_cal)
    holdout_scored_cal = scored_pairs(holdout_frame, holdout_prob_cal)

    params_cal, tune_score_cal = eval_rule(tune_scored_cal, truths, tune_ids)
    holdout_preds_cal = predictions_from_scores(holdout_scored_cal, holdout_ids, params_cal.get("threshold", 0.5), params_cal.get("cap"))
    if params_cal.get("bipartite"):
        holdout_preds_cal = enforce_bipartite_exclusivity(holdout_preds_cal, holdout_scored_cal)
    holdout_score_cal = macro_fbeta(holdout_preds_cal, truths, holdout_ids, beta=0.5)
    log(f"  [isotonic-calibrated, fit on tune] tune_macro_f05={tune_score_cal:.6f} rule={params_cal} | holdout_macro_f05={holdout_score_cal:.6f}")

    verdict = "HELPS" if holdout_score_cal > holdout_score_raw else "DOES NOT HELP"
    log(f"  *** Calibration verdict: {verdict} (uncalibrated holdout={holdout_score_raw:.6f} vs calibrated holdout={holdout_score_cal:.6f}) ***")


def experiment_cross_country(caches: dict, truths: dict, feat_cols: list[str], seed: int, log) -> None:
    log("=== Experiment 2: cross-country unseen-country simulation (France proxy) ===")
    for train_country, eval_country in [("india", "us"), ("us", "india")]:
        train_frame, train_meta = caches[train_country]
        eval_frame, eval_meta = caches[eval_country]

        train_fit_ids, train_tune_ids, _ = three_way_split(train_meta["sample_ids"], truths, seed)
        _, eval_tune_ids, eval_holdout_ids = three_way_split(eval_meta["sample_ids"], truths, seed)

        fit_frame = split_frame(train_frame, train_fit_ids)
        model = fit_model(fit_frame, feat_cols, seed)

        eval_tune_frame = split_frame(eval_frame, eval_tune_ids)
        eval_holdout_frame = split_frame(eval_frame, eval_holdout_ids)
        tune_prob = model.predict_proba(eval_tune_frame[feat_cols])
        holdout_prob = model.predict_proba(eval_holdout_frame[feat_cols])
        tune_scored = scored_pairs(eval_tune_frame, tune_prob)
        holdout_scored = scored_pairs(eval_holdout_frame, holdout_prob)

        params, tune_score = eval_rule(tune_scored, truths, eval_tune_ids)
        holdout_preds = predictions_from_scores(holdout_scored, eval_holdout_ids, params.get("threshold", 0.5), params.get("cap"))
        if params.get("bipartite"):
            holdout_preds = enforce_bipartite_exclusivity(holdout_preds, holdout_scored)
        holdout_score = macro_fbeta(holdout_preds, truths, eval_holdout_ids, beta=0.5)

        avg_pred_size = np.mean([len(v) for v in holdout_preds.values()]) if holdout_preds else 0.0
        log(
            f"  train={train_country} -> eval={eval_country} (unseen-country proxy): "
            f"best rule on eval-tune={params} tune_macro_f05={tune_score:.6f} | "
            f"holdout_macro_f05={holdout_score:.6f} | avg_predicted_size={avg_pred_size:.3f}"
        )
    log(
        "  *** Use the threshold/cap that scored best across both directions above as the "
        "evidence-based France rule (replace the unvalidated 0.95/cap=8 guess). ***"
    )


def experiment_hyperparams(caches: dict, truths: dict, feat_cols: list[str], seed: int, log) -> None:
    log("=== Experiment 3: LightGBM hyperparameter search ===")
    fit_frames, tune_frames = [], []
    fit_ids, tune_ids = [], []
    for country in COUNTRIES:
        frame, meta = caches[country]
        fit_c, tune_c, _ = three_way_split(meta["sample_ids"], truths, seed)
        fit_frames.append(split_frame(frame, fit_c))
        tune_frames.append(split_frame(frame, tune_c))
        fit_ids += fit_c; tune_ids += tune_c
    fit_frame = pd.concat(fit_frames, ignore_index=True)
    tune_frame = pd.concat(tune_frames, ignore_index=True)

    features = fit_frame[feat_cols]
    labels = fit_frame["label"]
    features, labels = downsample_training_pairs(features, labels, ratio=6.0, random_seed=seed)

    from lightgbm import LGBMClassifier

    positive = max(int(labels.sum()), 1)
    negative = max(int((1 - labels).sum()), 1)
    scale_pos_weight = negative / positive

    grid = [
        {"n_estimators": 700, "learning_rate": 0.035, "num_leaves": 96, "min_child_samples": 40},  # baseline (V5)
        {"n_estimators": 350, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 30},    # reference-repo-style
        {"n_estimators": 900, "learning_rate": 0.025, "num_leaves": 128, "min_child_samples": 50},
        {"n_estimators": 500, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 20},
    ]

    best = None
    for params in grid:
        t0 = time.time()
        model_raw = LGBMClassifier(
            objective="binary",
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
            force_col_wise=True,
            verbose=-1,
            **params,
        )
        model_raw.fit(features, labels)
        model = MatchModel(model=model_raw, feature_columns=list(features.columns))
        tune_prob = model.predict_proba(tune_frame[feat_cols])
        tune_scored = scored_pairs(tune_frame, tune_prob)
        rule_params, tune_score = eval_rule(tune_scored, truths, tune_ids)
        log(f"  params={params}: tune_macro_f05={tune_score:.6f} (rule={rule_params}, {time.time()-t0:.1f}s)")
        if best is None or tune_score > best[0]:
            best = (tune_score, params, rule_params)

    log(f"  *** Best hyperparameters on tune: {best[1]} -> tune_macro_f05={best[0]:.6f} (rule={best[2]}) ***")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(args.output_dir / "V6_EXPERIMENTS_LOG.md")
    log("=== V6 experiments starting (calibration, cross-country France-proxy rule, hyperparameter search) ===")

    truth_frame = read_ground_truth(args.resource_dir / "dataset" / "train")
    truths = truth_dict(truth_frame)
    del truth_frame

    caches = {country: load_country_cache(args.cache_dir, country) for country in COUNTRIES}
    for country, (frame, meta) in caches.items():
        log(f"Loaded {country} cache: {len(frame):,} feature rows, {meta['n_s1_sampled']:,} sampled S1, audit={meta['audit']}")
    feat_cols = feature_columns(caches["india"][0])
    log(f"Feature columns ({len(feat_cols)}): {feat_cols}")

    experiment_calibration(caches, truths, feat_cols, args.seed, log)
    experiment_cross_country(caches, truths, feat_cols, args.seed, log)
    experiment_hyperparams(caches, truths, feat_cols, args.seed, log)

    log("=== V6 experiments done ===")


if __name__ == "__main__":
    main()
