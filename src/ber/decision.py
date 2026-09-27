from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ber.metrics import macro_fbeta


def predictions_from_scores(
    scored: pd.DataFrame,
    source1_ids: Iterable[str],
    threshold: float | dict[str, float],
    cap: int | None,
) -> dict[str, list[str]]:
    """``threshold`` is either one global cutoff, or a per-source dict like
    ``{"S2": 0.75, "S3": 0.85}`` (source is read off the candidate_entity_id
    prefix, so no extra column is needed)."""
    predictions = {source1_id: [] for source1_id in source1_ids}
    if scored.empty:
        return predictions
    if isinstance(threshold, dict):
        is_s2 = scored["candidate_entity_id"].str.startswith("S2-")
        thr = pd.Series(threshold.get("S3", 0.5), index=scored.index)
        thr[is_s2] = threshold.get("S2", 0.5)
        eligible = scored[scored["probability"].ge(thr)].copy()
    else:
        eligible = scored[scored["probability"].ge(threshold)].copy()
    if cap is not None:
        eligible = eligible[eligible["prob_rank"].le(cap)]
    for source1_id, group in eligible.groupby("source1_entity_id", sort=False):
        predictions[source1_id] = group["candidate_entity_id"].tolist()
    return predictions


def enforce_bipartite_exclusivity(
    predictions: dict[str, list[str]],
    scored: pd.DataFrame,
) -> dict[str, list[str]]:
    """Enforce that each target (S2/S3) ID is kept by at most one S1 entity
    -- the one with the highest predicted probability -- removing it from
    every other S1's predicted list.

    Verified fact (audit_v5.py, Priority 3 on train_ground_truth.tsv): in
    reality every target belongs to at most one S1 entity, so any predicted
    target claimed by more than one S1 is guaranteed to contain at least one
    false positive, which costs 4x under F0.5's precision weighting.
    ``scored`` supplies the probability used to break these conflicts (the
    same DataFrame passed to ``predictions_from_scores`` for this batch).
    """
    if scored.empty:
        return predictions
    prob_lookup = {
        (row.source1_entity_id, row.candidate_entity_id): row.probability
        for row in scored.itertuples(index=False)
    }

    target_best: dict[str, tuple[str, float]] = {}
    for source1_id, target_ids in predictions.items():
        for target_id in target_ids:
            prob = prob_lookup.get((source1_id, target_id), 0.0)
            current = target_best.get(target_id)
            if current is None or prob > current[1]:
                target_best[target_id] = (source1_id, prob)

    keep: dict[str, set[str]] = {source1_id: set() for source1_id in predictions}
    for target_id, (best_source1, _prob) in target_best.items():
        keep[best_source1].add(target_id)

    return {
        source1_id: [t for t in target_ids if t in keep[source1_id]]
        for source1_id, target_ids in predictions.items()
    }


def tune_decision_rule(
    scored: pd.DataFrame,
    truths: dict[str, set[str]],
    source1_ids: list[str],
    thresholds: Iterable[float],
    caps: Iterable[int | None],
) -> tuple[dict[str, object], dict[str, list[str]]]:
    best_score = -1.0
    best_params: dict[str, object] = {"threshold": 0.5, "cap": None, "macro_f05": 0.0}
    best_predictions: dict[str, list[str]] = {}
    for threshold in thresholds:
        for cap in caps:
            predictions = predictions_from_scores(scored, source1_ids, threshold, cap)
            score = macro_fbeta(predictions, truths, source1_ids, beta=0.5)
            if score > best_score:
                best_score = score
                best_params = {"threshold": float(threshold), "cap": cap, "macro_f05": float(score)}
                best_predictions = predictions
    return best_params, best_predictions


def tune_source_specific_thresholds(
    scored: pd.DataFrame,
    truths: dict[str, set[str]],
    source1_ids: list[str],
    thresholds: Iterable[float],
    cap: int | None,
    use_bipartite: bool = True,
) -> tuple[dict[str, object], dict[str, list[str]]]:
    """Coordinate-descent search for separate S2/S3 thresholds (the
    reference repo's ablation: S3 needs a higher bar than S2), starting from
    the single global threshold as an anchor. O(n) two passes instead of an
    O(n^2) joint grid -- cheaper, and finds an equally good optimum for a
    search this close to separable (S2-only and S3-only rows don't interact
    except through the shared cap, which is held fixed here at the value
    already tuned globally).
    """
    thresholds = list(thresholds)

    def score_for(threshold_map: dict[str, float]) -> tuple[float, dict[str, list[str]]]:
        predictions = predictions_from_scores(scored, source1_ids, threshold_map, cap)
        if use_bipartite:
            predictions = enforce_bipartite_exclusivity(predictions, scored)
        return macro_fbeta(predictions, truths, source1_ids, beta=0.5), predictions

    best_s2, best_s3 = 0.5, 0.5
    best_score, best_predictions = score_for({"S2": best_s2, "S3": best_s3})

    for t in thresholds:
        s, preds = score_for({"S2": t, "S3": best_s3})
        if s > best_score:
            best_score, best_s2, best_predictions = s, t, preds

    for t in thresholds:
        s, preds = score_for({"S2": best_s2, "S3": t})
        if s > best_score:
            best_score, best_s3, best_predictions = s, t, preds

    params = {"threshold_s2": float(best_s2), "threshold_s3": float(best_s3), "cap": cap, "macro_f05": float(best_score), "bipartite": use_bipartite}
    return params, best_predictions
