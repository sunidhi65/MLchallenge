from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MatchModel:
    model: object
    feature_columns: list[str]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        frame = features.reindex(columns=self.feature_columns, fill_value=0)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(frame)[:, 1]
        raw = self.model.predict(frame)
        return np.asarray(raw, dtype=float)


def train_match_model(features: pd.DataFrame, labels: pd.Series, random_seed: int = 42) -> MatchModel:
    """Train the main pairwise matcher.

    LightGBM is preferred. A scikit-learn fallback is provided for environments
    where LightGBM is unavailable.
    """

    feature_columns = list(features.columns)
    x = features[feature_columns]
    y = labels.astype(int)

    from sklearn.calibration import CalibratedClassifierCV

    try:
        from lightgbm import LGBMClassifier

        positive = max(int(y.sum()), 1)
        negative = max(int((1 - y).sum()), 1)
        scale_pos_weight = negative / positive
        base_model = LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=128,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=50,
            reg_alpha=0.05,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_seed,
            n_jobs=-1,
            force_col_wise=True,
            verbose=-1,
        )
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    except Exception as lightgbm_error:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            base_model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=450,
                max_leaf_nodes=63,
                l2_regularization=0.1,
                random_state=random_seed,
            )
            model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install requirements.txt. The matcher needs lightgbm or scikit-learn."
            ) from exc
        print(f"LightGBM unavailable; using scikit-learn HistGradientBoostingClassifier. ({lightgbm_error})")

    model.fit(x, y)
    return MatchModel(model=model, feature_columns=feature_columns)


def downsample_training_pairs(
    features: pd.DataFrame,
    labels: pd.Series,
    ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep all positives and downsample negatives to a configurable ratio."""

    if ratio <= 0:
        return features, labels
    rng = np.random.default_rng(random_seed)
    labels = labels.astype(int).reset_index(drop=True)
    features = features.reset_index(drop=True)
    pos_idx = labels[labels.eq(1)].index.to_numpy()
    neg_idx = labels[labels.eq(0)].index.to_numpy()
    max_neg = min(len(neg_idx), int(max(len(pos_idx), 1) * ratio))
    chosen_neg = rng.choice(neg_idx, size=max_neg, replace=False) if len(neg_idx) > max_neg else neg_idx
    chosen = np.concatenate([pos_idx, chosen_neg])
    rng.shuffle(chosen)
    return features.iloc[chosen].reset_index(drop=True), labels.iloc[chosen].reset_index(drop=True)


def scored_pairs(candidates: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    out = candidates[["source1_entity_id", "candidate_entity_id", "candidate_rank"]].copy()
    out["probability"] = probabilities
    out = out.sort_values(
        ["source1_entity_id", "probability", "candidate_rank"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    out["prob_rank"] = out.groupby("source1_entity_id").cumcount() + 1
    return out
