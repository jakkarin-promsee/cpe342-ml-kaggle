"""Task 3 candidates built only from competition development rows."""

from __future__ import annotations

import numpy as np


class PrefitBlend:
    """Fixed weighted average of two previously fitted spending models."""

    def __init__(self, catboost_model, lightgbm_model, catboost_weight: float):
        if not 0 <= catboost_weight <= 1:
            raise ValueError("catboost_weight must be in [0, 1]")
        self.catboost_model = catboost_model
        self.lightgbm_model = lightgbm_model
        self.catboost_weight = float(catboost_weight)

    def predict(self, X):
        cat = np.asarray(self.catboost_model.predict(X), dtype=float).reshape(-1)
        lgb = np.asarray(self.lightgbm_model.predict(X), dtype=float).reshape(-1)
        if cat.shape != lgb.shape or not (np.isfinite(cat).all() and np.isfinite(lgb).all()):
            raise ValueError("Component predictions are misaligned or nonfinite")
        return np.maximum(self.catboost_weight * cat + (1 - self.catboost_weight) * lgb, 0)


class PositiveGate:
    """Predict zero below a development-selected nonzero probability cutoff."""

    def __init__(self, occurrence_model, positive_amount_model, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.occurrence_model = occurrence_model
        self.positive_amount_model = positive_amount_model
        self.threshold = float(threshold)

    def predict(self, X):
        probability = np.asarray(self.occurrence_model.predict_proba(X), dtype=float)[:, 1]
        positive_amount = np.asarray(self.positive_amount_model.predict(X), dtype=float).reshape(-1)
        if probability.shape != positive_amount.shape or not (
            np.isfinite(probability).all() and np.isfinite(positive_amount).all()
        ):
            raise ValueError("Component predictions are misaligned or nonfinite")
        return np.where(probability >= self.threshold, np.maximum(positive_amount, 0), 0)


class WeightedPair:
    """Fixed average of two nonnegative, previously fitted regressors."""

    def __init__(self, first_model, second_model, first_weight: float):
        if not 0 <= first_weight <= 1:
            raise ValueError("first_weight must be in [0, 1]")
        self.first_model = first_model
        self.second_model = second_model
        self.first_weight = float(first_weight)

    def predict(self, X):
        first = np.asarray(self.first_model.predict(X), dtype=float).reshape(-1)
        second = np.asarray(self.second_model.predict(X), dtype=float).reshape(-1)
        if first.shape != second.shape or not (np.isfinite(first).all() and np.isfinite(second).all()):
            raise ValueError("Component predictions are misaligned or nonfinite")
        return np.maximum(self.first_weight * first + (1 - self.first_weight) * second, 0)
