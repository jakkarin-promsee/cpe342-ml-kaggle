"""Task 5 unlabeled behavior scores; no hidden labels or external data."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


class PeerChangeScorer(BaseEstimator):
    """Rank large within-account changes against similar baseline accounts."""

    def __init__(self, change_columns, cohort_columns, neighbors=128, chunk_size=512):
        self.change_columns = change_columns
        self.cohort_columns = cohort_columns
        self.neighbors = neighbors
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        self.change_columns_ = list(self.change_columns)
        self.cohort_columns_ = list(self.cohort_columns)
        if not self.change_columns_ or not self.cohort_columns_:
            raise ValueError("Both change and cohort columns are required")
        if len(X) < 2:
            raise ValueError("At least two unlabeled accounts are required")
        self.cohort_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        cohort = self.cohort_imputer_.fit_transform(X[self.cohort_columns_])
        self.cohort_scaler_ = RobustScaler().fit(cohort)
        cohort = self.cohort_scaler_.transform(cohort)
        self.change_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.reference_changes_ = self.change_imputer_.fit_transform(X[self.change_columns_])
        self.global_floor_ = np.maximum(
            np.subtract(*np.percentile(self.reference_changes_, [75, 25], axis=0)) * 0.25,
            1e-6,
        )
        self.neighbors_ = NearestNeighbors(n_neighbors=min(self.neighbors, len(X)))
        self.neighbors_.fit(cohort)
        return self

    def anomaly_score(self, X):
        cohort = self.cohort_scaler_.transform(
            self.cohort_imputer_.transform(X[self.cohort_columns_])
        )
        changes = self.change_imputer_.transform(X[self.change_columns_])
        output = np.empty(len(X), dtype=float)
        for start in range(0, len(X), self.chunk_size):
            end = min(start + self.chunk_size, len(X))
            indexes = self.neighbors_.kneighbors(cohort[start:end], return_distance=False)
            peer = self.reference_changes_[indexes]
            center = np.median(peer, axis=1)
            spread = np.percentile(peer, 75, axis=1) - np.percentile(peer, 25, axis=1)
            excess = np.maximum(changes[start:end] - center, 0)
            z = excess / (spread + self.global_floor_)
            width = min(3, z.shape[1])
            output[start:end] = np.partition(z, -width, axis=1)[:, -width:].mean(axis=1)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite peer-change scores")
        return output


class BotRegularityScorer(BaseEstimator):
    """Rank simultaneous high activity and regularity using unlabeled ECDFs."""

    def __init__(self, activity_columns, regularity_high, regularity_low):
        self.activity_columns = activity_columns
        self.regularity_high = regularity_high
        self.regularity_low = regularity_low

    def fit(self, X, y=None):
        self.activity_columns_ = list(self.activity_columns)
        self.regularity_high_ = list(self.regularity_high)
        self.regularity_low_ = list(self.regularity_low)
        self.columns_ = self.activity_columns_ + self.regularity_high_ + self.regularity_low_
        if len(set(self.columns_)) != len(self.columns_) or not all(
            (self.activity_columns_, self.regularity_high_, self.regularity_low_)
        ):
            raise ValueError("Nonempty, distinct score columns are required")
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer_.fit_transform(X[self.columns_])
        self.sorted_ = np.sort(values, axis=0)
        return self

    def anomaly_score(self, X):
        values = self.imputer_.transform(X[self.columns_])
        ranks = np.empty_like(values, dtype=float)
        for j in range(values.shape[1]):
            ranks[:, j] = np.searchsorted(self.sorted_[:, j], values[:, j], side="right") / len(self.sorted_)
        a = len(self.activity_columns_)
        h = len(self.regularity_high_)
        activity = ranks[:, :a].mean(axis=1)
        regularity = np.concatenate([ranks[:, a:a+h], 1 - ranks[:, a+h:]], axis=1).mean(axis=1)
        return np.minimum(activity, regularity)
