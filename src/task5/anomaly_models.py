"""Small out-of-sample anomaly score adapters for Task 5 notebooks."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class PCAReconstruction(BaseEstimator):
    def __init__(self, n_components: int = 12, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state).fit(X)
        return self

    def score_samples(self, X):
        projected = self.pca_.inverse_transform(self.pca_.transform(X))
        return -np.mean((np.asarray(X) - projected) ** 2, axis=1)


class EmpiricalTail(BaseEstimator):
    """Simple ECDF tail baseline; not the published ECOD implementation."""

    def __init__(self, top_k: int = 8):
        self.top_k = top_k

    def fit(self, X, y=None):
        self.sorted_ = np.sort(np.asarray(X), axis=0)
        return self

    def score_samples(self, X):
        X = np.asarray(X)
        n = len(self.sorted_)
        tails = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            # Inclusive tails keep common tied values (binary flags, imputed
            # medians) from looking maximally rare.
            left = np.searchsorted(self.sorted_[:, j], X[:, j], side="right")
            right = n - np.searchsorted(self.sorted_[:, j], X[:, j], side="left")
            tails[:, j] = -np.log((np.minimum(left, right) + 1) / (n + 1))
        k = min(self.top_k, X.shape[1])
        return -np.partition(tails, -k, axis=1)[:, -k:].sum(axis=1)


class BinaryAnomalyModel(BaseEstimator):
    """Fit preprocessing with the detector and expose 0/1 predictions."""

    def __init__(self, detector, anomaly_fraction: float = 0.05):
        self.detector = detector
        self.anomaly_fraction = anomaly_fraction

    def fit(self, X, y=None):
        if not 0 < self.anomaly_fraction < 0.5:
            raise ValueError("anomaly_fraction must be between 0 and 0.5")
        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer_.fit_transform(X)
        self.scaler_ = RobustScaler().fit(values)
        values = self.scaler_.transform(values)
        self.detector_ = clone(self.detector).fit(values)
        # The threshold is a fixed operational alert budget, not a tuned F3 threshold.
        self.threshold_ = float(np.quantile(-self.detector_.score_samples(values), 1 - self.anomaly_fraction))
        return self

    def anomaly_score(self, X):
        values = self.imputer_.transform(X[self.columns_])
        values = self.scaler_.transform(values)
        return -np.asarray(self.detector_.score_samples(values), dtype=float)

    def predict(self, X):
        return (self.anomaly_score(X) >= self.threshold_).astype("int8")
