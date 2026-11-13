"""Task 2 decision rules selected from development-fold probabilities."""

from __future__ import annotations

import numpy as np


def restore_categorical_nulls(frame, categorical_columns):
    """Match the saved native categorical imputer's None missing marker."""
    result = frame.copy()
    columns = list(categorical_columns)
    result[columns] = result[columns].astype(object).where(result[columns].notna(), None)
    return result


class ClassBiasClassifier:
    """Apply fixed class log-probability biases to a fitted classifier."""

    def __init__(self, base_model, biases, categorical_columns):
        self.base_model = base_model
        self.biases = np.asarray(biases, dtype=float)
        self.categorical_columns = tuple(categorical_columns)
        self.classes_ = np.asarray(base_model.named_steps["classifier"].classes_)
        if self.biases.shape != self.classes_.shape or not np.isfinite(self.biases).all():
            raise ValueError("One finite bias is required for each class")

    def predict_proba(self, X):
        X = restore_categorical_nulls(X, self.categorical_columns)
        probabilities = np.asarray(self.base_model.predict_proba(X), dtype=float)
        if probabilities.shape[1] != len(self.classes_) or not np.isfinite(probabilities).all():
            raise ValueError("Unexpected class probabilities")
        return probabilities

    def predict(self, X):
        log_probabilities = np.log(np.clip(self.predict_proba(X), 1e-12, 1.0))
        return self.classes_[np.argmax(log_probabilities + self.biases, axis=1)]
