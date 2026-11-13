"""Small from-scratch PyTorch regressors for Task 3 tabular experiments."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


NOMINAL = ("primary_game", "platform", "seasonal_spending_pattern", "segment")


def signed_log1p(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x + self.layers(x))


class TabularNet(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, dropout: float,
                 architecture: str, heads: int):
        super().__init__()
        if architecture == "plain":
            layers = []
            dim = input_dim
            for _ in range(depth):
                layers.extend((nn.Linear(dim, width), nn.LayerNorm(width),
                               nn.GELU(), nn.Dropout(dropout)))
                dim = width
            self.body = nn.Sequential(*layers)
        elif architecture == "residual":
            self.body = nn.Sequential(
                nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU(),
                *[ResidualBlock(width, dropout) for _ in range(depth)],
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.head = nn.Linear(width, heads)

    def forward(self, x):
        return self.head(self.body(x))


class NeuralSpendingRegressor(BaseEstimator, RegressorMixin):
    """Impute/scale tabular features, then fit a direct or zero-aware MLP.

    An internal validation split controls early stopping. The outer evaluation
    rows are never used to fit preprocessing, weights, or epoch count.
    """

    def __init__(self, *, architecture="plain", objective="direct", width=128,
                 depth=2, dropout=0.1, learning_rate=1e-3, weight_decay=1e-4,
                 batch_size=1024, max_epochs=40, patience=6, target_scale=10000.0,
                 gate_threshold=0.5, random_state=42, device="auto"):
        self.architecture = architecture
        self.objective = objective
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.target_scale = target_scale
        self.gate_threshold = gate_threshold
        self.random_state = random_state
        self.device = device

    def _preprocessor(self, columns):
        nominal = [c for c in NOMINAL if c in columns]
        numeric = [c for c in columns if c not in nominal]
        return ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("signed_log", FunctionTransformer(signed_log1p)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("nominal", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), nominal),
        ], remainder="drop", sparse_threshold=0)

    def _device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _decode(self, output):
        if self.objective == "direct":
            amount = torch.nn.functional.softplus(output[:, 0]) * self.target_scale
            return None, amount
        if self.objective == "hurdle":
            probability = torch.sigmoid(output[:, 0])
            amount = torch.nn.functional.softplus(output[:, 1]) * self.target_scale
            return probability, amount
        raise ValueError(f"Unknown objective: {self.objective}")

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame with named columns")
        target = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(X) != len(target) or not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("Target must be finite, nonnegative, and aligned with X")
        if self.objective not in ("direct", "hurdle"):
            raise ValueError("objective must be direct or hurdle")
        if not 0 <= self.gate_threshold <= 1:
            raise ValueError("gate_threshold must be in [0, 1]")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        train_idx, valid_idx = train_test_split(
            np.arange(len(target)), test_size=0.12, random_state=self.random_state,
            stratify=(target > 0).astype(int),
        )
        self.preprocessor_ = self._preprocessor(X.columns)
        train_x = np.asarray(self.preprocessor_.fit_transform(X.iloc[train_idx]), dtype=np.float32)
        valid_x = np.asarray(self.preprocessor_.transform(X.iloc[valid_idx]), dtype=np.float32)
        train_y, valid_y = target[train_idx], target[valid_idx]
        device = self._device()
        self.net_ = TabularNet(train_x.shape[1], self.width, self.depth, self.dropout,
                               self.architecture, 2 if self.objective == "hurdle" else 1).to(device)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
            pin_memory=device.type == "cuda", num_workers=0,
        )
        val_tensor = torch.from_numpy(valid_x).to(device)
        best, best_state, stale = float("inf"), None, 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            self.net_.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = self.net_(batch_x)
                if self.objective == "direct":
                    _, amount = self._decode(output)
                    loss = torch.mean(torch.abs(amount - batch_y)) / self.target_scale
                else:
                    probability, amount = self._decode(output)
                    occurrence = (batch_y > 0).float()
                    bce = torch.nn.functional.binary_cross_entropy(probability, occurrence)
                    positive = occurrence > 0
                    amount_loss = (torch.mean(torch.abs(amount[positive] - batch_y[positive]))
                                   / self.target_scale) if positive.any() else 0.0
                    loss = bce + amount_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                optimizer.step()

            self.net_.eval()
            with torch.no_grad():
                output = self.net_(val_tensor)
                probability, amount = self._decode(output)
                if self.objective == "hurdle":
                    prediction = torch.where(probability >= self.gate_threshold, amount, 0.0)
                else:
                    prediction = amount
                validation_mae = float(np.mean(np.abs(
                    prediction.cpu().numpy() - valid_y)))
            self.history_.append({"epoch": epoch, "internal_validation_MAE_thb": validation_mae})
            if validation_mae < best - 0.1:
                best, best_state, stale = validation_mae, copy.deepcopy(self.net_.state_dict()), 0
                self.best_epoch_ = epoch
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.to("cpu")
        self.net_.eval()
        self.internal_validation_MAE_thb_ = best
        return self

    def predict_components(self, X):
        if not hasattr(self, "net_"):
            raise ValueError("Fit the model before predicting")
        array = np.asarray(self.preprocessor_.transform(X), dtype=np.float32)
        pieces = []
        with torch.no_grad():
            for start in range(0, len(array), self.batch_size):
                output = self.net_(torch.from_numpy(array[start:start+self.batch_size]))
                probability, amount = self._decode(output)
                pieces.append((None if probability is None else probability.numpy(), amount.numpy()))
        amount = np.concatenate([part[1] for part in pieces])
        if self.objective == "direct":
            return None, np.maximum(amount, 0)
        probability = np.concatenate([part[0] for part in pieces])
        return probability, np.maximum(amount, 0)

    def predict(self, X):
        probability, amount = self.predict_components(X)
        if probability is None:
            return amount
        return np.where(probability >= self.gate_threshold, amount, 0)
