"""NumPy-only early-warning research with mature labels and fixed alarm budgets."""
from __future__ import annotations
import numpy as np
from . import early_warning as EW


def average_precision(y, score):
    """Non-interpolated AP, grouping score ties (same definition as sklearn)."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    if not y.any():
        return 0.0
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(y) - 1]
    tp = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / y.sum())


def features(L):
    n, T = L.shape
    d = EW.rel_level(L)
    scores = {"jump": EW.score_jump(L, seasonal=False), "seasonal_jump": EW.score_jump(L),
              "bocpd": EW.score_bocpd(L), "cusum": EW.score_cusum(L),
              "page_hinkley": EW.score_page_hinkley(L)}
    arrays = {k: np.nan_to_num(v) for k, v in scores.items()}
    for k in ["slope", "acceleration", "dispersion", "trend_consistency", "distance", "past_event"]:
        arrays[k] = np.zeros((n, T))
    for t in range(6, T):
        sig = EW.robust_sigma(d[:, :t])
        delta = np.diff(d[:, t - 3:t + 1], axis=1)
        arrays["slope"][:, t] = np.abs(delta.mean(axis=1)) / sig
        arrays["acceleration"][:, t] = np.abs(delta[:, -1] - delta[:, -2]) / sig
        arrays["dispersion"][:, t] = np.std(delta, axis=1) / sig
        arrays["trend_consistency"][:, t] = np.abs(np.sign(delta).sum(axis=1)) / 3
        arrays["distance"][:, t] = np.abs(d[:, t] - np.median(d[:, t - 6:t], axis=1)) / sig
        arrays["past_event"][:, t] = EW.events(L[:, :t + 1])[:, max(0, t - 7):t - 1].any(axis=1)
    # Modest nonlinear transforms, no series identifiers or forward information.
    return {k: np.log1p(np.maximum(v, 0)) for k, v in arrays.items()}


def logistic_predict(X, y, Z, penalty=0.03):
    """Regularized logistic IRLS. Standardization is fitted on training rows only."""
    X, y, Z = np.nan_to_num(X), np.asarray(y, float), np.nan_to_num(Z)
    if not y.any() or y.all():
        return np.full(len(Z), (y.sum() + 1) / (len(y) + 2))
    mu, sd = X.mean(axis=0), np.maximum(X.std(axis=0), 0.1)
    X = np.column_stack([np.ones(len(X)), np.clip((X - mu) / sd, -8, 8)])
    Z = np.column_stack([np.ones(len(Z)), np.clip((Z - mu) / sd, -8, 8)])
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean() / (1 - y.mean()))
    reg = penalty * np.eye(len(b)); reg[0, 0] = 1e-8
    for _ in range(25):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-5)
        grad = X.T @ (p - y) / len(y) + reg @ b
        H = (X.T * w) @ X / len(y) + reg
        step = np.linalg.solve(H, grad)
        b -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))


def stack(f, names, months):
    return np.column_stack([f[k][:, list(months)].T.ravel() for k in names])


def top_budget(score, fraction):
    alarm = np.zeros(len(score), dtype=bool)
    k = max(1, int(np.floor(len(score) * fraction)))
    alarm[np.argsort(-score, kind="stable")[:k]] = True
    return alarm
