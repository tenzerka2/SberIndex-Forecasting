"""Structural-change early warning: event definition, online detectors and a supervised classifier.

Why the previous labelling was replaced
---------------------------------------
pipeline.change_points() scored level shifts on the RAW log level. Municipal spending has a strong
national seasonality, so "shocks" were mostly calendar effects: 23% of all series were flagged in
2023-11 (December spike), 11% in 2023-04, 12% in 2024-11 and ~0% in May. A classifier trained on
that target mainly learns the calendar. Here the signal is the panel-relative level
d_it = log y_it - median_j log y_jt, which removes the common seasonal/macro path, and an extra
"seasonal echo" filter drops shifts that repeat 12 months apart (municipality-specific seasonality,
e.g. resort areas).

Event (ground truth) at month t for series i:
    shift_t = median(d[t:t+3]) - median(d[t-3:t]),   z_t = shift_t / sigma_i
    event if |z_t| >= KAPPA, all three post points lie beyond half the shift on the same side, no
    same-sign shift of >= half the size at t-12 / t+12 (when observable), local maximum within +/-2.
Labels need months up to t+2. They are always computed from data truncated at a given cutoff, so a
model trained at time t never sees labels that depend on data after t.

Online protocol: a detector score s_it uses only months <= t. Row (i, t) is positive if an event
started at tau in [t-2, t] (the change is detectable with delay 0..2 months).
"""
from __future__ import annotations

import numpy as np

KAPPA = 3.0
MIN_SIGMA = 0.01


def rel_level(L: np.ndarray) -> np.ndarray:
    return L - np.median(L, axis=0)


def robust_sigma(d: np.ndarray) -> np.ndarray:
    """1.4826 * MAD of month-to-month changes, per series (uses all columns given)."""
    dd = np.diff(d, axis=1)
    med = np.median(dd, axis=1, keepdims=True)
    return np.maximum(1.4826 * np.median(np.abs(dd - med), axis=1), MIN_SIGMA)


def _shift(d: np.ndarray, t: int) -> np.ndarray:
    return np.median(d[:, t : t + 3], axis=1) - np.median(d[:, t - 3 : t], axis=1)


def events(L: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """Binary (n x T) event matrix computed from the given (possibly truncated) history only."""
    d = rel_level(L)
    n, T = d.shape
    sig = robust_sigma(d)
    Z = np.zeros((n, T))
    S = np.full((n, T), np.nan)
    for t in range(3, T - 2):
        S[:, t] = _shift(d, t)
        Z[:, t] = S[:, t] / sig
    ev = np.zeros((n, T), dtype=bool)
    for t in range(3, T - 2):
        s = S[:, t]
        pre = np.median(d[:, t - 3 : t], axis=1)
        post = d[:, t : t + 3] - pre[:, None]
        persistent = np.all(np.sign(post) == np.sign(s)[:, None], axis=1) & np.all(
            np.abs(post) >= 0.5 * np.abs(s)[:, None], axis=1
        )
        echo = np.zeros(n, dtype=bool)
        for u in (t - 12, t + 12):
            if 3 <= u < T - 2:
                echo |= (np.sign(S[:, u]) == np.sign(s)) & (np.abs(S[:, u]) >= 0.5 * np.abs(s))
        ev[:, t] = (np.abs(Z[:, t]) >= kappa) & persistent & ~echo
    # keep the strongest of events within +/- 2 months
    absz = np.where(ev, np.abs(Z), 0)
    keep = ev.copy()
    for t in range(T):
        lo, hi = max(0, t - 2), min(T, t + 3)
        keep[:, t] &= absz[:, t] >= absz[:, lo:hi].max(axis=1)
    return keep


def row_labels(ev: np.ndarray, delay: int = 2) -> np.ndarray:
    """Row (i,t) positive if an event started in [t-delay, t]."""
    n, T = ev.shape
    y = np.zeros((n, T), dtype=bool)
    for t in range(T):
        y[:, t] = ev[:, max(0, t - delay) : t + 1].any(axis=1)
    return y


# ---------------------------------------------------------------------------------------------
# Online detectors: every function returns score[:, t] computed from columns <= t only.
# ---------------------------------------------------------------------------------------------
def _online(L: np.ndarray, t_from: int, fn) -> np.ndarray:
    n, T = L.shape
    out = np.full((n, T), np.nan)
    for t in range(t_from, T):
        out[:, t] = fn(L[:, : t + 1])
    return out


def score_jump(L, t_from=6, seasonal=True):
    """Robust standardized jump of the last k=1..3 months vs the 3 months before (max over k).

    seasonal=True subtracts the same statistic 12 months earlier (when observable), which removes
    municipality-specific seasonality.
    """

    def fn(H):
        d = rel_level(H)
        t = d.shape[1] - 1
        sig = robust_sigma(d)
        best = np.zeros(d.shape[0])
        for k in (1, 2, 3):
            if t - k - 2 < 0:
                continue
            j = np.median(d[:, t - k + 1 : t + 1], axis=1) - np.median(d[:, t - k - 2 : t - k + 1], axis=1)
            if seasonal and t - 12 - k - 2 >= 0:
                j = j - (np.median(d[:, t - 12 - k + 1 : t - 11], axis=1) - np.median(d[:, t - 12 - k - 2 : t - 12 - k + 1], axis=1))
            best = np.maximum(best, np.abs(j) / sig)
        return best

    return _online(L, t_from, fn)


def _residual_stream(H):
    """Standardized deviations of d from its running median of the previous 6 months."""
    d = rel_level(H)
    sig = robust_sigma(d)
    T = d.shape[1]
    r = np.zeros_like(d)
    for t in range(3, T):
        r[:, t] = (d[:, t] - np.median(d[:, max(0, t - 6) : t], axis=1)) / sig
    return r


def score_cusum(L, t_from=6, k=0.5):
    """Two-sided CUSUM on standardized residuals (data <= t)."""

    def fn(H):
        r = _residual_stream(H)
        sp = np.zeros(r.shape[0]); sn = np.zeros(r.shape[0])
        for t in range(3, r.shape[1]):
            sp = np.maximum(0, sp + r[:, t] - k)
            sn = np.maximum(0, sn - r[:, t] - k)
        return np.maximum(sp, sn)

    return _online(L, t_from, fn)


def score_page_hinkley(L, t_from=6, delta=0.5):
    def fn(H):
        r = _residual_stream(H)[:, 3:]
        out = np.zeros(r.shape[0])
        for sgn in (1, -1):
            x = sgn * r
            mean = np.cumsum(x, axis=1) / np.arange(1, x.shape[1] + 1)
            m = np.cumsum(x - mean - delta, axis=1)
            out = np.maximum(out, m[:, -1] - m.min(axis=1))
        return out

    return _online(L, t_from, fn)


def score_bocpd(L, t_from=6, hazard=1 / 24, prior_var_mult=25.0, recent=2):
    """Bayesian online change-point detection (Adams & MacKay 2007).

    Gaussian observations with known per-series noise variance (robust sigma^2 from data <= t) and a
    conjugate Normal prior on the segment mean. Score = posterior P(run length <= recent).
    """

    def fn(H):
        d = rel_level(H)
        n, T = d.shape
        s2 = robust_sigma(d) ** 2 / 2  # level noise ~ diff noise / sqrt(2)
        mu0 = d[:, :3].mean(axis=1)
        v0 = prior_var_mult * s2
        # run-length distribution R[n, r]; sufficient stats per run length
        R = np.ones((n, 1))
        mu = mu0[:, None].copy(); var = v0[:, None].copy()
        for t in range(T):
            x = d[:, t][:, None]
            pv = var + s2[:, None]
            pred = np.exp(-0.5 * (x - mu) ** 2 / pv) / np.sqrt(2 * np.pi * pv)
            growth = R * pred * (1 - hazard)
            cp = (R * pred * hazard).sum(axis=1, keepdims=True)
            R = np.concatenate([cp, growth], axis=1)
            R /= R.sum(axis=1, keepdims=True)
            # update posteriors: new run starts from prior
            post_var = 1 / (1 / var + 1 / s2[:, None])
            post_mu = post_var * (mu / var + x / s2[:, None])
            mu = np.concatenate([mu0[:, None], post_mu], axis=1)
            var = np.concatenate([v0[:, None], post_var], axis=1)
        return R[:, 1 : recent + 2].sum(axis=1)  # run length 0..recent (index 0 is "cp before x_t")

    return _online(L, t_from, fn)


def score_pelt(L, t_from=6, pen_mult=3.0, recent=2):
    """PELT (ruptures, L2 cost) on the expanding window; score = standardized size of a change point
    found in the last `recent`+1 months (0 if none)."""
    import ruptures as rpt

    def fn(H):
        d = rel_level(H)
        sig = robust_sigma(d)
        T = d.shape[1]
        out = np.zeros(d.shape[0])
        for i in range(d.shape[0]):
            x = d[i] / sig[i]
            bkps = rpt.Pelt(model="l2", min_size=2, jump=1).fit(x).predict(pen=pen_mult * np.log(T))
            for b in bkps[:-1]:
                if b >= T - 1 - recent:
                    out[i] = max(out[i], abs(np.median(x[b:]) - np.median(x[max(0, b - 3) : b])))
        return out

    return _online(L, t_from, fn)


def feature_tensor(L: np.ndarray, scores: dict[str, np.ndarray], extra: dict[str, np.ndarray] | None = None):
    """Per (i,t) features for the supervised classifier; all computed from data <= t."""
    d = rel_level(L)
    n, T = d.shape
    feats = {k: v for k, v in scores.items()}
    sig_online = np.full((n, T), np.nan)
    vol3 = np.full((n, T), np.nan)
    dev = np.full((n, T), np.nan)
    yoy_rel = np.full((n, T), np.nan)
    acc = np.full((n, T), np.nan)
    for t in range(3, T):
        dt = rel_level(L[:, : t + 1])
        sig_online[:, t] = robust_sigma(dt)
        vol3[:, t] = np.std(np.diff(dt[:, -4:], axis=1), axis=1) / sig_online[:, t]
        dev[:, t] = (dt[:, t] - np.median(dt[:, :t], axis=1)) / sig_online[:, t]
        acc[:, t] = ((dt[:, t] - dt[:, t - 1]) - (dt[:, t - 1] - dt[:, t - 2])) / sig_online[:, t]
        if t >= 12:
            yoy_rel[:, t] = (dt[:, t] - dt[:, t - 12]) / sig_online[:, t]
    feats.update(sigma=sig_online, vol3=vol3, dev=dev, yoy_rel=yoy_rel, accel=acc,
                 size=L.copy(), d_level=d)
    if extra:
        feats.update(extra)
    return feats


def pr_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import average_precision_score

    a = s >= thr
    tp = int((a & y).sum()); fp = int((a & ~y).sum()); fn = int((~a & y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {
        "PR_AUC": float(average_precision_score(y, s)) if y.any() else float("nan"),
        "precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-12),
        "false_alarms_per_100": 100 * fp / len(y), "alarms_per_100": 100 * a.sum() / len(y),
        "threshold": float(thr), "n_rows": int(len(y)), "positives": int(y.sum()),
    }


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[int(np.argmax(f1[:-1]))]) if len(t) else float("inf")
