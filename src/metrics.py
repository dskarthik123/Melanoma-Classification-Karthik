import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, f1_score, confusion_matrix)


def compute(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str = "roc_auc",
                 n: int = 1000, thr: float = 0.5, seed: int = 42):
    """Percentile bootstrap over validation samples.

    Resampling is at image level; note in the report that this does not account
    for within-patient correlation, so intervals are mildly optimistic.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(compute(y[idx], p[idx], thr)[metric])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)