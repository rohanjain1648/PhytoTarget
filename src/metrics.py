"""Virtual-screening evaluation metrics.

ROC-AUC alone is a poor description of a screening model. A screen is used by
taking the top fraction of a ranked list into the lab, so what matters is how
many real actives land in that top slice. Enrichment factor and BEDROC answer
that question directly; ROC-AUC averages over the whole list, including the
region nobody will ever look at.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values.

    Implemented here rather than imported so the project does not depend on a
    whole statistics package for twelve lines of sorting, and so the correction
    applied to thousands of taxon tests is readable in the repository that
    reports them.

    Sort p ascending, scale each by n/rank, then enforce monotonicity by taking a
    running minimum from the largest p downward - without that step a q-value can
    come out smaller than the q-value of a more significant test.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # monotone from the top
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.01) -> float:
    """EF@x%: how many times more actives the top x% holds than random selection.

    EF = 1.0 means the model is worthless. For a 1% slice the ceiling is
    1 / 0.01 = 100 (only reachable when actives are rarer than the slice).
    """
    n = len(y_true)
    n_sel = max(1, int(round(n * fraction)))
    n_act = int(y_true.sum())
    if n_act == 0 or n == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    hits = int(y_true[order[:n_sel]].sum())
    return float((hits / n_sel) / (n_act / n))


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly, 2007) — early-recognition-weighted ranking score.

    alpha=20 weights roughly the top 8% of the list. Returns [0, 1]; 0.5 is the
    expected value for a random ranking only when actives are rare, so it should
    be read next to the active fraction rather than against a fixed 0.5 baseline.
    """
    n = len(y_true)
    n_act = int(y_true.sum())
    if n_act == 0 or n_act == n:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    ranks = np.where(y_true[order] == 1)[0] + 1  # 1-indexed ranks of actives

    ra = n_act / n
    rie_num = np.sum(np.exp(-alpha * ranks / n)) / n_act
    rie_den = (1.0 / n) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    rie = rie_num / rie_den

    factor = ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))
    offset = 1.0 / (1 - np.exp(alpha * (1 - ra)))
    return float(rie * factor + offset)


def evaluate(y_true: np.ndarray, y_score: np.ndarray,
             ef_fractions=(0.01, 0.05), bedroc_alpha: float = 20.0) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    out = {
        "n": int(len(y_true)),
        "n_actives": int(y_true.sum()),
        "active_fraction": float(y_true.mean()) if len(y_true) else 0.0,
    }
    if len(np.unique(y_true)) < 2:
        out.update({"roc_auc": float("nan"), "pr_auc": float("nan")})
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    for frac in ef_fractions:
        out[f"ef_{int(frac * 100)}pct"] = enrichment_factor(y_true, y_score, frac)
    out[f"bedroc_{int(bedroc_alpha)}"] = bedroc(y_true, y_score, bedroc_alpha)
    return out


def decoy_augmented_eval(active_scores: np.ndarray, decoy_scores: np.ndarray) -> dict:
    """Enrichment when held-out actives are hidden in a realistic screening deck.

    Why this metric exists. ChEMBL is publication-biased: researchers report the
    compounds that worked, so a target's dataset is often 50-70% actives. Enrichment
    factor is bounded above by 1/active_fraction, so on a 64%-active test set EF@1%
    cannot exceed ~1.6 no matter how good the model is. Quoting that as a screening
    result is meaningless.

    Downsampling actives to reach a realistic prevalence does not fix it either: at
    1% of a 688-compound test set you are left with two actives, and EF becomes a
    coin flip with a standard deviation larger than its mean.

    So we do the opposite. We keep every held-out active and add the *real* natural
    product library as presumed-inactive decoys. Prevalence lands near 1% because the
    library is large, and the active count stays in the hundreds, so the estimate is
    stable. This is also precisely the deployment task: rank a natural-product library
    and see whether genuine actives surface near the top.

    Two honest caveats. The decoys are *presumed* inactive - unlabelled, not measured
    inactive - so a decoy that is really active counts against the model and makes this
    a conservative estimate. And the actives come from ChEMBL, which is not a random
    sample of chemical space.
    """
    active_scores = np.asarray(active_scores, dtype=float)
    decoy_scores = np.asarray(decoy_scores, dtype=float)
    if len(active_scores) == 0 or len(decoy_scores) == 0:
        return {}

    y = np.concatenate([np.ones(len(active_scores), int), np.zeros(len(decoy_scores), int)])
    s = np.concatenate([active_scores, decoy_scores])
    prevalence = float(y.mean())

    return {
        "n_held_out_actives": int(len(active_scores)),
        "n_library_decoys": int(len(decoy_scores)),
        "active_prevalence": round(prevalence, 5),
        "ef_1pct": round(float(enrichment_factor(y, s, 0.01)), 3),
        "ef_5pct": round(float(enrichment_factor(y, s, 0.05)), 3),
        "ef_1pct_ceiling": round(float(1.0 / prevalence), 1),
        "bedroc_20": round(float(bedroc(y, s, 20.0)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "decoys_are_presumed_inactive": True,
    }


def calibration_quality(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score plus expected calibration error.

    A model that says "0.9" should be right about 90% of the time. Screening
    triage decisions are made on the probability value, not just the rank, so an
    uncalibrated score is actively misleading to whoever reads the table.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    brier = float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    bins = np.linspace(0, 1, n_bins + 1)
    ece, total = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return {"brier": brier, "ece": float(ece)}
