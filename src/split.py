"""Dataset splitting and split-bias measurement.

The single most common way a virtual-screening model lies to you is the random
split. Congeneric series (a dozen analogues of one scaffold from one medicinal-
chemistry paper) get spread across train and test, so the model is graded on
compounds it has effectively already seen. Reported AUC goes up; real prospective
performance does not move at all.

This module provides both splits so the inflation can be *reported as a number*,
plus the AVE bias metric of Wallach & Heifets (2018), which quantifies how much
of the remaining separability is an artefact of how actives and inactives sit in
chemical space rather than genuine signal.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def random_split(n: int, test_fraction: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def scaffold_split(scaffolds, test_fraction: float, seed: int = 42):
    """Bemis-Murcko scaffold split: no scaffold appears in both train and test.

    Scaffold groups are filled largest-first into train, which keeps the common
    chemotypes in training and pushes rarer, more novel chemistry into test. That
    is deliberately the harder direction, and the one that resembles prospective use.

    Acyclic molecules (empty scaffold) are each treated as their own group rather
    than lumped together, since they share no ring system by definition.
    """
    groups: dict[str, list[int]] = {}
    for i, scaf in enumerate(scaffolds):
        key = scaf if scaf else f"__acyclic_{i}"
        groups.setdefault(key, []).append(i)

    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))

    n = len(scaffolds)
    n_test_target = int(round(n * test_fraction))
    train_idx: list[int] = []
    test_idx: list[int] = []

    # The largest 5% of scaffold groups anchor the training set: these are the
    # dense congeneric series, and putting any of them in test would make the
    # test set trivially easy for the wrong reason.
    n_big = max(1, len(ordered) // 20)
    big, tail = ordered[:n_big], ordered[n_big:]
    for group in big:
        train_idx.extend(group)

    # The long tail is shuffled so the test set is not just "every singleton in
    # dataset order", then filled until the target size is reached.
    rng.shuffle(tail)
    for group in tail:
        if len(test_idx) + len(group) <= n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    # Guard against a degenerate split when one scaffold dominates.
    if not test_idx or not train_idx:
        return random_split(n, test_fraction, seed)
    log.info("scaffold split: %d train / %d test over %d scaffold groups",
             len(train_idx), len(test_idx), len(groups))
    return np.sort(np.asarray(train_idx)), np.sort(np.asarray(test_idx))


def _nn_similarity(query: np.ndarray, ref: np.ndarray, block: int = 512) -> np.ndarray:
    """Nearest-neighbour Tanimoto of each query row to the reference set."""
    if len(query) == 0 or len(ref) == 0:
        return np.zeros(len(query), dtype=np.float32)
    q = (query > 0).astype(np.float32)
    r = (ref > 0).astype(np.float32)
    r_sum = r.sum(axis=1)
    out = np.zeros(len(q), dtype=np.float32)
    for s in range(0, len(q), block):
        chunk = q[s:s + block]
        inter = chunk @ r.T
        union = chunk.sum(axis=1)[:, None] + r_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[s:s + block] = sim.max(axis=1)
    return out


def _cdf_auc(sims: np.ndarray, n_points: int = 100) -> float:
    """Area under the empirical CDF of nearest-neighbour similarities."""
    if len(sims) == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, n_points)
    frac = np.array([(sims <= t).mean() for t in thresholds])
    return float(np.trapezoid(frac, thresholds))


def ave_bias(fps: np.ndarray, labels: np.ndarray,
             train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """Asymmetric Validation Embedding bias (Wallach & Heifets, J Chem Inf Model 2018).

    Measures whether test actives sit closer to training actives than to training
    inactives (and symmetrically for inactives). A value near 0 means the split
    carries no spatial giveaway; large positive values mean a nearest-neighbour
    lookup would score well without learning any pharmacology.

    Returned in roughly [-2, 2]. We report it rather than optimise it away, and
    interpret the model's AUC in light of it.
    """
    fps = (fps > 0).astype(np.float32)
    tr_a = fps[train_idx][labels[train_idx] == 1]
    tr_i = fps[train_idx][labels[train_idx] == 0]
    te_a = fps[test_idx][labels[test_idx] == 1]
    te_i = fps[test_idx][labels[test_idx] == 0]

    aa = _cdf_auc(_nn_similarity(te_a, tr_a))
    ai = _cdf_auc(_nn_similarity(te_a, tr_i))
    ii = _cdf_auc(_nn_similarity(te_i, tr_i))
    ia = _cdf_auc(_nn_similarity(te_i, tr_a))

    bias = (ai - aa) + (ia - ii)
    return {"ave_bias": float(bias), "AA": aa, "AI": ai, "II": ii, "IA": ia}
