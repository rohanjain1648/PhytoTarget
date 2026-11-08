"""Cached featurisation of the natural-product library.

Both training (for the decoy-augmented evaluation) and screening need the same
23k x 2060 feature matrix. Computing Morgan fingerprints twice costs a minute of
wall clock for no reason, so the first caller writes an .npz keyed on the
fingerprint settings and later callers load it.

The cache key includes the fingerprint configuration, so changing radius or bit
count in config.yaml produces a different cache file rather than silently
reusing features built under the old settings.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import featurize

log = logging.getLogger(__name__)


def _cache_path(data_dir: Path, fp_cfg: dict) -> Path:
    tag = f"r{fp_cfg['radius']}_b{fp_cfg['n_bits']}_d{int(bool(fp_cfg['use_descriptors']))}"
    return data_dir / f"libfeatures_{tag}.npz"


def library_features(compounds: pd.DataFrame, data_dir: Path, fp_cfg: dict):
    """(X, keep_mask) for the collapsed library, cached on disk.

    `keep_mask` marks the rows RDKit could parse, so callers can filter their own
    dataframe to match the feature matrix.
    """
    path = _cache_path(data_dir, fp_cfg)
    smiles = compounds["smiles"].tolist()

    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if int(cached["n_input"]) == len(smiles):
            log.info("loaded cached library features from %s", path.name)
            return cached["X"], cached["keep"].astype(bool)
        log.info("cache %s was built for %d compounds, not %d - rebuilding",
                 path.name, int(cached["n_input"]), len(smiles))

    log.info("featurising %d library compounds (first run, will be cached)", len(smiles))
    X, keep = featurize(smiles, fp_cfg["radius"], fp_cfg["n_bits"], fp_cfg["use_descriptors"])
    np.savez_compressed(path, X=X.astype(np.float32), keep=np.asarray(keep),
                        n_input=np.asarray(len(smiles)))
    log.info("cached library features to %s", path.name)
    return X, np.asarray(keep, dtype=bool)


def library_pains(compounds: pd.DataFrame, data_dir: Path) -> np.ndarray:
    """PAINS alert counts for the collapsed library, cached on disk.

    Matching ~480 PAINS SMARTS against 24k molecules is the slowest step in the
    whole screen, and the answer does not depend on the target, so recomputing it
    for every target is pure waste.
    """
    from .chem import pains_alerts

    path = data_dir / "libpains.npz"
    smiles = compounds["smiles"].tolist()
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if int(cached["n"]) == len(smiles):
            log.info("loaded cached PAINS counts from %s", path.name)
            return cached["alerts"]

    log.info("screening %d compounds for PAINS alerts (first run, will be cached)",
             len(smiles))
    alerts = np.array([pains_alerts(s) for s in smiles], dtype=np.int16)
    np.savez_compressed(path, alerts=alerts, n=np.asarray(len(smiles)))
    log.info("cached PAINS counts to %s", path.name)
    return alerts
