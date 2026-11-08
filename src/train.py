"""Train and honestly evaluate a ligand-based activity model for one target.

What this script produces is not just a model but a *claim about how much the
model can be trusted*, written to metrics_<TARGET>.json:

  1. Scaffold-split performance   - the headline, prospective-ish number.
  2. Random-split performance     - the inflated number, reported alongside so the
                                    gap is visible instead of hidden.
  3. AVE bias                     - how much of the split is spatial artefact.
  4. y-scrambling control         - labels permuted; AUC must collapse to ~0.5.
                                    If it does not, there is leakage somewhere and
                                    every other number on this page is void.
  5. Calibration (Brier, ECE)     - are the probabilities meaningful as probabilities.

The model shipped for screening is trained on all available data and calibrated,
because at screening time there is no reason to hold data back.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from .chem import featurize
from .metrics import calibration_quality, decoy_augmented_eval, evaluate
from .split import ave_bias, random_split, scaffold_split
from .util import load_config, setup_logging, target_config, write_json

log = logging.getLogger(__name__)


def make_model(cfg: dict, seed: int = 42) -> LGBMClassifier:
    m = cfg["model"]
    return LGBMClassifier(
        n_estimators=m["n_estimators"],
        learning_rate=m["learning_rate"],
        num_leaves=m["num_leaves"],
        min_child_samples=m["min_child_samples"],
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.6,
        random_state=seed,
        # Bounded rather than -1: CalibratedClassifierCV fits k models, and letting
        # each spawn one thread per core oversubscribes the machine badly enough
        # to get the process killed on a laptop.
        n_jobs=2,
        verbose=-1,
    )


def _fit_score(X, y, train_idx, test_idx, cfg, seed=42):
    model = make_model(cfg, seed)
    model.fit(X[train_idx], y[train_idx])
    scores = model.predict_proba(X[test_idx])[:, 1]
    val = cfg["validation"]
    return model, scores, evaluate(y[test_idx], scores,
                                   tuple(val["ef_fractions"]), val["bedroc_alpha"])


def y_scramble(X, y, scaffolds, cfg) -> dict:
    """Negative control: permute the labels and retrain.

    Any real signal must vanish. A scrambled AUC meaningfully above 0.5 means the
    pipeline is learning something about the *data layout* rather than the
    chemistry - duplicate molecules across the split, a leaked index, an
    identifier encoded in a feature.
    """
    n_runs = cfg["validation"]["y_scramble_runs"]
    frac = cfg["split"]["test_fraction"]
    seed0 = cfg["split"]["random_seed"]
    aucs = []
    for run in range(n_runs):
        rng = np.random.default_rng(seed0 + 1000 + run)
        y_perm = rng.permutation(y)
        tr, te = scaffold_split(scaffolds, frac, seed=seed0 + run)
        if len(np.unique(y_perm[te])) < 2:
            continue
        _, _, res = _fit_score(X, y_perm, tr, te, cfg, seed=seed0 + run)
        aucs.append(res["roc_auc"])
        log.info("  y-scramble run %d: ROC-AUC = %.3f", run + 1, res["roc_auc"])
    if not aucs:
        return {"mean_roc_auc": float("nan"), "std": float("nan"), "n_runs": 0}
    return {"mean_roc_auc": float(np.mean(aucs)), "std": float(np.std(aucs)),
            "n_runs": len(aucs), "runs": [float(a) for a in aucs]}


def run(target_id: str, cfg: dict, data_dir: Path, out_dir: Path, model_dir: Path) -> dict:
    tcfg = target_config(cfg, target_id)
    df = pd.read_csv(data_dir / f"bioactivity_{target_id}.csv")
    log.info("%s: %d molecules (%d active)", target_id, len(df), int(df.label.sum()))

    fp_cfg = cfg["fingerprint"]
    X, keep = featurize(df["smiles"].tolist(), fp_cfg["radius"], fp_cfg["n_bits"],
                        fp_cfg["use_descriptors"])
    df = df[keep].reset_index(drop=True)
    y = df["label"].to_numpy(dtype=int)
    scaffolds = df["scaffold"].fillna("").tolist()
    log.info("featurised: X=%s, %d active / %d inactive", X.shape, int(y.sum()), int((1 - y).sum()))

    frac = cfg["split"]["test_fraction"]
    seed = cfg["split"]["random_seed"]
    n_bits = fp_cfg["n_bits"]
    fps_only = X[:, :n_bits]

    results = {}

    log.info("--- random split (bias control) ---")
    rtr, rte = random_split(len(df), frac, seed)
    _, _, results["random"] = _fit_score(X, y, rtr, rte, cfg, seed)
    log.info("random  ROC-AUC=%.3f PR-AUC=%.3f EF@1%%=%.1f",
             results["random"]["roc_auc"], results["random"]["pr_auc"],
             results["random"]["ef_1pct"])

    log.info("--- scaffold split (headline) ---")
    str_, ste = scaffold_split(scaffolds, frac, seed)
    smodel, sscores, results["scaffold"] = _fit_score(X, y, str_, ste, cfg, seed)
    log.info("scaffold ROC-AUC=%.3f PR-AUC=%.3f EF@1%%=%.1f BEDROC=%.3f",
             results["scaffold"]["roc_auc"], results["scaffold"]["pr_auc"],
             results["scaffold"]["ef_1pct"], results["scaffold"]["bedroc_20"])

    log.info("--- enrichment with held-out actives hidden in the real NP library ---")
    realistic = _decoy_augmented(X, y, ste, smodel, cfg, data_dir, target_id)
    if realistic:
        log.info("EF@1%% = %.1f (ceiling %.0f) with %d held-out actives among %d "
                 "library decoys, prevalence %.3f%%",
                 realistic["ef_1pct"], realistic["ef_1pct_ceiling"],
                 realistic["n_held_out_actives"], realistic["n_library_decoys"],
                 100 * realistic["active_prevalence"])

    log.info("--- AVE split-bias ---")
    ave = ave_bias(fps_only, y, str_, ste)
    log.info("AVE bias = %+.3f (0 = unbiased)", ave["ave_bias"])

    log.info("--- y-scrambling null control (%d runs) ---", cfg["validation"]["y_scramble_runs"])
    scramble = y_scramble(X, y, scaffolds, cfg)
    log.info("scrambled mean ROC-AUC = %.3f +/- %.3f (expect ~0.5)",
             scramble["mean_roc_auc"], scramble["std"])

    log.info("--- calibration on scaffold-held-out set ---")
    cal_eval = CalibratedClassifierCV(make_model(cfg, seed), method=cfg["model"]["calibration"], cv=3)
    cal_eval.fit(X[str_], y[str_])
    cal_probs = cal_eval.predict_proba(X[ste])[:, 1]
    calib = calibration_quality(y[ste], cal_probs)
    calib["method"] = cfg["model"]["calibration"]
    log.info("Brier=%.4f ECE=%.4f", calib["brier"], calib["ece"])

    log.info("--- fitting final calibrated model on all data ---")
    final = CalibratedClassifierCV(make_model(cfg, seed), method=cfg["model"]["calibration"], cv=3)
    final.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "config": cfg, "target": tcfg,
                 "n_train": len(df), "n_actives": int(y.sum())},
                model_dir / f"model_{target_id}.joblib")
    # Training fingerprints are the applicability domain reference set.
    np.savez_compressed(model_dir / f"trainfps_{target_id}.npz",
                        fps=fps_only.astype(np.uint8), labels=y)
    log.info("saved model + AD reference set to %s", model_dir)

    metrics = {
        "target_chembl_id": target_id,
        "target_name": tcfg.get("name", target_id),
        "disease": tcfg.get("disease", ""),
        "n_molecules": int(len(df)),
        "n_actives": int(y.sum()),
        "n_inactives": int((1 - y).sum()),
        "n_scaffolds": int(len(set(s for s in scaffolds if s))),
        "label_thresholds": {"active_pchembl": cfg["labels"]["active_threshold"],
                             "inactive_pchembl": cfg["labels"]["inactive_threshold"]},
        "splits": results,
        "decoy_augmented": realistic,
        "auc_inflation_random_minus_scaffold": float(
            results["random"]["roc_auc"] - results["scaffold"]["roc_auc"]),
        "ave_bias": ave,
        "y_scramble": scramble,
        "calibration": calib,
        "applicability_domain": {
            "threshold": cfg["applicability_domain"]["similarity_threshold"]},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics, out_dir / f"metrics_{target_id}.json")
    log.info("wrote %s", out_dir / f"metrics_{target_id}.json")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train + validate activity model")
    ap.add_argument("--target")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    targets = [args.target] if args.target else [t["chembl_id"] for t in cfg["targets"]]
    for tid in targets:
        path = Path(args.data_dir) / f"bioactivity_{tid}.csv"
        if not path.exists():
            log.error("missing %s - run fetch_chembl first", path)
            continue
        log.info("================ %s ================", tid)
        run(tid, cfg, Path(args.data_dir), Path(args.out_dir), Path(args.model_dir))
    return 0



def _decoy_augmented(X, y, test_idx, model, cfg: dict, data_dir: Path,
                     target_id: str) -> dict:
    """Score the real NP library with the scaffold-split model and use it as decoys.

    The scaffold model never saw the test actives, and it never saw the library at
    all, so this is an honest estimate of the only thing that matters in deployment:
    if you hand the model 23,000 plant compounds with a few hundred real actives
    mixed in, how many of those actives land in the slice you could afford to assay?

    Library compounds whose InChIKey matches a training molecule are removed first -
    leaving them in would plant known actives among the decoys and penalise the model
    for ranking them correctly.
    """
    from .screen import collapse_library
    from .libfeatures import library_features
    from .chem import inchikey

    lib_path = data_dir / "np_library.csv"
    if not lib_path.exists():
        log.warning("no NP library yet - skipping decoy-augmented evaluation")
        return {}

    compounds = collapse_library(pd.read_csv(lib_path))
    fp_cfg = cfg["fingerprint"]
    Xlib, keep = library_features(compounds, data_dir, fp_cfg)
    compounds = compounds[keep].reset_index(drop=True)

    # Drop library compounds that are in the training data at all, actives or not:
    # they are not decoys, they are labelled molecules.
    train_keys = {inchikey(s) for s in pd.read_csv(
        data_dir / f"bioactivity_{target_id}.csv")["smiles"].dropna()}
    train_keys.discard("")
    lib_keys = compounds["inchikey"].fillna("").tolist()
    is_decoy = np.array([k == "" or k not in train_keys for k in lib_keys], dtype=bool)
    Xlib, n_dropped = Xlib[is_decoy], int((~is_decoy).sum())
    log.info("library decoys: %d (dropped %d already present in training data)",
             int(is_decoy.sum()), n_dropped)

    decoy_scores = model.predict_proba(Xlib)[:, 1]
    active_scores = model.predict_proba(X[test_idx][y[test_idx] == 1])[:, 1]
    return decoy_augmented_eval(active_scores, decoy_scores)



if __name__ == "__main__":
    sys.exit(main())
