"""Score the plant natural-product library against a trained target model.

Three things happen here that a naive `model.predict_proba()` screen does not do,
and each of them exists to stop the output table from overpromising:

* **Applicability domain.** A probability for a molecule unlike anything in
  training is an extrapolation. We compute max Tanimoto to the training set and
  flag everything below threshold, rather than silently ranking it.
* **Known-active flagging.** Plant compounds already in the target's ChEMBL
  training actives will of course score highly. They are kept (they are the
  retrospective validation) but flagged, so nobody mistakes a rediscovery for
  a novel hit.
* **PAINS alerts.** Frequent-hitter substructures get counted, because the
  classic natural-product screening failure is a promiscuous polyphenol that
  "hits" every assay it touches.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .chem import featurize, inchikey, inchikey_skeleton, pains_alerts, tanimoto_max
from .fetch_lotus import LABELS_QUERY, _qid, _sparql
from .libfeatures import library_features, library_pains
from .util import load_config, setup_logging, target_config

log = logging.getLogger(__name__)


def annotate_names(df: pd.DataFrame, top_n: int = 3000, extra_qids=()) -> pd.DataFrame:
    """Resolve Wikidata labels for the top-ranked compounds.

    The full library is 43k compounds and labelling all of them costs ~100 extra
    SPARQL round trips for names nobody will read. We label the top slice (what
    appears in the UI) plus any specific compounds the retrospective check needs.
    """
    qids = list(dict.fromkeys(list(df.head(top_n)["compound_qid"]) + list(extra_qids)))
    names: dict[str, str] = {}
    chunk = 400
    for i in range(0, len(qids), chunk):
        values = " ".join(f"wd:{q}" for q in qids[i:i + chunk])
        try:
            for b in _sparql(LABELS_QUERY.format(values=values)):
                names[_qid(b["item"]["value"])] = b["label"]["value"]
        except RuntimeError:
            log.warning("label chunk %d failed; continuing", i)
    # Only overwrite where the lookup actually returned something: compounds outside
    # the labelled slice keep whatever name the library already carried.
    resolved = df["compound_qid"].map(names)
    df["compound_name"] = resolved.fillna(df.get("compound_name")).fillna("")
    log.info("resolved %d compound names", int((df["compound_name"] != "").sum()))
    return df


def resolve_by_name(names: list[str]) -> dict[str, str]:
    """Wikidata QIDs for named compounds, so retrospective checks work by rank.

    Matching on the name inside the screening table would only find molecules in
    the labelled top slice; resolving the QID first means we can report an
    honest rank even if the compound landed in position 9000.
    """
    if not names:
        return {}
    # Resolving a label through SPARQL means scanning every rdfs:label in Wikidata,
    # which WDQS answers with a 504. The entity-search endpoint does it in one call.
    from .augment_library import resolve_qid

    out: dict[str, str] = {}
    for n in names:
        hit = resolve_qid(n)
        if hit:
            out[n.lower()] = hit[0]
        else:
            log.warning("could not resolve %s to a Wikidata structure", n)
    return out


def collapse_library(lib: pd.DataFrame) -> pd.DataFrame:
    """One row per unique compound, with its source taxa gathered up.

    The library is a (compound, taxon) edge list: a compound found in twelve
    plants appears twelve times. Scoring it twelve times would be wasted work and
    would let widely-distributed compounds dominate any top-k slice.
    """
    lib = lib.copy()
    lib["inchikey"] = lib["inchikey"].fillna("")
    key = np.where(lib["inchikey"].str.len() > 0, lib["inchikey"], lib["smiles"])
    lib["_key"] = key

    grouped = (lib.groupby("_key")
                  .agg(compound_qid=("compound_qid", "first"),
                       compound_name=("compound_name", "first"),
                       smiles=("smiles", "first"),
                       inchikey=("inchikey", "first"),
                       n_source_taxa=("taxon_name", "nunique"),
                       top_taxa=("taxon_name", lambda s: ";".join(sorted(set(s))[:5])))
                  .reset_index(drop=True))
    log.info("collapsed %d pairs -> %d unique compounds", len(lib), len(grouped))
    return grouped


def known_active_keys(bioactivity_path: Path) -> set[str]:
    """Skeleton InChIKeys of the target's training actives, for the rediscovery flag.

    Skeletons, not full keys: Wikidata SMILES are frequently drawn without
    stereochemistry while ChEMBL's are stereo-specified, so full-key matching
    recovers almost nothing (5 of 2,216 actives on AChE). See
    `chem.inchikey_skeleton` for why the looser match is the right one here.
    """
    if not bioactivity_path.exists():
        return set()
    df = pd.read_csv(bioactivity_path)
    actives = df[df["label"] == 1]["smiles"].dropna()
    keys = {inchikey_skeleton(s) for s in actives}
    keys.discard("")
    log.info("%d known actives available for dedup flagging", len(keys))
    return keys


def run(target_id: str, cfg: dict, data_dir: Path, out_dir: Path, model_dir: Path) -> pd.DataFrame:
    bundle = joblib.load(model_dir / f"model_{target_id}.joblib")
    model = bundle["model"]
    ref = np.load(model_dir / f"trainfps_{target_id}.npz")
    train_fps = ref["fps"]

    lib = pd.read_csv(data_dir / "np_library.csv")
    compounds = collapse_library(lib)

    fp_cfg = cfg["fingerprint"]
    # Cached: screening a second target reuses the first target's feature matrix.
    X, keep = library_features(compounds, data_dir, fp_cfg)
    compounds = compounds[keep].reset_index(drop=True)
    log.info("featurised %d library compounds", len(compounds))

    log.info("scoring...")
    compounds["prob_active"] = model.predict_proba(X)[:, 1]

    log.info("computing applicability domain against %d training molecules", len(train_fps))
    n_bits = fp_cfg["n_bits"]
    compounds["max_train_similarity"] = tanimoto_max(X[:, :n_bits], train_fps)
    ad_thresh = cfg["applicability_domain"]["similarity_threshold"]
    compounds["in_domain"] = compounds["max_train_similarity"] >= ad_thresh
    log.info("%.1f%% of library inside applicability domain (Tanimoto >= %.2f)",
             100 * compounds["in_domain"].mean(), ad_thresh)

    compounds["pains_alerts"] = library_pains(compounds, data_dir)
    log.info("%d compounds carry at least one PAINS alert", int((compounds.pains_alerts > 0).sum()))

    known = known_active_keys(data_dir / f"bioactivity_{target_id}.csv")
    compounds["is_known_active"] = compounds["smiles"].map(inchikey_skeleton).isin(known)
    log.info("%d library compounds are already known actives for this target",
             int(compounds.is_known_active.sum()))

    compounds = compounds.sort_values("prob_active", ascending=False).reset_index(drop=True)
    compounds.insert(0, "rank", np.arange(1, len(compounds) + 1))

    tcfg = target_config(cfg, target_id)
    positives = tcfg.get("retrospective_positives") or []
    name_to_qid = resolve_by_name([p["name"] for p in positives])
    log.info("resolving compound names for the top slice...")
    compounds = annotate_names(compounds, top_n=3000, extra_qids=name_to_qid.values())

    cols =["rank", "compound_qid", "compound_name", "smiles", "inchikey",
            "prob_active", "max_train_similarity", "in_domain", "pains_alerts",
            "is_known_active", "n_source_taxa", "top_taxa"]
    out = compounds[cols]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"screen_{target_id}.csv"
    out.to_csv(path, index=False)
    log.info("wrote %s (%d compounds)", path, len(out))

    _report_retrospective(out, tcfg, name_to_qid, out_dir, target_id)
    return out


def _report_retrospective(screen: pd.DataFrame, tcfg: dict, name_to_qid: dict,
                          out_dir: Path, target_id: str) -> None:
    """Where did the known plant-derived drugs for this target land?

    This is the headline validation: the model was never told these molecules
    are special, so their rank is an independent check that the ranking means
    something to a pharmacologist.
    """
    positives = tcfg.get("retrospective_positives") or []
    if not positives:
        return
    n = len(screen)
    log.info("--- retrospective recovery of known plant-derived leads ---")
    records = []
    for entry in positives:
        name = entry["name"]
        qid = name_to_qid.get(name.lower())
        hit = screen[screen["compound_qid"] == qid] if qid else screen.iloc[0:0]
        if hit.empty:
            hit = screen[screen["compound_name"].fillna("").str.lower() == name.lower()]
        if hit.empty:
            log.info("  %-14s not present in the LOTUS library", name)
            records.append({"name": name, "found": False,
                            "source_plant": entry.get("source_plant", "")})
            continue
        row = hit.iloc[0]
        pct = 100.0 * int(row["rank"]) / n
        log.info("  %-14s rank %5d / %d (top %.2f%%)  P=%.3f  [%s]",
                 name, int(row["rank"]), n, pct, row["prob_active"],
                 entry.get("source_plant", ""))
        records.append({"name": name, "found": True, "rank": int(row["rank"]),
                        "n_library": n, "percentile": round(pct, 3),
                        "prob_active": round(float(row["prob_active"]), 4),
                        "in_domain": bool(row["in_domain"]),
                        "is_known_active": bool(row["is_known_active"]),
                        "source_plant": entry.get("source_plant", "")})
    # Fold into the metrics file so the validation tab and the README can cite it.
    mpath = out_dir / f"metrics_{target_id}.json"
    if mpath.exists():
        import json
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
        metrics["retrospective_recovery"] = records
        mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Screen NP library against a target model")
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
        if not (Path(args.model_dir) / f"model_{tid}.joblib").exists():
            log.error("no model for %s - run train first", tid)
            continue
        log.info("================ screening %s ================", tid)
        run(tid, cfg, Path(args.data_dir), Path(args.out_dir), Path(args.model_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
