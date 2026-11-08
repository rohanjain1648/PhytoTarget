"""Pull labelled bioactivity data for one protein target from ChEMBL.

Source: ChEMBL 34 REST API (https://www.ebi.ac.uk/chembl/api/data/).
Licence: CC BY-SA 3.0. Attribution is recorded in README.md.

Design notes that matter for the science:

* One molecule can have many measurements against the same target, from different
  labs and assay formats. We take the **median** pChEMBL per molecule, not the max.
  Taking the max is the common shortcut and it systematically inflates the active
  class with single optimistic outliers.
* Activities whose median falls in the ambiguous band are **dropped**, not pushed
  into the inactive class. Calling an untested-looking 1 uM compound "inactive"
  teaches the model a fact that is not in evidence.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from .chem import murcko_scaffold, standardize
from .util import load_config, setup_logging

log = logging.getLogger(__name__)

BASE = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS = {"User-Agent": "PhytoTarget/0.1 (EthnoHack 2026 research prototype)"}
PAGE = 1000


def _get(path: str, params: dict, retries: int = 4) -> dict:
    url = f"{BASE}/{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=90)
            if resp.status_code == 200:
                return resp.json()
            log.warning("ChEMBL %s -> HTTP %s (attempt %d)", path, resp.status_code, attempt + 1)
        except requests.RequestException as exc:
            log.warning("ChEMBL %s -> %s (attempt %d)", path, exc, attempt + 1)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"ChEMBL request failed after {retries} attempts: {url} {params}")


def fetch_target_name(target_id: str) -> str:
    try:
        data = _get(f"target/{target_id}.json", {})
        return data.get("pref_name") or target_id
    except Exception:
        return target_id


def fetch_activities(target_id: str, cfg: dict) -> pd.DataFrame:
    """Page through every pChEMBL-bearing activity for the target."""
    lab = cfg["labels"]
    params = {
        "target_chembl_id": target_id,
        "pchembl_value__isnull": "false",
        "limit": PAGE,
        "offset": 0,
    }
    rows, total = [], None
    while True:
        data = _get("activity.json", params)
        meta = data["page_meta"]
        if total is None:
            total = meta["total_count"]
            log.info("ChEMBL reports %d pChEMBL activities for %s", total, target_id)
        for act in data["activities"]:
            rows.append({
                "molecule_chembl_id": act.get("molecule_chembl_id"),
                "smiles": act.get("canonical_smiles"),
                "pchembl": act.get("pchembl_value"),
                "standard_type": act.get("standard_type"),
                "assay_type": act.get("assay_type"),
                "target_organism": act.get("target_organism"),
            })
        params["offset"] += PAGE
        log.info("  fetched %d / %s", len(rows), total)
        if params["offset"] >= (total or 0):
            break
    df = pd.DataFrame(rows)
    log.info("raw activities: %d", len(df))

    df = df.dropna(subset=["molecule_chembl_id", "smiles", "pchembl"])
    df["pchembl"] = pd.to_numeric(df["pchembl"], errors="coerce")
    df = df.dropna(subset=["pchembl"])

    df = df[df["standard_type"].isin(lab["allowed_types"])]
    df = df[df["assay_type"].isin(lab["allowed_assay_types"])]
    if lab.get("require_human"):
        human = df["target_organism"].eq("Homo sapiens")
        if human.sum() >= 200:
            df = df[human]
        else:
            log.warning("only %d human activities; keeping all organisms", int(human.sum()))
    log.info("after assay/type filters: %d", len(df))
    return df


def build_dataset(target_id: str, cfg: dict) -> pd.DataFrame:
    raw = fetch_activities(target_id, cfg)
    if raw.empty:
        raise RuntimeError(f"no usable activities for {target_id}")

    agg = (raw.groupby("molecule_chembl_id")
              .agg(smiles=("smiles", "first"),
                   pchembl=("pchembl", "median"),
                   assay_type=("assay_type", "first"),
                   n_measurements=("pchembl", "size"))
              .reset_index())
    log.info("unique molecules: %d", len(agg))

    lab = cfg["labels"]
    agg["label"] = pd.NA
    agg.loc[agg["pchembl"] >= lab["active_threshold"], "label"] = 1
    agg.loc[agg["pchembl"] <= lab["inactive_threshold"], "label"] = 0
    dropped = agg["label"].isna().sum()
    agg = agg.dropna(subset=["label"])
    agg["label"] = agg["label"].astype(int)
    log.info("dropped %d molecules in the ambiguous band (%.1f-%.1f)",
             dropped, lab["inactive_threshold"], lab["active_threshold"])

    agg["smiles"] = agg["smiles"].map(standardize)
    agg = agg.dropna(subset=["smiles"])
    agg = agg.drop_duplicates(subset=["smiles"])
    agg["scaffold"] = agg["smiles"].map(murcko_scaffold)

    log.info("final: %d actives, %d inactives",
             int((agg.label == 1).sum()), int((agg.label == 0).sum()))
    return agg[["molecule_chembl_id", "smiles", "pchembl", "label",
                "assay_type", "scaffold"]].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch ChEMBL bioactivity for a target")
    ap.add_argument("--target", help="ChEMBL target id; default = all targets in config")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    targets = [args.target] if args.target else [t["chembl_id"] for t in cfg["targets"]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tid in targets:
        out = out_dir / f"bioactivity_{tid}.csv"
        if out.exists():
            log.info("%s already present, skipping (delete to refetch)", out)
            continue
        log.info("=== %s (%s) ===", tid, fetch_target_name(tid))
        df = build_dataset(tid, cfg)
        df.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
