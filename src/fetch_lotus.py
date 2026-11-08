"""Build the plant natural-product library from LOTUS, via the Wikidata Query Service.

Source: LOTUS (https://lotus.naturalproducts.net/), mirrored into Wikidata as
compound--`found in taxon` (P703) statements. Licence: CC0.

Why Wikidata rather than the LOTUS dump: the compound->organism link is the whole
point of this project, and on Wikidata it is a first-class, individually referenced
statement we can cite back to a source publication. It is also CC0, which removes
any redistribution question for the small cached snapshot we ship.

The transitive "is a plant" filter (`wdt:P171* wd:Q25314`) times the endpoint out,
so we pull the pairs flat and resolve taxon names in a second, cheaper pass.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from .chem import inchikey, standardize
from .util import load_config, setup_logging

log = logging.getLogger(__name__)

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "PhytoTarget/0.1 (EthnoHack 2026 research prototype)",
    "Accept": "application/sparql-results+json",
}

PAIRS_QUERY = """
SELECT ?c ?smiles ?t WHERE {{
  ?c wdt:P703 ?t .
  ?c wdt:P233 ?smiles .
}}
LIMIT {limit} OFFSET {offset}
"""

# Resolved in chunks. rdfs:label is far cheaper than the label SERVICE here.
LABELS_QUERY = """
SELECT ?item ?label WHERE {{
  VALUES ?item {{ {values} }}
  ?item rdfs:label ?label .
  FILTER(LANG(?label) = "en")
}}
"""


def _sparql(query: str, retries: int = 4) -> list[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(ENDPOINT, params={"query": query}, headers=HEADERS, timeout=180)
            if resp.status_code == 200:
                return resp.json()["results"]["bindings"]
            log.warning("WDQS HTTP %s (attempt %d)", resp.status_code, attempt + 1)
        except (requests.RequestException, ValueError) as exc:
            log.warning("WDQS %s (attempt %d)", exc, attempt + 1)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("Wikidata query failed")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fetch_pairs(max_pairs: int, page_size: int) -> pd.DataFrame:
    rows, offset = [], 0
    while len(rows) < max_pairs:
        limit = min(page_size, max_pairs - len(rows))
        log.info("WDQS pairs: offset=%d limit=%d (have %d)", offset, limit, len(rows))
        batch = _sparql(PAIRS_QUERY.format(limit=limit, offset=offset))
        if not batch:
            log.info("endpoint returned no more rows")
            break
        for b in batch:
            rows.append({
                "compound_qid": _qid(b["c"]["value"]),
                "smiles": b["smiles"]["value"],
                "taxon_qid": _qid(b["t"]["value"]),
            })
        offset += limit
        if len(batch) < limit:
            break
        time.sleep(0.5)  # be polite to a free public endpoint
    return pd.DataFrame(rows)


def fetch_labels(qids: list[str], chunk: int = 400) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(qids), chunk):
        part = qids[i:i + chunk]
        values = " ".join(f"wd:{q}" for q in part)
        log.info("WDQS labels: %d / %d", min(i + chunk, len(qids)), len(qids))
        try:
            for b in _sparql(LABELS_QUERY.format(values=values)):
                out[_qid(b["item"]["value"])] = b["label"]["value"]
        except RuntimeError:
            log.warning("label chunk failed; continuing without those names")
        time.sleep(0.3)
    return out


def build_library(cfg: dict) -> pd.DataFrame:
    lot = cfg["lotus"]
    df = fetch_pairs(lot["max_pairs"], lot["page_size"])
    log.info("raw compound-taxon pairs: %d", len(df))
    if df.empty:
        raise RuntimeError("no pairs returned from Wikidata")

    # Standardise once per unique SMILES rather than once per pair.
    uniq = pd.DataFrame({"smiles": df["smiles"].unique()})
    log.info("standardising %d unique SMILES", len(uniq))
    uniq["std"] = uniq["smiles"].map(standardize)
    uniq = uniq.dropna(subset=["std"])
    uniq["inchikey"] = uniq["std"].map(inchikey)
    df = df.merge(uniq, on="smiles", how="inner")
    df["smiles"] = df["std"]
    df = df.drop(columns=["std"])
    log.info("after standardisation: %d pairs", len(df))

    taxa = sorted(df["taxon_qid"].unique())
    log.info("resolving %d taxon names", len(taxa))
    tax_names = fetch_labels(taxa)
    df["taxon_name"] = df["taxon_qid"].map(tax_names).fillna("")

    # A taxon we cannot name is a taxon we cannot report to a botanist.
    df = df[df["taxon_name"].str.strip() != ""]
    df["genus"] = df["taxon_name"].str.split().str[0]
    df["family"] = ""
    df["compound_name"] = ""

    df = df.drop_duplicates(subset=["compound_qid", "taxon_qid"])
    log.info("final library: %d pairs, %d compounds, %d taxa",
             len(df), df["compound_qid"].nunique(), df["taxon_qid"].nunique())
    return df[["compound_qid", "compound_name", "smiles", "inchikey",
               "taxon_qid", "taxon_name", "genus", "family"]].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build plant NP library from LOTUS/Wikidata")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/processed/np_library.csv")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    setup_logging()
    out = Path(args.out)
    if out.exists() and not args.force:
        log.info("%s already present; use --force to refetch", out)
        return 0
    cfg = load_config(args.config)
    df = build_library(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
