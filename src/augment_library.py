"""Guarantee library coverage for the retrospective validation compounds.

Why this step exists
--------------------
`fetch_lotus` pages through LOTUS occurrence statements with LIMIT/OFFSET and stops
at a configured cap, because the full set is ~1M statements and WDQS will not serve
it reliably in one pass. The result is a large but *incomplete* sample of LOTUS, and
WDQS paging without a global ORDER BY is not even guaranteed to be stable between
runs.

That sampling is fine for screening - 23k standardised natural products is a real
library - but it broke the retrospective check: *Galanthus* and *Huperzia* are both
in the sample, yet galantamine and huperzine A themselves are not, so there was
nothing for the recovery test to find.

This script closes that specific gap by fetching occurrence statements for the named
retrospective compounds directly, rather than hoping they fall inside an arbitrary
offset window.

Why this is not cheating
------------------------
We ensure the *library* contains the compounds, which is a data-coverage fix. We do
not tell the model which ones they are, and we do not touch their scores: they are
ranked among all ~24k compounds like everything else, and the screen marks each one
`is_known_active` because they are in the target's ChEMBL training set. A high rank
for a flagged known active is a rediscovery, not a discovery - which is exactly why
`enrich.py` removes known actives before testing organism enrichment.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from .chem import inchikey, standardize
from .fetch_lotus import _qid, _sparql
from .util import load_config, setup_logging

log = logging.getLogger(__name__)

# Resolving a name to a QID through SPARQL means scanning every rdfs:label in
# Wikidata, which WDQS answers with a 504. The entity-search REST endpoint does the
# same job in one cheap call, so names are resolved there and SPARQL is used only
# for the narrow follow-up: "for THESE qids, what are the P703 taxa?"
SEARCH_URL = "https://www.wikidata.org/w/api.php"

OCCURRENCE_QUERY = """
SELECT ?compound ?smiles ?taxon ?taxonLabel WHERE {{
  VALUES ?compound {{ {qids} }}
  ?compound wdt:P233 ?smiles .
  ?compound wdt:P703 ?taxon .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""


def resolve_qid(name: str) -> tuple[str, str] | None:
    """(qid, label) for a compound name, verified to carry a SMILES string.

    The search endpoint ranks by relevance but will happily return a disambiguation
    page or an unrelated concept, so each candidate is checked for P233 (canonical
    SMILES) before being accepted. A "compound" with no structure is not a compound
    we can score.
    """
    try:
        r = requests.get(SEARCH_URL, params={
            "action": "wbsearchentities", "search": name, "language": "en",
            "uselang": "en", "type": "item", "limit": 8, "format": "json",
        }, timeout=30, headers={"User-Agent": "PhytoTarget/1.0 (research prototype)"})
        r.raise_for_status()
        candidates = [h["id"] for h in r.json().get("search", [])]
    except Exception as exc:
        log.warning("search failed for %s: %s", name, exc)
        return None
    if not candidates:
        return None

    try:
        r = requests.get(SEARCH_URL, params={
            "action": "wbgetentities", "ids": "|".join(candidates),
            "props": "claims|labels", "languages": "en", "format": "json",
        }, timeout=45, headers={"User-Agent": "PhytoTarget/1.0 (research prototype)"})
        r.raise_for_status()
        entities = r.json().get("entities", {})
    except Exception as exc:
        log.warning("entity lookup failed for %s: %s", name, exc)
        return None

    for qid in candidates:                      # keep search relevance order
        ent = entities.get(qid, {})
        claims = ent.get("claims", {})
        if "P233" not in claims and "P2017" not in claims:
            continue                            # no SMILES -> not a structure
        label = ent.get("labels", {}).get("en", {}).get("value", name)
        return qid, label
    return None


def fetch_compound_occurrences(names: list[str]) -> pd.DataFrame:
    if not names:
        return pd.DataFrame()

    resolved: dict[str, str] = {}
    for n in names:
        hit = resolve_qid(n)
        if hit:
            resolved[hit[0]] = hit[1]
            log.info("  %-16s -> %s (%s)", n, hit[0], hit[1])
        else:
            log.warning("  %-16s could not be resolved to a structure on Wikidata", n)
    if not resolved:
        return pd.DataFrame()

    values = " ".join(f"wd:{q}" for q in resolved)
    try:
        bindings = _sparql(OCCURRENCE_QUERY.format(qids=values))
    except RuntimeError as exc:
        log.error("WDQS occurrence query failed: %s", exc)
        return pd.DataFrame()

    rows = []
    for b in bindings:
        smi = standardize(b["smiles"]["value"])
        if not smi:
            continue
        qid = _qid(b["compound"]["value"])
        taxon = b.get("taxonLabel", {}).get("value", "")
        rows.append({
            "compound_qid": qid,
            "compound_name": resolved.get(qid, ""),
            "smiles": smi,
            "inchikey": inchikey(smi),
            "taxon_qid": _qid(b["taxon"]["value"]),
            "taxon_name": taxon,
            "genus": taxon.split()[0] if taxon else "",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("no occurrence statements returned")
        return df
    log.info("retrieved %d pairs covering %d compounds across %d taxa",
             len(df), df.compound_qid.nunique(), df.taxon_qid.nunique())
    return df


def run(cfg: dict, data_dir: Path) -> pd.DataFrame:
    names = sorted({p["name"] for t in cfg["targets"]
                    for p in (t.get("retrospective_positives") or [])})
    log.info("retrospective compounds to guarantee: %s", ", ".join(names))

    extra = fetch_compound_occurrences(names)
    path = data_dir / "np_library.csv"
    lib = pd.read_csv(path)
    before_pairs, before_cmpds = len(lib), lib.compound_qid.nunique()

    if extra.empty:
        log.warning("nothing to add; library unchanged")
        return lib

    lib = pd.concat([lib, extra[lib.columns.intersection(extra.columns)]],
                    ignore_index=True)
    lib = lib.drop_duplicates(subset=["compound_qid", "taxon_qid"]).reset_index(drop=True)
    lib.to_csv(path, index=False)
    log.info("library: %d -> %d pairs, %d -> %d compounds",
             before_pairs, len(lib), before_cmpds, lib.compound_qid.nunique())

    # Report, per compound, whether it is now present and in which plants.
    for n in names:
        hit = extra[extra["compound_name"].fillna("").str.lower() == n.lower()]
        if hit.empty:
            log.warning("  %-16s still absent - no P703 statement on Wikidata", n)
        else:
            plants = ", ".join(sorted(set(hit["taxon_name"]))[:4])
            log.info("  %-16s present in %d taxa (%s)", n, hit.taxon_qid.nunique(), plants)

    # The cached feature matrix was built for the old row count; drop it so the
    # next screen rebuilds rather than silently scoring a stale library.
    for cache in data_dir.glob("libfeatures_*.npz"):
        cache.unlink()
        log.info("invalidated feature cache %s", cache.name)
    return lib


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    args = ap.parse_args(argv)
    setup_logging()
    run(load_config(args.config), Path(args.data_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
