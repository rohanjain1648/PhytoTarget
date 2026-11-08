"""Source-organism enrichment: which plants are over-represented among predicted hits.

This is the step that turns a compound ranking back into something an
ethnopharmacologist can act on. A list of 500 promising molecules is hard to use.
"These eleven genera contain significantly more predicted actives than chance
would give you" is a field collection plan.

Statistics
----------
For each taxon we ask: given a library of M compounds of which n come from this
taxon, and a top-k slice of N compounds, is observing k hits from the taxon more
than chance? That is exactly the hypergeometric tail probability (sampling
without replacement), the same test used for GO-term enrichment.

Two corrections matter and both are applied:

* **Multiple testing.** We test thousands of taxa. Benjamini-Hochberg FDR, not
  raw p-values, decides what gets reported.
* **Rediscovery bias.** Compounds already in the target's training actives are
  excluded from the top-k before testing. Otherwise a genus like *Galanthus*
  scores as "enriched" purely because its alkaloid is already a licensed drug
  that the model memorised, which tells a prospector nothing new.

Taxa with fewer than `min_compounds_per_taxon` library entries are not tested at
all. With one or two compounds the test has no power, and including them just
inflates the multiple-testing burden.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .metrics import benjamini_hochberg
from .util import load_config, setup_logging

log = logging.getLogger(__name__)

# Curated ethnobotanical context for genera that commonly surface. These are
# literature notes for human interpretation only - they are NEVER used as model
# input, so they cannot leak into the statistics they are displayed next to.
TRADITIONAL_USE_NOTES: dict[str, str] = {
    "Galanthus": "Snowdrop; source of galantamine, a licensed anti-Alzheimer's AChE inhibitor.",
    "Huperzia": "Chinese club moss (Qian Ceng Ta); source of huperzine A, long used for memory complaints.",
    "Physostigma": "Calabar bean; source of physostigmine, the archetypal carbamate AChE inhibitor.",
    "Narcissus": "Amaryllidaceae; shares the galantamine-type alkaloid biosynthesis of Galanthus.",
    "Lycoris": "Amaryllidaceae; lycorine-type alkaloids, traditional Chinese medicine.",
    "Crinum": "Amaryllidaceae; alkaloid-rich, used in African and Asian traditional medicine.",
    "Peganum": "Syrian rue; harmine/harmaline, reversible MAO-A/B inhibitors, ritual and medicinal use.",
    "Banisteriopsis": "Ayahuasca vine; beta-carboline MAO inhibitors central to its pharmacology.",
    "Passiflora": "Passionflower; traditional anxiolytic, contains beta-carboline alkaloids.",
    "Berberis": "Barberry; berberine, broad traditional use across Ayurveda and TCM.",
    "Coptis": "Huang Lian; berberine-type alkaloids, classical TCM antimicrobial/metabolic herb.",
    "Curcuma": "Turmeric; curcuminoids, extensively studied but notorious for assay interference.",
    "Glycyrrhiza": "Liquorice; prenylated flavonoids, one of the most-used herbs in TCM formulae.",
    "Scutellaria": "Baikal skullcap; baicalein/wogonin flavones, TCM neuroprotective use.",
    "Salvia": "Sage/Danshen; diterpenoids and phenolics, memory and circulatory traditional uses.",
    "Withania": "Ashwagandha; withanolides, Ayurvedic nootropic and adaptogen.",
    "Bacopa": "Brahmi; bacosides, classical Ayurvedic memory tonic.",
    "Centella": "Gotu kola; triterpenes, Ayurvedic cognitive and wound-healing use.",
    "Ginkgo": "Maidenhair tree; ginkgolides/flavonoids, widely used for cognitive complaints.",
    "Panax": "Ginseng; ginsenosides, central tonic in Chinese and Korean medicine.",
    "Rhodiola": "Golden root; salidroside/rosavins, traditional adaptogen.",
    "Corydalis": "Yan Hu Suo; isoquinoline alkaloids, TCM analgesic.",
    "Stephania": "Fangji; bisbenzylisoquinoline alkaloids, TCM use.",
    "Tabernaemontana": "Apocynaceae; monoterpene indole alkaloids, pantropical traditional medicine.",
    "Rauvolfia": "Sarpagandha; reserpine-type alkaloids, Ayurvedic hypertension/psychiatric use.",
    "Catharanthus": "Madagascar periwinkle; bisindole alkaloids, source of vinca anticancer drugs.",
    "Papaver": "Poppy; benzylisoquinoline alkaloids, ancient analgesic use.",
    "Nelumbo": "Sacred lotus; aporphine alkaloids, TCM and Ayurvedic sedative use.",
    "Piper": "Pepper genus; amides and lignans, pan-tropical culinary and medicinal use.",
    "Zanthoxylum": "Prickly ash; isoquinoline alkaloids and amides, widespread traditional use.",
}


def _compound_taxa(lib: pd.DataFrame) -> pd.DataFrame:
    """(compound key, taxon_name, genus) edge list keyed the same way as screen.py."""
    lib = lib.copy()
    lib["inchikey"] = lib["inchikey"].fillna("")
    lib["_key"] = np.where(lib["inchikey"].str.len() > 0, lib["inchikey"], lib["smiles"])
    return lib[["_key", "taxon_name", "genus"]].drop_duplicates()


def enrich_level(edges: pd.DataFrame, level_col: str, level_name: str,
                 scored_keys: set, top_keys: set, cfg: dict) -> pd.DataFrame:
    """Hypergeometric over-representation for one taxonomic level."""
    e = edges[edges["_key"].isin(scored_keys)]
    e = e[e[level_col].astype(str).str.strip() != ""]

    pop_size = len(scored_keys)
    n_draws = len(top_keys)
    min_n = cfg["screening"]["min_compounds_per_taxon"]

    rows = []
    for taxon, grp in e.groupby(level_col):
        members = set(grp["_key"])
        n_taxon = len(members)
        if n_taxon < min_n:
            continue
        observed = len(members & top_keys)
        expected = n_draws * n_taxon / pop_size if pop_size else 0.0
        # sf(k-1) = P(X >= k): probability of seeing at least this many by chance.
        p = float(hypergeom.sf(observed - 1, pop_size, n_taxon, n_draws)) if observed > 0 else 1.0
        rows.append({
            "taxon_level": level_name,
            "taxon_name": taxon,
            "n_compounds_in_library": n_taxon,
            "n_in_top_k": observed,
            "expected": round(expected, 3),
            "fold_enrichment": round(observed / expected, 3) if expected > 0 else 0.0,
            "p_value": p,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = benjamini_hochberg(df["p_value"].to_numpy())
    return df


def run(target_id: str, cfg: dict, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    screen = pd.read_csv(out_dir / f"screen_{target_id}.csv")
    lib = pd.read_csv(data_dir / "np_library.csv")
    edges = _compound_taxa(lib)

    screen["inchikey"] = screen["inchikey"].fillna("")
    screen["_key"] = np.where(screen["inchikey"].str.len() > 0, screen["inchikey"], screen["smiles"])
    scored_keys = set(screen["_key"])

    top_k = cfg["screening"]["top_k_for_enrichment"]
    novel = screen[~screen["is_known_active"].astype(bool)]
    top_keys = set(novel.nsmallest(top_k, "rank")["_key"])
    log.info("population %d compounds; top-%d slice taken from %d novel predictions "
             "(%d known actives excluded)",
             len(scored_keys), top_k, len(novel), int(screen.is_known_active.sum()))

    parts = [
        enrich_level(edges, "genus", "genus", scored_keys, top_keys, cfg),
        enrich_level(edges, "taxon_name", "species", scored_keys, top_keys, cfg),
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if df.empty:
        log.warning("no taxa met the minimum-size requirement")
        return df

    df["traditional_use_note"] = df["taxon_name"].map(
        lambda t: TRADITIONAL_USE_NOTES.get(str(t).split()[0] if t else "", ""))
    df = df.sort_values(["q_value", "p_value", "fold_enrichment"],
                        ascending=[True, True, False]).reset_index(drop=True)

    alpha = cfg["screening"]["fdr_alpha"]
    sig = df[df["q_value"] < alpha]
    log.info("%d taxa tested, %d significant at FDR < %.2f", len(df), len(sig), alpha)
    for _, r in sig.head(12).iterrows():
        note = f"  <- {r['traditional_use_note']}" if r["traditional_use_note"] else ""
        log.info("  %-8s %-28s %2d/%5.2f  fold=%4.1f  q=%.2e%s",
                 r["taxon_level"], r["taxon_name"], r["n_in_top_k"],
                 r["expected"], r["fold_enrichment"], r["q_value"], note)

    cols = ["taxon_level", "taxon_name", "n_compounds_in_library", "n_in_top_k",
            "expected", "fold_enrichment", "p_value", "q_value", "traditional_use_note"]
    path = out_dir / f"enrichment_{target_id}.csv"
    df[cols].to_csv(path, index=False)
    log.info("wrote %s", path)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Source-organism enrichment over screening hits")
    ap.add_argument("--target")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    targets = [args.target] if args.target else [t["chembl_id"] for t in cfg["targets"]]
    for tid in targets:
        if not (Path(args.out_dir) / f"screen_{tid}.csv").exists():
            log.error("no screen output for %s - run screen first", tid)
            continue
        log.info("================ enrichment %s ================", tid)
        run(tid, cfg, Path(args.data_dir), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
