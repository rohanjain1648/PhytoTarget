"""Assemble the plant -> compound -> target -> disease knowledge graph.

A ranked CSV answers "what should I test". It does not answer "why", and "why"
is what makes a result arguable in front of a pharmacognosist. The graph makes
the chain explicit and walkable:

    Huperzia serrata --CONTAINS--> huperzine A --PREDICTED_ACTIVE(p=0.94)-->
    Acetylcholinesterase --IMPLICATED_IN--> Alzheimer's disease

Every edge carries its evidence: CONTAINS comes from a referenced Wikidata/LOTUS
occurrence statement, PREDICTED_ACTIVE carries the calibrated probability *and*
the applicability-domain flag, KNOWN_ACTIVE carries the measured pChEMBL. A path
through the graph is therefore a chain of claims, each of which can be checked
rather than taken on trust.

The graph is deliberately built from the top-N predictions, not the whole
library: a graph of 200k compounds is not a graph anyone reads.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .util import load_config, setup_logging, target_config, write_json

log = logging.getLogger(__name__)


def build(target_id: str, cfg: dict, data_dir: Path, out_dir: Path,
          top_n: int = 300) -> nx.DiGraph:
    tcfg = target_config(cfg, target_id)
    screen = pd.read_csv(out_dir / f"screen_{target_id}.csv")
    lib = pd.read_csv(data_dir / "np_library.csv")

    enr_path = out_dir / f"enrichment_{target_id}.csv"
    enrichment = pd.read_csv(enr_path) if enr_path.exists() else pd.DataFrame()
    q_by_taxon: dict[str, float] = {}
    if not enrichment.empty:
        for _, r in enrichment.iterrows():
            q_by_taxon[str(r["taxon_name"])] = float(r["q_value"])

    top = screen.nsmallest(top_n, "rank").copy()
    top["inchikey"] = top["inchikey"].fillna("")
    top["_key"] = np.where(top["inchikey"].str.len() > 0, top["inchikey"], top["smiles"])

    lib = lib.copy()
    lib["inchikey"] = lib["inchikey"].fillna("")
    lib["_key"] = np.where(lib["inchikey"].str.len() > 0, lib["inchikey"], lib["smiles"])
    lib = lib[lib["_key"].isin(set(top["_key"]))]

    G = nx.DiGraph()

    target_node = f"target:{target_id}"
    G.add_node(target_node, type="target", label=tcfg.get("name", target_id),
               chembl_id=target_id)

    disease = tcfg.get("disease") or ""
    if disease:
        disease_node = f"disease:{disease}"
        G.add_node(disease_node, type="disease", label=disease)
        G.add_edge(target_node, disease_node, type="IMPLICATED_IN", weight=1.0,
                   evidence="curated target-disease association")

    for _, row in top.iterrows():
        cnode = f"compound:{row['compound_qid']}"
        label = row["compound_name"] if isinstance(row["compound_name"], str) and row["compound_name"].strip() else row["compound_qid"]
        G.add_node(cnode, type="compound", label=label,
                   smiles=row["smiles"], inchikey=row["inchikey"],
                   rank=int(row["rank"]),
                   prob_active=round(float(row["prob_active"]), 4),
                   in_domain=bool(row["in_domain"]),
                   max_train_similarity=round(float(row["max_train_similarity"]), 3),
                   pains_alerts=int(row["pains_alerts"]),
                   is_known_active=bool(row["is_known_active"]))

        edge_type = "KNOWN_ACTIVE" if bool(row["is_known_active"]) else "PREDICTED_ACTIVE"
        G.add_edge(cnode, target_node, type=edge_type,
                   weight=round(float(row["prob_active"]), 4),
                   in_domain=bool(row["in_domain"]),
                   evidence=("measured in ChEMBL" if edge_type == "KNOWN_ACTIVE"
                             else "model prediction, calibrated"))

    for _, row in lib.iterrows():
        taxon = str(row["taxon_name"]).strip()
        if not taxon:
            continue
        tnode = f"taxon:{row['taxon_qid']}"
        if tnode not in G:
            G.add_node(tnode, type="taxon", label=taxon,
                       genus=str(row.get("genus", "")),
                       enrichment_q=q_by_taxon.get(taxon, None))
        cnode = f"compound:{row['compound_qid']}"
        if cnode in G:
            G.add_edge(tnode, cnode, type="CONTAINS", weight=1.0,
                       evidence="LOTUS/Wikidata occurrence statement (P703)")

    counts = {}
    for _, attrs in G.nodes(data=True):
        counts[attrs.get("type", "?")] = counts.get(attrs.get("type", "?"), 0) + 1
    log.info("graph: %d nodes %s, %d edges", G.number_of_nodes(), counts, G.number_of_edges())
    return G


def explain_path(G: nx.DiGraph, taxon_label: str, max_paths: int = 5) -> list[list[str]]:
    """Human-readable chains from a plant to the disease node.

    This is the query a domain expert actually wants: "you say this plant is
    interesting - show me the reasoning".
    """
    taxon_nodes = [n for n, a in G.nodes(data=True)
                   if a.get("type") == "taxon" and taxon_label.lower() in str(a.get("label", "")).lower()]
    disease_nodes = [n for n, a in G.nodes(data=True) if a.get("type") == "disease"]
    if not taxon_nodes or not disease_nodes:
        return []

    out = []
    for tn in taxon_nodes:
        for dn in disease_nodes:
            try:
                for path in nx.all_simple_paths(G, tn, dn, cutoff=4):
                    readable = []
                    for i, node in enumerate(path):
                        label = G.nodes[node].get("label", node)
                        if i + 1 < len(path):
                            etype = G.edges[node, path[i + 1]].get("type", "")
                            w = G.edges[node, path[i + 1]].get("weight", "")
                            label = f"{label} --{etype}({w})-->"
                        readable.append(label)
                    out.append(readable)
                    if len(out) >= max_paths:
                        return out
            except nx.NetworkXNoPath:
                continue
    return out


def run(target_id: str, cfg: dict, data_dir: Path, out_dir: Path, top_n: int = 300):
    G = build(target_id, cfg, data_dir, out_dir, top_n)
    data = nx.node_link_data(G, edges="links")
    write_json(data, out_dir / f"kg_{target_id}.json")
    log.info("wrote %s", out_dir / f"kg_{target_id}.json")

    nx.write_graphml(_graphml_safe(G), out_dir / f"kg_{target_id}.graphml")
    log.info("wrote %s (open in Cytoscape/Gephi)", out_dir / f"kg_{target_id}.graphml")

    tcfg = target_config(cfg, target_id)
    for entry in (tcfg.get("retrospective_positives") or []):
        plant = entry.get("source_plant", "")
        if not plant:
            continue
        paths = explain_path(G, plant, max_paths=2)
        if paths:
            log.info("example path for %s:", plant)
            for p in paths:
                log.info("   %s", " ".join(p))
    return G


def _graphml_safe(G: nx.DiGraph) -> nx.DiGraph:
    """GraphML rejects None values, so drop them rather than lose the export."""
    H = G.copy()
    for _, attrs in H.nodes(data=True):
        for k in [k for k, v in attrs.items() if v is None]:
            del attrs[k]
    for _, _, attrs in H.edges(data=True):
        for k in [k for k, v in attrs.items() if v is None]:
            del attrs[k]
    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the PhytoTarget knowledge graph")
    ap.add_argument("--target")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--top-n", type=int, default=300)
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    targets = [args.target] if args.target else [t["chembl_id"] for t in cfg["targets"]]
    for tid in targets:
        if not (Path(args.out_dir) / f"screen_{tid}.csv").exists():
            log.error("no screen output for %s", tid)
            continue
        log.info("================ graph %s ================", tid)
        run(tid, cfg, Path(args.data_dir), Path(args.out_dir), args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
