"""PhytoTarget — interactive screening dashboard.

Run with:  streamlit run app/streamlit_app.py

The dashboard is organised around one principle: never show a prediction without
showing how much to trust it. Every ranked table carries the applicability-domain
flag, the PAINS count and the known-active marker, and the Validation tab exists
so a reviewer can check the model's honesty before looking at a single hit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

st.set_page_config(page_title="PhytoTarget", page_icon="🌿", layout="wide")

ACCENT = "#2E7D5B"
WARN = "#B4541E"


# ----------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def available_targets() -> list[str]:
    return sorted(p.stem.replace("screen_", "") for p in OUT.glob("screen_*.csv"))


@st.cache_data(show_spinner=False)
def load_screen(tid: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"screen_{tid}.csv")


@st.cache_data(show_spinner=False)
def load_metrics(tid: str) -> dict:
    path = OUT / f"metrics_{tid}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_enrichment(tid: str) -> pd.DataFrame:
    path = OUT / f"enrichment_{tid}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_kg(tid: str) -> dict:
    path = OUT / f"kg_{tid}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def mol_image(smiles: str, size=(260, 200)):
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        return Draw.MolToImage(mol, size=size) if mol else None
    except Exception:
        return None


# ------------------------------------------------------------------------ setup
targets = available_targets()
if not targets:
    st.title("🌿 PhytoTarget")
    st.warning("No screening results found in `outputs/`. Run the pipeline first:\n\n"
               "```bash\npython -m src.fetch_chembl\npython -m src.fetch_lotus\n"
               "python -m src.train\npython -m src.screen\npython -m src.enrich\n"
               "python -m src.kg\n```")
    st.stop()

with st.sidebar:
    st.markdown("## 🌿 PhytoTarget")
    st.caption("Bias-corrected screening of natural products, with source-organism enrichment.")
    tid = st.selectbox("Disease target", targets,
                       format_func=lambda t: f"{load_metrics(t).get('target_name', t)} ({t})")
    metrics = load_metrics(tid)
    if metrics.get("disease"):
        st.info(f"**Indication**\n\n{metrics['disease']}")
    st.divider()
    st.caption("Data: ChEMBL (CC BY-SA 3.0) · LOTUS/Wikidata (CC0)")
    st.caption("Research prototype. Not clinical advice.")

screen = load_screen(tid)
enrichment = load_enrichment(tid)

st.title(f"{metrics.get('target_name', tid)} — plant natural-product screen")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Compounds screened", f"{len(screen):,}")
c2.metric("Scaffold-split ROC-AUC", f"{metrics.get('splits', {}).get('scaffold', {}).get('roc_auc', float('nan')):.3f}")
c3.metric("EF @ 1% (decoy-augmented)",
          f"{metrics.get('decoy_augmented', {}).get('ef_1pct', float('nan')):.0f}×")
c4.metric("In applicability domain", f"{100 * screen['in_domain'].mean():.1f}%")
sig = int((enrichment["q_value"] < 0.05).sum()) if not enrichment.empty else 0
c5.metric("Enriched taxa (FDR<0.05)", sig)

tab_hits, tab_taxa, tab_valid, tab_graph, tab_about = st.tabs(
    ["🔬 Ranked hits", "🌱 Organism enrichment", "📊 Validation", "🕸 Knowledge graph", "ℹ️ Method"])

# -------------------------------------------------------------------- hits tab
with tab_hits:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Ranked predictions")
        f1, f2, f3 = st.columns(3)
        only_domain = f1.checkbox("Applicability domain only", value=True,
                                  help="Hide compounds too dissimilar to training data for the "
                                       "probability to be meaningful.")
        hide_known = f2.checkbox("Hide known actives", value=True,
                                 help="Known actives are rediscoveries, not new leads.")
        clean_only = f3.checkbox("No PAINS alerts", value=False,
                                 help="Exclude frequent-hitter substructures.")

        view = screen.copy()
        if only_domain:
            view = view[view["in_domain"]]
        if hide_known:
            view = view[~view["is_known_active"].astype(bool)]
        if clean_only:
            view = view[view["pains_alerts"] == 0]

        st.caption(f"{len(view):,} compounds after filters")
        show = view.head(200)[["rank", "compound_name", "prob_active",
                               "max_train_similarity", "pains_alerts",
                               "n_source_taxa", "top_taxa", "compound_qid"]]
        st.dataframe(
            show, use_container_width=True, height=440, hide_index=True,
            column_config={
                "prob_active": st.column_config.ProgressColumn(
                    "P(active)", min_value=0.0, max_value=1.0, format="%.3f"),
                "max_train_similarity": st.column_config.NumberColumn(
                    "Max train sim.", format="%.2f"),
                "pains_alerts": st.column_config.NumberColumn("PAINS"),
                "n_source_taxa": st.column_config.NumberColumn("# plants"),
                "top_taxa": st.column_config.TextColumn("Source plants", width="medium"),
            })
        st.download_button("⬇ Download filtered hits (CSV)",
                           view.to_csv(index=False).encode(),
                           file_name=f"phytotarget_hits_{tid}.csv", mime="text/csv")

    with right:
        st.subheader("Inspect compound")
        if len(view):
            options = view.head(200)
            idx = st.selectbox(
                "Pick a hit", range(len(options)),
                format_func=lambda i: f"#{options.iloc[i]['rank']} "
                                      f"{options.iloc[i]['compound_name'] or options.iloc[i]['compound_qid']}")
            row = options.iloc[idx]
            img = mol_image(row["smiles"])
            if img:
                st.image(img, use_container_width=False)
            st.markdown(f"**P(active)** `{row['prob_active']:.3f}`  \n"
                        f"**Max similarity to training set** `{row['max_train_similarity']:.2f}`  \n"
                        f"**PAINS alerts** `{int(row['pains_alerts'])}`  \n"
                        f"**Found in** {row['n_source_taxa']} plant taxa")
            st.caption(f"Source plants: {row['top_taxa']}")
            st.code(row["smiles"], language=None)
            st.link_button("Open in Wikidata",
                           f"https://www.wikidata.org/wiki/{row['compound_qid']}")
            if row["max_train_similarity"] < 0.35:
                st.warning("Outside applicability domain — the model is extrapolating here.")
            if int(row["pains_alerts"]) > 0:
                st.warning("Carries a PAINS substructure: assay readout may be unreliable.")

# -------------------------------------------------------------------- taxa tab
with tab_taxa:
    if enrichment.empty:
        st.info("No enrichment results. Run `python -m src.enrich`.")
    else:
        st.subheader("Which plants are over-represented among predicted actives?")
        st.caption("Hypergeometric over-representation of source taxa in the top-k predictions, "
                   "Benjamini–Hochberg corrected. Known actives are excluded before testing so "
                   "enrichment reflects novel predictions rather than memorised drugs.")
        level = st.radio("Taxonomic level", ["genus", "species"], horizontal=True)
        sub = enrichment[enrichment["taxon_level"] == level].copy()
        sig_only = st.checkbox("Significant only (FDR < 0.05)", value=True)
        if sig_only:
            sub = sub[sub["q_value"] < 0.05]

        if sub.empty:
            st.warning("Nothing significant at this level.")
        else:
            plot = sub.nsmallest(20, "q_value").sort_values("fold_enrichment")
            fig = px.bar(plot, x="fold_enrichment", y="taxon_name", orientation="h",
                         color="q_value", color_continuous_scale="Greens_r",
                         labels={"fold_enrichment": "Fold enrichment",
                                 "taxon_name": "", "q_value": "FDR q"},
                         hover_data=["n_in_top_k", "n_compounds_in_library", "expected"])
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                sub[["taxon_name", "n_in_top_k", "n_compounds_in_library", "expected",
                     "fold_enrichment", "q_value", "traditional_use_note"]].head(60),
                use_container_width=True, hide_index=True,
                column_config={
                    "n_in_top_k": st.column_config.NumberColumn("Hits in top-k"),
                    "n_compounds_in_library": st.column_config.NumberColumn("In library"),
                    "expected": st.column_config.NumberColumn("Expected", format="%.2f"),
                    "fold_enrichment": st.column_config.NumberColumn("Fold", format="%.1f×"),
                    "q_value": st.column_config.NumberColumn("FDR q", format="%.2e"),
                    "traditional_use_note": st.column_config.TextColumn(
                        "Ethnobotanical context (literature, not model input)", width="large"),
                })

# -------------------------------------------------------------- validation tab
with tab_valid:
    st.subheader("How much should you trust this model?")
    splits = metrics.get("splits", {})
    rnd, scaf = splits.get("random", {}), splits.get("scaffold", {})

    st.markdown("#### 1. Random split vs scaffold split")
    st.caption("A random split lets analogues of the same scaffold sit in both train and test, so "
               "the model is graded on chemistry it has effectively already seen. The gap between "
               "these two bars is the amount by which a random-split result would have overstated "
               "performance.")
    comp = pd.DataFrame({
        "Split": ["Random (optimistic)", "Scaffold (honest)"],
        "ROC-AUC": [rnd.get("roc_auc", 0), scaf.get("roc_auc", 0)],
        "PR-AUC": [rnd.get("pr_auc", 0), scaf.get("pr_auc", 0)],
    })
    fig = go.Figure()
    fig.add_bar(name="ROC-AUC", x=comp["Split"], y=comp["ROC-AUC"], marker_color=ACCENT)
    fig.add_bar(name="PR-AUC", x=comp["Split"], y=comp["PR-AUC"], marker_color="#89B8A0")
    fig.update_layout(barmode="group", height=320, yaxis_range=[0, 1],
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.metric("AUC inflation from random splitting",
              f"+{metrics.get('auc_inflation_random_minus_scaffold', 0):.3f}")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("#### 2. y-scrambling")
        ys = metrics.get("y_scramble", {})
        mean = ys.get("mean_roc_auc", float("nan"))
        st.metric("Scrambled-label ROC-AUC", f"{mean:.3f}",
                  delta=f"{mean - 0.5:+.3f} vs chance")
        st.caption(f"{ys.get('n_runs', 0)} runs with permuted labels. Must collapse to ≈0.50; "
                   "anything higher means the pipeline is exploiting data layout, not chemistry.")
    with v2:
        st.markdown("#### 3. AVE split bias")
        ave = metrics.get("ave_bias", {})
        val = ave.get("ave_bias", float("nan")) if isinstance(ave, dict) else ave
        st.metric("AVE bias", f"{val:+.3f}")
        st.caption("Wallach & Heifets (2018). Near 0 means actives and inactives are not trivially "
                   "separable by nearest-neighbour geometry alone.")
    with v3:
        st.markdown("#### 4. Calibration")
        cal = metrics.get("calibration", {})
        st.metric("Brier score", f"{cal.get('brier', float('nan')):.4f}")
        st.caption(f"ECE {cal.get('ece', float('nan')):.4f}, {cal.get('method', '—')} calibration. "
                   "A compound shown at P=0.9 should be active about 90% of the time.")

    st.markdown("#### 5. Early recognition, measured where it means something")
    da = metrics.get("decoy_augmented", {})
    raw_ceiling = 1.0 / scaf.get("active_fraction", 1) if scaf.get("active_fraction") else float("nan")
    st.caption(
        f"Enrichment factor is capped at 1/active_fraction. This test set is "
        f"{100 * scaf.get('active_fraction', 0):.0f}% actives because ChEMBL is "
        f"publication-biased, so EF@1% here **cannot exceed {raw_ceiling:.2f}** no matter how "
        "good the model is — quoting it would be meaningless. Instead the held-out scaffold "
        "actives are hidden among the real natural-product library, which puts prevalence near "
        "1% with the active count still in the hundreds.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{scaf.get('roc_auc', 0):.3f}")
    m2.metric("PR-AUC", f"{scaf.get('pr_auc', 0):.3f}")
    m3.metric("EF @ 1%, raw test set",
              f"{scaf.get('ef_1pct', 0):.2f}×",
              delta=f"ceiling {raw_ceiling:.2f}× — saturated", delta_color="off")
    m4.metric("EF @ 1%, decoy-augmented",
              f"{da.get('ef_1pct', float('nan')):.1f}×",
              delta=f"ceiling {da.get('ef_1pct_ceiling', float('nan')):.0f}×", delta_color="off")
    if da:
        st.caption(
            f"{da.get('n_held_out_actives', 0)} held-out actives among "
            f"{da.get('n_library_decoys', 0):,} library decoys "
            f"({100 * da.get('active_prevalence', 0):.2f}% prevalence). Decoys are *presumed* "
            "inactive rather than measured inactive, so a decoy that is truly active counts "
            "against the model — this is a conservative estimate.")

    with st.expander("Dataset composition"):
        st.json({k: metrics.get(k) for k in
                 ["n_molecules", "n_actives", "n_inactives", "n_scaffolds", "label_thresholds"]})

# ------------------------------------------------------------------- graph tab
with tab_graph:
    kg = load_kg(tid)
    if not kg:
        st.info("No graph found. Run `python -m src.kg`.")
    else:
        nodes, links = kg.get("nodes", []), kg.get("links", [])
        g1, g2, g3 = st.columns(3)
        g1.metric("Nodes", f"{len(nodes):,}")
        g2.metric("Edges", f"{len(links):,}")
        g3.metric("Plant taxa", sum(1 for n in nodes if n.get("type") == "taxon"))

        st.caption("Each edge carries its evidence: CONTAINS from a referenced LOTUS/Wikidata "
                   "occurrence statement, PREDICTED_ACTIVE from the calibrated model with its "
                   "applicability-domain flag. A path is a chain of checkable claims.")

        taxa = sorted({n["label"] for n in nodes if n.get("type") == "taxon"})
        pick = st.selectbox("Trace the reasoning from a plant", taxa) if taxa else None
        if pick:
            label_by_id = {n["id"]: n.get("label", n["id"]) for n in nodes}
            type_by_id = {n["id"]: n.get("type") for n in nodes}
            node_by_id = {n["id"]: n for n in nodes}
            taxon_ids = [n["id"] for n in nodes if n.get("label") == pick]
            rows = []
            for lk in links:
                if lk["source"] in taxon_ids and lk.get("type") == "CONTAINS":
                    cid = lk["target"]
                    cnode = node_by_id.get(cid, {})
                    rows.append({
                        "Compound": label_by_id.get(cid, cid),
                        "Rank": cnode.get("rank"),
                        "P(active)": cnode.get("prob_active"),
                        "In domain": cnode.get("in_domain"),
                        "PAINS": cnode.get("pains_alerts"),
                        "Known active": cnode.get("is_known_active"),
                    })
            if rows:
                st.markdown(f"**{pick}** → contains → these ranked compounds → "
                            f"**{metrics.get('target_name', tid)}** → implicated in → "
                            f"**{metrics.get('disease', '—')}**")
                st.dataframe(pd.DataFrame(rows).sort_values("Rank"),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No top-ranked compounds recorded for this taxon.")

        st.download_button("⬇ Download graph (JSON)",
                           json.dumps(kg).encode(),
                           file_name=f"phytotarget_kg_{tid}.json", mime="application/json")

# ------------------------------------------------------------------- about tab
with tab_about:
    st.markdown("""
### What this pipeline does

1. **Labels** — bioactivity for the chosen target is pulled from ChEMBL. Measurements are
   aggregated per molecule by **median** pChEMBL (not maximum, which cherry-picks optimistic
   outliers). Molecules between 10 µM and 316 nM are **dropped**, not relabelled as inactive:
   that band is where assay noise makes the class genuinely ambiguous.

2. **Features** — ECFP4 (Morgan radius 2, 2048 bits) plus twelve interpretable physicochemical
   descriptors. Every molecule is standardised once (largest fragment, neutralised) so the same
   compound arriving from ChEMBL and from Wikidata collapses to one InChIKey.

3. **Validation** — the headline number comes from a **Bemis–Murcko scaffold split**. The
   random-split number is computed too and reported next to it, so the inflation is visible
   rather than hidden. A **y-scrambling** null control and the **AVE bias** score test whether
   the signal is real.

4. **Screening** — the LOTUS natural-product library (plant→compound occurrences, via Wikidata)
   is scored with a calibrated model. Each prediction carries an **applicability-domain** flag,
   a **PAINS** count and a **known-active** marker.

5. **Organism enrichment** — hypergeometric over-representation of source taxa among the top-k
   *novel* predictions, FDR-corrected. This is what turns a compound list into a collection plan.

6. **Knowledge graph** — plant → compound → target → disease, every edge carrying its evidence.

### Limitations

- Ligand-based models interpolate from known chemistry. Genuinely novel scaffolds fall outside
  the applicability domain, which is exactly where natural products often sit — the flag is
  there to keep that honest rather than to hide it.
- LOTUS occurrence records vary in how carefully the species was identified, and a compound
  reported from a plant says nothing about the concentration present or whether a traditional
  preparation would extract it.
- A predicted binder is not a drug. Nothing here addresses absorption, metabolism, toxicity,
  or whether the target engagement helps the patient.
- **This is a research prototype and not clinical advice.**

### Attribution

- **ChEMBL** (EMBL-EBI) — bioactivity data, CC BY-SA 3.0
- **LOTUS** via **Wikidata** — compound–organism occurrences, CC0
- **RDKit** — cheminformatics toolkit, BSD-3
- PAINS filters: Baell & Holloway (2010). AVE bias: Wallach & Heifets (2018).
  BEDROC: Truchon & Bayly (2007).
""")
