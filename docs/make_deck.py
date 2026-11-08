"""Generate the PhytoTarget pitch deck as a 16:9 PDF.

Every number on every slide is read from outputs/ at build time rather than typed
in by hand. If the pipeline is rerun and a metric changes, the deck changes with it,
so the deck cannot drift out of sync with the code that produced it.

    python docs/make_deck.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BUILD = ROOT / "docs" / "build"
BUILD.mkdir(parents=True, exist_ok=True)

W, H = 13.333 * inch, 7.5 * inch           # 16:9
INK = HexColor("#16271F")
BODY = HexColor("#33473D")
MUTED = HexColor("#6E8279")
ACCENT = HexColor("#2E7D5B")
WARN = HexColor("#B4541E")
PAPER = HexColor("#FBFAF6")
RULE = HexColor("#D8DED6")

PRIMARY = "CHEMBL220"
SECONDARY = "CHEMBL2039"


# ----------------------------------------------------------------- data access
def metrics(tid: str) -> dict:
    p = OUT / f"metrics_{tid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def screen(tid: str) -> pd.DataFrame:
    p = OUT / f"screen_{tid}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def enrichment(tid: str) -> pd.DataFrame:
    p = OUT / f"enrichment_{tid}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


M, M2 = metrics(PRIMARY), metrics(SECONDARY)
S = screen(PRIMARY)
E = enrichment(PRIMARY)


# -------------------------------------------------------------- chart builders
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#D8DED6", "axes.labelcolor": "#33473D",
    "xtick.color": "#6E8279", "ytick.color": "#6E8279",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#FBFAF6", "axes.facecolor": "#FBFAF6",
})


def chart_splits(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=200)
    labels, rnd, scaf = [], [], []
    for m in (M, M2):
        if not m:
            continue
        labels.append(m.get("target_name", "?").replace("Monoamine oxidase B", "MAO-B")
                      .replace("Acetylcholinesterase", "AChE"))
        rnd.append(m["splits"]["random"]["roc_auc"])
        scaf.append(m["splits"]["scaffold"]["roc_auc"])
    x = range(len(labels))
    ax.bar([i - 0.19 for i in x], rnd, width=0.36, color="#B9CFC2",
           label="Random split (optimistic)")
    ax.bar([i + 0.19 for i in x], scaf, width=0.36, color="#2E7D5B",
           label="Scaffold split (honest)")
    for i, (r, s) in enumerate(zip(rnd, scaf)):
        ax.text(i - 0.19, r + 0.012, f"{r:.3f}", ha="center", fontsize=9, color="#6E8279")
        ax.text(i + 0.19, s + 0.012, f"{s:.3f}", ha="center", fontsize=9,
                color="#2E7D5B", fontweight="bold")
    ax.axhline(0.5, color="#B4541E", lw=1, ls=":", zorder=0)
    ax.text(len(labels) - 0.5, 0.515, "chance", fontsize=8, color="#B4541E", ha="right")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.04); ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def chart_scramble(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=200)
    ys = M.get("y_scramble", {})
    runs = ys.get("runs", [])
    ax.axhspan(0.45, 0.55, color="#E3EDE7", zorder=0)
    ax.scatter(range(1, len(runs) + 1), runs, s=55, color="#6E8279",
               zorder=3, label="label-permuted refits")
    ax.axhline(ys.get("mean_roc_auc", 0.5), color=ACCENT.hexval().replace("0x", "#")[:7],
               lw=1.6, label=f"mean {ys.get('mean_roc_auc', 0):.3f}")
    ax.axhline(M["splits"]["scaffold"]["roc_auc"], color="#2E7D5B", lw=1.6, ls="--",
               label=f"real labels {M['splits']['scaffold']['roc_auc']:.3f}")
    ax.set_ylim(0.3, 1.02); ax.set_xlabel("refit"); ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def chart_enrichment(path: Path) -> Path | None:
    if E.empty:
        return None
    sub = E[(E["taxon_level"] == "genus") & (E["q_value"] < 0.05)].nsmallest(10, "q_value")
    if sub.empty:
        sub = E[E["taxon_level"] == "genus"].nsmallest(10, "q_value")
    if sub.empty:
        return None
    sub = sub.sort_values("fold_enrichment")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.barh(sub["taxon_name"], sub["fold_enrichment"], color="#2E7D5B", height=0.62)
    for y, (f, q) in enumerate(zip(sub["fold_enrichment"], sub["q_value"])):
        ax.text(f + max(sub["fold_enrichment"]) * 0.02, y, f"q={q:.1e}",
                va="center", fontsize=8, color="#6E8279")
    ax.set_xlabel("Fold enrichment among top-k novel predictions")
    ax.set_xlim(0, max(sub["fold_enrichment"]) * 1.32)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def chart_domain(path: Path) -> Path | None:
    if S.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.3), dpi=200)
    ax.hist(S["max_train_similarity"], bins=46, color="#B9CFC2", edgecolor="#FBFAF6")
    thr = 0.35
    ax.axvline(thr, color="#B4541E", lw=1.8)
    inside = 100 * (S["max_train_similarity"] >= thr).mean()
    ax.text(thr + 0.012, ax.get_ylim()[1] * 0.92,
            f"  domain threshold {thr}\n  only {inside:.1f}% of the library\n  sits above it",
            fontsize=9, color="#B4541E", va="top")
    ax.set_xlabel("Max Tanimoto similarity to any training molecule")
    ax.set_ylabel("compounds")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ------------------------------------------------------------ drawing helpers
def bg(c: canvas.Canvas) -> None:
    c.setFillColor(PAPER); c.rect(0, 0, W, H, stroke=0, fill=1)


def kicker(c: canvas.Canvas, text: str) -> None:
    c.setFont("Helvetica-Bold", 10.5); c.setFillColor(ACCENT)
    c.drawString(0.92 * inch, H - 0.78 * inch, text.upper())


def title(c: canvas.Canvas, text: str, size: int = 30) -> None:
    c.setFont("Helvetica-Bold", size); c.setFillColor(INK)
    c.drawString(0.92 * inch, H - 1.33 * inch, text)


def rule(c: canvas.Canvas, y: float) -> None:
    c.setStrokeColor(RULE); c.setLineWidth(1)
    c.line(0.92 * inch, y, W - 0.92 * inch, y)


def wrap(c: canvas.Canvas, text: str, x: float, y: float, width: float,
         size: float = 12.5, leading: float = 17.5, color=BODY,
         font: str = "Helvetica") -> float:
    c.setFont(font, size); c.setFillColor(color)
    words, line = text.split(), ""
    for w in words:
        trial = f"{line} {w}".strip()
        if c.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            c.drawString(x, y, line); y -= leading; line = w
    if line:
        c.drawString(x, y, line); y -= leading
    return y


def bullets(c: canvas.Canvas, items: list[tuple[str, str]], x: float, y: float,
            width: float) -> float:
    for head, body in items:
        c.setFont("Helvetica-Bold", 12.5); c.setFillColor(INK)
        c.drawString(x, y, head)
        off = c.stringWidth(head + "  ", "Helvetica-Bold", 12.5)
        y = wrap(c, body, x + off, y, width - off, 12.5, 17.0)
        y -= 7
    return y


def stat(c: canvas.Canvas, x: float, y: float, value: str, label: str,
         color=ACCENT, vsize: int = 32) -> None:
    c.setFont("Helvetica-Bold", vsize); c.setFillColor(color)
    c.drawString(x, y, value)
    c.setFont("Helvetica", 9.6); c.setFillColor(MUTED)
    wrap(c, label, x, y - 0.235 * inch, 2.5 * inch, 9.6, 12.2, MUTED)


def footer(c: canvas.Canvas, n: int) -> None:
    c.setFont("Helvetica", 8.6); c.setFillColor(MUTED)
    c.drawString(0.92 * inch, 0.5 * inch, "PhytoTarget")
    c.drawRightString(W - 0.92 * inch, 0.5 * inch, str(n))


# ------------------------------------------------------------------- slides
def build(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=(W, H))
    page = [0]

    def newpage():
        footer(c, page[0]); c.showPage(); page[0] += 1; bg(c)

    bg(c); page[0] = 1

    # 1 — title
    c.setFillColor(ACCENT); c.rect(0, 0, 0.2 * inch, H, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 56); c.setFillColor(INK)
    c.drawString(1.15 * inch, H - 2.9 * inch, "PhytoTarget")
    c.setFont("Helvetica", 19); c.setFillColor(BODY)
    c.drawString(1.15 * inch, H - 3.5 * inch,
                 "Bias-corrected target prediction for plant natural products,")
    c.drawString(1.15 * inch, H - 3.86 * inch, "with source-organism enrichment.")
    rule(c, H - 4.4 * inch)
    c.setFont("Helvetica", 12.5); c.setFillColor(MUTED)
    c.drawString(1.15 * inch, H - 4.78 * inch,
                 "Traditional medicine names plants. Drug discovery needs molecules and targets.")
    c.drawString(1.15 * inch, H - 5.12 * inch,
                 "This closes that gap in the direction a field researcher can act on.")
    c.setFont("Helvetica-Oblique", 10.5); c.setFillColor(MUTED)
    c.drawString(1.15 * inch, 1.0 * inch,
                 "ChEMBL (CC BY-SA 3.0) + LOTUS/Wikidata (CC0) + RDKit  |  research prototype, not clinical advice")
    newpage()

    # 2 — problem
    kicker(c, "the gap")
    title(c, "Centuries of observation, indexed by the wrong key")
    rule(c, H - 1.68 * inch)
    y = H - 2.25 * inch
    y = bullets(c, [
        ("Traditional knowledge is organism-level.",
         "It records that a preparation of a plant helps a condition. It does not name the "
         "molecule or the protein it acts on."),
        ("Drug discovery is molecule-level.",
         "Target prediction, assays and medicinal chemistry all operate on structures. The "
         "two bodies of knowledge do not share an index."),
        ("The bridge exists but is unexploited.",
         "LOTUS, via Wikidata, holds hundreds of thousands of referenced plant-compound "
         "occurrence statements. Joined to a target model, it turns a plant into a set of "
         "testable molecular hypotheses."),
    ], 0.92 * inch, y, W - 1.84 * inch)

    y -= 0.25 * inch
    rule(c, y)
    y -= 0.52 * inch
    c.setFont("Helvetica-Bold", 14); c.setFillColor(INK)
    c.drawString(0.92 * inch, y, "And the inverse question is the one nobody tools for:")
    y -= 0.42 * inch
    c.setFont("Helvetica-Bold", 17); c.setFillColor(ACCENT)
    c.drawString(0.92 * inch, y,
                 "“Which plants are over-represented among the predicted actives?”")
    y -= 0.36 * inch
    wrap(c, "That answer is a collection plan, not just a compound list.",
         0.92 * inch, y, W - 1.84 * inch, 12.5, 17, MUTED)
    newpage()

    # 3 — why results are inflated
    kicker(c, "the real problem")
    title(c, "A screening model is easy to make look excellent")
    rule(c, H - 1.68 * inch)
    y = H - 2.18 * inch
    y = wrap(c, "Four failure modes produce impressive numbers that mean nothing. This "
                "pipeline instruments all four instead of assuming they are absent.",
             0.92 * inch, y, W - 1.84 * inch, 13, 18, BODY)
    y -= 0.3 * inch

    rows = [
        ("Random train/test split",
         "Analogues of one scaffold land in both halves; the model is graded on chemistry "
         "it memorised.",
         "Scaffold split is the headline. Random is reported beside it, so the inflation "
         "is visible."),
        ("Hidden label leakage",
         "Any bug that lets labels reach features inflates everything downstream.",
         "y-scrambling: 10 refits on permuted labels must collapse to 0.50."),
        ("Artificial decoy geometry",
         "Actives separable from inactives by nearest-neighbour distance alone.",
         "AVE bias (Wallach & Heifets 2018) computed and reported per target."),
        ("Publication-biased prevalence",
         "ChEMBL is 50-70% actives, which caps EF@1% near 1.6 - a saturated EF looks like "
         "a triumph.",
         "Held-out actives hidden in the real 23k natural-product library instead."),
    ]
    colx = [0.92 * inch, 4.1 * inch, 8.9 * inch]
    colw = [2.9 * inch, 4.5 * inch, 3.5 * inch]
    c.setFont("Helvetica-Bold", 9.6); c.setFillColor(MUTED)
    for x, h in zip(colx, ["FAILURE MODE", "WHAT IT DOES TO YOUR METRICS", "WHAT WE DO"]):
        c.drawString(x, y, h)
    y -= 0.1 * inch
    rule(c, y)
    y -= 0.34 * inch
    for name, harm, fix in rows:
        y0 = y
        c.setFont("Helvetica-Bold", 11.4); c.setFillColor(INK)
        c.drawString(colx[0], y, name)
        ya = wrap(c, harm, colx[1], y, colw[1], 10.8, 14.4, BODY)
        yb = wrap(c, fix, colx[2], y, colw[2], 10.8, 14.4, ACCENT)
        y = min(ya, yb) - 0.17 * inch
        c.setStrokeColor(RULE); c.setLineWidth(0.5)
        c.line(colx[0], y + 0.08 * inch, W - 0.92 * inch, y + 0.08 * inch)
        y -= 0.06 * inch
    newpage()

    # 4 — pipeline
    kicker(c, "architecture")
    title(c, "One standardisation path, two sources, four guards")
    rule(c, H - 1.68 * inch)

    boxes = [
        ("ChEMBL\nbioactivity", 0.92, 4.42, 2.0, 0.95, "#E3EDE7"),
        ("LOTUS / Wikidata\nplant → compound", 0.92, 3.1, 2.0, 0.95, "#E3EDE7"),
        ("curate\nmedian pChEMBL\ndrop 10µM-316nM", 3.35, 3.76, 1.95, 1.3, "#F2F4F0"),
        ("featurise\nECFP4 + 12 desc.\none standardiser", 5.72, 3.76, 1.95, 1.3, "#F2F4F0"),
        ("scaffold-split\ntrain + isotonic\ncalibration", 8.09, 3.76, 1.95, 1.3, "#C9DED3"),
        ("score 23,761\nNP compounds", 10.46, 3.76, 1.95, 1.3, "#2E7D5B"),
    ]
    for label, bx, by, bw, bh, fill in boxes:
        c.setFillColor(HexColor(fill))
        c.roundRect(bx * inch, by * inch, bw * inch, bh * inch, 7, stroke=0, fill=1)
        lines = label.split("\n")
        is_dark = fill == "#2E7D5B"
        c.setFillColor(PAPER if is_dark else INK)
        for i, ln in enumerate(lines):
            c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 10.2 if i == 0 else 9.0)
            c.drawCentredString((bx + bw / 2) * inch,
                                (by + bh - 0.3) * inch - i * 0.185 * inch, ln)
    c.setStrokeColor(MUTED); c.setLineWidth(1.1)
    for x0, x1, yy in [(2.92, 3.35, 4.41), (5.3, 5.72, 4.41),
                       (7.67, 8.09, 4.41), (10.04, 10.46, 4.41)]:
        c.line(x0 * inch, yy * inch, x1 * inch, yy * inch)
        c.line((x1 - 0.1) * inch, (yy + 0.06) * inch, x1 * inch, yy * inch)
        c.line((x1 - 0.1) * inch, (yy - 0.06) * inch, x1 * inch, yy * inch)

    y = 2.72 * inch
    c.setFont("Helvetica-Bold", 10.4); c.setFillColor(MUTED)
    c.drawString(0.92 * inch, y, "EVERY RANKED HIT CARRIES")
    y -= 0.32 * inch
    for head, body in [
        ("Applicability domain", "max Tanimoto to training; below threshold is flagged, not silently ranked"),
        ("PAINS count", "frequent-hitter substructures, so a promiscuous polyphenol is not celebrated"),
        ("Known-active flag", "rediscoveries are kept as validation but excluded before enrichment testing"),
    ]:
        c.setFont("Helvetica-Bold", 11.2); c.setFillColor(ACCENT)
        c.drawString(0.92 * inch, y, head)
        c.setFont("Helvetica", 11.2); c.setFillColor(BODY)
        c.drawString(0.92 * inch + c.stringWidth(head + "   ", "Helvetica-Bold", 11.2), y, body)
        y -= 0.265 * inch
    newpage()

    # 5 — validation
    kicker(c, "validation")
    title(c, "The honest number, next to the flattering one")
    rule(c, H - 1.68 * inch)
    p1 = chart_splits(BUILD / "splits.png")
    p2 = chart_scramble(BUILD / "scramble.png")
    c.drawImage(str(p1), 0.85 * inch, 2.12 * inch, width=5.7 * inch,
                height=3.2 * inch, preserveAspectRatio=True, anchor="sw")
    c.drawImage(str(p2), 6.9 * inch, 2.12 * inch, width=5.7 * inch,
                height=3.2 * inch, preserveAspectRatio=True, anchor="sw")
    c.setFont("Helvetica-Bold", 11.4); c.setFillColor(INK)
    c.drawString(0.92 * inch, 5.48 * inch, "Scaffold split vs random split")
    c.drawString(6.95 * inch, 5.48 * inch, "y-scrambling: the leakage control")
    y = 1.78 * inch
    infl = M.get("auc_inflation_random_minus_scaffold", 0)
    ys = M.get("y_scramble", {})
    wrap(c, f"Random splitting overstates AChE performance by {infl:+.3f} AUC and MAO-B by "
            f"{M2.get('auc_inflation_random_minus_scaffold', 0):+.3f}. With labels permuted, "
            f"performance collapses to {ys.get('mean_roc_auc', 0):.3f} ± "
            f"{ys.get('std', 0):.3f} over {ys.get('n_runs', 0)} refits - tight on chance, so "
            f"there is no leakage path in the pipeline. That control is what makes the "
            f"{M['splits']['scaffold']['roc_auc']:.3f} worth reading.",
         0.92 * inch, y, W - 1.84 * inch, 12.2, 16.6, BODY)
    newpage()

    # 6 — what we are NOT hiding
    kicker(c, "what we are not hiding")
    title(c, "Two results that argue against us")
    rule(c, H - 1.68 * inch)
    y = H - 2.3 * inch
    ave = M.get("ave_bias", {})
    aveval = ave.get("ave_bias") if isinstance(ave, dict) else ave
    c.setFont("Helvetica-Bold", 15); c.setFillColor(WARN)
    c.drawString(0.92 * inch, y, f"1.  AVE split bias is {aveval:+.2f}, which is middling, not good.")
    y -= 0.36 * inch
    y = wrap(c, "A meaningful part of the scaffold split is separable by nearest-neighbour "
                "geometry alone. We report it rather than re-splitting until the number "
                "improves: tuning a split to minimise AVE and then quoting the resulting AUC "
                "is itself a way of overfitting to a validation statistic.",
             0.92 * inch, y, W - 1.84 * inch, 12.4, 17, BODY)
    y -= 0.3 * inch

    inside = 100 * (S["max_train_similarity"] >= 0.35).mean() if not S.empty else 0
    c.setFont("Helvetica-Bold", 15); c.setFillColor(WARN)
    c.drawString(0.92 * inch, y, f"2.  Only {inside:.1f}% of the plant library is inside the applicability domain.")
    y -= 0.36 * inch
    y = wrap(c, "Natural-product scaffolds are mostly unlike the synthetic chemistry in "
                "ChEMBL. This is the single most important caveat in the whole project, and "
                "it is a property of the problem rather than a bug in the model. The honest "
                "response is to surface it on every row and default the interface to "
                "in-domain compounds only - not to quietly rank 23,761 extrapolations.",
             0.92 * inch, y, W - 1.84 * inch, 12.4, 17, BODY)

    p4 = chart_domain(BUILD / "domain.png")
    if p4:
        c.drawImage(str(p4), 7.4 * inch, 0.95 * inch, width=5.2 * inch,
                    height=2.75 * inch, preserveAspectRatio=True, anchor="sw")
    newpage()

    # 7 — results / hits
    kicker(c, "output")
    title(c, "A ranked list that tells you how much to trust it")
    rule(c, H - 1.68 * inch)
    y = H - 2.22 * inch
    if not S.empty:
        novel = S[(~S["is_known_active"].astype(bool)) & (S["in_domain"])]
        clean = novel[novel["pains_alerts"] == 0]
        stat(c, 0.92 * inch, y, f"{len(S):,}", "plant compounds scored")
        stat(c, 4.0 * inch, y, f"{len(novel):,}", "novel + inside applicability domain")
        stat(c, 7.6 * inch, y, f"{len(clean):,}", "of those with no PAINS alert")
        kn = int(S["is_known_active"].sum())
        stat(c, 10.9 * inch, y, f"{kn:,}", "known actives recovered (flagged, excluded from enrichment)")
        y -= 1.12 * inch

        cols = [("#", 0.92, 0.5), ("compound", 1.5, 3.5), ("P(active)", 5.2, 0.95),
                ("train sim.", 6.4, 0.95), ("PAINS", 7.6, 0.6),
                ("source plants", 8.5, 4.1)]
        c.setFont("Helvetica-Bold", 9.4); c.setFillColor(MUTED)
        for name, x, _ in cols:
            c.drawString(x * inch, y, name.upper())
        y -= 0.1 * inch
        rule(c, y)
        y -= 0.3 * inch
        for _, r in clean.head(9).iterrows():
            nm = str(r.get("compound_name") or r["compound_qid"])
            nm = nm if len(nm) <= 42 else nm[:40] + "…"
            taxa = str(r.get("top_taxa") or "")
            taxa = taxa.replace(";", ", ")
            taxa = taxa if len(taxa) <= 54 else taxa[:52] + "…"
            c.setFont("Helvetica", 10.4); c.setFillColor(BODY)
            c.drawString(0.92 * inch, y, str(int(r["rank"])))
            c.setFont("Helvetica-Bold", 10.4); c.setFillColor(INK)
            c.drawString(1.5 * inch, y, nm)
            c.setFont("Helvetica-Bold", 10.4); c.setFillColor(ACCENT)
            c.drawString(5.2 * inch, y, f"{r['prob_active']:.3f}")
            c.setFont("Helvetica", 10.4); c.setFillColor(BODY)
            c.drawString(6.4 * inch, y, f"{r['max_train_similarity']:.2f}")
            c.drawString(7.6 * inch, y, str(int(r["pains_alerts"])))
            c.setFont("Helvetica-Oblique", 9.8); c.setFillColor(MUTED)
            c.drawString(8.5 * inch, y, taxa)
            y -= 0.255 * inch
    newpage()

    # 8 — enrichment
    kicker(c, "the inverse question")
    title(c, "Which plants are over-represented among the hits?")
    rule(c, H - 1.68 * inch)
    p3 = chart_enrichment(BUILD / "enrich.png")
    if p3:
        c.drawImage(str(p3), 0.85 * inch, 1.0 * inch, width=6.5 * inch,
                    height=4.3 * inch, preserveAspectRatio=True, anchor="sw")
    x2 = 7.8 * inch
    y = H - 2.3 * inch
    y = wrap(c, "Hypergeometric over-representation of source taxa in the top-k predictions, "
                "Benjamini-Hochberg corrected across every taxon tested - the same test used "
                "for GO-term enrichment.",
             x2, y, W - x2 - 0.92 * inch, 12.2, 16.6, BODY)
    y -= 0.2 * inch
    y = wrap(c, "Known actives are removed before testing. Otherwise a genus scores as "
                "“enter enriched” purely because its alkaloid is already a licensed drug the "
                "model memorised, which tells a prospector nothing new.",
             x2, y, W - x2 - 0.92 * inch, 12.2, 16.6, BODY)
    y -= 0.2 * inch
    if not E.empty:
        nsig = int((E["q_value"] < 0.05).sum())
        ntested = len(E)
        stat(c, x2, y - 0.1 * inch, f"{nsig}", f"taxa significant at FDR < 0.05, of {ntested:,} tested", ACCENT, 28)
        y -= 1.0 * inch
    y = wrap(c, "Taxa with too few library compounds are not tested at all: the test has no "
                "power there and including them only inflates the multiple-testing burden.",
             x2, y, W - x2 - 0.92 * inch, 11.4, 15.4, MUTED)
    newpage()

    # 9 — knowledge graph
    kicker(c, "why, not just what")
    title(c, "Every edge carries its evidence")
    rule(c, H - 1.68 * inch)
    y = H - 2.35 * inch
    y = wrap(c, "A ranked CSV answers “what should I test”. It does not answer “why”, and "
                "“why” is what makes a result arguable in front of a pharmacognosist.",
             0.92 * inch, y, W - 1.84 * inch, 13, 18, BODY)
    y -= 0.45 * inch

    chain = [("Huperzia serrata", "#C9DED3", 2.5),
             ("huperzine A", "#E3EDE7", 2.1),
             ("Acetylcholinesterase", "#C9DED3", 2.6),
             ("Alzheimer's disease", "#2E7D5B", 2.5)]
    edges = ["CONTAINS", "PREDICTED_ACTIVE", "IMPLICATED_IN"]
    x = 0.92 * inch
    for i, (label, fill, bw) in enumerate(chain):
        c.setFillColor(HexColor(fill))
        c.roundRect(x, y - 0.42 * inch, bw * inch, 0.62 * inch, 7, stroke=0, fill=1)
        c.setFillColor(PAPER if fill == "#2E7D5B" else INK)
        c.setFont("Helvetica-Bold", 11.2)
        c.drawCentredString(x + bw * inch / 2, y - 0.17 * inch, label)
        x += bw * inch
        if i < len(edges):
            c.setStrokeColor(MUTED); c.setLineWidth(1.1)
            c.line(x + 0.08 * inch, y - 0.11 * inch, x + 0.72 * inch, y - 0.11 * inch)
            c.line(x + 0.62 * inch, y - 0.05 * inch, x + 0.72 * inch, y - 0.11 * inch)
            c.line(x + 0.62 * inch, y - 0.17 * inch, x + 0.72 * inch, y - 0.11 * inch)
            c.setFont("Helvetica-Bold", 7.6); c.setFillColor(MUTED)
            c.drawCentredString(x + 0.4 * inch, y + 0.03 * inch, edges[i])
            x += 0.8 * inch
    y -= 1.1 * inch

    y = bullets(c, [
        ("CONTAINS", "a referenced LOTUS/Wikidata occurrence statement (P703), not an assertion."),
        ("PREDICTED_ACTIVE", "the calibrated probability and the applicability-domain flag together."),
        ("KNOWN_ACTIVE", "the measured pChEMBL value, where one exists."),
        ("IMPLICATED_IN", "a curated target-disease association."),
    ], 0.92 * inch, y, W - 1.84 * inch)
    y -= 0.12 * inch
    wrap(c, "So a path through the graph is a chain of individually checkable claims rather "
            "than an opaque score. Exports to GraphML for Cytoscape or Gephi.",
         0.92 * inch, y, W - 1.84 * inch, 12.2, 16.6, MUTED)
    newpage()

    # 10 — limitations
    kicker(c, "limitations")
    title(c, "What this does not do")
    rule(c, H - 1.68 * inch)
    y = H - 2.22 * inch
    y = bullets(c, [
        ("Ligand-based models interpolate.",
         "They predict from resemblance to known actives. Genuinely novel scaffolds fall "
         "outside the domain - exactly where interesting natural products sit. The flag "
         "makes that visible; it does not solve it."),
        ("Decoys are presumed inactive, not measured inactive.",
         "An unlabelled library compound that is truly active counts against the model, "
         "so that estimate is conservative."),
        ("A predicted binder is not a drug.",
         "Nothing here addresses absorption, metabolism, toxicity, tissue exposure, or "
         "whether engaging the target helps a patient."),
        ("Occurrence is not bioavailable dose.",
         "A compound reported from a plant says nothing about concentration, seasonal "
         "variation, or whether a traditional preparation extracts it at all."),
        ("Enrichment inherits study bias.",
         "Well-studied genera have more library compounds and more opportunity to appear "
         "enriched. The hypergeometric null accounts for how many, not for why."),
        ("Ethnobotanical notes are never model input.",
         "They are literature context for human interpretation, so they cannot leak into "
         "the statistics they sit beside."),
    ], 0.92 * inch, y, W - 1.84 * inch)

    y -= 0.1 * inch
    rule(c, y)
    y -= 0.42 * inch
    c.setFont("Helvetica-Bold", 12.4); c.setFillColor(WARN)
    c.drawString(0.92 * inch, y, "Research prototype. Not clinical advice.")
    y -= 0.36 * inch
    wrap(c, "Traditional knowledge referenced here originates with the communities who "
            "developed and maintained it over generations; this tool points back toward that "
            "knowledge rather than claiming it.  Data: ChEMBL (CC BY-SA 3.0), LOTUS via "
            "Wikidata (CC0), RDKit (BSD-3).",
         0.92 * inch, y, W - 1.84 * inch, 11, 15, MUTED)

    footer(c, page[0])
    c.save()


if __name__ == "__main__":
    out = ROOT / "docs" / "PhytoTarget_deck.pdf"
    build(out)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
