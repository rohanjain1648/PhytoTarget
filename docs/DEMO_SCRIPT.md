# PhytoTarget — 3-minute demo script

Timings assume a 3:00 cap. The structure front-loads the one thing most screening
demos skip: showing the validation *before* showing the hits. That ordering is the
argument, so don't reorder it.

**Setup before recording**

```bash
streamlit run app/streamlit_app.py
```

Have the browser already on the dashboard with **Acetylcholinesterase** selected, and
have `docs/PhytoTarget_deck.pdf` open in a second tab. Zoom the browser to ~110% so
table text is readable in a compressed recording.

---

## 0:00 – 0:25 · The gap

> "Traditional medicine encodes centuries of pharmacological observation. But it
> indexes that knowledge by *plant* — not by molecule, and not by protein target.
> Drug discovery works the other way round. The two bodies of knowledge don't share
> an index.
>
> PhytoTarget bridges them, and then asks the question nobody tools for: *which
> plants are statistically over-represented among the predicted actives?* That
> answer isn't a compound list. It's a collection plan."

*On screen: the dashboard header with the five metric tiles.*

---

## 0:25 – 1:15 · Why you should believe any of it (Validation tab)

Open **📊 Validation** immediately. This is the differentiator — spend real time here.

> "Before any hits, here's why the numbers mean something.
>
> **Left:** the same model scored two ways. A random train/test split gives 0.974 —
> but it lets analogues of the same scaffold sit in both halves, so the model is
> graded on chemistry it already memorised. A Bemis–Murcko scaffold split gives
> 0.943. That gap, +0.031, is exactly how much a random split would have overstated
> this model. Most submissions report the 0.974 and don't mention the other number
> exists.
>
> **Right:** y-scrambling. Ten refits with the labels shuffled. Performance collapses
> to 0.501 — dead on chance, tight standard deviation. That's the control proving
> there's no leakage path in the pipeline. Without it, a good AUC is just a claim."

Point at the AVE bias tile.

> "And here's a number that argues *against* us: AVE split bias is +0.50, which is
> middling. Part of the split really is separable by nearest-neighbour geometry. We
> report it rather than re-splitting until it looks better — because tuning a split
> to minimise AVE and then quoting the AUC is just overfitting a validation
> statistic."

---

## 1:15 – 1:50 · The honest ceiling (Ranked hits tab)

Switch to **🔬 Ranked hits**. Leave *"Applicability domain only"* checked.

> "Now the output. 23,765 plant compounds from LOTUS via Wikidata — a hundred
> thousand referenced plant–compound occurrence statements.
>
> Critically: **only about 16% of this library is inside the model's applicability
> domain.** Natural-product scaffolds mostly don't look like the synthetic chemistry
> in ChEMBL. That's the single biggest caveat in the project, so the interface
> defaults to hiding the extrapolations rather than ranking 23,000 of them and hoping
> nobody checks."

Uncheck **Hide known actives**, then point to a flagged row.

> "Known actives are flagged, not deleted — they're the retrospective check. But
> they're *excluded* before enrichment testing, so a plant can't score as interesting
> just because its alkaloid is already a licensed drug the model memorised."

Click a hit in the inspector panel on the right.

> "Every row carries its own trust signals: calibrated probability, similarity to the
> nearest training molecule, PAINS count — because the classic natural-product
> failure is a promiscuous polyphenol that 'hits' every assay it touches."

---

## 1:50 – 2:30 · The inverse question (Organism enrichment tab)

Switch to **🌱 Organism enrichment**.

> "And here's the part that's actually new. For every genus and species: given how
> many library compounds come from that taxon, is its overlap with the top predictions
> more than chance? That's a hypergeometric test — the same one used for GO-term
> enrichment — Benjamini–Hochberg corrected across every taxon tested.
>
> These genera are over-represented among *novel* predictions. The right-hand column
> is ethnobotanical context from the literature — and it is never model input, so it
> can't leak into the statistics it's sitting next to. It's there so a
> pharmacognosist can sanity-check the output against what the plant is actually
> traditionally used for."

---

## 2:30 – 3:00 · Why, not just what (Knowledge graph tab + close)

Switch to **🕸 Knowledge graph**, pick a taxon.

> "Finally — a ranked CSV answers *what to test*. It doesn't answer *why*. The graph
> does: plant, contains, compound, predicted active with its probability and domain
> flag, target, implicated in disease. Every edge carries its evidence, so a path is
> a chain of individually checkable claims rather than an opaque score.
>
> It's a research prototype, not clinical advice — a predicted binder isn't a drug.
> But everything here is reproducible from two open datasets with one command, and
> every number on every slide is read from the pipeline output at build time, so the
> deck can't drift from the code.
>
> That's PhytoTarget."

---

## Recording notes

- **Don't** narrate clicking ("now I'll click on…"). Say what the number means.
- The validation tab is the whole pitch. If you overrun, cut from the hits section,
  never from validation.
- Mention "23,765 compounds, 21,377 plant taxa" once — concrete figures land better
  than "a large library".
- If a judge asks the obvious hostile question — *"isn't finding galantamine just
  memorisation?"* — the answer is yes, and that's precisely why it's flagged
  `is_known_active` and stripped out before enrichment. Say that plainly; it's a
  strength, not a dodge.
