# 🌿 PhytoTarget

> **EthnoHack 2026 — Track 2: Natural Products & Drug Discovery**

<p align="center">
  <img src="docs/cover_image.jpg" alt="PhytoTarget Cover" width="100%" />
</p>

**Bias-corrected target prediction for natural products, with source-organism enrichment.**

Traditional medicine encodes centuries of pharmacological observation, but it names
*plants*, not *molecules* or *protein targets*. PhytoTarget closes that gap in the
direction a field researcher actually needs: it screens the plant natural-product
universe against a disease target, then asks **which plants are statistically
over-represented among the predicted actives** — turning a ranked compound list into
a collection plan.

The hard part is not building a classifier. It is building one whose numbers are not
quietly inflated. Most of the engineering here is about that.

---

## The problem with most virtual screening results

A ligand-based activity model is easy to make look excellent and hard to make
trustworthy. Four failure modes produce impressive-looking numbers that mean nothing,
and this pipeline instruments all four rather than hoping they are absent.

| Failure mode | What it does to your metrics | What PhytoTarget does |
|---|---|---|
| **Random train/test split** | Analogues of the same scaffold land in both halves. The model is graded on chemistry it has effectively memorised. | Headline metric is a **Bemis–Murcko scaffold split**. The random-split number is computed too and reported *beside* it, so the inflation is visible instead of hidden. |
| **Hidden label leakage** | Any pipeline bug that lets labels reach features inflates everything downstream. | **y-scrambling**: 10 refits on permuted labels. Performance must collapse to ≈0.50. |
| **Artificial decoy geometry** | Actives and inactives separable by nearest-neighbour distance alone — the model learns the split, not the pharmacology. | **AVE bias** (Wallach & Heifets, 2018) reported for every target. |
| **Publication-biased prevalence** | ChEMBL is 50–70% actives because people publish what works. EF@1% is capped at 1/active_fraction ≈ **1.6**, so a saturated EF looks like a ceiling-breaking result. | **Decoy-augmented evaluation**: held-out scaffold actives hidden among the real 23k-compound natural-product library, giving ~1% prevalence with the active count still in the hundreds. |

Two further guards apply at screening time rather than training time:

- **Applicability domain.** A probability for a molecule unlike anything in training is
  an extrapolation. Max Tanimoto to the training set is computed for every library
  compound and anything below threshold is flagged, not silently ranked. This matters
  enormously for natural products, whose scaffolds are frequently nothing like the
  synthetic chemistry in ChEMBL.
- **Rediscovery flagging.** Plant compounds already among the target's ChEMBL actives
  will of course score highly. They are kept — they are the retrospective validation —
  but flagged, and **excluded before enrichment testing**, so a genus does not score as
  "enriched" purely because its alkaloid is already a licensed drug the model memorised.

---

## Results

### Model honesty (the numbers that decide whether to read the rest)

| | Acetylcholinesterase (CHEMBL220) | Monoamine oxidase B (CHEMBL2039) |
|---|---|---|
| Indication | Alzheimer's disease | Parkinson's disease / depression |
| Molecules after curation | 3,439 (2,216 act / 1,223 inact) | 2,818 (1,568 act / 1,250 inact) |
| Distinct scaffolds | 1,799 | 1,037 |
| **ROC-AUC, scaffold split** (headline) | **0.943** | **0.922** |
| ROC-AUC, random split (optimistic) | 0.974 | 0.973 |
| **AUC inflation from random splitting** | **+0.031** | **+0.050** |
| PR-AUC, scaffold split | 0.966 | 0.877 |
| **y-scrambled ROC-AUC** (must be ≈0.50) | **0.501 ± 0.022** | **0.505 ± 0.022** |
| AVE split bias | +0.502 | +0.376 |
| Brier score / ECE (isotonic) | 0.109 / 0.073 | 0.130 / 0.112 |
| EF@1%, raw test set (*uninformative*) | 1.56 of a 1.56 ceiling | 2.21 of a 2.21 ceiling |
| **EF@1%, decoy-augmented** | **54.9** of a 55 ceiling | **66.7** of a 94 ceiling |

On enrichment factor specifically: the raw-test-set EF of 1.56 is not a weak result, it
is a **meaningless** one — it is the arithmetic maximum for a 64%-active set. Hiding
held-out actives in the real library instead puts prevalence at 1–2% and gives EF@1% of
54.9 and 66.7. The AChE figure still sits at its ceiling, so the top 1% of that ranked
library is essentially all held-out actives; the MAO-B figure, at 71% of its ceiling, is
the more informative of the two.

The y-scrambling result is the single most important row: with labels permuted,
performance sits on chance with a tight standard deviation. There is no leakage path
in the pipeline. The scaffold-vs-random gap is modest but real, and reporting it is the
point — a submission quoting only the 0.974 would be overstating its model by a margin
that matters.

**The AVE bias values are honestly middling, not good.** +0.50 on AChE says a
meaningful part of the scaffold split *is* separable by nearest-neighbour geometry.
We report it rather than tuning the split until the number looks better, because
re-splitting to minimise AVE and then quoting the resulting AUC is itself a way of
overfitting to a validation statistic.

### Library

Built live from **LOTUS via Wikidata** (SPARQL): **102,470** referenced
organism–compound occurrence pairs covering **43,093** compounds across **21,377**
source taxa, collapsing to **23,765** unique standardised structures after InChIKey
deduplication.

**An important scope correction.** The intent was a plant-only library, but the
transitive plant filter (`wdt:P171* wd:Q25314`) times the WDQS endpoint out, so the
library is the full set of LOTUS *source organisms*. It is predominantly plants, but
it also contains fungi, algae, cyanobacteria and bacteria. This is visible in the
results — *Lyngbya majuscula* is a cyanobacterium and *Cladosporium* a fungus — and it
is a real limitation of this build rather than a presentation choice. It cuts both
ways: it widens the library beyond the ethnobotanical framing, and it is also how the
strongest AChE hits were found (see below).

### Screening output

| | AChE (CHEMBL220) | MAO-B (CHEMBL2039) |
|---|---|---|
| Compounds scored | 23,765 | 23,765 |
| **Inside applicability domain** | **15.7%** | **7.8%** |
| Novel + in-domain | 3,726 | 1,856 |
| ...and free of PAINS alerts | 3,265 | 1,532 |
| Known actives recovered and flagged | 8 | 0 |
| Taxa tested / significant at FDR < 0.05 | 12,203 / **62** | 12,203 / **46** |

The applicability-domain figure is the most important number in this table and the
least flattering. **Over 84% of the natural-product library is too dissimilar from
ChEMBL's AChE chemistry for a probability to mean anything**, so the dashboard
defaults to hiding those rows rather than ranking 23,765 extrapolations.

### Retrospective recovery: what the model does with known plant-derived leads

These compounds are in the library and their ranks are honest — the model was not told
which rows they were.

| Compound | Source organism | Rank of 23,765 | Percentile | P(active) |
|---|---|---|---|---|
| **galantamine** (licensed AChE drug) | *Galanthus nivalis* | **626** | top 2.6% | 0.536 |
| **harmine** (MAO inhibitor) | *Peganum harmala* | **927** | top 3.9% | 0.482 |
| huperzine A | *Huperzia serrata* | 3,730 | top 15.7% | 0.403 |
| harmaline | *Peganum harmala* | 5,358 | top 22.6% | 0.281 |
| physostigmine | *Physostigma venenosum* | 8,287 | top 34.9% | 0.299 |

Two of five land in the top 4%. **Three do not, and the physostigmine result is a
genuine, explainable failure worth stating plainly**: physostigmine is a carbamate
that inhibits AChE by covalently carbamylating the catalytic serine. A 2D
fingerprint model has no representation of a covalent mechanism — it can only ask
whether a structure resembles known binders — so a mechanistically distinct inhibitor
is exactly the case it should be expected to miss. Reporting only galantamine here
would have been straightforward and dishonest.

Note also that galantamine is among the 8 flagged known actives, so its rank is partly
**rediscovery, not discovery**. That is precisely why known actives are stripped out
before organism enrichment is computed.

### Top novel hits (AChE, in-domain, no PAINS)

The highest-scoring novel compounds are **terreulactone D** (P = 0.843) and
**territrem E** (P = 0.818), both meroterpenoids from *Aspergillus terreus*. The
territrem family is independently documented in the literature as a class of
exceptionally potent AChE inhibitors. The model surfaced them without them appearing
among its training actives — a checkable result rather than a claim, and the clearest
single piece of evidence that the ranking carries real signal.

It is also a direct consequence of the scope slip described above: had the plant-only
filter worked, these fungal compounds would not have been in the library at all.

---

## How it works

```
ChEMBL bioactivity ──┐
                     ├─► curate ─► featurise ─► scaffold-split train ─► calibrate ─► model
LOTUS / Wikidata ────┘                                                                 │
   (plant → compound)                                                                  ▼
                     ┌──────────────────── score 23,761 NP compounds ──────────────────┘
                     │
                     ├─► applicability domain + PAINS + known-active flags
                     ├─► hypergeometric organism enrichment (BH-FDR)
                     └─► plant → compound → target → disease knowledge graph
```

**1. Labels.** Bioactivity is pulled from the ChEMBL API with
`standard_type` in {IC50, Ki, Kd, EC50} and assay confidence ≥ 8. Multiple measurements
per molecule are aggregated by **median pChEMBL, not maximum** — taking the max
cherry-picks the most optimistic outlier from every dose–response series ever run.
Molecules falling between 10 µM and 316 nM are **dropped rather than relabelled
inactive**: that band is where assay-to-assay noise makes the class genuinely
ambiguous, and forcing a label there teaches the model to reproduce noise.

**2. Features.** ECFP4 (Morgan radius 2, 2048 bits) plus twelve interpretable
physicochemical descriptors. Every structure — from ChEMBL and from Wikidata alike —
passes through **one** standardisation routine (largest organic fragment, neutralised,
canonical InChIKey) so the same compound arriving from two sources collapses to one row
instead of appearing as two.

**3. Model.** LightGBM with isotonic calibration. Calibration is not cosmetic here:
the whole output is a ranked list a human will threshold, so a compound shown at
P = 0.9 needs to be active about 90% of the time.

**4. Organism enrichment.** For each genus and species: given a library of *M*
compounds of which *n* come from that taxon, and a top-*k* slice of novel predictions,
is the observed overlap greater than chance? That is the hypergeometric tail
probability — the same test used for GO-term enrichment — with
**Benjamini–Hochberg FDR** across the thousands of taxa tested, and a minimum taxon
size below which the test has no power and is not run.

**5. Knowledge graph.** `plant --CONTAINS--> compound --PREDICTED_ACTIVE(p)--> target
--IMPLICATED_IN--> disease`, where every edge carries its evidence: the occurrence
statement behind CONTAINS, the calibrated probability *and* domain flag behind
PREDICTED_ACTIVE, the measured pChEMBL behind KNOWN_ACTIVE. A path through the graph is
a chain of individually checkable claims rather than an opaque score. Exports to
GraphML for Cytoscape/Gephi.

---

## Quickstart

```bash
pip install -r requirements.txt

python -m src.fetch_chembl      # bioactivity for each configured target
python -m src.fetch_lotus       # plant-compound library from Wikidata (~10 min)
python -m src.train             # scaffold split, y-scramble, AVE, calibration
python -m src.screen            # score the library, flag domain/PAINS/known
python -m src.enrich            # hypergeometric organism enrichment
python -m src.kg                # knowledge graph

streamlit run app/streamlit_app.py
```

Targets, thresholds and model settings live in `config.yaml`; adding a target is a
six-line block, no code change.

> **Note on a low-memory machine.** If training dies with an OpenBLAS allocation
> error, cap the thread pools — several parallel LightGBM fits each spawning one
> thread per core will exhaust a laptop:
> `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`

---

## Limitations

Stated plainly, because a screening tool that hides these is worse than no tool.

- **Ligand-based models interpolate.** They predict from resemblance to known actives.
  Genuinely novel scaffolds fall outside the applicability domain — which is exactly
  where interesting natural products often sit. The domain flag exists to keep that
  visible, not to solve it.
- **Decoys are *presumed* inactive, not measured inactive.** In the decoy-augmented
  evaluation an unlabelled library compound that is truly active counts against the
  model, making that estimate conservative.
- **A predicted binder is not a drug.** Nothing here addresses absorption, metabolism,
  toxicity, tissue exposure, or whether engaging the target helps the patient.
- **Occurrence is not bioavailable dose.** A compound reported from a plant says nothing
  about the concentration present, seasonal and geographic variation, or whether a
  traditional preparation would extract it at all.
- **LOTUS records vary in taxonomic rigour**, and enrichment inherits any sampling bias
  in which plants have been phytochemically studied — well-studied genera have more
  compounds in the library and more opportunity to appear enriched. The hypergeometric
  null accounts for library size per taxon, but not for *why* that size differs.
- **Ethnobotanical notes in the UI are literature context for human interpretation
  only.** They are never model input, so they cannot leak into the statistics they are
  displayed beside.
- **This is a research prototype, not clinical advice.**

---

## Attribution

Traditional knowledge referenced here originates with the communities who developed
and maintained it over generations; this tool points back toward that knowledge rather
than claiming it.

- **ChEMBL** (EMBL-EBI) — bioactivity data, CC BY-SA 3.0
- **LOTUS** via **Wikidata** — compound–organism occurrences, CC0
- **RDKit** — cheminformatics, BSD-3

Methods: PAINS filters, Baell & Holloway (2010) · AVE bias, Wallach & Heifets (2018) ·
BEDROC, Truchon & Bayly (2007) · scaffolds, Bemis & Murcko (1996).

## License

MIT (code). Data retains its upstream licences as listed above.
