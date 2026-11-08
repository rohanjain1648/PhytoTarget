"""Tests for the invariants the project's claims rest on.

These are deliberately not coverage-chasing. Each one checks a property that, if it
broke silently, would make a headline number wrong:

* scaffold splitting must not leak a scaffold across the train/test boundary -
  otherwise the "honest" metric is just the optimistic one
* standardisation must collapse salt and charge variants to one key - otherwise the
  same molecule is counted twice and "deduplicated" libraries are not
* enrichment factor must respect its own mathematical ceiling - the bug that makes
  a saturated EF look like a triumph
* the hypergeometric test must not call a taxon enriched when the overlap is exactly
  what chance predicts

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.chem import inchikey, murcko_scaffold, standardize
from src.metrics import bedroc, enrichment_factor
from src.split import scaffold_split


# ------------------------------------------------------------- standardisation
def test_salt_and_parent_collapse_to_one_key():
    """A hydrochloride salt and its free base are the same compound for screening."""
    free_base = standardize("CN1CCC[C@H]1c1cccnc1")          # nicotine
    salt = standardize("CN1CCC[C@H]1c1cccnc1.Cl")            # nicotine HCl
    assert free_base and salt
    assert inchikey(free_base) == inchikey(salt)


def test_charged_and_neutral_collapse():
    # Benzoate rather than acetate: standardize() deliberately drops anything under
    # five heavy atoms, because solvents and fragments are not screenable leads.
    acid, anion = standardize("c1ccccc1C(=O)O"), standardize("c1ccccc1C(=O)[O-]")
    assert acid and anion
    assert inchikey(acid) == inchikey(anion)


def test_fragments_and_solvents_are_rejected():
    assert standardize("CC(=O)O") is None, "acetic acid is a solvent, not a lead"
    assert standardize("CO") is None


def test_unparseable_smiles_returns_none_rather_than_raising():
    # The library has 100k+ rows from a public wiki; malformed SMILES are expected
    # and must be skipped, not crash a 20-minute screen.
    assert standardize("not_a_molecule[[[") is None
    assert standardize("") is None
    assert standardize(None) is None


def test_stereochemistry_is_preserved():
    a, b = standardize("C[C@H](N)C(=O)O"), standardize("C[C@@H](N)C(=O)O")
    assert a != b, "enantiomers must not be merged"


# ----------------------------------------------------------------- scaffolding
def test_scaffold_split_leaks_no_scaffold():
    """The central methodological claim: no scaffold appears on both sides."""
    rng = np.random.default_rng(0)
    scaffolds = np.array([f"S{i % 37}" for i in range(400)])
    rng.shuffle(scaffolds)

    train, test = scaffold_split(scaffolds, test_fraction=0.2, seed=7)

    assert len(set(train) & set(test)) == 0, "an index landed in both halves"
    assert len(train) + len(test) == len(scaffolds), "indices were lost"
    overlap = set(scaffolds[train]) & set(scaffolds[test])
    assert not overlap, f"scaffolds leaked across the split: {sorted(overlap)[:5]}"


def test_scaffold_split_test_fraction_is_approximately_honoured():
    scaffolds = np.array([f"S{i}" for i in range(500)])   # all distinct
    train, test = scaffold_split(scaffolds, test_fraction=0.2, seed=1)
    assert 0.12 <= len(test) / 500 <= 0.30


def test_murcko_scaffold_groups_analogues():
    """Two benzodiazepine-ish analogues differing only in a substituent share a scaffold."""
    a = murcko_scaffold("c1ccc2c(c1)NC(=O)CN=C2c1ccccc1")
    b = murcko_scaffold("c1ccc2c(c1)NC(=O)CN=C2c1ccc(Cl)cc1")
    assert a and b
    # Same ring system; the chloro substituent is stripped by Murcko reduction.
    assert a == b


# --------------------------------------------------------------------- metrics
def test_enrichment_factor_respects_its_ceiling():
    """EF cannot exceed 1/active_fraction - the bug that flatters balanced datasets."""
    y = np.array([1] * 64 + [0] * 36)            # 64% actives, like the ChEMBL sets
    perfect = np.concatenate([np.ones(64), np.zeros(36)])   # flawless ranking
    ef = enrichment_factor(y, perfect, 0.01)
    ceiling = 1.0 / y.mean()
    assert ef <= ceiling + 1e-9
    assert ef == pytest.approx(ceiling, rel=0.05), (
        "a perfect ranking on a 64%-active set should sit at the ceiling (~1.56), "
        "which is why EF is reported against the real library instead"
    )


def test_enrichment_factor_is_one_for_random_ranking():
    rng = np.random.default_rng(3)
    y = np.array([1] * 50 + [0] * 4950)
    scores = rng.random(len(y))
    efs = [enrichment_factor(y, rng.random(len(y)), 0.1) for _ in range(40)]
    assert 0.5 < float(np.mean(efs)) < 1.6, "random ranking should average ~1x"


def test_bedroc_spans_zero_to_one():
    y = np.array([1] * 20 + [0] * 980)
    best = np.concatenate([np.ones(20), np.zeros(980)])
    worst = np.concatenate([np.zeros(20), np.ones(980)])
    assert bedroc(y, best, 20.0) > 0.95
    assert bedroc(y, worst, 20.0) < 0.05


# ----------------------------------------------------------------- enrichment
def test_hypergeometric_does_not_flag_chance_overlap():
    """A taxon whose overlap equals its expectation must not come out significant."""
    from scipy.stats import hypergeom

    population, taxon_size, draws = 20000, 200, 500
    expected = draws * taxon_size / population          # = 5
    p = hypergeom.sf(int(expected) - 1, population, taxon_size, draws)
    assert p > 0.2, "chance-level overlap should not be significant"

    # Strong enrichment: 40 of the taxon's 200 compounds in a 500-compound slice.
    p_strong = hypergeom.sf(40 - 1, population, taxon_size, draws)
    assert p_strong < 1e-20


def test_bh_correction_is_more_conservative_than_raw_p():
    from src.metrics import benjamini_hochberg

    pvals = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5, 0.9]
    q = benjamini_hochberg(pvals)
    assert all(qi >= pi - 1e-12 for qi, pi in zip(q, pvals)), (
        "a BH q-value is never smaller than its raw p-value"
    )
    assert all(q[i] <= q[i + 1] + 1e-12 for i in range(len(q) - 1)), (
        "q-values must stay monotone in p - the step people forget"
    )
    assert benjamini_hochberg([]).size == 0
