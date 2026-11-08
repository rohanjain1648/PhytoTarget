"""Cheminformatics primitives: standardisation, fingerprints, scaffolds, structural alerts.

Every molecule entering PhytoTarget passes through `standardize()` exactly once, so
that a compound arriving from ChEMBL and the same compound arriving from Wikidata
collapse to the same InChIKey. Without this the dedup and the "is it already known"
flag are both wrong.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


@lru_cache(maxsize=1)
def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    for cat in (
        FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalogParams.FilterCatalogs.PAINS_C,
    ):
        params.AddCatalog(cat)
    return FilterCatalog(params)


def standardize(smiles: str) -> str | None:
    """Canonical SMILES of the largest neutral fragment, or None if unparseable.

    Salts and counter-ions are stripped (largest fragment), charges are neutralised
    where chemically sensible. Stereochemistry is preserved.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _LARGEST.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumHeavyAtoms() < 5:
        return None  # fragments / solvents are not screenable leads
    return Chem.MolToSmiles(mol, canonical=True)


def to_mol(smiles: str):
    return Chem.MolFromSmiles(smiles) if smiles else None


def inchikey(smiles: str) -> str:
    mol = to_mol(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES. Empty string for acyclic molecules."""
    mol = to_mol(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True) if scaf.GetNumAtoms() else ""
    except Exception:
        return ""


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    mol = to_mol(smiles)
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    fp = gen.GetFingerprint(mol)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "RingCount", "NumHeteroatoms", "QED",
]


def descriptors(smiles: str) -> np.ndarray | None:
    """Twelve interpretable physicochemical descriptors.

    These sit alongside the fingerprint so the model can learn gross property
    constraints (a 900 Da glycoside is not getting into the CNS) that bit vectors
    express only awkwardly.
    """
    mol = to_mol(smiles)
    if mol is None:
        return None
    try:
        vals = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol), float(mol.GetNumHeavyAtoms()),
            Descriptors.RingCount(mol), Descriptors.NumHeteroatoms(mol), QED.qed(mol),
        ]
    except Exception:
        return None
    arr = np.asarray(vals, dtype=np.float32)
    return arr if np.isfinite(arr).all() else None


def featurize(smiles_list, radius=2, n_bits=2048, use_descriptors=True):
    """Return (X, keep_mask). Rows that fail featurisation are dropped, not imputed."""
    feats, keep = [], []
    for smi in smiles_list:
        fp = morgan_fp(smi, radius, n_bits)
        if fp is None:
            keep.append(False)
            continue
        if use_descriptors:
            d = descriptors(smi)
            if d is None:
                keep.append(False)
                continue
            feats.append(np.concatenate([fp.astype(np.float32), d]))
        else:
            feats.append(fp.astype(np.float32))
        keep.append(True)
    X = np.vstack(feats) if feats else np.empty((0, n_bits + (12 if use_descriptors else 0)), np.float32)
    return X, np.asarray(keep, dtype=bool)


def feature_names(n_bits=2048, use_descriptors=True):
    names = [f"ecfp_{i}" for i in range(n_bits)]
    return names + DESCRIPTOR_NAMES if use_descriptors else names


def pains_alerts(smiles: str) -> int:
    """Count of PAINS (pan-assay interference) substructure matches.

    A non-zero count does not mean the compound is inactive. It means the
    *assay readout* is likely to be unreliable, so the hit should be deprioritised
    for follow-up rather than celebrated.
    """
    mol = to_mol(smiles)
    if mol is None:
        return 0
    try:
        return len(_pains_catalog().GetMatches(mol))
    except Exception:
        return 0


def tanimoto_max(query_fps: np.ndarray, ref_fps: np.ndarray, block: int = 512) -> np.ndarray:
    """Max Tanimoto similarity of each query row against all reference rows.

    Binary fingerprints only. Blocked so a 100k x 10k comparison does not
    allocate a 4 GB intermediate.
    """
    if query_fps.size == 0 or ref_fps.size == 0:
        return np.zeros(len(query_fps), dtype=np.float32)
    q = (query_fps > 0).astype(np.float32)
    r = (ref_fps > 0).astype(np.float32)
    r_sum = r.sum(axis=1)
    out = np.zeros(len(q), dtype=np.float32)
    for start in range(0, len(q), block):
        chunk = q[start:start + block]
        inter = chunk @ r.T
        union = chunk.sum(axis=1)[:, None] + r_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[start:start + block] = sim.max(axis=1)
    return out
