"""Tests for samap.io.homology — gnnm_from_pairs and homology_from_eggnog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from samap.io.homology import _parse_og_at_taxon, gnnm_from_pairs, homology_from_eggnog

EXAMPLE = Path(__file__).parent.parent.parent / "example_data"


# ---------------------------------------------------------------------------
# gnnm_from_pairs
# ---------------------------------------------------------------------------


def test_gnnm_from_pairs_with_ids() -> None:
    pairs = [("A1", "B1"), ("A2", "B1"), ("A2", "B2")]
    gnnm, gns, gd = gnnm_from_pairs(
        pairs, ids={"aa": ["A1", "A2"], "bb": ["B1", "B2"]}
    )
    assert isinstance(gnnm, sp.csr_matrix)
    assert gnnm.shape == (4, 4)
    assert sorted(gd) == ["aa", "bb"]
    assert set(gns) == {"aa_A1", "aa_A2", "bb_B1", "bb_B2"}
    # symmetric → 6 edges
    assert gnnm.nnz == 6
    # all weights 1.0, on-diagonal blocks zero
    idx = pd.Index(gns)
    a = gnnm.toarray()
    assert a[idx.get_loc("aa_A1"), idx.get_loc("bb_B1")] == pytest.approx(1.0)
    assert a[idx.get_loc("aa_A1"), idx.get_loc("aa_A2")] == 0.0


def test_gnnm_from_pairs_prefixed_infer_species() -> None:
    pairs = [("hu_SOX2", "mm_Sox2"), ("hu_TP53", "mm_Trp53")]
    gnnm, _gns, gd = gnnm_from_pairs(pairs)
    assert sorted(gd) == ["hu", "mm"]
    assert gnnm.nnz == 4


def test_gnnm_from_pairs_weights_clip() -> None:
    pairs = [("x_A", "y_B"), ("x_A", "y_B")]  # duplicate → would sum to 1.6
    gnnm, _gns, _ = gnnm_from_pairs(pairs, weights=[0.8, 0.8])
    assert gnnm.toarray().max() == pytest.approx(1.0)


def test_gnnm_from_pairs_errors() -> None:
    with pytest.raises(ValueError, match="empty"):
        gnnm_from_pairs([])
    with pytest.raises(ValueError, match="exactly two"):
        gnnm_from_pairs([("a", "b", "c")])
    with pytest.raises(ValueError, match="not found in any"):
        gnnm_from_pairs([("A", "Z")], ids={"x": ["A"], "y": ["B"]})


# ---------------------------------------------------------------------------
# eggnog
# ---------------------------------------------------------------------------


def test_parse_og_at_taxon() -> None:
    cell = "38ERC@33154,3NUD8@4751,KOG2877@2759"
    assert _parse_og_at_taxon(cell, "2759") == "KOG2877"
    assert _parse_og_at_taxon(cell, "33154") == "38ERC"
    assert _parse_og_at_taxon(cell, "9999") is None
    assert _parse_og_at_taxon(float("nan"), "2759") is None


def test_homology_from_eggnog_synthetic() -> None:
    # Two species, 3 genes each; one shared OG, one species-specific.
    a = pd.DataFrame(
        {"eggNOG_OGs": ["OG1@2759", "OG2@2759", "OG3@2759,X@33154"]},
        index=["gA1", "gA2", "gA3"],
    )
    b = pd.DataFrame(
        {"eggNOG_OGs": ["OG1@2759", "OG1@2759", "OG4@2759"]},
        index=["gB1", "gB2", "gB3"],
    )
    gnnm, gns, gd = homology_from_eggnog({"aa": a, "bb": b})
    # OG1 has aa:[gA1], bb:[gB1, gB2] → 2 cross-species edges → 4 nnz symmetric
    assert gnnm.nnz == 4
    assert set(gns) == {"aa_gA1", "bb_gB1", "bb_gB2"}
    assert sorted(gd) == ["aa", "bb"]


def test_homology_from_eggnog_max_og_size() -> None:
    a = pd.DataFrame({"eggNOG_OGs": ["BIG@2759"] * 10}, index=[f"a{i}" for i in range(10)])
    b = pd.DataFrame({"eggNOG_OGs": ["BIG@2759"] * 10}, index=[f"b{i}" for i in range(10)])
    with pytest.raises(ValueError, match="No cross-species"):
        homology_from_eggnog({"aa": a, "bb": b}, max_og_size=5)
    # but no cap works
    gnnm, _, _ = homology_from_eggnog({"aa": a, "bb": b}, max_og_size=None)
    assert gnnm.nnz == 200  # 10×10 pairs × 2 (symmetric)


def test_homology_from_eggnog_missing_column() -> None:
    a = pd.DataFrame({"wrong": ["X@2759"]}, index=["g1"])
    with pytest.raises(KeyError, match="eggNOG_OGs"):
        homology_from_eggnog({"aa": a, "bb": a})


# ---------------------------------------------------------------------------
# Live: bundled eggnog TSVs vs the BLAST graph
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not (EXAMPLE / "eggnog").exists(), reason="example_data/eggnog missing")
def test_eggnog_bundled_three_species() -> None:
    tsvs = {
        "hy": EXAMPLE / "eggnog" / "hydra.tsv",
        "pl": EXAMPLE / "eggnog" / "planarian.tsv",
        "sc": EXAMPLE / "eggnog" / "schistosome.tsv",
    }
    gnnm, _gns, gd = homology_from_eggnog(tsvs, taxon=2759)
    assert set(gd) == {"hy", "pl", "sc"}
    # Expect tens of thousands of edges across 3 species.
    assert gnnm.nnz > 50_000
    # All edges weight 1.0 and matrix symmetric.
    assert np.allclose(gnnm.data, 1.0)
    diff = (gnnm - gnnm.T).tocoo()
    assert abs(diff.data).max() < 1e-6 if diff.nnz else True
    # Per-species gene counts roughly match TSV row counts (≤, since
    # only genes with an OG@2759 are included).
    for sid in gd:
        assert gd[sid].size > 1000
