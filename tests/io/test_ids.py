"""Tests for samap.io.ids.detect_id_flavor."""

from __future__ import annotations

import pytest

from samap.io.ids import FLAVOR_PATTERNS, detect_id_flavor


@pytest.mark.parametrize(
    ("ids", "expected", "versioned"),
    [
        (["ENSG00000139618", "ENSG00000141510", "ENSG00000012048"], "ensembl_gene", False),
        (["ENSMUSG00000017146.5", "ENSMUSG00000059552.9"], "ensembl_gene", True),
        (["ENST00000380152", "ENST00000269305.9"], "ensembl_tx", True),
        (["ENSP00000369497", "ENSDARP00000153304"], "ensembl_protein", False),
        (["NM_007294.4", "NM_000546", "NR_046018.2"], "refseq_rna", True),
        (["NP_000537.3", "XP_011515983.1"], "refseq_protein", True),
        (["672", "7157", "11576"], "ncbi_geneid", False),
        (["P04637", "Q9Y2B4", "A0A024R161"], "uniprot", False),
        (["FBgn0000490", "FBgn0003996"], "flybase", False),
        (["WBGene00006763", "WBGene00000001"], "wormbase", False),
        (["ZDB-GENE-990415-270", "ZDB-GENE-040426-1"], "zfin", False),
        (["MGI:88180", "MGI:98834"], "mgi", False),
        (["HGNC:11998", "HGNC:1100"], "hgnc", False),
        (["Smp_175590", "Smp_051920", "Smp_000020.1"], "wbps", True),
        (["SOX2", "TP53", "CD3D"], "symbol", False),
    ],
)
def test_single_flavor(ids: list[str], expected: str, versioned: bool) -> None:
    r = detect_id_flavor(ids)
    assert r.flavor == expected, str(r)
    assert r.confidence == 1.0
    assert r.has_version is versioned
    assert r.sample_size == len(ids)
    assert expected in FLAVOR_PATTERNS or expected == "unknown"


def test_unknown_namespace() -> None:
    r = detect_id_flavor(["dd_Smed_v4_10001_0_1", "t33417aep", "TRINITY_DN1234_c0_g1"])
    assert r.flavor == "unknown"
    assert r.counts["unknown"] == 3


def test_unknown_beats_minority_specific() -> None:
    # 8 de-novo contig IDs + 2 Ensembl → still report unknown (not mixed),
    # because the dominant signal is "consistently unrecognized".
    ids = [f"dd_Smed_v4_{i}_0_1" for i in range(8)] + ["ENSG00000139618"] * 2
    r = detect_id_flavor(ids)
    assert r.flavor == "unknown"
    assert r.confidence == pytest.approx(0.8)


def test_mixed_below_threshold() -> None:
    ids = ["ENSG00000139618"] * 3 + ["NM_007294"] * 3 + ["7157"] * 4
    r = detect_id_flavor(ids, min_confidence=0.7)
    assert r.flavor == "mixed"
    assert r.counts == {"ensembl_gene": 3, "refseq_rna": 3, "ncbi_geneid": 4}


def test_symbol_does_not_shadow_specific() -> None:
    # 8 Ensembl + 2 symbol-like → ensembl should win even though symbol
    # could also match short uppercase tokens.
    ids = [f"ENSG{n:011d}" for n in range(8)] + ["SOX2", "BRCA1"]
    r = detect_id_flavor(ids)
    assert r.flavor == "ensembl_gene"
    assert r.confidence == pytest.approx(0.8)


def test_empty_input() -> None:
    r = detect_id_flavor([])
    assert r.flavor == "unknown"
    assert r.sample_size == 0
    assert r.confidence == 0.0


def test_sample_cap_is_respected() -> None:
    ids = (f"ENSG{n:011d}" for n in range(10_000))
    r = detect_id_flavor(ids, sample=50)
    assert r.sample_size == 50
    assert r.flavor == "ensembl_gene"


def test_examples_populated() -> None:
    r = detect_id_flavor(["ENSG00000139618", "NM_007294", "ENSG00000141510"])
    assert "ENSG00000139618" in r.examples["ensembl_gene"]
    assert "NM_007294" in r.examples["refseq_rna"]
