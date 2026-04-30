"""Tests for samap.io.match.match_fasta."""

from __future__ import annotations

from pathlib import Path

import pytest

from samap.io.match import TRANSFORMS, _gtf_tx2gene, match_fasta

EXAMPLE = Path(__file__).parent.parent.parent / "example_data"


# ---------------------------------------------------------------------------
# Transform unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "header", "expected"),
    [
        ("first_token", "ENSG00000139618 some desc", "ENSG00000139618"),
        ("strip_version", "ENST00000380152.7 desc", "ENST00000380152"),
        ("uniprot_pipe", "sp|P04637|P53_HUMAN", "P04637"),
        ("ncbi_lcl", "lcl|NM_007294.4 BRCA1", "NM_007294.4"),
        ("kv_gene", "TX1 gene:ENSG00000139618 biotype:protein_coding", "ENSG00000139618"),
        ("kv_gene", "TX1 gene_id=Smp_000040", "Smp_000040"),
        ("kv_transcript", "P1 transcript:ENST00000380152", "ENST00000380152"),
        ("after_last_pipe", "abc|def|ENSG00000139618", "ENSG00000139618"),
        ("before_first_pipe", "ENSG00000139618|abc|def", "ENSG00000139618"),
    ],
)
def test_builtin_transforms(name: str, header: str, expected: str) -> None:
    assert TRANSFORMS[name](header) == expected


def test_transforms_return_none_when_inapplicable() -> None:
    assert TRANSFORMS["strip_version"]("ENSG00000139618") is None
    assert TRANSFORMS["uniprot_pipe"]("plain_id") is None
    assert TRANSFORMS["ncbi_lcl"]("plain_id") is None
    assert TRANSFORMS["kv_gene"]("no kv here") is None
    assert TRANSFORMS["after_last_pipe"]("nopipe") is None


# ---------------------------------------------------------------------------
# match_fasta — synthetic
# ---------------------------------------------------------------------------


def _write_fa(path: Path, recs: dict[str, str]) -> Path:
    with path.open("w") as f:
        for h, s in recs.items():
            f.write(f">{h}\n{s}\n")
    return path


def test_match_first_token_wins(tmp_path: Path) -> None:
    var = ["g1", "g2", "g3"]
    fa = _write_fa(tmp_path / "in.fa", {"g1 desc x": "AAA", "g2 desc": "CCCC", "gX": "GG"})
    rep = match_fasta(var, fa, write=tmp_path / "out.fa")

    assert rep.transform == "first_token"
    assert rep.n_matched == 2
    assert rep.frac == pytest.approx(2 / 3)
    assert rep.sample_unmatched == ["g3"]
    # scores DataFrame has all transforms, first_token best
    assert rep.scores.index[0] == "first_token"

    out = (tmp_path / "out.fa").read_text()
    assert ">g1\n" in out and ">g2\n" in out
    assert ">g3" not in out


def test_match_strip_version(tmp_path: Path) -> None:
    var = ["ENSG00000139618", "ENSG00000141510"]
    fa = _write_fa(
        tmp_path / "in.fa",
        {"ENSG00000139618.17": "M" * 5, "ENSG00000141510.19": "M" * 5},
    )
    rep = match_fasta(var, fa)
    assert rep.transform == "strip_version"
    assert rep.n_matched == 2
    assert rep.names is not None
    assert {tuple(r) for r in rep.names.tolist()} == {
        ("ENSG00000139618.17", "ENSG00000139618"),
        ("ENSG00000141510.19", "ENSG00000141510"),
    }


def test_match_kv_gene_ensembl_style(tmp_path: Path) -> None:
    # Ensembl pep FASTA header style:
    # >ENSP00000369497 pep ... gene:ENSG00000139618 transcript:ENST00000380152 ...
    var = ["ENSG00000139618"]
    fa = _write_fa(
        tmp_path / "in.fa",
        {"ENSP00000369497 pep gene:ENSG00000139618 transcript:ENST00000380152": "M" * 30},
    )
    rep = match_fasta(var, fa)
    assert rep.transform == "kv_gene"
    assert rep.n_matched == 1
    assert rep.names[0, 0] == "ENSP00000369497"
    assert rep.names[0, 1] == "ENSG00000139618"


def test_match_via_gtf(tmp_path: Path) -> None:
    var = ["GENE_A", "GENE_B"]
    # FASTA keyed on transcript IDs; GTF supplies tx → gene
    fa = _write_fa(tmp_path / "in.fa", {"tx1.1": "MMM", "tx2.1": "MMMM", "tx3.1": "M"})
    gtf = tmp_path / "ann.gtf"
    gtf.write_text(
        'chr1\tsrc\ttranscript\t1\t10\t.\t+\t.\tgene_id "GENE_A"; transcript_id "tx1";\n'
        'chr1\tsrc\ttranscript\t1\t10\t.\t+\t.\tgene_id "GENE_A"; transcript_id "tx2";\n'
        'chr1\tsrc\ttranscript\t1\t10\t.\t+\t.\tgene_id "GENE_B"; transcript_id "tx3";\n'
    )
    rep = match_fasta(var, fa, gtf=gtf, write=tmp_path / "out.fa")
    assert rep.transform == "gtf"
    assert rep.n_matched == 2
    # tx1 and tx2 both → GENE_A; longest (tx2, 4 aa) chosen for output
    from samap.io.fetch import _iter_fasta
    out = dict(_iter_fasta((tmp_path / "out.fa").read_text()))
    assert len(out["GENE_A"]) == 4
    # names array maps each fasta first-token → gene
    assert rep.names.shape[0] == 3


def test_match_explicit_mapping(tmp_path: Path) -> None:
    var = ["A", "B"]
    fa = _write_fa(tmp_path / "in.fa", {"weird1": "MM", "weird2": "MMMM"})
    rep = match_fasta(var, fa, mapping={"weird1": "A", "weird2": "B"})
    assert rep.transform == "mapping"
    assert rep.n_matched == 2


def test_match_extra_transform(tmp_path: Path) -> None:
    # Hydra-style: var_names have a baked-in 'hy_' prefix the FASTA lacks.
    var = ["hy_t1aep", "hy_t2aep", "hy_t3aep"]
    fa = _write_fa(tmp_path / "in.fa", {"t1aep": "MM", "t2aep": "MM"})
    rep = match_fasta(var, fa)
    assert rep.n_matched == 0  # built-ins can't add a prefix
    rep2 = match_fasta(var, fa, extra_transforms={"add_hy": lambda h: f"hy_{h.split()[0]}"})
    assert rep2.transform == "add_hy"
    assert rep2.n_matched == 2


def test_gtf_parser_styles(tmp_path: Path) -> None:
    gtf = tmp_path / "x.gtf"
    gtf.write_text(
        '# comment\n'
        'c\ts\ttx\t1\t2\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
        'c\ts\ttx\t1\t2\t.\t+\t.\ttranscript_id=T2;gene_id=G2\n'
    )
    m = _gtf_tx2gene(gtf)
    assert m == {"T1": "G1", "T2": "G2"}


# ---------------------------------------------------------------------------
# match_fasta — bundled example data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_data not available")
@pytest.mark.parametrize(
    ("h5ad", "fasta", "min_frac", "expect_tx"),
    [
        ("planarian.h5ad", "transcriptomes/planarian_transcriptome.fasta", 0.99, "identity"),
        ("schistosome.h5ad", "transcriptomes/schistosome_proteome.fasta", 0.99, "identity"),
    ],
)
def test_match_on_bundled(h5ad: str, fasta: str, min_frac: float, expect_tx: str) -> None:
    import anndata as ad
    a = ad.read_h5ad(EXAMPLE / h5ad)
    rep = match_fasta(a, EXAMPLE / fasta)
    assert rep.frac >= min_frac, str(rep)
    # identity and first_token should tie at the top for these clean cases
    assert rep.scores.loc[expect_tx, "frac"] >= min_frac
