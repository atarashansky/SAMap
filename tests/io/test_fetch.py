"""Tests for samap.io.fetch.fetch_proteome.

Network is mocked for unit tests; a small ``@pytest.mark.network`` smoke
test hits the real Ensembl/NCBI endpoints when explicitly enabled.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pytest

from samap.io import fetch as fetch_mod
from samap.io.fetch import _iter_fasta, _strip_version, fetch_proteome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_headers(path: Path) -> list[str]:
    return [ln[1:].strip() for ln in path.read_text().splitlines() if ln.startswith(">")]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_strip_version() -> None:
    assert _strip_version("ENSG00000139618.17") == "ENSG00000139618"
    assert _strip_version("ENSG00000139618") == "ENSG00000139618"
    assert _strip_version("NM_007294.4") == "NM_007294"


def test_iter_fasta_roundtrip() -> None:
    txt = ">a desc\nACDE\nFG\n>b\nWWW\n"
    recs = list(_iter_fasta(txt))
    assert recs == [("a desc", "ACDEFG"), ("b", "WWW")]


# ---------------------------------------------------------------------------
# Ensembl backend (mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ensembl(monkeypatch):
    """Replace _post_json with canned Ensembl responses."""
    seq_db = {
        "ENSG00000139618": [("ENSP1", "M" * 30), ("ENSP2", "M" * 50)],  # 50 wins
        "ENSG00000141510": [("ENSP3", "M" * 10)],
        "ENST00000380152": [("ENSP4", "M" * 22)],
    }
    parent_db = {"ENST00000380152": "ENSG00000139618"}

    def fake_post(url, body, *, timeout):
        if "/sequence/id" in url:
            out = []
            for q in body["ids"]:
                for acc, s in seq_db.get(q, []):
                    out.append({"id": acc, "query": q, "seq": s})
            return out
        if "/lookup/id" in url:
            return {
                q: ({"object_type": "Transcript", "Parent": parent_db[q]} if q in parent_db else None)
                for q in body["ids"]
            }
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(fetch_mod, "_post_json", fake_post)
    return seq_db


def test_ensembl_gene_canonical_longest(mock_ensembl, tmp_path: Path) -> None:
    ids = ["ENSG00000139618.17", "ENSG00000141510", "ENSG99999999999"]
    rep = fetch_proteome(ids, tmp_path / "out.fa")

    assert rep.flavor == "ensembl_gene"
    assert rep.n_requested == 3
    assert rep.n_fetched == 2
    assert rep.missing == ["ENSG99999999999"]
    assert rep.names is None  # gene-level → no tx→gene map

    heads = _read_headers(rep.fasta_path)
    # Headers are the ORIGINAL (versioned) input ids — match var_names by construction.
    assert heads == ["ENSG00000139618.17", "ENSG00000141510"]

    # Longest isoform was chosen for ENSG00000139618.
    txt = rep.fasta_path.read_text()
    seqs = dict(_iter_fasta(txt))
    assert len(seqs["ENSG00000139618.17"]) == 50


def test_ensembl_tx_builds_names_map(mock_ensembl, tmp_path: Path) -> None:
    rep = fetch_proteome(["ENST00000380152"], tmp_path / "out.fa")
    assert rep.flavor == "ensembl_tx"
    assert rep.n_fetched == 1
    assert rep.names is not None
    assert rep.names.shape == (1, 2)
    assert rep.names[0, 0] == "ENST00000380152"
    assert rep.names[0, 1] == "ENSG00000139618"


def test_ensembl_non_canonical_writes_isoforms(mock_ensembl, tmp_path: Path) -> None:
    rep = fetch_proteome(["ENSG00000139618"], tmp_path / "out.fa", canonical=False)
    heads = _read_headers(rep.fasta_path)
    assert heads == ["ENSG00000139618|ENSP1", "ENSG00000139618|ENSP2"]
    assert rep.n_fetched == 2


def test_anndata_input(mock_ensembl, tmp_path: Path) -> None:
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(
        X=np.zeros((1, 2), dtype=np.float32),
        var={"gene_id": ["ENSG00000139618", "ENSG00000141510"]},
    )
    a.var_names = ["ENSG00000139618", "ENSG00000141510"]
    rep = fetch_proteome(a, tmp_path / "out.fa")
    assert rep.n_fetched == 2


def test_unsupported_flavor_raises(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match=r"unknown"):
        fetch_proteome(["dd_Smed_v4_10001_0_1"], tmp_path / "out.fa")


# ---------------------------------------------------------------------------
# NCBI backend (mocked)
# ---------------------------------------------------------------------------


def _make_ncbi_zip(records: dict[str, list[tuple[str, str]]]) -> bytes:
    """Build a fake datasets/v2 zip with a protein.faa."""
    faa = StringIO()
    for gid, isos in records.items():
        for acc, seq in isos:
            faa.write(f">{acc} stuff [GeneID={gid}]\n{seq}\n")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ncbi_dataset/data/protein.faa", faa.getvalue())
    return buf.getvalue()


@pytest.fixture
def mock_ncbi(monkeypatch):
    db = {"672": [("NP_A", "M" * 40), ("NP_B", "M" * 12)], "7157": [("NP_C", "M" * 30)]}

    class _Resp:
        def __init__(self, data: bytes) -> None:
            self._d = data

        def read(self) -> bytes:
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode())
        wanted = {str(g): db[str(g)] for g in body["gene_ids"] if str(g) in db}
        return _Resp(_make_ncbi_zip(wanted))

    monkeypatch.setattr(fetch_mod, "urlopen", fake_urlopen)
    return db


def test_ncbi_geneid(mock_ncbi, tmp_path: Path) -> None:
    rep = fetch_proteome(["672", "7157", "999999999"], tmp_path / "out.fa")
    assert rep.flavor == "ncbi_geneid"
    assert rep.n_fetched == 2
    assert "999999999" in rep.missing
    heads = _read_headers(rep.fasta_path)
    assert heads == ["672", "7157"]
    seqs = dict(_iter_fasta(rep.fasta_path.read_text()))
    assert len(seqs["672"]) == 40  # longest isoform


# ---------------------------------------------------------------------------
# Live smoke test (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_ensembl_tiny(tmp_path: Path) -> None:  # pragma: no cover
    rep = fetch_proteome(
        ["ENSG00000139618", "ENSG00000141510"], tmp_path / "out.fa", source="ensembl"
    )
    assert rep.n_fetched == 2
    seqs = dict(_iter_fasta(rep.fasta_path.read_text()))
    assert len(seqs["ENSG00000139618"]) > 1000  # BRCA2 protein length
