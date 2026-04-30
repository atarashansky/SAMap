"""Tests for the `samap` CLI entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from samap.__main__ import _build_parser, main

EXAMPLE = Path(__file__).parent.parent.parent / "example_data"


def test_parser_has_all_subcommands() -> None:
    p = _build_parser()
    sub = next(
        a for a in p._actions if getattr(a, "choices", None) and isinstance(a.choices, dict)
    )
    assert set(sub.choices) >= {"detect-ids", "fetch-proteome", "match-fasta", "blast"}


def test_parser_blast_species_repeated() -> None:
    p = _build_parser()
    ns = p.parse_args(
        [
            "blast",
            "--species", "hu", "hu.fa", "prot",
            "--species", "mm", "mm.fa", "prot",
            "--maps", "out/",
        ]
    )
    assert ns.species == [["hu", "hu.fa", "prot"], ["mm", "mm.fa", "prot"]]
    assert ns.maps == "out/"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_data not available")
def test_cli_detect_ids(capsys) -> None:
    rc = main(["detect-ids", str(EXAMPLE / "schistosome.h5ad")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wbps" in out


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_data not available")
def test_cli_match_fasta(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "match-fasta",
            str(EXAMPLE / "schistosome.h5ad"),
            str(EXAMPLE / "transcriptomes" / "schistosome_proteome.fasta"),
            "-o", str(tmp_path / "out.fa"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "identity" in out
    assert (tmp_path / "out.fa").exists()
