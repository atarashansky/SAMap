"""Tests for samap.io.blast — gnnm cache and run_blast (mocked subprocess)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from samap.io import blast as blast_mod
from samap.io.blast import _resolve_engine, load_gnnm, run_blast, save_gnnm


# ---------------------------------------------------------------------------
# gnnm cache round-trip
# ---------------------------------------------------------------------------


def _toy_gnnm() -> tuple[sp.csr_matrix, np.ndarray, dict[str, np.ndarray]]:
    gns = np.array(["aa_g1", "aa_g2", "bb_h1", "bb_h2"], dtype=object)
    m = sp.csr_matrix(
        np.array(
            [[0, 0, 0.9, 0], [0, 0, 0, 0.5], [0.9, 0, 0, 0], [0, 0.5, 0, 0]],
            dtype=np.float32,
        )
    )
    gd = {"aa": gns[:2], "bb": gns[2:]}
    return m, gns, gd


def test_save_load_gnnm_roundtrip(tmp_path: Path) -> None:
    g = _toy_gnnm()
    p = save_gnnm(g, tmp_path / "g.npz")
    m2, gns2, gd2 = load_gnnm(p)
    assert (m2 != g[0]).nnz == 0
    np.testing.assert_array_equal(gns2, g[1])
    assert sorted(gd2) == ["aa", "bb"]
    np.testing.assert_array_equal(gd2["aa"], g[2]["aa"])


def test_load_gnnm_feeds_SAMAP(tmp_path: Path, monkeypatch) -> None:
    """The loaded tuple has the exact shape SAMAP(gnnm=...) unpacks."""
    g = _toy_gnnm()
    p = save_gnnm(g, tmp_path / "g.npz")
    m, gns, gd = load_gnnm(p)
    # SAMAP.__init__ does: gnnm_matrix, gns, gns_dict = gnnm
    # — verify we can unpack exactly 3.
    a, b, c = (m, gns, gd)
    assert sp.issparse(a) and isinstance(b, np.ndarray) and isinstance(c, dict)


# ---------------------------------------------------------------------------
# run_blast — engine resolution
# ---------------------------------------------------------------------------


def _patch_path(monkeypatch, present: set[str]) -> None:
    monkeypatch.setattr(
        blast_mod, "_which", lambda exe: f"/bin/{exe}" if exe in present else None
    )


def test_resolve_engine_auto_preference(monkeypatch) -> None:
    # All available: prot DB → diamond, nucl DB → mmseqs.
    _patch_path(monkeypatch, {"diamond", "mmseqs", "blastp", "blastx",
                              "tblastn", "tblastx", "makeblastdb"})
    assert _resolve_engine("auto", "prot", "prot") == "diamond"
    assert _resolve_engine("auto", "nucl", "prot") == "diamond"  # blastx
    assert _resolve_engine("auto", "prot", "nucl") == "mmseqs"
    assert _resolve_engine("auto", "nucl", "nucl") == "mmseqs"
    # Explicit engines respected.
    assert _resolve_engine("mmseqs", "prot", "prot") == "mmseqs"
    assert _resolve_engine("blast", "prot", "prot") == "blast"


def test_resolve_engine_auto_skips_missing(monkeypatch) -> None:
    _patch_path(monkeypatch, {"mmseqs"})
    # diamond not on PATH → mmseqs picked even for prot DB
    assert _resolve_engine("auto", "prot", "prot") == "mmseqs"
    _patch_path(monkeypatch, {"tblastx", "makeblastdb"})
    # only blast+ → blast for nucl↔nucl
    assert _resolve_engine("auto", "nucl", "nucl") == "blast"


def test_resolve_engine_diamond_nucl_fallback(monkeypatch) -> None:
    _patch_path(monkeypatch, {"diamond", "mmseqs"})
    # diamond has no tblastx → falls back via auto → mmseqs
    assert _resolve_engine("diamond", "nucl", "nucl") == "mmseqs"


def test_resolve_engine_missing_raises(monkeypatch) -> None:
    _patch_path(monkeypatch, set())
    with pytest.raises(RuntimeError, match="No aligner found"):
        _resolve_engine("auto", "prot", "prot")
    # Explicit-but-missing → falls back to auto → still nothing → raise
    with pytest.raises(RuntimeError, match="No aligner found"):
        _resolve_engine("mmseqs", "prot", "prot")


# ---------------------------------------------------------------------------
# run_blast — full call with mocked subprocess
# ---------------------------------------------------------------------------


@pytest.fixture
def two_fastas(tmp_path: Path) -> dict[str, tuple[Path, str]]:
    a = tmp_path / "a.fa"
    b = tmp_path / "b.fa"
    a.write_text(">g1\nMMMM\n>g2\nMMMM\n")
    b.write_text(">h1\nMMMM\n>h2\nMMMM\n")
    return {"aa": (a, "prot"), "bb": (b, "prot")}


def _record_runner(calls: list[list[str]]):
    """Fake subprocess.run that records calls and touches output files."""
    def _run(cmd, check):
        calls.append(list(cmd))
        for i, tok in enumerate(cmd):
            if tok in ("-o", "-out", "-d") and i + 1 < len(cmd):
                Path(cmd[i + 1]).touch()
        if cmd[:2] == ["mmseqs", "easy-search"]:
            Path(cmd[4]).touch()  # positional out.m8
        if cmd[:2] == ["mmseqs", "createdb"]:
            Path(cmd[3] + ".dbtype").touch()
        if cmd[0] == "makeblastdb":
            outp = cmd[cmd.index("-out") + 1]
            ext = ".phr" if cmd[cmd.index("-dbtype") + 1] == "prot" else ".nhr"
            Path(outp + ext).touch()
    return _run


def test_run_blast_diamond_commands(tmp_path: Path, two_fastas, monkeypatch) -> None:
    _patch_path(monkeypatch, {"diamond", "mmseqs"})
    calls: list[list[str]] = []
    out = run_blast(two_fastas, f_maps=tmp_path / "maps", _runner=_record_runner(calls))
    assert (out / "aabb").is_dir()
    kinds = [c[:2] for c in calls]
    # Lazy DB build: only DBs actually used as targets get built (aa & bb).
    assert kinds.count(["diamond", "makedb"]) == 2
    assert kinds.count(["diamond", "blastp"]) == 2
    # No mmseqs calls for prot↔prot under auto.
    assert not any(c[0] == "mmseqs" for c in calls)
    align = next(c for c in calls if c[:2] == ["diamond", "blastp"])
    assert align[align.index("--outfmt") + 1] == "6"
    assert "--max-hsps" in align
    assert "--very-sensitive" in align
    assert (out / "aabb" / "aa_to_bb.txt").exists()
    assert (out / "aabb" / "bb_to_aa.txt").exists()


def test_run_blast_skips_existing(tmp_path: Path, two_fastas, monkeypatch) -> None:
    _patch_path(monkeypatch, {"diamond", "mmseqs", "blastp", "makeblastdb"})
    pair = tmp_path / "maps" / "aabb"
    pair.mkdir(parents=True)
    (pair / "aa_to_bb.txt").write_text("x\n")
    (pair / "bb_to_aa.txt").write_text("x\n")
    calls: list[list[str]] = []
    run_blast(two_fastas, f_maps=tmp_path / "maps", _runner=_record_runner(calls))
    # Lazy DB build → nothing at all should run.
    assert calls == []


def test_run_blast_mmseqs_nucl(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a.fa"
    b = tmp_path / "b.fa"
    a.write_text(">g\nACGTACGT\n")
    b.write_text(">h\nACGTACGT\n")
    fastas = {"aa": (a, "nucl"), "bb": (b, "nucl")}

    _patch_path(monkeypatch, {"diamond", "mmseqs"})
    monkeypatch.setattr(blast_mod, "_has_cuda", lambda: False)
    calls: list[list[str]] = []
    out = run_blast(fastas, f_maps=tmp_path / "maps",
                    sensitivity="ultra-sensitive", _runner=_record_runner(calls))
    kinds = [c[:2] for c in calls]
    assert kinds.count(["mmseqs", "createdb"]) == 2
    assert kinds.count(["mmseqs", "easy-search"]) == 2
    es = next(c for c in calls if c[:2] == ["mmseqs", "easy-search"])
    assert es[es.index("--format-mode") + 1] == "0"
    assert es[es.index("-s") + 1] == "8.5"  # ultra-sensitive mapping
    assert "--search-type" in es and es[es.index("--search-type") + 1] == "2"
    assert "--gpu" not in es
    assert (out / "aabb" / "aa_to_bb.txt").exists()


def test_run_blast_mmseqs_gpu_flag(tmp_path: Path, two_fastas, monkeypatch) -> None:
    _patch_path(monkeypatch, {"mmseqs"})
    monkeypatch.setattr(blast_mod, "_has_cuda", lambda: True)
    calls: list[list[str]] = []
    run_blast(two_fastas, f_maps=tmp_path / "maps", engine="mmseqs",
              _runner=_record_runner(calls))
    es = next(c for c in calls if c[:2] == ["mmseqs", "easy-search"])
    assert "--gpu" in es and es[es.index("--gpu") + 1] == "1"
    # prot↔prot → no --search-type
    assert "--search-type" not in es


def test_run_blast_ncbi_tblastx(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a.fa"
    b = tmp_path / "b.fa"
    a.write_text(">g\nACGTACGT\n")
    b.write_text(">h\nACGTACGT\n")
    fastas = {"aa": (a, "nucl"), "bb": (b, "nucl")}
    # Only BLAST+ available.
    _patch_path(monkeypatch, {"tblastx", "makeblastdb"})
    calls: list[list[str]] = []
    run_blast(fastas, f_maps=tmp_path / "maps", _runner=_record_runner(calls))
    assert any(c[0] == "makeblastdb" and "-dbtype" in c for c in calls)
    assert any(c[0] == "tblastx" for c in calls)


def test_run_blast_input_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_blast({"aa": (tmp_path / "a.fa", "prot")})
    a = tmp_path / "a.fa"
    a.write_text(">x\nM\n")
    with pytest.raises(ValueError, match=r"prot.*nucl"):
        run_blast({"aa": (a, "dna"), "bb": (a, "prot")})  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        run_blast({"aa": (a, "prot"), "bb": (tmp_path / "nope.fa", "prot")})
