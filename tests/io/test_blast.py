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


def test_resolve_engine_diamond_preferred(monkeypatch) -> None:
    monkeypatch.setattr(blast_mod, "_which", lambda exe: f"/bin/{exe}")
    assert _resolve_engine("auto", "prot", "prot") == "diamond"
    assert _resolve_engine("auto", "nucl", "prot") == "diamond"  # blastx
    # tblastx → no diamond mode → blast
    assert _resolve_engine("auto", "nucl", "nucl") == "blast"
    assert _resolve_engine("blast", "prot", "prot") == "blast"


def test_resolve_engine_missing_raises(monkeypatch) -> None:
    monkeypatch.setattr(blast_mod, "_which", lambda exe: None)
    with pytest.raises(RuntimeError, match="Neither DIAMOND nor NCBI BLAST"):
        _resolve_engine("auto", "prot", "prot")


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


def test_run_blast_diamond_commands(tmp_path: Path, two_fastas, monkeypatch) -> None:
    monkeypatch.setattr(blast_mod, "_which", lambda exe: f"/bin/{exe}")
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(list(cmd))
        # touch output files so skip-existing logic can be exercised
        for i, tok in enumerate(cmd):
            if tok in ("-o", "-out", "-d") and i + 1 < len(cmd):
                Path(cmd[i + 1]).touch()
        return None

    out = run_blast(two_fastas, f_maps=tmp_path / "maps", _runner=fake_run)
    assert (out / "aabb").is_dir()
    # 2 makedb + 2 alignments (aa→bb, bb→aa)
    kinds = [c[0:2] for c in calls]
    assert kinds.count(["diamond", "makedb"]) == 2
    assert kinds.count(["diamond", "blastp"]) == 2
    # outfmt 6, max-hsps 1, sensitivity flag present
    align = next(c for c in calls if c[:2] == ["diamond", "blastp"])
    assert "--outfmt" in align and align[align.index("--outfmt") + 1] == "6"
    assert "--max-hsps" in align
    assert "--very-sensitive" in align
    # output files named correctly
    assert (out / "aabb" / "aa_to_bb.txt").exists()
    assert (out / "aabb" / "bb_to_aa.txt").exists()


def test_run_blast_skips_existing(tmp_path: Path, two_fastas, monkeypatch) -> None:
    monkeypatch.setattr(blast_mod, "_which", lambda exe: f"/bin/{exe}")
    pair = tmp_path / "maps" / "aabb"
    pair.mkdir(parents=True)
    (pair / "aa_to_bb.txt").write_text("x\n")
    (pair / "bb_to_aa.txt").write_text("x\n")
    calls: list[list[str]] = []
    run_blast(
        two_fastas,
        f_maps=tmp_path / "maps",
        _runner=lambda c, check: calls.append(list(c)),
    )
    # DBs may still be built; alignments must be skipped.
    assert not any(c[:2] == ["diamond", "blastp"] for c in calls), calls
    assert not any(c[0] in {"blastp", "blastx", "tblastn", "tblastx"} for c in calls)


def test_run_blast_ncbi_tblastx(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a.fa"
    b = tmp_path / "b.fa"
    a.write_text(">g\nACGTACGT\n")
    b.write_text(">h\nACGTACGT\n")
    fastas = {"aa": (a, "nucl"), "bb": (b, "nucl")}

    monkeypatch.setattr(
        blast_mod, "_which", lambda exe: None if exe == "diamond" else f"/bin/{exe}"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(list(cmd))
        for i, tok in enumerate(cmd):
            if tok in ("-out", "-o") and i + 1 < len(cmd):
                Path(cmd[i + 1]).touch()
        if cmd[0] == "makeblastdb":
            outp = cmd[cmd.index("-out") + 1]
            Path(outp + ".nhr").touch()

    run_blast(fastas, f_maps=tmp_path / "maps", _runner=fake_run)
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
