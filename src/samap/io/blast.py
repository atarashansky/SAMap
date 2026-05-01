"""Reciprocal sequence search and ``gnnm`` caching.

Two concerns live here:

1. :func:`run_blast` — a Python port of ``map_genes.sh``. Runs all
   N-choose-2 reciprocal alignments for a set of FASTAs and writes
   ``maps/{a}{b}/{a}_to_{b}.txt`` tables in the BLAST ``-outfmt 6``
   layout that :func:`samap.core.homology._calculate_blast_graph`
   already consumes. Three engines:

   - **DIAMOND** — fastest CPU prot↔prot/nucl↔prot. Default for
     protein databases.
   - **MMseqs2** — covers *all four* modes including translated
     nucl↔nucl (tblastx-equivalent), and adds ``--gpu`` on CUDA
     hardware. Default for nucleotide databases.
   - **NCBI BLAST+** — reference; last-resort fallback.

   All three emit the same 12-column ``-outfmt 6`` table. Install with
   ``conda install -c bioconda diamond mmseqs2 blast``.

2. :func:`save_gnnm` / :func:`load_gnnm` — serialize the
   ``(gnnm, gns, gns_dict)`` tuple to a single ``.npz``. The 3-species
   example takes ~38 s to parse from BLAST tables on every SAMAP init;
   round-tripping the cached graph is sub-second.

Typical usage::

    from samap.io import run_blast, save_gnnm, load_gnnm
    from samap.core.homology import _calculate_blast_graph

    run_blast(
        {"hu": ("hu.fa", "prot"), "mm": ("mm.fa", "prot")},
        f_maps="maps/", engine="diamond", threads=16,
    )
    gnnm = _calculate_blast_graph(["hu", "mm"], f_maps="maps/", reciprocate=True)
    save_gnnm(gnnm, "maps/gnnm.npz")

    # … later, on every SAMAP init:
    sm = SAMAP(sams, gnnm=load_gnnm("maps/gnnm.npz"))
"""

from __future__ import annotations

import shutil
import subprocess
from itertools import combinations
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp

from samap._logging import logger

__all__ = ["load_gnnm", "run_blast", "save_gnnm"]

GnnmTuple = tuple[sp.csr_matrix, np.ndarray, dict[str, np.ndarray]]
SeqType = Literal["prot", "nucl"]
Engine = Literal["auto", "diamond", "mmseqs", "blast"]

CONDA_INSTALL_HINT = "conda install -c bioconda diamond mmseqs2 blast"


# ---------------------------------------------------------------------------
# gnnm cache
# ---------------------------------------------------------------------------


def save_gnnm(gnnm: GnnmTuple, path: str | PathLike[str]) -> Path:
    """Serialize a ``(gnnm, gns, gns_dict)`` tuple to a single ``.npz``.

    Parameters
    ----------
    gnnm
        The tuple returned by :func:`samap.core.homology._calculate_blast_graph`,
        :func:`samap.io.homology.gnnm_from_pairs`, etc.
    path
        Output ``.npz`` path.
    """
    m, gns, gns_dict = gnnm
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sids = np.array(list(gns_dict), dtype=object)
    np.savez_compressed(
        p,
        data=m.data,
        indices=m.indices,
        indptr=m.indptr,
        shape=np.array(m.shape),
        gns=np.asarray(gns, dtype=object),
        sids=sids,
        **{f"gns__{sid}": np.asarray(v, dtype=object) for sid, v in gns_dict.items()},
    )
    logger.info("save_gnnm: wrote %s (%d genes, %d edges)", p, gns.size, m.nnz)
    return p


def load_gnnm(path: str | PathLike[str]) -> GnnmTuple:
    """Load a ``(gnnm, gns, gns_dict)`` tuple from :func:`save_gnnm`."""
    with np.load(path, allow_pickle=True) as z:
        m = sp.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        gns = z["gns"]
        sids = list(z["sids"])
        gns_dict = {sid: z[f"gns__{sid}"] for sid in sids}
    logger.info("load_gnnm: %s → %d genes, %d edges", path, gns.size, m.nnz)
    return m, gns, gns_dict


# ---------------------------------------------------------------------------
# run_blast — port of map_genes.sh
# ---------------------------------------------------------------------------

# Dispatch table: (query_type, db_type) → per-engine subcommand/args. ``None``
# means the engine has no mode for that direction.
#
#   diamond:  blastp/blastx only (protein DB).
#   mmseqs:   easy-search covers all four; nucl query is auto-6-frame
#             translated; ``--search-type 2`` forces translated–translated
#             (tblastx-equivalent) for nucl↔nucl.
#   blast:    reference exe per direction.
#
_DISPATCH: dict[tuple[SeqType, SeqType], dict[str, str | tuple[str, ...] | None]] = {
    ("prot", "prot"): {"blast": "blastp",  "diamond": "blastp", "mmseqs": ()},
    ("nucl", "prot"): {"blast": "blastx",  "diamond": "blastx", "mmseqs": ()},
    ("prot", "nucl"): {"blast": "tblastn", "diamond": None,     "mmseqs": ()},
    ("nucl", "nucl"): {"blast": "tblastx", "diamond": None,
                       "mmseqs": ("--search-type", "2")},
}

# ``auto`` preference: DIAMOND for protein DBs (fastest CPU prot↔prot),
# MMseqs2 otherwise (only fast option for translated nucl), BLAST+ last.
_AUTO_PREFERENCE: dict[SeqType, tuple[str, ...]] = {
    "prot": ("diamond", "mmseqs", "blast"),
    "nucl": ("mmseqs", "blast"),
}


def _which(exe: str) -> str | None:
    return shutil.which(exe)


def _has_cuda() -> bool:
    """Best-effort check for an NVIDIA GPU (so MMseqs2 ``--gpu 1`` is added)."""
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    return _which("nvidia-smi") is not None


def _engine_available(eng: str, qtype: SeqType, dbtype: SeqType) -> bool:
    row = _DISPATCH[(qtype, dbtype)]
    if eng == "diamond":
        return row["diamond"] is not None and _which("diamond") is not None
    if eng == "mmseqs":
        return _which("mmseqs") is not None
    if eng == "blast":
        return _which(row["blast"]) is not None and _which("makeblastdb") is not None
    raise ValueError(f"unknown engine: {eng!r}")


def _resolve_engine(engine: Engine, qtype: SeqType, dbtype: SeqType) -> str:
    """Pick concrete engine for one direction; raise if unavailable."""
    row = _DISPATCH[(qtype, dbtype)]
    if engine == "auto":
        for cand in _AUTO_PREFERENCE[dbtype]:
            if _engine_available(cand, qtype, dbtype):
                return cand
        raise RuntimeError(
            f"No aligner found on PATH for {qtype} query vs {dbtype} db "
            f"(tried {', '.join(_AUTO_PREFERENCE[dbtype])}). Install with "
            f"`{CONDA_INSTALL_HINT}`."
        )
    # Explicit engine: honour if possible, else fall back loudly.
    if _engine_available(engine, qtype, dbtype):
        return engine
    blast_exe = row["blast"]
    if engine == "diamond" and row["diamond"] is None:
        logger.warning(
            "DIAMOND has no %s mode (%s vs %s); falling back via auto.",
            blast_exe, qtype, dbtype,
        )
    else:
        logger.warning(
            "engine=%r not found on PATH for %s vs %s; falling back via auto.",
            engine, qtype, dbtype,
        )
    return _resolve_engine("auto", qtype, dbtype)


def run_blast(
    fastas: dict[str, tuple[str | PathLike[str], SeqType]],
    *,
    f_maps: str | PathLike[str] = "maps/",
    engine: Engine = "auto",
    threads: int = 8,
    evalue: float = 1e-6,
    sensitivity: str = "very-sensitive",
    gpu: bool | None = None,
    overwrite: bool = False,
    _runner: callable = subprocess.run,  # test seam
) -> Path:
    """Run all N-choose-2 reciprocal alignments for a set of FASTAs.

    Output layout matches what ``SAMAP(f_maps=...)`` /
    :func:`samap.core.homology._calculate_blast_graph` expects::

        <f_maps>/<a><b>/<a>_to_<b>.txt
        <f_maps>/<a><b>/<b>_to_<a>.txt

    Parameters
    ----------
    fastas
        ``{species_id: (fasta_path, "prot" | "nucl")}``. At least two
        species. Use :func:`samap.io.fetch_proteome` or
        :func:`samap.io.match_fasta` upstream to ensure headers match
        ``adata.var_names``.
    f_maps
        Output directory.
    engine
        - ``"auto"`` (default): DIAMOND for protein-DB targets, MMseqs2
          for nucleotide-DB targets, NCBI BLAST+ as last resort.
        - ``"diamond"`` / ``"mmseqs"`` / ``"blast"``: force one engine
          (falls back via ``auto`` per-direction if it has no mode or
          isn't on PATH).
    threads
        Per-process thread count.
    evalue
        E-value cutoff (passed to the aligner).
    sensitivity
        DIAMOND sensitivity flag (``"sensitive"`` / ``"very-sensitive"``
        / ``"ultra-sensitive"``). For MMseqs2 this maps to ``-s``
        (5.7 / 7.5 / 8.5 respectively). Ignored for BLAST+.
    gpu
        MMseqs2 only. ``None`` (default) auto-detects via
        ``nvidia-smi``; ``True``/``False`` forces. Ignored otherwise.
    overwrite
        If False (default), skip a direction whose output table already
        exists and is non-empty.

    Returns
    -------
    Path
        ``f_maps`` as a :class:`~pathlib.Path`.

    Notes
    -----
    All three engines emit the 12-column BLAST ``-outfmt 6`` table, so
    ``_calculate_blast_graph`` consumes any of them without changes.
    Install with ``conda install -c bioconda diamond mmseqs2 blast``.
    """
    if len(fastas) < 2:
        raise ValueError("need at least two species in `fastas`")
    for sid, (fa, t) in fastas.items():
        if t not in ("prot", "nucl"):
            raise ValueError(f"{sid}: type must be 'prot' or 'nucl', got {t!r}")
        if not Path(fa).exists():
            raise FileNotFoundError(f"{sid}: FASTA not found: {fa}")

    out_root = Path(f_maps)
    out_root.mkdir(parents=True, exist_ok=True)
    dbdir = out_root / "_db"
    dbdir.mkdir(exist_ok=True)
    tmpdir = out_root / "_tmp"

    use_gpu = _has_cuda() if gpu is None else bool(gpu)
    mmseqs_s = {"sensitive": "5.7", "very-sensitive": "7.5",
                "ultra-sensitive": "8.5"}.get(sensitivity, "7.5")

    # DBs are built lazily per (species, engine) the first time they're needed.
    dbs: dict[tuple[str, str], Path] = {}

    def _ensure_db(sid: str, eng: str) -> Path:
        key = (sid, eng)
        if key in dbs:
            return dbs[key]
        fa, t = fastas[sid]
        if eng == "diamond":
            p = dbdir / f"{sid}.dmnd"
            if overwrite or not p.exists():
                logger.info("diamond makedb: %s → %s", fa, p)
                _runner(["diamond", "makedb", "--in", str(fa), "-d", str(p),
                         "--threads", str(threads)], check=True)
        elif eng == "mmseqs":
            p = dbdir / f"mm_{sid}"
            sentinel = dbdir / f"mm_{sid}.dbtype"
            if overwrite or not sentinel.exists():
                logger.info("mmseqs createdb: %s (%s) → %s", fa, t, p)
                _runner(["mmseqs", "createdb", str(fa), str(p),
                         "--dbtype", "1" if t == "prot" else "2"], check=True)
        elif eng == "blast":
            p = dbdir / sid
            sentinel = dbdir / f"{sid}.{'phr' if t == 'prot' else 'nhr'}"
            if overwrite or not sentinel.exists():
                logger.info("makeblastdb: %s (%s) → %s", fa, t, p)
                _runner(["makeblastdb", "-in", str(fa), "-dbtype", t,
                         "-out", str(p)], check=True)
        else:  # pragma: no cover — guarded by _resolve_engine
            raise ValueError(eng)
        dbs[key] = p
        return p

    # All ordered (query, db) pairs across distinct species.
    for a, b in combinations(fastas, 2):
        pair_dir = out_root / f"{a}{b}"
        pair_dir.mkdir(exist_ok=True)
        for q, d in ((a, b), (b, a)):
            qfa, qtype = fastas[q]
            _, dtype = fastas[d]
            out = pair_dir / f"{q}_to_{d}.txt"
            if not overwrite and out.exists() and out.stat().st_size > 0:
                logger.info("skip (exists): %s", out)
                continue
            eng = _resolve_engine(engine, qtype, dtype)
            row = _DISPATCH[(qtype, dtype)]
            if eng == "diamond":
                db = _ensure_db(d, "diamond")
                cmd = [
                    "diamond", row["diamond"],
                    "-q", str(qfa), "-d", str(db), "-o", str(out),
                    "--outfmt", "6",
                    "--evalue", str(evalue),
                    "--max-hsps", "1",
                    "--threads", str(threads),
                    f"--{sensitivity}",
                ]
            elif eng == "mmseqs":
                db = _ensure_db(d, "mmseqs")
                tmpdir.mkdir(exist_ok=True)
                cmd = [
                    "mmseqs", "easy-search",
                    str(qfa), str(db), str(out), str(tmpdir),
                    "-e", str(evalue),
                    "-s", mmseqs_s,
                    "--threads", str(threads),
                    "--format-mode", "0",
                    *row["mmseqs"],
                ]
                if use_gpu:
                    cmd += ["--gpu", "1"]
            else:  # blast
                db = _ensure_db(d, "blast")
                cmd = [
                    row["blast"],
                    "-query", str(qfa), "-db", str(db), "-out", str(out),
                    "-outfmt", "6",
                    "-evalue", str(evalue),
                    "-max_hsps", "1",
                    "-num_threads", str(threads),
                ]
            logger.info("[%s] %s → %s : %s", eng, q, d, " ".join(cmd[:2]))
            _runner(cmd, check=True)

    return out_root
