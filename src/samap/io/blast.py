"""Reciprocal sequence search and ``gnnm`` caching.

Two concerns live here:

1. :func:`run_blast` — a Python port of ``map_genes.sh``. Runs all
   N-choose-2 reciprocal alignments for a set of FASTAs and writes
   ``maps/{a}{b}/{a}_to_{b}.txt`` tables in the BLAST ``-outfmt 6``
   layout that :func:`samap.core.homology._calculate_blast_graph`
   already consumes. DIAMOND is the default engine (orders of magnitude
   faster on protein-vs-protein); NCBI BLAST+ is the fallback and the
   only option for nucleotide queries (``tblastx``).

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
Engine = Literal["auto", "diamond", "blast"]


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

# Dispatch table: (query_type, db_type) → (blast_exe, diamond_subcmd_or_None).
# DIAMOND covers blastp and blastx; tblastn/tblastx need NCBI BLAST+.
_DISPATCH: dict[tuple[SeqType, SeqType], tuple[str, str | None]] = {
    ("prot", "prot"): ("blastp", "blastp"),
    ("nucl", "prot"): ("blastx", "blastx"),
    ("prot", "nucl"): ("tblastn", None),
    ("nucl", "nucl"): ("tblastx", None),
}


def _which(exe: str) -> str | None:
    return shutil.which(exe)


def _resolve_engine(engine: Engine, qtype: SeqType, dbtype: SeqType) -> str:
    """Pick concrete engine for one direction; raise if unavailable."""
    blast_exe, diamond_sub = _DISPATCH[(qtype, dbtype)]
    if engine in ("auto", "diamond") and diamond_sub and _which("diamond"):
        return "diamond"
    if engine == "diamond" and not diamond_sub:
        logger.warning(
            "DIAMOND has no %s mode (%s vs %s); falling back to NCBI BLAST+.",
            blast_exe, qtype, dbtype,
        )
    if _which(blast_exe) and _which("makeblastdb"):
        return "blast"
    raise RuntimeError(
        f"Neither DIAMOND nor NCBI BLAST+ ({blast_exe}/makeblastdb) found on "
        f"PATH for {qtype} query vs {dbtype} db. Install with e.g. "
        f"`conda install -c bioconda diamond blast`."
    )


def run_blast(
    fastas: dict[str, tuple[str | PathLike[str], SeqType]],
    *,
    f_maps: str | PathLike[str] = "maps/",
    engine: Engine = "auto",
    threads: int = 8,
    evalue: float = 1e-6,
    sensitivity: str = "very-sensitive",
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
        ``"auto"`` (DIAMOND if available and applicable, else BLAST+),
        ``"diamond"`` (force DIAMOND, falls back per-direction where it
        has no mode), or ``"blast"`` (force NCBI BLAST+).
    threads
        Per-process thread count.
    evalue
        E-value cutoff (passed to the aligner).
    sensitivity
        DIAMOND sensitivity flag (e.g. ``"sensitive"``,
        ``"very-sensitive"``, ``"ultra-sensitive"``). Ignored for BLAST+.
    overwrite
        If False (default), skip a direction whose output table already
        exists and is non-empty.

    Returns
    -------
    Path
        ``f_maps`` as a :class:`~pathlib.Path`.

    Notes
    -----
    DIAMOND ``-outfmt 6`` matches BLAST's tab layout column-for-column,
    so ``_calculate_blast_graph`` consumes either without changes.
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

    # Build per-species DBs once.
    dbdir = out_root / "_db"
    dbdir.mkdir(exist_ok=True)
    dbs: dict[str, dict[str, Path]] = {}
    for sid, (fa, t) in fastas.items():
        dbs[sid] = {}
        # DIAMOND db (protein only — diamond makedb wants AA)
        if t == "prot" and _which("diamond"):
            dpath = dbdir / f"{sid}.dmnd"
            if overwrite or not dpath.exists():
                logger.info("diamond makedb: %s → %s", fa, dpath)
                _runner(
                    ["diamond", "makedb", "--in", str(fa), "-d", str(dpath),
                     "--threads", str(threads)],
                    check=True,
                )
            dbs[sid]["diamond"] = dpath
        # NCBI BLAST db
        if _which("makeblastdb"):
            bpath = dbdir / sid
            sentinel = dbdir / f"{sid}.{'phr' if t == 'prot' else 'nhr'}"
            if overwrite or not sentinel.exists():
                logger.info("makeblastdb: %s (%s) → %s", fa, t, bpath)
                _runner(
                    ["makeblastdb", "-in", str(fa), "-dbtype", t, "-out", str(bpath)],
                    check=True,
                )
            dbs[sid]["blast"] = bpath

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
            blast_exe, diamond_sub = _DISPATCH[(qtype, dtype)]
            if eng == "diamond":
                cmd = [
                    "diamond", diamond_sub,
                    "-q", str(qfa),
                    "-d", str(dbs[d]["diamond"]),
                    "-o", str(out),
                    "--outfmt", "6",
                    "--evalue", str(evalue),
                    "--max-hsps", "1",
                    "--threads", str(threads),
                    f"--{sensitivity}",
                ]
            else:
                cmd = [
                    blast_exe,
                    "-query", str(qfa),
                    "-db", str(dbs[d]["blast"]),
                    "-out", str(out),
                    "-outfmt", "6",
                    "-evalue", str(evalue),
                    "-max_hsps", "1",
                    "-num_threads", str(threads),
                ]
            logger.info("[%s] %s → %s : %s", eng, q, d, " ".join(cmd[:2]))
            _runner(cmd, check=True)

    return out_root
