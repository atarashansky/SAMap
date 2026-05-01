"""SAMap command-line interface.

Thin argparse wrapper around :mod:`samap.io` so the FASTA-preparation
and BLAST steps can be scripted without writing a Python driver.

Examples
--------
::

    samap detect-ids data.h5ad
    samap fetch-proteome data.h5ad -o data.fa
    samap match-fasta data.h5ad transcriptome.fa --gtf ann.gtf -o renamed.fa
    samap blast --species hu hu.fa prot --species mm mm.fa prot \\
        --maps maps/ --threads 16
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from samap import __version__


def _cmd_detect_ids(ns: argparse.Namespace) -> int:
    import anndata as ad

    from samap.io.ids import detect_id_flavor

    a = ad.read_h5ad(ns.h5ad)
    src = a.var[ns.var_key] if ns.var_key else a.var_names
    rep = detect_id_flavor(src, sample=ns.sample)
    print(rep)
    print(rep.counts)
    return 0


def _cmd_fetch_proteome(ns: argparse.Namespace) -> int:
    import anndata as ad

    from samap.io.fetch import fetch_proteome

    a = ad.read_h5ad(ns.h5ad)
    rep = fetch_proteome(
        a,
        ns.out,
        var_key=ns.var_key,
        source=ns.source,
        canonical=not ns.all_isoforms,
    )
    print(rep)
    return 0 if rep.n_fetched else 1


def _cmd_match_fasta(ns: argparse.Namespace) -> int:
    import anndata as ad

    from samap.io.match import match_fasta

    a = ad.read_h5ad(ns.h5ad)
    rep = match_fasta(
        a,
        ns.fasta,
        var_key=ns.var_key,
        gtf=ns.gtf,
        write=ns.out,
    )
    print(rep)
    print(rep.scores.to_string())
    if rep.sample_unmatched:
        print("sample unmatched var_names:", rep.sample_unmatched)
    return 0


def _cmd_blast(ns: argparse.Namespace) -> int:
    from samap.io.blast import run_blast, save_gnnm

    fastas = {sid: (path, kind) for sid, path, kind in ns.species}
    out = run_blast(
        fastas,
        f_maps=ns.maps,
        engine=ns.engine,
        threads=ns.threads,
        evalue=ns.evalue,
        sensitivity=ns.sensitivity,
        gpu=(True if ns.gpu else False if ns.no_gpu else None),
        overwrite=ns.overwrite,
    )
    print(f"maps written under {out}")
    if ns.cache:
        from samap.core.homology import _calculate_blast_graph

        sids = list(fastas)
        g = _calculate_blast_graph(
            sids, f_maps=str(out) + "/", reciprocate=True, eval_thr=ns.evalue
        )
        p = save_gnnm(g, ns.cache)
        print(f"gnnm cache written: {p}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="samap",
        description="SAMap I/O helpers — detect IDs, fetch/reconcile FASTAs, run BLAST.",
    )
    p.add_argument("--version", action="version", version=f"samap {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # detect-ids -----------------------------------------------------------
    s = sub.add_parser("detect-ids", help="Classify var_names identifier namespace.")
    s.add_argument("h5ad", help="Input .h5ad file.")
    s.add_argument("--var-key", default=None, help="adata.var column to use instead of var_names.")
    s.add_argument("--sample", type=int, default=200)
    s.set_defaults(func=_cmd_detect_ids)

    # fetch-proteome -------------------------------------------------------
    s = sub.add_parser(
        "fetch-proteome",
        help="Derive a protein FASTA whose headers are var_names (Ensembl / NCBI GeneID).",
    )
    s.add_argument("h5ad")
    s.add_argument("-o", "--out", required=True, help="Output FASTA path.")
    s.add_argument("--var-key", default=None)
    s.add_argument("--source", choices=["auto", "ensembl", "ncbi"], default="auto")
    s.add_argument("--all-isoforms", action="store_true", help="Write every isoform (default: longest only).")
    s.set_defaults(func=_cmd_fetch_proteome)

    # match-fasta ----------------------------------------------------------
    s = sub.add_parser(
        "match-fasta",
        help="Score header transforms vs var_names; optionally write a renamed FASTA.",
    )
    s.add_argument("h5ad")
    s.add_argument("fasta")
    s.add_argument("--var-key", default=None)
    s.add_argument("--gtf", default=None, help="GTF/GFF for transcript→gene mapping.")
    s.add_argument("-o", "--out", default=None, help="Write renamed FASTA here.")
    s.set_defaults(func=_cmd_match_fasta)

    # blast ----------------------------------------------------------------
    s = sub.add_parser(
        "blast",
        help="Run all N-choose-2 reciprocal alignments (DIAMOND-first).",
    )
    s.add_argument(
        "--species",
        nargs=3,
        action="append",
        metavar=("SID", "FASTA", "TYPE"),
        required=True,
        help="Repeat per species: --species hu hu.fa prot --species mm mm.fa prot",
    )
    s.add_argument("--maps", default="maps/", help="Output maps directory.")
    s.add_argument(
        "--engine",
        choices=["auto", "diamond", "mmseqs", "blast"],
        default="auto",
        help=(
            "auto: DIAMOND for protein DBs, MMseqs2 for nucleotide DBs, "
            "BLAST+ last resort. Install with "
            "`conda install -c bioconda diamond mmseqs2 blast`."
        ),
    )
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--evalue", type=float, default=1e-6)
    s.add_argument(
        "--sensitivity",
        choices=["sensitive", "very-sensitive", "ultra-sensitive"],
        default="very-sensitive",
    )
    g = s.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="Force MMseqs2 --gpu 1.")
    g.add_argument("--no-gpu", action="store_true", help="Disable MMseqs2 GPU even if detected.")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--cache", default=None, help="Also write gnnm.npz cache here.")
    s.set_defaults(func=_cmd_blast)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
