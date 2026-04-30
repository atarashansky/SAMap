"""Reconcile FASTA headers with ``adata.var_names``.

When a user must bring their own FASTA (PlanMine, AEP hydra, de-novo
Trinity assembly), the headers almost never match ``var_names`` exactly.
:func:`match_fasta` runs a small cascade of header→ID transforms, scores
each by overlap fraction against the var_names, picks the best, and
optionally writes a renamed FASTA whose headers *are* the var_names.

The transforms are deliberately conservative and individually invertible
where possible, so the user can see *which* transform won and trust it.
For the common transcript-FASTA / gene-var_names case, supply a GTF and
the transcript→gene map is built automatically and returned as a
``names`` array suitable for ``SAMAP(names={sid: report.names})``.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from samap._logging import logger
from samap.io.fetch import _iter_fasta  # re-use the simple FASTA reader

__all__ = ["MatchReport", "match_fasta"]

# ---------------------------------------------------------------------------
# Header → identifier transforms
# ---------------------------------------------------------------------------
# Each transform takes the *full* header line (without the leading '>')
# and returns a candidate identifier, or None if it doesn't apply. The
# cascade is scored independently — the best one is reported, not the
# first one to match.

_VERSION_RE = re.compile(r"\.\d+$")
# `gene:` / `gene=` only — `gene_` is NOT a kv separator (it appears in
# `gene_id`, `gene_biotype`, etc.). `gene_id` / `geneid` is matched
# separately and tried first.
_KV_GENEID_RE = re.compile(r"\bgene_?id[:=]([\w.\-]+)", re.IGNORECASE)
_KV_GENE_RE = re.compile(r"\bgene[:=]([\w.\-]+)", re.IGNORECASE)
_KV_TRANSCRIPT_RE = re.compile(r"\btranscript(?:_id)?[:=]([\w.\-]+)", re.IGNORECASE)
_UNIPROT_PIPE_RE = re.compile(r"^(?:sp|tr)\|([^|]+)\|")
_NCBI_LCL_RE = re.compile(r"^lcl\|")


def _t_identity(h: str) -> str | None:
    return h


def _t_first_token(h: str) -> str | None:
    return h.split()[0] if h else None


def _t_strip_version(h: str) -> str | None:
    tok = h.split()[0]
    return _VERSION_RE.sub("", tok) if _VERSION_RE.search(tok) else None


def _t_uniprot_pipe(h: str) -> str | None:
    m = _UNIPROT_PIPE_RE.match(h)
    return m.group(1) if m else None


def _t_ncbi_lcl(h: str) -> str | None:
    if _NCBI_LCL_RE.match(h):
        return _NCBI_LCL_RE.sub("", h.split()[0])
    return None


def _t_kv_gene(h: str) -> str | None:
    m = _KV_GENEID_RE.search(h) or _KV_GENE_RE.search(h)
    return m.group(1) if m else None


def _t_kv_transcript(h: str) -> str | None:
    m = _KV_TRANSCRIPT_RE.search(h)
    return m.group(1) if m else None


def _t_after_last_pipe(h: str) -> str | None:
    tok = h.split()[0]
    return tok.rsplit("|", 1)[-1] if "|" in tok else None


def _t_before_first_pipe(h: str) -> str | None:
    tok = h.split()[0]
    return tok.split("|", 1)[0] if "|" in tok else None


#: Built-in transform cascade. Order is for reporting only — every
#: transform is scored and the best wins.
TRANSFORMS: dict[str, Callable[[str], str | None]] = {
    "identity": _t_identity,
    "first_token": _t_first_token,
    "strip_version": _t_strip_version,
    "uniprot_pipe": _t_uniprot_pipe,
    "ncbi_lcl": _t_ncbi_lcl,
    "kv_gene": _t_kv_gene,
    "kv_transcript": _t_kv_transcript,
    "after_last_pipe": _t_after_last_pipe,
    "before_first_pipe": _t_before_first_pipe,
}


@dataclass(frozen=True)
class MatchReport:
    """Result of :func:`match_fasta`.

    Attributes
    ----------
    transform
        Name of the winning transform (a key of :data:`TRANSFORMS`, or a
        user-supplied name, or ``"gtf"`` / ``"mapping"``).
    n_var
        Number of var_names.
    n_fasta
        Number of FASTA records.
    n_matched
        var_names that have at least one FASTA record after the transform.
    frac
        ``n_matched / n_var``.
    scores
        Per-transform DataFrame: columns ``[n_matched, frac, example_in,
        example_out]`` indexed by transform name. Sorted by ``frac``
        descending. This is the diagnostic to show the user.
    names
        ``(M, 2)`` array of ``[fasta_header_first_token, var_name]`` for
        the winning transform. Feed to ``SAMAP(names={sid: report.names})``
        when you want to use the *original* FASTA for BLAST.
    out_fasta
        Path to the renamed FASTA (headers = var_names), or ``None`` if
        ``write=False``.
    sample_unmatched
        Up to 5 example var_names with no FASTA match under the winning
        transform.
    """

    transform: str
    n_var: int
    n_fasta: int
    n_matched: int
    frac: float
    scores: pd.DataFrame
    names: np.ndarray | None
    out_fasta: Path | None
    sample_unmatched: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover — convenience only
        return (
            f"MatchReport(transform={self.transform!r}, "
            f"matched={self.n_matched}/{self.n_var} ({100*self.frac:.1f}%), "
            f"fasta_records={self.n_fasta})"
        )


# ---------------------------------------------------------------------------
# GTF transcript_id → gene_id parser
# ---------------------------------------------------------------------------

_GTF_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')


def _gtf_tx2gene(path: str | Path) -> dict[str, str]:
    """Parse a GTF/GFF and return ``{transcript_id: gene_id}``.

    Only the attribute column is inspected; both GTF (``key "val";``) and
    GFF3 (``key=val;``) styles are handled.
    """
    out: dict[str, str] = {}
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = cols[8]
            tx = gid = None
            # GTF style
            for k, v in _GTF_ATTR_RE.findall(attrs):
                if k == "transcript_id":
                    tx = v
                elif k == "gene_id":
                    gid = v
            # GFF3 style
            if tx is None or gid is None:
                for kv in attrs.split(";"):
                    if "=" in kv:
                        k, _, v = kv.strip().partition("=")
                        if k.lower() in ("transcript_id", "id") and tx is None:
                            tx = v
                        elif k.lower() in ("gene_id", "parent") and gid is None:
                            gid = v.removeprefix("gene:")
            if tx and gid:
                out[tx] = gid
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_fasta(
    adata: Any | Iterable[str],
    fasta: str | Path,
    *,
    var_key: str | None = None,
    gtf: str | Path | None = None,
    mapping: dict[str, str] | None = None,
    extra_transforms: dict[str, Callable[[str], str | None]] | None = None,
    write: str | Path | None = None,
) -> MatchReport:
    """Score FASTA-header transforms against var_names and pick the best.

    Parameters
    ----------
    adata
        ``AnnData`` (uses ``var_names`` or ``var[var_key]``) or any
        iterable of identifier strings.
    fasta
        Path to a (optionally gzipped) FASTA file.
    var_key
        Column of ``adata.var`` to use instead of ``var_names``.
    gtf
        Optional GTF/GFF path. If given, a ``"gtf"`` transform is added:
        ``first_token`` → strip version → ``transcript_id → gene_id``.
        This is the canonical fix for "Cell Ranger keyed on gene_id,
        FASTA keyed on transcript_id, both from the same GTF."
    mapping
        Optional explicit ``{header_first_token: var_name}`` map. Added
        as a ``"mapping"`` transform.
    extra_transforms
        Additional header→id callables to score alongside the built-ins.
    write
        If given, write a renamed FASTA at this path: one record per
        matched var_name (longest sequence wins on collisions), header =
        var_name.

    Returns
    -------
    MatchReport
    """
    var = _coerce_var(adata, var_key=var_key)
    var_set = set(var)
    n_var = len(var)

    # Read FASTA once into memory — proteomes are small enough.
    p = Path(fasta)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        records = list(_iter_fasta(f.read()))
    n_fa = len(records)
    headers = [h for h, _ in records]

    # Build the candidate transform set.
    cands: dict[str, Callable[[str], str | None]] = dict(TRANSFORMS)
    if extra_transforms:
        cands.update(extra_transforms)
    if mapping:
        m = dict(mapping)
        cands["mapping"] = lambda h: m.get(h.split()[0])
    if gtf is not None:
        tx2g = _gtf_tx2gene(gtf)
        cands["gtf"] = lambda h: tx2g.get(_VERSION_RE.sub("", h.split()[0]))

    # Score: for each transform, how many var_names are covered?
    rows = []
    best: tuple[str, dict[str, list[int]]] | None = None  # (name, var_name → [record idx])
    for name, fn in cands.items():
        applied = [fn(h) for h in headers]
        hit: dict[str, list[int]] = {}
        ex_in = ex_out = None
        for i, a in enumerate(applied):
            if a is None:
                continue
            if ex_in is None:
                ex_in, ex_out = headers[i], a
            if a in var_set:
                hit.setdefault(a, []).append(i)
        n_match = len(hit)
        rows.append((name, n_match, n_match / n_var if n_var else 0.0, ex_in, ex_out))
        if best is None or n_match > len(best[1]):
            best = (name, hit)

    scores = (
        pd.DataFrame(rows, columns=["transform", "n_matched", "frac", "example_in", "example_out"])
        .set_index("transform")
        .sort_values("frac", ascending=False)
    )
    logger.info("match_fasta: top transforms\n%s", scores.head(5).to_string())

    assert best is not None
    win_name, win_hit = best
    matched_var = list(win_hit.keys())
    n_match = len(matched_var)

    # names array: original FASTA first-token → var_name (one row per record
    # that mapped). Suitable for SAMAP(names=...) if the user keeps the
    # original FASTA for BLAST.
    names_rows: list[tuple[str, str]] = []
    for v, idxs in win_hit.items():
        for i in idxs:
            names_rows.append((headers[i].split()[0], v))
    names_arr = np.array(names_rows, dtype=object) if names_rows else None

    out_path: Path | None = None
    if write is not None:
        out_path = Path(write)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for v in var:
                idxs = win_hit.get(v)
                if not idxs:
                    continue
                # longest sequence wins on collision
                i = max(idxs, key=lambda j: len(records[j][1]))
                seq = records[i][1]
                f.write(f">{v}\n")
                for k in range(0, len(seq), 80):
                    f.write(seq[k : k + 80] + "\n")
        logger.info("match_fasta: wrote %d records → %s", n_match, out_path)

    unmatched = [v for v in var if v not in win_hit][:5]

    return MatchReport(
        transform=win_name,
        n_var=n_var,
        n_fasta=n_fa,
        n_matched=n_match,
        frac=n_match / n_var if n_var else 0.0,
        scores=scores,
        names=names_arr,
        out_fasta=out_path,
        sample_unmatched=unmatched,
    )


def _coerce_var(adata: Any | Iterable[str], *, var_key: str | None) -> list[str]:
    if hasattr(adata, "var_names"):
        src = adata.var[var_key] if var_key is not None else adata.var_names
        return [str(x) for x in src]
    return [str(x) for x in adata]
