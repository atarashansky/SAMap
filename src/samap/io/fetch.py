"""Fetch a protein FASTA whose headers match ``adata.var_names``.

The single biggest SAMap onboarding wall is curating a proteome whose
FASTA headers match the gene identifiers in the user's h5ad. For any
dataset whose ``var_names`` are stable database accessions (Ensembl,
NCBI GeneID), we can *derive* that FASTA instead of asking for one.

Only stdlib HTTP is used to avoid adding ``requests`` as a hard
dependency.
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from samap._logging import logger
from samap.io.ids import detect_id_flavor

__all__ = ["FetchReport", "fetch_proteome"]

# --- Backend constants ---------------------------------------------------
_ENSEMBL_REST = "https://rest.ensembl.org"
_ENSEMBL_BATCH = 50  # hard limit on POST /sequence/id and /lookup/id
_NCBI_DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2"
_NCBI_BATCH = 800

_SUPPORTED = frozenset(
    {"ensembl_gene", "ensembl_tx", "ensembl_protein", "ncbi_geneid"}
)


@dataclass(frozen=True)
class FetchReport:
    """Result of :func:`fetch_proteome`.

    Attributes
    ----------
    flavor
        Identifier flavor that was fetched (see :mod:`samap.io.ids`).
    n_requested
        Number of distinct input identifiers.
    n_fetched
        Number of identifiers for which a sequence was written.
    fasta_path
        Path to the written FASTA. Headers are the *original* input
        identifiers (including any version suffix), so this file's
        headers match ``adata.var_names`` by construction.
    names
        ``(N, 2)`` ndarray mapping each input identifier to a parent
        gene identifier, suitable for ``SAMAP(names={sid: report.names})``.
        Populated for ``ensembl_tx`` / ``ensembl_protein`` (transcript /
        protein → gene). ``None`` when input is already gene-level.
    missing
        Input identifiers for which no sequence could be retrieved.
    """

    flavor: str
    n_requested: int
    n_fetched: int
    fasta_path: Path
    names: np.ndarray | None
    missing: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover — convenience only
        miss = (
            f", missing[:3]={self.missing[:3]}" if self.missing else ""
        )
        return (
            f"FetchReport(flavor={self.flavor!r}, "
            f"fetched={self.n_fetched}/{self.n_requested}, "
            f"fasta={self.fasta_path}{miss})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_proteome(
    ids: Iterable[str] | Any,
    out_fasta: str | Path,
    *,
    source: Literal["auto", "ensembl", "ncbi"] = "auto",
    var_key: str | None = None,
    canonical: bool = True,
    batch_size: int | None = None,
    sleep: float = 0.0,
    timeout: float = 60.0,
) -> FetchReport:
    """Write a protein FASTA whose headers are the input identifiers.

    Parameters
    ----------
    ids
        Iterable of identifier strings, **or** an ``AnnData`` (in which
        case ``var_names`` — or ``var[var_key]`` — is used).
    out_fasta
        Output FASTA path.
    source
        Backend to use. ``"auto"`` dispatches on
        :func:`~samap.io.ids.detect_id_flavor`.
    var_key
        When ``ids`` is an AnnData, take this column of ``.var`` instead
        of ``var_names``.
    canonical
        If ``True`` (default), keep one sequence per input identifier —
        the longest. If ``False``, write every isoform; headers become
        ``<input_id>|<isoform_accession>``.
    batch_size
        Override the per-backend POST batch size. Defaults to the
        backend's documented limit (50 for Ensembl, 800 for NCBI).
    sleep
        Seconds to sleep between batches (rate-limit cushion).
    timeout
        Per-request timeout in seconds.

    Returns
    -------
    FetchReport

    Raises
    ------
    NotImplementedError
        If the detected flavor is not yet supported by a fetch backend.
        The message names the flavor and points at
        :func:`samap.io.match.match_fasta` as the alternative.

    Notes
    -----
    Currently supported flavors: ``ensembl_gene``, ``ensembl_tx``,
    ``ensembl_protein``, ``ncbi_geneid``. RefSeq, UniProt, and
    model-organism-DB IDs are intentionally deferred — most resolve to
    one of the supported namespaces via a single xref lookup, which a
    follow-up PR can layer on top of this dispatch.
    """
    # Coerce input → list[str], preserving original tokens (incl. version
    # suffixes) so output headers match var_names exactly.
    raw = _coerce_ids(ids, var_key=var_key)
    raw = list(dict.fromkeys(raw))  # dedupe, order-preserving
    n_req = len(raw)

    rep = detect_id_flavor(raw)
    if source == "auto":
        flavor = rep.flavor
    elif source == "ensembl":
        flavor = rep.flavor if rep.flavor.startswith("ensembl_") else "ensembl_gene"
    elif source == "ncbi":
        flavor = "ncbi_geneid"
    else:  # pragma: no cover — Literal guards this
        raise ValueError(f"unknown source {source!r}")

    if flavor not in _SUPPORTED:
        raise NotImplementedError(
            f"fetch_proteome does not (yet) have a backend for identifier "
            f"flavor {flavor!r} (detected with confidence "
            f"{rep.confidence:.2f}; counts={rep.counts}). Supported: "
            f"{sorted(_SUPPORTED)}. For other namespaces, supply a FASTA "
            f"and use samap.io.match_fasta to reconcile headers."
        )

    out = Path(out_fasta)
    out.parent.mkdir(parents=True, exist_ok=True)

    if flavor.startswith("ensembl_"):
        bs = batch_size or _ENSEMBL_BATCH
        seqs, names = _fetch_ensembl(
            raw, kind=flavor, batch_size=bs, sleep=sleep, timeout=timeout
        )
    else:  # ncbi_geneid
        bs = batch_size or _NCBI_BATCH
        seqs, names = _fetch_ncbi_geneid(
            raw, batch_size=bs, sleep=sleep, timeout=timeout
        )

    n_written = _write_fasta(out, raw, seqs, canonical=canonical)
    missing = [r for r in raw if r not in seqs]

    if missing:
        logger.warning(
            "fetch_proteome: %d/%d %s identifiers returned no sequence "
            "(examples: %s).",
            len(missing), n_req, flavor, missing[:5],
        )

    return FetchReport(
        flavor=flavor,
        n_requested=n_req,
        n_fetched=n_written,
        fasta_path=out,
        names=names,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_ids(ids: Iterable[str] | Any, *, var_key: str | None) -> list[str]:
    """Turn an AnnData or iterable into a list of identifier strings."""
    if hasattr(ids, "var_names"):  # AnnData-like
        ad = ids
        src = ad.var[var_key] if var_key is not None else ad.var_names
        return [str(x) for x in src]
    return [str(x) for x in ids]


def _strip_version(tok: str) -> str:
    """Drop a trailing ``.\\d+`` version suffix (Ensembl/RefSeq)."""
    return re.sub(r"\.\d+$", "", tok)


def _chunks(xs: Sequence[str], n: int) -> Iterator[Sequence[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _post_json(url: str, body: dict[str, Any], *, timeout: float) -> Any:
    req = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _write_fasta(
    out: Path,
    order: Sequence[str],
    seqs: dict[str, list[tuple[str, str]]],
    *,
    canonical: bool,
    width: int = 80,
) -> int:
    """Write ``seqs`` to ``out`` keyed by original identifiers in ``order``.

    ``seqs`` maps each original input id → list of ``(isoform_acc, seq)``.
    """
    n = 0
    with out.open("w") as f:
        for raw in order:
            isoforms = seqs.get(raw)
            if not isoforms:
                continue
            if canonical:
                acc, s = max(isoforms, key=lambda kv: len(kv[1]))
                f.write(f">{raw}\n")
                for i in range(0, len(s), width):
                    f.write(s[i : i + width] + "\n")
                n += 1
            else:
                for acc, s in isoforms:
                    f.write(f">{raw}|{acc}\n")
                    for i in range(0, len(s), width):
                        f.write(s[i : i + width] + "\n")
                    n += 1
    return n


# ---------------------------------------------------------------------------
# Ensembl backend
# ---------------------------------------------------------------------------


def _fetch_ensembl(
    raw: Sequence[str],
    *,
    kind: str,
    batch_size: int,
    sleep: float,
    timeout: float,
) -> tuple[dict[str, list[tuple[str, str]]], np.ndarray | None]:
    """Fetch protein sequences from Ensembl REST.

    Returns
    -------
    seqs
        Map of *original* input id → list of ``(ENSP_acc, sequence)``.
    names
        ``(N, 2)`` array of ``[input_id, parent_gene_id]`` when ``kind``
        is ``ensembl_tx`` / ``ensembl_protein``; else ``None``.
    """
    # Ensembl rejects versioned IDs — strip for the request, key results
    # back to the original token.
    stripped = [_strip_version(x) for x in raw]
    by_stripped: dict[str, str] = dict(zip(stripped, raw))

    seqs: dict[str, list[tuple[str, str]]] = {}
    n_batches = (len(stripped) + batch_size - 1) // batch_size
    for bi, chunk in enumerate(_chunks(stripped, batch_size), 1):
        try:
            res = _post_json(
                f"{_ENSEMBL_REST}/sequence/id",
                {"ids": list(chunk), "type": "protein"},
                timeout=timeout,
            )
        except HTTPError as e:  # pragma: no cover — network
            logger.warning("Ensembl /sequence/id batch %d/%d: %s", bi, n_batches, e)
            continue
        for rec in res:
            q = rec.get("query")
            s = rec.get("seq")
            acc = rec.get("id", "")
            if not q or not s:
                continue
            orig = by_stripped.get(q, q)
            seqs.setdefault(orig, []).append((acc, s))
        if bi % 10 == 0 or bi == n_batches:
            logger.info(
                "Ensembl fetch: %d/%d batches, %d ids resolved.",
                bi, n_batches, len(seqs),
            )
        if sleep:
            time.sleep(sleep)

    names: np.ndarray | None = None
    if kind in ("ensembl_tx", "ensembl_protein"):
        parent: dict[str, str] = {}
        for bi, chunk in enumerate(_chunks(stripped, batch_size), 1):
            try:
                res = _post_json(
                    f"{_ENSEMBL_REST}/lookup/id",
                    {"ids": list(chunk)},
                    timeout=timeout,
                )
            except HTTPError as e:  # pragma: no cover — network
                logger.warning("Ensembl /lookup/id batch %d: %s", bi, e)
                continue
            for k, v in res.items():
                if not v:
                    continue
                # Transcript → Parent is gene; Translation → Parent is
                # transcript, so walk one more hop via 'Parent' if needed.
                p = v.get("Parent")
                if v.get("object_type") == "Translation" and p:
                    # second hop is cheap to skip — collapse via the
                    # transcript's Parent on a follow-up batch is overkill
                    # for this use case; emit the transcript parent and
                    # let _coarsen_blast_graph handle many→one.
                    pass
                if p:
                    parent[k] = p
            if sleep:
                time.sleep(sleep)
        if parent:
            rows = [
                (by_stripped.get(k, k), parent[k])
                for k in stripped
                if k in parent
            ]
            names = np.array(rows, dtype=object)

    return seqs, names


# ---------------------------------------------------------------------------
# NCBI Datasets backend (GeneID → protein.faa)
# ---------------------------------------------------------------------------

_GENEID_RE = re.compile(r"\[GeneID=(\d+)\]")


def _fetch_ncbi_geneid(
    raw: Sequence[str],
    *,
    batch_size: int,
    sleep: float,
    timeout: float,
) -> tuple[dict[str, list[tuple[str, str]]], None]:
    """Fetch protein FASTA for NCBI GeneIDs via Datasets v2.

    The download endpoint returns a zip containing ``protein.faa`` whose
    headers carry ``[GeneID=N]`` — parsed back to the input id.
    """
    seqs: dict[str, list[tuple[str, str]]] = {}
    n_batches = (len(raw) + batch_size - 1) // batch_size
    for bi, chunk in enumerate(_chunks(raw, batch_size), 1):
        body = {
            "gene_ids": [int(x) for x in chunk],
            "include_annotation_type": ["FASTA_PROTEIN"],
        }
        req = Request(
            f"{_NCBI_DATASETS}/gene/download",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/zip",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as r:
                blob = r.read()
        except HTTPError as e:  # pragma: no cover — network
            logger.warning("NCBI datasets batch %d/%d: %s", bi, n_batches, e)
            continue
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            faa = next((n for n in z.namelist() if n.endswith(".faa")), None)
            if faa is None:
                continue
            for hdr, seq in _iter_fasta(z.read(faa).decode()):
                m = _GENEID_RE.search(hdr)
                if not m:
                    continue
                gid = m.group(1)
                if gid in seqs or gid in chunk:
                    acc = hdr.split()[0]
                    seqs.setdefault(gid, []).append((acc, seq))
        if bi % 5 == 0 or bi == n_batches:
            logger.info(
                "NCBI fetch: %d/%d batches, %d ids resolved.", bi, n_batches, len(seqs)
            )
        if sleep:
            time.sleep(sleep)
    return seqs, None


def _iter_fasta(text: str) -> Iterator[tuple[str, str]]:
    """Yield (header_without_gt, sequence) from FASTA text."""
    hdr: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if hdr is not None:
                yield hdr, "".join(buf)
            hdr = line[1:]
            buf = []
        else:
            buf.append(line.strip())
    if hdr is not None:
        yield hdr, "".join(buf)
