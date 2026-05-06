"""Build a SAMap homology graph (``gnnm``) without running BLAST.

``SAMAP(..., gnnm=(matrix, gns, gns_dict))`` bypasses BLAST-table parsing
entirely. This module provides builders that produce that tuple from
ortholog sources other than reciprocal BLAST:

- :func:`gnnm_from_pairs` — the primitive. Turn any ``(N, 2)`` table of
  cross-species gene pairs into the ``(csr_matrix, gns, gns_dict)`` tuple.
  Use this when you already have orthologs from BioMart, OrthoDB, OMA, or
  a local pipeline.
- :func:`homology_from_eggnog` — parse eggNOG-mapper annotation TSVs and
  build the graph from orthogroup (OG) co-membership at a chosen taxon
  level. Works for *any* species emapper covers, including non-model
  organisms with no stable gene IDs (emapper takes raw sequences).

Ensembl Compara is intentionally **not** wrapped here: pulling all
orthologs for a species pair via the REST ``/homology`` endpoint is one
request per gene; the practical path is a single BioMart bulk export
(``*_homolog_ensembl_gene`` + ``*_homolog_perc_id``) which the user can
feed to :func:`gnnm_from_pairs` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from samap._logging import logger

__all__ = ["gnnm_from_pairs", "homology_from_eggnog"]

GnnmTuple = tuple[sp.csr_matrix, np.ndarray, dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Primitive: pairs → gnnm
# ---------------------------------------------------------------------------


def gnnm_from_pairs(
    pairs: Iterable[Sequence[str]],
    *,
    ids: dict[str, Iterable[str]] | None = None,
    weights: Iterable[float] | None = None,
    prefix: bool = True,
    symmetric: bool = True,
) -> GnnmTuple:
    """Build a ``(gnnm, gns, gns_dict)`` tuple from cross-species gene pairs.

    Parameters
    ----------
    pairs
        Iterable of ``(gene_a, gene_b)`` rows. Genes may be either
        already prefixed (``"hu_SOX2"``) or bare — see ``ids`` / ``prefix``.
    ids
        Optional ``{species_id: iterable_of_genes}`` declaring which
        genes belong to which species. If given and ``prefix`` is True,
        every gene appearing in ``pairs`` is prefixed with ``"<sid>_"``
        according to membership. If ``ids`` is None, genes in ``pairs``
        **must** already carry a ``"<sid>_"`` prefix and species are
        inferred from the token before the first underscore.
    weights
        Optional per-pair edge weights in ``(0, 1]``. Default ``1.0``.
    prefix
        Whether to prefix genes from ``ids`` (default True). Set False
        when ``pairs`` are already prefixed and ``ids`` is provided only
        to fix the species set.
    symmetric
        If True (default), add the reverse edge for every pair so the
        resulting graph is symmetric — SAMap expects this.

    Returns
    -------
    tuple
        ``(gnnm, gns, gns_dict)`` exactly as ``SAMAP(gnnm=...)`` consumes:

        - ``gnnm``: ``(G, G)`` CSR matrix of edge weights
        - ``gns``: length-G ndarray of prefixed gene names (the row/col index)
        - ``gns_dict``: ``{sid: ndarray}`` partitioning ``gns`` by species

    Examples
    --------
    >>> pairs = [("SOX2", "Sox2"), ("TP53", "Trp53")]
    >>> gnnm, gns, gd = gnnm_from_pairs(
    ...     pairs, ids={"hu": ["SOX2", "TP53"], "mm": ["Sox2", "Trp53"]}
    ... )
    >>> gnnm.shape
    (4, 4)
    >>> sorted(gd)
    ['hu', 'mm']
    """
    pairs_list = [tuple(p) for p in pairs]
    if not pairs_list:
        raise ValueError("pairs is empty")
    if any(len(p) != 2 for p in pairs_list):
        raise ValueError("each pair must have exactly two elements")

    w = (
        np.ones(len(pairs_list), dtype=np.float32)
        if weights is None
        else np.asarray(list(weights), dtype=np.float32)
    )
    if w.shape[0] != len(pairs_list):
        raise ValueError("weights length must match pairs length")

    if ids is not None:
        gene2sid: dict[str, str] = {}
        for sid, genes in ids.items():
            for g in genes:
                gene2sid[str(g)] = sid
        if prefix:
            pairs_list = [
                (f"{gene2sid.get(a, '?')}_{a}", f"{gene2sid.get(b, '?')}_{b}")
                for a, b in pairs_list
            ]
            unknown = [g for g in {x for p in pairs_list for x in p} if g.startswith("?_")]
            if unknown:
                raise ValueError(
                    f"{len(unknown)} genes in `pairs` not found in any `ids` "
                    f"species (examples: {unknown[:5]})."
                )
        sids = list(ids)
    else:
        # Infer species from prefix before first underscore.
        sids = sorted({g.split("_", 1)[0] for p in pairs_list for g in p})

    a = np.array([p[0] for p in pairs_list])
    b = np.array([p[1] for p in pairs_list])
    gns = np.unique(np.concatenate([a, b]))
    idx = pd.Index(gns)
    ia = idx.get_indexer(a)
    ib = idx.get_indexer(b)

    if symmetric:
        rows = np.concatenate([ia, ib])
        cols = np.concatenate([ib, ia])
        data = np.concatenate([w, w])
    else:
        rows, cols, data = ia, ib, w

    gnnm = sp.coo_matrix((data, (rows, cols)), shape=(gns.size, gns.size)).tocsr()
    # Collapse duplicate edges to their max weight (coo→csr sums them).
    gnnm.sum_duplicates()
    gnnm.data = np.minimum(gnnm.data, 1.0)

    sps = np.array([g.split("_", 1)[0] for g in gns])
    gns_dict = {sid: gns[sps == sid] for sid in sids}

    logger.info(
        "gnnm_from_pairs: %d genes across %d species, %d edges (%s).",
        gns.size, len(sids), gnnm.nnz,
        "symmetric" if symmetric else "directed",
    )
    return gnnm, gns, gns_dict


# ---------------------------------------------------------------------------
# eggNOG-mapper → gnnm
# ---------------------------------------------------------------------------


def _parse_og_at_taxon(cell: Any, taxon: str) -> str | None:
    """Extract the OG id at ``@<taxon>`` from an ``eggNOG_OGs`` cell.

    The cell is a comma-separated list like
    ``"38ERC@33154,3NUD8@4751,KOG2877@2759"``. Returns the OG token whose
    ``@`` suffix matches ``taxon``, or ``None``.
    """
    if not isinstance(cell, str):
        return None
    suffix = "@" + taxon
    for tok in cell.split(","):
        tok = tok.strip()
        if tok.endswith(suffix):
            return tok[: -len(suffix)]
    return None


def homology_from_eggnog(
    tsvs: dict[str, str | PathLike[str] | pd.DataFrame],
    *,
    taxon: int | str = 2759,
    og_key: str = "eggNOG_OGs",
    max_og_size: int | None = 200,
) -> GnnmTuple:
    """Build a homology graph from eggNOG-mapper annotation TSVs.

    Genes that share an orthogroup (OG) at the given ``taxon`` level get
    an edge of weight 1.0. This is coarser than BLAST bitscore but works
    for *any* species emapper has been run on — including non-model
    organisms with no stable gene IDs.

    Parameters
    ----------
    tsvs
        ``{species_id: path_or_DataFrame}``. Each TSV must have the
        query/gene id in column 0 and an ``og_key`` column (default
        ``"eggNOG_OGs"`` — emapper's standard output).
    taxon
        NCBI taxon id at which to take OG membership. Default 2759
        (Eukaryota). Common alternatives: 33208 (Metazoa), 33154
        (Opisthokonta), 7742 (Vertebrata).
    og_key
        Column holding the comma-separated ``OG@taxon`` list.
    max_og_size
        Skip OGs whose total membership across all species exceeds this
        — very large groups (e.g. zinc-fingers) generate O(n²) edges of
        low specificity. ``None`` disables the cap.

    Returns
    -------
    tuple
        ``(gnnm, gns, gns_dict)`` for ``SAMAP(gnnm=...)``.
    """
    taxon_s = str(taxon)
    sids = list(tsvs)

    # Load: per-species map gene → OG
    by_og: dict[str, dict[str, list[str]]] = {}  # OG → {sid: [genes]}
    n_genes: dict[str, int] = {}
    for sid, src in tsvs.items():
        df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, sep="\t", index_col=0)
        if og_key not in df.columns:
            raise KeyError(
                f"column {og_key!r} not found in eggnog table for {sid!r}; "
                f"available: {list(df.columns)[:10]}..."
            )
        n = 0
        for gene, cell in df[og_key].items():
            og = _parse_og_at_taxon(cell, taxon_s)
            if og is None:
                continue
            by_og.setdefault(og, {}).setdefault(sid, []).append(f"{sid}_{gene}")
            n += 1
        n_genes[sid] = n
        logger.info("eggnog[%s]: %d/%d genes have an OG @%s", sid, n, len(df), taxon_s)

    # Emit cross-species pairs (skip within-species).
    pairs: list[tuple[str, str]] = []
    n_skipped = 0
    for og, members in by_og.items():
        total = sum(len(v) for v in members.values())
        if max_og_size is not None and total > max_og_size:
            n_skipped += 1
            continue
        ks = [k for k in sids if k in members]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                for ga in members[ks[i]]:
                    for gb in members[ks[j]]:
                        pairs.append((ga, gb))

    if n_skipped:
        logger.info(
            "eggnog: skipped %d OGs with > %d members (low-specificity).",
            n_skipped, max_og_size,
        )
    if not pairs:
        raise ValueError(
            f"No cross-species OG co-memberships found at taxon {taxon_s}. "
            f"Try a broader taxon (e.g. 33154 Opisthokonta or 1 root)."
        )

    return gnnm_from_pairs(pairs, prefix=False)
