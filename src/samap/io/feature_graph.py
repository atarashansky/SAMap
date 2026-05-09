"""Feature-graph builders for non-BLAST homology priors.

SAMap's iteration is a feature-graph-guided manifold alignment: it consumes
any symmetric ``(gnnm, gns, gns_dict)`` tuple over disjoint feature spaces,
not specifically a BLAST bitscore graph over protein-coding genes. This
module provides builders for feature graphs that are *not* derived from
sequence alignment:

- :func:`gnnm_from_gtf` — build a peak↔gene graph from a GTF annotation,
  for scATAC↔scRNA alignment with peaks as native ATAC features.
- :func:`compose_gnnm` — chain two feature graphs through a shared
  feature-space block, e.g. peak→gene(species A) ∘ ortholog(A→B) for
  cross-species ATAC↔RNA.
- :func:`prepare_atac_sam` — wrap a cells × peaks AnnData in a SAM object
  with TF-IDF + LSI loadings, so ``SAMAP`` can consume it without running
  the log-CPM/dispersion preprocessing that assumes count-like RNA features.

All builders return the same ``GnnmTuple`` shape that
:func:`samap.io.homology.gnnm_from_pairs` produces and ``SAMAP(gnnm=...)``
consumes.
"""

from __future__ import annotations

import re
from os import PathLike
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from samap._logging import logger
from samap.io.homology import GnnmTuple, gnnm_from_pairs

if TYPE_CHECKING:
    from anndata import AnnData

    from samap.sam import SAM

__all__ = ["compose_gnnm", "gnnm_from_gtf", "prepare_atac_sam"]


# ---------------------------------------------------------------------------
# Peak-interval parsing
# ---------------------------------------------------------------------------

# Accepts "chr1:1000-2000", "chr1-1000-2000", "chr1_1000_2000".
_PEAK_RE = re.compile(r"^(.+?)[:\-_](\d+)[\-_](\d+)$")


def _parse_peak_intervals(adata: AnnData) -> pd.DataFrame:
    """Extract (Chromosome, Start, End, peak) from an ATAC AnnData.

    Tries, in order: explicit ``var`` columns (any of ``chrom``/``chr``/
    ``Chromosome`` + ``start``/``Start`` + ``end``/``End``), then parses
    ``var_names`` of the form ``"chr:start-end"`` (or ``-``/``_``-delimited).
    """
    var = adata.var
    chrom_col = next((c for c in ("Chromosome", "chrom", "chr", "seqnames") if c in var), None)
    start_col = next((c for c in ("Start", "start", "chromStart") if c in var), None)
    end_col = next((c for c in ("End", "end", "chromEnd") if c in var), None)

    if chrom_col and start_col and end_col:
        df = pd.DataFrame(
            {
                "Chromosome": var[chrom_col].astype(str).values,
                "Start": var[start_col].astype(int).values,
                "End": var[end_col].astype(int).values,
                "peak": np.asarray(adata.var_names),
            }
        )
        return df

    names = np.asarray(adata.var_names)
    chroms = np.empty(names.size, dtype=object)
    starts = np.empty(names.size, dtype=np.int64)
    ends = np.empty(names.size, dtype=np.int64)
    bad: list[str] = []
    for i, n in enumerate(names):
        m = _PEAK_RE.match(str(n))
        if m is None:
            bad.append(str(n))
            chroms[i] = ""
            starts[i] = -1
            ends[i] = -1
        else:
            chroms[i] = m.group(1)
            starts[i] = int(m.group(2))
            ends[i] = int(m.group(3))
    if bad:
        if len(bad) == names.size:
            raise ValueError(
                "Could not parse peak coordinates from var_names and no "
                "chrom/start/end columns found in adata.var. Expected "
                "var_names like 'chr1:1000-2000'; got e.g. "
                f"{bad[:3]}."
            )
        logger.warning(
            "gnnm_from_gtf: %d/%d peak var_names did not parse as "
            "'chr:start-end' (examples: %s); these peaks will be dropped.",
            len(bad),
            names.size,
            bad[:3],
        )
    df = pd.DataFrame({"Chromosome": chroms, "Start": starts, "End": ends, "peak": names})
    return df[df["Start"] >= 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# GTF parsing → gene table with TSS
# ---------------------------------------------------------------------------


def _read_gene_tss(gtf: str | PathLike[str] | pd.DataFrame) -> pd.DataFrame:
    """Read a GTF and return one row per gene with its TSS.

    Returns columns ``Chromosome, Start, End, Strand, gene_id, tss`` where
    ``Start``/``End`` are the gene-body extent and ``tss`` is the strand-aware
    transcription start (gene ``Start`` on ``+``, ``End`` on ``-``).

    Uses :mod:`pyranges` if available; otherwise falls back to a minimal
    line reader that extracts only ``feature == "gene"`` rows.
    """
    if isinstance(gtf, pd.DataFrame):
        genes = gtf.copy()
    else:
        try:
            import pyranges as pr

            gr = pr.read_gtf(str(gtf))
            df = gr.df if hasattr(gr, "df") else pd.DataFrame(gr)
            genes = df[df["Feature"] == "gene"][
                ["Chromosome", "Start", "End", "Strand", "gene_id"]
            ].copy()
        except ImportError:
            genes = _read_gtf_minimal(gtf)

    if genes.empty:
        raise ValueError(f"No 'gene' features found in GTF {gtf!r}.")

    genes = genes.drop_duplicates("gene_id").reset_index(drop=True)
    genes["Chromosome"] = genes["Chromosome"].astype(str)
    genes["Start"] = genes["Start"].astype(int)
    genes["End"] = genes["End"].astype(int)
    genes["Strand"] = genes["Strand"].astype(str)
    genes["tss"] = np.where(genes["Strand"] == "-", genes["End"], genes["Start"])
    return genes


def _read_gtf_minimal(path: str | PathLike[str]) -> pd.DataFrame:
    """Tiny GTF reader: only ``feature == 'gene'`` rows, only the columns we need."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    rows: list[tuple[str, int, int, str, str]] = []
    gid_re = re.compile(r'gene_id\s+"([^"]+)"')
    with opener(path, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = gid_re.search(f[8])
            if m is None:
                continue
            rows.append((f[0], int(f[3]) - 1, int(f[4]), f[6], m.group(1)))
    return pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "Strand", "gene_id"])


# ---------------------------------------------------------------------------
# Peak ↔ gene feature-graph builder
# ---------------------------------------------------------------------------


def gnnm_from_gtf(
    atac: AnnData,
    rna: AnnData,
    gtf: str | PathLike[str] | pd.DataFrame,
    *,
    ids: tuple[str, str] = ("at", "rn"),
    kind: Literal["tss_window", "powerlaw", "coaccess"] = "tss_window",
    upstream: int = 2_000,
    downstream: int = 0,
    window: int = 150_000,
    gamma: float = 0.87,
    coaccess: pd.DataFrame | None = None,
) -> GnnmTuple:
    """Build a peak↔gene feature graph from a GTF annotation.

    Produces the ``(gnnm, gns, gns_dict)`` tuple for ``SAMAP(gnnm=...)``
    with peaks as the ATAC-side features and genes as the RNA-side
    features. Edges and weights depend on ``kind``:

    - ``'tss_window'`` — peak overlaps gene-body ∪ [TSS − ``upstream``,
      TSS + ``downstream``]; weight 1.0. This is the Signac/Seurat
      ``GeneActivity`` rule.
    - ``'powerlaw'`` — peak within ±``window`` of a gene's TSS; weight
      :math:`(1 + d)^{-\\gamma}` where *d* is the peak-centre→TSS distance
      in bp. Mirrors GLUE's ``rna_anchored_guidance_graph`` (default
      ``window=150_000``, ``gamma=0.87``).
    - ``'coaccess'`` — edges are taken from ``coaccess`` (a Cicero /
      Signac ``LinkPeaks`` table with columns ``peak``, ``gene``,
      ``score``); the GTF is unused. Strongest prior, fewest edges.

    Parameters
    ----------
    atac
        Cells × peaks AnnData. Peak coordinates are read from
        ``var_names`` (``"chr:start-end"``) or from ``var`` columns
        ``chrom``/``start``/``end``.
    rna
        Cells × genes AnnData. Only ``var_names`` are used, to restrict
        the gene side of the graph to genes actually measured.
    gtf
        Path to a GTF/GFF file (optionally gzipped), or a DataFrame with
        columns ``Chromosome, Start, End, Strand, gene_id``.
    ids
        Two-letter dataset IDs ``(atac_id, rna_id)`` used as the SAMap
        species/dataset prefixes. Default ``("at", "rn")``.
    kind
        Edge rule, see above.
    upstream, downstream
        Promoter window around the TSS for ``kind='tss_window'``
        (bp upstream / downstream of the TSS, strand-aware). Default
        2000 / 0, matching Signac.
    window, gamma
        Power-law parameters for ``kind='powerlaw'``.
    coaccess
        Cicero/LinkPeaks output for ``kind='coaccess'``: a DataFrame with
        columns ``peak``, ``gene``, ``score`` (in ``(0, 1]``).

    Returns
    -------
    tuple
        ``(gnnm, gns, gns_dict)`` exactly as ``SAMAP(gnnm=...)`` consumes.
        Feature names are prefixed ``"<atac_id>_<peak>"`` /
        ``"<rna_id>_<gene>"``.
    """
    atac_id, rna_id = ids
    rna_genes = set(map(str, rna.var_names))

    if kind == "coaccess":
        if coaccess is None:
            raise ValueError("kind='coaccess' requires `coaccess=` DataFrame.")
        df = coaccess.rename(columns={c: c.lower() for c in coaccess.columns})
        for col in ("peak", "gene", "score"):
            if col not in df.columns:
                raise ValueError(
                    f"`coaccess` must have columns 'peak', 'gene', 'score'; missing {col!r}."
                )
        atac_peaks = set(map(str, atac.var_names))
        df = df[df["peak"].astype(str).isin(atac_peaks) & df["gene"].astype(str).isin(rna_genes)]
        pairs = df[["peak", "gene"]].astype(str).to_numpy()
        weights = df["score"].to_numpy(dtype=float)
    else:
        peaks = _parse_peak_intervals(atac)
        genes = _read_gene_tss(gtf)
        genes = genes[genes["gene_id"].astype(str).isin(rna_genes)].reset_index(drop=True)
        if genes.empty:
            raise ValueError(
                "No GTF gene_id matched rna.var_names. Check that the GTF "
                "uses the same gene-ID namespace as the RNA AnnData "
                f"(examples from GTF: {list(_read_gene_tss(gtf)['gene_id'][:3])}; "
                f"from rna.var_names: {list(rna.var_names[:3])})."
            )
        pairs, weights = _peak_gene_edges(
            peaks,
            genes,
            kind=kind,
            upstream=upstream,
            downstream=downstream,
            window=window,
            gamma=gamma,
        )

    if pairs.size == 0:
        raise ValueError(
            f"gnnm_from_gtf produced zero peak↔gene edges (kind={kind!r}). "
            "Check chromosome naming ('chr1' vs '1') matches between "
            "atac.var_names and the GTF."
        )

    logger.info(
        "gnnm_from_gtf[%s]: %d edges over %d peaks x %d genes.",
        kind,
        pairs.shape[0],
        np.unique(pairs[:, 0]).size,
        np.unique(pairs[:, 1]).size,
    )

    return gnnm_from_pairs(
        pairs,
        ids={atac_id: list(atac.var_names), rna_id: list(rna.var_names)},
        weights=weights,
    )


def _peak_gene_edges(
    peaks: pd.DataFrame,
    genes: pd.DataFrame,
    *,
    kind: str,
    upstream: int,
    downstream: int,
    window: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (pairs, weights) for the genomic-overlap edge rules.

    Uses :mod:`pyranges` for the interval join when available, else a
    pandas groupby-per-chromosome fallback (O(n log n) per chromosome).
    """
    if kind == "tss_window":
        # Target interval = gene body ∪ strand-aware promoter window.
        plus = genes["Strand"] != "-"
        ext_start = np.where(plus, genes["Start"] - upstream, genes["Start"] - downstream)
        ext_end = np.where(plus, genes["End"] + downstream, genes["End"] + upstream)
        tgt = pd.DataFrame(
            {
                "Chromosome": genes["Chromosome"].values,
                "Start": np.maximum(ext_start, 0),
                "End": ext_end,
                "gene_id": genes["gene_id"].values,
            }
        )
        hits = _overlap_join(peaks, tgt)
        return (
            hits[["peak", "gene_id"]].to_numpy(),
            np.ones(len(hits), dtype=float),
        )

    if kind == "powerlaw":
        # Target interval = TSS ± window; weight = (1 + |peak_mid - TSS|)^-gamma.
        tgt = pd.DataFrame(
            {
                "Chromosome": genes["Chromosome"].values,
                "Start": np.maximum(genes["tss"].values - window, 0),
                "End": genes["tss"].values + window,
                "gene_id": genes["gene_id"].values,
                "tss": genes["tss"].values,
            }
        )
        hits = _overlap_join(peaks, tgt)
        peak_mid = (hits["Start"].to_numpy() + hits["End"].to_numpy()) / 2.0
        d = np.abs(peak_mid - hits["tss"].to_numpy())
        w = np.power(1.0 + d, -gamma)
        # Normalise so a peak sitting on the TSS has weight 1.0 (it already does).
        return hits[["peak", "gene_id"]].to_numpy(), w

    raise ValueError(f"unknown kind={kind!r}")


def _overlap_join(peaks: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Inner overlap-join of peak intervals against target intervals.

    Returns one row per (peak, target) overlap with all columns of both
    inputs (peak Start/End kept as ``Start``/``End``; target extras kept
    by name).
    """
    try:
        import pyranges as pr

        a = pr.PyRanges(peaks)
        b = pr.PyRanges(targets)
        j = a.join(b, apply_strand_suffix=False)
        df = j.df if hasattr(j, "df") else pd.DataFrame(j)
        return df
    except ImportError:
        pass

    # Pure-pandas fallback: per-chromosome sort + sweep.
    out: list[pd.DataFrame] = []
    extra_cols = [c for c in targets.columns if c not in ("Chromosome", "Start", "End")]
    for chrom, pk in peaks.groupby("Chromosome", observed=True):
        tg = targets[targets["Chromosome"] == chrom]
        if tg.empty:
            continue
        pk = pk.sort_values("Start").reset_index(drop=True)
        tg = tg.sort_values("Start").reset_index(drop=True)
        ts = tg["Start"].to_numpy()
        te = tg["End"].to_numpy()
        ps = pk["Start"].to_numpy()
        pe = pk["End"].to_numpy()
        # candidate target indices: those with Start < peak.End
        hi = np.searchsorted(ts, pe, side="left")
        rows_p: list[int] = []
        rows_t: list[int] = []
        for i in range(len(pk)):
            cand = np.arange(hi[i])
            cand = cand[te[cand] > ps[i]]
            rows_p.extend([i] * cand.size)
            rows_t.extend(cand.tolist())
        if not rows_p:
            continue
        merged = pk.iloc[rows_p].reset_index(drop=True)
        merged[extra_cols] = tg.iloc[rows_t][extra_cols].reset_index(drop=True)
        out.append(merged)
    if not out:
        return pd.DataFrame(columns=["Chromosome", "Start", "End", "peak", *extra_cols])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Graph composition
# ---------------------------------------------------------------------------


def compose_gnnm(a: GnnmTuple, b: GnnmTuple, *, via: str) -> GnnmTuple:
    """Chain two feature graphs through a shared feature-space block.

    Given ``a = (G_a, gns_a, gd_a)`` over feature spaces ``{X, via}`` and
    ``b = (G_b, gns_b, gd_b)`` over ``{via, Y}``, return a graph over
    ``{X, Y}`` whose ``X↔Y`` block is the sparse product
    ``G_a[X, via] @ G_b[via, Y]`` restricted to features present in both
    ``via`` blocks.

    Typical use: cross-species ATAC↔RNA, where ``a`` is a peak→gene graph
    on species *S* (from :func:`gnnm_from_gtf`) and ``b`` is the
    S→T ortholog graph (from BLAST / eggNOG / :func:`gnnm_from_pairs`).
    Compose with ``via=S`` to get a peak(S)↔gene(T) graph.

    Weights are the matrix product (sum over intermediate features),
    clipped to ``[0, 1]``.
    """
    Ga, gns_a, gd_a = a
    Gb, gns_b, gd_b = b
    if via not in gd_a or via not in gd_b:
        raise ValueError(
            f"via={via!r} must be a key in both gns_dicts (a has {list(gd_a)}, b has {list(gd_b)})."
        )
    other_a = [k for k in gd_a if k != via]
    other_b = [k for k in gd_b if k != via]
    if len(other_a) != 1 or len(other_b) != 1:
        raise ValueError(
            "compose_gnnm currently supports exactly one non-shared "
            "feature space on each side; got "
            f"a={list(gd_a)}, b={list(gd_b)}."
        )
    xid, yid = other_a[0], other_b[0]

    idx_a = pd.Index(gns_a)
    idx_b = pd.Index(gns_b)
    via_shared = np.intersect1d(gd_a[via], gd_b[via])
    if via_shared.size == 0:
        raise ValueError(
            f"The two graphs share no features in the '{via}' block. "
            "Check that both were built with the same dataset id and "
            "the same gene-name namespace."
        )

    rows_x = idx_a.get_indexer(gd_a[xid])
    cols_va = idx_a.get_indexer(via_shared)
    rows_vb = idx_b.get_indexer(via_shared)
    cols_y = idx_b.get_indexer(gd_b[yid])

    # X×via block from a, via×Y block from b → X×Y product.
    Axv = Ga.tocsr()[rows_x, :][:, cols_va]
    Bvy = Gb.tocsr()[rows_vb, :][:, cols_y]
    Cxy = (Axv @ Bvy).tocoo()
    Cxy.data = np.minimum(Cxy.data, 1.0)

    x_names = np.asarray(gd_a[xid])
    y_names = np.asarray(gd_b[yid])
    gns_out = np.concatenate([x_names, y_names])
    nx = x_names.size

    rows = np.concatenate([Cxy.row, Cxy.col + nx])
    cols = np.concatenate([Cxy.col + nx, Cxy.row])
    data = np.concatenate([Cxy.data, Cxy.data])
    G = sp.coo_matrix((data, (rows, cols)), shape=(gns_out.size, gns_out.size)).tocsr()
    G.eliminate_zeros()

    gns_dict = {xid: x_names, yid: y_names}
    logger.info(
        "compose_gnnm: %s(%d) ∘ %s(%d shared) → %s(%d), %d edges.",
        xid,
        nx,
        via,
        via_shared.size,
        yid,
        y_names.size,
        Cxy.nnz,
    )
    return G, gns_out, gns_dict


# ---------------------------------------------------------------------------
# LSI preprocessing arm for binary peak matrices
# ---------------------------------------------------------------------------


def prepare_atac_sam(
    adata: AnnData,
    *,
    n_components: int = 50,
    drop_first: bool = True,
    k: int = 20,
    min_cells: int = 1,
    log_idf: bool = True,
) -> SAM:
    """Wrap a cells × peaks AnnData in a SAM object via TF-IDF + LSI.

    This is the (A2) preprocessing arm for binary/near-binary peak
    matrices: it bypasses SAM's log-CPM/dispersion path (which assumes
    count-like RNA features) and instead populates exactly the slots
    SAMap reads from a per-dataset SAM object —

    - ``adata.X`` — TF-IDF-normalised peak matrix (used by the
      correlation step after kNN smoothing);
    - ``adata.var['weights']`` — IDF term, scaled to ``[0, 1]``
      (down-weights ubiquitous promoter peaks, up-weights cell-type-
      specific distal elements; same role as SAM dispersion weights);
    - ``adata.varm['PCs_SAMap']`` — right singular vectors of the TF-IDF
      matrix (LSI loadings), with SV1 dropped by default since it tracks
      log-depth;
    - ``adata.obsp['connectivities']`` — within-dataset kNN on the LSI
      embedding;
    - ``adata.obs['leiden_clusters']`` — placeholder cluster column so
      ``SAMAP`` can use ``keys={'at': 'leiden_clusters'}`` without
      re-running Leiden on the TF-IDF matrix (call
      ``sam.leiden_clustering()`` afterwards if you want real clusters).

    Parameters
    ----------
    adata
        Cells × peaks AnnData with raw fragment/insertion counts (or a
        binary accessibility matrix) in ``.X``.
    n_components
        Number of LSI components to keep (after optionally dropping SV1).
    drop_first
        Drop the first singular vector (depth component). Default True.
    k
        kNN graph degree for the within-dataset connectivities.
    min_cells
        Drop peaks open in fewer than this many cells before TF-IDF.
    log_idf
        Use ``log(1 + N/n_i)`` IDF (Signac ``RunTFIDF`` method 1). If
        False, uses raw ``N/n_i``.

    Returns
    -------
    SAM
        A SAM object whose ``.adata`` carries the slots above. Pass it
        directly into ``SAMAP({..., 'at': sam}, gnnm=...)``.
    """
    from scipy.sparse.linalg import svds

    from samap.sam import SAM
    from samap.sam.knn import calc_nnm

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    # Filter near-empty peaks (IDF blows up otherwise).
    n_per_peak = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = n_per_peak >= max(min_cells, 1)
    if keep.sum() < keep.size:
        logger.info(
            "prepare_atac_sam: dropping %d/%d peaks open in <%d cells.",
            (~keep).sum(),
            keep.size,
            min_cells,
        )
    ad = adata[:, keep].copy()
    X = X[:, keep]
    n_per_peak = n_per_peak[keep]

    n_cells = X.shape[0]
    # TF: per-cell L1 normalisation. IDF: log(1 + N/n_i).
    cell_sums = np.asarray(X.sum(axis=1)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    tf = sp.diags(1.0 / cell_sums) @ X
    idf_raw = n_cells / n_per_peak
    idf = np.log1p(idf_raw) if log_idf else idf_raw
    tfidf = (tf @ sp.diags(idf)).tocsr().astype(np.float32)

    # LSI via truncated SVD.
    n_sv = n_components + (1 if drop_first else 0)
    n_sv = min(n_sv, min(tfidf.shape) - 1)
    u, s, vt = svds(tfidf, k=n_sv)
    order = np.argsort(-s)
    u, s, vt = u[:, order], s[order], vt[order, :]
    if drop_first:
        u, s, vt = u[:, 1:], s[1:], vt[1:, :]
    embedding = (u * s).astype(np.float32)
    loadings = vt.T.astype(np.float32)  # (peaks × components)

    # Within-dataset kNN on the LSI embedding.
    nnm = calc_nnm(embedding, k=min(k, n_cells - 1), distance="cosine")

    # IDF → [0, 1] weights.
    w = idf.astype(np.float32)
    w = w / (w.max() if w.max() > 0 else 1.0)

    ad.X = tfidf
    ad.layers["X_disp"] = tfidf
    ad.var["weights"] = w
    ad.var["mask_genes"] = True
    ad.varm["PCs_SAMap"] = loadings
    ad.obsm["X_lsi"] = embedding
    ad.obsp["connectivities"] = nnm
    ad.uns["neighbors"] = {"params": {"n_neighbors": k, "method": "umap", "metric": "cosine"}}
    ad.uns["preprocess_args"] = {"norm": "tfidf"}
    ad.uns["run_args"] = {
        "preprocessing": "StandardScaler",
        "weight_PCs": False,
        "npcs": loadings.shape[1],
    }
    ad.uns["modality"] = "atac"
    if "leiden_clusters" not in ad.obs:
        ad.obs["leiden_clusters"] = pd.Categorical(np.zeros(n_cells, dtype=int).astype(str))

    sam = SAM(counts=ad, inplace=True)
    # SAM's __init__ re-derives X_disp from raw; restore the TF-IDF layer.
    sam.adata = ad
    sam.adata_raw = ad
    logger.info(
        "prepare_atac_sam: %d cells x %d peaks -> %d LSI components "
        "(SV1 %s); IDF-weight range [%.3f, %.3f].",
        n_cells,
        ad.n_vars,
        loadings.shape[1],
        "dropped" if drop_first else "kept",
        float(w.min()),
        float(w.max()),
    )
    return sam
