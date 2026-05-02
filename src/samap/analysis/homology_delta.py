"""Homology-graph delta: sequence-similarity vs expression-correlation residuals.

After ``sm.run()`` two gene-gene graphs live on the combined ``adata``:

- ``varp['homology_graph']`` — the input BLAST/DIAMOND-derived graph; edge
  weight is normalized sequence similarity (bitscore-derived).
- ``varp['homology_graph_reweighted']`` — same edge set, weights replaced by
  cross-species expression correlation on the aligned manifold.

The interesting biology is in *how the two disagree*. A high-correlation /
low-bitscore edge is a candidate paralog substitution or co-option (the gene
that took over the job despite weaker sequence homology); a low-correlation /
high-bitscore edge is a 1-1 ortholog whose expression has diverged.

This module exposes the per-edge residual table and a paralog-substitution
finder that doesn't require an external orthology database (cf.
``ParalogSubstitutions`` which does).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

from samap.utils import q as _q

if TYPE_CHECKING:
    from samap.core.mapping import SAMAP


def _running_median_residual(x: np.ndarray, y: np.ndarray, n_bins: int = 40) -> np.ndarray:
    """Residual of y after subtracting a binned running-median fit y~f(x).

    Cheap, robust, dependency-free LOESS substitute.
    """
    if x.size == 0:
        return np.zeros(0)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n_bins = max(4, min(n_bins, max(4, x.size // 25)))
    edges = np.quantile(xs, np.linspace(0, 1, n_bins + 1))
    edges[-1] = edges[-1] + 1e-12
    which = np.searchsorted(edges, xs, side="right") - 1
    which = np.clip(which, 0, n_bins - 1)
    med = np.zeros(n_bins)
    for b in range(n_bins):
        m = which == b
        med[b] = np.median(ys[m]) if m.any() else (med[b - 1] if b else 0.0)
    fitted = med[which]
    resid = np.empty_like(y)
    resid[order] = ys - fitted
    return resid


def homology_graph_delta(
    sm: SAMAP,
    *,
    species_pair: tuple[str, str] | None = None,
    include_dropped: bool = True,
) -> pd.DataFrame:
    """Per-edge comparison of sequence-similarity vs expression-correlation.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    species_pair : tuple[str, str], optional
        Restrict to edges between these two species. Default: all
        cross-species edges (for >2 species this is the union of all
        species-pair blocks).
    include_dropped : bool, optional
        Include BLAST edges that were pruned to zero in the reweighted graph
        (their ``corr`` will be 0). Default True.

    Returns
    -------
    pandas.DataFrame
        Columns: ``a`` (species of gene_a), ``b``, ``gene_a``, ``gene_b``,
        ``seq`` (homology_graph weight), ``corr`` (reweighted weight),
        ``rank_seq``, ``rank_corr`` (per-gene_a rank: 0 = best), ``resid``
        (corr − running-median(corr | seq)), ``dropped`` (bool).
    """
    ad = sm.samap.adata
    G0 = ad.varp["homology_graph"].tocoo()
    G1 = ad.varp.get("homology_graph_reweighted")
    if G1 is None:
        raise ValueError(
            "No reweighted homology graph found — run sm.run() with n_iterations >= 2 first."
        )
    G1 = sp.csr_matrix(G1)

    var = _q(ad.var_names)
    sp_of = np.array([g.split("_", 1)[0] for g in var])
    name_of = np.array([g.split("_", 1)[1] for g in var])

    row, col, seq = G0.row, G0.col, G0.data
    a, b = sp_of[row], sp_of[col]
    cross = a != b
    # Keep one direction per undirected edge (graph is symmetric).
    upper = row < col
    keep = cross & upper
    if species_pair is not None:
        s1, s2 = species_pair
        keep &= ((a == s1) & (b == s2)) | ((a == s2) & (b == s1))
    row, col, seq = row[keep], col[keep], seq[keep]

    corr = np.asarray(G1[row, col]).ravel()
    dropped = corr == 0
    if not include_dropped:
        m = ~dropped
        row, col, seq, corr, dropped = row[m], col[m], seq[m], corr[m], dropped[m]

    # Per-gene_a rank in each metric (0 = best partner). Same gene_a may
    # appear with multiple partners; rank is over those partners only.
    df = pd.DataFrame(
        {
            "a": sp_of[row],
            "b": sp_of[col],
            "gene_a": name_of[row],
            "gene_b": name_of[col],
            "seq": seq.astype(float),
            "corr": corr.astype(float),
            "dropped": dropped,
        }
    )
    df["rank_seq"] = df.groupby(["a", "gene_a"])["seq"].rank(method="min", ascending=False) - 1
    df["rank_corr"] = df.groupby(["a", "gene_a"])["corr"].rank(method="min", ascending=False) - 1
    df["resid"] = _running_median_residual(df["seq"].values, df["corr"].values)
    return df.sort_values("resid", ascending=False).reset_index(drop=True)


def find_paralog_substitutions(
    sm: SAMAP,
    *,
    species_pair: tuple[str, str] | None = None,
    min_corr: float = 0.3,
    min_seq_gap: float = 0.0,
) -> pd.DataFrame:
    """Genes whose best-correlated partner is *not* their best-sequence partner.

    For each gene with ≥2 cross-species edges, compares the partner that
    wins on sequence similarity (the "expected ortholog") to the partner
    that wins on expression correlation (the "functional partner"). When
    these differ and the correlation winner is convincingly correlated, the
    gene is a paralog-substitution candidate — its expression role has been
    taken over by a different homolog than the closest-by-sequence one.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    species_pair : tuple[str, str], optional
        Restrict to one species pair.
    min_corr : float, optional
        Minimum correlation of the corr-winner to report. Default 0.3.
    min_seq_gap : float, optional
        Require the seq-winner's ``seq`` to exceed the corr-winner's ``seq``
        by at least this much (otherwise the two are sequence-equivalent and
        the "substitution" is ambiguous). Default 0.0.

    Returns
    -------
    pandas.DataFrame
        Columns: ``a``, ``gene_a``, ``b``, ``seq_best`` (gene_b winning on
        seq), ``seq_best_seq``, ``seq_best_corr``, ``corr_best`` (gene_b
        winning on corr), ``corr_best_seq``, ``corr_best_corr``, ``corr_gap``
        (corr_best_corr − seq_best_corr), ``resid`` (of the corr-winning
        edge). Sorted by ``corr_gap`` descending.
    """
    delta = homology_graph_delta(sm, species_pair=species_pair, include_dropped=True)
    # Need ≥2 partners to have a substitution to talk about.
    g = delta.groupby(["a", "gene_a"])
    multi = g["gene_b"].transform("size") >= 2
    d = delta[multi]

    seq_best = d.loc[d.groupby(["a", "gene_a"])["seq"].idxmax()].set_index(["a", "gene_a"])
    corr_best = d.loc[d.groupby(["a", "gene_a"])["corr"].idxmax()].set_index(["a", "gene_a"])

    out = seq_best[["b", "gene_b", "seq", "corr"]].rename(
        columns={"gene_b": "seq_best", "seq": "seq_best_seq", "corr": "seq_best_corr"}
    )
    out = out.join(
        corr_best[["gene_b", "seq", "corr", "resid"]].rename(
            columns={
                "gene_b": "corr_best",
                "seq": "corr_best_seq",
                "corr": "corr_best_corr",
            }
        )
    )
    out["corr_gap"] = out["corr_best_corr"] - out["seq_best_corr"]
    out = out[out["seq_best"] != out["corr_best"]]
    out = out[
        (out["corr_best_corr"] >= min_corr)
        & ((out["seq_best_seq"] - out["corr_best_seq"]) >= min_seq_gap)
    ]
    return out.reset_index().sort_values("corr_gap", ascending=False).reset_index(drop=True)
