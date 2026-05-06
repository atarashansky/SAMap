"""Uniform-granularity clustering and mapping-degeneracy metrics.

Author cell-type annotations vary in granularity by 1–2 orders of magnitude
(10 dev stages vs 260 fine clusters), which makes alignment scores
incomparable across pairs. ``cluster_to_k`` produces a leiden clustering at
a *fixed* target cluster count per species so mapping-score matrices are
directly comparable, and ``mapping_degeneracy`` summarises the structure of
a score matrix (1:1-ness, entropy, effective rank) — these are the metrics
that should track phylogenetic distance, unlike ``max_score`` which
saturates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from samap.analysis.scores import get_mapping_scores

if TYPE_CHECKING:
    from samap.core.mapping import SAMAP
    from samap.sam import SAM


def cluster_to_k(
    sam: SAM,
    k: int,
    *,
    tol: int = 2,
    res_range: tuple[float, float] = (0.05, 20.0),
    max_iter: int = 25,
    seed: int = 0,
) -> str:
    """Leiden-cluster a SAM object to a target cluster count via resolution search.

    Binary-searches the leiden ``resolution`` parameter until the resulting
    number of clusters is within ``tol`` of ``k`` (or the search budget is
    exhausted). Writes the result to ``sam.adata.obs[f'leiden_k{k}']`` and
    returns that key.

    Parameters
    ----------
    sam : SAM
        A SAM object with ``run()`` already called (needs ``obsp['nnm']``).
    k : int
        Target number of clusters.
    tol : int, optional
        Accept any solution with ``|n_clusters − k| ≤ tol``. Default 2.
    res_range : tuple[float, float], optional
        Initial (lo, hi) bracket for resolution. Default (0.05, 20.0).
    max_iter : int, optional
        Maximum bisection steps. Default 25.
    seed : int, optional
        Leiden RNG seed. Default 0.

    Returns
    -------
    str
        The obs column name written (``f'leiden_k{k}'``).
    """
    lo, hi = res_range
    best_res, best_k, best_lab = None, None, None

    def _try(res: float) -> tuple[int, np.ndarray]:
        sam.leiden_clustering(res=res, seed=seed)
        lab = sam.adata.obs["leiden_clusters"].values.copy()
        return int(pd.Series(lab).nunique()), lab

    n_lo, lab_lo = _try(lo)
    n_hi, lab_hi = _try(hi)
    # Expand bracket if needed
    while n_hi < k and hi < 200:
        hi *= 2
        n_hi, lab_hi = _try(hi)
    while n_lo > k and lo > 1e-3:
        lo /= 2
        n_lo, lab_lo = _try(lo)

    for cand_k, cand_lab, cand_res in [(n_lo, lab_lo, lo), (n_hi, lab_hi, hi)]:
        if best_k is None or abs(cand_k - k) < abs(best_k - k):
            best_res, best_k, best_lab = cand_res, cand_k, cand_lab

    for _ in range(max_iter):
        if best_k is not None and abs(best_k - k) <= tol:
            break
        mid = float(np.sqrt(lo * hi))  # geometric midpoint — leiden k~log(res)-ish
        n_mid, lab_mid = _try(mid)
        if abs(n_mid - k) < abs((best_k or 10**9) - k):
            best_res, best_k, best_lab = mid, n_mid, lab_mid
        if n_mid < k:
            lo = mid
        else:
            hi = mid

    key = f"leiden_k{k}"
    sam.adata.obs[key] = pd.Categorical(best_lab.astype(str))
    sam.adata.uns[f"{key}_resolution"] = float(best_res)
    sam.adata.uns[f"{key}_n_clusters"] = int(best_k)
    return key


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def mapping_degeneracy(
    sm: SAMAP,
    keys: dict[str, str],
    *,
    thr: float = 0.1,
    n_top: int = 0,
) -> dict:
    """Structural metrics on the cross-species mapping-score matrix.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    keys : dict[str, str]
        Annotation key per species (typically the ``leiden_k{k}`` columns
        from ``cluster_to_k`` so granularity is matched).
    thr : float, optional
        Edge threshold for the reciprocal-best calculation. Default 0.1.
    n_top : int, optional
        Passed through to ``get_mapping_scores``.

    Returns
    -------
    dict
        ``score_matrix`` (DataFrame, species-a rows × species-b cols),
        ``rbh_frac`` (float, fraction of a-clusters whose best b-partner
        also picks them as best), ``row_entropy`` / ``col_entropy`` (mean
        Shannon entropy of row/column-normalized score distributions, in
        bits), ``eff_rank`` (∑σ / σ_max of the score matrix), ``max_score``,
        ``mean_top1``.
    """
    if len(keys) != 2:
        raise ValueError("mapping_degeneracy is defined for two-species comparisons.")
    a, b = list(keys.keys())
    _, MT = get_mapping_scores(sm, keys, n_top=n_top)
    ar = [r for r in MT.index if r.startswith(f"{a}_")]
    bc = [c for c in MT.columns if c.startswith(f"{b}_")]
    A = MT.loc[ar, bc].copy()
    A.index = [r[len(a) + 1 :] for r in A.index]
    A.columns = [c[len(b) + 1 :] for c in A.columns]

    M = A.values.astype(float)
    n_a, n_b = M.shape

    # reciprocal-best fraction (above thr)
    Mthr = M.copy()
    Mthr[Mthr < thr] = 0.0
    best_b = np.argmax(Mthr, axis=1)
    best_a = np.argmax(Mthr, axis=0)
    has_hit = Mthr.max(axis=1) > 0
    rbh = has_hit & (best_a[best_b] == np.arange(n_a))
    rbh_frac = float(rbh.mean())

    # row/col entropy of the (positive part of the) score distribution
    row_ent = float(np.mean([_entropy(r) for r in np.maximum(M, 0)]))
    col_ent = float(np.mean([_entropy(c) for c in np.maximum(M, 0).T]))

    # effective rank
    if M.any():
        s = np.linalg.svd(M, compute_uv=False)
        eff_rank = float(s.sum() / s.max()) if s.max() > 0 else 0.0
    else:
        eff_rank = 0.0

    return {
        "score_matrix": A,
        "rbh_frac": rbh_frac,
        "row_entropy": row_ent,
        "col_entropy": col_ent,
        "eff_rank": eff_rank,
        "max_score": float(M.max()),
        "mean_top1": float(M.max(axis=1).mean()),
        "n_a": n_a,
        "n_b": n_b,
    }
