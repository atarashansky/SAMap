"""Label-permutation null for SAMap mapping scores.

The mapping score (alignment score) is the mean fraction of cross-species
mutual-nearest-neighbour mass landing in each target cluster. With the
manifold *fixed* after ``sm.run()``, permuting the per-species cell-type
labels gives an empirical null for "how high would the score be if cluster
membership were random". This is cheap — no manifold recomputation, just
re-aggregating the existing connectivity matrix under shuffled labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from samap.analysis.scores import _compute_csim
from samap.sam import SAM
from samap.utils import q as _q

if TYPE_CHECKING:
    from samap.core.mapping import SAMAP


def permutation_null_scores(
    sm: SAMAP,
    keys: dict[str, str],
    n_perm: int = 200,
    seed: int = 0,
    n_top: int = 0,
    which_iter: int | str = "final",
) -> pd.DataFrame:
    """Empirical null distribution for cross-species mapping scores.

    Permutes cell-type labels *within* each species (so cluster sizes are
    preserved) ``n_perm`` times, recomputes mapping scores against the
    fixed cell-cell connectivity, and reports the observed score alongside
    the null mean / std / 95th percentile / empirical p-value for every
    cross-species cell-type pair.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    keys : dict[str, str]
        Annotation key per species, as for ``get_mapping_scores``.
    n_perm : int, optional
        Number of permutations. Default 200.
    seed : int, optional
        RNG seed. Default 0.
    n_top : int, optional
        Passed through to ``_compute_csim``. Default 0.
    which_iter : int or 'final', optional
        Which iteration's connectivity to use; see ``get_mapping_scores``.

    Returns
    -------
    pandas.DataFrame
        One row per cross-species (type_a, type_b) pair with columns
        ``a``, ``b``, ``type_a``, ``type_b``, ``score``, ``null_mean``,
        ``null_std``, ``null_q95``, ``p_emp`` (one-sided, fraction of
        permutations with null ≥ observed; floored at ``1/(n_perm+1)``).
    """
    rng = np.random.default_rng(seed)

    if len(list(keys.keys())) < len(list(sm.sams.keys())):
        samap = SAM(
            counts=sm.samap.adata[np.isin(sm.samap.adata.obs["species"], list(keys.keys()))]
        )
    else:
        samap = sm.samap

    splabels = _q(samap.adata.obs["species"])
    ix = np.unique(splabels, return_index=True)[1]
    skeys = splabels[np.sort(ix)]

    # observed labels (species-prefixed) and per-species index blocks
    obs_lab = np.empty(splabels.size, dtype=object)
    blocks: dict[str, np.ndarray] = {}
    for sid in skeys:
        m = splabels == sid
        blocks[sid] = np.where(m)[0]
        obs_lab[m] = sid + "_" + _q(sm.sams[sid].adata.obs[keys[sid]].astype(str))

    label_col = "_perm_null_tmp"
    samap.adata.obs[label_col] = pd.Categorical(obs_lab)

    X = None
    if which_iter != "final":
        if not hasattr(sm, "nnm_per_iter") or not sm.nnm_per_iter:
            raise ValueError("which_iter requires sm.nnm_per_iter; call sm.run() first.")
        X = sm.nnm_per_iter[int(which_iter)]
        if len(list(keys.keys())) < len(list(sm.sams.keys())):
            mask = np.isin(sm.samap.adata.obs["species"], list(keys.keys()))
            X = X[mask, :][:, mask]

    # observed
    obs_csim, clu = _compute_csim(samap, label_col, X=X, n_top=n_top, prepend=False)
    sp_of = np.array([c.split("_", 1)[0] for c in clu])

    # null draws — accumulate per-cell sums to keep memory O(k^2) not O(n_perm * k^2)
    null_sum = np.zeros_like(obs_csim)
    null_sumsq = np.zeros_like(obs_csim)
    null_ge = np.zeros_like(obs_csim)  # count of perms with null >= observed
    null_max = np.full_like(obs_csim, -np.inf)
    q95_buf = np.empty((n_perm, *obs_csim.shape), dtype=np.float32) if n_perm <= 500 else None

    for p in range(n_perm):
        perm_lab = obs_lab.copy()
        for sid in skeys:
            idx = blocks[sid]
            perm_lab[idx] = obs_lab[rng.permutation(idx)]
        samap.adata.obs[label_col] = pd.Categorical(perm_lab)
        csim_p, clu_p = _compute_csim(samap, label_col, X=X, n_top=n_top, prepend=False)
        # _compute_csim's clu order is np.unique → stable across permutations,
        # but assert to be safe
        if not np.array_equal(clu_p, clu):  # pragma: no cover
            reidx = pd.Series(np.arange(len(clu_p)), index=clu_p).reindex(clu).values
            csim_p = csim_p[reidx][:, reidx]
        null_sum += csim_p
        null_sumsq += csim_p**2
        null_ge += (csim_p >= obs_csim).astype(np.float64)
        null_max = np.maximum(null_max, csim_p)
        if q95_buf is not None:
            q95_buf[p] = csim_p

    null_mean = null_sum / n_perm
    null_var = np.maximum(0.0, null_sumsq / n_perm - null_mean**2)
    null_std = np.sqrt(null_var)
    if q95_buf is not None:
        null_q95 = np.quantile(q95_buf, 0.95, axis=0)
    else:
        # fall back to mean + 1.645·std for very large n_perm
        null_q95 = null_mean + 1.645 * null_std
    p_emp = (null_ge + 1.0) / (n_perm + 1.0)

    # restore observed labels & tidy
    samap.adata.obs[label_col] = pd.Categorical(obs_lab)

    rows = []
    for i in range(len(clu)):
        for j in range(len(clu)):
            if sp_of[i] == sp_of[j]:
                continue
            rows.append(
                {
                    "a": sp_of[i],
                    "b": sp_of[j],
                    "type_a": clu[i].split("_", 1)[1],
                    "type_b": clu[j].split("_", 1)[1],
                    "score": float(obs_csim[i, j]),
                    "null_mean": float(null_mean[i, j]),
                    "null_std": float(null_std[i, j]),
                    "null_q95": float(null_q95[i, j]),
                    "null_max": float(null_max[i, j]),
                    "p_emp": float(p_emp[i, j]),
                }
            )
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    del samap.adata.obs[label_col]
    return out
