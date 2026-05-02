"""Tests for #6 (label-permutation null)."""

from __future__ import annotations

import numpy as np

from samap.analysis import permutation_null_scores


def test_null_separates_planted_from_unplanted(tiny_samap):
    out = permutation_null_scores(
        tiny_samap, {"aa": "cell_type", "bb": "cell_type"}, n_perm=50, seed=0
    )
    # 3×3×2 directions = 18 cross-species rows
    assert len(out) == 18
    diag = out[out.type_a == out.type_b]
    off = out[out.type_a != out.type_b]
    # Planted matches: observed >> null_q95 and minimal p
    assert (diag["score"] > diag["null_q95"]).all()
    assert np.allclose(diag["p_emp"], 1.0 / 51)
    # Unplanted: score ≈ 0 << null mean
    assert (off["score"] < off["null_mean"]).all()


def test_null_reproducible(tiny_samap):
    a = permutation_null_scores(
        tiny_samap, {"aa": "cell_type", "bb": "cell_type"}, n_perm=20, seed=7
    )
    b = permutation_null_scores(
        tiny_samap, {"aa": "cell_type", "bb": "cell_type"}, n_perm=20, seed=7
    )
    np.testing.assert_allclose(a["null_mean"].values, b["null_mean"].values)


def test_null_does_not_clobber_obs(tiny_samap):
    cols_before = set(tiny_samap.samap.adata.obs.columns)
    _ = permutation_null_scores(tiny_samap, {"aa": "cell_type", "bb": "cell_type"}, n_perm=5)
    assert "_perm_null_tmp" not in tiny_samap.samap.adata.obs.columns
    # Only the standard get_mapping_scores label may have been added; nothing else
    new = set(tiny_samap.samap.adata.obs.columns) - cols_before
    assert all("_mapping_scores" in c or False for c in new) or not new
