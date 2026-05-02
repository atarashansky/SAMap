"""Tests for #4 (uniform-k clustering + mapping degeneracy)."""

from __future__ import annotations

from samap.analysis import cluster_to_k, mapping_degeneracy


def test_cluster_to_k_hits_target(tiny_samap):
    sm = tiny_samap
    for k in (3, 5):
        key = cluster_to_k(sm.sams["aa"], k=k, tol=1, seed=0)
        n = sm.sams["aa"].adata.uns[f"{key}_n_clusters"]
        assert abs(n - k) <= 1, f"k={k}: got {n} clusters"
        assert key in sm.sams["aa"].adata.obs.columns


def test_degeneracy_on_planted_1to1(tiny_samap):
    sm = tiny_samap
    # The planted cell_type labels are perfectly 1:1 by construction.
    deg = mapping_degeneracy(sm, {"aa": "cell_type", "bb": "cell_type"})
    assert deg["rbh_frac"] == 1.0
    assert deg["row_entropy"] < 0.05  # near-zero off-diagonal
    assert deg["col_entropy"] < 0.05
    assert abs(deg["eff_rank"] - 3.0) < 0.1
    assert deg["score_matrix"].shape == (3, 3)


def test_degeneracy_at_finer_k_is_higher_entropy(tiny_samap):
    sm = tiny_samap
    ka = cluster_to_k(sm.sams["aa"], k=6, tol=1)
    kb = cluster_to_k(sm.sams["bb"], k=6, tol=1)
    deg6 = mapping_degeneracy(sm, {"aa": ka, "bb": kb})
    deg3 = mapping_degeneracy(sm, {"aa": "cell_type", "bb": "cell_type"})
    # Splitting each planted cluster in two creates 2:2 blocks → entropy rises
    assert deg6["row_entropy"] > deg3["row_entropy"]
    assert deg6["rbh_frac"] <= deg3["rbh_frac"]
