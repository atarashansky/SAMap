"""Tests for #1 (homology-graph delta) and #7 (iter-0 connectivity)."""

from __future__ import annotations

import numpy as np
import pytest

from samap.analysis import (
    find_paralog_substitutions,
    get_mapping_scores,
    homology_graph_delta,
)


class TestIterZeroConnectivity:
    def test_nnm_per_iter_stored(self, tiny_samap):
        sm = tiny_samap
        assert hasattr(sm, "nnm_per_iter")
        assert len(sm.nnm_per_iter) == sm.samap.adata.uns["n_iterations"]
        assert "connectivities_iter0" in sm.samap.adata.obsp

    def test_iter0_nnm_is_iteration_zero(self, tiny_samap):
        sm = tiny_samap
        assert (sm.samap.adata.obsp["connectivities_iter0"] - sm.nnm_per_iter[0]).nnz == 0

    def test_get_mapping_scores_which_iter(self, tiny_samap):
        sm = tiny_samap
        keys = {"aa": "cell_type", "bb": "cell_type"}
        _, A_final = get_mapping_scores(sm, keys)
        _, A0 = get_mapping_scores(sm, keys, which_iter=0)
        _, A_last = get_mapping_scores(sm, keys, which_iter=len(sm.nnm_per_iter) - 1)
        # final == last iteration's nnm
        np.testing.assert_allclose(A_final.values, A_last.values)
        # iter0 ≤ final on the planted diagonal (planted structure means
        # refinement can only help or be neutral)
        ar = [r for r in A0.index if r.startswith("aa_")]
        bc = [c for c in A0.columns if c.startswith("bb_")]
        d0 = np.diag(A0.loc[ar, bc].values)
        df = np.diag(A_final.loc[ar, bc].values)
        assert (df >= d0 - 1e-6).all()

    def test_which_iter_out_of_range(self, tiny_samap):
        with pytest.raises(ValueError, match="out of range"):
            get_mapping_scores(tiny_samap, {"aa": "cell_type", "bb": "cell_type"}, which_iter=99)


class TestHomologyDelta:
    def test_delta_shape_and_columns(self, tiny_samap):
        df = homology_graph_delta(tiny_samap)
        assert {"a", "b", "gene_a", "gene_b", "seq", "corr", "resid", "dropped"} <= set(df.columns)
        # cross-species, upper-triangle only → one row per undirected edge
        assert (df["a"] != df["b"]).all()
        assert df["seq"].between(0, 1.0).all()

    def test_decoy_edges_dropped(self, tiny_samap):
        """Cross-cluster decoy edges (g in cluster i ↔ g in cluster j≠i,
        seq=0.3) should be killed by reweighting; identity edges (seq=0.9)
        should survive with high corr."""
        df = homology_graph_delta(tiny_samap)
        identity = df[np.isclose(df["seq"], 0.9)]
        decoy = df[np.isclose(df["seq"], 0.3)]
        assert identity["corr"].mean() > 0.9
        assert decoy["dropped"].mean() > 0.5, (
            f"expected most cross-cluster decoy edges pruned, "
            f"got {decoy['dropped'].mean():.2%} dropped"
        )

    def test_paralog_substitution_finder(self, tiny_samap):
        ps = find_paralog_substitutions(tiny_samap, min_corr=0.2)
        assert {"gene_a", "seq_best", "corr_best", "corr_gap"} <= set(ps.columns)
        # By construction every gene has both an identity edge (seq=0.9) and
        # a paralog-ring edge (seq=0.4). When the ring partner wins on corr,
        # seq_best and corr_best must differ.
        assert (ps["seq_best"] != ps["corr_best"]).all()
        assert (ps["corr_gap"] > 0).all()
