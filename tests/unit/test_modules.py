"""Tests for #2 (module-factored scores) and #3 (joint_weights flag)."""

from __future__ import annotations

import numpy as np

from samap.analysis import gene_modules, module_factored_scores


def test_gene_modules_partitions_homology_graph(tiny_samap):
    mods = gene_modules(tiny_samap, resolution=1.0, seed=0)
    n = mods[mods >= 0].nunique()
    assert n >= 3  # at least the three planted marker blocks
    # genes with no homology edge get -1
    assert (mods == -1).sum() > 0


def test_module_factored_scores_multi_module_support(tiny_samap):
    mf = module_factored_scores(tiny_samap, {"aa": "cell_type", "bb": "cell_type"}, align_thr=0.1)
    assert len(mf) == 3  # the three planted 1:1 mappings
    assert {"n_modules", "top_module_frac", "module_entropy"} <= set(mf.columns)
    assert (mf["n_modules"] >= 2).all()
    assert (mf["module_entropy"] > 0).all()
    assert (mf["module_entropy"] <= np.log2(mf["n_gene_pairs"])).all()


def test_joint_weights_flag_runs_and_differs():
    """joint_weights=True should change per-species var['weights'] but
    leave the default-off path bit-identical."""
    from tests.fixtures.tiny_samap import build_tiny_samap

    sm_off = build_tiny_samap(seed=1, run=False)
    sm_off.run(n_iterations=2, umap=False, ncpus=2)
    w_off = sm_off.sams["aa"].adata.var["weights"].values.copy()

    sm_on = build_tiny_samap(seed=1, run=False)
    sm_on.run(n_iterations=2, umap=False, ncpus=2, joint_weights=True)
    w_on = sm_on.sams["aa"].adata.var["weights"].values.copy()

    assert not np.allclose(w_off, w_on), "joint_weights=True did not alter weights"
    assert "connectivities_iter0" in sm_on.samap.adata.obsp
