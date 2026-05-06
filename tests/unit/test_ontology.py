"""Tests for samap.analysis.ontology — disk-based re-scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from samap.analysis import get_mapping_scores
from samap.analysis.ontology import (
    build_union_graph,
    cluster_families,
    family_phylogenetic_signal,
    persist_pair,
    score_from_connectivities,
)


@pytest.fixture(scope="module")
def persisted(tmp_path_factory, tiny_samap):
    out = tmp_path_factory.mktemp("ontology") / "aabb"
    meta = persist_pair(tiny_samap, str(out))
    return str(out), meta, tiny_samap


class TestPersistAndRescore:
    def test_persist_writes_files(self, persisted):
        prefix, meta, _ = persisted
        import os

        assert os.path.exists(f"{prefix}_obsp.npz")
        assert os.path.exists(f"{prefix}_obs.parquet")
        assert os.path.exists(f"{prefix}_meta.json")
        assert meta["mapping_K"] > 0
        assert set(meta["species"]) == {"aa", "bb"}

    def test_roundtrip_matches_get_mapping_scores(self, persisted):
        prefix, _, sm = persisted
        keys = {"aa": "cell_type", "bb": "cell_type"}
        _, MT = get_mapping_scores(sm, keys)
        # Persisted obs_name is the combined-adata name (which SAMAP
        # species-prefixes); index labels by that so the round-trip is
        # exact regardless of how the per-species sams name their cells.
        obs = pd.read_parquet(f"{prefix}_obs.parquet")
        labels = {}
        for s in ("aa", "bb"):
            mask = obs.species == s
            labels[s] = pd.Series(
                sm.sams[s].adata.obs["cell_type"].astype(str).values,
                index=obs.obs_name[mask].values,
            )
        S = score_from_connectivities(prefix, labels)
        # cross-block should match MT to within float tolerance
        ar = [r for r in MT.index if r.startswith("aa_")]
        bc = [c for c in MT.columns if c.startswith("bb_")]
        np.testing.assert_allclose(S.loc[ar, bc].values, MT.loc[ar, bc].values, atol=1e-6)

    def test_rescore_arbitrary_labels(self, persisted):
        prefix, _, _sm = persisted
        obs = pd.read_parquet(f"{prefix}_obs.parquet")
        rng = np.random.default_rng(0)
        labels = {
            s: pd.Series(
                rng.integers(0, 5, (obs.species == s).sum()).astype(str),
                index=obs.obs_name[obs.species == s].values,
            )
            for s in ("aa", "bb")
        }
        S = score_from_connectivities(prefix, labels)
        # Random labels → no strong block structure; max score well below 1
        ar = [r for r in S.index if r.startswith("aa_")]
        bc = [c for c in S.columns if c.startswith("bb_")]
        assert S.loc[ar, bc].values.max() < 0.6
        assert S.loc[ar, bc].values.min() >= 0.0

    def test_handles_make_unique_suffix(self, persisted, tiny_samap):
        """SAMAP's combined adata calls ``obs_names_make_unique`` which appends
        ``-N`` to colliding obs-names; ``score_from_connectivities`` must accept
        labels indexed by the *original* per-species names."""
        prefix, _, _sm = persisted
        obs = pd.read_parquet(f"{prefix}_obs.parquet")
        # Simulate a collision: rewrite the first 5 bb obs-names with a -1 suffix.
        bb_mask = obs["species"] == "bb"
        bb_idx = np.where(bb_mask)[0][:5]
        orig = obs.copy()
        obs.iloc[bb_idx, obs.columns.get_loc("obs_name")] = (
            obs.iloc[bb_idx]["obs_name"].astype(str) + "-1"
        )
        obs.to_parquet(f"{prefix}_obs.parquet")
        try:
            labels = {
                "aa": tiny_samap.sams["aa"].adata.obs["cell_type"],
                "bb": tiny_samap.sams["bb"].adata.obs["cell_type"],
            }
            # Should NOT raise — suffix is stripped on miss-then-retry.
            S = score_from_connectivities(prefix, labels)
            assert S.shape[0] == S.shape[1]
            assert (S.values.diagonal() == 0).all()
        finally:
            orig.to_parquet(f"{prefix}_obs.parquet")


class TestFamilyPhylogeneticSignal:
    def test_lineage_vs_program_classification(self):
        """A family whose within-family score decays with divergence is
        classified ``lineage``; one whose score is flat/increasing is
        ``program``. Uses a synthetic 5-species edge set so the test is
        independent of the tiny_samap fixture."""
        sp = ["s0", "s1", "s2", "s3", "s4"]
        # divergence: s0..s4 on a ladder, 100 Mya per step
        div = {(sp[i], sp[j]): (j - i) * 100.0 for i in range(5) for j in range(i + 1, 5)}
        # Family A (lineage-like): score = 1 - 0.002·div, all 10 pairs
        # Family B (program-bin): score = 0.3 + 0.001·div
        edges = []
        fams = []
        for f, nA, fn in [
            ("A", "0", lambda d: 1.0 - 0.002 * d),
            ("B", "1", lambda d: 0.3 + 0.001 * d),
        ]:
            for s in sp:
                fams.append({"node": f"{s}_{nA}", "species": s, "label": nA, "family": f})
            for i in range(5):
                for j in range(i + 1, 5):
                    d = div[(sp[i], sp[j])]
                    edges.append(
                        {
                            "src": f"{sp[i]}_{nA}",
                            "dst": f"{sp[j]}_{nA}",
                            "score": fn(d),
                            "rbh": True,
                        }
                    )
        E = pd.DataFrame(edges)
        F = pd.DataFrame(fams)
        out = family_phylogenetic_signal(E, F, div, min_species=5, min_pairs=8)
        out = out.set_index("family")
        assert out.loc["A", "rho"] < -0.9
        assert out.loc["A", "classification"] == "lineage"
        assert out.loc["B", "rho"] > 0.9
        assert out.loc["B", "classification"] == "program"

    def test_exclude_pairs_control(self):
        """A family whose apparent signal comes only from a tight same-study
        clade collapses to ambiguous after ``exclude_pairs``."""
        sp = ["s0", "s1", "s2", "s3", "s4"]
        div = {(sp[i], sp[j]): (j - i) * 100.0 for i in range(5) for j in range(i + 1, 5)}
        # Flat score 0.4 everywhere EXCEPT s0-s1 (the close pair) at 0.95.
        edges, fams = [], []
        for s in sp:
            fams.append({"node": f"{s}_0", "species": s, "label": "0", "family": "X"})
        for i in range(5):
            for j in range(i + 1, 5):
                sc = 0.95 if (i, j) == (0, 1) else 0.4
                edges.append({"src": f"{sp[i]}_0", "dst": f"{sp[j]}_0", "score": sc, "rbh": True})
        E, F = pd.DataFrame(edges), pd.DataFrame(fams)
        out_all = family_phylogenetic_signal(E, F, div, min_species=5, min_pairs=8)
        out_ex = family_phylogenetic_signal(
            E, F, div, min_species=5, min_pairs=8, exclude_pairs={"s0s1"}
        )
        assert out_all.iloc[0]["rho"] < -0.3
        assert abs(out_ex.iloc[0]["rho_ex"]) < 0.2 or np.isnan(out_ex.iloc[0]["rho_ex"])
        assert out_ex.iloc[0]["classification"] == "ambiguous"


class TestUnionGraph:
    def test_build_and_cluster(self, persisted, tmp_path):
        prefix, _, sm = persisted
        import shutil

        d = tmp_path / "persisted"
        d.mkdir()
        for ext in ("_obsp.npz", "_obs.parquet", "_meta.json"):
            shutil.copy(f"{prefix}{ext}", d / f"aabb{ext}")
        obs = pd.read_parquet(d / "aabb_obs.parquet")
        labels = {
            s: pd.Series(
                sm.sams[s].adata.obs["cell_type"].astype(str).values,
                index=obs.obs_name[obs.species == s].values,
            )
            for s in ("aa", "bb")
        }
        E = build_union_graph(str(d), ["aa", "bb"], labels)
        # tiny_samap has 3 planted 1:1 clusters → exactly 3 RBH edges, all
        # near-1.0, and at thr=0.1 there are no off-diagonal edges.
        assert E["rbh"].sum() == 3
        assert (E[E["rbh"]]["score"] > 0.9).all()
        fams = cluster_families(E, rbh_only=True, resolution=1.0)
        # 3 RBH pairs → 3 size-2 families (each aa+bb)
        sizes = fams.groupby("family").size()
        assert (sizes == 2).sum() == 3
        for _, grp in fams[fams.family.isin(sizes[sizes == 2].index)].groupby("family"):
            assert set(grp.species) == {"aa", "bb"}
