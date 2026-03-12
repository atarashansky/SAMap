"""Equivalence tests for the precomposed feature-translation path.

These pin the new :func:`_mapping_window_fast` / :func:`_compute_sigma` /
:func:`_projection_precompute` against a direct reimplementation of the
legacy materialise-Xtr path. Both must agree to ~1e-6 rtol — the rewrite
is an algebraic reshuffling, not an approximation.

If these tests start failing after the backward-compat ``_mapping_window``
wrapper is removed, they should still pass: they exercise
:func:`_mapping_window_fast` directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler

from samap.core._backend import Backend
from samap.core.projection import (
    _compute_sigma,
    _mapping_window_fast,
    _projection_precompute,
)

# --------------------------------------------------------------------------- #
# Fixtures — small synthetic 2- and 3-species inputs                          #
# --------------------------------------------------------------------------- #


class _MockAdata:
    """Minimal adata stand-in: just the fields projection.py reads."""

    def __init__(
        self,
        X: spp.csr_matrix,
        var_names: np.ndarray,
        weights: np.ndarray,
        PCs: np.ndarray,
    ) -> None:
        self.X = X
        self.var_names = var_names
        import pandas as pd

        self.var = pd.DataFrame({"weights": weights}, index=var_names)
        self.varm = {"PCs_SAMap": PCs}

    def __getitem__(self, key):  # adata[:, gene_names]
        _, cols = key
        # map gene names → column indices (projection.py slices by var_names match)
        name_to_ix = {n: i for i, n in enumerate(self.var_names)}
        ix = np.array([name_to_ix[c] for c in cols])
        return _MockAdata(
            X=self.X[:, ix],
            var_names=self.var_names[ix],
            weights=self.var["weights"].values[ix],
            PCs=self.varm["PCs_SAMap"][ix],
        )


class _MockSAM:
    def __init__(self, adata: _MockAdata) -> None:
        self.adata = adata


def _make_species(
    sid: str,
    n_cells: int,
    n_genes: int,
    npcs: int,
    rng: np.random.Generator,
) -> tuple[_MockSAM, np.ndarray]:
    """Build one mock species with random sparse counts, weights, and PC loadings."""
    var_names = np.array([f"{sid}_gene{i:03d}" for i in range(n_genes)])
    X = spp.random(n_cells, n_genes, density=0.35, format="csr", random_state=rng)
    X.data *= 10
    X = X.astype(np.float64)
    weights = rng.uniform(0.1, 1.0, n_genes)
    PCs = rng.standard_normal((n_genes, npcs)).astype(np.float64)
    sam = _MockSAM(_MockAdata(X, var_names, weights, PCs))
    return sam, var_names


def _make_gnnm(
    gns_list: list[np.ndarray],
    rng: np.random.Generator,
    density: float = 0.2,
) -> tuple[spp.csr_matrix, np.ndarray]:
    """Random block-off-diagonal homology graph (no within-species edges)."""
    gns = np.concatenate(gns_list)
    g_total = gns.size
    # build per-species-pair off-diagonal blocks
    sizes = [g.size for g in gns_list]
    offsets = np.cumsum([0, *sizes])
    A = spp.lil_matrix((g_total, g_total), dtype=np.float64)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if i == j:
                continue
            r0, r1 = offsets[i], offsets[i + 1]
            c0, c1 = offsets[j], offsets[j + 1]
            block = spp.random(sizes[i], sizes[j], density=density, format="csr", random_state=rng)
            block.data = np.abs(block.data) + 0.01  # strictly positive edges
            A[r0:r1, c0:c1] = block
    return A.tocsr(), gns


@pytest.fixture
def bk() -> Backend:
    return Backend("cpu")


@pytest.fixture
def synth2(bk):  # 2 species
    rng = np.random.default_rng(42)
    sam_a, gns_a = _make_species("aa", n_cells=60, n_genes=25, npcs=8, rng=rng)
    sam_b, gns_b = _make_species("bb", n_cells=45, n_genes=20, npcs=6, rng=rng)
    sams = {"aa": sam_a, "bb": sam_b}
    gnnm, gns = _make_gnnm([gns_a, gns_b], rng)
    return sams, gnnm, gns


@pytest.fixture
def synth3(bk):  # 3 species
    rng = np.random.default_rng(123)
    sam_a, gns_a = _make_species("aa", n_cells=40, n_genes=18, npcs=5, rng=rng)
    sam_b, gns_b = _make_species("bb", n_cells=35, n_genes=15, npcs=5, rng=rng)
    sam_c, gns_c = _make_species("cc", n_cells=30, n_genes=12, npcs=4, rng=rng)
    sams = {"aa": sam_a, "bb": sam_b, "cc": sam_c}
    gnnm, gns = _make_gnnm([gns_a, gns_b, gns_c], rng)
    return sams, gnnm, gns


# --------------------------------------------------------------------------- #
# Reference implementation — legacy materialise-Xtr path                      #
# --------------------------------------------------------------------------- #


def _legacy_wpca(sams, gnnm, gns, pairwise: bool):
    """Direct transcription of the pre-refactor _mapping_window wpca logic.

    Kept here as a test oracle — the production path no longer materialises Xtr.
    """
    from samap.core.homology import _tanh_scale
    from samap.utils import q as _q

    std = StandardScaler(with_mean=False)

    gnnm_corr = gnnm.copy()
    gnnm_corr.data[:] = _tanh_scale(gnnm_corr.data)

    gs, adatas, Ws, ss = {}, {}, {}, {}
    species_indexer, genes_indexer = [], []
    for sid in sams:
        gs[sid] = gns[np.isin(gns, _q(sams[sid].adata.var_names))]
        adatas[sid] = sams[sid].adata[:, gs[sid]]
        Ws[sid] = adatas[sid].var["weights"].values
        ss[sid] = std.fit_transform(adatas[sid].X).multiply(Ws[sid][None, :]).tocsr()
        species_indexer.append(np.arange(ss[sid].shape[0]))
        genes_indexer.append(np.arange(gs[sid].size))
    for i in range(1, len(species_indexer)):
        species_indexer[i] += species_indexer[i - 1].max() + 1
        genes_indexer[i] += genes_indexer[i - 1].max() + 1

    su = np.asarray(gnnm_corr.sum(0))
    su[su == 0] = 1
    gnnm_corr = gnnm_corr.multiply(1 / su).tocsr()

    X = spp.block_diag(list(ss.values())).tocsr()
    W = np.concatenate(list(Ws.values())).flatten()

    if pairwise:
        Xtr_rows = []
        for i in range(len(sams)):
            xtr = []
            for j in range(len(sams)):
                if i != j:
                    gsub = gnnm_corr[genes_indexer[i]][:, genes_indexer[j]]
                    su = np.asarray(gsub.sum(0))
                    su[su == 0] = 1
                    gsub = gsub.multiply(1 / su).tocsr()
                    x = X[species_indexer[i]][:, genes_indexer[i]].dot(gsub)
                    xtr.append(std.fit_transform(x).multiply(W[genes_indexer[j]][None, :]))
                else:
                    xtr.append(
                        spp.csr_matrix((species_indexer[i].size, genes_indexer[i].size))
                    )
            Xtr_rows.append(spp.hstack(xtr))
        Xtr = spp.vstack(Xtr_rows)
    else:
        Xtr_rows = []
        for i in range(len(sams)):
            x = X[species_indexer[i]].dot(gnnm_corr)
            Xtr_rows.append(std.fit_transform(x).multiply(W[None, :]))
        Xtr = spp.vstack(Xtr_rows)
    Xc = (X + Xtr).tocsr()

    mus = [np.asarray(Xc[species_indexer[i]].mean(0)).flatten() for i in range(len(sams))]

    import scipy as sp_full

    C = sp_full.linalg.block_diag(*[adatas[sid].varm["PCs_SAMap"] for sid in sams])
    M = np.vstack(mus).dot(C)
    it = 0
    PCAs = []
    for sid in sams:
        PCAs.append(Xc[:, it : it + gs[sid].size].dot(adatas[sid].varm["PCs_SAMap"]))
        it += gs[sid].size
    wpca = np.hstack(PCAs)
    for i in range(len(sams)):
        wpca[species_indexer[i]] -= M[i]

    return wpca, gnnm_corr


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


class TestComputeSigma:
    """Sigma quadratic-form must match sklearn's StandardScaler exactly."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 999])
    def test_matches_sklearn(self, bk, seed):
        rng = np.random.default_rng(seed)
        n, g1, g2 = 50, 30, 20
        X = spp.random(n, g1, density=0.3, format="csr", random_state=rng).astype(np.float64)
        G = spp.random(g1, g2, density=0.25, format="csr", random_state=rng).astype(np.float64)

        truth = StandardScaler(with_mean=False).fit(X @ G).scale_

        XtX = (X.T @ X).tocsr()
        mu = np.asarray(X.mean(0)).flatten()
        sigma = _compute_sigma(XtX, mu, G, n, bk)

        np.testing.assert_allclose(sigma, truth, rtol=1e-12, atol=1e-14)

    def test_zero_variance_columns_map_to_one(self, bk):
        """StandardScaler replaces zero-variance scale with 1.0; so must we."""
        n, g1, g2 = 20, 10, 5
        X = spp.random(n, g1, density=0.3, format="csr", random_state=0).astype(np.float64)
        # G with an all-zero column → zero-variance output column
        G = spp.random(g1, g2, density=0.3, format="csr", random_state=1).astype(np.float64).tolil()
        G[:, 2] = 0
        G = G.tocsr()

        truth = StandardScaler(with_mean=False).fit(X @ G).scale_

        XtX = (X.T @ X).tocsr()
        mu = np.asarray(X.mean(0)).flatten()
        sigma = _compute_sigma(XtX, mu, G, n, bk)

        assert sigma[2] == 1.0
        np.testing.assert_allclose(sigma, truth, rtol=1e-12, atol=1e-14)


class TestWPCAEquivalence:
    """The full wpca output must match the legacy materialise-Xtr path."""

    def test_2species_pairwise(self, synth2, bk):
        sams, gnnm, gns = synth2
        pre = _projection_precompute(sams, gns, bk)
        out = _mapping_window_fast(gnnm, pre, K=5, pairwise=True)
        wpca_old, gnnm_corr_old = _legacy_wpca(sams, gnnm, gns, pairwise=True)

        np.testing.assert_allclose(out["wPCA"], wpca_old, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(
            out["gnnm_corr"].toarray(), gnnm_corr_old.toarray(), rtol=1e-12, atol=1e-14
        )

    def test_2species_all_to_all(self, synth2, bk):
        sams, gnnm, gns = synth2
        pre = _projection_precompute(sams, gns, bk)
        out = _mapping_window_fast(gnnm, pre, K=5, pairwise=False)
        wpca_old, _ = _legacy_wpca(sams, gnnm, gns, pairwise=False)

        np.testing.assert_allclose(out["wPCA"], wpca_old, rtol=1e-6, atol=1e-10)

    def test_3species_pairwise(self, synth3, bk):
        sams, gnnm, gns = synth3
        pre = _projection_precompute(sams, gns, bk)
        out = _mapping_window_fast(gnnm, pre, K=5, pairwise=True)
        wpca_old, _ = _legacy_wpca(sams, gnnm, gns, pairwise=True)

        np.testing.assert_allclose(out["wPCA"], wpca_old, rtol=1e-6, atol=1e-10)

    def test_3species_all_to_all(self, synth3, bk):
        """3+ species: pairwise vs all-to-all differ due to normalisation scope.

        With 2 species the global and per-pair column-normalisations of the
        homology graph coincide; with 3+ they don't (each column gets
        contributions from multiple species). This test guards the all-to-all
        branch specifically.
        """
        sams, gnnm, gns = synth3
        pre = _projection_precompute(sams, gns, bk)
        out = _mapping_window_fast(gnnm, pre, K=5, pairwise=False)
        wpca_old, _ = _legacy_wpca(sams, gnnm, gns, pairwise=False)

        np.testing.assert_allclose(out["wPCA"], wpca_old, rtol=1e-6, atol=1e-10)

    def test_precompute_is_iteration_invariant(self, synth2, bk):
        """Precompute dict shouldn't depend on gnnm — reuse across iterations."""
        sams, gnnm, gns = synth2
        pre = _projection_precompute(sams, gns, bk)

        # Two different homology graphs, same precompute
        rng = np.random.default_rng(7)
        gnnm2 = gnnm.copy()
        gnnm2.data = rng.uniform(0.01, 1.0, gnnm2.data.size)

        out1 = _mapping_window_fast(gnnm, pre, K=5, pairwise=True)
        out2 = _mapping_window_fast(gnnm2, pre, K=5, pairwise=True)

        # Different inputs → different outputs (sanity: precompute isn't stale-caching)
        assert not np.allclose(out1["wPCA"], out2["wPCA"])

        # But both match their respective legacy oracles
        wpca_old1, _ = _legacy_wpca(sams, gnnm, gns, pairwise=True)
        wpca_old2, _ = _legacy_wpca(sams, gnnm2, gns, pairwise=True)
        np.testing.assert_allclose(out1["wPCA"], wpca_old1, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(out2["wPCA"], wpca_old2, rtol=1e-6, atol=1e-10)


class TestBackwardCompatWrapper:
    """The old _mapping_window signature still works (via internal precompute)."""

    def test_wrapper_equivalent_to_fast_path(self, synth2, bk):
        from samap.core.projection import _mapping_window

        sams, gnnm, gns = synth2
        out_wrapper = _mapping_window(sams, gnnm, gns, K=5, pairwise=True)

        pre = _projection_precompute(sams, gns, bk)
        out_fast = _mapping_window_fast(gnnm, pre, K=5, pairwise=True)

        np.testing.assert_allclose(out_wrapper["wPCA"], out_fast["wPCA"], rtol=1e-12)
        # knn structure should be identical since wpca is identical
        np.testing.assert_allclose(
            out_wrapper["knn"].toarray(), out_fast["knn"].toarray(), rtol=1e-12
        )
