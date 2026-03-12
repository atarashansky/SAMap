"""Equivalence tests for streaming mutual-NN construction in coarsening.

The streaming implementation in ``_compute_mutual_graph`` replaces the
original monolithic ``D = B @ nnm_internal.T`` with per-species-pair block
computation. These tests verify the output is numerically identical.

The reference implementation here (``_reference_mutual_graph``) reproduces the
*original* code path verbatim: build full block matrices, materialise D,
mutualise, scale, top-k. This is the spec we test against.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as spp

from samap.core.coarsening import _compute_mutual_graph
from samap.core.correlation import _replace
from samap.core.homology import _tanh_scale
from samap.utils import sparse_knn

# ---------------------------------------------------------------------------
# Reference implementation (reproduces the original _mapper body)
# ---------------------------------------------------------------------------


def _reference_mutual_graph(
    nnms_in: dict[str, Any],
    neigh_from_keys: dict[str, bool],
    B: spp.csr_matrix,
    offsets: dict[str, int],
    n_cells: dict[str, int],
    sids: list[str],
    k1: int,
    N: int,
    *,
    pairwise: bool,
    threshold: float,
    scale_edges_by_corr: bool,
    wPCA: Any,
) -> spp.csr_matrix:
    """Original monolithic D = B @ nnm_internal.T path, for comparison."""
    any_nfk = any(neigh_from_keys[sid] for sid in sids)

    # Build block-diag nnm_internal. For nfk species, the effective block
    # is M @ M.T (co-clustering), otherwise the expanded kNN directly.
    eff_blocks: list[Any] = []
    for sid in sids:
        blk = nnms_in[sid]
        if neigh_from_keys[sid]:
            eff_blocks.append(blk.dot(blk.T))
        else:
            eff_blocks.append(blk)
    nnm_internal = spp.block_diag(eff_blocks).tocsr()

    D = B.dot(nnm_internal.T).tocsr()
    if not any_nfk and threshold > 0:
        D.data[D.data < threshold] = 0
        D.eliminate_zeros()

    D = D.multiply(D.T).tocsr()
    D.data[:] = D.data**0.5

    if scale_edges_by_corr:
        x, y = D.nonzero()
        vals = _replace(wPCA, x, y)
        vals[vals < 1e-3] = 1e-3
        F = D.copy()
        F.data[:] = vals
        ma = np.asarray(F.max(1).todense())
        ma[ma == 0] = 1
        F = F.multiply(1 / ma).tocsr()
        F.data[:] = _tanh_scale(F.data, center=0.7, scale=10)
        ma = np.asarray(D.max(1).todense())
        ma[ma == 0] = 1
        D = F.multiply(D).tocsr()
        D.data[:] = np.sqrt(D.data)
        ma2 = np.asarray(D.max(1).todense())
        ma2[ma2 == 0] = 1
        D = D.multiply(ma / ma2).tocsr()

    if not pairwise or len(sids) == 2:
        return sparse_knn(D, k1).tocsr()

    # pairwise top-k per species pair
    row = np.array([], dtype="int64")
    col = np.array([], dtype="int64")
    data = np.array([], dtype="float64")
    for a in sids:
        ra = np.arange(offsets[a], offsets[a] + n_cells[a])
        for b in sids:
            if a == b:
                continue
            rb = np.arange(offsets[b], offsets[b] + n_cells[b])
            Dsub = sparse_knn(D[ra][:, rb], k1).tocoo()
            row = np.append(row, ra[Dsub.row])
            col = np.append(col, rb[Dsub.col])
            data = np.append(data, Dsub.data)
    return spp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sym_knn(
    rng: np.random.Generator, n: int, k: int, values: bool = False
) -> spp.csr_matrix:
    """Build a symmetric kNN-ish sparse matrix with positive entries.

    If ``values`` is False the matrix is binary {0,1}; if True, nonzeros are
    drawn uniform in (0.5, 1.5) to exercise the threshold floor.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        nbrs = rng.choice(n, size=min(k, n), replace=False)
        nbrs = nbrs[nbrs != i][: max(k - 1, 1)]
        for j in nbrs:
            v = float(rng.uniform(0.5, 1.5)) if values else 1.0
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([v, v])
    M = spp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    M.sum_duplicates()
    if not values:
        M.data[:] = 1.0
    M.setdiag(0)
    M.eliminate_zeros()
    return M


def _make_cross_B(
    rng: np.random.Generator,
    sids: list[str],
    n_cells: dict[str, int],
    offsets: dict[str, int],
    N: int,
    k: int,
) -> spp.csr_matrix:
    """Build a block-off-diagonal cross-species kNN (like mdata['knn'])."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for a in sids:
        na = n_cells[a]
        off_a = offsets[a]
        for b in sids:
            if a == b:
                continue
            nb = n_cells[b]
            off_b = offsets[b]
            for i in range(na):
                nbrs = rng.choice(nb, size=min(k, nb), replace=False)
                for j in nbrs:
                    rows.append(off_a + i)
                    cols.append(off_b + int(j))
                    vals.append(float(rng.uniform(0.2, 1.0)))
    return spp.csr_matrix((vals, (rows, cols)), shape=(N, N))


@pytest.fixture
def two_species_inputs(
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Synthetic 2-species input: ~80 and ~120 cells, random kNN structure."""
    sids = ["spA", "spB"]
    n_cells = {"spA": 80, "spB": 120}
    offsets = {"spA": 0, "spB": 80}
    N = 200

    nnms_in = {
        "spA": _make_sym_knn(rng, 80, k=8, values=False),
        "spB": _make_sym_knn(rng, 120, k=8, values=False),
    }
    neigh_from_keys = {"spA": False, "spB": False}
    B = _make_cross_B(rng, sids, n_cells, offsets, N, k=10)

    return {
        "sids": sids,
        "n_cells": n_cells,
        "offsets": offsets,
        "N": N,
        "nnms_in": nnms_in,
        "neigh_from_keys": neigh_from_keys,
        "B": B,
    }


@pytest.fixture
def three_species_inputs(
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Synthetic 3-species input for pairwise top-k testing."""
    sids = ["x", "y", "z"]
    n_cells = {"x": 60, "y": 70, "z": 50}
    offsets = {"x": 0, "y": 60, "z": 130}
    N = 180

    nnms_in = {
        sid: _make_sym_knn(rng, n_cells[sid], k=6, values=False) for sid in sids
    }
    neigh_from_keys = dict.fromkeys(sids, False)
    B = _make_cross_B(rng, sids, n_cells, offsets, N, k=8)

    return {
        "sids": sids,
        "n_cells": n_cells,
        "offsets": offsets,
        "N": N,
        "nnms_in": nnms_in,
        "neigh_from_keys": neigh_from_keys,
        "B": B,
    }


@pytest.fixture
def nfk_inputs(rng: np.random.Generator) -> dict[str, Any]:
    """2-species input where one species uses the coclustering path."""
    sids = ["a", "b"]
    n_cells = {"a": 90, "b": 70}
    offsets = {"a": 0, "b": 90}
    N = 160

    # species a: 4 clusters, one-hot membership
    cl_a = rng.integers(0, 4, size=90)
    M_a = np.zeros((90, 4))
    M_a[np.arange(90), cl_a] = 1
    M_a = spp.csr_matrix(M_a)

    nnms_in = {
        "a": M_a,
        "b": _make_sym_knn(rng, 70, k=6, values=False),
    }
    neigh_from_keys = {"a": True, "b": False}
    B = _make_cross_B(rng, sids, n_cells, offsets, N, k=8)

    return {
        "sids": sids,
        "n_cells": n_cells,
        "offsets": offsets,
        "N": N,
        "nnms_in": nnms_in,
        "neigh_from_keys": neigh_from_keys,
        "B": B,
    }


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------


def _assert_sparse_equal(A: spp.spmatrix, B: spp.spmatrix, atol: float = 1e-12) -> None:
    """Assert two sparse matrices are numerically identical."""
    assert A.shape == B.shape, f"shape mismatch: {A.shape} vs {B.shape}"
    diff = (A - B).tocoo()
    if diff.nnz:
        max_abs = float(np.abs(diff.data).max())
        assert max_abs <= atol, (
            f"max abs diff {max_abs} > atol {atol}; "
            f"{diff.nnz} entries differ; "
            f"nnz(A)={A.nnz}, nnz(B)={B.nnz}"
        )


class TestTwoSpecies:
    """Equivalence tests for the common 2-species case."""

    @pytest.mark.parametrize("chunksize", [10_000, 30, 7])
    def test_basic(
        self, two_species_inputs: dict[str, Any], chunksize: int
    ) -> None:
        """Streaming == reference, 2 species, no scaling, various chunk sizes.

        chunksize=7 forces many small chunks within each species to exercise
        the chunk-boundary index bookkeeping.
        """
        inp = two_species_inputs
        k1 = 15

        ref = _reference_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, threshold=0.1, scale_edges_by_corr=False, wPCA=None,
        )
        got = _compute_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, chunksize=chunksize, threshold=0.1,
            scale_edges_by_corr=False, wPCA=None,
        )
        _assert_sparse_equal(got, ref)

    def test_with_scale_edges_by_corr(
        self, two_species_inputs: dict[str, Any], rng: np.random.Generator
    ) -> None:
        """Streaming == reference with correlation-based edge rescaling.

        wPCA rows must correlate with the kNN structure for nonzero effect;
        a pure random wPCA exercises the path regardless.
        """
        inp = two_species_inputs
        k1 = 12
        wPCA = rng.standard_normal((inp["N"], 50))

        ref = _reference_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, threshold=0.1, scale_edges_by_corr=True, wPCA=wPCA,
        )
        got = _compute_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, chunksize=25, threshold=0.1,
            scale_edges_by_corr=True, wPCA=wPCA,
        )
        _assert_sparse_equal(got, ref)

    def test_non_pairwise(
        self, two_species_inputs: dict[str, Any]
    ) -> None:
        """pairwise=False (global per-row top-k) matches reference."""
        inp = two_species_inputs
        k1 = 10

        ref = _reference_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=False, threshold=0.1, scale_edges_by_corr=False, wPCA=None,
        )
        got = _compute_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=False, chunksize=1000, threshold=0.1,
            scale_edges_by_corr=False, wPCA=None,
        )
        _assert_sparse_equal(got, ref)


class TestThreeSpecies:
    """Multi-species with pairwise per-block top-k."""

    def test_pairwise_topk(
        self, three_species_inputs: dict[str, Any]
    ) -> None:
        inp = three_species_inputs
        k1 = 10

        ref = _reference_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, threshold=0.1, scale_edges_by_corr=False, wPCA=None,
        )
        got = _compute_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, chunksize=20, threshold=0.1,
            scale_edges_by_corr=False, wPCA=None,
        )
        _assert_sparse_equal(got, ref)

    def test_with_scale_edges_by_corr(
        self, three_species_inputs: dict[str, Any], rng: np.random.Generator
    ) -> None:
        inp = three_species_inputs
        k1 = 8
        wPCA = rng.standard_normal((inp["N"], 40))

        ref = _reference_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, threshold=0.1, scale_edges_by_corr=True, wPCA=wPCA,
        )
        got = _compute_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, chunksize=15, threshold=0.1,
            scale_edges_by_corr=True, wPCA=wPCA,
        )
        _assert_sparse_equal(got, ref)


class TestCoclustering:
    """The neigh_from_keys (nfk) coclustering path."""

    def test_nfk_one_species(
        self, nfk_inputs: dict[str, Any]
    ) -> None:
        """One species uses coclustering; threshold is disabled (matches original)."""
        inp = nfk_inputs
        k1 = 12

        ref = _reference_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, threshold=0.0, scale_edges_by_corr=False, wPCA=None,
        )
        got = _compute_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], inp["B"],
            inp["offsets"], inp["n_cells"], inp["sids"], k1, inp["N"],
            pairwise=True, chunksize=30, threshold=0.0,
            scale_edges_by_corr=False, wPCA=None,
        )
        _assert_sparse_equal(got, ref)


class TestEdgeCases:
    """Degenerate inputs."""

    def test_single_species_no_cross(self, rng: np.random.Generator) -> None:
        """One species → no cross-species pairs → empty Dk."""
        sids = ["only"]
        n_cells = {"only": 50}
        offsets = {"only": 0}
        N = 50
        nnms_in = {"only": _make_sym_knn(rng, 50, k=5)}
        neigh_from_keys = {"only": False}
        B = spp.csr_matrix((N, N))  # empty

        got = _compute_mutual_graph(
            nnms_in, neigh_from_keys, B, offsets, n_cells, sids, 10, N,
            pairwise=True, chunksize=1000, threshold=0.1,
            scale_edges_by_corr=False, wPCA=None,
        )
        assert got.shape == (N, N)
        assert got.nnz == 0

    def test_empty_cross_species(
        self, two_species_inputs: dict[str, Any]
    ) -> None:
        """B has no entries → Dk should be empty."""
        inp = two_species_inputs
        B_empty = spp.csr_matrix((inp["N"], inp["N"]))

        got = _compute_mutual_graph(
            inp["nnms_in"], inp["neigh_from_keys"], B_empty,
            inp["offsets"], inp["n_cells"], inp["sids"], 10, inp["N"],
            pairwise=True, chunksize=1000, threshold=0.1,
            scale_edges_by_corr=False, wPCA=None,
        )
        assert got.nnz == 0
