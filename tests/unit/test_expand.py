"""Unit tests for :mod:`samap.core.expand`.

Compares the BFS neighbourhood expansion against the original matrix-power
implementation. The two are exactly equivalent when every cell's budget is
large enough to absorb its full reachable-within-NH-hops set. When budgets
truncate, they may pick different marginal neighbours (see module docstring).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

from samap.core.expand import (
    _smart_expand,
    _smart_expand_bfs,
    _smart_expand_matpow,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make_knn_graph(
    n_cells: int, k: int, n_clusters: int, rng: np.random.Generator
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build a weighted kNN graph over synthetic blob data.

    Returns the (symmetrised, row-normalised) connectivity matrix and an
    integer cluster-label array. Row normalisation mimics what scanpy's
    connectivities look like (weights in (0, 1], diagonal absent).
    """
    # Place cluster centres on a circle so they're well-separated.
    centres = (
        np.stack(
            [
                np.cos(2 * np.pi * np.arange(n_clusters) / n_clusters),
                np.sin(2 * np.pi * np.arange(n_clusters) / n_clusters),
            ],
            axis=1,
        )
        * 10.0
    )
    labels = rng.integers(0, n_clusters, size=n_cells)
    pts = centres[labels] + rng.normal(scale=1.0, size=(n_cells, 2))

    # Distance-weighted kNN, exclude self.
    A = kneighbors_graph(pts, n_neighbors=k, mode="distance", include_self=False)
    # Convert distances → similarities (Gaussian-ish), symmetrise, drop diag.
    A.data = np.exp(-A.data / A.data.mean())
    A = A.maximum(A.T).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, labels


def _edge_set(A: sp.spmatrix) -> set[tuple[int, int]]:
    A = A.tocoo()
    return set(zip(A.row.tolist(), A.col.tolist(), strict=True))


# ---------------------------------------------------------------------------
# Exact-equivalence regime: budget ≥ reachable set.
# ---------------------------------------------------------------------------


def test_bfs_matches_matpow_when_budget_covers_reachable(rng: np.random.Generator) -> None:
    """With budget ≥ reachable-within-NH-hops, BFS and matpow must agree exactly.

    Here NH=1 (two hops total) on a small sparse graph; the reachable set per
    cell is well under the budget, so both algorithms collect the full set
    and the choice of in-ring ranking is irrelevant.
    """
    n, k = 120, 4
    nnm, _ = _make_knn_graph(n, k, n_clusters=6, rng=rng)

    # Generous budget — well above the 2-hop reachable count for k=4.
    K = np.full(n, 60, dtype=np.int64)

    out_old = _smart_expand_matpow(nnm, K.copy(), NH=1)
    out_new = _smart_expand_bfs(nnm, K.copy(), NH=1)

    old_edges = _edge_set(out_old)
    new_edges = _edge_set(out_new)

    # matpow can include self-loops at even hops (an nnm^2 diagonal entry
    # survives the ring subtraction if nnm itself has no diagonal). BFS
    # never collects self. Strip self-loops from the matpow output before
    # comparing.
    old_edges = {(r, c) for (r, c) in old_edges if r != c}

    assert old_edges == new_edges, (
        f"edge-set mismatch: "
        f"{len(old_edges - new_edges)} matpow-only, "
        f"{len(new_edges - old_edges)} bfs-only"
    )


def test_bfs_matches_matpow_single_hop(rng: np.random.Generator) -> None:
    """With NH=0 (direct neighbours only), both algorithms are pure top-k.

    This is the trivial case — no multi-hop expansion, so the in-ring
    ranking is identical (both use the edge weights directly).
    """
    n, k = 200, 10
    nnm, labels = _make_knn_graph(n, k, n_clusters=5, rng=rng)

    # Per-cell budget = cluster size (typical usage).
    _, ix, counts = np.unique(labels, return_inverse=True, return_counts=True)
    K = counts[ix].astype(np.int64)

    out_old = _smart_expand_matpow(nnm, K.copy(), NH=0)
    out_new = _smart_expand_bfs(nnm, K.copy(), NH=0)

    assert _edge_set(out_old) == _edge_set(out_new)


# ---------------------------------------------------------------------------
# Truncation regime: budget < reachable set. Characterise divergence.
# ---------------------------------------------------------------------------


def test_bfs_near_matpow_when_budget_truncates(rng: np.random.Generator) -> None:
    """With tight budgets, BFS and matpow may differ at the margin.

    Both prioritise by hop distance, so divergence is confined to the *last*
    ring a cell draws from and only when that ring overflows the remaining
    budget. We assert high Jaccard similarity and that per-cell output
    sizes match exactly (both fill the same budget).
    """
    n, k = 500, 20
    nnm, labels = _make_knn_graph(n, k, n_clusters=8, rng=rng)

    _, ix, counts = np.unique(labels, return_inverse=True, return_counts=True)
    K = counts[ix].astype(np.int64)

    out_old = _smart_expand_matpow(nnm, K.copy(), NH=3)
    out_new = _smart_expand_bfs(nnm, K.copy(), NH=3)

    # Strip self-loops from matpow (see exact-equivalence test).
    old_edges = {(r, c) for (r, c) in _edge_set(out_old) if r != c}
    new_edges = _edge_set(out_new)

    inter = len(old_edges & new_edges)
    union = len(old_edges | new_edges)
    jaccard = inter / union if union else 1.0

    # Per-cell output cardinality should be very close (both fill budget or
    # exhaust reachable set; matpow may have +1 from a self-loop).
    old_nnz = np.asarray((out_old != 0).sum(axis=1)).ravel()
    new_nnz = np.asarray((out_new != 0).sum(axis=1)).ravel()
    # Allow matpow up to +1 per cell (the self-loop).
    assert np.all(new_nnz <= old_nnz)
    assert np.all(old_nnz - new_nnz <= 1)

    # In practice Jaccard is ~0.95+ here. 0.9 is a conservative floor —
    # if this fails the algorithms have diverged meaningfully and should be
    # investigated, not just have the threshold lowered.
    assert jaccard > 0.9, f"Jaccard={jaccard:.3f} — BFS diverged from matpow"


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_bfs_output_binarized(rng: np.random.Generator) -> None:
    """BFS output data is all 1.0 — structure is the only signal."""
    n, k = 100, 8
    nnm, _ = _make_knn_graph(n, k, n_clusters=4, rng=rng)
    K = np.full(n, 30, dtype=np.int64)
    out = _smart_expand_bfs(nnm, K, NH=2)
    assert out.nnz > 0
    np.testing.assert_array_equal(out.data, np.ones(out.nnz))


def test_bfs_no_self_loops(rng: np.random.Generator) -> None:
    """BFS never collects a cell as its own neighbour."""
    n, k = 100, 8
    nnm, _ = _make_knn_graph(n, k, n_clusters=4, rng=rng)
    K = np.full(n, 50, dtype=np.int64)
    out = _smart_expand_bfs(nnm, K, NH=3)
    assert out.diagonal().sum() == 0


def test_bfs_respects_budget(rng: np.random.Generator) -> None:
    """No cell collects more neighbours than its budget."""
    n, k = 200, 12
    nnm, labels = _make_knn_graph(n, k, n_clusters=5, rng=rng)
    _, ix, counts = np.unique(labels, return_inverse=True, return_counts=True)
    K = counts[ix].astype(np.int64)
    out = _smart_expand_bfs(nnm, K, NH=3)
    nnz_per_row = np.asarray((out != 0).sum(axis=1)).ravel()
    assert np.all(nnz_per_row <= K)


def test_bfs_zero_budget(rng: np.random.Generator) -> None:
    """Cells with K=0 contribute nothing."""
    n, k = 50, 5
    nnm, _ = _make_knn_graph(n, k, n_clusters=3, rng=rng)
    K = np.zeros(n, dtype=np.int64)
    K[::3] = 10  # only every third cell gets a budget
    out = _smart_expand_bfs(nnm, K, NH=2)
    nnz_per_row = np.asarray((out != 0).sum(axis=1)).ravel()
    assert np.all(nnz_per_row[K == 0] == 0)
    assert np.any(nnz_per_row[K > 0] > 0)


def test_bfs_disconnected_component(rng: np.random.Generator) -> None:
    """BFS on a disconnected node collects nothing (gracefully)."""
    n = 50
    nnm, _ = _make_knn_graph(n, k=5, n_clusters=3, rng=rng)
    # Isolate the last cell.
    nnm = nnm.tolil()
    nnm[n - 1, :] = 0
    nnm[:, n - 1] = 0
    nnm = nnm.tocsr()
    nnm.eliminate_zeros()
    K = np.full(n, 20, dtype=np.int64)
    out = _smart_expand_bfs(nnm, K, NH=3)
    # Isolated cell has no neighbours to collect.
    assert out.getrow(n - 1).nnz == 0
    # Other cells still work.
    assert out.getrow(0).nnz > 0


def test_bfs_empty_graph() -> None:
    """Zero-cell input → zero-cell output."""
    nnm = sp.csr_matrix((0, 0), dtype=np.float64)
    K = np.array([], dtype=np.int64)
    out = _smart_expand_bfs(nnm, K, NH=3)
    assert out.shape == (0, 0)
    assert out.nnz == 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_dispatch_legacy_true_calls_matpow(rng: np.random.Generator) -> None:
    """_smart_expand(legacy=True) delegates to matpow."""
    n, k = 80, 6
    nnm, _ = _make_knn_graph(n, k, n_clusters=4, rng=rng)
    K = np.full(n, 30, dtype=np.int64)
    out_dispatch = _smart_expand(nnm, K.copy(), NH=2, legacy=True)
    out_direct = _smart_expand_matpow(nnm, K.copy(), NH=2)
    assert _edge_set(out_dispatch) == _edge_set(out_direct)


def test_dispatch_legacy_false_calls_bfs(rng: np.random.Generator) -> None:
    """_smart_expand(legacy=False) delegates to BFS."""
    n, k = 80, 6
    nnm, _ = _make_knn_graph(n, k, n_clusters=4, rng=rng)
    K = np.full(n, 30, dtype=np.int64)
    out_dispatch = _smart_expand(nnm, K.copy(), NH=2, legacy=False)
    out_direct = _smart_expand_bfs(nnm, K.copy(), NH=2)
    assert _edge_set(out_dispatch) == _edge_set(out_direct)
