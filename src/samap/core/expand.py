"""Neighborhood expansion for cluster-adaptive kNN stitching.

Two implementations:

* :func:`_smart_expand_matpow` — the original algorithm. Computes iterated
  sparse matrix powers ``nnm^i`` to materialize hop-``i`` rings, then trims
  each ring to a per-cell budget. Simple but densifies badly at scale: the
  nnz of ``nnm^NH`` grows geometrically with NH and the kNN degree.

* :func:`_smart_expand_bfs` — a per-cell budget-capped BFS. Each cell
  independently walks its neighbourhood hop by hop, collecting nodes in
  (hop, weight) priority order up to its budget. Never materializes matrix
  powers; working set per cell is O(budget * k). Numba-parallel over cells.

Both return a sparse matrix whose *structure* is what matters — the caller
immediately binarizes the output. The two algorithms agree exactly when
every cell's budget ≥ its reachable-within-NH-hops count. When budgets
truncate, they may select different neighbours at the margin because
``matpow`` ranks ring members by path-sum weight while ``bfs`` ranks by
max incoming edge weight. See ``tests/unit/test_expand.py`` for a
characterisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
from numba import njit, prange

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Legacy matpow implementation
# ---------------------------------------------------------------------------


def _sparse_knn_ks(D: sp.sparse.coo_matrix, ks: NDArray[Any]) -> sp.sparse.coo_matrix:
    """Keep variable top-k values per row in sparse matrix."""
    D1 = D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:] = D1.row[idr]
    D1.col[:] = D1.col[idr]
    D1.data[:] = D1.data[idr]

    row, ind = np.unique(D1.row, return_index=True)
    ind = np.append(ind, D1.data.size)
    for i in range(ind.size - 1):
        idx = np.argsort(D1.data[ind[i] : ind[i + 1]])
        k = ks[row[i]]
        if idx.size > k:
            idx = idx[:-k] if k != 0 else idx
            D1.data[np.arange(ind[i], ind[i + 1])[idx]] = 0
    D1.eliminate_zeros()
    return D1


def _smart_expand_matpow(
    nnm: sp.sparse.csr_matrix, K: NDArray[Any], NH: int = 3
) -> sp.sparse.csr_matrix:
    """Original matrix-power neighbourhood expansion.

    Builds hop-``i`` rings via sparse matrix powers, then greedily fills
    each cell's budget ring by ring. Kept for regression testing and as a
    fallback.
    """
    stage0 = nnm.copy()
    S = [stage0]
    running = stage0
    for i in range(1, NH + 1):
        stage = running.dot(stage0)
        running = stage
        stage = stage.tolil()
        for j in range(i):
            stage[S[j].nonzero()] = 0
        stage = stage.tocsr()
        S.append(stage)

    for i in range(len(S)):
        s = _sparse_knn_ks(S[i], K).tocsr()
        a, c = np.unique(s.nonzero()[0], return_counts=True)
        numnz = np.zeros(s.shape[0], dtype="int32")
        numnz[a] = c
        K = K - numnz
        K[K < 0] = 0
        S[i] = s
    res = S[0]
    for i in range(1, len(S)):
        res = res + S[i]
    return res


# ---------------------------------------------------------------------------
# BFS implementation
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _bfs_expand_kernel(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    K_arr: NDArray[np.int64],
    NH: np.int64,
    n_cells: np.int64,
    buf_size: np.int64,
    out_cols: NDArray[np.int64],
    out_offsets: NDArray[np.int64],
    out_counts: NDArray[np.int64],
) -> None:
    """Per-cell budget-capped BFS over a CSR adjacency matrix.

    For each cell ``c`` (parallelized over cells):

    1. Seed the frontier with ``c``'s direct neighbours, weighted by the
       corresponding edge weights in ``nnm``.
    2. At each hop, sort the frontier by weight (descending) and collect
       unvisited nodes in that order until the cell's budget is met or the
       ring is exhausted.
    3. Expand only from *collected* nodes — their neighbours form the next
       frontier. Duplicates are resolved at collection time (first hit by
       highest weight wins, since the frontier is sorted).
    4. Stop after ``NH+1`` hops or when the budget is filled.

    Writes collected column indices for cell ``c`` into
    ``out_cols[out_offsets[c] : out_offsets[c] + out_counts[c]]``. Slots
    beyond ``out_counts[c]`` are unused (budget unfilled).
    """
    for c in prange(n_cells):
        budget = K_arr[c]
        if budget == 0:
            out_counts[c] = 0
            continue

        # Per-cell visited mask. Allocated inside the prange body so each
        # parallel iteration gets its own — numba makes this thread-local.
        visited = np.zeros(n_cells, dtype=np.bool_)
        visited[c] = True

        # Frontier double-buffer. buf_size is a safe upper bound on the
        # number of (node, weight) entries in a single hop's frontier:
        # at most ``budget`` nodes are collected per hop, each contributing
        # at most ``max_deg`` neighbours → budget * max_deg. The extra
        # +max_deg headroom covers the initial seed from ``c`` itself.
        front_idx = np.empty(buf_size, dtype=np.int64)
        front_w = np.empty(buf_size, dtype=np.float64)
        next_idx = np.empty(buf_size, dtype=np.int64)
        next_w = np.empty(buf_size, dtype=np.float64)

        # Seed: direct neighbours of c.
        n_front = 0
        for p in range(indptr[c], indptr[c + 1]):
            front_idx[n_front] = indices[p]
            front_w[n_front] = data[p]
            n_front += 1

        out_base = out_offsets[c]
        n_collected = 0

        for _hop in range(NH + 1):
            if n_front == 0 or n_collected >= budget:
                break

            # Rank this hop's candidates by weight, descending.
            # argsort is ascending → negate to get descending.
            order = np.argsort(-front_w[:n_front])

            n_next = 0
            for oi in range(n_front):
                node = front_idx[order[oi]]
                if visited[node]:
                    continue
                if n_collected >= budget:
                    break

                # Collect.
                out_cols[out_base + n_collected] = node
                n_collected += 1
                visited[node] = True

                # Expand: push this node's neighbours into next frontier.
                # Visited-filtering here is a *conservative* prune — more
                # filtering happens at collection time on the next hop
                # (handles duplicates within next_idx too).
                for p in range(indptr[node], indptr[node + 1]):
                    nb = indices[p]
                    if not visited[nb] and n_next < buf_size:
                        next_idx[n_next] = nb
                        next_w[n_next] = data[p]
                        n_next += 1

            # Swap buffers.
            front_idx, next_idx = next_idx, front_idx
            front_w, next_w = next_w, front_w
            n_front = n_next

        out_counts[c] = n_collected


def _smart_expand_bfs(
    nnm: sp.sparse.csr_matrix, K: NDArray[Any], NH: int = 3
) -> sp.sparse.csr_matrix:
    """BFS-based neighbourhood expansion.

    Algorithmically equivalent to :func:`_smart_expand_matpow` *when every
    cell's budget covers its full reachable set within ``NH+1`` hops*. When
    budgets truncate, the two may pick different marginal neighbours because
    they rank ring members differently (path-sum vs. max-edge weight). The
    output is binarized by the caller so only membership matters.

    Parameters
    ----------
    nnm
        ``(n, n)`` CSR adjacency / connectivity matrix. Need not be
        symmetric; only outgoing edges are walked.
    K
        ``(n,)`` per-cell collection budget (typically the cell's cluster
        size).
    NH
        Maximum number of *extra* hops beyond direct neighbours. Total
        hops walked is ``NH + 1``.

    Returns
    -------
    ``(n, n)`` CSR matrix with ``1.0`` at every collected ``(cell,
    neighbour)`` pair.
    """
    nnm = nnm.tocsr()
    n = nnm.shape[0]

    indptr = np.ascontiguousarray(nnm.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(nnm.indices, dtype=np.int64)
    data = np.ascontiguousarray(nnm.data, dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.int64)

    if n == 0:
        return sp.sparse.csr_matrix((0, 0), dtype=np.float64)

    # Preallocate output by budget. A cell may collect fewer than its
    # budget if its reachable component is small; out_counts records actuals.
    total = int(K.sum())
    out_cols = np.empty(max(total, 1), dtype=np.int64)
    out_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(K, out=out_offsets[1:])
    out_counts = np.zeros(n, dtype=np.int64)

    # Kernel buffer sizing: each hop expands from ≤ max_K collected nodes,
    # each with ≤ max_deg outgoing edges.
    degs = np.diff(indptr)
    max_deg = int(degs.max()) if degs.size else 0
    max_K = int(K.max()) if K.size else 0
    buf_size = max(max_K * max_deg + max_deg, 1)

    if total > 0 and max_deg > 0:
        _bfs_expand_kernel(
            indptr,
            indices,
            data,
            K,
            np.int64(NH),
            np.int64(n),
            np.int64(buf_size),
            out_cols,
            out_offsets,
            out_counts,
        )

    # Compact output: each cell's block in out_cols is sized by budget but
    # only the first out_counts[c] entries are valid. Build a mask.
    if total == 0:
        return sp.sparse.csr_matrix((n, n), dtype=np.float64)

    block_ids = np.repeat(np.arange(n, dtype=np.int64), K)
    within = np.arange(total, dtype=np.int64) - out_offsets[block_ids]
    valid = within < out_counts[block_ids]

    rows = block_ids[valid]
    cols = out_cols[:total][valid]
    vals = np.ones(rows.size, dtype=np.float64)

    return sp.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def _smart_expand(
    nnm: sp.sparse.csr_matrix,
    K: NDArray[Any],
    NH: int = 3,
    *,
    legacy: bool = False,
    bk: Any = None,
) -> sp.sparse.csr_matrix:
    """Expand each cell's neighbourhood to a per-cell budget via multi-hop walk.

    Parameters
    ----------
    nnm
        ``(n, n)`` sparse connectivity matrix.
    K
        ``(n,)`` per-cell budget (number of neighbours to collect).
    NH
        Number of extra hops beyond direct neighbours (default 3 → walks up
        to 4 hops).
    legacy
        If ``False`` (default), use the BFS algorithm — ~5× faster at 3k cells
        and memory-bounded. If ``True``, use the original matrix-power
        algorithm. Note: matpow wastes ~1 budget slot per cell on self-loops
        (a cell's 2-hop neighbourhood always includes itself); BFS avoids this
        and is arguably more correct, but will select slightly different
        marginal neighbours (~1% edge difference on the golden-suite data).
        Set ``legacy=True`` only if you need bit-exact reproduction of
        pre-3.0 SAMap output.
    bk
        Array backend. Currently unused (both paths are CPU-only numba);
        threaded through for future GPU work.
    """
    if legacy:
        return _smart_expand_matpow(nnm, K, NH=NH)
    return _smart_expand_bfs(nnm, K, NH=NH)
