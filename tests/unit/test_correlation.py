"""Equivalence tests for batched correlation computation in correlation.py.

The streaming path in ``_compute_pair_corrs`` computes ``Xavg`` in
per-pair-batch tiles instead of materialising the full N × G matrix. These
tests verify the output matches the materialised path bit-identically (or
to machine precision — the arithmetic is identical, order is the same).

They also verify the dict-free kernel against a reference Pearson/Xi
implementation built directly from NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as spp

from samap.core.correlation import _compute_pair_corrs, _corr_kernel, _xicorr

# ---------------------------------------------------------------------------
# Reference (pure-NumPy) correlation for a single pair
# ---------------------------------------------------------------------------


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    """Textbook Pearson — matches the kernel's formula exactly."""
    return float(
        ((x - x.mean()) * (y - y.mean()) / x.std() / y.std()).sum() / x.size
    )


def _ref_corr(
    nnms: spp.csr_matrix,
    Xs: spp.csc_matrix,
    p: np.ndarray,
    ps_int: np.ndarray,
    sp_starts: np.ndarray,
    sp_lens: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Reference: materialise Xavg, loop in pure Python, correlate."""
    Xavg = np.asarray(nnms.dot(Xs).todense())
    n_pairs = p.shape[0]
    res = np.zeros(n_pairs)
    for j in range(n_pairs):
        g1, g2 = p[j]
        s1, s2 = ps_int[j]
        st1, ln1 = sp_starts[s1], sp_lens[s1]
        st2, ln2 = sp_starts[s2], sp_lens[s2]

        xcol = Xavg[:, g1]
        ycol = Xavg[:, g2]
        xx = np.concatenate((xcol[st1 : st1 + ln1], xcol[st2 : st2 + ln2]))
        yy = np.concatenate((ycol[st1 : st1 + ln1], ycol[st2 : st2 + ln2]))

        if mode == "pearson":
            res[j] = _pearson_np(xx, yy)
        else:
            res[j] = _xicorr(xx, yy)
    return res


# ---------------------------------------------------------------------------
# Fixtures: synthetic 2-species inputs
# ---------------------------------------------------------------------------


def _make_knn(rng: np.random.Generator, n: int, k: int) -> spp.csr_matrix:
    """Sparse symmetric kNN with self-loops (averaging operator)."""
    rows, cols = [], []
    for i in range(n):
        nbrs = rng.choice(n, size=min(k, n), replace=False)
        if i not in nbrs:
            nbrs[0] = i
        for j in nbrs:
            rows.append(i)
            cols.append(j)
    M = spp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    M.sum_duplicates()
    M.data[:] = 1.0
    return M


@pytest.fixture
def corr_inputs(rng: np.random.Generator) -> dict[str, Any]:
    """Synthetic 2-species input for correlation testing.

    ~500 cells split 300/200, 200 genes split 120/80, ~500 gene pairs.
    Expression is sparse random (20% density) → smoothed Xavg is moderately
    dense (realistic).
    """
    n_a, n_b = 300, 200
    g_a, g_b = 120, 80
    N = n_a + n_b
    G = g_a + g_b

    # Row-normalised averaging operator over full manifold
    knn = _make_knn(rng, N, k=15)
    rs = np.asarray(knn.sum(1)).flatten()
    rs[rs == 0] = 1
    nnms = knn.multiply(1.0 / rs[:, None]).tocsr()

    # Block-diagonal expression: species A uses genes [0, g_a), B uses [g_a, G)
    Xa = spp.random(n_a, g_a, density=0.2, format="csr", random_state=1, dtype=np.float32)
    Xb = spp.random(n_b, g_b, density=0.2, format="csr", random_state=2, dtype=np.float32)
    Xs = spp.block_diag([Xa, Xb]).tocsc()

    # Species layout
    sp_starts = np.array([0, n_a], dtype=np.int64)
    sp_lens = np.array([n_a, n_b], dtype=np.int64)

    # Gene pairs: each is cross-species (gene from [0,g_a) × gene from [g_a,G))
    n_pairs = 500
    p1 = rng.integers(0, g_a, size=n_pairs)
    p2 = rng.integers(g_a, G, size=n_pairs)
    p = np.column_stack((p1, p2)).astype(np.int64)
    # species IDs: gene < g_a → species 0, else species 1
    ps_int = np.column_stack(
        (np.zeros(n_pairs, dtype=np.int64), np.ones(n_pairs, dtype=np.int64))
    )

    return {
        "nnms": nnms,
        "Xs": Xs,
        "p": p,
        "ps_int": ps_int,
        "sp_starts": sp_starts,
        "sp_lens": sp_lens,
        "N": N,
    }


@pytest.fixture
def corr_inputs_3sp(rng: np.random.Generator) -> dict[str, Any]:
    """3-species variant: pairs span all three species combinations."""
    n = [150, 100, 120]
    g = [60, 50, 40]
    N = sum(n)

    knn = _make_knn(rng, N, k=12)
    rs = np.asarray(knn.sum(1)).flatten()
    rs[rs == 0] = 1
    nnms = knn.multiply(1.0 / rs[:, None]).tocsr()

    X_blocks = [
        spp.random(n[i], g[i], density=0.2, format="csr", random_state=i + 10, dtype=np.float32)
        for i in range(3)
    ]
    Xs = spp.block_diag(X_blocks).tocsc()

    n_off = np.cumsum([0, *n])
    g_off = np.cumsum([0, *g])
    sp_starts = n_off[:-1].astype(np.int64)
    sp_lens = np.array(n, dtype=np.int64)

    # Generate pairs across all three species combinations
    n_pairs_per = 120
    p_list, ps_list = [], []
    combos = [(0, 1), (0, 2), (1, 2)]
    for s1, s2 in combos:
        p1 = rng.integers(g_off[s1], g_off[s1 + 1], size=n_pairs_per)
        p2 = rng.integers(g_off[s2], g_off[s2 + 1], size=n_pairs_per)
        p_list.append(np.column_stack((p1, p2)))
        ps_list.append(np.column_stack(
            (np.full(n_pairs_per, s1), np.full(n_pairs_per, s2))
        ))
    p = np.vstack(p_list).astype(np.int64)
    ps_int = np.vstack(ps_list).astype(np.int64)
    # shuffle so batches don't align with species combos
    perm = rng.permutation(p.shape[0])
    p, ps_int = p[perm], ps_int[perm]

    return {
        "nnms": nnms,
        "Xs": Xs,
        "p": p,
        "ps_int": ps_int,
        "sp_starts": sp_starts,
        "sp_lens": sp_lens,
        "N": N,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKernelAgainstReference:
    """Dict-free kernel matches a pure-NumPy reference."""

    def test_pearson_vs_numpy(self, corr_inputs: dict[str, Any]) -> None:
        inp = corr_inputs
        ref = _ref_corr(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], "pearson",
        )
        got = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "pearson", None,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)

    def test_xi_vs_numpy(self, corr_inputs: dict[str, Any]) -> None:
        inp = corr_inputs
        ref = _ref_corr(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], "xi",
        )
        got = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "xi", None,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


class TestStreamingEquivalence:
    """Streaming path (batch_size=int) matches materialised (batch_size=None)."""

    @pytest.mark.parametrize("batch_size", [1, 7, 64, 256, 10_000])
    def test_pearson_batched(
        self, corr_inputs: dict[str, Any], batch_size: int
    ) -> None:
        """Streaming Pearson == materialised, across a range of batch sizes.

        batch_size=1 is the strictest correctness check (every pair isolated);
        batch_size=10_000 exercises the single-batch fallthrough.
        """
        inp = corr_inputs
        ref = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "pearson", None,
        )
        got = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "pearson", batch_size,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("batch_size", [1, 32, 500])
    def test_xi_batched(
        self, corr_inputs: dict[str, Any], batch_size: int
    ) -> None:
        inp = corr_inputs
        ref = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "xi", None,
        )
        got = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "xi", batch_size,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("batch_size", [1, 50, 200])
    def test_three_species_pearson(
        self, corr_inputs_3sp: dict[str, Any], batch_size: int
    ) -> None:
        """3-species, shuffled pairs across all combos — exercises mixed-batch
        species indexing and gene-overlap between batches."""
        inp = corr_inputs_3sp
        ref = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "pearson", None,
        )
        got = _compute_pair_corrs(
            inp["nnms"], inp["Xs"], inp["p"], inp["ps_int"],
            inp["sp_starts"], inp["sp_lens"], inp["N"], "pearson", batch_size,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


class TestKernelDirect:
    """Low-level kernel sanity checks."""

    def test_kernel_empty_pairs(self) -> None:
        """Zero pairs → zero-length result."""
        sp_starts = np.array([0, 10], dtype=np.int64)
        sp_lens = np.array([10, 10], dtype=np.int64)
        # dummy CSC
        M = spp.csc_matrix((20, 5))
        res = _corr_kernel(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            sp_starts, sp_lens, M.indptr, M.indices, M.data, 20, True,
        )
        assert res.size == 0
