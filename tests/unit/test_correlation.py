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

from samap.core._backend import Backend
from samap.core.correlation import (
    _compute_pair_corrs,
    _corr_kernel,
    _replace,
    _replace_vectorized,
    _resolve_batch_size,
    _xicorr,
    replace_corr,
)

# ---------------------------------------------------------------------------
# Reference (pure-NumPy) correlation for a single pair
# ---------------------------------------------------------------------------


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    """Textbook Pearson — matches the kernel's formula exactly."""
    return float(((x - x.mean()) * (y - y.mean()) / x.std() / y.std()).sum() / x.size)


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
    ps_int = np.column_stack((np.zeros(n_pairs, dtype=np.int64), np.ones(n_pairs, dtype=np.int64)))

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
        ps_list.append(np.column_stack((np.full(n_pairs_per, s1), np.full(n_pairs_per, s2))))
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
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            "pearson",
        )
        got = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "pearson",
            None,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)

    def test_xi_vs_numpy(self, corr_inputs: dict[str, Any]) -> None:
        inp = corr_inputs
        ref = _ref_corr(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            "xi",
        )
        got = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "xi",
            None,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


class TestStreamingEquivalence:
    """Streaming path (batch_size=int) matches materialised (batch_size=None)."""

    @pytest.mark.parametrize("batch_size", [1, 7, 64, 256, 10_000])
    def test_pearson_batched(self, corr_inputs: dict[str, Any], batch_size: int) -> None:
        """Streaming Pearson == materialised, across a range of batch sizes.

        batch_size=1 is the strictest correctness check (every pair isolated);
        batch_size=10_000 exercises the single-batch fallthrough.
        """
        inp = corr_inputs
        ref = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "pearson",
            None,
        )
        got = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "pearson",
            batch_size,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("batch_size", [1, 32, 500])
    def test_xi_batched(self, corr_inputs: dict[str, Any], batch_size: int) -> None:
        inp = corr_inputs
        ref = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "xi",
            None,
        )
        got = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "xi",
            batch_size,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("batch_size", [1, 50, 200])
    def test_three_species_pearson(self, corr_inputs_3sp: dict[str, Any], batch_size: int) -> None:
        """3-species, shuffled pairs across all combos — exercises mixed-batch
        species indexing and gene-overlap between batches."""
        inp = corr_inputs_3sp
        ref = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "pearson",
            None,
        )
        got = _compute_pair_corrs(
            inp["nnms"],
            inp["Xs"],
            inp["p"],
            inp["ps_int"],
            inp["sp_starts"],
            inp["sp_lens"],
            inp["N"],
            "pearson",
            batch_size,
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
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            sp_starts,
            sp_lens,
            M.indptr,
            M.indices,
            M.data,
            20,
            True,
        )
        assert res.size == 0


# ---------------------------------------------------------------------------
# _replace (per-pair Pearson over dense wPCA rows)
# ---------------------------------------------------------------------------


@pytest.fixture
def replace_inputs(rng: np.random.Generator) -> dict[str, Any]:
    """Dense embedding + random index pairs for _replace tests."""
    n, d = 800, 50
    X = rng.standard_normal((n, d)).astype(np.float64)
    n_pairs = 1000
    xi = rng.integers(0, n, size=n_pairs).astype(np.int64)
    yi = rng.integers(0, n, size=n_pairs).astype(np.int64)
    return {"X": X, "xi": xi, "yi": yi, "n": n, "d": d, "n_pairs": n_pairs}


class TestReplaceVectorized:
    """_replace_vectorized matches numba _replace and pure-numpy reference."""

    def test_against_numpy_corrcoef(self, replace_inputs: dict[str, Any]) -> None:
        """Vectorised form matches np.corrcoef pairwise (rtol=1e-12)."""
        inp = replace_inputs
        bk = Backend("cpu")

        got = _replace_vectorized(inp["X"], inp["xi"], inp["yi"], bk)

        # Reference via np.corrcoef — O(n_pairs) loop, but authoritative
        ref = np.array(
            [np.corrcoef(inp["X"][i], inp["X"][j])[0, 1] for i, j in zip(inp["xi"], inp["yi"])]
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)

    def test_against_numba(self, replace_inputs: dict[str, Any]) -> None:
        """Vectorised form matches numba _replace (the CPU fast path)."""
        inp = replace_inputs
        bk = Backend("cpu")

        numba_res = _replace(inp["X"], inp["xi"], inp["yi"])
        vec_res = _replace_vectorized(inp["X"], inp["xi"], inp["yi"], bk)

        np.testing.assert_allclose(vec_res, numba_res, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("batch_size", [1, 7, 100, 500, 2000])
    def test_batched_matches_full(self, replace_inputs: dict[str, Any], batch_size: int) -> None:
        """Chunked vectorised == single-shot vectorised (all batch sizes).

        batch_size=1 is the tightest correctness probe; 2000 > n_pairs
        exercises the fallthrough.
        """
        inp = replace_inputs
        bk = Backend("cpu")

        full = _replace_vectorized(inp["X"], inp["xi"], inp["yi"], bk, batch_size=None)
        chunked = _replace_vectorized(inp["X"], inp["xi"], inp["yi"], bk, batch_size=batch_size)
        np.testing.assert_allclose(chunked, full, rtol=0, atol=0)

    def test_float32_input(self, replace_inputs: dict[str, Any]) -> None:
        """float32 input → float64 output, matches float64 input path.

        wPCA is often stored float32 for memory; the vectorised form
        upcasts internally to match _replace's float64 arithmetic.
        """
        inp = replace_inputs
        bk = Backend("cpu")
        X32 = inp["X"].astype(np.float32)

        res32 = _replace_vectorized(X32, inp["xi"], inp["yi"], bk)
        res64 = _replace_vectorized(inp["X"], inp["xi"], inp["yi"], bk)

        assert res32.dtype == np.float64
        # float32 input has less precision → looser tolerance
        np.testing.assert_allclose(res32, res64, rtol=1e-5, atol=1e-7)

    def test_zero_variance_row(self, rng: np.random.Generator) -> None:
        """Constant row → std=0 → nan (matches _replace behaviour)."""
        bk = Backend("cpu")
        X = rng.standard_normal((10, 20))
        X[3, :] = 5.0  # constant → zero variance

        xi = np.array([3, 0], dtype=np.int64)
        yi = np.array([1, 2], dtype=np.int64)

        vec = _replace_vectorized(X, xi, yi, bk)
        numba = _replace(X, xi, yi)

        assert np.isnan(vec[0])
        assert np.isnan(numba[0])
        np.testing.assert_allclose(vec[1], numba[1], rtol=1e-12)


class TestReplaceCorrDispatcher:
    """replace_corr routes to numba on CPU, vectorised on GPU."""

    def test_cpu_backend_uses_numba(self, replace_inputs: dict[str, Any]) -> None:
        """CPU backend → numba path; result matches _replace directly."""
        inp = replace_inputs
        bk = Backend("cpu")

        disp = replace_corr(inp["X"], inp["xi"], inp["yi"], bk)
        numba = _replace(inp["X"], inp["xi"], inp["yi"])
        # CPU dispatch IS the numba path → bit-identical
        np.testing.assert_array_equal(disp, numba)

    def test_bk_none_defaults_to_numba(self, replace_inputs: dict[str, Any]) -> None:
        """bk=None (backward-compat) → numba path."""
        inp = replace_inputs
        disp = replace_corr(inp["X"], inp["xi"], inp["yi"], bk=None)
        numba = _replace(inp["X"], inp["xi"], inp["yi"])
        np.testing.assert_array_equal(disp, numba)

    def test_mock_gpu_uses_vectorized(self, replace_inputs: dict[str, Any]) -> None:
        """Mock Backend with gpu=True → vectorised path.

        We can't test a real GPU path on CI; this verifies the dispatch
        logic by constructing a duck-typed backend with gpu=True + xp=numpy.
        """

        class _MockGPUBackend:
            gpu = True
            xp = np  # numpy stands in for cupy here

        inp = replace_inputs
        bk = _MockGPUBackend()

        disp = replace_corr(inp["X"], inp["xi"], inp["yi"], bk, batch_size=100)
        ref = _replace_vectorized(inp["X"], inp["xi"], inp["yi"], bk, batch_size=100)
        np.testing.assert_array_equal(disp, ref)
        # and that it matches numba to fp tolerance
        numba = _replace(inp["X"], inp["xi"], inp["yi"])
        np.testing.assert_allclose(disp, numba, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# _resolve_batch_size auto-selection heuristic
# ---------------------------------------------------------------------------


class TestResolveBatchSize:
    """Auto-selection of materialised vs streaming based on estimated memory."""

    def test_explicit_passthrough(self, rng: np.random.Generator) -> None:
        """Explicit batch_size (non-'auto') is returned unchanged."""
        nnms = spp.eye(100, format="csr")
        Xs = spp.random(100, 50, density=0.1, format="csc")

        # All explicit values pass through untouched — including None.
        assert _resolve_batch_size(None, nnms, Xs) is None
        assert _resolve_batch_size(32, nnms, Xs) == 32
        assert _resolve_batch_size(9999, nnms, Xs) == 9999

    def test_tiny_data_materialises(self, rng: np.random.Generator) -> None:
        """Toy-scale data (hundreds of cells) → auto picks materialised.

        At 500 cells × 200 genes, even 100% density is 800 KB — far under
        the 2 GB default threshold.
        """
        # Realistic toy: 500 cells, avg 15 neighbours, 20% expression density
        nnms = spp.random(500, 500, density=15 / 500, format="csr")
        Xs = spp.random(500, 200, density=0.2, format="csc")

        got = _resolve_batch_size("auto", nnms, Xs, mem_threshold_gb=2.0)
        assert got is None

    def test_million_cell_streams(self) -> None:
        """Million-cell scale → auto picks streaming.

        1M cells × 10k genes × ~50% density (after kNN fill-in from 5%
        input density with k~20) ≈ 60 GB CSC. Well over any threshold.
        We mock shapes/nnz rather than allocating a real million-entry
        matrix — _resolve_batch_size only reads .shape and .nnz.
        """

        class _MockSparse:
            def __init__(self, shape: tuple[int, int], nnz: int) -> None:
                self.shape = shape
                self.nnz = nnz

        n_cells, n_genes = 1_000_000, 10_000
        k = 20
        expr_nnz = int(n_cells * n_genes * 0.05)  # 5% expression density

        nnms = _MockSparse(shape=(n_cells, n_cells), nnz=n_cells * k)
        Xs = _MockSparse(shape=(n_cells, n_genes), nnz=expr_nnz)

        got = _resolve_batch_size("auto", nnms, Xs, mem_threshold_gb=2.0)
        assert got == 512

    def test_threshold_boundary(self) -> None:
        """Crossing the threshold flips the decision.

        Fixed shapes → fixed estimate. Vary mem_threshold_gb above and
        below the estimate to verify the boundary logic.
        """

        class _MockSparse:
            def __init__(self, shape: tuple[int, int], nnz: int) -> None:
                self.shape = shape
                self.nnz = nnz

        # 100k cells × 5k genes, k=15, density 10% → output density ~80%
        # → est = 100k * 5k * 0.8 * 12 bytes ≈ 4.8 GB
        n_cells, n_genes = 100_000, 5_000
        nnms = _MockSparse(shape=(n_cells, n_cells), nnz=n_cells * 15)
        Xs = _MockSparse(shape=(n_cells, n_genes), nnz=int(n_cells * n_genes * 0.10))

        # threshold above estimate → materialise
        assert _resolve_batch_size("auto", nnms, Xs, mem_threshold_gb=10.0) is None
        # threshold below estimate → stream
        assert _resolve_batch_size("auto", nnms, Xs, mem_threshold_gb=1.0) == 512

    def test_zero_sized_inputs(self) -> None:
        """Degenerate 0-cell / 0-gene inputs → materialise (trivial)."""
        nnms = spp.csr_matrix((0, 0))
        Xs = spp.csc_matrix((0, 0))
        assert _resolve_batch_size("auto", nnms, Xs) is None

        nnms = spp.eye(50, format="csr")
        Xs = spp.csc_matrix((50, 0))
        assert _resolve_batch_size("auto", nnms, Xs) is None
