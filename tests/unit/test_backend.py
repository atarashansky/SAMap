"""Unit tests for samap.core._backend — the CPU/GPU dispatch layer.

CPU tests run unconditionally. GPU tests are skipped unless cupy is installed
*and* a CUDA device is visible (they provide coverage on CUDA CI only).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from samap.core._backend import HAS_CUPY, Backend, COOBuilder

if HAS_CUPY:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse

    _CUDA = cp.is_available()
else:
    _CUDA = False

gpu_only = pytest.mark.skipif(
    not _CUDA, reason="requires cupy + a CUDA device"
)


# ---------------------------------------------------------------------------
# Backend construction / device selection
# ---------------------------------------------------------------------------


class TestBackendInit:
    def test_cpu_backend(self) -> None:
        bk = Backend("cpu")
        assert bk.device == "cpu"
        assert bk.gpu is False
        assert bk.xp is np
        assert bk.sp is sp
        # spla should expose svds/LinearOperator
        assert hasattr(bk.spla, "svds")
        assert hasattr(bk.spla, "LinearOperator")

    def test_auto_without_cuda(self) -> None:
        """On a machine without cupy/CUDA, auto → cpu silently."""
        bk = Backend("auto")
        # Either we have a GPU (CI) or we don't (dev laptop); either is valid.
        assert bk.device in ("cpu", "cuda")
        assert bk.gpu == (bk.device == "cuda")

    @pytest.mark.skipif(HAS_CUPY, reason="tests the no-cupy error path")
    def test_cuda_without_cupy_raises(self) -> None:
        with pytest.raises(RuntimeError, match="cupy is not installed"):
            Backend("cuda")

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'cpu', 'cuda', or 'auto'"):
            Backend("tpu")  # type: ignore[arg-type]

    def test_repr(self) -> None:
        bk = Backend("cpu")
        assert "cpu" in repr(bk)
        assert "gpu=False" in repr(bk)

    @gpu_only
    def test_cuda_backend(self) -> None:
        bk = Backend("cuda")
        assert bk.device == "cuda"
        assert bk.gpu is True
        assert bk.xp is cp
        assert bk.sp is cpx_sparse


# ---------------------------------------------------------------------------
# Compat shims — CPU
# ---------------------------------------------------------------------------


@pytest.fixture
def bk_cpu() -> Backend:
    return Backend("cpu")


@pytest.fixture
def small_csr() -> sp.csr_matrix:
    """A fixed 3x3 CSR with known structure."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    row = np.array([0, 0, 1, 2])
    col = np.array([0, 2, 1, 0])
    return sp.csr_matrix((data, (row, col)), shape=(3, 3))


class TestNonzeroCPU:
    def test_nonzero_on_csr(self, bk_cpu: Backend, small_csr: sp.csr_matrix) -> None:
        rows, cols = bk_cpu.nonzero(small_csr)
        # scipy returns sorted (row-major) for CSR
        np.testing.assert_array_equal(rows, [0, 0, 1, 2])
        np.testing.assert_array_equal(cols, [0, 2, 1, 0])

    def test_nonzero_on_dense(self, bk_cpu: Backend) -> None:
        A = np.array([[0, 5, 0], [0, 0, 7]])
        rows, cols = bk_cpu.nonzero(A)
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(cols, [1, 2])


class TestSparseFromCoo:
    def test_basic_csr(self, bk_cpu: Backend) -> None:
        data = [1.0, 2.0, 3.0]
        row = [0, 1, 2]
        col = [2, 1, 0]
        A = bk_cpu.sparse_from_coo(data, row, col, shape=(3, 3), fmt="csr")
        assert A.format == "csr"
        expected = np.array([[0, 0, 1], [0, 2, 0], [3, 0, 0]], dtype=float)
        np.testing.assert_array_equal(A.toarray(), expected)

    def test_csc_format(self, bk_cpu: Backend) -> None:
        A = bk_cpu.sparse_from_coo([5.0], [1], [1], shape=(2, 2), fmt="csc")
        assert A.format == "csc"
        assert A[1, 1] == 5.0

    def test_duplicates_are_summed(self, bk_cpu: Backend) -> None:
        # Two entries at (0, 0) → summed
        A = bk_cpu.sparse_from_coo(
            [1.0, 2.0, 10.0], [0, 0, 1], [0, 0, 1], shape=(2, 2)
        )
        assert A[0, 0] == 3.0
        assert A[1, 1] == 10.0


class TestSetdiag:
    def test_setdiag_zero_eliminates(self, bk_cpu: Backend) -> None:
        A = sp.csr_matrix(np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype=float))
        nnz_before = A.nnz
        bk_cpu.setdiag(A, 0)
        np.testing.assert_array_equal(A.diagonal(), [0, 0, 0])
        # Diagonal entries removed from structure
        assert A.nnz == nnz_before - 3

    def test_setdiag_scalar_nonzero(self, bk_cpu: Backend) -> None:
        A = sp.csr_matrix(np.eye(3))
        bk_cpu.setdiag(A, 7.0)
        np.testing.assert_array_equal(A.diagonal(), [7, 7, 7])
        # Off-diagonal untouched
        assert A[0, 1] == 0.0

    def test_setdiag_array(self, bk_cpu: Backend) -> None:
        A = sp.csr_matrix(np.zeros((3, 3)))
        bk_cpu.setdiag(A, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(A.diagonal(), [10, 20, 30])

    def test_setdiag_converts_to_csr(self, bk_cpu: Backend) -> None:
        A = sp.csc_matrix(np.eye(3))
        out = bk_cpu.setdiag(A, 0)
        assert out.format == "csr"

    def test_setdiag_no_warning(self, bk_cpu: Backend) -> None:
        """The shim suppresses SparseEfficiencyWarning on structural change."""
        A = sp.csr_matrix((3, 3))  # all-zero, so setdiag changes structure
        with warnings_error(sp.SparseEfficiencyWarning):
            bk_cpu.setdiag(A, 1.0)


def warnings_error(*categories):
    """Context manager: turn given warning categories into errors."""
    import warnings

    class _Ctx:
        def __enter__(self):
            self._mgr = warnings.catch_warnings()
            self._mgr.__enter__()
            for cat in categories:
                warnings.simplefilter("error", cat)

        def __exit__(self, *exc):
            return self._mgr.__exit__(*exc)

    return _Ctx()


class TestSvds:
    def test_cpu_passes_solver(self, bk_cpu: Backend) -> None:
        """On CPU, solver kwarg passes through to scipy."""
        # Build a rank-2 5x4 matrix
        rng = np.random.default_rng(42)
        A = sp.random(5, 4, density=0.6, random_state=rng)
        u, s, vt = bk_cpu.svds(A, k=2, solver="arpack")
        assert s.shape == (2,)
        assert u.shape == (5, 2)
        assert vt.shape == (2, 4)
        # Singular values are non-negative
        assert (s >= 0).all()

    def test_cpu_svds_numerics(self, bk_cpu: Backend) -> None:
        """Sanity check: recovered singular values match numpy.linalg.svd."""
        rng = np.random.default_rng(0)
        M = rng.standard_normal((6, 4))
        A = sp.csr_matrix(M)
        _, s, _ = bk_cpu.svds(A, k=3)
        s_true = np.linalg.svd(M, compute_uv=False)
        # svds returns ascending; np.linalg.svd descending
        np.testing.assert_allclose(np.sort(s)[::-1], s_true[:3], rtol=1e-6)


class TestLinearOperator:
    def test_dispatch_to_scipy(self, bk_cpu: Backend) -> None:
        # 3x3 identity via matvec
        lo = bk_cpu.LinearOperator(
            shape=(3, 3), matvec=lambda x: x, rmatvec=lambda x: x, dtype=np.float64
        )
        assert isinstance(lo, ScipyLinearOperator)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(lo @ x, x)

    def test_in_svds(self, bk_cpu: Backend) -> None:
        """LinearOperator can feed into svds (implicit-matrix SVD)."""
        M = np.diag([5.0, 3.0, 1.0, 0.1])
        lo = bk_cpu.LinearOperator(
            shape=(4, 4),
            matvec=lambda x: M @ x,
            rmatvec=lambda x: M.T @ x,
            dtype=np.float64,
        )
        _, s, _ = bk_cpu.svds(lo, k=2)
        np.testing.assert_allclose(sorted(s), [3.0, 5.0], rtol=1e-6)


# ---------------------------------------------------------------------------
# Data movement — CPU backend (identity ops)
# ---------------------------------------------------------------------------


class TestDataMovementCPU:
    def test_to_device_noop_dense(self, bk_cpu: Backend) -> None:
        x = np.arange(5.0)
        assert bk_cpu.to_device(x) is x

    def test_to_device_noop_sparse(self, bk_cpu: Backend, small_csr) -> None:
        assert bk_cpu.to_device(small_csr) is small_csr

    def test_to_host_noop(self, bk_cpu: Backend) -> None:
        x = np.arange(5.0)
        assert bk_cpu.to_host(x) is x

    def test_asfortran_noop_cpu(self, bk_cpu: Backend) -> None:
        x = np.zeros((3, 4), order="C")
        out = bk_cpu.asfortran_if_gpu(x)
        assert out is x
        assert out.flags.c_contiguous  # unchanged

    def test_free_pool_noop(self, bk_cpu: Backend) -> None:
        # Should not raise
        bk_cpu.free_pool()


# ---------------------------------------------------------------------------
# COOBuilder
# ---------------------------------------------------------------------------


class TestCOOBuilder:
    def test_single_adds(self, bk_cpu: Backend) -> None:
        b = COOBuilder(bk_cpu, shape=(3, 3))
        b.add(0, 1, 5.0)
        b.add(1, 2, 3.0)
        b.add(2, 0, 7.0)
        A = b.finalize(fmt="csr")
        expected = np.array([[0, 5, 0], [0, 0, 3], [7, 0, 0]], dtype=float)
        np.testing.assert_array_equal(A.toarray(), expected)
        assert A.format == "csr"

    def test_batch_adds(self, bk_cpu: Backend) -> None:
        b = COOBuilder(bk_cpu, shape=(2, 2))
        b.add_batch([0, 1], [0, 1], [1.0, 2.0])
        A = b.finalize()
        np.testing.assert_array_equal(A.toarray(), [[1, 0], [0, 2]])

    def test_mixed_adds(self, bk_cpu: Backend) -> None:
        b = COOBuilder(bk_cpu, shape=(3, 3))
        b.add(0, 0, 1.0)
        b.add_batch([1, 2], [1, 2], [2.0, 3.0])
        b.add(0, 2, 4.0)
        A = b.finalize()
        expected = np.array([[1, 0, 4], [0, 2, 0], [0, 0, 3]], dtype=float)
        np.testing.assert_array_equal(A.toarray(), expected)

    def test_empty_builder(self, bk_cpu: Backend) -> None:
        b = COOBuilder(bk_cpu, shape=(4, 4))
        A = b.finalize()
        assert A.shape == (4, 4)
        assert A.nnz == 0

    def test_custom_dtype(self, bk_cpu: Backend) -> None:
        b = COOBuilder(bk_cpu, shape=(2, 2), dtype=np.float32)
        b.add(0, 0, 1.0)
        A = b.finalize()
        assert A.dtype == np.float32

    def test_duplicates_summed(self, bk_cpu: Backend) -> None:
        b = COOBuilder(bk_cpu, shape=(2, 2))
        b.add(0, 0, 1.0)
        b.add(0, 0, 2.0)
        b.add_batch([0], [0], [3.0])
        A = b.finalize()
        assert A[0, 0] == 6.0

    def test_csc_finalize(self, bk_cpu: Backend) -> None:
        b = COOBuilder(bk_cpu, shape=(3, 3))
        b.add(1, 1, 9.0)
        A = b.finalize(fmt="csc")
        assert A.format == "csc"
        assert A[1, 1] == 9.0

    def test_matches_lil_fancy_assign(self, bk_cpu: Backend) -> None:
        """COOBuilder should produce the same matrix as the lil_matrix
        fancy-index pattern it replaces."""
        rows = np.array([0, 2, 1, 3, 2])
        cols = np.array([1, 0, 3, 2, 2])
        vals = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        # Reference: lil fancy assignment (note: lil OVERWRITES on duplicate
        # keys, not sums — so we use unique keys here to keep semantics
        # aligned with COO summing).
        lil = sp.lil_matrix((4, 4))
        lil[rows, cols] = vals
        ref = lil.tocsr()

        # COOBuilder path
        b = COOBuilder(bk_cpu, shape=(4, 4))
        b.add_batch(rows, cols, vals)
        ours = b.finalize(fmt="csr")

        np.testing.assert_array_equal(ours.toarray(), ref.toarray())

    def test_end_to_end_dot(self, bk_cpu: Backend) -> None:
        """Build two matrices with COOBuilder and multiply them."""
        b1 = COOBuilder(bk_cpu, shape=(2, 3))
        b1.add_batch([0, 0, 1], [0, 2, 1], [1.0, 2.0, 3.0])
        A = b1.finalize()

        b2 = COOBuilder(bk_cpu, shape=(3, 2))
        b2.add_batch([0, 1, 2], [0, 1, 0], [4.0, 5.0, 6.0])
        B = b2.finalize()

        C = A.dot(B)
        # Manually: A = [[1,0,2],[0,3,0]], B = [[4,0],[0,5],[6,0]]
        # C = [[1*4+2*6, 0], [0, 3*5]] = [[16, 0], [0, 15]]
        np.testing.assert_array_equal(C.toarray(), [[16, 0], [0, 15]])


# ---------------------------------------------------------------------------
# GPU tests — skipped on machines without CUDA
# ---------------------------------------------------------------------------


@gpu_only
class TestBackendGPU:
    @pytest.fixture
    def bk_gpu(self) -> Backend:
        return Backend("cuda")

    def test_nonzero_via_coo(self, bk_gpu: Backend) -> None:
        # Build a small cupy CSR directly
        data = cp.array([1.0, 2.0, 3.0])
        row = cp.array([0, 1, 2])
        col = cp.array([2, 1, 0])
        A = cpx_sparse.csr_matrix((data, (row, col)), shape=(3, 3))
        r, c = bk_gpu.nonzero(A)
        # Row indices may come back in sorted order depending on cupy's COO
        # internals — check as a set of pairs.
        pairs = set(zip(cp.asnumpy(r).tolist(), cp.asnumpy(c).tolist()))
        assert pairs == {(0, 2), (1, 1), (2, 0)}

    def test_sparse_from_coo_on_gpu(self, bk_gpu: Backend) -> None:
        A = bk_gpu.sparse_from_coo([1.0, 2.0], [0, 1], [1, 0], shape=(2, 2))
        assert cpx_sparse.issparse(A)
        dense = cp.asnumpy(A.toarray())
        np.testing.assert_array_equal(dense, [[0, 1], [2, 0]])

    def test_svds_strips_solver(self, bk_gpu: Backend) -> None:
        """On GPU, solver= should be silently dropped, not passed to cupy."""
        M = cp.asarray(np.diag([5.0, 3.0, 1.0]).astype(np.float64))
        A = cpx_sparse.csr_matrix(M)
        # If solver/v0 were NOT stripped, cupy would raise TypeError.
        _, s, _ = bk_gpu.svds(A, k=2, solver="arpack", v0=np.ones(3))
        s_host = cp.asnumpy(s)
        np.testing.assert_allclose(sorted(s_host), [3.0, 5.0], rtol=1e-5)

    def test_linear_operator_dispatch(self, bk_gpu: Backend) -> None:
        lo = bk_gpu.LinearOperator(
            shape=(3, 3),
            matvec=lambda x: 2 * x,
            rmatvec=lambda x: 2 * x,
            dtype=np.float64,
        )
        # Should be a cupy LinearOperator
        from cupyx.scipy.sparse.linalg import LinearOperator as CupyLO

        assert isinstance(lo, CupyLO)

    def test_free_pool(self, bk_gpu: Backend) -> None:
        # Allocate something, free, check no error
        _ = cp.zeros(1000)
        bk_gpu.free_pool()


@gpu_only
class TestDataMovementGPU:
    @pytest.fixture
    def bk_gpu(self) -> Backend:
        return Backend("cuda")

    def test_dense_roundtrip(self, bk_gpu: Backend) -> None:
        x_host = np.arange(6.0).reshape(2, 3)
        x_dev = bk_gpu.to_device(x_host)
        assert isinstance(x_dev, cp.ndarray)
        x_back = bk_gpu.to_host(x_dev)
        assert isinstance(x_back, np.ndarray)
        np.testing.assert_array_equal(x_back, x_host)

    def test_sparse_csr_roundtrip(self, bk_gpu: Backend) -> None:
        A_host = sp.random(5, 5, density=0.3, format="csr", random_state=0)
        A_dev = bk_gpu.to_device(A_host)
        assert cpx_sparse.issparse(A_dev)
        assert A_dev.format == "csr"
        A_back = bk_gpu.to_host(A_dev)
        assert sp.issparse(A_back)
        np.testing.assert_allclose(A_back.toarray(), A_host.toarray())

    def test_sparse_csc_roundtrip(self, bk_gpu: Backend) -> None:
        A_host = sp.random(4, 4, density=0.4, format="csc", random_state=1)
        A_dev = bk_gpu.to_device(A_host)
        assert A_dev.format == "csc"
        A_back = bk_gpu.to_host(A_dev)
        np.testing.assert_allclose(A_back.toarray(), A_host.toarray())

    def test_sparse_coo_roundtrip(self, bk_gpu: Backend) -> None:
        A_host = sp.random(4, 4, density=0.4, format="coo", random_state=2)
        A_dev = bk_gpu.to_device(A_host)
        assert A_dev.format == "coo"
        A_back = bk_gpu.to_host(A_dev)
        np.testing.assert_allclose(A_back.toarray(), A_host.toarray())

    def test_lil_converted_to_csr(self, bk_gpu: Backend) -> None:
        """LIL (no GPU equivalent) should be silently routed through CSR."""
        A_host = sp.lil_matrix((3, 3))
        A_host[0, 1] = 5.0
        A_host[2, 2] = 7.0
        A_dev = bk_gpu.to_device(A_host)
        assert A_dev.format == "csr"
        np.testing.assert_allclose(
            cp.asnumpy(A_dev.toarray()), A_host.toarray()
        )

    def test_to_device_idempotent(self, bk_gpu: Backend) -> None:
        """Calling to_device on already-device data should be a no-op."""
        x_dev = cp.arange(5.0)
        assert bk_gpu.to_device(x_dev) is x_dev

    def test_to_host_on_host_is_noop(self, bk_gpu: Backend) -> None:
        x = np.arange(5.0)
        assert bk_gpu.to_host(x) is x

    def test_asfortran_converts(self, bk_gpu: Backend) -> None:
        x = cp.zeros((3, 4), order="C")
        assert x.flags.c_contiguous
        out = bk_gpu.asfortran_if_gpu(x)
        assert out.flags.f_contiguous

    def test_asfortran_noop_if_already_f(self, bk_gpu: Backend) -> None:
        x = cp.zeros((3, 4), order="F")
        assert bk_gpu.asfortran_if_gpu(x) is x


@gpu_only
class TestCOOBuilderGPU:
    @pytest.fixture
    def bk_gpu(self) -> Backend:
        return Backend("cuda")

    def test_result_on_device(self, bk_gpu: Backend) -> None:
        b = COOBuilder(bk_gpu, shape=(3, 3))
        b.add(0, 1, 5.0)
        b.add_batch([1, 2], [2, 0], [3.0, 7.0])
        A = b.finalize()
        assert cpx_sparse.issparse(A)
        dense = cp.asnumpy(A.toarray())
        np.testing.assert_array_equal(
            dense, [[0, 5, 0], [0, 0, 3], [7, 0, 0]]
        )

    def test_cpu_gpu_builders_agree(self, bk_gpu: Backend) -> None:
        """Same triplets → same dense matrix regardless of backend."""
        bk_cpu = Backend("cpu")
        rng = np.random.default_rng(42)
        n = 10
        rows = rng.integers(0, n, size=30)
        cols = rng.integers(0, n, size=30)
        vals = rng.standard_normal(30)

        b_cpu = COOBuilder(bk_cpu, shape=(n, n))
        b_cpu.add_batch(rows, cols, vals)
        A_cpu = b_cpu.finalize().toarray()

        b_gpu = COOBuilder(bk_gpu, shape=(n, n))
        b_gpu.add_batch(rows, cols, vals)
        A_gpu = cp.asnumpy(b_gpu.finalize().toarray())

        np.testing.assert_allclose(A_gpu, A_cpu, rtol=1e-12)

    def test_accepts_device_arrays_in_batch(self, bk_gpu: Backend) -> None:
        """add_batch should accept cupy arrays (brought to host internally)."""
        b = COOBuilder(bk_gpu, shape=(3, 3))
        b.add_batch(cp.array([0, 1]), cp.array([1, 2]), cp.array([4.0, 6.0]))
        A = b.finalize()
        dense = cp.asnumpy(A.toarray())
        np.testing.assert_array_equal(dense, [[0, 4, 0], [0, 0, 6], [0, 0, 0]])
