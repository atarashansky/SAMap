"""Unit tests for samap.core.knn — CPU/GPU kNN dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from samap.core._backend import HAS_CUPY, Backend
from samap.core.knn import HAS_FAISS, _hnswlib_knn, approximate_knn

if HAS_CUPY:
    import cupy as cp

    _CUDA = cp.is_available()
else:
    _CUDA = False

# The FAISS GPU path needs cupy (for device arrays), cuda, and a GPU-enabled
# faiss build. On the macOS dev machine none of these hold.
_FAISS_GPU_AVAILABLE = _CUDA and HAS_FAISS
if _FAISS_GPU_AVAILABLE:
    import faiss

    _FAISS_GPU_AVAILABLE = hasattr(faiss, "StandardGpuResources")

gpu_only = pytest.mark.skipif(not _FAISS_GPU_AVAILABLE, reason="requires cupy + CUDA + faiss-gpu")


# ---------------------------------------------------------------------------
# Reference exact cosine kNN (numpy brute-force)
# ---------------------------------------------------------------------------


def _brute_force_cosine_knn(
    queries: np.ndarray, database: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine kNN by dense matmul — reference for recall / exactness."""
    qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    dn = database / np.linalg.norm(database, axis=1, keepdims=True)
    sims = qn @ dn.T  # (n_q, n_d) cosine similarities
    # Top-k by similarity = smallest-k by distance
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    # Sort within each row so neighbours are ordered near → far
    row_sims = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-row_sims, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    dist_sorted = 1.0 - np.take_along_axis(row_sims, order, axis=1)
    return idx_sorted, dist_sorted


@pytest.fixture
def bk_cpu() -> Backend:
    return Backend("cpu")


@pytest.fixture
def small_data() -> tuple[np.ndarray, np.ndarray]:
    """~500 database points, 50 queries, 16 dims — small enough for exact ref."""
    rng = np.random.default_rng(42)
    n_db, n_q, dim = 500, 50, 16
    db = rng.standard_normal((n_db, dim)).astype(np.float32)
    q = rng.standard_normal((n_q, dim)).astype(np.float32)
    return q, db


# ---------------------------------------------------------------------------
# Output-format contracts
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_shapes(self, bk_cpu: Backend, small_data) -> None:
        q, db = small_data
        k = 5
        idx, dist = approximate_knn(q, db, k=k, metric="cosine", bk=bk_cpu)
        assert idx.shape == (q.shape[0], k)
        assert dist.shape == (q.shape[0], k)

    def test_indices_are_int(self, bk_cpu: Backend, small_data) -> None:
        q, db = small_data
        idx, _ = approximate_knn(q, db, k=5, metric="cosine", bk=bk_cpu)
        assert np.issubdtype(idx.dtype, np.integer)
        # All in bounds
        assert idx.min() >= 0
        assert idx.max() < db.shape[0]

    def test_cosine_distances_in_range(self, bk_cpu: Backend, small_data) -> None:
        q, db = small_data
        _, dist = approximate_knn(q, db, k=5, metric="cosine", bk=bk_cpu)
        # cosine distance = 1 - cos_sim, always in [0, 2]
        assert dist.min() >= 0.0 - 1e-6
        assert dist.max() <= 2.0 + 1e-6

    def test_distances_sorted_ascending(self, bk_cpu: Backend, small_data) -> None:
        q, db = small_data
        _, dist = approximate_knn(q, db, k=10, metric="cosine", bk=bk_cpu)
        # Each row should be non-decreasing (closest first)
        assert (np.diff(dist, axis=1) >= -1e-6).all()

    def test_default_bk_is_cpu(self, small_data) -> None:
        """bk=None should create a CPU backend silently."""
        q, db = small_data
        idx, _dist = approximate_knn(q, db, k=3)
        assert idx.shape == (q.shape[0], 3)


# ---------------------------------------------------------------------------
# CPU HNSW recall vs brute force
# ---------------------------------------------------------------------------


class TestHnswRecall:
    def test_recall_above_95pct(self, small_data) -> None:
        """With ef=200, M=48 and k=10 on 500 points, HNSW recall is near-perfect."""
        q, db = small_data
        k = 10
        idx_hnsw, _ = _hnswlib_knn(q, db, k=k, metric="cosine")
        idx_exact, _ = _brute_force_cosine_knn(q, db, k=k)

        # Recall@k: for each query, fraction of HNSW neighbours that appear
        # in the exact top-k.
        hits = 0
        for row_h, row_e in zip(idx_hnsw, idx_exact):
            hits += len(set(row_h.tolist()) & set(row_e.tolist()))
        recall = hits / (q.shape[0] * k)
        assert recall > 0.95, f"HNSW recall too low: {recall:.3f}"

    def test_distances_close_to_exact(self, small_data) -> None:
        """HNSW distances should match brute-force to float32 precision when
        the same neighbour is found."""
        q, db = small_data
        _idx_hnsw, dist_hnsw = _hnswlib_knn(q, db, k=1, metric="cosine")
        _, dist_exact = _brute_force_cosine_knn(q, db, k=1)

        # For queries where HNSW found the true nearest neighbour, the
        # distance should match.
        np.testing.assert_allclose(dist_hnsw, dist_exact, atol=1e-5)

    def test_deterministic_single_thread(self, small_data) -> None:
        """num_threads=1 + fixed seed gives reproducible index → reproducible
        results. (Proxy for golden-test determinism.)"""
        q, db = small_data
        idx_a, dist_a = _hnswlib_knn(q, db, k=5, num_threads=1)
        idx_b, dist_b = _hnswlib_knn(q, db, k=5, num_threads=1)
        np.testing.assert_array_equal(idx_a, idx_b)
        np.testing.assert_array_equal(dist_a, dist_b)


# ---------------------------------------------------------------------------
# GPU brute-force (FAISS) — exact
# ---------------------------------------------------------------------------


@gpu_only
class TestFaissGPU:
    @pytest.fixture
    def bk_gpu(self) -> Backend:
        return Backend("cuda")

    def test_exact_vs_brute_force(self, bk_gpu: Backend, small_data) -> None:
        """GpuIndexFlatIP is exact — neighbour sets must match brute-force."""
        q, db = small_data
        k = 10

        idx_faiss, dist_faiss = approximate_knn(q, db, k=k, metric="cosine", bk=bk_gpu)
        idx_exact, dist_exact = _brute_force_cosine_knn(q, db, k=k)

        # Neighbour sets should be identical (exact search). Use sets per row
        # to ignore tie-breaking order differences.
        for i in range(q.shape[0]):
            assert set(idx_faiss[i].tolist()) == set(idx_exact[i].tolist()), (
                f"row {i}: FAISS {idx_faiss[i]} != exact {idx_exact[i]}"
            )
        # Distances match to float32 precision.
        np.testing.assert_allclose(
            np.sort(dist_faiss, axis=1),
            np.sort(dist_exact, axis=1),
            atol=1e-5,
        )

    def test_accepts_cupy_arrays(self, bk_gpu: Backend) -> None:
        rng = np.random.default_rng(0)
        q = cp.asarray(rng.standard_normal((20, 8)).astype(np.float32))
        db = cp.asarray(rng.standard_normal((100, 8)).astype(np.float32))
        idx, dist = approximate_knn(q, db, k=5, metric="cosine", bk=bk_gpu)
        # Results are returned on host
        assert isinstance(idx, np.ndarray)
        assert isinstance(dist, np.ndarray)
        assert idx.shape == (20, 5)

    def test_resources_cached_on_backend(self, bk_gpu: Backend) -> None:
        res1 = bk_gpu.faiss_gpu_resources()
        res2 = bk_gpu.faiss_gpu_resources()
        assert res1 is res2
        assert res1 is not None

    def test_non_cosine_metric_raises(self, bk_gpu: Backend, small_data) -> None:
        """The FAISS-GPU path is cosine-only; other metrics must fail loud.

        (Dispatching via approximate_knn would work because the non-cosine
        branch never reaches _faiss_gpu_knn — this tests the internal directly.)
        """
        from samap.core.knn import _faiss_gpu_knn

        q, db = small_data
        with pytest.raises(ValueError, match="only supports metric='cosine'"):
            _faiss_gpu_knn(q, db, k=5, metric="l2", bk=bk_gpu)

    def test_faiss_matches_hnsw_distances(self, bk_gpu: Backend, small_data) -> None:
        """Sanity cross-check: FAISS and HNSW agree on nearest-neighbour distance."""
        q, db = small_data
        _, dist_gpu = approximate_knn(q, db, k=1, metric="cosine", bk=bk_gpu)
        _, dist_cpu = _hnswlib_knn(q, db, k=1, metric="cosine")
        np.testing.assert_allclose(dist_gpu, dist_cpu, atol=1e-5)


# ---------------------------------------------------------------------------
# Graceful fallback — GPU backend without faiss-gpu
# ---------------------------------------------------------------------------


class TestFallback:
    def test_cpu_backend_always_uses_hnswlib(self, bk_cpu: Backend, small_data, caplog) -> None:
        """On CPU backend, no fallback warning — hnswlib is the direct path."""
        q, db = small_data
        with caplog.at_level("WARNING"):
            approximate_knn(q, db, k=3, metric="cosine", bk=bk_cpu)
        assert "faiss" not in caplog.text.lower()

    @pytest.mark.skipif(
        not _CUDA or _FAISS_GPU_AVAILABLE,
        reason="needs CUDA but *without* faiss-gpu to test the fallback",
    )
    def test_gpu_without_faiss_warns_and_falls_back(self, small_data, caplog) -> None:
        """GPU backend + no faiss-gpu → warning + hnswlib path."""
        q, db = small_data
        bk = Backend("cuda")
        with caplog.at_level("WARNING"):
            idx, _ = approximate_knn(q, db, k=3, metric="cosine", bk=bk)
        assert "faiss" in caplog.text.lower()
        assert idx.shape == (q.shape[0], 3)
