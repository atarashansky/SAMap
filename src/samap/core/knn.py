"""Cross-species k-nearest-neighbour dispatch: CPU HNSW vs GPU brute-force.

Why GPU brute-force instead of a GPU approximate index
------------------------------------------------------
SAMap's ``_united_proj`` rebuilds its kNN index **every iteration** because
the joint-embedding ``wpca`` changes on each pass. HNSW graph construction is
O(n log n · M) with M≈48 — all of which is discarded after one query batch.
For n in the hundreds of thousands and d≈600, a single GPU GEMM (N_q × d
times d × N_d) followed by a per-row top-k is faster than building a CPU
HNSW graph and querying it once. ``GpuIndexFlatIP`` is also **exact**, so
there is no recall trade-off.

TODO: at >1M points the O(N_q · N_d) brute-force memory for the distance
matrix starts to hurt. At that scale switch to ``GpuIndexIVFFlat`` (coarse
quantiser + short inverted lists), trading a small amount of recall for a
linear-in-N footprint. Not implemented here — current SAMap datasets top out
well below that threshold.

Both FAISS and its GPU extensions are **optional** dependencies. If FAISS is
absent or is a CPU-only build, the GPU path gracefully falls back to hnswlib
with a warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hnswlib
import numpy as np

from samap._logging import logger
from samap.core._backend import Backend

if TYPE_CHECKING:
    from numpy.typing import NDArray

# --- Optional faiss import --------------------------------------------------
# faiss may be absent, or present as a CPU-only build (no StandardGpuResources).
# We detect both conditions at module load and gate the GPU path on them.

try:
    import faiss as _faiss

    HAS_FAISS: bool = True
    _FAISS_GPU: bool = hasattr(_faiss, "StandardGpuResources")
except ImportError:
    _faiss = None  # type: ignore[assignment]
    HAS_FAISS = False
    _FAISS_GPU = False


__all__ = ["HAS_FAISS", "approximate_knn"]


def approximate_knn(
    queries: Any,
    database: Any,
    k: int,
    metric: str = "cosine",
    bk: Backend | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Cross-species approximate k-nearest-neighbours.

    Dispatches to FAISS-GPU brute-force on a CUDA backend (when faiss-gpu is
    available) and falls back to hnswlib otherwise. Returns results in
    hnswlib's convention: ``(indices, distances)`` where distances are
    ``1 - cos(q, d)`` for the cosine metric.

    Parameters
    ----------
    queries : array-like, shape (n_q, d)
        Query vectors. May be numpy or cupy.
    database : array-like, shape (n_d, d)
        Database (index) vectors. May be numpy or cupy.
    k : int
        Neighbours to return per query.
    metric : str
        Distance metric. Only ``'cosine'`` is supported on the GPU path;
        hnswlib additionally supports ``'l2'`` and ``'ip'``.
    bk : Backend or None
        Backend instance. ``None`` → a fresh CPU backend.

    Returns
    -------
    indices : int ndarray, shape (n_q, k)
        Row indices into ``database``.
    distances : float ndarray, shape (n_q, k)
        Distances — for cosine, ``1 - cos(q, d)`` in ``[0, 2]``.
        Always returned on host (numpy).
    """
    if bk is None:
        bk = Backend("cpu")

    if bk.gpu and _FAISS_GPU:
        return _faiss_gpu_knn(queries, database, k, metric, bk)

    if bk.gpu and not _FAISS_GPU:
        logger.warning(
            "GPU backend requested but faiss-gpu is not available; "
            "falling back to CPU hnswlib for kNN."
        )

    return _hnswlib_knn(queries, database, k, metric)


# ---------------------------------------------------------------------------
# CPU path — HNSW via hnswlib
# ---------------------------------------------------------------------------

# Default HNSW parameters — kept identical to the legacy inline implementation
# in projection._united_proj so the golden regression test is bit-stable.
_HNSW_EF: int = 200
_HNSW_M: int = 48


def _hnswlib_build(
    database: Any,
    metric: str = "cosine",
    *,
    ef: int = _HNSW_EF,
    M: int = _HNSW_M,
    num_threads: int = -1,
) -> Any:
    """Build (and return) an HNSW index over ``database`` without querying.

    P0.3: lets callers reuse one index across many query batches when the
    database is shared. Returns the populated ``hnswlib.Index``.
    """
    db = np.ascontiguousarray(np.asarray(database, dtype=np.float32))
    n_d, dim = db.shape
    labels = np.arange(n_d)
    index = hnswlib.Index(space=metric, dim=dim)
    index.init_index(max_elements=n_d, ef_construction=ef, M=M)
    index.add_items(db, labels, num_threads=num_threads)
    index.set_ef(ef)
    return index


def _hnswlib_query(
    index: Any,
    queries: Any,
    k: int,
    *,
    num_threads: int = -1,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Query a prebuilt HNSW index. Returns ``(indices, distances)``."""
    q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    return index.knn_query(q, k=k, num_threads=num_threads)


def _hnswlib_knn(
    queries: Any,
    database: Any,
    k: int,
    metric: str = "cosine",
    *,
    ef: int = _HNSW_EF,
    M: int = _HNSW_M,
    num_threads: int = -1,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """CPU approximate kNN via hnswlib.

    Builds a fresh HNSW index over ``database`` and queries it. The index is
    discarded on return — callers that need the same database across many
    queries should use hnswlib directly.

    ``num_threads`` controls parallelism for both index construction and
    querying (``-1`` → all cores). The golden regression test monkeypatches
    the ``hnswlib`` module reference in this file to force single-threaded
    deterministic behaviour; keep the top-level ``import hnswlib`` intact
    for that patch to work.
    """
    # hnswlib requires host numpy arrays (any float dtype).
    q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    db = np.ascontiguousarray(np.asarray(database, dtype=np.float32))

    n_d, dim = db.shape
    labels = np.arange(n_d)

    index = hnswlib.Index(space=metric, dim=dim)
    index.init_index(max_elements=n_d, ef_construction=ef, M=M)
    index.add_items(db, labels, num_threads=num_threads)
    index.set_ef(ef)

    idx, dist = index.knn_query(q, k=k, num_threads=num_threads)
    return idx, dist


# ---------------------------------------------------------------------------
# GPU path — FAISS GpuIndexFlatIP (exact brute-force)
# ---------------------------------------------------------------------------


def _faiss_gpu_knn(
    queries: Any,
    database: Any,
    k: int,
    metric: str,
    bk: Backend,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """GPU exact kNN via FAISS ``GpuIndexFlatIP``.

    For cosine similarity we L2-normalise both ``queries`` and ``database``
    so that their inner product equals ``cos(q, d)``. FAISS returns the
    top-k by *descending* IP; we convert to cosine *distance* (``1 - ip``) to
    match hnswlib's output convention.

    FAISS requires ``float32`` C-contiguous inputs. Both numpy and cupy
    arrays are accepted — FAISS reads cupy arrays directly via the CUDA
    array interface (zero-copy when the layout already matches).
    """
    if metric != "cosine":
        raise ValueError(
            f"_faiss_gpu_knn only supports metric='cosine', got {metric!r}. "
            "Use the CPU (hnswlib) path for other metrics."
        )

    res = bk.faiss_gpu_resources()
    if res is None:
        # Should not happen — caller checks _FAISS_GPU — but be defensive.
        logger.warning("faiss_gpu_resources() returned None; falling back to hnswlib.")
        return _hnswlib_knn(queries, database, k, metric)

    # FAISS insists on float32, C-contiguous. We upload to device first so
    # normalisation runs on GPU, then hand device arrays to FAISS.
    xp = bk.xp
    q_dev = xp.ascontiguousarray(bk.to_device(queries), dtype=xp.float32)
    db_dev = xp.ascontiguousarray(bk.to_device(database), dtype=xp.float32)

    # L2-normalise rows in-place (safe: we own these copies)
    q_norm = xp.linalg.norm(q_dev, axis=1, keepdims=True)
    q_norm = xp.where(q_norm == 0, 1.0, q_norm)
    q_dev /= q_norm

    db_norm = xp.linalg.norm(db_dev, axis=1, keepdims=True)
    db_norm = xp.where(db_norm == 0, 1.0, db_norm)
    db_dev /= db_norm

    dim = int(db_dev.shape[1])

    # Flat inner-product index — exact search, no training needed.
    cfg = _faiss.GpuIndexFlatConfig()
    cfg.device = 0  # TODO: multi-GPU device selection
    index = _faiss.GpuIndexFlatIP(res, dim, cfg)
    index.add(db_dev)

    sims, idx = index.search(q_dev, k)
    # sims is inner-product == cos(q,d) since inputs are unit-norm.
    # Convert to cosine distance; bring to host for downstream CSR assembly
    # which runs on CPU regardless of backend.
    dists_host = 1.0 - bk.to_host(sims)
    idx_host = bk.to_host(idx)
    return idx_host, dists_host
