"""k-NN graph construction for the vendored SAM algorithm.

Vendored from samalg.utilities with `gen_sparse_knn` rewritten for
direct CSR construction (no lil_matrix scatter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from samap.core._backend import Backend


def gen_sparse_knn(
    knni: NDArray[np.int64],
    knnd: NDArray[np.floating[Any]],
    shape: tuple[int, int] | None = None,
) -> sparse.csr_matrix:
    """Generate sparse k-NN matrix from indices and distances.

    Direct CSR construction via COO. Replaces the original lil_matrix +
    fancy-index scatter, which was O(n*k) Python-loop overhead inside
    scipy's lil assignment path.

    Parameters
    ----------
    knni : NDArray
        k-NN indices (n x k).
    knnd : NDArray
        k-NN distances (n x k).
    shape : tuple | None, optional
        Output shape. If None, uses (n, n).

    Returns
    -------
    sparse.csr_matrix
        Sparse k-NN matrix.
    """
    n, k = knni.shape
    if shape is None:
        shape = (n, n)
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = knni.ravel().astype(np.int32, copy=False)
    data = knnd.ravel()
    # COO -> CSR handles duplicate (row, col) pairs by summing, and sorts
    # column indices within each row automatically.
    return sparse.csr_matrix((data, (rows, cols)), shape=shape)


def nearest_neighbors_hnsw(
    x: NDArray[np.floating[Any]],
    ef: int = 200,
    M: int = 48,
    n_neighbors: int = 100,
) -> tuple[NDArray[np.int64], NDArray[np.floating[Any]]]:
    """Compute approximate nearest neighbors using HNSW algorithm.

    Parameters
    ----------
    x : NDArray
        Input data matrix.
    ef : int, optional
        HNSW ef parameter (search quality). Default is 200.
    M : int, optional
        HNSW M parameter (graph connectivity). Default is 48.
    n_neighbors : int, optional
        Number of neighbors. Default is 100.

    Returns
    -------
    tuple
        (indices, distances) arrays of shape (n, k).
    """
    import hnswlib

    labels = np.arange(x.shape[0])
    p = hnswlib.Index(space="cosine", dim=x.shape[1])
    p.init_index(max_elements=x.shape[0], ef_construction=ef, M=M)
    p.add_items(x, labels)
    p.set_ef(ef)
    idx, dist = p.knn_query(x, k=n_neighbors)
    return idx, dist


def _nearest_neighbors_umap(
    X: NDArray[np.floating[Any]],
    n_neighbors: int = 15,
    metric: str = "correlation",
    random_state: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.floating[Any]]]:
    """Fallback k-NN via UMAP's nearest_neighbors (pynndescent)."""
    from umap.umap_ import nearest_neighbors

    rs = np.random.RandomState(random_state)
    return nearest_neighbors(X, n_neighbors, metric, {}, True, rs)[:2]


def calc_nnm(
    g_weighted: NDArray[np.floating[Any]],
    k: int,
    distance: str | None = None,
    bk: Backend | None = None,
) -> sparse.csr_matrix:
    """Calculate k-nearest neighbor matrix.

    Parameters
    ----------
    g_weighted : NDArray
        Input coordinates (typically PCA-reduced).
    k : int
        Number of neighbors.
    distance : str | None, optional
        Distance metric. If 'cosine', dispatches to
        :func:`samap.core.knn.approximate_knn` (FAISS-GPU on a CUDA
        backend, hnswlib otherwise). For other metrics falls back to
        UMAP's pynndescent — CPU only.
    bk : Backend, optional
        GPU/CPU dispatch for the cosine path. Ignored for non-cosine
        metrics (FAISS-GPU is cosine-only).

    Returns
    -------
    sparse.csr_matrix
        Sparse k-NN matrix with distances as values.
    """
    if distance == "cosine":
        # approximate_knn dispatches FAISS-GPU ↔ hnswlib. SAM's kNN is
        # symmetric (self-query) so queries == database.
        from samap.core.knn import approximate_knn

        nnm, dists = approximate_knn(g_weighted, g_weighted, k, metric="cosine", bk=bk)
    else:
        nnm, dists = _nearest_neighbors_umap(g_weighted, n_neighbors=k, metric=distance)
    return gen_sparse_knn(nnm, dists)
