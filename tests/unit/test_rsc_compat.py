"""Tests for the rapids-singlecell optional-dispatch layer.

Only the CPU fallback path is testable here (rsc not installed in CI).
The GPU path is a thin passthrough to rsc — trusted upstream.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from samap._rsc_compat import HAS_RSC, leiden, umap
from samap.core._backend import Backend


def _tiny_adata_with_neighbors() -> AnnData:
    """Minimal AnnData with pre-set neighbors — enough for sc.tl.umap/leiden."""
    rng = np.random.default_rng(0)
    n = 40
    # 2D blobs so UMAP/Leiden have something to find
    X = np.vstack([
        rng.normal([0, 0], 0.1, (n // 2, 2)),
        rng.normal([5, 5], 0.1, (n // 2, 2)),
    ]).astype(np.float32)
    adata = AnnData(X)
    # Fake a connectivity graph (scanpy's neighbors output)
    from sklearn.neighbors import kneighbors_graph

    conn = kneighbors_graph(X, 5, mode="connectivity", include_self=False)
    adata.obsp["connectivities"] = sp.csr_matrix(conn)
    adata.obsp["distances"] = kneighbors_graph(X, 5, mode="distance")
    adata.uns["neighbors"] = {"params": {"n_neighbors": 5, "method": "umap"}}
    return adata


class TestRscCompatModule:
    def test_has_rsc_false_in_test_env(self):
        """rsc is not installed in the CI env — dispatch takes CPU path."""
        assert HAS_RSC is False

    def test_imports_cleanly_without_rsc(self):
        """Module must import without rsc present (optional dependency)."""
        # Re-import to prove the import itself doesn't require rsc
        import importlib

        import samap._rsc_compat as mod

        importlib.reload(mod)
        assert mod.HAS_RSC is False


class TestCPUFallback:
    """With a CPU backend and no rsc, both wrappers should call scanpy."""

    def test_umap_cpu_path(self):
        adata = _tiny_adata_with_neighbors()
        bk = Backend("cpu")
        # Should not raise; writes X_umap
        umap(adata, bk)
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (40, 2)

    def test_leiden_cpu_path(self):
        adata = _tiny_adata_with_neighbors()
        bk = Backend("cpu")
        leiden(adata, bk, resolution=0.5)
        assert "leiden" in adata.obs
        # Two well-separated blobs → should find ≥2 clusters
        assert adata.obs["leiden"].nunique() >= 2

    def test_leiden_respects_key_added(self):
        adata = _tiny_adata_with_neighbors()
        bk = Backend("cpu")
        leiden(adata, bk, key_added="my_clusters", resolution=0.5)
        assert "my_clusters" in adata.obs
        assert "leiden" not in adata.obs  # default key not used
