"""Optional rapids-singlecell dispatch for UMAP and Leiden.

rapids-singlecell (rsc) provides GPU-accelerated implementations of the
scanpy tools suite. When both a CUDA backend is active *and* rsc is
installed, we dispatch to it — otherwise fall back to CPU scanpy.

This module imports cleanly on machines without rsc: ``HAS_RSC`` is False
and all wrappers take the CPU path. No GPU dependency is imposed.

Known upstream issues handled here:

* rsc's Leiden occasionally returns a degenerate clustering (one cluster
  per cell, or a single cluster) at certain resolution values — a known
  cugraph edge case. When detected we fall back to CPU scanpy, which is
  slower but always well-behaved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from samap._logging import get_logger

if TYPE_CHECKING:
    from anndata import AnnData

    from samap.core._backend import Backend

logger = get_logger("samap.rsc")

# --- Optional import --------------------------------------------------------

try:
    import rapids_singlecell as rsc

    HAS_RSC: bool = True
except ImportError:
    rsc = None  # type: ignore[assignment]
    HAS_RSC = False


# --- Dispatch wrappers ------------------------------------------------------


def umap(adata: AnnData, bk: Backend, **kwargs: Any) -> None:
    """Compute UMAP embedding — rsc on GPU, scanpy on CPU.

    Both implementations write to ``adata.obsm['X_umap']`` in place.

    Parameters
    ----------
    adata
        Annotated data with neighbors already computed.
    bk
        Active backend. rsc is used only when ``bk.gpu and HAS_RSC``.
    **kwargs
        Forwarded to ``rsc.tl.umap`` or ``scanpy.tl.umap`` (same signature).
    """
    if bk.gpu and HAS_RSC:
        rsc.tl.umap(adata, **kwargs)
    else:
        import scanpy as sc

        sc.tl.umap(adata, **kwargs)


def leiden(adata: AnnData, bk: Backend, key_added: str = "leiden", **kwargs: Any) -> None:
    """Compute Leiden clustering — rsc on GPU with fallback, scanpy on CPU.

    rapids-singlecell's cugraph-backed Leiden has a known failure mode
    where it collapses to one cluster or explodes to one-cluster-per-cell
    at certain resolution values. When we detect that, we warn and re-run
    on CPU. A successful clustering is defined heuristically as producing
    between 2 and n_cells/2 clusters.

    Parameters
    ----------
    adata
        Annotated data with neighbors already computed.
    bk
        Active backend. rsc is attempted only when ``bk.gpu and HAS_RSC``.
    key_added
        Key under which the cluster assignments are stored in ``adata.obs``.
        Defaults to ``"leiden"`` (matching scanpy's default).
    **kwargs
        Forwarded to ``rsc.tl.leiden`` or ``scanpy.tl.leiden``.
    """
    if bk.gpu and HAS_RSC:
        rsc.tl.leiden(adata, key_added=key_added, **kwargs)
        n_clusters = adata.obs[key_added].nunique()
        n_cells = adata.shape[0]
        # Degenerate: everything in one bucket, or everything in its own bucket.
        # A reasonable clustering of single-cell data sits well inside this band.
        if n_clusters < 2 or n_clusters > n_cells // 2:
            logger.warning(
                "rsc leiden returned %d clusters for %d cells (degenerate); "
                "falling back to CPU scanpy.",
                n_clusters,
                n_cells,
            )
            import scanpy as sc

            sc.tl.leiden(adata, key_added=key_added, **kwargs)
    else:
        import scanpy as sc

        sc.tl.leiden(adata, key_added=key_added, **kwargs)
