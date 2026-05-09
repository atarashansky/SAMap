"""Per-species projection precompute cache.

The two dominant per-species costs in :func:`samap.core.projection._projection_precompute`
are the sparse-PCA loadings (:func:`~samap.core.projection.prepare_SAMap_loadings`,
~46% of a 2-species run's wall time) and the Gram matrix ``XᵀX`` (~19%). Both
depend only on a single SAM object, yet a 210-pair sweep over 21 species
recomputes each one 20×. This module persists those two arrays to a per-species
``{code}_precompute.npz`` so subsequent runs can load-and-slice instead of
recompute.

The cached arrays are stored over the species' **full** ``var_names`` so they
are partner-agnostic; :func:`_projection_precompute` slices them to the
per-pair homology-connected gene set on load. The cheap per-species pieces
(``ss``, ``mu_ss``, ``wpca_own``, ``M_own`` — all <2 s) are not cached; they
either depend on the per-pair gene slice (``wpca_own``) or are trivially
recomputed from in-memory state.

A content fingerprint (``n_cells``, ``n_genes``, ``X.nnz``, SAM gene-weight
bytes, ``npcs``) is stored alongside; :func:`load_precompute` returns ``None``
on mismatch so callers fall through to recomputation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler

from samap._logging import logger
from samap.core.projection import prepare_SAMap_loadings

if TYPE_CHECKING:
    from typing import Any

    from samap.sam import SAM


_FORMAT_VERSION = 1


def _species_fingerprint(sam: SAM, npcs: int) -> str:
    """Hash of the inputs that determine ``PCs_SAMap`` and ``XtX``.

    Covers shape, ``X.nnz`` and the SAM gene-weight vector. A change in any of
    these invalidates both cached arrays.
    """
    adata = sam.adata
    h = hashlib.sha1()
    h.update(np.int64(adata.shape[0]).tobytes())
    h.update(np.int64(adata.shape[1]).tobytes())
    h.update(np.int64(npcs).tobytes())
    X = adata.X
    nnz = X.nnz if sps.issparse(X) else int(np.count_nonzero(X))
    h.update(np.int64(nnz).tobytes())
    W = np.ascontiguousarray(adata.var["weights"].values, dtype=np.float64)
    h.update(W.tobytes())
    return h.hexdigest()


def _strip_prefix(names: np.ndarray, code: str) -> np.ndarray:
    pre = code + "_"
    if names.size and str(names[0]).startswith(pre):
        return np.asarray([n[len(pre) :] for n in names], dtype=object)
    return np.asarray(names, dtype=object)


def precompute_species(
    sam: SAM,
    code: str,
    out_dir: str | Path,
    npcs: int = 300,
) -> Path:
    """Compute and persist the per-species SAMap projection precompute.

    Runs :func:`prepare_SAMap_loadings` (sparse PCA, ``npcs`` components) and
    the full-gene Gram matrix ``XᵀX`` of the standardised, gene-weighted
    expression — the two expensive per-species inputs to
    :func:`samap.core.projection._projection_precompute` — and writes them to
    ``{out_dir}/{code}_precompute.npz``.

    Parameters
    ----------
    sam : SAM
        A SAM object that has been through :meth:`SAM.run` (i.e. has
        ``adata.var['weights']`` and ``adata.uns['run_args']``). May or may
        not already have ``adata.varm['PCs_SAMap']``.
    code : str
        Species code; used for the output filename and to strip any
        ``{code}_`` var-name prefix so the cache is prefix-agnostic.
    out_dir : str or Path
        Directory to write the ``.npz`` into. Created if missing.
    npcs : int, optional
        Number of PC loadings. Must match what ``SAMAP`` will use. Default 300.

    Returns
    -------
    Path
        Path to the written ``.npz``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{code}_precompute.npz"

    adata = sam.adata
    if "weights" not in adata.var:
        raise KeyError(
            f"'{code}': adata.var['weights'] missing — run SAM.run() first."
        )

    if "PCs_SAMap" not in adata.varm or adata.varm["PCs_SAMap"].shape[1] != npcs:
        logger.info("precompute_species[%s]: computing PCs_SAMap (npcs=%d)", code, npcs)
        prepare_SAMap_loadings(sam, npcs=npcs)
    PCs = np.asarray(adata.varm["PCs_SAMap"], dtype=np.float64)

    logger.info(
        "precompute_species[%s]: computing XtX over %d genes", code, adata.shape[1]
    )
    std = StandardScaler(with_mean=False)
    W = adata.var["weights"].values
    ss_full = std.fit_transform(adata.X).multiply(W[None, :]).tocsr()
    XtX = (ss_full.T @ ss_full).tocsr()

    fp = _species_fingerprint(sam, npcs)
    var_names = _strip_prefix(np.asarray(adata.var_names), code)

    np.savez(
        path,
        format_version=np.int64(_FORMAT_VERSION),
        fingerprint=np.asarray(fp),
        n_cells=np.int64(adata.shape[0]),
        n_genes=np.int64(adata.shape[1]),
        npcs=np.int64(npcs),
        var_names=var_names,
        PCs_SAMap=PCs,
        XtX_data=XtX.data,
        XtX_indices=XtX.indices.astype(np.int32, copy=False),
        XtX_indptr=XtX.indptr.astype(np.int64, copy=False),
        XtX_shape=np.asarray(XtX.shape, dtype=np.int64),
    )
    logger.info(
        "precompute_species[%s]: wrote %s (%.1f MB)",
        code,
        path,
        path.stat().st_size / 1e6,
    )
    return path


def load_precompute(
    code: str,
    cache_dir: str | Path,
    sam: SAM | None = None,
) -> dict[str, Any] | None:
    """Load a per-species precompute cache written by :func:`precompute_species`.

    Parameters
    ----------
    code : str
        Species code.
    cache_dir : str or Path
        Directory containing ``{code}_precompute.npz``.
    sam : SAM, optional
        If given, the cache fingerprint is checked against this SAM and the
        function returns ``None`` on mismatch (so the caller falls through to
        recomputation).

    Returns
    -------
    dict or None
        On hit: ``{'PCs_SAMap', 'XtX', 'var_names', 'npcs', 'n_cells',
        'n_genes', 'fingerprint', 'path'}``. ``XtX`` is a full-gene
        ``scipy.sparse.csr_matrix`` (f64) — slice it to the per-pair gene set
        before use. Returns ``None`` if the file is absent, unreadable, or
        stale.
    """
    path = Path(cache_dir) / f"{code}_precompute.npz"
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=True)
        if int(z["format_version"]) != _FORMAT_VERSION:
            logger.warning(
                "precompute cache '%s' has format_version=%s, expected %d — ignoring.",
                path,
                z["format_version"],
                _FORMAT_VERSION,
            )
            return None
        npcs = int(z["npcs"])
        fp = str(z["fingerprint"])
        if sam is not None:
            fp_now = _species_fingerprint(sam, npcs)
            if fp_now != fp:
                logger.warning(
                    "precompute cache '%s' is stale (fingerprint mismatch) — ignoring.",
                    path,
                )
                return None
        shape = tuple(int(x) for x in z["XtX_shape"])
        XtX = sps.csr_matrix(
            (z["XtX_data"], z["XtX_indices"], z["XtX_indptr"]), shape=shape
        )
        return {
            "PCs_SAMap": z["PCs_SAMap"],
            "XtX": XtX,
            "var_names": z["var_names"],
            "npcs": npcs,
            "n_cells": int(z["n_cells"]),
            "n_genes": int(z["n_genes"]),
            "fingerprint": fp,
            "path": str(path),
        }
    except Exception as e:  # noqa: BLE001 — cache miss must never raise
        logger.warning("Failed to load precompute cache '%s': %s — ignoring.", path, e)
        return None
