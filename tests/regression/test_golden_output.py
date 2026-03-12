"""Golden regression test: full 3-species SAMap pipeline.

Captures the numeric outputs of the current SAMap implementation so that
subsequent optimization/refactoring can be verified to produce identical
results (to floating-point tolerance).

Regenerate the fixture with:
    pytest tests/regression/test_golden_output.py -m slow --regenerate-golden

Run the comparison with:
    pytest tests/regression/test_golden_output.py -m slow
"""

from __future__ import annotations

import os
import random
import types
from pathlib import Path
from typing import Any

# Nudge thread counts toward determinism. These must be set before numpy/
# numba import, but since conftest.py imports numpy first there's no hard
# guarantee. The numba-parallel functions in SAMap write to distinct array
# indices (no reductions), so thread count there is benign — these env vars
# are belt-and-suspenders.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_FIXTURES = _HERE / "fixtures"
_GOLDEN = _FIXTURES / "golden_3species.npz"
_EXAMPLE_DATA = _HERE.parent.parent / "example_data"

_SEED = 42
_RTOL = 1e-4
_ATOL = 1e-6

_SPECIES = {
    "pl": "planarian.h5ad",
    "sc": "schistosome.h5ad",
    "hy": "hydra.h5ad",
}


# ---------------------------------------------------------------------------
# Deterministic hnswlib wrapper
# ---------------------------------------------------------------------------


class _DeterministicHNSWIndex:
    """Thin wrapper around hnswlib.Index that forces deterministic behaviour.

    hnswlib's multi-threaded ``add_items`` produces a non-deterministic index
    (insertion order races). We force single-threaded insertion and query, and
    pin the construction seed. The wrapper delegates everything else.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import hnswlib as _real_hnswlib

        self._index = _real_hnswlib.Index(*args, **kwargs)

    def init_index(self, *args: Any, **kwargs: Any) -> Any:
        # hnswlib defaults random_seed to 100 already; make it explicit.
        kwargs.setdefault("random_seed", _SEED)
        return self._index.init_index(*args, **kwargs)

    def add_items(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["num_threads"] = 1
        return self._index.add_items(*args, **kwargs)

    def knn_query(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["num_threads"] = 1
        return self._index.knn_query(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._index, name)


def _patch_hnswlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the hnswlib module seen by samap with a deterministic shim.

    ``samap.core.projection`` does a top-level ``import hnswlib`` — that's the
    only call site reached during the golden pipeline (samap.sam's hnswlib
    usage is gated behind ``SAM.run()``/``calculate_nnm``, which the pipeline
    does not call — input SAMs are pre-computed and loaded from h5ad).
    """
    import samap.core.projection as projection

    fake = types.SimpleNamespace(Index=_DeterministicHNSWIndex)
    monkeypatch.setattr(projection, "hnswlib", fake)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _fix_seeds() -> None:
    np.random.seed(_SEED)
    random.seed(_SEED)


def _run_pipeline(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run the full 3-species SAMap pipeline with determinism controls."""
    _patch_hnswlib(monkeypatch)
    _fix_seeds()

    from samap import SAMAP
    from samap.sam import SAM

    sams: dict[str, Any] = {}
    for sid, fname in _SPECIES.items():
        sam = SAM()
        sam.load_data(str(_EXAMPLE_DATA / fname))
        sams[sid] = sam

    sm = SAMAP(sams, f_maps=str(_EXAMPLE_DATA / "maps") + os.sep)
    # umap=False: UMAP is stochastic and we don't pin its output anyway.
    sm.run(n_iterations=3, umap=False)
    return sm


def _pack_sparse(prefix: str, mat: sp.spmatrix, out: dict[str, np.ndarray]) -> None:
    csr = sp.csr_matrix(mat)
    csr.sort_indices()
    # Drop explicit zeros so structural comparison is stable.
    csr.eliminate_zeros()
    out[f"{prefix}_data"] = np.ascontiguousarray(csr.data, dtype=np.float64)
    out[f"{prefix}_indices"] = np.ascontiguousarray(csr.indices, dtype=np.int64)
    out[f"{prefix}_indptr"] = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    out[f"{prefix}_shape"] = np.asarray(csr.shape, dtype=np.int64)


def _extract_outputs(sm: Any) -> dict[str, np.ndarray]:
    """Capture all numeric outputs we want to pin."""
    out: dict[str, np.ndarray] = {}

    # Stitched cross-species kNN graph (feeds UMAP / downstream analysis).
    _pack_sparse("conn", sm.samap.adata.obsp["connectivities"], out)

    # Refined gene-homology graph (correlation-reweighted).
    _pack_sparse("gnnm_refined", sm.gnnm_refined, out)

    # Original BLAST homology graph (reindexed post-run). Should be fully
    # deterministic irrespective of hnswlib — good sanity anchor.
    _pack_sparse("gnnm", sm.gnnm, out)

    # Per-species SAM gene weights. These come from the input h5ad (SAMap
    # does not modify them) and serve as a load-integrity check.
    for sid in _SPECIES:
        w = sm.sams[sid].adata.var["weights"].to_numpy()
        out[f"weights_{sid}"] = np.ascontiguousarray(w, dtype=np.float64)

    return out


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _compare_sparse_strict(
    prefix: str, golden: Any, actual: dict[str, np.ndarray]
) -> None:
    """Assert sparse-matrix equality: exact structure + allclose data."""
    np.testing.assert_array_equal(
        golden[f"{prefix}_shape"], actual[f"{prefix}_shape"]
    ), f"{prefix}: shape mismatch"
    np.testing.assert_array_equal(
        golden[f"{prefix}_indptr"], actual[f"{prefix}_indptr"]
    ), f"{prefix}: indptr mismatch (different nnz pattern)"
    np.testing.assert_array_equal(
        golden[f"{prefix}_indices"], actual[f"{prefix}_indices"]
    ), f"{prefix}: indices mismatch (different sparsity pattern)"
    np.testing.assert_allclose(
        golden[f"{prefix}_data"],
        actual[f"{prefix}_data"],
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{prefix}: nonzero values diverge",
    )


def _compare_sparse_as_dense(
    prefix: str, golden: Any, actual: dict[str, np.ndarray]
) -> None:
    """Fallback: compare sparse matrices as dense, elementwise allclose.

    Used if structural comparison fails due to tiny values crossing the
    zero threshold. Reconstructs both CSR matrices and compares the
    elementwise difference without materializing a huge dense array.
    """
    g_shape = tuple(golden[f"{prefix}_shape"])
    a_shape = tuple(actual[f"{prefix}_shape"])
    assert g_shape == a_shape, f"{prefix}: shape mismatch {g_shape} vs {a_shape}"

    g = sp.csr_matrix(
        (golden[f"{prefix}_data"], golden[f"{prefix}_indices"], golden[f"{prefix}_indptr"]),
        shape=g_shape,
    )
    a = sp.csr_matrix(
        (actual[f"{prefix}_data"], actual[f"{prefix}_indices"], actual[f"{prefix}_indptr"]),
        shape=a_shape,
    )
    diff = (g - a).tocsr()
    # Absolute-difference check is enough here — relative tolerance on a
    # sparse graph with many tiny edges is noisy.
    max_abs = np.abs(diff.data).max() if diff.nnz else 0.0
    assert max_abs <= max(_ATOL, _RTOL), (
        f"{prefix}: max abs elementwise difference {max_abs:.3e} exceeds "
        f"tolerance (checked {diff.nnz} differing entries)"
    )


def _report_sparse_mismatch(
    prefix: str, golden: Any, actual: dict[str, np.ndarray]
) -> str:
    """Produce a diagnostic string when structural comparison fails."""
    g_nnz = len(golden[f"{prefix}_data"])
    a_nnz = len(actual[f"{prefix}_data"])
    g_sum = float(golden[f"{prefix}_data"].sum())
    a_sum = float(actual[f"{prefix}_data"].sum())
    return (
        f"  {prefix}: nnz golden={g_nnz} actual={a_nnz} "
        f"(Δ={a_nnz - g_nnz}), sum golden={g_sum:.6g} actual={a_sum:.6g}"
    )


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_golden_3species(
    regenerate_golden: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin full 3-species SAMap pipeline output against a golden fixture.

    Runs the pipeline end-to-end on the example hydra/planarian/schistosome
    data and asserts that the stitched kNN graph, the refined gene-homology
    graph, and the per-species gene weights match the stored golden within
    rtol=1e-4.

    Determinism notes
    -----------------
    The only known source of run-to-run variation in the core algorithm is
    hnswlib's multi-threaded ``add_items`` in ``_united_proj``. We patch the
    ``hnswlib`` module reference inside ``samap.core.projection`` with a wrapper
    that forces single-threaded index construction and a fixed seed.

    If this test fails after a refactor with *structural* differences in the
    kNN graph (different nnz pattern) but numerically similar overall weight
    distribution, it likely means the refactor changed hnswlib invocation in
    a way the shim no longer covers. Either extend the shim or loosen to a
    top-k-overlap comparison.
    """
    if not _EXAMPLE_DATA.exists():
        pytest.skip("Example data not available at example_data/")

    sm = _run_pipeline(monkeypatch)
    actual = _extract_outputs(sm)

    if regenerate_golden:
        _FIXTURES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_GOLDEN, **actual)
        pytest.skip(f"Regenerated golden fixture → {_GOLDEN}")

    if not _GOLDEN.exists():
        pytest.fail(
            f"Golden fixture missing: {_GOLDEN}\n"
            "Generate it with:  pytest tests/regression/test_golden_output.py "
            "-m slow --regenerate-golden"
        )

    golden = np.load(_GOLDEN)

    # --- Dense vectors: per-species gene weights ------------------------
    for sid in _SPECIES:
        key = f"weights_{sid}"
        np.testing.assert_allclose(
            golden[key],
            actual[key],
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"Gene weights for species '{sid}' diverged from golden",
        )

    # --- Sparse graphs ---------------------------------------------------
    # gnnm (BLAST homology, reindexed) should be bit-for-bit deterministic
    # — it has no stochastic inputs. Hard structural check.
    _compare_sparse_strict("gnnm", golden, actual)

    # gnnm_refined and conn depend on hnswlib. Try strict first; if the
    # sparsity pattern shifts (e.g. a few edges crossing the zero threshold)
    # fall back to elementwise comparison.
    failures: list[str] = []
    for prefix in ("gnnm_refined", "conn"):
        try:
            _compare_sparse_strict(prefix, golden, actual)
        except AssertionError as e_strict:
            try:
                _compare_sparse_as_dense(prefix, golden, actual)
            except AssertionError as e_dense:
                failures.append(
                    f"{prefix} failed both strict and dense comparison.\n"
                    f"  strict: {e_strict}\n"
                    f"  dense:  {e_dense}\n"
                    f"{_report_sparse_mismatch(prefix, golden, actual)}"
                )

    if failures:
        pytest.fail(
            "Golden regression mismatch:\n"
            + "\n".join(failures)
            + "\n\nIf this divergence is expected (e.g. after an intentional "
            "algorithmic change), regenerate the fixture with "
            "--regenerate-golden."
        )
