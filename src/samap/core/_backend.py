"""Backend dispatch layer for CPU (numpy/scipy) ↔ GPU (cupy/cupyx).

This module provides a thin abstraction so SAMap's hot path can run on either
CPU or GPU with a single code path. The ``Backend`` class exposes the active
array and sparse namespaces (``xp``, ``sp``) and provides compatibility shims
for operations where the scipy and cupy APIs diverge.

cupy is an **optional** dependency. If it is not installed, importing this
module still succeeds; ``Backend("cpu")`` and ``Backend("auto")`` work, and
``Backend("cuda")`` raises a clear error.

Known scipy ↔ cupy divergences handled here:

* No ``.nonzero()`` on cupy sparse matrices — use ``nonzero()`` shim.
* No LIL format on cupy — use :class:`COOBuilder` instead of ``lil_matrix``.
* ``svds`` on cupy accepts no ``solver=`` / ``v0=`` / ``random_state=`` kwargs
  (cupy uses a thick-restart Lanczos on the normal equations; scipy uses
  ARPACK). The shim filters unsupported kwargs on GPU.
* cuSPARSE SpMM internally forces a Fortran-order copy of the dense RHS; use
  :meth:`Backend.asfortran_if_gpu` to pre-convert and avoid an implicit copy
  at matmul time.
* ``sum_duplicates()`` on cupy sorts indices in a different order than scipy.
  This usually does not affect numerical results but can break exact-bytes
  golden-output comparisons — compare with a tolerance instead.
"""

from __future__ import annotations

import warnings
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_spla

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# --- Optional cupy import ---------------------------------------------------
# cupy may not be installed (e.g. on macOS dev machines with no CUDA support).
# We try to import it at module load and set flags accordingly. All cupy-using
# code paths are guarded on these flags so the module imports cleanly either
# way.

_cupy: ModuleType | None
_cupyx_sparse: ModuleType | None
_cupyx_spla: ModuleType | None

try:
    import cupy as _cupy
    import cupyx.scipy.sparse as _cupyx_sparse
    import cupyx.scipy.sparse.linalg as _cupyx_spla

    HAS_CUPY: bool = True
except ImportError:
    _cupy = None
    _cupyx_sparse = None
    _cupyx_spla = None
    HAS_CUPY = False


def _cuda_available() -> bool:
    """Return True iff cupy is importable *and* a CUDA device is present."""
    if not HAS_CUPY:
        return False
    try:
        return bool(_cupy.is_available())  # type: ignore[union-attr]
    except Exception:
        # cupy can raise from is_available() if the CUDA driver is present but
        # incompatible. Treat any failure as "no GPU".
        return False


# Kwargs that scipy.sparse.linalg.svds accepts but cupyx does not.
# cupy's svds signature: (a, k=6, *, ncv, tol, which, maxiter,
#                         return_singular_vectors)
_CUPY_SVDS_UNSUPPORTED = frozenset({"solver", "v0", "random_state", "rng", "options"})


__all__ = ["HAS_CUPY", "Backend", "COOBuilder"]


class Backend:
    """Dispatch between numpy/scipy and cupy/cupyx.

    Parameters
    ----------
    device
        ``"cpu"`` forces the numpy/scipy backend. ``"cuda"`` forces the
        cupy/cupyx backend (raises :class:`RuntimeError` if cupy is not
        installed or no CUDA device is visible). ``"auto"`` (default) picks
        cuda when available, else cpu.

    Attributes
    ----------
    xp
        Array namespace — :mod:`numpy` or :mod:`cupy`.
    sp
        Sparse namespace — :mod:`scipy.sparse` or :mod:`cupyx.scipy.sparse`.
    spla
        Sparse linear-algebra namespace — :mod:`scipy.sparse.linalg` or
        :mod:`cupyx.scipy.sparse.linalg`.
    gpu
        ``True`` if the cuda backend is active.
    device
        The resolved device string, ``"cpu"`` or ``"cuda"``.
    """

    __slots__ = ("device", "gpu", "sp", "spla", "xp")

    def __init__(self, device: Literal["cpu", "cuda", "auto"] = "auto") -> None:
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"

        if device == "cuda":
            if not HAS_CUPY:
                raise RuntimeError(
                    "Backend('cuda') requested but cupy is not installed. "
                    "Install cupy (e.g. 'pip install cupy-cuda12x') or use "
                    "device='cpu'."
                )
            if not _cuda_available():
                raise RuntimeError(
                    "Backend('cuda') requested but no CUDA device is "
                    "available. Check your GPU drivers or use device='cpu'."
                )
            self.xp = _cupy
            self.sp = _cupyx_sparse
            self.spla = _cupyx_spla
            self.gpu = True
        elif device == "cpu":
            self.xp = np
            self.sp = scipy_sparse
            self.spla = scipy_spla
            self.gpu = False
        else:
            raise ValueError(
                f"device must be 'cpu', 'cuda', or 'auto'; got {device!r}"
            )

        self.device: str = device

    def __repr__(self) -> str:
        return f"Backend(device={self.device!r}, gpu={self.gpu})"

    # -----------------------------------------------------------------------
    # Compat shims for scipy/cupy API differences
    # -----------------------------------------------------------------------

    def nonzero(self, A: Any) -> tuple[ArrayLike, ArrayLike]:
        """Return (row, col) indices of stored entries.

        cupy sparse matrices lack a ``.nonzero()`` method. On GPU this goes
        through COO format; on CPU it calls the native method. In both cases
        the result includes *explicit zeros* (i.e. this is structural
        nonzero, matching scipy's behaviour on sparse matrices).
        """
        if self.gpu and self.sp.issparse(A):
            coo = A.tocoo()
            return coo.row, coo.col
        return A.nonzero()

    def sparse_from_coo(
        self,
        data: ArrayLike,
        row: ArrayLike,
        col: ArrayLike,
        shape: tuple[int, int],
        fmt: str = "csr",
    ) -> Any:
        """Build a sparse matrix from COO triplets on the active backend.

        Duplicate (row, col) entries are summed. Note that cupy's
        ``sum_duplicates`` sorts indices differently from scipy's, which
        should not affect numerical results but may break byte-exact
        comparisons.
        """
        data = self.xp.asarray(data)
        row = self.xp.asarray(row)
        col = self.xp.asarray(col)
        coo = self.sp.coo_matrix((data, (row, col)), shape=shape)
        return coo.asformat(fmt)

    def setdiag(self, A: Any, val: Any) -> Any:
        """Set the main diagonal of ``A`` to ``val``, in CSR, no LIL round-trip.

        scipy's CSR ``setdiag`` works but emits a ``SparseEfficiencyWarning``
        when it has to change the sparsity structure. We suppress that
        warning and call ``eliminate_zeros()`` when zeroing the diagonal so
        the structural nnz shrinks.

        cupy CSR supports ``setdiag`` directly as well; the same path
        handles both backends.

        Returns
        -------
        The input matrix ``A`` (modified in place), converted to CSR if it
        was not already.
        """
        if A.format != "csr":
            A = A.tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy_sparse.SparseEfficiencyWarning)
            A.setdiag(val)
        if np.isscalar(val) and val == 0:
            A.eliminate_zeros()
        return A

    def svds(self, A: Any, k: int, **kwargs: Any) -> Any:
        """Compute ``k`` largest singular values/vectors.

        On GPU, strips kwargs that cupy does not support (``solver``, ``v0``,
        ``random_state``, ``rng``, ``options``). cupy's implementation uses a
        thick-restart Lanczos on the normal equations (AᴴA or AAᴴ), not the
        Golub-Kahan bidiagonalisation that scipy's ARPACK path uses. This
        squares the condition number — fine for top-k singular values, less
        accurate for the smallest.
        """
        if self.gpu:
            kwargs = {k_: v for k_, v in kwargs.items() if k_ not in _CUPY_SVDS_UNSUPPORTED}
        return self.spla.svds(A, k=k, **kwargs)

    def LinearOperator(
        self,
        shape: tuple[int, int],
        matvec: Any,
        rmatvec: Any = None,
        matmat: Any = None,
        rmatmat: Any = None,
        dtype: Any = None,
    ) -> Any:
        """Dispatch to the backend's ``LinearOperator`` constructor.

        Both scipy and cupy share the same constructor signature, so this is
        a straight pass-through to the active ``spla`` namespace.
        """
        return self.spla.LinearOperator(
            shape=shape,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=dtype,
        )

    # -----------------------------------------------------------------------
    # Data movement
    # -----------------------------------------------------------------------

    def to_device(self, A: Any) -> Any:
        """Move array or sparse matrix to the active backend.

        * On a CPU backend this is a no-op (returns ``A`` unchanged).
        * On a GPU backend, uploads numpy/scipy data to cupy/cupyx. Objects
          already on-device are returned unchanged.

        Handles dense ndarrays and scipy CSR/CSC/COO sparse matrices.
        """
        if not self.gpu:
            return A

        # Already on device?
        if isinstance(A, _cupy.ndarray) or _cupyx_sparse.issparse(A):
            return A

        if scipy_sparse.issparse(A):
            # cupy sparse constructors accept scipy sparse matrices directly
            # and copy to device.
            fmt = A.format
            if fmt == "csr":
                return _cupyx_sparse.csr_matrix(A)
            if fmt == "csc":
                return _cupyx_sparse.csc_matrix(A)
            if fmt == "coo":
                return _cupyx_sparse.coo_matrix(A)
            # Unsupported format on GPU (lil, dok, bsr, dia handled via csr)
            return _cupyx_sparse.csr_matrix(A.tocsr())

        # Dense array → cupy
        return _cupy.asarray(A)

    def to_host(self, A: Any) -> Any:
        """Move array or sparse matrix back to numpy/scipy (host memory).

        If ``A`` is already host-resident (numpy/scipy), returns it unchanged.
        If ``A`` is a cupy array or cupyx sparse matrix, calls ``.get()`` to
        transfer to host.
        """
        if HAS_CUPY and isinstance(A, _cupy.ndarray):
            return A.get()
        if HAS_CUPY and _cupyx_sparse.issparse(A):
            return A.get()
        return A

    def asfortran_if_gpu(self, A: Any) -> Any:
        """Pre-convert a dense array to Fortran (column-major) order on GPU.

        cuSPARSE's SpMM (sparse-times-dense) path requires the dense RHS to be
        Fortran-ordered and will silently make a copy if it is not. When the
        same dense block is re-used across several SpMM calls (e.g. inside a
        Lanczos loop), pre-converting once avoids repeated implicit copies.

        On CPU this is a no-op (scipy's sparse dot handles C-order fine).
        """
        if not self.gpu:
            return A
        if isinstance(A, _cupy.ndarray) and not A.flags.f_contiguous:
            return _cupy.asfortranarray(A)
        return A

    def free_pool(self) -> None:
        """Release unused blocks from cupy's memory pools.

        cupy caches GPU allocations in a memory pool for reuse. After a large
        transient allocation, calling this reclaims device memory. No-op on
        CPU.
        """
        if not self.gpu:
            return
        _cupy.get_default_memory_pool().free_all_blocks()
        _cupy.get_default_pinned_memory_pool().free_all_blocks()


class COOBuilder:
    """Accumulate (row, col, val) triplets on the host, finalise to CSR/CSC.

    This replaces the ``lil_matrix`` + fancy-index-assignment pattern, which
    does not work on GPU because cupy has no LIL format. Triplets are buffered
    in Python lists (host side, O(1) append), then concatenated and converted
    to the target format in one shot at :meth:`finalize` time.

    Parameters
    ----------
    bk
        Backend that determines the target namespace for :meth:`finalize`.
    shape
        Shape of the output matrix.
    dtype
        Data dtype for values. Defaults to ``float64``.

    Examples
    --------
    >>> bk = Backend("cpu")
    >>> b = COOBuilder(bk, shape=(3, 3))
    >>> b.add(0, 1, 5.0)
    >>> b.add_batch([1, 2], [2, 0], [3.0, 7.0])
    >>> A = b.finalize(fmt="csr")
    >>> A.toarray()
    array([[0., 5., 0.],
           [0., 0., 3.],
           [7., 0., 0.]])
    """

    __slots__ = ("_bk", "_cols", "_dtype", "_rows", "_shape", "_vals")

    def __init__(
        self, bk: Backend, shape: tuple[int, int], dtype: Any = None
    ) -> None:
        self._bk = bk
        self._shape = shape
        self._dtype = np.float64 if dtype is None else np.dtype(dtype)
        # Buffer as lists of numpy arrays — cheap to append, single concat at end.
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []

    def add(self, i: int, j: int, v: Any) -> None:
        """Add a single (row, col, value) triplet."""
        self._rows.append(np.asarray([i], dtype=np.int64))
        self._cols.append(np.asarray([j], dtype=np.int64))
        self._vals.append(np.asarray([v], dtype=self._dtype))

    def add_batch(self, ii: ArrayLike, jj: ArrayLike, vv: ArrayLike) -> None:
        """Add arrays of (row, col, value) triplets at once.

        Inputs may be on host or device; they are brought to host for
        buffering. This keeps accumulation cheap and defers the single
        host→device transfer to :meth:`finalize`.
        """
        ii_h = self._bk.to_host(ii)
        jj_h = self._bk.to_host(jj)
        vv_h = self._bk.to_host(vv)
        self._rows.append(np.ascontiguousarray(ii_h, dtype=np.int64))
        self._cols.append(np.ascontiguousarray(jj_h, dtype=np.int64))
        self._vals.append(np.ascontiguousarray(vv_h, dtype=self._dtype))

    def finalize(self, fmt: str = "csr") -> Any:
        """Concatenate buffered triplets and build a sparse matrix.

        Duplicate (row, col) entries are summed (standard COO semantics).
        On a GPU backend the result lives on device.
        """
        if self._rows:
            row = np.concatenate(self._rows)
            col = np.concatenate(self._cols)
            val = np.concatenate(self._vals)
        else:
            row = np.empty(0, dtype=np.int64)
            col = np.empty(0, dtype=np.int64)
            val = np.empty(0, dtype=self._dtype)
        return self._bk.sparse_from_coo(val, row, col, self._shape, fmt=fmt)
