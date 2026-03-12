"""PCA implementations for the vendored SAM algorithm.

Vendored from samalg.utilities. Provides:
- `_pca_with_sparse`: Implicit-centering sparse PCA via LinearOperator + svds.
  Avoids densifying a sparse matrix for mean-subtraction. Now dispatches to
  either ARPACK (serial Lanczos, original) or randomized SVD (block SpMM,
  GPU-friendly) via the `svd_solver` parameter.
- `randomized_svd_implicit_center`: fbpca-style randomized SVD with implicit
  mean-centering. O(k) SpMM passes instead of hundreds of serial matvecs.
- `weighted_PCA`: Dense PCA with optional eigenvalue weighting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sklearn.utils.sparsefuncs as sf
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import svd_flip

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from samap.core._backend import Backend


# ---------------------------------------------------------------------------
# Randomized SVD with implicit centering
# ---------------------------------------------------------------------------


def randomized_svd_implicit_center(
    X: Any,
    k: int,
    mu: NDArray[np.floating[Any]] | None = None,
    n_oversamples: int = 10,
    n_power: int = 4,
    seed: int = 0,
    bk: Backend | None = None,
) -> dict[str, NDArray[np.floating[Any]]]:
    r"""Top-k SVD of ``(X - 1·μ)`` via randomized range-finding, never densifying.

    Every occurrence of ``(X - 1·μ) @ M`` is computed as
    ``X @ M - 1 · (μ @ M)`` — one SpMM plus a rank-1 broadcast correction.
    Symmetrically, ``(X - 1·μ).T @ N = X.T @ N - μ.T · (1.T @ N)``.
    This replaces ARPACK's hundreds of serial matvecs with O(n_power) block
    SpMM calls, which is dramatically better for GPU utilisation
    (cuSPARSE SpMM >> repeated SpMV).

    Algorithm (Halko–Martinsson–Tropp, adapted for implicit centering)::

        Ω = random(m, k + p)                       # Gaussian sketch
        Y = (X - 1μ) @ Ω                           # range sketch
        for q power iterations:
            Y = qr(Y).Q                            # stabilise
            Z = (X - 1μ).T @ Y                     # one SpMM + correction
            Z = qr(Z).Q                            # stabilise
            Y = (X - 1μ) @ Z                       # one SpMM + correction
        Q = qr(Y).Q                                # (n, k+p) orthonormal
        B = Q.T @ (X - 1μ)                         # small: (k+p, m)
        U_b, s, Vt = svd(B)                        # dense, fits in memory
        U = Q @ U_b

    Parameters
    ----------
    X : sparse matrix
        Input (n_cells × n_genes), CSR or CSC. On GPU, a cupyx sparse matrix.
    k : int
        Number of singular components to return.
    mu : array-like, optional
        Column means (1 × n_genes or flat). If None, computed from X.
    n_oversamples : int, default 10
        Extra sketch dimensions beyond k. More improves accuracy at the cost
        of wider SpMM blocks.
    n_power : int, default 4
        Power iterations. Each adds two SpMM passes. 4 is conservative for
        SAM's conditioning (single-cell data has a long singular-value tail).
    seed : int, default 0
        Seeds the random sketch matrix for reproducibility.
    bk : Backend, optional
        CPU/GPU dispatch. If None, a CPU backend is created.

    Returns
    -------
    dict with keys:
        - ``X_pca``: (n, k) projected scores, i.e. ``(X - 1μ) @ V.T`` == ``U @ diag(s)``
        - ``components``: (k, m) right singular vectors ``Vt``
        - ``variance``: (k,) explained variances ``s**2 / (n-1)``
        - ``variance_ratio``: variances normalised by total column variance
    """
    if bk is None:
        from samap.core._backend import Backend as _Backend

        bk = _Backend("cpu")

    xp = bk.xp
    n, m = X.shape

    # --- Mean vector as a flat (m,) array on the active backend ---
    if mu is None:
        mu_flat = xp.asarray(X.mean(axis=0)).ravel()
    else:
        mu_flat = bk.to_device(np.asarray(mu).ravel())
    mu_row = mu_flat[None, :]  # (1, m) for broadcasting
    mu_col = mu_flat[:, None]  # (m, 1)

    # --- Implicit-centering SpMM helpers -------------------------------------
    # (X - 1μ) @ M  ==  X @ M - 1 · (μ @ M)
    #   The second term is a (1, r) row vector, broadcast-subtracted from
    #   every row of the (n, r) SpMM result. No (n, m) dense materialised.
    def A_matmul(M: Any) -> Any:
        M = bk.asfortran_if_gpu(M)  # cuSPARSE SpMM prefers F-order RHS
        return X @ M - mu_row @ M  # (n, r) - (1, r) broadcast

    # (X - 1μ).T @ N  ==  X.T @ N - μ.T · (1.T @ N)
    #   (1.T @ N) is just the column sums of N → (1, r). Correction is a
    #   rank-1 outer product μ.T @ col_sums.
    def At_matmul(N: Any) -> Any:
        N = bk.asfortran_if_gpu(N)
        col_sums = N.sum(axis=0, keepdims=True)  # (1, r)
        return X.T @ N - mu_col @ col_sums  # (m, r) - (m, 1)@(1, r)

    # --- Range finder --------------------------------------------------------
    k_os = k + n_oversamples
    # numpy Generator path works on both backends; cupy mirrors the API.
    rng = xp.random.default_rng(seed)
    Omega = rng.standard_normal((m, k_os)).astype(X.dtype, copy=False)

    Y = A_matmul(Omega)  # (n, k_os)

    # Power iterations with QR stabilisation on both sides. Without the
    # intermediate re-orthonormalisation the smaller singular directions wash
    # out in floating point after ~2 iterations.
    for _ in range(n_power):
        Y, _ = xp.linalg.qr(Y)
        Z = At_matmul(Y)  # (m, k_os)
        Z, _ = xp.linalg.qr(Z)
        Y = A_matmul(Z)  # (n, k_os)

    Q, _ = xp.linalg.qr(Y)  # (n, k_os), orthonormal columns

    # --- Project and small SVD ----------------------------------------------
    # B = Q.T @ (X - 1μ). This is the *transpose* of (X - 1μ).T @ Q, which
    # our At_matmul helper already computes as an SpMM.
    B = At_matmul(Q).T  # (k_os, m), dense — small by construction

    U_b, s, Vt = xp.linalg.svd(B, full_matrices=False)
    U = Q @ U_b  # (n, k_os)

    # Truncate to k, sign-fix for determinism. svd_flip is numpy-side, so
    # pull back to host if on GPU. SVD output is already sorted descending.
    U_h = bk.to_host(U[:, :k])
    s_h = bk.to_host(s[:k])
    Vt_h = bk.to_host(Vt[:k, :])
    U_h, Vt_h = svd_flip(U_h, Vt_h)

    X_pca = U_h * s_h  # (n, k) — scores == U @ diag(s)
    ev = s_h**2 / (n - 1)

    # Total variance of the *uncentered* sparse X — matches `_pca_with_sparse`
    # which uses sf.mean_variance_axis on X (not on X - μ). This is the
    # correct denominator: sum of column variances.
    X_host = bk.to_host(X)
    if sp.issparse(X_host):
        total_var = sf.mean_variance_axis(X_host, axis=0)[1].sum()
    else:
        total_var = X_host.var(axis=0, ddof=0).sum()
    ev_ratio = ev / total_var

    return {
        "X_pca": X_pca,
        "variance": ev,
        "variance_ratio": ev_ratio,
        "components": Vt_h,
    }


# ---------------------------------------------------------------------------
# ARPACK path (original) + dispatch
# ---------------------------------------------------------------------------


def _pca_with_sparse(
    X: sp.spmatrix,
    npcs: int,
    solver: str = "arpack",
    mu: NDArray[np.floating[Any]] | None = None,
    seed: int = 0,
    mu_axis: int = 0,
    svd_solver: Literal["arpack", "randomized"] = "arpack",
    bk: Backend | None = None,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Perform PCA on sparse matrices using iterative SVD with implicit centering.

    Uses a LinearOperator to represent (X - mu) without ever materializing the
    dense centered matrix. The matvec/rmatvec closures compute Xv - mu·v on the
    fly, keeping memory at O(nnz) instead of O(n*m).

    Parameters
    ----------
    X : sparse.spmatrix
        Input sparse matrix.
    npcs : int
        Number of principal components.
    solver : str, optional
        SVD solver passed to scipy's ``svds`` (ARPACK path only). Default is 'arpack'.
    mu : NDArray | None, optional
        Pre-computed mean. If None, computed from X.
    seed : int, optional
        Random seed. Default is 0.
    mu_axis : int, optional
        Axis along which mean was computed. Default is 0.
    svd_solver : {"arpack", "randomized"}, optional
        Algorithm dispatch. ``"arpack"`` (default) uses the original
        LinearOperator + iterative Lanczos path — hundreds of serial matvecs,
        tight convergence, but poor GPU utilisation. ``"randomized"`` uses
        block SpMM (see :func:`randomized_svd_implicit_center`) — O(k) passes,
        each a wide sparse-dense matmul, typically 5-10× faster and far
        better suited to GPU. Randomized requires ``mu_axis=0``.
    bk : Backend, optional
        Backend for the randomized path. Ignored for ARPACK.

    Returns
    -------
    dict
        Dictionary with keys 'X_pca', 'variance', 'variance_ratio', 'components'.
    """
    # --- Randomized solver dispatch ------------------------------------------
    # The randomized path only supports column centering (mu_axis=0) — the
    # only case SAM actually uses. For mu_axis=1, fall through to ARPACK.
    if svd_solver == "randomized":
        if mu_axis != 0:
            raise ValueError("randomized svd_solver only supports mu_axis=0 (column centering)")
        return randomized_svd_implicit_center(X, npcs, mu=mu, seed=seed, bk=bk)

    # --- ARPACK path (original) ----------------------------------------------
    random_state = check_random_state(seed)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))
    X = check_array(X, accept_sparse=["csr", "csc"])

    if mu is None:
        if mu_axis == 0:
            mu = np.asarray(X.mean(0)).flatten()[None, :]
        else:
            mu = np.asarray(X.mean(1)).flatten()[:, None]

    if mu_axis == 0:
        mdot = mu.dot
        mmat = mdot
        mhdot = mu.T.dot
        mhmat = mu.T.dot
        Xdot = X.dot
        Xmat = Xdot
        XHdot = X.T.conj().dot
        XHmat = XHdot
        ones = np.ones(X.shape[0])[None, :].dot

        def matvec(x: NDArray[Any]) -> NDArray[Any]:
            return Xdot(x) - mdot(x)

        def matmat(x: NDArray[Any]) -> NDArray[Any]:
            return Xmat(x) - mmat(x)

        def rmatvec(x: NDArray[Any]) -> NDArray[Any]:
            return XHdot(x) - mhdot(ones(x))

        def rmatmat(x: NDArray[Any]) -> NDArray[Any]:
            return XHmat(x) - mhmat(ones(x))

    else:
        mdot = mu.dot
        mmat = mdot
        mhdot = mu.T.dot
        mhmat = mu.T.dot
        Xdot = X.dot
        Xmat = Xdot
        XHdot = X.T.conj().dot
        XHmat = XHdot
        ones = np.ones(X.shape[1])[None, :].dot

        def matvec(x: NDArray[Any]) -> NDArray[Any]:
            return Xdot(x) - mdot(ones(x))

        def matmat(x: NDArray[Any]) -> NDArray[Any]:
            return Xmat(x) - mmat(ones(x))

        def rmatvec(x: NDArray[Any]) -> NDArray[Any]:
            return XHdot(x) - mhdot(x)

        def rmatmat(x: NDArray[Any]) -> NDArray[Any]:
            return XHmat(x) - mhmat(x)

    XL = spla.LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    u, s, v = spla.svds(XL, solver=solver, k=npcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    v = v[idx, :]

    X_pca = (u * s)[:, idx]
    ev = s[idx] ** 2 / (X.shape[0] - 1)

    total_var = sf.mean_variance_axis(X, axis=0)[1].sum()
    ev_ratio = ev / total_var

    return {
        "X_pca": X_pca,
        "variance": ev,
        "variance_ratio": ev_ratio,
        "components": v,
    }


def weighted_PCA(
    mat: NDArray[np.floating[Any]],
    do_weight: bool = True,
    npcs: int | None = None,
    solver: str = "auto",
    seed: int = 0,
) -> tuple[NDArray[np.floating[Any]], PCA]:
    """Perform PCA with optional eigenvalue weighting.

    Parameters
    ----------
    mat : NDArray
        Input data matrix.
    do_weight : bool, optional
        If True, weight PCs by eigenvalues. Default is True.
    npcs : int | None, optional
        Number of components. If None, uses min(mat.shape).
    solver : str, optional
        SVD solver. Default is 'auto'.
    seed : int, optional
        Random seed. Default is 0.

    Returns
    -------
    tuple
        (reduced_weighted, pca_object)
    """
    if do_weight:
        ncom = min(mat.shape) if npcs is None else min((min(mat.shape), npcs))

        pca = PCA(svd_solver=solver, n_components=ncom, random_state=check_random_state(seed))
        reduced = pca.fit_transform(mat)
        scaled_eigenvalues = pca.explained_variance_
        scaled_eigenvalues = scaled_eigenvalues / scaled_eigenvalues.max()
        reduced_weighted = reduced * scaled_eigenvalues[None, :] ** 0.5
    else:
        pca = PCA(n_components=npcs, svd_solver=solver, random_state=check_random_state(seed))
        reduced = pca.fit_transform(mat)
        if reduced.shape[1] == 1:
            pca = PCA(n_components=2, svd_solver=solver, random_state=check_random_state(seed))
            reduced = pca.fit_transform(mat)
        reduced_weighted = reduced

    return reduced_weighted, pca
