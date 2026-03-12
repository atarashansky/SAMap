"""PCA implementations for the vendored SAM algorithm.

Vendored from samalg.utilities. Provides:
- `_pca_with_sparse`: Implicit-centering sparse PCA via LinearOperator + svds.
  Avoids densifying a sparse matrix for mean-subtraction.
- `weighted_PCA`: Dense PCA with optional eigenvalue weighting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sklearn.utils.sparsefuncs as sf
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import svd_flip

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _pca_with_sparse(
    X: sp.spmatrix,
    npcs: int,
    solver: str = "arpack",
    mu: NDArray[np.floating[Any]] | None = None,
    seed: int = 0,
    mu_axis: int = 0,
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
        SVD solver to use. Default is 'arpack'.
    mu : NDArray | None, optional
        Pre-computed mean. If None, computed from X.
    seed : int, optional
        Random seed. Default is 0.
    mu_axis : int, optional
        Axis along which mean was computed. Default is 0.

    Returns
    -------
    dict
        Dictionary with keys 'X_pca', 'variance', 'variance_ratio', 'components'.
    """
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
