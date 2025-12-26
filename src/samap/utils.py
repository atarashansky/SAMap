"""Utility functions for SAMap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from samap._logging import logger

if TYPE_CHECKING:
    from typing import Any

    import scipy.sparse as sp
    from numpy.typing import NDArray
    from samalg import SAM


def save_samap(sm: Any, fn: str) -> None:
    """Save a SAMAP object to a pickle file.

    Parameters
    ----------
    sm : SAMAP
        The SAMAP object to save.
    fn : str
        File path. If it doesn't end with '.pkl', the extension is added.
    """
    import dill

    if not fn.endswith(".pkl"):
        fn = fn + ".pkl"
    sm.path_to_file = fn

    # Clean up GUI and UMAP objects that can't be pickled
    _cleanup_for_serialization(sm)

    with open(fn, "wb") as f:
        dill.dump(sm, f)
    logger.info("Saved SAMAP object to %s", fn)


def _cleanup_for_serialization(sm: Any) -> None:
    """Remove unpicklable attributes from a SAMAP object.

    Parameters
    ----------
    sm : SAMAP
        The SAMAP object to clean up.
    """
    # Clean up GUI references
    for attr in ("sam1", "sam2", "samap"):
        if hasattr(sm, attr):
            sam_obj = getattr(sm, attr)
            if hasattr(sam_obj, "SamGui"):
                delattr(sam_obj, "SamGui")
            if hasattr(sam_obj, "umap_obj"):
                delattr(sam_obj, "umap_obj")

    if hasattr(sm, "SamapGui"):
        delattr(sm, "SamapGui")


def load_samap(fn: str) -> Any:
    """Load a SAMAP object from a pickle file.

    Parameters
    ----------
    fn : str
        File path. If it doesn't end with '.pkl', the extension is added.

    Returns
    -------
    SAMAP
        The loaded SAMAP object.
    """
    import dill

    if not fn.endswith(".pkl"):
        fn = fn + ".pkl"
    with open(fn, "rb") as f:
        sm = dill.load(f)
    logger.info("Loaded SAMAP object from %s", fn)
    return sm


def prepend_var_prefix(s: SAM, pre: str) -> None:
    """Add species prefix to variable names in a SAM object.

    Parameters
    ----------
    s : SAM
        SAM object to modify.
    pre : str
        Prefix to add (e.g., species ID).
    """
    x = [str(name).split("_")[0] for name in s.adata.var_names]
    vn = []
    for i, g in enumerate(s.adata.var_names):
        if x[i] != pre:
            vn.append(pre + "_" + g)
        else:
            vn.append(g)
    s.adata.var_names = pd.Index(vn)
    s.adata_raw.var_names = pd.Index(vn)


def df_to_dict(
    df: pd.DataFrame,
    key_key: str | None = None,
    val_key: list[str] | None = None,
) -> dict[str, NDArray[Any]]:
    """Convert a DataFrame to a dictionary mapping keys to value arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    key_key : str, optional
        Column to use as keys. If None, uses the index.
    val_key : list of str, optional
        Columns to use as values. If empty, uses all columns.

    Returns
    -------
    dict
        Dictionary mapping keys to arrays of values.
    """
    if val_key is None:
        val_key = []

    if key_key is None:
        index = list(df.index)
    else:
        index = list(df[key_key].values)

    if len(val_key) == 0:
        val_key = list(df.columns)

    a = []
    b = []
    for key in val_key:
        if key != key_key:
            a.extend(index)
            b.extend(list(df[key].values))
    a_arr = np.array(a)
    b_arr = np.array(b)

    idx = np.argsort(a_arr)
    a_arr = a_arr[idx]
    b_arr = b_arr[idx]
    bounds = np.where(a_arr[:-1] != a_arr[1:])[0] + 1
    bounds = np.append(np.append(0, bounds), a_arr.size)
    bounds_left = bounds[:-1]
    bounds_right = bounds[1:]
    slists = [b_arr[bounds_left[i] : bounds_right[i]] for i in range(bounds_left.size)]
    return dict(zip(np.unique(a_arr), slists))


def to_vn(op: NDArray[Any]) -> NDArray[np.str_]:
    """Convert a 2D array of pairs to semicolon-separated strings.

    Parameters
    ----------
    op : ndarray
        Nx2 array of pairs.

    Returns
    -------
    ndarray
        1D array of strings in format "a;b".
    """
    return np.array(list(op[:, 0].astype("object") + ";" + op[:, 1].astype("object")))


def to_vo(op: NDArray[Any]) -> NDArray[Any]:
    """Convert semicolon-separated strings to a 2D array of pairs.

    Parameters
    ----------
    op : ndarray
        1D array of strings in format "a;b".

    Returns
    -------
    ndarray
        Nx2 array of pairs.
    """
    import samalg.utilities as ut

    return np.vstack((ut.extract_annotation(op, None, ";"))).T


def substr(
    x: NDArray[Any] | list[str],
    s: str = "_",
    ix: int | None = None,
    obj: bool = False,
) -> NDArray[Any] | list[NDArray[Any]]:
    """Split strings by a separator and optionally extract specific indices.

    Parameters
    ----------
    x : array-like
        Array of strings to split.
    s : str, optional
        Separator, by default "_".
    ix : int, optional
        Index to extract. If None, returns all parts as separate arrays.
    obj : bool, optional
        If True, return as object dtype.

    Returns
    -------
    ndarray or list of ndarray
        Extracted parts or list of all parts.
    """
    m = []
    if ix is not None:
        for i in range(len(x)):
            f = x[i].split(s)
            safe_ix = min(len(f) - 1, ix)
            m.append(f[safe_ix])
        return np.array(m).astype("object") if obj else np.array(m)
    else:
        ms = []
        ls = []
        for i in range(len(x)):
            f = x[i].split(s)
            m = []
            for idx in range(len(f)):
                m.append(f[idx])
            ms.append(m)
            ls.append(len(m))
        ml = max(ls)
        for i in range(len(ms)):
            ms[i].extend([""] * (ml - len(ms[i])))
            if ml - len(ms[i]) > 0:
                ms[i] = np.concatenate(ms[i])
        result = np.vstack(ms)
        if obj:
            result = result.astype("object")
        return [result[:, i] for i in range(result.shape[1])]


def sparse_knn(D: sp.coo_matrix, k: int) -> sp.coo_matrix:
    """Keep only the top-k values per row in a sparse matrix.

    Parameters
    ----------
    D : scipy.sparse matrix
        Input sparse matrix.
    k : int
        Number of neighbors to keep per row.

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse matrix with only top-k values per row.
    """
    D1 = D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:] = D1.row[idr]
    D1.col[:] = D1.col[idr]
    D1.data[:] = D1.data[idr]

    _, ind = np.unique(D1.row, return_index=True)
    ind = np.append(ind, D1.data.size)
    for i in range(ind.size - 1):
        idx = np.argsort(D1.data[ind[i] : ind[i + 1]])
        if idx.size > k:
            idx = idx[:-k]
            D1.data[np.arange(ind[i], ind[i + 1])[idx]] = 0
    D1.eliminate_zeros()
    return D1
