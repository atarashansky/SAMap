"""Small helpers for the vendored SAM algorithm.

Vendored from samalg.utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def convert_annotations(A: NDArray[Any]) -> NDArray[np.int64]:
    """Convert categorical annotations to integer codes.

    Parameters
    ----------
    A : NDArray
        Array of categorical values.

    Returns
    -------
    NDArray
        Integer codes.
    """
    x = np.unique(A)
    y = np.zeros(A.size)
    for z, i in enumerate(x):
        y[i == A] = z
    return y.astype("int")


def extract_annotation(
    cn: NDArray[Any],
    x: int | None,
    c: str = "_",
) -> NDArray[Any] | list[NDArray[Any]]:
    """Extract annotations from cell names by splitting on delimiter.

    Parameters
    ----------
    cn : NDArray
        Array of cell names.
    x : int | None
        Index of annotation field to extract. If None, returns all fields.
    c : str, optional
        Delimiter character. Default is '_'.

    Returns
    -------
    NDArray | list
        Extracted annotations.
    """
    m = []
    if x is not None:
        for i in range(cn.size):
            f = cn[i].split(c)
            x = min(len(f) - 1, x)
            m.append(f[x])
        return np.array(m)
    else:
        ms: list[list[str]] = []
        ls = []
        for i in range(cn.size):
            f = cn[i].split(c)
            m_inner = []
            for field_x in range(len(f)):
                m_inner.append(f[field_x])
            ms.append(m_inner)
            ls.append(len(m_inner))
        ml = max(ls)
        for i in range(len(ms)):
            ms[i].extend([""] * (ml - len(ms[i])))
            if ml - len(ms[i]) > 0:
                ms[i] = list(np.concatenate(ms[i]))
        ms_arr = np.vstack(ms)
        MS = []
        for i in range(ms_arr.shape[1]):
            MS.append(ms_arr[:, i])
        return MS
