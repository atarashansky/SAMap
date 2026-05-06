"""Shared pytest fixtures for SAMap tests."""

from __future__ import annotations

# Thread-count pinning MUST happen before numpy/numba are first imported
# anywhere in the test session. pytest loads the root conftest before any
# test module is collected, so this is the only safe place. Previously these
# lived at the top of ``tests/regression/test_golden_output.py``, which
# pytest collects *after* ``tests/integration`` (alphabetical), so numba had
# already cached NUMBA_NUM_THREADS=<ncpu> by the time the setdefault("1")
# fired, and any subsequent JIT re-read raised "Cannot set NUMBA_NUM_THREADS
# to a different value once the threads have been launched".
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_sparse_matrix(rng: np.random.Generator) -> sp.csr_matrix:
    """Create a sample sparse matrix for testing."""
    data = rng.random((100, 100))
    data[data < 0.9] = 0  # Make it sparse
    return sp.csr_matrix(data)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "gene": ["geneA", "geneA", "geneB", "geneC", "geneC", "geneC"],
            "value": [1, 2, 3, 4, 5, 6],
        },
        index=["idx1", "idx2", "idx3", "idx4", "idx5", "idx6"],
    )


@pytest.fixture
def sample_gene_pairs() -> NDArray[np.str_]:
    """Create sample gene pairs for testing."""
    return np.array(
        [
            ["hu_SOX2", "ms_Sox2"],
            ["hu_OCT4", "ms_Oct4"],
            ["hu_NANOG", "ms_Nanog"],
        ]
    )


@pytest.fixture(scope="module")
def tiny_samap():
    """Two-species synthetic SAMAP with planted 1:1 cluster structure (~15s)."""
    from tests.fixtures.tiny_samap import build_tiny_samap

    return build_tiny_samap(seed=0, run=True)


@pytest.fixture
def sample_species_ids() -> list[str]:
    """Sample species identifiers."""
    return ["hu", "ms"]
