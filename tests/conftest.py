"""Shared pytest fixtures for SAMap tests."""

from __future__ import annotations

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


@pytest.fixture
def sample_species_ids() -> list[str]:
    """Sample species identifiers."""
    return ["hu", "ms"]
