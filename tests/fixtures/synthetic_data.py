"""Synthetic data generators for SAMap tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from numpy.typing import NDArray


def generate_synthetic_expression_matrix(
    n_cells: int = 100,
    n_genes: int = 500,
    sparsity: float = 0.8,
    seed: int = 42,
) -> sp.csr_matrix:
    """Generate synthetic single-cell expression matrix.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    sparsity : float
        Fraction of zeros in the matrix.
    seed : int
        Random seed.

    Returns
    -------
    scipy.sparse.csr_matrix
        Synthetic expression matrix.
    """
    rng = np.random.default_rng(seed)
    data = rng.poisson(lam=2, size=(n_cells, n_genes)).astype(float)
    mask = rng.random((n_cells, n_genes)) < sparsity
    data[mask] = 0
    return sp.csr_matrix(data)


def generate_synthetic_gene_names(
    n_genes: int = 500,
    species_prefix: str = "hu",
) -> NDArray[np.str_]:
    """Generate synthetic gene names.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    species_prefix : str
        Species prefix to add.

    Returns
    -------
    ndarray
        Array of gene names.
    """
    return np.array([f"{species_prefix}_GENE{i:04d}" for i in range(n_genes)])


def generate_synthetic_cell_names(
    n_cells: int = 100,
    prefix: str = "cell",
) -> NDArray[np.str_]:
    """Generate synthetic cell names.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    prefix : str
        Cell name prefix.

    Returns
    -------
    ndarray
        Array of cell names.
    """
    return np.array([f"{prefix}_{i:04d}" for i in range(n_cells)])


def generate_synthetic_homology_graph(
    gene_names_1: NDArray[np.str_],
    gene_names_2: NDArray[np.str_],
    n_homologs: int = 100,
    seed: int = 42,
) -> tuple[sp.csr_matrix, NDArray[np.str_]]:
    """Generate synthetic gene homology graph.

    Parameters
    ----------
    gene_names_1 : ndarray
        Gene names from species 1.
    gene_names_2 : ndarray
        Gene names from species 2.
    n_homologs : int
        Number of homologous gene pairs.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (homology_matrix, combined_gene_names)
    """
    rng = np.random.default_rng(seed)

    n_genes_1 = len(gene_names_1)
    n_genes_2 = len(gene_names_2)
    n_total = n_genes_1 + n_genes_2

    # Create combined gene names
    all_genes = np.concatenate([gene_names_1, gene_names_2])

    # Create sparse homology matrix
    gnnm = sp.lil_matrix((n_total, n_total))

    # Add random homolog connections
    n_homologs = min(n_homologs, min(n_genes_1, n_genes_2))
    idx1 = rng.choice(n_genes_1, n_homologs, replace=False)
    idx2 = rng.choice(n_genes_2, n_homologs, replace=False) + n_genes_1

    for i, j in zip(idx1, idx2):
        score = rng.random() * 0.5 + 0.5  # Score between 0.5 and 1.0
        gnnm[i, j] = score
        gnnm[j, i] = score

    return gnnm.tocsr(), all_genes
