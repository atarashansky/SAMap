"""Tiny end-to-end SAMAP fixture for analysis-module unit tests.

Builds two synthetic species (~250 cells × ~300 genes each) with:
  - 3 planted cell-type clusters per species, 1:1 correspondence by design
  - a homology graph that's near-identity on the marker genes plus a layer
    of weaker "paralog" edges so the homology-delta / module code has
    non-trivial structure to chew on

Runtime: ≲15s on a laptop. Safe for unit tests; intentionally NOT
representative of real data — structure is *planted* so assertions can be
exact-ish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from samap import SAMAP
from samap.sam import SAM


def _make_species(
    code: str,
    n_per_cluster: int = 80,
    n_marker_per_cluster: int = 40,
    n_noise_genes: int = 180,
    seed: int = 0,
) -> tuple[SAM, dict]:
    """One synthetic species with 3 clusters and disjoint marker-gene blocks."""
    rng = np.random.default_rng(seed)
    n_clusters = 3
    n_cells = n_per_cluster * n_clusters
    n_markers = n_marker_per_cluster * n_clusters
    n_genes = n_markers + n_noise_genes

    # Base poisson background
    X = rng.poisson(lam=0.3, size=(n_cells, n_genes)).astype(np.float32)
    labels = np.repeat([f"type_{i}" for i in range(n_clusters)], n_per_cluster)

    # Per-cluster marker blocks: cluster i lights up genes
    # [i*n_marker_per_cluster : (i+1)*n_marker_per_cluster]
    for i in range(n_clusters):
        rows = slice(i * n_per_cluster, (i + 1) * n_per_cluster)
        cols = slice(i * n_marker_per_cluster, (i + 1) * n_marker_per_cluster)
        X[rows, cols] += rng.poisson(lam=6, size=(n_per_cluster, n_marker_per_cluster))

    var = pd.DataFrame(index=[f"{code}_g{i:04d}" for i in range(n_genes)])
    obs = pd.DataFrame({"cell_type": labels}, index=[f"{code}_c{i:04d}" for i in range(n_cells)])
    adata = AnnData(X=sp.csr_matrix(X), obs=obs, var=var)

    sam = SAM(counts=adata)
    sam.preprocess_data()
    sam.run()
    return sam, {
        "n_marker_per_cluster": n_marker_per_cluster,
        "n_clusters": n_clusters,
        "n_genes": n_genes,
    }


def _make_homology(
    g_a: list[str], g_b: list[str], n_marker_per_cluster: int, n_clusters: int, seed: int = 0
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, np.ndarray]]:
    """Build a 2-species homology graph.

    - identity edges on all marker genes (strong, score 0.9)
    - "paralog" edges: marker_i ↔ marker_{i+1 in same cluster} (weak, 0.4)
    - a few cross-cluster decoy edges (0.3)
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = len(g_a), len(g_b)
    n_markers = n_marker_per_cluster * n_clusters
    rows, cols, data = [], [], []

    def add(i, j, s):
        rows.extend([i, n_a + j])
        cols.extend([n_a + j, i])
        data.extend([s, s])

    # identity on markers
    for i in range(n_markers):
        add(i, i, 0.9)
    # paralog ring within each cluster's marker block
    for c in range(n_clusters):
        base = c * n_marker_per_cluster
        for k in range(n_marker_per_cluster):
            j = base + (k + 1) % n_marker_per_cluster
            add(base + k, j, 0.4)
    # cross-cluster decoys (these should be killed by reweighting)
    for _ in range(30):
        i = int(rng.integers(0, n_markers))
        j = int(rng.integers(0, n_markers))
        if i // n_marker_per_cluster != j // n_marker_per_cluster:
            add(i, j, 0.3)
    # weak hits on a few noise genes
    for i in range(n_markers, min(n_a, n_b), 5):
        add(i, i, 0.2)

    gns = np.asarray(list(g_a) + list(g_b))
    G = sp.csr_matrix((data, (rows, cols)), shape=(len(gns), len(gns)), dtype=np.float64)
    G.sum_duplicates()
    gns_dict = {"aa": np.asarray(g_a), "bb": np.asarray(g_b)}
    return G, gns, gns_dict


def build_tiny_samap(seed: int = 0, run: bool = True) -> SAMAP:
    """Construct + (optionally) run a tiny 2-species SAMAP."""
    sa, meta = _make_species("aa", seed=seed)
    sb, _ = _make_species("bb", seed=seed + 1)
    G, gns, gns_dict = _make_homology(
        list(sa.adata.var_names),
        list(sb.adata.var_names),
        n_marker_per_cluster=meta["n_marker_per_cluster"],
        n_clusters=meta["n_clusters"],
        seed=seed,
    )
    sm = SAMAP(
        {"aa": sa, "bb": sb},
        gnnm=(G, gns, gns_dict),
        keys={"aa": "cell_type", "bb": "cell_type"},
    )
    if run:
        sm.run(n_iterations=3, umap=False, ncpus=2)
    return sm
