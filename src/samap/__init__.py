"""SAMap: Self-Assembling Manifold Mapping for cross-species single-cell analysis.

SAMap is an algorithm for mapping single-cell datasets across species using
manifold alignment and gene homology information.

Example
-------
>>> from samap import SAMAP
>>> sm = SAMAP(
...     sams={'hu': 'human.h5ad', 'ms': 'mouse.h5ad'},
...     f_maps='maps/',
... )
>>> sm.run()

For more information, see the documentation at:
https://github.com/atarashansky/SAMap
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sc-samap")
except PackageNotFoundError:
    # Package not installed (e.g. running from source checkout)
    __version__ = "2.0.2"

# Core imports
# Analysis imports
from samap.analysis import (
    GOEA,
    CellTypeTriangles,
    FunctionalEnrichment,
    GenePairFinder,
    GeneTriangles,
    ParalogSubstitutions,
    cluster_to_k,
    convert_eggnog_to_homologs,
    find_cluster_markers,
    find_paralog_substitutions,
    gene_modules,
    get_mapping_scores,
    homology_graph_delta,
    mapping_degeneracy,
    module_factored_scores,
    permutation_null_scores,
    sankey_plot,
)
from samap.core.mapping import SAMAP
from samap.io import load_samap, save_samap

# Utilities
from samap.utils import (
    df_to_dict,
    prepend_var_prefix,
    sparse_knn,
    substr,
    to_vn,
    to_vo,
)

__all__ = [
    # Analysis
    "GOEA",
    # Core
    "SAMAP",
    "CellTypeTriangles",
    "FunctionalEnrichment",
    "GenePairFinder",
    "GeneTriangles",
    "ParalogSubstitutions",
    # Version
    "__version__",
    "cluster_to_k",
    "convert_eggnog_to_homologs",
    "df_to_dict",
    "find_cluster_markers",
    "find_paralog_substitutions",
    "gene_modules",
    "get_mapping_scores",
    "homology_graph_delta",
    "load_samap",
    "mapping_degeneracy",
    "module_factored_scores",
    "permutation_null_scores",
    # Utilities
    "prepend_var_prefix",
    "sankey_plot",
    # I/O
    "save_samap",
    "sparse_knn",
    "substr",
    "to_vn",
    "to_vo",
]
