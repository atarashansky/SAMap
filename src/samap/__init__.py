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

__version__ = "2.0.0"

# Core imports
from samap.core.mapping import SAMAP
from samap.io import load_samap, save_samap

# Analysis imports
from samap.analysis import (
    GOEA,
    CellTypeTriangles,
    FunctionalEnrichment,
    GenePairFinder,
    GeneTriangles,
    ParalogSubstitutions,
    convert_eggnog_to_homologs,
    find_cluster_markers,
    get_mapping_scores,
    sankey_plot,
)

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
    # Version
    "__version__",
    # Core
    "SAMAP",
    # I/O
    "save_samap",
    "load_samap",
    # Analysis
    "GOEA",
    "FunctionalEnrichment",
    "GenePairFinder",
    "find_cluster_markers",
    "get_mapping_scores",
    "sankey_plot",
    "ParalogSubstitutions",
    "convert_eggnog_to_homologs",
    "CellTypeTriangles",
    "GeneTriangles",
    # Utilities
    "prepend_var_prefix",
    "df_to_dict",
    "to_vn",
    "to_vo",
    "substr",
    "sparse_knn",
]
