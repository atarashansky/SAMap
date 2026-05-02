"""Analysis functions for SAMap."""

from __future__ import annotations

from samap.analysis.degeneracy import cluster_to_k, mapping_degeneracy
from samap.analysis.enrichment import GOEA, FunctionalEnrichment
from samap.analysis.gene_pairs import GenePairFinder, find_cluster_markers
from samap.analysis.homology_delta import find_paralog_substitutions, homology_graph_delta
from samap.analysis.modules import gene_modules, module_factored_scores
from samap.analysis.null import permutation_null_scores
from samap.analysis.plotting import sankey_plot
from samap.analysis.scores import (
    CellTypeTriangles,
    GeneTriangles,
    ParalogSubstitutions,
    convert_eggnog_to_homologs,
    get_mapping_scores,
)

__all__ = [
    "GOEA",
    "CellTypeTriangles",
    "FunctionalEnrichment",
    "GenePairFinder",
    "GeneTriangles",
    "ParalogSubstitutions",
    "cluster_to_k",
    "convert_eggnog_to_homologs",
    "find_cluster_markers",
    "find_paralog_substitutions",
    "gene_modules",
    "get_mapping_scores",
    "homology_graph_delta",
    "mapping_degeneracy",
    "module_factored_scores",
    "permutation_null_scores",
    "sankey_plot",
]
