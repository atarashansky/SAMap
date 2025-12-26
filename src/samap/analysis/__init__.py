"""Analysis functions for SAMap."""

from __future__ import annotations

from samap.analysis.enrichment import GOEA, FunctionalEnrichment
from samap.analysis.gene_pairs import GenePairFinder, find_cluster_markers
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
    "convert_eggnog_to_homologs",
    "find_cluster_markers",
    "get_mapping_scores",
    "sankey_plot",
]
