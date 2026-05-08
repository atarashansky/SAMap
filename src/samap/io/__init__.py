"""I/O utilities for SAMap.

Submodules
----------
ids
    Identifier-namespace detection (:func:`detect_id_flavor`).
fetch
    Derive a protein FASTA from var_names (:func:`fetch_proteome`).
"""

from __future__ import annotations

from samap.io.blast import load_gnnm, run_blast, save_gnnm
from samap.io.feature_graph import compose_gnnm, gnnm_from_gtf, prepare_atac_sam
from samap.io.fetch import FetchReport, fetch_proteome
from samap.io.homology import gnnm_from_pairs, homology_from_eggnog
from samap.io.ids import FLAVOR_PATTERNS, IdFlavorReport, detect_id_flavor
from samap.io.match import MatchReport, match_fasta
from samap.utils import load_samap, save_samap

__all__ = [
    "FLAVOR_PATTERNS",
    "FetchReport",
    "IdFlavorReport",
    "MatchReport",
    "compose_gnnm",
    "detect_id_flavor",
    "fetch_proteome",
    "gnnm_from_gtf",
    "gnnm_from_pairs",
    "homology_from_eggnog",
    "load_gnnm",
    "load_samap",
    "match_fasta",
    "prepare_atac_sam",
    "run_blast",
    "save_gnnm",
    "save_samap",
]
