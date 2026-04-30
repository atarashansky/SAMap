"""I/O utilities for SAMap.

Submodules
----------
ids
    Identifier-namespace detection (:func:`detect_id_flavor`).
fetch
    Derive a protein FASTA from var_names (:func:`fetch_proteome`).
"""

from __future__ import annotations

from samap.io.fetch import FetchReport, fetch_proteome
from samap.io.ids import FLAVOR_PATTERNS, IdFlavorReport, detect_id_flavor
from samap.utils import load_samap, save_samap

__all__ = [
    "FLAVOR_PATTERNS",
    "FetchReport",
    "IdFlavorReport",
    "detect_id_flavor",
    "fetch_proteome",
    "load_samap",
    "save_samap",
]
