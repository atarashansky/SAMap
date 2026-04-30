"""Gene-identifier flavor detection.

SAMap's homology graph is built by joining ``adata.var_names`` against the
sequence headers in a BLAST/DIAMOND output. The single biggest onboarding
failure is a namespace mismatch between those two — Ensembl gene vs
transcript IDs, version suffixes, RefSeq vs GeneID, etc.

This module classifies a sample of identifiers by regex so downstream
helpers (:mod:`samap.io.fetch`, :mod:`samap.io.match`) can pick the right
fetch backend or header transform automatically.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = ["FLAVOR_PATTERNS", "IdFlavorReport", "detect_id_flavor"]


# Order matters: more specific patterns first. Each entry is
# (flavor, compiled_regex, human-readable description).
#
# The regexes are intentionally anchored at ^ and (where safe) $ so that
# concatenated headers like ``ENSG00000123456|SOX2`` don't false-positive —
# callers that need fuzzier matching should use :mod:`samap.io.match`.
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # --- Ensembl ----------------------------------------------------------
    # Core stable IDs: ENS{species?}{G|T|P}\d{11} optionally with .version.
    # Species infix is 3–6 upper letters (MUS, DAR, MMUR, ORLG, ...).
    (
        "ensembl_gene",
        re.compile(r"^ENS([A-Z]{3,6})?G\d{11}(\.\d+)?$"),
        "Ensembl gene stable ID",
    ),
    (
        "ensembl_tx",
        re.compile(r"^ENS([A-Z]{3,6})?T\d{11}(\.\d+)?$"),
        "Ensembl transcript stable ID",
    ),
    (
        "ensembl_protein",
        re.compile(r"^ENS([A-Z]{3,6})?P\d{11}(\.\d+)?$"),
        "Ensembl protein stable ID",
    ),
    # --- NCBI RefSeq ------------------------------------------------------
    (
        "refseq_rna",
        re.compile(r"^[NX][MR]_\d{6,}(\.\d+)?$"),
        "RefSeq mRNA / ncRNA accession",
    ),
    (
        "refseq_protein",
        re.compile(r"^[NXY]P_\d{6,}(\.\d+)?$"),
        "RefSeq protein accession",
    ),
    # --- NCBI GeneID (bare integers — keep AFTER everything else with
    # digits, and require the whole token to be numeric) ------------------
    (
        "ncbi_geneid",
        re.compile(r"^\d{3,9}$"),
        "NCBI Entrez GeneID",
    ),
    # --- UniProt ----------------------------------------------------------
    (
        "uniprot",
        re.compile(
            r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]"
            r"|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
        ),
        "UniProtKB accession",
    ),
    # --- Model-organism DBs ----------------------------------------------
    ("flybase", re.compile(r"^FBgn\d{7}$"), "FlyBase gene ID"),
    ("wormbase", re.compile(r"^WBGene\d{8}$"), "WormBase gene ID"),
    ("zfin", re.compile(r"^ZDB-GENE-\d{6}-\d+$"), "ZFIN gene ID"),
    ("mgi", re.compile(r"^MGI:\d+$"), "MGI gene ID"),
    ("hgnc", re.compile(r"^HGNC:\d+$"), "HGNC gene ID"),
    # --- WormBase parasite (Schistosoma, etc.) ---------------------------
    ("wbps", re.compile(r"^Smp_\d{6}([a-z]?(\.\d+)?)?$"), "WormBase ParaSite Smp ID"),
    # --- Heuristic: HGNC gene symbol -------------------------------------
    # Last resort for well-formed symbols (SOX2, CD3D, TP53). Requires
    # leading letter, 2–10 alnum/-, at least one upper. Very permissive,
    # so this is reported as `symbol` not `hgnc_symbol` and callers should
    # treat it as a hint only.
    (
        "symbol",
        re.compile(r"^[A-Z][A-Za-z0-9\-]{1,10}$"),
        "Gene-symbol-like token (heuristic)",
    ),
]

#: Public mapping of flavor name → human-readable description.
FLAVOR_PATTERNS: dict[str, str] = {name: desc for name, _, desc in _PATTERNS}


@dataclass(frozen=True)
class IdFlavorReport:
    """Result of :func:`detect_id_flavor`.

    Attributes
    ----------
    flavor
        Dominant identifier flavor (a key of :data:`FLAVOR_PATTERNS`, or
        ``"unknown"``).
    confidence
        Fraction of the *sampled* identifiers matching ``flavor``.
    counts
        Per-flavor match counts over the sample (includes ``"unknown"``).
    has_version
        ``True`` if at least one matched identifier carried a ``.\\d+``
        version suffix (relevant for Ensembl/RefSeq fetch backends).
    sample_size
        Number of identifiers actually inspected.
    examples
        Up to 3 example identifiers per flavor, for diagnostics.
    """

    flavor: str
    confidence: float
    counts: dict[str, int]
    has_version: bool
    sample_size: int
    examples: dict[str, list[str]] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover — convenience only
        top = ", ".join(
            f"{k}={v}" for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1])[:4]
        )
        return (
            f"IdFlavorReport(flavor={self.flavor!r}, "
            f"confidence={self.confidence:.2f}, n={self.sample_size}, "
            f"counts={{{top}}}, has_version={self.has_version})"
        )


def _classify(token: str) -> tuple[str, bool]:
    """Return (flavor, has_version_suffix) for a single identifier."""
    for name, pat, _ in _PATTERNS:
        m = pat.match(token)
        if m:
            return name, "." in token and bool(re.search(r"\.\d+$", token))
    return "unknown", False


def detect_id_flavor(
    ids: Iterable[str],
    *,
    sample: int = 200,
    min_confidence: float = 0.7,
) -> IdFlavorReport:
    """Classify the dominant identifier namespace of a collection of gene IDs.

    Parameters
    ----------
    ids
        Iterable of identifier strings — typically ``adata.var_names`` or a
        column of ``adata.var``.
    sample
        Maximum number of identifiers to inspect. The first ``sample``
        entries are used (deterministic — callers should shuffle first if
        the input is sorted by a confounding key).
    min_confidence
        If the dominant flavor's fraction is below this threshold the
        report's ``flavor`` is set to ``"mixed"`` (counts still populated).

    Returns
    -------
    IdFlavorReport
        See class docstring.

    Examples
    --------
    >>> r = detect_id_flavor(["ENSG00000139618", "ENSG00000141510.17"])
    >>> r.flavor, r.has_version
    ('ensembl_gene', True)
    """
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    has_version = False
    n = 0

    for tok in ids:
        if n >= sample:
            break
        s = str(tok)
        flav, ver = _classify(s)
        counts[flav] += 1
        has_version |= ver
        if len(examples.setdefault(flav, [])) < 3:
            examples[flav].append(s)
        n += 1

    if n == 0:
        return IdFlavorReport("unknown", 0.0, {}, False, 0, {})

    # Winner selection precedence:
    #   1. a specific flavor (anything but symbol/unknown) meeting min_confidence
    #   2. else `unknown` meeting min_confidence (consistently unrecognized →
    #      actionable: use samap.io.match_fasta)
    #   3. else `symbol` meeting min_confidence
    #   4. else `mixed`
    def _best(keys: list[str]) -> tuple[str, float] | None:
        sub = {k: counts[k] for k in keys if counts.get(k)}
        if not sub:
            return None
        k, v = max(sub.items(), key=lambda kv: kv[1])
        return k, v / n

    specific_keys = [k for k in counts if k not in ("symbol", "unknown")]
    for cand in (_best(specific_keys), _best(["unknown"]), _best(["symbol"])):
        if cand is not None and cand[1] >= min_confidence:
            flav, conf = cand
            break
    else:
        # report the overall top fraction as `confidence` even when mixed
        _, conf = max(((k, v / n) for k, v in counts.items()), key=lambda kv: kv[1])
        flav = "mixed"

    return IdFlavorReport(
        flavor=flav,
        confidence=conf,
        counts=dict(counts),
        has_version=has_version,
        sample_size=n,
        examples=examples,
    )
