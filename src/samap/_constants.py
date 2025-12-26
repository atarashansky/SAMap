"""Constants used throughout SAMap."""

from __future__ import annotations

# BLAST graph filtering
DEFAULT_EVAL_THRESHOLD: float = 1e-6
DEFAULT_FILTER_THRESHOLD: float = 0.25

# UMAP parameters
UMAP_MIN_DIST: float = 0.1
UMAP_SIZE_THRESHOLD: int = 10000
UMAP_MAXITER_LARGE: int = 200
UMAP_MAXITER_SMALL: int = 500

# Mapping parameters
DEFAULT_NUM_ITERATIONS: int = 3
DEFAULT_CROSS_K: int = 20
DEFAULT_NEIGHBORHOOD_SIZE: int = 3
DEFAULT_LEIDEN_RESOLUTION: float = 3.0

# Correlation parameters
DEFAULT_CORRELATION_THRESHOLD: float = 0.0
DEFAULT_ALIGNMENT_THRESHOLD: float = 0.1
DEFAULT_EXPRESSION_THRESHOLD: float = 0.05

# SAM preprocessing parameters
DEFAULT_THRESH_LOW: float = 0.0
DEFAULT_THRESH_HIGH: float = 0.96
DEFAULT_MIN_EXPRESSION: int = 1
DEFAULT_N_PCS: int = 100
DEFAULT_K_NEIGHBORS: int = 20
DEFAULT_N_GENES: int = 3000

# Marker gene parameters
DEFAULT_WEIGHT_THRESHOLD: float = 0.2
DEFAULT_PVAL_THRESHOLD: float = 1e-2
DEFAULT_N_TOP_GENES: int = 1000

# KOG category descriptions
KOG_TABLE: dict[str, str] = {
    "A": "RNA processing and modification",
    "B": "Chromatin structure and dynamics",
    "C": "Energy production and conversion",
    "D": "Cell cycle control, cell division, chromosome partitioning",
    "E": "Amino acid transport and metabolism",
    "F": "Nucleotide transport and metabolism",
    "G": "Carbohydrate transport and metabolism",
    "H": "Coenzyme transport and metabolism",
    "I": "Lipid transport and metabolism",
    "J": "Translation, ribosomal structure and biogenesis",
    "K": "Transcription",
    "L": "Replication, recombination, and repair",
    "M": "Cell wall membrane/envelope biogenesis",
    "N": "Cell motility",
    "O": "Post-translational modification, protein turnover, chaperones",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis, transport and catabolism",
    "R": "General function prediction only",
    "S": "Function unknown",
    "T": "Signal transduction mechanisms",
    "U": "Intracellular trafficking, secretion, and vesicular transport",
    "V": "Defense mechanisms",
    "W": "Extracellular structures",
    "Y": "Nuclear structure",
    "Z": "Cytoskeleton",
}
