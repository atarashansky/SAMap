"""Cross-species cell-type ontology builder from persisted SAMap connectivities.

The standard SAMap workflow holds the full ``SAMAP`` object in memory and
recomputes mapping scores from the joint manifold. For pan-metazoan corpora
(100+ species pairs) that's prohibitive — but each pair's cell–cell
connectivity matrix is small (≤100k×100k sparse) and, once persisted to
disk, can be re-scored against *any* per-species label set in ~0.2s.

This module provides:

- ``score_from_connectivities`` — recompute the symmetric mapping-score
  matrix (the same quantity ``get_mapping_scores`` returns) from a
  persisted ``obsp['connectivities']`` plus per-species label vectors,
  without instantiating SAM/SAMAP objects.
- ``persist_pair`` — write a SAMAP run's connectivity + obs metadata to
  the on-disk format ``score_from_connectivities`` consumes.
- ``build_union_graph`` — assemble the all-pairs cell-type graph (one node
  per species×cluster, weighted edges from every pairwise score matrix).
- ``cluster_families`` — Leiden community detection on the union graph,
  optionally restricted to reciprocal-best edges.

Typical usage::

    # one-time, after each pairwise SAMAP run:
    persist_pair(sm, "atlas/persisted/mmhs")

    # then, with arbitrary per-species label sets:
    labels = {sp: anndata.read_h5ad(f"{sp}.h5ad").obs["leiden_k40"] for sp in SPECIES}
    G = build_union_graph("atlas/persisted", SPECIES, labels)
    fams = cluster_families(G, rbh_only=True, resolution=2.0)
"""

from __future__ import annotations

import os
import re
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:
    from samap.core.mapping import SAMAP


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_pair(sm: SAMAP, out_prefix: str) -> dict:
    """Persist a SAMAP run's connectivity for later disk-based re-scoring.

    Writes three files at ``out_prefix``:

    - ``{out_prefix}_obsp.npz`` — ``adata.obsp['connectivities']`` (CSR).
    - ``{out_prefix}_obs.parquet`` — per-cell ``species`` and bare obs name
      (the ``species`` column is the authoritative row→species map; do
      *not* infer it from obs-name prefixes — those are not guaranteed
      to be species-prefixed).
    - ``{out_prefix}_meta.json`` — ``mapping_K``, species list, cell counts.

    Parameters
    ----------
    sm : SAMAP
        Run SAMAP object.
    out_prefix : str
        Path prefix (no extension). Directory is created if missing.

    Returns
    -------
    dict
        The metadata written to ``_meta.json``.
    """
    import json

    ad = sm.samap.adata
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    sp.save_npz(f"{out_prefix}_obsp.npz", ad.obsp["connectivities"].tocsr())
    obs = pd.DataFrame(
        {
            "obs_name": np.asarray(ad.obs_names, dtype=str),
            "species": np.asarray(ad.obs["species"], dtype=str),
        }
    )
    obs.to_parquet(f"{out_prefix}_obs.parquet")
    species = list(sm.sams.keys())
    meta = {
        "species": species,
        "n_cells": {s: int((obs.species == s).sum()) for s in species},
        "mapping_K": int(ad.uns.get("mapping_K", 20)),
        "n_obs": int(ad.n_obs),
        "obsp_nnz": int(ad.obsp["connectivities"].nnz),
    }
    with open(f"{out_prefix}_meta.json", "w") as f:
        json.dump(meta, f)
    return meta


def _load_pair(prefix: str) -> tuple[sp.csr_matrix, pd.DataFrame, dict]:
    import json

    X = sp.load_npz(f"{prefix}_obsp.npz").tocsr()
    if os.path.exists(f"{prefix}_obs.parquet"):
        obs = pd.read_parquet(f"{prefix}_obs.parquet")
    else:  # back-compat: bare obsnames CSV, no species column
        names = pd.read_csv(f"{prefix}_obsnames.csv", header=None)[0].values
        obs = pd.DataFrame({"obs_name": names, "species": None})
    meta = {}
    if os.path.exists(f"{prefix}_meta.json"):
        with open(f"{prefix}_meta.json") as f:
            meta = json.load(f)
    return X, obs, meta


# ---------------------------------------------------------------------------
# Core re-scorer
# ---------------------------------------------------------------------------


def score_from_connectivities(
    prefix: str,
    labels: dict[str, pd.Series],
    *,
    mapping_K: int | None = None,
) -> pd.DataFrame:
    """Recompute the mapping-score matrix from a persisted SAMap connectivity.

    This reproduces the per-pair block of ``get_mapping_scores`` without
    instantiating SAM/SAMAP objects: per-cell cross-species kNN mass is
    aggregated by target label, mean-pooled by source label,
    max-symmetrised across the two directions, and divided by
    ``mapping_K`` (the cross-species-k used at run time).

    .. note::
       The persisted ``connectivities`` matrix is the *directed* kNN graph
       (each row's nonzeros are that cell's outgoing neighbours). It is
       **not** symmetric — ``X[ia][:, ib]`` (a→b mass) and ``X[ib][:, ia]``
       (b→a mass) must be read separately; using the transpose of one for
       the other gives wrong results when the two species have different
       cell counts.

    Parameters
    ----------
    prefix : str
        Path prefix written by ``persist_pair`` (i.e., the string before
        ``_obsp.npz`` / ``_obs.parquet`` / ``_meta.json``).
    labels : dict[str, pandas.Series]
        ``{species_id: Series}`` where each Series is indexed by *bare*
        obs names (matching ``_obs.parquet['obs_name']``) and values are
        cluster labels. Exactly two species are expected for a pairwise
        run; for multi-species runs the score matrix is built over all
        species in ``labels``.
    mapping_K : int, optional
        Override the persisted ``mapping_K``. Defaults to the value in
        ``_meta.json`` (or 20 if missing).

    Returns
    -------
    pandas.DataFrame
        Symmetric mapping-score matrix; rows and columns are
        species-prefixed cluster labels (``"{species}_{label}"``). For
        two-species input the species-a×species-b block is what matters;
        within-species blocks are zero.
    """
    X, obs, meta = _load_pair(prefix)
    species = list(labels.keys())
    if mapping_K is None:
        mapping_K = int(meta.get("mapping_K", 20))

    # Row→species assignment.
    if obs["species"].notna().all():
        sp_of = obs["species"].values
    else:
        # No species column persisted (legacy). Fall back to obs-name
        # membership in the per-species label index. SAMAP concatenates
        # species in input order, so a positional split is safe when
        # obs-names alone are ambiguous, but only if the caller's label
        # indexes cover their species exactly.
        sp_of = np.full(len(obs), "", dtype=object)
        names = obs["obs_name"].values
        for s in species:
            hit = pd.Index(labels[s].index).get_indexer(names) >= 0
            sp_of[hit & (sp_of == "")] = s
        if (sp_of == "").any():
            # last resort: positional split by per-species label sizes
            sizes = {s: len(labels[s]) for s in species}
            if sum(sizes.values()) == len(obs):
                pos = 0
                for s in species:
                    sp_of[pos : pos + sizes[s]] = s
                    pos += sizes[s]
            else:
                raise ValueError(
                    "Cannot assign rows to species: persist_pair() should "
                    "be re-run to write the species column."
                )

    # Per-species cell index + label vector + indicator matrix.
    #
    # SAMAP concatenates per-species AnnDatas and calls
    # ``obs_names_make_unique()``, which appends ``-1``, ``-2``, … to any
    # obs-name that collides across species (common when two 10x datasets
    # share raw barcodes). The label Series users supply is indexed by the
    # *original* per-species obs-names, so a direct reindex against the
    # persisted (suffixed) names misses those cells. We therefore retry the
    # reindex with the trailing ``-N`` stripped — but only for cells that
    # missed on the direct lookup, so a barcode that legitimately ends in
    # ``-1`` is not perturbed.
    suffix_re = re.compile(r"-\d+$")
    blocks: dict[str, dict] = {}
    for s in species:
        idx = np.where(sp_of == s)[0]
        names = obs["obs_name"].values[idx]
        lab_s = labels[s].reindex(names)
        miss = lab_s.isna()
        if miss.any():
            stripped = pd.Index(names)[miss].map(lambda x: suffix_re.sub("", x))
            retry = labels[s].reindex(stripped)
            lab_s.values[np.where(miss)[0]] = retry.values
        lab = lab_s.astype(str).values
        if pd.isna(lab).any() or (lab == "nan").any():
            missing = int((lab == "nan").sum() + pd.isna(lab).sum())
            raise ValueError(
                f"{missing} cells of species {s!r} have no label — "
                "label index must cover every persisted obs_name "
                "(after stripping any obs_names_make_unique '-N' suffix)."
            )
        uniq = np.unique(lab)
        ind = sp.csr_matrix(
            (np.ones(idx.size), (np.arange(idx.size), pd.Categorical(lab, uniq).codes)),
            shape=(idx.size, uniq.size),
        )
        cnt = np.asarray(ind.sum(0)).ravel()
        cnt[cnt == 0] = 1
        blocks[s] = {"idx": idx, "uniq": uniq, "I": ind, "cnt": cnt}

    # Directed cluster→cluster mass for every ordered species pair.
    M_dir: dict[tuple[str, str], np.ndarray] = {}
    for a in species:
        for b in species:
            if a == b:
                continue
            ba, bb = blocks[a], blocks[b]
            Xab = X[ba["idx"]][:, bb["idx"]]
            M_dir[(a, b)] = (ba["I"].T @ (Xab @ bb["I"])).toarray() / ba["cnt"][:, None]

    # Assemble symmetric matrix.
    all_labels = np.concatenate([[f"{s}_{u}" for u in blocks[s]["uniq"]] for s in species])
    pos = {}
    off = 0
    for s in species:
        pos[s] = (off, off + blocks[s]["uniq"].size)
        off += blocks[s]["uniq"].size
    M = np.zeros((off, off))
    for a in species:
        for b in species:
            if a == b:
                continue
            ra, rb = pos[a], pos[b]
            M[ra[0] : ra[1], rb[0] : rb[1]] = np.maximum(M_dir[(a, b)], M_dir[(b, a)].T) / mapping_K
    return pd.DataFrame(M, index=all_labels, columns=all_labels)


# ---------------------------------------------------------------------------
# Union graph + family detection
# ---------------------------------------------------------------------------


def build_union_graph(
    persisted_dir: str,
    species: list[str],
    labels: dict[str, pd.Series],
    *,
    pair_name: callable = lambda a, b: f"{a}{b}",
    score_thr: float = 0.1,
    mapping_K: int | None = None,
) -> pd.DataFrame:
    """Assemble the all-pairs cell-type graph from persisted connectivities.

    For every unordered species pair ``(a, b)`` with a persisted
    ``{pair_name(a,b)}_obsp.npz`` (or ``{pair_name(b,a)}_obsp.npz``)
    under ``persisted_dir``, computes the score matrix at the supplied
    labels and emits one edge per ``(cluster_a, cluster_b)`` with score
    above ``score_thr``, plus a per-edge ``rbh`` flag.

    Parameters
    ----------
    persisted_dir : str
        Directory containing ``{pair}_obsp.npz`` etc.
    species : list[str]
        Species IDs in the corpus.
    labels : dict[str, pandas.Series]
        Per-species label vectors (bare-obs-name indexed).
    pair_name : callable, optional
        ``(a, b) -> str`` mapping species pair to file-prefix stem.
        Default concatenates the two IDs.
    score_thr : float, optional
        Minimum score to emit an edge. Default 0.1.
    mapping_K : int, optional
        Override the persisted per-pair mapping_K.

    Returns
    -------
    pandas.DataFrame
        Columns ``src``, ``dst`` (species-prefixed labels), ``score``,
        ``rbh`` (bool — reciprocal-best within that pair's score matrix
        at ``score_thr``).
    """
    edges = []
    for a, b in combinations(species, 2):
        prefix = None
        for stem in (pair_name(a, b), pair_name(b, a)):
            if os.path.exists(os.path.join(persisted_dir, f"{stem}_obsp.npz")):
                prefix = os.path.join(persisted_dir, stem)
                break
        if prefix is None:
            continue
        # Species order in the file may be (a,b) or (b,a) — we always pass
        # both species' labels and let score_from_connectivities sort it
        # out via the persisted species column.
        S = score_from_connectivities(prefix, {a: labels[a], b: labels[b]}, mapping_K=mapping_K)
        ar = [r for r in S.index if r.startswith(f"{a}_")]
        bc = [c for c in S.columns if c.startswith(f"{b}_")]
        Mv = S.loc[ar, bc].values
        bb_ = Mv.argmax(1)
        ba_ = Mv.argmax(0)
        rbh = np.zeros_like(Mv, dtype=bool)
        rbh[np.arange(Mv.shape[0]), bb_] = (ba_[bb_] == np.arange(Mv.shape[0])) & (
            Mv.max(1) > score_thr
        )
        ii, jj = np.where(Mv > score_thr)
        for i, j in zip(ii, jj):
            edges.append((ar[i], bc[j], float(Mv[i, j]), bool(rbh[i, j])))
    return pd.DataFrame(edges, columns=["src", "dst", "score", "rbh"])


def family_phylogenetic_signal(
    edges: pd.DataFrame,
    families: pd.DataFrame,
    divergence: dict[tuple[str, str], float],
    *,
    min_species: int = 8,
    min_pairs: int = 8,
    exclude_pairs: set[str] | None = None,
) -> pd.DataFrame:
    """Per-family Spearman correlation of within-family score with divergence.

    For each cell-type family (as returned by :func:`cluster_families`),
    take the per-species-pair *maximum* alignment score over within-family
    edges and correlate it against phylogenetic divergence. A family whose
    internal scores decay with divergence (negative ρ) behaves like a
    conserved cell-type lineage; one whose scores are flat or *increase*
    with divergence (positive ρ) is a generic-program bin — any cell type
    expressing the program lands there regardless of lineage, and at deep
    divergence the program match is the only signal left.

    The ``exclude_pairs`` control matters: a tight same-study clade (e.g.
    four placozoans from one paper) will dominate the rank correlation for
    every family it anchors. Passing those pair names here gives the
    clade-independent ρ.

    Parameters
    ----------
    edges : pandas.DataFrame
        Output of :func:`build_union_graph` (columns ``src``, ``dst``,
        ``score``; species-prefixed node names).
    families : pandas.DataFrame
        Output of :func:`cluster_families` (columns ``node``, ``family``).
    divergence : dict[tuple[str, str], float]
        Species-pair → divergence (any symmetric scalar, e.g. Mya). Either
        key order is looked up.
    min_species : int, optional
        Only test families spanning at least this many species. Default 8.
    min_pairs : int, optional
        Minimum within-family species-pairs to compute ρ. Default 8.
    exclude_pairs : set[str], optional
        Species-pair names (concatenated codes, e.g. ``"tahh"``) to drop
        before computing ρ_ex — typically same-study pairs.

    Returns
    -------
    pandas.DataFrame
        One row per family with columns ``family``, ``n_species``,
        ``n_pairs``, ``rho``, ``p``, ``rho_ex``, ``p_ex``, ``classification``
        (``"lineage"`` if ρ_ex < −0.15 and p_ex < 0.05; ``"program"`` if
        ρ_ex > +0.15 and p_ex < 0.05; else ``"ambiguous"``).
    """
    from scipy.stats import spearmanr

    fam_of = dict(zip(families["node"], families["family"]))
    e = edges.copy()
    e["src_sp"] = e["src"].str.split("_", n=1).str[0]
    e["dst_sp"] = e["dst"].str.split("_", n=1).str[0]
    e["src_fam"] = e["src"].map(fam_of)
    e["dst_fam"] = e["dst"].map(fam_of)
    e["pair"] = [
        a + b if (a, b) in divergence or (a + b) in (exclude_pairs or set()) else b + a
        for a, b in zip(e["src_sp"], e["dst_sp"])
    ]

    def _div(a: str, b: str) -> float | None:
        return divergence.get((a, b), divergence.get((b, a)))

    fam_nsp = families.groupby("family")["species"].nunique()
    excl = exclude_pairs or set()
    rows = []
    for fam, nsp in fam_nsp.items():
        if nsp < min_species:
            continue
        sub = e[(e["src_fam"] == fam) & (e["dst_fam"] == fam)]
        pp = sub.groupby(["src_sp", "dst_sp", "pair"])["score"].max().reset_index()
        pp["div"] = [_div(a, b) for a, b in zip(pp["src_sp"], pp["dst_sp"])]
        pp = pp.dropna(subset=["div"])
        if len(pp) < min_pairs:
            continue
        rho, p = spearmanr(pp["div"], pp["score"])
        pp_ex = pp[~pp["pair"].isin(excl)]
        if len(pp_ex) >= min_pairs:
            rho_ex, p_ex = spearmanr(pp_ex["div"], pp_ex["score"])
        else:
            rho_ex, p_ex = float("nan"), float("nan")
        if p_ex < 0.05 and rho_ex < -0.15:
            cls = "lineage"
        elif p_ex < 0.05 and rho_ex > 0.15:
            cls = "program"
        else:
            cls = "ambiguous"
        rows.append(
            {
                "family": fam,
                "n_species": int(nsp),
                "n_pairs": len(pp),
                "rho": float(rho),
                "p": float(p),
                "rho_ex": float(rho_ex),
                "p_ex": float(p_ex),
                "classification": cls,
            }
        )
    return pd.DataFrame(rows).sort_values("rho_ex").reset_index(drop=True)


def cluster_families(
    edges: pd.DataFrame,
    *,
    rbh_only: bool = True,
    resolution: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Leiden community detection on a union cell-type graph.

    Parameters
    ----------
    edges : pandas.DataFrame
        Output of ``build_union_graph`` (columns ``src``, ``dst``,
        ``score``, ``rbh``).
    rbh_only : bool, optional
        Partition over reciprocal-best edges only. Default True.
    resolution : float, optional
        Leiden resolution. Default 2.0.
    seed : int, optional
        Leiden RNG seed. Default 0.

    Returns
    -------
    pandas.DataFrame
        One row per node: ``node`` (species-prefixed label), ``species``,
        ``label``, ``family`` (community id), ``rbh_degree``
        (within-family RBH degree).
    """
    import igraph as ig
    import leidenalg

    nodes = sorted(set(edges["src"]) | set(edges["dst"]))
    nidx = {n: i for i, n in enumerate(nodes)}
    sub = edges[edges["rbh"]] if rbh_only else edges
    g = ig.Graph(n=len(nodes))
    g.add_edges([(nidx[r.src], nidx[r.dst]) for r in sub.itertuples()])
    g.es["weight"] = sub["score"].tolist()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight",
        seed=seed,
    )
    fam = pd.Series(part.membership, index=nodes, name="family")
    out = pd.DataFrame({"node": nodes, "family": fam.values})
    out["species"] = out["node"].str.split("_", n=1).str[0]
    out["label"] = out["node"].str.split("_", n=1).str[1]
    # within-family RBH degree
    rbh_e = edges[edges["rbh"]].copy()
    rbh_e["fam_src"] = rbh_e["src"].map(fam)
    rbh_e["fam_dst"] = rbh_e["dst"].map(fam)
    within = rbh_e[rbh_e["fam_src"] == rbh_e["fam_dst"]]
    deg = pd.concat([within["src"], within["dst"]]).value_counts()
    out["rbh_degree"] = out["node"].map(deg).fillna(0).astype(int)
    return out
