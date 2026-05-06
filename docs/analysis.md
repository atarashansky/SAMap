# Interpreting mappings (`samap.analysis`)

A mapping score tells you *that* two cell types align; it doesn't tell you
*why*, *how confidently*, or *how that confidence should change as you push
across deeper divergence*. The `samap.analysis` module exists to make the
score interpretable: which gene programs carry it, how it compares to a
permutation null, how degenerate the mapping is, and how it behaves when you
chain SAMap across many species.

```text
                    ┌────────────────────────────────────────┐
   sm.run() ─────▶  │  get_mapping_scores(which_iter=0|final) │  iter-0 vs converged
                    └────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
  permutation_null_scores   mapping_degeneracy        gene_modules
   (empirical p per pair)   (1:1-ness, entropy,            │
                              effective rank)              ▼
                                                  module_factored_scores
                                                  (which programs carry it)
                                                           │
                                                           ▼
                                              homology_graph_delta /
                                              find_paralog_substitutions
                                              (gene-level evolution)

   persist_pair ──▶ score_from_connectivities ──▶ build_union_graph ──▶ cluster_families
   (one-time write)        (re-score in 0.2s)        (all-pairs graph)   (Leiden families)
                                                                              │
                                                                              ▼
                                                                  family_phylogenetic_signal
                                                                  (lineage vs program)
```

## 1. Iter-0 vs converged scores

SAMap iteratively refines the cross-species manifold. The converged score
measures alignment on the *learned* joint embedding; the iteration-0 score
measures it on the *raw BLAST homology graph* — i.e., before SAMap has had a
chance to find anything. The two answer different questions: iter-0 is
"could a naive ortholog projection see this?", converged is "what does SAMap
see?". The delta between them is the value-add of manifold alignment.

```python
sm.run(joint_weights=True)            # default; stores nnm_per_iter
D0, M0 = get_mapping_scores(sm, keys, which_iter=0)
Df, Mf = get_mapping_scores(sm, keys, which_iter="final")
delta = Mf - M0                       # per-cluster-pair gain from alignment
```

`SAMAP.run` now writes `obsp['connectivities_iter0']` so the iter-0 graph is
recoverable post hoc.

## 2. Permutation null

The mapping score is a mean fraction of cross-species mutual-nearest-
neighbour mass landing in each target cluster. Holding the *manifold* fixed
and permuting per-species cluster labels gives an empirical null without
re-running alignment:

```python
from samap.analysis import permutation_null_scores

null = permutation_null_scores(sm, keys, n_perm=200)
# columns: ct_a, ct_b, score_obs, p_perm, q95_null, ...
```

On a 90 Mya pair (mouse↔human, k≈40 clusters) the q95 null is ≈0.01–0.05,
so any score above ~0.1 is very confidently non-random. Across a 750 Mya
pair the q95 null doesn't shift much — what shifts is how *many* clusters
exceed it. That's the degeneracy story.

## 3. Mapping degeneracy

Author-supplied annotations vary in granularity by 1–2 orders of magnitude
(10 dev stages vs 260 fine clusters). To compare alignment structure across
pairs you need a fixed cluster count, then a summary of how 1:1 the mapping
is:

```python
from samap.analysis import cluster_to_k, mapping_degeneracy

key = cluster_to_k(sam_a, k=40)                      # leiden at fixed k
deg = mapping_degeneracy(sm, {"hu": key_hu, "mm": key_mm})
# deg = {"mean_top1": ..., "rbh_frac": ..., "row_entropy": ..., "eff_rank": ...}
```

These metrics are the ones that track phylogenetic distance — they decay
cleanly out to ≈550 Mya and then plateau, while `max_score` saturates almost
immediately. The plateau defines the **tile-ability horizon**: how far apart
two atlases can be before SAMap can no longer recover a confident
near-bijective mapping. With careful same-protocol data the horizon is
≈90–160 Mya; cross-lab heterogeneous data hits the floor by ~160 Mya.

## 4. Module-factored scores

A high alignment score can be carried by a *single* shared gene program (the
ctenophore colloblast aligns to bilaterian cardiomyocytes through the
sarcomere module) or by *many independent* programs. Only the latter is
strong evidence of cell-type homology in the [Arendt CoRC] sense.

```python
from samap.analysis import gene_modules, module_factored_scores

mods = gene_modules(sm, with_coexpression=15, snn=15)   # ~30 program-sized modules
mf = module_factored_scores(sm, keys, modules=mods)
# adds: n_modules, top_module, top_module_frac, module_entropy
```

`with_coexpression` is load-bearing: the BLAST homology graph alone fragments
into thousands of disconnected components — e.g. 6,697 components for a
3-species mouse–human–zebrafish graph of 58k genes, including 1,295 singleton
genes, ~3,000 of size 2–3, and one giant component of ~20k genes — so a
direct Leiden partition is dominated by trivial size-2/3 modules and a single
unstructured giant. Bridging it with within-species gene–gene correlation kNN
consolidates these into interpretable program-sized modules (sarcomere,
ciliome, ECM, neural, lysosomal, …). `top_module_frac > 0.7` flags
single-program edges.

## 5. Homology-graph delta and paralog substitutions

After `sm.run()` the combined `adata` carries two gene–gene graphs: the input
BLAST graph (sequence similarity) and the reweighted graph (cross-species
expression correlation on the aligned manifold). Disagreements between them
are the gene-level evolution signal:

```python
from samap.analysis import homology_graph_delta, find_paralog_substitutions

delta = homology_graph_delta(sm)
# columns: gene_a, gene_b, seq_w, expr_w, residual

subs = find_paralog_substitutions(sm, min_corr=0.3)
# gene families where a non-best-hit paralog took over the expression role
```

Unlike the original `ParalogSubstitutions` (which needs an external
orthology DB), `find_paralog_substitutions` works directly off the SAMap
homology graph.

## 6. Disk-based ontology building

For pan-metazoan corpora (100+ pairs) holding `SAMAP` objects in memory is
prohibitive. After each run, persist the connectivity once and re-score
against any label set in milliseconds:

```python
from samap.analysis import (
    persist_pair, score_from_connectivities, build_union_graph,
    cluster_families, family_phylogenetic_signal,
)

# one-time, after each pairwise run:
persist_pair(sm, "atlas/persisted/mmhs")

# later, with arbitrary labels (no SAMAP objects):
labels = {sp: ad.read_h5ad(f"{sp}.h5ad").obs["leiden_k40"] for sp in SPECIES}
S = score_from_connectivities("atlas/persisted/mmhs", labels)   # ~0.2 s

# all-pairs union graph + Leiden families:
edges = build_union_graph("atlas/persisted", SPECIES, labels)
families = cluster_families(edges, rbh_only=True, resolution=2.0)
```

`cluster_families` returns one row per cell-type cluster with a `family`
assignment. RBH-only clustering at `resolution=2.0` is the recommended
starting point; lower-confidence edges fragment families at deep divergence.

## 7. Lineage signal vs program signal

A family whose internal scores decay with divergence behaves like a
conserved cell-type lineage; one whose scores are flat or *increase* with
divergence is a generic-program bin where any cell type expressing the
program lands regardless of phylogeny:

```python
sig = family_phylogenetic_signal(edges, families, divergence,
                                 exclude_pairs={"tath", "tahh", ...})
# rho       — Spearman ρ of within-family score vs divergence
# rho_ex    — same after dropping exclude_pairs
# classification — "lineage" / "program" / "ambiguous"
```

The `exclude_pairs` control matters: a tight same-study clade (e.g. four
placozoans from one paper) will dominate the rank correlation for every
family it anchors. Always report both `rho` and `rho_ex` — a family that
looks like a strong lineage but collapses on clade exclusion is a sampling
artifact, not biology.

## Quick reference

| Function                       | Input             | Output                               | Question                              |
|--------------------------------|-------------------|--------------------------------------|---------------------------------------|
| `get_mapping_scores`           | SAMAP + keys      | score DataFrames                     | How well do these cell types align?   |
| `permutation_null_scores`      | SAMAP + keys      | per-pair p-values                    | Is this score above chance?           |
| `cluster_to_k`                 | SAM + k           | obs key                              | Comparable granularity across species |
| `mapping_degeneracy`           | SAMAP + keys      | dict of metrics                      | How 1:1 is this mapping?              |
| `gene_modules`                 | SAMAP             | gene → module                        | What programs structure the homology? |
| `module_factored_scores`       | SAMAP + keys      | scores + module decomposition        | Which programs carry each match?      |
| `homology_graph_delta`         | SAMAP             | per-edge residuals                   | Where do seq and expr disagree?       |
| `find_paralog_substitutions`   | SAMAP             | per-gene-family events               | Which paralogs swapped roles?         |
| `persist_pair`                 | SAMAP + path      | files on disk                        | (one-time, enables disk re-scoring)   |
| `score_from_connectivities`    | path + labels     | score matrix                         | Re-score without SAMAP objects        |
| `build_union_graph`            | dir + labels      | edge DataFrame                       | All-pairs cell-type graph             |
| `cluster_families`             | edges             | node → family                        | Cross-species cell-type families      |
| `family_phylogenetic_signal`   | edges + families  | per-family ρ + classification        | Lineage or program?                   |
