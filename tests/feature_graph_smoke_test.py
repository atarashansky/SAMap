"""Smoke test for samap.io.feature_graph builders.

Constructs a tiny synthetic ATAC (100 cells × 500 peaks, binary) and RNA
(100 cells × 200 genes) pair plus a 10-gene GTF, then verifies:

1. ``gnnm_from_gtf`` produces a non-empty, symmetric GnnmTuple under both
   ``kind='tss_window'`` and ``kind='powerlaw'``.
2. ``compose_gnnm`` chains a peak→gene graph with a gene-ortholog graph.
3. ``prepare_atac_sam`` populates the SAM slots SAMap reads
   (``var['weights']``, ``varm['PCs_SAMap']``, ``obsp['connectivities']``).
4. ``SAMAP(gnnm=...)`` accepts the tuple and survives ``__init__`` —
   the homology graph and per-dataset SAM objects wire together.

This is plumbing verification only; the synthetic data is random and a
full ``SAMAP.run()`` on it would not be meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

import samap.io as sio
from samap import SAMAP
from samap.sam import SAM


def _make_synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)

    # --- 10-gene GTF on chr1/chr2 ---------------------------------------
    gene_ids = [f"GENE{i:03d}" for i in range(10)]
    chroms = ["chr1"] * 6 + ["chr2"] * 4
    starts = np.array(
        [10_000 + i * 50_000 for i in range(6)] + [10_000 + i * 50_000 for i in range(4)]
    )
    ends = starts + 5_000
    strands = list("+-" * 5)
    genes = pd.DataFrame(
        {
            "Chromosome": chroms,
            "Start": starts,
            "End": ends,
            "Strand": strands,
            "gene_id": gene_ids,
        }
    )

    # --- RNA: 100 cells × 200 genes (the 10 GTF genes + 190 fillers) ----
    rna_gene_names = gene_ids + [f"FILLER{i:03d}" for i in range(190)]
    rna_X = rng.poisson(2.0, size=(100, 200)).astype(np.float32)
    rna = AnnData(
        X=sp.csr_matrix(rna_X),
        obs=pd.DataFrame(index=[f"rcell{i}" for i in range(100)]),
        var=pd.DataFrame(index=rna_gene_names),
    )

    # --- ATAC: 100 cells × 500 peaks, binary, peaks tile chr1/chr2 ------
    peak_chrom = rng.choice(["chr1", "chr2"], size=500, p=[0.6, 0.4])
    span = np.where(peak_chrom == "chr1", 6 * 50_000 + 10_000, 4 * 50_000 + 10_000)
    peak_start = (rng.random(500) * span).astype(int)
    # Guarantee unique (chrom, start) so var_names are unique.
    seen: set[tuple[str, int]] = set()
    for i in range(500):
        while (peak_chrom[i], int(peak_start[i])) in seen:
            peak_start[i] += 1
        seen.add((peak_chrom[i], int(peak_start[i])))
    peak_end = peak_start + 500
    peak_names = [f"{c}:{s}-{e}" for c, s, e in zip(peak_chrom, peak_start, peak_end)]
    atac_X = (rng.random((100, 500)) < 0.1).astype(np.float32)
    atac = AnnData(
        X=sp.csr_matrix(atac_X),
        obs=pd.DataFrame(index=[f"acell{i}" for i in range(100)]),
        var=pd.DataFrame(index=peak_names),
    )

    return atac, rna, genes


def main() -> None:
    atac, rna, genes = _make_synthetic()
    print(f"synthetic: atac {atac.shape}, rna {rna.shape}, gtf {len(genes)} genes")

    # ---- gnnm_from_gtf: tss_window ------------------------------------
    g_tss = sio.gnnm_from_gtf(atac, rna, genes, ids=("at", "rn"), kind="tss_window")
    G, gns, gd = g_tss
    assert G.nnz > 0, "tss_window produced zero edges"
    assert set(gd) == {"at", "rn"}
    assert (abs(G - G.T)).nnz == 0, "graph not symmetric"
    assert all(g.startswith(("at_", "rn_")) for g in gns)
    print(
        f"[ok] gnnm_from_gtf tss_window: {G.shape}, nnz={G.nnz}, "
        f"|at|={gd['at'].size}, |rn|={gd['rn'].size}"
    )

    # ---- gnnm_from_gtf: powerlaw --------------------------------------
    g_pl = sio.gnnm_from_gtf(
        atac, rna, genes, ids=("at", "rn"), kind="powerlaw", window=150_000, gamma=0.87
    )
    Gp, _, _ = g_pl
    assert Gp.nnz > 0
    assert Gp.data.min() > 0 and Gp.data.max() <= 1.0
    print(
        f"[ok] gnnm_from_gtf powerlaw: nnz={Gp.nnz}, "
        f"weight range=[{Gp.data.min():.3g}, {Gp.data.max():.3g}]"
    )

    # ---- compose_gnnm: peak→gene(at,rn) ∘ ortholog(rn,mm) -------------
    ortho_pairs = [(f"GENE{i:03d}", f"mmGene{i}") for i in range(10)]
    g_orth = sio.gnnm_from_pairs(
        ortho_pairs,
        ids={"rn": [f"GENE{i:03d}" for i in range(10)], "mm": [f"mmGene{i}" for i in range(10)]},
    )
    g_comp = sio.compose_gnnm(g_tss, g_orth, via="rn")
    Gc, _gns_c, gd_c = g_comp
    assert set(gd_c) == {"at", "mm"}
    assert Gc.nnz > 0
    print(f"[ok] compose_gnnm: at({gd_c['at'].size}) ∘ rn → mm({gd_c['mm'].size}), nnz={Gc.nnz}")

    # ---- prepare_atac_sam --------------------------------------------
    atac_sam = sio.prepare_atac_sam(atac, n_components=20, k=10)
    ad = atac_sam.adata
    assert "weights" in ad.var
    assert "PCs_SAMap" in ad.varm
    assert "connectivities" in ad.obsp
    assert ad.varm["PCs_SAMap"].shape[1] == 20
    assert ad.uns.get("modality") == "atac"
    print(
        f"[ok] prepare_atac_sam: X={ad.X.shape}, "
        f"PCs_SAMap={ad.varm['PCs_SAMap'].shape}, "
        f"weights∈[{ad.var['weights'].min():.3f},{ad.var['weights'].max():.3f}]"
    )

    # ---- SAMAP init accepts the tuple --------------------------------
    rna_sam = SAM(counts=rna)
    rna_sam.preprocess_data(
        sum_norm="cell_median", norm="log", thresh_low=0.0, thresh_high=1.0, min_expression=0
    )
    rna_sam.run(
        preprocessing="StandardScaler",
        npcs=20,
        weight_PCs=False,
        k=10,
        n_genes=200,
        weight_mode="rms",
        verbose=False,
    )

    # Rebuild gnnm against the (possibly filtered) atac_sam var_names so
    # the overlap diagnostic doesn't fire.
    g_init = sio.gnnm_from_gtf(
        atac_sam.adata, rna_sam.adata, genes, ids=("at", "rn"), kind="tss_window"
    )

    sm = SAMAP(
        {"at": atac_sam, "rn": rna_sam},
        gnnm=g_init,
        keys={"at": "leiden_clusters", "rn": "leiden_clusters"},
    )
    assert sm.gnnm.nnz > 0
    assert set(sm.gns_dict) == {"at", "rn"}
    sizes = {k: int(v.size) for k, v in sm.gns_dict.items()}
    print(f"[ok] SAMAP.__init__: gnnm={sm.gnnm.shape} nnz={sm.gnnm.nnz}, gns_dict sizes={sizes}")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
