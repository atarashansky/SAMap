"""Unit test: SAMAP init warns on low var_names↔BLAST overlap."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from samap import SAMAP


class _StubSAM:
    """Minimal SAM stand-in: just enough for SAMAP.__init__ up to _Samap_Iter."""

    def __init__(self, var_names: list[str]) -> None:
        import anndata as ad
        n_var = len(var_names)
        X = sp.csr_matrix(np.ones((4, n_var), dtype=np.float32))
        self.adata = ad.AnnData(X=X, var=pd.DataFrame(index=var_names))
        self.adata.obs["leiden_clusters"] = ["a", "a", "b", "b"]
        # Populate varm so SAMAP.__init__ skips prepare_SAMap_loadings.
        self.adata.varm["PCs_SAMap"] = np.zeros((n_var, 2), dtype=np.float32)
        # prepend_var_prefix touches adata_raw too.
        self.adata_raw = self.adata

    def preprocess_data(self, **kw): ...
    def run(self, **kw): ...
    def leiden_clustering(self, **kw): ...
    def dispersion_ranking_NN(self, **kw): ...


def _gnnm_for(gns_a: list[str], gns_b: list[str]):
    """Build a tiny dense homology graph linking every a↔b."""
    gns = np.array([f"aa_{g}" for g in gns_a] + [f"bb_{g}" for g in gns_b])
    n = gns.size
    m = np.zeros((n, n))
    m[: len(gns_a), len(gns_a) :] = 0.5
    m[len(gns_a) :, : len(gns_a)] = 0.5
    gns_dict = {"aa": gns[: len(gns_a)], "bb": gns[len(gns_a) :]}
    return sp.csr_matrix(m), gns, gns_dict


def test_low_overlap_emits_warning(caplog, monkeypatch):
    # bb var_names share 1/10 with the BLAST graph → 10% < 30% threshold
    sams = {
        "aa": _StubSAM([f"g{i}" for i in range(10)]),
        "bb": _StubSAM(["h0"] + [f"X{i}" for i in range(9)]),
    }
    gnnm = _gnnm_for([f"g{i}" for i in range(10)], [f"h{i}" for i in range(10)])

    # Don't actually spin up the iterator — we only care about the init-time
    # diagnostic, and _Samap_Iter does real work (precompute) on stub data.
    import samap.core.mapping as mapping
    monkeypatch.setattr(
        mapping, "_Samap_Iter", lambda *a, **k: object()
    )

    with caplog.at_level(logging.WARNING, logger="samap"):
        SAMAP(sams, gnnm=gnnm)

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Only 10.0% of 'bb'" in m for m in msgs), msgs
    # examples should be raw (prefix stripped)
    assert any("'X0'" in m or "X0" in m for m in msgs)
    assert any("h1" in m for m in msgs)


def test_high_overlap_no_warning(caplog, monkeypatch):
    sams = {
        "aa": _StubSAM([f"g{i}" for i in range(10)]),
        "bb": _StubSAM([f"h{i}" for i in range(10)]),
    }
    gnnm = _gnnm_for([f"g{i}" for i in range(10)], [f"h{i}" for i in range(10)])

    import samap.core.mapping as mapping
    monkeypatch.setattr(mapping, "_Samap_Iter", lambda *a, **k: object())

    with caplog.at_level(logging.WARNING, logger="samap"):
        SAMAP(sams, gnnm=gnnm)

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("matched the homology graph" in m for m in msgs), msgs
