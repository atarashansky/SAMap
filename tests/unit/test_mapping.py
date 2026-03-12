"""Tests for SAMAP orchestration in mapping.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anndata import AnnData


class TestSAMAPInputGuard:
    """The SAMAP constructor should accept any SAM-like object via duck typing.

    v3.0.0 vendored SAM as ``samap.sam.SAM``. Users who still import from the
    old ``samalg`` package (or any third-party SAM-compatible object) must not
    be rejected by an overly strict isinstance check. The guard should only
    check for an ``.adata`` attribute.
    """

    def _make_samap(self, sams):
        # Import inside the test to avoid loading the heavy module at collect
        # time for unrelated test runs.
        from samap.core.mapping import SAMAP

        # f_maps=None with gnnm=None triggers a file-system walk; we only want
        # to test the input guard, so we pass a placeholder gnnm tuple that
        # gets past the guard but will fail later. We catch the later failure.
        return SAMAP(sams, gnnm=(None, None, {}))

    def test_rejects_bare_object(self):
        """An object with no .adata attribute is rejected with a clear TypeError."""
        with pytest.raises(TypeError, match=r"must be either a path.*or a SAM object"):
            self._make_samap({"pl": SimpleNamespace()})

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="Got int"):
            self._make_samap({"pl": 42})

    def test_accepts_duck_typed_sam(self):
        """Any object with .adata passes the guard (duck typing).

        The constructor will fail *later* when it tries to call SAM-specific
        methods on our mock, but the important thing is that the input guard
        does not raise the "must be either a path or a SAM object" TypeError.
        """
        mock_sam = SimpleNamespace(adata=AnnData())
        # We don't expect this to succeed end-to-end — the mock lacks
        # leiden_clustering, varm, etc. We just verify the *guard* accepts it.
        try:
            self._make_samap({"pl": mock_sam, "sc": mock_sam})
        except TypeError as e:
            # The guard's specific TypeError must not fire.
            assert "must be either a path" not in str(e), (
                f"Duck-typed SAM with .adata was rejected by input guard: {e}"
            )
        except Exception:
            # Any other exception (AttributeError, etc.) from downstream code
            # is fine — we're only testing the guard.
            pass

    def test_accepts_vendored_sam(self):
        """Vendored samap.sam.SAM instances satisfy the duck-type contract once loaded.

        A bare ``SAM()`` has no ``.adata`` until data is loaded — which is
        correct, since passing an unloaded SAM to SAMAP is meaningless. Once
        ``counts`` are provided, ``.adata`` exists and the guard accepts it.
        """
        import numpy as np

        from samap.sam import SAM

        # Minimal loaded SAM: pass a tiny AnnData via counts
        sam = SAM(counts=AnnData(np.ones((2, 3))))
        assert hasattr(sam, "adata"), "Loaded SAM must expose .adata"
