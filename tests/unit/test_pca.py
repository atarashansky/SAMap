"""Unit tests for samap.sam.pca — ARPACK vs randomized SVD with implicit centering.

Randomized SVD and ARPACK find *different orthonormal bases* for the same
leading singular subspace — so we compare subspace angles and reconstruction
error, not raw component matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

from samap.core._backend import HAS_CUPY, Backend
from samap.sam.pca import _pca_with_sparse, randomized_svd_implicit_center

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def low_rank_sparse():
    """Sparse matrix with a strong low-rank signal.

    Built as (scaled low-rank) + small noise, then sparsified to ~10% density.
    The signal is boosted 100× relative to the noise so that the random
    sparsification (which zeros 90% of entries — a *large* structured
    perturbation) does not swamp the top singular directions. Without that
    boost the post-sparsification spectrum collapses to an almost-flat noise
    floor after the first 3-4 modes, making subspace comparison meaningless.
    """
    rng = np.random.default_rng(42)
    n, m = 500, 200
    rank = 15
    # Exponential singular spectrum, scaled large so it survives sparsification.
    svals = 100.0 * np.exp(-0.3 * np.arange(rank))
    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((m, rank))
    # Orthonormalise so svals are the actual pre-sparsification singular values
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)
    dense = (U * svals) @ V.T + 0.01 * rng.standard_normal((n, m))
    mask = rng.random((n, m)) > 0.10
    dense[mask] = 0.0
    X = sp.csr_matrix(dense.astype(np.float64))
    # rank=15 is the true signal rank — tests should compare subspaces up to
    # ~10-12 PCs; beyond that the noise-floor PCs have no unique "right
    # answer" and randomized vs ARPACK will legitimately differ.
    return X, n, m


# Within-signal-rank PC count for subspace comparisons. Beyond this the
# singular spectrum flattens out (noise floor from sparsification) and there
# is no unique correct basis — randomized and ARPACK will find different but
# equally valid directions.
_SIGNAL_PCS = 12


# ---------------------------------------------------------------------------
# Core equivalence — ARPACK vs randomized
# ---------------------------------------------------------------------------


class TestArpackVsRandomized:
    """Both solvers should recover the same leading singular subspace."""

    def test_explained_variance_close(self, low_rank_sparse):
        """Top-PC explained variances agree within ~1%.

        ARPACK converges tightly; randomized SVD is approximate but with 4
        power iterations and 10 oversamples it should match the top modes
        very well.
        """
        X, _, _ = low_rank_sparse
        k = 50
        out_arp = _pca_with_sparse(X, k, svd_solver="arpack", seed=0)
        out_rnd = _pca_with_sparse(X, k, svd_solver="randomized", seed=0)

        var_arp = out_arp["variance"]
        var_rnd = out_rnd["variance"]

        # Top 10 PCs: should be very close (< 1% relative error)
        rel_err_top = np.abs(var_arp[:10] - var_rnd[:10]) / var_arp[:10]
        assert rel_err_top.max() < 0.01, (
            f"Top-10 variance relative error {rel_err_top.max():.4f} > 1%"
        )

        # All k PCs: bound the aggregate. The tail PCs can drift more
        # (randomized SVD is less accurate there) but the *sum* of variances
        # should be within a few percent — that's what matters for PCA.
        var_sum_arp = var_arp.sum()
        var_sum_rnd = var_rnd.sum()
        assert abs(var_sum_arp - var_sum_rnd) / var_sum_arp < 0.05

    def test_subspace_angle_small(self, low_rank_sparse):
        """Principal angles between the two component bases should be near zero.

        We compare only the top `_SIGNAL_PCS` — within the true signal rank.
        Beyond that the noise floor has no unique basis (any orthonormal set
        of noise directions is equally correct) so randomized and ARPACK will
        find different but equally valid ones. The full k=50 subspaces would
        show a ~1 rad max angle purely from that ambiguity.
        """
        X, _, _ = low_rank_sparse
        k = 50  # oversampled — we still truncate the comparison below
        out_arp = _pca_with_sparse(X, k, svd_solver="arpack", seed=0)
        out_rnd = _pca_with_sparse(X, k, svd_solver="randomized", seed=0)

        # subspace_angles takes column vectors; components are row vectors
        angles = subspace_angles(
            out_arp["components"][:_SIGNAL_PCS].T,
            out_rnd["components"][:_SIGNAL_PCS].T,
        )
        # With n_power=4 and a clear spectral gap, the top subspace should
        # align to well under 0.1 rad.
        assert angles.max() < 0.1, (
            f"Max principal angle {angles.max():.4f} rad — subspaces disagree"
        )

    def test_reconstruction_error_similar(self, low_rank_sparse):
        """Low-rank reconstruction ``X_pca @ components`` should be near-identical.

        This is basis-invariant: the product ``U·Σ·Vᵀ`` is the same for any
        pair of orthonormal bases spanning the same singular subspace.
        Restricted to the signal PCs only — noise-floor PCs reconstruct
        different (but equally valid) noise approximations.
        """
        X, _, _ = low_rank_sparse
        k = 50
        out_arp = _pca_with_sparse(X, k, svd_solver="arpack", seed=0)
        out_rnd = _pca_with_sparse(X, k, svd_solver="randomized", seed=0)

        rec_arp = out_arp["X_pca"][:, :_SIGNAL_PCS] @ out_arp["components"][:_SIGNAL_PCS]
        rec_rnd = out_rnd["X_pca"][:, :_SIGNAL_PCS] @ out_rnd["components"][:_SIGNAL_PCS]

        diff = np.linalg.norm(rec_arp - rec_rnd)
        ref = np.linalg.norm(rec_arp)
        assert diff / ref < 0.05, f"Reconstruction differs by {diff/ref:.2%}"

    def test_output_shapes_and_dtypes(self, low_rank_sparse):
        """Both paths return the same dict schema."""
        X, n, m = low_rank_sparse
        k = 20
        for solver in ("arpack", "randomized"):
            out = _pca_with_sparse(X, k, svd_solver=solver, seed=0)
            assert out["X_pca"].shape == (n, k)
            assert out["components"].shape == (k, m)
            assert out["variance"].shape == (k,)
            assert out["variance_ratio"].shape == (k,)
            # Variances should be descending
            assert (np.diff(out["variance"]) <= 1e-10).all()

    def test_variance_ratio_matches_variance(self, low_rank_sparse):
        """variance_ratio should be variance / total_var, same total for both."""
        X, _, _ = low_rank_sparse
        k = 10
        out_arp = _pca_with_sparse(X, k, svd_solver="arpack", seed=0)
        out_rnd = _pca_with_sparse(X, k, svd_solver="randomized", seed=0)
        # Both divide by the same total (sum of column variances of X)
        total_arp = out_arp["variance"][0] / out_arp["variance_ratio"][0]
        total_rnd = out_rnd["variance"][0] / out_rnd["variance_ratio"][0]
        assert abs(total_arp - total_rnd) / total_arp < 1e-6


# ---------------------------------------------------------------------------
# Ground truth — compare against sklearn dense PCA
# ---------------------------------------------------------------------------


class TestVsSklearnDense:
    """sklearn.PCA on the densified matrix is the ground-truth reference."""

    @pytest.fixture(scope="class")
    def sklearn_ref(self, low_rank_sparse):
        X, _, _ = low_rank_sparse
        pca = PCA(n_components=50, svd_solver="full")
        X_pca = pca.fit_transform(X.toarray())
        return {
            "X_pca": X_pca,
            "components": pca.components_,
            "variance": pca.explained_variance_,
        }

    @pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
    def test_variance_vs_sklearn(self, low_rank_sparse, sklearn_ref, svd_solver):
        """Explained variances match sklearn within 1% on the top PCs."""
        X, _, _ = low_rank_sparse
        out = _pca_with_sparse(X, 50, svd_solver=svd_solver, seed=0)
        var_ref = sklearn_ref["variance"][:10]
        var_got = out["variance"][:10]
        rel_err = np.abs(var_ref - var_got) / var_ref
        assert rel_err.max() < 0.01, (
            f"{svd_solver}: top-10 variance differs from sklearn by {rel_err.max():.4f}"
        )

    @pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
    def test_subspace_vs_sklearn(self, low_rank_sparse, sklearn_ref, svd_solver):
        """Component subspace aligns with sklearn's (within the signal rank)."""
        X, _, _ = low_rank_sparse
        out = _pca_with_sparse(X, 50, svd_solver=svd_solver, seed=0)
        angles = subspace_angles(
            sklearn_ref["components"][:_SIGNAL_PCS].T,
            out["components"][:_SIGNAL_PCS].T,
        )
        assert angles.max() < 0.1


# ---------------------------------------------------------------------------
# API & edge cases
# ---------------------------------------------------------------------------


class TestAPIAndEdgeCases:
    def test_arpack_is_default(self, low_rank_sparse):
        """Omitting svd_solver picks ARPACK — no behaviour change for callers."""
        X, _, _ = low_rank_sparse
        out_default = _pca_with_sparse(X, 10, seed=0)
        out_arpack = _pca_with_sparse(X, 10, svd_solver="arpack", seed=0)
        np.testing.assert_array_equal(out_default["variance"], out_arpack["variance"])

    def test_randomized_rejects_mu_axis_1(self, low_rank_sparse):
        """Row-centering (mu_axis=1) is not supported on the randomized path."""
        X, _, _ = low_rank_sparse
        with pytest.raises(ValueError, match="mu_axis=0"):
            _pca_with_sparse(X, 10, svd_solver="randomized", mu_axis=1)

    def test_randomized_accepts_precomputed_mu(self, low_rank_sparse):
        """Passing an explicit mean should give the same result as auto-computing it."""
        X, _, _ = low_rank_sparse
        mu = np.asarray(X.mean(axis=0))
        out_auto = randomized_svd_implicit_center(X, 10, mu=None, seed=0)
        out_manual = randomized_svd_implicit_center(X, 10, mu=mu, seed=0)
        np.testing.assert_allclose(out_auto["variance"], out_manual["variance"], rtol=1e-10)

    def test_randomized_seeded_determinism(self, low_rank_sparse):
        """Same seed → identical output; different seed → different sketch."""
        X, _, _ = low_rank_sparse
        out1 = randomized_svd_implicit_center(X, 10, seed=7)
        out2 = randomized_svd_implicit_center(X, 10, seed=7)
        out3 = randomized_svd_implicit_center(X, 10, seed=8)
        np.testing.assert_array_equal(out1["X_pca"], out2["X_pca"])
        # With well-separated spectrum the *variances* converge regardless of
        # seed, but X_pca bytes will differ through the random sketch.
        assert not np.array_equal(out1["X_pca"], out3["X_pca"])

    def test_n_power_improves_tail_accuracy(self, low_rank_sparse):
        """More power iterations → tighter match to ARPACK on the tail PCs.

        This is a sanity check on the power-iteration plumbing. With
        n_power=0 the tail diverges; with n_power=4 it should tighten up.
        """
        X, _, _ = low_rank_sparse
        out_ref = _pca_with_sparse(X, 30, svd_solver="arpack", seed=0)

        out_p0 = randomized_svd_implicit_center(X, 30, n_power=0, seed=0)
        out_p4 = randomized_svd_implicit_center(X, 30, n_power=4, seed=0)

        # Compare tail (PCs 20-30) explained-variance error
        err_p0 = np.abs(out_p0["variance"][20:] - out_ref["variance"][20:]).sum()
        err_p4 = np.abs(out_p4["variance"][20:] - out_ref["variance"][20:]).sum()
        # Power iterations must help, or the plumbing is broken
        assert err_p4 < err_p0


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not installed")
class TestGPU:
    """Same comparisons with a CUDA backend."""

    def test_gpu_matches_cpu_randomized(self, low_rank_sparse):
        """GPU and CPU randomized SVD recover the same subspace.

        We don't compare bytes (cupy and numpy RNG streams differ, and float
        accumulation order differs on GPU) — we compare subspace angles.
        """
        X, _, _ = low_rank_sparse
        bk_gpu = Backend("cuda")
        X_gpu = bk_gpu.to_device(X)

        out_cpu = randomized_svd_implicit_center(X, 30, seed=0, bk=Backend("cpu"))
        out_gpu = randomized_svd_implicit_center(X_gpu, 30, seed=0, bk=bk_gpu)

        angles = subspace_angles(
            out_cpu["components"][:_SIGNAL_PCS].T,
            out_gpu["components"][:_SIGNAL_PCS].T,
        )
        assert angles.max() < 0.1

        # Variances should agree to within numerical tolerance — both are
        # approximating the same singular values.
        np.testing.assert_allclose(
            out_cpu["variance"][:10], out_gpu["variance"][:10], rtol=0.02
        )

    def test_gpu_vs_arpack(self, low_rank_sparse):
        """GPU randomized SVD matches CPU ARPACK ground truth."""
        X, _, _ = low_rank_sparse
        bk_gpu = Backend("cuda")
        X_gpu = bk_gpu.to_device(X)

        out_ref = _pca_with_sparse(X, 30, svd_solver="arpack", seed=0)
        out_gpu = randomized_svd_implicit_center(X_gpu, 30, seed=0, bk=bk_gpu)

        rel_err = np.abs(out_ref["variance"][:10] - out_gpu["variance"][:10])
        rel_err /= out_ref["variance"][:10]
        assert rel_err.max() < 0.02
