"""Self-Assembling Manifold (SAM) algorithm — vendored core.

Vendored from samalg.sam (sc-sam v2.0.2). Contains the SAM class with
only the methods SAMap actually uses. Heavy features dropped during
vendoring:
    - run_umap, run_tsne, run_diff_map, run_diff_umap
    - kmeans_clustering, hdbscan_clustering, louvain_clustering
    - identify_marker_genes_*
    - save/load (dill-based)

Fixes applied:
    - Removed obsm["X_processed"] = D_sub (n_cells x n_genes stored every
      iteration, never read — pure memory waste).
    - Replaced .tolil() + .setdiag() + .tocsr() cycles with direct CSR
      setdiag (no format round-trip).
    - Dropped numba import (SAM has no @jit functions; import existed
      only to suppress a warning).

Copyright 2018, Alexander J. Tarashansky.
"""

from __future__ import annotations

import contextlib
import gc
import time
import warnings
from typing import TYPE_CHECKING, Any, Literal

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.utils.sparsefuncs as sf
from anndata import AnnData
from scipy.sparse import SparseEfficiencyWarning
from sklearn.preprocessing import Normalizer

from .._logging import get_logger
from .knn import calc_nnm
from .pca import _pca_with_sparse, weighted_PCA
from .utils import convert_annotations

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from samap.core._backend import Backend

logger = get_logger("samap.sam")


class DataNotLoadedError(RuntimeError):
    """Raised when an operation requires data that has not been loaded."""

    def __init__(self, msg: str | None = None) -> None:
        super().__init__(
            msg or "No data has been loaded. Use load_data() or pass data to the constructor."
        )


class InvalidParameterError(ValueError):
    """Raised when a parameter has an invalid value."""

    def __init__(self, param: str, value: Any, valid_values: list[Any] | None = None) -> None:
        msg = f"Invalid value for '{param}': {value!r}."
        if valid_values:
            msg += f" Valid values: {valid_values}."
        super().__init__(msg)


def _csr_setdiag(mat: sp.csr_matrix, val: float) -> sp.csr_matrix:
    """Set the diagonal of a CSR matrix in place, suppressing efficiency warnings.

    scipy's CSR setdiag works natively; the lil round-trip in the original
    SAM code was unnecessary. The SparseEfficiencyWarning fires only if
    the diagonal entries don't already exist in the sparsity structure.
    For SAM's k-NN matrices, the diagonal is always present (hnswlib
    returns each point as its own nearest neighbor), so this is a no-op
    structural change — just a data overwrite.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        mat.setdiag(val)
    if val == 0:
        mat.eliminate_zeros()
    return mat


class SAM:
    """Self-Assembling Manifolds single-cell RNA sequencing analysis tool.

    SAM iteratively rescales the input gene expression matrix to emphasize
    genes that are spatially variable along the intrinsic manifold of the data.
    It outputs the gene weights, nearest neighbor matrix, and a 2D projection.

    Parameters
    ----------
    counts : tuple | list | pd.DataFrame | AnnData | None
        Input data in one of the following formats:
        - tuple/list: (data, gene_names, cell_names) where data is sparse/dense matrix
        - pd.DataFrame: cells x genes expression matrix
        - AnnData: annotated data object
    inplace : bool, optional
        If True and counts is AnnData, use the object directly without copying.
        Default is False.

    Attributes
    ----------
    preprocess_args : dict
        Dictionary of arguments used for the 'preprocess_data' function.
    run_args : dict
        Dictionary of arguments used for the 'run' function.
    adata_raw : AnnData
        An AnnData object containing the raw, unfiltered input data.
    adata : AnnData
        An AnnData object containing all processed data and SAM outputs.
    """

    def __init__(
        self,
        counts: (
            tuple[sp.spmatrix | NDArray[np.floating[Any]], NDArray[Any], NDArray[Any]]
            | list[Any]
            | pd.DataFrame
            | AnnData
            | None
        ) = None,
        inplace: bool = False,
        bk: Backend | None = None,
    ) -> None:
        # Backend for GPU dispatch in the iteration-loop hot spots
        # (dispersion SpMM, sparse PCA, kNN). Lazy-construct Backend("cpu")
        # if not provided so importing SAM doesn't force a dependency on
        # samap.core._backend at module-load time.
        if bk is None:
            from samap.core._backend import Backend as _Backend

            bk = _Backend("cpu")
        self._bk = bk

        self.run_args: dict[str, Any] = {}
        self.preprocess_args: dict[str, Any] = {}

        if isinstance(counts, (tuple, list)):
            raw_data, all_gene_names, all_cell_names = counts
            if isinstance(raw_data, np.ndarray):
                raw_data = sp.csr_matrix(raw_data)

            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

        elif isinstance(counts, pd.DataFrame):
            raw_data = sp.csr_matrix(counts.values)
            all_gene_names = np.array(list(counts.columns.values))
            all_cell_names = np.array(list(counts.index.values))

            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

        elif isinstance(counts, AnnData):
            all_cell_names = np.array(list(counts.obs_names))
            all_gene_names = np.array(list(counts.var_names))
            if counts.is_view:
                counts = counts.copy()

            if inplace:
                self.adata_raw = counts
            else:
                self.adata_raw = counts.copy()

        elif counts is not None:
            raise TypeError(
                "'counts' must be either a tuple/list of "
                "(data, gene IDs, cell IDs), a Pandas DataFrame of "
                "cells x genes, or an AnnData object."
            )

        if counts is not None:
            if np.unique(all_gene_names).size != all_gene_names.size:
                self.adata_raw.var_names_make_unique()
            if np.unique(all_cell_names).size != all_cell_names.size:
                self.adata_raw.obs_names_make_unique()

            if inplace:
                self.adata = self.adata_raw
            else:
                self.adata = self.adata_raw.copy()

            if "X_disp" not in self.adata_raw.layers:
                self.adata.layers["X_disp"] = self.adata.X

    def preprocess_data(
        self,
        div: float = 1,
        downsample: float = 0,
        sum_norm: str | float | None = "cell_median",
        norm: str | None = "log",
        min_expression: float = 1,
        thresh_low: float = 0.0,
        thresh_high: float = 0.96,
        thresh: float | None = None,
        filter_genes: bool = True,
    ) -> None:
        """Log-normalize and filter the expression data.

        Parameters
        ----------
        div : float, optional
            The factor by which the gene expression will be divided prior to
            normalization. Default is 1.
        downsample : float, optional
            The factor by which to randomly downsample the data. If 0, the
            data will not be downsampled. Default is 0.
        sum_norm : str | float | None, optional
            Library normalization method. Options:
            - float: Normalize each cell to this total count
            - 'cell_median': Normalize to median total count per cell
            - 'gene_median': Normalize genes to median total count per gene
            - None: No normalization
            Default is 'cell_median'.
        norm : str | None, optional
            Data transformation method. Options:
            - 'log': log2(x + 1) transformation
            - 'ftt': Freeman-Tukey variance-stabilizing transformation
            - 'asin': arcsinh transformation
            - 'multinomial': Pearson residual transformation (experimental)
            - None: No transformation
            Default is 'log'.
        min_expression : float, optional
            Threshold above which a gene is considered expressed. Values below
            this are set to zero. Default is 1.
        thresh_low : float, optional
            Keep genes expressed in greater than thresh_low*100% of cells.
            Default is 0.0.
        thresh_high : float, optional
            Keep genes expressed in less than thresh_high*100% of cells.
            Default is 0.96.
        thresh : float | None, optional
            If provided, sets thresh_low=thresh and thresh_high=1-thresh.
        filter_genes : bool, optional
            Whether to apply gene filtering. Default is True.
        """
        if thresh is not None:
            thresh_low = thresh
            thresh_high = 1 - thresh

        if not hasattr(self, "adata_raw"):
            raise DataNotLoadedError()

        self.preprocess_args = {
            "div": div,
            "sum_norm": sum_norm,
            "norm": norm,
            "min_expression": min_expression,
            "thresh_low": thresh_low,
            "thresh_high": thresh_high,
            "filter_genes": filter_genes,
        }

        self.run_args = self.adata.uns.get("run_args", {})

        D = self.adata_raw.X
        self.adata = self.adata_raw.copy()

        D = self.adata.X
        if isinstance(D, np.ndarray):
            D = sp.csr_matrix(D, dtype="float32")
        else:
            if str(D.dtype) != "float32":
                D = D.astype("float32")
            D.sort_indices()

        if D.getformat() == "csc":
            D = D.tocsr()

        # Sum-normalize
        if sum_norm == "cell_median" and norm != "multinomial":
            s = np.asarray(D.sum(1)).flatten()
            sum_norm_val = np.median(s)
            D = D.multiply(1 / s[:, None] * sum_norm_val).tocsr()
        elif sum_norm == "gene_median" and norm != "multinomial":
            s = np.asarray(D.sum(0)).flatten()
            sum_norm_val = np.median(s[s > 0])
            s[s == 0] = 1
            D = D.multiply(1 / s[None, :] * sum_norm_val).tocsr()
        elif sum_norm is not None and norm != "multinomial":
            D = D.multiply(1 / np.asarray(D.sum(1)).flatten()[:, None] * sum_norm).tocsr()

        # Normalize
        self.adata.X = D
        if norm is None:
            D.data[:] = D.data / div

        elif norm.lower() == "log":
            D.data[:] = np.log2(D.data / div + 1)

        elif norm.lower() == "ftt":
            D.data[:] = np.sqrt(D.data / div) + np.sqrt(D.data / div + 1) - 1

        elif norm.lower() == "asin":
            D.data[:] = np.arcsinh(D.data / div)

        elif norm.lower() == "multinomial":
            ni = np.asarray(D.sum(1)).flatten()  # cells
            pj = np.asarray(D.sum(0) / D.sum()).flatten()  # genes
            col = D.indices
            row = []
            for i in range(D.shape[0]):
                row.append(i * np.ones(D.indptr[i + 1] - D.indptr[i]))
            row = np.concatenate(row).astype("int32")
            mu = sp.coo_matrix((ni[row] * pj[col], (row, col))).tocsr()
            mu2 = mu.copy()
            mu2.data[:] = mu2.data**2
            mu2 = mu2.multiply(1 / ni[:, None])
            mu.data[:] = (D.data - mu.data) / np.sqrt(mu.data - mu2.data)

            self.adata.X = mu
            if sum_norm is None:
                sum_norm = np.median(ni)
            D = D.multiply(1 / ni[:, None] * sum_norm).tocsr()
            D.data[:] = np.log2(D.data / div + 1)

        else:
            D.data[:] = D.data / div

        # Zero-out low-expressed genes
        idx = np.where(D.data <= min_expression)[0]
        D.data[idx] = 0

        # Filter genes
        idx_genes = np.arange(D.shape[1])
        if filter_genes:
            a, ct = np.unique(D.indices, return_counts=True)
            c = np.zeros(D.shape[1])
            c[a] = ct

            keep = np.where(
                np.logical_and(c / D.shape[0] > thresh_low, c / D.shape[0] <= thresh_high)
            )[0]

            idx_genes = np.array(list(set(keep) & set(idx_genes)), dtype=np.intp)

        mask_genes = np.zeros(D.shape[1], dtype="bool")
        mask_genes[idx_genes] = True

        self.adata.X = self.adata.X.multiply(mask_genes[None, :]).tocsr()
        self.adata.X.eliminate_zeros()
        self.adata.var["mask_genes"] = mask_genes

        if norm == "multinomial":
            self.adata.layers["X_disp"] = D.multiply(mask_genes[None, :]).tocsr()
            self.adata.layers["X_disp"].eliminate_zeros()
        else:
            self.adata.layers["X_disp"] = self.adata.X

        self.calculate_mean_var()

        self.adata.uns["preprocess_args"] = self.preprocess_args
        self.adata.uns["run_args"] = self.run_args

    def calculate_mean_var(self, adata: AnnData | None = None) -> None:
        """Calculate mean and variance for each gene.

        Parameters
        ----------
        adata : AnnData | None, optional
            The AnnData object to calculate statistics for.
            If None, uses self.adata.
        """
        if adata is None:
            adata = self.adata

        if sp.issparse(adata.X):
            mu, var = sf.mean_variance_axis(adata.X, axis=0)
        else:
            mu = adata.X.mean(0)
            var = adata.X.var(0)

        adata.var["means"] = mu
        adata.var["variances"] = var

    def get_labels(self, key: str) -> NDArray[Any]:
        """Get labels from obs.

        Parameters
        ----------
        key : str
            Key in adata.obs.

        Returns
        -------
        NDArray
            Array of labels.
        """
        if key not in list(self.adata.obs.keys()):
            logger.warning("Key '%s' does not exist in `obs`.", key)
            return np.array([])
        return np.array(list(self.adata.obs[key]))

    def load_data(
        self,
        filename: str,
        transpose: bool = True,
        sep: str = ",",
        calculate_avg: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load expression data from file.

        Parameters
        ----------
        filename : str
            Path to the data file. Supported formats:
            - .csv/.txt: Tabular format (genes x cells by default)
            - .h5ad: AnnData format
        transpose : bool, optional
            If True (default), assumes file is genes x cells.
            Set to False if file is cells x genes.
        sep : str, optional
            Delimiter for CSV/TXT files. Default is ','.
        calculate_avg : bool, optional
            If True and loading .h5ad with existing neighbors, perform
            kNN averaging. Default is False.
        **kwargs
            Additional arguments passed to file loading functions.
        """
        ext = filename.split(".")[-1]

        if ext != "h5ad":
            df = pd.read_csv(filename, sep=sep, index_col=0, **kwargs)
            dataset = df.T if transpose else df

            raw_data = sp.csr_matrix(dataset.values)
            all_cell_names = np.array(list(dataset.index.values))
            all_gene_names = np.array(list(dataset.columns.values))

            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

            if np.unique(all_gene_names).size != all_gene_names.size:
                self.adata_raw.var_names_make_unique()
            if np.unique(all_cell_names).size != all_cell_names.size:
                self.adata_raw.obs_names_make_unique()

            self.adata = self.adata_raw.copy()
            self.adata.layers["X_disp"] = raw_data

        else:
            self.adata = anndata.read_h5ad(filename, **kwargs)
            if self.adata.raw is not None:
                self.adata_raw = AnnData(X=self.adata.raw.X)
                self.adata_raw.var_names = self.adata.var_names
                self.adata_raw.obs_names = self.adata.obs_names
                self.adata_raw.obs = self.adata.obs

                del self.adata.raw

                if (
                    "X_knn_avg" not in self.adata.layers
                    and "connectivities" in self.adata.obsp
                    and calculate_avg
                ):
                    self.dispersion_ranking_NN(save_avgs=True)
            else:
                self.adata_raw = self.adata

            if "X_disp" not in list(self.adata.layers.keys()):
                self.adata.layers["X_disp"] = self.adata.X

        filename = ".".join(filename.split(".")[:-1]) + ".h5ad"
        self.adata.uns["path_to_file"] = filename
        self.adata_raw.uns["path_to_file"] = filename

    def save_anndata(self, fname: str = "", save_knn: bool = False, **kwargs: Any) -> None:
        """Save adata to an h5ad file.

        Parameters
        ----------
        fname : str, optional
            Output file path. If empty, uses path from adata.uns['path_to_file'].
        save_knn : bool, optional
            If True, include X_knn_avg layer. Default is False (layer can be large).
        **kwargs
            Additional arguments passed to AnnData.write_h5ad().
        """
        Xknn = None
        if not save_knn and "X_knn_avg" in self.adata.layers:
            Xknn = self.adata.layers["X_knn_avg"]
            del self.adata.layers["X_knn_avg"]

        if fname == "":
            if "path_to_file" not in self.adata.uns:
                raise KeyError("Path to file not known.")
            fname = self.adata.uns["path_to_file"]

        x = self.adata
        x.raw = self.adata_raw

        # Fix weird issues when index name is an integer
        for y in [
            x.obs.columns,
            x.var.columns,
            x.obs.index,
            x.var.index,
            x.raw.var.index,
            x.raw.var.columns,
        ]:
            y.name = str(y.name) if y.name is not None else None

        x.write_h5ad(fname, **kwargs)
        del x.raw

        if Xknn is not None:
            self.adata.layers["X_knn_avg"] = Xknn

    def dispersion_ranking_NN(
        self,
        nnm: sp.spmatrix | None = None,
        num_norm_avg: int = 50,
        weight_mode: Literal["dispersion", "variance", "rms", "combined"] = "combined",
        save_avgs: bool = False,
        adata: AnnData | None = None,
    ) -> NDArray[np.float64]:
        """Compute spatial dispersion factors for each gene.

        Parameters
        ----------
        nnm : scipy.sparse.spmatrix | None, optional
            Cell-to-cell nearest-neighbor matrix. If None, uses
            adata.obsp['connectivities'].
        num_norm_avg : int, optional
            Number of top dispersions to average for normalization. Default is 50.
        weight_mode : str, optional
            Weight calculation method. One of 'dispersion', 'variance', 'rms',
            'combined'. Default is 'combined'.
        save_avgs : bool, optional
            If True, save kNN-averaged values to layers['X_knn_avg']. Default is False.
        adata : AnnData | None, optional
            AnnData object to use. If None, uses self.adata.

        Returns
        -------
        NDArray[np.float64]
            Vector of gene weights.
        """
        if adata is None:
            adata = self.adata

        if nnm is None:
            nnm = adata.obsp["connectivities"]
        f = np.asarray(nnm.sum(1))
        f[f == 0] = 1

        bk = self._bk

        # --- SpMM: D_avg = (nnm / row_sums) @ X_disp ----------------------
        # This is the dominant cost of the SAM iteration. On GPU we upload
        # both sparse operands once, do a cuSPARSE SpGEMM, compute column
        # mean/var on the device, and pull back only the (n_genes,) vectors.
        # The rest of the dispersion arithmetic is cheap numpy on host.
        if bk.gpu:
            # Row-normalise nnm before upload so we do one SpGEMM on device.
            nnm_norm = nnm.multiply(1.0 / f)
            nnm_g = bk.to_device(nnm_norm.tocsr())
            Xd_g = bk.to_device(adata.layers["X_disp"].tocsr())
            D_avg_g = nnm_g @ Xd_g  # cuSPARSE sparse-sparse → sparse (n_cells, n_genes)

            xp = bk.xp
            n = D_avg_g.shape[0]
            # Mean: column sums / n. Var: E[x²] - E[x]² (population variance,
            # matching sklearn.sparsefuncs.mean_variance_axis axis=0 ddof=0).
            col_sum = xp.asarray(D_avg_g.sum(axis=0)).ravel()
            mu = col_sum / n
            # E[x²] via squaring the data buffer. We need D_avg_g.data² summed
            # per column — reuse the matrix structure.
            D_sq_g = D_avg_g.copy()
            D_sq_g.data = D_sq_g.data**2
            ex2 = xp.asarray(D_sq_g.sum(axis=0)).ravel() / n
            var = ex2 - mu**2

            mu2 = None
            if weight_mode in ("rms", "combined"):
                # RMS = sqrt(E[x²]) — we already have ex2.
                mu_rms = xp.sqrt(ex2)
                if weight_mode == "rms":
                    mu = mu_rms
                else:  # combined
                    mu2 = mu_rms

            mu = bk.to_host(mu)
            var = bk.to_host(var)
            if mu2 is not None:
                mu2 = bk.to_host(mu2)

            if save_avgs:
                adata.layers["X_knn_avg"] = bk.to_host(D_avg_g)
            del nnm_g, Xd_g, D_avg_g, D_sq_g
            bk.free_pool()

        else:
            # --- CPU path (original) --------------------------------------
            D_avg = (nnm.multiply(1 / f)).dot(adata.layers["X_disp"])

            if save_avgs:
                adata.layers["X_knn_avg"] = D_avg.copy()

            if sp.issparse(D_avg):
                mu, var = sf.mean_variance_axis(D_avg, axis=0)
                if weight_mode == "rms":
                    D_avg.data[:] = D_avg.data**2
                    mu, _ = sf.mean_variance_axis(D_avg, axis=0)
                    mu = mu**0.5

                if weight_mode == "combined":
                    D_avg.data[:] = D_avg.data**2
                    mu2, _ = sf.mean_variance_axis(D_avg, axis=0)
                    mu2 = mu2**0.5
            else:
                mu = D_avg.mean(0)
                var = D_avg.var(0)
                if weight_mode == "rms":
                    mu = (D_avg**2).mean(0) ** 0.5
                if weight_mode == "combined":
                    mu2 = (D_avg**2).mean(0) ** 0.5

            if not save_avgs:
                del D_avg
                gc.collect()

        if weight_mode in ("dispersion", "rms", "combined"):
            dispersions = np.zeros(var.size)
            dispersions[mu > 0] = var[mu > 0] / mu[mu > 0]
            adata.var["spatial_dispersions"] = dispersions.copy()

            if weight_mode == "combined":
                dispersions2 = np.zeros(var.size)
                dispersions2[mu2 > 0] = var[mu2 > 0] / mu2[mu2 > 0]

        elif weight_mode == "variance":
            dispersions = var
            adata.var["spatial_variances"] = dispersions.copy()
        else:
            raise InvalidParameterError(
                "weight_mode",
                weight_mode,
                valid_values=["dispersion", "variance", "rms", "combined"],
            )

        ma = np.sort(dispersions)[-num_norm_avg:].mean()
        dispersions[dispersions >= ma] = ma

        weights = ((dispersions / dispersions.max()) ** 0.5).flatten()

        if weight_mode == "combined":
            ma = np.sort(dispersions2)[-num_norm_avg:].mean()
            dispersions2[dispersions2 >= ma] = ma

            weights2 = ((dispersions2 / dispersions2.max()) ** 0.5).flatten()
            weights = np.vstack((weights, weights2)).max(0)

        return weights

    def run(
        self,
        max_iter: int = 10,
        verbose: bool = True,
        projection: str | None = None,
        stopping_condition: float = 1e-2,
        num_norm_avg: int = 50,
        k: int = 20,
        distance: Literal["correlation", "euclidean", "cosine"] = "cosine",
        preprocessing: Literal["StandardScaler", "Normalizer"] | None = "StandardScaler",
        npcs: int = 150,
        n_genes: int | None = 3000,
        weight_PCs: bool = False,
        sparse_pca: bool = False,
        proj_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        weight_mode: Literal["dispersion", "variance", "rms", "combined"] = "rms",
        components: NDArray[np.floating[Any]] | None = None,
        batch_key: str | None = None,
    ) -> None:
        """Run the Self-Assembling Manifold algorithm.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations. Default is 10.
        verbose : bool, optional
            If True, print progress. Default is True.
        projection : str | None, optional
            Projection method. In this vendored version, projections are not
            computed; pass None (default). A non-None value logs a warning.
        stopping_condition : float, optional
            RMSE threshold for convergence. Default is 1e-2.
        num_norm_avg : int, optional
            Top dispersions to average for normalization. Default is 50.
        k : int, optional
            Number of nearest neighbors. Default is 20.
        distance : str, optional
            Distance metric: 'correlation', 'euclidean', 'cosine'. Default is 'cosine'.
        preprocessing : str | None, optional
            Preprocessing method: 'StandardScaler', 'Normalizer', None.
            Default is 'StandardScaler'.
        npcs : int, optional
            Number of principal components. Default is 150.
        n_genes : int | None, optional
            Number of genes to use. Default is 3000. If None, uses all genes.
        weight_PCs : bool, optional
            Weight PCs by eigenvalues. Default is False.
        sparse_pca : bool, optional
            Use sparse PCA implementation. Default is False.
        proj_kwargs : dict | None, optional
            Unused in vendored version. Kept for signature compatibility.
        seed : int, optional
            Random seed. Default is 0.
        weight_mode : str, optional
            Weight calculation mode. Default is 'rms'.
        components : NDArray | None, optional
            Pre-computed PCA components. Default is None.
        batch_key : str | None, optional
            Key in obs for batch correction with Harmony. Default is None.
        """
        if proj_kwargs is None:
            proj_kwargs = {}

        D = self.adata.X
        if k < 5:
            k = 5
        if k > D.shape[0] - 1:
            k = D.shape[0] - 2

        if preprocessing not in ("StandardScaler", "Normalizer", None, "None"):
            raise InvalidParameterError(
                "preprocessing",
                preprocessing,
                valid_values=["StandardScaler", "Normalizer", None],
            )
        if weight_mode not in ("dispersion", "variance", "rms", "combined"):
            raise InvalidParameterError(
                "weight_mode",
                weight_mode,
                valid_values=["dispersion", "variance", "rms", "combined"],
            )

        if self.adata.layers["X_disp"].min() < 0 and weight_mode == "dispersion":
            logger.warning(
                "`X_disp` layer contains negative values. Setting `weight_mode` to 'rms'."
            )
            weight_mode = "rms"

        numcells = D.shape[0]

        if n_genes is None:
            n_genes = self.adata.shape[1]
            if not sparse_pca and numcells > 10000:
                warnings.warn(
                    "All genes are being used. It is recommended "
                    "to set `sparse_pca=True` to satisfy memory "
                    "constraints for datasets with more than "
                    "10,000 cells. Setting `sparse_pca` to True.",
                    stacklevel=2,
                )
                sparse_pca = True

        if not sparse_pca:
            n_genes = min(n_genes, (D.sum(0) > 0).sum())

        self.run_args = {
            "max_iter": max_iter,
            "verbose": verbose,
            "projection": projection,
            "stopping_condition": stopping_condition,
            "num_norm_avg": num_norm_avg,
            "k": k,
            "distance": distance,
            "preprocessing": preprocessing,
            "npcs": npcs,
            "n_genes": n_genes,
            "weight_PCs": weight_PCs,
            "proj_kwargs": proj_kwargs,
            "sparse_pca": sparse_pca,
            "weight_mode": weight_mode,
            "seed": seed,
            "components": components,
        }
        self.adata.uns["run_args"] = self.run_args

        tinit = time.time()
        np.random.seed(seed)

        if verbose:
            logger.info("Running SAM algorithm")

        W = np.ones(D.shape[1])
        self.adata.var["weights"] = W

        old = np.zeros(W.size)
        new = W

        i = 0
        err = ((new - old) ** 2).mean() ** 0.5

        if max_iter < 5:
            max_iter = 5

        nnas = num_norm_avg

        while i < max_iter and err > stopping_condition:
            conv = err
            if verbose:
                logger.info("Iteration: %d, Convergence: %.6f", i, conv)

            i += 1
            old = new
            first = i == 1

            W = self.calculate_nnm(
                batch_key=batch_key,
                n_genes=n_genes,
                preprocessing=preprocessing,
                npcs=npcs,
                num_norm_avg=nnas,
                weight_PCs=weight_PCs,
                sparse_pca=sparse_pca,
                weight_mode=weight_mode,
                seed=seed,
                components=components,
                first=first,
            )
            gc.collect()
            new = W
            err = ((new - old) ** 2).mean() ** 0.5
            self.adata.var["weights"] = W

        all_gene_names = np.array(list(self.adata.var_names))
        indices = np.argsort(-W)
        ranked_genes = all_gene_names[indices]

        self.adata.uns["ranked_genes"] = ranked_genes

        # Projections (umap/tsne/diff_umap) stripped in vendored version.
        # SAMap computes its own projections on the combined manifold.
        if projection is not None:
            logger.warning(
                "projection=%r requested but projection methods are not included "
                "in the vendored SAM. Compute projections separately if needed.",
                projection,
            )

        elapsed = time.time() - tinit
        if verbose:
            logger.info("Elapsed time: %.2f seconds", elapsed)

    def calculate_nnm(
        self,
        adata: AnnData | None = None,
        batch_key: str | None = None,
        g_weighted: NDArray[np.floating[Any]] | None = None,
        n_genes: int = 3000,
        preprocessing: str | None = "StandardScaler",
        npcs: int = 150,
        num_norm_avg: int = 50,
        weight_PCs: bool = False,
        sparse_pca: bool = False,
        update_manifold: bool = True,
        weight_mode: str = "dispersion",
        seed: int = 0,
        components: NDArray[np.floating[Any]] | None = None,
        first: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Calculate nearest neighbor matrix and update weights.

        This is the core iteration step of the SAM algorithm.

        Parameters
        ----------
        adata : AnnData | None
            AnnData object to use.
        batch_key : str | None
            Key for batch correction.
        g_weighted : NDArray | None
            Pre-computed weighted coordinates.
        n_genes : int
            Number of genes to use.
        preprocessing : str | None
            Preprocessing method.
        npcs : int
            Number of PCs.
        num_norm_avg : int
            Normalization averaging.
        weight_PCs : bool
            Weight by eigenvalues.
        sparse_pca : bool
            Use sparse PCA.
        update_manifold : bool
            Update manifold structure.
        weight_mode : str
            Weight calculation mode.
        seed : int
            Random seed.
        components : NDArray | None
            Pre-computed components.
        first : bool
            Is this the first iteration.

        Returns
        -------
        NDArray | tuple
            Gene weights, or (PCs, weighted_coords) if not updating manifold.
        """
        if adata is None:
            adata = self.adata

        numcells = adata.shape[0]
        k = adata.uns["run_args"].get("k", 20)
        distance = adata.uns["run_args"].get("distance", "correlation")

        D = adata.X
        W = adata.var["weights"].values

        if "means" not in adata.var or "variances" not in adata.var:
            self.calculate_mean_var(adata)

        if n_genes is None:
            gkeep = np.arange(W.size)
        else:
            if first:
                mu = np.array(list(adata.var["means"]))
                var = np.array(list(adata.var["variances"]))
                mu[mu == 0] = 1
                dispersions = var / mu
                gkeep = np.sort(np.argsort(-dispersions)[:n_genes])
            else:
                gkeep = np.sort(np.argsort(-W)[:n_genes])

        if g_weighted is None:
            if preprocessing == "Normalizer":
                Ds = D[:, gkeep]
                if sp.issparse(Ds) and not sparse_pca:
                    Ds = Ds.toarray()

                Ds = Normalizer().fit_transform(Ds)

            elif preprocessing == "StandardScaler":
                if not sparse_pca:
                    Ds = D[:, gkeep]
                    if sp.issparse(Ds):
                        Ds = Ds.toarray()

                    v = adata.var["variances"].values[gkeep]
                    m = adata.var["means"].values[gkeep]
                    v[v == 0] = 1
                    Ds = (Ds - m) / v**0.5

                    Ds[Ds > 10] = 10
                    Ds[Ds < -10] = -10
                else:
                    Ds = D[:, gkeep]
                    v = adata.var["variances"].values[gkeep]
                    v[v == 0] = 1
                    Ds = Ds.multiply(1 / v**0.5).tocsr()

            else:
                Ds = D[:, gkeep].toarray()

            D_sub = Ds.multiply(W[gkeep]).tocsr() if sp.issparse(Ds) else Ds * W[gkeep]

            if components is None:
                if not sparse_pca:
                    npcs = min(npcs, min((D.shape[0], gkeep.size)))
                    if numcells > 500:
                        g_weighted, pca = weighted_PCA(
                            D_sub,
                            npcs=npcs,
                            do_weight=weight_PCs,
                            solver="auto",
                            seed=seed,
                        )
                    else:
                        g_weighted, pca = weighted_PCA(
                            D_sub,
                            npcs=npcs,
                            do_weight=weight_PCs,
                            solver="full",
                            seed=seed,
                        )
                    components = pca.components_

                else:
                    npcs = min(npcs, min((D.shape[0], gkeep.size)) - 1)
                    v = adata.var["variances"].values[gkeep]
                    v[v == 0] = 1
                    m = adata.var["means"].values[gkeep] * W[gkeep]
                    if preprocessing == "StandardScaler":
                        no = m / v**0.5
                    else:
                        no = np.asarray(D_sub.mean(0)).flatten()
                    mean_correction = no
                    output = _pca_with_sparse(D_sub, npcs, mu=(no)[None, :], seed=seed, bk=self._bk)
                    components = output["components"]
                    g_weighted = output["X_pca"]

                    if weight_PCs:
                        ev = output["variance"]
                        ev = ev / ev.max()
                        g_weighted = g_weighted * (ev**0.5)
            else:
                components = components[:, gkeep]
                v = adata.var["variances"].values[gkeep]
                v[v == 0] = 1
                m = adata.var["means"].values[gkeep] * W[gkeep]
                if preprocessing == "StandardScaler":
                    ns = m / v**0.5
                else:
                    ns = np.asarray(D_sub.mean(0)).flatten()
                mean_correction = ns

                if sp.issparse(D_sub):
                    g_weighted = D_sub.dot(components.T) - ns.flatten().dot(components.T)
                else:
                    g_weighted = (D_sub - ns).dot(components.T)
                if weight_PCs:
                    ev = g_weighted.var(0)
                    ev = ev / ev.max()
                    g_weighted = g_weighted * (ev**0.5)

            adata.varm["PCs"] = np.zeros(shape=(adata.n_vars, npcs))
            adata.varm["PCs"][gkeep] = components.T
            # NOTE: original SAM stored D_sub in obsm["X_processed"] here —
            # an (n_cells x n_genes) matrix written every iteration and never
            # read back. Dropped during vendoring.
            adata.uns["dimred_indices"] = gkeep
            if sparse_pca:
                mc = np.zeros(adata.shape[1])
                mc[gkeep] = mean_correction
                adata.var["mean_correction"] = mc

        if batch_key is not None:
            try:
                import harmonypy

                harmony_out = harmonypy.run_harmony(g_weighted, adata.obs, batch_key, verbose=False)
                g_weighted = harmony_out.Z_corr.T
            except ImportError as err:
                raise ImportError(
                    "harmonypy is required for batch correction. "
                    "Install it with: pip install harmonypy"
                ) from err

        if update_manifold:
            edm = calc_nnm(g_weighted, k, distance, bk=self._bk)

            # Distances matrix: zero out self-distances on the diagonal.
            edm_dist = edm.copy()
            _csr_setdiag(edm_dist, 0)
            adata.obsp["distances"] = edm_dist

            # Connectivities: binary adjacency with self-loops.
            EDM = edm.copy()
            EDM.data[:] = 1
            _csr_setdiag(EDM, 1)
            adata.obsp["connectivities"] = EDM

            # nnm: similarity-weighted adjacency for correlation/cosine.
            if distance in ("correlation", "cosine"):
                edm.data[:] = 1 - edm.data
                _csr_setdiag(edm, 1)
                edm.data[edm.data < 0] = 0.001
                adata.obsp["nnm"] = edm
            else:
                adata.obsp["nnm"] = EDM

            W = self.dispersion_ranking_NN(
                EDM, weight_mode=weight_mode, num_norm_avg=num_norm_avg, adata=adata
            )
            adata.obsm["X_pca"] = g_weighted
            return W
        else:
            logger.info("Not updating the manifold...")
            PCs = np.zeros(shape=(adata.n_vars, npcs))
            PCs[gkeep] = components.T
            return PCs, g_weighted

    def leiden_clustering(
        self,
        X: sp.spmatrix | None = None,
        res: float = 1,
        method: Literal["modularity", "significance"] = "modularity",
        seed: int = 0,
    ) -> NDArray[np.int64] | None:
        """Perform Leiden clustering.

        On a CUDA backend with rapids-singlecell installed, dispatches to
        the cugraph-backed GPU implementation for the common case
        (``X=None``, ``method='modularity'``). Otherwise uses CPU
        leidenalg/igraph — which also handles custom adjacency matrices
        and the significance-based partition.

        Parameters
        ----------
        X : sparse matrix | None, optional
            Adjacency matrix. If None, uses connectivities.
        res : float, optional
            Resolution parameter. Default is 1.
        method : str, optional
            Optimization method. Default is 'modularity'.
        seed : int, optional
            Random seed. Default is 0.

        Returns
        -------
        NDArray | None
            Cluster labels if X provided, None otherwise.
        """
        # --- GPU fast path ----------------------------------------------------
        # rsc.tl.leiden only handles the modularity partition on an AnnData
        # with pre-computed neighbors. That covers SAM's default invocation
        # (X=None, method='modularity'). Custom-X and significance paths fall
        # through to CPU leidenalg below, which is the only implementation
        # that supports them.
        if X is None and method == "modularity" and self._bk.gpu:
            from samap import _rsc_compat

            if _rsc_compat.HAS_RSC:
                _rsc_compat.leiden(
                    self.adata,
                    self._bk,
                    resolution=res,
                    key_added="leiden_clusters",
                    random_state=seed,
                )
                return None

        if X is None:
            X = self.adata.obsp["connectivities"]
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False

        import igraph as ig
        import leidenalg

        adjacency = X
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = np.asarray(weights).flatten()
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets, strict=False)))
        with contextlib.suppress(ValueError, TypeError):
            g.es["weight"] = weights

        if method == "significance":
            cl = leidenalg.find_partition(g, leidenalg.SignificanceVertexPartition, seed=seed)
        else:
            cl = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed
            )

        if save:
            if method == "modularity":
                self.adata.obs["leiden_clusters"] = pd.Categorical(np.array(cl.membership))
            elif method == "significance":
                self.adata.obs["leiden_sig_clusters"] = pd.Categorical(np.array(cl.membership))
            return None
        return np.array(cl.membership)

    def scatter(
        self,
        projection: str | NDArray[np.floating[Any]] | None = None,
        c: str | NDArray[Any] | None = None,
        colorspec: str | NDArray[Any] | None = None,
        cmap: str = "rainbow",
        linewidth: float = 0.0,
        edgecolor: str = "k",
        axes: Any | None = None,
        colorbar: bool = True,
        s: float = 10,
        **kwargs: Any,
    ) -> Any:
        """Display a scatter plot.

        Parameters
        ----------
        projection : str | NDArray | None, optional
            Key in adata.obsm or 2D coordinates array. Default is UMAP.
        c : str | NDArray | None, optional
            Color data - key in adata.obs or array.
        colorspec : str | NDArray | None, optional
            Direct color specification.
        cmap : str, optional
            Colormap name. Default is 'rainbow'.
        linewidth : float, optional
            Marker edge width. Default is 0.0.
        edgecolor : str, optional
            Marker edge color. Default is 'k'.
        axes : matplotlib.axes.Axes | None, optional
            Existing axes to plot on.
        colorbar : bool, optional
            Whether to show colorbar. Default is True.
        s : float, optional
            Marker size. Default is 10.
        **kwargs
            Additional arguments passed to matplotlib.pyplot.scatter.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed!")
            return None

        if isinstance(projection, str):
            if projection not in self.adata.obsm:
                logger.error("Projection %r not found in adata.obsm", projection)
                return None
            dt = self.adata.obsm[projection]

        elif projection is None:
            if "X_umap" in self.adata.obsm:
                dt = self.adata.obsm["X_umap"]
            elif "X_tsne" in self.adata.obsm:
                dt = self.adata.obsm["X_tsne"]
            else:
                logger.error("No projection found. Pass one via `projection=`.")
                return None
        else:
            dt = projection

        if axes is None:
            plt.figure()
            axes = plt.gca()

        if colorspec is not None:
            axes.scatter(
                dt[:, 0],
                dt[:, 1],
                s=s,
                linewidth=linewidth,
                edgecolor=edgecolor,
                c=colorspec,
                **kwargs,
            )
        elif c is None:
            axes.scatter(
                dt[:, 0],
                dt[:, 1],
                s=s,
                linewidth=linewidth,
                edgecolor=edgecolor,
                **kwargs,
            )
        else:
            if isinstance(c, str):
                with contextlib.suppress(KeyError):
                    c = self.get_labels(c)

            if (isinstance(c[0], (str, np.str_))) and (isinstance(c, (np.ndarray, list))):
                i = convert_annotations(c)
                ui, ai = np.unique(i, return_index=True)
                cax = axes.scatter(
                    dt[:, 0],
                    dt[:, 1],
                    c=i,
                    cmap=cmap,
                    s=s,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    **kwargs,
                )

                if colorbar:
                    cbar = plt.colorbar(cax, ax=axes, ticks=ui)
                    cbar.ax.set_yticklabels(c[ai])
            else:
                if not isinstance(c, (np.ndarray, list)):
                    colorbar = False
                i = c

                scatter_kwargs: dict[str, Any] = {
                    "c": i,
                    "s": s,
                    "linewidth": linewidth,
                    "edgecolor": edgecolor,
                    **kwargs,
                }
                if isinstance(i, np.ndarray) and np.issubdtype(i.dtype, np.number):
                    scatter_kwargs["cmap"] = cmap

                cax = axes.scatter(dt[:, 0], dt[:, 1], **scatter_kwargs)

                if colorbar:
                    plt.colorbar(cax, ax=axes)
        return axes
