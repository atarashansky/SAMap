"""Unit tests for samap.utils module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from samap.utils import df_to_dict, sparse_knn, substr, to_vn, to_vo


class TestToVn:
    """Tests for to_vn function."""

    def test_basic_conversion(self) -> None:
        """Test basic array to semicolon-separated strings."""
        arr = np.array([["a", "b"], ["c", "d"], ["e", "f"]])
        result = to_vn(arr)
        expected = np.array(["a;b", "c;d", "e;f"])
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self) -> None:
        """Test with empty array."""
        arr = np.array([]).reshape(0, 2)
        result = to_vn(arr)
        assert len(result) == 0

    def test_single_row(self) -> None:
        """Test with single row."""
        arr = np.array([["gene1", "gene2"]])
        result = to_vn(arr)
        expected = np.array(["gene1;gene2"])
        np.testing.assert_array_equal(result, expected)


class TestSubstr:
    """Tests for substr function."""

    def test_split_with_index(self) -> None:
        """Test splitting strings and extracting specific index."""
        strings = np.array(["hu_SOX2", "ms_Oct4", "hu_NANOG"])
        result = substr(strings, "_", 0)
        expected = np.array(["hu", "ms", "hu"])
        np.testing.assert_array_equal(result, expected)

    def test_split_second_index(self) -> None:
        """Test extracting second part after split."""
        strings = np.array(["hu_SOX2", "ms_Oct4", "hu_NANOG"])
        result = substr(strings, "_", 1)
        expected = np.array(["SOX2", "Oct4", "NANOG"])
        np.testing.assert_array_equal(result, expected)

    def test_split_without_index(self) -> None:
        """Test splitting without index returns all parts."""
        strings = np.array(["a_b_c", "d_e_f"])
        result = substr(strings, "_")
        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], np.array(["a", "d"]))
        np.testing.assert_array_equal(result[1], np.array(["b", "e"]))
        np.testing.assert_array_equal(result[2], np.array(["c", "f"]))

    def test_split_with_object_dtype(self) -> None:
        """Test with object dtype output."""
        strings = np.array(["hu_SOX2", "ms_Oct4"])
        result = substr(strings, "_", 0, obj=True)
        assert result.dtype == object

    def test_index_out_of_bounds(self) -> None:
        """Test that out-of-bounds index returns last element."""
        strings = np.array(["hu_SOX2", "ms"])
        result = substr(strings, "_", 5)  # Index beyond available parts
        # Should return last available part
        assert result[0] == "SOX2"
        assert result[1] == "ms"


class TestDfToDict:
    """Tests for df_to_dict function."""

    def test_basic_conversion(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic DataFrame to dict conversion."""
        result = df_to_dict(sample_dataframe, key_key="gene", val_key=["value"])
        assert "geneA" in result
        assert "geneB" in result
        assert "geneC" in result
        np.testing.assert_array_equal(result["geneA"], np.array([1, 2]))
        np.testing.assert_array_equal(result["geneB"], np.array([3]))
        np.testing.assert_array_equal(result["geneC"], np.array([4, 5, 6]))

    def test_with_index_as_key(self) -> None:
        """Test using index as key."""
        df = pd.DataFrame({"col1": [1, 2, 3]}, index=["a", "b", "c"])
        result = df_to_dict(df, key_key=None, val_key=["col1"])
        assert "a" in result
        assert "b" in result
        assert "c" in result


class TestSparseKnn:
    """Tests for sparse_knn function."""

    def test_keeps_top_k(self) -> None:
        """Test that function keeps top-k values per row."""
        # Create a dense matrix and convert to sparse
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=float)
        sparse = sp.csr_matrix(data)

        result = sparse_knn(sparse, k=2)

        # Each row should have at most k non-zero values
        for i in range(result.shape[0]):
            row = result.getrow(i).toarray().flatten()
            assert np.count_nonzero(row) <= 2

    def test_preserves_largest_values(self) -> None:
        """Test that largest values are preserved."""
        data = np.array([[1, 5, 2, 4], [3, 8, 1, 6]], dtype=float)
        sparse = sp.csr_matrix(data)

        result = sparse_knn(sparse, k=2)

        # First row: 5 and 4 should be kept
        row0 = result.getrow(0).toarray().flatten()
        assert 5 in row0
        assert 4 in row0

        # Second row: 8 and 6 should be kept
        row1 = result.getrow(1).toarray().flatten()
        assert 8 in row1
        assert 6 in row1

    def test_handles_sparse_input(self, sample_sparse_matrix: sp.csr_matrix) -> None:
        """Test with actual sparse input."""
        result = sparse_knn(sample_sparse_matrix, k=5)

        # Result should be sparse
        assert sp.issparse(result)

        # Each row should have at most k values
        for i in range(result.shape[0]):
            row = result.getrow(i).toarray().flatten()
            assert np.count_nonzero(row) <= 5


class TestToVo:
    """Tests for to_vo function."""

    def test_converts_back_from_vn(self) -> None:
        """Test that to_vo converts semicolon strings back to pairs."""
        # This test requires samalg to be installed
        pytest.importorskip("samalg")

        vn_strings = np.array(["gene1;gene2", "gene3;gene4"])
        result = to_vo(vn_strings)

        assert result.shape == (2, 2)
        assert result[0, 0] == "gene1"
        assert result[0, 1] == "gene2"
        assert result[1, 0] == "gene3"
        assert result[1, 1] == "gene4"
