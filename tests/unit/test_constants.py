"""Unit tests for samap._constants module."""

from __future__ import annotations

from samap._constants import (
    DEFAULT_ALIGNMENT_THRESHOLD,
    DEFAULT_CROSS_K,
    DEFAULT_EVAL_THRESHOLD,
    DEFAULT_FILTER_THRESHOLD,
    DEFAULT_NUM_ITERATIONS,
    KOG_TABLE,
    UMAP_MIN_DIST,
    UMAP_SIZE_THRESHOLD,
)


class TestConstants:
    """Tests for constant values."""

    def test_eval_threshold_is_float(self) -> None:
        """Test that eval threshold is a float."""
        assert isinstance(DEFAULT_EVAL_THRESHOLD, float)
        assert DEFAULT_EVAL_THRESHOLD == 1e-6

    def test_filter_threshold_is_float(self) -> None:
        """Test that filter threshold is a float."""
        assert isinstance(DEFAULT_FILTER_THRESHOLD, float)
        assert DEFAULT_FILTER_THRESHOLD == 0.25

    def test_umap_constants(self) -> None:
        """Test UMAP-related constants."""
        assert isinstance(UMAP_MIN_DIST, float)
        assert isinstance(UMAP_SIZE_THRESHOLD, int)
        assert UMAP_SIZE_THRESHOLD == 10000

    def test_default_parameters(self) -> None:
        """Test default algorithm parameters."""
        assert DEFAULT_NUM_ITERATIONS == 3
        assert DEFAULT_CROSS_K == 20
        assert DEFAULT_ALIGNMENT_THRESHOLD == 0.1

    def test_kog_table_completeness(self) -> None:
        """Test that KOG table has expected entries."""
        assert isinstance(KOG_TABLE, dict)
        # Check some known KOG categories
        assert "A" in KOG_TABLE
        assert "J" in KOG_TABLE
        assert "Z" in KOG_TABLE
        # Check content
        assert "RNA processing" in KOG_TABLE["A"]
        assert "Translation" in KOG_TABLE["J"]
        assert "Cytoskeleton" in KOG_TABLE["Z"]
