"""Unit tests for samap.exceptions module."""

from __future__ import annotations

import pytest

from samap.exceptions import (
    BlastGraphError,
    DataError,
    DependencyError,
    GeneNotFoundError,
    MappingError,
    SAMapError,
    SpeciesNotFoundError,
)


class TestExceptions:
    """Tests for custom exception classes."""

    def test_samap_error_is_base(self) -> None:
        """Test that SAMapError is the base exception."""
        with pytest.raises(SAMapError):
            raise SAMapError("test error")

    def test_blast_graph_error_inherits(self) -> None:
        """Test that BlastGraphError inherits from SAMapError."""
        with pytest.raises(SAMapError):
            raise BlastGraphError("blast error")
        with pytest.raises(BlastGraphError):
            raise BlastGraphError("blast error")

    def test_mapping_error_inherits(self) -> None:
        """Test that MappingError inherits from SAMapError."""
        with pytest.raises(SAMapError):
            raise MappingError("mapping error")

    def test_data_error_inherits(self) -> None:
        """Test that DataError inherits from SAMapError."""
        with pytest.raises(SAMapError):
            raise DataError("data error")

    def test_species_not_found_inherits(self) -> None:
        """Test that SpeciesNotFoundError inherits from SAMapError."""
        with pytest.raises(SAMapError):
            raise SpeciesNotFoundError("species not found")

    def test_gene_not_found_inherits(self) -> None:
        """Test that GeneNotFoundError inherits from SAMapError."""
        with pytest.raises(SAMapError):
            raise GeneNotFoundError("gene not found")

    def test_dependency_error_inherits(self) -> None:
        """Test that DependencyError inherits from SAMapError."""
        with pytest.raises(SAMapError):
            raise DependencyError("dependency missing")

    def test_exception_messages(self) -> None:
        """Test that exception messages are preserved."""
        msg = "custom error message"
        try:
            raise SAMapError(msg)
        except SAMapError as e:
            assert str(e) == msg
