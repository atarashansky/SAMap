"""Custom exceptions for SAMap."""

from __future__ import annotations


class SAMapError(Exception):
    """Base exception for SAMap errors."""


class BlastGraphError(SAMapError):
    """Error related to BLAST graph construction or processing."""


class MappingError(SAMapError):
    """Error during SAMap mapping process."""


class DataError(SAMapError):
    """Error related to input data format or content."""


class SpeciesNotFoundError(SAMapError):
    """Species ID not found in the SAMAP object."""


class GeneNotFoundError(SAMapError):
    """Gene not found in the dataset."""


class DependencyError(SAMapError):
    """Required dependency is not installed."""
