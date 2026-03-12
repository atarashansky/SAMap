"""Pytest config for regression tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --regenerate-golden CLI flag."""
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help="Regenerate golden fixture(s) instead of comparing against them.",
    )


@pytest.fixture
def regenerate_golden(request: pytest.FixtureRequest) -> bool:
    """True when the user asked to regenerate golden fixtures."""
    return request.config.getoption("--regenerate-golden")
