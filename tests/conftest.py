"""Pytest fixtures for latincy-readers tests."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tesserae_dir(fixtures_dir) -> Path:
    """Path to Tesserae test fixtures."""
    return fixtures_dir / "tesserae"


@pytest.fixture
def sample_tess_file(tesserae_dir) -> Path:
    """Path to sample .tess file."""
    return tesserae_dir / "tesserae.tess"


@pytest.fixture
def tei_dir(fixtures_dir) -> Path:
    """Path to TEI test fixtures."""
    return fixtures_dir / "tei"


@pytest.fixture
def sample_tei_file(tei_dir) -> Path:
    """Path to sample TEI XML file."""
    return tei_dir / "sample.xml"
