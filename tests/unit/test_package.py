"""Smoke tests for package metadata and import surface."""

import importlib.metadata
import importlib.resources
import re

import ketjet

SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.dev\d+)?$")


def test_version_is_pep440_semver() -> None:
    assert SEMVER.match(ketjet.__version__)


def test_version_matches_installed_metadata() -> None:
    assert importlib.metadata.version("ketjet") == ketjet.__version__


def test_package_is_typed() -> None:
    # PEP 561 marker must ship inside the package so type checkers use our annotations.
    assert importlib.resources.files("ketjet").joinpath("py.typed").is_file()


def test_public_api_is_explicit() -> None:
    assert "__version__" in ketjet.__all__
