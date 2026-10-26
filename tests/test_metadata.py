"""Release metadata must agree with the package version.

A Zenodo DOI is permanent, so a version mismatch between the code,
CITATION.cff and .zenodo.json would be archived for good. These checks run
only from a source checkout (the files are not shipped in the wheel).
"""

import json
import re
from pathlib import Path

import pytest

import terrdiag

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not found (not a source checkout)")
    return path.read_text(encoding="utf-8")


def test_citation_cff_version_matches_package():
    match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", _read("CITATION.cff"), re.MULTILINE)
    assert match, "CITATION.cff has no version field"
    assert match.group(1) == terrdiag.__version__


def test_zenodo_json_version_matches_package():
    meta = json.loads(_read(".zenodo.json"))
    assert meta["version"] == terrdiag.__version__


def test_changelog_lists_current_version():
    assert f"[{terrdiag.__version__}]" in _read("CHANGELOG.md")
