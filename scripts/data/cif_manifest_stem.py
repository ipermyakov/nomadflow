"""Map mmCIF file paths to manifest-style lowercase ids (stem after gzip)."""
from __future__ import annotations

from pathlib import Path


def manifest_stem_from_path(path: Path) -> str | None:
    """Return manifest id stem for a structure file, or None if not a supported mmCIF.

    Matches export_manifest_ids.py compare-dir logic, plus .mmcif.gz.
    """
    suf = path.suffix.lower()
    if suf == ".gz":
        stem = path.stem.lower()
        if stem.endswith(".cif"):
            return stem[:-4]
        if stem.endswith(".mmcif"):
            return stem[:-6]
        return None
    if suf in (".cif", ".mmcif"):
        return path.stem.lower()
    return None
