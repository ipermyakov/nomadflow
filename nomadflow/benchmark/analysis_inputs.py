"""Load benchmark run inputs for motif scaffolding analysis."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from nomadflow.benchmark.success_criteria import expected_group_count

logger = logging.getLogger(__name__)

_ESMFOLD_PMPNN_SAMPLE_RE = re.compile(r"sample=(\d+)")
_CRITERIA_SUFFIX_RE = re.compile(r"(__criteria_v\d+)$")
_REPO_PATH_MARKERS = ("article_results/", "configs/")
_STAGE_UNDERSCORE_RE = re.compile(r"_stage([12])$")
PREFERRED_CRITERIA_VERSIONS = ("v3", "v2")


def resolve_samples_dir(sampling_run: Path) -> Path:
    """Prefer samples_eval/ for downstream MPNN/ESMFold when present."""
    sampling_run = Path(sampling_run)
    samples_eval = sampling_run / "samples_eval"
    if samples_eval.is_dir():
        return samples_eval
    samples = sampling_run / "samples"
    if samples.is_dir():
        return samples
    raise FileNotFoundError(
        f"Neither samples_eval/ nor samples/ found under {sampling_run}"
    )


def prefer_samples_eval_pdb(path: Path | str | None) -> Path | None:
    """If ``path`` is ``samples/X.pdb`` and ``samples_eval/X.pdb`` exists, use eval.

    RMSD/group_rmsd already scores the single scaffold chain in ``samples_eval/``.
    Viz ``samples/`` PDBs may keep a native motif copy on another chain.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    original = Path(text)
    if original.parent.name != "samples":
        return original
    eval_path = original.parent.parent / "samples_eval" / original.name
    if eval_path.is_file():
        return eval_path
    return original


def load_samples_metadata(sampling_run: Path) -> dict[str, Any]:
    """Load merged samples metadata from a sampling run directory."""
    sampling_run = Path(sampling_run)
    root_metadata = sampling_run / "samples_metadata.json"
    if root_metadata.is_file():
        with open(root_metadata, encoding="utf-8") as handle:
            return json.load(handle)

    metadata_dir = sampling_run / "metadata"
    shard_paths = sorted(metadata_dir.glob("samples_metadata_rank*_batch*.json"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No samples_metadata.json or metadata shards found under {sampling_run}"
        )

    merged: dict[str, Any] = {}
    for shard_path in shard_paths:
        with open(shard_path, encoding="utf-8") as handle:
            merged.update(json.load(handle))
    return merged


def build_esmfold_design_path_index(folding_df: pd.DataFrame) -> dict[str, Path]:
    """Map sample name to ESMFold folded PDB path from folding_rmsd.csv."""
    if folding_df.empty or "folded_path" not in folding_df.columns or "name" not in folding_df.columns:
        return {}
    index: dict[str, Path] = {}
    for _, row in folding_df.iterrows():
        folded_path = Path(str(row["folded_path"]))
        if folded_path.is_file():
            index[str(row["name"])] = folded_path
    return index


def _esmfold_structure_label(path: Path) -> str:
    name = path.name
    if "model_name=v_48_020" in name:
        return "one_shot"
    match = _ESMFOLD_PMPNN_SAMPLE_RE.search(name)
    if match is not None:
        return f"pmpnn_sample_{match.group(1)}"
    return path.stem.replace(" ", "_")


def list_esmfold_structure_paths(esmfold_dir: Path, sample_name: str) -> list[tuple[str, Path]]:
    """List all ESMFold PDB variants for one design under esmfold/<sample_name>/."""
    folder = Path(esmfold_dir) / sample_name
    if not folder.is_dir():
        return []
    entries: list[tuple[str, Path]] = []
    for path in sorted(folder.glob("*.pdb")):
        if not path.is_file():
            continue
        entries.append((_esmfold_structure_label(path), path))
    return entries


def load_esmfold_tables(sampling_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load folding_rmsd.csv and sc_rmsd.csv from esmfold/."""
    sampling_run = Path(sampling_run)
    esmfold_dir = sampling_run / "esmfold"
    folding_path = esmfold_dir / "folding_rmsd.csv"
    sc_path = esmfold_dir / "sc_rmsd.csv"
    missing = [path for path in (folding_path, sc_path) if not path.is_file()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing ESMFold RMSD tables under {esmfold_dir}: {missing_str}"
        )
    return pd.read_csv(folding_path), pd.read_csv(sc_path)


def infer_num_samples_per_task(
    samples_metadata: dict[str, Any],
    default: int = 200,
) -> int:
    for metadata in samples_metadata.values():
        num_samples = metadata.get("num_samples")
        if num_samples is not None:
            return int(num_samples)
    return default


_STAGE2_LAYOUT_MARKER = "__len_"


def parent_motif_task_id(
    task_or_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Map a Stage-2 per-sample task id to the parent motif task.

    Preference order:
    1. ``source_stage1_task_name`` from sample/task metadata when present
    2. strip Stage-2 layout suffix at ``__len_`` (``1QJG_ext10__len_100_…`` → ``1QJG_ext10``)
    3. otherwise return the input unchanged (Stage-1 / vanilla)
    """
    if metadata:
        source = metadata.get("source_stage1_task_name")
        if source is not None and str(source).strip():
            return str(source).strip()
    text = str(task_or_name)
    marker_at = text.find(_STAGE2_LAYOUT_MARKER)
    if marker_at > 0:
        return text[:marker_at]
    return text


def task_sample_counts(samples_metadata: dict[str, Any]) -> dict[str, int]:
    """Count samples per analysis task id (parent motif for Stage-2)."""
    counts: dict[str, int] = {}
    for metadata in samples_metadata.values():
        if not isinstance(metadata, dict):
            continue
        raw_name = metadata.get("name")
        if raw_name is None:
            continue
        task = parent_motif_task_id(str(raw_name), metadata)
        counts[task] = counts.get(task, 0) + 1
    return counts


def remap_dataframe_tasks_to_parent(
    df: pd.DataFrame,
    samples_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Rewrite ``task`` column to parent motif ids using sample metadata."""
    if df.empty or "task" not in df.columns:
        return df
    out = df.copy()

    def _remap_row(row: pd.Series) -> str:
        sample_name = row.get("name")
        metadata = None
        if sample_name is not None and sample_name in samples_metadata:
            meta = samples_metadata[sample_name]
            if isinstance(meta, dict):
                metadata = meta
        raw_task = row["task"]
        if metadata is not None and metadata.get("name") is not None:
            return parent_motif_task_id(str(metadata["name"]), metadata)
        return parent_motif_task_id(str(raw_task), metadata)

    out["task"] = out.apply(_remap_row, axis=1)
    return out


def repo_root_from(start: Path | None = None) -> Path:
    """Walk parents until a directory containing article_results/ is found."""
    start = (start or Path.cwd()).resolve()
    if (start / "article_results").is_dir():
        return start
    for parent in start.parents:
        if (parent / "article_results").is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not detect repository root containing article_results/ from {start}"
    )


def rewrite_repo_path(
    path: str | Path | None, *, repo_root: Path | None = None
) -> Path | None:
    """Rewrite known repo-relative prefixes onto ``repo_root`` without stating files.

    Use this for bulk CSV columns (``group_rmsd`` paths). ``remap_repo_path`` still
    stats files because callers that open a PDB need a path that exists locally.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    original = Path(text)
    if repo_root is None:
        try:
            repo_root = repo_root_from()
        except FileNotFoundError:
            return original
    for marker in _REPO_PATH_MARKERS:
        idx = text.find(marker)
        if idx < 0:
            continue
        return Path(repo_root) / text[idx:]
    return original


def remap_path_column(
    values: pd.Series, *, repo_root: Path | None = None
) -> pd.Series:
    """Rewrite path strings in a Series without per-row filesystem stats."""
    if values.empty:
        return values
    mapped: dict[object, object] = {}
    for value in pd.unique(values.dropna()):
        rewritten = rewrite_repo_path(value, repo_root=repo_root)
        mapped[value] = str(rewritten) if rewritten is not None else value
    return values.map(lambda value: mapped.get(value, value))


def _existing_samples_eval_pdb(path: Path) -> Path | None:
    if path.parent.name != "samples":
        return None
    preferred = prefer_samples_eval_pdb(path)
    if preferred is not None and preferred != path:
        return preferred
    return None


def remap_repo_path(path: str | Path | None, *, repo_root: Path | None = None) -> Path | None:
    """Return an existing path, rewriting server prefixes onto this repo when needed."""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    original = Path(text)
    if original.exists():
        return original
    preferred = _existing_samples_eval_pdb(original)
    if preferred is not None:
        return preferred
    rewritten = rewrite_repo_path(original, repo_root=repo_root)
    if rewritten is None or rewritten == original:
        return original
    if rewritten.exists():
        return rewritten
    preferred = _existing_samples_eval_pdb(rewritten)
    if preferred is not None:
        return preferred
    return original


def strip_criteria_version_suffix(run_id: str) -> str:
    """Drop ``__criteria_v2`` / ``__criteria_v3`` / … from an analysis run id."""
    return _CRITERIA_SUFFIX_RE.sub("", str(run_id))


def criteria_version_from_run_id(run_id: str) -> str | None:
    """Return ``v2`` / ``v3`` / … encoded in the directory name, else ``None``."""
    match = _CRITERIA_SUFFIX_RE.search(str(run_id))
    if match is None:
        return None
    return match.group(1).removeprefix("__criteria_")


def criteria_dir_name(run_id: str, version: str) -> str:
    base = strip_criteria_version_suffix(run_id)
    ver = str(version).strip()
    if ver.startswith("v"):
        return f"{base}__criteria_{ver}"
    return f"{base}__criteria_v{ver}"


def sampling_run_id_from_analysis_id(analysis_id: str) -> str:
    """Map analysis dir names (stage1 / __criteria_v2) to sampling_runs ids (stage_1)."""
    name = strip_criteria_version_suffix(str(analysis_id))
    return _STAGE_UNDERSCORE_RE.sub(r"_stage_\1", name)


def resolve_sampling_run_path(
    *,
    manifest_sampling_run: str | Path | None,
    analysis_run_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Resolve a sampling-run directory from a manifest path, with local fallbacks."""
    if repo_root is None and analysis_run_dir is not None:
        try:
            repo_root = repo_root_from(Path(analysis_run_dir))
        except FileNotFoundError:
            repo_root = None
    if repo_root is None:
        try:
            repo_root = repo_root_from()
        except FileNotFoundError:
            repo_root = None

    if manifest_sampling_run:
        remapped = remap_repo_path(manifest_sampling_run, repo_root=repo_root)
        if remapped is not None and remapped.exists():
            return remapped

    if analysis_run_dir is not None and repo_root is not None:
        sampling_id = sampling_run_id_from_analysis_id(Path(analysis_run_dir).name)
        fallback = repo_root / "article_results" / "sampling_runs" / sampling_id
        if fallback.exists():
            return fallback
    return None


def remap_samples_metadata_paths(
    samples_metadata: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Rewrite server-absolute path/native_pdb fields onto this repo when they exist locally."""
    remapped: dict[str, Any] = {}
    for sample_name, metadata in samples_metadata.items():
        if not isinstance(metadata, dict):
            remapped[sample_name] = metadata
            continue
        entry = dict(metadata)
        for key in ("path", "viz_path", "native_pdb"):
            if key not in entry:
                continue
            new_path = remap_repo_path(entry[key], repo_root=repo_root)
            if new_path is not None:
                entry[key] = str(new_path)
        if "path" in entry:
            preferred = prefer_samples_eval_pdb(entry["path"])
            if preferred is not None:
                entry["path"] = str(preferred)
        remapped[sample_name] = entry
    return remapped


def expected_group_counts(samples_metadata: dict[str, Any]) -> dict[str, int]:
    """Map sample name → annotated motif group count."""
    return {
        sample_name: expected_group_count(metadata)
        for sample_name, metadata in samples_metadata.items()
    }


def load_group_rmsd_table(
    sampling_run: Path,
    *,
    samples_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load esmfold/group_rmsd.csv and remap task ids to parent motifs."""
    sampling_run = Path(sampling_run)
    path = sampling_run / "esmfold" / "group_rmsd.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if samples_metadata is not None:
        df = remap_dataframe_tasks_to_parent(df, samples_metadata)
    return df


NATIVE_GROUP_CSV = "group_fold_vs_native.csv"


def load_native_group_table(
    analysis_out: Path,
    *,
    samples_metadata: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Load fold-vs-native group RMSDs for publication ``pass_final``.

    Prefers ``analysis_out/group_fold_vs_native.csv``, else the sibling
    ``article_results/analysis_runs/<id>__criteria_v3/group_fold_vs_native.csv``.
    Missing file → empty frame (callers must fail ``pass_all_groups``, not fall
    back to fold-vs-design).
    """
    analysis_out = Path(analysis_out)
    candidates = [analysis_out / NATIVE_GROUP_CSV]
    if repo_root is None:
        try:
            repo_root = repo_root_from(analysis_out)
        except FileNotFoundError:
            repo_root = None
    if repo_root is not None:
        v3_dir = (
            Path(repo_root)
            / "article_results"
            / "analysis_runs"
            / criteria_dir_name(analysis_out.name, "v3")
        )
        v3_csv = v3_dir / NATIVE_GROUP_CSV
        if v3_csv not in candidates:
            candidates.append(v3_csv)
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if samples_metadata is not None:
        df = remap_dataframe_tasks_to_parent(df, samples_metadata)
    return df
