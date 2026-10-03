#!/usr/bin/env python3
"""
Fetch individual AlphaFold v4 mmCIF files for ids listed in a file (e.g. missing.txt).

Primary URL pattern (when it still works):
  https://alphafold.ebi.ac.uk/files/AF-<ACCESSION>-F1-model_v4.cif

In practice the EBI ``/files/`` endpoint often returns HTTP 404 for programmatic
access; per-file v4 coordinates are hosted on Google Cloud Public Datasets:
  gs://public-datasets-deepmind-alphafold-v4/AF-<ACCESSION>-F1-model_v4.cif
See: https://github.com/google-deepmind/alphafold/blob/main/afdb/README.md

Use ``--source gcloud`` with the Google Cloud SDK (**``gcloud``** or **``gsutil``**
on ``PATH``), or pass ``--gcs-cli /path/to/gcloud``. Authenticate with
``gcloud auth login`` (or application-default credentials). Plain HTTPS cannot
read this bucket (403 without credentials).

Limitations:
  HTTPS gapfill only works when the server returns 200. For 404, use
  ``--source gcloud`` or bulk FTP tar extraction (``bulk_extract_afdbv4_tars.py``).
  Per-structure recovery from FTP without knowing the tar is not implemented.

No automated FTP fallback is implemented here.

Parallel downloads: pass ``-j N`` (default 8) to run ``N`` worker threads. Each
id uses one HTTPS request or one ``gcloud``/``gsutil`` subprocess; for millions
of ids, ``--max-inflight`` caps queued tasks (default ``max(64, 8*jobs)``).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from functools import partial
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

LOG = logging.getLogger("fetch_missing_afdbv4")


@dataclass(frozen=True)
class _OneIdResult:
    """Outcome of processing a single structure id (used by parallel workers)."""

    ok: bool = False
    skipped: bool = False
    failed_parse: bool = False
    not_found: bool = False
    https_404: bool = False
    detail_url: str | None = None
    fatal: bool = False
    fatal_exc: BaseException | None = None


def _maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def output_cif_path(out_dir: Path, struct_id: str, shard: bool) -> Path:
    sid = struct_id.strip().lower()
    if not shard:
        return out_dir / f"{sid}.cif"
    if "af-" in sid:
        mid = sid[6:8]
    else:
        mid = sid[1:3]
    return out_dir / mid / f"{sid}.cif"


def structure_id_to_url(structure_id: str) -> str | None:
    """Map manifest-style id to EBI /files/ URL.

    Accepts ``af-<acc>-f1`` or ``af-<acc>-f1-model_v4`` (and ``AF-...`` variants).
    """
    raw = structure_id.strip()
    low = raw.lower()
    if low.endswith("-model_v4"):
        raw = raw[: -len("-model_v4")]
        low = raw.lower()
    parts = [p for p in raw.split("-") if p]
    if len(parts) == 3 and parts[0].lower() == "af" and parts[2].lower() == "f1":
        acc = parts[1].upper()
        return f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif"
    if raw.upper().startswith("AF-") and raw.upper().endswith("-F1"):
        segs = low.split("-")
        if len(segs) == 3 and segs[0] == "af" and segs[2] == "f1":
            return f"https://alphafold.ebi.ac.uk/files/AF-{segs[1].upper()}-F1-model_v4.cif"
    return None


GCS_V4_BUCKET = "gs://public-datasets-deepmind-alphafold-v4"


def structure_id_to_gs_uri(structure_id: str) -> str | None:
    """Map manifest id to gs:// URI for model_v4.cif in the public DeepMind bucket."""
    raw = structure_id.strip()
    low = raw.lower()
    if low.endswith("-model_v4"):
        raw = raw[: -len("-model_v4")]
        low = raw.lower()
    parts = [p for p in raw.split("-") if p]
    if len(parts) == 3 and parts[0].lower() == "af" and parts[2].lower() == "f1":
        acc = parts[1].upper()
        return f"{GCS_V4_BUCKET}/AF-{acc}-F1-model_v4.cif"
    if raw.upper().startswith("AF-") and raw.upper().endswith("-F1"):
        segs = low.split("-")
        if len(segs) == 3 and segs[0] == "af" and segs[2] == "f1":
            return f"{GCS_V4_BUCKET}/AF-{segs[1].upper()}-F1-model_v4.cif"
    return None


def gcs_cp_argv(gcs_cli: str | None, gs_uri: str, dest_tmp: Path) -> list[str] | None:
    """Build argv for ``gcloud storage cp`` or ``gsutil cp``. Returns None if no CLI found."""
    if gcs_cli:
        exe = gcs_cli
        base = Path(exe).name.lower()
        if base == "gsutil":
            return [exe, "cp", gs_uri, str(dest_tmp)]
        return [exe, "storage", "cp", gs_uri, str(dest_tmp)]
    exe = shutil.which("gcloud")
    if exe:
        return [exe, "storage", "cp", gs_uri, str(dest_tmp)]
    exe = shutil.which("gsutil")
    if exe:
        return [exe, "cp", gs_uri, str(dest_tmp)]
    return None


def fetch_gcs_cp(gs_uri: str, dest: Path, gcs_cli: str | None) -> bool:
    """Copy one object from GCS using gcloud or gsutil. Returns False if object missing or copy fails."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    argv = gcs_cp_argv(gcs_cli, gs_uri, tmp)
    if argv is None:
        LOG.error(
            "Neither 'gcloud' nor 'gsutil' found on PATH. "
            "See https://cloud.google.com/sdk/docs/install or use --gcs-cli PATH."
        )
        return False
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
        os.replace(tmp, dest)
    except FileNotFoundError:
        LOG.error("Executable not found: %s", argv[0])
        return False
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        if tmp.is_file():
            tmp.unlink()
        LOG.debug("GCS copy failed for %s: %s", gs_uri, err)
        return False
    return True


def fetch_url_to_file(url: str, dest: Path, timeout_sec: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "zenprotein2-afdb512-gapfill/1.0"})
    with urlopen(req, timeout=timeout_sec) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def fetch_with_retries(
    url: str,
    dest: Path,
    *,
    timeout_sec: float,
    max_retries: int,
    backoff_sec: float,
) -> bool:
    """Return True on success. False on 404. Raises on unrecoverable error after retries."""
    attempt = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    while True:
        attempt += 1
        try:
            fetch_url_to_file(url, tmp, timeout_sec=timeout_sec)
            os.replace(tmp, dest)
            return True
        except HTTPError as e:
            if e.code == 404:
                if tmp.is_file():
                    tmp.unlink()
                return False
            if e.code in (429, 500, 502, 503, 504) and attempt <= max_retries:
                wait = backoff_sec * (2 ** (attempt - 1))
                LOG.warning(
                    "HTTP %s for %s, retry in %.1fs (%d/%d)",
                    e.code,
                    url,
                    wait,
                    attempt,
                    max_retries,
                )
                time.sleep(wait)
                continue
            if tmp.is_file():
                tmp.unlink()
            raise
        except URLError as e:
            if attempt <= max_retries:
                wait = backoff_sec * (2 ** (attempt - 1))
                LOG.warning("network error for %s: %s; retry in %.1fs", url, e, wait)
                time.sleep(wait)
                continue
            if tmp.is_file():
                tmp.unlink()
            raise
        except OSError:
            if tmp.is_file():
                tmp.unlink()
            raise


def _process_one_id(
    sid: str,
    *,
    source: str,
    out_dir: Path,
    shard: bool,
    gcs_cli: str | None,
    timeout_sec: float,
    max_retries: int,
    backoff_sec: float,
    sleep_sec: float,
    verbose: bool,
) -> _OneIdResult:
    """Download or skip one id. Safe to call from worker threads."""
    if source == "https":
        loc = structure_id_to_url(sid)
    else:
        loc = structure_id_to_gs_uri(sid)
    if loc is None:
        LOG.warning("skip (unrecognized id format): %s", sid)
        _maybe_sleep(sleep_sec)
        return _OneIdResult(failed_parse=True)

    dest = output_cif_path(out_dir, sid, shard=shard)
    if dest.is_file():
        LOG.debug("skip existing: %s", dest)
        _maybe_sleep(sleep_sec)
        return _OneIdResult(skipped=True)

    try:
        if source == "https":
            got = fetch_with_retries(
                loc,
                dest,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
            )
            if got:
                if verbose:
                    LOG.info("saved %s", dest)
                else:
                    LOG.debug("saved %s", dest)
                _maybe_sleep(sleep_sec)
                return _OneIdResult(ok=True)
            _maybe_sleep(sleep_sec)
            return _OneIdResult(not_found=True, https_404=True, detail_url=loc)
        got = fetch_gcs_cp(loc, dest, gcs_cli)
        if got:
            if verbose:
                LOG.info("saved %s", dest)
            else:
                LOG.debug("saved %s", dest)
            _maybe_sleep(sleep_sec)
            return _OneIdResult(ok=True)
        LOG.warning("not in bucket or gcloud failed: %s", loc)
        _maybe_sleep(sleep_sec)
        return _OneIdResult(not_found=True)
    except (HTTPError, URLError, OSError) as e:
        LOG.error("failed %s: %s", loc, e)
        return _OneIdResult(fatal=True, fatal_exc=e)


def _bounded_parallel_results(
    fn,
    items: list[str],
    *,
    max_workers: int,
    max_inflight: int,
):
    """Run ``fn(item)`` with up to ``max_workers`` threads and at most ``max_inflight`` queued futures."""
    max_inflight = max(max_inflight, max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        pending: dict[concurrent.futures.Future, None] = {}
        it = iter(items)

        def submit_some() -> None:
            while len(pending) < max_inflight:
                try:
                    sid = next(it)
                except StopIteration:
                    break
                fut = ex.submit(fn, sid)
                pending[fut] = None

        submit_some()
        while pending:
            done, _ = concurrent.futures.wait(
                pending.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                del pending[fut]
                yield fut.result()
            submit_some()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--missing-ids",
        type=Path,
        required=True,
        help="Text file with one structure id per line.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write .cif files.",
    )
    parser.add_argument(
        "--shard",
        dest="shard",
        action="store_true",
        default=True,
        help="Shard outputs like bulk_extract (default: on).",
    )
    parser.add_argument(
        "--no-shard",
        dest="shard",
        action="store_false",
        help="Flat out_dir/<id>.cif layout.",
    )
    parser.add_argument(
        "--source",
        choices=("https", "gcloud"),
        default="https",
        help="https: EBI /files/ URL (often 404). gcloud: copy from gs://public-datasets-deepmind-alphafold-v4 (needs gcloud or gsutil on PATH + auth).",
    )
    parser.add_argument(
        "--gcs-cli",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to gcloud or gsutil if not on PATH (e.g. /opt/google-cloud-sdk/bin/gcloud).",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional delay in seconds after each id (rate limit); default 0 (no pause).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (plain iteration).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for transient HTTP 5xx / network errors (not 404).",
    )
    parser.add_argument(
        "--backoff-sec",
        type=float,
        default=2.0,
        help="Initial backoff between retries (exponential).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        metavar="N",
        help="Parallel worker threads (one HTTPS request or one gcloud/gsutil subprocess per id in flight). Default: 8. Use 1 for strictly sequential runs.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        metavar="N",
        help="Cap on queued work items (limits memory vs huge id lists). Default: max(64, jobs*8).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.missing_ids.is_file():
        LOG.error("file not found: %s", args.missing_ids)
        return 1

    if args.source == "gcloud":
        probe_tmp = Path(tempfile.gettempdir()) / "afdb512_gcs_cli_probe.part"
        if gcs_cp_argv(args.gcs_cli, f"{GCS_V4_BUCKET}/AF-P05067-F1-model_v4.cif", probe_tmp) is None:
            LOG.error(
                "Neither 'gcloud' nor 'gsutil' found on PATH.\n"
                "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install\n"
                "Then run: gcloud auth login\n"
                "Or pass --gcs-cli /full/path/to/gcloud (or to gsutil)."
            )
            return 1

    ids: list[str] = []
    with args.missing_ids.open("r", encoding="utf-8") as fp:
        for line in fp:
            s = line.strip()
            if s:
                ids.append(s)

    max_inflight = args.max_inflight
    if max_inflight is None:
        max_inflight = max(64, args.jobs * 8)

    worker = partial(
        _process_one_id,
        source=args.source,
        out_dir=args.out_dir,
        shard=args.shard,
        gcs_cli=args.gcs_cli,
        timeout_sec=args.timeout,
        max_retries=args.max_retries,
        backoff_sec=args.backoff_sec,
        sleep_sec=args.sleep_sec,
        verbose=args.verbose,
    )

    use_pbar = tqdm is not None and not args.no_progress
    pbar = (
        tqdm(total=len(ids), unit="id", desc="fetch_missing", smoothing=0.05)
        if use_pbar
        else None
    )

    ok = skipped = failed_parse = not_found = 0
    https_404_hint = False

    def _apply_one(r: _OneIdResult) -> bool:
        """Update counters from one result. Returns True if the run must abort."""
        nonlocal ok, skipped, failed_parse, not_found, https_404_hint
        if r.fatal:
            return True
        if r.ok:
            ok += 1
        elif r.skipped:
            skipped += 1
        elif r.failed_parse:
            failed_parse += 1
        elif r.not_found:
            not_found += 1
            if r.https_404:
                if not https_404_hint:
                    https_404_hint = True
                    LOG.warning(
                        "HTTPS 404 for %s — alphafold.ebi.ac.uk/files/ often returns 404; "
                        "retry with: --source gcloud (Google Cloud SDK + auth).",
                        r.detail_url or "?",
                    )
                else:
                    LOG.debug("HTTPS 404: %s", r.detail_url)
        if pbar is not None:
            pbar.set_postfix(
                ok=ok,
                skip=skipped,
                miss=not_found,
                bad=failed_parse,
                refresh=False,
            )
            pbar.update(1)
        return False

    try:
        if args.jobs <= 1:
            for sid in ids:
                if _apply_one(worker(sid)):
                    return 1
        else:
            for r in _bounded_parallel_results(
                worker,
                ids,
                max_workers=args.jobs,
                max_inflight=max_inflight,
            ):
                if _apply_one(r):
                    return 1
    finally:
        if pbar is not None:
            pbar.close()

    LOG.info(
        "done: downloaded=%d skipped_existing=%d bad_id=%d not_found_or_failed=%d",
        ok,
        skipped,
        failed_parse,
        not_found,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
