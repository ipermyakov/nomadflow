#!/usr/bin/env python3
"""
Batch-download AlphaFold v4 mmCIF objects from the DeepMind public GCS bucket
using a small number of ``gcloud storage cp --read-paths-from-stdin`` invocations
instead of one subprocess per structure id.

Reads manifest-style ids (same rules as ``fetch_missing_afdbv4.py`` /
``structure_id_to_gs_uri``), builds ``gs://public-datasets-deepmind-alphafold-v4/...``
URIs, copies each chunk into a temporary directory, then moves files to
``--out-dir`` using the same shard layout and ``{id}.cif`` lowercase names as
``fetch_missing_afdbv4.py`` / ``bulk_extract_afdbv4_tars.py``.

Requires **Google Cloud SDK** ``gcloud`` with the ``storage`` subcommand
(``gsutil`` alone does not support this stdin mode). Authenticate with
``gcloud auth login`` or application-default credentials.

Idempotency: ids whose target ``.cif`` already exists are skipped and never
sent to ``gcloud``.

By default each ``gcloud`` invocation uses ``--no-user-output-enabled`` so the
SDK does not print one ``Copying …`` line per object (which breaks tqdm). Use
``--gcloud-verbose`` to restore that output for debugging.

Progress: phase **read_lines** counts physical lines in ``--missing-ids`` (``wc -l``
sets ``total``). That phase is usually much faster than GCS. After the file is
fully read, if chunk jobs are still running, the bar switches to **gcs_chunks**
(until all pending chunk futures finish). Postfix ``eta_ub`` during the line
phase is pessimistic (see earlier); during **gcs_chunks** it is ``<=`` remaining
chunks times the recent average chunk wall time.

Use ``-j N`` (default 1) to run up to ``N`` chunk transfers concurrently (each chunk
is still one ``gcloud storage cp -I``). Raise ``--max-inflight`` if you need more
than ``max(2*j, j+1)`` chunks queued; watch GCS / CPU / disk limits.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

import fetch_missing_afdbv4 as fm

LOG = logging.getLogger("batch_fetch_missing_afdbv4_gcs")


def _file_line_count_wc(path: Path) -> int | None:
    """Return line count via ``wc -l`` (fast); ``None`` if unavailable."""
    try:
        proc = subprocess.run(
            ["wc", "-l", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        return int(proc.stdout.split()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError) as e:
        LOG.debug("wc -l failed (%s); tqdm will have no total/ETA", e)
        return None


def _format_duration(sec: float) -> str:
    if sec >= 86400:
        return f"{sec / 86400:.1f}d"
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    if sec >= 60:
        return f"{sec / 60:.0f}m"
    return f"{sec:.0f}s"


def _gs_uri_basename(gs_uri: str) -> str:
    return gs_uri.rsplit("/", 1)[-1]


def _resolve_gcloud_exe(gcs_cli: str | None) -> str:
    if gcs_cli:
        p = Path(gcs_cli)
        if not p.is_file():
            raise FileNotFoundError(f"--gcs-cli not found or not a file: {gcs_cli}")
        if p.name.lower() != "gcloud":
            raise ValueError(
                "Batch mode requires the `gcloud` binary (with `storage cp -I`). "
                f"Got {gcs_cli!r}; `gsutil` does not support --read-paths-from-stdin."
            )
        return str(p.resolve())
    exe = shutil.which("gcloud")
    if not exe:
        raise FileNotFoundError(
            "`gcloud` not found on PATH. Install Google Cloud SDK: "
            "https://cloud.google.com/sdk/docs/install "
            "Then run: gcloud auth login"
        )
    return exe


def _gcloud_batch_cp(
    gcloud_exe: str,
    gs_uris: list[str],
    dest_dir: Path,
    *,
    continue_on_error: bool,
    quiet_user_output: bool,
) -> int:
    """Run ``gcloud storage cp -I dest_dir`` with URIs on stdin. Returns process return code."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [gcloud_exe]
    if quiet_user_output:
        argv.append("--no-user-output-enabled")
    argv.extend(["storage", "cp"])
    if continue_on_error:
        argv.append("--continue-on-error")
    argv.extend(["--read-paths-from-stdin", str(dest_dir)])
    proc = subprocess.run(
        argv,
        input="\n".join(gs_uris) + ("\n" if gs_uris else ""),
        text=True,
    )
    if proc.returncode != 0:
        hint = (
            " Re-run with --gcloud-verbose for per-object logs."
            if quiet_user_output
            else " See gcloud output above."
        )
        LOG.warning("gcloud exited %s.%s", proc.returncode, hint)
    return int(proc.returncode)


def _move_downloaded_chunk(
    ids: list[str],
    gs_uris: list[str],
    staging: Path,
    out_dir: Path,
    *,
    shard: bool,
) -> tuple[int, int, int]:
    """Move ``AF-*-F1-model_v4.cif`` from staging into sharded ``{id}.cif`` paths.

    Returns ``(moved_ok, missing_src, dest_exists_skipped)``.
    """
    moved = missing = skipped = 0
    for sid, uri in zip(ids, gs_uris, strict=True):
        src = staging / _gs_uri_basename(uri)
        dest = fm.output_cif_path(out_dir, sid, shard=shard)
        if dest.is_file():
            skipped += 1
            if src.is_file():
                src.unlink()
            continue
        if not src.is_file():
            missing += 1
            LOG.debug("expected download missing: %s -> %s", uri, src)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move works across filesystems; os.rename/os.replace does not.
        shutil.move(str(src), str(dest))
        moved += 1
    return moved, missing, skipped


def _run_gcloud_chunk_worker(
    gcloud_exe: str,
    chunk_ids: list[str],
    chunk_uris: list[str],
    out_dir: Path,
    *,
    shard: bool,
    continue_on_error: bool,
    gcloud_verbose: bool,
) -> tuple[int, int, int, float, int]:
    """Download one chunk in a worker thread. Returns (moved, missing, skipped, wall_sec, gcloud_rc)."""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="afdbv4_gcs_batch_") as tmp:
        staging = Path(tmp)
        rc = _gcloud_batch_cp(
            gcloud_exe,
            chunk_uris,
            staging,
            continue_on_error=continue_on_error,
            quiet_user_output=not gcloud_verbose,
        )
        moved, missing, skipped = _move_downloaded_chunk(
            chunk_ids,
            chunk_uris,
            staging,
            out_dir,
            shard=shard,
        )
    wall = time.monotonic() - t0
    if rc != 0 and missing:
        LOG.debug("gcloud rc=%s missing_src=%d (chunk %d ids)", rc, missing, len(chunk_ids))
    return moved, missing, skipped, wall, rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--missing-ids",
        type=Path,
        required=True,
        help="Text file with one structure id per line (manifest style).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for extracted .cif files (e.g. data/afdb_cifs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        metavar="N",
        help="Max gs:// objects per gcloud invocation (stdin size). Default: 50000.",
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
        "--gcs-cli",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to gcloud if not on PATH.",
    )
    parser.add_argument(
        "--no-continue-on-error",
        dest="continue_on_error",
        action="store_false",
        help="Do not pass --continue-on-error to gcloud (fail fast on first error).",
    )
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
        help="Pass --continue-on-error to gcloud (default: on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse ids and print counts only; do not call gcloud or write files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm line scan progress (and chunk postfix).",
    )
    parser.add_argument(
        "--no-wc-total",
        action="store_true",
        help="Do not run wc -l on --missing-ids (tqdm will have no line total / no line-scan ETA).",
    )
    parser.add_argument(
        "--gcloud-verbose",
        action="store_true",
        help="Do not pass gcloud --no-user-output-enabled (prints one line per object; very noisy).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent chunk jobs (each = one gcloud stdin batch + move). Default: 1 (sequential).",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        metavar="N",
        help="Max chunk futures queued (default: max(2*j, j+1), never below j).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.max_inflight is not None and args.max_inflight < args.jobs:
        parser.error("--max-inflight must be >= --jobs")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.jobs > 1 and not args.dry_run:
        _mi = args.max_inflight if args.max_inflight is not None else max(args.jobs * 2, args.jobs + 1)
        _mi = max(_mi, args.jobs)
        LOG.info("parallel chunks: jobs=%d max_inflight=%d", args.jobs, _mi)

    if not args.missing_ids.is_file():
        LOG.error("file not found: %s", args.missing_ids)
        return 1

    try:
        gcloud_exe = _resolve_gcloud_exe(args.gcs_cli)
    except (FileNotFoundError, ValueError) as e:
        LOG.error("%s", e)
        return 1

    file_line_total: int | None = None
    if not args.no_wc_total:
        file_line_total = _file_line_count_wc(args.missing_ids)
        if file_line_total is not None:
            LOG.info("missing-ids file: %d lines (wc -l)", file_line_total)

    total_lines = bad_id = skipped_existing = 0
    chunks = 0
    moved_total = missing_after_dl = 0
    dry_run_would_fetch = 0
    chunk_ids: list[str] = []
    chunk_uris: list[str] = []
    chunk_sid_seen: set[str] = set()
    chunk_wall_times: list[float] = []
    chunks_submitted_total = 0

    use_pbar = tqdm is not None and not args.no_progress
    scan_pbar: tqdm | None = None
    chunk_pbar: tqdm | None = None
    pending: set[concurrent.futures.Future] = set()

    def refresh_progress_postfix() -> None:
        bar = chunk_pbar if chunk_pbar is not None else scan_pbar
        if bar is None:
            return
        postfix: dict[str, object] = {
            "ch": chunks,
            "ok": moved_total,
            "gcs_miss": missing_after_dl,
        }
        if not args.dry_run:
            postfix["q"] = len(pending)
        if chunk_wall_times:
            if chunk_pbar is not None and chunks_submitted_total > 0:
                rem_c = chunks_submitted_total - chunks
                if rem_c > 0:
                    avg = sum(chunk_wall_times) / len(chunk_wall_times)
                    postfix["eta_ub"] = "<= " + _format_duration(rem_c * avg)
            elif (
                scan_pbar is not None
                and file_line_total is not None
                and scan_pbar.n < file_line_total
            ):
                rem_lines = file_line_total - scan_pbar.n
                if rem_lines > 0:
                    chunks_ub = (rem_lines + args.chunk_size - 1) // args.chunk_size
                    avg = sum(chunk_wall_times) / len(chunk_wall_times)
                    postfix["eta_ub"] = "<= " + _format_duration(chunks_ub * avg)
        bar.set_postfix(postfix, refresh=False)

    def _apply_chunk_result(
        moved: int, missing: int, _sk: int, wall: float, rc: int
    ) -> None:
        nonlocal chunks, moved_total, missing_after_dl
        chunks += 1
        moved_total += moved
        missing_after_dl += missing
        chunk_wall_times.append(wall)
        while len(chunk_wall_times) > 8:
            chunk_wall_times.pop(0)
        if rc != 0 and missing:
            LOG.debug("after chunk %s: gcloud rc=%s missing_src=%d", chunks, rc, missing)
        refresh_progress_postfix()
        if chunk_pbar is not None:
            chunk_pbar.update(1)

    def _drain_one_future(*, block: bool) -> None:
        if not pending:
            return
        if block:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
        else:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=0,
            )
        if not done:
            return
        for fut in done:
            pending.discard(fut)
            _apply_chunk_result(*fut.result())

    def dispatch_chunk(executor: concurrent.futures.ThreadPoolExecutor | None) -> None:
        nonlocal chunks, chunk_ids, chunk_uris, dry_run_would_fetch, chunks_submitted_total
        if not chunk_ids:
            return
        if args.dry_run:
            chunks += 1
            dry_run_would_fetch += len(chunk_uris)
            LOG.info(
                "dry-run: would run gcloud chunk %d (%d objects)",
                chunks,
                len(chunk_uris),
            )
            chunk_ids.clear()
            chunk_uris.clear()
            chunk_sid_seen.clear()
            refresh_progress_postfix()
            return
        assert executor is not None
        ids_copy = list(chunk_ids)
        uris_copy = list(chunk_uris)
        chunk_ids.clear()
        chunk_uris.clear()
        chunk_sid_seen.clear()
        max_inflight = args.max_inflight
        if max_inflight is None:
            max_inflight = max(args.jobs * 2, args.jobs + 1)
        max_inflight = max(max_inflight, args.jobs)
        while len(pending) >= max_inflight:
            _drain_one_future(block=True)
        fut = executor.submit(
            _run_gcloud_chunk_worker,
            gcloud_exe,
            ids_copy,
            uris_copy,
            args.out_dir,
            shard=args.shard,
            continue_on_error=args.continue_on_error,
            gcloud_verbose=args.gcloud_verbose,
        )
        pending.add(fut)
        chunks_submitted_total += 1
        refresh_progress_postfix()

    executor: concurrent.futures.ThreadPoolExecutor | None = None
    try:
        if not args.dry_run:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
        try:
            with args.missing_ids.open("r", encoding="utf-8") as fp:
                if use_pbar:
                    if file_line_total is not None:
                        scan_pbar = tqdm(
                            fp,
                            total=file_line_total,
                            unit="lines",
                            desc="read_lines",
                            dynamic_ncols=True,
                            smoothing=0.08,
                            bar_format=(
                                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                "[{elapsed}, {rate_fmt}{postfix}]"
                            ),
                        )
                    else:
                        scan_pbar = tqdm(
                            fp,
                            unit="lines",
                            desc="read_lines",
                            dynamic_ncols=True,
                            smoothing=0.08,
                        )
                    line_iter = scan_pbar
                else:
                    line_iter = fp
                for line in line_iter:
                    sid = line.strip()
                    if not sid:
                        continue
                    total_lines += 1
                    dest = fm.output_cif_path(args.out_dir, sid, shard=args.shard)
                    if dest.is_file():
                        skipped_existing += 1
                        continue
                    uri = fm.structure_id_to_gs_uri(sid)
                    if uri is None:
                        bad_id += 1
                        LOG.warning("skip (unrecognized id format): %s", sid)
                        continue
                    dedupe_key = sid.strip().lower()
                    if dedupe_key in chunk_sid_seen:
                        continue
                    chunk_sid_seen.add(dedupe_key)
                    chunk_ids.append(sid)
                    chunk_uris.append(uri)
                    if len(chunk_ids) >= args.chunk_size:
                        dispatch_chunk(executor)

                dispatch_chunk(executor)
            if not args.dry_run and chunks_submitted_total > 0:
                LOG.info(
                    "Finished reading missing-id lines: chunk jobs submitted=%d "
                    "completed=%d still_pending=%d (read_lines bar is only file I/O)",
                    chunks_submitted_total,
                    chunks,
                    len(pending),
                )
            if use_pbar and tqdm is not None and not args.dry_run and pending:
                if scan_pbar is not None:
                    scan_pbar.close()
                    scan_pbar = None
                chunk_pbar = tqdm(
                    total=chunks_submitted_total,
                    initial=chunks,
                    unit="chunk",
                    desc="gcs_chunks",
                    dynamic_ncols=True,
                    smoothing=0.05,
                    bar_format=(
                        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}, {rate_fmt}{postfix}]"
                    ),
                )
                refresh_progress_postfix()
            if executor is not None:
                while pending:
                    _drain_one_future(block=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
    finally:
        if chunk_pbar is not None:
            chunk_pbar.close()
            chunk_pbar = None
        if scan_pbar is not None:
            scan_pbar.close()

    LOG.info(
        "done: lines=%d skipped_existing=%d bad_id=%d chunks=%d chunks_submitted=%d "
        "moved_to_out=%d missing_object_after_gcloud=%d dry_run=%s dry_run_objects=%d",
        total_lines,
        skipped_existing,
        bad_id,
        chunks,
        chunks_submitted_total,
        moved_total,
        missing_after_dl,
        args.dry_run,
        dry_run_would_fetch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
