#!/usr/bin/env python3
"""
Download AlphaFold v4 bulk .tar archives and extract mmCIF members that appear in a
manifest id list, writing decompressed .cif files under --out-dir.

Uses streaming-friendly tar iteration where possible. Requires a line-delimited
id file (e.g. data/afdb512_manifest_ids.txt).
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.error import ContentTooShortError, HTTPError, URLError
from urllib.request import Request, urlopen

_SCRIPTS_DATA = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(_SCRIPTS_DATA))
from cif_manifest_stem import manifest_stem_from_path  # noqa: E402

LOG = logging.getLogger("bulk_extract_afdbv4")


def load_manifest_ids(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            s = line.strip()
            if s:
                out.add(s.lower())
    return out


def output_cif_path(out_dir: Path, struct_id: str, shard: bool) -> Path:
    """Match rcsb/datamodule layout when sharding is enabled."""
    sid = struct_id.lower()
    if not shard:
        return out_dir / f"{sid}.cif"
    if "af-" in sid:
        mid = sid[6:8]
    else:
        mid = sid[1:3]
    return out_dir / mid / f"{sid}.cif"


def _url_basename(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def download_url(url: str, dest: Path, timeout_sec: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "zenprotein2-afdb512-bulk/1.0"})
    with urlopen(req, timeout=timeout_sec) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(tmp, dest)


def try_download_url(url: str, dest: Path, timeout_sec: float) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        download_url(url, dest, timeout_sec=timeout_sec)
    except ContentTooShortError:
        if tmp.is_file():
            tmp.unlink()
        raise


def member_text(m: tarfile.TarInfo, tf: tarfile.TarFile) -> str:
    bio = tf.extractfile(m)
    if bio is None:
        raise OSError(f"not a readable file member: {m.name}")
    raw = bio.read()
    name_l = m.name.lower()
    if name_l.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def process_one_tar(
    url: str,
    staging_dir: Path,
    manifest_ids: set[str],
    out_dir: Path,
    shard: bool,
    timeout_sec: float,
    force_download_tar: bool,
) -> tuple[int, int, int]:
    """Returns (written, skipped_existing, skipped_not_in_manifest)."""
    name = _url_basename(url)
    tar_path = staging_dir / name

    if (
        tar_path.is_file()
        and tar_path.stat().st_size > 0
        and not force_download_tar
    ):
        LOG.info("skip download (tar exists): %s", tar_path)
    else:
        LOG.info("download: %s -> %s", url, tar_path)
        try_download_url(url, tar_path, timeout_sec=timeout_sec)

    written = 0
    skipped_existing = 0
    skipped_nomatch = 0

    with tarfile.open(tar_path, mode="r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            rel = Path(m.name)
            stem = manifest_stem_from_path(rel)
            if stem is None:
                continue
            if stem not in manifest_ids:
                skipped_nomatch += 1
                continue
            out_path = output_cif_path(out_dir, stem, shard=shard)
            if out_path.is_file():
                skipped_existing += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            text = member_text(m, tf)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(out_path.parent),
                suffix=".cif.tmp",
            ) as fp:
                fp.write(text)
                tmp_name = fp.name
            try:
                os.replace(tmp_name, out_path)
            except OSError:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise
            written += 1
            if written % 500 == 0:
                LOG.info("progress %s: written %d ...", name, written)

    LOG.info(
        "done %s: written=%d skip_existing=%d member_skipped_not_in_manifest=%d",
        name,
        written,
        skipped_existing,
        skipped_nomatch,
    )
    return written, skipped_existing, skipped_nomatch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urls-file",
        type=Path,
        required=True,
        help="Text file with one tar URL per line.",
    )
    parser.add_argument(
        "--manifest-ids",
        type=Path,
        required=True,
        help="Text file with one structure id per line (lowercased when matched).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for decompressed .cif files.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help="Directory to store downloaded .tar archives.",
    )
    parser.add_argument(
        "--shard",
        dest="shard",
        action="store_true",
        default=True,
        help="Shard outputs like rcsb/datamodule (default: on).",
    )
    parser.add_argument(
        "--no-shard",
        dest="shard",
        action="store_false",
        help="Write flat out_dir/<id>.cif without mid subdirectories.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds for tar download.",
    )
    parser.add_argument(
        "--force-download-tar",
        action="store_true",
        help="Re-download tar even if it already exists in staging.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.urls_file.is_file():
        LOG.error("urls file not found: %s", args.urls_file)
        return 1
    if not args.manifest_ids.is_file():
        LOG.error("manifest ids file not found: %s", args.manifest_ids)
        return 1

    urls = []
    with args.urls_file.open("r", encoding="utf-8") as fp:
        for line in fp:
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)

    if not urls:
        LOG.error("no URLs in %s", args.urls_file)
        return 1

    LOG.info("loading manifest ids from %s ...", args.manifest_ids)
    manifest_ids = load_manifest_ids(args.manifest_ids)
    LOG.info("manifest id count: %d", len(manifest_ids))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    total_w = total_se = total_nm = 0
    for url in urls:
        try:
            w, se, nm = process_one_tar(
                url=url,
                staging_dir=args.staging_dir,
                manifest_ids=manifest_ids,
                out_dir=args.out_dir,
                shard=args.shard,
                timeout_sec=args.timeout,
                force_download_tar=args.force_download_tar,
            )
            total_w += w
            total_se += se
            total_nm += nm
        except (HTTPError, URLError, OSError, tarfile.TarError) as e:
            LOG.exception("failed on URL %s: %s", url, e)
            return 1

    LOG.info(
        "ALL DONE: written=%d skipped_existing=%d skipped_members_not_in_manifest=%d",
        total_w,
        total_se,
        total_nm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
