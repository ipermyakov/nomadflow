#!/usr/bin/env python3
"""
List AlphaFold DB v4 bulk tar URLs by parsing the EBI FTP directory index.

The JSON metadata cache (e.g. download_metadata.json) from EBI typically lists
current archives (e.g. v6) only; v4 root listings use proteome UP*_v4.tar plus
swissprot_*, mane_* archives and must be scraped from the directory index.

Index: https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

V4_INDEX_URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/"
# Apache directory listing: capture every href; keep only root-level *_v4.tar (UP*, swissprot*, mane_*, …).
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def fetch_index_html(url: str, timeout_sec: float) -> str:
    req = Request(url, headers={"User-Agent": "zenprotein2-afdb512-list/1.0"})
    with urlopen(req, timeout=timeout_sec) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def extract_tar_urls(html: str, base_url: str) -> list[str]:
    """Collect HTTPS URLs for *.tar whose path is .../alphafold/v4/<name>_v4.tar (not subfolders)."""
    found: set[str] = set()
    for m in _HREF_RE.finditer(html):
        raw = m.group(1).strip()
        if not raw or raw.startswith("?"):
            continue
        url = unquote(urljoin(base_url, raw))
        path = urlparse(url).path
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            continue
        parent, name = parts[-2], parts[-1]
        if parent.lower() != "v4":
            continue
        if not name.lower().endswith("_v4.tar"):
            continue
        found.add(url)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=V4_INDEX_URL,
        help="Directory index URL to parse (default: EBI AlphaFold v4).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write one URL per line (default: stdout).",
    )
    args = parser.parse_args()

    base = args.url
    if not base.endswith("/"):
        base = base + "/"

    try:
        html = fetch_index_html(base, args.timeout)
    except URLError as e:
        print(f"error: failed to fetch index: {e}", file=sys.stderr)
        return 1

    urls = extract_tar_urls(html, base)
    if not urls:
        print("error: no *_v4.tar links found (index format changed?)", file=sys.stderr)
        return 1

    out_fp = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        for u in urls:
            out_fp.write(f"{u}\n")
        if args.out:
            print(f"wrote {len(urls)} URLs to {args.out}", file=sys.stderr)
    finally:
        if args.out:
            out_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
