#!/usr/bin/env bash
# Compare extracted mmCIF coverage against AFDB512 manifest; write full missing/extra lists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${MANIFEST:-$ROOT/data/afdb512_clusters_metadata/afdb_metadata/manifest.json}"
WRITE_MISSING="${WRITE_MISSING:-$ROOT/data/afdb512_missing_ids.txt}"
WRITE_EXTRA="${WRITE_EXTRA:-$ROOT/data/afdb512_extra_ids.txt}"

if [[ $# -ge 1 ]]; then
  COMPARE_DIR="$1"
fi
: "${COMPARE_DIR:?Usage: $0 <COMPARE_DIR>  or  COMPARE_DIR=/path/to/cifs $0}"

exec python "$ROOT/scripts/data/export_manifest_ids.py" \
  --manifest "$MANIFEST" \
  --compare-dir "$COMPARE_DIR" \
  --write-missing "$WRITE_MISSING" \
  --write-extra "$WRITE_EXTRA"
