# AFDB512 v4 mmCIF: commands cheat sheet

All paths below are from the **repository root** (`zenprotein2/`).

Prerequisites:

```bash
pip install -r scripts/data/requirements.txt
```

For gap-fill from Google Cloud you also need [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) and `gcloud auth login`.

---

## Default files

| File | Role |
|------|------|
| `data/afdb512_clusters_metadata/afdb_metadata/manifest.json` | Boltz-style manifest (ids) |
| `data/afdb512_manifest_ids.txt` | One id per line (export from manifest) |
| `data/afdb_cifs/` | Extracted `.cif` (sharded for `af-` ids) |
| `data/afdb512_missing_ids.txt` | Ids in manifest but no local `.cif` |
| `data/afdb512_extra_ids.txt` | Local stems not in manifest |
| `data/ccd/ccd.rdb` | Redis dump for the CCD dictionary |
| `data/afdb512_clusters_metadata/afdb_metadata/afdb_clusters.rdb` | Redis dump for AFDB Foldseek clusters |
| `data/afdb_512_clusters_boltz/` | Parsed dataset output for training |

---

## 0. Download metadata and CCD

The AFDB manifest and cluster mapping come from the dataset metadata archive linked in [scripts/data/README.md](../data/README.md). The repository currently documents this as a Box share rather than a stable direct `wget` URL. Download the metadata archive from that link, then unpack or copy it so these files exist:

```text
data/afdb512_clusters_metadata/afdb_metadata/manifest.json
data/afdb512_clusters_metadata/afdb_metadata/clustering.json
data/afdb512_clusters_metadata/afdb_metadata/afdb_clusters.rdb
```

For example, after downloading the archive locally:

```bash
mkdir -p data/afdb512_clusters_metadata

# If the Box download is a .tar/.tar.gz:
tar -xf /path/to/afdb512_clusters_metadata.tar.gz -C data/afdb512_clusters_metadata --strip-components=1

# If the Box download is a .zip:
unzip /path/to/afdb512_clusters_metadata.zip -d data/afdb512_clusters_metadata
```

The preprocessed CCD Redis dump is provided by Boltz:

```bash
mkdir -p data/ccd
wget -O data/ccd/ccd.rdb https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb
```

If you only have `clustering.json` and not `afdb_clusters.rdb`, build the Redis dump once:

**Terminal A — temporary cluster Redis**

```bash
cd data/afdb512_clusters_metadata/afdb_metadata
redis-server --dbfilename afdb_clusters.rdb --port 7778
```

**Terminal B — load JSON into Redis**

```bash
python scripts/data/json_to_rdb.py \
  --dict data/afdb512_clusters_metadata/afdb_metadata/clustering.json \
  --port 7778 \
  --no-pickle-values
```

Then stop Terminal A. Future runs can start Redis directly from `afdb_clusters.rdb`.

---

## 1. Export ids from manifest (once or after manifest change)

```bash
python scripts/data/export_manifest_ids.py \
  --manifest data/afdb512_clusters_metadata/afdb_metadata/manifest.json \
  --out data/afdb512_manifest_ids.txt
```

---

## 2. Download CIFs from GCS

EBI `https://alphafold.ebi.ac.uk/files/...` is often **404**. Prefer Google Cloud Storage against `gs://public-datasets-deepmind-alphafold-v4/`.

For a fresh dataset, use the manifest id list directly. The batch GCS copier is the primary download path because most AFDB512 files come from this bucket:

```bash
python scripts/afdb512_download/batch_fetch_missing_afdbv4_gcs.py \
  --missing-ids data/afdb512_manifest_ids.txt \
  --out-dir data/afdb_cifs
```

### Chunking and parallelism (`batch_fetch_missing_afdbv4_gcs.py`)

| Flag | Default | Role |
|------|---------|------|
| `--chunk-size N` | `50000` | Max `gs://` paths passed on stdin per single `gcloud storage cp -I` invocation. Larger chunks = fewer subprocess calls, larger stdin batches. |
| `-j N` / `--jobs N` | `1` | How many chunk jobs run **concurrently** (each job is one chunk: one `gcloud` batch + file moves). `1` = strictly sequential chunks. |
| `--max-inflight N` | `max(2*j, j+1)` (never below `j`) | Upper bound on **queued** chunk futures ahead of completed work; caps memory when `jobs` > 1. Must be `>= --jobs`. |

Example with wider chunks and four parallel chunk jobs:

```bash
python scripts/afdb512_download/batch_fetch_missing_afdbv4_gcs.py \
  --missing-ids data/afdb512_manifest_ids.txt \
  --out-dir data/afdb_cifs \
  --chunk-size 100000 \
  -j 4 \
  --max-inflight 10
```

Other useful flags (same script): `--dry-run`, `--no-continue-on-error`, `--gcloud-verbose`, `--no-wc-total`, `--no-progress`, `--gcs-cli PATH`, `--shard` / `--no-shard`, `-v` / `--verbose`.

Existing `.cif` targets are skipped automatically, so the same command is safe to re-run.

---

## 3. Refresh `data/afdb512_missing_ids.txt` (and extra list)

After adding or changing files under `data/afdb_cifs/`, regenerate **full** missing/extra lists:

**Option A — wrapper (writes defaults including `data/afdb512_missing_ids.txt`):**

```bash
scripts/afdb512_download/verify_manifest_coverage.sh data/afdb_cifs
```

**Option B — explicit Python (same result, paths configurable):**

```bash
python scripts/data/export_manifest_ids.py \
  --manifest data/afdb512_clusters_metadata/afdb_metadata/manifest.json \
  --compare-dir data/afdb_cifs \
  --write-missing data/afdb512_missing_ids.txt \
  --write-extra data/afdb512_extra_ids.txt
```

**Overrides** (wrapper only): set `MANIFEST`, `COMPARE_DIR`, `WRITE_MISSING`, `WRITE_EXTRA` before calling the shell script.

---

## 4. Gap-fill remaining missing ids from GCS

After coverage refresh, run the same batch copier on the generated missing-id list:

```bash
python scripts/afdb512_download/batch_fetch_missing_afdbv4_gcs.py \
  --missing-ids data/afdb512_missing_ids.txt \
  --out-dir data/afdb_cifs
```

Use the same **chunking / parallelism** flags as in [§2](#2-download-cifs-from-gcs) (`--chunk-size`, `-j` / `--jobs`, `--max-inflight`).

For very small missing lists, per-id GCS is simpler:

```bash
python scripts/afdb512_download/fetch_missing_afdbv4.py \
  --source gcloud \
  --missing-ids data/afdb512_missing_ids.txt \
  --out-dir data/afdb_cifs
```

Parallelism for **per-id** mode (`fetch_missing_afdbv4.py`): `-j N` / `--jobs N` (default `8`) runs that many worker threads (each in-flight id = one HTTPS request or one `gcloud`/`gsutil` subprocess); `--max-inflight N` caps queued work (default `max(64, jobs*8)`). Other flags: `--gcs-cli /path/to/gcloud`, `--sleep-sec …`, `--no-progress`, `-v` / `--verbose`.

After gap-fill, re-run coverage:

```bash
scripts/afdb512_download/verify_manifest_coverage.sh data/afdb_cifs
```

---

## 5. Start Redis for final preprocessing

Run these in separate terminals and keep both Redis servers running while `rcsb.py` runs.

**Terminal 1 — CCD Redis (`7777`)**

```bash
cd data/ccd
redis-server --dbfilename ccd.rdb --port 7777
```

Check from another terminal:

```bash
redis-cli -p 7777 ping
```

**Terminal 2 — AFDB cluster Redis (`7778`)**

```bash
cd data/afdb512_clusters_metadata/afdb_metadata
redis-server --dbfilename afdb_clusters.rdb --port 7778
```

Check from another terminal:

```bash
redis-cli -p 7778 ping
```

---

## 6. Featurize mmCIF for training (`rcsb.py`)

Run from `scripts/data` because `rcsb.py` imports the local `mmcif.py` module:

```bash
cd scripts/data

python rcsb.py \
  --datadir ../../data/afdb_cifs \
  --outdir ../../data/afdb_512_clusters_boltz \
  --ccd-redis-host localhost \
  --ccd-redis-port 7777 \
  --clusters-redis-host localhost \
  --clusters-redis-port 7778 \
  --num-processes "$(nproc)"
```

The output directory should contain:

```text
data/afdb_512_clusters_boltz/manifest.json
data/afdb_512_clusters_boltz/records/
data/afdb_512_clusters_boltz/structures/
```

Re-runs skip structures whose `.npz` and record `.json` already exist, then rebuild `manifest.json` from all records at the end.

---

## Optional: EBI bulk tar path

The older EBI bulk tar flow is not the recommended path for AFDB512 because most required files are fetched from GCS above. If you still want to try it first, use:

```bash
python scripts/afdb512_download/list_v4_tar_urls.py --out data/afdb512_v4_tar_urls.txt

python scripts/afdb512_download/bulk_extract_afdbv4_tars.py \
  --urls-file data/afdb512_v4_tar_urls.txt \
  --manifest-ids data/afdb512_manifest_ids.txt \
  --out-dir data/afdb_cifs \
  --staging-dir data/tar_staging
```

Re-runs skip existing `.tar` (unless `--force-download-tar`) and existing `.cif`.

---

## See also

- [scripts/data/README.md](../data/README.md) — CCD, clustering Redis, full Boltz-style preprocessing.
- [DeepMind AlphaFold DB README](https://github.com/google-deepmind/alphafold/blob/main/afdb/README.md) — bulk GCS layout (`gs://public-datasets-deepmind-alphafold-v4/`).
