"""
Modified from Boltz
https://github.com/jwohlwend/boltz/blob/main/scripts/process/rcsb.py
"""
import os
import argparse
import json
import multiprocessing
import pickle
import traceback
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Set

import numpy as np
import rdkit
from mmcif import parse_mmcif
from p_tqdm import p_umap
from redis import Redis
from tqdm import tqdm

from nomadflow.boltz.data.filter.static.filter import StaticFilter
from nomadflow.boltz.data.filter.static.ligand import ExcludedLigands
from nomadflow.boltz.data.filter.static.polymer import (
    ClashingChainsFilter,
    ConsecutiveCA,
    MinimumLengthFilter,
    UnknownFilter,
)
from nomadflow.boltz.data.types import ChainInfo, InterfaceInfo, Record, Target
from nomadflow.boltz.data import const

from cif_manifest_stem import manifest_stem_from_path


@dataclass(frozen=True, slots=True)
class PDB:
    """A raw MMCIF PDB file."""

    id: str
    path: str


class CCDResource:
    """A shared resource for processing."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the redis database."""
        self._redis = Redis(host=host, port=port)

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the Redis database."""
        value = self._redis.get(key)
        if value is not None:
            value = pickle.loads(value)  # noqa: S301
        return value

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the resource."""
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out


class ClusterResource:
    """A shared resource for processing."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the redis database."""
        self._redis = Redis(host=host, port=port)

    def get(self, key: str, default=None) -> Any:  # noqa: ANN401
        """Get an item from the Redis database."""
        value = self._redis.get(key)
        if value is not None:
            return value.decode("utf-8")
        else:
            return default

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the resource."""
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out


def fetch(datadir: Path, max_file_size: Optional[int] = None) -> list[PDB]:
    """Fetch the PDB files."""
    data = []
    excluded = 0
    seen: Set[Path] = set()

    def iter_structure_files():
        for pattern in ("*.cif*", "*.mmcif*"):
            for file in datadir.rglob(pattern):
                if file in seen:
                    continue
                seen.add(file)
                yield file

    for file in tqdm(iter_structure_files(), unit="file"):
        pdb_id = manifest_stem_from_path(file)
        if pdb_id is None:
            continue

        # Check file size and skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue

        # Create the target
        target = PDB(id=pdb_id, path=str(file))
        data.append(target)

    print(f"Excluded {excluded} files due to size.")  # noqa: T201
    return data


def finalize(outdir: Path) -> None:
    """Run post-processing in main thread.

    Parameters
    ----------
    outdir : Path
        The output directory.

    """
    # Group records into a manifest
    records_dir = outdir / "records"

    failed_count = 0
    records = []
    for record in records_dir.rglob("*.json"):
        path = record
        try:
            with path.open("r") as f:
                records.append(json.load(f))
        except:  # noqa: E722
            failed_count += 1
            print(f"Failed to parse {record}")  # noqa: T201
    if failed_count > 0:
        print(f"Failed to parse {failed_count} entries.")  # noqa: T201
    else:
        print("All entries parsed successfully.")

    # Save manifest
    outpath = outdir / "manifest.json"
    with outpath.open("w") as f:
        json.dump(records, f)


def parse(data: PDB, ccd_resource: CCDResource, clusters: ClusterResource) -> Target:
    """Process a structure.

    Parameters
    ----------
    data : PDB
        The raw input data.
    resource: Resource
        The shared resource.

    Returns
    -------
    Target
        The processed data.

    """
    # Get the PDB id
    pdb_id = data.id.lower()

    # Parse structure
    parsed = parse_mmcif(data.path, ccd_resource, ignore_connections=True)
    structure = parsed.data
    structure_info = parsed.info

    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(structure.chains):
        key = f"{pdb_id}_{chain['entity_id'] + 1}"  # TODO: this is if we 1-index entities as the PDB but the parsing script doesn't
        # print(key, int(chain["res_num"]))
        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain["name"],
                msa_id="",  # FIX
                mol_type=int(chain["mol_type"]),
                cluster_id=clusters.get(key, -1),
                num_residues=int(chain["res_num"]),
                entity_id=int(chain['entity_id']),
            )
        )

    # Get interface metadata
    interface_info = []
    for interface in structure.interfaces:
        chain_1 = int(interface["chain_1"])
        chain_2 = int(interface["chain_2"])
        interface_info.append(
            InterfaceInfo(
                chain_1=chain_1,
                chain_2=chain_2,
            )
        )

    # Create record
    record = Record(
        id=data.id,
        structure=structure_info,
        chains=chain_info,
        interfaces=interface_info,
    )

    return Target(structure=structure, record=record)


def process_structure(
    data: PDB,
    get_ccd_resource: Callable[[], CCDResource],
    outdir: Path,
    filters: list[StaticFilter],
    get_clusters_resource: Callable[[], ClusterResource],
) -> None:
    """Process a target.

    Parameters
    ----------
    item : PDB
        The raw input data.
    resource: Resource
        The shared resource.
    outdir : Path
        The output directory.

    """

    ccd = get_ccd_resource()
    clusters = get_clusters_resource()

    # Check if we need to process
    if "af-" in data.id:
        mid = data.id[6:8]
    else:
        mid = data.id[1:3]

    mid_out_dir = outdir / "structures" / mid
    mid_record_dir = outdir / "records" / mid
    if not mid_out_dir.exists():
        mid_out_dir.mkdir(parents=True, exist_ok=True)
    if not mid_record_dir.exists():
        mid_record_dir.mkdir(parents=True, exist_ok=True)

    struct_path = outdir / "structures" / mid /f"{data.id}.npz"
    record_path = outdir / "records" / mid / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        return

    try:
        # Parse the target
        target: Target = parse(data, ccd, clusters)
        structure = target.structure

        # Apply the filters
        mask = structure.mask
        if filters is not None:
            for f in filters:
                filter_mask = f.filter(structure)
                mask = mask & filter_mask
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"Failed to parse {data.id}")
        return

    # Replace chains and interfaces
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))

    interfaces = []
    for interface in target.record.interfaces:
        chain_1 = bool(mask[interface.chain_1])
        chain_2 = bool(mask[interface.chain_2])
        interfaces.append(replace(interface, valid=(chain_1 and chain_2)))

    # Replace structure and record
    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains, interfaces=interfaces)
    target = replace(target, structure=structure, record=record)

    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))

    # Dump record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)


def process(args) -> None:
    """Run the data processing task."""
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    records_dir = args.outdir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = args.outdir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)

    # Load filters
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Set default pickle properties
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    # Load shared data from redis
    def get_ccd_resource() -> CCDResource:
        return CCDResource(host=args.ccd_redis_host, port=args.ccd_redis_port)
    def get_cluster_resource() -> ClusterResource:
        return ClusterResource(host=args.clusters_redis_host, port=args.clusters_redis_port)

    # Get data points
    print("Fetching data...")
    data = fetch(args.datadir)

    # Check if we can run in parallel
    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(data)))
    parallel = num_processes > 1

    # Run processing
    print("Processing data...")
    if parallel:
        # Create processing function
        fn = partial(
            process_structure,
            get_ccd_resource=get_ccd_resource,
            outdir=args.outdir,
            get_clusters_resource=get_cluster_resource,
            filters=filters,
        )
        # Run processing in parallel
        p_umap(fn, data, num_cpus=num_processes)
    else:
        for item in tqdm(data):
            process_structure(
                item,
                get_ccd_resource=get_ccd_resource,
                outdir=args.outdir,
                get_clusters_resource=get_cluster_resource,
                filters=filters,
            )

    # Finalize
    finalize(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MSA data.")
    parser.add_argument(
        "--datadir",
        type=Path,
        required=True,
        help="The data containing the MMCIF files.",
    )
    parser.add_argument(
        "--clusters-redis-host",
        type=str,
        default="localhost",
        help="The Redis host.",
    )
    parser.add_argument(
        "--clusters-redis-port",
        type=int,
        default=7778,
        help="The Redis port.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="data",
        help="The output directory.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of processes.",
    )
    parser.add_argument(
        "--ccd-redis-host",
        type=str,
        default="localhost",
        help="The Redis host.",
    )
    parser.add_argument(
        "--ccd-redis-port",
        type=int,
        default=7777,
        help="The Redis port.",
    )
    parser.add_argument(
        "--load-redis-into-memory",
        action="store_true"
    )
    parser.add_argument(
        "--use-assembly",
        action="store_true",
        help="Whether to use assembly 1.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    process(args)
