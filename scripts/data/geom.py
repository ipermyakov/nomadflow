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
from typing import Any, Optional

import numpy as np
import rdkit
from mmcif import parse_mmcif
from p_tqdm import p_umap
from redis import Redis
from tqdm import tqdm
import hashlib

from nomadflow.boltz.data.types import ChainInfo, StructureInfo, InterfaceInfo, Record, Target

from nomadflow.data.featurize.mol.sampling import mol_to_struct

def hash_sequence(seq: str) -> str:
    """Hash a sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()


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


def parse(datadir: Path, data: dict) -> Target:
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
    data_id = data['data_id']
    data_path = datadir / data['pickle_path']
    with open(data_path, 'rb') as fp:
        data_dict = pickle.load(fp)
    # Parse structure
    conformer_data = data_dict['conformers']
    # we just take the first conformer for now
    rd_mol = conformer_data[0]['rd_mol']
    structure = mol_to_struct(rd_mol, noise_ligand=False)

    structure_info = StructureInfo(
        deposited=None,
        revised=None,
        released=None,
        resolution=None,
        method=f"QM9:{data_dict['smiles']}",
        num_chains=1,
        num_interfaces=0,
    )

    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(structure.chains):
        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain["name"],
                msa_id="",  # FIX
                mol_type=int(chain["mol_type"]),
                cluster_id=-1,
                num_residues=int(chain["res_num"]),
            )
        )

    # Get interface metadata
    interface_info = []

    # Create record
    record = Record(
        id=data_id,
        structure=structure_info,
        chains=chain_info,
        interfaces=interface_info,
    )

    return Target(structure=structure, record=record)


def process_structure(
    data: tuple[str, dict],
    datadir: Path,
    outdir: Path,
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
    smiles = data[0]
    data_dict = data[1]
    data_id = hash_sequence(smiles)
    data_dict['data_id'] = data_id
    mid = data_id[1:3]

    mid_out_dir = outdir / "structures" / mid
    mid_record_dir = outdir / "records" / mid
    if not mid_out_dir.exists():
        mid_out_dir.mkdir(parents=True, exist_ok=True)
    if not mid_record_dir.exists():
        mid_record_dir.mkdir(parents=True, exist_ok=True)

    struct_path = outdir / "structures" / mid /f"{data_id}.npz"
    record_path = outdir / "records" / mid / f"{data_id}.json"

    if struct_path.exists() and record_path.exists():
        return

    try:
        # Parse the target
        target: Target = parse(datadir, data_dict)
        structure = target.structure
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"Failed to parse {smiles}")
        print(data_dict)
        return

    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))

    record = target.record
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

    datadir = Path(args.datadir)
    with open(datadir / f"summary_{args.dataset}.json") as fp:
        metadata = json.load(fp)
        metadata = [t for t in metadata.items() if len(t[0]) > 1]

    # Set default pickle properties
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    # Check if we can run in parallel
    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(metadata)))
    parallel = num_processes > 1

    # Run processing
    print("Processing data...")
    if parallel:
        # Create processing function
        fn = partial(
            process_structure,
            datadir=datadir,
            outdir=args.outdir,
        )
        # Run processing in parallel
        p_umap(fn, metadata, num_cpus=num_processes)
    else:
        for item in tqdm(metadata):
            process_structure(
                item,
                datadir=datadir,
                outdir=args.outdir,
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
        "--dataset",
        type=str,
        required=True,
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
    args = parser.parse_args()
    assert args.dataset in ['qm9', 'drugs']
    process(args)
