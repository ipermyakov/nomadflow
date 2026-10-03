""" Compute the bbRMSD between structures """
import argparse
import os
import glob
import json
import dataclasses
import shutil
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import tqdm
import p_tqdm
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.PDBIO import PDBIO
import pyrosetta


from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import PerResidueRMSDMetric
from pyrosetta.rosetta.core.scoring import superimpose_pose, rms_at_corresponding_atoms_no_super

from nomadflow.openfold.utils.feats import atom14_to_atom37
from nomadflow.openfold.utils.tensor_utils import tree_map, tensor_tree_map
from nomadflow.benchmark.success_criteria import filter_one_shot, filter_pmpnn_esmfold
from nomadflow.benchmark.rmsd_mapping import (
    build_rmsd_mapping,
    build_motif_group_mappings,
    pose_indices_from_residue_pairs,
)
from nomadflow.benchmark.group_rmsd_rows import build_group_rmsd_row, pooled_rmsd_outputs_enabled

RMSD_PROTEIN_CA_TYPE = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca
RMSD_ALL_HEAVY_TYPE = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_all_heavy
RMSD_PROTEIN_BB_HEAVY_TYPE = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy


def atom14_to_atom37_np(atom14, batch):
    batch = tree_map(lambda n: torch.tensor(n), batch, np.ndarray)
    out = atom14_to_atom37(torch.as_tensor(atom14), batch)
    return (x.numpy() for x in out)


def parse_esm_log(lines):
    ret = {}
    samples = {}
    name = None
    for l in lines:
        l = l.strip()
        if "Reading sequences" in l:
            if name is not None:
                ret[name] = samples
            name = os.path.splitext(
                l.split("/")[-1]
            )[0]
            samples = {}
        elif "model_name=v_48_020" in l and name is not None:
            sample_dict = {}
            sample_num = 0

            other_vals = l.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pTM" in substring:
                    sample_dict["ptm"] = float(substring.split(" ")[1])
                elif "pAE" in substring:
                    sample_dict["pae"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict

        elif "sample=" in l and name is not None:
            tail = l.split("sample=")[-1]
            sample_dict = {}
            sample_num = int(tail.split(",")[0])

            other_vals = tail.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pAE" in substring:
                    sample_dict["pae"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict
    if name is not None:
        ret[name] = samples

    return ret

def parse_dict_to_df(d):
    keys = ["name", "sample", "plddt", "ptm"]
    pd_dict = {k: [] for k in keys}
    for name, samples in d.items():
        for sample_num, sample_dict in samples.items():
            pd_dict["name"].append(name)
            pd_dict["sample"].append(sample_num)
            pd_dict["plddt"].append(sample_dict['plddt'])
            pd_dict["ptm"].append(sample_dict['ptm'])

    return pd.DataFrame.from_dict(pd_dict)


# RMSDMetric spazzes out sometimes and I dont know why so here's a numpy version for C-alpha rmsd
def calc_ca_rmsd(
    mod_pose,
    ref_pose,
    residue_selector_ref,
    residue_selector,
    residue_selector_super_ref,
    residue_selector_super,
):
    ref_residues = residue_selector_super_ref.get_residues(ref_pose)
    mod_residues = residue_selector_super.get_residues(mod_pose)
    mod_ca_coords = np.array([mod_pose.residue(res).atom("CA").xyz() for res in mod_residues])
    ref_ca_coords = np.array([ref_pose.residue(res).atom("CA").xyz() for res in ref_residues])

    mod_com = mod_ca_coords.mean(axis=0)
    mod_ca_coords = mod_ca_coords - mod_com[None]
    ref_com = ref_ca_coords.mean(axis=0)
    ref_ca_coords = ref_ca_coords - ref_com[None]

    rot, _ = Rotation.align_vectors(ref_ca_coords, mod_ca_coords)

    ref_residues = residue_selector_ref.get_residues(ref_pose)
    mod_residues = residue_selector.get_residues(mod_pose)
    mod_ca_coords = np.array([mod_pose.residue(res).atom("CA").xyz() for res in mod_residues])
    ref_ca_coords = np.array([ref_pose.residue(res).atom("CA").xyz() for res in ref_residues])

    mod_com = mod_ca_coords.mean(axis=0)
    mod_ca_coords = mod_ca_coords - mod_com[None]
    ref_com = ref_ca_coords.mean(axis=0)
    ref_ca_coords = ref_ca_coords - ref_com[None]

    mod_ca_coords = rot.apply(mod_ca_coords)
    mse = np.mean(
        np.sum((mod_ca_coords - ref_ca_coords) ** 2, axis=-1)
    )
    rms = np.sqrt(mse)
    return rms


def _existing_pose_indices(pose, residues, chain):
    pdb_info = pose.pdb_info()
    valid = []
    for residue in residues:
        pose_idx = int(pdb_info.pdb2pose(chain, int(residue)))
        if pose_idx > 0:
            valid.append((int(residue), chain))
    return valid


def _selector_from_pose_indices(pose, pairs):
    pdb_info = pose.pdb_info()
    indices = pose_indices_from_residue_pairs(
        pairs,
        pdb_info.pdb2pose,
    )
    selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    selector.set_index(",".join(str(i) for i in indices))
    return selector


def _compute_single_group_rmsd_row(
    *,
    design_pose,
    query_pose,
    mapping,
    group_mapping,
    name,
    task,
    path,
    sample,
):
    row = build_group_rmsd_row(
        name=name,
        task=task,
        path=path,
        sample=sample,
        group_id=group_mapping["group_id"],
        num_group_res=len(group_mapping["reference_residues"]),
        all_atom_rmsd=10.0,
        ca_rmsd=10.0,
        bb_rmsd=10.0,
        rmsd_status="failed:RuntimeError",
    )
    try:
        ref_pairs = _existing_pose_indices(
            design_pose,
            group_mapping["reference_residues"],
            mapping["reference_chain"],
        )
        folded_pairs = _existing_pose_indices(
            query_pose,
            group_mapping["folded_residues"],
            mapping["folded_chain"],
        )
        bb_ref_residues = group_mapping["bb_reference_residues"] or group_mapping["reference_residues"]
        bb_folded_residues = group_mapping["bb_folded_residues"] or group_mapping["folded_residues"]
        bb_ref_pairs = _existing_pose_indices(
            design_pose,
            bb_ref_residues,
            mapping["reference_chain"],
        )
        bb_folded_pairs = _existing_pose_indices(
            query_pose,
            bb_folded_residues,
            mapping["folded_chain"],
        )
        if not ref_pairs or not folded_pairs:
            raise RuntimeError(
                "Invalid group selectors: "
                f"group_id={group_mapping['group_id']} "
                f"reference_chain={mapping['reference_chain']} folded_chain={mapping['folded_chain']} "
                f"ref_pairs={len(ref_pairs)} folded_pairs={len(folded_pairs)}"
            )
        fixed_res_selector = _selector_from_pose_indices(design_pose, ref_pairs)
        folded_res_selector = _selector_from_pose_indices(query_pose, folded_pairs)
        fixed_bb_selector = (
            _selector_from_pose_indices(design_pose, bb_ref_pairs) if bb_ref_pairs else fixed_res_selector
        )
        folded_bb_selector = (
            _selector_from_pose_indices(query_pose, bb_folded_pairs) if bb_folded_pairs else folded_res_selector
        )

        rmsd_metric = RMSDMetric()
        rmsd_metric.set_comparison_pose(design_pose)
        rmsd_metric.set_run_superimpose(True)

        rmsd_metric.set_rmsd_type(RMSD_ALL_HEAVY_TYPE)
        rmsd_metric.set_residue_selector(folded_res_selector)
        rmsd_metric.set_residue_selector_reference(fixed_res_selector)
        row["motif_group_all_atom_rmsd"] = rmsd_metric.calculate(query_pose)

        row["motif_group_ca_rmsd"] = calc_ca_rmsd(
            query_pose,
            design_pose,
            fixed_bb_selector,
            folded_bb_selector,
            fixed_bb_selector,
            folded_bb_selector,
        )
        rmsd_metric.set_rmsd_type(RMSD_PROTEIN_BB_HEAVY_TYPE)
        rmsd_metric.set_residue_selector(folded_bb_selector)
        rmsd_metric.set_residue_selector_reference(fixed_bb_selector)
        row["motif_group_bb_rmsd"] = rmsd_metric.calculate(query_pose)
        row = build_group_rmsd_row(
            name=name,
            task=task,
            path=path,
            sample=sample,
            group_id=group_mapping["group_id"],
            num_group_res=len(group_mapping["reference_residues"]),
            all_atom_rmsd=row["motif_group_all_atom_rmsd"],
            ca_rmsd=row["motif_group_ca_rmsd"],
            bb_rmsd=row["motif_group_bb_rmsd"],
            rmsd_status="ok",
        )
    except Exception as e:
        row["rmsd_status"] = f"failed:{type(e).__name__}"
    return row


def _compute_group_rmsd_rows(
    *,
    design_pose,
    query_pose,
    mapping,
    group_mappings,
    name,
    task,
    path,
    sample,
):
    return [
        _compute_single_group_rmsd_row(
            design_pose=design_pose,
            query_pose=query_pose,
            mapping=mapping,
            group_mapping=group_mapping,
            name=name,
            task=task,
            path=path,
            sample=sample,
        )
        for group_mapping in group_mappings
    ]


def compute_rmsds(
    design_path,
    folded_path,
    pmpnn_paths,
    pmpnn_paths_idx,
    sample_metadata,
    task_name,
    sample_name,
):

    design_pose = pyrosetta.pose_from_pdb(design_path)

    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(design_pose)
    rmsd_metric.set_run_superimpose(True)

    design_data = {
        "name": os.path.splitext(os.path.basename(design_path))[0],
        "design_path": design_path,
        "folded_path": folded_path,
        "rmsd_status": "ok",
    }
    # compute all-atom design metrics
    # we have to do this workaround cuz pyrosetta does not like the esmfold/proteinmpnn output file names
    # and its a bit of a pain to rename all the files each time
    with open(folded_path) as fp:
        folded_pose = pyrosetta.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(folded_pose, fp.read())
    mapping = build_rmsd_mapping(sample_metadata)
    group_mappings = build_motif_group_mappings(sample_metadata, mapping)
    group_rmsd_rows = []
    if not group_mappings:
        print(f"No motif groups in metadata for sample={sample_name}; skipping per-group RMSD")

    try:
        ref_pairs = _existing_pose_indices(
            design_pose,
            mapping["motif_reference_residues"],
            mapping["reference_chain"],
        )
        folded_pairs = _existing_pose_indices(
            folded_pose,
            mapping["motif_folded_residues"],
            mapping["folded_chain"],
        )
        bb_ref_pairs = _existing_pose_indices(
            design_pose,
            mapping["motif_bb_reference_residues"],
            mapping["reference_chain"],
        )
        bb_folded_pairs = _existing_pose_indices(
            folded_pose,
            mapping["motif_bb_folded_residues"],
            mapping["folded_chain"],
        )
        if not ref_pairs or not folded_pairs:
            raise RuntimeError(
                "Invalid motif selectors: "
                f"reference_chain={mapping['reference_chain']} folded_chain={mapping['folded_chain']} "
                f"ref_pairs={len(ref_pairs)} folded_pairs={len(folded_pairs)}"
            )
        fixed_res_selector = _selector_from_pose_indices(design_pose, ref_pairs)
        esmfold_fixed_res_selector = _selector_from_pose_indices(folded_pose, folded_pairs)
        fixed_bb_selector = (
            _selector_from_pose_indices(design_pose, bb_ref_pairs) if bb_ref_pairs else None
        )
        esmfold_fixed_bb_selector = (
            _selector_from_pose_indices(folded_pose, bb_folded_pairs) if bb_folded_pairs else None
        )
        all_res_selector = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
        rmsd_metric.set_rmsd_type(RMSD_ALL_HEAVY_TYPE)
        rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
        rmsd_metric.set_residue_selector_reference(fixed_res_selector)
        design_data['motif_all_atom_rmsd'] = rmsd_metric.calculate(folded_pose)
        rmsd_metric.set_residue_selector(all_res_selector)
        rmsd_metric.set_residue_selector_reference(all_res_selector)
        design_data['global_all_atom_rmsd'] = rmsd_metric.calculate(folded_pose)

        # rand_aug(folded_pose)

        # rmsd_metric.set_rmsd_type(RMSD_PROTEIN_CA_TYPE)
        if fixed_bb_selector is not None:
            rmsd_metric.set_residue_selector(esmfold_fixed_bb_selector)
            rmsd_metric.set_residue_selector_reference(fixed_bb_selector)
        else:
            rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
            rmsd_metric.set_residue_selector_reference(fixed_res_selector)
        # design_data['motif_ca_rmsd'] = rmsd_metric.calculate(folded_pose)

        # rand_aug(folded_pose)
        design_data['motif_ca_rmsd'] = calc_ca_rmsd(
            folded_pose,
            design_pose,
            fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
            esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
            fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
            esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
        )


        rmsd_metric.set_rmsd_type(RMSD_PROTEIN_BB_HEAVY_TYPE)
        design_data['motif_bb_rmsd'] = rmsd_metric.calculate(folded_pose)
        rmsd_metric.set_residue_selector(all_res_selector)
        rmsd_metric.set_residue_selector_reference(all_res_selector)
        design_data['global_bb_rmsd'] = rmsd_metric.calculate(folded_pose)
    except Exception as e:
        print(f"got error {e}, for path {folded_path}, assigning rmsds to 10")
        print(mapping)
        design_data["rmsd_status"] = f"failed:{type(e).__name__}"
        design_data['motif_all_atom_rmsd'] = 10.
        design_data['global_all_atom_rmsd'] = 10.
        design_data['motif_bb_rmsd'] = 10.
        design_data['motif_ca_rmsd'] = 10.
        design_data['global_bb_rmsd'] = 10.


    pmpnn_data = []
    # rmsd_metric.set_corresponding_atoms_robust(True)
    for path in pmpnn_paths:
        # print(path)
        _data = {}
        try:
            with open(path) as fp:
                pmpnn_pose = pyrosetta.Pose()
                pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pmpnn_pose, fp.read())
            rmsd_metric.set_rmsd_type(RMSD_ALL_HEAVY_TYPE)
            rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
            rmsd_metric.set_residue_selector_reference(fixed_res_selector)
            _data['motif_all_atom_rmsd'] = rmsd_metric.calculate(pmpnn_pose)

            # rmsd_metric.set_rmsd_type(RMSD_PROTEIN_CA_TYPE)
            # rmsd_metric.set_rmsd_type(RMSD_PROTEIN_CA_TYPE)
            if fixed_bb_selector is not None:
                rmsd_metric.set_residue_selector(esmfold_fixed_bb_selector)
                rmsd_metric.set_residue_selector_reference(fixed_bb_selector)
            else:
                rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
                rmsd_metric.set_residue_selector_reference(fixed_res_selector)

            # rand_aug(pmpnn_pose)

            # # _data['motif_ca_rmsd'] = rmsd_metric.calculate(pmpnn_pose)
            _data['motif_ca_rmsd'] = calc_ca_rmsd(
                pmpnn_pose,
                design_pose,
                fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
                esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
                fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
                esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
            )

            # rand_aug(pmpnn_pose)

            rmsd_metric.set_rmsd_type(RMSD_PROTEIN_BB_HEAVY_TYPE)
            _data['motif_bb_rmsd'] = rmsd_metric.calculate(pmpnn_pose)
            rmsd_metric.set_residue_selector(all_res_selector)
            rmsd_metric.set_residue_selector_reference(all_res_selector)
            _data['global_bb_rmsd'] = rmsd_metric.calculate(pmpnn_pose)
            _data['rmsd_status'] = "ok"
        except Exception as e:
            print(f"got error {e}, for path {path}, assigning rmsds to 10")
            _data = {
                'motif_all_atom_rmsd': 10.,
                'motif_bb_rmsd': 10.,
                'motif_ca_rmsd': 10.,
                'global_bb_rmsd': 10.,
                'rmsd_status': f"failed:{type(e).__name__}",
            }
        pmpnn_data.append(_data)

    if group_mappings:
        group_rmsd_rows.extend(
            _compute_group_rmsd_rows(
                design_pose=design_pose,
                query_pose=folded_pose,
                mapping=mapping,
                group_mappings=group_mappings,
                name=sample_name,
                task=task_name,
                path=os.path.abspath(folded_path),
                sample=0,
            )
        )
    for sample_idx, path in zip(pmpnn_paths_idx, pmpnn_paths, strict=False):
        if not group_mappings:
            continue
        try:
            with open(path) as fp:
                pmpnn_pose = pyrosetta.Pose()
                pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pmpnn_pose, fp.read())
            group_rmsd_rows.extend(
                _compute_group_rmsd_rows(
                    design_pose=design_pose,
                    query_pose=pmpnn_pose,
                    mapping=mapping,
                    group_mappings=group_mappings,
                    name=sample_name,
                    task=task_name,
                    path=os.path.abspath(path),
                    sample=sample_idx,
                )
            )
        except Exception as e:
            print(
                f"Failed to compute group RMSD for pmpnn sample={sample_idx} path={path}: "
                f"{type(e).__name__}"
            )

    return design_data, pmpnn_data, group_rmsd_rows


POOLED_RMSD_CSV_NAMES = (
    "sc_rmsd.csv",
    "folding_rmsd.csv",
    "best_sc_motif_rmsd.csv",
    "best_sc_global_rmsd.csv",
)


def build_rmsd_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Compute the ")
    parser.add_argument("--esmlog", help="esmfold output log")
    parser.add_argument("--folded_folders", help="emfold folded structures")
    parser.add_argument(
        "--samples",
        help="gen model samples directory (defaults to samples_eval/ with fallback to samples/)",
    )
    parser.add_argument("--samples_metadata", help="gen model samples")
    parser.add_argument(
        "--sampling-run",
        help="Sampling run root; used to resolve --samples when omitted",
    )
    parser.add_argument(
        "--layout-manifest",
        help="Stage-1 layout_manifest.jsonl (required for Stage-2 group mapping)",
    )
    parser.add_argument(
        "--only-group-rmsd",
        action="store_true",
        help="Write only group_rmsd.csv; do not overwrite folding_rmsd.csv or sc_rmsd.csv",
    )
    return parser


if __name__ == '__main__':
    parser = build_rmsd_arg_parser()
    args = parser.parse_args()
    pyrosetta.init()

    if args.samples is not None:
        samples_dir = args.samples
    elif args.sampling_run is not None:
        from nomadflow.benchmark.analysis_inputs import resolve_samples_dir

        samples_dir = str(resolve_samples_dir(Path(args.sampling_run)))
    else:
        parser.error("Provide --samples or --sampling-run")

    with open(args.esmlog) as fp:
        lines = fp.readlines()
    parse_dict = parse_esm_log(lines)

    with open(args.samples_metadata) as fp:
        samples_metadata = json.load(fp)

    if args.sampling_run is not None or args.layout_manifest is not None:
        from nomadflow.benchmark.layout_manifest_eval import (
            enrich_samples_metadata_from_layout_manifest,
        )

        # In-memory only — never persist enriched metadata into sampling_runs.
        samples_metadata = enrich_samples_metadata_from_layout_manifest(
            samples_metadata,
            sampling_run=Path(args.sampling_run) if args.sampling_run else None,
            layout_manifest_path=(
                Path(args.layout_manifest) if args.layout_manifest else None
            ),
        )

    sc_df_dict = []
    folding_df_dict = []
    group_df_dict = []

    def process(parse_dict_item):
        name, sample_data = parse_dict_item
        ret_sc_df_dicts = []

        sample_path = os.path.join(
            samples_dir,
            name + ".pdb"
        )
        folded_path_folder = os.path.join(
            args.folded_folders,
            name
        )
        design_folded_path = None
        pmpnn_paths = []
        pmpnn_paths_idx = []
        for path in glob.glob(os.path.join(folded_path_folder, "*")):
            if "model_name=v_48_020" in path:
                design_folded_path = path
            else:
                sample_num = int(path.split("sample=")[-1][0])
                pmpnn_paths.append(path)
                pmpnn_paths_idx.append(sample_num)

        try:
            design_rmsds, pmpnn_rmsds, group_rows = compute_rmsds(
                sample_path,
                design_folded_path,
                pmpnn_paths,
                pmpnn_paths_idx,
                samples_metadata[name],
                samples_metadata[name]['name'],
                name,
            )
            print(design_rmsds)
            print(pmpnn_rmsds)
        except Exception as e:
            print(f"Error in {name}")
            raise e
        for i, _data in sample_data.items():
            if i == 0:
                folding_dict = {
                    "name": name,
                    "path": os.path.abspath(sample_path),
                    "task": samples_metadata[name]['name']
                }
                folding_dict.update(_data)
                folding_dict.update(design_rmsds)
                ret_folding_df_dict = folding_dict
            else:
                _idx = pmpnn_paths_idx.index(i)
                sc_dict = {
                    "name": name,
                    "path": os.path.abspath(pmpnn_paths[_idx]),
                    "sample": i,
                    "task": samples_metadata[name]['name']
                }
                sc_dict.update(_data)
                sc_dict.update(pmpnn_rmsds[_idx])
                ret_sc_df_dicts.append(sc_dict)

        return ret_folding_df_dict, ret_sc_df_dicts, group_rows

    _num_cpus = int(
        os.environ.get("RMSD_NUM_CPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_CPUS_ON_NODE")
        or 16
    )
    processed_data = p_tqdm.p_map(process, parse_dict.items(), num_cpus=_num_cpus)
    for _data in processed_data:
        folding_df_dict.append(_data[0])
        sc_df_dict.extend(_data[1])
        group_df_dict.extend(_data[2])

    # for name, sample_data in tqdm.tqdm(parse_dict.items()):
    #     sample_path = os.path.join(
    #         args.samples,
    #         name + ".pdb"
    #     )
    #     folded_path_folder = os.path.join(
    #         args.folded_folders,
    #         name
    #     )
    #     design_folded_path = None
    #     pmpnn_paths = []
    #     pmpnn_paths_idx = []
    #     for path in glob.glob(os.path.join(folded_path_folder, "*")):
    #         if "model_name=v_48_020" in path:
    #             design_folded_path = path
    #         else:
    #             sample_num = int(path.split("sample=")[-1][0])
    #             pmpnn_paths.append(path)
    #             pmpnn_paths_idx.append(sample_num)

    #     fixed_seq_idx = samples_metadata[name]['fixed_seq_res_idx']
    #     fixed_seq_chain = samples_metadata[name]['fixed_seq_chain']
    #     fixed_bb_res_idx = samples_metadata[name]['fixed_bb_res_idx']
    #     fixed_bb_chain = samples_metadata[name]['fixed_bb_chain']

    #     try:
    #         design_rmsds, pmpnn_rmsds = compute_rmsds(
    #             sample_path,
    #             design_folded_path,
    #             pmpnn_paths,
    #             fixed_seq_idx,
    #             fixed_seq_chain,
    #             fixed_bb_res_idx,
    #             fixed_bb_chain,
    #         )
    #         print(design_rmsds)
    #         print(pmpnn_rmsds)
    #     except Exception as e:
    #         print(f"Error in {name}")
    #         raise e
    #     for i, _data in sample_data.items():
    #         if i == 0:
    #             folding_dict = {
    #                 "name": name,
    #                 "path": os.path.abspath(sample_path),
    #                 "task": samples_metadata[name]['name']
    #             }
    #             folding_dict.update(_data)
    #             folding_dict.update(design_rmsds)
    #             folding_df_dict.append(folding_dict)
    #         else:
    #             _idx = pmpnn_paths_idx.index(i)
    #             sc_dict = {
    #                 "name": name,
    #                 "path": os.path.abspath(pmpnn_paths[_idx]),
    #                 "sample": i,
    #                 "task": samples_metadata[name]['name']
    #             }
    #             sc_dict.update(_data)
    #             sc_dict.update(pmpnn_rmsds[_idx])
    #             sc_df_dict.append(sc_dict)

    sc_df = pd.DataFrame(sc_df_dict)
    folding_df = pd.DataFrame(folding_df_dict)
    group_df = pd.DataFrame(group_df_dict)
    if group_df.empty:
        group_df = pd.DataFrame(
            columns=[
                "name",
                "task",
                "path",
                "sample",
                "group_id",
                "num_group_res",
                "motif_group_all_atom_rmsd",
                "motif_group_ca_rmsd",
                "motif_group_bb_rmsd",
                "rmsd_status",
            ]
        )
    group_df.to_csv("group_rmsd.csv", index=False)

    if not pooled_rmsd_outputs_enabled(args.only_group_rmsd):
        raise SystemExit(0)

    sc_df.to_csv("sc_rmsd.csv")
    folding_df.to_csv("folding_rmsd.csv")

    collapsed_motif_df = []
    collapsed_global_df = []
    for name in parse_dict.keys():
        sub_df = sc_df[sc_df['name'] == name]
        motif_min_rmsd = sub_df['motif_all_atom_rmsd'].min()
        collapsed_motif_df.append(sub_df[sub_df['motif_all_atom_rmsd'] == motif_min_rmsd])
        global_min_rmsd = sub_df['global_bb_rmsd'].min()
        collapsed_global_df.append(sub_df[sub_df['global_bb_rmsd'] == global_min_rmsd])
    collapsed_motif_df = pd.concat(collapsed_motif_df)
    collapsed_motif_df.to_csv("best_sc_motif_rmsd.csv")
    collapsed_global_df = pd.concat(collapsed_global_df)
    collapsed_global_df.to_csv("best_sc_global_rmsd.csv")

    pmpnn_pass = filter_pmpnn_esmfold(sc_df)
    pmpnn_pass = pmpnn_pass.groupby("name").sample(1)
    pmpnn_task_pass = pmpnn_pass.groupby("task")
    pz_pass = filter_one_shot(folding_df)
    pz_task_pass = pz_pass.groupby("task")

    with open("../num_designable.txt", 'w') as fp:
        fp.write(f"Num designable: {len(pmpnn_pass)}\n")
        fp.write(f"Num tasks passed: {len(pmpnn_task_pass)}\n")
        fp.write(f"Num consistent: {len(pz_pass)}\n")
        fp.write(f"Num tasks consistent: {len(pz_task_pass)}\n")

