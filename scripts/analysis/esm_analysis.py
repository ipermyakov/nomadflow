""" Compute the bbRMSD between structures """
import argparse
import os
import glob
import dataclasses
import shutil

import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import tqdm
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
import pyrosetta

from nomadflow.openfold.utils.feats import atom14_to_atom37
from nomadflow.openfold.utils.tensor_utils import tree_map, tensor_tree_map

def pdb_to_structure(pdb_path, silent=False):
    parser = PDBParser(QUIET=silent)
    name = os.path.splitext(
        os.path.basename(pdb_path)
    )[0]
    struct = parser.get_structure(name, pdb_path)
    return struct


def atom14_to_atom37_np(atom14, batch):
    batch = tree_map(lambda n: torch.tensor(n), batch, np.ndarray)
    out = atom14_to_atom37(torch.as_tensor(atom14), batch)
    return (x.numpy() for x in out)


def parse_esm_log(lines, fold_original=False):
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
        elif fold_original and "model_name=v_48_020" in l:
            sample_dict = {}
            sample_num = 0

            other_vals = l.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pTM" in substring:
                    sample_dict["ptm"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict

        elif "sample=" in l:
            tail = l.split("sample=")[-1]
            sample_dict = {}
            sample_num = int(tail.split(",")[0])

            other_vals = tail.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pTM" in substring:
                    sample_dict["ptm"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute the ")
    parser.add_argument("--esmlog", help="esmfold output log")
    parser.add_argument("--folded_folders", help="emfold folded structures")
    parser.add_argument("--samples", help="gen model samples")
    parser.add_argument("--fold_original", default=True, action='store_false')
    args = parser.parse_args()


    with open(args.esmlog) as fp:
        lines = fp.readlines()
    parse_dict = parse_esm_log(lines, fold_original=args.fold_original)
    df = parse_dict_to_df(parse_dict)
    # df = df[~df['name'].str.contains("clean_traj")]
    print(df)

    rmsds = []
    all_atom_rmsds = []
    core_all_atom_rmsds = []
    surface_all_atom_rmsds = []
    superimpose = Superimposer()
    pdbio = PDBIO()

    pyrosetta.init()

    for _, row in tqdm.tqdm(list(df.iterrows())):
        sample_path = os.path.join(
            args.samples,
            row['name'] + ".pdb"
        )
        folded_path = os.path.join(
            args.folded_folders,
            row['name']
        )
        current_folded = None
        folded_paths = glob.glob(os.path.join(folded_path, "*"))
        for path in folded_paths:
            if args.fold_original and row['sample'] == 0:
                if "model_name=v_48_020" in path:
                    current_folded = path
                    break
            else:
                if f"sample={row['sample']}" in path:
                    current_folded = path
                    break

        sample = pdb_to_structure(sample_path, silent=True)
        folded = pdb_to_structure(current_folded, silent=True)
        sample_ca = [atom for atom in sample.get_atoms() if atom.get_name() == 'CA']
        folded_ca = [atom for atom in folded.get_atoms() if atom.get_name() == 'CA']
        for _ in range(5):
            superimpose.set_atoms(
                fixed=sample_ca,
                moving=folded_ca
            )
            superimpose.apply(folded_ca)
        rmsds.append(superimpose.rms)

        if args.fold_original and row['sample'] == 0:
            TMPFILE = f"{os.environ.get('TMPDIR', '/tmp')}/folded_intermediate-{hash(current_folded)}.pdb"
            shutil.copyfile(current_folded, TMPFILE)

            sample_pose = pyrosetta.io.pose_from_pdb(sample_path)
            folded_pose = pyrosetta.io.pose_from_pdb(TMPFILE)
            rmsd = pyrosetta.rosetta.core.scoring.all_atom_rmsd(sample_pose, folded_pose)
            all_atom_rmsds.append(rmsd)

            core_res_selector = pyrosetta.rosetta.core.select.residue_selector.NumNeighborsSelector()
            # i have no idea how ur supposed to properly do this
            core_res = list(pyrosetta.rosetta.core.select.get_residues_from_subset(core_res_selector.apply(sample_pose)))
            core_res_list = pyrosetta.rosetta.std.list_unsigned_long_t()
            for i in core_res:
                core_res_list.append(i)
            core_rmsd = pyrosetta.rosetta.core.scoring.all_atom_rmsd(sample_pose, folded_pose, core_res_list)
            core_all_atom_rmsds.append(core_rmsd)

            not_core_res_selector = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector()
            not_core_res_selector.set_residue_selector(core_res_selector)
            # i have no idea how ur supposed to properly do this
            not_core_res = list(pyrosetta.rosetta.core.select.get_residues_from_subset(not_core_res_selector.apply(sample_pose)))
            not_core_res_list = pyrosetta.rosetta.std.list_unsigned_long_t()
            for i in not_core_res:
                not_core_res_list.append(i)
            surface_rmsd = pyrosetta.rosetta.core.scoring.all_atom_rmsd(sample_pose, folded_pose, not_core_res_list)
            surface_all_atom_rmsds.append(surface_rmsd)

            print(rmsd, core_rmsd, surface_rmsd)


            if os.path.exists(TMPFILE):
                os.remove(TMPFILE)


            # sample = pdb_to_structure(sample_path, silent=True)
            # folded = pdb_to_structure(current_folded, silent=True)
            # sample_all_atom = [atom for atom in sample.get_atoms()]
            # folded_all_atom = [atom for atom in folded.get_atoms()]

            # # an initial alignment
            # superimpose.set_atoms(
            #     fixed=sample_all_atom,
            #     moving=folded_all_atom
            # )
            # superimpose.apply(folded_all_atom)

            # stringio = StringIO()
            # pdbio.set_structure(folded)
            # with open(TMPFILE, 'w') as fp:
            #     pdbio.save(fp)
            # with open(TMPFILE) as fp:
            #     folded_str = fp.read()
            # folded_protein = from_pdb_string(folded_str)
            # folded_protein = dataclasses.asdict(folded_protein)
            # folded_protein.update({
            #     'all_atom_positions': folded_protein['atom_positions'],
            #     'all_atom_mask': folded_protein['atom_mask']
            # })
            # folded_protein = make_atom14_masks_np(folded_protein)
            # folded_protein = make_atom14_positions_np(folded_protein)
            # folded_atom14 = folded_protein['atom14_gt_positions']

            # with open(sample_path) as fp:
            #     sample_protein = from_pdb_string(fp.read())
            # sample_protein_dict = dataclasses.asdict(sample_protein)
            # sample_protein_dict.update({
            #     'all_atom_positions': sample_protein.atom_positions,
            #     'all_atom_mask': sample_protein.atom_mask
            # })
            # sample_protein_dict = make_atom14_masks_np(sample_protein_dict)
            # sample_protein_dict = make_atom14_positions_np(sample_protein_dict)
            # sample_gt_atom14 = sample_protein_dict['atom14_gt_positions']
            # sample_alt_atom14 = sample_protein_dict['atom14_alt_gt_positions']

            # gt_mse = np.square(folded_atom14 - sample_gt_atom14).sum(axis=-1)
            # alt_mse = np.square(folded_atom14 - sample_gt_atom14).sum(axis=-1)
            # alt_mse_better = alt_mse < gt_mse
            # sample_opt_atom14 = sample_gt_atom14 * (~alt_mse_better[..., None]) + sample_alt_atom14 * alt_mse_better[..., None]
            # sample_opt_atom37, sample_atom37_mask = atom14_to_atom37_np(sample_opt_atom14, sample_protein_dict)

            # sample_protein = Protein(
            #     atom_positions=sample_opt_atom37,
            #     aatype=sample_protein.aatype,
            #     atom_mask=sample_protein.atom_mask,
            #     residue_index=sample_protein.residue_index,
            #     chain_index=sample_protein.chain_index,
            #     b_factors=sample_protein.b_factors
            # )
            # stringio = StringIO()
            # with open(TMPFILE, 'w') as fp:
            #     fp.write(to_pdb(sample_protein))
            # sample = pdb_to_structure(TMPFILE, silent=True)
            # sample_all_atom = [atom for atom in sample.get_atoms()]
            # folded_all_atom = [atom for atom in folded.get_atoms()]

            # # second alignment
            # superimpose.set_atoms(
            #     fixed=sample_all_atom,
            #     moving=folded_all_atom
            # )
            # superimpose.apply(folded_all_atom)

            # all_atom_rmsds.append(superimpose.rms)

    df['sc_rmsd'] = rmsds

    sc_df = df[df['sample'] != 0]
    sc_df.to_csv("sc_rmsd.csv")

    folding_df = df[df['sample'] == 0]
    if len(all_atom_rmsds) == len(folding_df):
        folding_df['sc_rmsd_all_atom'] = all_atom_rmsds
    if len(core_all_atom_rmsds) == len(folding_df):
        folding_df['sc_rmsd_core_all_atom'] = core_all_atom_rmsds
    if len(core_all_atom_rmsds) == len(folding_df):
        folding_df['sc_rmsd_surface_all_atom'] = surface_all_atom_rmsds
    folding_df.to_csv("folding_rmsd.csv")

    collapsed_df = []
    for name in parse_dict.keys():
        sub_df = sc_df[sc_df['name'] == name]
        min_rmsd = sub_df['sc_rmsd'].min()
        collapsed_df.append(sub_df[sub_df['sc_rmsd'] == min_rmsd])
    collapsed_df = pd.concat(collapsed_df)
    collapsed_df.to_csv("best_sc_rmsd.csv")
    with open("../num_designable.txt", 'w') as fp:
        fp.write(f"Num designable: {len(collapsed_df[collapsed_df.sc_rmsd < 2])}\n")
        fp.write(f"Num folded correctly (bb): {len(folding_df[folding_df.sc_rmsd < 2])}\n")
        fp.write(f"Num folded correctly (aa): {len(folding_df[folding_df.sc_rmsd_all_atom < 2])}\n")
        fp.write(f"Num folded correctly (core): {len(folding_df[(folding_df.sc_rmsd_core_all_atom < 2) & (folding_df.sc_rmsd < 2)])}")

