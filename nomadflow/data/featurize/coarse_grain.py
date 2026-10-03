import torch
import numpy as np
from scipy.optimize import milp, LinearConstraint
from scipy.spatial.transform import Rotation
from rdkit import Chem

from nomadflow.openfold.utils import rigid_utils as ru

RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')


def get_rigid_group(atom, rotatable_bonds, current_group):
    neighbors = atom.GetNeighbors()
    for neighbor in neighbors:
        if neighbor.GetIdx() not in current_group:
            current_group = list(set(current_group + [neighbor.GetIdx()]))
            bond_tuple = tuple(sorted((atom.GetIdx(), neighbor.GetIdx())))
            if bond_tuple not in rotatable_bonds:
                neighbor_group = get_rigid_group(neighbor, rotatable_bonds, current_group)
                current_group = list(set(current_group + neighbor_group))
    return current_group


def get_rigid_groups(mol):
    possible_groups = []
    visited_atoms = []
    rotatable_bonds = mol.GetSubstructMatches(RotatableBond)
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in visited_atoms:
            rigid_group = tuple(sorted(get_rigid_group(atom, rotatable_bonds, [atom.GetIdx()])))
            possible_groups.append(rigid_group)

    return set(possible_groups)


def compute_coarse_grain_groups(mol):
    cg_groups = get_rigid_groups(mol)
    num_cg_groups = len(cg_groups)
    num_atoms = len(mol.GetAtoms())
    A = np.zeros((num_atoms, num_cg_groups))

    for i, cg_group in enumerate(cg_groups):
        for atom in cg_group:
            A[atom][i] = 1

    atom_cover_constraint = LinearConstraint(A, lb=np.ones(num_atoms))
    c = np.ones(num_cg_groups)

    res = milp(c, bounds=(0, 1), integrality=np.ones(num_cg_groups), constraints=atom_cover_constraint)
    cg_group_list =  [sorted(g) for i, g in enumerate(cg_groups) if res.x[i] == 1]
    return sorted(cg_group_list, key=lambda x: len(x))


def compute_coarse_grain_frames(
        mol,
        cg_groups,
        conf_id=-1,
        colinear_eps=1e-6,
        frame_select_aug=True
    ):
    conformer = mol.GetConformer(conf_id)
    atom_pos = conformer.GetPositions()
    atom_pos = torch.as_tensor(atom_pos)
    rigids = []
    ref_pos = []
    gt_atom_pos = []
    group_idx = torch.full((atom_pos.shape[0],), torch.nan) # []
    same_group_idx = torch.zeros((atom_pos.shape[0], atom_pos.shape[0]), dtype=torch.bool) # []
    dummy_rot_mask = []

    covered_atoms = []

    for idx, cg_group in enumerate(cg_groups):
        cg_group = sorted(cg_group)
        group_atom_pos = atom_pos[cg_group]
        center_of_mass = torch.mean(group_atom_pos, axis=0)
        if len(cg_group) < 3:
            # we can't define a reference frame with less than 3 atoms
            # so we sample a random rotation instead
            rotquat = torch.as_tensor(Rotation.random().to_quat())
            rigid = ru.Rigid(
                rots=ru.Rotation(quats=rotquat[None]),
                trans=center_of_mass[None]
            )
            dummy_rot_mask.append(True)
        else:
            center_of_mass = torch.mean(group_atom_pos, axis=0)
            if frame_select_aug:
                # add 0.05A noise when selecting whcih atoms to use for the frame
                dist_to_com = torch.linalg.vector_norm(
                    group_atom_pos + torch.randn_like(group_atom_pos) * 0.05 - center_of_mass[None],
                    axis=-1
                )
            else:
                dist_to_com = torch.linalg.vector_norm(group_atom_pos - center_of_mass[None], axis=-1)
            closest_atoms = torch.argsort(dist_to_com)
            closest_atom_pos = group_atom_pos[closest_atoms]

            # case 1: the closest 2 are co-linear with the COM
            cross_prod_1_2 = torch.cross(
                closest_atom_pos[0] - center_of_mass,
                closest_atom_pos[1] - center_of_mass,
                dim=-1
            )
            # case 2: the closest 3 are co-linear
            cross_prod_1_2_3 = torch.cross(
                closest_atom_pos[0] - closest_atom_pos[1],
                closest_atom_pos[2] - closest_atom_pos[1],
                dim=-1
            )
            if torch.linalg.vector_norm(cross_prod_1_2) > colinear_eps:
                rigid = ru.Rigid.from_3_points(
                    closest_atom_pos[[0]],
                    center_of_mass[None],
                    closest_atom_pos[[1]],
                )
            elif torch.linalg.vector_norm(cross_prod_1_2_3) > colinear_eps:
                rigid = ru.Rigid.from_3_points(
                    closest_atom_pos[[1]],
                    closest_atom_pos[[0]],
                    closest_atom_pos[[2]],
                )
            else:
                # check if 1 2 4 are colinear
                cross_prod_1_2_4 = torch.cross(
                    closest_atom_pos[1] - closest_atom_pos[0],
                    closest_atom_pos[3] - closest_atom_pos[0],
                    dim=-1
                )
                if torch.linalg.vector_norm(cross_prod_1_2_4) > colinear_eps:
                    rigid = ru.Rigid.from_3_points(
                        closest_atom_pos[[1]],
                        closest_atom_pos[[0]],
                        closest_atom_pos[[3]],
                    )
                else:
                    raise ValueError("it appears your molecule has >3 co-linear atoms, which we do not support frame construction for")

            dummy_rot_mask.append(False)

        for atom_idx in cg_group:
            if atom_idx not in covered_atoms:
                group_idx[atom_idx] = idx
                covered_atoms.append(idx)

        rigids.append(rigid)

    for cg_group in enumerate(cg_groups):
        for atom_idx1 in cg_group:
            for atom_idx2 in cg_group:
                same_group_idx[covered_atoms.index(atom_idx1)][covered_atoms.index(atom_idx2)] = True

    assert torch.isfinite(group_idx).all()

    ret = {
        "rigids": ru.Rigid.cat(rigids, dim=0),
        "group_idx": group_idx.long(),
        "same_group_idx": same_group_idx,
        "ref_pos": torch.stack(ref_pos, dim=0),
        "gt_atom_pos": torch.stack(gt_atom_pos, dim=0),
        "dummy_rot_mask": torch.as_tensor(dummy_rot_mask),
        "atompos_idx_from_rdkit_idx": torch.as_tensor(covered_atoms, dtype=torch.long)
    }
    return ret


def compute_atomic_frames(
    mol,
    cg_groups,
    conf_id=-1,
    colinear_eps=1e-6,
    frame_select_aug=True
):
    cg_data = compute_coarse_grain_frames(
        mol,
        cg_groups,
        conf_id,
        colinear_eps,
        frame_select_aug
    )
    cg_rigids = cg_data['rigids']
    group_idx = cg_data['group_idx']
    gt_atom_pos = cg_data['gt_atom_pos']
    atom_rigids = ru.Rigid(
        rots=cg_rigids.get_rots()[group_idx],
        trans=gt_atom_pos
    )
    atom_dummy_rot = cg_data['dummy_rot_mask'][group_idx]

    return {
        "atom_rigids": atom_rigids,
        "atom_dummy_rot_mask": atom_dummy_rot,
        "atompos_idx_from_rdkit_idx": cg_data['atompos_idx_from_rdkit_idx']
    }