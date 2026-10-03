""" Coarse grain constants

Adapted from https://github.com/aqlaboratory/openfold/blob/2134cc09b3994b6280e6e3c569dd7d761e4da7a0/openfold/np/residue_constants.py
"""
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants used in AlphaFold."""

import collections
import functools
from typing import Mapping, List, Tuple
from importlib import resources

import numpy as np
import tree
import torch

from nomadflow.openfold.data import residue_constants
from nomadflow.openfold.utils import rigid_utils


# Atoms positions relative to the 4 CG groups as inspired by Equifold
# https://www.biorxiv.org/content/10.1101/2022.10.07.511322v2.full.pdf
# 0: 'backbone group',
# 1: 'psi-group',
# 2, 3: sidechain cg groups
# I'm generating these semi-manually here from openfold residue_constants, but
# at some point I should export them

# group 2 and 3 atoms
coarse_grain_sidechain_groups = {
    "ALA": {2: [], 3: []},
    "ARG": {
        2: ["NE", "NH1", "NH2", "CZ"],
        3: ["CB", "CG", "CD"],
    },
    "ASN": {
        2: ["CG", "ND2", "OD1"],
        3: []
    },
    "ASP": {
        2: ["CG", "OD1", "OD2"],
        3: []
    },
    "CYS": {
        2: ["CA", "CB", "SG"],
        3: []
    },
    "GLN": {
        2: ["CG", "CD", "NE2", "OE1"],
        3: []
    },
    "GLU": {
        2: ["CG", "CD", "OE1", "OE2"],
        3: []
    },
    "GLY": {2: [], 3: []},
    "HIS": {
        2: ["CG", "CD2", "ND1", "CE1", "NE2"],
        3: []
    },
    "ILE": {
        2: ["CB", "CG1", "CD1"],
        3: ["CB", "CG1", "CG2"],
    },
    "LEU": {
        2: ["CG", "CD1", "CD2"],
        3: []
    },
    "LYS": {
        2: ["CD", "CE", "NZ"],
        3: ["CB", "CG", "CD"],
    },
    "MET": {
        2: ["CG", "SD", "CE"],
        3: []
    },
    "PHE": {
        2: ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        3: []
    },
    "PRO": {
        2: ["CB", "CG", "CD"],
        3: []
    },
    "SER": {
        2: ["CA", "CB", "OG"],
        3: []
    },
    "THR": {
        2: ["CB", "CG2", "OG1"],
        3: []
    },
    "TRP": {
        2: ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"],
        3: []
    },
    "TYR": {
        2: ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"],
        3: []
    },
    "VAL": {
        2: ["CB", "CG1", "CG2"],
        3: []
    },
}

# groups 2 and 3 atoms which compose the frame
# in order of that fed into Rigid.from_3_points
# these atoms are roughly chosen to maximize the chance that the identity
# of the residue can be determined purely off of the relative positionings of the
# coarse grain frames
#
# e.g.
# - ASP/ASN and GLU/GLN are differentiated by centering frame 2 of ASN/GLN on N rather than CB/D
# - VAL/THR are differentated by centering frame 2 of THR on O rather than CB
# - we rely on the relative angles between the bb frame and frame 2 to differentate ASP/ASN from VAL/THR
# - similar logic applies to LEU and GLU/GLN
# - PHE/TYR are differentated by centering frame 2 of TYR on O
# - the CYS frame is centered on CA to differentated it from SER which is centered on CB

coarse_grain_sidechain_axes = {
    "ALA": {2: [], 3: []},
    "ARG": {
        2: ["NH1", "CZ", "NH2"],
        3: ["CB", "CG", "CD"],
    },
    "ASN": {
        2: ["CG", "ND2", "OD1"],
        3: []
    },
    "ASP": {
        2: ["OD1", "CG", "OD2"],
        3: []
    },
    "CYS": {
        2: ["CA", "CB", "SG"],
        3: []
    },
    "GLN": {
        2: ["CD", "NE2", "OE1"],
        3: []
    },
    "GLU": {
        2: ["OE1", "CD", "OE2"],
        3: []
    },
    "GLY": {2: [], 3: []},
    "HIS": {
        2: ["ND1", "CE1", "NE2"],
        3: []
    },
    "ILE": {
        2: ["CB", "CG1", "CD1"],
        3: ["CG2", "CB", "CG1"],
    },
    "LEU": {
        2: ["CD1", "CG", "CD2"],
        3: []
    },
    "LYS": {
        2: ["CD", "CE", "NZ"],
        3: ["CB", "CG", "CD"],
    },
    "MET": {
        2: ["CG", "SD", "CE"],
        3: []
    },
    "PHE": {
        2: ["CE1", "CZ", "CE2"],
        3: []
    },
    "PRO": {
        2: ["CB", "CG", "CD"],
        3: []
    },
    "SER": {
        2: ["CA", "CB", "OG"],
        3: []
    },
    "THR": {
        2: ["CB", "OG1", "CG2"],
        3: []
    },
    "TRP": {
        2: ["CH2", "CZ2", "CZ3"],
        3: []
    },
    "TYR": {
        2: ["CE2", "OH", "CE1"],
        3: []
    },
    "VAL": {
        2: ["CB", "CG1", "CG2"],
        3: []
    },
}

cg_group_mask = {
    resname: [
        1.0,
        1.0,
        float(len(coarse_grain_sidechain_groups[resname][2]) > 0),
        float(len(coarse_grain_sidechain_groups[resname][3]) > 0)
    ]
    for resname in coarse_grain_sidechain_groups
}

# fill out with backbone first since that stays the same
coarse_grain_atom_positions = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 1, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 1, (0.626, 1.062, 0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 1, (0.625, 1.062, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 1, (0.626, 1.062, -0.000)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 1, (0.625, 1.062, -0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 1, (0.626, 1.062, -0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 1, (0.626, 1.062, 0.000)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 1, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 1, (0.625, 1.063, 0.000)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 1, (0.627, 1.062, -0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 1, (0.625, 1.063, -0.000)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 1, (0.626, 1.062, -0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 1, (0.625, 1.062, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 1, (0.626, 1.062, -0.000)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 1, (0.621, 1.066, 0.000)],
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 1, (0.626, 1.062, -0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 1, (0.626, 1.062, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 1, (0.627, 1.062, 0.000)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 1, (0.627, 1.062, -0.000)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 1, (0.627, 1.062, -0.000)],
    ],
}

def _make_rigid_transformation_4x4(ex, ey, translation):
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # Normalize ex.
    ex_normalized = ex / np.linalg.norm(ex)

    # make ey perpendicular to ex
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # compute ez as cross product
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack(
        [ex_normalized, ey_normalized, eznorm, translation]
    ).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m

# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom37_to_cg_group = np.zeros([21, 37], dtype=int)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
restype_atom37_cg_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_cg_group = np.zeros([21, 14], dtype=int)
restype_atom14_cg_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_cg_group_default_frame = np.zeros([21, 4, 4, 4], dtype=np.float32)


def _make_coarse_grain_atom_positions():
    # default_positions = torch.zeros(21, 14, 3)
    # for i, (_, atom_pos_list) in enumerate(residue_constants.rigid_group_atom_positions.items()):
    #     for j, (_, _, coord) in enumerate(atom_pos_list):
    #         default_positions[i, j] = torch.as_tensor(coord)

    IDEALIZED_POS = torch.tensor(residue_constants.restype_atom14_rigid_group_positions)
    DEFAULT_FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame)
    ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
    GROUP_IDX = torch.tensor(residue_constants.restype_atom14_to_rigid_group)

    default_rigids = rigid_utils.Rigid.from_tensor_4x4(DEFAULT_FRAMES)

    default_rigids_4 = default_rigids[..., 4]
    default_rigids_5 = default_rigids_4.compose(default_rigids[..., 5])
    default_rigids_6 = default_rigids_5.compose(default_rigids[..., 6])
    default_rigids_7 = default_rigids_6.compose(default_rigids[..., 7])
    default_rigids = rigid_utils.Rigid.cat([
        default_rigids[..., :4],
        default_rigids_4[..., None],
        default_rigids_5[..., None],
        default_rigids_6[..., None],
        default_rigids_7[..., None],
    ], dim=-1)

    # generate default positions which we'll reconstruct the CG frames from
    # co-opting some of the model code to make this easier
    aatype = torch.arange(21)
    # [21, 14]
    group_mask = GROUP_IDX.to(aatype.device)[aatype, ...]

    # [21, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_FRAMES.shape[-3],
    ).to(default_rigids.device)

    # [21, 14, 8]
    t_atoms_to_global = default_rigids[..., None, :] * group_mask

    # [21, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )
    # [21, 14, 1]
    frame_atom_mask = ATOM_MASK.to(aatype.device)[aatype, ...].unsqueeze(-1).to(default_rigids.device)
    # [21, 14, 3]
    frame_null_pos = IDEALIZED_POS.to(aatype.device)[aatype, ...].to(default_rigids.device)
    default_positions = t_atoms_to_global.apply(frame_null_pos)
    default_positions = default_positions * frame_atom_mask

    # this is extremely sloppy rip
    for cg_group in [2, 3]:
        for restype, restype_letter in enumerate(residue_constants.restypes):
            ref_atom14 = default_positions[restype]
            resname = residue_constants.restype_1to3[restype_letter]
            cg_group_atoms = coarse_grain_sidechain_groups[resname][cg_group]
            cg_frame_atoms = coarse_grain_sidechain_axes[resname][cg_group]

            if len(cg_group_atoms) > 0:
                def _get_idx(atom):
                    atom_list = residue_constants.restype_name_to_atom14_names[resname]
                    return atom_list.index(atom)
                cg_group_atoms = sorted(cg_group_atoms, key=_get_idx)

                cg_frame = rigid_utils.Rigid.from_3_points(
                    ref_atom14[_get_idx(cg_frame_atoms[0])],
                    ref_atom14[_get_idx(cg_frame_atoms[1])],
                    ref_atom14[_get_idx(cg_frame_atoms[2])],
                )
                if cg_group == 3:
                    prev_tensor_4x4 = torch.as_tensor(restype_cg_group_default_frame[restype, cg_group-1])
                    prev_cg_frame = rigid_utils.Rigid.from_tensor_4x4(prev_tensor_4x4)
                    rel_cg_frame = prev_cg_frame.invert().compose(cg_frame)
                    tensor_4x4 = rel_cg_frame.to_tensor_4x4()
                else:
                    tensor_4x4 = cg_frame.to_tensor_4x4()
                tensor_4x4 = cg_frame.to_tensor_4x4()
                restype_cg_group_default_frame[restype, cg_group] = tensor_4x4.numpy()

                for atomname in cg_group_atoms:
                    atompos = ref_atom14[_get_idx(atomname)]
                    if atomname not in [data[0] for data in coarse_grain_atom_positions[resname]]:
                        coarse_grain_atom_positions[resname].append(
                            [atomname, cg_group, tuple(cg_frame.invert_apply(atompos).tolist())]
                        )

_make_coarse_grain_atom_positions()


def _make_cg_group_constants():
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(residue_constants.restypes):
        resname = residue_constants.restype_1to3[restype_letter]
        for atomname, group_idx, atom_position in coarse_grain_atom_positions[
            resname
        ]:
            atomtype = residue_constants.atom_order[atomname]
            restype_atom37_to_cg_group[restype, atomtype] = group_idx
            restype_atom37_cg_group_positions[
                restype, atomtype, :
            ] = atom_position

            atom14idx = residue_constants.restype_name_to_atom14_names[resname].index(atomname)
            restype_atom14_to_cg_group[restype, atom14idx] = group_idx
            restype_atom14_cg_group_positions[
                restype, atom14idx, :
            ] = atom_position

    for restype, restype_letter in enumerate(residue_constants.restypes):
        resname = residue_constants.restype_1to3[restype_letter]
        atom_positions = {
            name: np.array(pos)
            for name, _, pos in coarse_grain_atom_positions[resname]
        }

        # backbone to backbone is the identity transform
        restype_cg_group_default_frame[restype, 0, :, :] = np.eye(4)

        # psi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"] - atom_positions["N"],
            translation=atom_positions["C"],
        )
        restype_cg_group_default_frame[restype, 1, :, :] = mat

_make_cg_group_constants()