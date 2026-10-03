""" Methods for generating proper inputs for sampling. """
import copy
from dataclasses import astuple, asdict, dataclass, replace
from typing import Tuple, Optional

from rdkit import Chem

import numpy as np
import torch
import networkx as nx
from scipy.spatial.transform import Rotation

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import (
    Residue, Atom, Chain, Structure, SamplingResidue
)

from nomadflow.data.featurize.tokenize import RES_TO_AA, Tokenized, convert_atom_str_to_tuple
from nomadflow.openfold.data import residue_constants as rc
from nomadflow.data.constants import coarse_grain as cg
from nomadflow.data.featurize.motif_se3_augment import apply_multi_motif_se3_augmentation
from nomadflow.utils import coarse_grain as cg_utils
from nomadflow.openfold.utils import rigid_utils as ru

from .tokenize import (
    Rigid,
    RigidData,
    Token,
    TokenBond,
    TokenData,
    Tokenized,
    append_motif_group_aggregators,
    standard_residue_to_frames,
)


# TODO: Revisit grouping/selection strategy in this helper.
# Current contiguous-fragment assignment is a pragmatic default and may need
# task-aware logic (e.g., explicit motif partitioning controls) for sampling.
def _assign_copy_motif_group_ids(
    token_data: np.ndarray,
    rigid_data: np.ndarray,
    explicit_group_by_copy_key: Optional[dict[tuple[int, int], int]] = None,
) -> None:
    """Assign rigid-level motif groups for copied motif residues in-place."""
    if explicit_group_by_copy_key is not None:
        missing_keys: list[tuple[int, int]] = []
        for token_idx in range(token_data.shape[0]):
            token = token_data[token_idx]
            if not bool(token["is_copy"]):
                continue
            key = (int(token["asym_id"]), int(token["res_idx"]))
            group_id = explicit_group_by_copy_key.get(key)
            if group_id is None:
                missing_keys.append(key)
                continue
            rigid_start = int(token["rigid_idx"])
            rigid_end = rigid_start + int(token["rigid_num"])
            rigid_data["motif_group_id"][rigid_start:rigid_end] = np.int32(group_id)
        if missing_keys:
            preview = ", ".join(f"(asym_id={a}, res_idx={r})" for a, r in missing_keys[:5])
            raise ValueError(
                "motif_groups does not cover all copied motif residues. Missing keys: "
                + preview
            )
        return

    # Default fallback: contiguous copied fragments form independent groups.
    next_group_id = 1
    last_key = None
    current_group_id = 0

    for token_idx in range(token_data.shape[0]):
        token = token_data[token_idx]
        if not bool(token["is_copy"]):
            continue

        key = (int(token["asym_id"]), int(token["res_idx"]))
        if last_key is None:
            current_group_id = next_group_id
            next_group_id += 1
        else:
            last_asym, last_res = last_key
            if key[0] != last_asym or key[1] != last_res + 1:
                current_group_id = next_group_id
                next_group_id += 1

        rigid_start = int(token["rigid_idx"])
        rigid_end = rigid_start + int(token["rigid_num"])
        rigid_slice = slice(rigid_start, rigid_end)
        rigid_data["motif_group_id"][rigid_slice] = np.int32(current_group_id)
        last_key = key

@dataclass(frozen=True)
class AtomData:
    """Tokenized datatype."""

    name: np.ndarray
    element: int
    charge: int
    coords: np.ndarray
    conformer: np.ndarray
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class ResidueData:
    """Tokenized datatype."""

    name: str
    res_type: int
    res_idx: int
    atom_idx: int
    atom_num: int
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool
    is_copy: bool


@dataclass(frozen=True)
class ChainData:
    """Tokenized datatype."""

    name: str
    mol_type: int
    entity_id: int
    sym_id: int
    asym_id: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_num: int
    cyclic_period: int



def generate_protein_structure_template(  # noqa: C901, PLR0915
    chain_lens: dict[str, int],
    extra_mols: Optional[Structure] = None,
    extra_mols_residue_is_unindexed_mask: Optional[np.ndarray] = None
) -> Structure:
    """ Generate a structure template which can be filled in by other methods
    depending on downstream sampling task.

    Parameters
    ----------
    chain_lens: dict[str, int]
        A mapping from chain ID to number of residues in that chain.
        Specifies the new protein chains to generate.
    extra_mols: Optional[Structure]
        Input molecules to include in the structure template.
        These either specify protein generation conditioning (e.g. binder target, motif to scaffold)
        or inclusion of known molecules (e.g. fixed small molecule ligand)
    extra_mols_residue_is_unindexed: Optional[np.ndarray]
        Specifies which residues in `extra_mols` will be an unindexed (protein) residue.
        These residues will have their residue indices overwritten

    Returns
    -------
    struct : Structure
        The structure template.

    """
    if extra_mols is not None:
        extra_mols = copy.deepcopy(extra_mols)

    chain_names = list(chain_lens.keys())
    if extra_mols is not None:
        chain_names.extend(extra_mols.chains['name'].tolist())
    chain_names = sorted(set(chain_names))

    chain_data = {}
    residue_data = []
    res_idx = 0

    for chain, chain_len in chain_lens.items():
        chain_idx = chain_names.index(chain)
        if chain not in chain_data:
            chain_data[chain] = {
                "name": chain,
                "mol_type": const.chain_type_ids["PROTEIN"],
                "entity_id": chain_idx,
                "sym_id": chain_idx,
                "asym_id": chain_idx,
                "atom_idx": 0,
                "atom_num": 0,
                "res_idx": res_idx,
                "res_num": chain_len,
                "cyclic_period": 0,
            }
        else:
            chain_data[chain]["res_num"] = chain_data[chain]["res_num"] + chain_len

        for _ in range(chain_len):
            res = ResidueData(
                name=const.unk_token["PROTEIN"],
                res_type=const.unk_token_ids["PROTEIN"],
                res_idx=res_idx,
                atom_idx=0,  # placeholder
                atom_num=0,  # placeholder
                atom_center=1,
                atom_disto=1,
                is_standard=True,
                is_present=True,
                is_copy=False
            )
            residue_data.append(astuple(res))
            res_idx += 1

    residues = np.array(residue_data, dtype=SamplingResidue)
    chain_data = [astuple(ChainData(**d)) for d in chain_data.values()]
    chains = np.array(chain_data, dtype=Chain)
    atoms = np.array([])
    bonds = np.array([])

    if extra_mols is not None:
        assert extra_mols_residue_is_unindexed_mask is not None
        # offset residue index for residues
        extra_mols.residues['res_idx'][extra_mols_residue_is_unindexed_mask] += len(residues)
        extra_mols.chains['res_idx'] += len(residues)
        # rename the chain IDs of the extra_mols to be consistent with the template residue chains
        for chain in extra_mols.chains:
            new_chain_idx = chain_names.index(chain['name'])
            chain["entity_id"] = new_chain_idx
            chain["sym_id"] = new_chain_idx
            chain["asym_id"] = new_chain_idx
        residues = np.concatenate([residues, extra_mols.residues], axis=0)
        chains = np.concatenate([chains, extra_mols.chains], axis=0)
        # TODO: we might need to think about renumbering some of these
        atoms = extra_mols.atoms
        bonds = extra_mols.bonds

    struct = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=np.array([]),
        interfaces=np.array([]),
        mask=np.ones(chains.shape[0], dtype=bool),
    )
    return struct


def sample_noise_from_struct_template(  # noqa: C901, PLR0915
    struct: Structure,
    task_masks: Optional[dict[str, np.ndarray]] = None,
    trans_std=16,
) -> Tuple[np.ndarray, np.array, np.ndarray, np.ndarray]:
    """ Sample a t=0 initial datapoint from a structure template.

    Parameters
    ----------
    struct: Structure
        A structure template generated by `generate_protein_structure_template` (or other equivalent means)
    task_masks: Optional[dict[str, np.ndarray]]
        Specifies different modes of featurization. If not None, must include the following keys:
        - atom_noising_mask (np.ndarray, dtype=bool): for each atom in `struct`, specify whether to noise the atom or not
        - res_type_noising_mask (np.ndarray, dtype=bool): for each residue in `struct`, specify whether to mask the residue type (sequence)
        - residue_is_unindexed_mask (np.ndarray, dtype=bool): for each residue in `struct`, specify whether to the residue is unindexed
    trans_std : int, default=16
        The standard deviation of the translation prior distribution.

    Returns:
        Tuple[np.ndarray, np.array, np.ndarray, np.ndarray]: _description_
    """
    bond_graph = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in struct.bonds])

    # Create token data and rigid data
    token_data = []
    rigid_data = []

    # Keep track of atom_idx, rigid_idx to token_idx
    token_idx = 0
    rigid_idx = 0
    atom_to_rigid = {}
    rigid_to_token = {}

    # Filter to valid chains only
    chains = struct.chains[struct.mask]

    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

        for i, res in enumerate(struct.residues[res_start:res_end]):
            # Get atom indices
            atom_start = res["atom_idx"]
            atom_end = res["atom_idx"] + res["atom_num"]

            # Standard residues with unknown sequence are completely designed
            if res["is_standard"] and (res['name'] == 'UNK') and is_protein:
                # Token is present if centers are
                is_present = res["is_present"]

                rigid_trans = np.random.randn(3, 3) * trans_std
                rigid_quat = Rotation.random(3).as_quat(canonical=True)
                rigid_tensor7 = np.concatenate([rigid_quat, rigid_trans], axis=-1)
                rigid_mask = np.ones(3, dtype=bool)

                # Create token
                token = TokenData(
                    token_idx=token_idx,
                    rigid_idx=rigid_idx,
                    rigid_num=3,
                    res_idx=res["res_idx"],
                    res_type=res["res_type"],
                    res_name=res["name"],
                    sym_id=chain["sym_id"],
                    asym_id=chain["asym_id"],
                    entity_id=chain["entity_id"],
                    mol_type=chain["mol_type"],
                    resolved_mask=is_present,
                    center_coords=rigid_tensor7[0, 4:],
                    is_copy=False,
                    is_unindexed=False,
                    is_atomized=False,
                    seq_noising_mask=True,
                    motif_group_id=0,
                )
                token_data.append(astuple(token))

                # Update rigid_idx to token_idx
                for j in range(3):
                    rigid = RigidData(
                        rigid_idx=rigid_idx,
                        token_idx=token_idx,
                        sidechain_idx=j,
                        is_atomized=False,
                        element=-1,
                        charge=0,
                        tensor7=rigid_tensor7[j],
                        is_present=rigid_mask[j],
                        rigids_noising_mask=True,
                        num_real_input_axes=2
                    )
                    rigid_data.append(astuple(rigid))
                    rigid_to_token[rigid_idx] = token_idx

                    rigid_idx += 1

                token_idx += 1

            # Standard residues with known sequence are read in and optionally (partially) redesigned
            elif res["is_standard"] and (res['name'] != 'UNK') and is_protein:
                # If we're provided a standard token, we're currently assuming its a copy
                # since we're not doing multi-chain sampling
                # TODO: we should have a separate mask for this at some point

                # Token is present if centers are
                is_present = res["is_present"]

                # Get frame atoms
                atoms = struct.atoms[atom_start:atom_end]
                assert task_masks is not None
                atom_noising_mask = task_masks['atom_noising_mask']
                res_type_noising_mask = task_masks['res_type_noising_mask']
                residue_is_unindexed_mask = task_masks['residue_is_unindexed_mask']

                is_unindexed = residue_is_unindexed_mask[res_start+i]
                noise_seq = res_type_noising_mask[res_start + i]

                rigid_tensor7, rigid_mask, dummy_rigid_idx = standard_residue_to_frames(
                    res, atoms
                )

                # Create token
                token = TokenData(
                    token_idx=token_idx,
                    rigid_idx=rigid_idx,
                    rigid_num=3,
                    res_idx=res["res_idx"],
                    res_type=res["res_type"],
                    res_name=res["name"],
                    sym_id=chain["sym_id"],
                    asym_id=chain["asym_id"],
                    entity_id=chain["entity_id"],
                    mol_type=chain["mol_type"],
                    resolved_mask=is_present,
                    center_coords=rigid_tensor7[0, 4:],
                    is_copy=True,
                    is_unindexed=bool(is_unindexed),
                    is_atomized=False,
                    seq_noising_mask=bool(noise_seq),
                    motif_group_id=0,
                )
                token_data.append(astuple(token))

                noise_atoms = atom_noising_mask[atom_start:atom_end]

                atom_name_to_cg_idx = {}
                cg_idx_to_atom_idx = {0: [], 1: [], 2: []}
                atom_order = rc.restype_name_to_atom14_names[res["name"]]

                for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    atom_name_to_cg_idx[atom_name] = 0
                    if atom_name == 'CB' and res["name"] == "GLY":
                        continue
                    cg_idx_to_atom_idx[0].append(atom_order.index(atom_name))

                for atom_name in cg.coarse_grain_sidechain_groups[res["name"]][2]:
                    atom_name_to_cg_idx[atom_name] = 1
                    cg_idx_to_atom_idx[1].append(atom_order.index(atom_name))
                for atom_name in cg.coarse_grain_sidechain_groups[res["name"]][3]:
                    atom_name_to_cg_idx[atom_name] = 2
                    cg_idx_to_atom_idx[2].append(atom_order.index(atom_name))

                for j, atom in enumerate(atoms):
                    atom_name = atom['name']
                    atom_name = [chr(c + 32) for c in atom_name if c != 0]
                    atom_name = "".join(atom_name)
                    atom_to_rigid[atom_start + j] = rigid_idx + atom_name_to_cg_idx[atom_name]

                _noise_rigid = []
                # Update rigid_idx to token_idx
                for j in range(3):
                    cg_atom_idxs = cg_idx_to_atom_idx[j]
                    # we figure out if we're noising the rigid
                    # by checking if any of its component atoms are being noised
                    # we also need to check if its a dummy rigid
                    # and if so, copy the noising status of its corresponding non-dummy rigid
                    if len(cg_atom_idxs) > 0:
                        noise_rigid = noise_atoms[cg_atom_idxs].any()
                    else:
                        noise_rigid = _noise_rigid[dummy_rigid_idx[j]]
                    _noise_rigid.append(noise_rigid)

                    noise_rigid_trans = np.random.randn(3) * trans_std
                    noise_rigid_quat = Rotation.random().as_quat(canonical=True)
                    noise_rigid_tensor7 = np.concatenate([noise_rigid_quat, noise_rigid_trans], axis=-1)

                    rigid = RigidData(
                        rigid_idx=rigid_idx,
                        token_idx=token_idx,
                        sidechain_idx=j,
                        is_atomized=False,
                        element=-1,
                        charge=0,
                        tensor7=noise_rigid_tensor7 if noise_rigid else rigid_tensor7[j],
                        is_present=True if noise_rigid else rigid_mask[j],
                        # Diffusion mask semantics: copy rigids still participate in denoising.
                        rigids_noising_mask=True,
                        num_real_input_axes=2
                    )
                    rigid_data.append(astuple(rigid))
                    rigid_to_token[rigid_idx] = token_idx

                    rigid_idx += 1

                token_idx += 1

            # Non-standard are tokenized per atom
            else:
                # We use the unk protein token as res_type
                unk_token = const.unk_token["PROTEIN"]
                unk_id = const.token_ids[unk_token]

                # Get atom coordinates
                atom_data = struct.atoms[atom_start:atom_end]
                atom_coords = atom_data["coords"]

                valid_neighbors = [i for i in range(res["atom_num"]) if atom_data[i]["is_present"]]
                valid_neighbor_coords = atom_coords[np.array(valid_neighbors, dtype=int)]

                # Tokenize each atom
                for j, atom in enumerate(atom_data):
                    # Token is present if atom is
                    is_present = res["is_present"] & atom["is_present"]
                    atom_idx = atom_start + j

                    atom_trans = np.random.randn(3) * trans_std
                    atom_quat = Rotation.random().as_quat(canonical=True)
                    atom_tensor7 = np.concatenate([atom_quat, atom_trans], axis=-1)

                    # Create token
                    token = TokenData(
                        token_idx=token_idx,
                        rigid_idx=rigid_idx,
                        rigid_num=1,
                        res_idx=res["res_idx"],
                        res_type=unk_id,
                        res_name=res["name"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        resolved_mask=is_present,
                        center_coords=atom_tensor7[4:],
                        is_copy=False,
                        is_unindexed=False,
                        is_atomized=True,
                        seq_noising_mask=False,
                        motif_group_id=0,
                    )
                    token_data.append(astuple(token))

                    num_real_input_axes = len(list(bond_graph.neighbors(atom_idx)))

                    # Update atom_idx to token_idx
                    atom_to_rigid[atom_start + j] = rigid_idx
                    rigid = RigidData(
                        rigid_idx=rigid_idx,
                        token_idx=token_idx,
                        sidechain_idx=0,
                        is_atomized=True,
                        element=atom["element"],
                        charge=atom["charge"],
                        tensor7=atom_tensor7,
                        is_present=atom["is_present"],
                        rigids_noising_mask=True,
                        num_real_input_axes=2 if num_real_input_axes > 2 else num_real_input_axes
                    )
                    rigid_data.append(astuple(rigid))
                    rigid_to_token[rigid_idx] = token_idx

                    rigid_idx += 1
                    token_idx += 1

    # Create token bonds
    token_bonds = []

    # Add atom-atom bonds from ligands
    for bond in struct.bonds:
        atom1 = bond["atom_1"]
        atom2 = bond["atom_2"]
        if atom1 not in atom_to_rigid or atom2 not in atom_to_rigid:
            continue
        rigid1 = atom_to_rigid[atom1]
        rigid2 = atom_to_rigid[atom2]
        if rigid1 not in rigid_to_token or rigid2 not in rigid_to_token:
            continue
        token_bond = (
            rigid_to_token[rigid1],
            rigid_to_token[rigid2],
            bond["type"] + 1,
        )
        token_bonds.append(token_bond)

    token_data = np.array(token_data, dtype=Token)
    token_bonds = np.array(token_bonds, dtype=TokenBond)
    rigid_data = np.array(rigid_data, dtype=Rigid)
    multi_motif_se3 = bool(
        task_masks.get("multi_motif_se3", False) if task_masks is not None else False
    )
    if multi_motif_se3:
        explicit_group_by_copy_key = None
        if task_masks is not None:
            explicit_group_by_copy_key = task_masks.get("motif_group_by_copy_key")
        _assign_copy_motif_group_ids(
            token_data,
            rigid_data,
            explicit_group_by_copy_key=explicit_group_by_copy_key,
        )

    # Center on motif-group rigids when available; fall back to truly fixed rigids.
    center_rigids_mask = rigid_data["motif_group_id"] > 0
    if not center_rigids_mask.any():
        center_rigids_mask = ~rigid_data["rigids_noising_mask"]
    if center_rigids_mask.any():
        center_rigids = rigid_data[center_rigids_mask]
        fixed_rigids_com = center_rigids["tensor7"][:, 4:].mean(axis=0)
        tensor7_edit = np.zeros((rigid_data.shape[0], 7))
        tensor7_edit[center_rigids_mask, 4:] = fixed_rigids_com[None]
        rigid_data["tensor7"] -= tensor7_edit
    else:
        fixed_rigids_com = np.zeros(3)
    # fixed_rigids_com = np.zeros(3)

    if task_masks is not None:
        apply_multi_motif_se3_augmentation(
            token_data,
            rigid_data,
            enabled=multi_motif_se3,
            np_rng=task_masks.get("np_rng", np.random),
            trans_std=float(task_masks.get("multi_motif_se3_trans_std", trans_std)),
        )

    if multi_motif_se3:
        token_data, rigid_data = append_motif_group_aggregators(
            token_data,
            rigid_data,
        )

    return token_data, rigid_data, token_bonds, fixed_rigids_com


def construct_atoms(
    data: Tokenized,
    struct: Structure,
    take_copy_structure: bool = True # False
):
    """ Given a structure template and corresponding tokenized data, construct a full Structure object with the properly shaped arrays.

    NOTE: This function is primarily intended to generate a Structure object that has all the correctly shaped arrays e.g.
    with all the memory pre-allocated. Users should not rely on this function to *generate coordinates* (the current implementation
    does some coordinate updating as a side effect and may not do in future iterations). To update coordinates, use `update_structure`.


    Parameters
    ===========
    data: Tokenized
        A Tokenized object specifying the necessary data to reconstruct a full biomolecule Structure
    struct: Structure
        A structure template (such as that generated by `generate_protein_structure_template`)
    take_copy_structure: bool, default=True
        When given a motif structure, whether or not to impute the copy structure at the locations specified by `data`

    Returns
    =======
    struct: Structure
        The full Structure object.
    """

    token_data = data.tokens
    rigids_data = data.rigids

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    atom_data = []
    tokens_to_residues = {}
    copy_tokens_to_residues = {}
    new_residues = []

    # res_idx = 1
    atom_idx = 0

    unindexed_to_motif_idx = {}

    for token in token_data:
        res_idx = token['res_idx']
        if token['is_copy']:
            record_dict = copy_tokens_to_residues

            # TODO: we're currently placing "inferred seq pos" in the token_idx slot
            # if the motif residue is unindexed, we need to record the which residue has been assigned to represent the residue
            # so we can determine whether the resultant residue needs to copy the motif coords
            if token['is_unindexed']:
                unindexed_to_motif_idx[token['token_idx']] = res_idx

        else:
            record_dict = tokens_to_residues

        if res_idx not in record_dict:
            record_dict[res_idx] = [token]
        else:
            record_dict[res_idx].append(token)

    # print(unindexed_to_motif_idx)

    has_residue_is_copy = "is_copy" in struct.residues.dtype.names
    residue_token_sets = []
    standard_tokens = []

    for residue in struct.residues:
        res_idx = residue['res_idx']
        residue_is_copy = bool(residue["is_copy"]) if has_residue_is_copy else False
        if residue_is_copy:
            token_set = copy_tokens_to_residues[res_idx]
        elif take_copy_structure:
            # use the copy's data instead of the denoiser's data for residues which have been copied
            if res_idx in unindexed_to_motif_idx:
                # we check if an unindexed motif residue to be scaffolded maps to the current residue
                token_set = copy_tokens_to_residues[unindexed_to_motif_idx[res_idx]]
                # print(residue, token_set)
            elif res_idx in copy_tokens_to_residues:
                # otherwise we just assume that the mapping is indexed
                token_set = copy_tokens_to_residues[res_idx]
                # print(residue, token_set)
            else:
                token_set = tokens_to_residues[res_idx]
        else:
            token_set = tokens_to_residues[res_idx]
        residue_token_sets.append(token_set)
        if len(token_set) == 1:
            token = token_set[0]
            token_is_standard_protein = (
                token['mol_type'] == const.chain_type_ids["PROTEIN"]
                and token['rigid_num'] == 3
            )
            if token_is_standard_protein:
                standard_tokens.append(token)

    if standard_tokens:
        standard_tensor7 = np.stack(
            [
                rigids_data["tensor7"][
                    token["rigid_idx"]: token["rigid_idx"] + token["rigid_num"]
                ].copy()
                for token in standard_tokens
            ],
            axis=0,
        )
        standard_tensor7 = torch.as_tensor(standard_tensor7)
        standard_rigids = ru.Rigid.from_tensor_7(standard_tensor7)
        standard_seq = torch.as_tensor([RES_TO_AA[token["res_type"]] for token in standard_tokens])
        standard_mask = torch.ones_like(standard_seq, dtype=torch.bool)
        standard_atom14, standard_atom14_mask = cg_utils.compute_atom14_from_cg_frames(
            standard_rigids,
            standard_mask,
            standard_seq,
            return_atom_mask=True,
        )
        if standard_atom14_mask.dim() > 0 and standard_atom14_mask.shape[-1] == 1:
            standard_atom14_mask = standard_atom14_mask.squeeze(-1)
        # Only add a batch dimension for a single standard residue [14, 3].
        # A plain dim()==2 check would also match [N, 14] and break indexing.
        if standard_atom14.dim() == 2 and standard_atom14.shape[-1] == 3:
            standard_atom14 = standard_atom14.unsqueeze(0)
        if standard_atom14_mask.dim() == 1:
            standard_atom14_mask = standard_atom14_mask.unsqueeze(0)
        standard_atom14_mask = standard_atom14_mask.bool()
    else:
        standard_atom14 = None
        standard_atom14_mask = None

    standard_idx = 0
    for residue, token_set in zip(struct.residues, residue_token_sets):
        res_idx = residue["res_idx"]

        if len(token_set) == 1:
            token = token_set[0]
            token_is_standard_protein = (
                token['mol_type'] == const.chain_type_ids["PROTEIN"]
                and
                token['rigid_num'] == 3  ## TODO: we should probably use an is_atomized field instead
            )
            assert token_is_standard_protein

            if res_idx in unindexed_to_motif_idx:
                copy_token = copy_tokens_to_residues[unindexed_to_motif_idx[res_idx]][0]
                res_type = copy_token['res_type']
                res_name = const.tokens[res_type]
            elif res_idx in copy_tokens_to_residues:
                copy_token = copy_tokens_to_residues[res_idx][0]
                res_type = copy_token['res_type']
                res_name = const.tokens[res_type]
            elif token['seq_noising_mask']:
                res_type = token['res_type']
                res_name = const.tokens[res_type]
            else:
                res_type = residue['res_type']
                res_name = residue['name']
            # print(residue, token['res_type'], const.tokens[token['res_type']], residue['res_type'], residue['name'], token['seq_noising_mask'])
            atom_num = len(rc.residue_atoms[res_name])


            res = ResidueData(
                name=res_name,
                res_type=res_type,
                res_idx=res_idx,
                atom_idx=atom_idx,
                atom_num=atom_num,
                atom_center=1,
                atom_disto=1,
                is_standard=True,
                is_present=True,
                is_copy=token['is_copy']
            )
            new_residues.append(astuple(res))
            assert standard_atom14 is not None
            assert standard_atom14_mask is not None

            num_atoms = len(rc.residue_atoms[res_name])
            res_atom14 = standard_atom14[standard_idx]
            res_atom14_mask = standard_atom14_mask[standard_idx]
            standard_idx += 1
            atom_coords = res_atom14[:num_atoms].numpy(force=True)
            atom_is_present = res_atom14_mask[:num_atoms].numpy(force=True)


            # construct the rest of the atom data
            res_atom_names = [i for i in rc.restype_name_to_atom14_names[res_name] if len(i) > 0]
            for i, res_atom_name in enumerate(res_atom_names):
                atom_name = np.array(convert_atom_str_to_tuple(res_atom_name))
                atom_element = res_atom_name[0]  # TODO: im pretty sure this works but might be good to have smth less error prone
                atom_element = periodic_table.GetAtomicNumber(atom_element)
                atom_charge = 0
                atom_conformer = np.zeros_like(atom_coords)
                atom_chirality = 0
                coords = atom_coords[i]
                is_present = atom_is_present[i]
                atom_conformer = np.zeros_like(coords)

                atom = AtomData(
                    name=atom_name,
                    element=atom_element,
                    charge=atom_charge,
                    coords=coords,
                    conformer=atom_conformer,
                    is_present=is_present,
                    chirality=atom_chirality
                )
                atom_data.append(astuple(atom))
                atom_idx += 1

        else:
            res = ResidueData(
                name=residue['name'],
                res_type=residue['res_type'],
                res_idx=res_idx,
                atom_idx=atom_idx,
                atom_num=residue['atom_num'],
                atom_center=0,
                atom_disto=0,
                is_standard=False,
                is_present=True,
                is_copy=False
            )
            new_residues.append(astuple(res))
            atom_num = len(token_set)

            atom_start = residue['atom_idx']
            atom_end = residue['atom_idx'] + residue['atom_num']
            atoms = struct.atoms[atom_start:atom_end]

            # construct the rest of the atom data
            for i, token in enumerate(token_set):
                rigid = rigids_data[token['rigid_idx']]
                new_atom = atoms[i].copy()
                new_atom['coords'] = rigid['tensor7'][4:]
                atom_data.append(tuple(new_atom.tolist()))
                atom_idx += 1


    new_struct = Structure(
        atoms=np.array(atom_data, dtype=Atom),
        bonds=np.array([]),
        residues=np.array(new_residues, dtype=SamplingResidue),
        chains=struct.chains,
        connections=np.array([]),
        interfaces=np.array([]),
        mask=struct.mask,
    )
    # print(struct.residues)
    return new_struct

# TODO: this function seems pretty prone to bugs since we make many assumptions about how rigid_tensor7 is organized
def update_structure(
    struct: Structure,
    rigid_tensor7,
    *,
    copy_struct: bool = True,
):
    """ Update the coordinates in a Structure object from a set of rigids.

    NOTE: This function makes many assumptions about the organization of `rigid_tensor7`. For proper functionality,
    ensure you are sourcing the rigids tensor7 from the proper Tokenized object

    Parameters
    ==========
    struct: Structure
        Structure object
    rigid_tensor7: torch.Tensor (n_rigids x 7)
        Rigids which correspond to all coordinates in the structure. This assumes the tensor is organized
        the same way the rigids['tensor7'] tensor of the corresponding Tokenized object would be organized.

    Returns
    =======
    Structure
        Structure object with updated coordinates
    """
    if copy_struct:
        struct = copy.deepcopy(struct)
    rigid_tensor7_is_torch = torch.is_tensor(rigid_tensor7)
    rigid_coords_np = None
    if rigid_tensor7_is_torch:
        rigid_coords_np = rigid_tensor7[:, 4:].detach().cpu().numpy()
    chains = struct.chains[struct.mask]

    rigid_idx = 0

    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

        tensor7_rigid_starts = []
        tensor7_idx = []
        tensor7_seq = []
        _rigid_idx = rigid_idx
        for i, res in enumerate(struct.residues[res_start:res_end]):
            if res["is_standard"] and is_protein:
                rigid_start = _rigid_idx
                tensor7_rigid_starts.append(rigid_start)
                tensor7_idx.append(i)

                res_type = res["res_type"]
                aa_type = RES_TO_AA[int(res_type)]
                tensor7_seq.append(aa_type)

                _rigid_idx += 3
            else:
                _rigid_idx += res["atom_num"]

        if len(tensor7_idx) > 0:
            if rigid_tensor7_is_torch:
                index = torch.as_tensor(tensor7_rigid_starts, device=rigid_tensor7.device, dtype=torch.long)
                offsets = torch.tensor([0, 1, 2], device=rigid_tensor7.device, dtype=torch.long)
                gather_index = (index[:, None] + offsets[None, :]).reshape(-1)
                tensor7_for_atom14 = rigid_tensor7.index_select(0, gather_index).view(-1, 3, 7)
            else:
                index = np.asarray(tensor7_rigid_starts, dtype=np.int64)
                gather_index = (index[:, None] + np.array([0, 1, 2], dtype=np.int64)[None, :]).reshape(-1)
                tensor7_for_atom14 = np.asarray(rigid_tensor7[gather_index], dtype=np.float32).reshape(-1, 3, 7)
                tensor7_for_atom14 = torch.as_tensor(tensor7_for_atom14)
            rigids_for_atom14 = ru.Rigid.from_tensor_7(tensor7_for_atom14)
            tensor7_seq = torch.as_tensor(tensor7_seq)
            dummy_mask = torch.ones_like(tensor7_seq, dtype=torch.bool)
            atom14, atom14_mask = cg_utils.compute_atom14_from_cg_frames(rigids_for_atom14, dummy_mask, tensor7_seq, return_atom_mask=True)
            if atom14_mask.dim() > 0 and atom14_mask.shape[-1] == 1:
                atom14_mask = atom14_mask.squeeze(-1)
            if atom14.dim() == 2 and atom14.shape[-1] == 3:
                atom14 = atom14.unsqueeze(0)
            if atom14_mask.dim() == 1:
                atom14_mask = atom14_mask.unsqueeze(0)
            atom14_mask = atom14_mask.bool()
            atom14_np = atom14.detach().cpu().numpy()
            atom14_mask_np = atom14_mask.detach().cpu().numpy()
            tensor7_idx_map = {res_i: idx for idx, res_i in enumerate(tensor7_idx)}
        else:
            atom14_np = None
            atom14_mask_np = None
            tensor7_idx_map = {}

        for i, res in enumerate(struct.residues[res_start:res_end]):
            # Get atom indices
            atom_start = res["atom_idx"]
            atom_end = res["atom_idx"] + res["atom_num"]

            # Standard residues are tokens
            if res["is_standard"] and is_protein:
                assert atom14_np is not None
                assert atom14_mask_np is not None
                # Token is present if centers are
                is_present = res["is_present"]

                # Get frame atoms
                atoms = struct.atoms[atom_start:atom_end]

                # rigid_start = rigid_idx
                # rigid_end = rigid_idx + 3
                # res_tensor7 = rigid_tensor7[rigid_start:rigid_end].copy()
                # atom14, atom14_mask = frames_to_atom14(res, res_tensor7)
                # atom14 = atom14.squeeze()
                # atom14_mask = atom14_mask.squeeze()
                # atom14_mask = atom14_mask.bool()
                _idx = tensor7_idx_map[i]
                res_atom14 = atom14_np[_idx]
                res_atom14_mask = atom14_mask_np[_idx]
                # Update the full standard-residue atom block for this frame.
                # Partial updates can leak coordinates from another frame/template.
                atom_count = atoms.shape[0]
                atoms["coords"] = res_atom14[:atom_count]
                atoms["is_present"] = res_atom14_mask[:atom_count]

                rigid_idx += 3

            # Non-standard are tokenized per atom
            else:
                # Get atom coordinates
                atom_data = struct.atoms[atom_start:atom_end]

                # Tokenize each atom
                for i, atom in enumerate(atom_data):
                    # Token is present if atom is
                    is_present = res["is_present"] & atom["is_present"]
                    if rigid_tensor7_is_torch:
                        assert rigid_coords_np is not None
                        coords = rigid_coords_np[rigid_idx]
                    else:
                        coords = rigid_tensor7[rigid_idx, 4:]
                    atom_data[i]["coords"] = coords

                    rigid_idx += 1

    return struct
