import copy
from dataclasses import astuple, asdict, dataclass, replace
from typing import Tuple, List, Optional, Union
import functools as fn


import numpy as np
import torch
import networkx as nx
from scipy.spatial.transform import Rotation

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import (
    Structure
)

from nomadflow.openfold.data import residue_constants as rc
from nomadflow.data.constants import coarse_grain as cg
from nomadflow.data.constants.atomize import get_standard_protein_residue_bonds
from nomadflow.utils import coarse_grain as cg_utils
from nomadflow.openfold.utils import rigid_utils as ru
from nomadflow.openfold.utils.rigid_utils import rot_to_quat

Token = [
    ("token_idx", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i4")),
    ("res_name", np.dtype("<U8")),
    ("rigid_idx", np.dtype("i4")),
    ("rigid_num", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i4")),  # the total bytes need to be divisible by 4
    ("resolved_mask", np.dtype("?")),
    ("center_coords", np.dtype("3f4")),
    ("is_copy", np.dtype("?")),
    ("is_unindexed", np.dtype("?")),
    ("is_atomized", np.dtype("?")),
    ("seq_noising_mask", np.dtype("?")),
    ("motif_group_id", np.dtype("i4")),
    ("is_motif_group_aggregator", np.dtype("?")),
]

Rigid = [
    ("rigid_idx", np.dtype("i4")),
    ("token_idx", np.dtype("i4")),
    ("sidechain_idx", np.dtype("i1")),
    ("is_atomized", np.dtype("?")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("tensor7", np.dtype("7f4")),
    ("is_present", np.dtype("?")),
    ("rigids_noising_mask", np.dtype("?")),
    ("num_real_input_axes", np.dtype("i1")),
    ("motif_group_id", np.dtype("i4")),
    ("is_motif_group_aggregator", np.dtype("?")),
]

TokenBond = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

IDENTITY_TENSOR7 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

RES_TO_AA = {}
for i, aa in enumerate(rc.resnames):
    RES_TO_AA[const.token_ids[aa]] = i
AA_TO_RES = {j: i for i, j in RES_TO_AA.items()}

# TODO: probably move mmcif.py from data processing into nomadflow so i can just import this
def convert_atom_str_to_tuple(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip()
    name_tuple = [ord(c) - 32 for c in name]
    name_tuple = name_tuple + [0] * (4 - len(name))
    return tuple(name_tuple)

def convert_atom_tuple_to_str(name: tuple[int, int, int, int]) -> str:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    tuple[int, int, int, int]
        The converted atom name.

    """
    atom_name = [chr(c + 32) for c in name if c != 0]
    atom_name = "".join(atom_name)
    return atom_name

@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    rigids: np.ndarray
    bonds: np.ndarray
    structure: Structure

@dataclass
class TokenData:
    """TokenData datatype."""

    token_idx: int
    res_idx: int
    res_type: int
    res_name: str
    rigid_idx: int
    rigid_num: int
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    resolved_mask: bool
    center_coords: np.array
    is_copy: bool
    is_unindexed: bool
    is_atomized: bool
    seq_noising_mask: bool
    motif_group_id: int = 0
    is_motif_group_aggregator: bool = False


@dataclass
class RigidData:
    """RigidData datatype."""

    rigid_idx: int
    token_idx: int
    sidechain_idx: int
    is_atomized: bool
    element: int
    charge: int
    tensor7: np.ndarray
    is_present: bool
    rigids_noising_mask: bool
    num_real_input_axes: int
    motif_group_id: int = 0
    is_motif_group_aggregator: bool = False


def _sample_random_quat(np_rng=None) -> np.ndarray:
    if np_rng is None or np_rng is np.random:
        return Rotation.random().as_quat(canonical=True).astype(np.float32)
    return Rotation.random(random_state=np_rng).as_quat(canonical=True).astype(np.float32)


def append_motif_group_aggregators(
    token_data: np.ndarray,
    rigid_data: np.ndarray,
    np_rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Append one aggregator token/rigid for each motif group.

    The helper is idempotent for arrays using the current dtype: existing rows
    marked with ``is_motif_group_aggregator`` reserve their ``motif_group_id`` and
    are not duplicated on repeated calls.
    """
    source_mask = (rigid_data["motif_group_id"] > 0) & ~rigid_data["is_motif_group_aggregator"]
    existing_aggregator_groups = set(
        int(group_id)
        for group_id in rigid_data["motif_group_id"][rigid_data["is_motif_group_aggregator"]]
        if int(group_id) > 0
    )
    missing_group_ids = [
        int(group_id)
        for group_id in np.unique(rigid_data["motif_group_id"][source_mask])
        if int(group_id) not in existing_aggregator_groups
    ]
    if len(missing_group_ids) == 0:
        return token_data, rigid_data

    aggregator_tokens = np.zeros(len(missing_group_ids), dtype=Token)
    aggregator_rigids = np.zeros(len(missing_group_ids), dtype=Rigid)
    next_token_idx = int(token_data["token_idx"].max()) + 1 if len(token_data) > 0 else 0
    next_rigid_idx = int(rigid_data["rigid_idx"].max()) + 1 if len(rigid_data) > 0 else 0

    for aggregator_offset, group_id in enumerate(missing_group_ids):
        group_rigids = rigid_data[source_mask & (rigid_data["motif_group_id"] == group_id)]
        present_mask = group_rigids["is_present"]
        centroid_source = group_rigids[present_mask] if bool(present_mask.any()) else group_rigids
        centroid = centroid_source["tensor7"][:, 4:].mean(axis=0).astype(np.float32)
        tensor7 = IDENTITY_TENSOR7.copy()
        tensor7[:4] = _sample_random_quat(np_rng)
        tensor7[4:] = centroid

        token_idx = next_token_idx + aggregator_offset
        rigid_idx = next_rigid_idx + aggregator_offset

        aggregator_tokens[aggregator_offset] = astuple(TokenData(
            token_idx=token_idx,
            res_idx=-1,
            res_type=0,
            res_name="VGM",
            rigid_idx=rigid_idx,
            rigid_num=1,
            sym_id=-1,
            asym_id=-1,
            entity_id=-1,
            mol_type=-1,
            resolved_mask=bool(present_mask.any()),
            center_coords=centroid,
            is_copy=False,
            is_unindexed=True,
            is_atomized=False,
            seq_noising_mask=False,
            motif_group_id=group_id,
            is_motif_group_aggregator=True,
        ))
        aggregator_rigids[aggregator_offset] = astuple(RigidData(
            rigid_idx=rigid_idx,
            token_idx=token_idx,
            sidechain_idx=0,
            is_atomized=False,
            element=-1,
            charge=0,
            tensor7=tensor7,
            is_present=bool(present_mask.any()),
            rigids_noising_mask=True,
            num_real_input_axes=0,
            motif_group_id=group_id,
            is_motif_group_aggregator=True,
        ))

    return (
        np.concatenate([token_data, aggregator_tokens]),
        np.concatenate([rigid_data, aggregator_rigids]),
    )


def compute_frame(
    n: np.ndarray,
    ca: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the frame for a residue.

    Parameters
    ----------
    n : np.ndarray
        The N atom.
    ca : np.ndarray
        The C atom.
    c : np.ndarray
        The CA atom.

    Returns
    -------
    np.ndarray
        The frame.

    """
    v1 = c - ca
    v2 = n - ca
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    u2 = v2 - e1 * np.dot(e1.T, v2)
    e2 = u2 / (np.linalg.norm(u2) + 1e-10)
    e3 = np.cross(e1, e2)
    rot = np.column_stack([e1, e2, e3])
    t = ca
    return rot, t


def get_unk_token(chain: np.ndarray) -> int:
    """Get the unk token for a residue.

    Parameters
    ----------
    chain : np.ndarray
        The chain.

    Returns
    -------
    int
        The unk token.

    """
    if chain["mol_type"] == const.chain_type_ids["DNA"]:
        unk_token = const.unk_token["DNA"]
    elif chain["mol_type"] == const.chain_type_ids["RNA"]:
        unk_token = const.unk_token["RNA"]
    else:
        unk_token = const.unk_token["PROTEIN"]

    res_id = const.token_ids[unk_token]
    return res_id


def standard_residue_to_frames(residue, atoms):
    """ Generate the frame presentation for a canonical protein residue.

    Parameters
    ==========
    residue: nomadflow.boltz.data.types.Residue
        The residue to featurize
    atoms: np.ndarray, dtype=nomadflow.boltz.data.types.Atom
        The atoms in the residue to be featurized

    Returns
    =======
    rigids_tensor7: np.ndarray
        Stack of residue frames in tensor7 format
    rigids_mask: np.ndarray, dtype=bool
        Mask specifying if all the atoms needed to define each frame exists
    dummy_rigid_idx:  list[int]
        Specifies the source frame for any duplicate frames generated. If all frames
        are unique, this is [0, 1, 2].
    """
    res_name = residue['name']
    atoms = atoms[atoms['is_present']]  # only select present atoms
    dummy_rigid_idx = [0]

    bb_frame = ['N', 'CA', 'C']
    # bb_frame = ['C', 'CA', 'N']
    frame2 = cg.coarse_grain_sidechain_axes[res_name][2]
    frame3 = cg.coarse_grain_sidechain_axes[res_name][3]
    # construct dummy frames as necessary
    # use bb frame if frame2 doesn't exist
    if len(frame2) == 0:
        frame2 = bb_frame
        dummy_rigid_idx.append(dummy_rigid_idx[-1])
    else:
        dummy_rigid_idx.append(dummy_rigid_idx[-1]+1)
    # use frame2 frame if frame3 doesn't exist
    if len(frame3) == 0:
        frame3 = frame2
        dummy_rigid_idx.append(dummy_rigid_idx[-1])
    else:
        dummy_rigid_idx.append(dummy_rigid_idx[-1]+1)

    frame_atom_names = [bb_frame, frame2, frame3]
    # figure out which frames are resolved enough for us to model
    # this is a little roundabout because the atom ids are stored as arrays of 4 ints
    frame_atom_ids = [
        np.array([convert_atom_str_to_tuple(c) for c in atom_set])  # shape (3, 4)
        for atom_set in frame_atom_names
    ]
    atom_ids = atoms["name"]  # shape of (n_atom, 4)

    rigid_mask = []
    rigid_tensor7 = []
    for i, atom_id_set in enumerate(frame_atom_ids):
        atom_match = (atom_ids[..., None, :] == atom_id_set[None]).all(axis=-1)  # shape (n_atom, 3)
        frame_exists = atom_match.any(axis=0).all()
        rigid_mask.append(frame_exists)
        if frame_exists:
            frame_atom_coords = atoms["coords"][..., None, :] * atom_match[..., None]
            frame_atom_coords = frame_atom_coords.sum(axis=0)
            # print(frame_atom_coords)
            # TODO: i'm not sure why i need to do this, this is really jenk...
            if i == 0:
                frame_rot, frame_trans = compute_frame(
                    frame_atom_coords[0],
                    frame_atom_coords[1],
                    frame_atom_coords[2],
                )
                rigid = ru.Rigid(
                    rots=ru.Rotation(torch.as_tensor(frame_rot)),
                    trans=torch.as_tensor(frame_trans)
                )
            else:
                rigid = ru.Rigid.from_3_points(
                    torch.as_tensor(frame_atom_coords[0]),
                    torch.as_tensor(frame_atom_coords[1]),
                    torch.as_tensor(frame_atom_coords[2]),
                )
            tensor7 = rigid.to_tensor_7().numpy(force=True)
            rigid_tensor7.append(tensor7)
        else:
            rigid_tensor7.append(IDENTITY_TENSOR7.copy())

    return np.stack(rigid_tensor7, axis=0), np.array(rigid_mask), dummy_rigid_idx


def is_colinear(point1, point2, point3, tol=1e-2):
    """ Check if three points are colinear"""
    v1 = point1 - point2
    v2 = point3 - point1
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    e2 = v2 / (np.linalg.norm(v2) + 1e-10)
    e3 = np.cross(e1, e2)
    return np.linalg.norm(e3) < tol


def select_axes(atom_coord, neighbors, valid_neighbors, valid_neighbor_coords):
    np.random.shuffle(neighbors)
    for i, neighbor1 in enumerate(neighbors[:-1]):
        for neighbor2 in neighbors[i+1:]:
            coord1 = valid_neighbor_coords[valid_neighbors.index(neighbor1)]
            coord2 = valid_neighbor_coords[valid_neighbors.index(neighbor2)]
            if not is_colinear(coord1, atom_coord, coord2):
                return (neighbor1, neighbor2)
    return None

def gen_rand_rot_frame(trans):
    quat = Rotation.random().as_quat(canonical=True)
    return np.concatenate([quat, trans], axis=0), 0

def gen_semirand_rot_frame(center, x_axis_point):
    x_axis = x_axis_point - center
    x_axis = x_axis / np.linalg.norm(x_axis + 1e-6)
    # sample y vecs until we get one which is suitable to make an axis from
    while True:
        y_vec = np.random.randn(3)
        y_vec = y_vec / np.linalg.norm(y_vec)
        if np.dot(x_axis, y_vec) < 1 - 1e-6:
            break
    y_axis = y_vec - np.dot(x_axis, y_vec) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    y_axis_point = y_axis + center
    rot, trans = compute_frame(
        x_axis_point,
        center,
        y_axis_point,
    )
    quat = rot_to_quat(torch.as_tensor(rot)).numpy(force=True)
    tensor7 = np.concatenate([quat, trans], axis=-1)
    if np.isnan(tensor7).any():
        raise ValueError("encountered nan in computing semi-random rotation")
    return tensor7, 1

def gen_det_rot_frame(center, point1, point2):
    rot, trans = compute_frame(
        point1,
        center,
        point2,
    )
    quat = rot_to_quat(torch.as_tensor(rot)).numpy(force=True)
    tensor7 = np.concatenate([quat, trans], axis=-1)
    return tensor7, 2


def arbitrary_atom_to_frame(
    atom,
    atom_idx,
    valid_neighbors: list[int],
    valid_neighbor_coords: np.ndarray,
    neighbor_graph: nx.Graph
):
    """ For an arbitrary atom, compute a frame to use for that atom. This function will try its best to
    construct a frame from real axes given the particular input chemical graph. If it cannot, it'll either
    construct a semi-random rotation (one axis is fixed) or sample a random rotation for the frame.

    Parameters
    ==========
    atom: nomadflow.boltz.data.types.Atom
        Data about the atom to featurize
    atom_idx: int
        The identifier of `atom` in this particular chemical subgraph.
    valid_neighbors: list[int]
        A list of atom ids which are valid connections to `atom` to use for frame construction.
    valid_neighbor_coords: np.ndarray
        The coordinates for every atom in `valid_neighbors`.
    neighbor_graph: nx.Graph
        A networkX graph which specifies the connectivity between atoms in this particular chemical subgraph.

    Returns
    =======
    tensor7: np.ndarray
        The frame for the atom, represented in tensor7 format.
    num_real_input_axes: int
        The number of real axes used to construct the frame. Possible values {0, 1, 2}.
    """
    if not atom["is_present"]:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0

    # atom_idx = atom["atom_idx"]
    # print(atom_idx, neighbor_graph.nodes)
    if atom_idx in neighbor_graph.nodes:
        neighbors = [n for n in neighbor_graph.neighbors(atom_idx) if n in valid_neighbors]
    else:
        atom_name = atom['name']
        atom_name = [chr(c + 32) for c in atom_name if c != 0]
        atom_name = "".join(atom_name)
        # print(atom, atom_name)
        neighbors = []

    _select_axes = fn.partial(select_axes, valid_neighbors=valid_neighbors, valid_neighbor_coords=valid_neighbor_coords)

    if len(neighbors) == 0:
        quat = Rotation.random().as_quat(canonical=True)
        trans = atom["coords"]
        return np.concatenate([quat, trans], axis=0), 0
    elif len(neighbors) == 1:
        neighbor_idx = neighbors[0]
        neighbor_neighbors = [n for n in neighbor_graph.neighbors(neighbor_idx) if n in valid_neighbors and n != atom_idx]
        # print(atom, atom_idx, neighbor_idx, neighbor_neighbors, list(neighbor_graph.edges))
        if len(neighbor_neighbors) > 0:
            # if we can get a second hop neighbor to define the frame, use that
            axes = _select_axes(atom["coords"], [neighbor_idx] + neighbor_neighbors)
            if axes is not None:
                # print(atom, 2)
                return gen_det_rot_frame(atom['coords'], *axes)
        try:
            neighbor_coord = valid_neighbor_coords[valid_neighbors.index(neighbor_idx)]
            # print(atom, 1)
            return gen_semirand_rot_frame(atom['coords'], neighbor_coord)
        except Exception as e:
            print(f"Caught exception '{e}', replacing with random rotation")
            # print(atom, 0)
            return gen_rand_rot_frame(atom['coords'])
    else:
        axes = _select_axes(atom["coords"], neighbors)
        if axes is not None:
            return gen_det_rot_frame(atom["coords"], *axes)

        for neighbor_idx in neighbors:
            try:
                neighbor_coord = valid_neighbor_coords[valid_neighbors.index(neighbor_idx)]
                return gen_semirand_rot_frame(atom['coords'], neighbor_coord)
            except Exception:
                pass
        print("Error in featurizing rotation, replacing with random rotation")
        return gen_rand_rot_frame(atom['coords'])



class StructureTokenizer:
    """ A class object for tokenization, mainly to help code organization"""
    def __init__(
        self,
        struct: Structure,
        task_data: dict[str, np.ndarray],
        shuffle_chains: bool = False,
        shuffle_copied_fragments: bool = True,
        np_rng=None,
    ):
        self.struct = struct
        self.task_data = task_data
        self.shuffle_chains = shuffle_chains
        self.shuffle_copied_fragments = shuffle_copied_fragments
        self.np_rng = np_rng
        self.bond_graph = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in struct.bonds])

        self.token_idx = 0
        self.rigid_idx = 0
        self.atom_to_rigid = {}
        self.rigid_to_token = {}

        self.copy_data = []
        self.atomized_bond_store = {}

        self.processed = False

    def _get_standard_protein_residue_data(
        self,
        chain,
        res,
        noise_seq,
        *,
        residue_array_idx: int,
    ):
        # Get atom indices
        atom_start = res["atom_idx"]
        atom_end = res["atom_idx"] + res["atom_num"]
        # store returned rigids
        ret_rigids = []
        # Token is present if centers are
        is_present = res["is_present"]

        # If protein, compute frame, only used for templates
        rigid_tensor7 = np.stack([IDENTITY_TENSOR7.copy() for _ in range(3)], axis=0)
        rigid_mask = np.zeros(3, dtype=bool)

        # Get frame atoms
        atoms = self.struct.atoms[atom_start:atom_end]
        atom_noising_mask = self.task_data['atom_noising_mask']
        noise_atoms = atom_noising_mask[atom_start:atom_end]
        # get residue frames
        rigid_tensor7, rigid_mask, dummy_rigid_idx = standard_residue_to_frames(
            res, atoms
        )

        mg = self.task_data.get("motif_group_id_residue")
        if mg is None:
            motif_gid = 0
        else:
            motif_gid = int(mg[int(residue_array_idx)])
        mg_rigid = self.task_data.get("motif_group_id_rigid_by_residue_frame")
        if mg_rigid is None:
            motif_rigid_gids = np.full(3, motif_gid, dtype=np.int32)
        else:
            motif_rigid_gids = np.asarray(
                mg_rigid[int(residue_array_idx)],
                dtype=np.int32,
            )

        # Create token
        token = TokenData(
            token_idx=self.token_idx,
            rigid_idx=self.rigid_idx,
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
            seq_noising_mask=noise_seq,
            motif_group_id=motif_gid,
        )

        # compute a mapping for which rigids correspond to which atoms
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
        # update the global atom_to_rigid mapping
        for i, atom in enumerate(atoms):
            atom_name = atom['name']
            atom_name = [chr(c + 32) for c in atom_name if c != 0]
            atom_name = "".join(atom_name)
            self.atom_to_rigid[atom_start + i] = self.rigid_idx + atom_name_to_cg_idx[atom_name]

        _noise_rigid : List[bool] = []
        # Update rigid_idx to token_idx
        for i in range(3):
            cg_atom_idxs = cg_idx_to_atom_idx[i]
            # we figure out if we're noising the rigid
            # by checking if any of its component atoms are being noised
            # we also need to check if its a dummy rigid
            if len(cg_atom_idxs) > 0:
                noise_rigid = bool(noise_atoms[cg_atom_idxs].any())
            else:
                # if rigid is a dummy rigid, copy the noising status of its corresponding non-dummy rigid
                noise_rigid = _noise_rigid[dummy_rigid_idx[i]]
            _noise_rigid.append(noise_rigid)

            rigid = RigidData(
                rigid_idx=self.rigid_idx,
                token_idx=self.token_idx,
                sidechain_idx=i,
                is_atomized=False,
                element=-1,
                charge=0,
                tensor7=rigid_tensor7[i],
                is_present=rigid_mask[i],
                rigids_noising_mask=noise_rigid,
                num_real_input_axes=2,
                motif_group_id=int(motif_rigid_gids[i]),
            )
            ret_rigids.append(rigid)
            self.rigid_to_token[self.rigid_idx] = self.token_idx

            self.rigid_idx = self.rigid_idx + 1
        self.token_idx = self.token_idx + 1
        return token, ret_rigids

    def _get_nonstandard_residue_data(
        self,
        chain,
        res,
        bond_graph_override=None,
        process_isolated=False,
    ):
        if bond_graph_override is None:
            bond_graph_override = self.bond_graph

        # when we copy an atomized residue
        # we don't want to leak information about
        # residue positioning based on the current chain index
        # so we have the option to process this residue
        # independent of the current tokenizer state
        if process_isolated:
            token_idx = 0
            rigid_idx = 0
            atom_to_rigid = {}
            rigid_to_token = {}
        else:
            token_idx = self.token_idx
            rigid_idx = self.rigid_idx
            atom_to_rigid = self.atom_to_rigid
            rigid_to_token = self.rigid_to_token

        ret_tokens = []
        ret_rigids = []
        # Get atom indices
        atom_start = res["atom_idx"]
        atom_end = res["atom_idx"] + res["atom_num"]

        # We use the unk protein token as res_type
        unk_token = const.unk_token["PROTEIN"]
        unk_id = const.token_ids[unk_token]

        # Get atom coordinates
        atom_data = self.struct.atoms[atom_start:atom_end]
        atom_coords = atom_data["coords"]
        atom_noising_mask = self.task_data['atom_noising_mask']

        valid_neighbors = [i for i in range(res["atom_num"]) if atom_data[i]["is_present"]]
        valid_neighbor_coords = atom_coords[np.array(valid_neighbors, dtype=int)]

        # Tokenize each atom
        for i, atom in enumerate(atom_data):
            # Token is present if atom is
            is_present = res["is_present"] & atom["is_present"]
            atom_idx = atom_start + i
            # print(bond_graph_override.edges.data(), i)

            atom_tensor7, num_real_input_axes = arbitrary_atom_to_frame(
                atom,
                atom_idx if not process_isolated else i,
                valid_neighbors,
                valid_neighbor_coords,
                bond_graph_override
            )

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
            ret_tokens.append(token)

            # Update atom_idx to token_idx
            atom_to_rigid[atom_start + i] = rigid_idx
            rigid = RigidData(
                rigid_idx=rigid_idx,
                token_idx=token_idx,
                sidechain_idx=0,
                is_atomized=True,
                element=atom["element"],
                charge=atom["charge"],
                tensor7=atom_tensor7,
                is_present=atom["is_present"],
                rigids_noising_mask=bool(atom_noising_mask[atom_start + i]),
                num_real_input_axes=num_real_input_axes,
                motif_group_id=0,
            )
            ret_rigids.append(rigid)
            rigid_to_token[rigid_idx] = token_idx

            rigid_idx = rigid_idx + 1
            token_idx = token_idx + 1

        if not process_isolated:
            self.token_idx = token_idx
            self.rigid_idx = rigid_idx

        return ret_tokens, ret_rigids

    def _append_copy_features(
        self,
        chain,
        res,
        token,
        rigids,
        *,
        residue_array_idx: int,
    ):
        copy_indexed_residue_mask = self.task_data['copy_indexed_residue_mask']
        copy_unindexed_residue_mask = self.task_data['copy_unindexed_residue_mask']
        copy_atomized_residue_mask = self.task_data['copy_atomized_residue_mask']
        is_indexed = copy_indexed_residue_mask[residue_array_idx]
        is_unindexed = copy_unindexed_residue_mask[residue_array_idx]
        assert not is_indexed & is_unindexed, (
            f"residue copy of array_idx={residue_array_idx} "
            f"(local res_idx={res['res_idx']}) cannot be both indexed and unindexed!"
        )

        # if we're not copying anything, return
        if not (is_indexed | is_unindexed):
            return

        if copy_atomized_residue_mask[residue_array_idx]:
            aa_bond_data = get_standard_protein_residue_bonds(res['name'], atom_idx=0)
            _bond_graph_override = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in aa_bond_data])
            copy_tokens, copy_rigids = self._get_nonstandard_residue_data(
                chain, res,
                bond_graph_override=_bond_graph_override,
                process_isolated=True
            )

            self.atomized_bond_store[(chain['asym_id'], res['res_idx'])] = aa_bond_data

            for i, (t, r) in enumerate(zip(copy_tokens, copy_rigids)):
                _t = copy.deepcopy(t)
                _t = replace(_t, is_unindexed = not bool(is_indexed))
                _t = replace(_t, is_atomized = True)
                self.copy_data.append(
                    {"token": _t, "rigids": [r], "res_internal_idx": i}
                )
        else:
            copy_token = copy.deepcopy(token)
            copy_token = replace(copy_token, is_unindexed = not bool(is_indexed))
            copy_token = replace(copy_token, is_atomized = False)
            self.copy_data.append({
                "token": copy_token, "rigids": rigids, "res_internal_idx": 0
            })

    def _shuffle_copy_fragments(
        self,
        copy_data
    ):
        # we swap around the order of copied segments
        # since the ordering of these segments in the model inputs
        # could potentially serve to leak the ground truth ordering of these segments
        frag_idx = 0
        token_0 = asdict(copy_data[0]['token'])
        last_res_idx = token_0['res_idx']
        last_chain_id = token_0['asym_id']
        frag_mapping = {}
        for i, copy_dict in enumerate(copy_data):
            token = asdict(copy_dict['token'])
            chain_id = token['asym_id']
            if abs(last_res_idx - token['res_idx']) > 1 or chain_id != last_chain_id:
                frag_idx += 1

            if frag_idx not in frag_mapping:
                frag_mapping[frag_idx] = [copy_dict]
            else:
                frag_mapping[frag_idx].append(copy_dict)

            last_res_idx = token['res_idx']
            last_chain_id = chain_id

        rng = self.np_rng if self.np_rng is not None else np.random
        frag_order = rng.permutation(frag_idx+1)
        _new_copy_data = []
        for i in frag_order:
            _new_copy_data.extend(frag_mapping[i])
        return _new_copy_data

    def tokenize(
        self,
    ):
        assert not self.processed, (
            "im surprised you got here, "
            "this is a one-time use function, and we have already tokenized this data. "
            "please recreate this object if you wish to re-tokenize this data"
        )
        # Create token data and rigid data
        token_data = []
        rigid_data = []
        # Create token bonds
        token_bonds = []

        res_type_noising_mask = self.task_data['res_type_noising_mask']

        # Filter to valid chains only
        chains = self.struct.chains[self.struct.mask]

        if self.shuffle_chains:
            np.random.shuffle(chains)

        for chain in chains:
            # Get residue indices
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

            for i, res in enumerate(self.struct.residues[res_start:res_end]):
                residue_array_idx = int(res_start + i)
                # Standard residues are tokens
                if res["is_standard"] and (res['name'] != 'UNK') and is_protein:
                    token, ret_rigids = self._get_standard_protein_residue_data(
                        chain, res,
                        noise_seq=res_type_noising_mask[residue_array_idx],
                        residue_array_idx=residue_array_idx,
                    )

                    self._append_copy_features(
                        chain, res, token, ret_rigids,
                        residue_array_idx=residue_array_idx,
                    )

                    token = replace(token, seq_noising_mask=True, motif_group_id=0)
                    ret_rigids = [
                        replace(r, rigids_noising_mask=True, motif_group_id=0)
                        for r in ret_rigids
                    ]

                    token_data.append(astuple(token))
                    rigid_data.extend([astuple(r) for r in ret_rigids])

                # Non-standard are tokenized per atom
                else:
                    ret_tokens, ret_rigids = self._get_nonstandard_residue_data(
                        chain, res,
                    )
                    token_data.extend([astuple(t) for t in ret_tokens])
                    rigid_data.extend([astuple(r) for r in ret_rigids])

        if self.shuffle_copied_fragments and len(self.copy_data) > 0:
            copy_data = self._shuffle_copy_fragments(self.copy_data)
        else:
            copy_data = self.copy_data

        copy_atomized_bonds = []
        # append our copied residues onto the tokenized data
        # this preserves res_idx but assigns the correct token_idx and rigid_idx
        # as well as specifying proper bookkeeping values
        for copy_dict in copy_data:
            # copy_dict = copy.deepcopy(copy_dict)  # TODO: idek if this is necessary
            token = asdict(copy_dict['token'])
            rigids = [asdict(r) for r in copy_dict['rigids']]

            token['is_copy'] = True
            token['token_idx'] = self.token_idx
            token['rigid_idx'] = self.rigid_idx
            token_data.append(astuple(TokenData(**token)))

            token_res_tag = (token['asym_id'], token['res_idx'])
            if token_res_tag in self.atomized_bond_store and copy_dict['res_internal_idx'] == 0:
                copy_bond_data = self.atomized_bond_store[token_res_tag]
                copy_bond_data['atom_1'] += self.token_idx
                copy_bond_data['atom_2'] += self.token_idx
                copy_bond_data['type'] += 1
                copy_atomized_bonds.append(copy_bond_data.astype(TokenBond))

            for r in rigids:
                r['token_idx'] = self.token_idx
                r['rigid_idx'] = self.rigid_idx
                rigid_data.append(astuple(RigidData(**r)))
                self.rigid_idx += 1

            self.token_idx += 1

        # Add atom-atom bonds from ligands
        for bond in self.struct.bonds:
            atom1 = bond["atom_1"]
            atom2 = bond["atom_2"]
            if atom1 not in self.atom_to_rigid or atom2 not in self.atom_to_rigid:
                continue
            rigid1 = self.atom_to_rigid[atom1]
            rigid2 = self.atom_to_rigid[atom2]
            if rigid1 not in self.rigid_to_token or rigid2 not in self.rigid_to_token:
                continue
            token_bond = (
                self.rigid_to_token[rigid1],
                self.rigid_to_token[rigid2],
                bond["type"] + 1,
            )
            token_bonds.append(token_bond)

        token_data = np.array(token_data, dtype=Token)
        token_bonds = np.array(token_bonds, dtype=TokenBond)
        rigid_data = np.array(rigid_data, dtype=Rigid)

        if len(copy_atomized_bonds) > 0:
            token_bonds = np.concatenate([token_bonds] + copy_atomized_bonds)

        # mark that we've run this function already
        self.processed = True
        return token_data, rigid_data, token_bonds


def tokenize_structure(  # noqa: C901, PLR0915
    struct: Structure,
    task_data: dict[str, np.ndarray],
    shuffle_chains: bool = False,
    shuffle_copied_fragments: bool = True,
    np_rng=None,
) -> Tuple[np.ndarray, np.array, np.ndarray]:
    """Tokenize a structure.

    Parameters
    ----------
    struct : Structure
        The structure to tokenize.

    Returns
    -------
    np.ndarray
        The tokenized data.
    np.ndarray
        The rigids data.
    np.ndarray
        The tokenized bonds.

    """

    tokenizer = StructureTokenizer(
        struct,
        task_data,
        shuffle_chains,
        shuffle_copied_fragments,
        np_rng=np_rng,
    )
    return tokenizer.tokenize()

