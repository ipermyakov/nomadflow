"""Utilities for calculating all atom representations."""
import torch
from nomadflow.openfold.utils import rigid_utils as ru
from nomadflow.openfold.utils.tensor_utils import batched_gather
from nomadflow.openfold.data import residue_constants
from nomadflow.data.constants import coarse_grain


Rigid = ru.Rigid
Rotation = ru.Rotation

# Residue Constants from OpenFold/AlphaFold2.
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)

def _gen_version_dict(module):
    return {
        "IDEALIZED_POS37": torch.tensor(module.restype_atom37_cg_group_positions),
        "IDEALIZED_POS37_MASK": torch.any(torch.tensor(module.restype_atom37_cg_group_positions), axis=-1),
        "IDEALIZED_POS": torch.tensor(module.restype_atom14_cg_group_positions),
        "DEFAULT_FRAMES": torch.tensor(module.restype_cg_group_default_frame),
        "GROUP_IDX": torch.tensor(module.restype_atom14_to_cg_group),
    }

cg_version_constants = {
    1: _gen_version_dict(coarse_grain),
}


def frames_to_atom14_pos(
        r: Rigid,
        aatype: torch.Tensor,
        cg_version: int=1
    ):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """
    GROUP_IDX = cg_version_constants[cg_version]["GROUP_IDX"]
    DEFAULT_FRAMES = cg_version_constants[cg_version]["DEFAULT_FRAMES"]
    IDEALIZED_POS = cg_version_constants[cg_version]["IDEALIZED_POS"]

    # [*, N, 14]
    group_mask = GROUP_IDX.to(aatype.device)[aatype, ...]

    # [*, N, 14, 4]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 14, 4]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    frame_atom_mask = ATOM_MASK.to(aatype.device)[aatype, ...].unsqueeze(-1).to(r.device)

    # [*, N, 14, 3]
    frame_null_pos = IDEALIZED_POS.to(aatype.device)[aatype, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions, frame_atom_mask


def compute_atom14_from_cg_frames(rigids, res_mask, seq, cg_version=1, return_atom_mask=False):
    rigids = ru.Rigid.cat([
        rigids[..., 0:1],
        rigids[..., 0:1],
        rigids[..., 1:],
    ], dim=-1)
    atom14_pos, atom14_mask = frames_to_atom14_pos(
        rigids,
        seq,
        cg_version
    )
    atom37_bb_pos = torch.zeros(rigids[..., 0].shape + (37, 3), device=atom14_pos.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37 = adjust_oxygen_pos(atom37_bb_pos.view(-1, 37, 3), res_mask.view(-1)).view(atom37_bb_pos.shape)
    atom14_pos[..., 3, :] = atom37[..., 4, :]

    if return_atom_mask:
        return atom14_pos, atom14_mask
    else:
        return atom14_pos


def compute_atom14_frames_from_cg_frames(rigids, res_mask, seq, cg_version=1):
    rigids_4 = ru.Rigid.cat([
        rigids[..., 0:1],
        rigids[..., 0:1],
        rigids[..., 1:],
    ], dim=-1)
    atom14_pos, _ = frames_to_atom14_pos(
        rigids_4,
        seq,
        cg_version
    )
    atom37_bb_pos = torch.zeros(rigids[..., 0].shape + (37, 3), device=atom14_pos.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37 = adjust_oxygen_pos(atom37_bb_pos.view(-1, 37, 3), res_mask.view(-1)).view(atom37_bb_pos.shape)
    atom14_pos[..., 3, :] = atom37[..., 4, :]

    GROUP_IDX = cg_version_constants[cg_version]["GROUP_IDX"].to(seq.device)
    atom14_mask = ATOM_MASK.to(seq.device)[seq].bool()
    rigid_quats = rigids_4.get_rots().get_quats()
    atom_quats = batched_gather(
        rigid_quats,
        GROUP_IDX[seq],
        1,
        1
    )
    atom_quats = atom_quats * atom14_mask[..., None]
    atom_quats[atom14_mask, 0] = 1

    atom_rigids = ru.Rigid(
        rots=ru.Rotation(quats=atom_quats),
        trans=atom14_pos,
    )

    return atom_rigids, atom14_mask


def adjust_oxygen_pos(
    atom_37: torch.Tensor, pos_is_known = None
) -> torch.Tensor:
    """
    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7
    )

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = (
        calpha_to_carbonyl_term + calpha_to_nitrogen_term
    )  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((atom_37.shape[0],), dtype=torch.int64, device=atom_37.device)

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0
    )  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = (
        atom_37[next_res_gone, 2, :]
        + carbonyl_to_oxygen_term[next_res_gone, :] * 1.23
    )

    return atom_37