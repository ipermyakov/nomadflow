import torch


def motif_pair_state(motif_group_id_q: torch.Tensor, motif_group_id_k: torch.Tensor) -> torch.Tensor:
    """Map motif-group ids to pair-state classes.

    Pair-state IDs:
      0 = both not motif
      1 = one motif
      2 = same motif group
      3 = different motif groups
    """
    motif_i = motif_group_id_q[..., None]
    motif_j = motif_group_id_k[..., None, :]
    motif_i_pos = motif_i > 0
    motif_j_pos = motif_j > 0
    both_motif = motif_i_pos & motif_j_pos
    one_motif = motif_i_pos ^ motif_j_pos
    same_group = both_motif & (motif_i == motif_j)
    different_group = both_motif & (motif_i != motif_j)
    pair_state = torch.zeros_like(motif_i + motif_j, dtype=torch.long)
    pair_state = pair_state + one_motif.long()
    pair_state = torch.where(same_group, torch.full_like(pair_state, 2), pair_state)
    pair_state = torch.where(different_group, torch.full_like(pair_state, 3), pair_state)
    return pair_state
