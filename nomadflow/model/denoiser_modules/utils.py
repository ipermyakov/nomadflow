import torch


def gather_helper(tensor, token_gather_idx):
    new_dims = tensor.dim() - token_gather_idx.dim()
    idx_expand = token_gather_idx.view(
        *token_gather_idx.shape, *[1 for _ in range(new_dims)]
    ).expand(
        *[-1 for _ in token_gather_idx.shape],
        *tensor.shape[-new_dims:]
    ).long()
    return torch.gather(
        tensor,
        1,
        idx_expand
    )
