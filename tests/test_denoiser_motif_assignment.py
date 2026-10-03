from __future__ import annotations

import torch

from nomadflow.model.denoiser import scaffold_candidate_mask


def test_scaffold_candidate_mask_excludes_copies_aggregators_and_padding():
    token_data = {
        "token_is_copy_mask": torch.tensor(
            [[False, True, False, False, False]]
        ),
        "token_mask": torch.tensor([[True, True, True, True, False]]),
        "token_to_rep_rigid": torch.tensor([[0, 1, 2, 3, 4]]),
    }
    rigids_data = {
        "rigids_is_motif_group_aggregator": torch.tensor(
            [[False, False, True, False, False]]
        )
    }

    mask = scaffold_candidate_mask(token_data, rigids_data)

    assert mask.tolist() == [[True, False, False, True, False]]


def test_scaffold_candidate_mask_supports_runs_without_aggregators():
    token_data = {
        "token_is_copy_mask": torch.tensor([[False, True, False]]),
        "token_mask": torch.tensor([[True, True, True]]),
        "token_to_rep_rigid": torch.tensor([[0, 1, 2]]),
    }

    mask = scaffold_candidate_mask(token_data, {})

    assert mask.tolist() == [[True, False, True]]
