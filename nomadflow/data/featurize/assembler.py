""" Assemble input features to nomadflow from Tokenized objects """
import random
from typing import Optional
from dataclasses import asdict

import numpy as np
import torch
from torch import Tensor, from_numpy
import torch.nn.functional as F
from torch.utils.data import default_collate

from nomadflow.boltz.data import const
from nomadflow.boltz.data.pad import pad_dim

from nomadflow.data.featurize.tokenize import Tokenized

# Tasks without early lock omit motif_t_lock; collate still stacks a scalar per sample.
MOTIF_T_LOCK_DISABLED = -1.0


def process_token_features(
    data: Tokenized,
    max_tokens: Optional[int] = None,
) -> dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, Tensor]
        The token features.

    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    seq_index_ref = {}
    curr_seq_idx = 0
    seq_index = []
    res_index = []
    max_res_idx = token_data['res_idx'].max()

    token_to_rep_rigid = []
    rigid_idx = 0
    for _, token in enumerate(token_data):
        res_idx = token['res_idx']
        if token['is_unindexed']:
            seq_index.append(curr_seq_idx)
            curr_seq_idx += 1
            res_index.append(
                max(*res_index, max_res_idx) + 1
            )
        elif res_idx not in seq_index_ref:
            seq_index_ref[res_idx] = curr_seq_idx
            seq_index.append(curr_seq_idx)
            curr_seq_idx += 1
            res_index.append(res_idx)
        else:
            seq_index.append(seq_index_ref[res_idx])
            res_index.append(res_idx)

        token_to_rep_rigid.append(rigid_idx)
        rigid_idx += token['rigid_num']


    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = torch.as_tensor(res_index).long()
    # to avoid exceeding our sinusoidal embedding capacity with really large residue indices
    residue_index = residue_index - residue_index.min()

    asym_id = from_numpy(token_data["asym_id"].copy()).long()
    entity_id = from_numpy(token_data["entity_id"].copy()).long()
    sym_id = from_numpy(token_data["sym_id"].copy()).long()
    mol_type = from_numpy(token_data["mol_type"].copy()).long()
    res_type = from_numpy(token_data["res_type"].copy()).long()
    token_to_rep_rigid = torch.as_tensor(token_to_rep_rigid).long()
    seq_index = torch.as_tensor(seq_index, dtype=torch.long)

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"].copy()).float()
    is_copy_token_mask = from_numpy(token_data['is_copy'].copy()).bool()
    token_is_unindexed_mask = from_numpy(token_data['is_unindexed'].copy()).bool()
    seq_noising_mask = from_numpy(token_data["seq_noising_mask"].copy()).bool()
    # is_copy_token_mask = from_numpy(is_copy_token_mask.copy()).bool()
    # token_is_unindexed_mask = from_numpy(token_is_unindexed_mask.copy()).bool()

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens = len(token_data)

    # TODO: this does not preserve bonds for copied residues
    # i think this is not a problem for now but might need to consider it in the future
    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = token_bond["type"]
        bonds[token_2, token_1] = token_bond["type"]

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            seq_index = pad_dim(seq_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            token_to_rep_rigid = pad_dim(token_to_rep_rigid, 0, pad_len)
            seq_noising_mask = pad_dim(seq_noising_mask, 0, pad_len)

    token_features = {
        "token_idx": token_index,
        "token_seq_idx": seq_index,
        "residue_idx": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "token_to_rep_rigid": token_to_rep_rigid,
        "token_bonds": bonds.long(),
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_is_copy_mask": is_copy_token_mask,
        "token_is_unindexed_mask": token_is_unindexed_mask,
        "seq_noising_mask": seq_noising_mask,
        # aggregate / renamed masks for convenience
        "token_mask": resolved_mask.bool() & pad_mask.bool(),
        "seq": res_type,
    }
    return token_features


def process_rigid_features(
    data: Tokenized,
    rigids_per_window_queries: int = 16,
    max_rigids: Optional[int] = None,
) -> dict[str, Tensor]:
    """Get the rigid features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    rigids_per_window_queries: int
        Number of rigids per query window. Output features will be padded to multiples of this number.
    max_rigids : int, optional
        The maximum number of rigids.

    Returns
    -------
    dict[str, Tensor]
        The rigid features.

    """
    # Filter to tokens' atoms
    token_data = data.tokens
    rigid_data = data.rigids
    rigid_to_token = []
    rigid_to_seq_idx = []

    seq_index_ref = {}
    curr_seq_idx = 0
    seq_index = []
    for _, token in enumerate(token_data):
        res_idx = token['res_idx']
        if token['is_unindexed']:
            seq_index.append(curr_seq_idx)
            curr_seq_idx += 1
        elif res_idx not in seq_index_ref:
            seq_index_ref[res_idx] = curr_seq_idx
            seq_index.append(curr_seq_idx)
            curr_seq_idx += 1
        else:
            seq_index.append(seq_index_ref[res_idx])
    token_to_seq_idx = np.array(seq_index)

    for token_id, token in enumerate(token_data):
        rigid_to_token.extend([token_id] * token["rigid_num"])
        rigid_to_seq_idx.extend([int(token_to_seq_idx[token_id])] * token["rigid_num"])
    rigid_to_seq_idx = np.array(rigid_to_seq_idx)

    # Compute features
    ref_element = from_numpy(rigid_data["element"].copy()).long()
    ref_charge = from_numpy(rigid_data["charge"].copy())
    sidechain_idx = from_numpy(rigid_data["sidechain_idx"].copy())
    tensor7 = from_numpy(rigid_data["tensor7"].copy())
    resolved_mask = from_numpy(rigid_data["is_present"].copy())
    is_atom_mask = from_numpy(rigid_data["is_atomized"].copy())
    pad_mask = torch.ones(len(rigid_data), dtype=torch.float)
    rigid_to_token = torch.tensor(rigid_to_token, dtype=torch.long)
    new_rigids_noising_mask = from_numpy(rigid_data['rigids_noising_mask'].copy()).bool()
    rigids_seq_idx = from_numpy(rigid_to_seq_idx.copy()).long()
    if "motif_group_id" in rigid_data.dtype.names:
        motif_group_id = from_numpy(rigid_data["motif_group_id"].copy()).long()
    else:
        motif_group_id = torch.zeros(len(rigid_data), dtype=torch.long)
    if "is_motif_group_aggregator" in rigid_data.dtype.names:
        is_motif_group_aggregator = from_numpy(rigid_data["is_motif_group_aggregator"].copy()).bool()
    else:
        is_motif_group_aggregator = torch.zeros(len(rigid_data), dtype=torch.bool)

    # Compute padding and apply
    if max_rigids is not None:
        assert max_rigids % rigids_per_window_queries == 0
        pad_len = max_rigids - len(rigid_data)
    else:
        pad_len = (
            (len(rigid_data) - 1) // rigids_per_window_queries + 1
        ) * rigids_per_window_queries - len(rigid_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        rigid_to_token = pad_dim(rigid_to_token, 0, pad_len)
        sidechain_idx = pad_dim(sidechain_idx, 0, pad_len)
        new_rigids_noising_mask = pad_dim(new_rigids_noising_mask, 0, pad_len, value=True)
        rigids_seq_idx = pad_dim(rigids_seq_idx, 0, pad_len)
        motif_group_id = pad_dim(motif_group_id, 0, pad_len)
        is_motif_group_aggregator = pad_dim(is_motif_group_aggregator, 0, pad_len)

        tensor7_pad = torch.zeros((pad_len, 7), device=tensor7.device, dtype=tensor7.dtype)
        tensor7_pad[:, 0] = 1
        tensor7 = torch.cat([
            tensor7, tensor7_pad
        ], dim=0)

    return {
        "rigids_seq_idx": rigids_seq_idx,
        "rigids_noising_mask": new_rigids_noising_mask,
        "rigids_sidechain_idx": sidechain_idx.long(),
        "rigids_is_atom_mask": is_atom_mask,
        "rigids_ref_element": ref_element.long(),
        "rigids_ref_charge": ref_charge.float(),
        "rigids_resolved_mask": resolved_mask.bool(),
        "rigids_pad_mask": pad_mask.bool(),
        "rigids_to_token": rigid_to_token,
        "motif_group_id": motif_group_id,
        "rigids_is_motif_group_aggregator": is_motif_group_aggregator,
        "rigids_1": tensor7,
        # aggregate mask used in the model
        "rigids_mask": resolved_mask.bool() & pad_mask.bool()
    }


# TODO: i should fuse this and featurize_inference
def featurize_training(
    data: Tokenized,
    task_data,
    max_tokens: Optional[int] = None,
    max_rigids: Optional[int] = None,
    rigids_per_window_queries: int = 16,
):
    token_features = process_token_features(
        data,
        max_tokens
    )
    rigid_features = process_rigid_features(
        data,
        rigids_per_window_queries,
        max_rigids
    )

    out = {
        "t": task_data['t'],
        "multi_motif_se3": torch.as_tensor(
            bool(task_data.get("multi_motif_se3", False)),
            dtype=torch.bool,
        ),
        "multi_motif_se3_trans_std": torch.as_tensor(
            float(task_data.get("multi_motif_se3_trans_std", 16.0)),
            dtype=torch.float32,
        ),
        "token": token_features,
        "rigids": rigid_features,
        "motif_t_lock": torch.as_tensor(
            float(task_data["motif_t_lock"])
            if task_data.get("motif_t_lock") is not None
            else MOTIF_T_LOCK_DISABLED,
            dtype=torch.float32,
        ),
    }
    return out

def featurize_inference(
    data: Tokenized,
    task_data,
    task_name: Optional[str] = 'sample',
    max_tokens: Optional[int] = None,
    max_rigids: Optional[int] = None,
    rigids_per_window_queries: int = 16,
    task_metadata: Optional[dict] = None,
    sample_metadata: Optional[dict] = None,
):
    token_features = process_token_features(
        data,
        max_tokens
    )
    rigid_features = process_rigid_features(
        data,
        rigids_per_window_queries,
        max_rigids
    )

    out = {
        "t": task_data['t'],
        "multi_motif_se3": torch.as_tensor(
            bool(task_data.get("multi_motif_se3", False)),
            dtype=torch.bool,
        ),
        "multi_motif_se3_trans_std": torch.as_tensor(
            float(task_data.get("multi_motif_se3_trans_std", 16.0)),
            dtype=torch.float32,
        ),
        "task": task_name,
        "input_data": data,
        "token": token_features,
        "rigids": rigid_features,
        "motif_t_lock": torch.as_tensor(
            float(task_data["motif_t_lock"])
            if task_data.get("motif_t_lock") is not None
            else MOTIF_T_LOCK_DISABLED,
            dtype=torch.float32,
        ),
    }
    if task_metadata is not None:
        out["task_metadata"] = task_metadata
    if sample_metadata is not None:
        out["sample_metadata"] = sample_metadata
    return out


def collate(data_list):
    def pad(data, token_max_len, rigids_max_len):
        padded_data = {
            't': data['t'],
            'multi_motif_se3': data.get(
                'multi_motif_se3',
                torch.as_tensor(False, dtype=torch.bool),
            ),
            'multi_motif_se3_trans_std': data.get(
                'multi_motif_se3_trans_std',
                torch.as_tensor(16.0, dtype=torch.float32),
            ),
            'token': {},
            'rigids': {},
            'motif_t_lock': data.get(
                'motif_t_lock',
                torch.tensor(MOTIF_T_LOCK_DISABLED, dtype=torch.float32),
            ),
        }
        ## pad token data
        for key, value in data['token'].items():
            pad_len = token_max_len - value.shape[0]
            if not isinstance(value, torch.Tensor):
                print(key, value)
                raise ValueError(f"key {key} is associated with value {value} which is not a torch tensor")
            if key == 'token_bonds':
                padded_data['token'][key] = F.pad(value, (0, pad_len, 0, pad_len))
            elif value.dim() == 1:
                padded_data['token'][key] = F.pad(value, (0, pad_len))
            elif value.dim() == 2:
                padded_data['token'][key] = F.pad(value, (0, 0, 0, pad_len))
            elif value.dim() == 3:
                # for now this should be the only possibility
                assert key == 'token_bonds'
                padded_data['token'][key] = F.pad(value, (0, pad_len, 0, pad_len))
            else:
                raise ValueError(f"we ran into a token input with dim not in (1, 2, 3): {key} with dim {value.dim()}")

        ## pad rigid data
        for key, value in data['rigids'].items():
            pad_len = rigids_max_len - value.shape[0]
            if value.dim() == 1:
                padded_data['rigids'][key] = F.pad(value, (0, pad_len))
            elif value.dim() == 2:
                padded_data['rigids'][key] = F.pad(value, (0, 0, 0, pad_len))
            else:
                raise ValueError(f"we ran into a rigid input with dim not in (1, 2): {key} with dim {value.dim()}")

            # we're padding tensor_7s so we need to ensure the quat isn't 0
            if key in ('rigids_t', 'rigids_1'):
                if pad_len > 0:
                    padded_data['rigids'][key][..., -pad_len:, 0] = 1

        return padded_data

    token_lens = []
    rigids_lens = []
    for data in data_list:
        token_lens.append(
            data['token']['token_idx'].numel()
        )
        rigids_lens.append(
            data['rigids']['rigids_seq_idx'].numel()
        )
    padded_data_list = [
        pad(data, max(token_lens), max(rigids_lens))
        for data in data_list
    ]
    batched_data_list = default_collate(padded_data_list)
    batched_data_list['task'] = [data['task'] for data in data_list]
    if 'name' in data_list[0]:
        batched_data_list['name'] = [data['name'] for data in data_list]
    if 'val_seed' in data_list[0]:
        batched_data_list['val_seed'] = default_collate([data['val_seed'] for data in data_list])
    if 'input_data' in data_list[0]:
        batched_data_list['input_data'] = [asdict(data['input_data']) for data in data_list]
    if 'task_metadata' in data_list[0]:
        batched_data_list['task_metadata'] = [data['task_metadata'] for data in data_list]
    if 'sample_metadata' in data_list[0]:
        batched_data_list['sample_metadata'] = [data['sample_metadata'] for data in data_list]
    return batched_data_list
