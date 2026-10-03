import logging
import math
from functools import partial
import copy
import warnings
from dataclasses import replace
import os
import json
import time
from typing import Optional
from pathlib import Path

import numpy as np
import tqdm
import torch
import torch.distributed as dist
import tree
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from nomadflow.boltz.data import const

from nomadflow.openfold.data import residue_constants
from nomadflow.openfold.utils import rigid_utils as ru

# from nomadflow.data.write.mmcif import to_mmcif
from nomadflow.data.write.trajectory import (
    prediction_frame_to_tokenized,
    write_eval_tokenized_structure_pdb,
    write_multimodel_pdb_batched,
    write_multimodel_pdb,
    write_tokenized_structure_pdb,
)

from .utils import gen_pbar_str
from .ema import EMAModel

from .loss.multiframe import multiframe_fm_loss_dense_batch
from .loss.common import seq_losses_dense_batch
from nomadflow.stoch_interp.motif_time_schedule import (
    LinearMotifTimeSchedule,
    MOTIF_T_LOCK_DISABLED,
    effective_rigids_noising_mask,
    motif_t_lock_is_enabled,
    stash_base_rigids_noising_mask,
)
from nomadflow.benchmark.pmpnn_mapping import (
    apply_eval_sample_metadata,
    build_pmpnn_fixed_pos_dict,
)
from nomadflow.runtime.sampling.benchmark_metadata import (
    attach_output_chains_to_sample_metadata,
    finalize_motif_sample_metadata,
    normalize_eval_source_residue_keys,
)
from nomadflow.runtime.sampling.layout_manifest import export_layout_manifest


DEFAULT_SEQ_WEIGHT = {
    c: 1.0
    for c in 'ACDEFGHIKLMNPQRSTVWY'
}
DEFAULT_SEQ_WEIGHT['X'] = 0.
for c in ['C', 'E', 'H', 'P', 'Q', 'R', 'W']:
    DEFAULT_SEQ_WEIGHT[c] = 2.0

DEFAULT_RESTYPE_WEIGHT = {
    c: 1.0
    for c in const.tokens
}
for c in ['CYS', 'GLU', 'HIS', 'PRO', 'GLN', 'ARG', 'TRP']:
    DEFAULT_RESTYPE_WEIGHT[c] = 2.0


def _motif_time_schedule_for_corrupter(corrupter):
    schedule = getattr(corrupter, '_motif_time_schedule', None)
    if schedule is not None:
        return schedule
    return LinearMotifTimeSchedule()


def _base_rigids_noising_mask(rigids_data):
    return stash_base_rigids_noising_mask(rigids_data)


def apply_effective_rigids_noising_mask(batch, corrupter, t=None):
    """Apply motif_t_lock to ``rigids_noising_mask`` (motif + aggregator fixed when locked)."""
    rigids_data = batch['rigids']
    t = batch['t'] if t is None else t
    motif_t_lock = batch.get('motif_t_lock', getattr(corrupter, 'motif_t_lock', None))
    rigids_data['rigids_noising_mask'] = effective_rigids_noising_mask(
        _base_rigids_noising_mask(rigids_data),
        rigids_data.get('motif_group_id'),
        t,
        motif_t_lock,
        schedule=_motif_time_schedule_for_corrupter(corrupter),
    )


def _motif_group_motion_locked(batch, corrupter, t) -> bool:
    """True when integration time is past ``motif_t_lock`` (skip group churn / euler)."""
    motif_t_lock = batch.get('motif_t_lock', getattr(corrupter, 'motif_t_lock', None))
    if not motif_t_lock_is_enabled(motif_t_lock):
        return False
    schedule = _motif_time_schedule_for_corrupter(corrupter)
    if isinstance(t, torch.Tensor):
        t_eval = t.detach().to(dtype=torch.float32)
    else:
        t_eval = torch.tensor(float(t), dtype=torch.float32)
    if t_eval.ndim == 0:
        t_eval = t_eval.unsqueeze(0)
    return bool(schedule.is_locked(t_eval, motif_t_lock).all().item())


def _normalize_sampling_time(t, batch_size, device):
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    if t.ndim == 0:
        return t.repeat(batch_size).view(batch_size, 1)
    if t.ndim == 1:
        if t.shape[0] == 1:
            return t.repeat(batch_size).view(batch_size, 1)
        if t.shape[0] == batch_size:
            return t[:, None]
    if t.ndim == 2 and t.shape[0] == batch_size:
        return t
    raise ValueError(f"sampling time must be scalar, [B], or [B, K], got {tuple(t.shape)}")


def _motif_local_integration_time(batch, corrupter, t, d_t):
    """Dense per-rigid sampler time, using local motif time for aggregators."""
    rigids_data = batch['rigids']
    motif_group_id = rigids_data.get("motif_group_id")
    motif_t_lock = batch.get('motif_t_lock', getattr(corrupter, 'motif_t_lock', None))
    if motif_group_id is None or not motif_t_lock_is_enabled(motif_t_lock):
        return t, d_t

    group_id = motif_group_id.to(dtype=torch.long)
    device = group_id.device
    batch_size = group_id.shape[0]
    t = _normalize_sampling_time(t, batch_size, device)
    d_t = _normalize_sampling_time(d_t, batch_size, device)
    t_next = torch.clamp(t + d_t, max=1.0)

    schedule = _motif_time_schedule_for_corrupter(corrupter)
    motif_t = schedule.motif_interp_time(t, motif_t_lock)
    motif_t_next = schedule.motif_interp_time(t_next, motif_t_lock)
    motif_d_t = (motif_t_next - motif_t).clamp_min(0.0)

    t_lock_t = torch.as_tensor(motif_t_lock, device=device, dtype=torch.float32)
    while t_lock_t.ndim < t.ndim:
        t_lock_t = t_lock_t.unsqueeze(-1)
    lock_enabled = t_lock_t > MOTIF_T_LOCK_DISABLED
    locked = schedule.is_locked(t, motif_t_lock) & lock_enabled
    active = lock_enabled & ~locked & (motif_d_t > 0)

    aggregator_mask = rigids_data.get("rigids_is_motif_group_aggregator")
    if aggregator_mask is None:
        motif_local_mask = group_id > 0
    else:
        motif_local_mask = (group_id > 0) & aggregator_mask.to(device=device).bool()
    motif_local_mask = motif_local_mask & active.expand_as(motif_local_mask)

    global_t = t.expand_as(group_id).to(dtype=torch.float32)
    global_d_t = d_t.expand_as(group_id).to(dtype=torch.float32)
    local_t = motif_t.expand_as(group_id).to(dtype=torch.float32)
    local_d_t = motif_d_t.expand_as(group_id).to(dtype=torch.float32)
    return (
        torch.where(motif_local_mask, local_t, global_t),
        torch.where(motif_local_mask, local_d_t, global_d_t),
    )


def _motif_aggregator_step_size_scale(batch, corrupter, t, d_t, local_d_t):
    """Scale aggregator Euler step size back to the global integration interval."""
    rigids_data = batch['rigids']
    motif_group_id = rigids_data.get("motif_group_id")
    motif_t_lock = batch.get('motif_t_lock', getattr(corrupter, 'motif_t_lock', None))
    if motif_group_id is None or not motif_t_lock_is_enabled(motif_t_lock):
        return 1

    group_id = motif_group_id.to(dtype=torch.long)
    device = group_id.device
    batch_size = group_id.shape[0]
    t = _normalize_sampling_time(t, batch_size, device)
    d_t = _normalize_sampling_time(d_t, batch_size, device)

    local_d_t = torch.as_tensor(local_d_t, device=device, dtype=torch.float32)
    if local_d_t.ndim == 0:
        return 1
    local_d_t = local_d_t.expand_as(group_id)

    schedule = _motif_time_schedule_for_corrupter(corrupter)
    t_lock_t = torch.as_tensor(motif_t_lock, device=device, dtype=torch.float32)
    while t_lock_t.ndim < t.ndim:
        t_lock_t = t_lock_t.unsqueeze(-1)
    lock_enabled = t_lock_t > MOTIF_T_LOCK_DISABLED
    locked = schedule.is_locked(t, motif_t_lock) & lock_enabled
    active = lock_enabled & ~locked & (local_d_t > 0)

    aggregator_mask = rigids_data.get("rigids_is_motif_group_aggregator")
    if aggregator_mask is None:
        motif_local_mask = group_id > 0
    else:
        motif_local_mask = (group_id > 0) & aggregator_mask.to(device=device).bool()
    motif_local_mask = motif_local_mask & active.expand_as(motif_local_mask)

    global_d_t = d_t.expand_as(group_id).to(dtype=torch.float32)
    scale = global_d_t / local_d_t.clamp_min(1e-12)
    return torch.where(motif_local_mask, scale, torch.ones_like(scale))


def _trans_churn_t_hat(corrupter, d_t, t):
    """Global grid time after churn backward step (for denoiser conditioning)."""
    if getattr(corrupter, "sampling_noise_mode", None) != "churn":
        return t
    if corrupter.churn_by_sigma:
        curr_sigma = corrupter.g_t(t, corrupter._trans_g_t_fn)
        new_sigma = (1 + corrupter.churn) * curr_sigma
        return torch.clamp(corrupter.g_t_inv(new_sigma, corrupter._trans_g_t_fn), min=0)
    return torch.clamp(t - corrupter.churn * d_t, min=0)


def _model_conditioning_time(corrupter, t_1, d_t, num_batch, device):
    """Per-batch ``[B, 1]`` time for denoiser; separate from per-rigid integrator time."""
    return _normalize_sampling_time(
        _trans_churn_t_hat(corrupter, d_t, t_1),
        num_batch,
        device,
    )


def _real_token_metadata_mask(token_data):
    mask = token_data['asym_id'] >= 0
    if 'is_motif_group_aggregator' in token_data.dtype.names:
        mask = mask & ~token_data['is_motif_group_aggregator']
    return mask


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = batch_t.float().numpy(force=True)
    batch_loss = batch_loss.float().numpy(force=True)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


# from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
def _get_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return 1.0


def get_linear_warmup_schedule(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _get_decay_lr_lambda(
    current_step: int, *, start_step: int, decay_step: int, decay_factor: float
):
    if current_step < start_step:
        return 1.0
    else:
        exponent = (current_step - start_step) // decay_step
        return decay_factor ** (exponent + 1)


def get_mult_decay_schedule(
    optimizer: torch.optim.Optimizer, start_step: int, decay_step: int, decay_factor: float
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_decay_lr_lambda,
        start_step=start_step,
        decay_step=decay_step,
        decay_factor=decay_factor
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



RES_TO_AA = {}
for i, aa in enumerate(residue_constants.resnames):
    RES_TO_AA[const.token_ids[aa]] = i
AA_TO_RES = {j: i for i, j in RES_TO_AA.items()}

class BiomoleculeModule(L.LightningModule):
    def __init__(self,
                 model,
                 corrupter,
                 optim,
                 use_cosine_lr_sched=False,
                 cosine_warmup_steps=0,
                 cosine_total_steps=1e6,
                 use_linear_warmup=False,
                 linear_warmup_steps=0,
                 use_lr_step_decay=False,
                 lr_step_decay_start=0,
                 lr_step_decay_step=1,
                 lr_step_decay_factor=0.95,
                 use_ema=True,
                 ema_decay=0.999,
                 use_posthoc_ema=False,
                 seq_weight=DEFAULT_RESTYPE_WEIGHT,
                 use_euclidean_for_rots=False,
                 learnable_noise_schedule=False,
                 direct_rot_vf_loss=False,
                 rot_angle_weight=0.5,
                 self_condition_rate=0.5,
                 atom_rigid_upweight=True,
                 motif_aggregator_weight=5.0,
                 motif_aggregator_loss_weight=0.2,
                 motif_aggregator_fafe_loss_weight=0.35,
                 motif_intergroup_fafe_loss_weight=0.2,
                 fafe_loss_weight=0.5,
                 seq_loss_weight=0.25,
                 compile_model=False,
                 apply_self_folding=False,
                 strict_weight_loading=True,
                 log_train_loss_every_step=False,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self._model_num_params = float(sum(p.numel() for p in self.model.parameters()))
        if compile_model:
            self.model.compile()
        self.corrupter = corrupter
        self.optim = optim
        self.self_condition_rate = self_condition_rate

        self.use_cosine_lr_sched = use_cosine_lr_sched
        self.use_linear_warmup = use_linear_warmup
        self.use_lr_step_decay = use_lr_step_decay
        self.linear_warmup_steps = linear_warmup_steps
        self.cosine_warmup_steps = cosine_warmup_steps
        self.cosine_total_steps = cosine_total_steps
        self.lr_step_decay_start = lr_step_decay_start
        self.lr_step_decay_step = lr_step_decay_step
        self.lr_step_decay_factor = lr_step_decay_factor

        self.use_ema = use_ema
        self.use_posthoc_ema = use_posthoc_ema
        self.use_euclidean_for_rots = use_euclidean_for_rots
        self.learnable_noise_schedule = learnable_noise_schedule
        self.direct_rot_vf_loss = direct_rot_vf_loss
        self.rot_angle_weight = rot_angle_weight
        self.atom_rigid_upweight = atom_rigid_upweight
        self.motif_aggregator_weight = motif_aggregator_weight
        self.motif_aggregator_loss_weight = motif_aggregator_loss_weight
        self.motif_aggregator_fafe_loss_weight = motif_aggregator_fafe_loss_weight
        self.motif_intergroup_fafe_loss_weight = motif_intergroup_fafe_loss_weight
        self.fafe_loss_weight = fafe_loss_weight
        self.seq_loss_weight = seq_loss_weight
        self.apply_self_folding = apply_self_folding
        self.log_train_loss_every_step = log_train_loss_every_step
        # Accumulators over grad-accumulation microbatches on this rank; cleared in on_before_optimizer_step.
        self._throughput_residue_accum = 0.0
        self._throughput_batch_accum = 0.0
        self._throughput_len_accum = 0.0
        self._throughput_microbatch_count = 0.0
        self._throughput_last_opt_time: Optional[float] = None
        if learnable_noise_schedule:
            self.automatic_optimization = False

        seq_weight_tensor = torch.as_tensor([seq_weight[c] for c in const.tokens])
        self.seq_weight = seq_weight_tensor

        if use_ema:
            self.ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        else:
            self.ema = None

        if use_posthoc_ema:
            self.ema_long = EMAModel(model, gamma=6.94)
            self.ema_short = EMAModel(model, gamma=16.97)
        else:
            self.ema_long = None
            self.ema_short = None


        self.aatype_to_restype_tensor = torch.zeros(const.num_tokens)
        for aatype, restype in AA_TO_RES.items():
            self.aatype_to_restype_tensor[aatype] = restype

        if not strict_weight_loading:
            warnings.warn("Model weights will be loaded with strict_loading=False, be sure you know what you're doing!")
            self.strict_loading = False

    def _generate_folding_batch(self, batch, pred_aatype):
        pred_seq_batch = copy.deepcopy(batch)

        # convert aatype to restype
        aa_to_res = self.aatype_to_restype_tensor.to(pred_aatype.device)
        pred_restype = aa_to_res[pred_aatype]
        restype = batch['token']['res_type'].clone()
        is_protein = batch['token']['mol_type'] == const.chain_type_ids["PROTEIN"]
        new_restype = restype * ~is_protein + pred_restype * is_protein

        # replace res_type (and seq which is here for legacy reasons)
        pred_seq_batch['token']['res_type'] = new_restype.long()
        pred_seq_batch['token']['seq'] = new_restype.long()
        # also don't noise sequence internally
        seq_noising_mask = pred_seq_batch['token']['seq_noising_mask'].clone()
        seq_noising_mask[is_protein] = False
        pred_seq_batch['token']['seq_noising_mask'] = seq_noising_mask

        # we'll also mask out any copy residues
        # since the idea here is to evaluate the denoiser output designability
        # without influence from conditioning
        token_mask = pred_seq_batch['token']['token_mask'].clone()
        token_is_copy_mask = pred_seq_batch['token']['token_is_copy_mask']
        token_mask[token_is_copy_mask] = False
        pred_seq_batch['token']['token_mask'] = token_mask
        # we also need to mask the corresponding rigids
        rigids_mask = pred_seq_batch['rigids']['rigids_mask'].clone()
        rigids_to_token = pred_seq_batch['rigids']['rigids_to_token']
        new_rigids_mask = torch.gather(
            token_mask,
            -1,
            rigids_to_token
        )
        pred_seq_batch['rigids']['rigids_mask'] = rigids_mask & new_rigids_mask

        return pred_seq_batch

    def _prepare_noisy_batch(self, batch, rng_seed=None):
        batch['trans_t'] = batch['t']
        batch['rot_t'] = batch['t']
        return self.corrupter.corrupt_dense_batch(batch, rng_seed=rng_seed)

    def _forward_with_self_conditioning(self, batch, enable_self_conditioning=True):
        apply_effective_rigids_noising_mask(batch, self.corrupter)
        model = self.model
        if enable_self_conditioning and model.self_conditioning and np.random.uniform() < self.self_condition_rate:
            with torch.no_grad():
                self_conditioning = model(batch)

                if self.apply_self_folding:
                    pred_seq_batch = self._generate_folding_batch(batch, self_conditioning['pred_seq'])
                    self_folding = model(pred_seq_batch)
                else:
                    self_folding = None
        else:
            self_conditioning = None
            self_folding = None
        return model(batch, self_conditioning, self_folding)

    def _validation_seed_from_batch(self, batch):
        if 'val_seed' not in batch:
            return None
        seeds = batch['val_seed']
        if torch.is_tensor(seeds):
            seed_values = seeds.detach().cpu().view(-1).tolist()
        else:
            seed_values = [int(seed) for seed in seeds]
        combined = 0
        for seed in seed_values:
            combined = (combined * 1315423911 + int(seed)) % (2**31 - 1)
        return combined

    def _log_validation_metrics(self, batch, loss_dict):
        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {
            ("val/" + key): value
            for key, value in sorted(log_dict.items(), key=lambda x: x[0])
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch['t'].shape[0],
            sync_dist=True,
        )

        loss_by_task = {}
        for i, task in enumerate(batch['task']):
            task_loss_key = f"val_task/{task.name}_loss"
            task_seq_key = f"val_task/{task.name}_seq_loss"
            task_frame_key = f"val_task/{task.name}_frame_vf_loss"
            loss_by_task.setdefault(task_loss_key, []).append(loss_dict['loss_per_batch'][i])
            loss_by_task.setdefault(task_seq_key, []).append(loss_dict['seq_loss'][i])
            loss_by_task.setdefault(task_frame_key, []).append(loss_dict['frame_vf_loss'][i])
        for key, values in loss_by_task.items():
            value = torch.stack(values)
            self.log(
                key,
                value.mean(),
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                batch_size=value.shape[0],
                sync_dist=True,
            )

    @staticmethod
    def _loss_contribution_fractions(loss_dict):
        """Per-sample share of total loss for each weighted term (diagnostic)."""
        loss_per_batch = loss_dict.get("loss_per_batch")
        if loss_per_batch is None or not torch.is_tensor(loss_per_batch):
            return {}
        required = (
            "frame_vf_loss",
            "aggregator_pose_loss_weighted",
            "aggregator_fafe_loss_weighted",
            "motif_intergroup_fafe_loss_weighted",
            "fafe_loss_weighted",
            "seq_loss_weighted",
        )
        if any(key not in loss_dict for key in required):
            return {}
        denom = loss_per_batch.detach().clamp(min=1e-8)
        return {
            "loss_frac_frame_vf": loss_dict["frame_vf_loss"] / denom,
            "loss_frac_aggregator_pose": loss_dict["aggregator_pose_loss_weighted"] / denom,
            "loss_frac_aggregator_fafe": loss_dict["aggregator_fafe_loss_weighted"] / denom,
            "loss_frac_motif_intergroup_fafe": loss_dict["motif_intergroup_fafe_loss_weighted"] / denom,
            "loss_frac_fafe": loss_dict["fafe_loss_weighted"] / denom,
            "loss_frac_seq": loss_dict["seq_loss_weighted"] / denom,
        }

    def training_step(self, batch, batch_idx):
        # update EMA
        if self.ema is not None and self.global_step > 0:
            self.ema.update_parameters(self.model)
        if self.ema_long is not None and self.global_step > 0:
            self.ema_long.update_parameters(self.model, self.global_step-1)
        if self.ema_short is not None and self.global_step > 0:
            self.ema_short.update_parameters(self.model, self.global_step-1)

        # corrupt data
        batch = self._prepare_noisy_batch(batch)

        # Global residues/sec ≈ (sum over ranks of Σ B·L per optimizer step) / Δt between optimizer steps.
        # Local Σ(B·L) over this rank; summed across ranks in on_before_optimizer_step → cluster-wide residues/s.
        sm = batch["token"]["seq_noising_mask"]
        bsz = float(sm.shape[0])
        seqlen = float(sm.shape[1])
        self._throughput_residue_accum += bsz * seqlen
        self._throughput_batch_accum += bsz
        self._throughput_len_accum += seqlen
        self._throughput_microbatch_count += 1.0

        # run denoiser, with self-conditioning as necessary
        outputs = self._forward_with_self_conditioning(batch)

        # compute loss
        loss_dict = self._loss_step(batch, outputs)

        # log loss
        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
            loss_dict
        )

        loss_by_task = {}
        for i, task in enumerate(batch['task']):
            if task.name + "_loss" not in loss_by_task:
                loss_by_task[task.name + "_loss"] = []
            if task.name + "_seq_loss" not in loss_by_task:
                loss_by_task[task.name + "_seq_loss"] = []
            if task.name + "_frame_vf_loss" not in loss_by_task:
                loss_by_task[task.name + "_frame_vf_loss"] = []
            if task.name + "_frame_vf_loss_unscaled" not in loss_by_task:
                loss_by_task[task.name + "_frame_vf_loss_unscaled"] = []

            loss_by_task[task.name + "_loss"].append(loss_dict['loss_per_batch'][i])
            loss_by_task[task.name + "_seq_loss"].append(loss_dict["seq_loss"][i])
            loss_by_task[task.name + "_frame_vf_loss"].append(loss_dict['frame_vf_loss'][i])
            loss_by_task[task.name + "_frame_vf_loss_unscaled"].append(loss_dict['frame_vf_loss_unscaled'][i])

        loss_by_task = {
            key: torch.stack(values)
            for key, values in loss_by_task.items()
        }
        for key, value in loss_by_task.items():
            self.log(
                "task/" + key,
                value.mean(),
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=value.shape[0],
                sync_dist=False)

        log_dict = {
            ("train/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }
        t = batch['t']
        for loss_name, loss_list in loss_dict.items():
            if loss_name in ['loss', 'frameflow_loss', "frame_vf_loss_unscaled", 'loss_per_batch']:
                continue
            if t.numel() != loss_list.numel():
                continue
            # if not loss_name.startswith("pt_") and not loss_name.startswith("latent_"):
            #     continue
            stratified_losses = t_stratified_loss(
                t, loss_list, loss_name=loss_name)
            stratified_losses = {
                f"train/{k}": torch.round(torch.as_tensor(v, device=log_dict['train/loss'].device), decimals=3)
                for k,v in stratified_losses.items()
            }
            self.log_dict(
                stratified_losses,
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=t.shape[0],
                sync_dist=False)

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=t.shape[0],
            sync_dist=True)

        # Per-step scalar for MLflow/other loggers (set lmodule.log_train_loss_every_step=true).
        # Default train/* above uses on_epoch=True, so loggers only get a point each epoch otherwise.
        if self.log_train_loss_every_step:
            self.log(
                "train/loss_step",
                loss_dict["loss"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=t.shape[0],
                sync_dist=False,
            )

        return loss_dict

    def validation_step(self, batch, batch_idx):
        del batch_idx
        batch = copy.deepcopy(batch)
        validation_seed = self._validation_seed_from_batch(batch)
        batch = self._prepare_noisy_batch(batch, rng_seed=validation_seed)
        outputs = self._forward_with_self_conditioning(batch, enable_self_conditioning=False)
        loss_dict = self._loss_step(batch, outputs)
        self._log_validation_metrics(batch, loss_dict)
        return loss_dict["loss"]

    def _loss_step(self, inputs, outputs):
        apply_effective_rigids_noising_mask(inputs, self.corrupter)
        token_seq = inputs['token']['seq']
        seq_weight = self.seq_weight.to(token_seq.device)
        rigids_seq_idx = inputs['rigids']['rigids_seq_idx']
        rigids_seq = torch.gather(
            token_seq,
            -1,
            rigids_seq_idx,
        )
        rigidwise_weight = seq_weight[rigids_seq]

        frame_fm_loss_dict = multiframe_fm_loss_dense_batch(
            inputs, outputs, sep_rot_loss=True, use_euclidean_for_rots=self.use_euclidean_for_rots,
            t_norm_clip=0.9,
            rot_vf_angle_loss_weight=self.rot_angle_weight,
            fafe_l2_block_mask_size=1,
            rigidwise_weight=rigidwise_weight,
            direct_rot_vf_loss=self.direct_rot_vf_loss,
            upweight_atomic=self.atom_rigid_upweight,
            motif_aggregator_weight=self.motif_aggregator_weight,
        )

        frame_vf_loss = (
            frame_fm_loss_dict["trans_vf_loss"] +
            frame_fm_loss_dict["rot_vf_loss"]
        )
        unscaled_frame_vf_loss = (
            frame_fm_loss_dict["unscaled_trans_vf_loss"] +
            frame_fm_loss_dict["unscaled_rot_vf_loss"]
        )
        aggregator_pose_loss = (
            self.motif_aggregator_loss_weight
            * frame_fm_loss_dict["aggregator_pose_loss"]
        )
        aggregator_fafe_loss = (
            self.motif_aggregator_fafe_loss_weight
            * frame_fm_loss_dict["scaled_aggregator_fafe"]
        )
        motif_intergroup_fafe_loss = (
            self.motif_intergroup_fafe_loss_weight
            * frame_fm_loss_dict["scaled_motif_intergroup_fafe"]
        )

        atomic_seq_weight = seq_weight[token_seq]
        atomic_loss_dict = seq_losses_dense_batch(
            inputs,
            outputs,
            seqwise_weight=atomic_seq_weight
        )

        seq_loss_weighted = self.seq_loss_weight * atomic_loss_dict["seq_loss"]
        if self.direct_rot_vf_loss:
            fafe_loss_weighted = torch.zeros_like(frame_vf_loss)
            loss = (
                frame_vf_loss
                + aggregator_pose_loss
                + aggregator_fafe_loss
                + motif_intergroup_fafe_loss
                + seq_loss_weighted
            )
        else:
            fafe_loss_weighted = self.fafe_loss_weight * frame_fm_loss_dict["scaled_fafe"]
            loss = (
                frame_vf_loss
                + aggregator_pose_loss
                + aggregator_fafe_loss
                + motif_intergroup_fafe_loss
                + fafe_loss_weighted
                + seq_loss_weighted
            )

        loss_dict = {
            "loss": loss.mean(),
            "frame_vf_loss": frame_vf_loss,
            "frame_vf_loss_unscaled": unscaled_frame_vf_loss,
            "aggregator_pose_loss_weighted": aggregator_pose_loss,
            "aggregator_fafe_loss_weighted": aggregator_fafe_loss,
            "motif_intergroup_fafe_loss_weighted": motif_intergroup_fafe_loss,
            "fafe_loss_weighted": fafe_loss_weighted,
            "seq_loss_weighted": seq_loss_weighted,
            "loss_per_batch": loss,
        }
        loss_dict.update(self._loss_contribution_fractions(loss_dict))

        # TODO: for some reason this does not play well with nccl between different tasks
        # if 'motif_idx' in outputs:
        #     pred_motif_idx = outputs['motif_idx']
        #     gt_motif_idx = inputs['token']['token_seq_idx']
        #     is_motif_mask = ~inputs['token']['token_is_protein_output_mask'] & ~inputs['token']['token_is_ligand_mask']
        #     motif_idx_correct = (pred_motif_idx == gt_motif_idx) * is_motif_mask
        #     if is_motif_mask.sum() > 0:
        #         loss_dict['motif_idx_correct'] = motif_idx_correct.sum() / is_motif_mask.sum()

        # loss_dict[inputs['task'].name + "_loss"] = loss
        # loss_dict[inputs['task'].name + "_seq_loss"] = atomic_loss_dict["seq_loss"]
        # loss_dict[inputs['task'].name + "_frame_vf_loss"] = frame_vf_loss
        # loss_dict[inputs['task'].name + "_frame_vf_loss_unscaled"] = unscaled_frame_vf_loss


        loss_dict.update(frame_fm_loss_dict)
        loss_dict.update(atomic_loss_dict)
        # loss_dict.update(loss_by_task)

        return loss_dict


    def on_before_optimizer_step(self, optimizer):
        # Residues processed this optimizer step (all grad-accum microbatches), summed over every rank / GPU.
        local_res = float(self._throughput_residue_accum)
        local_batch = float(self._throughput_batch_accum)
        local_len = float(self._throughput_len_accum)
        local_count = float(self._throughput_microbatch_count)
        self._throughput_residue_accum = 0.0
        self._throughput_batch_accum = 0.0
        self._throughput_len_accum = 0.0
        self._throughput_microbatch_count = 0.0
        totals = torch.tensor(
            [local_res, local_batch, local_len, local_count],
            device=self.device,
            dtype=torch.float64,
        )
        if self.trainer.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_res_f = float(totals[0].item())
        total_batch_f = float(totals[1].item())
        total_len_f = float(totals[2].item())
        total_count_f = float(totals[3].item())

        now = time.perf_counter()
        if self._throughput_last_opt_time is not None:
            dt = now - self._throughput_last_opt_time
            if dt > 0.0 and self.trainer.is_global_zero:
                self.log(
                    "train/total_residues_per_sec",
                    total_res_f / dt,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                total_res_per_sec = total_res_f / dt
                total_res_per_hour = total_res_per_sec * 3600.0
                total_res_per_day = total_res_per_hour * 24.0
                total_res_per_week = total_res_per_day * 7.0
                self.log(
                    "train/total_residues_per_hour",
                    total_res_per_hour,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                self.log(
                    "train/total_residues_per_day",
                    total_res_per_day,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                self.log(
                    "train/total_residues_per_week",
                    total_res_per_week,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                self.log(
                    "train/model_num_params",
                    self._model_num_params,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                if self._model_num_params > 0.0:
                    self.log(
                        "train/residues_per_day_per_param",
                        total_res_per_day / self._model_num_params,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        logger=True,
                        sync_dist=False,
                    )
                if total_count_f > 0.0:
                    self.log(
                        "train/batch_size_mean",
                        total_batch_f / total_count_f,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        logger=True,
                        sync_dist=False,
                    )
                    self.log(
                        "train/L_mean",
                        total_len_f / total_count_f,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        logger=True,
                        sync_dist=False,
                    )
        self._throughput_last_opt_time = now

        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        with torch.no_grad():
            norms = []
            norm_dict = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    n = torch.linalg.vector_norm(p.grad.view(-1), dim=-1)
                    norms.append(n)
                    norm_dict[name] = n.item()
            # import json
            # print(json.dumps(norm_dict, indent=4))
            total_norm = torch.linalg.vector_norm(
                torch.stack(norms, dim=0),
                dim=0
            )

            # print("grad norm", total_norm)

        self.log(
            "grad_norm",
            total_norm,
            prog_bar=False,
            logger=True,
            on_step=None,
            on_epoch=True,
            batch_size=1,
            sync_dist=False
        )

    def predict_step(self, batch, batch_idx):
        if self.use_ema:
            model = self.ema.module
        else:
            model = self.model
        outputs = self._predict_step(model, batch)
        return outputs

    def _predict_step(
        self,
        model,
        batch
    ):
        corrupter = self.corrupter
        # Set-up time
        ts = torch.linspace(0.0, 1.0, corrupter.num_timesteps)

        rigids_data = batch['rigids']
        rigids_data['rigids_t'] = rigids_data['rigids_1']
        token_data = batch['token']
        rigids_0 = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']

        t_1 = ts[0]

        num_batch, num_res = seq_noising_mask.shape
        device = self.device
        denoiser_out = None

        prot_traj = [(
            trans_0,
            rotmats_0,
            None
        )]

        clean_traj = []

        global_shift = torch.zeros_like(trans_0).mean(dim=-2)
        for t_2 in tqdm.tqdm(ts[1:]):
            d_t = t_2 - t_1
            # Run model.
            trans_t_1, rotmats_t_1, _ = prot_traj[-1]

            # trans_t_1_center = trans_t_1.mean(dim=-2)
            # trans_t_1 = trans_t_1 - trans_t_1_center[..., None, :]
            # global_shift += trans_t_1_center

            churn_flow_mask = corrupter.motif_churn_flow_mask(
                rigids_data.get("motif_group_id"),
                rigids_data["rigids_mask"],
                rigids_data["rigids_noising_mask"],
                rigids_data.get("rigids_is_motif_group_aggregator"),
                device=trans_t_1.device,
            )
            integration_t, integration_d_t = _motif_local_integration_time(
                batch,
                corrupter,
                t_1,
                d_t,
            )
            t_hat, d_t_hat, trans_t_hat = corrupter.trans_churn(
                integration_d_t,
                integration_t,
                trans_t_1,
                noising_mask=churn_flow_mask,
            )
            _, _, rotmats_t_hat = corrupter.rot_churn(
                integration_d_t,
                integration_t,
                rotmats_t_1,
                noising_mask=churn_flow_mask,
            )
            if not _motif_group_motion_locked(batch, corrupter, t_1):
                trans_t_hat, rotmats_t_hat = corrupter.motif_group_churn(
                    integration_t,
                    t_hat,
                    trans_t_1,
                    rotmats_t_1,
                    trans_t_hat,
                    rotmats_t_hat,
                    rigids_data.get("motif_group_id"),
                    rigids_data["rigids_mask"],
                    rigids_data["rigids_noising_mask"],
                    rigids_data.get("rigids_is_motif_group_aggregator"),
                )

            rigids_data["trans_t"] = trans_t_hat
            rigids_data["rotmats_t"] = rotmats_t_hat
            rigids_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_hat),
                trans=trans_t_hat
            ).to_tensor_7()
            t = _model_conditioning_time(corrupter, t_1, d_t, num_batch, device)
            batch["t"] = t
            apply_effective_rigids_noising_mask(batch, corrupter, t=t)

            if self.apply_self_folding and denoiser_out is not None:
                pred_seq_batch = self._generate_folding_batch(batch, denoiser_out['pred_seq'])
                folding_out = model(pred_seq_batch, self_folding=denoiser_out)
            else:
                folding_out = None

            denoiser_out = model(batch, self_condition=denoiser_out, self_folding=folding_out)
            rigids_noising_mask = rigids_data['rigids_noising_mask']

            # Process model output.
            pred_rigids = denoiser_out['denoised_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_quats_1 = pred_rigids.get_rots().get_cur_rot()
            if pred_quats_1.shape[-1] != 4:
                raise ValueError("Motif group Euler step expects quaternion-backed predicted rigids")
            quats_t_hat = rigids_data['rigids_t'][..., :4]

            clean_traj.append(
                (pred_trans_1 + global_shift[..., None, :],
                 pred_rotmats_1,
                 denoiser_out["pred_seq"].detach().cpu(),
                )
            )

            trans_d_t_hat = d_t_hat
            rot_d_t_hat = d_t_hat
            trans_time = t_hat
            rot_time = t_hat
            trans_step_size_scale = _motif_aggregator_step_size_scale(
                batch,
                corrupter,
                t_1,
                d_t,
                integration_d_t,
            )

            trans_t_2 = corrupter.trans_euler_step(
                trans_d_t_hat,
                trans_time,
                pred_trans_1,
                trans_t_hat,
                noising_mask=rigids_noising_mask,
                step_size_scale=trans_step_size_scale,
            )
            rotmats_t_2 = corrupter.rots_euler_step(
                rot_d_t_hat,
                rot_time,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                rot_vf=denoiser_out['pred_rot_vf']
            )
            if not _motif_group_motion_locked(batch, corrupter, trans_time):
                trans_t_2, rotmats_t_2 = corrupter.motif_group_euler_step(
                    d_t_hat,
                    trans_time,
                    pred_trans_1,
                    pred_quats_1,
                    trans_t_hat,
                    rotmats_t_hat,
                    quats_t_hat,
                    trans_t_2,
                    rotmats_t_2,
                    rigids_data.get("motif_group_id"),
                    rigids_data["rigids_mask"],
                    rigids_data.get("rigids_is_motif_group_aggregator"),
                )

            prot_traj.append(
                (trans_t_2 + global_shift[..., None, :],
                 rotmats_t_2,
                 denoiser_out["pred_seq"].detach().cpu(),
                )
            )
            t_1 = t_2


            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _= prot_traj[-1]
        rigids_data["trans_t"] = trans_t_1
        rigids_data["rotmats_t"] = rotmats_t_1
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_1
        batch["t"] = t

        denoiser_out = model(
            batch,
            self_condition=denoiser_out,
            sanitize_motif_idx=True,
        )

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans() + global_shift[..., None, :]
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        ret = []

        # data_list = batch.to_data_list()

        prot_traj = prot_traj[1:]

        for i, input_data in enumerate(batch['input_data']):
            # print(input_data)
            num_rigids = input_data['rigids']['tensor7'].shape[0]
            output_data = copy.deepcopy(input_data)
            tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
            tensor7 = tensor7[i, :num_rigids]
            output_data['rigids']['tensor7'] = tensor7

            num_tokens = input_data['tokens']['token_idx'].shape[0]
            pred_seq = denoiser_out["pred_seq"].numpy(force=True)
            pred_seq = pred_seq[i, :num_tokens]
            output_data['tokens']['res_type'] = pred_seq


            # if we copy any tokens, figure out what generated residue corresponds to these fixed tokens
            # select masks
            token_data = output_data['tokens']
            token_is_copy_mask = token_data['is_copy']
            motif_idx = denoiser_out["motif_idx"][i, :num_tokens]
            real_token_metadata_mask = _real_token_metadata_mask(token_data)
            motif_select_mask = (
                token_is_copy_mask & token_data['resolved_mask'] & real_token_metadata_mask)
            motif_seq_fixed = ~token_data['seq_noising_mask'] & real_token_metadata_mask
            # actual idxs
            fixed_bb_res_idx = motif_idx[motif_select_mask]
            fixed_seq_res_idx = motif_idx[motif_seq_fixed]
            fixed_bb_chain_idx = token_data['asym_id'][motif_select_mask]
            fixed_seq_chain_idx = token_data['asym_id'][motif_seq_fixed]

            # token_is_unindexed_mask = token_data['is_unindexed']
            # token_assign_index_mask = token_is_unindexed_mask & motif_select_mask
            # token_data['token_idx'][token_assign_index_mask] = motif_idx.numpy(force=True)[token_assign_index_mask]

            # TODO: this is kinda jenk, we're doing this to allow us to have access to both the "original unindexed index" and "new assigned index"
            # so that we can replace the motif into the structure
            token_data['token_idx'][motif_select_mask] = motif_idx.numpy(force=True)[motif_select_mask]
            # print(output_data['tokens'])

            prot_traj_i = [(_trans[i], _rot[i], _seq[i]) for _trans, _rot, _seq in prot_traj]
            ret_prot_traj = []
            for _trans, _rot, _seq in prot_traj_i:
                traj_data = copy.deepcopy(input_data)
                _quat = ru.rot_to_quat(_rot)
                _tensor7 = torch.cat([_quat, _trans], dim=-1)
                _tensor7 = _tensor7[:num_rigids].numpy(force=True)
                traj_data['rigids']['tensor7'] = _tensor7

                num_tokens = input_data['tokens']['token_idx'].shape[0]
                _seq = _seq[:num_tokens].numpy(force=True)
                traj_data['tokens']['res_type'] = _seq
                ret_prot_traj.append(traj_data)

            clean_traj_i = [(_trans[i], _rot[i], _seq[i]) for _trans, _rot, _seq in clean_traj]
            ret_clean_traj = []
            for _trans, _rot, _seq in clean_traj_i:
                traj_data = copy.deepcopy(input_data)
                _quat = ru.rot_to_quat(_rot)
                _tensor7 = torch.cat([_quat, _trans], dim=-1)
                _tensor7 = _tensor7[:num_rigids].numpy(force=True)
                traj_data['rigids']['tensor7'] = _tensor7

                num_tokens = input_data['tokens']['token_idx'].shape[0]
                _seq = _seq[:num_tokens].numpy(force=True)
                traj_data['tokens']['res_type'] = _seq
                ret_clean_traj.append(traj_data)

            ret.append({
                "input_data": input_data,
                "output_data": output_data,
                "prot_traj": ret_prot_traj,
                "clean_traj": ret_clean_traj,
                "fixed_bb_res_idx": fixed_bb_res_idx,
                "fixed_seq_res_idx": fixed_seq_res_idx,
                "fixed_bb_chain_idx": fixed_bb_chain_idx,
                "fixed_seq_chain_idx": fixed_seq_chain_idx,
                "name": batch["task"][i]
            })

        return ret


    def configure_optimizers(self):
        optimizer = self.optim(self.model.parameters())

        if self.use_lr_step_decay:
            scheduler = get_mult_decay_schedule(
                optimizer,
                start_step=self.lr_step_decay_start,
                decay_step=self.lr_step_decay_step,
                decay_factor=self.lr_step_decay_factor
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        elif self.use_cosine_lr_sched:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cosine_warmup_steps,
                num_training_steps=int(self.cosine_total_steps),
                num_cycles=1
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.use_linear_warmup:
            scheduler = get_linear_warmup_schedule(
                optimizer,
                num_warmup_steps=self.linear_warmup_steps,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        else:
            return optimizer


class BiomoleculeSamplingModule(L.LightningModule):
    def __init__(
        self,
        model,
        corrupter,
        run_cfg,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        # the actual ema params don't matter here, we just wanna be able to load the weights
        self.ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        self.corrupter = corrupter
        self.run_cfg = run_cfg

    def predict_step(self, batch, batch_idx):
        # Set-up time
        ts = torch.linspace(0.0, 1.0, self.corrupter.num_timesteps)

        rigids_data = batch['rigids']
        rigids_data['rigids_t'] = rigids_data['rigids_1']
        token_data = batch['token']
        rigids_0 = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']

        prot_traj = [(
            trans_0,
            rotmats_0,
            None
        )]

        clean_traj = []

        # set up initial integration conditions
        t_1 = ts[0]
        prev_denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            trans_t, rotmats_t, _ = prot_traj[-1]
            prev_denoiser_out, prot_traj_point, clean_traj_point = self._integration_step(
                self.ema.module,
                batch,
                trans_t,
                rotmats_t,
                t_1,
                t_2,
                self_conditioning=prev_denoiser_out
            )
            prot_traj.append(prot_traj_point)
            clean_traj.append(clean_traj_point)
            t_1 = t_2

        trans_t, rotmats_t, _ = prot_traj[-1]
        final_denoiser_out, _, _ = self._integration_step(
            self.ema.module,
            batch,
            trans_t,
            rotmats_t,
            ts[-1],
            ts[-1],
            self_conditioning=prev_denoiser_out,
            apply_step=False
        )

        # Process model output.
        prot_traj = prot_traj[1:]

        ret = self._post_process_outputs(
            batch,
            prot_traj,
            clean_traj,
            final_denoiser_out
        )

        return ret

    def _integration_step(
        self,
        model,
        batch,
        trans_t_1,
        rotmats_t_1,
        t_1,
        t_2,
        self_conditioning=None,
        apply_step=True
    ):
        d_t = t_2 - t_1
        rigids_data = batch['rigids']
        token_data = batch['token']
        seq_noising_mask = token_data['seq_noising_mask']
        num_batch, num_res = seq_noising_mask.shape
        device = self.device
        denoiser_out = None

        churn_flow_mask = self.corrupter.motif_churn_flow_mask(
            rigids_data.get("motif_group_id"),
            rigids_data["rigids_mask"],
            rigids_data["rigids_noising_mask"],
            rigids_data.get("rigids_is_motif_group_aggregator"),
            device=trans_t_1.device,
        )
        integration_t, integration_d_t = _motif_local_integration_time(
            batch,
            self.corrupter,
            t_1,
            d_t,
        )
        t_hat, d_t_hat, trans_t_hat = self.corrupter.trans_churn(
            integration_d_t,
            integration_t,
            trans_t_1,
            noising_mask=churn_flow_mask,
        )
        _, _, rotmats_t_hat = self.corrupter.rot_churn(
            integration_d_t,
            integration_t,
            rotmats_t_1,
            noising_mask=churn_flow_mask,
        )
        if not _motif_group_motion_locked(batch, self.corrupter, t_1):
            trans_t_hat, rotmats_t_hat = self.corrupter.motif_group_churn(
                integration_t,
                t_hat,
                trans_t_1,
                rotmats_t_1,
                trans_t_hat,
                rotmats_t_hat,
                rigids_data.get("motif_group_id"),
                rigids_data["rigids_mask"],
                rigids_data["rigids_noising_mask"],
                rigids_data.get("rigids_is_motif_group_aggregator"),
            )

        # Run model.
        rigids_data["trans_t"] = trans_t_hat
        rigids_data["rotmats_t"] = rotmats_t_hat
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_hat),
            trans=trans_t_hat
        ).to_tensor_7()
        t = _model_conditioning_time(self.corrupter, t_1, d_t, num_batch, device)
        batch["t"] = t
        apply_effective_rigids_noising_mask(batch, self.corrupter, t=t)
        rigids_noising_mask = rigids_data['rigids_noising_mask']

        denoiser_out = model(
            batch,
            self_condition=self_conditioning,
            self_folding=None,
            sanitize_motif_idx=not apply_step,
        )

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        pred_quats_1 = pred_rigids.get_rots().get_cur_rot()
        if pred_quats_1.shape[-1] != 4:
            raise ValueError("Motif group Euler step expects quaternion-backed predicted rigids")
        quats_t_hat = rigids_data['rigids_t'][..., :4]

        clean_traj_point = (
            pred_trans_1,
            pred_rotmats_1,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        trans_d_t_hat = d_t_hat
        rot_d_t_hat = d_t_hat
        trans_time = t_hat
        rot_time = t_hat
        trans_step_size_scale = _motif_aggregator_step_size_scale(
            batch,
            self.corrupter,
            t_1,
            d_t,
            integration_d_t,
        )

        if apply_step:
            trans_t_2 = self.corrupter.trans_euler_step(
                trans_d_t_hat,
                trans_time,
                pred_trans_1,
                trans_t_hat,
                noising_mask=rigids_noising_mask,
                step_size_scale=trans_step_size_scale,
            )
            rotmats_t_2 = self.corrupter.rots_euler_step(
                rot_d_t_hat,
                rot_time,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                rot_vf=denoiser_out['pred_rot_vf']
            )
            if not _motif_group_motion_locked(batch, self.corrupter, trans_time):
                trans_t_2, rotmats_t_2 = self.corrupter.motif_group_euler_step(
                    d_t_hat,
                    trans_time,
                    pred_trans_1,
                    pred_quats_1,
                    trans_t_hat,
                    rotmats_t_hat,
                    quats_t_hat,
                    trans_t_2,
                    rotmats_t_2,
                    rigids_data.get("motif_group_id"),
                    rigids_data["rigids_mask"],
                    rigids_data.get("rigids_is_motif_group_aggregator"),
                )

            prot_traj_point = (
                trans_t_2,
                rotmats_t_2,
                denoiser_out["pred_seq"].detach().cpu(),
            )
        else:
            prot_traj_point = clean_traj_point

        return denoiser_out, prot_traj_point, clean_traj_point

    def _post_process_outputs(
        self,
        batch,
        prot_traj,
        clean_traj,
        final_denoiser_out
    ):
        ret = []
        center_clean_traj_com = bool(self.run_cfg.get("center_clean_traj_com", False))
        center_prot_traj_com = bool(self.run_cfg.get("center_prot_traj_com", False))

        pred_rigids = final_denoiser_out['denoised_rigids']
        pred_tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
        pred_seq = final_denoiser_out["pred_seq"].numpy(force=True)


        for i, input_data in enumerate(batch['input_data']):
            # chop off any padding for pred_rigids and pred_seq
            num_rigids = input_data['rigids']['tensor7'].shape[0]
            output_data = copy.deepcopy(input_data)
            # tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
            _tensor7 = pred_tensor7[i, :num_rigids]
            output_data['rigids']['tensor7'] = _tensor7

            num_tokens = input_data['tokens']['token_idx'].shape[0]
            _seq = pred_seq[i, :num_tokens]
            output_data['tokens']['res_type'] = _seq

            # if we copy any tokens, figure out what generated residue corresponds to these fixed tokens
            # select masks
            token_data = output_data['tokens']
            token_is_copy_mask = token_data['is_copy']
            motif_idx = final_denoiser_out["motif_idx"][i, :num_tokens]
            real_token_metadata_mask = _real_token_metadata_mask(token_data)
            motif_select_mask = (
                token_is_copy_mask & token_data['resolved_mask'] & real_token_metadata_mask)
            motif_seq_fixed = ~token_data['seq_noising_mask'] & real_token_metadata_mask
            # actual idxs
            fixed_bb_res_idx = motif_idx[motif_select_mask]
            fixed_seq_res_idx = motif_idx[motif_seq_fixed]
            fixed_bb_chain_idx = token_data['asym_id'][motif_select_mask]
            fixed_seq_chain_idx = token_data['asym_id'][motif_seq_fixed]

            # TODO: this is kinda jenk, we're doing this to allow us to have access to both
            # the "original unindexed index" (which is stored in res_idx)
            # and "new assigned index" (which is overwritten into token_idx)
            # we currently use both to impute the copy motif into the generated structure
            token_data['token_idx'][motif_select_mask] = motif_idx.numpy(force=True)[motif_select_mask]

            prot_traj_i = [(_trans[i], _rot[i], _seq[i]) for _trans, _rot, _seq in prot_traj]
            ret_prot_traj = []
            for _trans, _rot, _seq in prot_traj_i:
                traj_data = copy.deepcopy(input_data)
                if center_prot_traj_com:
                    # Center each protein trajectory frame independently.
                    _trans = _trans - _trans[:num_rigids].mean(dim=-2, keepdim=True)
                _quat = ru.rot_to_quat(_rot)
                _tensor7 = torch.cat([_quat, _trans], dim=-1)
                _tensor7 = _tensor7[:num_rigids].numpy(force=True)
                traj_data['rigids']['tensor7'] = _tensor7

                num_tokens = input_data['tokens']['token_idx'].shape[0]
                _seq = _seq[:num_tokens].numpy(force=True)
                traj_data['tokens']['res_type'] = _seq
                ret_prot_traj.append(traj_data)

            clean_traj_i = [(_trans[i], _rot[i], _seq[i]) for _trans, _rot, _seq in clean_traj]
            ret_clean_traj = []
            for _trans, _rot, _seq in clean_traj_i:
                traj_data = copy.deepcopy(input_data)
                if center_clean_traj_com:
                    # Center each clean trajectory frame independently.
                    _trans = _trans - _trans[:num_rigids].mean(dim=-2, keepdim=True)
                _quat = ru.rot_to_quat(_rot)
                _tensor7 = torch.cat([_quat, _trans], dim=-1)
                _tensor7 = _tensor7[:num_rigids].numpy(force=True)
                traj_data['rigids']['tensor7'] = _tensor7

                num_tokens = input_data['tokens']['token_idx'].shape[0]
                _seq = _seq[:num_tokens].numpy(force=True)
                traj_data['tokens']['res_type'] = _seq
                ret_clean_traj.append(traj_data)

            sample_result = {
                "input_data": input_data,
                "output_data": output_data,
                "prot_traj": ret_prot_traj,
                "clean_traj": ret_clean_traj,
                "fixed_bb_res_idx": fixed_bb_res_idx,
                "fixed_seq_res_idx": fixed_seq_res_idx,
                "fixed_bb_chain_idx": fixed_bb_chain_idx,
                "fixed_seq_chain_idx": fixed_seq_chain_idx,
                "name": batch["task"][i]
            }
            if "task_metadata" in batch:
                sample_result["task_metadata"] = batch["task_metadata"][i]
            if "sample_metadata" in batch:
                sample_result["sample_metadata"] = finalize_motif_sample_metadata(
                    token_data,
                    motif_idx,
                    batch["sample_metadata"][i],
                )
            ret.append(sample_result)

        return ret


class PDBWriter(BasePredictionWriter):
    def __init__(self, output_dir, run_cfg):
        super().__init__(write_interval="batch_and_epoch")
        self._log = logging.getLogger(__name__)
        # Keep paths absolute. write_on_batch_end used to chdir into samples/,
        # and a relative out_dir was then opened from inside that directory.
        self.output_dir = os.path.abspath(output_dir)
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.write_eval_pdb = bool(run_cfg.get("write_eval_pdb", True))
        self.samples_eval_dir = (
            os.path.join(self.output_dir, "samples_eval") if self.write_eval_pdb else None
        )
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        self.traj_dir = os.path.join(self.output_dir, "traj")
        self.run_cfg = run_cfg

        os.makedirs(self.samples_dir, exist_ok=True)
        if self.samples_eval_dir is not None:
            os.makedirs(self.samples_eval_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)


        self.samples_metadata = {}

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        samples_metadata = {}

        curr_sample_id = 0
        rank = trainer.global_rank
        for sample_data in prediction:
            sample_output = prediction_frame_to_tokenized(sample_data['output_data'])
            sample_len = sample_output.tokens.shape[0]
            sample_name = f"len_{sample_len}_protein_id{rank}_{batch_idx}_{curr_sample_id}" #.pdb"
            sample_path = os.path.join(self.samples_dir, sample_name + ".pdb")
            struct = write_tokenized_structure_pdb(
                sample_path,
                sample_output,
                output_motif_chains=self.run_cfg['output_motif_chains'],
            )
            sample_path = os.path.abspath(sample_path)

            eval_sample_path = sample_path
            if self.samples_eval_dir is not None:
                eval_sample_path = os.path.join(self.samples_eval_dir, sample_name + ".pdb")
                sample_chain_name = "A"
                task_metadata = sample_data.get("task_metadata") or {}
                if task_metadata.get("sample_chain_name"):
                    sample_chain_name = str(task_metadata["sample_chain_name"])
                write_eval_tokenized_structure_pdb(
                    eval_sample_path,
                    sample_output,
                    sample_chain_name=sample_chain_name,
                )
                eval_sample_path = os.path.abspath(eval_sample_path)

            if self.run_cfg['save_traj']:
                clean_traj = sample_data['clean_traj']
                prot_traj = sample_data['prot_traj']
                use_cuda_traj_batch = bool(self.run_cfg.get("use_cuda_traj_batch", True))

                clean_traj_name = os.path.join(
                    self.traj_dir,
                    f"len_{sample_len}_protein_id{rank}_{batch_idx}_{curr_sample_id}_clean_traj.pdb"
                )
                prot_traj_name = os.path.join(
                    self.traj_dir,
                    f"len_{sample_len}_protein_id{rank}_{batch_idx}_{curr_sample_id}_prot_traj.pdb"
                )
                clean_frames = [prediction_frame_to_tokenized(traj_data) for traj_data in clean_traj]
                prot_frames = [prediction_frame_to_tokenized(traj_data) for traj_data in prot_traj]
                write_traj_fn = (
                    write_multimodel_pdb_batched if use_cuda_traj_batch else write_multimodel_pdb
                )
                clean_path, clean_conects, clean_models = write_traj_fn(
                    clean_traj_name,
                    clean_frames,
                    output_motif_chains=self.run_cfg['output_motif_chains'],
                )
                self._log.info(
                    "Post-processed trajectory %s with %s CONECT records across %s model(s).",
                    clean_path,
                    clean_conects,
                    clean_models,
                )
                prot_path, prot_conects, prot_models = write_traj_fn(
                    prot_traj_name,
                    prot_frames,
                    output_motif_chains=self.run_cfg['output_motif_chains'],
                )
                self._log.info(
                    "Post-processed trajectory %s with %s CONECT records across %s model(s).",
                    prot_path,
                    prot_conects,
                    prot_models,
                )

            chain_data = struct.chains
            chain_mapping = {
                c['asym_id']: c['name']
                for c in chain_data
            }

            samples_metadata[sample_name] = {
                "path": eval_sample_path if self.samples_eval_dir is not None else sample_path,
                "viz_path": sample_path,
                "name": sample_data['name'] if 'name' in sample_data else None,
                "length": sample_len,
                "fixed_bb_res_idx": [i+1 for i in sample_data['fixed_bb_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                "fixed_bb_chain": [chain_mapping[int(i)] for i in sample_data['fixed_bb_chain_idx']],
                "fixed_seq_res_idx": [i+1 for i in sample_data['fixed_seq_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                "fixed_seq_chain": [chain_mapping[int(i)] for i in sample_data['fixed_seq_chain_idx']],
            }
            task_metadata = sample_data.get("task_metadata")
            if task_metadata:
                samples_metadata[sample_name].update(task_metadata)
            sample_metadata = sample_data.get("sample_metadata")
            if sample_metadata:
                samples_metadata[sample_name].update(
                    attach_output_chains_to_sample_metadata(
                        sample_metadata,
                        chain_mapping,
                    )
                )
            sample_chain_name = str(samples_metadata[sample_name].get("sample_chain_name") or "A")
            # Eval metadata: motif_idx indices are scaffold-relative; remap chains off viz asym_id
            # (e.g. B) onto sample_chain_name before PMPNN/RMSD. Stage-2 conditioning uses stage2_* fields.
            samples_metadata[sample_name].update(
                normalize_eval_source_residue_keys(
                    samples_metadata[sample_name],
                    sample_chain_name,
                )
            )
            apply_eval_sample_metadata(samples_metadata[sample_name])
            curr_sample_id += 1

        with open(os.path.join(self.metadata_dir, f"samples_metadata_rank{rank}_batch{batch_idx}.json"), 'w') as fp:
            json.dump(samples_metadata, fp)
        self.samples_metadata.update(samples_metadata)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # collect metadata across all processes
        gathered = [None for _ in range(trainer.world_size)]
        # Call the collective on *every* rank or it will hang
        if dist.is_available() and dist.is_initialized() and trainer.world_size > 1:
            dist.all_gather_object(gathered, self.samples_metadata)   # blocks until all ranks participate
        else:
            gathered = [self.samples_metadata]

        samples_metadata = {}
        for d in gathered:
            samples_metadata.update(d)

        # Now only rank 0 writes/merges
        if trainer.global_rank == 0:
            # merged = ...  # flatten/concatenate/serialize as you like
            # write merged to disk
            samples_metadata_path = os.path.join(self.output_dir, "samples_metadata.json")
            with open(samples_metadata_path, 'w') as fp:
                json.dump(samples_metadata, fp)

            pmpnn_fixed_pos_dict = build_pmpnn_fixed_pos_dict(samples_metadata)

            with open(os.path.join(self.output_dir, "pmpnn_fixed_pos_dict.jsonl"), 'w') as fp:
                json.dump(pmpnn_fixed_pos_dict, fp)

            if bool(self.run_cfg.get("export_layout_manifest", False)):
                layout_manifest_path = os.path.join(self.output_dir, "layout_manifest.jsonl")
                num_entries = export_layout_manifest(
                    samples_metadata_path=Path(os.path.abspath(samples_metadata_path)),
                    output_path=Path(os.path.abspath(layout_manifest_path)),
                )
                self._log.info(
                    "Exported %s layout entries to %s",
                    num_entries,
                    layout_manifest_path,
                )

        # (optional) keep ranks in lockstep before exiting
        trainer.strategy.barrier()