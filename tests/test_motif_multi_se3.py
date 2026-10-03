"""Tests for multi-motif SE(3) grouping and augmentation."""

import numpy as np
import pytest

from nomadflow.runtime.training.motif_partition import build_motif_group_id_by_residue


class _DummyStruct:
    def __init__(self, residues, chains, atoms, mask):
        self.residues = residues
        self.chains = chains
        self.atoms = atoms
        self.mask = mask


def test_build_motif_group_id_mask_blocks():
    residues = np.zeros(
        5,
        dtype=[
            ("is_present", bool),
            ("name", "<U5"),
            ("res_idx", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("is_standard", bool),
        ],
    )
    residues["is_present"] = True
    residues["name"] = "ALA"
    residues["res_idx"] = np.arange(5, dtype=np.int32)
    residues["atom_idx"] = np.arange(5, dtype=np.int32) * 4
    residues["atom_num"] = 4
    residues["is_standard"] = True

    atoms = np.zeros(
        20,
        dtype=[("coords", np.float32, (3,)), ("name", "4i1"), ("is_present", bool)],
    )
    atoms["is_present"] = True
    atoms["coords"] = 0.0
    for residue_index in range(5):
        atoms["name"][residue_index * 4 + 1] = ord("a") - 32
        atoms["name"][residue_index * 4 + 2] = ord("c") - 32
        atoms["coords"][residue_index * 4 + 1] = [1.0, 0.0, 0.0]

    chains = np.array(
        [
            ("A", 0, 0, 0, 0, 0, 0, 0, 5, 0),
        ],
        dtype=[
            ("name", "<U5"),
            ("mol_type", np.int8),
            ("entity_id", np.int32),
            ("sym_id", np.int32),
            ("asym_id", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("res_idx", np.int32),
            ("res_num", np.int32),
            ("cyclic_period", np.int32),
        ],
    )
    data = _DummyStruct(residues, chains, atoms, np.array([True]))
    resolved = residues["is_present"]
    noise = np.ones(5, dtype=bool)
    motif_compressed = np.array([1, 1, 0, 1, 1], dtype=bool)
    noise[resolved] = ~motif_compressed

    out = build_motif_group_id_by_residue(
        data,
        noise,
        motif_compressed,
        resolved,
        enabled=True,
        partition_mode="mask_blocks",
        max_groups=8,
        np_rng=np.random.default_rng(0),
    )

    assert out[0] == 1 and out[1] == 1
    assert out[2] == 0
    assert out[3] == 2 and out[4] == 2


def test_se3_augmentation_preserves_intra_group_distance():
    pytest.importorskip("torch")
    from nomadflow.data.featurize.motif_se3_augment import apply_multi_motif_se3_augmentation
    from nomadflow.data.featurize.tokenize import Rigid, Token

    token = np.zeros(1, dtype=Token)
    token["token_idx"] = 0
    token["rigid_idx"] = 0
    token["rigid_num"] = 3
    token["resolved_mask"] = True

    rigids = np.zeros(3, dtype=Rigid)
    rigids["rigids_noising_mask"] = [True, True, True]
    rigids["motif_group_id"] = [1, 1, 0]
    rigids["token_idx"] = [0, 0, 0]
    rigids["tensor7"][:, 0] = 1.0
    rigids["tensor7"][0, 4:] = [0.0, 0.0, 0.0]
    rigids["tensor7"][1, 4:] = [3.0, 1.0, 0.0]
    rigids["tensor7"][2, 4:] = [9.0, 0.0, 0.0]
    free_rigid_before = rigids["tensor7"][2].copy()

    distance_before = float(
        np.linalg.norm(rigids["tensor7"][0, 4:] - rigids["tensor7"][1, 4:])
    )
    apply_multi_motif_se3_augmentation(
        token,
        rigids,
        enabled=True,
        np_rng=np.random.default_rng(42),
        trans_std=4.0,
    )
    distance_after = float(
        np.linalg.norm(rigids["tensor7"][0, 4:] - rigids["tensor7"][1, 4:])
    )

    assert abs(distance_before - distance_after) < 1e-4
    assert np.allclose(rigids["tensor7"][2], free_rigid_before)


def test_dense_corruption_moves_rigid_level_motif_group_to_target():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
    )
    rigids_1 = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    batch = {
        "token": {},
        "rigids": {
            "rigids_1": rigids_1,
            "rigids_mask": torch.tensor([[True, True, True]]),
            "rigids_noising_mask": torch.tensor([[True, True, True]]),
            "motif_group_id": torch.tensor([[1, 1, 0]], dtype=torch.long),
        },
        "multi_motif_se3": torch.tensor([True]),
        "multi_motif_se3_trans_std": torch.tensor([4.0]),
    }

    endpoints = corrupter.sample_dense_corruption_endpoints(batch, rng_seed=7)
    interp_start = corrupter.interpolate_dense_corruption(
        endpoints,
        torch.tensor([[0.0]], dtype=torch.float32),
    )
    interp_end = corrupter.interpolate_dense_corruption(
        endpoints,
        torch.tensor([[1.0]], dtype=torch.float32),
    )
    interp_mid = corrupter.interpolate_dense_corruption(
        endpoints,
        torch.tensor([[0.5]], dtype=torch.float32),
    )

    motif_start = interp_start["trans_t"][0, :2]
    motif_mid = interp_mid["trans_t"][0, :2]
    motif_end = interp_end["trans_t"][0, :2]
    motif_target = endpoints["trans_1"][0, :2]

    assert not torch.allclose(motif_start, motif_target)
    assert torch.allclose(motif_end, motif_target)

    start_dist = torch.linalg.vector_norm(motif_start[0] - motif_start[1]).item()
    mid_dist = torch.linalg.vector_norm(motif_mid[0] - motif_mid[1]).item()
    end_dist = torch.linalg.vector_norm(motif_end[0] - motif_end[1]).item()
    assert abs(start_dist - mid_dist) < 1e-4
    assert abs(start_dist - end_dist) < 1e-4

    motif_rot_start = interp_start["rotmats_t"][0, :2]
    motif_rot_mid = interp_mid["rotmats_t"][0, :2]
    motif_rot_end = interp_end["rotmats_t"][0, :2]
    rel_start = motif_rot_start[0].transpose(0, 1) @ motif_rot_start[1]
    rel_mid = motif_rot_mid[0].transpose(0, 1) @ motif_rot_mid[1]
    rel_end = motif_rot_end[0].transpose(0, 1) @ motif_rot_end[1]
    assert torch.allclose(rel_start, rel_mid, atol=1e-4, rtol=1e-4)
    assert torch.allclose(rel_start, rel_end, atol=1e-4, rtol=1e-4)


def _motif_corruption_batch():
    torch = pytest.importorskip("torch")
    rigids_1 = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    return {
        "token": {},
        "rigids": {
            "rigids_1": rigids_1,
            "rigids_mask": torch.tensor([[True, True, True]]),
            "rigids_noising_mask": torch.tensor([[True, True, True]]),
            "motif_group_id": torch.tensor([[1, 1, 0]], dtype=torch.long),
        },
        "multi_motif_se3": torch.tensor([True]),
        "multi_motif_se3_trans_std": torch.tensor([4.0]),
    }


def test_collate_stacks_motif_t_lock_when_tasks_differ():
    torch = pytest.importorskip("torch")
    from nomadflow.data.featurize.assembler import MOTIF_T_LOCK_DISABLED, collate

    def _sample(motif_t_lock_value):
        return {
            "task": "dummy",
            "t": torch.tensor(0.5, dtype=torch.float32),
            "multi_motif_se3": torch.tensor(False, dtype=torch.bool),
            "multi_motif_se3_trans_std": torch.tensor(16.0, dtype=torch.float32),
            "motif_t_lock": torch.tensor(motif_t_lock_value, dtype=torch.float32),
            "token": {"token_idx": torch.arange(4, dtype=torch.long)},
            "rigids": {
                "rigids_seq_idx": torch.arange(8, dtype=torch.long),
                "rigids_t": torch.zeros(8, 7, dtype=torch.float32),
            },
        }

    batch = collate([_sample(0.5), _sample(MOTIF_T_LOCK_DISABLED)])
    assert torch.allclose(
        batch["motif_t_lock"],
        torch.tensor([0.5, MOTIF_T_LOCK_DISABLED], dtype=torch.float32),
    )


def test_linear_motif_schedule_reaches_gt_at_t_lock():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.motif_time_schedule import LinearMotifTimeSchedule

    schedule = LinearMotifTimeSchedule()
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    t_lock = 0.5
    t_motif = schedule.motif_interp_time(t, t_lock)
    assert torch.allclose(t_motif, torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0]))
    locked = schedule.is_locked(t, t_lock)
    assert locked.tolist() == [False, False, True, True, True]


def test_effective_rigids_noising_mask_clears_motif_and_aggregator_when_locked():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.motif_time_schedule import (
        MOTIF_T_LOCK_DISABLED,
        effective_rigids_noising_mask,
    )

    base = torch.tensor([[False, True, True, True]])
    group_id = torch.tensor([[0, 1, 1, 1]])
    t = torch.tensor([[0.75]])
    effective = effective_rigids_noising_mask(base, group_id, t, 0.5)
    assert effective.tolist() == [[False, False, False, False]]

    t_unlocked = torch.tensor([[0.25]])
    effective_unlocked = effective_rigids_noising_mask(base, group_id, t_unlocked, 0.5)
    assert effective_unlocked.tolist() == [[False, True, True, True]]

    disabled = effective_rigids_noising_mask(base, group_id, t, MOTIF_T_LOCK_DISABLED)
    assert torch.equal(disabled, base)


def test_effective_rigids_noising_mask_batched_t_lock():
    torch = pytest.importorskip("torch")
    from nomadflow.data.featurize.assembler import MOTIF_T_LOCK_DISABLED
    from nomadflow.stoch_interp.motif_time_schedule import effective_rigids_noising_mask

    base = torch.tensor([[True, True], [True, True]])
    group_id = torch.tensor([[1, 0], [1, 1]])
    t = torch.tensor([[0.75], [0.25]])
    t_lock = torch.tensor([0.5, 0.5])
    effective = effective_rigids_noising_mask(base, group_id, t, t_lock)
    assert effective.tolist() == [[False, True], [True, True]]

    t_lock_mixed = torch.tensor([MOTIF_T_LOCK_DISABLED, 0.5])
    effective_mixed = effective_rigids_noising_mask(base, group_id, t, t_lock_mixed)
    assert effective_mixed.tolist() == [[True, True], [False, False]]


def test_multiframe_loss_uses_effective_noising_mask_when_motif_locked():
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    rigids_1 = torch.zeros(1, 4, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    pred = rigids_1.clone()
    pred[0, 2, 4] = 2.0
    pred[0, 3, 4] = 3.0
    inputs = {
        "t": torch.tensor([[0.75]], dtype=torch.float32),
        "motif_t_lock": torch.tensor(0.5, dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.tensor([[True, True, True, True]]),
            "rigids_noising_mask": torch.tensor([[True, True, True, True]]),
            "rigids_seq_idx": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "motif_group_id": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor([[False, False, True, False]]),
            "gt_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
    }
    losses = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=5.0
    )
    # Only scaffold rigid contributes when motif+aggregator are locked (noising_mask effective False).
    assert torch.allclose(losses["trans_vf_loss"], torch.tensor([0.36]), atol=1e-6)
    assert torch.allclose(losses["aggregator_trans_loss"], torch.tensor([0.0]), atol=1e-6)


def test_linear_motif_schedule_batched_t_lock():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.motif_time_schedule import LinearMotifTimeSchedule

    schedule = LinearMotifTimeSchedule()
    t = torch.tensor([[0.25], [0.75]])
    t_lock = torch.tensor([0.5, 0.5])
    t_motif = schedule.motif_interp_time(t, t_lock)
    assert torch.allclose(t_motif, torch.tensor([[0.5], [1.0]]))
    locked = schedule.is_locked(t, t_lock)
    assert locked.tolist() == [[False], [True]]


def test_motif_group_motion_locked_for_integration_time():
    torch = pytest.importorskip("torch")
    from nomadflow.runtime.lmod import _motif_group_motion_locked
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        motif_t_lock=0.5,
    )
    batch = {"motif_t_lock": torch.tensor(0.5, dtype=torch.float32)}
    assert not _motif_group_motion_locked(batch, corrupter, 0.25)
    assert _motif_group_motion_locked(batch, corrupter, 0.5)
    assert _motif_group_motion_locked(batch, corrupter, torch.tensor([[0.75]]))


def test_motif_local_integration_time_uses_aggregator_local_time_before_lock():
    torch = pytest.importorskip("torch")
    from nomadflow.data.featurize.assembler import MOTIF_T_LOCK_DISABLED
    from nomadflow.runtime.lmod import (
        _motif_aggregator_step_size_scale,
        _motif_local_integration_time,
    )
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        motif_t_lock=0.5,
    )
    batch = {
        "motif_t_lock": torch.tensor(0.5, dtype=torch.float32),
        "rigids": {
            "motif_group_id": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor([[False, True, False]]),
        },
    }

    time_map, dt_map = _motif_local_integration_time(
        batch,
        corrupter,
        torch.tensor([[0.0]], dtype=torch.float32),
        torch.tensor([[0.25]], dtype=torch.float32),
    )
    assert torch.allclose(
        time_map,
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    assert torch.allclose(dt_map, torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float32))
    assert torch.allclose(
        _motif_aggregator_step_size_scale(
            batch,
            corrupter,
            torch.tensor([[0.0]], dtype=torch.float32),
            torch.tensor([[0.25]], dtype=torch.float32),
            dt_map,
        ),
        torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float32),
    )

    crossing_time, crossing_dt = _motif_local_integration_time(
        batch,
        corrupter,
        torch.tensor([[0.4]], dtype=torch.float32),
        torch.tensor([[0.2]], dtype=torch.float32),
    )
    assert torch.allclose(
        crossing_time,
        torch.tensor([[0.4, 0.8, 0.4]], dtype=torch.float32),
    )
    assert torch.allclose(crossing_dt, torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float32))
    assert torch.allclose(
        _motif_aggregator_step_size_scale(
            batch,
            corrupter,
            torch.tensor([[0.4]], dtype=torch.float32),
            torch.tensor([[0.2]], dtype=torch.float32),
            crossing_dt,
        ),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    )

    locked_time, locked_dt = _motif_local_integration_time(
        batch,
        corrupter,
        torch.tensor([[0.5]], dtype=torch.float32),
        torch.tensor([[0.1]], dtype=torch.float32),
    )
    assert torch.allclose(locked_time, torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32))
    assert torch.allclose(locked_dt, torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32))

    batch["motif_t_lock"] = torch.tensor(MOTIF_T_LOCK_DISABLED, dtype=torch.float32)
    assert _motif_local_integration_time(batch, corrupter, 0.0, 0.25) == (0.0, 0.25)


def test_rot_churn_accepts_per_rigid_noise_scale_multiplier():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        sampling_noise_mode="churn",
    )
    rotmats_t = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 4, 1, 1)
    noising_mask = torch.tensor([[True, True, True, False]])
    multiplier = torch.tensor([[[2.0], [1.5], [1.0], [1.0]]], dtype=torch.float32)

    _, _, rotmats_t_hat = corrupter.rot_churn(
        torch.tensor(0.0025),
        torch.tensor(0.25),
        rotmats_t,
        noising_mask,
        noise_scale_multiplier=multiplier,
    )

    assert rotmats_t_hat.shape == rotmats_t.shape
    assert torch.allclose(rotmats_t_hat[0, 3], rotmats_t[0, 3])


def test_trans_euler_step_accepts_per_rigid_step_size_scale():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        sampling_noise_mode=None,
    )
    trans_t = torch.zeros(1, 3, 3, dtype=torch.float32)
    trans_1 = torch.ones(1, 3, 3, dtype=torch.float32)
    noising_mask = torch.tensor([[True, True, True]])

    trans_next = corrupter.trans_euler_step(
        torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        trans_1,
        trans_t,
        noising_mask=noising_mask,
        step_size_scale=torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float32),
    )

    assert torch.allclose(
        trans_next[..., 0],
        torch.tensor([[0.25, 0.25, 0.25]], dtype=torch.float32),
    )


def test_motif_local_integration_time_drives_churn_with_local_variance():
    torch = pytest.importorskip("torch")
    from nomadflow.data.featurize.assembler import MOTIF_T_LOCK_DISABLED
    from nomadflow.runtime.lmod import _motif_local_integration_time
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        sampling_noise_mode="churn",
        motif_t_lock=0.5,
    )
    batch = {
        "motif_t_lock": torch.tensor(0.5, dtype=torch.float32),
        "rigids": {
            "motif_group_id": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor([[False, True, False]]),
        },
    }
    t = torch.tensor([[0.25]], dtype=torch.float32)
    d_t = torch.tensor([[0.10]], dtype=torch.float32)

    time_map, dt_map = _motif_local_integration_time(batch, corrupter, t, d_t)
    t_hat, d_t_hat, _ = corrupter.trans_churn(
        dt_map,
        time_map,
        torch.zeros(1, 3, 3, dtype=torch.float32),
        noising_mask=torch.zeros(1, 3, dtype=torch.bool),
    )

    assert torch.allclose(
        t_hat,
        torch.tensor([[0.21, 0.42, 0.21]], dtype=torch.float32),
    )
    assert torch.allclose(d_t_hat, torch.tensor([[0.14, 0.28, 0.14]], dtype=torch.float32))

    batch["motif_t_lock"] = torch.tensor(MOTIF_T_LOCK_DISABLED, dtype=torch.float32)
    assert _motif_local_integration_time(batch, corrupter, 0.25, 0.10) == (0.25, 0.10)


def test_model_conditioning_time_stays_batch_scalar_not_per_rigid():
    torch = pytest.importorskip("torch")
    from nomadflow.runtime.lmod import (
        _model_conditioning_time,
        _motif_local_integration_time,
    )
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        sampling_noise_mode="churn",
        motif_t_lock=0.5,
    )
    batch = {
        "motif_t_lock": torch.tensor(0.5, dtype=torch.float32),
        "rigids": {
            "motif_group_id": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor([[False, True, False]]),
        },
    }
    t_1 = torch.tensor([[0.25]], dtype=torch.float32)
    d_t = torch.tensor([[0.10]], dtype=torch.float32)
    integration_t, _ = _motif_local_integration_time(batch, corrupter, t_1, d_t)
    assert integration_t.shape == (1, 3)

    model_t = _model_conditioning_time(corrupter, t_1, d_t, num_batch=1, device=t_1.device)
    assert model_t.shape == (1, 1)
    assert torch.allclose(model_t, torch.tensor([[0.21]], dtype=torch.float32))


def test_motif_early_lock_clamps_to_gt_from_t_lock():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        motif_t_lock=0.5,
    )
    batch = _motif_corruption_batch()
    endpoints = corrupter.sample_dense_corruption_endpoints(batch, rng_seed=7)
    motif_target = endpoints["trans_1"][0, :2]

    for t_val in (0.5, 0.75, 1.0):
        interp = corrupter.interpolate_dense_corruption(
            endpoints,
            torch.tensor([[t_val]], dtype=torch.float32),
        )
        assert torch.allclose(interp["trans_t"][0, :2], motif_target, atol=1e-5)
        assert torch.allclose(
            interp["rotmats_t"][0, :2],
            endpoints["rotmats_1"][0, :2],
            atol=1e-5,
        )

    interp_mid = corrupter.interpolate_dense_corruption(
        endpoints,
        torch.tensor([[0.25]], dtype=torch.float32),
    )
    assert not torch.allclose(interp_mid["trans_t"][0, :2], motif_target, atol=1e-3)


def test_spatial_fragment_task_passes_motif_t_lock_in_task_data():
    pytest.importorskip("torch")
    from nomadflow.runtime.training.motif_scaffold import SpatialFragmentMotifScaffolding

    task = SpatialFragmentMotifScaffolding(
        multi_motif_se3=True,
        motif_t_lock=0.5,
        max_num_fragments=2,
        max_num_res=16,
    )
    residues = np.zeros(
        8,
        dtype=[
            ("is_present", bool),
            ("name", "<U5"),
            ("res_idx", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("is_standard", bool),
        ],
    )
    residues["is_present"] = True
    residues["name"] = "ALA"
    residues["res_idx"] = np.arange(8, dtype=np.int32)
    residues["atom_idx"] = np.arange(8, dtype=np.int32) * 4
    residues["atom_num"] = 4
    residues["is_standard"] = True
    atoms = np.zeros(
        32,
        dtype=[("coords", np.float32, (3,)), ("name", "4i1"), ("is_present", bool)],
    )
    atoms["is_present"] = True
    for i in range(8):
        atoms["coords"][i * 4 + 1] = [float(i), 0.0, 0.0]
        atoms["name"][i * 4 + 1] = ord("a") - 32
    chains = np.array(
        [("A", 0, 0, 0, 0, 0, 0, 0, 8, 0)],
        dtype=[
            ("name", "<U5"),
            ("mol_type", np.int8),
            ("entity_id", np.int32),
            ("sym_id", np.int32),
            ("asym_id", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("res_idx", np.int32),
            ("res_num", np.int32),
            ("cyclic_period", np.int32),
        ],
    )
    data = type("S", (), {})()
    data.residues = residues
    data.chains = chains
    data.atoms = atoms
    data.mask = np.array([True])

    task_data = task.sample_t_and_mask(data, np_rng=np.random.default_rng(0))
    assert task_data["motif_t_lock"] == 0.5

    task_off = SpatialFragmentMotifScaffolding(multi_motif_se3=True, max_num_fragments=2)
    task_data_off = task_off.sample_t_and_mask(data, np_rng=np.random.default_rng(0))
    assert task_data_off["motif_t_lock"] is None


def test_motif_early_lock_matches_baseline_without_lock():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    batch = _motif_corruption_batch()
    baseline = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
    )
    locked = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        motif_t_lock=1.0,
    )
    endpoints = baseline.sample_dense_corruption_endpoints(batch, rng_seed=7)
    endpoints["motif_t_lock"] = 1.0
    for t_val in (0.0, 0.5, 1.0):
        t = torch.tensor([[t_val]], dtype=torch.float32)
        base_interp = baseline.interpolate_dense_corruption(endpoints, t)
        lock_interp = locked.interpolate_dense_corruption(endpoints, t)
        assert torch.allclose(
            base_interp["trans_t"][0, :2],
            lock_interp["trans_t"][0, :2],
            atol=1e-5,
        )
        assert torch.allclose(
            base_interp["rotmats_t"][0, :2],
            lock_interp["rotmats_t"][0, :2],
            atol=1e-5,
        )


def test_group_projection_uses_real_members_and_applies_to_aggregator():
    torch = pytest.importorskip("torch")
    from nomadflow.model.modules.frame_transformer import _project_rigid_groups_to_predicted_state
    from nomadflow.openfold.utils import rigid_utils as ru

    ref_tensor7 = torch.tensor(
        [[[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], [
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        ], [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ]]],
        dtype=torch.float32,
    )
    pred_tensor7 = ref_tensor7.clone()
    pred_tensor7[0, 0, 4:] = torch.tensor([10.0, 0.0, 0.0])
    pred_tensor7[0, 1, 4:] = torch.tensor([12.0, 0.0, 0.0])
    pred_tensor7[0, 2, 4:] = torch.tensor([100.0, 0.0, 0.0])

    projected = _project_rigid_groups_to_predicted_state(
        rigids_pred=ru.Rigid.from_tensor_7(pred_tensor7),
        rigids_ref=ru.Rigid.from_tensor_7(ref_tensor7),
        motif_group_id=torch.tensor([[1, 1, 1]], dtype=torch.long),
        rigids_mask=torch.tensor([[True, True, True]]),
        is_motif_group_aggregator=torch.tensor([[False, False, True]]),
    )

    assert torch.allclose(
        projected.get_trans(),
        torch.tensor([[[10.0, 0.0, 0.0], [12.0, 0.0, 0.0], [11.0, 0.0, 0.0]]]),
        atol=1e-6,
    )


def test_multiframe_flow_loss_excludes_real_motif_but_keeps_aggregator():
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    rigids_1 = torch.zeros(1, 4, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    pred = rigids_1.clone()
    pred[0, 0, 4] = 10.0
    pred[0, 1, 4] = 10.0
    pred[0, 2, 4] = 2.0
    pred[0, 3, 4] = 3.0
    inputs = {
        "t": torch.tensor([[0.0]], dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.tensor([[True, True, True, True]]),
            "rigids_noising_mask": torch.tensor([[True, True, True, True]]),
            "rigids_seq_idx": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "motif_group_id": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor([[False, False, True, False]]),
            "gt_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
    }

    losses = multiframe_fm_loss_dense_batch(
        inputs,
        outputs,
        direct_rot_vf_loss=True,
    )

    # flow rigids: aggregator (err^2=4, w=5) + scaffold (err^2=9, w=1) -> mean 29/4 * 0.01
    assert torch.allclose(losses["trans_vf_loss"], torch.tensor([0.0725]), atol=1e-6)
    assert torch.allclose(losses["aggregator_trans_loss"], torch.tensor([0.04]), atol=1e-6)
    assert torch.allclose(losses["motif_real_trans_loss"], torch.tensor([1.0]), atol=1e-6)
    assert "aggregator_rot_loss" in losses
    assert "motif_real_rot_loss" in losses


def test_motif_aggregator_weight_is_constant_across_group_sizes():
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    rigids_1 = torch.zeros(1, 8, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    pred = rigids_1.clone()
    # Two aggregators with the same trans error; groups have 2 and 6 real motif rigids.
    pred[0, 2, 4] = 2.0
    pred[0, 7, 4] = 2.0
    inputs = {
        "t": torch.tensor([[0.0]], dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.ones(1, 8, dtype=torch.bool),
            "rigids_noising_mask": torch.ones(1, 8, dtype=torch.bool),
            "rigids_seq_idx": torch.arange(8)[None],
            "motif_group_id": torch.tensor([[1, 1, 1, 0, 0, 0, 2, 2]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor(
                [[False, False, True, False, False, False, False, True]]
            ),
            "gt_rot_vf": torch.zeros(1, 8, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 8, 3, dtype=torch.float32),
    }

    losses_w5 = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=5.0
    )
    losses_w1 = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )

    # Only aggregator/scaffold flow rigids contribute; both aggregators have err^2=4.
    assert losses_w5["trans_vf_loss"] > losses_w1["trans_vf_loss"]
    assert torch.allclose(
        losses_w5["aggregator_trans_loss"],
        losses_w1["aggregator_trans_loss"],
        atol=1e-6,
    )


def test_motif_aggregator_fafe_tracks_inter_motif_geometry():
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    rigids_1 = torch.zeros(1, 8, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    rigids_1[0, 7, 4] = 10.0
    pred = rigids_1.clone()
    pred[0, 7, 4] = 12.0
    inputs = {
        "t": torch.tensor([[0.0]], dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.ones(1, 8, dtype=torch.bool),
            "rigids_noising_mask": torch.ones(1, 8, dtype=torch.bool),
            "rigids_seq_idx": torch.arange(8)[None],
            "motif_group_id": torch.tensor([[1, 1, 1, 0, 0, 0, 2, 2]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor(
                [[False, False, True, False, False, False, False, True]]
            ),
            "gt_rot_vf": torch.zeros(1, 8, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 8, 3, dtype=torch.float32),
    }

    losses = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )

    expected = torch.sqrt(torch.tensor([4.0 / 20.0**2 + 1e-6]))
    assert torch.allclose(losses["aggregator_fafe"], expected, atol=1e-6)
    assert torch.allclose(losses["scaled_aggregator_fafe"], expected, atol=1e-6)


def test_motif_aggregator_fafe_is_zero_with_single_aggregator():
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    rigids_1 = torch.zeros(1, 4, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    pred = rigids_1.clone()
    pred[0, 2, 4] = 2.0
    inputs = {
        "t": torch.tensor([[0.0]], dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.ones(1, 4, dtype=torch.bool),
            "rigids_noising_mask": torch.ones(1, 4, dtype=torch.bool),
            "rigids_seq_idx": torch.arange(4)[None],
            "motif_group_id": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor(
                [[False, False, True, False]]
            ),
            "gt_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
    }

    losses = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )

    assert torch.allclose(losses["aggregator_fafe"], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(losses["scaled_aggregator_fafe"], torch.tensor([0.0]), atol=1e-6)


def test_motif_intergroup_fafe_tracks_cross_group_geometry_only():
    """Inter-group FAFE uses real motif residues with group_i != group_j; ignores aggregators / intra-group."""
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    # Frames: g1 real, g1 real, g1 agg, scaffold, g2 real, g2 agg
    rigids_1 = torch.zeros(1, 6, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    rigids_1[0, 4, 4] = 10.0  # g2 real GT at x=10
    pred = rigids_1.clone()
    pred[0, 4, 4] = 12.0  # predicted g2 real at x=12 → relative error 2Å vs g1@0
    # Same-group relative error (and aggregator error) must not enter this term.
    pred[0, 1, 4] = 3.0
    pred[0, 5, 4] = 20.0

    inputs = {
        "t": torch.tensor([[0.0]], dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.ones(1, 6, dtype=torch.bool),
            "rigids_noising_mask": torch.ones(1, 6, dtype=torch.bool),
            "rigids_seq_idx": torch.arange(6)[None],
            "motif_group_id": torch.tensor([[1, 1, 1, 0, 2, 2]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor(
                [[False, False, True, False, False, True]]
            ),
            "gt_rot_vf": torch.zeros(1, 6, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 6, 3, dtype=torch.float32),
    }

    losses = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )

    # Cross-group real pairs only: (0,4)/(4,0) err=2; (1,4)/(4,1) err=|12-3|-|10-0|=1.
    # Mean sq err over 4 pairs = (4+1+4+1)/4 = 2.5 → sqrt(2.5/20^2 + eps).
    expected = torch.sqrt(torch.tensor([2.5 / 20.0**2 + 1e-6]))
    assert torch.allclose(losses["motif_intergroup_fafe"], expected, atol=1e-5)
    assert torch.allclose(losses["scaled_motif_intergroup_fafe"], expected, atol=1e-5)

    # Aggregator-only noise must not enter (same-group pairs are also excluded).
    # Perfect cross-group geometry still yields FAFE≈sqrt(eps) when pairs exist.
    pred_agg_only = rigids_1.clone()
    pred_agg_only[0, 5, 4] = 20.0
    outputs_agg = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred_agg_only),
        "pred_rot_vf": torch.zeros(1, 6, 3, dtype=torch.float32),
    }
    losses_agg = multiframe_fm_loss_dense_batch(
        inputs, outputs_agg, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )
    assert torch.allclose(
        losses_agg["motif_intergroup_fafe"],
        torch.sqrt(torch.tensor([1e-6])),
        atol=1e-6,
    )
    assert losses_agg["motif_intergroup_fafe"] < 0.1 * losses["motif_intergroup_fafe"]


def test_motif_intergroup_fafe_zero_with_single_group():
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    rigids_1 = torch.zeros(1, 4, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    pred = rigids_1.clone()
    pred[0, 1, 4] = 5.0
    inputs = {
        "t": torch.tensor([[0.0]], dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.ones(1, 4, dtype=torch.bool),
            "rigids_noising_mask": torch.ones(1, 4, dtype=torch.bool),
            "rigids_seq_idx": torch.arange(4)[None],
            "motif_group_id": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor(
                [[False, False, True, False]]
            ),
            "gt_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 4, 3, dtype=torch.float32),
    }
    losses = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )
    assert torch.allclose(losses["motif_intergroup_fafe"], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(losses["scaled_motif_intergroup_fafe"], torch.tensor([0.0]), atol=1e-6)


def test_motif_aggregator_pose_loss_is_group_normalized_and_t_lock_scaled():
    torch = pytest.importorskip("torch")
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    rigids_1 = torch.zeros(1, 8, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    pred = rigids_1.clone()
    # Two aggregators with the same trans error; real motif group sizes differ.
    pred[0, 2, 4] = 2.0
    pred[0, 7, 4] = 2.0
    inputs = {
        "t": torch.tensor([[0.25]], dtype=torch.float32),
        "motif_t_lock": torch.tensor(0.5, dtype=torch.float32),
        "rigids": {
            "rigids_t": rigids_1,
            "rigids_1": rigids_1,
            "rigids_mask": torch.ones(1, 8, dtype=torch.bool),
            "rigids_noising_mask": torch.ones(1, 8, dtype=torch.bool),
            "rigids_seq_idx": torch.arange(8)[None],
            "motif_group_id": torch.tensor([[1, 1, 1, 0, 0, 0, 2, 2]], dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.tensor(
                [[False, False, True, False, False, False, False, True]]
            ),
            "gt_rot_vf": torch.zeros(1, 8, 3, dtype=torch.float32),
        },
    }
    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, 8, 3, dtype=torch.float32),
    }

    losses = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )

    # Mean over two aggregators in motif time: err^2=4 / (1 - t_motif)^2,
    # Angstrom-to-nm scale 0.01, then active-window compensation 1 / motif_t_lock.
    assert torch.allclose(losses["aggregator_pose_loss"], torch.tensor([0.32]), atol=1e-6)

    inputs["t"] = torch.tensor([[0.75]], dtype=torch.float32)
    locked_losses = multiframe_fm_loss_dense_batch(
        inputs, outputs, direct_rot_vf_loss=True, motif_aggregator_weight=1.0
    )
    assert torch.allclose(locked_losses["aggregator_pose_loss"], torch.tensor([0.0]), atol=1e-6)


def test_sampling_churn_excludes_real_motif_and_reuses_aggregator_transform():
    torch = pytest.importorskip("torch")
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        sampling_noise_mode="churn",
    )
    trans_t = torch.tensor(
        [[[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    rotmats_t = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 4, 1, 1)
    delta_rot = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trans_t_hat = trans_t.clone()
    trans_t_hat[0, 2] = torch.tensor([1.0, 2.0, 0.0])
    trans_t_hat[0, 3] = torch.tensor([11.0, 0.0, 0.0])
    rotmats_t_hat = rotmats_t.clone()
    rotmats_t_hat[0, 2] = delta_rot

    motif_group_id = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    rigids_mask = torch.tensor([[True, True, True, True]])
    noising_mask = torch.tensor([[True, True, True, True]])
    aggregator_mask = torch.tensor([[False, False, True, False]])

    flow_mask = corrupter.motif_churn_flow_mask(
        motif_group_id,
        rigids_mask,
        noising_mask,
        aggregator_mask,
    )
    assert flow_mask.tolist() == [[False, False, True, True]]

    out_trans, out_rotmats = corrupter.motif_group_churn(
        torch.tensor([[0.5]], dtype=torch.float32),
        torch.tensor([[0.4]], dtype=torch.float32),
        trans_t,
        rotmats_t,
        trans_t_hat,
        rotmats_t_hat,
        motif_group_id,
        rigids_mask,
        noising_mask,
        aggregator_mask,
    )

    expected_trans = torch.tensor(
        [[[1.0, 3.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [11.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(out_trans, expected_trans, atol=1e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(out_trans[0, 0] - out_trans[0, 2]),
        torch.linalg.vector_norm(trans_t[0, 0] - trans_t[0, 2]),
        atol=1e-6,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(out_trans[0, 1] - out_trans[0, 2]),
        torch.linalg.vector_norm(trans_t[0, 1] - trans_t[0, 2]),
        atol=1e-6,
    )
    assert torch.allclose(out_rotmats[0, 0], delta_rot, atol=1e-6)
    assert torch.allclose(out_rotmats[0, 1], delta_rot, atol=1e-6)
    assert torch.allclose(out_rotmats[0, 2], delta_rot, atol=1e-6)
    assert torch.allclose(out_rotmats[0, 3], torch.eye(3), atol=1e-6)


def test_sampling_churn_matches_single_aggregator_projection():
    torch = pytest.importorskip("torch")
    from nomadflow.model.modules.frame_transformer import _apply_group_rigid_projection
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

    corrupter = MultiSE3Interpolant(
        prealign_noise=False,
        use_stochastic_centering=False,
        sampling_noise_mode="churn",
    )
    trans_t = torch.tensor(
        [[[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    quats_t = torch.tensor(
        [[
            [0.9238795, 0.3826834, 0.0, 0.0],
            [0.9238795, 0.0, 0.3826834, 0.0],
            [0.9238795, 0.0, 0.0, 0.3826834],
            [1.0, 0.0, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    rotmats_t = ru.quat_to_rot(quats_t)
    group_quat = torch.tensor([0.8660254, 0.0, 0.0, 0.5], dtype=torch.float32)
    aggregator_quat_t_hat = ru.quat_multiply(group_quat, quats_t[0, 2])
    rotmats_t_hat = rotmats_t.clone()
    rotmats_t_hat[0, 2] = ru.quat_to_rot(aggregator_quat_t_hat)
    trans_t_hat = trans_t.clone()
    trans_t_hat[0, 2] = torch.tensor([1.0, 2.0, 0.0])
    trans_t_hat[0, 3] = torch.tensor([11.0, 0.0, 0.0])

    motif_group_id = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    rigids_mask = torch.tensor([[True, True, True, True]])
    noising_mask = torch.tensor([[True, True, True, True]])
    aggregator_mask = torch.tensor([[False, False, True, False]])

    out_trans, out_rotmats = corrupter.motif_group_churn(
        torch.tensor([[0.5]], dtype=torch.float32),
        torch.tensor([[0.4]], dtype=torch.float32),
        trans_t,
        rotmats_t,
        trans_t_hat,
        rotmats_t_hat,
        motif_group_id,
        rigids_mask,
        noising_mask,
        aggregator_mask,
    )
    expected_trans, expected_quats = _apply_group_rigid_projection(
        ref_trans=trans_t[0, 2:3],
        pred_trans=trans_t_hat[0, 2:3],
        ref_quats=quats_t[0, 2:3],
        pred_quats=aggregator_quat_t_hat[None],
        apply_ref_trans=trans_t[0, :2],
        apply_ref_quats=quats_t[0, :2],
    )

    assert torch.allclose(out_trans[0, :2], expected_trans, atol=1e-5)
    assert torch.allclose(out_rotmats[0, :2], ru.quat_to_rot(expected_quats), atol=1e-5)
    assert torch.allclose(out_rotmats[0, 2], rotmats_t_hat[0, 2], atol=1e-6)


def test_embedder_token_motif_group_shapes_and_noop_for_zero_group():
    torch = pytest.importorskip("torch")
    from nomadflow.model.denoiser_modules.embedder import Embedder

    torch.manual_seed(0)
    embedder = Embedder(
        c_s=32,
        c_z=16,
        c_frame=24,
        c_framepair=12,
        c_hidden=16,
        num_heads=4,
        block_q=2,
        block_k=4,
        n_tfmr_blocks=1,
        n_pair_embed_blocks=1,
        max_rigids_idx=17,
        max_motif_group_id=8,
        use_sc_rigid_transformer=False,
        use_self_folding=False,
        use_qk_norm=False,
    )
    embedder.eval()

    assert torch.allclose(
        embedder.rigid_motif_group_embed.weight[0],
        torch.zeros_like(embedder.rigid_motif_group_embed.weight[0]),
    )
    assert torch.allclose(
        embedder.node_motif_group_embed.weight[0],
        torch.zeros_like(embedder.node_motif_group_embed.weight[0]),
    )

    B, N_tok, N_rig = 1, 3, 6
    token_to_rep_rigid = torch.tensor([[0, 2, 4]], dtype=torch.long)
    rigids_to_token = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    rigids = torch.zeros(B, N_rig, 7, dtype=torch.float32)
    rigids[..., 0] = 1.0
    rigids[..., 4:] = torch.tensor(
        [[[0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [2.0, 0.0, 0.0],
          [3.0, 0.0, 0.0],
          [4.0, 0.0, 0.0],
          [5.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    common_kwargs = dict(
        token_mask=torch.tensor([[True, True, True]]),
        token_seq=torch.tensor([[1, 2, 3]], dtype=torch.long),
        token_seq_idx=torch.tensor([[0, 1, 2]], dtype=torch.long),
        token_seq_noising_mask=torch.tensor([[False, False, False]]),
        token_seq_mask=torch.tensor([[True, True, True]]),
        token_chain_idx=torch.tensor([[0, 0, 0]], dtype=torch.long),
        token_is_unindexed_mask=torch.tensor([[False, False, False]]),
        token_gather_idx=token_to_rep_rigid,
        t=torch.tensor([[0.3]], dtype=torch.float32),
        rigids=rigids,
        rigids_element=torch.full((B, N_rig), -1, dtype=torch.long),
        rigids_charge=torch.zeros(B, N_rig, dtype=torch.float32),
        rigids_token_uid=rigids_to_token,
        rigids_idx=torch.zeros(B, N_rig, dtype=torch.long),
        rigids_motif_group_id=torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long),
        rigids_mask=torch.tensor([[True] * N_rig]),
        rigids_noising_mask=torch.tensor([[True] * N_rig]),
        rigids_is_atomized_mask=torch.tensor([[False] * N_rig]),
        rigids_is_motif_group_aggregator=torch.tensor([[False] * N_rig]),
        token_bonds=torch.zeros(B, N_tok, N_tok, dtype=torch.long),
    )

    zero_kwargs = dict(common_kwargs)
    zero_kwargs["rigids_motif_group_id"] = torch.zeros(B, N_rig, dtype=torch.long)
    out_zero = embedder(**zero_kwargs)

    out_nonzero = embedder(**common_kwargs)
    aggregator_kwargs = dict(common_kwargs)
    aggregator_kwargs["rigids_is_motif_group_aggregator"] = torch.tensor(
        [[False, False, True, False, False, False]]
    )
    out_aggregator = embedder(**aggregator_kwargs)

    assert out_zero["node_embed"].shape == (B, N_tok, 32)
    assert out_zero["edge_embed"].shape == (B, N_tok, N_tok, 16)
    assert out_zero["framepair_embed"].shape[0] == B
    assert out_zero["framepair_embed"].shape[-1] == 12

    # Zero-group path is deterministic and non-zero groups change activations.
    out_zero_again = embedder(
        **zero_kwargs,
    )
    assert torch.allclose(out_zero["node_embed"], out_zero_again["node_embed"], atol=1e-6)
    assert not torch.allclose(out_zero["node_embed"], out_nonzero["node_embed"], atol=1e-6)
    assert not torch.allclose(out_zero["edge_embed"], out_nonzero["edge_embed"], atol=1e-6)
    assert not torch.allclose(out_nonzero["rigids_embed"], out_aggregator["rigids_embed"], atol=1e-6)

    # Token-level motif groups are derived from all rigids in each token (amax aggregation).
    # token_to_rep_rigid picks [0, 2, 4], but rigid id at index 0 is zero while same token has id 5.
    derived_kwargs = dict(common_kwargs)
    derived_kwargs["rigids_motif_group_id"] = torch.tensor([[0, 5, 0, 2, 0, 0]], dtype=torch.long)
    out_derived = embedder(**derived_kwargs)
    assert out_derived["token_motif_group_id"].tolist() == [[5, 2, 0]]
