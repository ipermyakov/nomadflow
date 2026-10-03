import os

import tqdm
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from nomadflow.openfold.utils import rigid_utils as ru

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter
from torch_scatter import scatter_add
import numpy as np

from . import so3_utils
from .motif_time_schedule import MotifTimeSchedule, build_motif_time_schedule


def _resolve_cache_dir(cache_dir):
    if cache_dir:
        expanded = os.path.expanduser(os.path.expandvars(cache_dir))
        return os.path.abspath(expanded)
    repo_root = os.environ.get("REPO_ROOT")
    if repo_root:
        return os.path.join(repo_root, ".cache")
    return os.path.abspath(".cache")

def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]

@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices

# from eigenfold
class HarmonicPrior:
    def __init__(self, N = 256, a=3/(3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += a
            J[j,j] += a
            J[i,j] = J[j,i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1/D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)

    def sample(self, batch_dims=()):
        return self.P @ (torch.sqrt(self.D_inv)[:,None] * torch.randn(*batch_dims, self.N, 3, device=self.P.device))


def _centered_gaussian(batch, rigids_per_res, device):
    noise = torch.randn(batch.shape[0], rigids_per_res, 3, device=device)
    center = scatter(
        noise,
        index=batch,
        dim=0,
        reduce='mean'
    ).mean(dim=-2)
    return noise - center[batch][..., None, :]


def _uniform_so3(num_res, rigids_per_res, device):
    return torch.tensor(
        Rotation.random(num_res * rigids_per_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_res, rigids_per_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (~diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (
        ~diffuse_mask[..., None, None]
    )


def _unsqueeze_to_rank(tensor, rank):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    while tensor.ndim < rank:
        tensor = tensor.unsqueeze(-1)
    return tensor


def _sample_motif_group_motion_endpoints(
    batch,
    *,
    trans_1,
    rotmats_1,
    rigids_mask,
):
    rigids_data = batch.get("rigids")
    if rigids_data is None or "motif_group_id" not in rigids_data:
        return None

    enabled = batch.get("multi_motif_se3")
    if enabled is None:
        return None
    enabled = torch.as_tensor(enabled, device=trans_1.device, dtype=torch.bool).view(-1)
    if not bool(enabled.any()):
        return None

    trans_std = batch.get("multi_motif_se3_trans_std")
    if trans_std is None:
        trans_std = torch.full(
            (trans_1.shape[0],),
            16.0,
            device=trans_1.device,
            dtype=trans_1.dtype,
        )
    else:
        trans_std = torch.as_tensor(
            trans_std,
            device=trans_1.device,
            dtype=trans_1.dtype,
        ).view(-1)

    rigid_group_id = rigids_data["motif_group_id"].to(device=trans_1.device, dtype=torch.long)
    motif_move_mask = rigids_mask & (rigid_group_id > 0)
    motif_move_mask = motif_move_mask & enabled[:, None]
    if not bool(motif_move_mask.any()):
        return None

    motif_group_rot_0 = torch.eye(
        3,
        device=rotmats_1.device,
        dtype=rotmats_1.dtype,
    ).view(1, 1, 3, 3).expand_as(rotmats_1).clone()
    motif_group_trans_0 = torch.zeros_like(trans_1)
    motif_group_center = torch.zeros_like(trans_1)

    for batch_idx in range(trans_1.shape[0]):
        if not bool(enabled[batch_idx]):
            continue
        batch_group_ids = torch.unique(rigid_group_id[batch_idx, motif_move_mask[batch_idx]])
        for group_id in batch_group_ids.tolist():
            if group_id <= 0:
                continue
            group_mask = motif_move_mask[batch_idx] & (rigid_group_id[batch_idx] == int(group_id))
            if not bool(group_mask.any()):
                continue
            group_center = trans_1[batch_idx, group_mask].mean(dim=0)
            group_rot = torch.as_tensor(
                Rotation.random(random_state=np.random).as_matrix(),
                device=rotmats_1.device,
                dtype=rotmats_1.dtype,
            )
            # Sample an absolute random point for the group center at t=0.
            group_center_0 = torch.as_tensor(
                np.random.standard_normal(3),
                device=trans_1.device,
                dtype=trans_1.dtype,
            ) * trans_std[batch_idx]
            group_trans = group_center_0 - group_center
            motif_group_rot_0[batch_idx, group_mask] = group_rot
            motif_group_trans_0[batch_idx, group_mask] = group_trans
            motif_group_center[batch_idx, group_mask] = group_center

    return {
        "motif_move_mask": motif_move_mask,
        "motif_group_rot_0": motif_group_rot_0,
        "motif_group_trans_0": motif_group_trans_0,
        "motif_group_center": motif_group_center,
    }


def _average_incremental_global_quat(ref_quats, pred_quats):
    """Average global frame increments q_pred * inverse(q_ref)."""
    delta_quats = ru.quat_multiply(pred_quats, ru.invert_quat(ref_quats))
    ref_quat = delta_quats[0]
    quat_sign = torch.where(
        (delta_quats * ref_quat).sum(dim=-1, keepdim=True) < 0,
        delta_quats.new_tensor(-1.0),
        delta_quats.new_tensor(1.0),
    )
    mean_quat = F.normalize((delta_quats * quat_sign).mean(dim=0), dim=-1)
    return mean_quat

class MultiSE3Interpolant:
    def __init__(self,
                 min_t=1e-2,
                 separate_ot=True,
                 prealign_noise=True,
                 trans_preconditioning=False,
                 trans_prior_std=16,
                 rigids_per_res=3,#5,
                 lognorm_t_sched=False,
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 mixed_beta_t_sched=False,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 use_diffusion_forcing=False,
                 use_stochastic_centering=True,
                 center_on_motif=True,
                 sig_perturb=4.0,
                 use_uniform_rot_noise=False,
                 use_unwrapped_rot_noise=False,
                 unwrapped_rot_noise_sig=1.5,
                 trans_gamma=0.16,
                 trans_step_size=1.5,
                 rot_gamma=0.16,
                 rot_step_size=1.5,
                 rot_sample_schedule="linear",
                 sampling_noise_mode="churn",
                 sampling_churn=0.4,
                 churn_by_sigma=False,
                 num_timesteps=400,
                 use_euclidean_for_rots=False,
                 rot_sfm=False,
                 unconditional_guidance_alpha=0.0,
                 dynamic_cfg=False,
                 rot_g_t_fn="fn1",
                 trans_g_t_fn="fn1",
                 cache_dir=None,
                 motif_t_lock=None,
                 motif_time_schedule="linear",
    ):
        self._igso3 = None
        self.cache_dir = _resolve_cache_dir(cache_dir)

        self.min_t = min_t
        self.separate_ot = separate_ot
        self.prealign_noise = prealign_noise
        self.trans_preconditioning = trans_preconditioning
        self.trans_prior_std = trans_prior_std
        self.lognorm_t_sched = lognorm_t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.shift_time_scale = shift_time_scale
        self.mixed_beta_t_sched = mixed_beta_t_sched
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2

        self.trans_gamma = trans_gamma
        self.trans_step_size = trans_step_size
        self.rot_gamma = rot_gamma
        self.rot_step_size = rot_step_size
        assert rot_sample_schedule in ["linear", "exp"], rot_sample_schedule
        self.rot_sample_schedule = rot_sample_schedule
        assert sampling_noise_mode in ["churn", "euler", None], sampling_noise_mode
        self.sampling_noise_mode = sampling_noise_mode
        self.churn = sampling_churn
        self.churn_by_sigma = churn_by_sigma
        self.num_timesteps = num_timesteps

        print(self.igso3)

        self.rigids_per_res = rigids_per_res
        self.use_diffusion_forcing = use_diffusion_forcing
        self.use_stochastic_centering = use_stochastic_centering
        self.center_on_motif = center_on_motif
        self.sig_perturb = sig_perturb
        self.use_uniform_rot_noise = use_uniform_rot_noise
        self.use_euclidean_for_rots = use_euclidean_for_rots
        self.rot_sfm = rot_sfm
        self.use_unwrapped_rot_noise = use_unwrapped_rot_noise
        self.unwrapped_rot_noise_sig = unwrapped_rot_noise_sig

        self.unconditional_guidance_alpha = unconditional_guidance_alpha
        self.dynamic_cfg = dynamic_cfg

        assert rot_g_t_fn in ['fn1', 'fn2', 'fn3', 'fn4', 'fn5', 'fn6'] or rot_g_t_fn.startswith("poly") or rot_g_t_fn.startswith("tan")
        assert trans_g_t_fn in ['fn1', 'fn2', 'fn3', 'fn4', 'fn5', 'fn6'] or trans_g_t_fn.startswith("poly") or trans_g_t_fn.startswith("tan")
        self._rot_g_t_fn = rot_g_t_fn
        self._trans_g_t_fn = trans_g_t_fn

        self.motif_t_lock = motif_t_lock
        self.motif_time_schedule_name = motif_time_schedule
        self._motif_time_schedule = build_motif_time_schedule(motif_time_schedule)

    def _motif_schedule_context(
        self,
        trans_time: torch.Tensor,
        rot_time: torch.Tensor,
        motif_t_lock: float | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, MotifTimeSchedule | None]:
        if motif_t_lock is None or self._motif_time_schedule is None:
            return trans_time, rot_time, None, None
        schedule = self._motif_time_schedule
        motif_trans_time = schedule.motif_interp_time(trans_time, motif_t_lock)
        motif_rot_time = schedule.motif_interp_time(rot_time, motif_t_lock)
        locked = schedule.is_locked(trans_time, motif_t_lock)
        return motif_trans_time, motif_rot_time, locked, schedule

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=self.cache_dir)
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        if self.lognorm_t_sched:
            ln_sig = self.lognorm_mu + torch.randn(num_batch, device=self._device).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
            return t
        elif self.mixed_beta_t_sched:
            u = torch.rand(1)
            if u < 0.02:
                t = torch.rand(num_batch, device=self._device).float()
                return t * (1 - self.min_t)
            else:
                dist = torch.distributions.beta.Beta(self.beta_p1, self.beta_p2)
                t = dist.sample((num_batch,)).to(self._device)
                return t * (1 - self.min_t)
        else:
            t = torch.rand(num_batch, device=self._device).float()
            return t * (1 - self.min_t)

    def time_shift(self, t, n_res):
        shift_scale = np.sqrt(n_res / 100)
        return t / (t * (1 - shift_scale) + shift_scale)

    def _sample_trans_0(self, batch, device):
        trans_0 = _centered_gaussian(batch, self.rigids_per_res, device)
        trans_0 = trans_0 * self.trans_prior_std
        return trans_0.to(device)

    def _corrupt_trans(self, trans_1, trans_0, t, rigids_mask, diffuse_mask):
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * rigids_mask[..., None]

    def _sample_rotmats_0(self, rotmats_1):
        if self.use_uniform_rot_noise:
            rotmats_0 = _uniform_so3(rotmats_1.shape[0], rotmats_1.shape[1], rotmats_1.device)
        else:
            num_rigids = rotmats_1.shape[0] * rotmats_1.shape[1]
            noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_rigids).to(rotmats_1.device)
            noisy_rotmats = noisy_rotmats.view(*rotmats_1.shape[:2], 3, 3).float()
            rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        return rotmats_0

    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, rigids_mask, diffuse_mask):
        if self.use_unwrapped_rot_noise:
            noise_rotvec = torch.randn(rotmats_1.shape[:-1], device=rotmats_1.device) * self.unwrapped_rot_noise_sig

            # regularize the rotvec we apply to generate the rotation at time t
            noise_rotvec_t = noise_rotvec * (1 - t)[..., None]
            noise_rotvec_t_theta = torch.linalg.vector_norm(noise_rotvec_t, dim=-1)
            noise_rotvec_t_axis = F.normalize(noise_rotvec_t, dim=-1)
            k = torch.floor(noise_rotvec_t_theta / (2 * torch.pi))
            noise_rotvec_t_angle = noise_rotvec_t_theta - 2 * k * torch.pi
            reverse_axis = noise_rotvec_t_angle > torch.pi
            noise_rotvec_t_angle = noise_rotvec_t_angle * (~reverse_axis) + (2 * torch.pi - noise_rotvec_t_angle) * reverse_axis
            noise_rotvec_t_axis = noise_rotvec_t_axis * (1 - 2 * reverse_axis.float())[..., None]
            noise_rotvec_t = noise_rotvec_t_axis * noise_rotvec_t_angle[..., None]

            rotmats_t = so3_utils.apply_rotvec_to_rotmat(rotmats_1, noise_rotvec_t)

            # there's def a better way of doing this but im lazy
            rot_vf_axis = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
            rot_vf = F.normalize(rot_vf_axis, dim=-1) * torch.linalg.vector_norm(noise_rotvec, dim=-1)[..., None]

        elif self.use_euclidean_for_rots:
            rotvecs_1 = so3_utils.rotmat_to_rotvec(rotmats_1)
            # rotvecs_0 = torch.randn_like(rotvecs_1) * torch.pi
            rotvecs_0 = so3_utils.rotmat_to_rotvec(rotmats_0)
            rotvecs_t = (1 - t[..., None]) * rotvecs_0 + t[..., None] * rotvecs_1
            angle_t = torch.linalg.vector_norm(rotvecs_t + 1e-10, dim=-1)
            angle_t = torch.remainder(angle_t, 2 * torch.pi)
            flip_rotvec = angle_t >= torch.pi
            angle_t = (2 * torch.pi - angle_t) * flip_rotvec + angle_t * ~flip_rotvec
            axis_t = F.normalize(rotvecs_t, dim=-1) * (1 - 2 * flip_rotvec)[..., None]
            rotvecs_t = axis_t * angle_t[..., None]
            rotmats_t = so3_utils.rotvec_to_rotmat(rotvecs_t)
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)[..., None]
        else:
            rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)[..., None]

        if self.rot_sfm:
            eps_t = torch.sqrt(0.01 * t * (1-t) + 1e-4)
            self.igso3.to(rotmats_1.device)
            noisy_rotmats = self.igso3.sample(eps_t.view(-1), rotmats_1.shape[1]).to(rotmats_1.device)
            rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_t, noisy_rotmats)
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)[..., None]


        identity = torch.eye(3, device=t.device)
        rotmats_t = rotmats_t * rigids_mask[..., None, None] + identity[None, None] * (
            ~rigids_mask[..., None, None]
        )

        rot_vf = rot_vf * rigids_mask[..., None] + torch.zeros_like(rot_vf) * (
            ~rigids_mask[..., None]
        )
        rot_vf = rot_vf * diffuse_mask[..., None] + torch.zeros_like(rot_vf) * (
            ~diffuse_mask[..., None]
        )

        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask), rot_vf

    def _normalize_dense_time(self, t, batch_size, device):
        t = torch.as_tensor(t, device=device).float()
        if t.ndim == 0:
            return t.repeat(batch_size).view(batch_size, 1)
        if t.ndim == 1:
            if t.shape[0] == 1:
                return t.repeat(batch_size).view(batch_size, 1)
            if t.shape[0] == batch_size:
                return t[:, None]
            raise ValueError(f"dense time must have shape [1] or [{batch_size}], got {tuple(t.shape)}")
        if t.ndim == 2:
            if t.shape[0] != batch_size:
                raise ValueError(f"dense time batch mismatch: expected {batch_size}, got {tuple(t.shape)}")
            return t
        raise ValueError(f"dense time must be scalar, [B], or [B, K], got {tuple(t.shape)}")

    def sample_dense_corruption_endpoints(self, batch, rng_seed=None):
        rigids_data = batch["rigids"]

        rigids_1 = ru.Rigid.from_tensor_7(rigids_data["rigids_1"])
        trans_1 = rigids_1.get_trans()
        rotmats_1 = rigids_1.get_rots().get_rot_mats()

        rigids_mask = rigids_data["rigids_mask"]
        rigids_noising_mask = rigids_data["rigids_noising_mask"].bool()

        if rng_seed is not None:
            devices = [trans_1.device.index] if trans_1.is_cuda and trans_1.device.index is not None else []
            np_state = np.random.get_state()
            try:
                with torch.random.fork_rng(devices=devices):
                    np.random.seed(int(rng_seed))
                    torch.manual_seed(int(rng_seed))
                    if trans_1.is_cuda:
                        torch.cuda.manual_seed_all(int(rng_seed))
                    return self.sample_dense_corruption_endpoints(batch, rng_seed=None)
            finally:
                np.random.set_state(np_state)

        rotmats_0 = self._sample_rotmats_0(rotmats_1)
        trans_0 = torch.randn_like(trans_1) * self.trans_prior_std
        trans_0 = trans_0 - trans_0.mean(dim=1)[..., None, :]

        if self.center_on_motif:
            multi_motif_se3 = batch.get("multi_motif_se3")
            use_group_center = False
            if multi_motif_se3 is not None:
                multi_motif_se3 = torch.as_tensor(
                    multi_motif_se3, device=rigids_mask.device, dtype=torch.bool
                ).view(-1)
                use_group_center = bool(multi_motif_se3.any())
            motif_group_id = rigids_data.get("motif_group_id") if use_group_center else None
            if motif_group_id is not None and bool((motif_group_id > 0).any().item()):
                center_mask = rigids_mask & (motif_group_id.to(rigids_mask.device) > 0)
            else:
                center_mask = rigids_mask * (~rigids_noising_mask)
            fixed_trans_1 = trans_1 * center_mask[..., None]
            fixed_center = fixed_trans_1.sum(dim=1) / center_mask.long().sum(dim=1)[..., None].clip(min=1)
            global_center = (trans_1 * rigids_mask[..., None]).sum(dim=1) / rigids_mask.long().sum(dim=1)[..., None].clip(min=1)
            use_fixed_center = center_mask.any(dim=-1)
            center = fixed_center * use_fixed_center[..., None] + global_center * (~use_fixed_center[..., None])
        else:
            center = (trans_1 * rigids_mask[..., None]).sum(dim=1) / rigids_mask.long().sum(dim=1)[..., None].clip(min=1)

        trans_1 = trans_1 - center[..., None, :]
        if 'atom' in batch:
            atom_data = batch['atom']
            atom_data["atom14"] = atom_data["atom14"] - center[..., None, None, :]
            atom_data["atom14"] *= atom_data['atom14_mask'][..., None]
            atom_data['atom14_gt_positions'] = atom_data['atom14_gt_positions'] - center[..., None, None, :]
            atom_data['atom14_gt_positions'] *= atom_data['atom14_mask'][..., None]
            atom_data['atom14_alt_gt_positions'] = atom_data['atom14_alt_gt_positions'] - center[..., None, None, :]
            atom_data['atom14_alt_gt_positions'] *= atom_data['atom14_mask'][..., None]

        if self.prealign_noise:
            align_mask = (rigids_mask * rigids_noising_mask).bool()
            multi_motif_se3 = batch.get("multi_motif_se3")
            use_group_exclude = False
            if multi_motif_se3 is not None:
                multi_motif_se3 = torch.as_tensor(
                    multi_motif_se3, device=align_mask.device, dtype=torch.bool
                ).view(-1)
                use_group_exclude = bool(multi_motif_se3.any())
            motif_group_id = rigids_data.get("motif_group_id") if use_group_exclude else None
            if motif_group_id is not None:
                align_mask = align_mask & ~(motif_group_id.to(align_mask.device) > 0)
            align_batch = torch.tile(
                torch.arange(rigids_mask.shape[0])[..., None],
                (1, rigids_mask.shape[1])
            ).to(align_mask.device)
            align_batch = align_batch[align_mask]

            _, _, align_rot_mats = align_structures(
                trans_0[align_mask],
                align_batch,
                trans_1[align_mask]
            )
            num_batch = trans_0.shape[0]

            if align_rot_mats.shape[0] != num_batch:
                num_pad = num_batch - align_rot_mats.shape[0]
                eye = torch.eye(3, device=align_rot_mats.device, dtype=align_rot_mats.dtype)
                align_rot_mats_safe = torch.cat([
                    align_rot_mats,
                    eye[None].expand(num_pad, -1, -1)
                ], dim=0)
            else:
                align_rot_mats_safe = align_rot_mats

            trans_0 = torch.einsum("bni,bij->bnj", trans_0, align_rot_mats_safe)

        if self.use_stochastic_centering:
            stoch_center = torch.randn_like(center) * self.sig_perturb
            trans_0 = trans_0 + stoch_center[..., None, :]

        motif_motion = _sample_motif_group_motion_endpoints(
            batch,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            rigids_mask=rigids_mask,
        )

        rigids_0 = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_0), trans=trans_0
        )
        centered_rigids_1 = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_1), trans=trans_1
        )

        motif_motion = motif_motion or {}
        return {
            "trans_0": trans_0,
            "rotmats_0": rotmats_0,
            "trans_1": trans_1,
            "rotmats_1": rotmats_1,
            "rigids_0": rigids_0.to_tensor_7(),
            "rigids_1": centered_rigids_1.to_tensor_7(),
            "rigids_mask": rigids_mask,
            "rigids_noising_mask": rigids_noising_mask,
            "center": center,
            "motif_t_lock": batch.get("motif_t_lock", self.motif_t_lock),
            "motif_move_mask": motif_motion.get("motif_move_mask"),
            "motif_group_rot_0": motif_motion.get("motif_group_rot_0"),
            "motif_group_trans_0": motif_motion.get("motif_group_trans_0"),
            "motif_group_center": motif_motion.get("motif_group_center"),
        }

    def _apply_dense_endpoint_constraints(
        self,
        *,
        endpoints,
        trans_interp,
        rotmats_interp,
        rot_vf,
        trans_time,
        rot_time,
    ):
        trans_time_is_zero = trans_time.eq(0)[..., None]
        trans_time_is_one = trans_time.eq(1)[..., None]
        trans_0_masked = _trans_diffuse_mask(
            endpoints["trans_0"],
            endpoints["trans_1"],
            endpoints["rigids_noising_mask"],
        )
        trans_interp = torch.where(trans_time_is_zero, trans_0_masked, trans_interp)
        trans_interp = torch.where(trans_time_is_one, endpoints["trans_1"], trans_interp)

        rot_time_is_zero = rot_time.eq(0)[..., None, None]
        rot_time_is_one = rot_time.eq(1)[..., None, None]
        rotmats_0_masked = _rots_diffuse_mask(
            endpoints["rotmats_0"],
            endpoints["rotmats_1"],
            endpoints["rigids_noising_mask"],
        )
        rotmats_interp = torch.where(rot_time_is_zero, rotmats_0_masked, rotmats_interp)
        rotmats_interp = torch.where(rot_time_is_one, endpoints["rotmats_1"], rotmats_interp)

        rot_vf_endpoint_mask = rot_time.eq(0) | rot_time.eq(1)
        rot_vf = torch.where(rot_vf_endpoint_mask[..., None], torch.zeros_like(rot_vf), rot_vf)
        return trans_interp, rotmats_interp, rot_vf

    def _apply_motif_group_motion(
        self,
        *,
        endpoints,
        trans_interp,
        rotmats_interp,
        rot_vf,
        trans_time,
        rot_time,
    ):
        motif_move_mask = endpoints.get("motif_move_mask")
        if motif_move_mask is None or not bool(motif_move_mask.any()):
            return trans_interp, rotmats_interp, rot_vf

        group_rot_0 = endpoints.get("motif_group_rot_0")
        group_trans_0 = endpoints.get("motif_group_trans_0")
        group_center = endpoints.get("motif_group_center")
        if group_rot_0 is None or group_trans_0 is None or group_center is None:
            raise ValueError("Motif motion endpoints missing group transform tensors")

        motif_t_lock = endpoints.get("motif_t_lock")
        motif_trans_time, motif_rot_time, locked, _ = self._motif_schedule_context(
            trans_time,
            rot_time,
            motif_t_lock,
        )

        trans_time_is_zero = motif_trans_time.eq(0)[..., None]
        trans_time_is_one = motif_trans_time.eq(1)[..., None]
        rot_time_is_zero = motif_rot_time.eq(0)[..., None, None]
        rot_time_is_one = motif_rot_time.eq(1)[..., None, None]

        group_rotvec_0 = so3_utils.rotmat_to_rotvec(group_rot_0)
        group_rotvec_t = (1.0 - motif_rot_time[..., None]) * group_rotvec_0
        group_rot_t = so3_utils.rotvec_to_rotmat(group_rotvec_t)
        group_trans_t = (1.0 - motif_trans_time[..., None]) * group_trans_0

        trans_1_centered = endpoints["trans_1"] - group_center
        group_center_t = group_center + group_trans_t
        motif_trans_interp = (
            torch.einsum("bnij,bnj->bni", group_rot_t, trans_1_centered)
            + group_center_t
        )
        motif_rotmats_interp = torch.einsum(
            "bnij,bnjk->bnik",
            group_rot_t,
            endpoints["rotmats_1"],
        )
        group_center_0 = group_center + group_trans_0
        motif_trans_0 = (
            torch.einsum("bnij,bnj->bni", group_rot_0, trans_1_centered)
            + group_center_0
        )
        motif_rotmats_0 = torch.einsum(
            "bnij,bnjk->bnik",
            group_rot_0,
            endpoints["rotmats_1"],
        )

        motif_trans_interp = _trans_diffuse_mask(
            motif_trans_interp,
            endpoints["trans_1"],
            motif_move_mask,
        )
        motif_trans_interp = torch.where(trans_time_is_zero, motif_trans_0, motif_trans_interp)
        motif_trans_interp = torch.where(trans_time_is_one, endpoints["trans_1"], motif_trans_interp)

        motif_rotmats_interp = _rots_diffuse_mask(
            motif_rotmats_interp,
            endpoints["rotmats_1"],
            motif_move_mask,
        )
        motif_rotmats_interp = torch.where(rot_time_is_zero, motif_rotmats_0, motif_rotmats_interp)
        motif_rotmats_interp = torch.where(rot_time_is_one, endpoints["rotmats_1"], motif_rotmats_interp)

        if locked is not None:
            locked_trans = locked[..., None]
            locked_rot = locked[..., None, None]
            motif_trans_interp = torch.where(
                locked_trans,
                endpoints["trans_1"],
                motif_trans_interp,
            )
            motif_rotmats_interp = torch.where(
                locked_rot,
                endpoints["rotmats_1"],
                motif_rotmats_interp,
            )

        motif_rot_vf = torch.zeros_like(rot_vf)

        trans_interp = torch.where(motif_move_mask[..., None], motif_trans_interp, trans_interp)
        rotmats_interp = torch.where(
            motif_move_mask[..., None, None],
            motif_rotmats_interp,
            rotmats_interp,
        )
        rot_vf = torch.where(motif_move_mask[..., None], motif_rot_vf, rot_vf)
        return trans_interp, rotmats_interp, rot_vf

    def interpolate_dense_corruption(self, endpoints, trans_t, rot_t=None):
        batch_size = endpoints["rigids_1"].shape[0]
        device = endpoints["rigids_1"].device
        trans_time = self._normalize_dense_time(trans_t, batch_size, device)
        rot_time = trans_time if rot_t is None else self._normalize_dense_time(rot_t, batch_size, device)

        trans_interp = self._corrupt_trans(
            endpoints["trans_1"],
            endpoints["trans_0"],
            trans_time,
            endpoints["rigids_mask"],
            endpoints["rigids_noising_mask"],
        )
        rotmats_interp, rot_vf = self._corrupt_rotmats(
            endpoints["rotmats_1"],
            endpoints["rotmats_0"],
            rot_time,
            endpoints["rigids_mask"],
            endpoints["rigids_noising_mask"],
        )
        trans_interp, rotmats_interp, rot_vf = self._apply_dense_endpoint_constraints(
            endpoints=endpoints,
            trans_interp=trans_interp,
            rotmats_interp=rotmats_interp,
            rot_vf=rot_vf,
            trans_time=trans_time,
            rot_time=rot_time,
        )
        trans_interp, rotmats_interp, rot_vf = self._apply_motif_group_motion(
            endpoints=endpoints,
            trans_interp=trans_interp,
            rotmats_interp=rotmats_interp,
            rot_vf=rot_vf,
            trans_time=trans_time,
            rot_time=rot_time,
        )

        rigids_interp = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_interp),
            trans=trans_interp,
        )
        return {
            "rigids_t": rigids_interp.to_tensor_7(),
            "rotmats_t": rotmats_interp,
            "trans_t": trans_interp,
            "gt_rot_vf": rot_vf,
            "trans_time": trans_time,
            "rot_time": rot_time,
        }


    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]

        rigids_1 = ru.Rigid.from_tensor_7(res_data["rigids_1"])
        # [N, 5, 3]
        trans_1 = rigids_1.get_trans()
        # [N, 5, 3, 3]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()

        # [N]
        res_mask = res_data["res_mask"]
        rigids_noising_mask = res_data["rigids_noising_mask"]
        rigidwise_batch = res_data.batch[..., None].expand(-1, self.rigids_per_res)

        # [B]
        rigidwise_t = batch['rigidwise_t']
        if self.shift_time_scale:
            rigidwise_t = self.time_shift(rigidwise_t, (res_data.batch == 0).float().sum().item())

        rotmats_0 = self._sample_rotmats_0(rotmats_1)
        trans_0 = self._sample_trans_0(res_data.batch, trans_1.device)

        if self.center_on_motif:
            # center samples on the center of the fixed region
            # if there is no fixed region, center the whole sample
            center_mask = res_mask[..., None] * (~rigids_noising_mask)
            center_batch = rigidwise_batch[center_mask]
            fixed_trans_1 = trans_1[center_mask]
            fixed_center = scatter(
                fixed_trans_1,
                index=center_batch,
                dim_size=(res_data.batch.max().item() + 1),
                reduce='mean'
            )
            global_center = scatter(
                trans_1.flatten(0, 1),
                index=torch.repeat_interleave(res_data.batch, self.rigids_per_res),
                dim_size=(res_data.batch.max().item() + 1),
                reduce='mean'
            )
            use_fixed_center = scatter(
                torch.ones_like(center_batch, dtype=torch.bool),
                index=center_batch,
                dim_size=(res_data.batch.max().item() + 1),
                reduce='any'
            )
            center = fixed_center * use_fixed_center[..., None] + global_center * (~use_fixed_center[..., None])
        else:
            center = scatter(
                trans_1.flatten(0, 1),
                index=torch.repeat_interleave(res_data.batch, self.rigids_per_res),
                dim_size=(res_data.batch.max().item() + 1),
                reduce='mean'
            )
        trans_1 = trans_1 - center[rigidwise_batch]

        # this is just so we can calculate atom14 rmsds
        res_data["atom14"] = res_data["atom14"] - center[res_data.batch][..., None,:]
        res_data['atom14'] *= res_data['atom14_mask'][..., None]
        res_data['atom14_gt_positions'] = res_data['atom14_gt_positions'] - center[res_data.batch][..., None, :]
        res_data['atom14_gt_positions'] *= res_data['atom14_mask'][..., None]
        res_data['atom14_alt_gt_positions'] = res_data['atom14_alt_gt_positions'] - center[res_data.batch][..., None, :]
        res_data['atom14_alt_gt_positions'] *= res_data['atom14_mask'][..., None]


        if self.prealign_noise:
            # rotate each structure to align as best as possible with noise
            align_mask = res_mask[..., None] * rigids_noising_mask
            align_batch = rigidwise_batch[align_mask]

            _, _, align_rot_mats = align_structures(
                trans_0[align_mask],
                align_batch,
                trans_1[align_mask]
            )
            align_rot_mats_safe = align_rot_mats
            trans_0 = torch.einsum("nki,nij->nkj", trans_0, align_rot_mats_safe[res_data.batch])

        if self.use_stochastic_centering:
            stoch_center = torch.randn_like(center) * self.sig_perturb
            trans_0 = trans_0 + stoch_center[rigidwise_batch]


        trans_t = self._corrupt_trans(
            trans_1,
            trans_0,
            rigidwise_t,
            res_mask[..., None],
            rigids_noising_mask,
        )
        rotmats_t, rot_vf = self._corrupt_rotmats(
            rotmats_1,
            rotmats_0,
            rigidwise_t,
            res_mask[..., None],
            rigids_noising_mask,
        )
        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t), trans=trans_t
        )
        res_data["rigids_t"] = rigids_t.to_tensor_7()
        res_data["rotmats_t"] = rotmats_t
        res_data["trans_t"] = trans_t

        # we also overwrite the ground truth rigids_1 since we've done some centering
        rigids_1 = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_1), trans=trans_1
        )
        res_data["rigids_1"] = rigids_1.to_tensor_7()

        return batch

    # @torch.no_grad()
    def corrupt_dense_batch(self, batch, rng_seed=None):
        rigids_data = batch["rigids"]
        endpoints = self.sample_dense_corruption_endpoints(batch, rng_seed=rng_seed)
        interpolated = self.interpolate_dense_corruption(
            endpoints,
            batch['trans_t'],
            batch['rot_t'],
        )

        rigids_data["rigids_t"] = interpolated["rigids_t"]
        rigids_data["rotmats_t"] = interpolated["rotmats_t"]
        rigids_data["trans_t"] = interpolated["trans_t"]
        rigids_data["rigids_1"] = endpoints["rigids_1"]
        rigids_data["gt_rot_vf"] = interpolated["gt_rot_vf"]

        return batch

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        else:
            raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def g_t_inv(self, s, g_t_fn):
        assert g_t_fn == 'fn1'
        return (- (0.2 * s + 1) + ((0.2 * s + 1) ** 2 - 4 * s * (0.01 * s - 1)) ** 0.5) / (2 * s)

    def _center_trans(self, trans_t, batch, trans_noising_mask=None):
        if trans_noising_mask is not None:
            fixed_trans = trans_t[~trans_noising_mask]
            fixed_batch = batch[..., None].expand(-1, trans_t.shape[-2])
            fixed_batch = fixed_batch[~trans_noising_mask]
            center = scatter(
                fixed_trans,
                index=fixed_batch,
                dim=0,
                reduce='mean',
                dim_size=int(batch.max().item() + 1)
            )
        else:
            center = scatter(
                trans_t,
                index=batch,
                dim=0,
                reduce='mean'
            ).mean(dim=-2)
        return trans_t - center[batch][..., None, :], center

    def trans_churn(self, d_t, t, trans_t, noising_mask, noise_scale_multiplier=1):
        if self.sampling_noise_mode == "churn":
            if self.churn_by_sigma:
                curr_sigma = self.g_t(t, self._trans_g_t_fn)
                new_sigma = (1 + self.churn) * curr_sigma
                t_hat = self.g_t_inv(new_sigma, self._trans_g_t_fn)
                t_hat = torch.clamp(t_hat, min=0)
                d_t_hat = d_t + (t - t_hat)
            else:
                t_hat = torch.clamp(t - self.churn * d_t, min=0)
                d_t_hat = d_t * (1 + self.churn)
            noise_scale = torch.sqrt(
                2 * self.g_t(t_hat, self._trans_g_t_fn) * self.trans_gamma - 2 * self.g_t(t, self._trans_g_t_fn) * self.trans_gamma
            )
            if isinstance(noise_scale, torch.Tensor):
                noise_scale = noise_scale.to(device=trans_t.device, dtype=trans_t.dtype)
            while isinstance(noise_scale, torch.Tensor) and noise_scale.ndim < trans_t.ndim:
                noise_scale = noise_scale.unsqueeze(-1)
            trans_t_hat = trans_t + torch.randn_like(trans_t) * noise_scale * noise_scale_multiplier * 10
            trans_t_hat = _trans_diffuse_mask(trans_t_hat, trans_t, noising_mask)
            return t_hat, d_t_hat, trans_t_hat
        else:
            return t, d_t, trans_t

    def motif_churn_flow_mask(
            self,
            motif_group_id,
            rigids_mask,
            noising_mask,
            aggregator_mask=None,
            device=None,
    ):
        if device is None:
            device = noising_mask.device
        total_mask = (
            rigids_mask.to(device=device).bool()
            & noising_mask.to(device=device).bool()
        )
        if motif_group_id is None:
            return total_mask

        group_id = motif_group_id.to(device=device, dtype=torch.long)
        if aggregator_mask is None:
            aggregator_mask = torch.zeros_like(total_mask, dtype=torch.bool)
        else:
            aggregator_mask = aggregator_mask.to(device=device).bool()

        motif_real_mask = total_mask & (group_id > 0) & ~aggregator_mask
        return total_mask & ~motif_real_mask

    def _motif_group_churn_mask(
            self,
            motif_group_id,
            rigids_mask,
            noising_mask,
            aggregator_mask,
            device,
    ):
        group_id = motif_group_id.to(device=device, dtype=torch.long)
        total_mask = (
            rigids_mask.to(device=device).bool()
            & noising_mask.to(device=device).bool()
        )
        if aggregator_mask is None:
            aggregator_mask = torch.zeros_like(total_mask, dtype=torch.bool)
        else:
            aggregator_mask = aggregator_mask.to(device=device).bool()

        real_group_mask = total_mask & (group_id > 0) & ~aggregator_mask
        aggregator_group_mask = total_mask & (group_id > 0) & aggregator_mask
        return group_id, real_group_mask, aggregator_group_mask

    def _motif_group_churn_noise_scales(self, t, t_hat, batch_size, device, dtype):
        t = self._normalize_dense_time(t, batch_size, device)
        t_hat = self._normalize_dense_time(t_hat, batch_size, device)

        trans_noise_scale = torch.sqrt(
            2 * self.g_t(t_hat, self._trans_g_t_fn) * self.trans_gamma
            - 2 * self.g_t(t, self._trans_g_t_fn) * self.trans_gamma
        ).to(device=device, dtype=dtype)
        rot_noise_scale = torch.sqrt(
            2 * self.g_t(t_hat, self._rot_g_t_fn) * self.rot_gamma
            - 2 * self.g_t(t, self._rot_g_t_fn) * self.rot_gamma
        ).to(device=device, dtype=dtype)
        return trans_noise_scale, rot_noise_scale

    def _apply_single_motif_group_churn(
            self,
            *,
            trans_t,
            rotmats_t,
            out_trans,
            out_rotmats,
            batch_idx,
            group_mask,
            trans_noise_scale,
            rot_noise_scale,
    ):
        device = trans_t.device
        group_churn_scale = group_mask.float().sum().clamp(min=1).rsqrt()
        group_center = trans_t[batch_idx, group_mask].float().mean(dim=0)
        group_trans = (
            torch.randn(3, device=device, dtype=trans_t.dtype)
            * trans_noise_scale[batch_idx, 0]
            * group_churn_scale
            * 10
        )
        group_rotvec = (
            torch.randn(3, device=device, dtype=trans_t.dtype)
            * rot_noise_scale[batch_idx, 0]
            * group_churn_scale
        )
        group_rot = so3_utils.rotvec_to_rotmat(group_rotvec.float()[None])[0]

        group_trans_t = trans_t[batch_idx, group_mask].float()
        out_trans[batch_idx, group_mask] = (
            torch.einsum("ij,nj->ni", group_rot, group_trans_t - group_center)
            + group_center
            + group_trans.float()
        ).to(dtype=out_trans.dtype)
        out_rotmats[batch_idx, group_mask] = torch.einsum(
            "ij,njk->nik",
            group_rot,
            rotmats_t[batch_idx, group_mask].float(),
        ).to(dtype=out_rotmats.dtype)

    def motif_group_churn(
            self,
            t,
            t_hat,
            trans_t,
            rotmats_t,
            trans_t_hat,
            rotmats_t_hat,
            motif_group_id,
            rigids_mask,
            noising_mask,
            aggregator_mask=None,
    ):
        if self.sampling_noise_mode != "churn" or motif_group_id is None:
            return trans_t_hat, rotmats_t_hat

        group_id, real_group_mask, aggregator_group_mask = self._motif_group_churn_mask(
            motif_group_id,
            rigids_mask,
            noising_mask,
            aggregator_mask,
            trans_t.device,
        )
        if not bool(real_group_mask.any()) or not bool(aggregator_group_mask.any()):
            return trans_t_hat, rotmats_t_hat

        batch_size = trans_t.shape[0]

        out_trans = trans_t_hat.clone()
        out_rotmats = rotmats_t_hat.clone()

        for batch_idx in range(batch_size):
            batch_mask = real_group_mask[batch_idx]
            if not bool(batch_mask.any()):
                continue
            batch_group_ids = torch.unique(group_id[batch_idx, batch_mask])
            for curr_group_id in batch_group_ids.tolist():
                if curr_group_id <= 0:
                    continue
                curr_group_mask = batch_mask & (group_id[batch_idx] == int(curr_group_id))
                if not bool(curr_group_mask.any()):
                    continue

                curr_aggregator_mask = aggregator_group_mask[batch_idx] & (
                    group_id[batch_idx] == int(curr_group_id)
                )
                if not bool(curr_aggregator_mask.any()):
                    continue

                aggregator_idx = torch.nonzero(
                    curr_aggregator_mask,
                    as_tuple=False,
                ).view(-1)[0]
                aggregator_quat_t = ru.rot_to_quat(
                    rotmats_t[batch_idx, aggregator_idx].float()
                )
                aggregator_quat_t_hat = ru.rot_to_quat(
                    rotmats_t_hat[batch_idx, aggregator_idx].float()
                )
                group_delta_quat_t_hat = ru.quat_multiply(
                    aggregator_quat_t_hat,
                    ru.invert_quat(aggregator_quat_t),
                )
                group_delta_rot_t_hat = ru.quat_to_rot(group_delta_quat_t_hat)
                aggregator_trans_t_hat = trans_t_hat[batch_idx, aggregator_idx].float()

                curr_group_trans = trans_t[batch_idx, curr_group_mask].float()
                curr_group_center = curr_group_trans.mean(dim=0)
                out_trans[batch_idx, curr_group_mask] = (
                    torch.einsum(
                        "ij,nj->ni",
                        group_delta_rot_t_hat,
                        curr_group_trans - curr_group_center,
                    )
                    + aggregator_trans_t_hat
                ).to(dtype=out_trans.dtype)
                curr_group_quats = ru.rot_to_quat(
                    rotmats_t[batch_idx, curr_group_mask].float()
                )
                out_rotmats[batch_idx, curr_group_mask] = ru.quat_to_rot(
                    ru.quat_multiply(
                        group_delta_quat_t_hat,
                        curr_group_quats,
                    )
                ).to(dtype=out_rotmats.dtype)

        return out_trans, out_rotmats

    def _trans_score(self, t, trans_1, trans_t):
        t = _unsqueeze_to_rank(t, trans_t.ndim)
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return (t * trans_vf - trans_t) / (1-t)

    def trans_euler_step(
            self,
            d_t,
            t,
            trans_1,
            trans_t,
            noising_mask,
            vf_scale=1,
            step_size_scale=1,
    ):
        add_noise = (self.sampling_noise_mode == "euler")
        use_score = (self.sampling_noise_mode is not None)
        t_b = _unsqueeze_to_rank(t, trans_t.ndim)
        d_t_b = _unsqueeze_to_rank(d_t, trans_t.ndim)
        trans_vf = (trans_1 - trans_t) / (1 - t_b)

        step_size = self.trans_step_size if use_score else 1
        step_size = step_size * vf_scale
        step_size_scale = torch.as_tensor(
            step_size_scale,
            device=trans_t.device,
            dtype=trans_t.dtype,
        )
        step_size = step_size * _unsqueeze_to_rank(step_size_scale, trans_t.ndim)
        g_t = self.g_t(t, self._trans_g_t_fn)
        g_t = _unsqueeze_to_rank(g_t, trans_t.ndim)
        gamma = self.trans_gamma
        dW_t = torch.randn_like(trans_t) * torch.sqrt(d_t_b) * 10
        trans_score = self._trans_score(t, trans_1, trans_t) * use_score

        ode_mask = t_b > 0.99
        trans_next_ode = trans_t + (trans_vf * d_t_b) * step_size
        if add_noise:
            noise_t = torch.sqrt(2 * g_t * gamma) * dW_t
            trans_next_sde = trans_t + (trans_vf * d_t_b + trans_score * g_t * d_t_b) * step_size + noise_t
        else:
            trans_next_sde = trans_t + (trans_vf * d_t_b + trans_score * g_t * d_t_b) * step_size
        trans_next = torch.where(ode_mask, trans_next_ode, trans_next_sde)

        trans_next[~noising_mask] = trans_t[~noising_mask]
        return trans_next

    def motif_group_euler_step(
            self,
            d_t,
            t,
            pred_trans_1,
            pred_quats_1,
            trans_t,
            rotmats_t,
            quats_t,
            trans_next,
            rotmats_next,
            motif_group_id,
            rigids_mask,
            aggregator_mask=None,
            trans_vf_scale=1,
            rot_vf_scale=1,
    ):
        if motif_group_id is None:
            return trans_next, rotmats_next

        group_id = motif_group_id.to(device=pred_trans_1.device, dtype=torch.long)
        group_mask = rigids_mask.to(device=pred_trans_1.device).bool() & (group_id > 0)
        if aggregator_mask is None:
            aggregator_mask = torch.zeros_like(group_mask, dtype=torch.bool)
        else:
            aggregator_mask = aggregator_mask.to(device=pred_trans_1.device).bool()
        real_group_mask = group_mask & ~aggregator_mask
        aggregator_group_mask = group_mask & aggregator_mask
        if not bool(real_group_mask.any()) or not bool(aggregator_group_mask.any()):
            return trans_next, rotmats_next

        batch_size = pred_trans_1.shape[0]

        out_trans = trans_next.clone()
        out_rotmats = rotmats_next.clone()

        for batch_idx in range(batch_size):
            batch_mask = real_group_mask[batch_idx]
            if not bool(batch_mask.any()):
                continue
            batch_group_ids = torch.unique(group_id[batch_idx, batch_mask])
            for curr_group_id in batch_group_ids.tolist():
                if curr_group_id <= 0:
                    continue
                curr_group_mask = batch_mask & (group_id[batch_idx] == int(curr_group_id))
                if not bool(curr_group_mask.any()):
                    continue

                curr_aggregator_mask = aggregator_group_mask[batch_idx] & (
                    group_id[batch_idx] == int(curr_group_id)
                )
                if not bool(curr_aggregator_mask.any()):
                    continue

                aggregator_idx = torch.nonzero(
                    curr_aggregator_mask,
                    as_tuple=False,
                ).view(-1)[0]
                aggregator_quat_t = ru.rot_to_quat(
                    rotmats_t[batch_idx, aggregator_idx].float()
                )
                aggregator_quat_next = ru.rot_to_quat(
                    rotmats_next[batch_idx, aggregator_idx].float()
                )
                group_delta_quat_next = ru.quat_multiply(
                    aggregator_quat_next,
                    ru.invert_quat(aggregator_quat_t),
                )
                group_delta_rot_next = ru.quat_to_rot(group_delta_quat_next)
                aggregator_trans_next = trans_next[batch_idx, aggregator_idx].float()

                curr_group_trans = trans_t[batch_idx, curr_group_mask].float()
                curr_center = curr_group_trans.mean(dim=0)
                curr_centered = curr_group_trans - curr_center
                out_trans[batch_idx, curr_group_mask] = (
                    torch.einsum("ij,nj->ni", group_delta_rot_next, curr_centered)
                    + aggregator_trans_next
                ).to(dtype=out_trans.dtype)
                curr_group_quats = ru.rot_to_quat(
                    rotmats_t[batch_idx, curr_group_mask].float()
                )
                out_rotmats[batch_idx, curr_group_mask] = ru.quat_to_rot(
                    ru.quat_multiply(
                        group_delta_quat_next,
                        curr_group_quats,
                    )
                ).to(dtype=out_rotmats.dtype)

        return out_trans, out_rotmats


    def trans_guidance_step(
            self,
            d_t,
            t,
            trans_1_cond,
            trans_1_uncond,
            trans_t,
            noising_mask,
            vf_scale=1
    ):
        add_noise = (self.sampling_noise_mode == "euler")
        use_score = (self.sampling_noise_mode is not None)

        if self.dynamic_cfg:
            tau1 = 1 - 0.5
            tau2 = 1 - 0.9
            _t = t[0]
            if t < tau2:
                guidance_alpha = 0
            elif t < tau1:
                guidance_alpha = (tau1 - t) / (tau1 - tau2)
            else:
                guidance_alpha = 1

            if self.unconditional_guidance_alpha != 0:
                guidance_alpha = guidance_alpha * self.unconditional_guidance_alpha
        else:
            guidance_alpha = self.unconditional_guidance_alpha


        trans_vf_cond = (trans_1_cond - trans_t) / (1 - t)
        trans_vf_uncond = (trans_1_uncond - trans_t) / (1 - t)
        trans_vf = trans_vf_cond * (1 - guidance_alpha) + trans_vf_uncond * guidance_alpha

        step_size = self.trans_step_size if use_score else 1
        step_size = step_size * vf_scale
        g_t = self.g_t(t, self._trans_g_t_fn)
        gamma = self.trans_gamma
        dW_t = torch.randn_like(trans_t) * torch.sqrt(d_t) * 10
        trans_score_cond = self._trans_score(t, trans_1_cond, trans_t) * use_score
        trans_score_uncond = self._trans_score(t, trans_1_uncond, trans_t) * use_score
        trans_score = trans_score_cond * (1 - guidance_alpha) + trans_score_uncond * guidance_alpha

        # if t > 0.99:
        # TODO: fix hack (what happens if not all t > 0.99?)
        if (isinstance(t, torch.Tensor) and (t > 0.99).any()) or (not isinstance(t, torch.Tensor) and t > 0.99):
            trans_next = trans_t + (trans_vf * d_t) * step_size
        else:
            if add_noise:
                noise_t = torch.sqrt(2 * g_t * gamma) * dW_t
                trans_next = trans_t + (trans_vf * d_t + trans_score * g_t * d_t) * step_size + noise_t
            else:
                # print(trans_vf.shape, d_t.shape, trans_score.shape, g_t.shape, step_size.shape)
                trans_next = trans_t + (trans_vf * d_t + trans_score * g_t * d_t) * step_size

        trans_next[~noising_mask] = trans_t[~noising_mask]
        return trans_next


    def rot_churn(self, d_t, t, rotmats_t, noising_mask, noise_scale_multiplier=1):
        if self.sampling_noise_mode == "churn":
            gamma = self.rot_gamma
            if self.churn_by_sigma:
                curr_sigma = self.g_t(t, self._rot_g_t_fn)
                new_sigma = (1 + self.churn) * curr_sigma
                t_hat = self.g_t_inv(new_sigma, self._rot_g_t_fn)
                t_hat = torch.clamp(t_hat, min=0)
                d_t_hat = d_t + (t - t_hat)
            else:
                t_hat = torch.clamp(t - self.churn * d_t, min=0)
                d_t_hat = d_t * (1 + self.churn)

            noise_scale = torch.sqrt(
                2 * self.g_t(t_hat, self._rot_g_t_fn) * gamma - 2 * self.g_t(t, self._rot_g_t_fn) * gamma
            )
            if isinstance(noise_scale, torch.Tensor):
                noise_scale = noise_scale.to(device=rotmats_t.device, dtype=rotmats_t.dtype)
            dB_rot = torch.randn(
                *rotmats_t.shape[:-2],
                3,
                device=rotmats_t.device,
                dtype=rotmats_t.dtype,
            )
            while isinstance(noise_scale, torch.Tensor) and noise_scale.ndim < dB_rot.ndim:
                noise_scale = noise_scale.unsqueeze(-1)
            churn_multiplier = noise_scale_multiplier
            if isinstance(churn_multiplier, torch.Tensor):
                churn_multiplier = churn_multiplier.to(
                    device=rotmats_t.device,
                    dtype=rotmats_t.dtype,
                )
                while churn_multiplier.ndim < dB_rot.ndim:
                    churn_multiplier = churn_multiplier.unsqueeze(-1)
            dB_rot = dB_rot * noise_scale * churn_multiplier
            rotmats_t_hat = so3_utils.rot_mult(
                rotmats_t,
                so3_utils.rotvec_to_rotmat(dB_rot)
            )
            rotmats_t_hat = _rots_diffuse_mask(rotmats_t_hat, rotmats_t, noising_mask)
            return t_hat, d_t_hat, rotmats_t_hat
        else:
            return t, d_t, rotmats_t

    def _rots_score(self, t, rotmats_1, rotmats_t):
        ls = torch.arange(1000, device=rotmats_t.device)
        rel_rotmat = torch.einsum("...ij,...jk->...ik", rotmats_t.transpose(-1, -2), rotmats_1).view(-1, 3, 3)
        omega, _, _ = so3_utils.angle_from_rotmat(rel_rotmat)
        omega = omega.view(-1)
        sigma = (
            ((1-t) * 1.5).square()
            + (t * 0.1).square()
        ).sqrt()
        if isinstance(sigma, torch.Tensor):
            while sigma.ndim < rotmats_t.ndim - 2:
                sigma = sigma.unsqueeze(-1)
            sigma = sigma.expand(rotmats_t.shape[:-2]).reshape(-1).to(omega.device)
        else:
            sigma = torch.full_like(omega, float(sigma), device=omega.device)
        prefactor = so3_utils.dlog_igso3_expansion(omega, sigma, ls)
        prefactor = prefactor.view(rotmats_t.shape[:-2])
        omega = omega.view(rotmats_t.shape[:-2])
        rot_score = (prefactor / omega)[..., None] * so3_utils.calc_rot_vf(rotmats_1, rotmats_t)
        return rot_score

    def rots_euler_step(
            self,
            d_t,
            t,
            rotmats_1,
            rotmats_t,
            noising_mask,
            vf_scale=1,
            rot_vf=None
    ):
        add_noise = (self.sampling_noise_mode == "euler")
        use_score = (self.sampling_noise_mode is not None)

        if rot_vf is None:
            if self.rot_sample_schedule == "linear":
                t_b = _unsqueeze_to_rank(t, rotmats_t.ndim - 1)
                rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t_b)
            elif self.rot_sample_schedule == "exp":
                rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
            else:
                raise ValueError(f"unrecognized rot_sample_schedule {self.rot_sample_schedule}")
        rot_vf = rot_vf.float()
        t_b = _unsqueeze_to_rank(t, rot_vf.ndim)
        d_t_b = _unsqueeze_to_rank(d_t, rot_vf.ndim)

        # TODO: this is probably numerically unstable for fixed residues
        # i've seen this nan sometimes on fixed residues
        # which leads to some weird behavior downstream
        # current band-aid patch is just to force replace the fixed residues
        if use_score:
            rot_score = self._rots_score(t, rotmats_1, rotmats_t) * use_score
        else:
            rot_score = torch.zeros_like(rot_vf)

        g_t = self.g_t(t, self._rot_g_t_fn)
        g_t = _unsqueeze_to_rank(g_t, rot_vf.ndim)
        step_size = self.rot_step_size * vf_scale
        gamma = self.rot_gamma
        dB_rot = torch.randn_like(rot_vf) * torch.sqrt(d_t_b)
        ode_mask = t_b > 0.99
        rotvec_ode = d_t_b * rot_vf * step_size
        rotvec_t = d_t_b * rot_vf + d_t_b * g_t * rot_score
        if add_noise:
            noise_t = torch.sqrt(2 * g_t * gamma) * dB_rot
            rotvec_sde = rotvec_t * step_size + noise_t
        else:
            rotvec_sde = step_size * rotvec_t
        mat_t = so3_utils.rotvec_to_rotmat(torch.where(ode_mask, rotvec_ode, rotvec_sde))

        rotmats_next = torch.einsum("...ij,...jk->...ik", rotmats_t, mat_t)

        rotmats_next[~noising_mask] = rotmats_t[~noising_mask]
        return rotmats_next


    def rots_guidance_step(
            self,
            d_t,
            t,
            rotmats_1_cond,
            rotmats_1_uncond,
            rotmats_t,
            noising_mask,
            vf_scale=1
    ):
        add_noise = (self.sampling_noise_mode == "euler")
        use_score = (self.sampling_noise_mode is not None)

        if self.rot_sample_schedule == "linear":
            rot_vf_cond = so3_utils.calc_rot_vf(rotmats_t, rotmats_1_cond) / (1 - t)
            rot_vf_uncond = so3_utils.calc_rot_vf(rotmats_t, rotmats_1_uncond) / (1 - t)
        elif self.rot_sample_schedule == "exp":
            rot_vf_cond = so3_utils.calc_rot_vf(rotmats_t, rotmats_1_cond)
            rot_vf_uncond = so3_utils.calc_rot_vf(rotmats_t, rotmats_1_uncond)
        else:
            raise ValueError(f"unrecognized rot_sample_schedule {self.rot_sample_schedule}")

        # TODO: this is probably numerically unstable for fixed residues
        # i've seen this nan sometimes on fixed residues
        # which leads to some weird behavior downstream
        # current band-aid patch is just to force replace the fixed residues
        if use_score:
            rot_score_cond = self._rots_score(t, rotmats_1_cond, rotmats_t) * use_score
            rot_score_uncond = self._rots_score(t, rotmats_1_uncond, rotmats_t) * use_score
        else:
            rot_score_cond = torch.zeros_like(rot_vf_cond)
            rot_score_uncond = torch.zeros_like(rot_vf_uncond)

        rot_vf = rot_vf_cond * (1 - self.unconditional_guidance_alpha) + rot_vf_uncond * self.unconditional_guidance_alpha
        rot_score = rot_score_cond * (1 - self.unconditional_guidance_alpha) + rot_score_uncond * self.unconditional_guidance_alpha

        g_t = self.g_t(t, self._rot_g_t_fn)
        step_size = self.rot_step_size * vf_scale
        gamma = self.rot_gamma
        dB_rot = torch.randn_like(rot_vf) * torch.sqrt(d_t)
        # if t > 0.99:
        # TODO: fix hack (what happens if not all t > 0.99?)
        if (isinstance(t, torch.Tensor) and (t > 0.99).any()) or (not isinstance(t, torch.Tensor) and t > 0.99):
            mat_t = so3_utils.rotvec_to_rotmat(d_t * rot_vf * step_size)
        else:
            if add_noise:
                rotvec_t = d_t * rot_vf + d_t * g_t * rot_score
                noise_t = torch.sqrt(2 * g_t * gamma) * dB_rot
                mat_t = so3_utils.rotvec_to_rotmat(rotvec_t * step_size + noise_t)
            else:
                rotvec_t = d_t * rot_vf + d_t * g_t * rot_score
                mat_t = so3_utils.rotvec_to_rotmat(step_size * rotvec_t)

        rotmats_next = torch.einsum("...ij,...jk->...ik", rotmats_t, mat_t)

        rotmats_next[~noising_mask] = rotmats_t[~noising_mask]
        return rotmats_next
