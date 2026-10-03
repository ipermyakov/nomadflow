import torch
import numpy as np

from .task import TrainingTask, sample_task_t

class UnconditionalGeneration(TrainingTask):
    name: str = "unconditional"
    def __init__(
        self,
        prob=0.0,
        t_sched='lognorm',
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        shift_time_scale=False,
        t_min=0.01,
        t_max=0.99,
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        self.prob = prob
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

    def sample_t_and_mask(self, data, np_rng=None, torch_generator=None):
        residues = data.residues
        atoms = data.atoms

        t = sample_task_t(
            t_sched=self.t_sched,
            lognorm_mu=self.lognorm_mu,
            lognorm_sig=self.lognorm_sig,
            beta_p1=self.beta_p1,
            beta_p2=self.beta_p2,
            t_min=self.t_min,
            t_max=self.t_max,
            device='cpu',
            np_rng=np_rng,
            torch_generator=torch_generator,
        )

        atom_noising_mask = np.ones(atoms.shape[0], dtype=bool)
        res_type_noising_mask = np.ones(residues.shape[0], dtype=bool)
        copy_indexed_residue_mask = np.zeros_like(res_type_noising_mask)
        copy_unindexed_residue_mask = np.zeros_like(res_type_noising_mask)
        copy_atomized_residue_mask = np.zeros_like(res_type_noising_mask)
        motif_group_id_residue = np.zeros(residues.shape[0], dtype=np.int32)

        return {
            "t": t.numpy(force=True),
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
            "motif_group_id_residue": motif_group_id_residue,
            "multi_motif_se3": False,
            "multi_motif_se3_trans_std": 16.0,
        }

    def max_added_tokens(self, _):
        return 0
