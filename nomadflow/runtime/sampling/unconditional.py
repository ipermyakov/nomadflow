""" Code for generating unconditional generation inputs

"""
import copy
import string

from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import tree
from torch_geometric.data import HeteroData
import numpy as np

from nomadflow.boltz.data.types import Structure

from nomadflow.openfold.data import residue_constants
from nomadflow.openfold.data import data_transforms
from nomadflow.openfold.utils import rigid_utils as ru
from nomadflow.data.featurize.tokenize import Tokenized
from nomadflow.data.featurize.sampling import generate_protein_structure_template, sample_noise_from_struct_template
from nomadflow.data.featurize.assembler import featurize_inference


from .task import SamplingTask


def dict_cat(data):
    return tree.map_structure(torch.cat, *data)


def residue_to_atom37(residue):
    residue_37 = torch.full((91, 3), torch.nan)
    aa_3lt = residue.get_resname()
    atoms = list(residue.get_atoms())
    for atom in atoms:
        atom_name = atom.get_name()
        atom_idx = residue_constants.atom_types.index(atom_name)
        residue_37[atom_idx] = atom.get_coord()
    return residue_37, residue_constants.resname_to_idx[aa_3lt]


def _centered_gaussian(num_rigids):
    noise = torch.randn(num_rigids, 3)
    return noise - noise.mean(dim=0, keepdim=True)

def _uniform_so3(num_rigids):
    return torch.as_tensor(
        Rotation.random(num_rigids).as_quat(),
        dtype=torch.float32,
    )


class UnconditionalSampling(SamplingTask):
    task_name: str = "unconditional"

    def __init__(
        self,
        sample_length: int,
        num_samples: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sample_length = sample_length
        self.num_samples = num_samples

    def sample_data(self):
        for _ in range(self.num_samples):
            chain_lens = {
                'A': self.sample_length,
            }
            struct = generate_protein_structure_template(chain_lens)
            token_data, rigid_data, token_bonds, _ = sample_noise_from_struct_template(
                struct
            )
            data = Tokenized(
                tokens=token_data,
                rigids=rigid_data,
                bonds=token_bonds,
                structure=struct
            )

            rigids_1 = data.rigids['tensor7']
            rigids_noising_mask = np.ones(rigids_1.shape[:-1], dtype=bool)
            seq_noising_mask = np.ones(data.tokens['token_idx'].shape, dtype=bool)

            task_data = {
                "t": np.array([1.0], dtype=float),
                "rigids_noising_mask": rigids_noising_mask,
                "seq_noising_mask": seq_noising_mask,
                "copy_indexed_token_mask": None,
                "copy_unindexed_token_mask": None,
            }

            yield featurize_inference(data, task_data, task_name=self.kwargs.get("name", self.task_name))