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
from nomadflow.data.featurize.mol.sampling import smiles_to_struct
from nomadflow.data.featurize.tokenize import Tokenized
from nomadflow.data.featurize.sampling import sample_noise_from_struct_template
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


class UnconditionalSamplingFromSMILES(SamplingTask):
    task_name: str = "unconditional_smiles"

    def __init__(
        self,
        smiles: str,
        num_samples: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.smiles = smiles
        self.num_samples = num_samples

    def sample_data(self):
        struct = smiles_to_struct(self.smiles)
        for _ in range(self.num_samples):
            token_data, rigid_data, token_bonds, _ = sample_noise_from_struct_template(
                struct,
            )
            data = Tokenized(
                tokens=token_data,
                rigids=rigid_data,
                bonds=token_bonds,
                structure=struct
            )
            task_data = {
                "t": np.array([1.0], dtype=float),
            }

            yield featurize_inference(data, task_data, task_name=self.kwargs.get("name", self.task_name))