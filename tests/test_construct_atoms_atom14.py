import numpy as np
import pytest
import torch

pytest.importorskip("mashumaro")

from nomadflow.boltz.data import const
from nomadflow.data.featurize.sampling import (
    construct_atoms,
    generate_protein_structure_template,
    sample_noise_from_struct_template,
)
from nomadflow.data.featurize.tokenize import RES_TO_AA, Tokenized
from nomadflow.openfold.utils import rigid_utils as ru
from nomadflow.utils import coarse_grain as cg_utils


def _build_tokenized_protein(num_residues: int):
    struct = generate_protein_structure_template({"A": num_residues})
    token_data, rigid_data, _, _ = sample_noise_from_struct_template(struct)
    for token in token_data:
        token["res_type"] = const.token_ids["ALA"]
        token["seq_noising_mask"] = True
    data = Tokenized(
        tokens=token_data,
        rigids=rigid_data,
        bonds=np.zeros(0),
        structure=struct,
    )
    return struct, data, token_data, rigid_data


def _batched_atom14(token_data, rigid_data):
    standard_tensor7 = np.stack(
        [
            rigid_data["tensor7"][token["rigid_idx"] : token["rigid_idx"] + token["rigid_num"]]
            for token in token_data
        ],
        axis=0,
    )
    standard_rigids = ru.Rigid.from_tensor_7(torch.as_tensor(standard_tensor7))
    standard_seq = torch.as_tensor([RES_TO_AA[token["res_type"]] for token in token_data])
    standard_mask = torch.ones_like(standard_seq, dtype=torch.bool)
    atom14, _ = cg_utils.compute_atom14_from_cg_frames(
        standard_rigids,
        standard_mask,
        standard_seq,
        return_atom_mask=True,
    )
    return atom14


@pytest.mark.parametrize("num_residues", [1, 3, 5])
def test_construct_atoms_uses_per_residue_atom14_rows(num_residues):
    struct, data, token_data, rigid_data = _build_tokenized_protein(num_residues)
    batched_atom14 = _batched_atom14(token_data, rigid_data)

    out = construct_atoms(data, struct, take_copy_structure=False)

    assert batched_atom14.shape[0] == num_residues
    for residue_idx, res in enumerate(out.residues):
        atoms = out.atoms[res["atom_idx"] : res["atom_idx"] + res["atom_num"]]
        got = np.stack([atom["coords"] for atom in atoms])
        expected = batched_atom14[residue_idx, : got.shape[0]].numpy()
        assert np.allclose(got, expected, atol=1e-4)


def test_batched_atom14_unsqueeze_breaks_per_residue_indexing():
    batched_atom14 = torch.randn(3, 14, 3)
    wrongly_batched = batched_atom14.unsqueeze(0)

    assert wrongly_batched[0].shape == (3, 14, 3)
    with pytest.raises(IndexError):
        _ = wrongly_batched[1]

    assert batched_atom14[1].shape == (14, 3)
