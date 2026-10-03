import os
import pathlib

import torch

script_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(script_dir)
from . import common

def _mask_reswise_key(data, key):
    mask_data = data.clone()
    orig_mask = mask_data['residue'][key]
    select = (torch.rand_like(orig_mask.float()) < 0.8)
    mask_data['residue'][key] = orig_mask & select
    return mask_data


def _test_protein_res(model, data):
    # res_mask
    print("res_mask")
    mask_res_data = _mask_reswise_key(data, 'res_mask')
    out = model(mask_res_data)
    res_mask = mask_res_data['residue']['res_mask']
    assert torch.isclose(out['node_features'][~res_mask], torch.tensor(0.)).all()
    # res_noising_mask
    print("res_noising_mask")
    mask_res_data = _mask_reswise_key(data, 'res_noising_mask')
    out = model(mask_res_data)
    res_noising_mask = mask_res_data['residue']['res_noising_mask']
    assert torch.isclose(
        out['final_rigids'].to_tensor_7()[~res_noising_mask],
        data['residue']['rigids_t'][~res_noising_mask]
    ).all()


def _test_protein_seq(model, data):
    # seq_mask
    print("seq_mask")
    mask_seq_data = _mask_reswise_key(data, 'seq_mask')
    out = model(mask_seq_data)
    seq_mask = mask_seq_data['residue']['seq_mask']
    assert torch.isclose(out['seq_probs'][~seq_mask], torch.tensor(0.)).all()
    # seq_mask
    print("seq_noising_mask")
    mask_seq_data = _mask_reswise_key(data, 'seq_noising_mask')
    out = model(mask_seq_data)
    seq_noising_mask = mask_seq_data['residue']['seq_noising_mask']
    assert torch.isclose(
        out['seq_probs'][~seq_noising_mask],
        data['residue']['seq_probs_t'][~seq_noising_mask]
    ).all()


def _mask_reswise_vec_key(data, key):
    mask_data = data.clone()
    orig_mask = mask_data['residue'][key]
    select = (torch.rand_like(orig_mask[..., 0].float()) < 0.8)
    mask_data['residue'][key] = orig_mask & select[..., None]
    return mask_data


def _test_protein_chis(model, data):
    # seq_mask
    # print("chi_mask")
    # mask_chi_data = _mask_reswise_vec_key(data, 'chi_mask')
    # out = model(mask_chi_data)
    # chi_mask = mask_chi_data['residue']['chi_mask']
    # assert torch.isclose(
    #     out['seq_probs'][~chi_mask], torch.tensor(0.)
    # ).all()
    # seq_mask
    print("chi_noising_mask")
    mask_chi_data = _mask_reswise_vec_key(data, 'chi_noising_mask')
    out = model(mask_chi_data)
    chi_noising_mask = mask_chi_data['residue']['chi_noising_mask']
    assert torch.isclose(
        out['decoded_chis_gt_seq'][~chi_noising_mask],
        data['residue']['chis_t'][~chi_noising_mask]
    ).all()


def test_protein_fisher():
    model = common.MODELS['protein_fisher']()
    task = common.TASKS['protein_fisher']
    data = common.get_data(task)

    _test_protein_res(model, data)
    _test_protein_seq(model, data)

    model2 = common.MODELS['protein_fisher'](use_ipmp_trunk=True)
    _test_protein_res(model2, data)
    _test_protein_seq(model2, data)

    model3 = common.MODELS['protein_fisher'](
        use_ipmp_trunk=True,
        seq_matrix_updates=True
    )
    _test_protein_res(model3, data)
    _test_protein_seq(model3, data)


def test_protein_multichi_fisher():
    model = common.MODELS['protein_multichi_fisher']()
    task = common.TASKS['protein_multichi_fisher']
    data = common.get_data(task)

    _test_protein_res(model, data)
    _test_protein_seq(model, data)
    _test_protein_chis(model, data)


def test_protein_dirichlet():
    model = common.MODELS['protein_dirichlet']()
    task = common.TASKS['protein_dirichlet']
    data = common.get_data(task)

    _test_protein_res(model, data)
    _test_protein_seq(model, data)


def test_protein_multichi_dirichlet():
    model = common.MODELS['protein_multichi_dirichlet']()
    task = common.TASKS['protein_multichi_dirichlet']
    data = common.get_data(task)

    _test_protein_res(model, data)
    _test_protein_seq(model, data)
    _test_protein_chis(model, data)