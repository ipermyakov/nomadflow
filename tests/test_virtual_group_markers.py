from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tree")

from nomadflow.boltz.data.types import Atom, Chain, Interface, Residue, Structure
from nomadflow.data.datasets import datamodule as dm
from nomadflow.data.featurize.cropper import Cropper
from nomadflow.data.featurize.tokenize import (
    Rigid,
    Token,
    TokenBond,
    Tokenized,
    append_motif_group_aggregators,
)
from nomadflow.runtime.training.motif_scaffold import MotifScaffolding


def test_append_motif_group_aggregators_adds_centroids_once():
    tokens = np.zeros(2, dtype=Token)
    tokens["token_idx"] = [0, 1]
    tokens["rigid_idx"] = [0, 2]
    tokens["rigid_num"] = [2, 1]
    tokens["resolved_mask"] = True

    rigids = np.zeros(3, dtype=Rigid)
    rigids["rigid_idx"] = [0, 1, 2]
    rigids["token_idx"] = [0, 0, 1]
    rigids["is_present"] = True
    rigids["tensor7"][:, 0] = 1.0
    rigids["tensor7"][:, 4:] = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rigids["motif_group_id"] = [1, 1, 3]

    marked_tokens, marked_rigids = append_motif_group_aggregators(
        tokens,
        rigids,
        np_rng=np.random.default_rng(123),
    )
    marked_tokens_again, marked_rigids_again = append_motif_group_aggregators(
        marked_tokens,
        marked_rigids,
    )

    marker_tokens = marked_tokens[marked_tokens["is_motif_group_aggregator"]]
    marker_rigids = marked_rigids[marked_rigids["is_motif_group_aggregator"]]

    assert marker_tokens["motif_group_id"].tolist() == [1, 3]
    assert marker_rigids["motif_group_id"].tolist() == [1, 3]
    assert marker_tokens["rigid_idx"].tolist() == marker_rigids["rigid_idx"].tolist()
    assert marker_rigids["rigids_noising_mask"].tolist() == [True, True]
    assert np.allclose(marker_tokens[0]["center_coords"], [1.0, 0.0, 0.0])
    assert np.allclose(marker_rigids[0]["tensor7"][4:], [1.0, 0.0, 0.0])
    assert np.allclose(np.linalg.norm(marker_rigids["tensor7"][:, :4], axis=-1), 1.0)
    assert not np.allclose(marker_rigids["tensor7"][:, :4], np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(marked_tokens_again) == len(marked_tokens)
    assert len(marked_rigids_again) == len(marked_rigids)


def test_prepare_reserves_marker_slots_in_crop_budget(monkeypatch):
    tokens = np.zeros(4, dtype=Token)
    tokens["token_idx"] = np.arange(4)
    tokens["res_idx"] = np.arange(4)
    tokens["res_name"] = ["ALA", "GLY", "SER", "THR"]
    tokens["rigid_idx"] = np.arange(4)
    tokens["rigid_num"] = 1
    tokens["asym_id"] = 0
    tokens["resolved_mask"] = True
    tokens["center_coords"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    rigids = np.zeros(4, dtype=Rigid)
    rigids["rigid_idx"] = np.arange(4)
    rigids["token_idx"] = np.arange(4)
    rigids["tensor7"][:, 0] = 1.0
    rigids["tensor7"][:, 4:] = tokens["center_coords"]
    rigids["is_present"] = True
    rigids["motif_group_id"] = [1, 1, 2, 2]

    chains = np.zeros(1, dtype=Chain)
    chains["name"] = ["A"]
    chains["asym_id"] = [0]
    chains["res_num"] = [4]
    structure = Structure(
        atoms=np.zeros(0, dtype=[]),
        bonds=np.zeros(0, dtype=[]),
        residues=np.zeros(0, dtype=[]),
        chains=chains,
        connections=np.zeros(0, dtype=[]),
        interfaces=np.zeros(0, dtype=Interface),
        mask=np.array([True]),
    )

    class DummyTask:
        def sample_t_and_mask(self, struct, np_rng=None, torch_generator=None):
            del struct, np_rng, torch_generator
            return {}

        def max_added_tokens(self, real_token_count):
            assert real_token_count == 4
            return 0

    class MarkerAwareCropper:
        def __init__(self):
            self.max_tokens = None
            self.source_motif_group_count = None

        def crop(self, tokenized_data, max_tokens, random, chain_id=None, interface_id=None):
            del random, chain_id, interface_id
            self.max_tokens = max_tokens
            motif_group_ids = tokenized_data.rigids["motif_group_id"]
            self.source_motif_group_count = len(
                np.unique(motif_group_ids[motif_group_ids > 0])
            )

            cropped_tokens = tokenized_data.tokens[:max_tokens].copy()
            cropped_rigids = tokenized_data.rigids[:max_tokens].copy()
            cropped_tokens["token_idx"] = np.arange(len(cropped_tokens))
            cropped_tokens["rigid_idx"] = np.arange(len(cropped_tokens))
            cropped_rigids["token_idx"] = np.arange(len(cropped_rigids))
            cropped_rigids["rigid_idx"] = np.arange(len(cropped_rigids))
            return Tokenized(
                tokens=cropped_tokens,
                rigids=cropped_rigids,
                bonds=tokenized_data.bonds[:0],
                structure=tokenized_data.structure,
            )

    cropper = MarkerAwareCropper()
    monkeypatch.setattr(dm, "load_input", lambda record, data_dir: structure)
    monkeypatch.setattr(
        dm,
        "tokenize_structure",
        lambda struct, task_data, np_rng=None: (
            tokens.copy(),
            rigids.copy(),
            np.zeros(0, dtype=TokenBond),
        ),
    )

    prepared = dm.prepare_training_tokenized_example(
        dataset=SimpleNamespace(data_dir="unused"),
        sample=SimpleNamespace(record=SimpleNamespace(id="rec-a"), chain_id=None, interface_id=None),
        task=DummyTask(),
        cropper=cropper,
        max_crop_residues=3,
        remove_mol_types=[],
        mask_nonstandard=False,
        np_rng=np.random.RandomState(0),
    )

    final_tokens = prepared.tokenized.tokens
    marker_tokens = final_tokens[final_tokens["is_motif_group_aggregator"]]

    assert cropper.source_motif_group_count == 2
    assert cropper.max_tokens == 1
    assert len(final_tokens) <= 3
    assert marker_tokens["motif_group_id"].tolist() == [1]


def test_prepare_precrops_before_sampling_and_pins_all_motif_copies(monkeypatch):
    preview_tokens = np.zeros(4, dtype=Token)
    preview_tokens["token_idx"] = np.arange(4)
    preview_tokens["res_idx"] = np.arange(4)
    preview_tokens["rigid_idx"] = np.arange(4)
    preview_tokens["rigid_num"] = 1
    preview_tokens["asym_id"] = 0
    preview_tokens["resolved_mask"] = True
    preview_tokens["center_coords"][:, 0] = np.arange(4, dtype=np.float32)

    preview_rigids = np.zeros(4, dtype=Rigid)
    preview_rigids["rigid_idx"] = np.arange(4)
    preview_rigids["token_idx"] = np.arange(4)
    preview_rigids["tensor7"][:, 0] = 1.0
    preview_rigids["is_present"] = True
    preview_rigids["rigids_noising_mask"] = True

    residues = np.zeros(4, dtype=Residue)
    residues["name"] = "ALA"
    residues["res_idx"] = np.arange(4)
    residues["is_standard"] = True
    residues["is_present"] = True

    chains = np.zeros(1, dtype=Chain)
    chains["name"] = "A"
    chains["asym_id"] = 0
    chains["res_num"] = 4

    structure = Structure(
        atoms=np.zeros(0, dtype=Atom),
        bonds=np.zeros(0, dtype=[]),
        residues=residues,
        chains=chains,
        connections=np.zeros(0, dtype=[]),
        interfaces=np.zeros(0, dtype=Interface),
        mask=np.array([True]),
    )

    class DummyMotifTask(MotifScaffolding):
        def __init__(self):
            pass

        def max_motif_groups(self):
            return 2

        def sample_t_and_mask(self, struct, np_rng=None, torch_generator=None):
            del np_rng, torch_generator
            present = np.flatnonzero(struct.residues["is_present"])
            assert len(present) == 3
            groups = np.zeros(len(struct.residues), dtype=np.int32)
            groups[present[:2]] = [1, 2]
            return {"actual": True, "motif_group_id_residue": groups}

    def fake_tokenize(struct, task_data, **kwargs):
        del kwargs
        if not task_data.get("actual", False):
            return (
                preview_tokens.copy(),
                preview_rigids.copy(),
                np.zeros(0, dtype=TokenBond),
            )

        motif_residues = np.flatnonzero(task_data["motif_group_id_residue"])
        scaffold_residues = np.flatnonzero(struct.residues["is_present"])
        tokens = np.zeros(5, dtype=Token)
        tokens["token_idx"] = np.arange(5)
        tokens["res_idx"] = np.concatenate([scaffold_residues, motif_residues])
        tokens["rigid_idx"] = np.arange(5)
        tokens["rigid_num"] = 1
        tokens["asym_id"] = 0
        tokens["resolved_mask"] = True
        tokens["center_coords"][:, 0] = tokens["res_idx"]
        tokens["is_copy"][-2:] = True
        tokens["motif_group_id"][-2:] = [1, 2]

        rigids = np.zeros(5, dtype=Rigid)
        rigids["rigid_idx"] = np.arange(5)
        rigids["token_idx"] = np.arange(5)
        rigids["tensor7"][:, 0] = 1.0
        rigids["is_present"] = True
        rigids["rigids_noising_mask"] = True
        rigids["motif_group_id"][-2:] = [1, 2]
        return tokens, rigids, np.zeros(0, dtype=TokenBond)

    monkeypatch.setattr(dm, "load_input", lambda record, data_dir: structure)
    monkeypatch.setattr(dm, "tokenize_structure", fake_tokenize)

    prepared = dm.prepare_training_tokenized_example(
        dataset=SimpleNamespace(data_dir="unused"),
        sample=SimpleNamespace(
            record=SimpleNamespace(id="rec-a"),
            chain_id=None,
            interface_id=None,
        ),
        task=DummyMotifTask(),
        cropper=Cropper(min_neighborhood=0, max_neighborhood=0),
        max_crop_residues=5,
        remove_mol_types=[],
        mask_nonstandard=False,
        np_rng=np.random.RandomState(0),
    )

    final_tokens = prepared.tokenized.tokens
    copy_tokens = final_tokens[final_tokens["is_copy"]]
    marker_tokens = final_tokens[final_tokens["is_motif_group_aggregator"]]

    assert prepared.tokenized.structure.residues["is_present"].sum() == 3
    assert len(final_tokens) == 5
    assert copy_tokens["motif_group_id"].tolist() == [1, 2]
    assert marker_tokens["motif_group_id"].tolist() == [1, 2]
