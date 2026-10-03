from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from nomadflow.data.datasets import datamodule as dm
from nomadflow.runtime.lmod import BiomoleculeModule
from nomadflow.runtime.training.partial_seq import PartialSequenceConditioned


class DummyTask:
    def __init__(self, name):
        self.name = name

    def max_added_tokens(self, _):
        return 0


class DummyTaskSampler:
    def __init__(self, task_list):
        self.task_list = task_list


class CyclingSampler:
    def sample(self, records, random):
        del random
        while True:
            for record in records:
                yield SimpleNamespace(record=record, chain_id=None, interface_id=None)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]


class DummyCorrupter:
    def corrupt_dense_batch(self, batch, rng_seed=None):
        batch = dict(batch)
        batch["rng_seed"] = rng_seed
        return batch


class TinyModel(torch.nn.Module):
    self_conditioning = False

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)


@dataclass(frozen=True)
class DummyStructure:
    atoms: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray


def _make_fake_dataset(task_list, record_ids):
    manifest = [SimpleNamespace(id=record_id) for record_id in record_ids]
    return SimpleNamespace(
        task_sampler=DummyTaskSampler(task_list),
        data_sampler=CyclingSampler(),
        manifest=manifest,
        data_dir="unused",
    )


def test_validation_dataset_same_idx_is_deterministic_and_has_val_seed(monkeypatch):
    def fake_prepare_training_tokenized_example(**kwargs):
        np_rng = kwargs["np_rng"]
        torch_generator = kwargs["torch_generator"]
        metadata = kwargs["metadata"]
        return dm.PreparedTrainingExample(
            tokenized=None,
            task_data={},
            task=kwargs["task"],
            sample=kwargs["sample"],
            metadata={
                "record_id": kwargs["sample"].record.id,
                "task_name": kwargs["task"].name,
                "np_draw": int(np_rng.randint(0, 10_000)),
                "torch_draw": int(torch.randint(0, 10_000, (1,), generator=torch_generator).item()),
                "val_seed": metadata["val_seed"],
            },
        )

    monkeypatch.setattr(dm, "prepare_training_tokenized_example", fake_prepare_training_tokenized_example)
    monkeypatch.setattr(
        dm,
        "featurize_prepared_training_example",
        lambda prepared_example, **kwargs: prepared_example.metadata,
    )

    dataset = dm.ValidationDataset(
        datasets=[_make_fake_dataset([DummyTask("shared")], ["rec-a"])],
        max_crop_residues=16,
        max_crop_rigids=16,
        samples_per_task=2,
        use_cropper=False,
        base_seed=1234,
    )

    first = dataset[0]
    second = dataset[0]

    assert first["record_id"] == second["record_id"] == "rec-a"
    assert first["task_name"] == second["task_name"] == "shared"
    assert first["np_draw"] == second["np_draw"]
    assert first["torch_draw"] == second["torch_draw"]
    assert torch.equal(first["val_seed"], second["val_seed"])
    assert first["val_seed"].item() == dataset.items[0].seed


def test_validation_dataset_builds_task_balanced_items():
    shared_task = DummyTask("shared")
    unique_task = DummyTask("unique")
    dataset = dm.ValidationDataset(
        datasets=[
            _make_fake_dataset([shared_task], ["d0-rec"]),
            _make_fake_dataset([shared_task, unique_task], ["d1-rec"]),
        ],
        max_crop_residues=16,
        max_crop_rigids=16,
        samples_per_task=5,
        use_cropper=False,
        dataset_probs=[0.8, 0.2],
        base_seed=7,
    )

    counts = Counter((item.dataset_idx, item.task_idx) for item in dataset.items)

    assert counts[(0, 0)] == 4
    assert counts[(1, 0)] == 1
    assert counts[(1, 1)] == 5
    assert len(dataset) == 10


def test_build_validation_dataset_from_training_dataset_preserves_sampling_settings():
    training_dataset = dm.TrainingDataset(
        datasets=[_make_fake_dataset([DummyTask("shared")], ["rec-a"])],
        max_crop_residues=32,
        max_crop_rigids=64,
        use_cropper=True,
        crop_min_neighbors=4,
        crop_max_neighbors=10,
        dataset_probs=[1.0],
        remove_mol_types=["DNA"],
        mask_nonstandard=True,
    )

    validation_dataset = dm.build_validation_dataset_from_training_dataset(
        training_dataset,
        samples_per_task=3,
        base_seed=17,
    )

    assert validation_dataset.datasets is training_dataset.datasets
    assert validation_dataset.max_crop_residues == 32
    assert validation_dataset.max_crop_rigids == 64
    assert validation_dataset.samples_per_task == 3
    assert validation_dataset.base_seed == 17
    assert validation_dataset.dataset_probs == [1.0]
    assert validation_dataset.mask_nonstandard is True
    assert validation_dataset.remove_mol_types == [dm.const.chain_types.index("DNA")]
    assert validation_dataset.cropper.neighborhood_sizes == list(range(4, 11, 2))


def test_prepare_training_features_passes_rng_into_cropper(monkeypatch):
    residues = np.zeros(1, dtype=[("dummy", np.int32)])
    atoms = np.zeros(1, dtype=[("dummy", np.int32)])
    chains = np.zeros(1, dtype=[("mol_type", np.int32)])
    interfaces = np.zeros(0, dtype=[("chain_1", np.int32), ("chain_2", np.int32)])
    struct = DummyStructure(
        atoms=atoms,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=np.array([True]),
    )
    tokens = np.zeros(
        2,
        dtype=[
            ("token_idx", np.int32),
            ("resolved_mask", bool),
            ("asym_id", np.int32),
            ("res_idx", np.int32),
            ("rigid_num", np.int32),
            ("center_coords", np.float32, (3,)),
            ("is_motif_group_aggregator", bool),
        ],
    )
    tokens["token_idx"] = [0, 1]
    tokens["resolved_mask"] = True
    tokens["asym_id"] = 0
    tokens["res_idx"] = [0, 1]
    tokens["rigid_num"] = 1
    tokens["is_motif_group_aggregator"] = False
    bonds = np.zeros(0, dtype=[("token_1", np.int32), ("token_2", np.int32)])
    rigids = np.zeros(
        2,
        dtype=[
            ("rigids_noising_mask", bool),
            ("is_present", bool),
            ("tensor7", np.float32, (7,)),
            ("token_idx", np.int32),
            ("motif_group_id", np.int32),
            ("is_motif_group_aggregator", bool),
        ],
    )
    rigids["rigids_noising_mask"] = True
    rigids["is_present"] = True
    rigids["token_idx"] = [0, 1]
    rigids["motif_group_id"] = 0
    rigids["is_motif_group_aggregator"] = False

    class SpyTask(DummyTask):
        def __init__(self):
            super().__init__("shared")
            self.seen_np_rng = None

        def sample_t_and_mask(self, data, np_rng=None, torch_generator=None):
            del data, torch_generator
            self.seen_np_rng = np_rng
            return {"dummy": np.array([1.0], dtype=np.float32)}

    class SpyCropper:
        def __init__(self):
            self.random_arg = None

        def crop(self, tokenized_data, max_tokens, random, chain_id=None, interface_id=None):
            del max_tokens, chain_id, interface_id
            self.random_arg = random
            return tokenized_data

    spy_task = SpyTask()
    spy_cropper = SpyCropper()

    monkeypatch.setattr(dm, "load_input", lambda record, data_dir: struct)
    monkeypatch.setattr(
        dm,
        "tokenize_structure",
        lambda structure, task_data, np_rng=None, shuffle_chains=False, shuffle_copied_fragments=True: (
            tokens,
            rigids,
            bonds,
        ),
    )
    monkeypatch.setattr(dm, "featurize_training", lambda *args, **kwargs: {"ok": True})

    rng = np.random.RandomState(99)
    features = dm._prepare_training_features(
        dataset=SimpleNamespace(data_dir="unused"),
        sample=SimpleNamespace(record=SimpleNamespace(id="rec-a"), chain_id=None, interface_id=None),
        task=spy_task,
        cropper=spy_cropper,
        max_crop_residues=1,
        max_crop_rigids=4,
        remove_mol_types=[],
        mask_nonstandard=False,
        np_rng=rng,
        torch_generator=torch.Generator().manual_seed(99),
        metadata={"val_seed": torch.tensor(99)},
    )

    assert features["ok"] is True
    assert torch.equal(features["val_seed"], torch.tensor(99))
    assert spy_task.seen_np_rng is rng
    assert spy_cropper.random_arg is rng


def test_biomolecule_datamodule_exposes_val_dataloader(monkeypatch):
    monkeypatch.setattr(dm, "collate", lambda batch: {"items": list(batch)})

    dataset = DummyDataset([1, 2, 3])
    datamodule = dm.BiomoleculeDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=2,
        num_workers=0,
        val_num_workers=0,
    )

    loader = datamodule.val_dataloader()

    assert loader is not None
    assert next(iter(loader)) == {"items": [1, 2]}


def test_biomolecule_datamodule_allows_missing_val_dataset():
    dataset = DummyDataset([1, 2, 3])
    datamodule = dm.BiomoleculeDataModule(
        train_dataset=dataset,
        val_dataset=None,
        batch_size=2,
        num_workers=0,
        val_num_workers=0,
    )

    assert datamodule.val_dataloader() is None


def test_validation_step_logs_val_loss_and_uses_validation_seed(monkeypatch):
    module = BiomoleculeModule(
        model=TinyModel(),
        corrupter=DummyCorrupter(),
        optim=object(),
        use_ema=False,
        use_posthoc_ema=False,
    )

    expected_seed = module._validation_seed_from_batch({"val_seed": torch.tensor([11, 22])})
    seen = {"rng_seed": None, "log_dict": [], "log": []}

    def fake_prepare_noisy_batch(batch, rng_seed=None):
        seen["rng_seed"] = rng_seed
        return batch

    def fake_forward(batch, enable_self_conditioning=True):
        del batch, enable_self_conditioning
        return {"dummy": torch.tensor(1.0)}

    def fake_loss(batch, outputs):
        del batch, outputs
        return {
            "loss": torch.tensor(2.5),
            "loss_per_batch": torch.tensor([2.0, 3.0]),
            "seq_loss": torch.tensor([0.4, 0.6]),
            "frame_vf_loss": torch.tensor([1.6, 2.4]),
        }

    monkeypatch.setattr(module, "_prepare_noisy_batch", fake_prepare_noisy_batch)
    monkeypatch.setattr(module, "_forward_with_self_conditioning", fake_forward)
    monkeypatch.setattr(module, "_loss_step", fake_loss)
    monkeypatch.setattr(module, "log_dict", lambda values, **kwargs: seen["log_dict"].append((values, kwargs)))
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: seen["log"].append((name, value, kwargs)))

    batch = {
        "t": torch.tensor([0.25, 0.75]),
        "task": [SimpleNamespace(name="shared"), SimpleNamespace(name="shared")],
        "val_seed": torch.tensor([11, 22]),
    }

    loss = module.validation_step(batch, batch_idx=0)

    assert torch.isclose(loss, torch.tensor(2.5))
    assert seen["rng_seed"] == expected_seed
    assert any("val/loss" in values for values, _ in seen["log_dict"])
    assert any(torch.isclose(values["val/loss"], torch.tensor(2.5)) for values, _ in seen["log_dict"])
    assert any(name == "val_task/shared_loss" for name, _, _ in seen["log"])


def test_partial_seq_validation_rng_path_is_deterministic():
    data = SimpleNamespace(
        residues=np.zeros(8, dtype=[("dummy", np.int32)]),
        atoms=np.zeros(16, dtype=[("dummy", np.int32)]),
    )
    task = PartialSequenceConditioned(prob=1.0, t_sched="uniform")

    out_a = task.sample_t_and_mask(
        data,
        np_rng=np.random.RandomState(123),
        torch_generator=torch.Generator().manual_seed(123),
    )
    out_b = task.sample_t_and_mask(
        data,
        np_rng=np.random.RandomState(123),
        torch_generator=torch.Generator().manual_seed(123),
    )

    for key in out_a:
        np.testing.assert_array_equal(out_a[key], out_b[key])
