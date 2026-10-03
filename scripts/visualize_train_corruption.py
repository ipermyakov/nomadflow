import json
import logging
import os
import re
from pathlib import Path

from hydra_zen import zen
import omegaconf
import torch

from nomadflow.data.datasets.datamodule import (
    build_validation_dataset_from_training_dataset,
    featurize_prepared_training_example,
)
from nomadflow.data.featurize.assembler import collate
from nomadflow.data.write.trajectory import (
    adapt_tokenized_for_train_viz_copy_semantics,
    tokenized_with_updated_tensor7,
    write_multimodel_pdb,
    write_tokenized_structure_pdb,
)
from nomadflow.runtime.config import config_train_viz_hydra_store
from nomadflow.runtime.dataset_config import instantiate_dataset_from_config


log = logging.getLogger(__name__)


def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def _normalize_task_filter(task_filter) -> set[str]:
    if task_filter is None:
        return set()
    if isinstance(task_filter, str):
        return {task_filter}
    return {str(task_name) for task_name in task_filter}


def _select_validation_indices(dataset, task_filter: set[str]) -> list[int]:
    selected = []
    for idx, item in enumerate(dataset.items):
        task = dataset.datasets[item.dataset_idx].task_sampler.task_list[item.task_idx]
        if task_filter and task.name not in task_filter:
            continue
        selected.append(idx)
    return selected


def _frame_tokenized_examples(base_tokenized, endpoints, corrupter, num_frames: int):
    if num_frames < 2:
        raise ValueError(f"num_frames must be at least 2, got {num_frames}")

    num_rigids = len(base_tokenized.rigids)
    frame_t_grid = torch.linspace(0.0, 1.0, steps=num_frames, dtype=torch.float32)
    frame_examples = []
    for frame_idx, t_value in enumerate(frame_t_grid):
        if frame_idx == (num_frames - 1):
            tensor7 = endpoints["rigids_1"][0, :num_rigids]
        else:
            interpolated = corrupter.interpolate_dense_corruption(
                endpoints,
                torch.tensor([[float(t_value)]], dtype=torch.float32),
            )
            tensor7 = interpolated["rigids_t"][0, :num_rigids]
        frame_examples.append(tokenized_with_updated_tensor7(base_tokenized, tensor7))
    return frame_examples, frame_t_grid.tolist()


def main(corrupter, zen_cfg):
    log.info("Train-time corruption viz started in folder: %s", os.getcwd())

    zen_cfg = omegaconf.OmegaConf.to_container(zen_cfg, resolve=True)
    assert zen_cfg is not None

    dataset_config = zen_cfg["dataset_config"]
    train_dataset = instantiate_dataset_from_config(dataset_config)
    validation_dataset = build_validation_dataset_from_training_dataset(
        train_dataset,
        samples_per_task=int(zen_cfg["samples_per_task"]),
        base_seed=int(zen_cfg["base_seed"]),
    )

    task_filter = _normalize_task_filter(zen_cfg.get("task_filter"))
    selected_indices = _select_validation_indices(validation_dataset, task_filter)
    if not selected_indices:
        raise ValueError(
            f"No validation items matched task_filter={sorted(task_filter)} for dataset_config={dataset_config}"
        )

    out_dir = Path(zen_cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for dataset_idx in selected_indices:
        prepared_example = validation_dataset.prepare_item(dataset_idx)
        task = prepared_example.task
        val_seed = int(prepared_example.metadata["val_seed"].item())
        features = featurize_prepared_training_example(
            prepared_example,
            max_crop_residues=validation_dataset.max_crop_residues,
            max_crop_rigids=validation_dataset.max_crop_rigids,
        )
        batch = collate([features])
        endpoints = corrupter.sample_dense_corruption_endpoints(batch, rng_seed=val_seed)
        viz_tokenized = adapt_tokenized_for_train_viz_copy_semantics(prepared_example.tokenized)
        frame_examples, t_grid = _frame_tokenized_examples(
            viz_tokenized,
            endpoints,
            corrupter,
            num_frames=int(zen_cfg["num_frames"]),
        )

        sample = prepared_example.sample
        record_id = sample.record.id
        sample_id = _sanitize_path_component(f"{record_id}_seed{val_seed}_idx{dataset_idx}")
        task_dir = out_dir / _sanitize_path_component(task.name)
        task_dir.mkdir(parents=True, exist_ok=True)

        clean_example = tokenized_with_updated_tensor7(
            viz_tokenized,
            endpoints["rigids_1"][0, : len(viz_tokenized.rigids)],
        )
        clean_path = task_dir / f"{sample_id}.pdb"
        traj_path = task_dir / f"{sample_id}_train_corruption_traj.pdb"
        metadata_path = task_dir / f"{sample_id}.json"

        write_tokenized_structure_pdb(
            clean_path,
            clean_example,
            output_motif_chains=bool(zen_cfg["output_motif_chains"]),
            take_copy_structure=False,
        )
        _, total_conects, model_count = write_multimodel_pdb(
            traj_path,
            frame_examples,
            output_motif_chains=bool(zen_cfg["output_motif_chains"]),
            take_copy_structure=False,
            add_intra_rigid_conects=bool(zen_cfg["add_intra_rigid_conects"]),
        )

        metadata = {
            "dataset_config": dataset_config,
            "record_id": record_id,
            "task_name": task.name,
            "sample_index": dataset_idx,
            "val_seed": val_seed,
            "chain_id": sample.chain_id,
            "interface_id": sample.interface_id,
            "num_tokens": int(len(prepared_example.tokenized.tokens)),
            "num_rigids": int(len(prepared_example.tokenized.rigids)),
            "num_copy_tokens": int(features["token"]["token_is_copy_mask"].sum().item()),
            "num_noised_rigids": int(features["rigids"]["rigids_noising_mask"].sum().item()),
            "num_visible_rigids": int(features["rigids"]["rigids_mask"].sum().item()),
            "t_grid": t_grid,
            "output_motif_chains": bool(zen_cfg["output_motif_chains"]),
            "add_intra_rigid_conects": bool(zen_cfg["add_intra_rigid_conects"]),
            "clean_pdb": str(clean_path),
            "trajectory_pdb": str(traj_path),
            "trajectory_models": int(model_count),
            "trajectory_conects": int(total_conects),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest.append(metadata)
        log.info("Wrote train corruption trajectory for %s (%s) to %s", record_id, task.name, traj_path)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    config_train_viz_hydra_store()
    zen(main, unpack_kwargs=True).hydra_main(
        config_name="train_viz",
        version_base="1.2",
        config_path=".",
    )
