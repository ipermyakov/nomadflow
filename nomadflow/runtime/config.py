import os

import torch
from hydra_zen import store, make_custom_builds_fn, make_config, kwargs_of
from hydra.conf import HydraConf, RunDir
import omegaconf
from datetime import timedelta

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from nomadflow.data.datasets.datamodule import BiomoleculeDataModule, BiomoleculeSamplingDataModule

from nomadflow.stoch_interp.multiframe import MultiSE3Interpolant

from nomadflow.model.denoiser import IpaMultiRigidDenoiser


from nomadflow.runtime.lmod import BiomoleculeModule
from nomadflow.runtime.optim import get_std_opt

if os.environ.get("REPO_ROOT") is None:
    print("Didn't find REPO_ROOT as an env var, so let's try to load it from the repo")
    import pathlib
    # import pprint
    import shlex
    import subprocess
    script_dir = pathlib.Path(__file__).parent.resolve()
    script_dir = os.path.abspath(os.path.join(script_dir, "../.."))

    command = shlex.split(f"bash -c 'set -a && source {script_dir}/env_vars.sh && env'")
    proc = subprocess.Popen(command, stdout = subprocess.PIPE)
    for line in proc.stdout:
        (key, _, value) = line.partition(b"=")
        os.environ[key.decode()] = value.decode().strip()
    proc.communicate()

    # # pprint.pprint(dict(os.environ))
print("REPO_ROOT:", os.environ.get("REPO_ROOT"))

# targets ZenStore
# pylint: disable=not-callable

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
builds = make_custom_builds_fn(populate_full_signature=True)


def config_hydra_store():
    # change hydra conf defaults to avoid run collisions
    store(
        HydraConf(
            run=RunDir(dir="./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}")
    ))

    ## switches to allow for hot-swapping between paradigms and domains
    paradigm_store = store(group="paradigm")
    paradigm_store({"paradigm": "multiframefm"}, name="multiframefm")

    domain_store = store(group="domain")
    domain_store({"domain": "protein"}, name="protein")

    corruption_store = store(group="corrupter")
    corruption_store(
        MultiSE3Interpolant,
        name="multiframefm_protein")
    corruption_store(
        pbuilds(
            MultiSE3Interpolant,
            motif_t_lock=0.5,
            motif_time_schedule="linear",
        ),
        name="multiframefm_motif_t_lock_0_5",
    )

    datamodule_store = store(group="datamodule")
    datamodule_store(
        pbuilds(
            BiomoleculeDataModule,
            batch_size=1,
            num_workers=8,
            val_num_workers=0,
        ),
        name="default"
    )

    dataset_store = store(group="dataset")
    dataset_store(
        {
            "config": f"{os.environ.get('REPO_ROOT')}/configs/train/data/afdb_128.yaml",
            "val_config": None,
        },
        name="afdb_128")

    lmodule_store = store(group="lmodule")
    lmodule_store(
        pbuilds(BiomoleculeModule),
        name="protein"
    )

    # latent_fm_wrapper = ChimeraLatentWrapper
    model_store = store(group="model")
    model_store(IpaMultiRigidDenoiser, name="multiframefm_protein")

    tasks_store = store(group="tasks")
    tasks_store(
        {"config": f"{os.environ.get('REPO_ROOT')}/configs/train/tasks/train_128.yaml"},
        name="train_128"
    )

    exp_store = store(group="experiment")
    exp_store({
        "warm_start": None,
        # If true, load only model/EMA weights from warm_start (strict per
        # lmodule.strict_weight_loading) and start a fresh Lightning loop
        # (step/epoch/optimizer not restored). Needed for incomplete ckpts
        # and when max_steps is smaller than the ckpt global_step.
        "warm_start_weights_only": False,
    }, name="default")
    lightning_store = exp_store(group="experiment/lightning")
    lightning_store(
        pbuilds(
            Trainer,
            min_epochs=1,
            max_epochs=-1,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            use_distributed_sampler=False,
            # gradient_clip_val=1.0,
        ), name="default")

    optim_store = exp_store(group="experiment/optim")
    optim_store(pbuilds(torch.optim.Adam, lr=1e-4), name="adam")
    optim_store(pbuilds(get_std_opt, d_model=128), name="noam")

    exp_store(
        ModelCheckpoint,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last=True,
        train_time_interval=timedelta(days=1),
        group="experiment/checkpointer",
        name="protein")

    exp_store(
        {
            "enabled": True,
            "tracking_uri": os.path.join(os.environ.get("REPO_ROOT", "."), "mlruns"),
            "experiment_name": "nomadflow",
            "run_name": None,
            "tags": {},
            "artifact_location": None,
            "nested_run": False,
            "parent_run_id": None,
            "log_experiment_config": True,
            "log_hydra_artifacts": True,
            "log_checkpoint_artifacts": True,
        },
        group="experiment/mlflow",
        name="default"
    )

    hpo_store = store(group="hpo")
    hpo_store(
        {
            "study_name": "nomadflow_optuna_v1",
            "tracking_mode": "parent_child",
            "tracking_uri": None,
            "parent_run_name": "optuna_study_2",
            "storage": f"sqlite:///{os.path.join(os.environ.get('REPO_ROOT', '.'), 'outputs', 'optuna', 'optuna_v1.db')}",
            "direction": "minimize",
            "objective_metric": "val/loss",
            "n_trials": 10,
            "timeout": None,
            "execution_backend": "local",
            "trial_run_name_template": "trial_{trial_number:04d}",
            "trial_dir_template": "trials/trial_{trial_number:04d}",
            "trial_overrides": [],
            "search_space": {
                "experiment.optim.lr": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 5e-3,
                    "log": True,
                },
                # "datamodule.batch_size": {
                #     "type": "categorical",
                #     "choices": [4],
                # },
                "experiment.lightning.accumulate_grad_batches": {
                    "type": "categorical",
                    "choices": [1, 2, 4],
                },
                "experiment.lightning.gradient_clip_val": {
                    "type": "categorical",
                    "choices": [10.0, 20.0, 50.0],
                },
                # "model.num_blocks": {
                #     "type": "int",
                #     "low": 4,
                #     "high": 8,
                # },
            },
            "cluster": {
                "submit_script": None,
                "queue": None,
            },
        },
        name="default",
    )
    hpo_store({"name": "tpe", "seed": None}, group="hpo/sampler", name="tpe")
    hpo_store(
        {
            "name": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 1,
            "interval_steps": 1,
        },
        group="hpo/pruner",
        name="median",
    )
    hpo_store({"name": "none"}, group="hpo/pruner", name="none")

    # breaking the hydra_zen/hydra/omegaconf abstraction here a little
    # but this allows for nice swappable configs
    ExperimentConfig = make_config(
        debug=False,
        hydra_defaults=[
            {"paradigm": "multiframefm"},
            {"domain": "protein"},
            {"datamodule": "default"},
            {"dataset": "afdb_128"},
            {"lmodule": "${domain}"},
            {"corrupter": "${paradigm}_${domain}"},
            {"model": "${paradigm}_${domain}"},
            {"tasks": "train_128"},
            {"experiment": "default"},
            {"experiment/optim": "adam"},
            {"experiment/lightning": "default"},
            {"experiment/checkpointer": "${domain}"},
            {"experiment/mlflow": "default"},
            '_self_'
        ],
    )

    store(ExperimentConfig, name="main")

    HpoExperimentConfig = make_config(
        debug=False,
        hydra_defaults=[
            {"paradigm": "multiframefm"},
            {"domain": "protein"},
            {"datamodule": "default"},
            {"dataset": "afdb_128"},
            {"lmodule": "${domain}"},
            {"corrupter": "${paradigm}_${domain}"},
            {"model": "${paradigm}_${domain}"},
            {"tasks": "train_128"},
            {"experiment": "default"},
            {"experiment/optim": "adam"},
            {"experiment/lightning": "default"},
            {"experiment/checkpointer": "${domain}"},
            {"experiment/mlflow": "default"},
            {"hpo": "default"},
            {"hpo/sampler": "tpe"},
            {"hpo/pruner": "median"},
            "_self_",
        ],
    )
    store(HpoExperimentConfig, name="hpo_main")
    store.add_to_hydra_store()


def remove_zen_keys(cfg, keys=['_target_', '_partial_']):
    def remove_key(cfg):
        for key in keys:
            if hasattr(cfg, key):
                delattr(cfg, key)
            elif isinstance(cfg, dict) and key in cfg:
                del cfg[key]

        if isinstance(cfg, (omegaconf.DictConfig, dict)):
            for value in cfg.values():
                remove_key(value)
        elif isinstance(cfg, omegaconf.ListConfig):
            for value in cfg:
                remove_key(value)
        return cfg

    return remove_key(cfg.copy())


def config_sampling_hydra_store():
    # change hydra conf defaults to avoid run collisions
    store(
        HydraConf(
            run=RunDir(dir="./sampling_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}")
    ))

    sampler_store = store(group="sampler")
    sampler_store(
        builds(
            BiomoleculeSamplingDataModule
        ),
        name="default"
    )
    corrupter_store = store(group="corrupter")
    corrupter_store(
        MultiSE3Interpolant,
        name="default",
    )
    corrupter_store(
        builds(
            MultiSE3Interpolant,
            motif_t_lock=0.5,
            motif_time_schedule="linear",
        ),
        name="motif_t_lock_0_5",
    )

    SamplingConfig = make_config(
        model_dir="",
        out_dir="samples",
        save_traj=False,
        center_clean_traj_com=False,
        center_prot_traj_com=False,
        output_motif_chains=False,
        write_eval_pdb=True,
        export_layout_manifest=False,
        # If set, load this .ckpt directly. Architecture is read from the nearest
        # `.hydra/config.yaml` found by walking up from the checkpoint path (same
        # training run). If none is found, set model_dir to that run directory.
        # Without checkpoint_path, use checkpoint_idx under model_dir's
        # lightning_logs/version_0/checkpoints/.
        checkpoint_path="",
        checkpoint_idx=-1,
        hydra_defaults=[
            {"sampler": "default"},
            {"corrupter": "default"},
            '_self_'
        ],
    )

    store(SamplingConfig, name="main")
    store.add_to_hydra_store()


def config_train_viz_hydra_store():
    store(
        HydraConf(
            run=RunDir(dir="./train_viz_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}")
    ))

    corrupter_store = store(group="corrupter")
    corrupter_store(
        MultiSE3Interpolant,
        name="default",
    )

    TrainVizConfig = make_config(
        dataset_config=f"{os.environ.get('REPO_ROOT')}/configs/train/data/afdb_512_clusters_crop256.yaml",
        out_dir="train_corruption_viz",
        samples_per_task=2,
        base_seed=0,
        num_frames=12,
        task_filter=[],
        output_motif_chains=True,
        add_intra_rigid_conects=True,
        hydra_defaults=[
            {"corrupter": "default"},
            "_self_",
        ],
    )

    store(TrainVizConfig, name="train_viz")
    store.add_to_hydra_store()