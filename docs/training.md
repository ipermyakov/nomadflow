# Training

## Dataset preprocessing
See `scripts/data/README.md` for instructions on how to preprocess data.

## Training
To train, run
```bash
python train.py \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=<batch_size_per_gpu> \
    datamodule.num_workers=<num_cpu_workers> \
    dataset.config=<path_to_dataset_config> \
    experiment.lightning.devices=<num_gpus> \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \
```
Additional flags can be found in `nomadflow/runtime/config.py` within `config_hydra_store()`.
Example training scripts can be found in `scripts/submit_scripts/train/` and example dataset configs can be found in `configs/train/data/`.

Hyperparameter search: `python scripts/train_optuna.py` (see also `scripts/run_optuna_example.sh`).

## Experiment tracking (MLflow)

`train.py` logs to `${REPO_ROOT}/mlruns` by default (override with `experiment.mlflow.tracking_uri=...`). To inspect runs locally:

```bash
mlflow ui --backend-store-uri "${REPO_ROOT}/mlruns"
```

## Sampling from an MLflow run

After a training run logged to MLflow (with artifacts enabled), sample from that `run_id` via
`scripts/sample_from_mlflow_run.py` without manually locating the checkpoint or Hydra config.
Article multi-motif example:

```bash
python scripts/sample_from_mlflow_run.py \
    --run-id <mlflow_run_id> \
    --tracking-uri <mlflow_tracking_uri> \
    --tasks-yaml ${REPO_ROOT}/configs/sampling/protpardelle_motif_scaffolding/config_multi_motif_se3_catalytic_tlock_0.5.yaml \
    --out-dir <output_folder> \
    --batch-size 1 \
    --num-samples-per-task 16 \
    --ode-sampling \
    --output-motif-chains
```

The launcher downloads:
- `metadata/run_metadata.json`
- `hydra/config.yaml`
- the selected artifact under `checkpoints/`

It stages them into a local cache directory, writes provenance metadata under
`<output_folder>/metadata/mlflow_sampling_source.json`, and then calls `sample.py` with the
resolved `checkpoint_path` and staged run directory.

`--num-samples-per-task` rewrites `num_samples` in every task entry before sampling, so you can
shrink large task YAMLs without editing them by hand.
`--ode-sampling` expands to:
- `corrupter.sampling_noise_mode=null`
- `corrupter.trans_step_size=1`
- `corrupter.rot_step_size=10`
- `corrupter.num_timesteps=200`
