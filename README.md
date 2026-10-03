NomadFlow
==============================

NomadFlow generates an all-atom protein scaffold around several functional motifs at once. Each motif group is its own rigid body, so the model samples the scaffold together with the relative placement of the groups.

Preprint: link forthcoming.

The code extends [ProteinZen](https://github.com/alexjli/proteinzen), the all-atom SE(3) flow-matching model from ["All-atom protein generation via SE(3) flow matching with ProteinZen"](https://www.biorxiv.org/content/10.1101/2025.10.18.683228v1).

## Installation
Create a fresh conda environment and run the installer. It installs PyTorch 2.6 (CUDA 12.6) from the PyTorch wheel index, the matching PyG extensions from `data.pyg.org`, and the rest of the dependencies, then runs `pip install -e .`.

```bash
conda create -n nomadflow python=3.10
conda activate nomadflow
bash install_pt26.sh
```


## Pretrained weights

Published weights are not included in this repository. Download them from [10.5281/zenodo.22937971](https://doi.org/10.5281/zenodo.22937971) and place the checkpoint at `weights/0c62dc8a/nomadflow_0c62dc8a.ckpt`. The architecture config sits next to it at `weights/0c62dc8a/.hydra/config.yaml`.

## Training

Training (dataset preprocessing, `train.py`, Optuna, MLflow, and sampling from logged MLflow runs) is documented in [docs/training.md](docs/training.md).

## Sampling

Example below is the 1QJG catalytic site with +10 residues (`1QJG_ext10` from the article catalytic benchmark).

```bash
python sample.py \
    model_dir=weights/0c62dc8a \
    checkpoint_path=weights/0c62dc8a/nomadflow_0c62dc8a.ckpt \
    out_dir=outputs/1qjg_catalytic_plus10 \
    sampler.tasks_yaml=configs/sampling/examples/1qjg_catalytic_plus10.yaml \
    sampler.batch_size=1 \
    output_motif_chains=true
```

where
- `output_motif_chains=<bool, default=False>` which specifies whether motif copies are written into `samples/` as extra chains. This example sets it to `true`. Aggregator tokens are always written into `samples/`, each on its own chain.
- `write_eval_pdb=<bool, default=True>` which specifies whether to also write `samples_eval/` with the scaffold chain only, without motif copies or aggregators.
- `save_traj=<bool, default=False>` which specifies whether to write a multi-model PDB of every sampler step into `traj/`. Each sample gets two files: `*_clean_traj.pdb` is the denoised structure predicted at that step, and `*_prot_traj.pdb` is the structure after the integration step. This is off in the example above; writing both files slows inference.
- `center_clean_traj_com=<bool, default=False>` which specifies whether each frame of `*_clean_traj.pdb` is translated so its center of mass is at the origin. The final structure in `samples/` is unchanged.
- `center_prot_traj_com=<bool, default=False>` which specifies the same centering for each frame of `*_prot_traj.pdb`.

The task YAML is one scaffolding task with two motif groups:

```yaml
tasks:
- task: motif_scaffolding
  pdb: configs/sampling/protpardelle_motif_scaffolding/pdbs/1QJG.pdb
  pdb_contigs: A38,A14,A99,A22-31   # A38 is one residue, A22-31 a range
  motif_groups: [["A38", "A14", "A99"], ["A22-31"]]  # one SE(3) frame per group
  num_samples: 1                     # article benchmark uses 200
  total_length: 63-113              # inclusive length of the generated chain
  name: 1QJG_ext10
  multi_motif_se3: true             # each group is its own rigid body; default is one body
  multi_motif_se3_trans_std: 12     # translation scale between groups; default is 16
  motif_t_lock: 0.5                 # motif frames stay fixed until this fraction of the trajectory
```

Sampling uses the stochastic sampler (`churn`, 400 steps). By default it uses all visible GPUs; set `CUDA_VISIBLE_DEVICES` to restrict them. To record the trajectory for the example above, add `save_traj=true`. Add `center_clean_traj_com=true` and `center_prot_traj_com=true` when the frames should be centered.

