# p0/p1 Centering and Motif Group Frames

This note documents where `p0`/`p1` are centered now, and how motif-group centers are handled in training and sampling code paths.

## Quick definitions

- `p1`: target/clean endpoint (`trans_1`, `rotmats_1`).
- `p0`: noisy endpoint (`trans_0`, `rotmats_0`).
- `center`: per-sample translation used to recenter coordinates before corruption.
- `motif_group_center`: per-rigid tensor storing each motif group's center in the centered `p1` frame.

## Centering map (table)

| Stage / path | Where in code | How `p1` is centered | How `p0` is sampled/centered | Motif group center / motion | Notes |
|---|---|---|---|---|---|
| Featurization before runtime corruption | `nomadflow/data/featurize/sampling.py::sample_noise_from_struct_template` | Existing rigid coords are recentered by subtracting COM of motif-group rigids (`motif_group_id > 0`), fallback to fixed rigids (`~rigids_noising_mask`). | Per-rigid noise is sampled during tokenization (for generated parts), then same global recentering is applied to stored `tensor7` for selected center mask. | `motif_group_id` is assigned per copied motif residue. Optional `apply_multi_motif_se3_augmentation` applies independent SE(3) per group. | Produces centered template-like `rigids_1`-style geometry for downstream pipeline. |
| Dense training corruption endpoints | `nomadflow/stoch_interp/multiframe.py::sample_dense_corruption_endpoints` | `center` is computed from motif-group rigids when present and `center_on_motif=True`, else from non-noised rigids, else global COM. Then `trans_1 -= center`. | `trans_0 ~ N(0, trans_prior_std)` per rigid, then zero-mean per sample (`-mean`). Optional `prealign_noise` rotates `p0` to match selected rigids. Optional stochastic shift adds random global translation. | `_sample_motif_group_motion_endpoints` stores `motif_group_center` from centered `trans_1`. For each group, `p0` group center is sampled as absolute random point (`N(0, trans_std)`), then `group_trans_0 = center_0 - center_1`. | Current change: `prealign` excludes `motif_group_id > 0` rigids from align mask. |
| Dense interpolation constraints | `nomadflow/stoch_interp/multiframe.py::_apply_dense_endpoint_constraints` and `_apply_motif_group_motion` | Keeps `t=1` at centered `p1`. | Keeps `t=0` at masked `p0` (`_trans_diffuse_mask` / `_rots_diffuse_mask`). | Group trajectory is built around `motif_group_center`: rotate/translate centered group frame and interpolate to `p1`. | Ensures endpoint consistency and motif-group transform continuity. |
| Legacy (non-dense) training corruption path | `nomadflow/stoch_interp/multiframe.py::corrupt_batch` | Centering uses non-noised rigids (`~rigids_noising_mask`) when `center_on_motif=True`; else global COM. (No explicit `motif_group_id` branch here.) | `trans_0` from `_sample_trans_0` (`_centered_gaussian * trans_prior_std`). Optional prealign and stochastic shift. | No dense motif-group endpoint tensors are created in this branch. | Mostly relevant to older residue-shaped path. |
| Sampling-time noise injection (churn/euler helpers) | `nomadflow/stoch_interp/multiframe.py` (`motif_group_churn`, `motif_group_euler_step`) | Operates in already-centered runtime frame. | Applies incremental noise/updates on current `trans_t`/`rotmats_t`. | Uses per-group current center (`mean`) and aggregator rigid to keep grouped motion coherent during sampling updates. | This is step-wise sampling behavior, not initial `p0/p1` endpoint construction. |

## Current practical interpretation

- Training dense path uses motif-aware centering for `p1` and centered Gaussian prior for `p0`.
- Motif groups now have independent random `p0` group centers (absolute random points), represented as deltas from `p1` group centers.
- `prealign_noise` still rotates noisy `p0`, but now only on non-motif noised rigids (`motif_group_id <= 0`).
