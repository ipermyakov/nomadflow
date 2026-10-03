#!/usr/bin/env python3
"""Compare MixedMotifScaffolding old-flag masks + loss between branch and zenprotein-main.

Uses synthetic single-chain ARG structures so main's fix_tip[i] indexing matches
branch fix_tip[res_pos], and so we do not depend on dataset sampling.

Also dumps a fixed synthetic multiframe/seq loss batch with gid=0 and historical-style
fixed motif frames, then compares shared loss terms (and asserts branch aggregator≈0).

Example:
  conda activate nomadflow
  python scripts/compare_old_flag_motif_parity.py compare \\
    --main-root /path/to/proteinzen \\
    --branch-root /path/to/this/repo
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


LOSS_SHARED_KEYS = [
    "rot_vf_loss",
    "trans_vf_loss",
    "unscaled_rot_vf_loss",
    "unscaled_trans_vf_loss",
    "pred_trans_mse",
    "ref_trans_mse",
    "fafe",
    "scaled_fafe",
    "seq_loss",
    "per_seq_recov",
]

LOSS_AGGREGATOR_KEYS = [
    "aggregator_fafe",
    "scaled_aggregator_fafe",
    "aggregator_trans_loss",
    "aggregator_rot_loss",
    "aggregator_pose_loss",
]


CASES = [
    {
        "name": "yaml_indexed_mixed",
        "p_is_unindexed": 0.0,
        "p_fix_tip": 0.35,
        "p_fix_bb": 0.35,
        "p_fix_all": 0.30,
        "seed": 0,
    },
    {
        "name": "yaml_unindexed_mixed",
        "p_is_unindexed": 1.0,
        "p_fix_tip": 0.35,
        "p_fix_bb": 0.35,
        "p_fix_all": 0.30,
        "seed": 1,
    },
    {
        "name": "force_tip",
        "p_is_unindexed": 0.0,
        "p_fix_tip": 1.0,
        "p_fix_bb": 0.0,
        "p_fix_all": 0.0,
        "seed": 2,
        "force_all_motif": True,
    },
    {
        "name": "force_bb",
        "p_is_unindexed": 0.0,
        "p_fix_tip": 0.0,
        "p_fix_bb": 1.0,
        "p_fix_all": 0.0,
        "seed": 3,
        "force_all_motif": True,
    },
    {
        "name": "force_all",
        "p_is_unindexed": 0.0,
        "p_fix_tip": 0.0,
        "p_fix_bb": 0.0,
        "p_fix_all": 1.0,
        "seed": 4,
        "force_all_motif": True,
    },
]


def _make_arg_chain(num_residues: int = 16, *, spacing: float = 3.8):
    import torch

    from nomadflow.boltz.data import const
    from nomadflow.data.featurize.tokenize import convert_atom_str_to_tuple
    from nomadflow.openfold.data import residue_constants as rc

    atom_names = [name for name in rc.restype_name_to_atom14_names["ARG"] if name]
    atoms_per_residue = len(atom_names)
    total_atoms = num_residues * atoms_per_residue

    atoms = np.zeros(
        total_atoms,
        dtype=[("coords", np.float32, (3,)), ("name", "4i1"), ("is_present", bool)],
    )
    atoms["is_present"] = True

    residues = np.zeros(
        num_residues,
        dtype=[
            ("name", "<U5"),
            ("res_type", np.int8),
            ("res_idx", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("atom_center", np.int32),
            ("atom_disto", np.int32),
            ("is_standard", bool),
            ("is_present", bool),
        ],
    )
    residues["name"] = "ARG"
    residues["res_idx"] = np.arange(num_residues, dtype=np.int32)
    residues["atom_idx"] = np.arange(num_residues, dtype=np.int32) * atoms_per_residue
    residues["atom_num"] = atoms_per_residue
    residues["atom_center"] = 1
    residues["atom_disto"] = 1
    residues["is_standard"] = True
    residues["is_present"] = True

    ca_index = atom_names.index("CA")
    for res_idx in range(num_residues):
        atom_start = int(residues["atom_idx"][res_idx])
        for local_idx, atom_name in enumerate(atom_names):
            atoms["name"][atom_start + local_idx] = convert_atom_str_to_tuple(atom_name)
        atoms["coords"][atom_start + ca_index] = [spacing * res_idx, 0.0, 0.0]

    chains = np.array(
        [
            (
                "A",
                const.chain_type_ids["PROTEIN"],
                0,
                0,
                0,
                0,
                total_atoms,
                0,
                num_residues,
                0,
            )
        ],
        dtype=[
            ("name", "<U5"),
            ("mol_type", np.int8),
            ("entity_id", np.int32),
            ("sym_id", np.int32),
            ("asym_id", np.int32),
            ("atom_idx", np.int32),
            ("atom_num", np.int32),
            ("res_idx", np.int32),
            ("res_num", np.int32),
            ("cyclic_period", np.int32),
        ],
    )

    class _Struct:
        pass

    struct = _Struct()
    struct.residues = residues
    struct.chains = chains
    struct.atoms = atoms
    struct.mask = np.array([True])
    return struct


def _build_task(case: dict):
    import torch

    from nomadflow.runtime.training.motif_scaffold import MixedMotifScaffolding

    kwargs = dict(
        max_frac_res=1.0 if case.get("force_all_motif") else 0.5,
        max_num_res=16 if case.get("force_all_motif") else 20,
        max_num_segments=1,
        p_is_unindexed=case["p_is_unindexed"],
        p_fix_tip=case["p_fix_tip"],
        p_fix_bb=case["p_fix_bb"],
        p_fix_all=case["p_fix_all"],
    )
    init_params = inspect.signature(MixedMotifScaffolding.__init__).parameters
    if "multi_motif_se3" in init_params:
        kwargs["multi_motif_se3"] = False
    if "motif_t_lock" in init_params:
        kwargs["motif_t_lock"] = None

    task = MixedMotifScaffolding(**kwargs)
    if case.get("force_all_motif"):
        n_res = 16

        def _all_motif(N, np_rng=None, *args, **kwargs):
            del np_rng, args, kwargs
            return torch.ones(N, dtype=torch.bool)

        # main: generate_motif_mask only; branch: generate_motif_mask_and_groups
        if hasattr(task, "generate_motif_mask_and_groups"):
            task.generate_motif_mask_and_groups = (
                lambda N, np_rng=None, **kw: (torch.ones(N, dtype=torch.bool), None)
            )
        else:
            task.generate_motif_mask = lambda N, np_rng=None: torch.ones(N, dtype=torch.bool)
        del n_res
    return task


def _dump_case(case: dict, num_residues: int = 16) -> dict:
    import torch

    struct = _make_arg_chain(num_residues)
    task = _build_task(case)
    np_rng = np.random.RandomState(case["seed"])
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(case["seed"] + 10_000)
    out = task.sample_t_and_mask(struct, np_rng=np_rng, torch_generator=torch_generator)

    payload = {
        "atom_noising_mask": np.asarray(out["atom_noising_mask"], dtype=bool),
        "res_type_noising_mask": np.asarray(out["res_type_noising_mask"], dtype=bool),
        "copy_indexed_residue_mask": np.asarray(out["copy_indexed_residue_mask"], dtype=bool),
        "copy_unindexed_residue_mask": np.asarray(out["copy_unindexed_residue_mask"], dtype=bool),
        "copy_atomized_residue_mask": np.asarray(out["copy_atomized_residue_mask"], dtype=bool),
        "t": np.asarray(out["t"], dtype=np.float32),
    }
    if "motif_group_id_residue" in out:
        payload["motif_group_id_residue"] = np.asarray(out["motif_group_id_residue"], dtype=np.int32)
    if "motif_group_id_rigid_by_residue_frame" in out:
        payload["motif_group_id_rigid_by_residue_frame"] = np.asarray(
            out["motif_group_id_rigid_by_residue_frame"], dtype=np.int32
        )
    if "multi_motif_se3" in out:
        payload["multi_motif_se3"] = np.asarray(bool(out["multi_motif_se3"]))
    return payload


def _dump_loss_payload() -> dict:
    """Fixed synthetic old-flag loss batch: gid=0, motif frames fixed via noising mask."""
    import torch

    from nomadflow.boltz.data import const
    from nomadflow.openfold.utils import rigid_utils as ru
    from nomadflow.runtime.loss.common import seq_losses_dense_batch
    from nomadflow.runtime.loss.multiframe import multiframe_fm_loss_dense_batch

    # Frames 0-1: fixed motif (noising False). Frames 2-3: scaffold (noising True).
    # Large pred error on fixed frames must not affect loss if masking matches main.
    n_rig = 4
    rigids_1 = torch.zeros(1, n_rig, 7, dtype=torch.float32)
    rigids_1[..., 0] = 1.0
    rigids_1[0, :, 4] = torch.arange(n_rig, dtype=torch.float32) * 2.0

    rigids_t = rigids_1.clone()
    pred = rigids_1.clone()
    pred[0, 0, 4] = 100.0
    pred[0, 1, 4] = 100.0
    pred[0, 2, 4] = 5.0
    pred[0, 3, 4] = 7.0

    noising = torch.tensor([[False, False, True, True]])
    inputs = {
        "t": torch.tensor([[0.3]], dtype=torch.float32),
        "multi_motif_se3": torch.tensor([False]),
        "rigids": {
            "rigids_t": rigids_t,
            "rigids_1": rigids_1,
            "rigids_mask": torch.ones(1, n_rig, dtype=torch.bool),
            "rigids_noising_mask": noising,
            "rigids_seq_idx": torch.arange(n_rig, dtype=torch.long)[None],
            "gt_rot_vf": torch.zeros(1, n_rig, 3, dtype=torch.float32),
            # Old-flag parity: groups stay 0 (branch); main ignores these keys.
            "motif_group_id": torch.zeros(1, n_rig, dtype=torch.long),
            "rigids_is_motif_group_aggregator": torch.zeros(1, n_rig, dtype=torch.bool),
        },
    }
    try:
        from nomadflow.data.featurize.assembler import MOTIF_T_LOCK_DISABLED

        inputs["motif_t_lock"] = torch.tensor(
            float(MOTIF_T_LOCK_DISABLED), dtype=torch.float32
        )
    except Exception:
        pass

    outputs = {
        "denoised_rigids": ru.Rigid.from_tensor_7(pred),
        "pred_rot_vf": torch.zeros(1, n_rig, 3, dtype=torch.float32),
    }

    frame_kwargs = {"direct_rot_vf_loss": True}
    frame_sig = inspect.signature(multiframe_fm_loss_dense_batch)
    if "motif_aggregator_weight" in frame_sig.parameters:
        frame_kwargs["motif_aggregator_weight"] = 5.0
    frame_losses = multiframe_fm_loss_dense_batch(inputs, outputs, **frame_kwargs)

    n_tok = 4
    n_aa = len(const.tokens)
    seq = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)  # ALA/ARG/ASN/ASP-like
    logits = torch.zeros(1, n_tok, n_aa, dtype=torch.float32)
    for i, aa in enumerate(seq[0].tolist()):
        logits[0, i, aa] = 5.0
    # Wrong prediction on scaffold token 2 → nonzero seq_loss only if noising True there.
    logits[0, 2, seq[0, 2]] = 0.0
    logits[0, 2, (seq[0, 2] + 1) % n_aa] = 5.0
    # Huge wrong logit on fixed motif token 0 — must be ignored by seq_noising_mask.
    logits[0, 0, seq[0, 0]] = 0.0
    logits[0, 0, (seq[0, 0] + 1) % n_aa] = 20.0

    seq_batch = {
        "token": {
            "seq": seq,
            "token_mask": torch.ones(1, n_tok, dtype=torch.bool),
            "seq_noising_mask": torch.tensor([[False, False, True, True]]),
        }
    }
    seq_losses = seq_losses_dense_batch(
        seq_batch, {"decoded_seq_logits": logits}
    )

    payload = {}
    for key, value in frame_losses.items():
        payload[key] = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
    for key, value in seq_losses.items():
        payload[key] = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
    return payload


def cmd_dump(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "label": args.label,
        "pythonpath_cwd": os.getcwd(),
        "proteinzen_file": None,
        "cases": [],
        "loss": None,
    }
    import nomadflow

    meta["proteinzen_file"] = getattr(nomadflow, "__file__", None)

    for case in CASES:
        payload = _dump_case(case, num_residues=args.num_residues)
        path = out_dir / f"{args.label}__{case['name']}.npz"
        np.savez_compressed(path, **payload)
        summary = {
            "name": case["name"],
            "path": str(path),
            "atom_noising_false": int((~payload["atom_noising_mask"]).sum()),
            "copy_residues": int(
                payload["copy_indexed_residue_mask"].sum()
                + payload["copy_unindexed_residue_mask"].sum()
            ),
        }
        if "motif_group_id_residue" in payload:
            summary["max_motif_group_id"] = int(payload["motif_group_id_residue"].max())
        meta["cases"].append(summary)
        print(f"[dump:{args.label}] {case['name']}: wrote {path}")

    loss_payload = _dump_loss_payload()
    loss_path = out_dir / f"{args.label}__loss.npz"
    np.savez_compressed(loss_path, **loss_payload)
    meta["loss"] = {
        "path": str(loss_path),
        "keys": sorted(loss_payload.keys()),
        "values": {k: np.asarray(v).tolist() for k, v in loss_payload.items()},
    }
    print(f"[dump:{args.label}] loss: wrote {loss_path}")

    meta_path = out_dir / f"{args.label}__meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[dump:{args.label}] meta -> {meta_path}")
    return 0


def _load_npz(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _compare_payloads(case_name: str, main_p: dict, branch_p: dict) -> list[str]:
    errors: list[str] = []
    keys = [
        "atom_noising_mask",
        "res_type_noising_mask",
        "copy_indexed_residue_mask",
        "copy_unindexed_residue_mask",
        "copy_atomized_residue_mask",
    ]
    for key in keys:
        if key not in main_p or key not in branch_p:
            errors.append(f"{case_name}: missing key {key}")
            continue
        if not np.array_equal(main_p[key], branch_p[key]):
            diff = int(np.sum(main_p[key] != branch_p[key]))
            errors.append(f"{case_name}: {key} differs in {diff} entries")

    if not np.allclose(main_p["t"], branch_p["t"], atol=1e-6, rtol=0):
        errors.append(
            f"{case_name}: t differs main={main_p['t']} branch={branch_p['t']}"
        )

    if "motif_group_id_residue" in branch_p:
        if int(branch_p["motif_group_id_residue"].max()) != 0:
            errors.append(f"{case_name}: branch motif_group_id_residue not all 0")
    if "motif_group_id_rigid_by_residue_frame" in branch_p:
        if int(branch_p["motif_group_id_rigid_by_residue_frame"].max()) != 0:
            errors.append(
                f"{case_name}: branch motif_group_id_rigid_by_residue_frame not all 0"
            )
    if "multi_motif_se3" in branch_p and bool(branch_p["multi_motif_se3"]):
        errors.append(f"{case_name}: branch multi_motif_se3 is True")

    copy_mask = (
        branch_p["copy_indexed_residue_mask"] | branch_p["copy_unindexed_residue_mask"]
    )
    if case_name.startswith("force_") and not bool(copy_mask.any()):
        errors.append(f"{case_name}: expected copy residues after force_all_motif")
    if case_name == "force_all" and bool(branch_p["atom_noising_mask"].any()):
        # all motif residues fully fixed => no atom noise
        errors.append(f"{case_name}: expected all-False atom_noising_mask on branch")

    return errors


def _compare_loss_payloads(main_p: dict, branch_p: dict) -> list[str]:
    errors: list[str] = []
    for key in LOSS_SHARED_KEYS:
        if key not in main_p or key not in branch_p:
            errors.append(f"loss: missing shared key {key}")
            continue
        if not np.allclose(main_p[key], branch_p[key], atol=1e-5, rtol=1e-5):
            errors.append(
                f"loss: {key} differs main={np.asarray(main_p[key]).tolist()} "
                f"branch={np.asarray(branch_p[key]).tolist()}"
            )

    for key in LOSS_AGGREGATOR_KEYS:
        if key not in branch_p:
            errors.append(f"loss: branch missing aggregator key {key}")
            continue
        if not np.allclose(branch_p[key], 0.0, atol=1e-6):
            errors.append(
                f"loss: branch {key} expected ~0, got {np.asarray(branch_p[key]).tolist()}"
            )
    return errors


def _run_dump(repo_root: Path, script_path: Path, label: str, out_dir: Path, num_residues: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(script_path),
        "dump",
        "--label",
        label,
        "--out-dir",
        str(out_dir),
        "--num-residues",
        str(num_residues),
    ]
    print(f"[compare] dumping {label} from {repo_root}")
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def cmd_compare(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    branch_root = Path(args.branch_root).resolve()
    main_root = Path(args.main_root).resolve()

    _run_dump(branch_root, script_path, "branch", out_dir, args.num_residues)
    _run_dump(main_root, script_path, "main", out_dir, args.num_residues)

    all_errors: list[str] = []
    for case in CASES:
        name = case["name"]
        main_path = out_dir / f"main__{name}.npz"
        branch_path = out_dir / f"branch__{name}.npz"
        if not main_path.exists() or not branch_path.exists():
            all_errors.append(f"{name}: missing dump files")
            continue
        main_p = _load_npz(main_path)
        branch_p = _load_npz(branch_path)
        case_errors = _compare_payloads(name, main_p, branch_p)
        if case_errors:
            all_errors.extend(case_errors)
            print(f"[compare] FAIL {name}")
            for err in case_errors:
                print(f"  - {err}")
        else:
            print(f"[compare] OK   {name}")

    main_loss_path = out_dir / "main__loss.npz"
    branch_loss_path = out_dir / "branch__loss.npz"
    if not main_loss_path.exists() or not branch_loss_path.exists():
        all_errors.append("loss: missing dump files")
        print("[compare] FAIL loss (missing dumps)")
    else:
        loss_errors = _compare_loss_payloads(
            _load_npz(main_loss_path), _load_npz(branch_loss_path)
        )
        if loss_errors:
            all_errors.extend(loss_errors)
            print("[compare] FAIL loss")
            for err in loss_errors:
                print(f"  - {err}")
        else:
            print("[compare] OK   loss (shared terms + branch aggregator≈0)")

    report = {
        "branch_root": str(branch_root),
        "main_root": str(main_root),
        "out_dir": str(out_dir),
        "ok": len(all_errors) == 0,
        "errors": all_errors,
    }
    report_path = out_dir / "compare_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[compare] report -> {report_path}")
    if all_errors:
        print(f"[compare] FAILED with {len(all_errors)} error(s)")
        return 1
    print(
        "[compare] PASSED: old-flag MixedMotif masks + loss match main "
        "(synthetic single-chain / gid=0 batch)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dump_p = sub.add_parser("dump", help="Dump masks using the proteinzen on PYTHONPATH")
    dump_p.add_argument("--label", required=True)
    dump_p.add_argument("--out-dir", type=Path, required=True)
    dump_p.add_argument("--num-residues", type=int, default=16)

    cmp_p = sub.add_parser("compare", help="Dump branch+main and compare")
    cmp_p.add_argument(
        "--branch-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    cmp_p.add_argument(
        "--main-root",
        type=Path,
        default=Path("../proteinzen"),
    )
    cmp_p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/old_flag_motif_parity"),
    )
    cmp_p.add_argument("--num-residues", type=int, default=16)

    args = parser.parse_args(argv)
    if args.command == "dump":
        return cmd_dump(args)
    if args.command == "compare":
        return cmd_compare(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
