from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from Bio.PDB import PDBIO, Atom, Chain, Model, Residue, Structure

from nomadflow.benchmark.layout_pose_metrics import (
    LAYOUT_FRAME,
    apply_rigid,
    backbone_coords,
    bootstrap_median_pairwise,
    compute_pmpnn_layout_pair_metrics,
    compute_run_layout_metrics,
    compute_sample_layout_pair_metrics,
    frame_from_wxyz_trans,
    group_frame,
    group_residue_pairs,
    invert_compose,
    kabsch_backbone_frame,
    layout_pair_distance,
    pairwise_layout_distance_matrices,
    pairwise_layout_stats,
    relative_pose_delta,
)
from nomadflow.data.featurize.tokenize import compute_frame


def _rotation_about_z(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _write_nca_c_pdb_atoms(
    path,
    residues: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    structure = Structure.Structure("test")
    model = Model.Model(0)
    structure.add(model)
    chains: dict[str, Chain.Chain] = {}
    serial = 1
    for (chain_id, res_idx), (n_coord, ca_coord, c_coord) in residues.items():
        if chain_id not in chains:
            chain = Chain.Chain(chain_id)
            model.add(chain)
            chains[chain_id] = chain
        residue = Residue.Residue((" ", res_idx, " "), "ALA", "")
        for name, coord, element in (
            ("N", np.asarray(n_coord, dtype=np.float64), "N"),
            ("CA", np.asarray(ca_coord, dtype=np.float64), "C"),
            ("C", np.asarray(c_coord, dtype=np.float64), "C"),
        ):
            residue.add(
                Atom.Atom(name, coord, 1.0, 1.0, " ", f"{name:>4s}", serial, element)
            )
            serial += 1
        chains[chain_id].add(residue)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def _ca_to_nca_c(
    coords: dict[tuple[str, int], np.ndarray],
    *,
    n_offset=np.array([-0.824, 0.660, 0.0]),
    c_offset=np.array([1.526, 0.0, 0.0]),
    rot_by_res: dict[tuple[str, int], np.ndarray] | None = None,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    n_offset = np.asarray(n_offset, dtype=np.float64)
    c_offset = np.asarray(c_offset, dtype=np.float64)
    for key, ca in coords.items():
        ca = np.asarray(ca, dtype=np.float64)
        rot = np.eye(3) if rot_by_res is None else rot_by_res.get(key, np.eye(3))
        out[key] = (ca + rot @ n_offset, ca, ca + rot @ c_offset)
    return out


def _write_nca_c_pdb(
    path,
    residues: dict[tuple[str, int], np.ndarray],
    *,
    n_offset=np.array([-0.824, 0.660, 0.0]),
    c_offset=np.array([1.526, 0.0, 0.0]),
) -> None:
    _write_nca_c_pdb_atoms(
        path, _ca_to_nca_c(residues, n_offset=n_offset, c_offset=c_offset)
    )


def _two_group_keys(*, extra_group: bool = False, extra_size: int = 8) -> list[dict]:
    keys = [
        {
            "native_chain": "A",
            "native_res_idx": 10,
            "output_chain": "A",
            "output_res_idx": 10,
            "group_id": 1,
        },
        {
            "native_chain": "A",
            "native_res_idx": 11,
            "output_chain": "A",
            "output_res_idx": 11,
            "group_id": 1,
        },
        {
            "native_chain": "A",
            "native_res_idx": 40,
            "output_chain": "A",
            "output_res_idx": 40,
            "group_id": 2,
        },
        {
            "native_chain": "A",
            "native_res_idx": 41,
            "output_chain": "A",
            "output_res_idx": 41,
            "group_id": 2,
        },
    ]
    if extra_group:
        for idx in range(80, 80 + extra_size):
            keys.append(
                {
                    "native_chain": "A",
                    "native_res_idx": idx,
                    "output_chain": "A",
                    "output_res_idx": idx,
                    "group_id": 3,
                }
            )
    return keys


def _native_coords(*, extra_group: bool = False, extra_size: int = 8) -> dict:
    coords = {
        ("A", 10): np.array([0.0, 0.0, 0.0]),
        ("A", 11): np.array([3.8, 0.0, 0.0]),
        ("A", 40): np.array([20.0, 0.0, 0.0]),
        ("A", 41): np.array([23.8, 0.0, 0.0]),
    }
    if extra_group:
        for i in range(extra_size):
            coords[("A", 80 + i)] = np.array([50.0 + 3.8 * i, 5.0, 0.0])
    return coords


def test_identity_relative_pose_is_zero(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    folded = tmp_path / "folded.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(design, coords)
    _write_nca_c_pdb(folded, coords)
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "task1",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": _two_group_keys(),
        },
        folded_path=folded,
    )
    assert len(rows) == 1
    assert rows[0]["excursion_trans_A"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["excursion_rot_deg"] == pytest.approx(0.0, abs=1e-5)
    assert rows[0]["retention_trans_A"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["retention_rot_deg"] == pytest.approx(0.0, abs=1e-5)
    for prefix in ("native", "design", "fold"):
        quat = np.array(
            [rows[0][f"{prefix}_tij_{k}"] for k in ("qw", "qx", "qy", "qz")],
            dtype=np.float64,
        )
        assert float(np.linalg.norm(quat)) == pytest.approx(1.0, abs=1e-6)
        assert rows[0][f"{prefix}_tij_qw"] >= -1e-12


def test_pmpnn_retention_is_design_vs_pmpnn_fold(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    pmpnn_dir = tmp_path / "esmfold" / "s0"
    pmpnn_dir.mkdir(parents=True)
    pmpnn = pmpnn_dir / "T=0.1, sample=3, score=1.0.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(design, coords)
    _write_nca_c_pdb(pmpnn, coords)
    metadata = {
        "s0": {
            "name": "1QJG_ext10",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": _two_group_keys(),
        }
    }
    pmpnn_df = pd.DataFrame(
        [
            {
                "name": "s0",
                "sample": 3,
                "path": str(pmpnn),
                "pass_final": True,
                "protocol": "pmpnn_esmfold",
            },
            {
                "name": "s0",
                "sample": 1,
                "path": str(tmp_path / "missing.pdb"),
                "pass_final": False,
                "protocol": "pmpnn_esmfold",
            },
        ]
    )
    df = compute_pmpnn_layout_pair_metrics(
        metadata, pmpnn_df, esmfold_dir=tmp_path / "esmfold"
    )
    assert len(df) == 1
    assert int(df.iloc[0]["pmpnn_sample"]) == 3
    assert df.iloc[0]["fold_source"] == "pmpnn_esmfold"
    assert df.iloc[0]["retention_trans_A"] == pytest.approx(0.0, abs=1e-5)


def test_global_rigid_motion_preserves_relative_pose(tmp_path):
    coords = _native_coords()
    rot = _rotation_about_z(35.0)
    trans = np.array([12.0, -4.0, 7.0])
    moved = {
        key: apply_rigid(np.asarray(ca), rot, trans) for key, ca in coords.items()
    }
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(
        design,
        moved,
        n_offset=rot @ np.array([-0.824, 0.660, 0.0]),
        c_offset=rot @ np.array([1.526, 0.0, 0.0]),
    )
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "task1",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": _two_group_keys(),
        },
    )
    assert rows[0]["excursion_trans_A"] == pytest.approx(0.0, abs=5e-2)
    assert rows[0]["excursion_rot_deg"] == pytest.approx(0.0, abs=1e-1)


def test_one_rigid_shared_se3_has_zero_excursion(tmp_path):
    test_global_rigid_motion_preserves_relative_pose(tmp_path)


def test_duplicate_designs_have_zero_dispersion(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    d0 = tmp_path / "d0.pdb"
    d1 = tmp_path / "d1.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(d0, coords)
    _write_nca_c_pdb(d1, coords)
    keys = _two_group_keys()
    metadata = {
        "s0": {"name": "task1", "path": str(d0), "native_pdb": str(native), "source_residue_keys": keys},
        "s1": {"name": "task1", "path": str(d1), "native_pdb": str(native), "source_residue_keys": keys},
    }
    folding = pd.DataFrame(
        [
            {"name": "s0", "pass_final": True},
            {"name": "s1", "pass_final": True},
        ]
    )
    _pairs, summaries = compute_run_layout_metrics(metadata, folding_df=folding)
    pw = summaries["pairwise_all_by_task"]["task1"]
    assert pw["median_pairwise_layout_translation_distance"] == pytest.approx(0.0, abs=1e-6)
    assert pw["median_pairwise_layout_rotation_distance"] == pytest.approx(0.0, abs=1e-5)
    assert pw["num_design_pairs"] == 1
    assert summaries["pairwise_pass_final_by_task"]["task1"]["num_design_pairs"] == 1
    assert summaries["layout_frame"] == LAYOUT_FRAME


def test_dispersion_does_not_cross_tasks(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    d0 = tmp_path / "d0.pdb"
    d1 = tmp_path / "d1.pdb"
    d2 = tmp_path / "d2.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(d0, coords)
    _write_nca_c_pdb(d1, coords)
    _write_nca_c_pdb(d2, coords)
    keys = _two_group_keys()
    metadata = {
        "s0": {"name": "taskA", "path": str(d0), "native_pdb": str(native), "source_residue_keys": keys},
        "s1": {"name": "taskA", "path": str(d1), "native_pdb": str(native), "source_residue_keys": keys},
        "s2": {"name": "taskB", "path": str(d2), "native_pdb": str(native), "source_residue_keys": keys},
    }
    _pairs, summaries = compute_run_layout_metrics(metadata)
    assert summaries["pairwise_all_by_task"]["taskA"]["num_design_pairs"] == 1
    assert summaries["pairwise_all_by_task"]["taskB"]["num_design_pairs"] == 0


def test_pair_equal_weight_not_residue_count(tmp_path):
    extra_size = 12
    coords = _native_coords(extra_group=True, extra_size=extra_size)
    moved = dict(coords)
    shift = np.array([10.0, 0.0, 0.0])
    moved[("A", 40)] = coords[("A", 40)] + shift
    moved[("A", 41)] = coords[("A", 41)] + shift
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(design, moved)
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "task1",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": _two_group_keys(extra_group=True, extra_size=extra_size),
        },
    )
    trans = { (r["group_i"], r["group_j"]): r["excursion_trans_A"] for r in rows }
    assert trans[(1, 2)] == pytest.approx(10.0, abs=0.05)
    assert trans[(2, 3)] == pytest.approx(10.0, abs=0.05)
    assert trans[(1, 3)] == pytest.approx(0.0, abs=0.05)
    equal_mean = float(np.mean(list(trans.values())))
    residue_weights = {(1, 2): 4, (1, 3): 2 + extra_size, (2, 3): 2 + extra_size}
    weighted = sum(trans[pair] * residue_weights[pair] for pair in trans) / sum(
        residue_weights.values()
    )
    assert equal_mean == pytest.approx(20.0 / 3.0, abs=0.05)
    assert abs(equal_mean - weighted) > 0.5


def test_group_frame_origin_is_backbone_com(tmp_path):
    coords = {
        ("A", 1): np.array([0.0, 0.0, 0.0]),
        ("A", 50): np.array([8.0, 3.0, 0.0]),
    }
    path = tmp_path / "g.pdb"
    _write_nca_c_pdb(path, coords)
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("g", str(path))
    pairs = group_residue_pairs(
        [
            {
                "native_chain": "A",
                "native_res_idx": 50,
                "output_chain": "A",
                "output_res_idx": 50,
                "group_id": 1,
            },
            {
                "native_chain": "A",
                "native_res_idx": 1,
                "output_chain": "A",
                "output_res_idx": 1,
                "group_id": 1,
            },
        ],
        1,
    )
    rot, origin = group_frame(structure, pairs, use_native=True)
    com = np.mean(
        [
            np.asarray(structure[0]["A"][50][atom].coord)
            for atom in ("N", "CA", "C")
        ]
        + [
            np.asarray(structure[0]["A"][1][atom].coord)
            for atom in ("N", "CA", "C")
        ],
        axis=0,
    )
    assert origin == pytest.approx(com, abs=1e-6)
    assert rot == pytest.approx(np.eye(3), abs=1e-8)


def test_folded_single_chain_falls_back_when_output_chain_missing(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    folded = tmp_path / "folded.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(design, coords)
    _write_nca_c_pdb(folded, coords)
    keys = [
        {
            "native_chain": "A",
            "native_res_idx": 10,
            "output_chain": "F",
            "output_res_idx": 10,
            "group_id": 1,
        },
        {
            "native_chain": "A",
            "native_res_idx": 11,
            "output_chain": "F",
            "output_res_idx": 11,
            "group_id": 1,
        },
        {
            "native_chain": "A",
            "native_res_idx": 40,
            "output_chain": "F",
            "output_res_idx": 40,
            "group_id": 2,
        },
        {
            "native_chain": "A",
            "native_res_idx": 41,
            "output_chain": "F",
            "output_res_idx": 41,
            "group_id": 2,
        },
    ]
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "task1",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": keys,
        },
        folded_path=folded,
    )
    assert len(rows) == 1
    assert rows[0]["retention_trans_A"] == pytest.approx(0.0, abs=1e-5)


def test_design_frame_finds_scaffold_when_output_chain_is_motif_copy(tmp_path):
    """Viz PDB: scaffold F 1–60, native copy on A 61–84; keys still say A:idx."""
    native_coords = _native_coords()
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    _write_nca_c_pdb(native, native_coords)
    copy_shift = np.array([50.0, 0.0, 0.0])
    design_coords = {
        ("F", 10): native_coords[("A", 10)],
        ("F", 11): native_coords[("A", 11)],
        ("F", 40): native_coords[("A", 40)],
        ("F", 41): native_coords[("A", 41)],
        ("A", 61): native_coords[("A", 10)] + copy_shift,
        ("A", 62): native_coords[("A", 11)] + copy_shift,
        ("A", 80): native_coords[("A", 40)] + copy_shift,
        ("A", 81): native_coords[("A", 41)] + copy_shift,
    }
    for idx in range(1, 9):
        design_coords[("F", idx)] = np.array([float(idx), 20.0, 0.0])
    _write_nca_c_pdb(design, design_coords)
    keys = [
        {
            "native_chain": "A",
            "native_res_idx": native_idx,
            "output_chain": "A",
            "output_res_idx": output_idx,
            "group_id": group_id,
        }
        for native_idx, output_idx, group_id in (
            (10, 10, 1),
            (11, 11, 1),
            (40, 40, 2),
            (41, 41, 2),
        )
    ]
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "4JHW",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": keys,
        },
    )
    assert len(rows) == 1
    assert rows[0]["excursion_trans_A"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["excursion_rot_deg"] == pytest.approx(0.0, abs=1e-5)


def test_design_frame_prefers_samples_eval_single_chain(tmp_path):
    native_coords = _native_coords()
    native = tmp_path / "native.pdb"
    samples = tmp_path / "samples"
    samples_eval = tmp_path / "samples_eval"
    samples.mkdir()
    samples_eval.mkdir()
    viz = samples / "s0.pdb"
    eval_pdb = samples_eval / "s0.pdb"
    _write_nca_c_pdb(native, native_coords)
    copy_shift = np.array([50.0, 0.0, 0.0])
    _write_nca_c_pdb(
        viz,
        {
            ("F", 10): native_coords[("A", 10)],
            ("F", 11): native_coords[("A", 11)],
            ("F", 40): native_coords[("A", 40)],
            ("F", 41): native_coords[("A", 41)],
            ("A", 61): native_coords[("A", 10)] + copy_shift,
            ("A", 62): native_coords[("A", 11)] + copy_shift,
            ("A", 80): native_coords[("A", 40)] + copy_shift,
            ("A", 81): native_coords[("A", 41)] + copy_shift,
        },
    )
    _write_nca_c_pdb(eval_pdb, native_coords)
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "4JHW",
            "path": str(viz),
            "native_pdb": str(native),
            "source_residue_keys": _two_group_keys(),
        },
    )
    assert len(rows) == 1
    assert rows[0]["excursion_trans_A"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["excursion_rot_deg"] == pytest.approx(0.0, abs=1e-5)


def test_design_frame_prefers_scaffold_when_residue_on_several_chains(tmp_path):
    native_coords = _native_coords()
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    _write_nca_c_pdb(native, native_coords)
    copy_shift = np.array([50.0, 0.0, 0.0])
    design_coords = {
        ("F", 10): native_coords[("A", 10)],
        ("F", 11): native_coords[("A", 11)],
        ("F", 40): native_coords[("A", 40)],
        ("F", 41): native_coords[("A", 41)],
        ("C", 10): native_coords[("A", 10)] + copy_shift,
        ("C", 11): native_coords[("A", 11)] + copy_shift,
        ("C", 40): native_coords[("A", 40)] + copy_shift,
        ("C", 41): native_coords[("A", 41)] + copy_shift,
        ("A", 61): native_coords[("A", 10)] + copy_shift,
        ("A", 62): native_coords[("A", 11)] + copy_shift,
        ("A", 80): native_coords[("A", 40)] + copy_shift,
        ("A", 81): native_coords[("A", 41)] + copy_shift,
    }
    for idx in range(1, 9):
        design_coords[("F", idx)] = np.array([float(idx), 20.0, 0.0])
    _write_nca_c_pdb(design, design_coords)
    keys = [
        {
            "native_chain": "A",
            "native_res_idx": idx,
            "output_chain": "A",
            "output_res_idx": idx,
            "group_id": 1 if idx < 20 else 2,
        }
        for idx in (10, 11, 40, 41)
    ]
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "4JHW",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": keys,
        },
    )
    assert len(rows) == 1
    assert rows[0]["excursion_trans_A"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["excursion_rot_deg"] == pytest.approx(0.0, abs=1e-5)


def test_invert_compose_matches_relative_pose_formula():
    rot_i = _rotation_about_z(20.0)
    rot_j = _rotation_about_z(-15.0)
    o_i = np.array([1.0, 2.0, 3.0])
    o_j = np.array([4.0, 0.0, 3.0])
    r_ij, t_ij = invert_compose((rot_i, o_i), (rot_j, o_j))
    assert t_ij == pytest.approx(rot_i.T @ (o_j - o_i), abs=1e-10)
    assert r_ij == pytest.approx(rot_i.T @ rot_j, abs=1e-10)
    trans, rot_deg = relative_pose_delta((r_ij, t_ij), (r_ij, t_ij))
    assert trans == pytest.approx(0.0, abs=1e-12)
    assert rot_deg == pytest.approx(0.0, abs=1e-12)


def test_rigid_motion_of_group2_is_recovered(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    _write_nca_c_pdb(native, coords)
    rot = _rotation_about_z(30.0)
    trans = np.array([5.0, -2.0, 1.0])
    com2 = 0.5 * (coords[("A", 40)] + coords[("A", 41)])
    rot_by_res = {("A", 40): rot, ("A", 41): rot}
    moved_ca = dict(coords)
    for key in (("A", 40), ("A", 41)):
        moved_ca[key] = rot @ (coords[key] - com2) + com2 + trans
    _write_nca_c_pdb_atoms(design, _ca_to_nca_c(moved_ca, rot_by_res=rot_by_res))
    rows, _rel = compute_sample_layout_pair_metrics(
        "s0",
        {
            "name": "task1",
            "path": str(design),
            "native_pdb": str(native),
            "source_residue_keys": _two_group_keys(),
        },
    )
    assert rows[0]["excursion_rot_deg"] == pytest.approx(30.0, abs=0.5)
    assert rows[0]["excursion_trans_A"] == pytest.approx(float(np.linalg.norm(trans)), abs=0.2)


def test_kabsch_less_sensitive_to_first_residue_than_nca_c(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    design = tmp_path / "design.pdb"
    _write_nca_c_pdb(native, coords)
    atoms = _ca_to_nca_c(coords)
    n0, ca0, c0 = atoms[("A", 10)]
    atoms[("A", 10)] = (n0 + np.array([0.0, 0.0, 2.0]), ca0, c0)
    _write_nca_c_pdb_atoms(design, atoms)

    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    native_s = parser.get_structure("n", str(native))
    design_s = parser.get_structure("d", str(design))
    keys = _two_group_keys()
    pairs1 = group_residue_pairs(keys, 1)
    pairs2 = group_residue_pairs(keys, 2)

    def _first_res_frame(structure, pairs):
        chain_id, res_idx = pairs[0][0]
        residue = structure[0][chain_id][res_idx]
        rot, _ = compute_frame(
            np.asarray(residue["N"].coord),
            np.asarray(residue["CA"].coord),
            np.asarray(residue["C"].coord),
        )
        coords_bb = []
        for native_pair, _sample in pairs:
            res = structure[0][native_pair[0]][native_pair[1]]
            for atom in ("N", "CA", "C"):
                coords_bb.append(np.asarray(res[atom].coord))
        origin = np.mean(coords_bb, axis=0)
        return np.asarray(rot, dtype=np.float64), origin

    old_native = invert_compose(
        _first_res_frame(native_s, pairs1), _first_res_frame(native_s, pairs2)
    )
    old_design = invert_compose(
        _first_res_frame(design_s, pairs1), _first_res_frame(design_s, pairs2)
    )
    _old_trans, old_rot = relative_pose_delta(old_native, old_design)

    new_native = invert_compose(
        group_frame(native_s, pairs1, use_native=True),
        group_frame(native_s, pairs2, use_native=True),
    )
    nat1 = backbone_coords(native_s, pairs1, use_native=True)
    nat2 = backbone_coords(native_s, pairs2, use_native=True)
    new_design = invert_compose(
        group_frame(design_s, pairs1, use_native=True, reference_coords=nat1),
        group_frame(design_s, pairs2, use_native=True, reference_coords=nat2),
    )
    _new_trans, new_rot = relative_pose_delta(new_native, new_design)
    assert new_rot < old_rot
    assert old_rot > 5.0


def test_stage2_parent_task_pools_dispersion(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    d0 = tmp_path / "d0.pdb"
    d1 = tmp_path / "d1.pdb"
    _write_nca_c_pdb(native, coords)
    _write_nca_c_pdb(d0, coords)
    _write_nca_c_pdb(d1, coords)
    keys = _two_group_keys()
    metadata = {
        "s0": {
            "name": "1QJG_ext10__len_78_protein_id0_0_0",
            "path": str(d0),
            "native_pdb": str(native),
            "source_residue_keys": keys,
        },
        "s1": {
            "name": "1QJG_ext10__len_99_protein_id0_1_0",
            "path": str(d1),
            "native_pdb": str(native),
            "source_residue_keys": keys,
        },
    }
    pairs, summaries = compute_run_layout_metrics(metadata)
    assert set(pairs["task"]) == {"1QJG_ext10"}
    assert summaries["pairwise_all_by_task"]["1QJG_ext10"]["num_design_pairs"] == 1


def test_kabsch_backbone_frame_identity():
    coords = np.array(
        [
            [-0.824, 0.660, 0.0],
            [0.0, 0.0, 0.0],
            [1.526, 0.0, 0.0],
            [3.8 - 0.824, 0.660, 0.0],
            [3.8, 0.0, 0.0],
            [3.8 + 1.526, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rot, origin = kabsch_backbone_frame(coords)
    assert origin == pytest.approx(coords.mean(axis=0), abs=1e-8)
    assert rot == pytest.approx(np.eye(3), abs=1e-5)
    rot_self, _ = kabsch_backbone_frame(coords, reference_coords=coords)
    assert rot_self == pytest.approx(np.eye(3), abs=1e-5)


def test_same_translation_magnitude_different_direction_nonzero_distance(tmp_path):
    coords = _native_coords()
    native = tmp_path / "native.pdb"
    plus = tmp_path / "plus.pdb"
    minus = tmp_path / "minus.pdb"
    _write_nca_c_pdb(native, coords)
    shifted_p = dict(coords)
    shifted_m = dict(coords)
    shift = np.array([10.0, 0.0, 0.0])
    shifted_p[("A", 40)] = coords[("A", 40)] + shift
    shifted_p[("A", 41)] = coords[("A", 41)] + shift
    shifted_m[("A", 40)] = coords[("A", 40)] - shift
    shifted_m[("A", 41)] = coords[("A", 41)] - shift
    _write_nca_c_pdb(plus, shifted_p)
    _write_nca_c_pdb(minus, shifted_m)
    keys = _two_group_keys()
    meta = {"name": "task1", "native_pdb": str(native), "source_residue_keys": keys}
    rows_p, rel_p = compute_sample_layout_pair_metrics(
        "plus", {**meta, "path": str(plus)}
    )
    rows_m, rel_m = compute_sample_layout_pair_metrics(
        "minus", {**meta, "path": str(minus)}
    )
    assert rows_p[0]["excursion_trans_A"] == pytest.approx(rows_m[0]["excursion_trans_A"], abs=0.05)
    d_trans, _d_rot = layout_pair_distance(rel_p, rel_m)
    assert d_trans == pytest.approx(20.0, abs=0.2)


def test_quaternion_double_cover_rotation_distance_zero():
    r, t = np.eye(3), np.array([1.0, 2.0, 3.0])
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    pos = frame_from_wxyz_trans(qw, qx, qy, qz, *t)
    neg = frame_from_wxyz_trans(-qw, -qx, -qy, -qz, *t)
    trans, rot = relative_pose_delta(pos, neg)
    assert trans == pytest.approx(0.0, abs=1e-12)
    assert rot == pytest.approx(0.0, abs=1e-8)
    d_t, d_r = layout_pair_distance({(1, 2): pos}, {(1, 2): neg})
    assert d_t == pytest.approx(0.0, abs=1e-12)
    assert d_r == pytest.approx(0.0, abs=1e-8)
    _ = r


def test_atom_order_permutation_with_correspondence_preserved():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.0, 0.0, 0.2],
            [10.0, 1.0, 0.0],
            [11.0, 1.1, 0.0],
            [12.0, 1.0, 0.2],
        ],
        dtype=np.float64,
    )
    moved = coords + np.array([3.0, -1.0, 2.0])
    perm = np.array([2, 0, 1, 5, 3, 4])
    rot_a, _ = kabsch_backbone_frame(moved, reference_coords=coords)
    rot_b, _ = kabsch_backbone_frame(moved[perm], reference_coords=coords[perm])
    assert rot_a == pytest.approx(rot_b, abs=1e-6)


def test_bootstrap_resamples_designs_not_pair_rows():
    relatives = {
        "a": {(1, 2): (np.eye(3), np.array([0.0, 0.0, 0.0]))},
        "b": {(1, 2): (np.eye(3), np.array([10.0, 0.0, 0.0]))},
        "c": {(1, 2): (np.eye(3), np.array([10.0, 0.0, 0.0]))},
    }
    rng = np.random.default_rng(0)
    boot = bootstrap_median_pairwise(
        relatives, ["a", "b", "c"], n_boot=80, rng=rng
    )
    assert boot["n_boot"] == 80
    # Precomputed pair rows are {10, 10, 0}; resampling those rows cannot
    # produce a replicate whose only pair is the duplicate (median 0) as
    # often as resampling designs {b, c}. Design-bootstrap CI must include 0.
    assert boot["bootstrap_trans_ci_lo"] == pytest.approx(0.0, abs=1e-9)
    assert boot["bootstrap_trans_ci_hi"] >= 0.0
    clustered = bootstrap_median_pairwise(
        relatives,
        ["a", "b", "c"],
        cluster_ids={"a": "src0", "b": "src0", "c": "src1"},
        n_boot=40,
        rng=np.random.default_rng(1),
    )
    stats = pairwise_layout_stats(
        relatives, ["a", "b", "c"], cluster_ids={"a": "src0", "b": "src0", "c": "src1"}
    )
    assert stats["bootstrap_unit"] == "source_layout"
    assert stats["n_bootstrap_units"] == 2
    assert np.isfinite(clustered["bootstrap_trans_ci_lo"])


def test_pairwise_distance_matrices_match_pair_loop():
    relatives = {
        "a": {(1, 2): (np.eye(3), np.array([0.0, 0.0, 0.0]))},
        "b": {(1, 2): (np.eye(3), np.array([10.0, 0.0, 0.0]))},
        "c": {(1, 2): (np.eye(3), np.array([10.0, 4.0, 0.0]))},
    }
    names, trans_mat, rot_mat = pairwise_layout_distance_matrices(
        relatives, ["a", "b", "c"]
    )
    assert names == ["a", "b", "c"]
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                assert trans_mat[i, j] == pytest.approx(0.0)
                continue
            d_t, d_r = layout_pair_distance(relatives[name_a], relatives[name_b])
            assert trans_mat[i, j] == pytest.approx(d_t, abs=1e-9)
            assert rot_mat[i, j] == pytest.approx(d_r, abs=1e-8)
