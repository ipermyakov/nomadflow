"""Tests for eval PDB chain-ID contract across motif sampling modes."""

from __future__ import annotations

import json
from pathlib import Path

from nomadflow.benchmark.eval_layout import (
    check_eval_layout,
    pmpnn_seqs_is_complete,
    samples_eval_is_complete,
    sync_eval_chain_metadata,
)
from nomadflow.benchmark.pmpnn_mapping import build_pmpnn_fixed_pos_dict
from nomadflow.benchmark.rmsd_mapping import build_rmsd_mapping
from scripts.analysis.prepare_eval_pdbs_from_samples import (
    prepare_eval_pdbs,
    prepare_eval_pdbs_from_multichain_pdb,
)


def _atom(resname: str, chain: str, resseq: int, x: float, y: float, z: float, serial: int = 1) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
    )


def _write_pdb(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def _write_run(tmp_path: Path, samples: dict[str, dict], *, eval_pdbs: dict[str, list[str]] | None = None) -> Path:
    samples_dir = tmp_path / "samples"
    metadata: dict[str, dict] = {}
    for name, spec in samples.items():
        _write_pdb(samples_dir / f"{name}.pdb", spec["pdb_lines"])
        metadata[name] = spec["metadata"]
    (tmp_path / "samples_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if eval_pdbs:
        eval_dir = tmp_path / "samples_eval"
        for name, lines in eval_pdbs.items():
            _write_pdb(eval_dir / f"{name}.pdb", lines)
    return tmp_path


def _eval_chain(path: Path) -> str:
    chains = {
        line[21:22]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    }
    assert len(chains) == 1
    return next(iter(chains))


def test_prepare_keeps_native_chain_mode3(tmp_path):
    src = tmp_path / "native_b.pdb"
    _write_pdb(src, [_atom("ALA", "B", 7, 1.0, 2.0, 3.0)])
    out = tmp_path / "eval.pdb"
    written = prepare_eval_pdbs_from_multichain_pdb(src, out, sample_chain_name="A")
    assert written == "B"
    assert _eval_chain(out) == "B"


def test_prepare_mode1_extracts_sample_chain_p_not_relabel_a(tmp_path):
    src = tmp_path / "viz.pdb"
    _write_pdb(
        src,
        [
            _atom("ALA", "A", 1, 0.0, 0.0, 0.0, serial=1),
            _atom("GLY", "P", 1, 1.0, 0.0, 0.0, serial=2),
            _atom("VAL", "V", 1, 10.0, 0.0, 0.0, serial=3),
        ],
    )
    out = tmp_path / "eval.pdb"
    written = prepare_eval_pdbs_from_multichain_pdb(src, out, sample_chain_name="P")
    assert written == "P"
    assert _eval_chain(out) == "P"


def test_prepare_run_mode3_persists_native_chain_in_metadata(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "len_53_1ycr": {
                "pdb_lines": [_atom("ALA", "B", 19, 1.0, 2.0, 3.0)],
                "metadata": {
                    "name": "1YCR",
                    "fixed_seq_res_idx": [19],
                    "fixed_seq_chain": ["B"],
                    "fixed_bb_res_idx": [19],
                    "fixed_bb_chain": ["B"],
                },
            },
            "len_77_3ixt": {
                "pdb_lines": [_atom("GLY", "P", 1, 4.0, 5.0, 6.0)],
                "metadata": {
                    "name": "3IXT",
                    "fixed_seq_res_idx": [1],
                    "fixed_seq_chain": ["P"],
                    "fixed_bb_res_idx": [1],
                    "fixed_bb_chain": ["P"],
                },
            },
        },
    )
    assert prepare_eval_pdbs(run) == "prepared"
    metadata = json.loads((run / "samples_metadata.json").read_text(encoding="utf-8"))
    assert _eval_chain(run / "samples_eval" / "len_53_1ycr.pdb") == "B"
    assert metadata["len_53_1ycr"]["sample_chain_name"] == "B"
    assert metadata["len_53_1ycr"]["fixed_seq_chain"] == ["B"]
    assert metadata["len_53_1ycr"]["rmsd_reference_chain"] == "B"
    assert metadata["len_53_1ycr"]["rmsd_folded_chain"] == "A"
    assert _eval_chain(run / "samples_eval" / "len_77_3ixt.pdb") == "P"
    assert metadata["len_77_3ixt"]["sample_chain_name"] == "P"
    assert build_pmpnn_fixed_pos_dict(metadata)["len_77_3ixt"] == {"P": [1]}
    mapping = build_rmsd_mapping(metadata["len_77_3ixt"])
    assert mapping["reference_chain"] == "P"
    assert mapping["folded_chain"] == "A"
    assert check_eval_layout(run) == []


def test_prepare_run_mode1_keeps_sample_chain_name_p(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "len_60_3ixt": {
                "pdb_lines": [
                    _atom("ALA", "A", 1, 0.0, 0.0, 0.0, serial=1),
                    _atom("GLY", "P", 1, 1.0, 0.0, 0.0, serial=2),
                    _atom("VAL", "V", 1, 10.0, 0.0, 0.0, serial=3),
                ],
                "metadata": {
                    "name": "3IXT",
                    "sample_chain_name": "P",
                    "fixed_seq_res_idx": [1],
                    "fixed_seq_chain": ["P"],
                    "fixed_bb_res_idx": [1],
                    "fixed_bb_chain": ["P"],
                },
            }
        },
    )
    assert prepare_eval_pdbs(run) == "prepared"
    metadata = json.loads((run / "samples_metadata.json").read_text(encoding="utf-8"))
    assert _eval_chain(run / "samples_eval" / "len_60_3ixt.pdb") == "P"
    assert metadata["len_60_3ixt"]["sample_chain_name"] == "P"
    assert metadata["len_60_3ixt"]["rmsd_reference_chain"] == "P"
    assert metadata["len_60_3ixt"]["rmsd_folded_chain"] == "A"
    assert check_eval_layout(run) == []


def test_prepare_run_mode2_stage2_extracts_scaffold_a_not_motif_b(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "len_98_protein_id0_15_1": {
                "pdb_lines": [
                    _atom("ALA", "A", 1, 0.0, 0.0, 0.0, serial=1),
                    _atom("ALA", "A", 16, 1.0, 0.0, 0.0, serial=2),
                    _atom("ALA", "A", 48, 2.0, 0.0, 0.0, serial=3),
                    _atom("ALA", "A", 84, 3.0, 0.0, 0.0, serial=4),
                    _atom("GLY", "B", 86, 4.0, 0.0, 0.0, serial=5),
                    _atom("GLY", "B", 98, 5.0, 0.0, 0.0, serial=6),
                ],
                "metadata": {
                    "name": "1QJG_ext10",
                    "fixed_seq_res_idx": [16, 48, 84],
                    "fixed_seq_chain": ["B", "B", "B"],
                    "fixed_bb_res_idx": [16, 48, 84],
                    "fixed_bb_chain": ["B", "B", "B"],
                },
            }
        },
    )
    assert prepare_eval_pdbs(run) == "prepared"
    metadata = json.loads((run / "samples_metadata.json").read_text(encoding="utf-8"))
    sample = metadata["len_98_protein_id0_15_1"]
    assert _eval_chain(run / "samples_eval" / "len_98_protein_id0_15_1.pdb") == "A"
    assert sample["sample_chain_name"] == "A"
    assert sample["fixed_seq_res_idx"] == [16, 48, 84]
    assert sample["fixed_seq_chain"] == ["A", "A", "A"]
    assert sample["rmsd_reference_chain"] == "A"
    assert build_pmpnn_fixed_pos_dict(metadata)["len_98_protein_id0_15_1"] == {
        "A": [16, 48, 84]
    }
    assert check_eval_layout(run) == []


def test_prepare_rewrites_complete_mode2_eval_extracted_as_motif_b(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "len_98_protein_id0_15_1": {
                "pdb_lines": [
                    _atom("ALA", "A", 1, 0.0, 0.0, 0.0, serial=1),
                    _atom("ALA", "A", 16, 1.0, 0.0, 0.0, serial=2),
                    _atom("ALA", "A", 84, 2.0, 0.0, 0.0, serial=3),
                    _atom("GLY", "B", 86, 3.0, 0.0, 0.0, serial=4),
                    _atom("GLY", "B", 98, 4.0, 0.0, 0.0, serial=5),
                ],
                "metadata": {
                    "sample_chain_name": "B",
                    "fixed_seq_res_idx": [16, 84],
                    "fixed_seq_chain": ["B", "B"],
                    "fixed_bb_res_idx": [16, 84],
                    "fixed_bb_chain": ["B", "B"],
                },
            }
        },
        eval_pdbs={
            "len_98_protein_id0_15_1": [
                _atom("GLY", "B", 86, 3.0, 0.0, 0.0, serial=1),
                _atom("GLY", "B", 98, 4.0, 0.0, 0.0, serial=2),
            ]
        },
    )
    assert samples_eval_is_complete(run)
    assert prepare_eval_pdbs(run) == "prepared"
    metadata = json.loads((run / "samples_metadata.json").read_text(encoding="utf-8"))
    sample = metadata["len_98_protein_id0_15_1"]
    assert _eval_chain(run / "samples_eval" / "len_98_protein_id0_15_1.pdb") == "A"
    assert sample["sample_chain_name"] == "A"
    assert sample["fixed_seq_res_idx"] == [16, 84]
    assert sample["fixed_seq_chain"] == ["A", "A"]
    assert check_eval_layout(run) == []


def test_samples_eval_complete_requires_name_match_not_nonempty(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "sample_a": {
                "pdb_lines": [_atom("ALA", "A", 1, 0.0, 0.0, 0.0)],
                "metadata": {"fixed_seq_res_idx": [1], "fixed_seq_chain": ["A"]},
            },
            "sample_b": {
                "pdb_lines": [_atom("GLY", "A", 1, 1.0, 0.0, 0.0)],
                "metadata": {"fixed_seq_res_idx": [1], "fixed_seq_chain": ["A"]},
            },
        },
        eval_pdbs={"sample_a": [_atom("ALA", "A", 1, 0.0, 0.0, 0.0)]},
    )
    assert not samples_eval_is_complete(run)
    assert prepare_eval_pdbs(run) == "prepared"
    assert samples_eval_is_complete(run)
    assert (run / "samples_eval" / "sample_b.pdb").is_file()


def test_prepare_skips_complete_mode2_eval_and_does_not_rewrite(tmp_path):
    original = _atom("ALA", "A", 5, 9.0, 8.0, 7.0)
    run = _write_run(
        tmp_path,
        {
            "layout_0": {
                "pdb_lines": [
                    _atom("ALA", "A", 5, 1.0, 2.0, 3.0, serial=1),
                    _atom("GLY", "B", 1, 4.0, 5.0, 6.0, serial=2),
                ],
                "metadata": {
                    "sample_chain_name": "A",
                    "fixed_seq_res_idx": [5],
                    "fixed_seq_chain": ["A"],
                    "fixed_bb_res_idx": [5],
                    "fixed_bb_chain": ["A"],
                },
            }
        },
        eval_pdbs={"layout_0": [original]},
    )
    before = (run / "samples_eval" / "layout_0.pdb").read_text(encoding="utf-8")
    assert prepare_eval_pdbs(run) == "skipped"
    after = (run / "samples_eval" / "layout_0.pdb").read_text(encoding="utf-8")
    assert after == before
    assert check_eval_layout(run) == []


def test_prepare_creates_missing_pmpnn_dict_without_rewriting_eval(tmp_path):
    original = _atom("ALA", "A", 5, 9.0, 8.0, 7.0)
    run = _write_run(
        tmp_path,
        {
            "layout_0": {
                "pdb_lines": [
                    _atom("ALA", "A", 5, 1.0, 2.0, 3.0, serial=1),
                    _atom("GLY", "B", 1, 4.0, 5.0, 6.0, serial=2),
                ],
                "metadata": {
                    "sample_chain_name": "A",
                    "fixed_seq_res_idx": [5],
                    "fixed_seq_chain": ["A"],
                    "fixed_bb_res_idx": [5],
                    "fixed_bb_chain": ["A"],
                },
            }
        },
        eval_pdbs={"layout_0": [original]},
    )
    assert not (run / "pmpnn_fixed_pos_dict.jsonl").exists()
    before = (run / "samples_eval" / "layout_0.pdb").read_text(encoding="utf-8")
    assert prepare_eval_pdbs(run) == "skipped"
    assert (run / "samples_eval" / "layout_0.pdb").read_text(encoding="utf-8") == before
    assert json.loads((run / "pmpnn_fixed_pos_dict.jsonl").read_text(encoding="utf-8"))[
        "layout_0"
    ] == {"A": [5]}
    assert check_eval_layout(run) == []


def test_prepare_skips_rewrite_after_repairing_stale_pmpnn_dict(tmp_path):
    original = _atom("ALA", "A", 5, 9.0, 8.0, 7.0)
    run = _write_run(
        tmp_path,
        {
            "layout_0": {
                "pdb_lines": [
                    _atom("ALA", "A", 5, 1.0, 2.0, 3.0, serial=1),
                    _atom("GLY", "B", 1, 4.0, 5.0, 6.0, serial=2),
                ],
                "metadata": {
                    "sample_chain_name": "A",
                    "fixed_seq_res_idx": [5],
                    "fixed_seq_chain": ["A"],
                    "fixed_bb_res_idx": [5],
                    "fixed_bb_chain": ["A"],
                },
            }
        },
        eval_pdbs={"layout_0": [original]},
    )
    (run / "pmpnn_fixed_pos_dict.jsonl").write_text(
        json.dumps({"layout_0": {"B": [5]}}), encoding="utf-8"
    )
    before = (run / "samples_eval" / "layout_0.pdb").read_text(encoding="utf-8")
    assert prepare_eval_pdbs(run) == "skipped"
    assert (run / "samples_eval" / "layout_0.pdb").read_text(encoding="utf-8") == before
    assert json.loads((run / "pmpnn_fixed_pos_dict.jsonl").read_text(encoding="utf-8"))[
        "layout_0"
    ] == {"A": [5]}
    assert check_eval_layout(run) == []


def test_hotfix_always_a_artifact_syncs_metadata_without_rewrite(tmp_path):
    eval_a = _atom("ALA", "A", 19, 1.0, 2.0, 3.0)
    run = _write_run(
        tmp_path,
        {
            "len_53_1ycr": {
                "pdb_lines": [_atom("ALA", "B", 19, 1.0, 2.0, 3.0)],
                "metadata": {
                    "name": "1YCR",
                    "fixed_seq_res_idx": [19],
                    "fixed_seq_chain": ["A"],
                    "fixed_bb_res_idx": [19],
                    "fixed_bb_chain": ["A"],
                    "rmsd_reference_chain": "A",
                    "rmsd_folded_chain": "A",
                },
            }
        },
        eval_pdbs={"len_53_1ycr": [eval_a]},
    )
    before = (run / "samples_eval" / "len_53_1ycr.pdb").read_text(encoding="utf-8")
    assert prepare_eval_pdbs(run) == "skipped"
    assert (run / "samples_eval" / "len_53_1ycr.pdb").read_text(encoding="utf-8") == before
    metadata = json.loads((run / "samples_metadata.json").read_text(encoding="utf-8"))
    assert metadata["len_53_1ycr"]["sample_chain_name"] == "A"
    assert metadata["len_53_1ycr"]["fixed_seq_chain"] == ["A"]
    assert _eval_chain(run / "samples_eval" / "len_53_1ycr.pdb") == "A"
    assert check_eval_layout(run) == []


def test_pmpnn_seqs_complete_requires_fa_per_sample(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "s0": {
                "pdb_lines": [_atom("ALA", "A", 1, 0.0, 0.0, 0.0)],
                "metadata": {"fixed_seq_res_idx": [1], "fixed_seq_chain": ["A"]},
            },
            "s1": {
                "pdb_lines": [_atom("GLY", "A", 1, 1.0, 0.0, 0.0)],
                "metadata": {"fixed_seq_res_idx": [1], "fixed_seq_chain": ["A"]},
            },
        },
    )
    seqs = run / "seqs"
    seqs.mkdir()
    (seqs / "s0.fa").write_text(">s0\nAAA\n", encoding="utf-8")
    assert not pmpnn_seqs_is_complete(run)
    (seqs / "s1.fa").write_text(">s1\nGGG\n", encoding="utf-8")
    assert pmpnn_seqs_is_complete(run)


def test_sync_eval_chain_metadata_reads_eval_pdb_not_native_samples(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "len_53_1ycr": {
                "pdb_lines": [_atom("ALA", "B", 19, 1.0, 2.0, 3.0)],
                "metadata": {
                    "fixed_seq_res_idx": [19],
                    "fixed_seq_chain": ["B"],
                },
            }
        },
        eval_pdbs={"len_53_1ycr": [_atom("ALA", "A", 19, 1.0, 2.0, 3.0)]},
    )
    assert sync_eval_chain_metadata(run)
    metadata = json.loads((run / "samples_metadata.json").read_text(encoding="utf-8"))
    assert metadata["len_53_1ycr"]["sample_chain_name"] == "A"
    assert metadata["len_53_1ycr"]["fixed_seq_chain"] == ["A"]
