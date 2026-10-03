import pandas as pd

from nomadflow.benchmark.success_criteria import (
    CRITERIA_VERSION,
    GLOBAL_HA_RMSD_MAX,
    MOTIF_BB_RMSD_MAX,
    MOTIF_CA_RMSD_MAX,
    MOTIF_HA_RMSD_MAX,
    SSC_RMSD_MAX,
    all_groups_pass,
    expected_group_count,
    filter_one_shot,
    filter_pmpnn_esmfold,
    filter_pmpnn_final,
    one_shot_final_pass,
    one_shot_pass,
    pmpnn_esmfold_pass,
)


def _one_shot_row(
    motif_ha: float,
    motif_bb: float,
    global_ha: float,
) -> pd.Series:
    return pd.Series(
        {
            "motif_all_atom_rmsd": motif_ha,
            "motif_ca_rmsd": motif_bb,
            "motif_bb_rmsd": motif_bb,
            "global_all_atom_rmsd": global_ha,
        }
    )


def _pmpnn_row(
    motif_ha: float,
    motif_bb: float,
    global_bb: float,
) -> pd.Series:
    return pd.Series(
        {
            "motif_all_atom_rmsd": motif_ha,
            "motif_ca_rmsd": motif_bb,
            "motif_bb_rmsd": motif_bb,
            "global_bb_rmsd": global_bb,
        }
    )


def test_one_shot_pass_at_threshold():
    row = _one_shot_row(
        MOTIF_HA_RMSD_MAX - 0.01,
        MOTIF_BB_RMSD_MAX - 0.01,
        GLOBAL_HA_RMSD_MAX - 0.01,
    )
    assert one_shot_pass(row)


def test_one_shot_fail_on_motif_ha():
    row = _one_shot_row(
        MOTIF_HA_RMSD_MAX + 0.01,
        MOTIF_BB_RMSD_MAX - 0.01,
        GLOBAL_HA_RMSD_MAX - 0.01,
    )
    assert not one_shot_pass(row)


def test_one_shot_fail_on_motif_ca():
    row = _one_shot_row(
        MOTIF_HA_RMSD_MAX - 0.01,
        MOTIF_CA_RMSD_MAX + 0.01,
        GLOBAL_HA_RMSD_MAX - 0.01,
    )
    assert not one_shot_pass(row)


def test_pmpnn_esmfold_pass_at_threshold():
    row = _pmpnn_row(
        MOTIF_HA_RMSD_MAX - 0.01,
        MOTIF_BB_RMSD_MAX - 0.01,
        2.0 - 0.01,
    )
    assert pmpnn_esmfold_pass(row)


def test_pmpnn_esmfold_fail_on_ssc():
    row = _pmpnn_row(
        MOTIF_HA_RMSD_MAX - 0.01,
        MOTIF_BB_RMSD_MAX - 0.01,
        2.0 + 0.01,
    )
    assert not pmpnn_esmfold_pass(row)


def test_one_shot_fail_on_exact_threshold():
    row = _one_shot_row(MOTIF_HA_RMSD_MAX, MOTIF_BB_RMSD_MAX - 0.01, GLOBAL_HA_RMSD_MAX - 0.01)
    assert not one_shot_pass(row)
    row = _one_shot_row(MOTIF_HA_RMSD_MAX - 0.01, MOTIF_BB_RMSD_MAX, GLOBAL_HA_RMSD_MAX - 0.01)
    assert not one_shot_pass(row)
    row = _one_shot_row(MOTIF_HA_RMSD_MAX - 0.01, MOTIF_BB_RMSD_MAX - 0.01, GLOBAL_HA_RMSD_MAX)
    assert not one_shot_pass(row)


def test_pmpnn_fail_on_exact_threshold():
    row = _pmpnn_row(MOTIF_HA_RMSD_MAX, MOTIF_BB_RMSD_MAX - 0.01, 1.99)
    assert not pmpnn_esmfold_pass(row)


def test_one_shot_prefers_ca_over_bb():
    row = pd.Series(
        {
            "motif_all_atom_rmsd": 1.0,
            "motif_ca_rmsd": 0.5,
            "motif_bb_rmsd": 2.0,
            "global_all_atom_rmsd": 1.5,
        }
    )
    assert one_shot_pass(row)
    row["motif_ca_rmsd"] = 1.5
    row["motif_bb_rmsd"] = 0.5
    assert not one_shot_pass(row)


def test_paper_thresholds():
    assert MOTIF_HA_RMSD_MAX == 2.0
    assert MOTIF_CA_RMSD_MAX == 1.0
    assert MOTIF_BB_RMSD_MAX == MOTIF_CA_RMSD_MAX
    assert GLOBAL_HA_RMSD_MAX == 2.0
    assert SSC_RMSD_MAX == 2.0


def test_all_groups_pass_conjunction_and_missing():
    assert all_groups_pass([1.0, 1.0], [0.5, 0.5], expected_n_groups=2)
    # Paper HA cut is 2.0 Å; 1.5 would fail the old upstream HA 1.5 cut.
    assert all_groups_pass([1.0, 1.5], [0.5, 0.5], expected_n_groups=2)
    assert not all_groups_pass([1.0, 2.01], [0.5, 0.5], expected_n_groups=2)
    assert not all_groups_pass([MOTIF_HA_RMSD_MAX], [0.5], expected_n_groups=1)
    assert not all_groups_pass([1.0], [MOTIF_CA_RMSD_MAX], expected_n_groups=1)
    assert not all_groups_pass([1.0], [0.5], expected_n_groups=2)
    assert not all_groups_pass([], [], expected_n_groups=1)
    assert not all_groups_pass(None, None, expected_n_groups=1)
    assert not all_groups_pass([1.0], [0.5], expected_n_groups=None)
    assert not all_groups_pass([1.0], [0.5], expected_n_groups=0)


def test_m1_final_equals_pooled_when_group_matches():
    row = _one_shot_row(1.0, 0.5, 1.5)
    row["pass_all_groups"] = True
    assert one_shot_pass(row)
    assert one_shot_final_pass(row)
    row["pass_all_groups"] = False
    assert one_shot_pass(row)
    assert not one_shot_final_pass(row)


def test_final_pass_is_groups_only():
    pooled_fail = _one_shot_row(2.0, 0.5, 1.5)
    pooled_fail["pass_all_groups"] = True
    missing_groups = _one_shot_row(1.0, 0.5, 1.5)
    assert one_shot_final_pass(pooled_fail)
    assert not one_shot_pass(pooled_fail)
    assert not one_shot_final_pass(missing_groups)
    assert one_shot_pass(missing_groups)


def test_missing_groups_are_not_final():
    row = _one_shot_row(1.0, 0.5, 1.5)
    assert not one_shot_final_pass(row)
    row["pass_all_groups"] = False
    assert not one_shot_final_pass(row)


def test_criteria_version_and_expected_group_count():
    assert CRITERIA_VERSION == "v3_groups_vs_native"
    assert expected_group_count({"motif_groups": [["A1"], ["A2-3"]]}) == 2
    assert (
        expected_group_count(
            {
                "source_residue_keys": [
                    {"group_id": 1},
                    {"group_id": 1},
                    {"group_id": 2},
                ]
            }
        )
        == 2
    )
    assert expected_group_count({}) == 0


def test_filter_helpers():
    df = pd.DataFrame(
        [
            {
                "name": "pass",
                "motif_all_atom_rmsd": 1.0,
                "motif_ca_rmsd": 0.5,
                "motif_bb_rmsd": 0.5,
                "global_all_atom_rmsd": 1.5,
            },
            {
                "name": "fail",
                "motif_all_atom_rmsd": 3.0,
                "motif_ca_rmsd": 0.5,
                "motif_bb_rmsd": 0.5,
                "global_all_atom_rmsd": 1.5,
            },
        ]
    )
    passed = filter_one_shot(df)
    assert len(passed) == 1
    assert passed.iloc[0]["name"] == "pass"

    pmpnn_df = pd.DataFrame(
        [
            {
                "name": "pass",
                "motif_all_atom_rmsd": 1.0,
                "motif_ca_rmsd": 0.5,
                "motif_bb_rmsd": 0.5,
                "global_bb_rmsd": 1.5,
            },
            {
                "name": "fail",
                "motif_all_atom_rmsd": 1.0,
                "motif_ca_rmsd": 0.5,
                "motif_bb_rmsd": 0.5,
                "global_bb_rmsd": 2.5,
            },
        ]
    )
    pmpnn_passed = filter_pmpnn_esmfold(pmpnn_df)
    assert len(pmpnn_passed) == 1
    assert pmpnn_passed.iloc[0]["name"] == "pass"

    pmpnn_final_df = pd.DataFrame(
        [
            {"name": "and_pass", "pass_final": True, "pass_all_groups": True},
            {"name": "pooled_only", "pass_final": False, "pass_all_groups": False},
        ]
    )
    pmpnn_final = filter_pmpnn_final(pmpnn_final_df)
    assert list(pmpnn_final["name"]) == ["and_pass"]
