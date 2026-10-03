from pathlib import Path

import yaml


def test_pretrained_motif_hydra_config_has_model_target():
    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / "configs/sampling/pretrained/motif_scaffolding_model/.hydra/config.yaml"
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert payload["model"]["_target_"] == "nomadflow.model.denoiser.IpaMultiRigidDenoiser"
    assert payload["model"]["c_s"] == 768
    assert payload["model"]["num_blocks"] == 12


def test_0c62dc8a_hydra_config_has_nomadflow_target():
    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / "weights/0c62dc8a/.hydra/config.yaml"
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert payload["model"]["_target_"] == "nomadflow.model.denoiser.IpaMultiRigidDenoiser"
    assert payload["model"]["c_s"] == 768
    assert payload["model"]["num_blocks"] == 12
    dumped = cfg_path.read_text(encoding="utf-8")
    assert "_target_: proteinzen." not in dumped
    assert "/trinity/" not in dumped
    assert "/data/home/" not in dumped
