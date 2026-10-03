import os

import hydra
import omegaconf


def expand_env_in_dataset_config(cfg):
    """Resolve dataset YAML paths and verify expected manifests exist."""

    def walk(node):
        if omegaconf.OmegaConf.is_dict(node):
            for key, value in list(node.items()):
                if key == "data_dir" and isinstance(value, str):
                    if value.startswith("ENV:"):
                        env_name = value[4:]
                        path = os.environ.get(env_name, "") or ""
                        if not path:
                            raise FileNotFoundError(
                                f"Dataset data_dir uses ENV:{env_name} but the environment variable is unset or empty. "
                                "Export it to a directory that contains manifest.json (see scripts/data/README.md)."
                            )
                    else:
                        path = os.path.expandvars(os.path.expanduser(value))
                    node[key] = path
                    manifest = os.path.join(path, "manifest.json")
                    if not os.path.isfile(manifest):
                        raise FileNotFoundError(
                            f"Training data not found: expected {manifest}. "
                            "Set NOMADFLOW_DATA_FRAMEDIFF_BOLTZ and NOMADFLOW_DATA_AFDB_BOLTZ "
                            "to directories that contain manifest.json (preprocessed Boltz data; "
                            "see scripts/data/README.md)."
                        )
                else:
                    walk(value)
        elif omegaconf.OmegaConf.is_list(node):
            for item in node:
                walk(item)

    walk(cfg)
    return cfg


def load_dataset_config(config_path):
    dataset_config = omegaconf.OmegaConf.load(config_path)
    return expand_env_in_dataset_config(dataset_config)


def instantiate_dataset_from_config(config_path):
    dataset_config = load_dataset_config(config_path)
    return hydra.utils.instantiate(dataset_config)
