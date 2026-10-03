#!/bin/bash
# One-time setup for the local ESMFold conda env used by scripts/analysis/esmfold.sh.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../env_vars.sh"

_activate_conda() {
    local env_name="$1"
    if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${env_name}" ]; then
        return 0
    fi
    if [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Could not find conda.sh. Run 'conda init bash' once or export CONDA_EXE." >&2
        exit 1
    fi
    conda activate "${env_name}"
}

ESMFOLD_ENV="${ESMFOLD_ENV:-esmfold}"
OPENFOLD_REF="${ESMFOLD_OPENFOLD_REF:-git+https://github.com/aqlaboratory/openfold.git@main}"

_activate_conda "${ESMFOLD_ENV}"

python -m pip install --upgrade pip
python -m pip install "fair-esm[esmfold]"
# deepspeed==0.5.9 (from fair-esm[esmfold]) is incompatible with numpy>=2.
python -m pip install "numpy<2"
python -m pip install modelcif
python -m pip install "dllogger @ git+https://github.com/NVIDIA/dllogger.git"
python -m pip install "openfold @ ${OPENFOLD_REF}" --no-build-isolation

python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["REPO_ROOT"]) / "scripts" / "analysis"))
import types

if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.inf = float("inf")
    sys.modules["torch._six"] = torch_six

from batched_esmfold_inference import (
    DEFAULT_ESMFOLD_MODEL_PATH,
    download_esmfold_weights,
    ensure_esm2_weights,
)

fold_path = Path(os.environ.get("ESMFOLD_MODEL_PATH", str(DEFAULT_ESMFOLD_MODEL_PATH)))
if not fold_path.exists() or fold_path.stat().st_size == 0:
    download_esmfold_weights(fold_path)
ensure_esm2_weights()
print("ESMFold weight files are present.")
PY

python - <<'PY'
import sys
import types

if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.inf = float("inf")
    sys.modules["torch._six"] = torch_six

from openfold.data.data_transforms import make_atom14_masks
from esm.esmfold.v1.esmfold import ESMFold

print("ESMFold import check passed.")
PY

echo "ESMFold env '${ESMFOLD_ENV}' is ready."
