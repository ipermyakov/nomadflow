# conda create -n nomadflow python=3.10
# conda activate nomadflow
set -euo pipefail
#
# Pins match environment.yml. Do not leave these unpinned:
# cuequivariance-ops-torch-cu12>=0.12 requires torch>=2.11, and a later
# `pip install` (default PyPI index, not cu126) then upgrades torch to the
# latest CUDA 13 wheel and leaves torchvision 0.21 behind.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126 \

pip install \
    cuequivariance==0.6.1 \
    cuequivariance-torch==0.6.1 \
    cuequivariance-ops-torch-cu12==0.6.1 \

# PyG wheels are not on PyPI. Versions match environment.yml.
pip install \
    pyg-lib==0.4.0+pt26cu126 \
    torch-scatter==2.1.2+pt26cu126 \
    torch-sparse==0.6.18+pt26cu126 \
    torch-cluster==1.6.3+pt26cu126 \
    torch-spline-conv==1.2.2+pt26cu126 \
    torch-geometric==2.6.1 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# sample.py and train.py pass weights_only= to Trainer.predict/fit.
# That argument exists from Lightning 2.6. The lightning package depends on
# pytorch-lightning without a pin, so both stay on the same version.
pip install "lightning==2.6.6" "pytorch-lightning==2.6.6" "torchmetrics==1.7.3" pandas

conda install -y -c https://conda.rosettacommons.org \
    pyrosetta

pip install \
    mdtraj \
    biopython \
    black \
    darglint \
    dill \
    dm-tree \
    e3nn \
    geomstats \
    hydra-zen \
    isort \
    mypy \
    ninja \
    pylint \
    pytest \
    rdkit \
    seaborn \
    mlflow \
    optuna \
    mashumaro \
    p_tqdm

pip install -e .
