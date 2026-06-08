#!/usr/bin/env bash
# setup_conda.sh — create and populate the metarl-iccbf conda environment.
#
# Usage (from the repo root):
#   bash setup_conda.sh
#
# Requirements:
#   - conda / miniconda (searches common locations automatically)
#   - NVIDIA GPU with driver ≥ 520 (RTX 3090/4090/A100 all work)
#   - Internet access for pip/conda downloads
#
# What it does:
#   1. Creates the conda env from environment.yml (python 3.11 + nvcc 12.8)
#   2. Installs PyTorch 2.10.0+cu128 via pip
#   3. Builds causal-conv1d and mamba-ssm from source against that torch
#   4. Installs all remaining project dependencies
#   5. Installs this package in editable mode
#
# Note: mujoco-py is intentionally omitted — it is deprecated and unused.
#       mosek is installed but requires a separate licence to run solvers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="metarl-iccbf"

# ── locate conda ────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    CONDA_CMD="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
elif [[ -x "/mnt/data/$USER/miniconda3/bin/conda" ]]; then
    CONDA_CMD="/mnt/data/$USER/miniconda3/bin/conda"
elif [[ -x "/opt/conda/bin/conda" ]]; then
    CONDA_CMD="/opt/conda/bin/conda"
else
    echo "ERROR: conda not found. Install miniconda first." >&2
    exit 1
fi

echo "Using conda: $CONDA_CMD"

# ── helper: run a command inside the env ────────────────────────────────────
RUN() { "$CONDA_CMD" run --no-capture-output -n "$ENV_NAME" "$@"; }

# ── 1. create env from environment.yml ──────────────────────────────────────
echo
echo "==> Creating conda environment '${ENV_NAME}' …"
"$CONDA_CMD" env create -f "$SCRIPT_DIR/environment.yml" -n "$ENV_NAME" \
    --force --quiet

# CUDA_HOME for extension compilation = the conda env prefix (nvcc lives there)
ENV_PREFIX="$("$CONDA_CMD" run -n "$ENV_NAME" python -c \
    "import sys, os; print(os.path.dirname(os.path.dirname(sys.executable)))")"
export CUDA_HOME="$ENV_PREFIX"
echo "    CUDA_HOME=$CUDA_HOME  (nvcc $("$CONDA_CMD" run -n "$ENV_NAME" nvcc --version | grep release | awk '{print $6}' | tr -d ,))"

# ── 2. PyTorch (cu128 matches the nvcc 12.8 we installed) ───────────────────
echo
echo "==> Installing PyTorch 2.10.0+cu128 …"
RUN pip install \
    "torch>=2.6.0" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --quiet

# ── 3. causal-conv1d from source ────────────────────────────────────────────
echo
echo "==> Building causal-conv1d from source …"
CUDA_HOME="$CUDA_HOME" MAX_JOBS=8 \
    RUN pip install \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.0" \
        --no-build-isolation --quiet

# ── 4. mamba-ssm from source ────────────────────────────────────────────────
echo
echo "==> Building mamba-ssm from source …"
CUDA_HOME="$CUDA_HOME" MAX_JOBS=8 \
    RUN pip install \
        "git+https://github.com/state-spaces/mamba.git@v2.3.0" \
        --no-build-isolation --quiet

# ── 5. remaining project dependencies ───────────────────────────────────────
echo
echo "==> Installing remaining dependencies …"
RUN pip install --quiet \
    "jax[cuda12_pip]" \
    "stable-baselines3>=2.7.0" \
    "sb3-contrib>=2.6.0" \
    "gymnasium>=0.29.1" \
    "ale-py>=0.10.2" \
    "imitation>=1.0.0" \
    "seals>=0.2.1" \
    "shimmy>=0.2.1" \
    "huggingface-hub>=1.3.0" \
    "git+https://github.com/huggingface/huggingface_sb3.git@main" \
    "datasets>=4.0.0" \
    "mujoco>=3.3.5" \
    "cvxpy>=1.6.4" \
    "clarabel>=0.10.0" \
    "ecos>=2.0.14" \
    "osqp>=1.0.3" \
    "scs>=3.2.7.post2" \
    "mosek>=11.0.16" \
    "optuna>=4.4.0" \
    "sacred>=0.8.7" \
    "tensorboard>=2.19.0" \
    "matplotlib>=3.10.1" \
    "opencv-python>=4.11.0.86" \
    "imageio>=2.37.0" \
    "imageio-ffmpeg>=0.6.0" \
    "moviepy>=2.2.1" \
    "scikit-learn>=1.7.1" \
    "joblib>=1.4.2" \
    "protobuf>=6.30.2" \
    "requests>=2.32.4" \
    "rich>=14.0.0" \
    "tqdm" \
    "symengine>=0.14.1" \
    "daceypy>=1.3.0"

# ── 6. install the package itself in editable mode ──────────────────────────
echo
echo "==> Installing metarl-iccbf in editable mode …"
RUN pip install -e "$SCRIPT_DIR" --no-deps --quiet

# ── verify ──────────────────────────────────────────────────────────────────
echo
echo "==> Verifying key imports …"
RUN python -c "
import torch, mamba_ssm, stable_baselines3 as sb3, gymnasium, jax
print('  torch      :', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('  mamba_ssm  :', mamba_ssm.__version__)
print('  sb3        :', sb3.__version__)
print('  gymnasium  :', gymnasium.__version__)
print('  jax        :', jax.__version__)
import metarl_iccbf
print('  metarl_iccbf: OK')
"

echo
echo "==> Done!  Activate with:"
echo "    conda activate ${ENV_NAME}"
