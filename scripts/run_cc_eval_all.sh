#!/usr/bin/env bash
# Run parallel evaluation for all cruise control models.
# Results are saved to outputs/eval/cruise_control/.
# Usage: bash scripts/run_cc_eval_all.sh [--num-workers N] [--num-chunks C] [--skip-existing] [--force]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
ICS_PATH="${REPO_ROOT}/outputs/data/cruise_control/cruise_control_episode_bank.npz"
OUT_DIR="${REPO_ROOT}/outputs/eval/cruise_control"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_CHUNKS="${NUM_CHUNKS:-32}"
SKIP_EXISTING=false

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
    --num-chunks)     NUM_CHUNKS="$2";  shift 2 ;;
    --skip-existing)  SKIP_EXISTING=true; shift ;;
    --force)          SKIP_EXISTING=false; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}"

run_eval() {
  local label="$1"
  local model_path="$2"
  local policy_type="$3"
  local algo="$4"
  local model_name
  model_name="$(basename "$(dirname "${model_path}")")"

  if [[ "${SKIP_EXISTING}" == true ]]; then
    local existing
    existing="$(find "${OUT_DIR}" -maxdepth 1 -name "*${model_name}*.mat" 2>/dev/null | head -1)"
    if [[ -n "${existing}" ]]; then
      echo ""
      echo "=== Skipping: ${label} (found existing: $(basename "${existing}")) ==="
      return 0
    fi
  fi

  echo ""
  echo "=== Evaluating: ${label} ==="
  echo "    model: ${model_path}"
  "${PYTHON}" -m metarl_iccbf.cruise_control.evaluation.cli \
    --parallel \
    --model-path "${model_path}" \
    --policy-type "${policy_type}" \
    --algo "${algo}" \
    --env-type iccbf \
    --ics-path "${ICS_PATH}" \
    --out-dir "${OUT_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --num-chunks "${NUM_CHUNKS}"
}

# LSTM + PPO
run_eval "LSTM+PPO" \
  "${REPO_ROOT}/outputs/cruise_control/LSTMTunedICCBF_CruiseControl_20260320_102749/best_model.zip" \
  "LSTM" "ppo"

# LSTM + SAC
run_eval "LSTM+SAC" \
  "${REPO_ROOT}/outputs/cruise_control/LSTMSAC_CruiseControl_20260315_174623/best_model.zip" \
  "RNN" "sac"

# GRU + PPO
run_eval "GRU+PPO" \
  "${REPO_ROOT}/outputs/cruise_control/GRUTunedICCBF_CruiseControl_20260305_125057/best_model.zip" \
  "GRU" "ppo"

# GRU + SAC
run_eval "GRU+SAC" \
  "${REPO_ROOT}/outputs/cruise_control/GRUSAC_CruiseControl_20260314_171131/best_model.zip" \
  "GRU" "sac"

# Mamba2 + PPO
run_eval "Mamba2+PPO" \
  "${REPO_ROOT}/outputs/cruise_control/Mamba2TunedICCBF_CruiseControl_20260225_084139/best_model.zip" \
  "MAMBA" "ppo"

# Mamba2 + SAC
run_eval "Mamba2+SAC" \
  "${REPO_ROOT}/outputs/cruise_control/Mamba2SAC_CruiseControl_20260313_135132/best_model.zip" \
  "MAMBA" "sac"

echo ""
echo "All evaluations complete. Results in: ${OUT_DIR}"
