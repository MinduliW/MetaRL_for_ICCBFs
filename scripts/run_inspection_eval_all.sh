#!/usr/bin/env bash
# Run parallel evaluation for all inspection models (500 episodes).
# Results are saved to outputs/eval/inspection/.
# Usage: bash scripts/run_inspection_eval_all.sh [--num-workers N] [--num-chunks C] [--skip-existing] [--force]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/mnt/data/aposadasn/miniconda3/envs/iccbf/bin/python}"
ICS_PATH="${REPO_ROOT}/outputs/data/inspection/inspection_legacy_500_episode_bank.npz"
OUT_DIR="${REPO_ROOT}/outputs/eval/inspection"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_CHUNKS="${NUM_CHUNKS:-16}"
SKIP_EXISTING=false

ADVERSARIAL=false

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
    --num-chunks)     NUM_CHUNKS="$2";  shift 2 ;;
    --skip-existing)  SKIP_EXISTING=true; shift ;;
    --force)          SKIP_EXISTING=false; shift ;;
    --adversarial)    ADVERSARIAL=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "${ADVERSARIAL}" == true ]]; then
  OUT_DIR="${REPO_ROOT}/outputs/eval/inspection_adversarial"
fi

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
  local adv_flag=""
  if [[ "${ADVERSARIAL}" == true ]]; then
    adv_flag="--adversarial"
  fi

  "${PYTHON}" -m metarl_iccbf.inspection.evaluation.cli \
    --parallel \
    --model-path "${model_path}" \
    --policy-type "${policy_type}" \
    --algo "${algo}" \
    --env-type iccbf \
    --ics-path "${ICS_PATH}" \
    --out-dir "${OUT_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --num-chunks "${NUM_CHUNKS}" \
    ${adv_flag}
}

if [[ "${ADVERSARIAL}" == true ]]; then
  # LSTM + PPO
  run_eval "LSTM+PPO" \
    "${REPO_ROOT}/outputs/inspection/LSTMTunedICCBF_Inspection_20260523_114917_Adversarial/best_model.zip" \
    "LSTM" "ppo"

  # GRU + PPO
  run_eval "GRU+PPO" \
    "${REPO_ROOT}/outputs/inspection/GRUTunedICCBF_Inspection_20260324_083604_Adversarial/best_model.zip" \
    "GRU" "ppo"

  # Mamba2 + PPO
  run_eval "Mamba2+PPO" \
    "${REPO_ROOT}/outputs/inspection/Mamba2TunedICCBF_Inspection_20260606_154851_Adversarial/best_model.zip" \
    "MAMBA" "ppo"

else
  # LSTM + PPO
  run_eval "LSTM+PPO" \
    "${REPO_ROOT}/outputs/inspection/LSTMTunedICCBF_Inspection_20260427_203555/best_model.zip" \
    "LSTM" "ppo"

  # LSTM + SAC
  run_eval "LSTM+SAC" \
    "${REPO_ROOT}/outputs/inspection/LSTMSAC_Inspection_20260318_111857/best_model.zip" \
    "RNN" "sac"

  # GRU + PPO
  run_eval "GRU+PPO" \
    "${REPO_ROOT}/outputs/inspection/GRUTunedICCBF_Inspection_20260312_122626/best_model.zip" \
    "GRU" "ppo"

  # GRU + SAC
  run_eval "GRU+SAC" \
    "${REPO_ROOT}/outputs/inspection/GRUSAC_Inspection_20260315_114414/best_model.zip" \
    "GRU" "sac"

  # Mamba2 + PPO
  run_eval "Mamba2+PPO" \
    "${REPO_ROOT}/outputs/inspection/Mamba2TunedICCBF_Inspection_20260304_145728/best_model.zip" \
    "MAMBA" "ppo"

  # Mamba2 + SAC
  run_eval "Mamba2+SAC" \
    "${REPO_ROOT}/outputs/inspection/Mamba2SAC_Inspection_20260313_181827/best_model.zip" \
    "MAMBA" "sac"
fi

echo ""
echo "All evaluations complete. Results in: ${OUT_DIR}"
