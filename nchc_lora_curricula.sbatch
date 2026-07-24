#!/usr/bin/env bash
# Submit all five experiments with:
#   sbatch -A YOUR_NCHC_PROJECT_ID nchc_lora_curricula.sbatch
#
# Override settings without editing this file, for example:
#   sbatch -A YOUR_ID --export=ALL,MODEL_NAME=Qwen/Qwen3-1.7B,WANDB_MODE=online nchc_lora_curricula.sbatch

#SBATCH --job-name=lora-curricula
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%5
# Slurm opens these files before the script starts, so keep them in the
# submission directory rather than in a directory that may not exist yet.
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --error=slurm-%x_%A_%a.err

set -Eeuo pipefail

# Submit this file from the repository root, or export PROJECT_DIR explicitly.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL_NAME="${MODEL_NAME:-nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1}"
TRAIN_DATASET="${TRAIN_DATASET:-${PROJECT_DIR}/dataset/training_rows.flat.jsonl}"
TEST_DATASET="${TEST_DATASET:-${PROJECT_DIR}/dataset/curriculum_test_set.flat.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/nchc_runs}"
WANDB_MODE="${WANDB_MODE:-offline}"

# Shared training settings. These can all be overridden through sbatch --export.
EPOCHS="${EPOCHS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
SEED="${SEED:-42}"

# Shared evaluation settings.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_ROOT}"
cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not available. Load your NCHC Python/uv environment before sbatch." >&2
    exit 1
fi

for required_file in \
    "${TRAIN_DATASET}" \
    "${TEST_DATASET}" \
    "${PROJECT_DIR}/difficulty.json" \
    "${PROJECT_DIR}/time_difficulty.json" \
    "${PROJECT_DIR}/length_difficulty.josn" \
    "${PROJECT_DIR}/budget_difficulty.json"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

case "${SLURM_ARRAY_TASK_ID:-}" in
    0)
        EXPERIMENT="llm_complexity"
        TRAIN_MODE="curriculum"
        DIFFICULTY_PATH="${PROJECT_DIR}/difficulty.json"
        ;;
    1)
        EXPERIMENT="time_curriculum"
        TRAIN_MODE="curriculum"
        DIFFICULTY_PATH="${PROJECT_DIR}/time_difficulty.json"
        ;;
    2)
        EXPERIMENT="length_curriculum"
        TRAIN_MODE="curriculum"
        # The repository currently spells this filename with a .josn suffix.
        DIFFICULTY_PATH="${PROJECT_DIR}/length_difficulty.josn"
        ;;
    3)
        EXPERIMENT="budget_curriculum"
        TRAIN_MODE="curriculum"
        DIFFICULTY_PATH="${PROJECT_DIR}/budget_difficulty.json"
        ;;
    4)
        EXPERIMENT="random_order"
        TRAIN_MODE="random"
        DIFFICULTY_PATH=""
        ;;
    *)
        echo "ERROR: SLURM_ARRAY_TASK_ID must be in 0..4." >&2
        exit 1
        ;;
esac

RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT}"
ADAPTER_DIR="${RUN_DIR}/adapter"
EVAL_LOG_DIR="${RUN_DIR}/eval_logs"
MERGE_TMP_DIR="${SLURM_TMPDIR:-/tmp}/${USER}/lora_curricula_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${ADAPTER_DIR}" "${EVAL_LOG_DIR}" "${MERGE_TMP_DIR}"

echo "Experiment:       ${EXPERIMENT}"
echo "Project:          ${PROJECT_DIR}"
echo "Model:            ${MODEL_NAME}"
echo "Training data:    ${TRAIN_DATASET}"
echo "Test data:        ${TEST_DATASET}"
echo "Adapter output:   ${ADAPTER_DIR}"
echo "Difficulty file:  ${DIFFICULTY_PATH:-input-length fallback / unused}"
echo "CUDA devices:     ${CUDA_VISIBLE_DEVICES:-assigned by Slurm}"
nvidia-smi || true

train_args=(
    uv run --frozen train.py
    --dataset_path "${TRAIN_DATASET}"
    --model_name "${MODEL_NAME}"
    --output_dir "${ADAPTER_DIR}"
    --gpu_id 0
    --train_mode "${TRAIN_MODE}"
    --difficulty_path "${DIFFICULTY_PATH}"
    --epochs "${EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum "${GRAD_ACCUM}"
    --max_seq_len "${MAX_SEQ_LEN}"
    --seed "${SEED}"
    --wandb_mode "${WANDB_MODE}"
    --wandb_name "nchc_${EXPERIMENT}"
)

echo "Starting LoRA training at $(date --iso-8601=seconds)"
"${train_args[@]}"

# train.py and eval.py run in separate processes, so training GPU memory is
# released before eval.py temporarily merges the adapter and starts vLLM.
echo "Starting evaluation at $(date --iso-8601=seconds)"
uv run --frozen eval.py \
    --dataset_path "${TEST_DATASET}" \
    --model_name "${MODEL_NAME}" \
    --lora_checkpoint "${ADAPTER_DIR}" \
    --merged_model_dir "${MERGE_TMP_DIR}" \
    --gpu_id 0 \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_mem_util "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --log_dir "${EVAL_LOG_DIR}" \
    --run_name "${EXPERIMENT}"

echo "Completed ${EXPERIMENT} at $(date --iso-8601=seconds)"
echo "Evaluation logs: ${EVAL_LOG_DIR}"
