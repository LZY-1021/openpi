#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

policy_config=${1}
checkpoint_dir=${2}
task_list_file=${3}
split=${4:-pretrain}
result_dir=${5:-${checkpoint_dir}_baseline_eval}
gpu_id=${6:-0}
port=${7:-8000}
num_trials=${8:-50}
seed=${9:-7}

if [ -z "${policy_config}" ] || [ -z "${checkpoint_dir}" ] || [ -z "${task_list_file}" ]; then
    echo "Usage: bash examples/robocasa/eval_many_baseline.sh <policy_config> <checkpoint_dir> <task_list_file> [split] [result_dir] [gpu_id] [port] [num_trials] [seed]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PI0_MLP_REUSE=0
export PI05_DENOISE_KV_MODE=fresh

mkdir -p "${result_dir}"

cd "${REPO_ROOT}"

echo -e "\033[33mStarting RoboCasa pi05 baseline server on port ${port}, gpu=${gpu_id}\033[0m"
python scripts/serve_policy.py \
    --port="${port}" policy:checkpoint \
    --policy.config="${policy_config}" \
    --policy.dir="${checkpoint_dir}" &
server_pid=$!

cleanup() {
    if kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}"
        wait "${server_pid}" 2>/dev/null
    fi
}
trap cleanup EXIT

sleep "${ROBOCASA_SERVER_WAIT_SECONDS:-20}"

echo -e "\033[33mRunning RoboCasa baseline eval: split=${split}, num_trials=${num_trials}, tasks=${task_list_file}\033[0m"
python examples/robocasa/main.py \
    --args.port "${port}" \
    --args.task_list_file "${task_list_file}" \
    --args.split "${split}" \
    --args.log_dir "${result_dir}" \
    --args.num_trials "${num_trials}" \
    --args.seed "${seed}"

python examples/robocasa/summarize_task_list_stats.py \
    --result_dir "${result_dir}" \
    --split "${split}" \
    --task_list_file "${task_list_file}" || true
