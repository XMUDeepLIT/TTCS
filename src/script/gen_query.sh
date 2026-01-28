#!/usr/bin/env bash
set -euo pipefail

# Import process cleanup library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# Use environment variables set by run_with_gpus.sh

# Parameter validation
exp_name=$1
num_samples=$2
challenger_model_path=$3
challenger_path_dir=$4
solver_path_dir=$5
gen_question_func=$6
hybrid_data=$7
train_file=$8
real_data_ratio=$9

if [ -z "$exp_name" ] || [ -z "$num_samples" ] || [ -z "$challenger_model_path" ] || [ -z "$challenger_path_dir" ] || [ -z "$solver_path_dir" ]; then
    echo "Error: All parameters cannot be empty"
    exit 1
fi

if [ ! -d "$challenger_model_path" ]; then
    echo "Error: challenger_model_path does not exist: $challenger_model_path"
    exit 1
fi

storage_path=${challenger_path_dir}/${exp_name}/gen_data
save_path_dir=${solver_path_dir}/${exp_name}

export VLLM_DISABLE_COMPILE_CACHE=1

# Parse GPU list from environment variable
IFS=',' read -ra GPU_ARRAY <<< "$TTCS_GEN_QUERY_GPUS"

echo "Starting query generation..."
echo "  Experiment: $exp_name"
echo "  Num samples: $num_samples"
echo "  Storage path: $storage_path"
echo "  Using GPU: $TTCS_GEN_QUERY_GPUS (total ${#GPU_ARRAY[@]})"
echo " ========================================== train_file: $train_file =========================================="
# Launch query generation processes
GEN_PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$i]}"
    suffix="$i"
    
    echo "Launching query generation process: GPU=${gpu}, suffix=${suffix}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m ${TTCS_CODE_MODULE}.challenger_generate_query \
        --model "$challenger_model_path" \
        --suffix "$suffix" \
        --num_samples "$num_samples" \
        --storage_path="$storage_path" \
        --get_prompts_func="$gen_question_func" \
        --train_file="$train_file" &
    GEN_PIDS+=($!)
done

echo "Waiting for all query generation processes... (total ${#GEN_PIDS[@]})"
for pid in "${GEN_PIDS[@]}"; do
    wait "$pid"
done

echo "Query generation completed, starting data merge..."
sleep 5
if [ "$hybrid_data" == "True" ]; then
    python -m ${TTCS_CODE_MODULE}.data_merge --data_path_dir="$storage_path" --save_path_dir="$save_path_dir" --exp_name="$exp_name" --hybrid_data 
else
    python -m ${TTCS_CODE_MODULE}.data_merge --data_path_dir="$storage_path" --save_path_dir="$save_path_dir" --exp_name="$exp_name" 
fi
echo "Data merge completed"
