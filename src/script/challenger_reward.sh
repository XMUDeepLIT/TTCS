#!/usr/bin/env bash
set -euo pipefail

# Import process cleanup library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# Use environment variables set by run_with_gpus.sh

solver_model_path=$1

# Validate parameters
if [ -z "$solver_model_path" ]; then
    echo "Error: solver_model_path cannot be empty"
    exit 1
fi

if [ ! -d "$solver_model_path" ]; then
    echo "Error: solver_model_path does not exist: $solver_model_path"
    exit 1
fi

# Parse GPU and port lists from environment variables
IFS=',' read -ra GPU_ARRAY <<< "$TTCS_REWARD_GPUS"
IFS=',' read -ra PORT_ARRAY <<< "$TTCS_REWARD_PORTS"

echo "[Reward Server] GPU Config: ${TTCS_REWARD_GPUS}"
echo "[Reward Server] Port Config: ${TTCS_REWARD_PORTS}"

export VLLM_DISABLE_COMPILE_CACHE=1
cd ${TTCS_WORKING_DIR}/
echo "Starting vLLM reward server..."

# Launch multiple reward servers
for i in "${!PORT_ARRAY[@]}"; do
    port="${PORT_ARRAY[$i]}"
    gpu_idx=$((i % ${#GPU_ARRAY[@]}))
    gpu="${GPU_ARRAY[$gpu_idx]}"
    
    echo "Launching Reward Server: GPU=${gpu}, Port=${port}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m ${TTCS_CODE_MODULE}.start_vllm_server --port ${port} --model_path $solver_model_path &
done
