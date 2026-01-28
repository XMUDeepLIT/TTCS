#!/usr/bin/env bash
# =============================================================================
# run_with_gpus.sh - GPU-aware launcher for TTCS training
# =============================================================================
#
# Usage:
#   ./run_with_gpus.sh <n_gpus>
#   ./run_with_gpus.sh 4      # Use 4 GPUs
#   ./run_with_gpus.sh 8      # Use 8 GPUs
#
# Features:
#   1. Detect current server GPU status (memory usage, utilization)
#   2. Wait until idle GPU count meets n_gpus
#   3. Select idle GPUs and set environment variables
#   4. Call main.sh to start training
#
# Environment Variables Output:
#   TTCS_N_GPUS           - Total GPU count
#   TTCS_GPU_IDS          - Selected GPU ID list (comma separated)
#   TTCS_CHALLENGER_GPUS  - GPUs for Challenger training
#   TTCS_REWARD_GPUS      - GPUs for Reward Server
#   TTCS_SOLVER_GPUS      - GPUs for Solver training
#   TTCS_GEN_QUERY_GPUS   - GPUs for query generation
#   TTCS_REWARD_PORTS     - Reward Server port list
#   TTCS_REWARD_BATTCS_PORT - Reward Server base port
#
# =============================================================================
# sleep 7200
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration Parameters
# =============================================================================

# GPU idle threshold
GPU_MEMORY_THRESHOLD_MB="${GPU_MEMORY_THRESHOLD_MB:-4000}"      # Memory usage below this is idle (MB)
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-5}"                  # Utilization below this is idle (%)

# Polling configuration
POLL_INTERVAL="${POLL_INTERVAL:-1800}"                            # Check interval (seconds)
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-48}"                          # Max wait time (hours)

# Reward Server configuration
REWARD_BATTCS_PORT="${TTCS_REWARD_BATTCS_PORT:-5000}"                 # Reward Server base port

# =============================================================================
# Path Configuration - Managed via environment variables
# =============================================================================

# Base directory (can be overridden via environment variable)
TTCS_BATTCS_DIR="${TTCS_BATTCS_DIR:-/base_path}"
TTCS_PROJECT_NAME="${TTCS_PROJECT_NAME:-TTCS}"
TTCS_CODE_MODULE="${TTCS_CODE_MODULE:-src}"

# Derived paths
TTCS_WORKING_DIR="${TTCS_BATTCS_DIR}/${TTCS_PROJECT_NAME}"
TTCS_MODEL_DIR="${TTCS_MODEL_DIR:-/model_path}"
TTCS_DATA_DIR="${TTCS_DATA_DIR:-${TTCS_BATTCS_DIR}/data}"
TTCS_SAVED_RESULTS_DIR="${TTCS_SAVED_RESULTS_DIR:-${TTCS_BATTCS_DIR}/saved_results}"
TTCS_CHALLENGER_DIR="${TTCS_SAVED_RESULTS_DIR}/Synthesizer_ttrl"
TTCS_SOLVER_DIR="${TTCS_SAVED_RESULTS_DIR}/Solver_ttrl"
TTCS_TENSORBOARD_DIR="${TTCS_SAVED_RESULTS_DIR}/tensorboard_log"
TTCS_PROMPT_DIR="${TTCS_WORKING_DIR}/${TTCS_CODE_MODULE}"
mkdir -p ${TTCS_CHALLENGER_DIR} ${TTCS_SOLVER_DIR} ${TTCS_TENSORBOARD_DIR} 
# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo "=============================================="
    echo "  Self-evolving-Agent GPU Launcher"
    echo "=============================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

print_usage() {
    echo "Usage: $0 <n_gpus>"
    echo ""
    echo "Arguments:"
    echo "  n_gpus    Number of GPUs needed (4 or 8)"
    echo ""
    echo "Examples:"
    echo "  $0 4      # Use 4 GPUs"
    echo "  $0 8      # Use 8 GPUs"
    echo ""
    echo "Environment Variables:"
    echo "  GPU_MEMORY_THRESHOLD_MB   Memory idle threshold (default: 500 MB)"
    echo "  GPU_UTIL_THRESHOLD        Utilization idle threshold (default: 10%)"
    echo "  POLL_INTERVAL             Poll interval (default: 30 seconds)"
    echo "  MAX_WAIT_HOURS            Max wait time (default: 48 hours)"
    echo "  TTCS_REWARD_BATTCS_PORT   Reward Server base port (default: 5000)"
}

log_info() {
    echo "[INFO] $(date '+%H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%H:%M:%S') - $1" >&2
}

# =============================================================================
# GPU Detection Functions
# =============================================================================

get_gpu_info() {
    # Get all GPU info
    # Output format: GPU_ID,MEMORY_USED_MB,MEMORY_TOTAL_MB,GPU_UTIL%
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

get_idle_gpus() {
    # Get idle GPU list
    # Returns: space-separated GPU ID list
    local mem_threshold=$1
    local util_threshold=$2
    local idle_gpus=""
    
    while IFS=',' read -r gpu_id mem_used mem_total gpu_util; do
        if [ -z "$gpu_id" ]; then
            continue
        fi
        
        # Check memory usage and utilization
        if [ "$mem_used" -lt "$mem_threshold" ] && [ "$gpu_util" -lt "$util_threshold" ]; then
            if [ -z "$idle_gpus" ]; then
                idle_gpus="$gpu_id"
            else
                idle_gpus="$idle_gpus $gpu_id"
            fi
        fi
    done < <(get_gpu_info)
    
    echo "$idle_gpus"
}

count_idle_gpus() {
    local idle_gpus
    idle_gpus=$(get_idle_gpus "$GPU_MEMORY_THRESHOLD_MB" "$GPU_UTIL_THRESHOLD")
    if [ -z "$idle_gpus" ]; then
        echo "0"
    else
        echo "$idle_gpus" | wc -w | tr -d ' '
    fi
}

select_gpus() {
    # Select specified number of idle GPUs
    local n_needed=$1
    local idle_gpus
    idle_gpus=$(get_idle_gpus "$GPU_MEMORY_THRESHOLD_MB" "$GPU_UTIL_THRESHOLD")
    
    # Select first n_needed GPUs
    echo "$idle_gpus" | tr ' ' '\n' | head -n "$n_needed" | tr '\n' ' ' | sed 's/ $//'
}

print_gpu_status() {
    echo "Current GPU Status:"
    echo "----------------------------------------"
    printf "%-5s %-12s %-12s %-10s %-8s\n" "GPU" "Mem Used" "Mem Total" "Util" "Status"
    echo "----------------------------------------"
    
    while IFS=',' read -r gpu_id mem_used mem_total gpu_util; do
        if [ -z "$gpu_id" ]; then
            continue
        fi
        
        local status="Busy"
        if [ "$mem_used" -lt "$GPU_MEMORY_THRESHOLD_MB" ] && [ "$gpu_util" -lt "$GPU_UTIL_THRESHOLD" ]; then
            status="Idle"
        fi
        
        printf "%-5s %-12s %-12s %-10s %-8s\n" \
            "$gpu_id" "${mem_used}MB" "${mem_total}MB" "${gpu_util}%" "$status"
    done < <(get_gpu_info)
    
    echo "----------------------------------------"
    echo "Idle threshold: Memory < ${GPU_MEMORY_THRESHOLD_MB}MB, Utilization < ${GPU_UTIL_THRESHOLD}%"
}

# =============================================================================
# Wait for GPU Function
# =============================================================================

wait_for_gpus() {
    local n_needed=$1
    local max_iterations=$((MAX_WAIT_HOURS * 3600 / POLL_INTERVAL))
    local iteration=0
    
    log_info "Waiting for $n_needed idle GPUs..."
    log_info "Check interval: ${POLL_INTERVAL}s, Max wait: ${MAX_WAIT_HOURS}h"
    echo ""
    
    while true; do
        local n_idle
        n_idle=$(count_idle_gpus)
        
        log_info "Current idle GPUs: $n_idle / Needed: $n_needed"
        
        if [ "$n_idle" -ge "$n_needed" ]; then
            log_info "GPU resources meet requirements!"
            return 0
        fi
        
        iteration=$((iteration + 1))
        if [ "$iteration" -ge "$max_iterations" ]; then
            log_error "Timeout (${MAX_WAIT_HOURS}h), cannot get enough GPUs"
            return 1
        fi
        
        # Show detailed status every 10 iterations
        if [ $((iteration % 10)) -eq 1 ]; then
            echo ""
            print_gpu_status
            echo ""
        fi
        
        log_info "Waiting ${POLL_INTERVAL} seconds before retry..."
        sleep "$POLL_INTERVAL"
    done
}

# =============================================================================
# Environment Variable Setup Function
# =============================================================================

setup_gpu_env() {
    local n_gpus=$1
    local selected_gpus=$2
    
    # Convert space-separated to array
    local gpu_array=($selected_gpus)
    local n_selected=${#gpu_array[@]}
    
    if [ "$n_selected" -lt "$n_gpus" ]; then
        log_error "Not enough GPUs selected: $n_selected < $n_gpus"
        return 1
    fi
    
    # Calculate allocation
    local half=$((n_gpus / 2))
    
    # Challenger GPUs (first half)
    local challenger_gpus=""
    for ((i=0; i<half; i++)); do
        if [ -z "$challenger_gpus" ]; then
            challenger_gpus="${gpu_array[$i]}"
        else
            challenger_gpus="${challenger_gpus},${gpu_array[$i]}"
        fi
    done
    
    # Reward GPUs (second half)
    local reward_gpus=""
    for ((i=half; i<n_gpus; i++)); do
        if [ -z "$reward_gpus" ]; then
            reward_gpus="${gpu_array[$i]}"
        else
            reward_gpus="${reward_gpus},${gpu_array[$i]}"
        fi
    done
    
    # Solver GPUs (all)
    local solver_gpus=""
    for ((i=0; i<n_gpus; i++)); do
        if [ -z "$solver_gpus" ]; then
            solver_gpus="${gpu_array[$i]}"
        else
            solver_gpus="${solver_gpus},${gpu_array[$i]}"
        fi
    done
    
    # Gen Query GPUs (all)
    local gen_query_gpus="$solver_gpus"
    
    # Reward Ports (same count as reward GPUs)
    local n_reward_servers=$half
    local reward_ports=""
    for ((i=0; i<n_reward_servers; i++)); do
        local port=$((REWARD_BATTCS_PORT + i))
        if [ -z "$reward_ports" ]; then
            reward_ports="$port"
        else
            reward_ports="${reward_ports},${port}"
        fi
    done
    
    # Set environment variables
    export TTCS_N_GPUS="$n_gpus"
    export TTCS_GPU_IDS="$solver_gpus"
    export TTCS_CHALLENGER_GPUS="$challenger_gpus"
    export TTCS_REWARD_GPUS="$reward_gpus"
    export TTCS_SOLVER_GPUS="$solver_gpus"
    export TTCS_GEN_QUERY_GPUS="$gen_query_gpus"
    export TTCS_REWARD_PORTS="$reward_ports"
    export TTCS_REWARD_BATTCS_PORT="$REWARD_BATTCS_PORT"
    export TTCS_N_CHALLENGER_GPUS="$half"
    export TTCS_N_REWARD_GPUS="$half"
    export TTCS_N_SOLVER_GPUS="$n_gpus"
    export TTCS_N_REWARD_SERVERS="$n_reward_servers"
    
    # Export path environment variables
    export TTCS_BATTCS_DIR
    export TTCS_PROJECT_NAME
    export TTCS_CODE_MODULE
    export TTCS_WORKING_DIR
    export TTCS_MODEL_DIR
    export TTCS_DATA_DIR
    export TTCS_SAVED_RESULTS_DIR
    export TTCS_CHALLENGER_DIR
    export TTCS_SOLVER_DIR
    export TTCS_TENSORBOARD_DIR
    export TTCS_PROMPT_DIR
    
    # Print configuration
    echo ""
    echo "GPU Allocation Config:"
    echo "=============================================="
    echo "Total GPUs:        $TTCS_N_GPUS"
    echo "Selected GPU IDs:  $TTCS_GPU_IDS"
    echo ""
    echo "Challenger GPUs:   $TTCS_CHALLENGER_GPUS (total $TTCS_N_CHALLENGER_GPUS)"
    echo "Reward GPUs:       $TTCS_REWARD_GPUS (total $TTCS_N_REWARD_GPUS)"
    echo "Solver GPUs:       $TTCS_SOLVER_GPUS (total $TTCS_N_SOLVER_GPUS)"
    echo "Gen Query GPUs:    $TTCS_GEN_QUERY_GPUS"
    echo ""
    echo "Reward Ports:      $TTCS_REWARD_PORTS"
    echo "Reward Base Port:  $TTCS_REWARD_BATTCS_PORT"
    echo "=============================================="
    echo ""
    echo "Path Config:"
    echo "=============================================="
    echo "Base Dir:          $TTCS_BATTCS_DIR"
    echo "Project Name:      $TTCS_PROJECT_NAME"
    echo "Code Module:       $TTCS_CODE_MODULE"
    echo "Working Dir:       $TTCS_WORKING_DIR"
    echo "Model Dir:         $TTCS_MODEL_DIR"
    echo "Data Dir:          $TTCS_DATA_DIR"
    echo "Results Dir:       $TTCS_SAVED_RESULTS_DIR"
    echo "Prompt Dir:        $TTCS_PROMPT_DIR"
    echo "=============================================="
    echo ""
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    print_banner
    
    # Check arguments
    if [ $# -lt 1 ]; then
        print_usage
        exit 1
    fi
    
    local n_gpus=$1
    
    # Validate arguments
    if ! [[ "$n_gpus" =~ ^[0-9]+$ ]]; then
        log_error "n_gpus must be a positive integer: $n_gpus"
        print_usage
        exit 1
    fi
    
    if [ "$n_gpus" -lt 2 ]; then
        log_error "n_gpus must be at least 2 (needed for challenger and reward)"
        exit 1
    fi
    
    if [ $((n_gpus % 2)) -ne 0 ]; then
        log_error "n_gpus must be even (split between challenger and reward): $n_gpus"
        exit 1
    fi
    
    log_info "Requested GPU count: $n_gpus"
    echo ""
    
    # Show current status
    print_gpu_status
    echo ""
    
    # Wait for enough GPUs
    if ! wait_for_gpus "$n_gpus"; then
        log_error "Cannot get enough GPUs, exiting"
        exit 1
    fi
    
    # Select GPUs
    local selected_gpus
    selected_gpus=$(select_gpus "$n_gpus")
    log_info "Selected GPUs: $selected_gpus"
    
    # Set environment variables
    setup_gpu_env "$n_gpus" "$selected_gpus"
    
    # Call main.sh
    log_info "Starting training..."
    echo ""
    echo "=============================================="
    echo "  Starting main.sh"
    echo "=============================================="
    echo ""
    
    # Execute main.sh
    bash "${SCRIPT_DIR}/main.sh"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "Training completed!"
    else
        log_error "Training failed, exit code: $exit_code"
    fi
    
    exit $exit_code
}

# =============================================================================
# Entry Point
# =============================================================================

main "$@"
