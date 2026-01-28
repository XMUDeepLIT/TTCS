#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1

# Use environment variables set by run_with_gpus.sh
save_path_dir=${TTCS_SAVED_RESULTS_DIR}/evaluation
eval_path=${TTCS_WORKING_DIR}/evaluation
cd ${eval_path}

prefix=$1
temperature=$2
eval_step=$3
num_iter=$4
base_model_name=$5

# Base model evaluation results saved in global directory (evaluated only once)
global_base_model_eval_dir=${save_path_dir}
base_model_eval_results_dir=${global_base_model_eval_dir}/${base_model_name}

eval_saved_path_dir=${save_path_dir}/${prefix}_step${eval_step}_temperature${temperature}
tb_path_dir=${TTCS_SAVED_RESULTS_DIR}/eval_tb_log
mkdir -p ${eval_saved_path_dir} ${tb_path_dir} ${TTCS_WORKING_DIR}/eval_logs ${global_base_model_eval_dir}

model_list=()
BASE_MODEL_NEEDS_EVAL="false"
base_model_path="${TTCS_MODEL_DIR}/${base_model_name}"

# Check if base model needs evaluation
if [ ! -d "${base_model_path}" ]; then
    echo "Warning: Base model does not exist: ${base_model_path}, skipping..."
elif [ -d "${base_model_eval_results_dir}" ]; then
    # Check if evaluation results are complete (check key result files)
    if [ -f "${base_model_eval_results_dir}/temp_data_Overall_results.jsonl" ] && \
       [ -f "${base_model_eval_results_dir}/greedy_data_Overall_results.jsonl" ]; then
        echo "Base model ${base_model_name} evaluation results already exist: ${base_model_eval_results_dir}, skipping."
    else
        echo "Base model ${base_model_name} evaluation results incomplete, need to re-evaluate"
        BASE_MODEL_NEEDS_EVAL="true"
        model_list+=("${base_model_name}")
    fi
else
    echo "Base model ${base_model_name} needs evaluation, results will be saved to: ${base_model_eval_results_dir}"
    BASE_MODEL_NEEDS_EVAL="true"
    model_list+=("${base_model_name}")
fi

for i in $(seq 1 ${num_iter}); do
  model_list+=("${prefix}-V${i}")
done
echo "Final model_list: ${model_list[@]}"

function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
exec > >(tee -a "${TTCS_WORKING_DIR}/eval_logs/eval_${prefix}-$(now).log") 2>&1
TASKS=(
  "temp_data"
  "greedy_data"
)

# Additional evaluation datasets (run first)
ADDITIONAL_EVAL_DATASETS=(
  "eval_bbeh.py"
  "eval_mmlupro.py"
  "eval_supergpqa.py"
)

# Initialize GPU queue - use all available GPUs
GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
#GPU_QUEUE=(5 6 7)
if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi

echo "Available GPUs: ${GPU_QUEUE[@]} (Total: ${#GPU_QUEUE[@]})"

# Global tracking variables
declare -A gpu_status
declare -A pids
declare -A model_gpu_mapping  # Track which GPUs are assigned to which model
declare -A model_path_mapping  # Track model paths for each GPU
declare -A model_info           # Store model information for pending models
declare -a retry_task_queue=()  # Queue for failed tasks that need retry

# Initialize GPU status
for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status["$gpu_id"]="idle"
done

# Function to cleanup GPU processes and free memory
cleanup_gpu_processes() {
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up GPU processes to free memory..."
    
    # Show current GPU processes before cleanup
    echo "Current GPU processes:"
    nvidia-smi pmon -c 1 2>/dev/null || echo "nvidia-smi pmon not available"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9

    # Find and kill processes using GPU memory
    echo "Finding processes using GPU memory..."
    local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -n "$gpu_pids" ]; then
        echo "Killing GPU processes: $gpu_pids"
        for pid in $gpu_pids; do
            if [ -n "$pid" ] && [ "$pid" != "N/A" ]; then
                echo "Killing process $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        
        # Wait for processes to terminate gracefully
        sleep 3
        
        # Force kill if still running
        for pid in $gpu_pids; do
            if [ -n "$pid" ] && [ "$pid" != "N/A" ]; then
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force killing process $pid"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
    else
        echo "No GPU processes found"
    fi
    
    # Clear GPU memory cache using Python
    echo "Clearing GPU memory cache..."
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print('GPU cache cleared and garbage collected')
else:
    print('CUDA not available')
" 2>/dev/null || echo "Failed to clear GPU cache"
    
    # Wait a moment for memory to be freed
    sleep 2
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU cleanup completed"
    echo "GPU status after cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
}

# Function to get available GPUs
get_available_gpus() {
    local available_gpus=()
    for gpu_id in "${GPU_QUEUE[@]}"; do
        # Check both script status and actual GPU usage
        if [ "${gpu_status[$gpu_id]}" = "idle" ]; then
            # Check actual GPU memory usage
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            
            # Check GPU utilization
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            # Remove any whitespace from gpu_util
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            # Only consider GPU available if memory usage is low (< 5%) AND GPU utilization is low (< 5%)
            if [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                available_gpus+=("$gpu_id")
            else
                if [ "$memory_percent" -ge 5 ]; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] has high memory usage (${memory_percent}%), skipping..." >&2
                fi
                if [ "$gpu_util" -ge 5 ]; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] has high utilization (${gpu_util}%), skipping..." >&2
                fi
            fi
        fi
    done
    echo "${available_gpus[@]}"
}

# Function to check if a dataset evaluation is already completed
check_dataset_completed() {
    local model_name="$1"
    local eval_script="$2"
    
    # Extract dataset name from eval script (e.g., eval_bbeh.py -> bbeh)
    local dataset_name=$(basename "$eval_script" .py | sed 's/^eval_//')
    
    # Check if final results file exists
    local result_file
    if [ "$model_name" == "$base_model_name" ]; then
        # Base model saves to global directory
        result_file="${base_model_eval_results_dir}/${dataset_name}_final_results.json"
    else
        result_file="${eval_saved_path_dir}/${model_name}/${dataset_name}_final_results.json"
    fi
    
    if [ -f "$result_file" ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Dataset [${dataset_name}] for model [${model_name}] already completed (found: ${result_file})"
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# Function to check if main task (temp_data or greedy_data) is already completed
check_main_task_completed() {
    local model_name="$1"
    local dataset="$2"
    
    # Check if Overall_results.jsonl file exists
    local result_file
    if [ "$model_name" == "$base_model_name" ]; then
        # Base model saves to global directory
        result_file="${base_model_eval_results_dir}/${dataset}_Overall_results.jsonl"
    else
        result_file="${eval_saved_path_dir}/${model_name}/${dataset}_Overall_results.jsonl"
    fi
    
    if [ -f "$result_file" ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Main task [${dataset}] for model [${model_name}] already completed (found: ${result_file})"
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# Function to start an additional evaluation job on one GPU
start_additional_eval_job() {
    local gpu_id="$1"
    local model_path="$2"
    local model_name="$3"
    local eval_script="$4"
    
    # Base model saves to global directory
    local target_save_dir="${eval_saved_path_dir}"
    if [ "$model_name" == "$base_model_name" ]; then
        target_save_dir="${global_base_model_eval_dir}"
    fi
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation [${eval_script}] for model [${model_name}] on GPU [${gpu_id}] (save_dir: ${target_save_dir})"
    
    # Start evaluation script on GPU
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${eval_script}" --model_path "${model_path}" --model_name "${model_name}" --save_path_dir "${target_save_dir}" --data_path_dir "${TTCS_DATA_DIR}" &
    local pid=$!
    
    # Wait a moment to check if process started successfully
    sleep 10
    
    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        # Process started successfully, now mark GPU as busy
        gpu_status["${gpu_id}"]="busy"
        pids["${gpu_id}"]="$pid"
        model_gpu_mapping["${gpu_id}"]="${model_name}"
        model_path_mapping["${gpu_id}"]="${model_path}"
        gpu_dataset["${gpu_id}"]="${eval_script}"
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started evaluation [${eval_script}] for model [${model_name}]"
        return 0
    else
        # Process failed to start, clean up and add to retry queue
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to start evaluation [${eval_script}] for model [${model_name}]"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid" 2>/dev/null || true
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Cleaned up failed process for evaluation [${eval_script}], adding to retry queue"
        retry_task_queue+=("${model_name}|${model_path}|${eval_script}|additional")
        return 1
    fi
}

# Function to start a math task job on one GPU
start_math_task_job() {
    local gpu_id="$1"
    local model_path="$2"
    local model_name="$3"
    local dataset="$4"
    
    # Base model saves to global directory
    local target_save_dir="${eval_saved_path_dir}"
    if [ "$model_name" == "$base_model_name" ]; then
        target_save_dir="${global_base_model_eval_dir}"
    fi
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math task [${dataset}] for model [${model_name}] on GPU [${gpu_id}] (save_dir: ${target_save_dir})"
    
    # Determine n_samples and temperature based on dataset
    local n_samples=32
    local temp=$temperature
    if [ "$dataset" == "${TASKS[1]}" ]; then
        n_samples=1
        temp=0.0
    fi
    
    # Start evaluation script on GPU
    CUDA_VISIBLE_DEVICES="${gpu_id}" python eval_all_math_para.py --model_path "${model_path}" --model_name "${model_name}" --dataset "${dataset}" --save_path_dir "${target_save_dir}" --n_samples "${n_samples}" --temperature "${temp}" --data_path_dir "${TTCS_DATA_DIR}" &
    local pid=$!
    
    # Wait a moment to check if process started successfully
    sleep 10
    
    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        # Process started successfully, now mark GPU as busy
        gpu_status["${gpu_id}"]="busy"
        pids["${gpu_id}"]="$pid"
        model_gpu_mapping["${gpu_id}"]="${model_name}"
        model_path_mapping["${gpu_id}"]="${model_path}"
        gpu_dataset["${gpu_id}"]="${dataset}"
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started math task [${dataset}] for model [${model_name}]"
        return 0
    else
        # Process failed to start, clean up and add to retry queue
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to start math task [${dataset}] for model [${model_name}]"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid" 2>/dev/null || true
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Cleaned up failed process for math task [${dataset}], adding to retry queue"
        retry_task_queue+=("${model_name}|${model_path}|${dataset}|math")
        return 1
    fi
}

# Function to check for completed jobs and free up GPUs
check_completed_jobs() {
    # First pass: detect newly completed GPUs and run post_eval immediately for each completed task
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        # Check if process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            local model_path="${model_path_mapping[$gpu_id]}"
            
            # Check if we've already processed this GPU
            if [ -z "${task_completed[$gpu_id]}" ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished for model [${model_name}] with PID [${pid}]"
                task_completed["$gpu_id"]=1
                
                # Run post_eval immediately for this completed task (only for main TASKS, not additional eval scripts)
                local dataset="${gpu_dataset[$gpu_id]}"
                local task_verified=false
                
                if [[ " ${TASKS[@]} " =~ " ${dataset} " ]]; then
                    local n_samples=32
                    local temp=$temperature
                    if [ "$dataset" == "${TASKS[1]}" ]; then
                        n_samples=1
                        temp=0.0
                    fi
                    
                    # Base model saves to global directory
                    local target_save_dir="${eval_saved_path_dir}"
                    if [ "$model_name" == "$base_model_name" ]; then
                        target_save_dir="${global_base_model_eval_dir}"
                    fi
                    
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running post_eval for [${model_name}] on dataset [${dataset}] with n_samples [${n_samples}] and temperature [${temp}]..."
                    python post_eval.py --save_path_dir "${target_save_dir}" --dataset "${dataset}" --model_name "${model_name}" --n_samples "${n_samples}" --temperature "${temp}"
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Post-processing completed for GPU [${gpu_id}] (model [${model_name}], dataset [${dataset}], n_samples [${n_samples}], temperature [${temp}])"
                    
                    # Verify math task completion
                    if check_main_task_completed "$model_name" "$dataset"; then
                        task_verified=true
                        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✓ Verified: Math task [${dataset}] for model [${model_name}] completed successfully"
                    else
                        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✗ Verification failed: Math task [${dataset}] for model [${model_name}] - adding to retry queue"
                        retry_task_queue+=("${model_name}|${model_path}|${dataset}|math")
                    fi
                else
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Evaluation script [${dataset}] completed for model [${model_name}], verifying..."
                    
                    # Verify additional eval task completion
                    if check_dataset_completed "$model_name" "$dataset"; then
                        task_verified=true
                        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✓ Verified: Evaluation [${dataset}] for model [${model_name}] completed successfully"
                    else
                        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✗ Verification failed: Evaluation [${dataset}] for model [${model_name}] - adding to retry queue"
                        retry_task_queue+=("${model_name}|${model_path}|${dataset}|additional")
                    fi
                fi
                
                # Free this GPU
                free_gpu "$gpu_id"
            fi
        fi
    done
    
    # Second pass: check for stuck processes (original logic)
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        if kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            
            # Process is running, but check if it's actually using GPU resources
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            
            # Check GPU utilization
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            # If GPU memory usage AND utilization are very low for a long time, the process might be stuck
            if [ "$memory_percent" -lt 2 ] && [ "$memory_used" -lt 100 ] && [ "$gpu_util" -lt 5 ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Warning: GPU [${gpu_id}] has very low memory usage (${memory_percent}%) and utilization (${gpu_util}%) for model [${model_name}]"
                # Check if this is a stuck process by looking at CPU usage
                local cpu_usage=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
                if [ "$cpu_usage" = "0.0" ] || [ -z "$cpu_usage" ]; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Detected stuck process PID [${pid}] on GPU [${gpu_id}], terminating..."
                    kill -TERM "$pid" 2>/dev/null || true
                    sleep 2
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid" 2>/dev/null || true
                    fi
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
}

# Array to track which task (0 or 1) has completed for each GPU
declare -A task_completed
# Array to track which dataset each GPU is processing
declare -A gpu_dataset

# Function to properly free up a GPU
free_gpu() {
    local gpu_id="$1"
    local model_name="${model_gpu_mapping[$gpu_id]}"
    
    # Clean up tracking variables
    unset pids["$gpu_id"]
    gpu_status["$gpu_id"]="idle"
    unset model_gpu_mapping["$gpu_id"]
    unset model_path_mapping["$gpu_id"]
    unset task_completed["$gpu_id"]
    unset gpu_dataset["$gpu_id"]
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Freed GPU [${gpu_id}] (was running model [${model_name}])"
}

# Function to detect and clean up zombie processes
cleanup_zombie_processes() {
    # Find all eval_all_math.py processes that might not be tracked
    local orphaned_pids=()
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            local pid=$(echo "$line" | awk '{print $2}')
            local gpu_used=$(echo "$line" | grep -o 'CUDA_VISIBLE_DEVICES=[0-9]*' | cut -d'=' -f2)
            
            # Check if this PID is not in our tracking
            local is_tracked=false
            for tracked_gpu in "${!pids[@]}"; do
                if [ "${pids[$tracked_gpu]}" = "$pid" ]; then
                    is_tracked=true
                    break
                fi
            done
            
            if [ "$is_tracked" = false ] && [ -n "$gpu_used" ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Found orphaned process PID [${pid}] on GPU [${gpu_used}]"
                orphaned_pids+=("$pid")
            fi
        fi
    done < <(ps aux | grep -E "eval_all_math_para.py|eval_bbeh.py|eval_mmlupro.py|eval_supergpqa.py" | grep -v grep)
    
    # Kill orphaned processes
    for pid in "${orphaned_pids[@]}"; do
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Killing orphaned process PID [${pid}]"
        kill -TERM "$pid" 2>/dev/null || true
    done
}

# Function to perform comprehensive GPU state cleanup
comprehensive_cleanup() {
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Performing comprehensive GPU state cleanup..."
    
    # Check all GPUs and their actual usage
    for gpu_id in {0..7}; do
        if nvidia-smi -i "$gpu_id" >/dev/null 2>&1; then
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            
            # Check GPU utilization
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            # If GPU is marked as busy but has very low memory usage AND utilization, it might be stuck
            if [ "${gpu_status[$gpu_id]}" = "busy" ] && [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                local model_name="${model_gpu_mapping[$gpu_id]}"
                local pid="${pids[$gpu_id]}"
                
                # Check if the process is actually still running
                if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                    # Process has finished, but don't free GPU here - let check_completed_jobs handle it
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] process finished for model [${model_name}], will be handled by check_completed_jobs..."
                else
                    # Process is still running but stuck, force cleanup
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] marked as busy but has low memory usage (${memory_percent}%) and utilization (${gpu_util}%), forcing cleanup..."
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
    
    # Clean up any orphaned processes
    cleanup_zombie_processes
}

# Main execution loop
# Clean up GPU processes before starting
#cleanup_gpu_processes

# First, run additional evaluation datasets for all models
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting additional evaluation datasets for ${#model_list[@]} models"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Evaluation datasets: ${ADDITIONAL_EVAL_DATASETS[*]}"

# Build task queue: array of "model_name|model_path|eval_script"
declare -a task_queue=()

# Initialize task queue with all model-dataset combinations
for model_name in "${model_list[@]}"; do
    if [ "$model_name" == "$base_model_name" ]; then
        model_path=${TTCS_MODEL_DIR}/${base_model_name}
    elif [[ $model_name =~ V[0-9]+ ]]; then
        model_path=${TTCS_SOLVER_DIR}/${model_name}/ckpts/global_step_${eval_step}/actor/huggingface
    else
        echo "Warning: Model name [${model_name}] does not match any known pattern"
        echo "Skipping model: $model_name"
        continue
    fi
    
    # Verify model path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path does not exist: $model_path"
        echo "Skipping model: $model_name"
        continue
    fi
    
    # Add each evaluation dataset to the queue
    for eval_script in "${ADDITIONAL_EVAL_DATASETS[@]}"; do
        # Check if already completed before adding to queue
        if check_dataset_completed "$model_name" "$eval_script"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Skipping already completed task: model [${model_name}], dataset [${eval_script}]"
            continue
        fi
        task_queue+=("${model_name}|${model_path}|${eval_script}")
    done
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] General Task queue initialized with ${#task_queue[@]} tasks"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] General Task queue:"
for i in "${!task_queue[@]}"; do
  echo "[$i] ${task_queue[$i]}"
done
# Initialize cleanup counter
cleanup_counter=0
task_index=0

# Process tasks asynchronously
while [ $task_index -lt ${#task_queue[@]} ] || [ ${#pids[@]} -gt 0 ]; do
    # Perform periodic comprehensive cleanup every 10 iterations
    cleanup_counter=$((cleanup_counter + 1))
    if [ $((cleanup_counter % 10)) -eq 0 ]; then
        comprehensive_cleanup
    else
        cleanup_zombie_processes
    fi
    
    check_completed_jobs
    available_gpus=($(get_available_gpus))
    
    # Try to start new tasks if there are available GPUs and pending tasks
    while [ $task_index -lt ${#task_queue[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
        # Get next task from queue
        task="${task_queue[$task_index]}"
        IFS='|' read -r model_name model_path eval_script <<< "$task"
        
        # Double-check if completed (in case it was completed by another process)
        if check_dataset_completed "$model_name" "$eval_script"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Task already completed, skipping: model [${model_name}], dataset [${eval_script}]"
            task_index=$((task_index + 1))
            continue
        fi
        
        # Validate that GPU ID is a number
        if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
            # Start the evaluation job on the first available GPU
            if start_additional_eval_job "${available_gpus[0]}" "$model_path" "$model_name" "$eval_script"; then
                local current_task=$((task_index + 1))
                local total_tasks=${#task_queue[@]}
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started evaluation [${eval_script}] for model [${model_name}] - task ${current_task}/${total_tasks}"
                task_index=$((task_index + 1))
                # Update available GPUs after starting a job
                available_gpus=($(get_available_gpus))
            else
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Failed to start evaluation [${eval_script}] for model [${model_name}], will retry later"
                # Break to wait for GPU availability
                break
            fi
        else
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Invalid GPU ID detected: GPU='${available_gpus[0]}'"
            task_index=$((task_index + 1))
            continue
        fi
    done
    
    # If there are still running jobs or pending tasks, wait a bit
    if [ ${#pids[@]} -gt 0 ] || [ $task_index -lt ${#task_queue[@]} ]; then
        if [ ${#pids[@]} -gt 0 ]; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running: ${#pids[@]} jobs, Pending: $((${#task_queue[@]} - task_index)) tasks"
        fi
        sleep 30
    fi
done

# Process retry queue for additional evaluations
if [ ${#retry_task_queue[@]} -gt 0 ]; then
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Processing ${#retry_task_queue[@]} retry tasks for additional evaluations..."
    
    # Filter only additional eval tasks from retry queue
    additional_retry_tasks=()
    remaining_retry_tasks=()
    for task in "${retry_task_queue[@]}"; do
        IFS='|' read -r model_name model_path dataset task_type <<< "$task"
        if [ "$task_type" == "additional" ]; then
            additional_retry_tasks+=("${model_name}|${model_path}|${dataset}")
        else
            remaining_retry_tasks+=("$task")
        fi
    done
    retry_task_queue=("${remaining_retry_tasks[@]}")
    
    retry_index=0
    while [ $retry_index -lt ${#additional_retry_tasks[@]} ] || [ ${#pids[@]} -gt 0 ]; do
        check_completed_jobs
        available_gpus=($(get_available_gpus))
        
        while [ $retry_index -lt ${#additional_retry_tasks[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
            task="${additional_retry_tasks[$retry_index]}"
            IFS='|' read -r model_name model_path eval_script <<< "$task"
            
            if check_dataset_completed "$model_name" "$eval_script"; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Retry task already completed, skipping: model [${model_name}], dataset [${eval_script}]"
                retry_index=$((retry_index + 1))
                continue
            fi
            
            if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
                if start_additional_eval_job "${available_gpus[0]}" "$model_path" "$model_name" "$eval_script"; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Retry: Successfully started evaluation [${eval_script}] for model [${model_name}]"
                    retry_index=$((retry_index + 1))
                    available_gpus=($(get_available_gpus))
                else
                    break
                fi
            else
                retry_index=$((retry_index + 1))
            fi
        done
        
        if [ ${#pids[@]} -gt 0 ] || [ $retry_index -lt ${#additional_retry_tasks[@]} ]; then
            sleep 30
        fi
    done
fi

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All additional evaluations completed successfully!"

# Reset GPU status for main evaluation tasks
for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status["$gpu_id"]="idle"
done
pids=()
model_gpu_mapping=()
model_path_mapping=()
task_completed=()
gpu_dataset=()

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math tasks for ${#model_list[@]} models"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math datasets: ${TASKS[*]}"

# Build math task queue: array of "model_name|model_path|dataset"
declare -a math_task_queue=()

# Initialize math task queue with all model-dataset combinations
for model_name in "${model_list[@]}"; do
    if [ "$model_name" == "$base_model_name" ]; then
        model_path=${TTCS_MODEL_DIR}/${base_model_name}
    elif [[ $model_name =~ V[0-9]+ ]]; then
        model_path=${TTCS_SOLVER_DIR}/${model_name}/ckpts/global_step_${eval_step}/actor/huggingface
    else
        echo "Warning: Model name [${model_name}] does not match any known pattern"
        echo "Skipping model: $model_name"
        continue
    fi
    
    # Verify model path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path does not exist: $model_path"
        echo "Skipping model: $model_name"
        continue
    fi
    
    # Add each math dataset to the queue
    for dataset in "${TASKS[@]}"; do
        # Check if already completed before adding to queue
        if check_main_task_completed "$model_name" "$dataset"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Skipping already completed task: model [${model_name}], dataset [${dataset}]"
            continue
        fi
        math_task_queue+=("${model_name}|${model_path}|${dataset}")
    done
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math task queue initialized with ${#math_task_queue[@]} tasks"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math Task queue:"
for i in "${!math_task_queue[@]}"; do
  echo "[$i] ${math_task_queue[$i]}"
done
# Initialize cleanup counter
cleanup_counter=0
math_task_index=0

# Process math tasks asynchronously
while [ $math_task_index -lt ${#math_task_queue[@]} ] || [ ${#pids[@]} -gt 0 ]; do
    # Perform periodic comprehensive cleanup every 10 iterations
    cleanup_counter=$((cleanup_counter + 1))
    if [ $((cleanup_counter % 10)) -eq 0 ]; then
        comprehensive_cleanup
    else
        cleanup_zombie_processes
    fi
    
    check_completed_jobs
    available_gpus=($(get_available_gpus))
    
    # Try to start new tasks if there are available GPUs and pending tasks
    while [ $math_task_index -lt ${#math_task_queue[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
        # Get next task from queue
        task="${math_task_queue[$math_task_index]}"
        IFS='|' read -r model_name model_path dataset <<< "$task"
        
        # Double-check if completed (in case it was completed by another process)
        if check_main_task_completed "$model_name" "$dataset"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Task already completed, skipping: model [${model_name}], dataset [${dataset}]"
            math_task_index=$((math_task_index + 1))
            continue
        fi
        
        # Validate that GPU ID is a number
        if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
            # Start the math task job on the first available GPU
            if start_math_task_job "${available_gpus[0]}" "$model_path" "$model_name" "$dataset"; then
                local current_task=$((math_task_index + 1))
                local total_tasks=${#math_task_queue[@]}
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started math task [${dataset}] for model [${model_name}] - task ${current_task}/${total_tasks}"
                math_task_index=$((math_task_index + 1))
                # Update available GPUs after starting a job
                available_gpus=($(get_available_gpus))
            else
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Failed to start math task [${dataset}] for model [${model_name}], will retry later"
                # Break to wait for GPU availability
                break
            fi
        else
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Invalid GPU ID detected: GPU='${available_gpus[0]}'"
            math_task_index=$((math_task_index + 1))
            continue
        fi
    done
    
    # If there are still running jobs or pending tasks, wait a bit
    if [ ${#pids[@]} -gt 0 ] || [ $math_task_index -lt ${#math_task_queue[@]} ]; then
        if [ ${#pids[@]} -gt 0 ]; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running: ${#pids[@]} jobs, Pending: $((${#math_task_queue[@]} - math_task_index)) tasks"
        fi
        sleep 30
    fi
done

# Process retry queue for math tasks
if [ ${#retry_task_queue[@]} -gt 0 ]; then
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Processing ${#retry_task_queue[@]} retry tasks for math evaluations..."
    
    # Filter only math tasks from retry queue
    math_retry_tasks=()
    for task in "${retry_task_queue[@]}"; do
        IFS='|' read -r model_name model_path dataset task_type <<< "$task"
        if [ "$task_type" == "math" ]; then
            math_retry_tasks+=("${model_name}|${model_path}|${dataset}")
        fi
    done
    
    if [ ${#math_retry_tasks[@]} -gt 0 ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Found ${#math_retry_tasks[@]} math retry tasks"
        
        retry_index=0
        while [ $retry_index -lt ${#math_retry_tasks[@]} ] || [ ${#pids[@]} -gt 0 ]; do
            check_completed_jobs
            available_gpus=($(get_available_gpus))
            
            while [ $retry_index -lt ${#math_retry_tasks[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
                task="${math_retry_tasks[$retry_index]}"
                IFS='|' read -r model_name model_path dataset <<< "$task"
                
                if check_main_task_completed "$model_name" "$dataset"; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Retry task already completed, skipping: model [${model_name}], dataset [${dataset}]"
                    retry_index=$((retry_index + 1))
                    continue
                fi
                
                if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
                    if start_math_task_job "${available_gpus[0]}" "$model_path" "$model_name" "$dataset"; then
                        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Retry: Successfully started math task [${dataset}] for model [${model_name}]"
                        retry_index=$((retry_index + 1))
                        available_gpus=($(get_available_gpus))
                    else
                        break
                    fi
                else
                    retry_index=$((retry_index + 1))
                fi
            done
            
            if [ ${#pids[@]} -gt 0 ] || [ $retry_index -lt ${#math_retry_tasks[@]} ]; then
                sleep 30
            fi
        done
    fi
fi

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All math tasks completed successfully!"

# Final verification: check if all tasks are truly completed
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Final verification of all tasks..."
all_completed=true
failed_tasks=()

for model_name in "${model_list[@]}"; do
    # Check additional evaluations
    for eval_script in "${ADDITIONAL_EVAL_DATASETS[@]}"; do
        if ! check_dataset_completed "$model_name" "$eval_script" 2>/dev/null; then
            all_completed=false
            failed_tasks+=("${model_name}|${eval_script}")
        fi
    done
    
    # Check math tasks
    for dataset in "${TASKS[@]}"; do
        if ! check_main_task_completed "$model_name" "$dataset" 2>/dev/null; then
            all_completed=false
            failed_tasks+=("${model_name}|${dataset}")
        fi
    done
done

if [ "$all_completed" = true ]; then
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✓ All tasks verified complete!"
else
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ⚠ WARNING: The following tasks are still incomplete:"
    for task in "${failed_tasks[@]}"; do
        echo "    - $task"
    done
fi

# Aggregate evaluation results for all models
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Aggregating evaluation results..."
python aggregate_eval_results.py --save_path_dir "${eval_saved_path_dir}" --model_list "${model_list[@]}"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All evaluation results aggregated successfully!"

python tb.py  \
    --prefix=$prefix \
    --step=$eval_step \
    --temperature=$temperature \
    --eval_results_dir="${eval_saved_path_dir}" \
    --tb_path_dir="${tb_path_dir}" \
    --base_model="${base_model_name}" \
    --base_model_dir="${global_base_model_eval_dir}" \
    --step_interval=20

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All models eval results upload tb successfully!"
