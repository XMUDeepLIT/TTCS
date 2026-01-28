#!/usr/bin/env bash
# Process cleanup library
# Provides safe process cleanup in multi-user environments

# Configuration parameters
CLEANUP_TIMEOUT_CHECK=${CLEANUP_TIMEOUT_CHECK:-15}      # Residual process check timeout (seconds)
CLEANUP_TIMEOUT_FALLBACK=${CLEANUP_TIMEOUT_FALLBACK:-45} # Fallback cleanup timeout (seconds)
CLEANUP_PROCESS_TIMEOUT=${CLEANUP_PROCESS_TIMEOUT:-3}    # Single process check timeout (seconds)
CLEANUP_SLEEP_INTERVAL=${CLEANUP_SLEEP_INTERVAL:-2}      # Wait time after process termination (seconds)
CLEANUP_PORTS=${CLEANUP_PORTS:-"5000 5001 5002 5003"}   # Ports to clean up

# Get unique session identifier
get_session_id() {
    local USER_ID=$(whoami)
    local SESSION_ID="$$"
    local PROCESS_GROUP_ID=$(ps -o pgid= -p $$ | tr -d ' ')
    local SESSION_PID=$(ps -o sid= -p $$ | tr -d ' ')
    local UNIQUE_SESSION_ID="${USER_ID}_${SESSION_PID}_${PROCESS_GROUP_ID}_${SESSION_ID}"
    
    echo "$UNIQUE_SESSION_ID"
}

# Get process tracking file path
# Args: $1 - script name prefix (optional, default "solver")
get_process_track_file() {
    local script_prefix="${1:-solver}"  # Default to "solver" as prefix
    local tmp="${2:-/tmp}"              # Default to "/tmp", can be customized
    local session_id=$(get_session_id)
    echo "${tmp}/${script_prefix}_processes_${session_id}.txt"
}

# Check if process belongs to current user session
is_my_process() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    
    # Check if process exists
    if ! ps -p "$pid" > /dev/null 2>&1; then
        return 1
    fi
    
    # Get current session info
    local SESSION_ID="$$"
    local PROCESS_GROUP_ID=$(ps -o pgid= -p $$ | tr -d ' ')
    local SESSION_PID=$(ps -o sid= -p $$ | tr -d ' ')
    
    # Get process session ID and process group ID
    local proc_sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')
    local proc_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    
    # Check if belongs to current session or process group
    if [ "$proc_sid" = "$SESSION_PID" ] || [ "$proc_pgid" = "$PROCESS_GROUP_ID" ]; then
        return 0
    fi
    
    # Additional check: if started by us, check parent process chain
    local ppid=$pid
    local max_depth=10
    local depth=0
    
    while [ "$ppid" != "1" ] && [ "$depth" -lt "$max_depth" ]; do
        if [ "$ppid" = "$SESSION_ID" ] || [ "$ppid" = "$PROCESS_GROUP_ID" ]; then
            return 0
        fi
        ppid=$(ps -o ppid= -p "$ppid" 2>/dev/null | tr -d ' ')
        depth=$((depth + 1))
    done
    
    return 1
}

# Get process list belonging to current user
get_my_processes() {
    local pattern=$1
    local result=""
    while read -r pid; do
        if is_my_process "$pid"; then
            result="$result $pid"
        fi
    done < <(pgrep -f "$pattern" 2>/dev/null || true)
    echo "$result"
}

# Process tracking function
# Args: $1 - PID, $2 - description, $3 - script prefix (optional), $4 - tmp path (optional)
track_process() {
    local pid=$1
    local description=$2
    local script_prefix="${3:-solver}"  # Default to "solver" as prefix
    local tmp="${4:-/tmp}"              # Default to "/tmp", can be customized
    local track_file=$(get_process_track_file "$script_prefix" "$tmp")
    echo "$pid:$description" >> "$track_file"
    echo "Tracking process: PID=$pid, Description=$description"
}

# Show tracked process status
# Args: $1 - script prefix (optional), $2 - tmp path (optional)
show_tracked_processes() {
    local script_prefix="${1:-solver}"  # Default to "solver" as prefix
    local tmp="${2:-/tmp}"              # Default to "/tmp", can be customized
    local track_file=$(get_process_track_file "$script_prefix" "$tmp")
    if [ -f "$track_file" ]; then
        echo "Currently tracked processes (${script_prefix}):"
        while IFS=':' read -r pid description; do
            if [ -n "$pid" ] && [ -n "$description" ]; then
                if ps -p "$pid" > /dev/null 2>&1; then
                    echo "  ✓ PID $pid ($description) - Running"
                else
                    echo "  ✗ PID $pid ($description) - Finished"
                fi
            fi
        done < "$track_file"
    else
        echo "No tracked processes (${script_prefix})"
    fi
}

# Clean up tracked processes
# Args: $1 - script prefix (optional), $2 - tmp path (optional)
cleanup_tracked_processes() {
    local script_prefix="${1:-solver}"  # Default to "solver" as prefix
    local tmp="${2:-/tmp}"              # Default to "/tmp", can be customized
    local track_file=$(get_process_track_file "$script_prefix" "$tmp")
    local tracked_cleaned=false
    
    if [ -f "$track_file" ]; then
        echo "Phase 1: Cleaning up tracked processes..."
        while IFS=':' read -r pid description; do
            if [ -n "$pid" ] && [ -n "$description" ]; then
                if ps -p "$pid" > /dev/null 2>&1; then
                    echo "  Terminating process: PID=$pid ($description)"
                    recursive_kill_process "$pid" "TERM" "$description"
                    sleep 2
                    if ps -p "$pid" > /dev/null 2>&1; then
                        echo "  Force killing process: PID=$pid"
                        recursive_kill_process "$pid" "KILL" "$description"
                    fi
                    tracked_cleaned=true
                fi
            fi
        done < "$track_file"
        rm -f "$track_file"
        echo "Tracked process cleanup completed"
    fi
    
    return $([ "$tracked_cleaned" = true ] && echo 0 || echo 1)
}

# Process check function with timeout
check_processes_with_timeout() {
    local pattern=$1
    local timeout=${2:-$CLEANUP_PROCESS_TIMEOUT}  # Use configured timeout
    local result=""
    
    # Use timeout command to limit execution time
    result=$(timeout $timeout bash -c "
        local pids=\$(pgrep -f \"$pattern\" 2>/dev/null || true)
        if [ -n \"\$pids\" ]; then
            for pid in \$pids; do
                if is_my_process \"\$pid\"; then
                    echo \"\$pid\"
                fi
            done
        fi
    " 2>/dev/null || echo "")
    
    echo "$result"
}

# Recursively clean up process and all child processes (simplified version)
recursive_kill_process() {
    local target_pid="$1"
    local signal="${2:-TERM}"
    local description="${3:-process}"
    
    if ! ps -p "$target_pid" > /dev/null 2>&1; then
        return 0  # Process does not exist, return directly
    fi
    
    # Get process info
    local cmd=$(ps -p "$target_pid" -o cmd= 2>/dev/null || echo "unknown")
    echo "    Terminating ${description}: PID=$target_pid, CMD=$cmd"
    
    # Use pkill -P to recursively terminate all child processes
    # This terminates child processes first, then parent process
    pkill -P "$target_pid" -"$signal" 2>/dev/null || true
    
    # Give child processes time to exit
    sleep 0.5
    
    # Finally send signal to parent process
    kill -"$signal" "$target_pid" 2>/dev/null || true
}

# Check residual processes (optimized version)
check_residual_processes() {
    local need_fallback=false
    
    echo "  Checking residual processes (with timeout protection)..."
    
    # Check Ray processes
    local ray_pids=$(check_processes_with_timeout "ray::")
    if [ -n "$ray_pids" ]; then
        echo "  Found residual Ray processes: $ray_pids"
        need_fallback=true
    fi
    
    # Check training processes
    local training_pids=$(check_processes_with_timeout "python3 -m se.main_solver_dapo")
    if [ -n "$training_pids" ]; then
        echo "  Found residual training processes: $training_pids"
        need_fallback=true
    fi
    
    # Check challenger training processes
    local challenger_pids=$(check_processes_with_timeout "python3 -m se.main_challenger")
    if [ -n "$challenger_pids" ]; then
        echo "  Found residual challenger training processes: $challenger_pids"
        need_fallback=true
    fi
    
    # Check model merge processes
    local merge_pids=$(check_processes_with_timeout "python3 -m verl.model_merger")
    if [ -n "$merge_pids" ]; then
        echo "  Found residual model merge processes: $merge_pids"
        need_fallback=true
    fi
    
    # Check vLLM processes (using more precise pattern)
    local vllm_pids=$(check_processes_with_timeout "start_vllm_server.py")
    if [ -n "$vllm_pids" ]; then
        echo "  Found residual vLLM server processes: $vllm_pids"
        need_fallback=true
    fi
    
    # Check port usage (force check all processes)
    local port_pids=""
    for port in $CLEANUP_PORTS; do
        local port_users=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_users" ]; then
            for pid in $port_users; do
                # Check if process exists
                if ps -p "$pid" > /dev/null 2>&1; then
                    local user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
                    local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 100)
                    echo "    Found port ${port} in use: PID=$pid, USER=$user, CMD=$cmd"
                    port_pids="$port_pids $pid"
                fi
            done
        fi
    done
    if [ -n "$port_pids" ]; then
        echo "  Found port-using processes: $port_pids"
        need_fallback=true
    fi
    
    echo "$need_fallback"
}

# Force cleanup all vLLM related processes (integrated from force_cleanup_vllm.sh)
force_cleanup_vllm_processes() {
    echo "  Force cleaning up all vLLM related processes..."
    
    # Show process status before cleanup
    echo "    vLLM process status before cleanup:"
    ps aux | grep start_vllm_server.py | grep -v grep || echo "      No vLLM processes found"
    
    # 1. Get all start_vllm_server.py process PIDs
    local vllm_pids=$(ps aux | grep start_vllm_server.py | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    
    if [ -n "$vllm_pids" ]; then
        echo "    Found vLLM processes: $vllm_pids"
        
        # Send TERM signal
        echo "    Sending TERM signal..."
        for pid in $vllm_pids; do
            echo "      Terminating process: PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
        done
        
        # Wait for processes to exit
        echo "    Waiting for processes to exit..."
        sleep 3
        
        # Check if processes still exist
        local remaining_pids=$(ps aux | grep start_vllm_server.py | grep -v grep | awk '{print $2}' 2>/dev/null || true)
        if [ -n "$remaining_pids" ]; then
            echo "    Processes still running, sending KILL signal..."
            for pid in $remaining_pids; do
                echo "      Force killing process: PID=$pid"
                kill -KILL "$pid" 2>/dev/null || true
            done
            sleep 1
        fi
    else
        echo "    No vLLM processes to clean up"
    fi
    
    # 2. Clean up vLLM multiprocess child processes (orphaned processes)
    echo "    Cleaning up vLLM multiprocess child processes..."
    local vllm_orphan_pids=$(ps aux | grep "multiprocessing.spawn import spawn_main" | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    if [ -n "$vllm_orphan_pids" ]; then
        echo "    Found vLLM child processes: $vllm_orphan_pids"
        for pid in $vllm_orphan_pids; do
            echo "      Terminating vLLM child process: PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2
        # Check for remaining processes
        local remaining_orphan_pids=$(ps aux | grep "multiprocessing.spawn import spawn_main" | grep -v grep | awk '{print $2}' 2>/dev/null || true)
        if [ -n "$remaining_orphan_pids" ]; then
            echo "    Force killing remaining vLLM child processes..."
            for pid in $remaining_orphan_pids; do
                echo "      Force killing vLLM child process: PID=$pid"
                kill -KILL "$pid" 2>/dev/null || true
            done
        fi
    else
        echo "    No vLLM child processes to clean up"
    fi
    
    # Show process status after cleanup
    echo "    vLLM process status after cleanup:"
    ps aux | grep start_vllm_server.py | grep -v grep || echo "      No vLLM processes found"
    ps aux | grep "multiprocessing.spawn import spawn_main" | grep -v grep || echo "      No vLLM child processes found"
    
    # Show port usage status after cleanup
    echo "    Port usage status after cleanup:"
    for port in $CLEANUP_PORTS; do
        local port_pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_pids" ]; then
            echo "      Port $port still in use by: $port_pids"
        else
            echo "      Port $port is free"
        fi
    done
    
    echo "  vLLM process force cleanup completed"
}

# Unified vLLM process cleanup (backward compatible)
cleanup_vllm_processes() {
    force_cleanup_vllm_processes
}

# Port-based vLLM process cleanup (simple and reliable)
cleanup_vllm_ports() {
    echo "Cleaning up vLLM port processes..."
    for port in $CLEANUP_PORTS; do
        local port_pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_pids" ]; then
            echo "  Cleaning up port $port processes (recursive): $port_pids"
            for pid in $port_pids; do
                echo "    Terminating port $port process: PID=$pid"
                recursive_kill_process "$pid" "TERM" "port ${port} process"
            done
            sleep 1
            for pid in $port_pids; do
                recursive_kill_process "$pid" "KILL" "port ${port} process"
            done
        else
            echo "  Port $port: free"
        fi
    done
    echo "vLLM port cleanup completed"
}

# Fallback cleanup residual processes (optimized version)
cleanup_residual_processes() {
    echo "Phase 3: Executing fallback cleanup..."
    
    # Clean up residual Ray processes (recursive)
    local ray_pids=$(check_processes_with_timeout "ray::")
    if [ -n "$ray_pids" ]; then
        echo "  Cleaning up residual Ray processes..."
        for pid in $ray_pids; do
            recursive_kill_process "$pid" "TERM" "Ray process"
        done
        sleep 1
        for pid in $ray_pids; do
            recursive_kill_process "$pid" "KILL" "Ray process"
        done
    fi
    
    # Clean up residual training processes (recursive)
    local training_pids=$(check_processes_with_timeout "python3 -m se.main_solver_dapo")
    if [ -n "$training_pids" ]; then
        echo "  Cleaning up residual training processes..."
        for pid in $training_pids; do
            recursive_kill_process "$pid" "TERM" "training process"
        done
        sleep 1
        for pid in $training_pids; do
            recursive_kill_process "$pid" "KILL" "training process"
        done
    fi
    
    # Clean up residual challenger training processes (recursive)
    local challenger_pids=$(check_processes_with_timeout "python3 -m se.main_challenger")
    if [ -n "$challenger_pids" ]; then
        echo "  Cleaning up residual challenger training processes..."
        for pid in $challenger_pids; do
            recursive_kill_process "$pid" "TERM" "challenger training process"
        done
        sleep 1
        for pid in $challenger_pids; do
            recursive_kill_process "$pid" "KILL" "challenger training process"
        done
    fi
    
    # Clean up residual model merge processes (recursive)
    local merge_pids=$(check_processes_with_timeout "python3 -m verl.model_merger")
    if [ -n "$merge_pids" ]; then
        echo "  Cleaning up residual model merge processes..."
        for pid in $merge_pids; do
            recursive_kill_process "$pid" "TERM" "model merge process"
        done
        sleep 1
        for pid in $merge_pids; do
            recursive_kill_process "$pid" "KILL" "model merge process"
        done
    fi
    
    # Unified vLLM process cleanup
    cleanup_vllm_processes
    
    # Force cleanup port-using processes (recursive, check user permissions)
    echo "  Force cleaning up port-using processes..."
    for port in $CLEANUP_PORTS; do
        local port_users=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_users" ]; then
            echo "    Cleaning up port ${port} processes: $port_users"
            for pid in $port_users; do
                if ps -p "$pid" > /dev/null 2>&1; then
                    # Check if process belongs to current user
                    local proc_user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
                    local current_user=$(whoami)
                    if [ "$proc_user" = "$current_user" ]; then
                        recursive_kill_process "$pid" "KILL" "port ${port} process"
                    else
                        echo "      Skipping other user's process: PID=$pid, USER=$proc_user"
                    fi
                fi
            done
        fi
    done
    
    echo "Fallback cleanup completed"
}

# Show GPU status
show_gpu_status() {
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "Unable to get GPU status"
    echo ""
}

# Main cleanup function
# Args: $1 - script prefix (optional), $2 - tmp path (optional)
cleanup_processes() {
    local script_prefix="${1:-solver}"  # Default to "solver" as prefix
    local tmp="${2:-/tmp}"              # Default to "/tmp", can be customized
    set +e
    echo "=========================================="
    echo "Starting process cleanup (${script_prefix})..."
    
    # Show current GPU status
    show_gpu_status
    
    # Show tracked process status
    echo "Process status before cleanup:"
    show_tracked_processes "$script_prefix" "$tmp"
    echo ""
    
    # Phase 1: Clean up tracked processes
    cleanup_tracked_processes "$script_prefix" "$tmp"
    
    # Wait for processes to exit
    sleep 2
    
    # Phase 2: Check residual processes (with timeout protection)
    echo "Phase 2: Checking residual processes..."
    local need_fallback
    need_fallback=$(timeout $CLEANUP_TIMEOUT_CHECK check_residual_processes 2>/dev/null || echo "true")
    
    # Phase 3: Fallback cleanup (with timeout protection)
    if [ "$need_fallback" = true ]; then
        echo "Executing fallback cleanup (with timeout protection)..."
        if timeout $CLEANUP_TIMEOUT_FALLBACK cleanup_residual_processes 2>/dev/null; then
            echo "Fallback cleanup completed"
        else
            echo "Warning: Fallback cleanup timed out, but best effort was made"
        fi
    else
        echo "No residual processes found, skipping fallback cleanup"
    fi
    
    # Show GPU status after cleanup
    echo ""
    echo "GPU status after cleanup:"
    show_gpu_status
    
    echo "Process cleanup completed"
    echo "=========================================="
}





# Check if process cleanup is disabled
check_cleanup_option() {
    local arg_count="$1"
    local last_arg="$2"
    
    if [ $arg_count -eq 5 ] && [ "$last_arg" = "--no-cleanup" ]; then
        echo "false"
    else
        echo "true"
    fi
}




# Standalone force cleanup vLLM function (can be called directly)
force_cleanup_vllm() {
    echo "=========================================="
    echo "Force cleaning up all start_vllm_server.py processes"
    echo "=========================================="
    
    force_cleanup_vllm_processes
    
    echo ""
    echo "=========================================="
    echo "Force cleanup completed"
    echo "=========================================="
}

# If this script is executed directly, perform cleanup
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Check if force vLLM cleanup is specified
    if [ "$1" = "--force-vllm" ]; then
        force_cleanup_vllm
    else
        cleanup_processes "${1:-solver}"
    fi
fi

