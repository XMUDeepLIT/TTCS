#!/usr/bin/env bash

set -xeuo pipefail
# Get script directory
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use environment variables set by run_with_gpus.sh
base_model_name="${TTCS_BATTCS_MODEL_NAME:-Qwen2.5-Math-1.5B}"
base_model_path="${TTCS_BATTCS_MODEL_PATH:-${TTCS_MODEL_DIR}/${base_model_name}}"

question_reward=TTCS

dataset=Minerva
real_data_ratio=5.0
temperature=0.6
# dataset -> batch_size mapping table
declare -A dataset_batch_size_map=(
    ["AIME24"]=16
    ["AIME25"]=16
    ["AMC23"]=16
    ["MATH500"]=64
    ["Minerva"]=64
    ["OlympiadBench"]=64
)
declare -A dataset_rollout_n_map=(
    ["AIME24"]=16
    ["AIME25"]=16
    ["AMC23"]=16
    ["MATH500"]=8
    ["Minerva"]=8
    ["OlympiadBench"]=8
)
train_file=${TTCS_DATA_DIR}/ttrl/${dataset}.parquet
solver_batch_size=${dataset_batch_size_map[$dataset]:-8}
rollout_n=${dataset_rollout_n_map[$dataset]:-8}
echo "[Config] dataset: ${dataset}, solver_batch_size: ${solver_batch_size}, rollout_n: ${rollout_n}, train_file: ${train_file}"

group_question_repetion_penalty=True
gen_question_func=ttrl_icl
hybrid_data=True
if [ "$TTCS_N_GPUS" -eq 4 ]; then
    challenger_training_steps=5
else
    challenger_training_steps=5
fi
solver_training_steps=15

variant=data_${dataset}_${question_reward}_gq${gen_question_func}_${base_model_name}

#variant=ttrl_qr_${question_reward}_gq${gen_question_func}_${base_model_name}
exp_name=${variant}-V1

solver_retrain_steps=15
solver_eval_step=15
solver_eval_temperature=0.6
solver_eval_num_iter=15
mkdir -p ${TTCS_SAVED_RESULTS_DIR} ${TTCS_CHALLENGER_DIR} ${TTCS_SOLVER_DIR} ${TTCS_TENSORBOARD_DIR} ${TTCS_WORKING_DIR}/logs
function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
exec > >(tee -a "${TTCS_WORKING_DIR}/logs/train_${variant}-$(now).log") 2>&1

cd ${TTCS_WORKING_DIR}
echo "Starting round 1 training..."
echo "Training Challenger..."
bash ${SCRIPT_DIR}/challenger.sh $exp_name $base_model_path $base_model_path $challenger_training_steps $question_reward $group_question_repetion_penalty $gen_question_func $train_file|| {
    echo "Error: Round 1 Challenger training failed"
    exit 1
}

echo "Training Solver..."
bash ${SCRIPT_DIR}/solver.sh $exp_name ${TTCS_CHALLENGER_DIR}/${exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface $base_model_path $solver_training_steps $gen_question_func $hybrid_data  $train_file $solver_batch_size $real_data_ratio $rollout_n || {
    echo " Error: Round 1 Solver training failed"
    exit 1
}


for iter in $(seq 2 ${solver_eval_num_iter}); do
    prev=$((iter-1))
    prev_exp_name=${variant}-V${prev}
    exp_name=${variant}-V${iter}
    prev_challenger_model_path=${TTCS_CHALLENGER_DIR}/${prev_exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface
    cur_challenger_model_path=${TTCS_CHALLENGER_DIR}/${exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface
    prev_solver_model_path=${TTCS_SOLVER_DIR}/${prev_exp_name}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface
    cur_solver_model_path=${TTCS_SOLVER_DIR}/${exp_name}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface

    echo "Starting round ${iter} training..."
    echo "Training Challenger (${exp_name})..."
    bash ${SCRIPT_DIR}/challenger.sh \
        ${exp_name} ${prev_challenger_model_path} \
        ${prev_solver_model_path} $challenger_training_steps $question_reward $group_question_repetion_penalty $gen_question_func $train_file || {
        echo "Error: Round ${iter} Challenger training failed"
        exit 1
    }

    echo "Training Solver (${exp_name})..."
    bash ${SCRIPT_DIR}/solver.sh \
        ${exp_name} ${cur_challenger_model_path} \
        ${prev_solver_model_path} $solver_training_steps $gen_question_func $hybrid_data $train_file $solver_batch_size $real_data_ratio $rollout_n|| {
        echo "Error: Round ${iter} Solver training failed"
        exit 1
    }
    
    echo "Round ${iter} training completed"
done

echo "All training completed!"
echo "Starting evaluation..."
eval_script="${TTCS_EVAL_SCRIPT:-${TTCS_WORKING_DIR}/evaluation/eval_single_math_data.sh}"
bash "$eval_script" "${variant}"  "${solver_eval_step}" "${solver_eval_num_iter}" "${base_model_name}" "${dataset}" || {
    echo "Error: Evaluation failed"
    exit 1
}
bash "$eval_script" "${variant}"  "${solver_eval_step}" "${solver_eval_num_iter}" "${base_model_name}" "${dataset}"|| {
    echo "Error: Evaluation failed"
    exit 1
}

