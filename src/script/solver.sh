#!/usr/bin/env bash
set -euo pipefail

# Load process cleanup library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# Get parameters
exp_name="$1"
challenger_model_path="$2"
solver_model_path="$3"
solver_training_steps="$4"
gen_question_func="$5"
hybrid_data="$6"
train_file="$7"
solver_batch_size="$8"
real_data_ratio="$9"
rollout_n="${10}"
echo "[exp_name]:${exp_name}"
echo "[challenger_model_path]: ${challenger_model_path}"
echo "[solver_model_path]: ${solver_model_path}"
echo "[solver_training_steps]: ${solver_training_steps}"
echo "[gen_question_func]: ${gen_question_func}"
echo "[train_file]: ${train_file}"
echo "[solver_batch_size]: ${solver_batch_size}"
echo "[real_data_ratio]: ${real_data_ratio}"
echo "[rollout_n]: ${rollout_n}"
# Validate parameters
if [ -z "$exp_name" ]; then
    echo "Error: exp_name cannot be empty"
    exit 1
fi

if [ -z "$challenger_model_path" ]; then
    echo "Error: challenger_model_path cannot be empty"
    exit 1
fi

if [ -z "$solver_model_path" ]; then
    echo "Error: solver_model_path cannot be empty"
    exit 1
fi

if [ -z "$solver_training_steps" ]; then
    echo "Error: solver_training_steps cannot be empty"
    exit 1
fi

# Validate solver_training_steps is a number
if ! [[ "$solver_training_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: solver_training_steps must be a number, current value: $solver_training_steps"
    exit 1
fi

# Validate model paths exist
if [ ! -d "$challenger_model_path" ]; then
    echo "Error: challenger_model_path does not exist: $challenger_model_path"
    exit 1
fi

if [ ! -d "$solver_model_path" ]; then
    echo "Error: solver_model_path does not exist: $solver_model_path"
    exit 1
fi

# Use environment variables set by run_with_gpus.sh
storage_path=${TTCS_SOLVER_DIR}/${exp_name}
CKPTS_DIR=${storage_path}/ckpts/
tensorboard_path=${TTCS_TENSORBOARD_DIR}/Solver-${exp_name}
mkdir -p ${CKPTS_DIR} ${tensorboard_path}
export TENSORBOARD_DIR=${tensorboard_path}

echo "[Path Config] Working Dir: ${TTCS_WORKING_DIR}"
echo "[Path Config] Data Dir: ${TTCS_DATA_DIR}"
echo "[Path Config] Storage Path: ${storage_path}"

echo "[GPU Config] Solver GPUs: ${TTCS_SOLVER_GPUS} (total ${TTCS_N_SOLVER_GPUS})"
echo "[GPU Config] Gen Query GPUs: ${TTCS_GEN_QUERY_GPUS}"

gen_query_num=500
#gen_query_num=8
echo "Starting query data generation..."
bash ${SCRIPT_DIR}/gen_query.sh $exp_name $gen_query_num $challenger_model_path ${TTCS_CHALLENGER_DIR} ${TTCS_SOLVER_DIR} $gen_question_func $hybrid_data $train_file $real_data_ratio || {
    echo "Error: gen_query.sh failed, exiting..."
    exit 1
}
echo "Data generation completed"
sleep 10
adv_estimator=grpo

# DAPO related parameters

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 1))

max_response_length=$((1024 * 3))

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
filter_lower=0.25
filter_high=0.75

max_num_gen_batches=10 # Max batches to generate before filtering yields 1 batch
train_prompt_bsz=${solver_batch_size}
val_batch_size=512 # Validation batch size
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=${rollout_n}
train_prompt_mini_bsz=$((train_prompt_bsz / 2))

# Paths
TRAIN_FILE=${storage_path}/train_data.parquet
TEST_FILE=${TTCS_DATA_DIR}/ttrl/test_set.parquet

# Algorithm
temperature=1.0
top_p=0.99
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.01

# val
val_temperature=1.0
val_top_p=1.0
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_rollout_n=4


# Start training process
echo "Starting training process..."
echo "Using GPU: ${TTCS_SOLVER_GPUS}, total ${TTCS_N_SOLVER_GPUS}"
TRAINING_PID=""

cd ${TTCS_WORKING_DIR}
CUDA_VISIBLE_DEVICES=${TTCS_SOLVER_GPUS} python3 -m ${TTCS_CODE_MODULE}.main_solver_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=True \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.filter_lower=${filter_lower} \
    algorithm.filter_groups.filter_high=${filter_high} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${solver_model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_rollout_n} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=solver \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    reward_model.reward_kwargs.filter_lower=${filter_lower} \
    reward_model.reward_kwargs.filter_high=${filter_high} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${TTCS_PROJECT_NAME}" \
    trainer.experiment_name="Solver-${exp_name}" \
    trainer.n_gpus_per_node=${TTCS_N_SOLVER_GPUS} \
    trainer.nnodes=1 \
    trainer.total_training_steps=${solver_training_steps} \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto &

TRAINING_PID=$!

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?



# Check if training succeeded
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "Training failed, exit code: $TRAINING_EXIT_CODE"
    exit $TRAINING_EXIT_CODE
fi

echo "Training completed successfully"
sleep 10


echo "${exp_name} solver training finished"