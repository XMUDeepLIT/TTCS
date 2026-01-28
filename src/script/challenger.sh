#!/usr/bin/env bash
# Challenger training script
# Usage: ./challenger.sh <exp_name> <challenger_model_path> <solver_model_path> <challenger_training_steps> [--no-cleanup]

set -euo pipefail


# Import process cleanup library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# Force cleanup vLLM processes and ports before starting (prevent port conflicts)
echo "Force cleaning up vLLM processes and ports before starting..."
force_cleanup_vllm_processes


# Parse parameters
exp_name="$1"
challenger_model_path="$2"
solver_model_path="$3"
challenger_training_steps="$4"
question_reward="$5"
group_question_repetion_penalty="$6"
gen_question_func="$7"
ttrl_train_file="$8"
echo "[exp_name]:${exp_name}"
echo "[challenger_model_path]: ${challenger_model_path}"
echo "[solver_model_path]: ${solver_model_path}"
echo "[challenger_training_steps]: ${challenger_training_steps}"
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

if [ -z "$challenger_training_steps" ]; then
    echo "Error: challenger_training_steps cannot be empty"
    exit 1
fi

# Validate challenger_training_steps is a number
if ! [[ "$challenger_training_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: challenger_training_steps must be a number, current value: $challenger_training_steps"
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
storage_path=${TTCS_CHALLENGER_DIR}/${exp_name}
CKPTS_DIR=${storage_path}/ckpts/
tensorboard_path=${TTCS_TENSORBOARD_DIR}/Challenger-${exp_name}
mkdir -p ${CKPTS_DIR} ${tensorboard_path}
export TENSORBOARD_DIR=${tensorboard_path}

echo "[Path Config] Working Dir: ${TTCS_WORKING_DIR}"
echo "[Path Config] Prompt Dir: ${TTCS_PROMPT_DIR}"
echo "[Path Config] Storage Path: ${storage_path}"

echo "[GPU Config] Challenger GPUs: ${TTCS_CHALLENGER_GPUS} (total ${TTCS_N_CHALLENGER_GPUS})"
echo "[GPU Config] Reward GPUs: ${TTCS_REWARD_GPUS}"
echo "[GPU Config] Reward Ports: ${TTCS_REWARD_PORTS}"

bash ${SCRIPT_DIR}/challenger_reward.sh $solver_model_path 
# Wait for vLLM service to start
echo "Waiting for vLLM service to start..."
sleep 10  # Increase wait time to ensure service is fully started



#rollout_query_num=8
rollout_query_num=4
query_top_p=0.99
query_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
kl_loss_coef=0.01
query_temperature=1.0
if [ "$TTCS_N_GPUS" -eq 4 ]; then
    batch_size=32
else
    batch_size=32
fi
ppo_mini_batch_size=$((batch_size / 4))
micro_batch_size_per_gpu=$((ppo_mini_batch_size / 4))
num_query=$((batch_size * challenger_training_steps))
#num_query=8
tp=1

cd ${TTCS_WORKING_DIR}
# Start training process
echo "Starting main training process..."
echo "Using GPU: ${TTCS_CHALLENGER_GPUS}, total ${TTCS_N_CHALLENGER_GPUS}"
TRAINING_PID=""

CUDA_VISIBLE_DEVICES=${TTCS_CHALLENGER_GPUS} python3 -m ${TTCS_CODE_MODULE}.main_challenger \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.num_querys=${num_query} \
    +data.get_prompts_func=${gen_question_func} \
    +data.gen_question_func=${gen_question_func} \
    +data.prompt_path=${TTCS_PROMPT_DIR} \
    data.filter_overlong_prompts=True \
    data.dynamic_topics=False \
    data.truncation='error' \
    +data.ttrl_icl_files=${ttrl_train_file} \
    actor_rollout_ref.model.path=${challenger_model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${query_temperature} \
    actor_rollout_ref.rollout.top_p=${query_top_p} \
    actor_rollout_ref.rollout.top_k="${query_top_k}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=${rollout_query_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=challenger \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    +reward_model.reward_kwargs.question_reward=${question_reward} \
    +reward_model.reward_kwargs.group_question_repetion_penalty=${group_question_repetion_penalty} \
    +reward_model.reward_kwargs.gen_question_func=${gen_question_func} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${TTCS_PROJECT_NAME}" \
    trainer.experiment_name="Challenger-${exp_name}" \
    trainer.n_gpus_per_node=${TTCS_N_CHALLENGER_GPUS} \
    trainer.total_training_steps=${challenger_training_steps} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=${challenger_training_steps} \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 &

# Record training process PID
TRAINING_PID=$!

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?


# Check if training succeeded
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "Training failed, exit code: $TRAINING_EXIT_CODE"
    echo "Starting process cleanup..."
    exit $TRAINING_EXIT_CODE
fi

echo "Training completed successfully"
sleep 10


pkill python

echo "${exp_name} challenger training finished"
echo "model path: ${CKPTS_DIR}/${exp_name}"
