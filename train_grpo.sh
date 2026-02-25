export CUDA_VISIBLE_DEVICES=1,2,3
# ray start --head --node-ip-address=10.0.0.2 --dashboard-host=10.0.0.2 --include-dashboard=True
# 下面三个是我加的，跑原searchr1可以去掉
#    algorithm.use_diversity=true \
#    reward_model.use_format: True \
#    actor_rollout_ref.actor.clip_higher=0.1 \


#    trainer.total_epochs=35 \
#    trainer.total_training_steps=305 \

# 干掉tmp下多余的内容
# sudo find /tmp -mindepth 1 -maxdepth 1 -type d -exec du -sh {} + | awk '$1 ~ /G/ && $1+0 >= 1 {print $2}' | xargs -r sudo rm -rf
export DATA_DIR='/home/hwai/programs/Search-R1/data/nq_search'

export WAND_PROJECT='Search-R1'
export PYTHONUNBUFFERED=1
# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-it-em

#export BASE_MODEL='/home/hwai/weights/Qwen2.5-3B-Instruct'
export BASE_MODEL='/home/hwai/weights/Qwen2.5-3B'

#export BASE_MODEL='/home/hwai/weights/Qwen3-8b'
#export EXPERIMENT_NAME=nq-search-r1-grpo-qwen-2.5-3b-diver-clip-higher-mvavg
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen-2.5-3b-sac-base-prompt-e5
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export WANDB_MODE=online
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEBUGGER_HOST=0.0.0.0
# 🔒 只让本地和内网直连，其它都走代理
#export no_proxy="127.0.0.1,localhost,*.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
#export NO_PROXY="$no_proxy"

# 🔌 代理地址（按你的 Clash 调整）
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
#export ALL_PROXY="$https_proxy"
#export all_proxy="$https_proxy"

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])
# -m debugpy --listen 0.0.0.0:5678
nohup python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_diversity=True \
    reward_model.use_format=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_higher=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=25 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=35 \
    trainer.total_training_steps=305 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log &
