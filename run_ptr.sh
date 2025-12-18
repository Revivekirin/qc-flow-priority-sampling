#!/bin/bash
#SBATCH --job-name=qc
#SBATCH --nodelist=pat-t3
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00


# MUJOCO_GL=egl python main.py \
#  --run_group=reproduce \
#  --agent.alpha=100 \
#  --env_name=lift-mh-low \
#  --sparse=False --horizon_length=5 \
# #  --video_episodes=5


# MUJOCO_GL=egl python main_ptr.py \
#   --env_name=square-mh-low \
#   --use_ptr_backward=True \
#   --use_ptr_online_priority=True \
#   --sparse=False \
#   --agent.alpha=100 \
#   --horizon_length=5 \
#   --use_weighted_target=False \
#   --metric=success_binary \
#   --backward=True \


#     PRIORITY_METRICS = [
#         "uniform",          # Baseline (no priority)
#         "success_binary",   # Current (simple)
#         "avg_reward",       # PTR baseline
#         "uqm_reward",       # PTR best (Table 3)
#         "uhm_reward",       # PTR alternative
#         "min_reward",       # PTR defensive
#     ]


# MUJOCO_GL=egl python main_ptr_logging.py \
#   --env_name=square-mh-low \
#   --use_ptr_backward=True \
#   --use_ptr_online_priority=True \
#   --sparse=False \
#   --agent.alpha=100 \
#   --horizon_length=5 \
#   --metric=td_error_rank \
#   --backward=False \
#   --use_weighted_target=False \
  # --offline_steps=0

# MUJOCO_GL=egl python main_ptr_logging.py \
#   --env_name=square-mh-low \
#   --use_ptr_backward=True \
#   --use_ptr_online_priority=True \
#   --sparse=False \
#   --agent.alpha=100 \
#   --horizon_length=5 \
#   --metric=avg_reward \
#   --backward=False \
#   --use_weighted_target=False \

# MUJOCO_GL=egl python main_ptr_logging.py \
#   --env_name=square-mh-low \
#   --use_ptr_backward=True \
#   --use_ptr_online_priority=True \
#   --sparse=False \
#   --agent.alpha=100 \
#   --horizon_length=5 \
#   --metric=uqm_reward \
#   --backward=False \
#   --use_weighted_target=False \

# MUJOCO_GL=egl python main_ptr_logging.py \
#   --env_name=square-mh-low \
#   --use_ptr_backward=True \
#   --use_ptr_online_priority=True \
#   --sparse=False \
#   --agent.alpha=100 \
#   --horizon_length=5 \
#   --metric=success_binary \
#   --backward=False \
#   --use_weighted_target=False \
#   --cluster_sampler=True \
#   --cluster_sampler=True \
#   --cluster_use_curriculum=True \
#   --cluster_curriculum_steps=200000

# MUJOCO_GL=egl python main_ptr_logging.py \
#   --env_name=square-mh-low \
#   --use_ptr_backward=True \
#   --use_ptr_online_priority=True \
#   --sparse=False \
#   --agent.alpha=100 \
#   --horizon_length=5 \
#   --metric=success_binary \
#   --backward=False \
#   --use_weighted_target=False \
#   --cluster_sampler=True \
#   --entity=sophia435256-robros

MUJOCO_GL=egl python debug.py \
  --env_name=transport-mh-low \
  --use_ptr_backward=True \
  --use_ptr_online_priority=True \
  --sparse=False \
  --agent.alpha=100 \
  --horizon_length=5 \
  --metric=td_error_rank \
  --backward=False \
  --use_weighted_target=False \
  --cluster_sampler=False \
  --entity=sophia435256-robros \