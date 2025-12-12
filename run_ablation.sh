#!/bin/bash
#SBATCH --job-name=qc
#SBATCH --nodelist=pat-t3
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda activate qc
# =============================================================================
# Quick PTR Priority Metrics Test
# =============================================================================
# This script tests just the priority metrics (no weighted critic)
# Takes ~30 minutes total (3 variants × 10 min each)
# =============================================================================

ENV_NAME="square-mh-low_dim"
OFFLINE_STEPS=10000
ONLINE_STEPS=10000
EVAL_INTERVAL=5000
SEED=0

echo "========================================="
echo "Quick PTR Priority Test"
echo "========================================="
echo "Testing 3 priority variants:"
echo "  1. uniform (no priority baseline)"
echo "  2. success_binary (current implementation)"
echo "  3. reward_mean (enhanced PTR)"
echo ""
echo "Environment: $ENV_NAME"
echo "Steps: ${OFFLINE_STEPS} offline + ${ONLINE_STEPS} online"
echo ""

# Create logs directory
mkdir -p logs

# =============================================================================
# Variant 1: Uniform (No Priority)
# =============================================================================
echo ""
echo "========================================="
echo "[1/3] Testing: uniform (baseline)"
echo "========================================="
python main_ptr_ablation.py \
  --env_name=$ENV_NAME \
  --seed=$SEED \
  --use_ptr_backward=True \
  --use_ptr_online_priority=True \
  --ptr_priority_metric=uniform \
  --offline_steps=$OFFLINE_STEPS \
  --online_steps=$ONLINE_STEPS \
  --eval_interval=$EVAL_INTERVAL \
  --eval_episodes=10 \
  --run_group=PTR_QuickTest \
  2>&1 | tee logs/quick_uniform.log

# =============================================================================
# Variant 2: Success Binary (Current)
# =============================================================================
echo ""
echo "========================================="
echo "[2/3] Testing: success_binary (current)"
echo "========================================="
python main_ptr_ablation.py \
  --env_name=$ENV_NAME \
  --seed=$SEED \
  --use_ptr_backward=True \
  --use_ptr_online_priority=True \
  --ptr_priority_metric=success_binary \
  --offline_steps=$OFFLINE_STEPS \
  --online_steps=$ONLINE_STEPS \
  --eval_interval=$EVAL_INTERVAL \
  --eval_episodes=10 \
  --run_group=PTR_QuickTest \
  2>&1 | tee logs/quick_success_binary.log

# =============================================================================
# Variant 3: Reward Mean (Enhanced)
# =============================================================================
echo ""
echo "========================================="
echo "[3/3] Testing: reward_mean (enhanced)"
echo "========================================="
python main_ptr_ablation.py \
  --env_name=$ENV_NAME \
  --seed=$SEED \
  --use_ptr_backward=True \
  --use_ptr_online_priority=True \
  --ptr_priority_metric=reward_mean \
  --offline_steps=$OFFLINE_STEPS \
  --online_steps=$ONLINE_STEPS \
  --eval_interval=$EVAL_INTERVAL \
  --eval_episodes=10 \
  --run_group=PTR_QuickTest \
  2>&1 | tee logs/quick_reward_mean.log

echo ""
echo "========================================="
echo "Quick Test Complete!"
echo "========================================="
echo ""
echo "Check results in W&B:"
echo "  Project: qc"
echo "  Group: PTR_QuickTest"
echo ""
echo "Expected differences:"
echo "  - uniform: Lowest success rate (baseline)"
echo "  - success_binary: Medium success rate (~10-20% better)"
echo "  - reward_mean: Highest success rate (~20-40% better)"
echo ""
echo "Next steps:"
echo "  1. Check W&B dashboard for comparison"
echo "  2. If reward_mean performs best, run full ablation"
echo "  3. If needed, test weighted critic variant"