#!/bin/bash
### AIME24 & AIME25 完整评估流程（优化并发）

set -e

# ⭐ 配置并发参数
EVAL_CONCURRENCY=512          # 评估并发
PRACTICE_CONCURRENCY=512     # 训练并发
JUDGE_CONCURRENCY=512            # 评判并发

echo "==================================="
echo "  并发配置:"
echo "  - 评估并发: $EVAL_CONCURRENCY"
echo "  - 训练并发: $PRACTICE_CONCURRENCY"
echo "  - 评判并发: $JUDGE_CONCURRENCY"
echo "==================================="

# 1. 准备数据
echo "步骤 1/5: 准备训练和评估数据..."
uv run scripts/data/process_training_free_GRPO_data.py

# 2. 评估基线 - 使用自定义并发
echo "步骤 2/5: 评估基线性能..."

echo "  [2.1] 评估 AIME24 基线..."
uv run scripts/run_eval.py \
  --config_name math/math_AIME24 \
  --concurrency $EVAL_CONCURRENCY \
  --judge_concurrency $JUDGE_CONCURRENCY

echo "  [2.2] 评估 AIME25 基线..."
uv run scripts/run_eval.py \
  --config_name math/math_AIME25 \
  --concurrency $EVAL_CONCURRENCY \
  --judge_concurrency $JUDGE_CONCURRENCY

# 3. Training-Free GRPO - 使用自定义并发
echo "步骤 3/5: 运行 Training-Free GRPO..."
uv run scripts/run_training_free_GRPO.py \
  --config_name math_reasoning \
  --experiment_name my_first_practice \
  --epochs 3 \
  --batch_size 50 \
  --grpo_n 5 \
  --rollout_concurrency $PRACTICE_CONCURRENCY

# 4. 评估增强版
echo "步骤 4/5: 评估增强后性能..."

echo "  [4.1] 评估 AIME24（增强版）..."
uv run scripts/run_eval.py \
  --config_name math/math_practice_AIME24 \
  --agent_config agents/practice/math_agent_my_first_practice \
  --concurrency $EVAL_CONCURRENCY \
  --judge_concurrency $JUDGE_CONCURRENCY

echo "  [4.2] 评估 AIME25（增强版）..."
uv run scripts/run_eval.py \
  --config_name math/math_practice_AIME25 \
  --agent_config agents/practice/math_agent_my_first_practice \
  --concurrency $EVAL_CONCURRENCY \
  --judge_concurrency $JUDGE_CONCURRENCY

echo "步骤 5/5: 评估完成！"