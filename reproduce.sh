#!/bin/bash
### AIME24 & AIME25

# 使用 set -e 但允许某些命令失败
set -e
set -o pipefail

# 🔧 加载环境变量
if [ -f .env ]; then
    echo "正在加载 .env 文件..."
    set -a  # 自动导出所有变量
    source .env
    set +a
    echo "✅ 环境变量已加载"
else
    echo "⚠️  警告: .env 文件不存在！"
fi

echo ""

# ⭐ 配置并发参数
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-1}"          # 评估并发
PRACTICE_CONCURRENCY="${PRACTICE_CONCURRENCY:-1}"     # 训练并发
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-1}"            # 评判并发

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

echo "==================================="
echo "  环境变量配置:"
echo "  - LLM 类型: ${UTU_LLM_TYPE}"
echo "  - LLM 模型: ${UTU_LLM_MODEL}"
echo "  - LLM BASE URL: ${UTU_LLM_BASE_URL}"
echo "  - 数据库 URL: ${UTU_DB_URL}"
echo "  - Judge LLM 类型: ${JUDGE_LLM_TYPE:-未设置}"
echo "  - Judge LLM 模型: ${JUDGE_LLM_MODEL:-未设置}"
echo "==================================="
echo "  [2.1] 评估 AIME24 基线..."
uv run scripts/run_eval.py \
  --config_name math/math_AIME24 \
  --concurrency "$EVAL_CONCURRENCY" \
  --judge_concurrency "$JUDGE_CONCURRENCY"

echo "  [2.2] 评估 AIME25 基线..."
uv run scripts/run_eval.py \
  --config_name math/math_AIME25 \
  --concurrency "$EVAL_CONCURRENCY" \
  --judge_concurrency "$JUDGE_CONCURRENCY" || {
    echo "⚠️  AIME25 评估失败，继续执行..."
  }

# # # 3. Training-Free GRPO - 使用自定义并发
# echo "步骤 3/5: 运行 Training-Free GRPO..."
# uv run scripts/run_training_free_GRPO.py \
#   --config_name math_reasoning \
#   --experiment_name math_practice \
#   --epochs 3 \
#   --grpo_n 5 \
#   --rollout_concurrency $PRACTICE_CONCURRENCY

# # # 4. 评估增强版
# # echo "步骤 4/5: 评估增强后性能..."

# # echo "  [4.1] 评估 AIME24（增强版）..."
# uv run scripts/run_eval.py \
#   --config_name math/math_practice_AIME24 \
#   --concurrency $EVAL_CONCURRENCY \
#   --judge_concurrency $JUDGE_CONCURRENCY

# echo "  [4.2] 评估 AIME25（增强版）..."
# uv run scripts/run_eval.py \
#   --config_name math/math_practice_AIME25 \
#   --concurrency $EVAL_CONCURRENCY \
#   --judge_concurrency $JUDGE_CONCURRENCY

# echo "步骤 5/5: 评估完成！"

# ### WebWalkerQA 完整评估流程
# echo ""
# echo "==================================="
# echo "  WebWalkerQA 评估流程"
# echo "==================================="

# # 6. 评估 WebWalkerQA 基线
# echo "步骤 6/9: 评估 WebWalkerQA 基线性能..."
# uv run scripts/run_eval.py \
#   --config_name web/web \
#   --concurrency $EVAL_CONCURRENCY \
#   --judge_concurrency $JUDGE_CONCURRENCY

# # 7. Training-Free GRPO for WebWalkerQA
# echo "步骤 7/9: 运行 WebWalkerQA Training-Free GRPO..."
# uv run scripts/run_training_free_GRPO.py \
#   --config_name web_search \
#   --experiment_name web_practice \
#   --epochs 3 \
#   --grpo_n 5 \
#   --rollout_concurrency $PRACTICE_CONCURRENCY

# # 8. 评估 WebWalkerQA 增强版
# echo "步骤 8/9: 评估 WebWalkerQA 增强后性能..."
# uv run scripts/run_eval.py \
#   --config_name web/web_practice \
#   --concurrency $EVAL_CONCURRENCY \
#   --judge_concurrency $JUDGE_CONCURRENCY
# echo "步骤 9/9: 所有评估完成！"