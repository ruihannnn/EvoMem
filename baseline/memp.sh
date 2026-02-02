#!/bin/bash
### MemP Baseline Evaluation Script
### 仿照 training-free GRPO 的评测流程，评估 MemP 方法在数学和 Web 任务上的表现

# cd /Users/liruihan/youtu-agent

set -e
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
EVAL_CONCURRENCY=64          # 评估并发
PRACTICE_CONCURRENCY=64     # 训练并发
JUDGE_CONCURRENCY=16            # 评判并发

echo "==================================="
echo "  并发配置:"
echo "  - 评估并发: $EVAL_CONCURRENCY"
echo "  - 训练并发: $PRACTICE_CONCURRENCY"
echo "  - 评判并发: $JUDGE_CONCURRENCY"
echo "==================================="


echo "==================================="
echo "  环境变量配置:"
echo "  - LLM 类型: ${UTU_LLM_TYPE}"
echo "  - LLM 模型: ${UTU_LLM_MODEL}"
echo "  - LLM BASE URL: ${UTU_LLM_BASE_URL}"
echo "  - 数据库 URL: ${UTU_DB_URL}"
echo "  - Judge LLM 类型: ${JUDGE_LLM_TYPE:-未设置}"
echo "  - Judge LLM 模型: ${JUDGE_LLM_MODEL:-未设置}"
echo "==================================="


### 数学推理任务 (AIME24 & AIME25)
echo ""
echo "==================================="
echo "  数学推理任务评测流程"
echo "==================================="



# 3. 运行 MemP 学习
echo "步骤 4/9: 运行 MemP 学习过程..."
uv run scripts/run_memp.py \
  --config_name memp_math_reasoning \
  --experiment_name memp_math_practice \
  --epochs 3 \
  --batch_size 50 \
  --rollout_concurrency $PRACTICE_CONCURRENCY \
  --memory_build_policy direct \
  --memory_retrieve_policy query \
  --memory_update_policy reflect \
  --memory_retrieve_num 5 \
  --memory_size 300 \
  --memory_dir /Users/liruihan/youtu-agent/memory/memp/aime \
  --do_eval false

# 4. 评估 MemP 增强后的性能
echo "步骤 5/9: 评估 AIME24（MemP 增强版）..."
uv run scripts/run_memp.py \
  --config_name memp_math_reasoning \
  --experiment_name memp_math_eval_aime24 \
  --epochs 1 \
  --batch_size 50 \
  --rollout_concurrency $PRACTICE_CONCURRENCY \
  --memory_build_policy direct \
  --memory_retrieve_policy query \
  --memory_update_policy none \
  --memory_retrieve_num 5 \
  --memory_size 300 \
  --memory_dir /Users/liruihan/youtu-agent/memory/memp/aime \
  --do_eval true \
  --eval_use_memory true \
  --eval_dataset AIME24

echo "步骤 6/9: 评估 AIME25（MemP 增强版）..."
uv run scripts/run_memp.py \
  --config_name memp_math_reasoning \
  --experiment_name memp_math_eval_aime25 \
  --epochs 1 \
  --batch_size 50 \
  --rollout_concurrency $PRACTICE_CONCURRENCY \
  --memory_build_policy direct \
  --memory_retrieve_policy query \
  --memory_update_policy none \
  --memory_retrieve_num 5 \
  --memory_size 300 \
  --memory_dir /Users/liruihan/youtu-agent/memory/memp/aime \
  --do_eval true \
  --eval_use_memory true \
  --eval_dataset AIME25

echo "步骤 7/9: 数学推理任务评测完成！"

### WebWalkerQA 任务
echo ""
echo "==================================="
echo "  WebWalkerQA 评测流程"
echo "==================================="

# 6. 运行 MemP for WebWalkerQA
echo "步骤 9/9: 运行 WebWalkerQA MemP 学习..."
uv run scripts/run_memp.py \
  --config_name memp_web_search \
  --experiment_name memp_web_practice \
  --epochs 3 \
  --batch_size 50 \
  --rollout_concurrency $PRACTICE_CONCURRENCY \
  --memory_build_policy direct \
  --memory_retrieve_policy query \
  --memory_update_policy reflect \
  --memory_retrieve_num 5 \
  --memory_size 300 \
  --memory_dir /Users/liruihan/youtu-agent/memory/WebWalkerQA \
  --do_eval false

# 7. 评估 WebWalkerQA MemP 增强版
echo "步骤 10/11: 评估 WebWalkerQA（MemP 增强版）..."
uv run scripts/run_memp.py \
  --config_name memp_web_search \
  --experiment_name memp_web_eval \
  --epochs 1 \
  --batch_size 50 \
  --rollout_concurrency $PRACTICE_CONCURRENCY \
  --memory_build_policy direct \
  --memory_retrieve_policy query \
  --memory_update_policy none \
  --memory_retrieve_num 5 \
  --memory_size 300 \
  --memory_dir /Users/liruihan/youtu-agent/memory/WebWalkerQA \
  --do_eval true \
  --eval_use_memory true \
  --eval_dataset WebWalkerQA

echo "步骤 11/11: 所有评估完成！"

# 8. 生成对比报告
echo "==================================="
echo "  MemP Baseline 评测完成"
echo "==================================="
echo "请查看以下结果："
echo "  - 数学任务: configs/agents/practice/memp_math_practice_memp_agent.yaml"
echo "  - Web 任务: configs/agents/practice/memp_web_practice_memp_agent.yaml"
echo "  - 记忆存储: workspace/memp_memory/"
echo "==================================="
