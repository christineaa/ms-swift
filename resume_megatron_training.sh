#!/bin/bash
# Megatron 恢复训练脚本（带完整错误诊断）

# ============================================================
# 配置区域 - 请根据你的实际情况修改
# ============================================================

# Checkpoint 路径（必须是所有节点都能访问的共享路径）
CHECKPOINT_DIR="/shared/path/to/checkpoint/vx-xxx"  # 修改这里！

# 保存路径
SAVE_DIR="/shared/path/to/checkpoint"  # 修改这里！

# 缓存数据集路径（可选，如果使用缓存数据集）
CACHED_DATASET="/shared/cache_dir/train"  # 修改这里！或留空
CACHED_VAL_DATASET="/shared/cache_dir/val"  # 修改这里！或留空

# 多机配置
NNODES=2  # 节点数量，修改这里！
NPROC_PER_NODE=8  # 每个节点的GPU数量
MASTER_ADDR="10.0.0.1"  # 主节点IP，修改这里！
MASTER_PORT=29500  # 主节点端口

# ============================================================
# 启用详细错误日志
# ============================================================
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE="/tmp/torch_error_rank_${RANK:-0}.log"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 设置共享缓存（多机训练必须！）
export MODELSCOPE_CACHE="/shared/modelscope_cache"  # 修改这里！

# ============================================================
# 预检查
# ============================================================
echo "============================================================"
echo "Megatron 训练恢复 - 预检查"
echo "============================================================"

# 检查 checkpoint 目录
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ 错误: checkpoint 目录不存在: $CHECKPOINT_DIR"
    echo "   请检查路径是否正确，以及是否在所有节点上都能访问"
    exit 1
fi

# 检查 args.json
if [ ! -f "$CHECKPOINT_DIR/args.json" ]; then
    echo "❌ 错误: args.json 不存在: $CHECKPOINT_DIR/args.json"
    echo "   请确保使用正确的 checkpoint 目录（包含 args.json 的 vx-xxx 目录）"
    exit 1
fi

echo "✅ Checkpoint 目录: $CHECKPOINT_DIR"
echo "✅ args.json 存在"

# 检查最新的 iteration
if [ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]; then
    LATEST_ITER=$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")
    echo "✅ 最新 iteration: $LATEST_ITER"
else
    echo "⚠️  警告: latest_checkpointed_iteration.txt 不存在"
fi

# 检查缓存数据集（如果指定）
if [ -n "$CACHED_DATASET" ] && [ "$CACHED_DATASET" != "/shared/cache_dir/train" ]; then
    if [ ! -d "$CACHED_DATASET" ]; then
        echo "❌ 错误: 缓存数据集目录不存在: $CACHED_DATASET"
        exit 1
    fi
    echo "✅ 缓存数据集: $CACHED_DATASET"
fi

echo ""
echo "============================================================"
echo "开始恢复训练"
echo "============================================================"
echo "节点数: $NNODES"
echo "每节点GPU数: $NPROC_PER_NODE"
echo "主节点: $MASTER_ADDR:$MASTER_PORT"
echo "当前节点 RANK: ${NODE_RANK:-未设置}"
echo ""

# ============================================================
# 启动训练
# ============================================================

# 构建数据集参数
DATASET_ARGS=""
if [ -n "$CACHED_DATASET" ] && [ "$CACHED_DATASET" != "/shared/cache_dir/train" ]; then
    DATASET_ARGS="--cached_dataset '$CACHED_DATASET'"
    if [ -n "$CACHED_VAL_DATASET" ] && [ "$CACHED_VAL_DATASET" != "/shared/cache_dir/val" ]; then
        DATASET_ARGS="$DATASET_ARGS --cached_val_dataset '$CACHED_VAL_DATASET'"
    fi
else
    echo "⚠️  警告: 未指定缓存数据集，请确保在命令中添加 --dataset 参数"
fi

# 使用 PYTORCH_ALLOC_CONF 代替已废弃的 PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$NPROC_PER_NODE \
NNODES=$NNODES \
NODE_RANK=${NODE_RANK:-0} \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
megatron sft \
    --load "$CHECKPOINT_DIR" \
    --save "$SAVE_DIR" \
    $DATASET_ARGS \
    --no_load_optim true \
    --no_load_rng false \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save_interval 100 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng false

# ============================================================
# 错误处理
# ============================================================
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
    echo "============================================================"
    echo ""
    echo "请检查以下错误日志："
    echo "  1. 当前终端的输出"
    echo "  2. /tmp/torch_error_rank_*.log"
    echo "  3. tensorboard 日志（如果有）"
    echo ""
    echo "常见问题排查："
    echo "  1. 检查所有节点是否都能访问 checkpoint: $CHECKPOINT_DIR"
    echo "  2. 检查 args.json 是否存在: $CHECKPOINT_DIR/args.json"
    echo "  3. 检查数据集路径是否正确"
    echo "  4. 检查节点间网络连接"
    echo "  5. 运行单节点测试: NNODES=1 NODE_RANK=0 ./resume_megatron_training.sh"
    echo ""

    # 尝试显示错误日志
    if [ -f "/tmp/torch_error_rank_${NODE_RANK:-0}.log" ]; then
        echo "节点 ${NODE_RANK:-0} 的错误日志："
        cat "/tmp/torch_error_rank_${NODE_RANK:-0}.log"
    fi

    exit $EXIT_CODE
else
    echo ""
    echo "============================================================"
    echo "✅ 训练成功完成或正常退出"
    echo "============================================================"
fi
