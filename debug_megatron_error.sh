#!/bin/bash
# Megatron 训练错误诊断脚本

echo "============================================================"
echo "Megatron 分布式训练错误诊断"
echo "============================================================"
echo ""
echo "检测到 rank 51 (local_rank: 3) 训练失败"
echo ""
echo "常见原因："
echo "1. 多机节点间的 checkpoint 文件不一致或无法访问"
echo "2. args.json 文件在某些节点上不存在"
echo "3. 数据集路径在某些节点上无法访问"
echo "4. 网络连接问题导致节点间通信失败"
echo "5. 内存或显存不足"
echo ""
echo "============================================================"
echo "步骤 1: 启用详细错误日志"
echo "============================================================"
echo ""
echo "在你的训练命令前添加以下环境变量："
echo ""
echo "export TORCH_DISTRIBUTED_DEBUG=DETAIL"
echo "export TORCHELASTIC_ERROR_FILE=/tmp/torch_error.log"
echo "export NCCL_DEBUG=INFO"
echo ""
echo "完整示例："
echo ""
cat << 'EOF'
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE=/tmp/torch_error_\${RANK}.log
export NCCL_DEBUG=INFO

PYTORCH_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
NNODES=<你的节点数> \
NODE_RANK=\${NODE_RANK} \
MASTER_ADDR=\${MASTER_ADDR} \
MASTER_PORT=\${MASTER_PORT} \
megatron sft \
    --load <your_checkpoint> \
    --save <your_save_dir> \
    ...
EOF
echo ""
echo "============================================================"
echo "步骤 2: 检查多机环境"
echo "============================================================"
echo ""
echo "A. 检查所有节点是否能访问 checkpoint:"
echo "   在每个节点上运行："
echo "   ls -la /path/to/checkpoint/vx-xxx/args.json"
echo ""
echo "B. 检查共享存储是否正常挂载："
echo "   df -h | grep <共享存储挂载点>"
echo ""
echo "C. 检查节点间网络连接："
echo "   # 在 node0 上运行"
echo "   ping <node1_ip>"
echo ""
echo "============================================================"
echo "步骤 3: 常见问题快速修复"
echo "============================================================"
echo ""
echo "问题 1: args.json 不存在"
echo "  解决：确保所有节点都能访问到包含 args.json 的目录"
echo "  检查：python debug_resume.py /shared/path/to/checkpoint"
echo ""
echo "问题 2: model_type 未找到"
echo "  解决：显式指定 --model 和 --model_type"
echo "  示例：--model /path/to/model --model_type qwen2"
echo ""
echo "问题 3: 缓存数据集路径问题"
echo "  解决：使用绝对路径的共享存储路径"
echo "  示例：--cached_dataset '/shared/cache_dir/train'"
echo ""
echo "问题 4: 内存/显存不足"
echo "  检查：nvidia-smi"
echo "  解决：减少 micro_batch_size 或启用更多重计算"
echo ""
echo "============================================================"
echo "步骤 4: 单节点测试"
echo "============================================================"
echo ""
echo "先在单节点上测试是否能正常恢复："
echo ""
cat << 'EOF'
PYTORCH_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load /shared/checkpoint/vx-xxx \
    --save /shared/checkpoint \
    --cached_dataset '/shared/cache_dir/train' \
    --no_load_optim true \
    --no_load_rng false \
    ... (其他参数)
EOF
echo ""
echo "如果单节点成功，再尝试多节点"
echo ""
echo "============================================================"
echo "步骤 5: 查看详细错误日志"
echo "============================================================"
echo ""
echo "运行训练后，查看每个节点的错误日志："
echo "  cat /tmp/torch_error_*.log"
echo ""
echo "查看 rank 51 所在节点的具体错误"
echo ""
