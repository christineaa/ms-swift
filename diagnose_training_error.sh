#!/bin/bash
# Megatron 训练错误综合诊断脚本

echo "============================================================"
echo "Megatron 训练错误诊断工具"
echo "============================================================"
echo ""

# 检查 1: CUDA 和 GPU 状态
echo "【1】检查 GPU 状态"
echo "------------------------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits | \
    while IFS=, read -r idx name total free used; do
        echo "  GPU $idx: $name"
        echo "    总显存: ${total} MB"
        echo "    可用显存: ${free} MB"
        echo "    已用显存: ${used} MB"
        usage_pct=$((used * 100 / total))
        if [ $usage_pct -gt 90 ]; then
            echo "    ⚠️  警告: 显存使用率 ${usage_pct}%，可能OOM"
        fi
    done
else
    echo "  ❌ nvidia-smi 不可用"
fi
echo ""

# 检查 2: Python 环境和关键依赖
echo "【2】检查 Python 环境"
echo "------------------------------------------------------------"
echo "  Python 版本: $(python --version 2>&1)"
echo "  pip 路径: $(which pip)"
echo ""
echo "  关键依赖版本:"
for pkg in torch transformers datasets megatron-core ms-swift; do
    version=$(pip show $pkg 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
    if [ -n "$version" ]; then
        echo "    ✅ $pkg: $version"
    else
        echo "    ❌ $pkg: 未安装"
    fi
done
echo ""

# 检查 3: TransformerEngine
echo "【3】检查 TransformerEngine"
echo "------------------------------------------------------------"
te_version=$(pip show transformer-engine 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
if [ -n "$te_version" ]; then
    echo "  ✅ transformer-engine: $te_version"
    echo "  💡 提示: 如果遇到 attention backend 错误，使用 --attention_backend local"
else
    echo "  ℹ️  transformer-engine: 未安装 (使用 --attention_backend local)"
fi
echo ""

# 检查 4: 环境变量
echo "【4】检查环境变量"
echo "------------------------------------------------------------"
env_vars=(
    "CUDA_VISIBLE_DEVICES"
    "NPROC_PER_NODE"
    "NNODES"
    "NODE_RANK"
    "MASTER_ADDR"
    "MASTER_PORT"
    "PYTORCH_ALLOC_CONF"
    "PYTORCH_CUDA_ALLOC_CONF"
    "TORCH_DISTRIBUTED_DEBUG"
    "NCCL_DEBUG"
)

for var in "${env_vars[@]}"; do
    val="${!var}"
    if [ -n "$val" ]; then
        echo "  ✅ $var=$val"
    else
        echo "  ➖ $var: (未设置)"
    fi
done
echo ""

# 检查 5: 最近的错误日志
echo "【5】搜索最近的错误日志"
echo "------------------------------------------------------------"
echo "  检查 /tmp/ 目录..."
torch_errors=$(find /tmp -name "torch_error*.log" -mmin -60 2>/dev/null)
if [ -n "$torch_errors" ]; then
    echo "  ✅ 找到 torch 错误日志:"
    for log in $torch_errors; do
        echo "    - $log"
        echo "      最后 20 行:"
        tail -n 20 "$log" | sed 's/^/        /'
    done
else
    echo "  ℹ️  未找到最近的 torch 错误日志"
fi
echo ""

# 检查 6: 进程状态
echo "【6】检查训练相关进程"
echo "------------------------------------------------------------"
training_procs=$(ps aux | grep -E "(python.*megatron|swift.*sft)" | grep -v grep)
if [ -n "$training_procs" ]; then
    echo "  ⚠️  发现运行中的训练进程:"
    echo "$training_procs" | sed 's/^/    /'
else
    echo "  ✅ 没有运行中的训练进程"
fi
echo ""

# 检查 7: 常见问题诊断
echo "【7】常见问题快速诊断"
echo "============================================================"
echo ""

echo "问题 1: ChildFailedError exitcode 1"
echo "------------------------------------------------------------"
echo "  常见原因:"
echo "    ❌ CUDA OOM (显存不足)"
echo "       解决: 减少 micro_batch_size 或增加重计算"
echo ""
echo "    ❌ 注意力后端不兼容"
echo "       解决: 添加 --attention_backend local"
echo ""
echo "    ❌ Checkpoint 加载失败"
echo "       解决: 检查 checkpoint 路径和 args.json"
echo ""
echo "    ❌ 数据集路径错误"
echo "       解决: 检查 --cached_dataset 路径是否正确"
echo ""
echo "    ❌ 张量并行配置不匹配"
echo "       解决: 确保 tensor_model_parallel_size 与 checkpoint 一致"
echo ""

echo "问题 2: TransformerEngine attention backend 错误"
echo "------------------------------------------------------------"
echo "  错误信息: 'No dot product attention backend is available'"
echo "  解决方案:"
echo "    添加参数: --attention_backend local"
echo ""

echo "问题 3: 多机训练节点间通信失败"
echo "------------------------------------------------------------"
echo "  检查清单:"
echo "    □ 所有节点能访问 checkpoint 目录"
echo "    □ args.json 在所有节点上都存在"
echo "    □ MASTER_ADDR 和 MASTER_PORT 正确配置"
echo "    □ 节点间网络可达 (ping 测试)"
echo "    □ 共享存储正确挂载"
echo ""

echo "============================================================"
echo "建议的下一步操作"
echo "============================================================"
echo ""
echo "1️⃣  查看完整错误信息:"
echo "   运行训练时添加详细日志:"
echo "   export TORCH_DISTRIBUTED_DEBUG=DETAIL"
echo "   export TORCHELASTIC_ERROR_FILE=/tmp/torch_error_\${RANK:-0}.log"
echo "   export NCCL_DEBUG=INFO"
echo ""
echo "2️⃣  尝试最小化配置测试:"
echo "   - 单节点、单 GPU 测试"
echo "   - 减小 batch size"
echo "   - 使用 --attention_backend local"
echo ""
echo "3️⃣  如果是显存问题:"
echo "   - 减少 --micro_batch_size"
echo "   - 增加 --recompute_num_layers"
echo "   - 使用 --recompute_granularity full"
echo ""
echo "4️⃣  检查 checkpoint 完整性:"
echo "   python debug_resume.py /path/to/checkpoint/vx-xxx"
echo ""

echo "============================================================"
