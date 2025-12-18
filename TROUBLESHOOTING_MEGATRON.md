# Megatron 训练故障排查指南

## 快速诊断工具

```bash
# 运行综合诊断
./diagnose_training_error.sh

# 检查 checkpoint 状态
python debug_resume.py /path/to/checkpoint/vx-xxx

# 检查缓存数据集
python inspect_cached_dataset.py /path/to/cache_dir/train
```

---

## 常见错误及解决方案

### 1. ChildFailedError (exitcode: 1)

**症状**: 训练进程异常退出，错误信息为 `ChildFailedError`

**可能原因**:

#### 1.1 CUDA Out of Memory (显存不足)

**识别方法**:
```bash
nvidia-smi  # 检查显存使用情况
```

**解决方案**:
```bash
# 方法 1: 减小 batch size
--micro_batch_size 8  # 从 16 减到 8

# 方法 2: 增加重计算
--recompute_granularity full
--recompute_num_layers 4  # 增加重计算层数

# 方法 3: 使用梯度检查点
--use_checkpoint true
```

#### 1.2 TransformerEngine 注意力后端错误

**错误信息**: `No dot product attention backend is available`

**解决方案**:
```bash
# 添加以下参数
--attention_backend local
```

这会使用原生 Megatron 注意力实现，绕过 TransformerEngine。

#### 1.3 Checkpoint 加载失败

**识别方法**:
```bash
# 检查 checkpoint 完整性
python debug_resume.py /path/to/checkpoint/vx-xxx
```

**常见问题**:
- ✅ **args.json 不存在**: 确保使用包含 args.json 的 vx-xxx 目录
- ✅ **model_type 未找到**: 已修复（见 swift/megatron/argument/train_args.py）
- ✅ **多机训练时其他节点无法访问**: 使用共享存储，确保所有节点都能访问

**解决方案**:
```bash
# 确保使用正确的 checkpoint 目录
--load /shared/path/to/checkpoint/vx-xxx  # 包含 args.json 的目录

# 多机训练时检查每个节点
# 在每个节点上运行:
ls -la /shared/path/to/checkpoint/vx-xxx/args.json
```

#### 1.4 数据集路径错误

**症状**: 找不到数据集或数据加载失败

**解决方案**:
```bash
# 使用缓存数据集时，指向包含 .arrow 文件的目录
--cached_dataset '/path/to/cache_dir/train'
--cached_val_dataset '/path/to/cache_dir/val'

# 检查数据集内容
python inspect_cached_dataset.py /path/to/cache_dir/train
```

#### 1.5 张量并行配置不匹配

**症状**: 恢复训练时提示张量并行配置错误

**解决方案**:
```bash
# 确保与原训练配置一致
--tensor_model_parallel_size 2  # 必须与 checkpoint 保存时一致
--sequence_parallel true  # 必须与 checkpoint 保存时一致
```

---

### 2. 多机训练问题

#### 2.1 节点间通信失败

**检查清单**:
```bash
# 1. 检查网络连接
ping <other_node_ip>

# 2. 检查共享存储
df -h | grep /shared

# 3. 检查 checkpoint 访问（在每个节点上）
ls -la /shared/checkpoint/vx-xxx/args.json

# 4. 检查环境变量
echo $MASTER_ADDR
echo $MASTER_PORT
echo $NODE_RANK
```

**正确配置示例**:
```bash
# Node 0 (主节点)
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export NODE_RANK=0
export NNODES=2

# Node 1
export MASTER_ADDR=10.0.0.1  # 与主节点相同
export MASTER_PORT=29500      # 与主节点相同
export NODE_RANK=1            # 不同节点有不同的 RANK
export NNODES=2               # 与主节点相同
```

---

### 3. 数据采样和恢复问题

#### 3.1 确保从正确位置继续训练

**关键参数**:
```bash
# 推荐配置（续训）
--load /path/to/checkpoint
--no_load_optim true   # 不加载优化器（节省内存）
--no_load_rng false    # 加载 RNG 状态（保持数据顺序）

# 微调场景
--load /path/to/checkpoint
--no_load_optim true   # 不加载优化器
--no_load_rng true     # 重新打乱数据
--finetune true
```

**数据采样机制**:
- `iteration` 从 checkpoint 加载（例如 1400）
- `consumed_train_samples` = iteration × global_batch_size
- 数据加载器会自动跳过前面已消费的样本
- `RNG 状态` 控制数据打乱顺序

---

### 4. 性能优化建议

#### 4.1 减少显存使用
```bash
--micro_batch_size 8              # 减小 batch size
--recompute_granularity full      # 全重计算
--recompute_num_layers 4          # 增加重计算层数
--no_save_optim true              # 不保存优化器
```

#### 4.2 加速训练
```bash
--use_liger_kernel true           # 使用 liger 内核
--cross_entropy_loss_fusion true  # 损失融合
--sequence_parallel true          # 序列并行
```

---

## 调试技巧

### 启用详细日志

```bash
# 在训练前设置环境变量
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE=/tmp/torch_error_rank_${RANK:-0}.log
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 单节点测试

```bash
# 先在单节点上测试是否能正常恢复
PYTORCH_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load /path/to/checkpoint/vx-xxx \
    --save /path/to/save_dir \
    --cached_dataset '/path/to/cache_dir/train' \
    --no_load_optim true \
    --no_load_rng false \
    --attention_backend local \
    ... (其他参数)
```

### 查看错误日志

```bash
# 查看 torch 错误日志
cat /tmp/torch_error_rank_*.log

# 查看最近的日志
find /tmp -name "torch_error*.log" -mmin -60 -exec cat {} \;
```

---

## 完整的恢复训练示例

```bash
#!/bin/bash

# 环境配置
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE=/tmp/torch_error_rank_${RANK:-0}.log
export NCCL_DEBUG=INFO
export MODELSCOPE_CACHE=/shared/modelscope_cache

# 训练参数
CHECKPOINT_DIR="/shared/checkpoint/vx-xxx"
SAVE_DIR="/shared/checkpoint"
CACHED_DATASET="/shared/cache_dir/train"
CACHED_VAL_DATASET="/shared/cache_dir/val"

# 多机配置
NNODES=2
NPROC_PER_NODE=8
MASTER_ADDR="10.0.0.1"
MASTER_PORT=29500

# 启动训练
PYTORCH_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$NPROC_PER_NODE \
NNODES=$NNODES \
NODE_RANK=${NODE_RANK:-0} \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
megatron sft \
    --load "$CHECKPOINT_DIR" \
    --save "$SAVE_DIR" \
    --cached_dataset "$CACHED_DATASET" \
    --cached_val_dataset "$CACHED_VAL_DATASET" \
    --no_load_optim true \
    --no_load_rng false \
    --attention_backend local \
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
```

---

## 相关资源

- **诊断脚本**: `./diagnose_training_error.sh`
- **恢复脚本**: `./resume_megatron_training.sh`
- **Checkpoint 检查**: `python debug_resume.py <checkpoint_dir>`
- **数据集检查**: `python inspect_cached_dataset.py <cache_dir>`
- **数据采样说明**: `python check_resume_data.py <checkpoint_dir>`

---

## 常见问题 FAQ

**Q: 恢复训练时会从哪个 iteration 开始？**
A: 从 checkpoint 保存的 iteration 继续（例如从 1400 继续到 1401, 1402...）

**Q: 使用 `--no_load_optim true` 是否会影响训练效果？**
A: 不会影响数据采样位置，但优化器状态会重新初始化，学习率调度器会从当前 iteration 对应的值开始。

**Q: 多机训练时如何确保数据一致性？**
A: 使用共享存储路径，确保所有节点都能访问相同的 checkpoint 和数据集。

**Q: 如何减少显存占用？**
A: 减小 `micro_batch_size`，增加 `recompute_num_layers`，使用 `--recompute_granularity full`。

**Q: TransformerEngine 错误怎么办？**
A: 添加 `--attention_backend local` 参数使用原生 Megatron 注意力实现。
