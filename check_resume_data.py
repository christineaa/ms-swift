#!/usr/bin/env python3
"""
检查 Megatron checkpoint 恢复时的数据采样情况

这个脚本帮助理解：
1. checkpoint 保存了哪些状态
2. 使用 --no_load_optim 和 --no_load_rng 时的影响
3. 数据采样是否会从正确的位置开始
"""
import os
import sys
import json

def check_checkpoint_state(checkpoint_dir):
    """检查 checkpoint 保存的状态信息"""
    print("=" * 80)
    print("检查 Megatron Checkpoint 状态")
    print("=" * 80)

    # 查找最新的 checkpoint
    latest_file = os.path.join(checkpoint_dir, 'latest_checkpointed_iteration.txt')
    if not os.path.exists(latest_file):
        print(f"❌ 未找到 latest_checkpointed_iteration.txt")
        return

    with open(latest_file, 'r') as f:
        latest_iter = f.read().strip()

    print(f"\n✅ 最新 checkpoint: iteration {latest_iter}")

    # 检查 checkpoint 目录
    iter_dir = os.path.join(checkpoint_dir, f'iter_{latest_iter.zfill(7)}')
    if not os.path.exists(iter_dir):
        print(f"❌ checkpoint 目录不存在: {iter_dir}")
        return

    print(f"✅ checkpoint 目录: {iter_dir}\n")

    # 列出 checkpoint 中的文件
    print("=" * 80)
    print("Checkpoint 文件列表")
    print("=" * 80)

    files = []
    for root, dirs, filenames in os.walk(iter_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, iter_dir)
            size = os.path.getsize(filepath)
            files.append((rel_path, size))

    files.sort()
    for rel_path, size in files:
        size_mb = size / (1024 * 1024)
        print(f"  {rel_path:<50} {size_mb:>10.2f} MB")

    # 检查是否保存了 RNG 状态
    print("\n" + "=" * 80)
    print("关键状态检查")
    print("=" * 80)

    has_rng = any('rng' in f[0].lower() for f in files)
    has_optim = any('optim' in f[0].lower() or 'optimizer' in f[0].lower() for f in files)

    print(f"\n{'✅' if has_rng else '❌'} RNG 状态: {'已保存' if has_rng else '未保存'}")
    print(f"{'✅' if has_optim else '❌'} 优化器状态: {'已保存' if has_optim else '未保存'}")

    # 读取 args.json 查看训练参数
    args_file = os.path.join(checkpoint_dir, 'args.json')
    if os.path.exists(args_file):
        print("\n" + "=" * 80)
        print("训练参数 (args.json)")
        print("=" * 80)

        with open(args_file, 'r') as f:
            args = json.load(f)

        key_params = [
            'seed',
            'data_seed',
            'no_save_optim',
            'no_save_rng',
            'micro_batch_size',
            'global_batch_size',
            'train_iters',
        ]

        for key in key_params:
            if key in args:
                print(f"  {key}: {args[key]}")

    print("\n" + "=" * 80)
    print("恢复训练的建议")
    print("=" * 80)

    print("\n1️⃣  完全恢复（推荐用于续训）:")
    print("   不加载优化器，但保持数据顺序一致：")
    print("   --load /path/to/checkpoint")
    print("   --no_load_optim true")
    print("   --no_load_rng false  # 加载 RNG 状态，保证数据顺序")

    print("\n2️⃣  微调场景（从 checkpoint 开始新训练）:")
    print("   --load /path/to/checkpoint")
    print("   --no_load_optim true")
    print("   --no_load_rng true   # 不加载 RNG，数据会重新打乱")
    print("   --finetune true")

    print("\n3️⃣  完全续训（包括优化器）:")
    print("   --load /path/to/checkpoint")
    print("   --no_load_optim false")
    print("   --no_load_rng false")

    print("\n" + "=" * 80)
    print("关键知识点")
    print("=" * 80)

    print("""
📌 数据采样机制：

Megatron 使用以下方式确定数据采样位置：

1. **consumed_train_samples** (已消费的训练样本数)
   = iteration × global_batch_size

2. **RNG 状态** (随机数生成器状态)
   - 控制数据的打乱顺序
   - 影响 data shuffle、dropout 等随机操作

3. **iteration** (当前迭代次数)
   - 从 checkpoint 加载
   - 决定了已经训练了多少步

❓ 使用 --no_load_optim 时会怎样？

✅ **iteration 会正确恢复**
   - 从 1400 继续训练到 1401, 1402...

✅ **consumed_train_samples 会正确计算**
   - Megatron 会根据 iteration 计算已消费样本数
   - 数据加载器会跳过前 1400 * global_batch_size 个样本

❓ 使用 --no_load_rng 时会怎样？

⚠️  **数据顺序可能改变**
   - RNG 状态被重置，数据打乱顺序会不同
   - 但仍然会跳过前面的样本（基于 consumed_samples）
   - 相当于从随机打乱后的第 1400 批开始

💡 **总结：**

如果你想从 1400 step 继续训练，使用以下配置：

✅ 推荐（续训）:
   --no_load_optim true    # 不加载优化器（节省内存/加快启动）
   --no_load_rng false     # 加载 RNG 状态（保证数据顺序一致）

这样可以：
- ✅ 从正确的 iteration (1400) 开始
- ✅ 跳过正确数量的样本
- ✅ 保持数据打乱顺序一致（如果原训练也保存了 RNG）
- ✅ 优化器状态会重新初始化（学习率会从当前 iteration 对应值开始）

⚠️  如果原训练使用了 --no_save_rng true，则无法加载 RNG 状态，
    此时数据顺序会改变，但仍会从第 1400 批开始。
""")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <checkpoint_dir>")
        print("\n例如:")
        print(f"  python {sys.argv[0]} megatron_output/model/vx-xxx")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    check_checkpoint_state(checkpoint_dir)
