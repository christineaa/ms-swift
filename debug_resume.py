#!/usr/bin/env python3
"""诊断 Megatron 恢复训练问题的脚本"""
import os
import json
import sys

def check_checkpoint_dir(load_path, save_path):
    """检查 checkpoint 目录结构"""
    print("=" * 80)
    print("检查 checkpoint 目录结构")
    print("=" * 80)

    # 检查 load 路径
    print(f"\n1. 检查 --load 路径: {load_path}")
    if not os.path.exists(load_path):
        print(f"   ❌ 路径不存在！")
        return False
    else:
        print(f"   ✅ 路径存在")

    # 检查是否有 args.json
    args_json_path = os.path.join(load_path, 'args.json')
    print(f"\n2. 检查 args.json: {args_json_path}")
    if not os.path.exists(args_json_path):
        print(f"   ❌ args.json 不存在！")
        print(f"   💡 需要找到包含 args.json 的目录")

        # 尝试在父目录和子目录中查找
        print(f"\n   🔍 在当前路径及子目录中查找 args.json...")
        for root, dirs, files in os.walk(load_path):
            if 'args.json' in files:
                found_path = root
                print(f"   ✅ 找到: {found_path}")
                print(f"   💡 请使用: --load {found_path}")

        # 检查父目录
        parent_dir = os.path.dirname(load_path)
        if parent_dir and os.path.exists(os.path.join(parent_dir, 'args.json')):
            print(f"   ✅ 在父目录找到: {parent_dir}")
            print(f"   💡 请使用: --load {parent_dir}")

        return False
    else:
        print(f"   ✅ args.json 存在")

        # 读取并显示 args.json 中的关键信息
        with open(args_json_path, 'r') as f:
            args = json.load(f)

        print(f"\n3. args.json 中的关键参数:")
        print(f"   - model: {args.get('model', 'NOT FOUND')}")
        print(f"   - train_type: {args.get('train_type', 'NOT FOUND')}")
        print(f"   - load: {args.get('load', 'NOT FOUND')}")
        print(f"   - save: {args.get('save', 'NOT FOUND')}")

        if args.get('model') is None:
            print(f"\n   ⚠️  WARNING: args.json 中的 model 字段为 None!")
            print(f"   💡 你需要在命令行中显式指定 --model 参数")

    # 检查 latest_checkpointed_iteration.txt
    latest_iter_path = os.path.join(load_path, 'latest_checkpointed_iteration.txt')
    print(f"\n4. 检查最新的 checkpoint:")
    if os.path.exists(latest_iter_path):
        with open(latest_iter_path, 'r') as f:
            latest_iter = f.read().strip()
        print(f"   ✅ 最新 iteration: {latest_iter}")

        # 检查对应的 checkpoint 目录
        iter_dir = os.path.join(load_path, f'iter_{latest_iter.zfill(7)}')
        if os.path.exists(iter_dir):
            print(f"   ✅ checkpoint 目录存在: {iter_dir}")
        else:
            print(f"   ❌ checkpoint 目录不存在: {iter_dir}")
    else:
        print(f"   ⚠️  latest_checkpointed_iteration.txt 不存在")

        # 列出所有 iter_* 目录
        iter_dirs = [d for d in os.listdir(load_path) if d.startswith('iter_')]
        if iter_dirs:
            print(f"   📁 找到的 checkpoint 目录:")
            for d in sorted(iter_dirs):
                print(f"      - {d}")

    # 检查 save 路径
    if save_path:
        print(f"\n5. 检查 --save 路径: {save_path}")
        if os.path.exists(save_path):
            print(f"   ✅ 路径存在")
        else:
            print(f"   ⚠️  路径不存在（将会自动创建）")

    print("\n" + "=" * 80)
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <load_path> [save_path]")
        print("\n例如:")
        print(f"  python {sys.argv[0]} megatron_output/Qwen2.5-7B-Instruct")
        print(f"  python {sys.argv[0]} megatron_output/Qwen2.5-7B-Instruct/v0-xxx megatron_output/Qwen2.5-7B-Instruct")
        sys.exit(1)

    load_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    success = check_checkpoint_dir(load_path, save_path)

    if success:
        print("\n✅ 基本检查通过！")
        print("\n推荐的恢复训练命令格式:")
        print(f"""
megatron sft \\
    --load {load_path} \\
    --save {save_path or load_path} \\
    --dataset 'your_dataset' \\
    ... (其他参数保持与原训练一致)
""")
    else:
        print("\n❌ 检查发现问题，请根据上面的提示修正后重试")
