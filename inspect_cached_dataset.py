#!/usr/bin/env python3
"""检查 ms-swift 缓存数据集内容的脚本"""
import os
import sys
from pathlib import Path

try:
    from datasets import load_from_disk, Dataset
    import pyarrow.parquet as pq
except ImportError:
    print("请先安装依赖: pip install datasets pyarrow")
    sys.exit(1)


def inspect_cached_dataset(cache_dir, num_samples=5, show_fields=True):
    """
    检查缓存数据集的内容

    Args:
        cache_dir: 缓存数据集目录路径（包含 .arrow 文件的目录）
        num_samples: 显示多少个样本
        show_fields: 是否显示所有字段信息
    """
    print("=" * 80)
    print(f"检查缓存数据集: {cache_dir}")
    print("=" * 80)

    cache_path = Path(cache_dir)

    # 检查路径是否存在
    if not cache_path.exists():
        print(f"❌ 路径不存在: {cache_dir}")
        return

    print(f"✅ 路径存在\n")

    # 列出所有 .arrow 文件
    arrow_files = list(cache_path.glob("*.arrow"))
    if not arrow_files:
        print(f"❌ 未找到 .arrow 文件")
        return

    print(f"📁 找到 {len(arrow_files)} 个 .arrow 文件")
    print(f"   示例文件: {arrow_files[0].name}")
    if len(arrow_files) > 1:
        print(f"              {arrow_files[1].name}")
        print(f"              ...")
    print()

    # 加载数据集
    try:
        print("📖 正在加载数据集...")
        dataset = Dataset.from_file(str(arrow_files[0]))

        # 如果有多个文件，尝试加载整个目录
        if len(arrow_files) > 1:
            try:
                # 检查是否有 dataset_info.json
                if (cache_path / "dataset_info.json").exists():
                    dataset = load_from_disk(str(cache_path))
                    print(f"✅ 成功加载完整数据集（所有文件）\n")
                else:
                    print(f"✅ 成功加载第一个文件作为示例\n")
            except:
                print(f"✅ 成功加载第一个文件作为示例\n")
        else:
            print(f"✅ 成功加载数据集\n")

        # 显示数据集基本信息
        print("=" * 80)
        print("数据集基本信息")
        print("=" * 80)
        print(f"样本总数: {len(dataset)}")
        print(f"字段列表: {list(dataset.features.keys())}")
        print()

        # 显示字段详细信息
        if show_fields:
            print("=" * 80)
            print("字段详细信息")
            print("=" * 80)
            for field_name, field_type in dataset.features.items():
                print(f"  • {field_name}: {field_type}")
            print()

        # 显示数据统计
        print("=" * 80)
        print("数据统计")
        print("=" * 80)

        # 如果有 length 字段，显示长度统计
        if 'length' in dataset.column_names:
            lengths = dataset['length']
            print(f"序列长度统计:")
            print(f"  - 最小长度: {min(lengths)}")
            print(f"  - 最大长度: {max(lengths)}")
            print(f"  - 平均长度: {sum(lengths) / len(lengths):.2f}")
            print()

        # 显示样本示例
        print("=" * 80)
        print(f"样本示例 (前 {min(num_samples, len(dataset))} 个)")
        print("=" * 80)

        for i in range(min(num_samples, len(dataset))):
            print(f"\n【样本 {i+1}】")
            sample = dataset[i]

            for key, value in sample.items():
                if key == 'input_ids' or key == 'labels':
                    # 对于长列表，只显示前后几个元素
                    if isinstance(value, list) and len(value) > 20:
                        print(f"  {key}: [{value[0]}, {value[1]}, {value[2]}, ..., {value[-3]}, {value[-2]}, {value[-1]}] (长度: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
                elif key == 'length':
                    print(f"  {key}: {value}")
                elif isinstance(value, str) and len(value) > 200:
                    # 对于长字符串，只显示前后部分
                    print(f"  {key}: {value[:100]}...{value[-100:]} (长度: {len(value)})")
                else:
                    print(f"  {key}: {value}")

        print("\n" + "=" * 80)
        print("✅ 数据集检查完成！")
        print("=" * 80)

        # 返回数据集供进一步操作
        return dataset

    except Exception as e:
        print(f"❌ 加载数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def search_samples(dataset, keyword, field='input_ids', max_results=5):
    """
    在数据集中搜索包含关键词的样本

    Args:
        dataset: 数据集对象
        keyword: 要搜索的关键词
        field: 搜索的字段
        max_results: 最多返回多少个结果
    """
    print(f"\n🔍 搜索包含 '{keyword}' 的样本（字段: {field}）...")

    found = 0
    for i, sample in enumerate(dataset):
        if field in sample:
            value = str(sample[field])
            if keyword.lower() in value.lower():
                found += 1
                print(f"\n【找到匹配 #{found}】样本索引: {i}")
                for key, val in sample.items():
                    if isinstance(val, list) and len(val) > 20:
                        print(f"  {key}: [...] (长度: {len(val)})")
                    elif isinstance(val, str) and len(val) > 200:
                        print(f"  {key}: {val[:100]}...")
                    else:
                        print(f"  {key}: {val}")

                if found >= max_results:
                    break

    if found == 0:
        print(f"未找到匹配的样本")
    else:
        print(f"\n共找到 {found} 个匹配样本")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='检查 ms-swift 缓存数据集内容')
    parser.add_argument('cache_dir', type=str, help='缓存数据集目录路径（例如: ./cache_dir/train）')
    parser.add_argument('--num_samples', type=int, default=5, help='显示多少个样本示例（默认: 5）')
    parser.add_argument('--no_fields', action='store_true', help='不显示字段详细信息')
    parser.add_argument('--search', type=str, default=None, help='搜索包含指定关键词的样本')
    parser.add_argument('--search_field', type=str, default='input_ids', help='搜索的字段（默认: input_ids）')

    args = parser.parse_args()

    # 检查数据集
    dataset = inspect_cached_dataset(
        args.cache_dir,
        num_samples=args.num_samples,
        show_fields=not args.no_fields
    )

    # 如果指定了搜索关键词，执行搜索
    if args.search and dataset is not None:
        search_samples(dataset, args.search, field=args.search_field)

    # 进入交互模式
    if dataset is not None:
        print("\n💡 提示: 数据集已加载到变量 'dataset' 中")
        print("   你可以使用以下方式进一步探索数据:")
        print("   - dataset[i]: 查看第 i 个样本")
        print("   - len(dataset): 查看样本总数")
        print("   - dataset.column_names: 查看所有字段名")
        print("   - dataset.filter(...): 过滤数据")
        print()

        # 询问是否进入交互模式
        try:
            user_input = input("是否进入交互式 Python 环境? (y/n): ").strip().lower()
            if user_input == 'y':
                import code
                code.interact(local=locals())
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
