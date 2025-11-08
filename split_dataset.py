"""
数据集划分工具

将数据集划分为训练集和测试集，用于模型评估
"""

import json
import random
from pathlib import Path

# ==================== 配置 ====================
# 输入数据文件
INPUT_FILES = [
    "data/s1k.json",
    # "data/s2k.json",  # 可以添加更多文件
]

# 输出文件
TRAIN_OUTPUT = "data/train_set.json"
TEST_OUTPUT = "data/test_set.json"

# 测试集比例（0.0 到 1.0）
TEST_RATIO = 0.2  # 20% 作为测试集

# 随机种子（保证可复现）
RANDOM_SEED = 42
# ==============================================


def load_data(file_paths):
    """加载所有数据文件"""
    all_data = []

    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"Warning: File {file_path} not found, skipping...")
            continue

        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"  Loaded {len(data)} examples")

    return all_data


def split_dataset(data, test_ratio, random_seed):
    """
    将数据划分为训练集和测试集

    Args:
        data: 数据列表
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        train_data, test_data
    """
    # 设置随机种子
    random.seed(random_seed)

    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 计算划分点
    total_size = len(shuffled_data)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    # 划分
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    return train_data, test_data


def save_dataset(data, output_path):
    """保存数据集到文件"""
    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} examples to {output_path}")


def main():
    print("=" * 70)
    print("Dataset Splitting Tool")
    print("=" * 70)

    # 1. 加载数据
    print("\nLoading data...")
    all_data = load_data(INPUT_FILES)

    if not all_data:
        print("Error: No data loaded!")
        return

    print(f"\nTotal examples: {len(all_data)}")

    # 2. 划分数据集
    print(f"\nSplitting dataset (test ratio: {TEST_RATIO:.0%})...")
    train_data, test_data = split_dataset(all_data, TEST_RATIO, RANDOM_SEED)

    print(f"Training set: {len(train_data)} examples ({len(train_data) / len(all_data):.0%})")
    print(f"Test set: {len(test_data)} examples ({len(test_data) / len(all_data):.0%})")

    # 3. 保存数据集
    print("\nSaving datasets...")
    save_dataset(train_data, TRAIN_OUTPUT)
    save_dataset(test_data, TEST_OUTPUT)

    # 4. 显示统计信息
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)

    # 计算平均长度
    train_lengths = [len(ex.get("output", "")) for ex in train_data]
    test_lengths = [len(ex.get("output", "")) for ex in test_data]

    print(f"\nTraining set output length:")
    print(f"  Min: {min(train_lengths)} characters")
    print(f"  Max: {max(train_lengths)} characters")
    print(f"  Avg: {sum(train_lengths) / len(train_lengths):.1f} characters")

    print(f"\nTest set output length:")
    print(f"  Min: {min(test_lengths)} characters")
    print(f"  Max: {max(test_lengths)} characters")
    print(f"  Avg: {sum(test_lengths) / len(test_lengths):.1f} characters")

    # 5. 显示示例
    print("\n" + "=" * 70)
    print("Sample from Training Set")
    print("=" * 70)
    sample = train_data[0]
    print(f"Instruction: {sample.get('instruction', '')[:150]}...")
    print(f"Input: {sample.get('input', '')[:100] if sample.get('input') else 'N/A'}...")
    print(f"Output length: {len(sample.get('output', ''))} characters")

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print(f"1. Use {TRAIN_OUTPUT} for training")
    print(f"2. Use {TEST_OUTPUT} for evaluation")
    print(f"3. Update config.py to use the training set:")
    print(f"   DATA_FILES = ['{TRAIN_OUTPUT}']")
    print(f"4. After training, evaluate using evaluate_model.py")

    print("\nDataset splitting completed!")


if __name__ == "__main__":
    main()
