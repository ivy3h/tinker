"""
评估脚本 - 用于评估已训练的 Tinker 模型

使用方法:
1. 准备测试数据集（JSON 格式）
2. 运行: python evaluate_model.py
"""

import tinker
from tinker import types
import json
import os
from typing import List, Dict

# ==================== 配置 ====================
# 模型路径（训练时输出的 tinker:// 路径）
MODEL_PATH = "tinker://<your-model-id>/sampler_weights/final"

# 测试数据文件
TEST_DATA_FILE = "data/test_set.json"

# 基础模型（必须与训练时使用的相同）
BASE_MODEL = "Qwen/Qwen3-8B-Base"

# 采样参数
MAX_TOKENS = 512
TEMPERATURE = 0.0  # 0.0 表示贪婪解码，获得确定性输出
# ==============================================


def load_test_data(file_path: str) -> List[Dict]:
    """加载测试数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def check_answer_exact_match(model_output: str, expected_output: str) -> bool:
    """
    检查答案是否完全匹配

    可以根据任务类型自定义此函数
    """
    model_clean = model_output.strip().lower()
    expected_clean = expected_output.strip().lower()
    return model_clean == expected_clean


def check_answer_contains(model_output: str, expected_output: str) -> bool:
    """
    检查模型输出是否包含预期答案

    对于长文本输出（如数学证明）更宽松的评估方式
    """
    return expected_output.strip().lower() in model_output.strip().lower()


def extract_final_answer(text: str) -> str:
    """
    从模型输出中提取最终答案

    根据你的数据格式自定义此函数
    例如: "Final Answer: 42" -> "42"
    """
    # 简单示例：查找 "Final Answer:" 后的内容
    if "Final Answer:" in text:
        return text.split("Final Answer:")[-1].strip()

    # 如果没有明确标记，返回最后一段
    paragraphs = text.strip().split("\n\n")
    return paragraphs[-1].strip() if paragraphs else text.strip()


def evaluate_model(sampling_client, test_data: List[Dict], tokenizer, verbose: bool = True) -> Dict:
    """
    评估模型在测试集上的表现

    Args:
        sampling_client: Tinker sampling client
        test_data: 测试数据列表
        tokenizer: Tokenizer 对象
        verbose: 是否显示详细信息

    Returns:
        评估结果字典
    """
    results = {"exact_match": 0, "contains_answer": 0, "total": 0, "examples": []}

    print(f"\nEvaluating on {len(test_data)} examples...")
    print("=" * 70)

    for i, example in enumerate(test_data):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        expected_output = example.get("output", "")

        # 构建 prompt
        if input_text:
            prompt_text = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt_text = f"Instruction: {instruction}\nResponse:"

        # 生成模型输出
        try:
            prompt_tokens = tokenizer.encode(prompt_text)
            prompt = types.ModelInput.from_ints(prompt_tokens)

            params = types.SamplingParams(
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=["\n\n", "Instruction:"]
            )

            future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
            result = future.result()

            model_output = tokenizer.decode(result.sequences[0].tokens).strip()

            # 评估
            is_exact_match = check_answer_exact_match(model_output, expected_output)
            contains_answer = check_answer_contains(model_output, expected_output)

            if is_exact_match:
                results["exact_match"] += 1
            if contains_answer:
                results["contains_answer"] += 1

            results["total"] += 1

            # 保存示例
            results["examples"].append(
                {
                    "instruction": instruction[:100] + "..."
                    if len(instruction) > 100
                    else instruction,
                    "expected": expected_output[:100] + "..."
                    if len(expected_output) > 100
                    else expected_output,
                    "predicted": model_output[:100] + "..."
                    if len(model_output) > 100
                    else model_output,
                    "exact_match": is_exact_match,
                    "contains_answer": contains_answer,
                }
            )

            # 显示进度
            if verbose and (i + 1) % 10 == 0:
                current_accuracy = results["exact_match"] / results["total"]
                print(
                    f"Progress: {i + 1}/{len(test_data)} - "
                    f"Exact Match Accuracy: {current_accuracy:.2%}"
                )

        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
            continue

    # 计算最终指标
    total = results["total"]
    results["exact_match_accuracy"] = results["exact_match"] / total if total > 0 else 0
    results["contains_accuracy"] = results["contains_answer"] / total if total > 0 else 0

    return results


def print_evaluation_results(results: Dict):
    """打印评估结果"""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total examples: {results['total']}")
    print(f"Exact match: {results['exact_match']} ({results['exact_match_accuracy']:.2%})")
    print(f"Contains answer: {results['contains_answer']} ({results['contains_accuracy']:.2%})")

    # 显示一些示例
    print("\n" + "=" * 70)
    print("Sample Predictions (first 5)")
    print("=" * 70)

    for i, ex in enumerate(results["examples"][:5]):
        print(f"\nExample {i + 1}:")
        print(f"Instruction: {ex['instruction']}")
        print(f"Expected: {ex['expected']}")
        print(f"Predicted: {ex['predicted']}")
        print(f"Exact Match: {'✓' if ex['exact_match'] else '✗'}")
        print(f"Contains Answer: {'✓' if ex['contains_answer'] else '✗'}")
        print("-" * 70)


def save_evaluation_results(results: Dict, output_file: str = "evaluation_results.json"):
    """保存评估结果到文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved to: {output_file}")


def main():
    print("=" * 70)
    print("Tinker Model Evaluation")
    print("=" * 70)

    # 1. 加载测试数据
    print(f"\nLoading test data from: {TEST_DATA_FILE}")
    test_data = load_test_data(TEST_DATA_FILE)
    print(f"Loaded {len(test_data)} test examples")

    # 2. 初始化 Tinker
    print("\nInitializing Tinker service...")
    service_client = tinker.ServiceClient()

    # 3. 加载模型
    print(f"Loading model from: {MODEL_PATH}")
    sampling_client = service_client.create_sampling_client(model_path=MODEL_PATH)

    # 4. 获取 tokenizer
    # 注意：需要从基础模型获取 tokenizer
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=32,  # rank 值不影响 tokenizer
    )
    tokenizer = training_client.get_tokenizer()

    # 5. 运行评估
    results = evaluate_model(sampling_client, test_data, tokenizer)

    # 6. 显示结果
    print_evaluation_results(results)

    # 7. 保存结果
    save_evaluation_results(results)

    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
