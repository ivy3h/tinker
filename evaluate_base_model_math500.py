import tinker
from tinker import types
from datasets import load_dataset
import re
import json
import os

# ==================== 配置 ====================
# 基础模型
BASE_MODEL = "Qwen/Qwen3-8B-Base"

# 采样参数
MAX_TOKENS = 16384
TEMPERATURE = 0.0

# 是否保存结果
SAVE_RESULTS = True
model_name = BASE_MODEL.split("/")[-1]
save_dir = os.path.join("evaluation_results", model_name)
os.makedirs(save_dir, exist_ok=True)
OUTPUT_FILE = os.path.join(save_dir, f"{model_name}_MATH500.json")

MAX_SAMPLES = None  # 改成 50 可以快速测试
# ==============================================


def normalize_answer(answer: str) -> str:
    """标准化答案"""
    answer = re.sub(r"\\boxed{([^}]*)}", r"\1", answer)
    answer = answer.replace("\\$", "").replace("$", "").replace("\\", "")
    answer = answer.strip().lower()
    answer = re.sub(r"\s+", " ", answer)
    return answer


def extract_answer_from_solution(text: str) -> str:
    """从输出中提取答案"""
    boxed_match = re.search(r"\\boxed{([^}]*)}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    answer_patterns = [
        r"[Tt]he answer is:?\s*(.+?)(?:\.|$)",
        r"[Tt]herefore,?\s*(.+?)(?:\.|$)",
        r"[Ss]o,?\s*the answer is:?\s*(.+?)(?:\.|$)",
        r"[Ff]inal [Aa]nswer:?\s*(.+?)(?:\.|$)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return lines[-1]

    return text.strip()


def check_answer(predicted: str, expected: str) -> bool:
    """检查答案是否正确"""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if pred_norm == exp_norm:
        return True

    if exp_norm in pred_norm:
        return True

    try:
        pred_val = eval(pred_norm.replace(" ", ""))
        exp_val = eval(exp_norm.replace(" ", ""))
        return abs(pred_val - exp_val) < 1e-6
    except:
        pass

    return False


def evaluate_base_model():
    """评估基础模型"""
    print("=" * 70)
    print("Evaluating Base Model (No Fine-tuning)")
    print("=" * 70)

    # 1. 加载数据集
    print("\nLoading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    print(f"Loaded {len(dataset)} problems")

    # 2. 初始化 Tinker 和基础模型
    print(f"\nInitializing base model: {BASE_MODEL}")
    service_client = tinker.ServiceClient()

    # 创建一个 training client 只是为了获取 tokenizer
    # 我们不会用它来训练，只是用它的 tokenizer
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=32)
    tokenizer = training_client.get_tokenizer()

    # 保存权重并创建 sampling client（这实际上就是基础模型）
    print("Creating sampling client for base model...")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="base-model-eval")

    # 3. 评估
    print("\nStarting evaluation...")
    print("=" * 70)

    results = {
        "model": BASE_MODEL,
        "fine_tuned": False,
        "total": 0,
        "correct": 0,
        "by_subject": {},
        "by_level": {},
        "detailed_results": [],
    }

    for i, example in enumerate(dataset):
        problem = example["problem"]
        answer = example["answer"]
        subject = example["subject"]
        level = example["level"]

        # 构建 prompt
        prompt_text = f"Solve the following math problem step by step:\n\n{problem}\n\nProvide a detailed solution and clearly state your final answer."

        try:
            prompt_tokens = tokenizer.encode(prompt_text)
            prompt_input = types.ModelInput.from_ints(prompt_tokens)

            params = types.SamplingParams(
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=["\n\nProblem:", "\n\n#"]
            )

            future = sampling_client.sample(
                prompt=prompt_input, sampling_params=params, num_samples=1
            )
            result = future.result()

            model_output = tokenizer.decode(result.sequences[0].tokens).strip()
            predicted_answer = extract_answer_from_solution(model_output)
            is_correct = check_answer(predicted_answer, answer)

            # 更新统计
            results["total"] += 1
            if is_correct:
                results["correct"] += 1

            # 按主题
            if subject not in results["by_subject"]:
                results["by_subject"][subject] = {"total": 0, "correct": 0}
            results["by_subject"][subject]["total"] += 1
            if is_correct:
                results["by_subject"][subject]["correct"] += 1

            # 按难度
            if level not in results["by_level"]:
                results["by_level"][level] = {"total": 0, "correct": 0}
            results["by_level"][level]["total"] += 1
            if is_correct:
                results["by_level"][level]["correct"] += 1

            # 保存详细结果
            results["detailed_results"].append(
                {
                    "problem": problem[:200] + "..." if len(problem) > 200 else problem,
                    "expected_answer": answer,
                    "predicted_answer": predicted_answer,
                    "subject": subject,
                    "level": level,
                    "correct": is_correct,
                }
            )

            # 显示进度
            if (i + 1) % 10 == 0:
                current_acc = results["correct"] / results["total"]
                print(
                    f"Progress: {i + 1}/{len(dataset)} - "
                    f"Accuracy: {current_acc:.2%} ({results['correct']}/{results['total']})"
                )

        except Exception as e:
            print(f"Error evaluating problem {i}: {e}")
            continue

    # 4. 计算最终结果
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0

    for subject in results["by_subject"]:
        subj_stats = results["by_subject"][subject]
        subj_stats["accuracy"] = (
            subj_stats["correct"] / subj_stats["total"] if subj_stats["total"] > 0 else 0
        )

    for level in results["by_level"]:
        level_stats = results["by_level"][level]
        level_stats["accuracy"] = (
            level_stats["correct"] / level_stats["total"] if level_stats["total"] > 0 else 0
        )

    return results


def print_results(results):
    """打印结果"""
    print("\n" + "=" * 70)
    print("Base Model Evaluation Results")
    print("=" * 70)

    print(f"\nModel: {results['model']}")
    print(f"Fine-tuned: {results['fine_tuned']}")

    print(f"\nOverall Performance:")
    print(f"  Total: {results['total']}")
    print(f"  Correct: {results['correct']}")
    print(f"  Accuracy: {results['accuracy']:.2%}")

    print(f"\nPerformance by Subject:")
    subjects_sorted = sorted(
        results["by_subject"].items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    for subject, stats in subjects_sorted:
        print(f"  {subject:25s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    print(f"\nPerformance by Difficulty:")
    levels_sorted = sorted(results["by_level"].items(), key=lambda x: x[0])
    for level, stats in levels_sorted:
        print(f"  Level {level}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")


def save_results(results, filename):
    """保存结果"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    # 评估
    results = evaluate_base_model()

    # 显示结果
    print_results(results)

    # 保存结果
    if SAVE_RESULTS:
        save_results(results, OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
