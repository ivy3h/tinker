"""
在 HuggingFaceH4/MATH-500 数据集上评估 Tinker 模型

MATH-500 是 OpenAI 创建的数学基准测试的子集，包含 500 个问题
涵盖 7 个数学主题：代数、计数与概率、几何、中级代数、数论、预代数、预微积分

使用方法:
1. 安装依赖: pip install datasets
2. 配置模型路径和基础模型
3. 运行: python evaluate_math500.py
"""

import tinker
from tinker import types
from datasets import load_dataset
import re
import json
from typing import Dict, List
import os

# ==================== 配置 ====================
# 模型路径（训练时输出的 tinker:// 路径）
<<<<<<< HEAD
MODEL_PATH = "tinker://<your-model-id>/sampler_weights/final"
=======
MODEL_PATH = "tinker://f0340c70-1f22-4a72-8e46-fc48b2965178/sampler_weights/s1k_test-20251107-124036"
>>>>>>> e64db62 (update MATH500_EVALUATION)

# 基础模型（必须与训练时使用的相同）
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# 采样参数
<<<<<<< HEAD
MAX_TOKENS = 32768  # 数学问题可能需要长推理
TEMPERATURE = 0.0  # 贪婪解码，获得确定性输出

=======
MAX_TOKENS = 16384  # 数学问题可能需要长推理
TEMPERATURE = 0.0  # 贪婪解码，获得确定性输出

# 是否保存详细结果
>>>>>>> e64db62 (update MATH500_EVALUATION)
SAVE_DETAILED_RESULTS = True
model_name = BASE_MODEL.split("/")[-1]
save_dir = os.path.join("evaluation_results", model_name)
os.makedirs(save_dir, exist_ok=True)
<<<<<<< HEAD
=======

>>>>>>> e64db62 (update MATH500_EVALUATION)
OUTPUT_FILE = os.path.join(save_dir, f"{model_name}_{DATASET_NAME}.json")
# ==============================================

def normalize_answer(answer: str) -> str:
    """
    标准化答案格式，去除 LaTeX 标记和多余空格
    
    MATH 数据集的答案通常包含 LaTeX 格式
    """
    # 去除 \boxed{} 标记
    answer = re.sub(r'\\boxed{([^}]*)}', r'\1', answer)
    
    # 去除其他常见 LaTeX 标记
    answer = answer.replace('\\$', '')
    answer = answer.replace('$', '')
    answer = answer.replace('\\', '')
    
    # 去除空格并转小写
    answer = answer.strip().lower()
    answer = re.sub(r'\s+', ' ', answer)
    
    return answer

def extract_answer_from_solution(text: str) -> str:
    """
    从模型输出中提取最终答案
    
    尝试识别常见的答案模式：
    - 以 "The answer is" 或 "Therefore" 开头的句子
    - boxed{} 标记
    - 最后一行的数字或表达式
    """
    # 尝试找 boxed{} 标记
    boxed_match = re.search(r'\\boxed{([^}]*)}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 尝试找 "The answer is" 模式
    answer_patterns = [
        r'[Tt]he answer is:?\s*(.+?)(?:\.|$)',
        r'[Tt]herefore,?\s*(.+?)(?:\.|$)',
        r'[Ss]o,?\s*the answer is:?\s*(.+?)(?:\.|$)',
        r'[Ff]inal [Aa]nswer:?\s*(.+?)(?:\.|$)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # 如果都没找到，返回最后一个非空行
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return text.strip()

def check_answer(predicted: str, expected: str) -> bool:
    """
    检查预测答案是否正确
    
    比较标准化后的答案
    """
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    # 精确匹配
    if pred_norm == exp_norm:
        return True
    
    # 检查包含关系（处理更长的输出）
    if exp_norm in pred_norm:
        return True
    
    # 数值比较（处理小数/分数）
    try:
        # 尝试作为数值比较
        pred_val = eval(pred_norm.replace(' ', ''))
        exp_val = eval(exp_norm.replace(' ', ''))
        return abs(pred_val - exp_val) < 1e-6
    except:
        pass
    
    return False

def evaluate_on_math500(
    sampling_client,
    tokenizer,
    max_samples: int = None,
    verbose: bool = True
) -> Dict:
    """
    在 MATH-500 数据集上评估模型
    
    Args:
        sampling_client: Tinker sampling client
        tokenizer: Tokenizer 对象
        max_samples: 最大评估样本数（None = 全部）
        verbose: 是否显示详细信息
    
    Returns:
        评估结果字典
    """
    # 加载 MATH-500 数据集
    print("Loading MATH-500 dataset from HuggingFace...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} problems")
    print("=" * 70)
    
    # 初始化结果统计
    results = {
        'total': 0,
        'correct': 0,
        'by_subject': {},
        'by_level': {},
        'detailed_results': []
    }
    
    # 评估每个问题
    for i, example in enumerate(dataset):
        problem = example['problem']
        solution = example['solution']
        answer = example['answer']
        subject = example['subject']
        level = example['level']
        
        # 构建 prompt
        prompt_text = f"Solve the following math problem step by step:\n\n{problem}\n\nProvide a detailed solution and clearly state your final answer."
        
        try:
            # 生成模型输出
            prompt_tokens = tokenizer.encode(prompt_text)
            prompt_input = types.ModelInput.from_ints(prompt_tokens)
            
            params = types.SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop=["\n\nProblem:", "\n\n#"]  # 防止继续生成新问题
            )
            
            future = sampling_client.sample(
                prompt=prompt_input,
                sampling_params=params,
                num_samples=1
            )
            result = future.result()
            
            model_output = tokenizer.decode(result.sequences[0].tokens).strip()
            
            # 提取答案
            predicted_answer = extract_answer_from_solution(model_output)
            
            # 检查正确性
            is_correct = check_answer(predicted_answer, answer)
            
            # 更新统计
            results['total'] += 1
            if is_correct:
                results['correct'] += 1
            
            # 按主题统计
            if subject not in results['by_subject']:
                results['by_subject'][subject] = {'total': 0, 'correct': 0}
            results['by_subject'][subject]['total'] += 1
            if is_correct:
                results['by_subject'][subject]['correct'] += 1
            
            # 按难度统计
            if level not in results['by_level']:
                results['by_level'][level] = {'total': 0, 'correct': 0}
            results['by_level'][level]['total'] += 1
            if is_correct:
                results['by_level'][level]['correct'] += 1
            
            # 保存详细结果
            results['detailed_results'].append({
                'problem': problem[:200] + '...' if len(problem) > 200 else problem,
                'expected_answer': answer,
                'predicted_answer': predicted_answer,
                'full_output': model_output[:500] + '...' if len(model_output) > 500 else model_output,
                'subject': subject,
                'level': level,
                'correct': is_correct
            })
            
            # 显示进度
            if verbose and (i + 1) % 10 == 0:
                current_acc = results['correct'] / results['total']
                print(f"Progress: {i + 1}/{len(dataset)} - "
                      f"Accuracy: {current_acc:.2%} "
                      f"({results['correct']}/{results['total']})")
        
        except Exception as e:
            print(f"Error evaluating problem {i}: {e}")
            continue
    
    # 计算最终准确率
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    # 计算各主题准确率
    for subject in results['by_subject']:
        subj_stats = results['by_subject'][subject]
        subj_stats['accuracy'] = subj_stats['correct'] / subj_stats['total'] if subj_stats['total'] > 0 else 0
    
    # 计算各难度准确率
    for level in results['by_level']:
        level_stats = results['by_level'][level]
        level_stats['accuracy'] = level_stats['correct'] / level_stats['total'] if level_stats['total'] > 0 else 0
    
    return results

def print_results(results: Dict):
    """打印评估结果"""
    print("\n" + "=" * 70)
    print("MATH-500 Evaluation Results")
    print("=" * 70)
    
    print(f"\nOverall Performance:")
    print(f"  Total problems: {results['total']}")
    print(f"  Correct: {results['correct']}")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    
    print(f"\nPerformance by Subject:")
    subjects_sorted = sorted(results['by_subject'].items(), 
                            key=lambda x: x[1]['accuracy'], 
                            reverse=True)
    for subject, stats in subjects_sorted:
        print(f"  {subject:25s}: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    print(f"\nPerformance by Difficulty Level:")
    levels_sorted = sorted(results['by_level'].items(), key=lambda x: x[0])
    for level, stats in levels_sorted:
        print(f"  Level {level}: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    print("\n" + "=" * 70)
    print("Sample Predictions (first 5 problems):")
    print("=" * 70)
    
    for i, detail in enumerate(results['detailed_results'][:5]):
        print(f"\nProblem {i + 1}:")
        print(f"Subject: {detail['subject']}, Level: {detail['level']}")
        print(f"Problem: {detail['problem']}")
        print(f"Expected: {detail['expected_answer']}")
        print(f"Predicted: {detail['predicted_answer']}")
        print(f"Correct: {'✓' if detail['correct'] else '✗'}")
        print("-" * 70)

def save_results(results: Dict, filename: str):
    """保存评估结果到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {filename}")

def main():
    print("=" * 70)
    print("MATH-500 Evaluation for Tinker Models")
    print("=" * 70)
    
    # 1. 初始化 Tinker
    print("\nInitializing Tinker service...")
    service_client = tinker.ServiceClient()
    
    # 2. 加载模型
    print(f"Loading model from: {MODEL_PATH}")
    sampling_client = service_client.create_sampling_client(
        model_path=MODEL_PATH
    )
    
    # 3. 获取 tokenizer
    print(f"Loading tokenizer from base model: {BASE_MODEL}")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=32
    )
    tokenizer = training_client.get_tokenizer()
    
    # 4. 运行评估
    print("\nStarting evaluation on MATH-500...")
    print("This may take a while (500 problems)...")
    results = evaluate_on_math500(sampling_client, tokenizer)
    
    # 5. 显示结果
    print_results(results)
    
    # 6. 保存结果
    if SAVE_DETAILED_RESULTS:
        save_results(results, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
