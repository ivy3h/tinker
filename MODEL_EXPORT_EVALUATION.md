# 模型导出和评估指南

## 📦 模型导出

Tinker 支持导出训练好的 LoRA 权重，你可以在其他推理平台使用这些权重。

### 方法 1: 使用 model_path 导出（推荐）

训练完成后，`sampling_client` 会有一个 `model_path` 属性：

```python
# 训练完成后
sampling_client = training_client.save_weights_and_get_sampling_client(
    name='my-finetuned-model'
)

# 打印模型路径（类似 tinker://<unique_id>/sampler_weights/final）
print(f"Model path: {sampling_client.model_path}")

# 下载模型权重
import tinker

service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

# 下载权重
future = rest_client.download_checkpoint_archive_from_tinker_path(
    sampling_client.model_path
)
archive_data = future.result()

# 保存为 tar.gz 文件
with open("model-checkpoint.tar.gz", "wb") as f:
    f.write(archive_data)

print("Model downloaded successfully!")
```

### 方法 2: 使用 checkpoint ID 导出

如果你知道 checkpoint ID：

```python
import tinker

service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

# 替换 <unique_id> 为你的 checkpoint ID
checkpoint_path = "tinker://<unique_id>/sampler_weights/final"

future = rest_client.download_checkpoint_archive_from_tinker_path(
    checkpoint_path
)
archive_data = future.result()

with open("model-checkpoint.tar.gz", "wb") as f:
    f.write(archive_data)
```

### 导出的内容

下载的 `model-checkpoint.tar.gz` 文件包含：
- LoRA adapter 权重
- 配置文件
- 其他训练信息

### 解压和使用

```bash
# 解压文件
mkdir model_weights
tar -xzvf model-checkpoint.tar.gz -C model_weights

# 查看内容
ls model_weights/
```

### 与其他框架集成

**使用 HuggingFace transformers:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B-Base")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Base")

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "./model_weights")

# 合并权重（可选，用于部署）
merged_model = model.merge_and_unload()

# 保存合并后的模型
save_path = "./merged_model"
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

**使用 vLLM 进行推理:**

```python
from vllm import LLM, SamplingParams

# 加载合并后的模型
llm = LLM(model="./merged_model")

# 推理
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Your prompt here"], sampling_params)
```

---

## 📊 模型评估

Tinker 提供了评估工具，特别是与 InspectAI 的集成。

### 1. 基础评估方法

#### 在训练循环中评估

```python
# 在训练过程中定期评估
for epoch in range(num_epochs):
    # 训练步骤
    fwdbwd_future = training_client.forward_backward(
        processed_examples,
        "cross_entropy"
    )
    optim_future = training_client.optim_step(
        types.AdamParams(learning_rate=learning_rate)
    )

    # 计算训练损失
    fwdbwd_result = fwdbwd_future.result()
    # ... 计算损失 ...

    # 每 N 个 epoch 评估一次
    if epoch % 5 == 0:
        # 保存检查点并评估
        eval_sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f'checkpoint-epoch-{epoch}'
        )

        # 在验证集上评估
        eval_results = evaluate_model(eval_sampling_client, validation_data)
        print(f"Epoch {epoch} - Validation accuracy: {eval_results['accuracy']}")
```

#### 自定义评估函数

```python
def evaluate_model(sampling_client, test_data, tokenizer):
    """
    评估模型性能

    Args:
        sampling_client: Tinker sampling client
        test_data: 测试数据列表
        tokenizer: tokenizer 对象

    Returns:
        评估指标字典
    """
    correct = 0
    total = 0

    for example in test_data:
        # 构建 prompt
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        expected_output = example.get("output", "")

        if input_text:
            prompt_text = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt_text = f"Instruction: {instruction}\nResponse:"

        # 生成预测
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt = types.ModelInput.from_ints(prompt_tokens)

        params = types.SamplingParams(
            max_tokens=512,
            temperature=0.0,  # 使用贪婪解码获得确定性输出
            stop=["\n\n", "Instruction:"]
        )

        future = sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=1
        )
        result = future.result()

        # 获取模型输出
        model_output = tokenizer.decode(result.sequences[0].tokens).strip()

        # 评估（根据任务类型自定义）
        if check_answer(model_output, expected_output):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

def check_answer(model_output, expected_output):
    """
    检查答案是否正确
    根据你的任务类型自定义此函数
    """
    # 对于数学问题，可能需要提取最终答案
    # 对于分类任务，可能需要检查类别标签
    # 简单示例：字符串匹配
    return model_output.strip().lower() == expected_output.strip().lower()
```

### 2. 使用 Tinker Cookbook 的评估工具

Tinker Cookbook 提供了更高级的评估抽象：

```python
from tinker_cookbook.evaluation import evaluate_completions

# 创建评估器
evaluator = evaluate_completions(
    sampling_client=sampling_client,
    test_dataset=test_data,
    renderer=renderer,  # 来自 tinker_cookbook.renderers
    metrics=['accuracy', 'exact_match', 'f1']
)

# 运行评估
results = evaluator.run()

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Exact Match: {results['exact_match']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
```

### 3. 与 InspectAI 集成（标准基准测试）

Tinker Cookbook 支持与 InspectAI 集成，用于在标准基准测试上评估：

```python
from tinker_cookbook.inspect_evaluation import run_inspect_eval

# 在 MMLU、GSM8K 等标准基准上评估
results = run_inspect_eval(
    sampling_client=sampling_client,
    benchmark="gsm8k",  # 或 "mmlu", "hellaswag" 等
    batch_size=32
)

print(f"GSM8K Score: {results['score']:.2f}%")
```

### 4. 数学推理评估（针对你的数据）

对于数学证明和推理任务，可以使用特定的评估方法：

```python
def evaluate_math_reasoning(sampling_client, test_data, tokenizer):
    """
    评估数学推理任务

    评估指标：
    - 最终答案正确率
    - 推理步骤质量
    - 完整性
    """
    results = {
        'correct_answer': 0,
        'correct_steps': 0,
        'complete_proof': 0,
        'total': 0
    }

    for example in test_data:
        instruction = example.get("instruction", "")
        expected_output = example.get("output", "")

        # 生成模型输出
        prompt_text = f"Instruction: {instruction}\nResponse:"
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt = types.ModelInput.from_ints(prompt_tokens)

        params = types.SamplingParams(
            max_tokens=2048,  # 数学推理需要较长输出
            temperature=0.0
        )

        future = sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=1
        )
        result = future.result()
        model_output = tokenizer.decode(result.sequences[0].tokens)

        # 评估不同方面
        if check_final_answer(model_output, expected_output):
            results['correct_answer'] += 1

        if check_reasoning_steps(model_output, expected_output):
            results['correct_steps'] += 1

        if check_completeness(model_output):
            results['complete_proof'] += 1

        results['total'] += 1

    # 计算百分比
    total = results['total']
    return {
        'answer_accuracy': results['correct_answer'] / total,
        'step_accuracy': results['correct_steps'] / total,
        'completeness': results['complete_proof'] / total
    }

def check_final_answer(model_output, expected_output):
    """提取并比较最终答案"""
    # 实现答案提取逻辑
    pass

def check_reasoning_steps(model_output, expected_output):
    """检查推理步骤是否正确"""
    # 实现步骤验证逻辑
    pass

def check_completeness(model_output):
    """检查证明是否完整"""
    # 实现完整性检查逻辑
    pass
```

### 5. 评估最佳实践

#### 划分数据集

```python
# 将数据划分为训练集和测试集
from sklearn.model_selection import train_test_split

all_data = load_json_files(["data/s1k.json"])

train_data, test_data = train_test_split(
    all_data,
    test_size=0.2,  # 20% 作为测试集
    random_state=42
)

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")
```

#### 保存测试集

```python
import json

# 保存测试集以便后续使用
with open("data/test_set.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
```

#### 定期评估

```python
# 在训练过程中定期在测试集上评估
evaluation_history = []

for epoch in range(num_epochs):
    # 训练...

    # 每 5 个 epoch 评估一次
    if epoch % 5 == 0:
        eval_results = evaluate_model(sampling_client, test_data, tokenizer)
        evaluation_history.append({
            'epoch': epoch,
            'accuracy': eval_results['accuracy'],
            'loss': current_loss
        })

        print(f"Epoch {epoch} - Test Accuracy: {eval_results['accuracy']:.4f}")

# 保存评估历史
with open("evaluation_history.json", "w") as f:
    json.dump(evaluation_history, f, indent=2)
```

---

## 🎯 完整示例：训练、导出、评估

```python
import tinker
from tinker import types
import json

# 1. 训练模型
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-30B-A3B-Instruct",
    rank=32
)

# ... 训练过程 ...

# 2. 保存并获取 sampling client
sampling_client = training_client.save_weights_and_get_sampling_client(
    name='math-model-final'
)
model_path = sampling_client.model_path
print(f"Model saved at: {model_path}")

# 3. 在测试集上评估
test_data = load_json_files(["data/test_set.json"])
eval_results = evaluate_model(sampling_client, test_data, tokenizer)
print(f"Test Accuracy: {eval_results['accuracy']:.4f}")

# 4. 导出模型权重
rest_client = service_client.create_rest_client()
future = rest_client.download_checkpoint_archive_from_tinker_path(model_path)
archive_data = future.result()

with open("math-model-final.tar.gz", "wb") as f:
    f.write(archive_data)
print("Model exported successfully!")

# 5. 保存评估结果
results_summary = {
    'model_path': model_path,
    'test_accuracy': eval_results['accuracy'],
    'test_size': len(test_data),
    'model_name': 'math-model-final'
}

with open("model_evaluation.json", "w") as f:
    json.dump(results_summary, f, indent=2)
```

---

## 📝 总结

### 模型导出
- ✅ 使用 `download_checkpoint_archive_from_tinker_path()` 导出
- ✅ 导出的是 LoRA adapter 权重（.tar.gz 格式）
- ✅ 可以与 HuggingFace、vLLM 等框架集成
- ✅ 可以合并权重用于部署

### 评估方法
- ✅ 训练中评估：监控训练损失
- ✅ 测试集评估：计算准确率等指标
- ✅ Tinker Cookbook：提供高级评估工具
- ✅ InspectAI 集成：标准基准测试
- ✅ 自定义评估：针对特定任务

### 推荐工作流程
1. 划分训练集和测试集（80/20）
2. 在训练集上训练模型
3. 定期在测试集上评估（每 5-10 epochs）
4. 选择最佳检查点
5. 导出模型权重
6. 在其他平台部署和使用
