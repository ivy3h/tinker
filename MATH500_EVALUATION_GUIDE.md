# 在 MATH-500 数据集上评估模型指南

## 📊 关于 MATH-500

**MATH-500** 是 OpenAI 在论文 "Let's Verify Step by Step" 中创建的数学基准测试子集。

- **来源**: HuggingFaceH4/MATH-500
- **问题数量**: 500 个数学问题
- **难度**: 分为 5 个难度等级（1-5）
- **主题**: 7 个数学领域
  - Algebra（代数）
  - Counting & Probability（计数与概率）
  - Geometry（几何）
  - Intermediate Algebra（中级代数）
  - Number Theory（数论）
  - Prealgebra（预代数）
  - Precalculus（预微积分）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install datasets
```

### 2. 配置评估脚本

编辑 `evaluate_math500.py`:

```python
# 设置你的模型路径（训练时输出的路径）
MODEL_PATH = "tinker://abc123xyz/sampler_weights/final"

# 设置基础模型（必须与训练时相同）
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct"

# 采样参数
MAX_TOKENS = 2048  # 数学推理需要较长输出
TEMPERATURE = 0.0  # 确定性输出
```

### 3. 运行评估

```bash
python evaluate_math500.py
```

评估完成后会生成 `math500_evaluation_results.json`。

## 📈 输出结果

### 总体性能
```
Overall Performance:
  Total problems: 500
  Correct: 285
  Accuracy: 57.00%
```

### 按主题分类
```
Performance by Subject:
  Prealgebra              : 75.32% (58/77)
  Algebra                 : 68.42% (52/76)
  Number Theory           : 61.54% (40/65)
  Counting & Probability  : 58.33% (35/60)
  Geometry               : 54.17% (39/72)
  Intermediate Algebra    : 48.00% (36/75)
  Precalculus            : 33.33% (25/75)
```

### 按难度分类
```
Performance by Difficulty Level:
  Level 1: 82.35% (42/51)
  Level 2: 71.15% (74/104)
  Level 3: 58.90% (86/146)
  Level 4: 45.00% (54/120)
  Level 5: 36.71% (29/79)
```

## 🔍 结果分析

### 标准答案格式

MATH-500 的答案通常包含 LaTeX 格式，例如：
- `\frac{3}{4}` （分数）
- `\boxed{42}` （最终答案）
- `3\sqrt{2}` （根式）
- `x = 5` （方程）

脚本会自动处理这些格式。

### 答案提取

脚本会尝试从模型输出中提取最终答案：

1. 查找 `\boxed{}` 标记
2. 查找 "The answer is..." 模式
3. 查找 "Therefore..." 模式
4. 使用最后一个非空行

### 答案比较

脚本会：
1. 标准化答案格式（去除 LaTeX、空格）
2. 精确匹配
3. 包含关系检查
4. 数值比较（处理不同数值表示）

## 💡 提高准确率的建议

### 1. Prompt 优化

当前使用的 prompt:
```python
prompt_text = f"Solve the following math problem step by step:\n\n{problem}\n\nProvide a detailed solution and clearly state your final answer."
```

可以尝试的改进：
```python
# 更明确的格式要求
prompt_text = f"""Solve this math problem:

{problem}

Think step by step and show your work. At the end, clearly state your final answer in the format:

The answer is: [your answer here]
"""
```

### 2. 训练数据改进

- 包含更多数学推理步骤
- 使用标准的答案格式
- 增加难题样本
- 涵盖所有 7 个主题

### 3. 模型选择

- 对于数学任务，Instruct 模型通常表现更好
- 较大的模型（30B+）在复杂推理上表现更好
- 考虑使用专门为数学优化的基础模型

### 4. 训练超参数

```python
# 推荐配置
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct"
LEARNING_RATE = 5e-5  # 较小学习率，保留推理能力
NUM_EPOCHS = 5  # 充分训练
MAX_SEQUENCE_LENGTH = 16384  # 支持长推理链
```

## 🛠️ 自定义评估

### 评估部分数据

```python
# 只评估前 50 个问题（快速测试）
results = evaluate_on_math500(
    sampling_client,
    tokenizer,
    max_samples=50
)
```

### 按主题评估

```python
# 加载数据集
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# 筛选特定主题
algebra_problems = dataset.filter(lambda x: x['subject'] == 'Algebra')

# 评估...
```

### 自定义答案提取

修改 `extract_answer_from_solution()` 函数来适配你的模型输出格式。

### 自定义答案检查

修改 `check_answer()` 函数来调整答案匹配逻辑。

## 📊 基准对比

典型的性能范围（参考）：

| 模型类型 | 预期准确率 |
|---------|-----------|
| 基础模型（未微调） | 10-20% |
| 微调的小模型（7B-8B） | 20-35% |
| 微调的中型模型（30B） | 35-55% |
| 微调的大型模型（70B+） | 55-75% |
| 专门的数学模型 | 70-85% |

**注意**: 这些是大致范围，实际性能取决于训练数据质量和数量。

## 🔧 故障排除

### 问题：准确率很低（< 20%）

**可能原因**:
1. 答案提取逻辑不匹配模型输出格式
2. 模型没有学到数学推理能力
3. Prompt 格式与训练数据不一致

**解决方法**:
- 检查 `detailed_results` 中的实际输出
- 调整答案提取函数
- 改进训练数据和 prompt

### 问题：某些主题表现很差

**可能原因**:
- 训练数据中该主题样本不足
- 该主题需要特殊的问题解决技巧

**解决方法**:
- 增加该主题的训练样本
- 使用该主题的专门数据集进行额外训练

### 问题：评估太慢

**解决方法**:
- 减少 MAX_TOKENS
- 先用少量样本测试 (`max_samples=50`)
- 使用更快的基础模型

## 📚 延伸阅读

- [MATH 数据集论文](https://arxiv.org/abs/2103.03874)
- [Let's Verify Step by Step 论文](https://arxiv.org/abs/2305.20050)
- [HuggingFace MATH-500 数据集页面](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

## 🎯 使用流程总结

```bash
# 1. 安装依赖
pip install datasets

# 2. 训练模型（如果还没有）
python train_with_config.py

# 3. 记录 model_path
# 输出会显示类似: tinker://abc123/sampler_weights/final

# 4. 配置评估脚本
# 编辑 evaluate_math500.py，设置 MODEL_PATH

# 5. 运行评估
python evaluate_math500.py

# 6. 查看结果
cat math500_evaluation_results.json
```

## ✨ 提示

1. **先小规模测试**: 用 `max_samples=10` 快速验证流程
2. **检查输出格式**: 查看前几个样本的 `full_output` 了解模型行为
3. **调整 prompt**: 如果答案提取失败率高，调整 prompt 格式
4. **迭代改进**: 根据错误分析改进训练数据
5. **保存检查点**: 在不同 epoch 评估，选择最佳模型

祝评估顺利！🎉
