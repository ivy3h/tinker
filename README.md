# Tinker 微调脚本使用指南

本目录包含用于 Tinker API 微调的 Python 脚本，专门为处理包含 instruction/input/output 格式的 JSON 数据集设计。

## 文件说明

1. **tinker_finetune.py** - 基础微调脚本，所有配置都在代码中
2. **train_with_config.py** - 使用配置文件的微调脚本（推荐）
3. **config.py** - 配置文件，用于设置训练参数和数据文件

## 快速开始

### 1. 设置 API 密钥

```bash
export TINKER_API_KEY=7624312694161539072
```

或者在代码开头添加：
```python
import os
os.environ['TINKER_API_KEY'] = '7624312694161539072'
```

### 2. 安装依赖

```bash
pip install tinker numpy
```

### 3. 准备数据

确保你的 JSON 文件格式如下：
```json
[
  {
    "instruction": "问题或指令",
    "input": "额外的输入（可选，可以为空字符串）",
    "output": "期望的输出"
  },
  {
    "instruction": "另一个问题",
    "input": "",
    "output": "另一个输出"
  }
]
```

**文件组织方式：**

如果你的 JSON 文件在 `data` 文件夹下：
```
your_project/
├── train_with_config.py
├── config.py
└── data/
    ├── s1k.json
    ├── s2k.json
    └── other_data.json
```

在 `config.py` 中使用相对路径：
```python
DATA_FILES = [
    "data/s1k.json",
    "data/s2k.json",
]
```

如果 JSON 文件在同一目录：
```
your_project/
├── train_with_config.py
├── config.py
├── s1k.json
└── s2k.json
```

在 `config.py` 中直接使用文件名：
```python
DATA_FILES = [
    "s1k.json",
    "s2k.json",
]
```

### 4. 配置训练参数

编辑 `config.py` 文件：

```python
# 指定要使用的数据文件
DATA_FILES = [
    "s1k.json",
    "s2k.json",  # 可以添加多个文件
]

# 选择基础模型
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Base"

# 设置训练参数
LORA_RANK = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
```

### 5. 运行训练

```bash
python train_with_config.py
```

或者使用基础脚本（需要在代码中修改配置）：
```bash
python tinker_finetune.py
```

## 配置参数说明

### 数据配置
- **DATA_FILES**: 要使用的 JSON 文件列表，会自动合并所有数据

### 模型配置
- **BASE_MODEL**: 基础模型选择
  - `Qwen/Qwen3-30B-A3B-Base`: 基础预训练模型
  - `Qwen/Qwen3-30B-A3B-Instruct`: 已经过指令微调的模型
  - `meta-llama/Llama-3.1-8B`: 较小的模型

- **LORA_RANK**: LoRA 秩（推荐值: 8, 16, 32, 64）
  - 更大的值可能效果更好但训练更慢

### 训练超参数
- **LEARNING_RATE**: 学习率（推荐范围: 1e-5 到 1e-3）
  - 较大数据集可以使用较大学习率
  - 较小数据集应使用较小学习率

- **NUM_EPOCHS**: 训练轮数
  - 根据数据集大小和损失下降情况调整
  - 观察损失曲线，如果持续下降可以增加轮数

### 采样配置
- **SAMPLING_MAX_TOKENS**: 生成的最大 token 数
- **SAMPLING_TEMPERATURE**: 采样温度（0-1），越高越随机
- **SAMPLING_NUM_SAMPLES**: 测试时生成的样本数量

### 数据处理配置（重要：针对长文本）
- **MAX_SEQUENCE_LENGTH**: 最大序列长度（推荐值: 2048, 4096, 8192, 16384）
  - 如果你的 output 很长（可能达到 16384 tokens），设置为 16384
  - 更长的序列会消耗更多内存和训练时间
  - 大多数模型的上下文窗口是有限的，需要根据模型选择

- **TRUNCATE_LONG_SEQUENCES**: 是否截断过长序列
  - `True`: 自动截断超过最大长度的部分（保留 prompt，截断 output）
  - `False`: 跳过过长的样本

- **SHOW_LENGTH_WARNINGS**: 是否显示长度警告信息
  - `True`: 显示被截断或跳过的样本信息（前10个）
  - `False`: 静默处理

## 处理长文本数据（重要）

如果你的数据 output 很长（如数学证明可能有上千 tokens），需要特别注意：

### 1. 检查你的数据长度

运行训练时，脚本会显示序列长度统计：
```
Sequence length statistics:
  Min: 245 tokens
  Max: 15234 tokens
  Mean: 3456.7 tokens
  Median: 2891.0 tokens
```

### 2. 根据统计调整配置

```python
# 如果最大长度在 4000 左右
MAX_SEQUENCE_LENGTH = 4096

# 如果最大长度在 8000 左右
MAX_SEQUENCE_LENGTH = 8192

# 如果最大长度超过 10000
MAX_SEQUENCE_LENGTH = 16384
```

### 3. 选择处理策略

**策略 A: 截断（推荐用于数学推理）**
```python
MAX_SEQUENCE_LENGTH = 8192
TRUNCATE_LONG_SEQUENCES = True
```
优点：不会丢失样本，只是输出被截断
缺点：长证明的结尾可能被截断

**策略 B: 跳过**
```python
MAX_SEQUENCE_LENGTH = 8192
TRUNCATE_LONG_SEQUENCES = False
```
优点：保证所有训练样本都是完整的
缺点：会丢失一些长样本

### 4. 内存和速度考虑

序列长度对训练的影响：
- 2048 tokens: 快速，适合大批量训练
- 4096 tokens: 中等速度，适合大多数任务
- 8192 tokens: 较慢，需要较多内存
- 16384 tokens: 很慢，需要大量内存

建议：
- 如果大部分样本 < 4096，使用 4096
- 如果需要完整保留长证明，使用 16384
- 可以先用较短长度测试，再增加

## 使用多个数据集

你可以轻松使用多个 JSON 文件：

如果文件在 `data` 文件夹下：
```python
DATA_FILES = [
    "data/s1k.json",
    "data/s2k.json",
    "data/additional_data.json",
]
```

如果文件在当前目录：
```python
DATA_FILES = [
    "s1k.json",
    "s2k.json",
    "additional_data.json",
]
```

脚本会自动加载并合并所有数据。如果某个文件不存在，会显示警告但继续处理其他文件。

## 输出说明

训练完成后，你会看到：

1. **训练进度**: 每个 epoch 的损失值
2. **模型名称**: 保存的模型名称（包含时间戳）
3. **测试输出**: 在第一个训练样本上的模型响应

模型会保存在 Tinker 的云端，你可以通过返回的 `sampling_client` 进行推理。

## 训练监控

观察以下指标来判断训练效果：

- **损失下降**: 损失应该逐渐下降
- **过拟合**: 如果损失下降到接近 0，可能过拟合，减少 epochs
- **收敛**: 如果损失不再下降，可以停止训练或调整学习率

## 常见问题

### Q: 如何选择合适的模型？
A:
- 如果从头开始训练特定任务，使用 Base 模型
- 如果在已有指令模型基础上微调，使用 Instruct 模型
- 如果计算资源有限，选择较小的模型（如 Llama-3.1-8B）

### Q: 训练需要多长时间？
A: 取决于数据集大小、模型大小和 epochs 数量。通常：
- 1000 个样本，3 epochs: 约 10-30 分钟
- 10000 个样本，3 epochs: 约 1-3 小时

### Q: 如何判断是否需要更多训练数据？
A:
- 如果模型在训练集上表现好但测试效果差，需要更多数据
- 如果损失下降很慢，可能需要更多高质量数据
- 建议至少 500-1000 个高质量样本

### Q: 如何调整学习率？
A:
- 如果损失震荡，降低学习率
- 如果损失下降太慢，增加学习率
- 推荐从 1e-4 开始，根据情况调整

### Q: 如何处理非常长的 output（16384+ tokens）？
A:
1. 首先运行一次训练，查看序列长度统计
2. 根据实际情况设置 MAX_SEQUENCE_LENGTH
3. 如果内存不足，考虑：
   - 降低 MAX_SEQUENCE_LENGTH 并使用截断
   - 减少 LORA_RANK
   - 使用更小的基础模型
4. 对于数学证明这种长文本：
   - 推荐 MAX_SEQUENCE_LENGTH = 8192 或 16384
   - 使用 TRUNCATE_LONG_SEQUENCES = True

### Q: 截断会影响训练效果吗？
A:
- 对于数学推理，截断可能影响完整性
- 但通常前面的推理步骤最重要
- 建议：设置足够大的 MAX_SEQUENCE_LENGTH，尽量减少截断
- 查看统计信息，确保大部分样本没有被截断

### Q: 训练时显示 "Truncated example" 是什么意思？
A:
- 表示该样本的序列长度超过了 MAX_SEQUENCE_LENGTH
- 模型会自动截断输出的后半部分
- 如果截断太多样本，建议增加 MAX_SEQUENCE_LENGTH

## 下一步

训练完成后，你可以：

1. **导出模型权重**
   - 在训练脚本结束时选择导出
   - 或使用 `MODEL_EXPORT_EVALUATION.md` 中的代码手动导出
   - 导出的权重可以在 HuggingFace、vLLM 等平台使用

2. **评估模型性能**
   - 使用 `split_dataset.py` 划分训练集和测试集
   - 使用 `evaluate_model.py` 在测试集上评估模型
   - 查看准确率、匹配率等指标

3. **继续训练或调整**
   - 在更大的数据集上继续训练
   - 调整超参数重新训练
   - 使用不同的基础模型

4. **部署模型**
   - 导出合并后的权重
   - 使用 vLLM、HuggingFace Inference 等部署
   - 提供 API 服务

详细的导出和评估指南请查看 `MODEL_EXPORT_EVALUATION.md`。

## 📚 其他工具

### 数据集划分工具

使用 `split_dataset.py` 将数据划分为训练集和测试集：

```bash
python split_dataset.py
```

这会生成：
- `data/train_set.json` - 训练集（80%）
- `data/test_set.json` - 测试集（20%）

### 模型评估工具

使用 `evaluate_model.py` 评估已训练的模型：

```bash
python evaluate_model.py
```

需要配置：
- `MODEL_PATH`: 模型的 tinker:// 路径
- `TEST_DATA_FILE`: 测试数据文件
- `BASE_MODEL`: 基础模型名称

评估结果会保存到 `evaluation_results.json`。

## 支持

如有问题，请参考 Tinker 官方文档：
https://tinker-docs.thinkingmachines.ai/
