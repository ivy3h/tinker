# Tinker Fine-tuning Configuration

MAX_SEQUENCE_LENGTH = 32768  # ← 关键！设置为最大值
TRUNCATE_LONG_SEQUENCES = True  # 自动截断超长部分
SHOW_LENGTH_WARNINGS = True  # 显示统计信息

MODEL_NAME_PREFIX = "s1k"  # 保存的模型名称前缀

DATA_FILES = [
    "data/s1k.json",
    # "s2k.json",        # 取消注释以添加
    # "s3k.json",
    # "other_data.json",
]

# ==================== 模型配置 ====================
# 基础模型选择（从可用模型列表中选择）
# 常用选项:
#   - "Qwen/Qwen3-30B-A3B-Base" (基础模型)
#   - "Qwen/Qwen3-30B-A3B-Instruct" (已经过指令微调)
#   - "meta-llama/Llama-3.1-8B" (较小的模型)
BASE_MODEL = "Qwen/Qwen3-8B-Base"

# LoRA 配置
LORA_RANK = 32  # LoRA 秩，较大的值可能效果更好但训练更慢
                 # 推荐值: 8, 16, 32, 64

# ==================== 训练超参数 ====================
LEARNING_RATE = 1e-5  # 学习率
                       # 推荐范围: 1e-5 到 1e-3

NUM_EPOCHS = 5  # 训练轮数
                # 根据数据集大小调整，较小数据集可能需要更多轮

# ==================== 采样配置 ====================
# 测试时的采样参数
SAMPLING_MAX_TOKENS = 16384  # 生成的最大 token 数
SAMPLING_TEMPERATURE = 0.7  # 温度，0-1 之间，越高越随机
SAMPLING_NUM_SAMPLES = 3  # 生成的样本数量

# ==================== 其他配置 ====================