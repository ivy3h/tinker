# Tinker Fine-tuning Configuration

MAX_SEQUENCE_LENGTH = 32768
TRUNCATE_LONG_SEQUENCES = True
SHOW_LENGTH_WARNINGS = True

MODEL_NAME_PREFIX = "s1k"

DATA_FILES = [
    "data/s1k.json",
    # "s2k.json",       
    # "s3k.json",
    # "other_data.json",
]

BASE_MODEL = "Qwen/Qwen3-8B-Base"

LORA_RANK = 32  
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5

SAMPLING_MAX_TOKENS = 16384 
SAMPLING_TEMPERATURE = 0.7 
SAMPLING_NUM_SAMPLES = 3 