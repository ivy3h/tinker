# Tinker Fine-tuning Configuration

MAX_SEQUENCE_LENGTH = 16384
TRUNCATE_LONG_SEQUENCES = False
SHOW_LENGTH_WARNINGS = True

MODEL_NAME_PREFIX = "s1k_1.1"

DATA_FILES = [
    #"data/s1k.json",
    #"data/ms1k_all_languages.json",       
    "data/s1k_1.1_best.json",       
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