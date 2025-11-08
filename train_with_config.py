import tinker
from tinker import types
import numpy as np
import json
import os
from datetime import datetime
from config import *


def load_json_files(file_paths):
    """加载多个 JSON 文件并合并数据"""
    all_data = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue

        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"  Loaded {len(data)} examples from {file_path}")

    return all_data


def process_example(example: dict, tokenizer) -> types.Datum:
    """
    处理单个训练样本

    期望的 JSON 格式:
    {
        "instruction": "问题或指令",
        "input": "额外的上下文（可选）",
        "output": "期望的输出"
    }
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # 构建提示文本
    if input_text:
        prompt_text = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt_text = f"Instruction: {instruction}\nResponse:"

    completion_text = f" {output_text}\n\n"

    # Tokenize
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

    # 设置权重
    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def main():
    print("=" * 70)
    print("Tinker Fine-tuning Script")
    print("=" * 70)

    # 1. 加载配置和数据
    print("\nConfiguration:")
    print(f"  Data files: {DATA_FILES}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")

    print("\n" + "=" * 70)
    print("Loading training data...")
    print("=" * 70)

    training_data = load_json_files(DATA_FILES)

    if not training_data:
        print("Error: No training data loaded!")
        return

    print(f"\nTotal training examples: {len(training_data)}")

    # 2. 初始化 Tinker
    print("\n" + "=" * 70)
    print("Initializing Tinker service...")
    print("=" * 70)

    service_client = tinker.ServiceClient()

    print("\nCreating training client...")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=LORA_RANK
    )

    tokenizer = training_client.get_tokenizer()

    # 3. 处理数据
    print("\n" + "=" * 70)
    print("Processing training data...")
    print("=" * 70)

    processed_examples = []
    failed_count = 0

    for i, example in enumerate(training_data):
        try:
            processed = process_example(example, tokenizer)
            processed_examples.append(processed)
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # 只显示前5个错误
                print(f"Warning: Failed to process example {i}: {e}")

    print(f"Successfully processed: {len(processed_examples)} examples")
    if failed_count > 0:
        print(f"Failed to process: {failed_count} examples")

    # 4. 训练
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    losses = []

    for epoch in range(NUM_EPOCHS):
        fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")

        optim_future = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # 计算损失
        logprobs = np.concatenate(
            [output["logprobs"].tolist() for output in fwdbwd_result.loss_fn_outputs]
        )
        weights = np.concatenate(
            [example.loss_fn_inputs["weights"].tolist() for example in processed_examples]
        )
        loss = -np.dot(logprobs, weights) / weights.sum()
        losses.append(loss)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

    # 5. 保存模型
    print("\n" + "=" * 70)
    print("Saving model...")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{MODEL_NAME_PREFIX}-{timestamp}"

    sampling_client = training_client.save_weights_and_get_sampling_client(name=model_name)

    print(f"Model saved as: {model_name}")

    # 6. 测试模型
    print("\n" + "=" * 70)
    print("Testing model...")
    print("=" * 70)

    test_example = training_data[0]
    test_instruction = test_example.get("instruction", "")
    test_input = test_example.get("input", "")

    if test_input:
        test_prompt = f"Instruction: {test_instruction}\nInput: {test_input}\nResponse:"
    else:
        test_prompt = f"Instruction: {test_instruction}\nResponse:"

    print(f"\nTest instruction: {test_instruction[:150]}...")

    prompt_tokens = tokenizer.encode(test_prompt)
    prompt = types.ModelInput.from_ints(prompt_tokens)
    params = types.SamplingParams(
        max_tokens=SAMPLING_MAX_TOKENS,
        temperature=SAMPLING_TEMPERATURE,
        stop=["\n\n", "Instruction:"],
    )

    future = sampling_client.sample(
        prompt=prompt, sampling_params=params, num_samples=SAMPLING_NUM_SAMPLES
    )
    result = future.result()

    print("\nModel responses:")
    print("-" * 70)
    for i, seq in enumerate(result.sequences):
        response = tokenizer.decode(seq.tokens)
        print(f"\nSample {i + 1}:")
        print(response[:500])  # 限制输出长度
        if len(response) > 500:
            print("... (truncated)")
        print("-" * 70)

    # 7. 总结
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Model name: {model_name}")
    print(f"Training examples: {len(processed_examples)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
