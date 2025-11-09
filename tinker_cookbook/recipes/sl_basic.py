import chz
import sys
import os
from datetime import datetime
from pathlib import Path
from tinker_cookbook import cli_utils, model_info

# from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio
from analyze_results import analyze_training_results


def get_model_short_name(model_name: str) -> str:
    """Extract short name from model path."""
    return model_name.split("/")[-1]


def get_dataset_name(dataset_path: str) -> str:
    """Extract dataset name from path."""
    return Path(dataset_path).stem


def create_output_dir(model_name: str, dataset_name: str, config: dict) -> str:
    """
    Create output directory with timestamp and hyperparameters.
    Format: results/YYYYMMDD_HHMMSS_model_dataset_lr{lr}_ep{epochs}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = get_model_short_name(model_name)
    lr = config.get("learning_rate", 1e-5)
    epochs = config.get("num_epochs", 5)

    dir_name = f"{timestamp}_{model_short}_{dataset_name}_lr{lr:.0e}_ep{epochs}"
    output_dir = os.path.join("results", dir_name)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_path = "/srv/nlprx-lab/share6/jhe478/tinker-cookbook/data/ms1k_all_languages.jsonl"

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromConversationFileBuilder(common_config=common_config, file_path=dataset_path)

    hyperparams = {
        "log_path": "/tmp/tinker-examples/sl_basic",
        "model_name": model_name,
        "dataset_builder": dataset,
        "learning_rate": 1e-5,
        "lr_schedule": "cosine",
        "num_epochs": 5,
        "eval_every": 8,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 1e-4,
    }

    return chz.Blueprint(train.Config).apply(hyperparams), hyperparams, dataset_path


def main(config: train.Config, hyperparams: dict, dataset_path: str):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    # Run training
    print("Starting training...")
    asyncio.run(train.main(config))

    # Analyze results automatically
    print("\n" + "=" * 60)
    print("AUTOMATIC RESULT ANALYSIS")
    print("=" * 60)

    model_name = hyperparams["model_name"]
    dataset_name = get_dataset_name(dataset_path)
    output_dir = create_output_dir(model_name, dataset_name, hyperparams)

    print(f"Saving results to: {output_dir}")

    # Save hyperparameters
    hyperparam_file = os.path.join(output_dir, "hyperparameters.txt")
    with open(hyperparam_file, "w") as f:
        f.write("Training Hyperparameters\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:           {model_name}\n")
        f.write(f"Dataset:         {dataset_path}\n")
        f.write(f"Learning rate:   {hyperparams['learning_rate']}\n")
        f.write(f"LR schedule:     {hyperparams['lr_schedule']}\n")
        f.write(f"Num epochs:      {hyperparams['num_epochs']}\n")
        f.write(f"Batch size:      {hyperparams['dataset_builder'].common_config.batch_size}\n")
        f.write(f"Max length:      {hyperparams['dataset_builder'].common_config.max_length}\n")
        f.write(f"Adam beta1:      {hyperparams['adam_beta1']}\n")
        f.write(f"Adam beta2:      {hyperparams['adam_beta2']}\n")
        f.write(f"Weight decay:      {hyperparams['weight_decay']}\n")
        f.write(f"Eval every:      {hyperparams['eval_every']}\n")

    print(f"Hyperparameters saved to: {hyperparam_file}")

    # Analyze training results
    success = analyze_training_results(config.log_path, output_dir)

    if success:
        print(f"\nAll results saved to: {output_dir}")
    else:
        print("\nWarning: Result analysis failed")


if __name__ == "__main__":
    blueprint, hyperparams, dataset_path = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make(), hyperparams, dataset_path)
