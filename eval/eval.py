#!/usr/bin/env python3
"""
Multilingual reasoning evaluation script (Tinker platform).

This script evaluates language models via Tinker platform on multilingual reasoning tasks and writes two-phase results:
  1) RAW generations (*.raw.jsonl)
  2) Labeled results (*.jsonl) after answer verification + language compliance.

Generation uses Tinker platform. Answer verification may call OpenAI via
`utils.generate.generate_response_with_retries` (requires OPENAI_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import tinker
from tinker import types
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from lm_eval.tasks.hendrycks_math.utils import last_boxed_only_string, is_equiv, remove_boxed

# If this file lives in, e.g., repo_root/eval/, and utilities live in repo_root/utils/,
# the following line lets `python -m eval.local_eval ...` work without installation.
# If you package the repo (recommended), you can remove this and use proper relative imports.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.language_validation import strip_math_and_symbols, validate_language
from utils.generate import generate_response_with_retries
from utils.prompts import (
    LF_prefix_dict, LF_suffix_dict,
    lang_dict, SUPPORTED_LANGUAGES,
    system_prompt_dict, verify_answer_template
)

# -------------------------
# Global settings
# -------------------------

warnings.filterwarnings("ignore")

# -------------------------
# Small I/O helpers
# -------------------------

def atomic_write_jsonl(path: str, records: Iterable[dict]) -> None:
    """Atomically write an iterable of dicts as JSONL to `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8") as tmp:
        tmp_path = tmp.name
        for rec in records:
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, path)


def read_jsonl(path: str) -> List[dict]:
    """Read a JSONL file into a list of dicts."""
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def file_exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


# -------------------------
# Language Manager
# -------------------------

class LanguageManager:
    """Handles language validation and conversion operations."""

    @staticmethod
    def get_language_code(language: str) -> str:
        """Get dataset column code from language name or code."""
        name_to_code = {name: code for code, name in SUPPORTED_LANGUAGES.items()}

        if language in SUPPORTED_LANGUAGES:
            return language  # already a code

        if language in name_to_code:
            return name_to_code[language]

        raise ValueError(f"Unsupported language: {language}")

    @staticmethod
    def validate_dataset_language(dataset: str, query_language: str) -> bool:
        """Validate that the query language is supported for the given dataset."""
        if not query_language:
            return False
        if dataset not in lang_dict:
            return False
        return query_language in lang_dict[dataset]


# -------------------------
# Prompt Generator
# -------------------------

class PromptGenerator:
    """Handles prompt generation for different models and configurations."""

    @staticmethod
    def system_prompt_select(prompt: str) -> str:
        return prompt

    @staticmethod
    def generate_message(
        model: str,
        query: str,
        tokenizer,
        reasoning_language: str,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> str:
        if prefix:
            message = [{'role': 'user', 'content': f"{query}\n{prefix}"}]
        else:
            system_content = PromptGenerator.system_prompt_select(
                system_prompt_dict.get(reasoning_language, system_prompt_dict["English"])
            )
            message = [{'role': 'user', 'content': f"{query}\n{system_content}"}]

        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        if suffix:
            if "experiments" in model:
                return f"{prompt}<think> {suffix}"
            return f"{prompt} {suffix}"

        return prompt


# -------------------------
# Dataset Loader
# -------------------------

class DatasetLoader:
    """Handles loading and sampling of datasets."""

    @staticmethod
    def _default_benchmarks_dir() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")

    @staticmethod
    def load_dataset_by_name(dataset_name: str, sample: bool = False, benchmarks_dir: Optional[str] = None) -> pd.DataFrame:
        benchmarks_dir = benchmarks_dir or DatasetLoader._default_benchmarks_dir()

        if "math500" in dataset_name:
            df = load_from_disk(os.path.join(benchmarks_dir, "mt-math-500")).to_pandas()
        elif "mmlu" in dataset_name:
            df = load_from_disk(os.path.join(benchmarks_dir, "mmlu-prox-lite")).to_pandas()
        elif "aime2024" in dataset_name:
            df = load_dataset("amphora/MCLM", "MT-AIME2024", split="test").to_pandas()
            if "zh-cn" in df.columns:
                df = df.rename(columns={"zh-cn": "zh"})
        elif "aime2025" in dataset_name:
            df = load_from_disk(os.path.join(benchmarks_dir, "mt-aime-2025")).to_pandas()
        elif "aime-combined" in dataset_name:
            df_2024 = load_dataset("amphora/MCLM", "MT-AIME2024", split="test").to_pandas()
            if "zh-cn" in df_2024.columns:
                df_2024 = df_2024.rename(columns={"zh-cn": "zh"})
            df_2025 = load_from_disk(os.path.join(benchmarks_dir, "mt-aime-2025")).to_pandas()
            common_cols = list(set(df_2024.columns) & set(df_2025.columns))
            if 'id' in common_cols:
                common_cols.remove('id')
            df = pd.concat([df_2024[common_cols], df_2025[common_cols]], ignore_index=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if sample:
            df = df.sample(5, random_state=42)
        return df


# -------------------------
# Response Processing
# -------------------------

class ResponseProcessor:
    """Handles processing and validation of model responses."""

    @staticmethod
    def extract_answer_from_response(response: str, model_path: str) -> str:
        """Return ONLY the last \\boxed{...} answer; empty string if none."""
        if not response:
            return ""
        try:
            ans = last_boxed_only_string(response)
            return (ans or "").strip()
        except Exception:
            return ""

    @staticmethod
    def verify_answer(
        gold_answer: str,
        model_answer: str,
        problem: str = "",
    ) -> bool:
        """
        Single-response answer verification via gpt-5-mini (flex → auto fallback).
        Returns True iff is_equiv(gold_answer, model_answer) or verified 'Correct: true'.
        """
        if not model_answer:
            return False
        if is_equiv(gold_answer, remove_boxed(model_answer)):
            return True
        try:
            prompt = verify_answer_template.format(
                problem=problem,
                gt_solution=gold_answer,
                model_solution=model_answer
            )
            out = generate_response_with_retries(
                [prompt],
                "gpt-5-mini",
                max_completion_tokens=1024,
                temperature=1.0,
                reasoning_effort="medium",
                service_tier="flex"
            )[0]
            if not out:
                out = generate_response_with_retries(
                    [prompt],
                    "gpt-5-mini",
                    max_completion_tokens=1024,
                    temperature=1.0,
                    reasoning_effort="medium",
                    service_tier="auto"
                )[0]

            if not out:
                return False

            txt = out
            if "Correct:" in txt:
                txt = txt.split("Correct:")[1].strip().lower()
            return "true" in txt.lower()
        except Exception:
            return False


# -------------------------
# Results Writer (two-phase)
# -------------------------

class ResultsWriter:
    """Handles writing generation and labeled results."""

    @staticmethod
    def prepare_output_base(output_path: str, dataset: str, model_path: str, language: str, reasoning_language: str) -> Tuple[str, str]:
        """Return (raw_path, final_path)."""
        model_name = model_path.replace('/', '_')
        lang_clean = language.replace(" ", "_")
        reasoning_lang_clean = reasoning_language.replace(" ", "_")
        out_dir = os.path.join(output_path, dataset, model_name)
        os.makedirs(out_dir, exist_ok=True)
        final_path = os.path.join(out_dir, f"{lang_clean}-{reasoning_lang_clean}.jsonl")
        raw_path = final_path.replace(".jsonl", ".raw.jsonl")
        return raw_path, final_path

    @staticmethod
    def iter_raw_records(
        df: pd.DataFrame,
        col_name: str,
        responses: List[List[str]],
        meta: Dict[str, str]
    ) -> Iterable[dict]:
        """Yield raw records (one per sample)."""
        has_level_subject = "level" in df.columns and "subject" in df.columns
        for i in range(len(df)):
            for sample_idx, resp in enumerate(responses[i]):
                rec = {
                    "row_index": i,
                    "sample_idx": sample_idx,
                    "question": str(df[col_name].iloc[i]),
                    "response": str(resp),
                    "answer": str(df["answer"].iloc[i]),
                    **meta,
                }
                if has_level_subject:
                    rec.update({
                        "level": str(df["level"].iloc[i]),
                        "subject": str(df["subject"].iloc[i])
                    })
                yield rec


# -------------------------
# Model Evaluator (LOCAL generation only)
# -------------------------

class ModelEvaluator:
    """Main class for evaluating models on multilingual reasoning tasks via Tinker."""

    def __init__(
        self,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        n_samples: int = 1,
        benchmarks_dir: Optional[str] = None,
        confidence_threshold: float = 0.9,
        compliance_threshold: float = 0.9,
        echo_prompt: bool = False,
        stop_tokens: Optional[List[str]] = None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_samples = n_samples
        self.benchmarks_dir = benchmarks_dir
        self.confidence_threshold = confidence_threshold
        self.compliance_threshold = compliance_threshold
        self.echo_prompt = echo_prompt
        # Default: no stop tokens for math problems to avoid premature truncation
        self.stop_tokens = stop_tokens if stop_tokens is not None else []
        
        # Tinker service client (shared)
        self.service_client = tinker.ServiceClient()
        self.sampling_client = None
        self.tokenizer = None

    def _load_tinker_model(self, base_model: str, lora_path: Optional[str] = None):
        """Load model via Tinker platform."""
        print(f"Initializing Tinker model: {base_model}")
        
        if lora_path:
            # Load fine-tuned model
            print(f"Loading LoRA weights from: {lora_path}")
            training_client = self.service_client.create_lora_training_client(
                base_model=base_model, rank=32
            )
            self.tokenizer = training_client.get_tokenizer()
            self.sampling_client = training_client.load_weights_and_get_sampling_client(
                name=lora_path
            )
        else:
            # Load base model
            training_client = self.service_client.create_lora_training_client(
                base_model=base_model, rank=32
            )
            self.tokenizer = training_client.get_tokenizer()
            self.sampling_client = training_client.save_weights_and_get_sampling_client(
                name="base-model-eval"
            )
        
        print("Model loaded successfully")

    def _generate_with_tinker(
        self,
        questions: List[str],
        model_path: str,
        reasoning_language: str,
        translated_prefix: Optional[str],
        translated_suffix: Optional[str],
    ) -> List[List[str]]:
        """Generate responses using Tinker platform."""
        queries: List[str] = []
        for q in questions:
            queries.append(
                PromptGenerator.generate_message(
                    model_path, q, self.tokenizer, reasoning_language, translated_prefix, translated_suffix
                )
            )
        
        if self.echo_prompt and queries:
            print(f"[debug] Sample prompt:\n{queries[0]}\n---")

        responses: List[List[str]] = []
        
        for query in tqdm(queries, desc="Generating responses"):
            sample_responses = []
            
            # Generate n_samples for each query
            for _ in range(self.n_samples):
                try:
                    prompt_tokens = self.tokenizer.encode(query)
                    prompt_input = types.ModelInput.from_ints(prompt_tokens)
                    
                    params = types.SamplingParams(
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stop=self.stop_tokens if self.stop_tokens else None
                    )
                    
                    future = self.sampling_client.sample(
                        prompt=prompt_input,
                        sampling_params=params,
                        num_samples=1
                    )
                    result = future.result()
                    
                    # Get the generated tokens
                    generated_tokens = result.sequences[0].tokens
                    model_output = self.tokenizer.decode(generated_tokens).strip()
                    
                    # Check if output might be truncated
                    if len(generated_tokens) >= self.max_tokens * 0.95:
                        print(f"Warning: Generated {len(generated_tokens)} tokens (close to max_tokens={self.max_tokens}), output may be truncated")
                    
                    sample_responses.append(model_output)
                    
                except Exception as e:
                    print(f"Error generating response: {e}")
                    sample_responses.append("")
            
            responses.append(sample_responses)
        
        return responses

    def _label_raw_file(
        self,
        raw_path: str,
        final_path: str,
        reasoning_language: str,
        model_path: str,
        dataset_name: str
    ) -> None:
        """
        Read raw JSONL and write labeled final JSONL (atomic), per-record:
          - answer_alone via ResponseProcessor.verify_answer (gpt-5-mini)
          - language_compliance via utils.detect.validate_language (lingua)
          - correct = answer_alone & language_compliance
        """
        print(f"Labeling raw results → {final_path}")
        raw_records = read_jsonl(raw_path)
        labeled_records: List[dict] = []

        for rec in tqdm(raw_records):
            resp = rec.get("response", "") or ""
            # language compliance
            lang_ok, lang_score, lang_chunks, valid_chunks, invalid_chunks = validate_language(
                resp, reasoning_language, self.confidence_threshold, self.compliance_threshold
            )
            # answer verification
            ans = ResponseProcessor.extract_answer_from_response(resp, model_path)
            ans_ok = ResponseProcessor.verify_answer(
                gold_answer=rec.get("answer", ""),
                model_answer=ans,
                problem=rec.get("question", "")
            )
            correct = 1 if (ans_ok and (lang_ok == 1)) else 0

            out = {**rec}
            out["answer_alone"] = 1 if ans_ok else 0
            out["language_compliance"] = lang_ok
            out["language_compliance_score"] = lang_score
            out["language_filtered_chunks"] = lang_chunks
            out["language_invalid_chunks"] = invalid_chunks
            out["correct"] = correct
            labeled_records.append(out)

        atomic_write_jsonl(final_path, labeled_records)
        print(f"Labeled results written: {final_path}")

        # Delete the temporary raw file after successful labeling
        if os.path.exists(raw_path):
            os.remove(raw_path)
            print(f"Temporary raw file deleted: {raw_path}")

    def evaluate_language(
        self,
        df: pd.DataFrame,
        language: str,
        reasoning_language: str,
        model_path: str,
        output_path: str,
        dataset: str,
        language_forcing: bool = False,
        lora_path: Optional[str] = None,
    ) -> None:
        """Evaluate model for a specific language with two-phase persistence."""
        col_name = LanguageManager.get_language_code(language)
        if col_name not in df.columns:
            print(f"Error: Language column '{col_name}' not found. Skipping {language}...")
            return

        raw_path, final_path = ResultsWriter.prepare_output_base(
            output_path, dataset, model_path, language, reasoning_language
        )

        # Case 1: Already finished
        if file_exists_nonempty(final_path):
            print(f"[SKIP] Final labeled results already exist: {final_path}")
            return

        # Case 2: Label-only (no model load)
        if file_exists_nonempty(raw_path) and not file_exists_nonempty(final_path):
            print(f"[RESUME] Raw generations found → labeling only")
            self._label_raw_file(raw_path, final_path, reasoning_language, model_path, dataset)
            return

        # Case 3: Need to generate → load model lazily here
        print(f"[RUN] Generating responses for language='{language}', reasoning_language='{reasoning_language}'")
        questions = df[col_name].values.tolist()

        translated_prefix, translated_suffix = (None, None)
        if language_forcing:
            translated_prefix = LF_prefix_dict[reasoning_language]
            translated_suffix = LF_suffix_dict[reasoning_language]

        try:
            # Load Tinker model
            self._load_tinker_model(model_path, lora_path)
            
            # Generate responses
            responses = self._generate_with_tinker(
                questions, model_path, reasoning_language, translated_prefix, translated_suffix
            )
        finally:
            # Clean up
            self.sampling_client = None
            self.tokenizer = None

        # Phase 1 write: RAW (unlabeled)
        meta = {
            "language": language,
            "reasoning_language": reasoning_language,
            "model": model_path,
            "dataset": dataset,
            "n_samples": self.n_samples
        }
        print(f"Writing RAW generations → {raw_path}")
        raw_records_iter = ResultsWriter.iter_raw_records(df, col_name, responses, meta)
        atomic_write_jsonl(raw_path, raw_records_iter)
        print(f"Raw generations written: {raw_path}")

        # Phase 2: Label
        self._label_raw_file(raw_path, final_path, reasoning_language, model_path, dataset)

    def run_evaluation(
        self,
        model: str,
        dataset: str,
        query_language: str,
        reasoning_language: str,
        sample: bool,
        output_path: str,
        language_forcing: bool = False,
        lora_path: Optional[str] = None,
    ) -> None:
        # Parse model path - support format like "model_path***lora_path" or just "model_path"
        if "***" in model:
            model_path, lora_path = model.split("***", 1)
        else:
            model_path = model
        
        model_name = model_path.replace("/", "_")
        print(f"Preparing run for: {model_name}")
        if lora_path:
            print(f"Using LoRA weights: {lora_path}")

        # Validate language support early
        if not LanguageManager.validate_dataset_language(dataset, query_language):
            print(f"Error: Query language '{query_language}' is not supported for dataset '{dataset}'")
            return

        # Load dataset
        df = DatasetLoader.load_dataset_by_name(dataset, sample, benchmarks_dir=getattr(self, 'benchmarks_dir', None))

        # Delegate to per-language runner
        print(f"Running {model_name} for query language: {query_language} and reasoning language: {reasoning_language}")
        self.evaluate_language(
            df=df,
            language=query_language,
            reasoning_language=reasoning_language,
            model_path=model_path,
            output_path=output_path,
            dataset=dataset,
            language_forcing=language_forcing,
            lora_path=lora_path
        )


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate language models via Tinker platform on multilingual math reasoning tasks"
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name or path (e.g., 'Qwen/Qwen3-8B-Base'). Use format 'model***lora_path' for fine-tuned models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["math500", "mmlu", "aime2024", "aime2025", "aime-combined"],
        help="Dataset name"
    )
    parser.add_argument(
        "--query_language",
        type=str,
        required=True,
        choices=["English", "French", "Swahili", "Japanese", "Latvian", "Chinese", "Telugu", "Thai", "Afrikaans", "Marathi"],
        help="Query language for evaluation"
    )
    parser.add_argument(
        "--reasoning_language",
        type=str,
        required=True,
        choices=["English", "French", "Swahili", "Japanese", "Latvian", "Chinese", "Telugu", "Thai", "Afrikaans", "Marathi"],
        help="Language the model should reason in"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for results"
    )

    # Optional arguments
    parser.add_argument("--sample", action="store_true", help="Use sample data for testing (5 examples)")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per question")
    parser.add_argument("--language_forcing", action="store_true", help="Force model to reason in specified language")
    parser.add_argument("--benchmarks_dir", type=str, default=None, help="Directory containing benchmark datasets")
    parser.add_argument("--echo_prompt", action="store_true", help="Print one sample formatted prompt for debugging")
    parser.add_argument("--stop_tokens", type=str, nargs="*", default=None, 
                        help="Stop tokens for generation (default: none). Example: --stop_tokens '\\n\\nProblem:' '\\n\\n#'")

    # Thresholds for language validation
    parser.add_argument("--confidence_threshold", type=float, default=0.9,
                        help="Minimum lingua confidence for the top language to keep a section (default: 0.9)")
    parser.add_argument("--compliance_threshold", type=float, default=0.9,
                        help="Minimum fraction of kept sections that must be in the target language (default: 0.9)")

    return parser.parse_args()


def main():

    args = parse_args()

    # Require language forcing when reasoning language differs / is not English (as before)
    if args.reasoning_language != args.query_language:
        assert args.language_forcing, (
            f"--language_forcing must be True when reasoning_language ({args.reasoning_language}) differs from query_language ({args.query_language})"
        )
    if args.reasoning_language != "English":
        assert args.language_forcing, (
            f"--language_forcing must be True when reasoning_language ({args.reasoning_language}) is not English"
        )

    evaluator = ModelEvaluator(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n_samples=args.n_samples,
        benchmarks_dir=args.benchmarks_dir,
        confidence_threshold=args.confidence_threshold,
        compliance_threshold=args.compliance_threshold,
        echo_prompt=args.echo_prompt,
        stop_tokens=args.stop_tokens,
    )

    evaluator.run_evaluation(
        model=args.model,
        dataset=args.dataset,
        query_language=args.query_language,
        reasoning_language=args.reasoning_language,
        sample=args.sample,
        output_path=args.output_path,
        language_forcing=args.language_forcing,
    )
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()