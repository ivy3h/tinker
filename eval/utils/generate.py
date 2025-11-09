#!/usr/bin/env python3
"""
Unified text generation for:
  • Local vLLM (Python API)
  • OpenAI Chat Completions API

Features:
  - Optional stop token support on both backends
  - Simple retry with exponential backoff
  - Optional extraction between start/end tags

Environment:
  - OPENAI_API_KEY  (required for OpenAI calls)

Optional vLLM tuning via environment:
  - VLLM_TP_SIZE        (default: auto = number of visible GPUs)
  - VLLM_GPU_MEM_UTIL   (default: 0.90)
  - VLLM_MAX_MODEL_LEN  (default: 32768)

Public entry point:
  generate_response_with_retries(prompts, model, ...)

Notes:
  - If `model` is an OpenAI model (see OPENAI_MODELS), we call OpenAI.
  - Otherwise we assume `model` is a local HF identifier to load with vLLM.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Dict

from dotenv import load_dotenv

# Load .env if present (no-op otherwise)
load_dotenv()

# -------------------------
# OpenAI client (optional)
# -------------------------
OPENAI_MODELS = {
    # Keep in sync with what your pipeline might use
    "gpt-4o-mini", "gpt-4o",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1.nano",
    "gpt-5-nano", "gpt-5-mini", "gpt-5",
    "o1", "o3", "o3-mini", "o4", "o4-mini",
}

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_openai_client = None
try:
    if _OPENAI_KEY:
        from openai import OpenAI  # import only if key present
        _openai_client = OpenAI(api_key=_OPENAI_KEY)
except Exception:
    # If OpenAI SDK is missing or misconfigured, we'll surface a runtime error when used.
    _openai_client = None

# -------------------------
# vLLM local configuration
# -------------------------
_VLLM_ENGINES: Dict[str, "LLM"] = {}

def _get_env_tp_size() -> int:
    """Return tensor parallel size (0 → auto = number of GPUs)."""
    val = os.getenv("VLLM_TP_SIZE", "0")
    try:
        tp = int(val)
        if tp > 0:
            return tp
    except Exception:
        pass
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1

def _get_env_gpu_mem_util() -> float:
    try:
        return float(os.getenv("VLLM_GPU_MEM_UTIL", "0.90"))
    except Exception:
        return 0.90

def _get_env_max_model_len() -> int:
    try:
        return int(os.getenv("VLLM_MAX_MODEL_LEN", "32768"))
    except Exception:
        return 32786

def _get_vllm_engine(model: str) -> "LLM":
    """Lazy-create and cache a vLLM engine for a given model name/path."""
    if model in _VLLM_ENGINES:
        return _VLLM_ENGINES[model]
    try:
        from vllm import LLM
    except Exception as e:
        raise RuntimeError(
            "vLLM is required for local generation but is not installed or failed to import."
        ) from e

    engine = LLM(
        model=model,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=_get_env_tp_size(),
        gpu_memory_utilization=_get_env_gpu_mem_util(),
        max_model_len=_get_env_max_model_len(),
        enforce_eager=True,
    )
    _VLLM_ENGINES[model] = engine
    return engine

def _vllm_generate(
    prompts: List[str],
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    stop_tokens: Optional[List[str]],
) -> List[str]:
    """Generate with local vLLM (first candidate per prompt)."""
    try:
        from vllm import SamplingParams
    except Exception as e:
        raise RuntimeError(
            "vLLM is required for local generation but is not installed or failed to import."
        ) from e

    llm = _get_vllm_engine(model)
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_tokens or None,
    )
    outputs = llm.generate(prompts, sp)
    return [(o.outputs[0].text if o.outputs else "") for o in outputs]

# -------------------------
# Helpers
# -------------------------

def _between_tags(text: str, start: Optional[str], end: Optional[str]) -> str:
    """Return substring between start/end tags if both present; otherwise return original text."""
    if not text:
        return ""
    if start and end:
        s_idx = text.find(start)
        if s_idx != -1:
            s_idx += len(start)
            e_idx = text.find(end, s_idx)
            if e_idx != -1:
                return text[s_idx:e_idx].strip()
    return text.strip()

def _sleep_backoff(attempt: int) -> None:
    # exponential backoff with jitter
    base = 0.75
    delay = base * (2 ** attempt)
    time.sleep(min(delay, 8.0))

# -------------------------
# Public API
# -------------------------

def generate_response_with_retries(
    prompts: List[str],
    model: str,
    start_tag: Optional[str] = None,
    end_tag: Optional[str] = None,
    *,
    max_completion_tokens: int = 4096,
    temperature: float = 0.0,
    stop_tokens: Optional[List[str]] = None,
    num_retries: int = 3,
    reasoning_effort: Optional[str] = None,  # forwarded to OpenAI if provided
    service_tier: Optional[str] = None       # forwarded to OpenAI if provided
) -> List[str]:
    """
    Generate text for each prompt.

    Args:
      prompts: list of user-visible prompts (already formatted if needed)
      model:   OpenAI model name (in OPENAI_MODELS) or local HF model id/path for vLLM
      start_tag, end_tag: if both provided, extract substring between them from each output
      max_completion_tokens: tokens to generate (approx)
      temperature: sampling temperature
      stop_tokens: optional list of stop sequences
      num_retries: total retry attempts on transient errors
      reasoning_effort, service_tier: forwarded to OpenAI (ignored by vLLM)

    Returns:
      List[str] aligned with `prompts`.
    """
    if not prompts:
        return []

    for attempt in range(num_retries + 1):
        try:
            # ---- OpenAI path ----
            if model in OPENAI_MODELS:
                if _openai_client is None:
                    raise RuntimeError(
                        "OpenAI API key not configured or OpenAI SDK unavailable."
                    )
                outputs: List[str] = []
                for p in prompts:
                    resp = _openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": p}],
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                        stop=stop_tokens or None,
                        reasoning_effort=reasoning_effort if reasoning_effort else None,
                        service_tier=service_tier if service_tier else None,
                    )
                    txt = ""
                    try:
                        # SDK returns list of choices; take first message content
                        txt = (resp.choices[0].message.content or "") if resp.choices else ""
                    except Exception:
                        txt = ""
                    outputs.append(_between_tags(txt, start_tag, end_tag))
                return outputs

            # ---- Local vLLM path ----
            raw = _vllm_generate(
                prompts,
                model,
                max_tokens=max_completion_tokens,
                temperature=temperature,
                stop_tokens=stop_tokens,
            )
            return [_between_tags(x, start_tag, end_tag) for x in raw]

        except Exception as e:
            # Last attempt -> return blanks to preserve alignment
            print(f"[generate] Attempt {attempt + 1}/{num_retries} failed: {e}")
            if attempt == num_retries:
                return [""] * len(prompts)
            _sleep_backoff(attempt)

    # Unreachable, but keeps type-checkers happy
    return [""] * len(prompts)