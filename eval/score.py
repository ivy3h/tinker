#!/usr/bin/env python3
"""
Score multilingual reasoning evaluation outputs.

Reads the final labeled JSONL files produced by the evaluator:
results/<dataset>/<model>/<Language-ReasoningLanguage>.jsonl

Computes accuracy per language pair and writes one CSV per dataset:
score_result/<dataset>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm


# -------------------------
# Core scoring helpers
# -------------------------

def _read_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSONL file to DataFrame or raise a helpful error."""
    try:
        df = pd.read_json(path, lines=True)
        if df.empty:
            print(f"[warn] Empty file: {path}")
        return df
    except ValueError as e:
        raise ValueError(f"Failed to read JSONL at {path}: {e}") from e


def _compute_accuracy(df: pd.DataFrame) -> float:
    """
    Compute accuracy from a labeled results DataFrame.

    Preferred:
      - Column `correct` in {0,1}
    Fallbacks:
      - If `answer_alone` AND `language_compliance` present:
          correct = int(answer_alone == 1 and language_compliance == 1)
      - If only `answer_alone` present:
          correct = int(answer_alone == 1)

    Returns float in [0,1]. Returns 0.0 if df is empty.
    """
    if df is None or len(df) == 0:
        return 0.0

    if "correct" in df.columns:
        try:
            return float(pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int).sum()) / len(df)
        except Exception:
            pass

    if {"answer_alone", "language_compliance"}.issubset(df.columns):
        aa = pd.to_numeric(df["answer_alone"], errors="coerce").fillna(0).astype(int)
        lc = pd.to_numeric(df["language_compliance"], errors="coerce").fillna(0).astype(int)
        correct_series = (aa.eq(1) & lc.eq(1)).astype(int)
        return float(correct_series.sum()) / len(df)

    if "answer_alone" in df.columns:
        aa = pd.to_numeric(df["answer_alone"], errors="coerce").fillna(0).astype(int)
        return float(aa.sum()) / len(df)

    print("[warn] No suitable columns to compute accuracy (need `correct` or fallbacks). Returning 0.0")
    return 0.0


def _require_pair_language(lang: str) -> str:
    """
    Require new-style language pair strings (e.g., 'English-Japanese').
    Returns filename '<pair>.jsonl' or raises ValueError.
    """
    if "-" not in lang:
        raise ValueError(
            f"Invalid --languages entry '{lang}'. Expected a pair like 'English-Japanese'."
        )
    return f"{lang}.jsonl"


def _list_dirs(path: Path) -> List[str]:
    return sorted([p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _list_jsonl(path: Path) -> List[str]:
    return sorted([p.name for p in path.iterdir() if p.is_file() and p.suffix == ".jsonl" and "raw" not in p.name])


# -------------------------
# Main scoring routine
# -------------------------

def score_tree(
    root_path: Path,
    models: Optional[Sequence[str]],
    datasets: Optional[Sequence[str]],
    languages: Optional[Sequence[str]],
    output_dir: Path,
    decimals: int = 3,
    add_avg: bool = True,
) -> None:
    """
    Traverse results/<dataset>/<model> and compute accuracy per language file.
    Writes one CSV per dataset with columns: model, <Language-ReasoningLanguage...>, [AVG].
    """
    if not root_path.exists():
        raise FileNotFoundError(f"--root_path does not exist: {root_path}")

    ds_list = list(datasets) if datasets else _list_dirs(root_path)
    if not ds_list:
        print(f"[warn] No datasets found under {root_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in ds_list:
        ds_path = root_path / ds
        if not ds_path.exists():
            print(f"[warn] Dataset path does not exist: {ds_path}")
            continue

        print(f"\n=== Scoring dataset: {ds} ===")

        model_list = list(models) if models else _list_dirs(ds_path)
        if not model_list:
            print(f"[warn] No models found under {ds_path}; skipping")
            continue

        # Determine the union of language filenames to score
        all_lang_files: set[str] = set()
        if languages:
            for lang in languages:
                all_lang_files.add(_require_pair_language(lang))
        else:
            # Discover from filesystem
            for m in model_list:
                mdir = ds_path / m
                if mdir.exists():
                    for f in _list_jsonl(mdir):
                        all_lang_files.add(f)
        lang_files_sorted = sorted(all_lang_files)

        # Prepare results table
        results: Dict[str, List[Optional[float]]] = {"model": []}
        for lf in lang_files_sorted:
            results[lf.replace(".jsonl", "")] = []

        # Score each model
        for m in model_list:
            mdir = ds_path / m
            if not mdir.exists():
                print(f"[warn] Missing model dir: {mdir}")
                continue

            results["model"].append(m)
            for lf in lang_files_sorted:
                stem = lf.replace(".jsonl", "")
                fpath = mdir / lf
                if not fpath.exists():
                    print(f"[warn] Missing file: {fpath}")
                    results[stem].append(None)
                    continue

                try:
                    df = _read_jsonl(fpath)
                    acc = _compute_accuracy(df)
                    results[stem].append(round(acc, decimals))
                    print(f"[ok] {m} · {lf}: {round(acc, decimals)}")
                except Exception as e:
                    print(f"[error] {m} · {lf}: {e}")
                    results[stem].append(None)

        # Write CSV
        if results["model"]:
            df_out = pd.DataFrame(results)

            if add_avg:
                lang_cols = [c for c in df_out.columns if c != "model"]
                df_out["AVG"] = df_out[lang_cols].mean(axis=1, skipna=True).round(decimals)

            out_csv = output_dir / f"{ds}.csv"
            df_out.to_csv(out_csv, index=False)
            print(f"[done] Wrote {out_csv}")
        else:
            print(f"[info] No rows collected for dataset {ds}; nothing to write.")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score multilingual reasoning evaluation outputs into per-dataset CSVs."
    )
    parser.add_argument("--root_path", type=str, required=True,
                        help="Root directory containing results (e.g., 'results').")
    parser.add_argument("--models", type=str, nargs="*",
                        help="Model names to score (space-separated). If omitted, score all models.")
    parser.add_argument("--datasets", type=str, nargs="*",
                        help="Dataset names to score (space-separated). If omitted, score all datasets.")
    parser.add_argument("--languages", type=str, nargs="*",
                        help="Language PAIRS to score (e.g., 'English-Japanese' 'French-French').")
    parser.add_argument("--output_dir", type=str, default="score_result",
                        help="Directory to write CSVs (default: score_result).")
    parser.add_argument("--decimals", type=int, default=3,
                        help="Round accuracies to this many decimals (default: 3).")
    parser.add_argument("--no_avg", action="store_true",
                        help="Do not add AVG column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_path = Path(args.root_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"Root path   : {root_path}")
    print(f"Models      : {args.models if args.models else 'ALL'}")
    print(f"Datasets    : {args.datasets if args.datasets else 'ALL'}")
    print(f"Languages   : {args.languages if args.languages else 'ALL discovered'}")
    print(f"Output dir  : {output_dir}")
    print(f"Decimals    : {args.decimals}")
    print(f"Add AVG     : {not args.no_avg}")

    score_tree(
        root_path=root_path,
        models=args.models,
        datasets=args.datasets,
        languages=args.languages,
        output_dir=output_dir,
        decimals=args.decimals,
        add_avg=not args.no_avg,
    )
    print("Scoring completed successfully!")


if __name__ == "__main__":
    main()