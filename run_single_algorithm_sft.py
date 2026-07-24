#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def maybe_load_repo_config(repo_root: Path) -> Dict[str, Any]:
    cfg_path = repo_root / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_flat_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {path} at line {line_number}")
            rows.append(row)
    return rows


def parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def row_matches_model_filters(
    row: Dict[str, Any],
    *,
    include_model_families: set[str],
    exclude_model_families: set[str],
    include_model_keys: set[str],
    exclude_model_keys: set[str],
    include_model_sizes: set[str],
    exclude_model_sizes: set[str],
) -> bool:
    model_family = row.get("model_family")
    model_key = row.get("model_key")
    model_size = row.get("model_size")
    if include_model_families and model_family not in include_model_families:
        return False
    if model_family in exclude_model_families:
        return False
    if include_model_keys and model_key not in include_model_keys:
        return False
    if model_key in exclude_model_keys:
        return False
    if include_model_sizes and model_size not in include_model_sizes:
        return False
    if model_size in exclude_model_sizes:
        return False
    return True


def count_rows_for_algorithm(
    rows: List[Dict[str, Any]],
    algorithm: str,
    *,
    include_model_families: set[str] | None = None,
    exclude_model_families: set[str] | None = None,
    include_model_keys: set[str] | None = None,
    exclude_model_keys: set[str] | None = None,
    include_model_sizes: set[str] | None = None,
    exclude_model_sizes: set[str] | None = None,
) -> Dict[str, Any]:
    include_model_families = include_model_families or set()
    exclude_model_families = exclude_model_families or set()
    include_model_keys = include_model_keys or set()
    exclude_model_keys = exclude_model_keys or set()
    include_model_sizes = include_model_sizes or set()
    exclude_model_sizes = exclude_model_sizes or set()
    pair_counts: Counter[str] = Counter()
    model_family_counts: Counter[str] = Counter()
    model_key_counts: Counter[str] = Counter()
    model_size_counts: Counter[str] = Counter()
    total = 0
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        if not row_matches_model_filters(
            row,
            include_model_families=include_model_families,
            exclude_model_families=exclude_model_families,
            include_model_keys=include_model_keys,
            exclude_model_keys=exclude_model_keys,
            include_model_sizes=include_model_sizes,
            exclude_model_sizes=exclude_model_sizes,
        ):
            continue
        length = row.get("length")
        total += 1
        if isinstance(length, int):
            pair_counts[str(length)] += 1
        if row.get("model_family"):
            model_family_counts[str(row["model_family"])] += 1
        if row.get("model_key"):
            model_key_counts[str(row["model_key"])] += 1
        if row.get("model_size"):
            model_size_counts[str(row["model_size"])] += 1
    return {
        "rows": total,
        "length_counts": dict(sorted(pair_counts.items(), key=lambda item: int(item[0]))),
        "model_family_counts": dict(sorted(model_family_counts.items())),
        "model_key_counts": dict(sorted(model_key_counts.items())),
        "model_size_counts": dict(sorted(model_size_counts.items())),
    }


def resolve_python_executable(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def shell_join(parts: List[str]) -> str:
    return shlex.join(parts)


def run_command(cmd: List[str], *, cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    print(f"[CMD] {shell_join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def latest_json_in_dir(path: Path) -> Path | None:
    candidates = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def default_run_name(algorithm: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{algorithm}_unweighted_sft_{stamp}"


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    cfg = maybe_load_repo_config(repo_root)
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    testing_cfg = cfg.get("testing", {}) if isinstance(cfg.get("testing"), dict) else {}

    default_train_dataset = repo_root / "dataset" / "training_rows.flat.jsonl"
    default_test_dataset = repo_root / "dataset" / "curriculum_test_set.flat.jsonl"
    default_output_root = repo_root / "runs" / "single_algorithm_sft"

    ap = argparse.ArgumentParser(
        description="Train on one algorithm with unweighted sampling SFT, then evaluate that algorithm on the flattened test set.",
    )
    ap.add_argument("--algorithm", required=True, help="Algorithm name to train and evaluate, e.g. coin_change.")
    ap.add_argument("--train_dataset_path", default=str(training_cfg.get("dataset_path", default_train_dataset)))
    ap.add_argument("--test_dataset_path", default=str(testing_cfg.get("flat_dataset_path", default_test_dataset)))
    ap.add_argument("--model_name", default=model_cfg.get("model_name", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--output_root", default=str(default_output_root))
    ap.add_argument("--prompt_field", choices=["prompt", "immediate_answer_prompt"], default=training_cfg.get("prompt_field", "prompt"))
    ap.add_argument(
        "--eval_prompt_type",
        choices=["thinking_prompt", "immediate_answer_prompt"],
        default=testing_cfg.get("prompt_type", "thinking_prompt"),
    )
    ap.add_argument("--epochs", type=float, default=training_cfg.get("epochs", 2))
    ap.add_argument("--learning_rate", type=float, default=training_cfg.get("learning_rate", 2e-4))
    ap.add_argument("--batch_size", type=int, default=training_cfg.get("batch_size", 1))
    ap.add_argument("--grad_accum", type=int, default=training_cfg.get("grad_accum", 16))
    ap.add_argument("--max_seq_len", type=int, default=training_cfg.get("max_seq_len", 16384))
    ap.add_argument("--seed", type=int, default=training_cfg.get("seed", 42))
    ap.add_argument("--include_model_families", default=",".join(training_cfg.get("include_model_families", [])))
    ap.add_argument("--exclude_model_families", default=",".join(training_cfg.get("exclude_model_families", [])))
    ap.add_argument("--include_model_keys", default=",".join(training_cfg.get("include_model_keys", [])))
    ap.add_argument("--exclude_model_keys", default=",".join(training_cfg.get("exclude_model_keys", [])))
    ap.add_argument("--include_model_sizes", default=",".join(training_cfg.get("include_model_sizes", [])))
    ap.add_argument("--exclude_model_sizes", default=",".join(training_cfg.get("exclude_model_sizes", [])))
    ap.add_argument(
        "--gpu_id",
        default=str(training_cfg.get("gpu_id", testing_cfg.get("gpu_id", 1))),
        help='Default GPU id used for both training and evaluation unless stage-specific flags override it.',
    )
    ap.add_argument("--train_gpu_id", default=None, help='Sets CUDA_VISIBLE_DEVICES for training, e.g. "0".')
    ap.add_argument("--eval_gpu_id", default=testing_cfg.get("gpu_id"), help='Passed to eval.py --gpu_id, e.g. "1".')
    ap.add_argument("--eval_batch_size", type=int, default=testing_cfg.get("batch_size", 32))
    ap.add_argument("--max_new_tokens", type=int, default=testing_cfg.get("max_new_tokens", 8192))
    ap.add_argument("--temperature", type=float, default=testing_cfg.get("temperature", 0.0))
    ap.add_argument("--top_p", type=float, default=testing_cfg.get("top_p", 1.0))
    ap.add_argument("--max_model_len", type=int, default=testing_cfg.get("max_model_len", 8192))
    ap.add_argument("--gpu_mem_util", type=float, default=testing_cfg.get("gpu_mem_util", 0.90))
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "auto"], default=testing_cfg.get("dtype", "bfloat16"))
    ap.add_argument("--repetition_penalty", type=float, default=testing_cfg.get("repetition_penalty", 1.0))
    ap.add_argument("--system_prompt", default=testing_cfg.get("system_prompt", "Please solve the following algorithmic problem without using programming languages."))
    ap.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default=training_cfg.get("wandb_mode", "online"))
    ap.add_argument("--wandb_project", default=cfg.get("wandb", {}).get("project") if isinstance(cfg.get("wandb"), dict) else None)
    ap.add_argument("--wandb_entity", default=cfg.get("wandb", {}).get("entity") if isinstance(cfg.get("wandb"), dict) else None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    args = build_parser(repo_root).parse_args()

    algorithm = args.algorithm.strip()
    if not algorithm:
        raise ValueError("--algorithm must be non-empty")

    train_dataset_path = Path(args.train_dataset_path).expanduser()
    if not train_dataset_path.is_absolute():
        train_dataset_path = (repo_root / train_dataset_path).resolve()
    test_dataset_path = Path(args.test_dataset_path).expanduser()
    if not test_dataset_path.is_absolute():
        test_dataset_path = (repo_root / test_dataset_path).resolve()

    if not train_dataset_path.exists():
        raise FileNotFoundError(f"training dataset not found: {train_dataset_path}")
    if not test_dataset_path.exists():
        raise FileNotFoundError(f"test dataset not found: {test_dataset_path}")

    training_rows = load_flat_jsonl(train_dataset_path)
    test_rows = load_flat_jsonl(test_dataset_path)
    available_train_algorithms = sorted({row.get("algorithm") for row in training_rows if isinstance(row.get("algorithm"), str)})
    available_test_algorithms = sorted({row.get("algorithm") for row in test_rows if isinstance(row.get("algorithm"), str)})

    if algorithm not in available_train_algorithms:
        raise RuntimeError(
            f"Algorithm {algorithm!r} not found in training dataset. Available: {available_train_algorithms}"
        )
    if algorithm not in available_test_algorithms:
        raise RuntimeError(
            f"Algorithm {algorithm!r} not found in test dataset. Available: {available_test_algorithms}"
        )

    include_model_families = set(parse_csv_list(args.include_model_families))
    exclude_model_families = set(parse_csv_list(args.exclude_model_families))
    include_model_keys = set(parse_csv_list(args.include_model_keys))
    exclude_model_keys = set(parse_csv_list(args.exclude_model_keys))
    include_model_sizes = set(parse_csv_list(args.include_model_sizes))
    exclude_model_sizes = set(parse_csv_list(args.exclude_model_sizes))

    train_stats = count_rows_for_algorithm(
        training_rows,
        algorithm,
        include_model_families=include_model_families,
        exclude_model_families=exclude_model_families,
        include_model_keys=include_model_keys,
        exclude_model_keys=exclude_model_keys,
        include_model_sizes=include_model_sizes,
        exclude_model_sizes=exclude_model_sizes,
    )
    test_stats = count_rows_for_algorithm(test_rows, algorithm)
    if train_stats["rows"] <= 0:
        raise RuntimeError(f"No training rows found for algorithm {algorithm!r}")
    if test_stats["rows"] <= 0:
        raise RuntimeError(f"No test rows found for algorithm {algorithm!r}")

    run_name = args.run_name or default_run_name(algorithm)
    effective_train_gpu_id = args.train_gpu_id if args.train_gpu_id is not None else args.gpu_id
    effective_eval_gpu_id = args.eval_gpu_id if args.eval_gpu_id is not None else args.gpu_id
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    run_dir = output_root / run_name
    adapter_dir = run_dir / "adapter"
    eval_log_dir = run_dir / "eval_logs"
    summary_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    python_exec = resolve_python_executable(repo_root)

    train_cmd = [
        python_exec,
        str(repo_root / "train.py"),
        "--dataset_path",
        str(train_dataset_path),
        "--model_name",
        args.model_name,
        "--output_dir",
        str(adapter_dir),
        "--gpu_id",
        str(effective_train_gpu_id),
        "--epochs",
        str(args.epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum",
        str(args.grad_accum),
        "--max_seq_len",
        str(args.max_seq_len),
        "--seed",
        str(args.seed),
        "--prompt_field",
        args.prompt_field,
        "--train_mode",
        "random",
        "--include_algorithms",
        algorithm,
        "--exclude_algorithms",
        "",
        "--system_prompt",
        args.system_prompt,
        "--wandb_mode",
        args.wandb_mode,
    ]
    for flag_name in [
        "include_model_families",
        "exclude_model_families",
        "include_model_keys",
        "exclude_model_keys",
        "include_model_sizes",
        "exclude_model_sizes",
    ]:
        value = getattr(args, flag_name)
        if value:
            train_cmd.extend([f"--{flag_name}", value])
    if args.wandb_project:
        train_cmd.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity:
        train_cmd.extend(["--wandb_entity", args.wandb_entity])
    train_cmd.extend(["--wandb_name", args.wandb_name or run_name])

    eval_cmd = [
        python_exec,
        str(repo_root / "eval.py"),
        "--dataset_path",
        str(test_dataset_path),
        "--algorithms",
        algorithm,
        "--model_name",
        args.model_name,
        "--lora_checkpoint",
        str(adapter_dir),
        "--prompt_type",
        args.eval_prompt_type,
        "--batch_size",
        str(args.eval_batch_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_model_len",
        str(args.max_model_len),
        "--gpu_mem_util",
        str(args.gpu_mem_util),
        "--dtype",
        args.dtype,
        "--repetition_penalty",
        str(args.repetition_penalty),
        "--system_prompt",
        args.system_prompt,
        "--log_dir",
        str(eval_log_dir),
        "--run_name",
        run_name,
    ]
    if effective_eval_gpu_id is not None and str(effective_eval_gpu_id).strip():
        eval_cmd.extend(["--gpu_id", str(effective_eval_gpu_id).strip()])

    summary: Dict[str, Any] = {
        "algorithm": algorithm,
        "run_name": run_name,
        "train_dataset_path": str(train_dataset_path),
        "test_dataset_path": str(test_dataset_path),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "run_dir": str(run_dir),
        "adapter_dir": str(adapter_dir),
        "eval_log_dir": str(eval_log_dir),
        "train_mode": "random",
        "training_model_filters": {
            "include_model_families": sorted(include_model_families),
            "exclude_model_families": sorted(exclude_model_families),
            "include_model_keys": sorted(include_model_keys),
            "exclude_model_keys": sorted(exclude_model_keys),
            "include_model_sizes": sorted(include_model_sizes),
            "exclude_model_sizes": sorted(exclude_model_sizes),
        },
        "train_command": train_cmd,
        "eval_command": eval_cmd,
        "gpu_id": args.gpu_id,
        "train_gpu_id": effective_train_gpu_id,
        "eval_gpu_id": effective_eval_gpu_id,
        "started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] algorithm={algorithm}")
    print(f"[INFO] train_rows={train_stats['rows']} length_counts={train_stats['length_counts']}")
    if train_stats.get("model_family_counts"):
        print(f"[INFO] train_model_families={train_stats['model_family_counts']}")
    if train_stats.get("model_size_counts"):
        print(f"[INFO] train_model_sizes={train_stats['model_size_counts']}")
    print(f"[INFO] test_rows={test_stats['rows']} length_counts={test_stats['length_counts']}")
    print(f"[INFO] train_gpu_id={effective_train_gpu_id} eval_gpu_id={effective_eval_gpu_id}")
    print(f"[INFO] run_dir={run_dir}")

    train_env = os.environ.copy()
    if effective_train_gpu_id is not None and str(effective_train_gpu_id).strip():
        train_env["CUDA_VISIBLE_DEVICES"] = str(effective_train_gpu_id).strip()

    run_command(train_cmd, cwd=repo_root, env=train_env, dry_run=args.dry_run)
    run_command(eval_cmd, cwd=repo_root, env=os.environ.copy(), dry_run=args.dry_run)

    if args.dry_run:
        return

    eval_json_path = latest_json_in_dir(eval_log_dir)
    if eval_json_path is not None:
        try:
            eval_payload = json.loads(eval_json_path.read_text(encoding="utf-8"))
        except Exception:
            eval_payload = None
        if isinstance(eval_payload, dict):
            summary["eval_json_path"] = str(eval_json_path)
            summary["eval_results"] = eval_payload.get("results", [])
            summary["eval_pair_results"] = eval_payload.get("pair_results", [])

    summary["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    algorithm_results = summary.get("eval_results") or []
    metric_row = next((row for row in algorithm_results if row.get("algorithm") == algorithm), None)
    if metric_row is not None:
        print(
            f"[DONE] {algorithm} accuracy={metric_row.get('accuracy'):.4f} "
            f"({metric_row.get('correct')}/{metric_row.get('total')})"
        )
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
