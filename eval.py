#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import datetime
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import utils


NAME_RE = re.compile(r"^(?P<algorithm>.+)_(?P<length>\d+)_(?P<idx>\d+)$")

_NAME_RE_CACHE: Dict[str, re.Pattern] = {}
_NAME_PREFIX_ALIASES: Dict[str, Tuple[str, ...]] = {
    "max_avg_subarray_fixed_k_window": ("max_avg_subarray_fixed_k",),
    "max_avg_subarray_variable_k_window": ("max_avg_subarray_variable_k",),
}
_EXTRACTOR_PARAM_ALIASES = {
    "original_s": ("s",),
    "original_nums": ("nums",),
}


def maybe_load_repo_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).resolve().with_name("config.yaml")
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


def maybe_set_cuda_visible_devices(gpu_id: str | None) -> str | None:
    if gpu_id is None:
        return None
    gpu_id = str(gpu_id).strip()
    if not gpu_id:
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return gpu_id


def parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_name(name: str) -> Dict[str, Any] | None:
    if not isinstance(name, str):
        return None
    match = NAME_RE.match(name)
    if match is None:
        return None
    return {
        "algorithm": match.group("algorithm"),
        "length": int(match.group("length")),
        "idx": int(match.group("idx")),
    }


def _get_name_re(algorithm: str) -> re.Pattern:
    if algorithm not in _NAME_RE_CACHE:
        _NAME_RE_CACHE[algorithm] = re.compile(rf"^{re.escape(algorithm)}_(\d+)_(\d+)$")
    return _NAME_RE_CACHE[algorithm]


def iter_name_prefixes(algorithm: str) -> Tuple[str, ...]:
    aliases = _NAME_PREFIX_ALIASES.get(algorithm, ())
    return (algorithm, *aliases)


def parse_len_idx_from_name(algorithm: str, name: str) -> Tuple[int, int] | None:
    if not isinstance(name, str):
        return None
    for prefix in iter_name_prefixes(algorithm):
        match = _get_name_re(prefix).match(name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


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


def build_model_input(tok, prompt: str, system_prompt: str | None = None) -> str:
    system_prompt = (system_prompt or "").strip()
    if hasattr(tok, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    if system_prompt:
        return system_prompt + "\n\n" + prompt
    return prompt


def get_extractor(algorithm: str):
    candidate_names = [f"extract_{algorithm}_from_text"]
    alias_names = {
        "vector_mean": "extract_vector_mean_calculation_from_text",
    }
    alias_name = alias_names.get(algorithm)
    if alias_name and alias_name not in candidate_names:
        candidate_names.append(alias_name)

    for fn_name in candidate_names:
        fn = getattr(utils, fn_name, None)
        if fn is not None:
            return fn_name, fn

    return candidate_names[0], None


def _get_extractor_param(params: Dict[str, Any], name: str):
    if name in params:
        return params[name]
    for alias in _EXTRACTOR_PARAM_ALIASES.get(name, ()):
        if alias in params:
            return params[alias]
    raise KeyError(name)


def _normalize_extractor_output(result):
    if isinstance(result, tuple) and len(result) >= 2:
        cand, ok = result[0], result[1]
        return cand, bool(ok)
    raise TypeError(f"extractor returned unsupported result: {result!r}")


def call_extractor(fn, text: str, params):
    sig = inspect.signature(fn)
    tail = list(sig.parameters.values())[1:]

    if not isinstance(params, dict):
        return fn(text, params)

    if tail and tail[0].name == "params":
        kwargs = {}
        for param in tail[1:]:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            try:
                kwargs[param.name] = _get_extractor_param(params, param.name)
            except KeyError:
                continue
        return _normalize_extractor_output(fn(text, params, **kwargs))

    args = []
    kwargs = {}
    for param in tail:
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            try:
                args.append(_get_extractor_param(params, param.name))
            except KeyError:
                if param.default is inspect.Parameter.empty:
                    raise KeyError(f'missing param "{param.name}" for extractor {fn.__name__}')
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            try:
                kwargs[param.name] = _get_extractor_param(params, param.name)
            except KeyError:
                if param.default is inspect.Parameter.empty:
                    raise KeyError(f'missing param "{param.name}" for extractor {fn.__name__}')

    return _normalize_extractor_output(fn(text, *args, **kwargs))


def to_jsonable(value: Any) -> Any:
    if value is Ellipsis:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return repr(value)


def safe_json_dump(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2, default=repr),
        encoding="utf-8",
    )


def contains_ellipsis(value: Any) -> bool:
    if value is Ellipsis:
        return True
    if isinstance(value, dict):
        return any(contains_ellipsis(k) or contains_ellipsis(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(contains_ellipsis(v) for v in value)
    return False


def _try(cmd: List[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def get_env_info() -> Dict[str, Any]:
    info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": _try(["git", "rev-parse", "HEAD"]),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    try:
        import vllm

        info["vllm"] = getattr(vllm, "__version__", None)
    except Exception:
        info["vllm"] = None
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        except Exception:
            info["gpu_name"] = None
            info["gpu_count"] = None
    else:
        info["gpu_name"] = None
        info["gpu_count"] = 0
    return info


def hr_kv(title: str, data: Dict[str, Any]) -> str:
    lines = [title] if title else []
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def format_sample_block(sample: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"[Sample] algorithm={sample.get('algorithm')} length={sample.get('length')} "
        f"idx={sample.get('idx')} ok={sample.get('ok')}"
    )
    lines.append(f"[Meta] name={sample.get('name')} prompt_type={sample.get('prompt_type')}")
    if sample.get("gen_error"):
        lines.append(f"[Generation] ERROR: {sample.get('gen_error')}")
    if sample.get("extract_error"):
        lines.append(f"[Extraction] ERROR: {sample.get('extract_error')}")
    lines.append("-" * 80)
    lines.append("[Prompt]")
    lines.append(str(sample.get("prompt", "")))
    lines.append("-" * 80)
    lines.append("[Params]")
    lines.append(json.dumps(to_jsonable(sample.get("params", {})), ensure_ascii=False, indent=2))
    lines.append("-" * 80)
    lines.append("[Model Output]")
    lines.append(str(sample.get("text", "")))
    lines.append("-" * 80)
    lines.append("[Extractor Output]")
    lines.append(
        json.dumps(
            to_jsonable({"cand": sample.get("cand"), "ok": sample.get("ok")}),
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("=" * 80)
    return "\n".join(lines)


def write_human_log_txt(path: Path, run_record: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Flat-Dataset Evaluation Run\n")
    lines.append("## Environment")
    lines.append(hr_kv("", run_record.get("env", {})))
    lines.append("")
    lines.append("## Config")
    lines.append(hr_kv("", run_record.get("config", {})))
    lines.append("")
    lines.append("## Algorithm Results")
    for row in run_record.get("results", []):
        lines.append(
            f"- [{row['algorithm']}] acc={row['accuracy']:.4f} ({row['correct']}/{row['total']}) "
            f"gen_errors={row['gen_errors']} extract_errors={row['extract_errors']} "
            f"avg_gen_tokens={row['avg_gen_tokens']:.1f} tokens_per_s={row['gen_tokens_per_s']:.2f}"
        )
    lines.append("")
    lines.append("## Pair Results")
    for row in run_record.get("pair_results", []):
        lines.append(
            f"- [{row['algorithm']}:{row['length']}] acc={row['accuracy']:.4f} ({row['correct']}/{row['total']}) "
            f"gen_errors={row['gen_errors']} extract_errors={row['extract_errors']}"
        )
    lines.append("")
    lines.append("## Samples")
    lines.append(f"Total samples logged: {len(run_record.get('samples', []))}")
    lines.append("")
    for sample in run_record.get("samples", []):
        lines.append(format_sample_block(sample))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_final_summary(run_record: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (per algorithm)")
    print("=" * 80)
    for row in run_record.get("results", []):
        print(f"{row['algorithm']:20s}  acc={row['accuracy']:.4f}  ({row['correct']}/{row['total']})")
    print("=" * 80 + "\n")


def maybe_make_merged_model_dir(
    base_model_name_or_path: str,
    lora_dir: str | None,
    out_dir: Path | None,
    dtype_str: str,
) -> str:
    if not lora_dir:
        return base_model_name_or_path

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("You passed --lora_checkpoint but peft is not installed. `pip install peft`") from exc

    lora_path = Path(lora_dir).expanduser().resolve()
    if not lora_path.exists():
        raise FileNotFoundError(f"lora_checkpoint not found: {lora_path}")

    temp_parent = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_parent = str(out_dir)
    runtime_dir = Path(tempfile.mkdtemp(prefix="merged_model_", dir=temp_parent)).resolve()
    atexit.register(shutil.rmtree, runtime_dir, ignore_errors=True)

    if dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    tok = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    tok.save_pretrained(runtime_dir)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    peft_model = PeftModel.from_pretrained(base, str(lora_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(runtime_dir, safe_serialization=True)

    del peft_model
    del merged
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(runtime_dir)


def normalize_row_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    name = row.get("name", "")
    parsed = parse_name(name) or {}
    algorithm = row.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        algorithm = parsed.get("algorithm")
    length = row.get("length")
    if not isinstance(length, int):
        length = parsed.get("length")
    idx = row.get("idx")
    if not isinstance(idx, int):
        idx = parsed.get("idx")
    return {
        **row,
        "name": name,
        "algorithm": algorithm,
        "length": length,
        "idx": idx,
    }


def row_matches_filters(
    row: Dict[str, Any],
    *,
    selected_algorithms: set[str],
    prompt_type: str,
    len_min: int,
    len_max: int,
    idx_min: int,
    idx_max: int,
) -> bool:
    algorithm = row.get("algorithm")
    length = row.get("length")
    idx = row.get("idx")

    if not isinstance(algorithm, str) or not algorithm:
        return False
    if selected_algorithms and algorithm not in selected_algorithms:
        return False
    if prompt_type not in row or not isinstance(row.get(prompt_type), str):
        return False
    if not isinstance(length, int):
        return False
    if len_min >= 0 and length < len_min:
        return False
    if len_max >= 0 and length > len_max:
        return False
    if idx_min >= 0:
        if not isinstance(idx, int) or idx < idx_min:
            return False
    if idx_max >= 0:
        if not isinstance(idx, int) or idx > idx_max:
            return False
    return True


def group_rows_by_algorithm(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)
    return grouped


def init_metric_row(algorithm: str, length: int | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "algorithm": algorithm,
        "correct": 0,
        "total": 0,
        "gen_errors": 0,
        "extract_errors": 0,
        "total_gen_tokens": 0,
        "total_gen_time_s": 0.0,
    }
    if length is not None:
        row["length"] = length
    return row


def finalize_metric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    total = row["total"]
    total_gen_time = row["total_gen_time_s"]
    row["accuracy"] = (row["correct"] / total) if total else 0.0
    row["avg_gen_tokens"] = (row["total_gen_tokens"] / total) if total else 0.0
    row["gen_tokens_per_s"] = (row["total_gen_tokens"] / total_gen_time) if total_gen_time > 1e-9 else 0.0
    return row


def update_metric_row(
    row: Dict[str, Any],
    *,
    ok: bool,
    gen_error: str | None,
    extract_error: str | None,
    gen_tokens: int,
    gen_time_s: float,
) -> None:
    row["total"] += 1
    row["total_gen_tokens"] += int(gen_tokens)
    row["total_gen_time_s"] += float(gen_time_s)
    if ok:
        row["correct"] += 1
    if gen_error is not None:
        row["gen_errors"] += 1
    if extract_error is not None:
        row["extract_errors"] += 1


def sort_metric_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["algorithm"], row.get("length", -1)))


def main() -> None:
    cfg = maybe_load_repo_config()
    testing_cfg = cfg.get("testing", {}) if isinstance(cfg.get("testing"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}

    repo_root = Path(__file__).resolve().parent
    default_dataset_path = Path(
        testing_cfg.get("flat_dataset_path", repo_root / "dataset" / "curriculum_test_set.flat.jsonl")
    )
    default_algorithms = testing_cfg.get("inference_algorithm_list") or []
    if not isinstance(default_algorithms, list):
        default_algorithms = []

    ap = argparse.ArgumentParser(description="Evaluate a model against the flattened curriculum_test_set.flat.jsonl dataset.")
    ap.add_argument("--dataset_path", type=str, default=str(default_dataset_path))
    ap.add_argument("--algorithms", type=str, default=",".join(default_algorithms))
    ap.add_argument(
        "--prompt_type",
        type=str,
        default=testing_cfg.get("prompt_type", "thinking_prompt"),
        choices=["thinking_prompt", "immediate_answer_prompt"],
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional cap per algorithm after filtering.")
    ap.add_argument("--batch_size", type=int, default=testing_cfg.get("batch_size", 32))
    ap.add_argument("--max_new_tokens", type=int, default=testing_cfg.get("max_new_tokens", 8192))
    ap.add_argument("--temperature", type=float, default=testing_cfg.get("temperature", 0.0))
    ap.add_argument("--top_p", type=float, default=testing_cfg.get("top_p", 1.0))
    ap.add_argument("--max_model_len", type=int, default=testing_cfg.get("max_model_len", 8192))
    ap.add_argument("--gpu_mem_util", type=float, default=testing_cfg.get("gpu_mem_util", 0.90))
    ap.add_argument(
        "--dtype",
        type=str,
        default=testing_cfg.get("dtype", "bfloat16"),
        choices=["bfloat16", "float16", "auto"],
    )
    ap.add_argument(
        "--gpu_id",
        type=str,
        default=testing_cfg.get("gpu_id"),
        help='Value to set for CUDA_VISIBLE_DEVICES before model initialization, e.g. "0" or "1".',
    )
    ap.add_argument("--len_min", type=int, default=testing_cfg.get("len_min", -1))
    ap.add_argument("--len_max", type=int, default=testing_cfg.get("len_max", -1))
    ap.add_argument("--idx_min", type=int, default=testing_cfg.get("idx_min", -1))
    ap.add_argument("--idx_max", type=int, default=testing_cfg.get("idx_max", -1))
    ap.add_argument("--model_name", type=str, default=model_cfg.get("model_name", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--lora_checkpoint", type=str, default=None)
    ap.add_argument("--merged_model_dir", type=str, default=None)
    ap.add_argument("--log_dir", type=str, default=testing_cfg.get("log_dir", str(repo_root / "logs")))
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument(
        "--system_prompt",
        type=str,
        default=testing_cfg.get(
            "system_prompt",
            "Please solve the following algorithmic problem without using programming languages.",
        ),
    )
    ap.add_argument(
        "--repetition_penalty",
        type=float,
        default=testing_cfg.get("repetition_penalty", 1.1),
    )
    ap.add_argument("--debug_ellipsis", action="store_true")
    args = ap.parse_args()

    selected_gpu_id = maybe_set_cuda_visible_devices(args.gpu_id)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"flattened dataset not found: {dataset_path}")

    torch.backends.cuda.matmul.allow_tf32 = True

    all_rows = [normalize_row_metadata(row) for row in load_flat_jsonl(dataset_path)]
    dataset_algorithms = sorted({row["algorithm"] for row in all_rows if isinstance(row.get("algorithm"), str)})

    requested_algorithms = parse_csv_list(args.algorithms)
    selected_algorithms = set(requested_algorithms) if requested_algorithms else set(dataset_algorithms)

    filtered_rows = [
        row
        for row in all_rows
        if row_matches_filters(
            row,
            selected_algorithms=selected_algorithms,
            prompt_type=args.prompt_type,
            len_min=args.len_min,
            len_max=args.len_max,
            idx_min=args.idx_min,
            idx_max=args.idx_max,
        )
    ]
    grouped_rows = group_rows_by_algorithm(filtered_rows)

    eval_algorithms = [algorithm for algorithm in (requested_algorithms or dataset_algorithms) if algorithm in grouped_rows]
    skipped_algorithms = [algorithm for algorithm in (requested_algorithms or dataset_algorithms) if algorithm not in grouped_rows]

    if not filtered_rows:
        raise RuntimeError(
            "No rows matched the flattened dataset filters. "
            f"dataset_path={dataset_path} prompt_type={args.prompt_type} "
            f"len=[{args.len_min},{args.len_max}] idx=[{args.idx_min},{args.idx_max}] "
            f"algorithms={sorted(selected_algorithms)}"
        )

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (args.run_name + "_") if args.run_name else ""
    txt_log_path = log_dir / f"{prefix}new_eval_{run_id}.txt"
    json_run_path = log_dir / f"{prefix}new_eval_{run_id}.json"
    merged_dir = Path(args.merged_model_dir).expanduser().resolve() if args.merged_model_dir else None

    model_path_for_vllm = maybe_make_merged_model_dir(
        base_model_name_or_path=args.model_name,
        lora_dir=args.lora_checkpoint,
        out_dir=merged_dir,
        dtype_str=args.dtype,
    )

    tok = AutoTokenizer.from_pretrained(model_path_for_vllm, trust_remote_code=True)
    llm = LLM(
        model=model_path_for_vllm,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
    )

    run_record: Dict[str, Any] = {
        "env": get_env_info(),
        "config": {
            "dataset_path": str(dataset_path),
            "model_name": args.model_name,
            "model_path_for_vllm": model_path_for_vllm,
            "lora_checkpoint": args.lora_checkpoint,
            "merged_model_dir": None,
            "merged_model_dir_arg": str(merged_dir) if merged_dir is not None else None,
            "runtime_model_dir": model_path_for_vllm if args.lora_checkpoint else None,
            "persist_merged_model": False,
            "prompt_type": args.prompt_type,
            "algorithms_requested": requested_algorithms or dataset_algorithms,
            "algorithms_evaluated": eval_algorithms,
            "algorithms_skipped_after_filtering": skipped_algorithms,
            "rows_loaded": len(all_rows),
            "rows_after_filtering": len(filtered_rows),
            "limit_per_algorithm": args.limit,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_mem_util,
            "dtype": args.dtype,
            "gpu_id": selected_gpu_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "len_min": args.len_min,
            "len_max": args.len_max,
            "idx_min": args.idx_min,
            "idx_max": args.idx_max,
            "repetition_penalty": args.repetition_penalty,
            "txt_log_path": str(txt_log_path),
            "json_run_path": str(json_run_path),
        },
        "results": [],
        "pair_results": [],
        "samples": [],
    }

    pair_metrics: Dict[Tuple[str, int], Dict[str, Any]] = {}
    algo_bar = tqdm(eval_algorithms, desc="Algorithms", dynamic_ncols=True)

    for algorithm in algo_bar:
        rows = grouped_rows[algorithm]
        if args.limit is not None:
            rows = rows[: max(0, int(args.limit))]
        if not rows:
            continue

        fn_name, extractor = get_extractor(algorithm)
        if extractor is None:
            print(f"[WARN] Missing extractor: utils.{fn_name} (skipping {algorithm})")
            continue

        metric_row = init_metric_row(algorithm)
        pbar = tqdm(range(0, len(rows), args.batch_size), desc=f"Eval ({algorithm})", dynamic_ncols=True, leave=False)

        for start in pbar:
            batch = rows[start : start + args.batch_size]

            prompts_raw = []
            params_list = []
            meta_list = []
            for ex in batch:
                prompt = ex.get(args.prompt_type)
                if not isinstance(prompt, str):
                    raise ValueError(f"Prompt type '{args.prompt_type}' missing in row: {ex}")
                prompts_raw.append(prompt)
                params_list.append(ex.get("params", {}))
                meta_list.append(
                    {
                        "name": ex.get("name"),
                        "algorithm": ex.get("algorithm"),
                        "length": ex.get("length"),
                        "idx": ex.get("idx"),
                        "source_path": ex.get("source_path"),
                    }
                )

            prompts = [build_model_input(tok, prompt, system_prompt=args.system_prompt) for prompt in prompts_raw]

            t0 = time.perf_counter()
            try:
                outs = llm.generate(prompts, sp)
            except Exception:
                batch_dt = time.perf_counter() - t0
                fail_rows = []
                for prompt, params, meta in zip(prompts_raw, params_list, meta_list):
                    sample = {
                        **meta,
                        "prompt_type": args.prompt_type,
                        "ok": False,
                        "prompt": prompt,
                        "params": params,
                        "text": "",
                        "cand": None,
                        "gen_tokens": 0,
                        "time_est_s": None,
                        "gen_error": "batch_generate_failed",
                        "extract_error": None,
                    }
                    fail_rows.append(sample)
                    run_record["samples"].append(sample)
                    update_metric_row(
                        metric_row,
                        ok=False,
                        gen_error="batch_generate_failed",
                        extract_error=None,
                        gen_tokens=0,
                        gen_time_s=batch_dt / max(len(batch), 1),
                    )
                    pair_key = (sample["algorithm"], sample["length"])
                    pair_row = pair_metrics.setdefault(pair_key, init_metric_row(sample["algorithm"], sample["length"]))
                    update_metric_row(
                        pair_row,
                        ok=False,
                        gen_error="batch_generate_failed",
                        extract_error=None,
                        gen_tokens=0,
                        gen_time_s=batch_dt / max(len(batch), 1),
                    )

                if args.debug_ellipsis:
                    for sample in fail_rows:
                        if contains_ellipsis(sample):
                            print("[DEBUG] Found Ellipsis in fail row:", sample.get("algorithm"), sample.get("name"))

                finalized_metric = finalize_metric_row(dict(metric_row))
                run_record["results"] = [
                    row for row in run_record["results"] if row["algorithm"] != algorithm
                ] + [finalized_metric]
                run_record["results"] = sort_metric_rows(run_record["results"])
                run_record["pair_results"] = sort_metric_rows(finalize_metric_row(dict(row)) for row in pair_metrics.values())
                safe_json_dump(json_run_path, run_record)
                write_human_log_txt(txt_log_path, run_record)
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_dt = time.perf_counter() - t0
            time_est = batch_dt / max(len(outs), 1)

            batch_rows = []
            for prompt, params, meta, out in zip(prompts_raw, params_list, meta_list, outs):
                text = out.outputs[0].text.strip() if out.outputs else ""
                gen_tokens = len(out.outputs[0].token_ids) if (out.outputs and out.outputs[0].token_ids is not None) else 0

                cand, ok = None, False
                extract_error = None
                try:
                    cand, ok = call_extractor(extractor, text, params)
                except Exception as exc:
                    extract_error = repr(exc)

                sample = {
                    **meta,
                    "prompt_type": args.prompt_type,
                    "ok": bool(ok),
                    "prompt": prompt,
                    "params": params,
                    "text": text,
                    "cand": cand,
                    "gen_tokens": int(gen_tokens),
                    "time_est_s": float(time_est),
                    "gen_error": None,
                    "extract_error": extract_error,
                }
                batch_rows.append(sample)
                run_record["samples"].append(sample)

                update_metric_row(
                    metric_row,
                    ok=bool(ok),
                    gen_error=None,
                    extract_error=extract_error,
                    gen_tokens=int(gen_tokens),
                    gen_time_s=float(time_est),
                )
                pair_key = (sample["algorithm"], sample["length"])
                pair_row = pair_metrics.setdefault(pair_key, init_metric_row(sample["algorithm"], sample["length"]))
                update_metric_row(
                    pair_row,
                    ok=bool(ok),
                    gen_error=None,
                    extract_error=extract_error,
                    gen_tokens=int(gen_tokens),
                    gen_time_s=float(time_est),
                )

            if args.debug_ellipsis:
                for sample in batch_rows:
                    if contains_ellipsis(sample) or contains_ellipsis(sample.get("params")):
                        print("[DEBUG] Found Ellipsis in sample:", sample.get("algorithm"), sample.get("name"))

            pbar.set_postfix(acc=f"{metric_row['correct']/max(metric_row['total'],1):.4f}", ok=f"{metric_row['correct']}/{metric_row['total']}")

            finalized_results = [
                finalize_metric_row(dict(result_row))
                for result_row in [metric_row]
            ]
            run_record["results"] = [
                row
                for row in run_record["results"]
                if row["algorithm"] != algorithm
            ] + finalized_results
            run_record["results"] = sort_metric_rows(run_record["results"])
            run_record["pair_results"] = sort_metric_rows(
                finalize_metric_row(dict(result_row)) for result_row in pair_metrics.values()
            )
            safe_json_dump(json_run_path, run_record)
            write_human_log_txt(txt_log_path, run_record)

        finalized_metric = finalize_metric_row(dict(metric_row))
        run_record["results"] = [row for row in run_record["results"] if row["algorithm"] != algorithm] + [finalized_metric]
        run_record["results"] = sort_metric_rows(run_record["results"])
        run_record["pair_results"] = sort_metric_rows(
            finalize_metric_row(dict(result_row)) for result_row in pair_metrics.values()
        )
        safe_json_dump(json_run_path, run_record)
        write_human_log_txt(txt_log_path, run_record)

        print(
            f"Final accuracy for {algorithm}: {finalized_metric['correct']}/{finalized_metric['total']} = "
            f"{finalized_metric['accuracy']:.4f} | tokens/s={finalized_metric['gen_tokens_per_s']:.2f}"
        )

    print(f"[LOG] Human-readable: {txt_log_path}")
    print_final_summary(run_record)
    print(f"[LOG] Full JSON:       {json_run_path}")


if __name__ == "__main__":
    main()
