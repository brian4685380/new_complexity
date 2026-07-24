#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig

import config
from train import (
    CurriculumStepCallback,
    FlexibleSFTTrainer,
    build_pair_curriculum_sampler,
    csv_default,
    maybe_init_wandb,
    maybe_set_cuda_visible_devices,
    print_curriculum_sampler_summary,
    resolve_repo_path,
    resolve_optional_repo_path,
)
from training_data import build_chat_messages, load_training_rows, parse_csv_list


def bool_default(primary: Dict[str, Any], fallback: Dict[str, Any], key: str, default: bool) -> bool:
    value = primary.get(key, fallback.get(key, default))
    return bool(value)


def dtype_from_arg(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def build_arg_parser(cfg_file: Dict[str, Any], repo_root: Path) -> argparse.ArgumentParser:
    training_cfg = cfg_file.get("training", {}) if isinstance(cfg_file.get("training"), dict) else {}
    full_cfg = cfg_file.get("full_training", {}) if isinstance(cfg_file.get("full_training"), dict) else {}
    model_cfg = cfg_file.get("model", {}) if isinstance(cfg_file.get("model"), dict) else {}
    wandb_cfg = cfg_file.get("wandb", {}) if isinstance(cfg_file.get("wandb"), dict) else {}

    default_dataset_path = resolve_repo_path(
        full_cfg.get("dataset_path", training_cfg.get("dataset_path", repo_root / "dataset" / "training_rows.flat.jsonl")),
        repo_root,
    )
    default_output_dir = resolve_repo_path(
        full_cfg.get("output_dir", repo_root / "runs" / "full_finetune" / "qwen1_7b"),
        repo_root,
    )
    default_difficulty_path = resolve_optional_repo_path(
        full_cfg.get(
            "difficulty_path",
            training_cfg.get("difficulty_path", "/home/brianyuan/p-llm-complexity/difficulty.json"),
        ),
        repo_root,
    )

    ap = argparse.ArgumentParser(
        description="Full fine-tune Qwen/Qwen3-1.7B on the flattened training_rows dataset."
    )
    ap.add_argument("--dataset_path", type=str, default=str(default_dataset_path))
    ap.add_argument("--model_name", type=str, default=full_cfg.get("model_name", model_cfg.get("model_name", "Qwen/Qwen3-1.7B")))
    ap.add_argument("--output_dir", type=str, default=str(default_output_dir))
    ap.add_argument(
        "--gpu_id",
        type=str,
        default=full_cfg.get("gpu_id", training_cfg.get("gpu_id", 1)),
        help='Value to set for CUDA_VISIBLE_DEVICES before model initialization, e.g. "0" or "1".',
    )
    ap.add_argument("--seed", type=int, default=full_cfg.get("seed", training_cfg.get("seed", 42)))
    ap.add_argument("--epochs", type=float, default=full_cfg.get("epochs", training_cfg.get("epochs", 1)))
    ap.add_argument(
        "--learning_rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=full_cfg.get("learning_rate", 2e-5),
        help="Full fine-tuning should usually use a lower LR than LoRA.",
    )
    ap.add_argument("--batch_size", type=int, default=full_cfg.get("batch_size", training_cfg.get("batch_size", 1)))
    ap.add_argument("--grad_accum", type=int, default=full_cfg.get("grad_accum", training_cfg.get("grad_accum", 16)))
    ap.add_argument("--max_seq_len", type=int, default=full_cfg.get("max_seq_len", training_cfg.get("max_seq_len", 16384)))
    ap.add_argument(
        "--assistant_only_loss",
        action=argparse.BooleanOptionalAction,
        default=bool_default(full_cfg, training_cfg, "assistant_only_loss", True),
    )
    ap.add_argument("--prompt_field", type=str, default=full_cfg.get("prompt_field", training_cfg.get("prompt_field", "prompt")), choices=["prompt", "immediate_answer_prompt"])
    ap.add_argument("--include_algorithms", "--train_algorithms", dest="include_algorithms", type=str, default=csv_default(full_cfg.get("include_algorithms", training_cfg.get("include_algorithms", []))))
    ap.add_argument("--exclude_algorithms", type=str, default=csv_default(full_cfg.get("exclude_algorithms", training_cfg.get("exclude_algorithms", []))))
    ap.add_argument("--include_model_families", type=str, default=csv_default(full_cfg.get("include_model_families", training_cfg.get("include_model_families", []))))
    ap.add_argument("--exclude_model_families", type=str, default=csv_default(full_cfg.get("exclude_model_families", training_cfg.get("exclude_model_families", []))))
    ap.add_argument("--include_model_keys", type=str, default=csv_default(full_cfg.get("include_model_keys", training_cfg.get("include_model_keys", []))))
    ap.add_argument("--exclude_model_keys", type=str, default=csv_default(full_cfg.get("exclude_model_keys", training_cfg.get("exclude_model_keys", []))))
    ap.add_argument("--include_model_sizes", type=str, default=csv_default(full_cfg.get("include_model_sizes", training_cfg.get("include_model_sizes", []))))
    ap.add_argument("--exclude_model_sizes", type=str, default=csv_default(full_cfg.get("exclude_model_sizes", training_cfg.get("exclude_model_sizes", []))))
    ap.add_argument("--include_model_names", type=str, default=csv_default(full_cfg.get("include_model_names", training_cfg.get("include_model_names", []))))
    ap.add_argument("--exclude_model_names", type=str, default=csv_default(full_cfg.get("exclude_model_names", training_cfg.get("exclude_model_names", []))))
    ap.add_argument("--include_model_dirs", type=str, default=csv_default(full_cfg.get("include_model_dirs", training_cfg.get("include_model_dirs", []))))
    ap.add_argument("--exclude_model_dirs", type=str, default=csv_default(full_cfg.get("exclude_model_dirs", training_cfg.get("exclude_model_dirs", []))))
    ap.add_argument("--len_min", type=int, default=full_cfg.get("len_min", training_cfg.get("len_min", -1)))
    ap.add_argument("--len_max", type=int, default=full_cfg.get("len_max", training_cfg.get("len_max", -1)))
    ap.add_argument("--idx_min", type=int, default=full_cfg.get("idx_min", training_cfg.get("idx_min", -1)))
    ap.add_argument("--idx_max", type=int, default=full_cfg.get("idx_max", training_cfg.get("idx_max", -1)))
    ap.add_argument("--train_mode", type=str, default=full_cfg.get("train_mode", training_cfg.get("train_mode", "random")), choices=["ordered", "random", "curriculum"])
    ap.add_argument(
        "--difficulty_path",
        type=str,
        default="" if default_difficulty_path is None else str(default_difficulty_path),
        help=(
            "JSON mapping difficulty by (algorithm, length) for curriculum mode. "
            "Empty string falls back to length-based pair difficulty."
        ),
    )
    ap.add_argument("--lr_scheduler", type=str, default=full_cfg.get("lr_scheduler", training_cfg.get("lr_scheduler", "cosine")), choices=["linear", "cosine", "cosine_with_restarts", "constant", "constant_with_warmup"])
    ap.add_argument("--warmup_ratio", type=float, default=full_cfg.get("warmup_ratio", training_cfg.get("warmup_ratio", 0.03)))
    ap.add_argument("--warmup_steps", type=int, default=full_cfg.get("warmup_steps", training_cfg.get("warmup_steps", 0)))
    ap.add_argument("--weight_decay", type=float, default=full_cfg.get("weight_decay", 0.0))
    ap.add_argument("--max_grad_norm", type=float, default=full_cfg.get("max_grad_norm", 1.0))
    ap.add_argument("--optim", type=str, default=full_cfg.get("optim", "adamw_torch_fused"))
    ap.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default=full_cfg.get("torch_dtype", "bfloat16"))
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=bool_default(full_cfg, {}, "bf16", True))
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=bool_default(full_cfg, {}, "fp16", False))
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=bool_default(full_cfg, {}, "gradient_checkpointing", True))
    ap.add_argument("--gradient_checkpointing_use_reentrant", action=argparse.BooleanOptionalAction, default=bool_default(full_cfg, {}, "gradient_checkpointing_use_reentrant", False))
    ap.add_argument("--attn_implementation", choices=["auto", "flash_attention_2", "sdpa", "eager"], default=full_cfg.get("attn_implementation", "auto"))
    ap.add_argument("--dataloader_num_workers", type=int, default=full_cfg.get("dataloader_num_workers", 0))
    ap.add_argument("--tau0", type=float, default=full_cfg.get("tau0", training_cfg.get("tau0", 0.15)))
    ap.add_argument("--tau1", type=float, default=full_cfg.get("tau1", training_cfg.get("tau1", 0.60)))
    ap.add_argument("--epsilon", type=float, default=full_cfg.get("epsilon", training_cfg.get("epsilon", 0.05)))
    ap.add_argument(
        "--system_prompt",
        type=str,
        default=full_cfg.get(
            "system_prompt",
            training_cfg.get("system_prompt", "Please solve the following algorithmic problem without using programming languages."),
        ),
    )
    ap.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default=full_cfg.get("wandb_mode", training_cfg.get("wandb_mode", "disabled")))
    ap.add_argument("--wandb_project", default=wandb_cfg.get("project"))
    ap.add_argument("--wandb_entity", default=wandb_cfg.get("entity"))
    ap.add_argument("--wandb_name", default=full_cfg.get("wandb_name", wandb_cfg.get("name")))
    ap.add_argument("--dry_run", action="store_true", help="Load/filter the dataset and print configuration, but do not load or train the model.")
    return ap


def build_default_run_name(args: argparse.Namespace) -> str:
    include_algorithms = parse_csv_list(args.include_algorithms)
    include_model_families = parse_csv_list(args.include_model_families)
    algo_tag = "all" if not include_algorithms else f"algos{len(include_algorithms)}"
    model_tag = "allmodels" if not include_model_families else "_".join(include_model_families)
    return f"full-{Path(args.model_name).name}-{model_tag}-{algo_tag}-{args.train_mode}"


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    sig = inspect.signature(SFTConfig.__init__)
    allowed_keys = set(sig.parameters.keys()) - {"self"}

    sft_kwargs: Dict[str, Any] = {
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": 1,
        "save_strategy": "no",
        "save_only_model": True,
        "save_safetensors": True,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": ("wandb" if args.wandb_mode != "disabled" else "none"),
        "run_name": args.wandb_name or build_default_run_name(args),
        "dataset_text_field": "text",
        "max_length": args.max_seq_len,
        "eval_strategy": "no",
        "lr_scheduler_type": args.lr_scheduler,
        "assistant_only_loss": args.assistant_only_loss,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": args.gradient_checkpointing_use_reentrant},
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
    }

    if "warmup_steps" in allowed_keys and args.warmup_steps > 0:
        sft_kwargs["warmup_steps"] = args.warmup_steps
    elif "warmup_ratio" in allowed_keys:
        sft_kwargs["warmup_ratio"] = args.warmup_ratio

    return SFTConfig(**{key: value for key, value in sft_kwargs.items() if key in allowed_keys})


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg_file = config.load_config()
    args = build_arg_parser(cfg_file, repo_root).parse_args()

    selected_gpu_id = maybe_set_cuda_visible_devices(args.gpu_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 cannot both be enabled.")

    collection = load_training_rows(
        dataset_path=Path(args.dataset_path),
        prompt_field=args.prompt_field,
        include_algorithms=parse_csv_list(args.include_algorithms),
        exclude_algorithms=parse_csv_list(args.exclude_algorithms),
        include_model_families=parse_csv_list(args.include_model_families),
        exclude_model_families=parse_csv_list(args.exclude_model_families),
        include_model_keys=parse_csv_list(args.include_model_keys),
        exclude_model_keys=parse_csv_list(args.exclude_model_keys),
        include_model_sizes=parse_csv_list(args.include_model_sizes),
        exclude_model_sizes=parse_csv_list(args.exclude_model_sizes),
        include_model_names=parse_csv_list(args.include_model_names),
        exclude_model_names=parse_csv_list(args.exclude_model_names),
        include_model_dirs=parse_csv_list(args.include_model_dirs),
        exclude_model_dirs=parse_csv_list(args.exclude_model_dirs),
        len_min=args.len_min,
        len_max=args.len_max,
        idx_min=args.idx_min,
        idx_max=args.idx_max,
    )
    rows = collection.rows
    stats = collection.stats

    if not rows:
        raise RuntimeError(
            "No training rows matched the flattened dataset filters. "
            f"dataset_path={args.dataset_path} prompt_field={args.prompt_field} "
            f"include_algorithms={parse_csv_list(args.include_algorithms)} "
            f"include_model_families={parse_csv_list(args.include_model_families)} "
            f"include_model_keys={parse_csv_list(args.include_model_keys)} "
            f"include_model_sizes={parse_csv_list(args.include_model_sizes)}"
        )

    print(f"[DATA] Loaded {len(rows)} training rows from {stats['dataset_path']}")
    print(f"[DATA] Algorithms: {stats['num_algorithms']} | Lengths: {stats.get('min_length')}..{stats.get('max_length')}")
    if stats.get("model_family_counts"):
        print(f"[DATA] Model families: {stats['model_family_counts']}")
    if stats.get("model_key_counts"):
        print(f"[DATA] Model keys: {stats['model_key_counts']}")
    if stats.get("model_size_counts"):
        print(f"[DATA] Model sizes: {stats['model_size_counts']}")
    if selected_gpu_id is not None:
        print(f"[ENV] CUDA_VISIBLE_DEVICES={selected_gpu_id}")

    print(
        "[TRAIN] full fine-tuning "
        f"model={args.model_name} epochs={args.epochs} lr={args.learning_rate} "
        f"batch={args.batch_size} grad_accum={args.grad_accum} max_seq_len={args.max_seq_len}"
    )
    print(
        "[TRAIN] memory settings "
        f"dtype={args.torch_dtype} bf16={args.bf16} fp16={args.fp16} "
        f"gradient_checkpointing={args.gradient_checkpointing} attn={args.attn_implementation}"
    )

    curriculum_sampler = None
    if args.train_mode == "curriculum":
        curriculum_sampler = build_pair_curriculum_sampler(
            rows=rows,
            difficulty_path=args.difficulty_path,
            repo_root=repo_root,
            num_samples=len(rows),
            tau0=args.tau0,
            tau1=args.tau1,
            epsilon=args.epsilon,
            seed=args.seed,
        )
        print_curriculum_sampler_summary(curriculum_sampler)
        stats.update(curriculum_sampler.stats)

    if args.dry_run:
        print("[DRY-RUN] Skipping tokenizer/model load and training.")
        return

    pre_wandb_config = vars(args).copy()
    pre_wandb_config.update(stats)
    pre_wandb_config["gpu_id"] = selected_gpu_id
    pre_wandb_config["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    wb_run = maybe_init_wandb(args, pre_wandb_config)
    if wb_run is not None:
        wb_run.config.update(stats, allow_val_change=True)

    train_examples = [
        {
            "messages": build_chat_messages(
                prompt=row["prompt_text"],
                completion=row["completion"],
                system_prompt=args.system_prompt,
            )
        }
        for row in rows
    ]
    train_ds = Dataset.from_list(train_examples)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype_from_arg(args.torch_dtype),
        "trust_remote_code": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    callbacks = []
    if curriculum_sampler is not None:
        callbacks.append(CurriculumStepCallback(curriculum_sampler))

    trainer = FlexibleSFTTrainer(
        model=model,
        args=build_sft_config(args),
        train_dataset=train_ds,
        processing_class=tok,
        train_mode=args.train_mode,
        curriculum_sampler=curriculum_sampler,
        callbacks=callbacks,
    )

    trainer.train()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"\nFull fine-tuned model saved to: {output_dir}")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
