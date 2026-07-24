#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

import config
from training_data import build_chat_messages, load_training_rows, parse_csv_list

PairKey = Tuple[str, int]


def ensure_assistant_training_chat_template(tokenizer, *, assistant_only_loss: bool) -> None:
    """Ensure TRL can build assistant-token masks for chat SFT."""
    if not assistant_only_loss or "{% generation %}" in (tokenizer.chat_template or ""):
        return

    from trl.chat_template_utils import get_training_chat_template

    try:
        patched_template = get_training_chat_template(tokenizer)
    except ValueError as exc:
        # Nemotron Nano uses a customized Llama 3 template. TRL cannot match
        # that template by exact string, even though the tokenizer uses the
        # standard Llama 3 header/eot special tokens. Its supplied Llama 3
        # training template adds the generation markers required for masks.
        vocab = tokenizer.get_vocab()
        llama3_tokens = {"<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}
        if not llama3_tokens.issubset(vocab):
            raise ValueError(
                "assistant_only_loss requires a chat template with Jinja generation markers, "
                "and this tokenizer is not recognized as Llama 3 compatible."
            ) from exc
        from trl.chat_template_utils import llama3_training_chat_template

        patched_template = llama3_training_chat_template
        print("[TOKENIZER] Using TRL's Llama 3 training chat template for assistant-only loss.")

    if patched_template is not None:
        tokenizer.chat_template = patched_template


def maybe_set_cuda_visible_devices(gpu_id: str | None) -> str | None:
    if gpu_id is None:
        return None
    gpu_id = str(gpu_id).strip()
    if not gpu_id:
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return gpu_id


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _parse_pair_key(key: str) -> PairKey | None:
    if ":" in key:
        left, right = key.rsplit(":", 1)
        right_len = _parse_int(right)
        if right_len is not None:
            return left, right_len
        left_len = _parse_int(left)
        if left_len is not None:
            return right, left_len

    match = re.match(r"^(?P<algorithm>.+)_(?P<length>\d+)$", key)
    if match:
        return match.group("algorithm"), int(match.group("length"))
    return None


def _add_pair_difficulty(
    out: Dict[PairKey, float],
    algorithm: Any,
    length: Any,
    difficulty: Any,
    source: str,
) -> None:
    parsed_length = _parse_int(length)
    if parsed_length is None:
        raise ValueError(f"Invalid length in difficulty file {source}: {length!r}")
    try:
        parsed_difficulty = float(difficulty)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid difficulty for {algorithm!r}:{length!r} in {source}: {difficulty!r}"
        ) from exc
    out[(str(algorithm), parsed_length)] = parsed_difficulty


def load_pair_difficulty_map(path: Path) -> Dict[PairKey, float]:
    """
    Load difficulty values keyed by (algorithm, length).

    Supported schemas include:
    - {"10": {"sorting": 0.0, ...}, ...}
    - {"sorting": {"10": 0.0, ...}, ...}
    - {"sorting:10": 0.0, "sorting_20": 0.1, ...}
    - [{"algorithm": "sorting", "length": 10, "difficulty": 0.0}, ...]
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"difficulty file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[PairKey, float] = {}
    source = str(path)

    if isinstance(data, dict):
        for outer_key, outer_value in data.items():
            if isinstance(outer_value, dict):
                outer_length = _parse_int(outer_key)
                if outer_length is not None:
                    for algorithm, difficulty in outer_value.items():
                        _add_pair_difficulty(out, algorithm, outer_length, difficulty, source)
                else:
                    for length, difficulty in outer_value.items():
                        _add_pair_difficulty(out, outer_key, length, difficulty, source)
            else:
                pair = _parse_pair_key(str(outer_key))
                if pair is None:
                    raise ValueError(f"Cannot parse difficulty key {outer_key!r} in {source}")
                _add_pair_difficulty(out, pair[0], pair[1], outer_value, source)
    elif isinstance(data, list):
        for index, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in difficulty list at index {index} in {source}")
            algorithm = row.get("algorithm", row.get("task", row.get("name")))
            length = row.get("length", row.get("input_length", row.get("size")))
            difficulty = row.get("difficulty", row.get("score", row.get("value")))
            if algorithm is None or length is None or difficulty is None:
                raise ValueError(f"Missing algorithm/length/difficulty at index {index} in {source}")
            _add_pair_difficulty(out, algorithm, length, difficulty, source)
    else:
        raise ValueError(f"Unsupported difficulty JSON root type in {source}: {type(data).__name__}")

    if not out:
        raise ValueError(f"No difficulty entries loaded from {source}")
    return out


def _format_pairs(pairs: List[PairKey], limit: int = 8) -> str:
    rendered = [f"{algorithm}:{length}" for algorithm, length in pairs[:limit]]
    suffix = "" if len(pairs) <= limit else "..."
    return ", ".join(rendered) + suffix


class PairDifficultyCurriculumSampler(Sampler[int]):
    """
    Sample an (algorithm, length) pair by difficulty, then sample one instance
    uniformly from that nonempty pair bucket.
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        pair_difficulties: Dict[PairKey, float],
        num_samples: int,
        tau0: float,
        tau1: float,
        epsilon: float,
        difficulty_source: str,
        seed: int = 42,
    ):
        pair_to_indices: DefaultDict[PairKey, List[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            pair_to_indices[(str(row["algorithm"]), int(row["length"]))].append(index)

        data_pairs = set(pair_to_indices)
        if pair_difficulties:
            difficulty_pairs = set(pair_difficulties)
            pairs = sorted(data_pairs & difficulty_pairs, key=lambda item: (item[0], item[1]))
            missing_difficulty_pairs = sorted(data_pairs - difficulty_pairs, key=lambda item: (item[0], item[1]))
            empty_difficulty_pairs = sorted(difficulty_pairs - data_pairs, key=lambda item: (item[0], item[1]))
        else:
            pairs = sorted(data_pairs, key=lambda item: (item[0], item[1]))
            pair_difficulties = {pair: float(pair[1]) for pair in pairs}
            missing_difficulty_pairs = []
            empty_difficulty_pairs = []

        if not pairs:
            raise ValueError(
                "No nonempty (algorithm, length) buckets matched the difficulty map. "
                f"dataset_pairs={len(data_pairs)} difficulty_pairs={len(pair_difficulties)}"
            )

        self.pairs = pairs
        self.buckets = [pair_to_indices[pair] for pair in self.pairs]
        self.difficulties = torch.tensor([pair_difficulties[pair] for pair in self.pairs], dtype=torch.float32)
        self.num_pairs = len(self.pairs)
        self.num_samples = int(num_samples)
        self.tau0 = float(tau0)
        self.tau1 = float(tau1)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.global_step = 0
        self.max_steps = 1
        self.stats: Dict[str, Any] = {
            "curriculum_sampling": "pair_difficulty_then_uniform_instance",
            "difficulty_source": difficulty_source,
            "dataset_pairs": len(data_pairs),
            "difficulty_pairs": len(pair_difficulties),
            "used_pairs": len(self.pairs),
            "missing_difficulty_pairs": len(missing_difficulty_pairs),
            "empty_difficulty_pairs_ignored": len(empty_difficulty_pairs),
            "missing_difficulty_pair_examples": _format_pairs(missing_difficulty_pairs),
            "empty_difficulty_pair_examples": _format_pairs(empty_difficulty_pairs),
        }

    def set_step(self, step: int, max_steps: int) -> None:
        self.global_step = int(step)
        self.max_steps = max(int(max_steps), 1)

    def current_tau(self) -> float:
        progress = min(max(self.global_step / self.max_steps, 0.0), 1.0)
        tau = self.tau0 + (self.tau1 - self.tau0) * progress
        return max(float(tau), 1e-6)

    def _probs(self) -> torch.Tensor:
        tau = self.current_tau()
        scaled = -self.difficulties / tau
        scaled = scaled - torch.max(scaled)
        weights = torch.exp(scaled)
        probs = weights / weights.sum()
        if self.epsilon > 0.0:
            uniform = torch.ones_like(probs) / float(self.num_pairs)
            probs = (1.0 - self.epsilon) * probs + self.epsilon * uniform
        return probs / probs.sum()

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.global_step)

        remaining = self.num_samples
        while remaining > 0:
            batch_size = min(remaining, 4096)
            probs = self._probs()
            pair_indices = torch.multinomial(probs, num_samples=batch_size, replacement=True, generator=generator)
            for pair_index in pair_indices.tolist():
                bucket = self.buckets[int(pair_index)]
                offset = torch.randint(len(bucket), (1,), generator=generator).item()
                yield int(bucket[int(offset)])
            remaining -= batch_size

    def __len__(self) -> int:
        return self.num_samples


class CurriculumStepCallback(TrainerCallback):
    def __init__(self, sampler: PairDifficultyCurriculumSampler):
        self.sampler = sampler

    def on_step_begin(self, args, state, control, **kwargs):
        self.sampler.set_step(state.global_step, state.max_steps)
        return control


class FlexibleSFTTrainer(SFTTrainer):
    def __init__(self, *args, train_mode: str = "ordered", curriculum_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_mode = train_mode
        self._curriculum_sampler = curriculum_sampler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        batch_size = getattr(self, "_train_batch_size", None) or self.args.per_device_train_batch_size

        if self._train_mode == "curriculum":
            if self._curriculum_sampler is None:
                raise ValueError("curriculum mode requires curriculum_sampler")
            sampler = self._curriculum_sampler
        elif self._train_mode == "random":
            sampler = RandomSampler(train_dataset)
        else:
            sampler = SequentialSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=getattr(self.args, "dataloader_drop_last", False),
            num_workers=getattr(self.args, "dataloader_num_workers", 0),
            pin_memory=getattr(self.args, "dataloader_pin_memory", True),
        )


def maybe_init_wandb(args: argparse.Namespace, pre_config: Dict[str, Any]) -> Optional[Any]:
    if args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed but --wandb_mode is enabled. Install with: pip install wandb"
        ) from exc

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    return wandb.init(
        project=args.wandb_project or None,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        mode=args.wandb_mode,
        config=pre_config,
    )


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def resolve_optional_repo_path(value: str | Path | None, repo_root: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve_repo_path(text, repo_root)


def csv_default(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, str):
        return value
    return ""


def build_pair_curriculum_sampler(
    *,
    rows: List[Dict[str, Any]],
    difficulty_path: str | Path | None,
    repo_root: Path,
    num_samples: int,
    tau0: float,
    tau1: float,
    epsilon: float,
    seed: int,
) -> PairDifficultyCurriculumSampler:
    resolved_difficulty_path = resolve_optional_repo_path(difficulty_path, repo_root)
    if resolved_difficulty_path is None:
        pair_difficulties: Dict[PairKey, float] = {}
        difficulty_source = "dataset_length_fallback"
    else:
        pair_difficulties = load_pair_difficulty_map(resolved_difficulty_path)
        difficulty_source = str(resolved_difficulty_path)

    return PairDifficultyCurriculumSampler(
        rows=rows,
        pair_difficulties=pair_difficulties,
        num_samples=num_samples,
        tau0=tau0,
        tau1=tau1,
        epsilon=epsilon,
        difficulty_source=difficulty_source,
        seed=seed,
    )


def print_curriculum_sampler_summary(sampler: PairDifficultyCurriculumSampler) -> None:
    stats = sampler.stats
    print(
        "[CURRICULUM] Sampling (algorithm, length) by difficulty, "
        "then uniformly sampling one instance from that pair."
    )
    print(
        "[CURRICULUM] "
        f"difficulty_source={stats['difficulty_source']} "
        f"dataset_pairs={stats['dataset_pairs']} "
        f"difficulty_pairs={stats['difficulty_pairs']} "
        f"used_pairs={stats['used_pairs']}"
    )
    if stats["empty_difficulty_pairs_ignored"]:
        print(
            "[CURRICULUM] Ignored "
            f"{stats['empty_difficulty_pairs_ignored']} difficulty pair(s) with 0 matching training rows: "
            f"{stats['empty_difficulty_pair_examples']}"
        )
    if stats["missing_difficulty_pairs"]:
        print(
            "[CURRICULUM] WARNING: ignored "
            f"{stats['missing_difficulty_pairs']} nonempty dataset pair(s) missing from difficulty file: "
            f"{stats['missing_difficulty_pair_examples']}"
        )


def build_arg_parser(cfg_file: Dict[str, Any], repo_root: Path) -> argparse.ArgumentParser:
    training_cfg = cfg_file.get("training", {}) if isinstance(cfg_file.get("training"), dict) else {}
    model_cfg = cfg_file.get("model", {}) if isinstance(cfg_file.get("model"), dict) else {}
    wandb_cfg = cfg_file.get("wandb", {}) if isinstance(cfg_file.get("wandb"), dict) else {}

    default_dataset_path = resolve_repo_path(
        training_cfg.get("dataset_path", repo_root / "dataset" / "training_rows.flat.jsonl"),
        repo_root,
    )
    default_output_dir = resolve_repo_path(training_cfg.get("output_dir", "./random_order_lora/"), repo_root)
    default_difficulty_path = resolve_optional_repo_path(
        training_cfg.get("difficulty_path", "/home/brianyuan/p-llm-complexity/difficulty.json"),
        repo_root,
    )

    ap = argparse.ArgumentParser(description="Train a LoRA adapter on the flattened training_rows dataset.")
    ap.add_argument("--dataset_path", type=str, default=str(default_dataset_path))
    ap.add_argument("--model_name", type=str, default=model_cfg.get("model_name", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--output_dir", type=str, default=str(default_output_dir))
    ap.add_argument(
        "--gpu_id",
        type=str,
        default=training_cfg.get("gpu_id", 1),
        help='Value to set for CUDA_VISIBLE_DEVICES before model initialization, e.g. "0" or "1".',
    )
    ap.add_argument("--seed", type=int, default=training_cfg.get("seed", 42))
    ap.add_argument("--epochs", type=float, default=training_cfg.get("epochs", 2))
    ap.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=training_cfg.get("learning_rate", 2e-4))
    ap.add_argument("--batch_size", type=int, default=training_cfg.get("batch_size", 1))
    ap.add_argument("--grad_accum", type=int, default=training_cfg.get("grad_accum", 16))
    ap.add_argument("--max_seq_len", type=int, default=training_cfg.get("max_seq_len", 16384))
    ap.add_argument(
        "--assistant_only_loss",
        action=argparse.BooleanOptionalAction,
        default=training_cfg.get("assistant_only_loss", True),
    )
    ap.add_argument(
        "--prompt_field",
        type=str,
        default=training_cfg.get("prompt_field", "prompt"),
        choices=["prompt", "immediate_answer_prompt"],
    )
    ap.add_argument(
        "--include_algorithms",
        "--train_algorithms",
        dest="include_algorithms",
        type=str,
        default=csv_default(training_cfg.get("include_algorithms", [])),
        help="Comma-separated algorithms to train on. Empty means all algorithms.",
    )
    ap.add_argument(
        "--exclude_algorithms",
        type=str,
        default=csv_default(training_cfg.get("exclude_algorithms", [])),
        help="Comma-separated algorithms to exclude after applying include_algorithms.",
    )
    ap.add_argument(
        "--include_model_families",
        type=str,
        default=csv_default(training_cfg.get("include_model_families", [])),
        help="Comma-separated model families to train on, e.g. qwen,nemotron253b. Empty means all.",
    )
    ap.add_argument(
        "--exclude_model_families",
        type=str,
        default=csv_default(training_cfg.get("exclude_model_families", [])),
        help="Comma-separated model families to exclude.",
    )
    ap.add_argument(
        "--include_model_keys",
        type=str,
        default=csv_default(training_cfg.get("include_model_keys", [])),
        help="Comma-separated normalized model keys to train on. Empty means all.",
    )
    ap.add_argument(
        "--exclude_model_keys",
        type=str,
        default=csv_default(training_cfg.get("exclude_model_keys", [])),
        help="Comma-separated normalized model keys to exclude.",
    )
    ap.add_argument(
        "--include_model_sizes",
        type=str,
        default=csv_default(training_cfg.get("include_model_sizes", [])),
        help='Comma-separated model sizes to train on, e.g. "235b,253b". Empty means all.',
    )
    ap.add_argument(
        "--exclude_model_sizes",
        type=str,
        default=csv_default(training_cfg.get("exclude_model_sizes", [])),
        help="Comma-separated model sizes to exclude.",
    )
    ap.add_argument(
        "--include_model_names",
        type=str,
        default=csv_default(training_cfg.get("include_model_names", [])),
        help="Comma-separated raw model_name values to train on. Empty means all.",
    )
    ap.add_argument(
        "--exclude_model_names",
        type=str,
        default=csv_default(training_cfg.get("exclude_model_names", [])),
        help="Comma-separated raw model_name values to exclude.",
    )
    ap.add_argument(
        "--include_model_dirs",
        type=str,
        default=csv_default(training_cfg.get("include_model_dirs", [])),
        help="Comma-separated model_dir values to train on. Empty means all.",
    )
    ap.add_argument(
        "--exclude_model_dirs",
        type=str,
        default=csv_default(training_cfg.get("exclude_model_dirs", [])),
        help="Comma-separated model_dir values to exclude.",
    )
    ap.add_argument("--len_min", type=int, default=training_cfg.get("len_min", -1))
    ap.add_argument("--len_max", type=int, default=training_cfg.get("len_max", -1))
    ap.add_argument("--idx_min", type=int, default=training_cfg.get("idx_min", -1))
    ap.add_argument("--idx_max", type=int, default=training_cfg.get("idx_max", -1))
    ap.add_argument(
        "--train_mode",
        type=str,
        default=training_cfg.get("train_mode", "curriculum"),
        choices=["ordered", "random", "curriculum"],
    )
    ap.add_argument(
        "--difficulty_path",
        type=str,
        default="" if default_difficulty_path is None else str(default_difficulty_path),
        help=(
            "JSON mapping difficulty by (algorithm, length) for curriculum mode. "
            "Empty string falls back to length-based pair difficulty."
        ),
    )
    ap.add_argument(
        "--lr_scheduler",
        type=str,
        default=training_cfg.get("lr_scheduler", "cosine"),
        choices=["linear", "cosine", "cosine_with_restarts", "constant", "constant_with_warmup"],
    )
    ap.add_argument("--warmup_ratio", type=float, default=training_cfg.get("warmup_ratio", 0.03))
    ap.add_argument("--warmup_steps", type=int, default=training_cfg.get("warmup_steps", 0))
    ap.add_argument("--tau0", type=float, default=training_cfg.get("tau0", 0.15))
    ap.add_argument("--tau1", type=float, default=training_cfg.get("tau1", 0.60))
    ap.add_argument("--epsilon", type=float, default=training_cfg.get("epsilon", 0.05))
    ap.add_argument(
        "--system_prompt",
        type=str,
        default=training_cfg.get(
            "system_prompt",
            "Please solve the following algorithmic problem without using programming languages.",
        ),
    )
    ap.add_argument(
        "--wandb_mode",
        choices=["disabled", "offline", "online"],
        default=training_cfg.get("wandb_mode", "online"),
    )
    ap.add_argument("--wandb_project", default=wandb_cfg.get("project"))
    ap.add_argument("--wandb_entity", default=wandb_cfg.get("entity"))
    ap.add_argument("--wandb_name", default=wandb_cfg.get("name"))
    return ap


def build_default_run_name(args: argparse.Namespace) -> str:
    include_algorithms = parse_csv_list(args.include_algorithms)
    exclude_algorithms = parse_csv_list(args.exclude_algorithms)
    include_tag = "all" if not include_algorithms else f"incl{len(include_algorithms)}"
    exclude_tag = f"excl{len(exclude_algorithms)}"
    return (
        f"sft-{Path(args.model_name).name}-{args.train_mode}"
        f"-sched{args.lr_scheduler}-warm{args.warmup_steps or args.warmup_ratio}"
        f"-{include_tag}-{exclude_tag}"
    )


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
        "bf16": True,
        "report_to": ("wandb" if args.wandb_mode != "disabled" else "none"),
        "run_name": args.wandb_name or build_default_run_name(args),
        "dataset_text_field": "text",
        "max_length": args.max_seq_len,
        "eval_strategy": "no",
        "lr_scheduler_type": args.lr_scheduler,
        "assistant_only_loss": args.assistant_only_loss,
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
            f"len=[{args.len_min},{args.len_max}] idx=[{args.idx_min},{args.idx_max}] "
            f"include={parse_csv_list(args.include_algorithms)} exclude={parse_csv_list(args.exclude_algorithms)} "
            f"model_families={parse_csv_list(args.include_model_families)} "
            f"model_keys={parse_csv_list(args.include_model_keys)} model_sizes={parse_csv_list(args.include_model_sizes)}"
        )

    print(f"[DATA] Loaded {len(rows)} training rows from {stats['dataset_path']}")
    print(f"[DATA] Algorithms: {stats['num_algorithms']} | Lengths: {stats.get('min_length')}..{stats.get('max_length')}")
    if stats.get("model_family_counts"):
        print(f"[DATA] Model families: {stats['model_family_counts']}")
    if stats.get("model_size_counts"):
        print(f"[DATA] Model sizes: {stats['model_size_counts']}")
    if selected_gpu_id is not None:
        print(f"[ENV] CUDA_VISIBLE_DEVICES={selected_gpu_id}")

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
    raw_ds = Dataset.from_list(train_examples)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    ensure_assistant_training_chat_template(tok, assistant_only_loss=args.assistant_only_loss)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    train_ds = raw_ds

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    callbacks = []
    if curriculum_sampler is not None:
        callbacks.append(CurriculumStepCallback(curriculum_sampler))

    trainer = FlexibleSFTTrainer(
        model=model,
        args=build_sft_config(args),
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=peft_config,
        train_mode=args.train_mode,
        curriculum_sampler=curriculum_sampler,
        callbacks=callbacks,
    )

    trainer.train()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    print(f"\nAdapter saved to: {output_dir}")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
