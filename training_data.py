from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List


def parse_csv_list(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Unsupported list-like value: {type(value)!r}")


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


def build_chat_messages(prompt: str, completion: str, system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": completion})
    return messages


def format_training_example(example: Dict[str, Any], tok: Any, system_prompt: str) -> Dict[str, str]:
    messages = build_chat_messages(
        prompt=example["prompt_text"],
        completion=example["completion"],
        system_prompt=system_prompt,
    )
    return {"text": tok.apply_chat_template(messages, tokenize=False)}


def compute_dataset_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    algorithm_counts: DefaultDict[str, int] = defaultdict(int)
    pair_counts: DefaultDict[str, int] = defaultdict(int)
    model_family_counts: DefaultDict[str, int] = defaultdict(int)
    model_key_counts: DefaultDict[str, int] = defaultdict(int)
    model_size_counts: DefaultDict[str, int] = defaultdict(int)
    model_name_counts: DefaultDict[str, int] = defaultdict(int)
    model_dir_counts: DefaultDict[str, int] = defaultdict(int)
    lengths: List[int] = []
    idxs: List[int] = []

    for row in rows:
        algorithm = str(row["algorithm"])
        length = int(row["length"])
        idx = int(row["idx"])
        algorithm_counts[algorithm] += 1
        pair_counts[f"{algorithm}:{length}"] += 1
        if row.get("model_family"):
            model_family_counts[str(row["model_family"])] += 1
        if row.get("model_key"):
            model_key_counts[str(row["model_key"])] += 1
        if row.get("model_size"):
            model_size_counts[str(row["model_size"])] += 1
        if row.get("model_name"):
            model_name_counts[str(row["model_name"])] += 1
        if row.get("model_dir"):
            model_dir_counts[str(row["model_dir"])] += 1
        lengths.append(length)
        idxs.append(idx)

    stats: Dict[str, Any] = {
        "num_examples": len(rows),
        "num_algorithms": len(algorithm_counts),
        "algorithm_counts": dict(sorted(algorithm_counts.items())),
        "pair_counts": dict(sorted(pair_counts.items())),
        "model_family_counts": dict(sorted(model_family_counts.items())),
        "model_key_counts": dict(sorted(model_key_counts.items())),
        "model_size_counts": dict(sorted(model_size_counts.items())),
        "model_name_counts": dict(sorted(model_name_counts.items())),
        "model_dir_counts": dict(sorted(model_dir_counts.items())),
    }
    if lengths:
        stats["min_length"] = min(lengths)
        stats["max_length"] = max(lengths)
        stats["avg_length"] = sum(lengths) / len(lengths)
    if idxs:
        stats["min_idx"] = min(idxs)
        stats["max_idx"] = max(idxs)
    return stats


def _row_matches_filters(
    row: Dict[str, Any],
    *,
    include_algorithms: set[str],
    exclude_algorithms: set[str],
    include_model_families: set[str],
    exclude_model_families: set[str],
    include_model_keys: set[str],
    exclude_model_keys: set[str],
    include_model_sizes: set[str],
    exclude_model_sizes: set[str],
    include_model_names: set[str],
    exclude_model_names: set[str],
    include_model_dirs: set[str],
    exclude_model_dirs: set[str],
    prompt_field: str,
    len_min: int,
    len_max: int,
    idx_min: int,
    idx_max: int,
) -> bool:
    algorithm = row.get("algorithm")
    model_family = row.get("model_family")
    model_key = row.get("model_key")
    model_size = row.get("model_size")
    model_name = row.get("model_name")
    model_dir = row.get("model_dir")
    length = row.get("length")
    idx = row.get("idx")
    prompt = row.get(prompt_field)
    completion = row.get("completion")

    if not isinstance(algorithm, str) or not algorithm:
        return False
    if include_algorithms and algorithm not in include_algorithms:
        return False
    if algorithm in exclude_algorithms:
        return False
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
    if include_model_names and model_name not in include_model_names:
        return False
    if model_name in exclude_model_names:
        return False
    if include_model_dirs and model_dir not in include_model_dirs:
        return False
    if model_dir in exclude_model_dirs:
        return False
    if not isinstance(prompt, str) or not prompt:
        return False
    if not isinstance(completion, str) or not completion:
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


@dataclass
class TrainingCollectionResult:
    rows: List[Dict[str, Any]]
    stats: Dict[str, Any]


def load_training_rows(
    *,
    dataset_path: Path,
    prompt_field: str,
    include_algorithms: List[str] | None = None,
    exclude_algorithms: List[str] | None = None,
    include_model_families: List[str] | None = None,
    exclude_model_families: List[str] | None = None,
    include_model_keys: List[str] | None = None,
    exclude_model_keys: List[str] | None = None,
    include_model_sizes: List[str] | None = None,
    exclude_model_sizes: List[str] | None = None,
    include_model_names: List[str] | None = None,
    exclude_model_names: List[str] | None = None,
    include_model_dirs: List[str] | None = None,
    exclude_model_dirs: List[str] | None = None,
    len_min: int = -1,
    len_max: int = -1,
    idx_min: int = -1,
    idx_max: int = -1,
) -> TrainingCollectionResult:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"training dataset not found: {dataset_path}")

    include_set = set(parse_csv_list(include_algorithms))
    exclude_set = set(parse_csv_list(exclude_algorithms))
    include_model_family_set = set(parse_csv_list(include_model_families))
    exclude_model_family_set = set(parse_csv_list(exclude_model_families))
    include_model_key_set = set(parse_csv_list(include_model_keys))
    exclude_model_key_set = set(parse_csv_list(exclude_model_keys))
    include_model_size_set = set(parse_csv_list(include_model_sizes))
    exclude_model_size_set = set(parse_csv_list(exclude_model_sizes))
    include_model_name_set = set(parse_csv_list(include_model_names))
    exclude_model_name_set = set(parse_csv_list(exclude_model_names))
    include_model_dir_set = set(parse_csv_list(include_model_dirs))
    exclude_model_dir_set = set(parse_csv_list(exclude_model_dirs))

    raw_rows = load_flat_jsonl(dataset_path)
    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        if not _row_matches_filters(
            row,
            include_algorithms=include_set,
            exclude_algorithms=exclude_set,
            include_model_families=include_model_family_set,
            exclude_model_families=exclude_model_family_set,
            include_model_keys=include_model_key_set,
            exclude_model_keys=exclude_model_key_set,
            include_model_sizes=include_model_size_set,
            exclude_model_sizes=exclude_model_size_set,
            include_model_names=include_model_name_set,
            exclude_model_names=exclude_model_name_set,
            include_model_dirs=include_model_dir_set,
            exclude_model_dirs=exclude_model_dir_set,
            prompt_field=prompt_field,
            len_min=len_min,
            len_max=len_max,
            idx_min=idx_min,
            idx_max=idx_max,
        ):
            continue
        rows.append(
            {
                "name": row["name"],
                "algorithm": row["algorithm"],
                "length": int(row["length"]),
                "idx": int(row["idx"]),
                "prompt_text": row[prompt_field],
                "completion": row["completion"],
                "difficulty": float(row["length"]),
                "model_family": row.get("model_family"),
                "model_key": row.get("model_key"),
                "model_size": row.get("model_size"),
                "model_name": row.get("model_name"),
                "model_dir": row.get("model_dir"),
            }
        )

    rows.sort(
        key=lambda row: (
            str(row.get("model_family") or ""),
            str(row.get("model_key") or ""),
            row["length"],
            row["algorithm"],
            row["idx"],
            row["name"],
        )
    )

    stats = compute_dataset_stats(rows)
    stats.update(
        {
            "dataset_path": str(dataset_path),
            "prompt_field": prompt_field,
            "difficulty_source": "length",
            "include_algorithms": sorted(include_set),
            "exclude_algorithms": sorted(exclude_set),
            "include_model_families": sorted(include_model_family_set),
            "exclude_model_families": sorted(exclude_model_family_set),
            "include_model_keys": sorted(include_model_key_set),
            "exclude_model_keys": sorted(exclude_model_key_set),
            "include_model_sizes": sorted(include_model_size_set),
            "exclude_model_sizes": sorted(exclude_model_size_set),
            "include_model_names": sorted(include_model_name_set),
            "exclude_model_names": sorted(exclude_model_name_set),
            "include_model_dirs": sorted(include_model_dir_set),
            "exclude_model_dirs": sorted(exclude_model_dir_set),
            "len_min": len_min,
            "len_max": len_max,
            "idx_min": idx_min,
            "idx_max": idx_max,
        }
    )
    return TrainingCollectionResult(rows=rows, stats=stats)
