#!/usr/bin/env python3
"""Download MuSiQue multi-hop QA dataset and convert to benchmark JSONL format."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_musique(split: str):
    """Load MuSiQue from HuggingFace Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' package is required for this script.\n"
            "Install it with: pip install datasets"
        )
    return load_dataset("dgslibisey/MuSiQue", split=split)


def hop_count(example: dict) -> int:
    """Derive hop count from question_decomposition length."""
    return len(example["question_decomposition"])


def convert_to_jsonl(
    split: str,
    output_path: Path,
    hops_filter: list[int] | None,
    max_examples: int | None,
    include_unanswerable: bool,
    seed: int,
    shuffle: bool,
) -> int:
    dataset = load_musique(split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for example in dataset:
        if not include_unanswerable and not example["answerable"]:
            continue

        hops = hop_count(example)
        if hops_filter and hops not in hops_filter:
            continue

        record = {
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "answer_aliases": example["answer_aliases"],
            "hops": hops,
            "answerable": example["answerable"],
            "question_decomposition": example["question_decomposition"],
        }
        records.append(record)

    hop_dist = Counter(r["hops"] for r in records)
    print(f"Hop distribution (before limit): {dict(sorted(hop_dist.items()))}")

    if shuffle:
        random.seed(seed)
        random.shuffle(records)

    if max_examples is not None:
        records = records[:max_examples]

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MuSiQue multi-hop QA dataset and convert to benchmark JSONL format"
    )
    parser.add_argument(
        "--output",
        default="data/musique.jsonl",
        help="Output JSONL path (default: data/musique.jsonl)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="HuggingFace dataset split (default: validation)",
    )
    parser.add_argument(
        "--hops",
        type=int,
        nargs="*",
        default=None,
        help="Filter by hop count(s), e.g. --hops 3 4 (default: all)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of output examples (default: all)",
    )
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Include unanswerable questions (default: answerable only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before --max-examples truncation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    print(f"Loading MuSiQue split={args.split} from HuggingFace...")
    count = convert_to_jsonl(
        split=args.split,
        output_path=output_path,
        hops_filter=args.hops,
        max_examples=args.max_examples,
        include_unanswerable=args.include_unanswerable,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    filter_desc: list[str] = []
    if args.hops:
        filter_desc.append(f"hops={args.hops}")
    if not args.include_unanswerable:
        filter_desc.append("answerable_only")
    if args.max_examples:
        filter_desc.append(f"max={args.max_examples}")

    filters = ", ".join(filter_desc) if filter_desc else "none"
    print(f"Wrote {count} records to {output_path} (filters: {filters})")


if __name__ == "__main__":
    main()
