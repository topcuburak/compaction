from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BrowseCompExample:
    id: str
    question: str
    answer: str | None
    raw: dict[str, Any]


def _pick_text(record: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _to_example(record: dict[str, Any], index: int) -> BrowseCompExample:
    question = _pick_text(record, ["question", "query", "prompt", "input"])
    if question is None:
        raise ValueError(f"Record at index {index} has no question-like field: {record.keys()}")

    answer = _pick_text(record, ["answer", "gold", "target", "label", "expected_answer"])
    example_id = _pick_text(record, ["id", "qid", "uuid"]) or str(index)

    return BrowseCompExample(
        id=example_id,
        question=question,
        answer=answer,
        raw=record,
    )


def load_browsecomp_dataset(path: str | Path) -> list[BrowseCompExample]:
    """Load a BrowseComp-style dataset from .jsonl or .json.

    Expected fields per item:
    - question-like: one of [question, query, prompt, input]
    - answer-like (optional): one of [answer, gold, target, label, expected_answer]
    - id-like (optional): one of [id, qid, uuid]
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples: list[BrowseCompExample] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL line {idx + 1} is not an object")
                examples.append(_to_example(record, idx))
        return examples

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            if "questions" in payload and isinstance(payload["questions"], list):
                items = payload["questions"]
            elif "data" in payload and isinstance(payload["data"], list):
                items = payload["data"]
            else:
                raise ValueError("JSON dataset must contain a top-level list or one of ['questions', 'data'] lists")
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("JSON dataset payload must be a list or dict")

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {idx} is not an object")
            examples.append(_to_example(item, idx))

        return examples

    raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .jsonl or .json")
