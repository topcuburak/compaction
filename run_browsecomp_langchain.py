#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from browsecomp_benchmark.agent import AgentConfig, run_browsecomp_question
from browsecomp_benchmark.compaction import CompactionConfig, ConversationCompactor
from browsecomp_benchmark.dataset import load_browsecomp_dataset
from browsecomp_benchmark.metrics import exact_match_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compaction strategies on BrowseComp-style long-horizon QA using LangChain"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to .jsonl/.json dataset")
    parser.add_argument("--strategy", type=str, default="none", choices=["none", "trim", "summarize"])
    parser.add_argument("--max-questions", type=int, default=25)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=24)

    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--token-budget", type=int, default=110_000)
    parser.add_argument("--keep-last-messages", type=int, default=12)
    parser.add_argument("--summary-trigger-ratio", type=float, default=0.92)

    parser.add_argument("--max-search-results", type=int, default=5)
    parser.add_argument("--max-result-chars", type=int, default=700)

    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    all_examples = load_browsecomp_dataset(args.dataset)
    selected = all_examples[args.start_index : args.start_index + args.max_questions]

    if not selected:
        raise ValueError("No examples selected. Check --start-index / --max-questions.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"{timestamp}_{args.strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.json"

    agent_config = AgentConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_search_results=args.max_search_results,
        max_result_chars=args.max_result_chars,
    )

    compaction_config = CompactionConfig(
        strategy=args.strategy,
        token_budget=args.token_budget,
        keep_last_messages=args.keep_last_messages,
        summary_trigger_ratio=args.summary_trigger_ratio,
    )

    scores: list[float] = []
    steps_list: list[int] = []
    tool_calls_list: list[int] = []
    token_est_list: list[int] = []
    latency_list: list[float] = []

    with rows_path.open("w", encoding="utf-8") as rows_file:
        for example in tqdm(selected, desc=f"Running {args.strategy}"):
            compactor = ConversationCompactor(compaction_config)

            start = time.perf_counter()
            run = run_browsecomp_question(
                question_id=example.id,
                question=example.question,
                config=agent_config,
                compactor=compactor,
            )
            latency = time.perf_counter() - start

            score = None
            if example.answer is not None:
                score = exact_match_score(run.final_answer, example.answer)
                scores.append(score)

            steps_list.append(run.steps)
            tool_calls_list.append(run.tool_calls)
            token_est_list.append(run.context_tokens_est)
            latency_list.append(latency)

            row = {
                "id": example.id,
                "question": example.question,
                "gold": example.answer,
                "prediction": run.final_answer,
                "exact_match": score,
                "steps": run.steps,
                "tool_calls": run.tool_calls,
                "context_tokens_est": run.context_tokens_est,
                "finished_reason": run.finished_reason,
                "latency_sec": round(latency, 3),
                "strategy": args.strategy,
            }
            rows_file.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "num_examples": len(selected),
        "strategy": args.strategy,
        "model": args.model,
        "base_url": args.base_url,
        "accuracy_exact_match": (statistics.mean(scores) if scores else None),
        "avg_steps": statistics.mean(steps_list),
        "avg_tool_calls": statistics.mean(tool_calls_list),
        "avg_context_tokens_est": statistics.mean(token_est_list),
        "avg_latency_sec": statistics.mean(latency_list),
        "rows_path": str(rows_path.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
