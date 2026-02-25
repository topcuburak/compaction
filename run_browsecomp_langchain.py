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


def _mean_or_none(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


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
    parser.add_argument("--request-timeout", type=int, default=int(os.getenv("REQUEST_TIMEOUT", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "2")))
    parser.add_argument("--tool-mode", type=str, default=os.getenv("TOOL_MODE", "manual"), choices=["manual", "native"])

    parser.add_argument("--token-budget", type=int, default=110_000)
    parser.add_argument("--keep-last-messages", type=int, default=12)
    parser.add_argument("--summary-trigger-ratio", type=float, default=0.92)

    parser.add_argument("--max-search-results", type=int, default=5)
    parser.add_argument("--max-result-chars", type=int, default=700)
    parser.add_argument(
        "--only-over-budget",
        action="store_true",
        help="Keep/evaluate only rows where max_context_tokens_est exceeds a threshold",
    )
    parser.add_argument(
        "--over-budget-threshold",
        type=int,
        default=None,
        help="Threshold for --only-over-budget (defaults to --token-budget)",
    )
    parser.add_argument(
        "--save-over-budget-dataset",
        type=str,
        default="",
        help="Optional path to save dataset rows whose peak tokens exceed threshold",
    )
    parser.add_argument(
        "--stop-after-over-budget",
        type=int,
        default=0,
        help="Stop execution after finding this many over-budget samples (0 disables early stop)",
    )

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
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        tool_mode=args.tool_mode,
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
    over_budget_threshold = args.over_budget_threshold or args.token_budget

    scores: list[float] = []
    steps_list: list[int] = []
    tool_calls_list: list[int] = []
    token_est_list: list[int] = []
    max_token_est_list: list[int] = []
    latency_list: list[float] = []
    kept_rows = 0
    filtered_out = 0
    over_budget_examples: list[dict[str, str]] = []
    num_errors = 0
    num_examples_processed = 0
    stopped_early = False
    stop_reason = ""

    with rows_path.open("w", encoding="utf-8") as rows_file:
        for example in tqdm(selected, desc=f"Running {args.strategy}"):
            num_examples_processed += 1
            compactor = ConversationCompactor(compaction_config)

            start = time.perf_counter()
            try:
                run = run_browsecomp_question(
                    question_id=example.id,
                    question=example.question,
                    config=agent_config,
                    compactor=compactor,
                )
                latency = time.perf_counter() - start
            except Exception as exc:
                latency = time.perf_counter() - start
                num_errors += 1

                if args.only_over_budget:
                    filtered_out += 1
                    continue

                error_row = {
                    "id": example.id,
                    "question": example.question,
                    "gold": example.answer,
                    "prediction": "",
                    "exact_match": None,
                    "steps": 0,
                    "tool_calls": 0,
                    "context_tokens_est": 0,
                    "max_context_tokens_est": 0,
                    "crossed_token_budget": False,
                    "finished_reason": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc).replace("\n", " ")[:1000],
                    "latency_sec": round(latency, 3),
                    "strategy": args.strategy,
                    "kept_for_summary": False,
                }
                rows_file.write(json.dumps(error_row, ensure_ascii=True) + "\n")
                continue

            is_over_budget = run.max_context_tokens_est > over_budget_threshold
            keep_row = (not args.only_over_budget) or (run.max_context_tokens_est > over_budget_threshold)
            if is_over_budget:
                subset_record: dict[str, str] = {
                    "id": example.id,
                    "question": example.question,
                }
                if example.answer is not None:
                    subset_record["answer"] = example.answer
                over_budget_examples.append(subset_record)

            score = None
            if example.answer is not None:
                score = exact_match_score(run.final_answer, example.answer)
                if keep_row:
                    scores.append(score)

            if keep_row:
                kept_rows += 1
                steps_list.append(run.steps)
                tool_calls_list.append(run.tool_calls)
                token_est_list.append(run.context_tokens_est)
                max_token_est_list.append(run.max_context_tokens_est)
                latency_list.append(latency)
            else:
                filtered_out += 1

            row = {
                "id": example.id,
                "question": example.question,
                "gold": example.answer,
                "prediction": run.final_answer,
                "exact_match": score,
                "steps": run.steps,
                "tool_calls": run.tool_calls,
                "context_tokens_est": run.context_tokens_est,
                "max_context_tokens_est": run.max_context_tokens_est,
                "crossed_token_budget": run.crossed_token_budget,
                "finished_reason": run.finished_reason,
                "latency_sec": round(latency, 3),
                "strategy": args.strategy,
                "kept_for_summary": keep_row,
            }
            if keep_row:
                rows_file.write(json.dumps(row, ensure_ascii=True) + "\n")

            if args.stop_after_over_budget > 0 and len(over_budget_examples) >= args.stop_after_over_budget:
                stopped_early = True
                stop_reason = (
                    f"reached_over_budget_target:{args.stop_after_over_budget}"
                )
                break

    over_budget_dataset_path = None
    if args.save_over_budget_dataset:
        output_subset_path = Path(args.save_over_budget_dataset)
        output_subset_path.parent.mkdir(parents=True, exist_ok=True)
        with output_subset_path.open("w", encoding="utf-8") as subset_file:
            for record in over_budget_examples:
                subset_file.write(json.dumps(record, ensure_ascii=True) + "\n")
        over_budget_dataset_path = str(output_subset_path.resolve())

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "num_examples_total": len(selected),
        "num_examples_processed": num_examples_processed,
        "num_examples_kept": kept_rows,
        "num_examples_filtered_out": filtered_out,
        "num_examples_over_budget": len(over_budget_examples),
        "num_errors": num_errors,
        "stop_after_over_budget": args.stop_after_over_budget,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason or None,
        "strategy": args.strategy,
        "model": args.model,
        "base_url": args.base_url,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "tool_mode": args.tool_mode,
        "only_over_budget": args.only_over_budget,
        "over_budget_threshold": over_budget_threshold,
        "over_budget_dataset_path": over_budget_dataset_path,
        "accuracy_exact_match": _mean_or_none(scores),
        "avg_steps": _mean_or_none(steps_list),
        "avg_tool_calls": _mean_or_none(tool_calls_list),
        "avg_context_tokens_est": _mean_or_none(token_est_list),
        "avg_max_context_tokens_est": _mean_or_none(max_token_est_list),
        "avg_latency_sec": _mean_or_none(latency_list),
        "rows_path": str(rows_path.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
