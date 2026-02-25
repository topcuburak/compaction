# BrowseComp LangChain Compaction Benchmark

LangChain-based harness to benchmark memory compaction on long-horizon web QA tasks (e.g., BrowseComp-style questions).

Implemented compaction strategies:
- `none`: no memory compaction
- `trim`: keep recent messages under a token budget
- `summarize`: summarize older trajectory into a running summary and keep recent messages

Compaction is applied only when estimated context tokens exceed `--token-budget` (default `110000`).

## What this gives you

- ReAct-style tool loop with web search (`web_search` via DuckDuckGo)
- Pluggable compaction strategy (`--strategy`)
- Two tool execution modes (`--tool-mode`):
- `manual` (default): model emits `SEARCH: ...` / `FINAL_ANSWER: ...`; works with standard vLLM chat endpoints
- `native`: OpenAI-style tool-calling (`tools`); requires vLLM tool-call support flags
- Per-question logs (`rows.jsonl`) and aggregate metrics (`summary.json`)
- Exact-match evaluation when gold answers are present

## Setup

```bash
cd /home/topcuburak/Desktop/compaction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set endpoint/model in `.env` for your local OpenAI-compatible server (vLLM, etc.).

## Dataset format

Use `.jsonl` or `.json`. Required question fields can be one of:
- `question`, `query`, `prompt`, `input`

Optional answer fields for evaluation can be one of:
- `answer`, `gold`, `target`, `label`, `expected_answer`

Example `.jsonl`:

```json
{"id":"q1","question":"Who founded SpaceX?","answer":"Elon Musk"}
{"id":"q2","question":"What is the capital of Japan?","answer":"Tokyo"}
```

## Download official BrowseComp data

This repo includes a helper that downloads the official BrowseComp CSV and converts it to the JSONL format expected by this benchmark.

```bash
python scripts/prepare_browsecomp_dataset.py
```

This creates:
- `data/browse_comp_test_set.csv`
- `data/browsecomp.jsonl`

## Run one strategy

```bash
python run_browsecomp_langchain.py \
  --dataset data/sample_browsecomp.jsonl \
  --strategy summarize \
  --tool-mode manual \
  --max-questions 10 \
  --max-steps 24 \
  --token-budget 110000
```

Use `--tool-mode native` only if your server supports OpenAI tool-calling.

Example on official BrowseComp data:

```bash
python run_browsecomp_langchain.py \
  --dataset data/browsecomp.jsonl \
  --strategy summarize \
  --tool-mode manual \
  --max-questions 200 \
  --max-steps 24 \
  --token-budget 110000
```

Evaluate only high-context cases (where trajectory peak tokens exceed 110k):

```bash
python run_browsecomp_langchain.py \
  --dataset data/browsecomp.jsonl \
  --strategy summarize \
  --tool-mode manual \
  --max-questions 200 \
  --max-steps 24 \
  --token-budget 110000 \
  --only-over-budget \
  --over-budget-threshold 110000
```

`rows.jsonl` includes both:
- `context_tokens_est`: estimated tokens at finish
- `max_context_tokens_est`: peak estimated tokens during the trajectory

Build a reusable long-context subset dataset (one full pass, then reuse):

```bash
python run_browsecomp_langchain.py \
  --dataset data/browsecomp.jsonl \
  --strategy summarize \
  --tool-mode manual \
  --max-questions 1266 \
  --max-steps 24 \
  --token-budget 110000 \
  --save-over-budget-dataset data/browsecomp_over_110k.jsonl
```

Then use only that subset for later experiments:

```bash
python run_browsecomp_langchain.py \
  --dataset data/browsecomp_over_110k.jsonl \
  --strategy trim \
  --tool-mode manual \
  --max-steps 24 \
  --token-budget 110000
```

## Compare all three baseline strategies

```bash
./scripts/run_all_strategies.sh data/sample_browsecomp.jsonl
```

Each run writes to `results/<timestamp>_<strategy>/` with:
- `rows.jsonl`
- `summary.json`

## Notes

- This setup is intentionally simple and controlled for compaction experiments.
- Add your own method by extending `ConversationCompactor` in `src/browsecomp_benchmark/compaction.py`.
- If your model supports 128K context, set `--token-budget` accordingly (e.g., 100K-120K).
