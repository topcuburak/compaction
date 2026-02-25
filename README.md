# BrowseComp LangChain Compaction Benchmark

LangChain-based harness to benchmark memory compaction on long-horizon web QA tasks (e.g., BrowseComp-style questions).

Implemented compaction strategies:
- `none`: no memory compaction
- `trim`: keep recent messages under a token budget
- `summarize`: summarize older trajectory into a running summary and keep recent messages

## What this gives you

- ReAct-style tool loop with web search (`web_search` via DuckDuckGo)
- Pluggable compaction strategy (`--strategy`)
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

## Run one strategy

```bash
python run_browsecomp_langchain.py \
  --dataset data/sample_browsecomp.jsonl \
  --strategy summarize \
  --max-questions 10 \
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
