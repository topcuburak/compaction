from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from ddgs import DDGS
except ImportError:  # Backward compatibility for older environments.
    from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .compaction import ConversationCompactor, estimate_tokens


SYSTEM_PROMPT_NATIVE = """You are a persistent web research agent for hard factual QA.

Process:
1) Use web_search to gather evidence from multiple sources.
2) Cross-check key claims when uncertain.
3) Keep reasoning concise and continue searching until confident.
4) When ready, output exactly one line:
FINAL_ANSWER: <short answer>

Do not output FINAL_ANSWER until you are done searching.
"""


SYSTEM_PROMPT_MANUAL = """You are a persistent web research agent for hard factual QA.

You must output exactly one action per turn in one of these formats:
- SEARCH: <query>
- FINAL_ANSWER: <short answer>

Rules:
1) Use SEARCH to request web evidence.
2) After receiving search results, either SEARCH again or output FINAL_ANSWER.
3) Keep answers short and factual.
4) Do not output any other format.
"""


class SearchInput(BaseModel):
    query: str = Field(..., description="Web search query")


@dataclass(slots=True)
class AgentConfig:
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    request_timeout: int = 120
    max_retries: int = 2
    max_steps: int = 24
    max_search_results: int = 5
    max_result_chars: int = 700
    tool_mode: str = "manual"  # one of: manual, native
    min_searches: int = 0
    print_context_lengths: bool = False


@dataclass(slots=True)
class QuestionRunResult:
    question_id: str
    question: str
    final_answer: str
    steps: int
    tool_calls: int
    context_tokens_est: int
    max_context_tokens_est: int
    request_context_tokens_est: list[int]
    max_request_context_tokens_est: int
    crossed_token_budget: bool
    finished_reason: str


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _extract_final_answer(content: object) -> str | None:
    text = _content_to_text(content).strip()
    match = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        first_line = answer.splitlines()[0].strip()
        if first_line:
            return first_line
    return None


def _extract_search_query(content: object) -> str | None:
    text = _content_to_text(content)
    match = re.search(r"^\s*SEARCH\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    query = match.group(1).strip()
    if not query:
        return None
    return query.splitlines()[0].strip()


def _update_peak_tokens(messages: list[BaseMessage], current_peak: int) -> int:
    return max(current_peak, estimate_tokens(messages))


def _build_search_tool(max_results: int, max_result_chars: int) -> StructuredTool:
    def web_search(query: str) -> str:
        rows: list[dict[str, object]] = []
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))

        if not rows:
            return "No results found."

        rendered: list[str] = []
        for idx, row in enumerate(rows, start=1):
            title = str(row.get("title", "")).strip()
            href = str(row.get("href", "")).strip()
            body = str(row.get("body", "")).strip()
            if len(body) > max_result_chars:
                body = body[:max_result_chars] + "..."
            rendered.append(f"[{idx}] {title}\nURL: {href}\nSnippet: {body}")

        return "\n\n".join(rendered)

    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description="Search the web and return top snippets with URLs.",
        args_schema=SearchInput,
    )


def build_model(config: AgentConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
    )


def _run_with_native_tools(
    question_id: str,
    question: str,
    llm: ChatOpenAI,
    search_tool: StructuredTool,
    config: AgentConfig,
    compactor: ConversationCompactor,
) -> QuestionRunResult:
    llm_with_tools = llm.bind_tools([search_tool])
    token_budget = compactor.config.token_budget

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT_NATIVE),
        HumanMessage(content=f"Question: {question}"),
    ]
    tool_calls = 0
    max_tokens = estimate_tokens(messages)
    request_context_tokens: list[int] = []
    crossed_budget = max_tokens > token_budget

    for step in range(1, config.max_steps + 1):
        pre_compaction_tokens = estimate_tokens(messages)
        max_tokens = max(max_tokens, pre_compaction_tokens)
        crossed_budget = crossed_budget or pre_compaction_tokens > token_budget
        messages = compactor.compact(messages, summarizer=llm)
        max_tokens = _update_peak_tokens(messages, max_tokens)
        crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

        request_context_tokens.append(estimate_tokens(messages))
        if config.print_context_lengths:
            print(f"  [request {step}/{config.max_steps}] context_tokens_est={request_context_tokens[-1]}")
        ai_message = llm_with_tools.invoke(messages)
        if not isinstance(ai_message, AIMessage):
            ai_message = AIMessage(content=_content_to_text(ai_message))

        messages.append(ai_message)
        max_tokens = _update_peak_tokens(messages, max_tokens)
        crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

        if ai_message.tool_calls:
            for call in ai_message.tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("args", {})
                tool_call_id = call.get("id", "")

                if tool_name != search_tool.name:
                    observation = f"Tool error: unknown tool '{tool_name}'."
                else:
                    try:
                        if isinstance(tool_args, dict):
                            observation = search_tool.invoke(tool_args)
                        else:
                            observation = search_tool.invoke({"query": str(tool_args)})
                    except Exception as exc:
                        observation = f"Tool error: {exc}"

                tool_calls += 1
                messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))
                max_tokens = _update_peak_tokens(messages, max_tokens)
                crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

            continue

        final = _extract_final_answer(ai_message.content)
        if final:
            if tool_calls < config.min_searches:
                messages.append(
                    HumanMessage(
                        content=(
                            "You need to gather more evidence before answering. "
                            "Call web_search to find additional supporting information."
                        )
                    )
                )
                max_tokens = _update_peak_tokens(messages, max_tokens)
                crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget
                continue

            return QuestionRunResult(
                question_id=question_id,
                question=question,
                final_answer=final,
                steps=step,
                tool_calls=tool_calls,
                context_tokens_est=estimate_tokens(messages),
                max_context_tokens_est=max_tokens,
                request_context_tokens_est=request_context_tokens,
                max_request_context_tokens_est=max(request_context_tokens, default=0),
                crossed_token_budget=crossed_budget,
                finished_reason="final_answer",
            )

        messages.append(
            HumanMessage(
                content=(
                    "If more evidence is needed, call web_search. "
                    "Otherwise return final output as 'FINAL_ANSWER: <short answer>'."
                )
            )
        )
        max_tokens = _update_peak_tokens(messages, max_tokens)
        crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

    messages.append(
        HumanMessage(
            content="Stop searching and provide your best final output as 'FINAL_ANSWER: <short answer>'."
        )
    )
    max_tokens = _update_peak_tokens(messages, max_tokens)
    crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

    request_context_tokens.append(estimate_tokens(messages))
    forced = llm.invoke(messages)
    final = _extract_final_answer(forced.content)
    if not final:
        fallback = _content_to_text(forced.content).strip()
        final = fallback.splitlines()[0].strip() if fallback else ""

    return QuestionRunResult(
        question_id=question_id,
        question=question,
        final_answer=final,
        steps=config.max_steps,
        tool_calls=tool_calls,
        context_tokens_est=estimate_tokens(messages),
        max_context_tokens_est=max_tokens,
        request_context_tokens_est=request_context_tokens,
        max_request_context_tokens_est=max(request_context_tokens, default=0),
        crossed_token_budget=crossed_budget,
        finished_reason="max_steps",
    )


def _run_with_manual_actions(
    question_id: str,
    question: str,
    llm: ChatOpenAI,
    search_tool: StructuredTool,
    config: AgentConfig,
    compactor: ConversationCompactor,
) -> QuestionRunResult:
    token_budget = compactor.config.token_budget
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT_MANUAL),
        HumanMessage(content=f"Question: {question}"),
    ]
    tool_calls = 0
    max_tokens = estimate_tokens(messages)
    request_context_tokens: list[int] = []
    crossed_budget = max_tokens > token_budget

    for step in range(1, config.max_steps + 1):
        pre_compaction_tokens = estimate_tokens(messages)
        max_tokens = max(max_tokens, pre_compaction_tokens)
        crossed_budget = crossed_budget or pre_compaction_tokens > token_budget
        messages = compactor.compact(messages, summarizer=llm)
        max_tokens = _update_peak_tokens(messages, max_tokens)
        crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

        request_context_tokens.append(estimate_tokens(messages))
        if config.print_context_lengths:
            print(f"  [request {step}/{config.max_steps}] context_tokens_est={request_context_tokens[-1]}")
        ai_message = llm.invoke(messages)
        if not isinstance(ai_message, AIMessage):
            ai_message = AIMessage(content=_content_to_text(ai_message))

        messages.append(ai_message)
        max_tokens = _update_peak_tokens(messages, max_tokens)
        crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

        final = _extract_final_answer(ai_message.content)
        if final:
            if tool_calls < config.min_searches:
                messages.append(
                    HumanMessage(
                        content=(
                            "You need to gather more evidence before answering. "
                            "Use SEARCH: <query> to find additional supporting information."
                        )
                    )
                )
                max_tokens = _update_peak_tokens(messages, max_tokens)
                crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget
                continue

            return QuestionRunResult(
                question_id=question_id,
                question=question,
                final_answer=final,
                steps=step,
                tool_calls=tool_calls,
                context_tokens_est=estimate_tokens(messages),
                max_context_tokens_est=max_tokens,
                request_context_tokens_est=request_context_tokens,
                max_request_context_tokens_est=max(request_context_tokens, default=0),
                crossed_token_budget=crossed_budget,
                finished_reason="final_answer",
            )

        search_query = _extract_search_query(ai_message.content)
        if search_query:
            try:
                observation = search_tool.invoke({"query": search_query})
            except Exception as exc:
                observation = f"Tool error: {exc}"

            tool_calls += 1
            messages.append(
                HumanMessage(
                    content=(
                        "[WEB_SEARCH_RESULTS]\n"
                        f"Query: {search_query}\n"
                        f"{observation}\n\n"
                        "Respond with either SEARCH: <query> or FINAL_ANSWER: <short answer>."
                    )
                )
            )
            max_tokens = _update_peak_tokens(messages, max_tokens)
            crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget
            continue

        messages.append(
            HumanMessage(
                content=(
                    "Invalid format. Respond with exactly one action line:\n"
                    "SEARCH: <query>\n"
                    "or\n"
                    "FINAL_ANSWER: <short answer>"
                )
            )
        )
        max_tokens = _update_peak_tokens(messages, max_tokens)
        crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

    messages.append(
        HumanMessage(
            content=(
                "Stop searching and give your best response in exactly this format:\n"
                "FINAL_ANSWER: <short answer>"
            )
        )
    )
    max_tokens = _update_peak_tokens(messages, max_tokens)
    crossed_budget = crossed_budget or estimate_tokens(messages) > token_budget

    request_context_tokens.append(estimate_tokens(messages))
    forced = llm.invoke(messages)
    final = _extract_final_answer(forced.content)
    if not final:
        fallback = _content_to_text(forced.content).strip()
        final = fallback.splitlines()[0].strip() if fallback else ""

    return QuestionRunResult(
        question_id=question_id,
        question=question,
        final_answer=final,
        steps=config.max_steps,
        tool_calls=tool_calls,
        context_tokens_est=estimate_tokens(messages),
        max_context_tokens_est=max_tokens,
        request_context_tokens_est=request_context_tokens,
        max_request_context_tokens_est=max(request_context_tokens, default=0),
        crossed_token_budget=crossed_budget,
        finished_reason="max_steps",
    )


def run_browsecomp_question(
    question_id: str,
    question: str,
    config: AgentConfig,
    compactor: ConversationCompactor,
) -> QuestionRunResult:
    llm = build_model(config)
    search_tool = _build_search_tool(config.max_search_results, config.max_result_chars)

    mode = config.tool_mode.lower().strip()
    if mode == "native":
        return _run_with_native_tools(question_id, question, llm, search_tool, config, compactor)
    if mode == "manual":
        return _run_with_manual_actions(question_id, question, llm, search_tool, config, compactor)

    raise ValueError(f"Unknown tool mode: {config.tool_mode}. Use 'manual' or 'native'.")
