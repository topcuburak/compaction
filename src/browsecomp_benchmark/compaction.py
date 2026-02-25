from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def message_to_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        content_text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(json.dumps(part, ensure_ascii=True))
            else:
                parts.append(str(part))
        content_text = "\n".join(parts)
    else:
        content_text = str(content)

    role = message.__class__.__name__.replace("Message", "").lower()
    return f"{role}: {content_text}"


def estimate_tokens_from_text(text: str) -> int:
    # Fast approximation that is stable across model backends.
    return max(1, len(text) // 4)


def estimate_tokens(messages: Iterable[BaseMessage]) -> int:
    return sum(estimate_tokens_from_text(message_to_text(msg)) + 4 for msg in messages)


@dataclass(slots=True)
class CompactionConfig:
    strategy: str = "none"  # one of: none, trim, summarize
    token_budget: int = 110_000
    keep_last_messages: int = 12
    summary_trigger_ratio: float = 0.92
    summary_max_chars: int = 24_000


@dataclass(slots=True)
class ConversationCompactor:
    config: CompactionConfig
    running_summary: str = ""
    _summary_prefix: str = field(default="[RUNNING_SUMMARY]", init=False)

    def compact(self, messages: list[BaseMessage], summarizer: BaseChatModel | None = None) -> list[BaseMessage]:
        strategy = self.config.strategy.lower().strip()
        if strategy == "none":
            return messages
        if estimate_tokens(messages) <= self.config.token_budget:
            return messages
        if strategy == "trim":
            return self._trim(messages)
        if strategy == "summarize":
            return self._summarize(messages, summarizer)
        raise ValueError(f"Unknown compaction strategy: {self.config.strategy}")

    def _is_summary_message(self, message: BaseMessage) -> bool:
        return isinstance(message, SystemMessage) and isinstance(message.content, str) and message.content.startswith(
            self._summary_prefix
        )

    def _build_summary_message(self) -> SystemMessage:
        text = self.running_summary.strip() or "(no summary yet)"
        return SystemMessage(content=f"{self._summary_prefix}\n{text}")

    def _trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        budget = self.config.token_budget
        total = estimate_tokens(messages)
        if total <= budget:
            return messages

        keep: list[BaseMessage] = []
        start_index = 0
        if messages and isinstance(messages[0], SystemMessage):
            keep.append(messages[0])
            start_index = 1

        keep_tokens = estimate_tokens(keep)
        tail: list[BaseMessage] = []
        for message in reversed(messages[start_index:]):
            msg_tokens = estimate_tokens([message])
            if keep_tokens + msg_tokens > budget:
                break
            tail.append(message)
            keep_tokens += msg_tokens

        tail.reverse()
        if not tail and len(messages) > start_index:
            tail = [messages[-1]]

        dropped = len(messages) - (len(keep) + len(tail))
        if dropped > 0:
            keep.append(SystemMessage(content=f"[TRIM_NOTICE] Dropped {dropped} old messages due to token budget."))

        return keep + tail

    def _summarize(self, messages: list[BaseMessage], summarizer: BaseChatModel | None) -> list[BaseMessage]:
        base_messages = [m for m in messages if not self._is_summary_message(m)]
        budget = self.config.token_budget
        keep_last = max(2, self.config.keep_last_messages)

        if summarizer is None:
            return self._trim(messages)

        if len(base_messages) > 2 and len(base_messages) - 1 > keep_last:
            older_messages = base_messages[1:-keep_last]
            if older_messages:
                chunk_text = "\n".join(message_to_text(m) for m in older_messages)
                if len(chunk_text) > self.config.summary_max_chars:
                    chunk_text = chunk_text[: self.config.summary_max_chars]

                summary_prompt = [
                    SystemMessage(
                        content=(
                            "You compact an agent trajectory for a web-question-answering task. "
                            "Preserve factual leads, constraints, intermediate conclusions, and unresolved questions. "
                            "Write concise bullet points for future reasoning."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Current summary:\n"
                            f"{self.running_summary or '(none)'}\n\n"
                            "New trajectory chunk to merge:\n"
                            f"{chunk_text}\n\n"
                            "Return only the updated summary bullets."
                        )
                    ),
                ]

                try:
                    response = summarizer.invoke(summary_prompt)
                    new_summary = response.content if isinstance(response.content, str) else str(response.content)
                    new_summary = new_summary.strip()
                    if new_summary:
                        self.running_summary = new_summary
                except Exception:
                    # If summary fails, keep the run going with deterministic trimming.
                    return self._trim(messages)

        if self.running_summary and base_messages and isinstance(base_messages[0], SystemMessage):
            compacted = [
                base_messages[0],
                self._build_summary_message(),
                *base_messages[-keep_last:],
            ]
        else:
            compacted = base_messages

        if estimate_tokens(compacted) > budget:
            return self._trim(compacted)

        return compacted
