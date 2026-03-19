"""
providers/llm.py — Swappable LLM Backends
──────────────────────────────────────────────────────────────────────
AgentDesk is LLM-agnostic. Swap the backend by changing one env var:

    LLM_PROVIDER=anthropic   # default — uses claude-opus-4-5
    LLM_PROVIDER=openai      # uses gpt-4o (pip install agentdesk[openai])

All providers expose the same interface:
    provider.complete(messages, system, tools) → (text, tool_calls)

Adding a new provider:
    1. Subclass LLMProvider
    2. Implement complete()
    3. Add a branch in get_llm_provider()
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use"


class LLMProvider(ABC):
    """Base class for all LLM backends."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        ...


# ── Anthropic (default) ────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """
    Uses the Anthropic Claude API.
    Required env: ANTHROPIC_API_KEY
    Model can be overridden via ANTHROPIC_MODEL env var.
    """

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

    def complete(self, messages, system="", tools=None, max_tokens=2048) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = self.client.messages.create(**kwargs)

        text = next((b.text for b in resp.content if hasattr(b, "text")), "")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        return LLMResponse(text=text, tool_calls=tool_calls, stop_reason=resp.stop_reason)


# ── OpenAI (optional — pip install agentdesk[openai]) ─────────────────────────

class OpenAIProvider(LLMProvider):
    """
    Uses the OpenAI Chat Completions API.
    Required env: OPENAI_API_KEY
    Model can be overridden via OPENAI_MODEL env var (default: gpt-4o).
    """

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install agentdesk[openai]")
        import json
        self._json = json
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Anthropic-style tool schemas to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    def complete(self, messages, system="", tools=None, max_tokens=2048) -> LLMResponse:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for m in messages:
            if isinstance(m["content"], list):
                # Tool result messages
                for block in m["content"]:
                    if block.get("type") == "tool_result":
                        oai_messages.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })
            else:
                oai_messages.append({"role": m["role"], "content": m["content"]})

        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=max_tokens, messages=oai_messages
        )
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=self._json.loads(tc.function.arguments),
                ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(text=text, tool_calls=tool_calls, stop_reason=stop_reason)


# ── Factory ────────────────────────────────────────────────────────────────────

def get_llm_provider() -> LLMProvider:
    """
    Return the configured LLM provider.
    Set LLM_PROVIDER env var to switch backends.

    Supported values: "anthropic" (default), "openai"
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicProvider()
    if provider == "openai":
        return OpenAIProvider()
    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Supported: anthropic, openai. "
        "See providers/llm.py to add your own."
    )
