"""
core/memory.py — Memory Manager
─────────────────────────────────────────────────────
Concept: Memory & Context Management

Two memory tiers:
  • Short-term  — sliding window of recent messages (in-process list)
  • Long-term   — persistent JSON store; keyed facts the system can recall
"""
import json
import os
from datetime import datetime
from typing import Any

SHORT_TERM_LIMIT = 20
LONG_TERM_DB_PATH = "data/memory.json"


class Message:
    def __init__(self, role: str, content: str, agent: str = ""):
        self.role = role
        self.content = content
        self.agent = agent
        self.timestamp = datetime.utcnow().isoformat()


class MemoryManager:
    """
    Manages short-term (in-memory) and long-term (file-backed) memory.

    Usage:
        mem = MemoryManager()
        mem.add_message("user", "Schedule a meeting tomorrow at 10am")
        mem.save_fact("last_task", "meeting scheduled")
        history = mem.get_short_term()      # for passing to Claude API
        fact = mem.recall_fact("last_task")
    """

    def __init__(self):
        self._short_term: list[Message] = []
        self._long_term: dict[str, Any] = {}
        self._load_long_term()

    # ── Short-term ─────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, agent: str = "") -> None:
        """Add a message and enforce the sliding-window limit."""
        self._short_term.append(Message(role=role, content=content, agent=agent))
        if len(self._short_term) > SHORT_TERM_LIMIT:
            # Drop the oldest non-system message
            for i, m in enumerate(self._short_term):
                if m.role != "system":
                    self._short_term.pop(i)
                    break

    def get_short_term(self) -> list[dict]:
        """Return messages formatted for the Anthropic messages API."""
        return [{"role": m.role, "content": m.content}
                for m in self._short_term if m.role in ("user", "assistant")]

    def clear_short_term(self) -> None:
        self._short_term = []

    def get_context_summary(self) -> str:
        """Produce a plain-text summary of recent turns for system prompts."""
        if not self._short_term:
            return "No conversation history yet."
        lines = []
        for m in self._short_term[-6:]:
            label = m.agent or m.role
            text = m.content[:120] + "..." if len(m.content) > 120 else m.content
            lines.append(f"[{label}]: {text}")
        return "\n".join(lines)

    # ── Long-term ──────────────────────────────────────────────────────────────

    def save_fact(self, key: str, value: Any) -> None:
        """Persist a fact to long-term memory."""
        self._long_term[key] = {"value": value, "updated": datetime.utcnow().isoformat()}
        self._persist_long_term()

    def recall_fact(self, key: str) -> Any | None:
        """Retrieve a fact from long-term memory."""
        entry = self._long_term.get(key)
        return entry["value"] if entry else None

    def list_facts(self) -> dict:
        return {k: v["value"] for k, v in self._long_term.items()}

    def _load_long_term(self) -> None:
        if os.path.exists(LONG_TERM_DB_PATH):
            with open(LONG_TERM_DB_PATH) as f:
                self._long_term = json.load(f)

    def _persist_long_term(self) -> None:
        os.makedirs(os.path.dirname(LONG_TERM_DB_PATH), exist_ok=True)
        with open(LONG_TERM_DB_PATH, "w") as f:
            json.dump(self._long_term, f, indent=2)
