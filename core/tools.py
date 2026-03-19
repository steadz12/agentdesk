"""
core/tools.py — Tool Definitions & Registry
─────────────────────────────────────────────────────
Concept: Tool Use / Function Calling

This module defines:
  • TOOL_DEFINITIONS  — JSON schemas passed to Claude's `tools` parameter
  • ToolRegistry      — maps tool names → Python implementations
  • execute_tool()    — dispatch function called after Claude returns a tool_use block

To add a new tool:
  1. Add an entry to TOOL_DEFINITIONS
  2. Implement the function in ToolRegistry
  3. Register it in ToolRegistry.__init__
"""
import json
import math
import datetime
from typing import Any


# ── Tool Schemas (passed to Claude API) ────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '(12 * 4) / 3 + sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Return the current UTC date and time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone label for display (informational only).",
                    "default": "UTC"
                }
            },
            "required": []
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the data/ directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Relative path inside data/, e.g. 'report.txt'"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "write_file",
        "description": "Write text content to a file inside the data/output/ directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename inside data/output/, e.g. 'summary.txt'"
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write."
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the internal knowledge base for relevant information on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_memory",
        "description": "Save an important fact to long-term memory for future retrieval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "A short identifier for this fact, e.g. 'user_preference_language'"
                },
                "value": {
                    "type": "string",
                    "description": "The fact or value to remember."
                }
            },
            "required": ["key", "value"]
        }
    }
]


# ── Tool Implementations ────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Holds the Python implementations of each tool.
    Inject rag_engine and memory_manager at startup to give tools access.
    """

    def __init__(self, rag_engine=None, memory_manager=None):
        self.rag = rag_engine
        self.memory = memory_manager
        self._dispatch = {
            "calculate": self._calculate,
            "get_current_datetime": self._get_datetime,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_knowledge_base": self._search_kb,
            "save_memory": self._save_memory,
        }

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Route a tool call to its implementation. Returns a string result."""
        fn = self._dispatch.get(tool_name)
        if fn is None:
            return f"Error: unknown tool '{tool_name}'"
        try:
            return fn(**tool_input)
        except Exception as e:
            return f"Tool error: {e}"

    # ── Implementations ────────────────────────────────────────────────────────

    def _calculate(self, expression: str) -> str:
        # Safe eval: only math functions + operators
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_globals["__builtins__"] = {}
        result = eval(expression, safe_globals)  # noqa: S307
        return f"Result: {result}"

    def _get_datetime(self, timezone: str = "UTC") -> str:
        now = datetime.datetime.utcnow()
        return f"Current UTC datetime: {now.strftime('%Y-%m-%d %H:%M:%S')} (requested tz: {timezone})"

    def _read_file(self, filename: str) -> str:
        path = f"data/{filename}"
        if not path.startswith("data/"):
            return "Error: can only read files inside data/"
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            return f"File not found: {path}"

    def _write_file(self, filename: str, content: str) -> str:
        import os
        os.makedirs("data/output", exist_ok=True)
        path = f"data/output/{filename}"
        with open(path, "w") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"

    def _search_kb(self, query: str) -> str:
        if self.rag is None:
            return "Knowledge base not available."
        chunks = self.rag.retrieve(query)
        return self.rag.format_context(chunks)

    def _save_memory(self, key: str, value: str) -> str:
        if self.memory is None:
            return "Memory manager not available."
        self.memory.save_fact(key, value)
        return f"Saved to long-term memory: {key} = {value}"
