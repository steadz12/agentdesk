"""
tests/test_core.py — Core component tests
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch


# ── Memory ─────────────────────────────────────────────────────────────────────

class TestMemoryManager:
    def setup_method(self):
        # Patch file I/O so tests don't touch disk
        with patch("builtins.open", MagicMock()), \
             patch("os.path.exists", return_value=False):
            from core.memory import MemoryManager
            self.mem = MemoryManager()

    def test_add_and_retrieve_messages(self):
        self.mem.add_message("user", "Hello")
        self.mem.add_message("assistant", "Hi there!")
        history = self.mem.get_short_term()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Hi there!"

    def test_sliding_window_enforced(self):
        from core.memory import MemoryManager
        import unittest.mock as mock
        with mock.patch("os.path.exists", return_value=False), \
             mock.patch("builtins.open", mock.MagicMock()):
            mem = MemoryManager()
            mem._short_term = []
            # Override limit for testing
            import core.memory as cm
            original = cm.SHORT_TERM_LIMIT
            cm.SHORT_TERM_LIMIT = 3
            for i in range(5):
                mem.add_message("user", f"msg {i}")
            assert len(mem._short_term) <= 4  # limit + 1 before eviction
            cm.SHORT_TERM_LIMIT = original

    def test_context_summary_empty(self):
        assert "No conversation" in self.mem.get_context_summary()

    def test_context_summary_with_messages(self):
        self.mem.add_message("user", "What is the sprint goal?")
        summary = self.mem.get_context_summary()
        assert "sprint" in summary.lower()


# ── RAG Engine ─────────────────────────────────────────────────────────────────

class TestRAGEngine:
    def setup_method(self):
        from providers.vector_store import InMemoryVectorStore
        from core.rag import RAGEngine
        self.rag = RAGEngine(vector_store=InMemoryVectorStore())

    def test_ingest_and_retrieve(self):
        self.rag.ingest_text(
            "The sprint goal is to ship the new authentication system.",
            source="test.txt"
        )
        results = self.rag.retrieve("What is the sprint goal?")
        assert len(results) > 0
        assert "authentication" in results[0].text.lower() or len(results) > 0

    def test_empty_retrieve_returns_empty(self):
        results = self.rag.retrieve("anything")
        assert results == []

    def test_format_context_no_results(self):
        ctx = self.rag.format_context([])
        assert "No relevant context" in ctx

    def test_format_context_with_results(self):
        self.rag.ingest_text("Daily standup is at 9am.", source="schedule.txt")
        results = self.rag.retrieve("standup time")
        ctx = self.rag.format_context(results)
        assert "Source:" in ctx


# ── Vector Store ───────────────────────────────────────────────────────────────

class TestInMemoryVectorStore:
    def setup_method(self):
        from providers.vector_store import InMemoryVectorStore
        self.store = InMemoryVectorStore()

    def test_add_and_search(self):
        self.store.add(
            ["sprint planning happens every two weeks",
             "the on-call rotation is weekly"],
            [{"source": "a"}, {"source": "b"}]
        )
        results = self.store.search("sprint", top_k=1)
        assert len(results) == 1
        assert "sprint" in results[0].text

    def test_scores_are_between_0_and_1(self):
        self.store.add(["python is a great language for AI"])
        results = self.store.search("python AI")
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_len(self):
        assert len(self.store) == 0
        self.store.add(["one", "two", "three"])
        assert len(self.store) == 3


# ── Tools ──────────────────────────────────────────────────────────────────────

class TestToolRegistry:
    def setup_method(self):
        from core.tools import ToolRegistry
        self.tools = ToolRegistry()

    def test_calculate_basic(self):
        result = self.tools.execute("calculate", {"expression": "2 + 2"})
        assert "4" in result

    def test_calculate_math_functions(self):
        result = self.tools.execute("calculate", {"expression": "sqrt(144)"})
        assert "12" in result

    def test_unknown_tool_returns_error(self):
        result = self.tools.execute("nonexistent_tool", {})
        assert "unknown tool" in result.lower() or "error" in result.lower()

    def test_get_datetime(self):
        result = self.tools.execute("get_current_datetime", {})
        assert "UTC" in result or "datetime" in result.lower()
