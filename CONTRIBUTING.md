# Contributing to AgentDesk

First off — thank you! Whether it's a bug fix, a new provider, or improved docs, every contribution makes AgentDesk better for developers everywhere.

---

## Ways to contribute

- **Bug reports** — open an issue with steps to reproduce
- **New LLM providers** — add a class to `providers/llm.py`
- **New vector stores** — add a class to `providers/vector_store.py`
- **New tools** — add a schema + implementation to `core/tools.py`
- **Better demo data** — add `.txt` files to `data/knowledge_base/`
- **Documentation** — improve `docs/` or inline docstrings

---

## Development setup

```bash
git clone https://github.com/yourusername/agentdesk
cd agentdesk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your API key
pytest tests/ -v       # all tests should pass
```

---

## Adding a new LLM provider

1. Open `providers/llm.py`
2. Subclass `LLMProvider` and implement `complete()`
3. Add a branch in `get_llm_provider()`
4. Add an optional dependency in `pyproject.toml`
5. Write a test in `tests/test_providers.py`
6. Document it in `docs/providers.md`

```python
class MyProvider(LLMProvider):
    def complete(self, messages, system="", tools=None, max_tokens=2048) -> LLMResponse:
        # call your API here
        return LLMResponse(text="...", tool_calls=[], stop_reason="end_turn")
```

---

## Adding a new vector store

1. Open `providers/vector_store.py`
2. Subclass `VectorStore` and implement `add()` and `search()`
3. Add a branch in `get_vector_store()`
4. Write tests

---

## Adding a new tool

1. Open `core/tools.py`
2. Add a JSON schema to `TOOL_DEFINITIONS`
3. Add the Python implementation to `ToolRegistry`
4. Register it in `ToolRegistry.__init__`

```python
# Schema
{
    "name": "send_slack_message",
    "description": "Send a message to a Slack channel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {"type": "string"},
            "message": {"type": "string"}
        },
        "required": ["channel", "message"]
    }
}

# Implementation
def _send_slack(self, channel: str, message: str) -> str:
    # your implementation
    return f"Sent to #{channel}"
```

---

## Code style

- Formatter: `ruff format .`
- Linter: `ruff check .`
- Type hints: use them everywhere, run `mypy .`
- Tests: every new feature needs a test in `tests/`
- Docstrings: module-level + class-level docstrings are required

---

## Pull request checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] No lint errors: `ruff check .`
- [ ] Docstrings added/updated
- [ ] `.env.example` updated if new env vars added
- [ ] `CHANGELOG.md` entry added

---

## Commit message format

```
type: short summary (max 72 chars)

Optional longer body explaining why, not what.
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

Examples:
```
feat: add Mistral LLM provider
fix: handle empty RAG results gracefully
docs: add Pinecone setup guide to docs/providers.md
```

---

## Questions?

Open a [GitHub Discussion](https://github.com/yourusername/agentdesk/discussions) — we're happy to help.
