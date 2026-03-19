# AgentDesk 🤖

**A multi-agent AI productivity workspace — built for developers, designed to be hacked.**

AgentDesk is a working reference implementation of four core AI agent concepts — orchestration, RAG, tool use, and memory — applied to a real developer productivity use case: managing tasks, sprints, and team knowledge.

Clone it, run it in 60 seconds, then swap in your own LLM, vector store, tools, or knowledge base.

---

## Why AgentDesk?

Most multi-agent demos are either toy examples or opaque black boxes. AgentDesk is neither:

- **Every concept is in its own file** — no magic, no frameworks hiding the architecture
- **Swappable backends** — change `LLM_PROVIDER=openai` or `VECTOR_STORE=faiss` in your `.env` and you're done
- **Real demo data included** — sprint planning, standup notes, team handbook — so it's useful out of the box
- **Zero-config default** — works with just an API key and `pip install`

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/yourusername/agentdesk
cd agentdesk

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Install
pip install -e .

# 4. Configure
cp .env.example .env
# → Open .env and add: ANTHROPIC_API_KEY=sk-ant-...
# → Get a free key at https://console.anthropic.com

# 5. Run
python main.py
```

That's it. AgentDesk starts in interactive mode and automatically loads:
- **Sprint data** — tasks, blockers, velocity, standup notes (`data/knowledge_base/sprint_14.txt`)
- **Team handbook** — on-call rota, deployment process, code review rules (`data/knowledge_base/team_handbook.txt`)
- **Productivity tips** — developer workflow best practices (built-in demo text)

> **Python 3.11+ required.** Check with `python --version`.

---

## Try these prompts

These all work out of the box with the included demo data — no setup beyond your API key.

**Sprint & task questions (uses RAG agent):**
```
> What tasks are currently blocked in the sprint?
> Who is on-call this week?
> What is our deployment process?
> What are the code review rules?
```

**Calculations (uses Tool agent):**
```
> Calculate the percentage of story points completed — 8 done out of 26 total
> What is 8 divided by 26 as a percentage?
```

**Combined — look up then act (uses both agents):**
```
> Search the knowledge base for sprint story points, calculate the remaining points, and save to memory
> Summarise the sprint blockers and write a status report to output/report.txt
```

**Productivity tips (uses built-in demo knowledge):**
```
> What time should I do deep work today?
> How should I structure my day as a developer?
```

**Shell commands:**
```
memory   → show all saved long-term facts
clear    → wipe conversation history
quit     → exit
```

Or run a single task from the CLI:

```bash
python main.py --task "Summarise the sprint blockers and calculate remaining story points"
python main.py --task "Calculate 40 * 22.5" --json
```

---

## Architecture

```
agentdesk/
├── main.py                    Entry point & interactive shell
├── config.py                  All settings (env-var driven)
│
├── agents/
│   ├── orchestrator.py        Plans tasks, delegates to specialists
│   ├── rag_agent.py           Answers from your knowledge base
│   └── tool_agent.py          Executes tools in an agentic loop
│
├── core/
│   ├── memory.py              Short-term (sliding window) + long-term (JSON)
│   ├── rag.py                 Chunking + retrieval (delegates to VectorStore)
│   └── tools.py               Tool schemas + Python implementations
│
├── providers/
│   ├── llm.py                 LLM backends: Anthropic (default), OpenAI
│   └── vector_store.py        Vector stores: in-memory (default), FAISS, Pinecone
│
├── data/
│   ├── knowledge_base/        Drop .txt files here for RAG
│   └── demo/                  Example datasets
│
└── tests/
    └── test_core.py           Pytest test suite
```

**How a task flows:**

```
User prompt
    │
    ▼
OrchestratorAgent   ← calls Claude, gets a JSON plan
    │
    ├── RAGAgent       ← retrieves chunks from VectorStore, answers with context
    ├── ToolAgent      ← agentic loop: Claude ↔ Python tools ↔ Claude ...
    └── direct answer  ← for simple questions
    │
    ▼
Synthesise + return final answer
```

---

## Swapping components

All backends are controlled by environment variables. No code changes needed.

### LLM provider

| `LLM_PROVIDER` | Model | Install |
|---|---|---|
| `anthropic` (default) | claude-opus-4-5 | included |
| `openai` | gpt-4o | `pip install agentdesk[openai]` |

Override the model:
```bash
ANTHROPIC_MODEL=claude-sonnet-4-20250514  # faster, cheaper
OPENAI_MODEL=gpt-4o-mini
```

### Vector store

| `VECTOR_STORE` | Description | Install |
|---|---|---|
| `memory` (default) | TF-IDF, zero deps, great for demos | included |
| `faiss` | Fast local ANN search | `pip install agentdesk[faiss]` |
| `pinecone` | Managed cloud search | `pip install agentdesk[pinecone]` |

### Adding your own backend

**New LLM:**
```python
# providers/llm.py
class MyProvider(LLMProvider):
    def complete(self, messages, system="", tools=None, max_tokens=2048) -> LLMResponse:
        # call your API
        return LLMResponse(text="...", tool_calls=[], stop_reason="end_turn")
```

**New vector store:**
```python
# providers/vector_store.py
class MyStore(VectorStore):
    def add(self, texts, metadatas=None): ...
    def search(self, query, top_k=3) -> list[SearchResult]: ...
```

See `CONTRIBUTING.md` for the full guide.

---

## Using your own knowledge base

Drop `.txt` files into `data/knowledge_base/` — they are loaded automatically every time you run `python main.py`, alongside the built-in demo text.

```bash
# Just add your files and run — no flags needed
cp my_runbook.txt data/knowledge_base/
python main.py
```

To load **only** your own files and skip the built-in demo text:

```bash
python main.py --no-demo
```

Good sources to add:
- Sprint backlog exports
- Team runbooks and playbooks
- Architecture decision records (ADRs)
- Meeting notes and standup summaries
- Confluence / Notion exports (save as plain text)

---

## Adding a new tool

Tools are what the agent can *do*, not just know. Adding one takes three steps:

**1. Define the schema** (tells Claude what the tool does and what arguments it takes):
```python
# core/tools.py — TOOL_DEFINITIONS list
{
    "name": "create_jira_ticket",
    "description": "Create a Jira ticket and return its URL.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]}
        },
        "required": ["title"]
    }
}
```

**2. Implement it:**
```python
# core/tools.py — ToolRegistry class
def _create_jira_ticket(self, title: str, priority: str = "medium") -> str:
    # call Jira API here
    return f"Created PROJ-123: {title}"
```

**3. Register it:**
```python
# ToolRegistry.__init__
self._dispatch["create_jira_ticket"] = self._create_jira_ticket
```

---

## Environment variables

Copy `.env.example` to `.env`:

```bash
# Required (pick one)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...          # if using LLM_PROVIDER=openai

# Backend selection
LLM_PROVIDER=anthropic          # anthropic | openai
VECTOR_STORE=memory             # memory | faiss | pinecone
ANTHROPIC_MODEL=claude-opus-4-5
OPENAI_MODEL=gpt-4o

# RAG tuning
KNOWLEDGE_BASE_PATH=data/knowledge_base
CHUNK_SIZE=400
CHUNK_OVERLAP=60
TOP_K_RESULTS=3

# Memory
SHORT_TERM_LIMIT=20
LONG_TERM_DB_PATH=data/memory.json

# Pinecone (if VECTOR_STORE=pinecone)
PINECONE_API_KEY=...
PINECONE_INDEX=agentdesk
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap

- [ ] Async agent execution (parallel subtasks)
- [ ] Web UI (FastAPI + HTMX)
- [ ] GitHub Issues integration tool
- [ ] Notion / Confluence ingestion
- [ ] Streaming responses
- [ ] Google Calendar tool
- [ ] Mistral / Ollama provider (local LLMs)
- [ ] ChromaDB vector store

PRs welcome — see `CONTRIBUTING.md`.

---

## Contributing

We welcome contributions of all sizes. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to:
- Add a new LLM provider
- Add a new vector store
- Add a new tool
- Improve the demo dataset

---

## License

MIT — do whatever you want with it. See [LICENSE](LICENSE).
