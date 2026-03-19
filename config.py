"""
config.py — AgentDesk Configuration
──────────────────────────────────────────────────────────────────────
All settings can be overridden via environment variables.
Copy .env.example → .env to get started.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Backend ────────────────────────────────────────────────────────────────
# LLM_PROVIDER=anthropic  (default) | openai
# ANTHROPIC_MODEL=claude-opus-4-5   (default)
# OPENAI_MODEL=gpt-4o               (if using OpenAI)
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL  = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
MODEL            = ANTHROPIC_MODEL  # alias used by agents
MAX_TOKENS       = int(os.getenv("MAX_TOKENS", "2048"))

# ── Vector Store ───────────────────────────────────────────────────────────────
# VECTOR_STORE=memory   (default, zero deps)
# VECTOR_STORE=faiss    (pip install agentdesk[faiss])
# VECTOR_STORE=pinecone (pip install agentdesk[pinecone] + set PINECONE_API_KEY)
VECTOR_STORE         = os.getenv("VECTOR_STORE", "memory")
KNOWLEDGE_BASE_PATH  = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base")
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "60"))
TOP_K_RESULTS        = int(os.getenv("TOP_K_RESULTS", "3"))

# ── Memory ─────────────────────────────────────────────────────────────────────
SHORT_TERM_LIMIT  = int(os.getenv("SHORT_TERM_LIMIT", "20"))
LONG_TERM_DB_PATH = os.getenv("LONG_TERM_DB_PATH", "data/memory.json")

# ── API Keys (validated at startup) ────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

def validate_config() -> None:
    """Raise a clear error if required credentials are missing."""
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "  → Copy .env.example to .env and add your key.\n"
            "  → Get a key at https://console.anthropic.com"
        )
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "  → Add OPENAI_API_KEY=sk-... to your .env file."
        )
