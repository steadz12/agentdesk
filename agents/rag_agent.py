"""
agents/rag_agent.py — RAG Agent
─────────────────────────────────────────────────────
Concept: RAG (Retrieval-Augmented Generation)

This agent:
  1. Takes an instruction / question
  2. Retrieves relevant chunks from the knowledge base
  3. Injects them into its system prompt
  4. Calls Claude with the augmented context to answer

This pattern prevents hallucination by grounding answers in real documents.
"""
import anthropic
from core.rag import RAGEngine
from core.memory import MemoryManager
from config import MODEL, MAX_TOKENS


class RAGAgent:
    """
    Retrieval-Augmented Generation agent.

    Usage:
        agent = RAGAgent(rag_engine, memory)
        answer = agent.run("What is our refund policy?")
    """

    SYSTEM_TEMPLATE = """You are a knowledgeable assistant. Answer the user's question
using ONLY the context provided below. If the context does not contain enough information,
say so clearly — do not hallucinate.

=== RETRIEVED CONTEXT ===
{context}
=== END CONTEXT ===

Recent conversation:
{history}"""

    def __init__(self, rag_engine: RAGEngine, memory: MemoryManager):
        self.rag = rag_engine
        self.memory = memory
        self.client = anthropic.Anthropic()

    def run(self, instruction: str) -> str:
        # 1. Retrieve relevant chunks
        chunks = self.rag.retrieve(instruction)
        context = self.rag.format_context(chunks)

        # 2. Build system prompt with retrieved context + memory
        system = self.SYSTEM_TEMPLATE.format(
            context=context,
            history=self.memory.get_context_summary()
        )

        # 3. Call Claude with augmented context
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": instruction}]
        )

        answer = resp.content[0].text.strip()

        # 4. Store in short-term memory
        self.memory.add_message("user", instruction, agent="user")
        self.memory.add_message("assistant", answer, agent="rag_agent")

        return answer
