"""
agents/orchestrator.py — Orchestrator Agent
─────────────────────────────────────────────────────
Concept: Agent Orchestration & Planning

The Orchestrator is the "brain" of the system. It:
  1. Receives a high-level task from the user
  2. Produces a structured plan (list of subtasks)
  3. Delegates each subtask to the appropriate specialist agent
  4. Collects results and synthesises a final response

Orchestration pattern: Sequential (plan → execute → synthesise)
For parallel execution, use asyncio.gather() over agent.run() calls.
"""
import json
import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from core.memory import MemoryManager
from core.rag import RAGEngine
from core.tools import ToolRegistry, TOOL_DEFINITIONS
from config import MODEL, MAX_TOKENS

console = Console()


class OrchestratorAgent:
    """
    Top-level agent that plans and coordinates all other agents.

    Architecture:
        User Task
            │
            ▼
        OrchestratorAgent.run()
            │ generates plan
            ├──► RAGAgent (if knowledge retrieval needed)
            ├──► ToolAgent (if tool execution needed)
            └──► synthesise() → final answer
    """

    SYSTEM_PROMPT = """You are an Orchestrator AI agent. Your job is to:
1. Analyse the user's task
2. Break it into clear subtasks
3. Decide which agent handles each subtask:
   - RAG_AGENT: for questions requiring knowledge base lookup
   - TOOL_AGENT: for tasks requiring tool execution (math, files, datetime, saving memory)
   - SELF: for tasks you can answer directly

Respond ONLY with a valid JSON plan in this format:
{
  "goal": "one-sentence goal",
  "subtasks": [
    {"id": 1, "agent": "RAG_AGENT|TOOL_AGENT|SELF", "instruction": "what to do", "depends_on": []},
    ...
  ],
  "synthesis_instruction": "how to combine results into a final answer"
}"""

    def __init__(
        self,
        rag_agent,
        tool_agent,
        memory: MemoryManager,
    ):
        self.rag_agent = rag_agent
        self.tool_agent = tool_agent
        self.memory = memory
        self.client = anthropic.Anthropic()

    def run(self, task: str) -> str:
        console.print(Panel(f"[bold cyan]ORCHESTRATOR[/bold cyan]\nPlanning task: {task}"))

        # 1. Inject context from memory into planning
        long_term_facts = self.memory.list_facts()
        context = f"\nKnown facts: {json.dumps(long_term_facts)}" if long_term_facts else ""

        # 2. Generate plan
        plan = self._plan(task + context)
        console.print(f"[yellow]Plan:[/yellow] {json.dumps(plan, indent=2)}")

        # 3. Execute subtasks in order
        results: dict[int, str] = {}
        for subtask in plan.get("subtasks", []):
            sid = subtask["id"]
            agent_name = subtask["agent"]
            instruction = subtask["instruction"]

            # Inject results of dependencies
            dep_context = ""
            for dep in subtask.get("depends_on", []):
                if dep in results:
                    dep_context += f"\nPrevious result (step {dep}): {results[dep]}"

            full_instruction = instruction + dep_context

            console.print(f"\n[green]→ Subtask {sid} [{agent_name}]:[/green] {instruction}")

            if agent_name == "RAG_AGENT":
                result = self.rag_agent.run(full_instruction)
            elif agent_name == "TOOL_AGENT":
                result = self.tool_agent.run(full_instruction)
            else:  # SELF
                result = self._answer_directly(full_instruction)

            results[sid] = result
            console.print(f"  [dim]Result: {result[:200]}{'...' if len(result) > 200 else ''}[/dim]")

        # 4. Synthesise
        final = self._synthesise(task, plan, results)
        self.memory.add_message("assistant", final, agent="orchestrator")
        return final

    # ── Internal ───────────────────────────────────────────────────────────────

    def _plan(self, task: str) -> dict:
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task}]
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: treat as single SELF subtask
            return {
                "goal": task,
                "subtasks": [{"id": 1, "agent": "SELF", "instruction": task, "depends_on": []}],
                "synthesis_instruction": "Return the result directly."
            }

    def _answer_directly(self, instruction: str) -> str:
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": instruction}]
        )
        return resp.content[0].text.strip()

    def _synthesise(self, original_task: str, plan: dict, results: dict) -> str:
        results_text = "\n".join(
            f"Step {k}: {v}" for k, v in results.items()
        )
        synthesis_prompt = f"""Original task: {original_task}

Subtask results:
{results_text}

Synthesis instruction: {plan.get('synthesis_instruction', 'Combine results into a final answer.')}

Please produce a clear, concise final answer for the user."""

        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        return resp.content[0].text.strip()
