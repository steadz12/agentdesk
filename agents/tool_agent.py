"""
agents/tool_agent.py — Tool-Use Agent
─────────────────────────────────────────────────────
Concept: Tool Use / Function Calling

This agent implements the standard agentic tool-use loop:
  ┌─────────────────────────────────┐
  │  User instruction               │
  │         │                       │
  │         ▼                       │
  │   Claude (with tools)           │
  │         │                       │
  │   ┌─────┴─────┐                 │
  │   │tool_use?  │                 │
  │   │   YES     │   NO            │
  │   ▼           ▼                 │
  │ execute    return text          │
  │ tool(s)                         │
  │   │                             │
  │   └──► feed result back ──────► │
  └─────────────────────────────────┘

Claude may call multiple tools across multiple turns before producing
a final text response. The loop continues until stop_reason == "end_turn".
"""
import anthropic
from rich.console import Console

from core.tools import ToolRegistry, TOOL_DEFINITIONS
from core.memory import MemoryManager
from config import MODEL, MAX_TOKENS

console = Console()

MAX_TOOL_ROUNDS = 10   # Safety limit: prevent infinite loops


class ToolAgent:
    """
    Agentic loop agent that can call tools to complete a task.

    Usage:
        agent = ToolAgent(tool_registry, memory)
        result = agent.run("What is 1234 * 5678, and save the result to memory as 'calc_result'")
    """

    SYSTEM_PROMPT = """You are a capable AI assistant with access to tools.
Use tools whenever they help you complete the task accurately.
You can call multiple tools in sequence. Always provide a clear final answer once done.

Available tools: calculate, get_current_datetime, read_file, write_file,
                 search_knowledge_base, save_memory"""

    def __init__(self, tool_registry: ToolRegistry, memory: MemoryManager):
        self.tools = tool_registry
        self.memory = memory
        self.client = anthropic.Anthropic()

    def run(self, instruction: str) -> str:
        """Execute the agentic tool-use loop."""
        messages = self.memory.get_short_term()
        messages.append({"role": "user", "content": instruction})

        rounds = 0
        while rounds < MAX_TOOL_ROUNDS:
            rounds += 1

            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self.SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages
            )

            # Append Claude's full response to the conversation
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                # Extract final text
                final_text = next(
                    (block.text for block in resp.content if hasattr(block, "text")),
                    "Task completed."
                )
                self.memory.add_message("user", instruction, agent="user")
                self.memory.add_message("assistant", final_text, agent="tool_agent")
                return final_text

            if resp.stop_reason == "tool_use":
                # Execute all tool calls in this turn
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        console.print(f"  [magenta]🔧 Tool:[/magenta] {block.name}({block.input})")
                        result = self.tools.execute(block.name, block.input)
                        console.print(f"  [dim]   → {result[:150]}[/dim]")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                # Feed results back to Claude
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason — bail out
                break

        return "Agent reached maximum tool rounds without a final answer."
