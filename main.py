"""
main.py — AgentDesk Entry Point
──────────────────────────────────────────────────────────────────────
Usage:
    python main.py                          # interactive mode
    python main.py --task "your task"       # single task, then exit
    python main.py --task "..." --json      # output raw JSON result
    python main.py --no-demo               # load only data/knowledge_base/
"""
import argparse
import json as _json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from config import validate_config, KNOWLEDGE_BASE_PATH, VECTOR_STORE, LLM_PROVIDER
from core.memory import MemoryManager
from core.rag import RAGEngine
from core.tools import ToolRegistry
from agents.rag_agent import RAGAgent
from agents.tool_agent import ToolAgent
from agents.orchestrator import OrchestratorAgent
from providers.llm import get_llm_provider
from providers.vector_store import get_vector_store

console = Console()

DEMO_KNOWLEDGE = """
Personal Productivity Tips for Developers
==========================================
Use time-blocking: assign specific hours to deep work, meetings, and email.
The Pomodoro technique (25 min work, 5 min break) improves focus for coding tasks.
Capture all tasks in a trusted system so nothing lives in your head.
Batch similar tasks together (e.g., all code reviews after standup).
Write tomorrow's top 3 priorities at end of each day.
Automate repetitive tasks before they steal more than 30 minutes per week.
Do the hardest task first thing in the morning (eat the frog).

Developer Workflow Best Practices
===================================
Commit early, commit often — small commits are easier to review and revert.
Write the test first if you are unsure of the design (TDD as design tool).
Read error messages carefully — they usually tell you exactly what went wrong.
Use descriptive branch names: feature/add-rate-limiting not my-branch.
Leave code better than you found it (Boy Scout rule).
Document the why, not the what — code shows what, comments explain why.

Meeting & Communication Guidelines
=====================================
Default to async: write it down before scheduling a meeting.
Every meeting needs an agenda and a named facilitator.
Decisions made in meetings must be recorded in writing within 24 hours.
Use threads in Slack to keep channels readable.
"""


def build_system(demo_mode: bool = True) -> OrchestratorAgent:
    """Construct and wire all agents."""
    console.print(Rule("[bold]AgentDesk — Starting up[/bold]"))

    # Show active configuration
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("LLM provider", f"[cyan]{LLM_PROVIDER}[/cyan]")
    t.add_row("Vector store", f"[cyan]{VECTOR_STORE}[/cyan]")
    t.add_row("Knowledge base", f"[cyan]{'demo data' if demo_mode else KNOWLEDGE_BASE_PATH}[/cyan]")
    console.print(t)
    console.print()

    memory = MemoryManager()

    llm = get_llm_provider()
    vector_store = get_vector_store()

    rag_engine = RAGEngine(vector_store=vector_store)
    if demo_mode:
        n = rag_engine.ingest_text(DEMO_KNOWLEDGE, source="demo_knowledge.txt")
        console.print(f"  [green]✓[/green] Demo knowledge indexed ({n} chunks)")
    else:
        n = rag_engine.ingest_directory(KNOWLEDGE_BASE_PATH)
        console.print(f"  [green]✓[/green] Knowledge base indexed ({n} chunks from {KNOWLEDGE_BASE_PATH})")

    tool_registry = ToolRegistry(rag_engine=rag_engine, memory_manager=memory)
    rag_agent = RAGAgent(rag_engine=rag_engine, memory=memory)
    tool_agent = ToolAgent(tool_registry=tool_registry, memory=memory)
    orchestrator = OrchestratorAgent(
        rag_agent=rag_agent,
        tool_agent=tool_agent,
        memory=memory,
    )

    console.print(f"  [green]✓[/green] All agents ready\n")
    return orchestrator


def run_interactive(orchestrator: OrchestratorAgent) -> None:
    console.print(Panel(
        "[bold green]AgentDesk is ready[/bold green]\n\n"
        "Try: [italic]\"What tasks are blocked this sprint?\"[/italic]\n"
        "Try: [italic]\"Calculate how many story points are left and save to memory\"[/italic]\n"
        "Try: [italic]\"What is our on-call process?\"[/italic]\n\n"
        "Commands: [bold]memory[/bold] · [bold]clear[/bold] · [bold]help[/bold] · [bold]quit[/bold]",
        title="🤖 AgentDesk",
        border_style="bright_blue",
    ))

    while True:
        try:
            task = console.input("\n[bold blue]>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not task:
            continue

        match task.lower():
            case "quit" | "exit" | "q":
                break
            case "clear":
                orchestrator.memory.clear_short_term()
                console.print("[dim]Short-term memory cleared.[/dim]")
                continue
            case "memory":
                facts = orchestrator.memory.list_facts()
                if facts:
                    console.print(Panel(
                        "\n".join(f"[bold]{k}[/bold]: {v}" for k, v in facts.items()),
                        title="Long-term memory"
                    ))
                else:
                    console.print("[dim]No facts in long-term memory yet.[/dim]")
                continue
            case "help":
                console.print(Panel(
                    "Ask any productivity or task-related question.\n\n"
                    "[bold]memory[/bold]  — show persisted facts\n"
                    "[bold]clear[/bold]   — clear conversation history\n"
                    "[bold]quit[/bold]    — exit",
                    title="Help"
                ))
                continue

        orchestrator.memory.add_message("user", task)
        result = orchestrator.run(task)
        console.print(f"\n[bold green]AgentDesk:[/bold green]")
        console.print(Panel(result, border_style="green"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentDesk — Multi-agent AI productivity workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --task "What tasks are blocked?"
  python main.py --task "Calculate 40 * 22.5" --json
  python main.py --no-demo   # use data/knowledge_base/ instead of demo data
        """,
    )
    parser.add_argument("--task", type=str, help="Run a single task and exit")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--no-demo", action="store_true", help="Use data/knowledge_base/ instead of demo data")
    args = parser.parse_args()

    try:
        validate_config()
    except EnvironmentError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    orchestrator = build_system(demo_mode=not args.no_demo)

    if args.task:
        result = orchestrator.run(args.task)
        if args.json:
            print(_json.dumps({"result": result}))
        else:
            console.print(Panel(result, title="Result", border_style="green"))
    else:
        run_interactive(orchestrator)


if __name__ == "__main__":
    main()
