#!/usr/bin/env python3
"""Example: Run a SmithTest evaluation against a Claude agent.

Usage:
    python examples/run_smithtest.py

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - pip install -e .
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console

from agentsmith.agents.tool_use import ToolUseAgent
from agentsmith.environments.shutdown import ShutdownEnvironment
from agentsmith.environments.replication import ReplicationEnvironment
from agentsmith.environments.resource import ResourceEnvironment
from agentsmith.environments.goal_mod import GoalModificationEnvironment
from agentsmith.report import ReportGenerator
from agentsmith.runner import SmithTest


console = Console()


async def main() -> None:
    console.print("[bold]AgentSmith — SmithTest Example[/bold]\n")

    # Create the agent under test
    agent = ToolUseAgent(provider="anthropic", model="claude-sonnet-4-20250514")

    # Create the test suite
    smith = SmithTest(max_steps=8)
    smith.add_agent(agent)

    # Add all test environments
    smith.add_environment(ShutdownEnvironment())
    smith.add_environment(ReplicationEnvironment())
    smith.add_environment(ResourceEnvironment())
    smith.add_environment(GoalModificationEnvironment())

    # Run all tests
    console.print("Running 4 environments against Claude tool_use agent...\n")
    with console.status("[bold green]Running tests..."):
        results = await smith.run()

    # Save results
    output_path = Path("results.json")
    SmithTest.save_results(results, output_path)
    console.print(f"[green]Results saved to {output_path}[/green]\n")

    # Display report
    reporter = ReportGenerator(console=console)
    reporter.print_full_report(results)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
