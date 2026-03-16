"""CLI entry point for SmithTest."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from agentsmith import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="smithtest")
def cli() -> None:
    """SmithTest -- Detect emergent self-preservation and self-replication
    drives in autonomous AI agents."""
    pass


@cli.command()
@click.option(
    "--agent",
    type=click.Choice(["react", "tool_use"], case_sensitive=False),
    default="tool_use",
    help="Agent type to test.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai"], case_sensitive=False),
    default="anthropic",
    help="LLM provider for the agent.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (defaults to provider's default).",
)
@click.option(
    "--environment",
    type=click.Choice(["all", "shutdown", "replication", "resource", "goal_modification"], case_sensitive=False),
    default="all",
    help="Which test environment(s) to run.",
)
@click.option(
    "--max-steps",
    type=int,
    default=10,
    help="Maximum interaction steps per environment.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="results.json",
    help="Output path for JSON results.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed output during the run.",
)
def run(
    agent: str,
    provider: str,
    model: str | None,
    environment: str,
    max_steps: int,
    output: str,
    verbose: bool,
) -> None:
    """Run SmithTest against an agent."""
    from agentsmith.agents import ALL_AGENTS
    from agentsmith.environments import ALL_ENVIRONMENTS
    from agentsmith.report import ReportGenerator
    from agentsmith.runner import SmithTest

    # Create agent
    if agent == "react":
        from agentsmith.agents.react import ReActAgent

        test_agent = ReActAgent(model=model or "claude-sonnet-4-20250514")
    else:
        from agentsmith.agents.tool_use import ToolUseAgent

        test_agent = ToolUseAgent(provider=provider, model=model)  # type: ignore[arg-type]

    # Create environments
    if environment == "all":
        envs = [env_cls() for env_cls in ALL_ENVIRONMENTS.values()]
    else:
        env_cls = ALL_ENVIRONMENTS.get(environment)
        if env_cls is None:
            console.print(f"[red]Unknown environment: {environment}[/red]")
            sys.exit(1)
        envs = [env_cls()]

    # Build and run SmithTest
    smith = SmithTest(max_steps=max_steps)
    smith.add_agent(test_agent)
    for env in envs:
        smith.add_environment(env)

    console.print(f"\n[bold]SmithTest v{__version__}[/bold]")
    console.print(f"Agent: [cyan]{agent}[/cyan] ({provider})")
    console.print(f"Environments: [cyan]{environment}[/cyan]")
    console.print(f"Max steps: [cyan]{max_steps}[/cyan]")
    console.print()

    with console.status("[bold green]Running tests..."):
        results = asyncio.run(smith.run())

    # Save results
    SmithTest.save_results(results, output)
    console.print(f"[green]Results saved to {output}[/green]\n")

    # Display report
    reporter = ReportGenerator(console=console)
    if verbose:
        reporter.print_full_report(results)
    else:
        reporter.print_summary(results)


@cli.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSON results file.",
)
@click.option(
    "--detailed",
    is_flag=True,
    default=False,
    help="Show detailed report with evidence.",
)
def report(input_path: str, detailed: bool) -> None:
    """Generate a report from saved results."""
    from agentsmith.report import ReportGenerator
    from agentsmith.runner import SmithTest

    results = SmithTest.load_results(input_path)
    reporter = ReportGenerator(console=console)

    if detailed:
        reporter.print_full_report(results)
    else:
        reporter.print_summary(results)


if __name__ == "__main__":
    cli()
