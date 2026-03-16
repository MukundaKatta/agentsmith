"""Report generation — produces human-readable output from SmithTest results."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentsmith.environments.base import Severity
from agentsmith.runner import SmithTestResults, TestRun


SEVERITY_COLORS = {
    Severity.NONE: "green",
    Severity.LOW: "cyan",
    Severity.MEDIUM: "yellow",
    Severity.HIGH: "red",
    Severity.CRITICAL: "bold red",
}

RISK_COLORS = {
    "NONE": "green",
    "LOW": "cyan",
    "MEDIUM": "yellow",
    "HIGH": "red",
    "CRITICAL": "bold red",
}


def _risk_color(score: float) -> str:
    if score >= 0.8:
        return "bold red"
    elif score >= 0.6:
        return "red"
    elif score >= 0.4:
        return "yellow"
    elif score >= 0.2:
        return "cyan"
    return "green"


def _score_bar(score: float, width: int = 20) -> str:
    """Generate a text progress bar for a 0-1 score."""
    filled = int(score * width)
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}] {score:.1%}"


class ReportGenerator:
    """Generates formatted reports from SmithTest results using Rich."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_summary(self, results: SmithTestResults) -> None:
        """Print a high-level summary table."""
        self.console.print()

        # Header panel
        risk_color = _risk_color(results.aggregate_risk)
        header = Text()
        header.append("AGENTSMITH TEST RESULTS\n", style="bold white")
        header.append(f"Aggregate Risk: ", style="white")
        header.append(f"{results.aggregate_risk:.1%}", style=risk_color)
        header.append(f"\nAssessment: ", style="white")
        header.append(results.risk_assessment, style=risk_color)
        self.console.print(Panel(header, border_style=risk_color, padding=(1, 2)))

        # Summary table
        table = Table(title="Test Results Summary", show_lines=True)
        table.add_column("Agent", style="bold")
        table.add_column("Environment", style="bold")
        table.add_column("Behavior Detected", justify="center")
        table.add_column("Severity", justify="center")
        table.add_column("Risk Score", justify="center")
        table.add_column("Evidence", justify="center")

        for run in results.runs:
            env_result = run.environment_result
            severity_style = SEVERITY_COLORS.get(env_result.severity, "white")
            risk_style = _risk_color(run.risk_score)

            detected_text = Text("YES", style="red") if env_result.behavior_detected else Text("NO", style="green")

            table.add_row(
                run.agent_name,
                run.environment_name,
                detected_text,
                Text(env_result.severity.value.upper(), style=severity_style),
                Text(f"{run.risk_score:.1%}", style=risk_style),
                str(len(env_result.evidence)),
            )

        self.console.print(table)

    def print_detailed(self, results: SmithTestResults) -> None:
        """Print detailed results for each test run."""
        self.console.print()

        for run in results.runs:
            self._print_run_detail(run)

    def _print_run_detail(self, run: TestRun) -> None:
        """Print detailed results for a single test run."""
        env_result = run.environment_result
        detectors = run.detector_results

        # Run header
        risk_style = _risk_color(run.risk_score)
        title = f"{run.agent_name} x {run.environment_name}"
        self.console.print(Panel(
            Text(f"Risk Score: {run.risk_score:.1%}", style=risk_style),
            title=title,
            border_style=risk_style,
        ))

        # Detector scores
        det_table = Table(title="Detector Analysis", show_lines=True, width=80)
        det_table.add_column("Detector", style="bold", width=15)
        det_table.add_column("Dimension", width=20)
        det_table.add_column("Score", justify="center", width=30)
        det_table.add_column("Level", justify="center", width=10)

        # Survival
        det_table.add_row(
            "Survival",
            "Frequency",
            _score_bar(detectors.survival.frequency),
            "",
        )
        det_table.add_row("", "Sophistication", _score_bar(detectors.survival.sophistication), "")
        det_table.add_row("", "Subtlety", _score_bar(detectors.survival.subtlety), "")
        det_table.add_row(
            "",
            "Overall",
            _score_bar(detectors.survival.overall),
            Text(detectors.survival.risk_level, style=RISK_COLORS.get(detectors.survival.risk_level, "white")),
        )

        # Replication
        det_table.add_row("Replication", "Intent", _score_bar(detectors.replication.intent), "")
        det_table.add_row("", "Capability", _score_bar(detectors.replication.capability), "")
        det_table.add_row("", "Execution", _score_bar(detectors.replication.execution), "")
        det_table.add_row(
            "",
            "Overall",
            _score_bar(detectors.replication.overall),
            Text(detectors.replication.risk_level, style=RISK_COLORS.get(detectors.replication.risk_level, "white")),
        )

        # Deception
        det_table.add_row("Deception", "Concealment", _score_bar(detectors.deception.concealment), "")
        det_table.add_row("", "Misdirection", _score_bar(detectors.deception.misdirection), "")
        det_table.add_row("", "Manipulation", _score_bar(detectors.deception.manipulation), "")
        det_table.add_row(
            "",
            "Overall",
            _score_bar(detectors.deception.overall),
            Text(detectors.deception.risk_level, style=RISK_COLORS.get(detectors.deception.risk_level, "white")),
        )

        self.console.print(det_table)

        # Evidence table
        if env_result.evidence:
            ev_table = Table(title="Evidence", show_lines=True)
            ev_table.add_column("#", width=4, justify="right")
            ev_table.add_column("Category", width=22)
            ev_table.add_column("Description", width=40)
            ev_table.add_column("Severity", width=10, justify="center")

            for i, ev in enumerate(env_result.evidence, 1):
                sev_style = SEVERITY_COLORS.get(ev.severity, "white")
                ev_table.add_row(
                    str(i),
                    ev.category,
                    ev.description,
                    Text(ev.severity.value.upper(), style=sev_style),
                )

            self.console.print(ev_table)

        self.console.print()

    def print_full_report(self, results: SmithTestResults) -> None:
        """Print the complete report: summary followed by details."""
        self.print_summary(results)
        self.print_detailed(results)
