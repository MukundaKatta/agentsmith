"""Base agent interface for SmithTest."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """A single tool call requested by the agent."""

    name: str = Field(description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool call arguments")


class AgentResponse(BaseModel):
    """Structured response from an agent step."""

    content: str = Field(default="", description="Text content of the response")
    tool_calls: list[AgentAction] = Field(default_factory=list, description="Tool calls requested")
    tool_calls_raw: list[dict[str, Any]] = Field(default_factory=list, description="Raw tool call data for logging")
    stop: bool = Field(default=False, description="Whether the agent considers itself done")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseAgent(ABC):
    """Abstract base class for agents under test.

    Agents wrap an LLM provider (Claude, OpenAI, etc.) and expose a
    uniform ``step`` interface that the test environments call in a loop.
    """

    name: str = "base"
    provider: str = "unknown"

    @abstractmethod
    async def step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Execute a single reasoning step.

        Args:
            system_prompt: The system-level instruction for this environment.
            messages: Conversation history (user / assistant / tool turns).
            tools: Available tool definitions for this environment.

        Returns:
            AgentResponse containing the agent's text, tool calls, and stop signal.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state between test runs."""
        pass
