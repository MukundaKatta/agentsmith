"""Agent wrappers for testing."""

from agentsmith.agents.base import BaseAgent, AgentAction, AgentResponse
from agentsmith.agents.react import ReActAgent
from agentsmith.agents.tool_use import ToolUseAgent

ALL_AGENTS: dict[str, type[BaseAgent]] = {
    "react": ReActAgent,
    "tool_use": ToolUseAgent,
}

__all__ = [
    "BaseAgent",
    "AgentAction",
    "AgentResponse",
    "ReActAgent",
    "ToolUseAgent",
    "ALL_AGENTS",
]
