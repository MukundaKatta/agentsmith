"""ReAct (Reason + Act) agent wrapper.

Uses a text-based reasoning loop where the model produces Thought / Action /
Observation traces.  Tool calls are parsed from the model's text output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentsmith.agents.base import AgentAction, AgentResponse, BaseAgent

# Default ReAct prompt injected before the environment's system prompt.
REACT_PREAMBLE = """\
You follow the ReAct framework. For every step, output exactly one of:

Thought: <your reasoning>
Action: <tool_name>({"arg": "value", ...})
Answer: <final answer when done>

Do NOT output anything else outside these three formats.\
"""


def _parse_react_output(text: str) -> tuple[str, list[AgentAction], bool]:
    """Parse a ReAct-formatted response into components.

    Returns:
        Tuple of (thought_text, tool_calls, is_final_answer).
    """
    tool_calls: list[AgentAction] = []
    is_final = False

    # Extract Action lines
    action_pattern = re.compile(r"Action:\s*(\w+)\((\{.*?\})\)", re.DOTALL)
    for match in action_pattern.finditer(text):
        tool_name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            arguments = {"raw": match.group(2)}
        tool_calls.append(AgentAction(name=tool_name, arguments=arguments))

    # Check for final answer
    if re.search(r"Answer:", text):
        is_final = True

    return text, tool_calls, is_final


class ReActAgent(BaseAgent):
    """ReAct agent that wraps an Anthropic Claude model.

    Uses text-based Thought/Action/Observation traces rather than native
    tool_use mode.  This tests whether concerning behaviors emerge even
    in a text-only reasoning framework.
    """

    name = "react"
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
            except ImportError:
                raise RuntimeError("anthropic package is required for ReActAgent. Install with: pip install anthropic")
        return self._client

    def _format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions as text for the ReAct prompt."""
        lines = ["Available tools:"]
        for tool in tools:
            params = tool.get("parameters", {})
            param_desc = ", ".join(f'{k}: {v.get("type", "any")}' for k, v in params.items())
            lines.append(f"  - {tool['name']}({param_desc}): {tool.get('description', '')}")
        return "\n".join(lines)

    async def step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        client = self._get_client()
        tool_text = self._format_tools_for_prompt(tools)
        full_system = f"{REACT_PREAMBLE}\n\n{tool_text}\n\n{system_prompt}"

        # Convert messages to Anthropic format
        api_messages = []
        for msg in messages:
            role = msg["role"]
            if role in ("user", "assistant"):
                api_messages.append({"role": role, "content": msg["content"]})
            elif role == "tool":
                api_messages.append({"role": "user", "content": f"Observation: {msg['content']}"})

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=full_system,
            messages=api_messages,
        )

        text = response.content[0].text if response.content else ""
        thought, tool_calls, is_final = _parse_react_output(text)

        return AgentResponse(
            content=thought,
            tool_calls=tool_calls,
            tool_calls_raw=[tc.model_dump() for tc in tool_calls],
            stop=is_final,
            metadata={"model": self.model, "stop_reason": response.stop_reason},
        )

    def reset(self) -> None:
        pass
