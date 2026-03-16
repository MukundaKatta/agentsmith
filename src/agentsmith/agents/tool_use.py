"""Tool-use agent wrappers for Claude and OpenAI.

These agents use the native tool_use / function_calling APIs rather than
text-based parsing, giving the model structured tool invocation capability.
"""

from __future__ import annotations

from typing import Any, Literal

from agentsmith.agents.base import AgentAction, AgentResponse, BaseAgent


def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our simple tool format to Anthropic's tool schema."""
    converted = []
    for tool in tools:
        properties = {}
        for param_name, param_info in tool.get("parameters", {}).items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
        converted.append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": properties,
            },
        })
    return converted


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our simple tool format to OpenAI's function schema."""
    converted = []
    for tool in tools:
        properties = {}
        for param_name, param_info in tool.get("parameters", {}).items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        })
    return converted


class ToolUseAgent(BaseAgent):
    """Agent wrapping Claude or OpenAI with native tool_use / function_calling.

    Provides sandboxed tools (file_read, file_write, web_search, execute_code)
    whose calls are all logged and monitored by the test environment.
    """

    name = "tool_use"

    def __init__(
        self,
        provider: Literal["anthropic", "openai"] = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self._provider = provider
        self.provider = provider  # type: ignore[assignment]
        self.model = model or ("claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o")
        self._api_key = api_key
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self._provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
            except ImportError:
                raise RuntimeError("anthropic package required. Install with: pip install anthropic")
        elif self._provider == "openai":
            try:
                import openai

                self._client = openai.OpenAI(api_key=self._api_key) if self._api_key else openai.OpenAI()
            except ImportError:
                raise RuntimeError("openai package required. Install with: pip install openai")
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

        return self._client

    async def _step_anthropic(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        client = self._get_client()
        api_tools = _convert_tools_to_anthropic(tools)

        api_messages = []
        for msg in messages:
            role = msg["role"]
            if role in ("user", "assistant"):
                api_messages.append({"role": role, "content": msg["content"]})

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=api_messages,
            tools=api_tools,
        )

        text_parts: list[str] = []
        tool_calls: list[AgentAction] = []
        raw_tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                action = AgentAction(name=block.name, arguments=block.input)
                tool_calls.append(action)
                raw_tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return AgentResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            tool_calls_raw=raw_tool_calls,
            stop=response.stop_reason == "end_turn" and len(tool_calls) == 0,
            metadata={"model": self.model, "stop_reason": response.stop_reason},
        )

    async def _step_openai(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        import json

        client = self._get_client()
        api_tools = _convert_tools_to_openai(tools)

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"]
            if role in ("user", "assistant"):
                api_messages.append({"role": role, "content": msg["content"]})

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_messages,
            tools=api_tools if api_tools else None,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls: list[AgentAction] = []
        raw_tool_calls: list[dict[str, Any]] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}
                action = AgentAction(name=tc.function.name, arguments=arguments)
                tool_calls.append(action)
                raw_tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": arguments,
                })

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            tool_calls_raw=raw_tool_calls,
            stop=choice.finish_reason == "stop" and len(tool_calls) == 0,
            metadata={"model": self.model, "finish_reason": choice.finish_reason},
        )

    async def step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        if self._provider == "anthropic":
            return await self._step_anthropic(system_prompt, messages, tools)
        elif self._provider == "openai":
            return await self._step_openai(system_prompt, messages, tools)
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

    def reset(self) -> None:
        pass
