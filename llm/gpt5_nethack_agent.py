from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Type

from enum import Enum
from pydantic import BaseModel
from openai import OpenAI, pydantic_function_tool

# -----------------------
# Tool registry using pydantic_function_tool
# -----------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._funcs: Dict[str, Callable[..., Any]] = {}
        self._specs: List[Dict[str, Any]] = {}
        self._specs = []
        self._model_by_name: Dict[str, Type[BaseModel]] = {}

    def register_tool(
        self,
        model_cls: Type[BaseModel],
        func: Callable[..., Any],
        description: Optional[str] = None,
    ) -> None:
        spec = pydantic_function_tool(model_cls)
        # Optionally override description shown to the model
        if description:
            spec["function"]["description"] = description
        name = spec["function"]["name"]
        self._funcs[name] = func
        self._model_by_name[name] = model_cls
        self._specs.append(spec)

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return list(self._specs)

    def call_local(self, name: str, arguments: Dict[str, Any]) -> str:
        # Validate via Pydantic, then call
        model = self._model_by_name[name]
        parsed = model(**(arguments or {}))
        result = self._funcs[name](**parsed.model_dump())
        return result if isinstance(result, str) else json.dumps(result)


class StepKind(Enum):
    THINKING = 0
    FUNCTION_CALL = 1
    JSON = 2
    MESSAGE = 3
    OUTPUT_DELTA = 4

class LLMStep:
    def __init__(self, kind: StepKind, content: Any) -> None:
        self.kind = kind
        self.content = content

class FunctionCallStep(LLMStep):
    def __init__(self, function_name: str, arguments: Dict[str, Any], result : str) -> None:
        super().__init__(StepKind.FUNCTION_CALL, content=result)
        self.function_name = function_name
        self.arguments = arguments

class ThinkingStep(LLMStep):
    def __init__(self, thought_process: str) -> None:
        super().__init__(StepKind.THINKING, content=thought_process)
        self.thought_process = thought_process

class JsonStep(LLMStep):
    def __init__(self, result: str) -> None:
        super().__init__(StepKind.JSON, content=result)

class MessageStep(LLMStep):
    def __init__(self, message: str) -> None:
        super().__init__(StepKind.MESSAGE, content=message)
        self.message = message

class OutputDelta(LLMStep):
    def __init__(self, delta: str, thinking: bool) -> None:
        super().__init__(StepKind.OUTPUT_DELTA, content=delta)
        self.thinking = thinking

class GPT5NanoAgent:
    def __init__(
        self,
        instructions_text: str,
        *,
        model: str = "gpt-5-nano",          # or "gpt-5-thinking-nano" if available
        reasoning_effort: str = "medium",   # "minimal" | "medium" | "high"
        client: Optional[OpenAI] = None,
    ) -> None:
        self.client = client or OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._system_prompt = instructions_text  # keep identical for prompt caching
        self._tools = ToolRegistry()

    def register_tool(self, model_cls: Type[BaseModel], func: Callable[..., Any], description: Optional[str] = None):
        self._tools.register_tool(model_cls, func, description)

    def chat(self, text, is_done, *, output_callback, max_tool_rounds: int = 8):
        """Call the LLM with the given input and structure the output into steps."""
        # initial user/system messages for the *first* request only
        initial_input = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self._system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text if isinstance(text, str) else json.dumps(text)}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Think deeply about what to do, then call a NETHACK-ACTION tool."}]
            },
        ]

        steps = []
        tokens_used = {}
        previous_response_id = None
        tool_rounds = 0
        next_input_items = initial_input

        while True:
            if tool_rounds > max_tool_rounds:
                raise RuntimeError("Exceeded maximum tool rounds; possible tool loop.")

            pending_calls = []

            client_args = {
                "model": self.model,
                "tools": self._tools.to_openai_tools(),
                "tool_choice": "auto",
                "reasoning": {"effort": self.reasoning_effort, "summary": "detailed"},
                "input": next_input_items,
            }

            if previous_response_id:
                client_args["previous_response_id"] = previous_response_id

            next_input_items = None

            with self.client.responses.stream(**client_args) as stream:
                for event in stream:
                    if event.type == "response.output_item.done":
                        if event.item.type == "function_call":
                            # collect the function call; do NOT break yet
                            pending_calls.append(event.item)
                        elif event.item.type == "message":
                            message = event.item
                            if message.role == "assistant" and message.status == "completed":
                                for content in message.content:
                                    try:
                                        steps.append(JsonStep(json.loads(content.text)))
                                    except json.JSONDecodeError:
                                        steps.append(MessageStep(content.text))
                                    output_callback(steps[-1])

                    elif event.type == "response.reasoning_summary_text.delta":
                        output_callback(OutputDelta(event.delta, True))

                    elif event.type == "response.output_text.delta":
                        output_callback(OutputDelta(event.delta, False))

                    elif event.type == "response.reasoning_summary_text.done":
                        steps.append(ThinkingStep(event.text))
                        output_callback(steps[-1])

                    elif event.type == "response.completed":
                        previous_response_id = event.response.id
                        self._add_usage(tokens_used, event.response.usage)

            # If there were tool calls, execute them and feed results back
            if pending_calls:
                tool_rounds += 1
                outputs = []
                for fc in pending_calls:
                    self._call_function(outputs, steps, fc)
                    output_callback(steps[-1])

                # Next loop iteration will send these outputs with previous_response_id
                next_input_items = outputs
                continue

            return steps, tokens_used


    def _call_function(self, messages, steps, fc):
        call_id = fc.call_id
        args = json.loads(fc.arguments)
        name = fc.name
        tool_result = self._tools.call_local(name, args)
        messages.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": tool_result if isinstance(tool_result, str) else json.dumps(tool_result)
            })

        steps.append(FunctionCallStep(name, args, tool_result))

    def _add_usage(self, tokens_used, usage):
        tokens_used["input"] = tokens_used.get("input_tokens", 0) + usage.input_tokens
        tokens_used["output"] = tokens_used.get("output_tokens", 0) + usage.output_tokens
        tokens_used["total"] = tokens_used.get("total_tokens", 0) + usage.total_tokens
        reasoning = usage.output_tokens_details.reasoning_tokens
        tokens_used["reasoning"] = tokens_used.get("reasoning_tokens", 0) + reasoning
        cached_tokens = usage.input_tokens_details.cached_tokens
        tokens_used["cached"] = tokens_used.get("cached_tokens", 0) + cached_tokens
