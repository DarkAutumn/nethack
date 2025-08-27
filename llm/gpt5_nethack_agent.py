from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Literal

import openai  # for pydantic_function_tool
from openai import OpenAI
from pydantic import BaseModel, Field, conint, constr

# -----------------------
# Pydantic models for tools (add the rest of your tools as needed)
# -----------------------

class MoveArgs(BaseModel):
    """NETHACK-ACTION: Move in the specified direction."""
    direction: Literal["n","s","e","w","nw","ne","sw","se"] = Field(..., description="Compass direction to move")

class WaitArgs(BaseModel):
    """NETHACK-ACTION: Wait (rest) for a number of turns."""
    num_turns: conint(ge=1, le=1000) = Field(..., description="Number of turns to wait")

class SearchArgs(BaseModel):
    """NETHACK-ACTION: Search for hidden doors/traps around you."""
    num_turns: conint(ge=1, le=50) =  Field(..., description="Search turns (22 total is a common recommendation)")

class KickArgs(BaseModel):
    """NETHACK-ACTION: Kick in the given direction."""
    direction: Literal["n","s","e","w","nw","ne","sw","se"]

class EatArgs(BaseModel):
    """NETHACK-ACTION: Eat a food item from inventory or the floor."""
    inventory_id: constr(min_length=1, max_length=5) = Field(..., description="Letter of item, or 'floor'")

class RespondYNArgs(BaseModel):
    """NETHACK-ACTION: Respond to a yes/no prompt."""
    response: Literal["y","n","q"]

class RespondDirectionArgs(BaseModel):
    """NETHACK-ACTION: Respond to a direction prompt."""
    direction: Literal["here","n","s","e","w","nw","ne","sw","se"]

class RespondInventoryArgs(BaseModel):
    """NETHACK-ACTION: Respond with an inventory letter."""
    inventory_id: constr(min_length=1, max_length=1)


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
        spec = openai.pydantic_function_tool(model_cls)
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


# -----------------------
# GPT-5 nano agent with streaming + tool calls
# -----------------------

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

    def play_nethack(self, status: Any, *, print_stream: bool = True, max_tool_rounds: int = 8) -> Dict[str, Any]:
        """
        Fresh turn using Responses API streaming + tool calling (2025-style):
        - stream once, accumulate function-call args via *.arguments.delta
        - run local tools
        - stream again with previous_response_id and the tool outputs
        - repeat until no more tool calls; then final output_text must be {"action": ...}
        """

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self._system_prompt}]
            },
            {
                "role": "user", "content":
                [
                    {
                        "type": "input_text",
                        "text": status if isinstance(status, str) else json.dumps(status)
                    },
                ]
            },
            {
                "role": "user", "content":
                [
                    {
                        "type": "input_text",
                        "text": "Think deeply about what to do, then call a NETHACK-ACTION tool."
                    },
                ]
            }
        ]

        client_args = {
            "model": self.model,
            "tools": self._tools.to_openai_tools(),
            "tool_choice": "auto",
            "reasoning": {"effort": self.reasoning_effort, "summary": "detailed"},
            "input": messages
        }

        tools_called = []
        tokens_used = {}
        reasoning = []
        result = None

        while not result:
            if len(tools_called) >= max_tool_rounds:
                raise RuntimeError("Exceeded maximum tool rounds; possible tool loop.")
            with self.client.responses.stream(**client_args) as stream:
                for event in stream:
                    f.write(event.model_dump_json(indent=2) + "\n")
                    f.flush()
                    match event.type:
                        case "response.output_item.done":
                            match event.item.type:
                                case "function_call":
                                    function_call = event.item

                                    call_id = function_call.call_id
                                    args = json.loads(function_call.arguments)
                                    name = function_call.name
                                    tool_result = self._tools.call_local(name, args)
                                    messages.append({
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": tool_result if isinstance(tool_result, str) else json.dumps(tool_result)
                                    })
                                    tools_called.append({
                                        "id": call_id,
                                        "name": name,
                                        "args": args,
                                        "result": tool_result
                                    })

                                case "message":
                                    message = event.item
                                    if message.role == "assistant" and message.status == "completed":
                                        for content in message.content:
                                            try:
                                                result = json.loads(content.text)
                                            except json.JSONDecodeError:
                                                result = None

                        case "response.reasoning_summary_text.delta":
                            if print_stream:
                                print(f"\033[33m{event.delta}\033[0m", end="", flush=True)

                        case "response.output_text.delta":
                            if print_stream:
                                print(event.delta, end="", flush=True)

                        case "response.reasoning_summary_text.done":
                            reasoning.append(event.text)

                        case "response.completed":
                            response = event.response
                            client_args["previous_response_id"] = response.id

                            self._add_usage(tokens_used, response.usage)

        return {
            "tools_called": tools_called,
            "tokens_used": tokens_used,
            "reasoning": reasoning,
            "result": result
        }

    def _add_usage(self, tokens_used, usage):
        tokens_used["input"] = tokens_used.get("input_tokens", 0) + usage.input_tokens
        tokens_used["output"] = tokens_used.get("output_tokens", 0) + usage.output_tokens
        tokens_used["total"] = tokens_used.get("total_tokens", 0) + usage.total_tokens
        reasoning = usage.output_tokens_details.reasoning_tokens
        tokens_used["reasoning"] = tokens_used.get("reasoning_tokens", 0) + reasoning
        cached_tokens = usage.input_tokens_details.cached_tokens
        tokens_used["cached"] = tokens_used.get("cached_tokens", 0) + cached_tokens


# -----------------------
# Helpers
# -----------------------

def _response_to_message(resp: Any) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    if getattr(resp, "output", None):
        for item in resp.output:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []):
                    if c.get("type") == "output_text" and "text" in c:
                        parts.append({"type": "output_text", "text": c["text"]})
                    elif c.get("type") == "input_text" and "text" in c:
                        parts.append({"type": "text", "text": c["text"]})
    elif getattr(resp, "output_text", None):
        parts.append({"type": "output_text", "text": resp.output_text})
    if not parts:
        parts = [{"type": "output_text", "text": ""}]
    return {"role": "assistant", "content": parts}

def _best_effort_json(s: str) -> str:
    t = s.strip().strip("`")
    opens, closes = t.count("{"), t.count("}")
    if opens > closes:
        t += "}" * (opens - closes)
    return t
