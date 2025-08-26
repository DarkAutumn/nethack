# llm_gpt_oss.py

import threading
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from types import FunctionType
import inspect

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


class LLMWithTools:
    """Simple wrapper around openai/gpt-oss-20b with tool calling."""

    def __init__(
        self,
        system_prompt: str,
        model_id: str = "openai/gpt-oss-20b",
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,  # "low" | "medium" | "high"
        dtype: Any = "auto",
        device_map: str = "auto",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Optional[PreTrainedModel] = None,
        io_lock: Optional[threading.Lock] = None,
    ) -> None:
        """
        Args:
            system_prompt: The developer/system instructions to prepend.
            model_id: HF model id (defaults to openai/gpt-oss-20b).
            max_new_tokens: Generation cap for a single step.
            temperature: Sampling temperature.
            reasoning_effort: Optional Harmony knob ("low"|"medium"|"high").
            dtype: torch dtype or "auto".
            device_map: transformers device map ("auto" recommended).
        """
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

        # Load or reuse tokenizer/model
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            max_memory = {0: "14GiB"}
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder="./offload",
            )
        self.model.eval()

        # Shared generation lock (avoid concurrent .generate() on the same model)
        self._gen_lock = io_lock or threading.Lock()

        self._registry: dict[str, Callable[..., Any]] = {}             # name -> original callback for execution
        self._tools_for_template: dict[str, Callable[..., Any]] = {}   # name -> alias function with preserved signature

        # Tool-call parsing patterns (support Harmony & generic <tool_call>).
        self._re_harmony_tool = re.compile(
            r"<\|channel\|>commentary\s+to=functions\.([a-zA-Z0-9_]+).*?"
            r"<\|message\|>(\{.*?\})\s*<\|call\|>",
            re.DOTALL,
        )
        self._re_generic_tool = re.compile(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
        )

    # ---------- Public API ----------
    def register_tool(self, name: str, callback: Callable[..., Any]) -> None:
        """
        Register a tool. 'name' is what the model will call.
        'callback' must be a plain Python function with full type hints.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")

        # Validate: all parameters and return must be type-hinted
        sig = inspect.signature(callback)
        for p in sig.parameters.values():
            if p.annotation is inspect._empty:  # pylint: disable=protected-access
                raise TypeError(f"Parameter '{p.name}' of tool '{name}' lacks a type hint")
        if getattr(callback, "__annotations__", {}).get("return", inspect._empty) is inspect._empty:
            raise TypeError(f"Tool '{name}' is missing a return type hint")

        # Create an alias function with the SAME code/signature but a different __name__
        alias = FunctionType(
            code=callback.__code__,
            globals=callback.__globals__,
            name=name,
            argdefs=callback.__defaults__,
            closure=callback.__closure__,
        )
        alias.__annotations__ = dict(getattr(callback, "__annotations__", {}))
        alias.__doc__ = callback.__doc__
        alias.__kwdefaults__ = getattr(callback, "__kwdefaults__", None)
        alias.__module__ = callback.__module__
        alias.__qualname__ = name  # keep things tidy for schema display

        # Store for: (a) prompt schema, (b) runtime execution
        self._tools_for_template[name] = alias
        self._registry[name] = callback


    def chat(self, message: str, system_prompt : Optional[str] = None, max_calls=16) -> Tuple[str, str]:
        """
        Run a single, stateless chat turn.

        Returns:
            (thinking, result) where:
              - thinking is the model's chain-of-thought (Harmony 'analysis' + 'commentary')
              - result   is the user-facing final answer (Harmony 'final')

        The model may call at most a few tools (bounded loop). Each call is executed
        and its output is fed back before the model continues.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": message},
        ]

        thinking_chunks: List[str] = []

        # One or more "generate -> (maybe) tool call -> tool result -> generate ..." cycles
        for _ in range(max_calls + 1):
            text = self._generate_once(messages)

            # Collect analysis/commentary (thinking) if present
            analysis = self._extract_channel(text, "analysis")
            if analysis:
                thinking_chunks.append(analysis.strip())
            commentary = self._extract_channel(text, "commentary")
            if commentary:
                thinking_chunks.append(commentary.strip())

            # If the model is requesting a tool, execute it and loop again
            parsed = self._parse_tool_call(text)
            if parsed is not None:
                tool_name, tool_args = parsed
                tool_output = self._run_tool(tool_name, tool_args)

                # Append the assistant's *declared* tool call (structure the template expects)
                tool_call_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }
                    ],
                }
                messages.append(tool_call_message)

                # Append the tool result so the model can use it next
                messages.append({"role": "tool", "content": tool_output})
                continue

            # No more tool calls; extract final answer and return
            final = self._extract_channel(text, "final") or text.strip()
            thinking = "\n\n".join([t for t in thinking_chunks if t]).strip()
            return thinking, final.rstrip("<|return|>").strip()

        # Safety net: if loop exhausts without a final, return what we have
        LOGGER.warning("Exceeded max tool-call rounds without a final answer.")
        return "\n\n".join(thinking_chunks).strip(), ""

    # ---------- Internals ----------

    def _generate_once(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Apply chat template and run a single generation; return only the new text."""
        kwargs: Dict[str, Any] = dict(
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if self.reasoning_effort:
            # The GPT-OSS chat template allows overriding Harmony "Reasoning" level
            kwargs["reasoning_effort"] = self.reasoning_effort

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=list(self._tools_for_template.values()),  # <- use alias functions here
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
        ).to(self.model.device)

        with self._gen_lock, torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # Only decode the newly generated suffix after the prompt length
        prompt_len = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0][prompt_len:])

    def _parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Detect and parse a tool call from model output.

        Supports:
          - Harmony:  <|channel|>commentary to=functions.NAME <|message|>{...}<|call|>
          - Generic:  <tool_call> {"name": "...", "arguments": {...}} </tool_call>
        """
        m = self._re_harmony_tool.search(text)
        if m:
            name, args_json = m.group(1), m.group(2)
            try:
                return name, json.loads(args_json)
            except json.JSONDecodeError:
                LOGGER.error("Failed to parse Harmony tool args JSON: %s", args_json)
                return name, {}

        m = self._re_generic_tool.search(text)
        if m:
            try:
                data = json.loads(m.group(1))
                name = data.get("name") or ""
                args = data.get("arguments") or {}
                if name:
                    return name, args
            except json.JSONDecodeError:
                LOGGER.error("Failed to parse <tool_call> JSON.")
        return None

    def _run_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a registered tool and stringify the result."""
        if name not in self._registry:
            return json.dumps(
                {"error": f"Tool '{name}' is not registered", "args": args}
            )
        try:
            result = self._registry[name](**args)
        except TypeError as exc:
            return json.dumps(
                {"error": f"Bad arguments for tool '{name}': {exc}", "args": args}
            )
        except Exception as exc:  # pylint: disable=broad-except
            return json.dumps({"error": f"Tool '{name}' raised: {exc}"})

        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)

    @staticmethod
    def _extract_channel(text: str, channel: str) -> str:
        """
        Extract the content of a Harmony channel from the assistant output.
        Examples of delimiters:
          <|channel|>analysis<|message|> ... <|end|>
          <|channel|>final<|message|> ... <|end|>
        """
        # Try to isolate the desired channel block(s)
        pattern = re.compile(
            rf"<\|channel\|>{re.escape(channel)}<\|message\|>(.*?)(?:<\|end\|>|$)",
            re.DOTALL,
        )
        parts = pattern.findall(text)
        if parts:
            return "\n".join(p.strip() for p in parts if p.strip())
        # Fallback: if final not found, take everything after the tag once
        if channel == "final" and "<|channel|>final<|message|>" in text:
            return text.split("<|channel|>final<|message|>", maxsplit=1)[-1].split(
                "<|end|>", maxsplit=1
            )[0]
        return ""
