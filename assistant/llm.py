from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import LLMConfig
from .schemas import UsageStats
from .tokenizer import count_tokens

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Calls Gemini (default) or an OpenAI-compatible chat API."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._gemini = None
        self._openai = None
        if self.config.provider == "gemini":
            from google import genai

            self._gemini = genai.Client(api_key=self.config.api_key)
        else:
            from openai import OpenAI

            self._openai = OpenAI(
                api_key=self.config.api_key, base_url=self.config.base_url
            )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> tuple[str, UsageStats]:
        if self._gemini is not None:
            return self._complete_gemini(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        return self._complete_openai(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> tuple[T, UsageStats]:
        if self._gemini is not None:
            text, usage = self._complete_gemini(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                response_schema=schema,
            )
            return schema.model_validate_json(_extract_json(text)), usage

        parse = getattr(self._openai.chat.completions, "parse", None)
        if parse is not None:
            try:
                response = parse(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=schema,
                )
                parsed = response.choices[0].message.parsed
                if parsed is not None:
                    return parsed, _usage_from_openai(response, messages)
            except Exception:
                pass

        json_messages = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Return only valid JSON matching this schema:\n"
                    f"{schema.model_json_schema()}"
                ),
            },
        ]
        text, usage = self.complete(
            json_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return schema.model_validate_json(_extract_json(text)), usage

    def _complete_gemini(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        response_schema: type[BaseModel] | None = None,
    ) -> tuple[str, UsageStats]:
        from google.genai import types

        system, contents = _to_gemini_contents(messages)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if json_mode or response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
            if "Return only valid JSON" not in (system or ""):
                schema_hint = (
                    "Return only valid JSON matching this schema:\n"
                    f"{json.dumps(response_schema.model_json_schema())}"
                )
                config_kwargs["system_instruction"] = (
                    f"{system}\n\n{schema_hint}" if system else schema_hint
                )

        history, latest = contents[:-1], contents[-1]
        latest_text = "".join(part.text or "" for part in (latest.parts or []))
        try:
            response = self._send_gemini_chat(
                history, latest_text, config_kwargs
            )
        except Exception:
            if response_schema is None:
                raise
            config_kwargs.pop("response_schema", None)
            response = self._send_gemini_chat(
                history, latest_text, config_kwargs
            )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError(
                "Gemini returned an empty response. Try a different GEMINI_MODEL "
                "in .env (for example gemini-3.6-flash) or raise max tokens."
            )
        return text, _usage_from_gemini(response, messages)

    def _send_gemini_chat(
        self,
        history: list[Any],
        latest_text: str,
        config_kwargs: dict[str, Any],
    ) -> Any:
        from google.genai import types

        chat = self._gemini.chats.create(
            model=self.config.model,
            history=history,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return chat.send_message(latest_text)

    def _complete_openai(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, UsageStats]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._openai.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "").strip()
        return text, _usage_from_openai(response, messages)


def _to_gemini_contents(messages: list[dict[str, str]]) -> tuple[str | None, list[Any]]:
    from google.genai import types

    system_parts: list[str] = []
    contents: list[Any] = []
    for message in messages:
        role = message.get("role", "user")
        text = message.get("content", "")
        if role == "system":
            system_parts.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(role=gemini_role, parts=[types.Part.from_text(text=text)])
        )
    system = "\n\n".join(part for part in system_parts if part) or None
    if not contents:
        raise ValueError("At least one user message is required.")
    return system, contents


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fenced:
        return fenced.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _usage_from_openai(response: Any, messages: list[dict[str, str]]) -> UsageStats:
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
    return UsageStats(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        estimated_prompt_tokens=count_tokens(messages),
    )


def _usage_from_gemini(response: Any, messages: list[dict[str, str]]) -> UsageStats:
    usage = getattr(response, "usage_metadata", None)
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion = int(getattr(usage, "candidates_token_count", 0) or 0)
    total = int(getattr(usage, "total_token_count", 0) or (prompt + completion))
    return UsageStats(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        estimated_prompt_tokens=count_tokens(messages),
    )
