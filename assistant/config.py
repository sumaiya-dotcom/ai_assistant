from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("PROVIDER", "").strip().lower()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not provider:
            if gemini_key:
                provider = "gemini"
            elif groq_key and not openai_key:
                provider = "groq"
            else:
                provider = "openai"

        if provider == "gemini":
            if not gemini_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env and paste "
                    "your key from https://aistudio.google.com/apikey"
                )
            return cls(
                provider="gemini",
                api_key=gemini_key,
                base_url=os.getenv(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta",
                ),
                model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").removeprefix(
                    "models/"
                ),
            )

        if provider == "groq":
            if not groq_key:
                raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env.")
            return cls(
                provider="groq",
                api_key=groq_key,
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            )

        if not openai_key:
            raise RuntimeError(
                "No LLM API key found. Set GEMINI_API_KEY in .env "
                "(see .env.example). OpenAI and Groq keys also work."
            )
        return cls(
            provider="openai",
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
