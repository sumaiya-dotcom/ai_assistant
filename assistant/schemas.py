from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SupportTicket(BaseModel):
    """Structured output for an IT support intake use case."""

    category: Literal["network", "hardware", "software", "access", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    summary: str = Field(..., max_length=160)
    requester_team: str | None = None
    asset_id: str | None = None
    missing_info: list[str] = Field(default_factory=list)


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_prompt_tokens: int = 0

    def add(self, other: "UsageStats") -> "UsageStats":
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_prompt_tokens=self.estimated_prompt_tokens
            + other.estimated_prompt_tokens,
        )
