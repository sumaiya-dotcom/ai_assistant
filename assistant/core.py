from __future__ import annotations

from .llm import LLMClient
from .memory import ConversationMemory
from .prompts import DEFAULT_SYSTEM
from .schemas import SupportTicket, UsageStats


class AIAssistant:
    def __init__(
        self,
        *,
        system_instruction: str = DEFAULT_SYSTEM,
        temperature: float = 0.2,
        client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.client = client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.session_usage = UsageStats()

    def ask(self, question: str) -> str:
        self.memory.add("user", question)
        messages = self._messages()
        answer, usage = self.client.complete(
            messages, temperature=self.temperature
        )
        self.session_usage = self.session_usage.add(usage)
        self.memory.add("assistant", answer)
        return answer

    def extract_ticket(self, report: str) -> SupportTicket:
        """Structured-output use case: turn a free-text incident into a ticket."""
        messages = [
            {
                "role": "system",
                "content": (
                    self.system_instruction
                    + "\nClassify the user's incident as a support ticket."
                ),
            },
            *self.memory.history(),
            {"role": "user", "content": report},
        ]
        ticket, usage = self.client.complete_structured(
            messages, SupportTicket, temperature=0.0
        )
        self.session_usage = self.session_usage.add(usage)
        self.memory.add("user", report)
        self.memory.add("assistant", ticket.model_dump_json())
        return ticket

    def reset(self) -> None:
        self.memory.clear()
        self.session_usage = UsageStats()

    def _messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_instruction},
            *self.memory.history(),
        ]
