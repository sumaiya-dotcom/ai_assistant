from __future__ import annotations

from dataclasses import dataclass, field

from .tokenizer import count_tokens


Message = dict[str, str]


@dataclass
class ConversationMemory:
    """Stores chat turns and trims oldest user/assistant pairs when over budget."""

    max_messages: int = 20
    max_context_tokens: int = 4000
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim()

    def history(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)
        while (
            self.messages
            and count_tokens(self.messages) > self.max_context_tokens
        ):
            self.messages.pop(0)
