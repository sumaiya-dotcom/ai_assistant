from __future__ import annotations

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback if tiktoken data is missing
    _enc = None


def count_text_tokens(text: str) -> int:
    if _enc is None:
        return max(1, len(text) // 4)
    return len(_enc.encode(text))


def count_tokens(messages: list[dict[str, str]]) -> int:
    """Rough chat-format token estimate (not billed usage)."""
    total = 0
    for message in messages:
        total += 4  # role/name framing overhead
        total += count_text_tokens(message.get("content", ""))
    return total + 2
