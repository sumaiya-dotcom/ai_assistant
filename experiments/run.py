"""Compare prompt strategies on the same tasks.

Run:
  py -m experiments.run
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from assistant.config import LLMConfig
from assistant.core import AIAssistant
from assistant.llm import LLMClient
from assistant.memory import ConversationMemory
from assistant.prompts import (
    CONCISE_SYSTEM,
    DEFAULT_SYSTEM,
    FEW_SHOT_TICKET,
    VAGUE_SYSTEM,
    VERBOSE_SYSTEM,
    ZERO_SHOT_TICKET,
)
from assistant.schemas import SupportTicket
from assistant.tokenizer import count_tokens

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FACT_QUESTION = (
    "Our sprint is 2 weeks. We completed 18 story points last sprint and "
    "committed 24 this sprint. What is a reasonable daily standup talking point "
    "for a developer who finished 1 of 3 tickets?"
)

FOLLOW_UP = "Using only numbers I already gave you, what was last sprint's velocity?"

INCIDENT = (
    "Can't log into Jira since the Okta cutover. I'm on the Payments team. "
    "Happens on my laptop and phone. Need access today for a release."
)

GOLDEN_TICKET = SupportTicket(
    category="access",
    priority="high",
    summary="Cannot log into Jira after Okta cutover",
    requester_team="Payments",
    asset_id=None,
    missing_info=["username or Okta id", "error message"],
)


@dataclass
class TrialResult:
    name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_prompt_tokens: int
    latency_ms: int
    output: str
    notes: str = ""


def _run_chat(
    name: str,
    system: str,
    question: str,
    *,
    temperature: float,
    history: list[dict[str, str]] | None = None,
) -> TrialResult:
    client = LLMClient()
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    start = time.perf_counter()
    text, usage = client.complete(messages, temperature=temperature)
    elapsed = int((time.perf_counter() - start) * 1000)
    return TrialResult(
        name=name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        estimated_prompt_tokens=usage.estimated_prompt_tokens,
        latency_ms=elapsed,
        output=text,
    )


def _run_ticket(name: str, system: str) -> TrialResult:
    client = LLMClient()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": INCIDENT},
    ]
    start = time.perf_counter()
    ticket, usage = client.complete_structured(
        messages, SupportTicket, temperature=0.0
    )
    elapsed = int((time.perf_counter() - start) * 1000)
    notes = _score_ticket(ticket)
    return TrialResult(
        name=name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        estimated_prompt_tokens=count_tokens(messages),
        latency_ms=elapsed,
        output=ticket.model_dump_json(),
        notes=notes,
    )


def _score_ticket(ticket: SupportTicket) -> str:
    checks = {
        "category": ticket.category == GOLDEN_TICKET.category,
        "priority_high_or_critical": ticket.priority in {"high", "critical"},
        "team": (ticket.requester_team or "").lower().startswith("payment"),
        "no_fake_asset": ticket.asset_id in (None, "", "unknown"),
        "mentions_jira_or_okta": any(
            word in ticket.summary.lower() for word in ("jira", "okta", "login", "access")
        ),
    }
    passed = sum(1 for ok in checks.values() if ok)
    failed = [k for k, ok in checks.items() if not ok]
    return f"score {passed}/{len(checks)}; fail={failed}"


def run() -> list[TrialResult]:
    config = LLMConfig.from_env()
    results: list[TrialResult] = []

    results.append(
        _run_chat("vague_system", VAGUE_SYSTEM, FACT_QUESTION, temperature=0.2)
    )
    results.append(
        _run_chat("default_system", DEFAULT_SYSTEM, FACT_QUESTION, temperature=0.2)
    )
    results.append(
        _run_chat("concise_system", CONCISE_SYSTEM, FACT_QUESTION, temperature=0.2)
    )
    results.append(
        _run_chat("verbose_system", VERBOSE_SYSTEM, FACT_QUESTION, temperature=0.2)
    )
    results.append(
        _run_chat("temp_0", DEFAULT_SYSTEM, FACT_QUESTION, temperature=0.0)
    )
    results.append(
        _run_chat("temp_0.9", DEFAULT_SYSTEM, FACT_QUESTION, temperature=0.9)
    )

    memory_on = AIAssistant(
        system_instruction=DEFAULT_SYSTEM,
        temperature=0.0,
        client=LLMClient(),
        memory=ConversationMemory(),
    )
    first = memory_on.ask(FACT_QUESTION)
    follow = memory_on.ask(FOLLOW_UP)
    usage = memory_on.session_usage
    results.append(
        TrialResult(
            name="history_on_followup",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_prompt_tokens=usage.estimated_prompt_tokens,
            latency_ms=0,
            output=follow,
            notes=f"turn1_len={len(first)}",
        )
    )

    memory_off = AIAssistant(
        system_instruction=DEFAULT_SYSTEM,
        temperature=0.0,
        client=LLMClient(),
        memory=ConversationMemory(),
    )
    memory_off.ask(FACT_QUESTION)
    memory_off.memory.clear()
    follow_off = memory_off.ask(FOLLOW_UP)
    usage_off = memory_off.session_usage
    results.append(
        TrialResult(
            name="history_off_followup",
            prompt_tokens=usage_off.prompt_tokens,
            completion_tokens=usage_off.completion_tokens,
            total_tokens=usage_off.total_tokens,
            estimated_prompt_tokens=usage_off.estimated_prompt_tokens,
            latency_ms=0,
            output=follow_off,
            notes="history cleared before follow-up",
        )
    )

    results.append(_run_ticket("ticket_zero_shot", ZERO_SHOT_TICKET))
    results.append(_run_ticket("ticket_few_shot", FEW_SHOT_TICKET + "\n" + ZERO_SHOT_TICKET))

    payload = {
        "provider": config.provider,
        "model": config.model,
        "trials": [asdict(r) for r in results],
    }
    (RESULTS / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = RESULTS / "latest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    return results


if __name__ == "__main__":
    for trial in run():
        print(f"\n=== {trial.name} ===")
        print(
            f"tokens prompt={trial.prompt_tokens} completion={trial.completion_tokens} "
            f"total={trial.total_tokens} est_prompt={trial.estimated_prompt_tokens} "
            f"latency_ms={trial.latency_ms}"
        )
        if trial.notes:
            print(f"notes: {trial.notes}")
        print(trial.output[:600])
