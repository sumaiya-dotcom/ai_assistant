# Week 2 – Python AI Assistant (Gemini)

Chat assistant that sends questions to the **Gemini API** (Google AI Studio), keeps conversation history, follows a system instruction, and returns structured JSON for IT ticket intake.

## What this covers

- How an LLM API call is built: system instruction + user message + history
- Tokens and context (API usage + a local estimate)
- Temperature and other generation settings
- Structured JSON output (`SupportTicket`)
- Prompt variants and an experiment runner

## Setup

1. Create an API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and configure:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

3. Put your key in `.env`:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
PROVIDER=gemini
```

If that model name fails with 404, open AI Studio and copy a current Flash model id into `GEMINI_MODEL`.

## Run the assistant

```powershell
py cli.py
```

Example:

```
You: What should I say in standup if I finished 1 of 3 tickets?
You: /ticket Can't log into Jira after Okta cutover, Payments team, need access today
You: /usage
You: /reset
```

`/ticket` is the structured-output path: Gemini must return a `SupportTicket` (category, priority, summary, team, asset id, missing info).

## Experiments

```powershell
py -m experiments.run
```

Writes `experiments/results/latest.json` and `latest.csv`. How to interpret the runs is in [EXPERIMENTS.md](EXPERIMENTS.md).

## Layout

| Path | Role |
| --- | --- |
| `assistant/core.py` | Public `AIAssistant` API |
| `assistant/memory.py` | History + token-budget trim |
| `assistant/prompts.py` | System instructions and ticket prompts |
| `assistant/schemas.py` | Pydantic structured output |
| `assistant/llm.py` | Gemini client (OpenAI/Groq still supported) |
| `cli.py` | Interactive chat |
| `experiments/run.py` | Prompt / temperature / history / JSON comparisons |
