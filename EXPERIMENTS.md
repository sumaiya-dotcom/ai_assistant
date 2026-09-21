# Week 2 experiments

Run `py -m experiments.run` after `.env` has a working `GEMINI_API_KEY`. Compare the printed trials (also saved under `experiments/results/`).

## What each trial tests

| Trial | Idea |
| --- | --- |
| `vague_system` vs `default_system` vs `concise_system` vs `verbose_system` | Prompt structure / system instruction |
| `temp_0` vs `temp_0.9` | Temperature: consistency vs variety |
| `history_on_followup` vs `history_off_followup` | Conversation memory |
| `ticket_zero_shot` vs `ticket_few_shot` | Structured JSON + prompt engineering |

## How to score results

**Response quality** — Is the standup answer useful and specific, or generic filler?

**Accuracy** — Does the follow-up with history recall **18** story points? Does the ticket use category `access`, high/critical priority, team Payments, and **no invented asset id**?

**Consistency** — `temp_0` should stay closer across re-runs than `temp_0.9`. Few-shot JSON should match the schema more reliably than a vague prompt.

**Token / context usage** — Verbose system + long history raises `prompt_tokens`. `/usage` in the CLI shows billed Gemini tokens vs the local estimate.

Fill this in after you run the script:

| Trial | Quality (1–5) | Accurate? | Tokens | Notes |
| --- | --- | --- | --- | --- |
| vague_system |  |  |  |  |
| default_system |  |  |  |  |
| concise_system |  |  |  |  |
| verbose_system |  |  |  |  |
| temp_0 |  |  |  |  |
| temp_0.9 |  |  |  |  |
| history_on_followup |  |  |  |  |
| history_off_followup |  |  |  |  |
| ticket_zero_shot |  |  |  |  |
| ticket_few_shot |  |  |  |  |
