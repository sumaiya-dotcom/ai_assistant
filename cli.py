"""Interactive CLI for the workplace AI assistant."""

from __future__ import annotations

import json
import sys

from assistant import AIAssistant


HELP = """Commands:
  /ticket <text>   Extract a structured support ticket
  /reset           Clear conversation history
  /usage           Show token usage for this session
  /help            Show this help
  /exit            Quit
"""


def main() -> None:
    try:
        assistant = AIAssistant()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


    print("=======================")
    print(" AI Assistant ready.")
    print("=======================")
    print(f"Model: {assistant.client.config.provider}/{assistant.client.config.model}")

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        if user == "/help":
            print(HELP)
            continue
        if user == "/reset":
            assistant.reset()
            print("History cleared.")
            continue
        if user == "/usage":
            print(assistant.session_usage.model_dump_json(indent=2))
            continue
        
        if user.startswith("/ticket"):
            report = user[len("/ticket") :].strip()
            if not report:
                print("Usage: /ticket VPN keeps dropping on laptop L-12")
                continue

            try:
                ticket = assistant.extract_ticket(report)
            except Exception as exc:
                print(f"Ticket extraction failed: {exc}")
                continue
            print(json.dumps(ticket.model_dump(), indent=2))
            continue
        

        try:
            answer = assistant.ask(user)
            print("\n")
        except Exception as exc:
            print("NO INTERNET CONNECTION")
            continue
        print("=========================================")
        print(f"Assistant: \n{answer}")
        print("=========================================")



if __name__ == "__main__":
    main()
