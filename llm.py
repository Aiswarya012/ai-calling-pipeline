from __future__ import annotations

import json
import logging
from pathlib import Path

from config import Settings

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a showroom appointment for the customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["name", "date", "time"],
            },
        },
    }
]


def book_appointment(name: str, date: str, time: str) -> str:
    logger.info("Booked appointment for %s on %s at %s", name, date, time)
    return f"Appointment confirmed for {name} on {date} at {time}."


class ConversationLLM:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._system_prompt = self._load_system_prompt()
        self._client = self._build_client()

    def _load_system_prompt(self) -> str:
        base = "You are a helpful Hindi-speaking showroom assistant. Reply in Hindi."
        path: Path = self._settings.knowledge_path
        if path.exists():
            return f"{base}\n\nCompany knowledge:\n{path.read_text(encoding='utf-8')}"
        return base

    def _build_client(self):
        if not self._settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set; using echo fallback reply")
            return None
        from openai import OpenAI

        return OpenAI(api_key=self._settings.openai_api_key)

    def reply(self, user_text: str) -> str:
        if self._client is None:
            return f"आपने कहा: {user_text}"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_text},
        ]
        response = self._client.chat.completions.create(
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or ""
        return self._resolve_tool_calls(messages, message)

    def _resolve_tool_calls(self, messages: list[dict], message) -> str:
        messages.append(message.model_dump())
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = book_appointment(**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
        follow_up = self._client.chat.completions.create(
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            messages=messages,
        )
        return follow_up.choices[0].message.content or ""
