from __future__ import annotations

from typing import Any

import httpx

from .config import Settings


class GroqClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        payload: dict[str, Any] = {
            "model": self._settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
