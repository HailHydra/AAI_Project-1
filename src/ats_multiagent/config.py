from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is required")
        model = os.getenv("GROQ_MODEL", cls.groq_model)
        return cls(groq_api_key=key, groq_model=model)
