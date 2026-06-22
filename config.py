from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    stt_model_id: str = "ai4bharat/indic-conformer-600m-multilingual"
    stt_language: str = "hi"
    stt_decoding: str = "rnnt"
    stt_sample_rate: int = 16_000

    tts_lang_code: str = "h"
    tts_voice: str = "hf_alpha"
    tts_sample_rate: int = 24_000

    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    openai_api_key: str | None = Field(default=None)

    knowledge_path: Path = Path(__file__).parent / "knowledge.md"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(openai_api_key=os.environ.get("OPENAI_API_KEY"))
