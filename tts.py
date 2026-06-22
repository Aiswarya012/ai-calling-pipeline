from __future__ import annotations

import logging

import numpy as np
from kokoro import KPipeline

from config import Settings

logger = logging.getLogger(__name__)


class KokoroTTS:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        logger.info("Loading TTS pipeline lang_code=%s", settings.tts_lang_code)
        self._pipeline = KPipeline(lang_code=settings.tts_lang_code)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks = [
            audio
            for _, _, audio in self._pipeline(text, voice=self._settings.tts_voice)
        ]
        if not chunks:
            raise ValueError("TTS produced no audio")
        return np.concatenate(chunks), self._settings.tts_sample_rate
