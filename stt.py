from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModel

from audio import load_mono
from config import Settings

logger = logging.getLogger(__name__)


class IndicConformerSTT:
    def __init__(self, settings: Settings, device: str) -> None:
        self._settings = settings
        self._device = device
        logger.info("Loading STT model %s", settings.stt_model_id)
        self._model = AutoModel.from_pretrained(
            settings.stt_model_id, trust_remote_code=True
        ).to(device)
        self._model.eval()

    @torch.inference_mode()
    def transcribe(self, audio_path: Path) -> str:
        waveform = load_mono(audio_path, self._settings.stt_sample_rate).to(self._device)
        text = self._model(waveform, self._settings.stt_language, self._settings.stt_decoding)
        return text.strip() if isinstance(text, str) else str(text).strip()
