from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from audio import save_wav
from config import Settings
from llm import ConversationLLM
from stt import IndicConformerSTT
from tts import KokoroTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("pipeline")


class CallPipeline:
    def __init__(self, settings: Settings) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device %s", device)
        self._stt = IndicConformerSTT(settings, device)
        self._llm = ConversationLLM(settings)
        self._tts = KokoroTTS(settings)

    def run(self, input_path: Path, output_path: Path) -> None:
        started = time.perf_counter()

        transcript = self._stt.transcribe(input_path)
        logger.info("STT transcript: %s", transcript)

        reply = self._llm.reply(transcript)
        logger.info("LLM reply: %s", reply)

        audio, sample_rate = self._tts.synthesize(reply)
        save_wav(output_path, audio, sample_rate)
        logger.info("TTS audio written to %s", output_path)

        logger.info("Pipeline latency: %.2fs", time.perf_counter() - started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT -> LLM -> TTS wiring test")
    parser.add_argument("--input", type=Path, required=True, help="Input speech WAV")
    parser.add_argument("--output", type=Path, default=Path("reply.wav"), help="Output WAV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    CallPipeline(settings).run(args.input, args.output)


if __name__ == "__main__":
    main()
