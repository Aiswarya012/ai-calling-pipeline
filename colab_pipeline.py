from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from kokoro import KPipeline
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

STT_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
STT_LANGUAGE = "hi"
STT_DECODING = "rnnt"
STT_SAMPLE_RATE = 16_000

LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LLM_MAX_NEW_TOKENS = 200

TTS_LANG_CODE = "h"
TTS_VOICE = "hf_alpha"
TTS_SAMPLE_RATE = 24_000

SYSTEM_PROMPT = (
    "You are a polite Hindi-speaking car showroom assistant. "
    "Always reply in natural, conversational Hindi. Keep replies short, "
    "like a phone call. Showroom hours are Monday to Saturday, 10 AM to 7 PM. "
    "You can help book test-drive and sales appointments."
)


def load_mono(path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


class IndicConformerSTT:
    def __init__(self, device: str) -> None:
        logger.info("Loading STT model %s", STT_MODEL_ID)
        self._device = device
        self._model = AutoModel.from_pretrained(STT_MODEL_ID, trust_remote_code=True).to(device)
        self._model.eval()

    @torch.inference_mode()
    def transcribe(self, audio_path: Path) -> str:
        waveform = load_mono(audio_path, STT_SAMPLE_RATE).to(self._device)
        text = self._model(waveform, STT_LANGUAGE, STT_DECODING)
        return text.strip() if isinstance(text, str) else str(text).strip()


class OpenSourceLLM:
    def __init__(self) -> None:
        logger.info("Loading LLM %s (4-bit)", LLM_MODEL_ID)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, quantization_config=quant, device_map="auto"
        )
        self._model.eval()

    @torch.inference_mode()
    def reply(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        generated = self._model.generate(
            **inputs, max_new_tokens=LLM_MAX_NEW_TOKENS, do_sample=True, temperature=0.4
        )
        output = generated[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(output, skip_special_tokens=True).strip()


class KokoroTTS:
    def __init__(self) -> None:
        logger.info("Loading TTS pipeline lang_code=%s", TTS_LANG_CODE)
        self._pipeline = KPipeline(lang_code=TTS_LANG_CODE)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks = [audio for _, _, audio in self._pipeline(text, voice=TTS_VOICE)]
        if not chunks:
            raise ValueError("TTS produced no audio")
        return np.concatenate(chunks), TTS_SAMPLE_RATE


class CallPipeline:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device %s", device)
        self._stt = IndicConformerSTT(device)
        self._llm = OpenSourceLLM()
        self._tts = KokoroTTS()

    def run(self, input_path: Path, output_path: Path) -> None:
        started = time.perf_counter()

        transcript = self._stt.transcribe(input_path)
        logger.info("STT transcript: %s", transcript)

        reply = self._llm.reply(transcript)
        logger.info("LLM reply: %s", reply)

        audio, sample_rate = self._tts.synthesize(reply)
        sf.write(str(output_path), audio, sample_rate)
        logger.info("TTS audio written to %s", output_path)

        logger.info("Pipeline latency: %.2fs", time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="STT -> open-source LLM -> TTS wiring test")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reply.wav"))
    args = parser.parse_args()
    CallPipeline().run(args.input, args.output)


if __name__ == "__main__":
    main()
