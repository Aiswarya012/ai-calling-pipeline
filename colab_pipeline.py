from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from kokoro import KPipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

STT_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
STT_LANGUAGE = "hi"
STT_DECODING = "rnnt"
STT_SAMPLE_RATE = 16_000

EMBED_MODEL_ID = "intfloat/multilingual-e5-small"
KNOWLEDGE_PATH = Path(__file__).parent / "knowledge.md"
RAG_TOP_K = 3

LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_NEW_TOKENS = 200

TTS_LANG_CODE = "h"
TTS_VOICE = "hf_alpha"
TTS_SAMPLE_RATE = 24_000

SYSTEM_PROMPT = (
    "You are a polite Hindi-speaking assistant for Sharma Motors car showroom. "
    "Always reply in natural, conversational Hindi. Keep replies short, like a phone call. "
    "Answer only using the company information provided below. If the information is not "
    "there, politely say you will check and get back. Write times in words like 'चार बजे'."
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


class KnowledgeRetriever:
    def __init__(self, path: Path) -> None:
        logger.info("Loading embedding model %s", EMBED_MODEL_ID)
        self._model = SentenceTransformer(EMBED_MODEL_ID)
        self._chunks = self._load_chunks(path)
        self._embeddings = self._model.encode(
            [f"passage: {chunk}" for chunk in self._chunks], normalize_embeddings=True
        )
        logger.info("Indexed %d knowledge chunks", len(self._chunks))

    @staticmethod
    def _load_chunks(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        return [c for c in chunks if not c.startswith("#")]

    def retrieve(self, query: str, k: int = RAG_TOP_K) -> list[str]:
        query_embedding = self._model.encode([f"query: {query}"], normalize_embeddings=True)
        scores = (self._embeddings @ query_embedding.T).ravel()
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._chunks[i] for i in top_indices]


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
    def reply(self, user_text: str, context: list[str]) -> str:
        system = SYSTEM_PROMPT
        if context:
            system += "\n\nCompany information:\n" + "\n".join(f"- {c}" for c in context)
        messages = [
            {"role": "system", "content": system},
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
        self._kb = KnowledgeRetriever(KNOWLEDGE_PATH)
        self._llm = OpenSourceLLM()
        self._tts = KokoroTTS()

    def run(self, input_path: Path, output_path: Path) -> None:
        started = time.perf_counter()

        transcript = self._stt.transcribe(input_path)
        logger.info("STT transcript: %s", transcript)

        context = self._kb.retrieve(transcript)
        logger.info("Retrieved context: %s", context)

        reply = self._llm.reply(transcript, context)
        logger.info("LLM reply: %s", reply)

        audio, sample_rate = self._tts.synthesize(reply)
        sf.write(str(output_path), audio, sample_rate)
        logger.info("TTS audio written to %s", output_path)

        logger.info("Pipeline latency: %.2fs", time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="STT -> RAG -> LLM -> TTS wiring test")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reply.wav"))
    args = parser.parse_args()
    CallPipeline().run(args.input, args.output)


if __name__ == "__main__":
    main()
