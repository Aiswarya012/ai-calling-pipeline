# AI Calling Pipeline — Full Chain Test (Open-Source)

End-to-end test of the Deepgram + GPT replacement, all open-source:

**Voice in → IndicConformer (STT) → Qwen2.5-7B (LLM) → Kokoro (TTS) → Voice out**

A Hindi speech clip goes in; a spoken Hindi reply comes out. No telephony, no paid APIs.

## Run in Google Colab

1. Open a new Colab notebook, set **Runtime → Change runtime type → T4 GPU**.
2. Clone this repo and open the notebook:

```python
!git clone https://github.com/Aiswarya012/ai-calling-pipeline.git
%cd ai-calling-pipeline
```

3. Run `run_in_colab.ipynb` (or paste its cells). It will:
   - install dependencies,
   - log in to HuggingFace (needed for the gated IndicConformer model),
   - generate a Hindi test clip with Kokoro,
   - run the full chain and play the spoken reply.

### One-time setup for IndicConformer (gated model)
- Accept terms at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
- Create a Read token at https://huggingface.co/settings/tokens and paste it into the `login()` cell.

## What you'll see
The logs print the **STT transcript**, the **LLM reply (Hindi)**, and the **end-to-end latency**, then `reply.wav` plays.

## Notes
- LLM runs 4-bit quantized to fit a free Colab T4 (~16 GB).
- STT 16 kHz, Kokoro 24 kHz — resampling handled in code.
- To try a different LLM, change `LLM_MODEL_ID` in `colab_pipeline.py` (e.g. a Sarvam model).
