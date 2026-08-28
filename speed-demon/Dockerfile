# SPEED DEMON uses the packaged vLLM runtime and only replaces the
# FlashInfer attention backend with the tested PR #50885 overlay.
FROM vllm/vllm-openai:v0.28.0

LABEL org.opencontainers.image.title="LocalLLM SPEED DEMON"
LABEL org.opencontainers.image.description="Qwen3.8 AWQ + DFlash2 with FlashInfer full decode graphs"
LABEL org.opencontainers.image.source="https://github.com/Pukerud/LocalLLM"

COPY vllm/v1/attention/backends/flashinfer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py
