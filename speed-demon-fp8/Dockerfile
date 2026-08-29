# FP8 DFlash2 support from vLLM PR #53122, layered on the tested
# FlashInfer full-decode graph image. Keep this image separate from the
# production BF16-draft image so the fallback remains one environment switch.
FROM localllm/speed-demon:vllm-0.28.0-flashinfer-50885

LABEL org.opencontainers.image.title="LocalLLM SPEED DEMON FP8"
LABEL org.opencontainers.image.description="Qwen3.8 AWQ + FP8 DFlash2 with FlashInfer and vLLM PR #53122"
LABEL org.opencontainers.image.source="https://github.com/Pukerud/LocalLLM"

COPY vllm-only.patch /tmp/vllm-only.patch
RUN command -v patch \
    && patch --batch --forward -p1 \
       -d /usr/local/lib/python3.12/dist-packages < /tmp/vllm-only.patch \
    && rm -f /tmp/vllm-only.patch
