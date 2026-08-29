# ExLlamaV3/TabbyAPI image for the Qwen3.8 EXL3 vision quant.
# The public TabbyAPI image currently lags ExLlamaV3 1.4.4, which is the
# minimum version required by the quantized-vision model branch.
FROM ghcr.io/theroyallab/tabbyapi@sha256:27744353387eefefb6a3ed91a5bde99def32debe413b4a4fca1d53dc94d87859

USER root

# Triton compiles a small CUDA helper on first use and needs Python.h.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Pin the wheel to the CUDA 12.8 / Torch 2.9.0 combination in the base image.
RUN /opt/venv/bin/pip install --no-cache-dir --force-reinstall --no-deps \
    "https://github.com/turboderp-org/exllamav3/releases/download/v1.4.4/exllamav3-1.4.4%2Bcu128.torch2.9.0-cp312-cp312-linux_x86_64.whl" \
    && /opt/venv/bin/python -c "from importlib.metadata import version; assert version('exllamav3') == '1.4.4+cu128.torch2.9.0', version('exllamav3')"

LABEL org.opencontainers.image.title="LocalLLM ExLlamaV3 Qwen3.8" \
      org.opencontainers.image.description="TabbyAPI with ExLlamaV3 1.4.4 for Qwen3.8 EXL3 vision"
