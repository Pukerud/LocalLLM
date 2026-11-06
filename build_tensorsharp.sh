#!/usr/bin/env bash
set -Eeuo pipefail
[[ $(id -u) == 1000 ]]
base=/home/user/.local/share/localllm-tensorsharp
cd "$base/source-9d18a64dbf07"
export DOTNET_ROOT="$base/dotnet-10.0.403" DOTNET_CLI_HOME="$base/dotnet-home"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 DOTNET_NOLOGO=1
export NUGET_PACKAGES="$base/nuget-packages"
export CUDA_HOME="$base/cuda-13.0.2" CUDA_PATH="$base/cuda-13.0.2" CUDACXX="$base/cuda-13.0.2/bin/nvcc"
export PATH="$DOTNET_ROOT:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$CUDA_HOME/lib"
export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF
export TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES=86-real TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=4
# Upstream Docker's portable optional direct-CUDA PTX. The served backend is
# strictly GGML CUDA, whose actual native library above is compiled for SM86.
# The initial optional SM86 fatbin failed on upstream unguarded SM89 FP8 asm;
# no kernel/numerical code is patched to hide that failure.
export TENSORSHARP_CUDA_ARCH=compute_75 TENSORSHARP_CUDNN=OFF
export TENSORSHARP_GGML_GIT_REF=57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234 GIT_TERMINAL_PROMPT=0
[[ $(git rev-parse HEAD) == 9d18a64dbf07efa7b0fb5b7d34dda97f05e2c368 ]]
"$DOTNET_ROOT/dotnet" --version
"$CUDACXX" --version
"$DOTNET_ROOT/dotnet" build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -m:3 \
 -p:TensorSharpSkipMlxNative=true -p:CudaArch=compute_75 -p:CudaEmitFatbin=false -p:CudaEnableFastMath=false
