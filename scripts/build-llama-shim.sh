#!/usr/bin/env bash
# Build native/llama_shim against our patched llama.cpp clone (shared Vulkan build).
# Output: build/native/libllama_shim.so (rpath to the llama.cpp shared libs).
set -euo pipefail
LC=${LLAMA_CPP_DIR:-$HOME/.cache/xdna-relay/llama.cpp}
LIB=$LC/build-vulkan-shared/bin
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/build/native"
g++ -O2 -std=c++17 -shared -fPIC "$ROOT/native/llama_shim/llama_shim.cpp" \
    -I"$LC/include" -I"$LC/ggml/include" -L"$LIB" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LIB" -o "$ROOT/build/native/libllama_shim.so"
echo "built $ROOT/build/native/libllama_shim.so"
