# llama.cpp patches (base: 6b0f75b)

Apply on a clone, never on contio's tree (`~/.local/share/contio-spike/src/llama.cpp`):

    git clone --shared ~/.local/share/contio-spike/src/llama.cpp ~/.cache/xdna-relay/llama.cpp
    cd ~/.cache/xdna-relay/llama.cpp && git checkout -b xdna-relay 6b0f75b
    git apply <repo>/patches/llama.cpp/*.patch
    cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
    # the UI asset step downloads from HF; copy them from contio's build instead:
    cp -r ~/.local/share/contio-spike/src/llama.cpp/build-vulkan/tools/ui/dist* build-vulkan/tools/ui/
    cmake --build build-vulkan -j 8 --target llama-server llama-bench

## 0001 — draft-simple on hybrid (recurrent) models without replay
Stock: `draft-simple` gets no recurrent snapshots (`n_rs_seq = 0`), so every partial
rejection restores a ~70 MB checkpoint and replays the accepted tokens on the target.
Patch: target and standalone draft get `n_rs_seq = n_max`; the draft keeps its checkpoint
for undoing its own multi-batch drafting. Measured +5–11% over stock, outputs identical to
stock speculation (see `bench/results/2026-11-24-spec-igpu.md`).
