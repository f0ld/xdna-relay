// xdna-relay: a plain-C shim over libllama for the Python iGPU executor.
//
// One model + one context + one sequence (seq 0) per handle. Only C scalar types and
// pointers cross the boundary, so ctypes never mirrors llama.cpp's structs.
// Built by scripts/build-llama-shim.sh against ~/.cache/xdna-relay/llama.cpp (base 6b0f75b).

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Handle {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_batch         batch{};
    int32_t             n_batch = 0;
    int32_t             n_vocab = 0;
    int32_t             n_out   = 0;  // rows with logits in the last eval
    int32_t             out0    = 0;  // batch index of the first output row (rows are contiguous)
};

bool g_quiet = true;

void log_cb(ggml_log_level level, const char * text, void *) {
    if (!g_quiet || level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

}  // namespace

extern "C" {

void rl_init(int quiet) {
    g_quiet = quiet != 0;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
}

// n_gpu_layers: -1 = all. flash: -1 auto, 0 off, 1 on. n_rs_seq: recurrent snapshots (hybrid models).
void * rl_load(const char * path, int n_gpu_layers, int n_ctx, int n_batch, int n_rs_seq, int flash) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.progress_callback = [](float, void *) { return true; };  // no progress dots
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        return nullptr;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) n_batch;
    cp.n_ubatch        = (uint32_t) n_batch;
    cp.n_seq_max       = 1;
    cp.n_rs_seq        = (uint32_t) n_rs_seq;
    cp.flash_attn_type = (llama_flash_attn_type) flash;
    cp.no_perf         = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return nullptr;
    }
    auto * h    = new Handle;
    h->model    = model;
    h->ctx      = ctx;
    h->vocab    = llama_model_get_vocab(model);
    h->n_batch  = n_batch;
    h->n_vocab  = llama_vocab_n_tokens(h->vocab);
    h->batch    = llama_batch_init(n_batch, 0, 1);
    return h;
}

void rl_free(void * p) {
    auto * h = static_cast<Handle *>(p);
    if (!h) {
        return;
    }
    llama_batch_free(h->batch);
    llama_free(h->ctx);
    llama_model_free(h->model);
    delete h;
}

int rl_n_vocab(void * p) { return static_cast<Handle *>(p)->n_vocab; }

int rl_tokenize(void * p, const char * text, int len, int * out, int max, int add_special, int parse_special) {
    auto * h = static_cast<Handle *>(p);
    return llama_tokenize(h->vocab, text, len, out, max, add_special != 0, parse_special != 0);
}

int rl_piece(void * p, int tok, char * buf, int len) {
    auto * h = static_cast<Handle *>(p);
    return llama_token_to_piece(h->vocab, tok, buf, len, 0, false);
}

int rl_is_eog(void * p, int tok) {
    return llama_vocab_is_eog(static_cast<Handle *>(p)->vocab, tok) ? 1 : 0;
}

// Append tokens at positions pos0.. to seq 0. all_logits: every row of the LAST chunk gets
// logits (then n must be <= n_batch), else only the last token. Returns llama_decode's code
// (0 ok), or -100 for a bad request.
int rl_eval(void * p, const int * toks, int n, int pos0, int all_logits) {
    auto * h = static_cast<Handle *>(p);
    if (n <= 0 || (all_logits && n > h->n_batch)) {
        return -100;
    }
    for (int beg = 0; beg < n; beg += h->n_batch) {
        const int m = (n - beg < h->n_batch) ? n - beg : h->n_batch;
        const bool last_chunk = beg + m == n;
        llama_batch & b = h->batch;
        b.n_tokens = m;
        for (int i = 0; i < m; ++i) {
            b.token[i]     = toks[beg + i];
            b.pos[i]       = pos0 + beg + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = last_chunk && (all_logits || i == m - 1);
        }
        const int rc = llama_decode(h->ctx, b);
        if (rc != 0) {
            return rc;
        }
        h->n_out = last_chunk ? (all_logits ? m : 1) : 0;
        h->out0  = all_logits ? 0 : m - 1;  // llama_get_logits_ith takes the batch index
    }
    return 0;
}

// Greedy readout of the last eval's rows: argmax and (top1 - top2) logit gap per row.
int rl_argmax_rows(void * p, int * toks, float * gaps) {
    auto * h = static_cast<Handle *>(p);
    for (int r = 0; r < h->n_out; ++r) {
        const float * l = llama_get_logits_ith(h->ctx, h->out0 + r);
        int   best = 0;
        float b1 = l[0], b2 = -1e30f;
        for (int v = 1; v < h->n_vocab; ++v) {
            if (l[v] > b1) {
                b2 = b1; b1 = l[v]; best = v;
            } else if (l[v] > b2) {
                b2 = l[v];
            }
        }
        toks[r] = best;
        gaps[r] = b1 - b2;
    }
    return h->n_out;
}

// Copy one row of logits (row index within the last eval's output rows).
int rl_logits_row(void * p, int row, float * out) {
    auto * h = static_cast<Handle *>(p);
    if (row < 0 || row >= h->n_out) {
        return -1;
    }
    std::memcpy(out, llama_get_logits_ith(h->ctx, h->out0 + row), sizeof(float) * h->n_vocab);
    return 0;
}

// Keep the first n_keep positions of seq 0. Returns 1 on success, 0 if the memory cannot
// remove a partial range (recurrent models without enough n_rs_seq snapshots).
int rl_truncate(void * p, int n_keep) {
    auto * h = static_cast<Handle *>(p);
    return llama_memory_seq_rm(llama_get_memory(h->ctx), 0, n_keep, -1) ? 1 : 0;
}

void rl_clear(void * p) {
    llama_memory_clear(llama_get_memory(static_cast<Handle *>(p)->ctx), true);
}

int rl_pos_max(void * p) {
    return llama_memory_seq_pos_max(llama_get_memory(static_cast<Handle *>(p)->ctx), 0);
}

void rl_synchronize(void * p) { llama_synchronize(static_cast<Handle *>(p)->ctx); }

// Whole-sequence state (KV cache, recurrent state) of seq 0, for the host-RAM prefix cache.
size_t rl_state_size(void * p) {
    return llama_state_seq_get_size(static_cast<Handle *>(p)->ctx, 0);
}

size_t rl_state_get(void * p, uint8_t * dst, size_t size) {
    return llama_state_seq_get_data(static_cast<Handle *>(p)->ctx, dst, size, 0);
}

// Replaces seq 0 with the snapshot. Returns bytes read (0 = failure).
size_t rl_state_set(void * p, const uint8_t * src, size_t size) {
    auto * h = static_cast<Handle *>(p);
    llama_memory_seq_rm(llama_get_memory(h->ctx), 0, -1, -1);
    return llama_state_seq_set_data(h->ctx, src, size, 0);
}

}  // extern "C"
