// xdna-relay: a plain-C shim over libllama for the Python iGPU executor.
//
// One model + one context per handle; one sequence (seq 0, the rl_* calls) or several
// (rl_load_multi + the *_seq calls and rl_decode_multi, for batched decode across requests). Only C scalar types and
// pointers cross the boundary, so ctypes never mirrors llama.cpp's structs.
// Built by scripts/build-llama-shim.sh against ~/.cache/xdna-relay/llama.cpp (base 6b0f75b).

#include "llama-ext.h"  // llama_set_embeddings_nextn / llama_get_embeddings_nextn(_ith) (fork staging API)
#include <cstring>
#include <vector>
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// MTP drafting (T-043), as llama.cpp's common_speculative_impl_draft_mtp for single-head
// models (qwen35 / qwen35moe): a second context of the same model runs only the MTP layer.
// Its input at position p is token x_p paired with the target's hidden state at p-1, so
// after every target batch the MTP context processes the same tokens (keeping its KV
// current), and a draft step feeds back its own hidden state.
struct Mtp {
    llama_context *   ctx   = nullptr;
    llama_batch_ext * batch = nullptr;
    int32_t           n_embd = 0;
    std::vector<std::vector<float>> pending;   // [seq] hidden state of the last processed token
    std::vector<std::vector<float>> verify;    // [seq] rows of the last target batch
    std::vector<int32_t>            verify_pos0, verify_rows;
    std::vector<float>              hbuf;      // input rows of one MTP batch
};

struct Mtp;

struct Handle {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    Mtp *               mtp   = nullptr;  // the model's own next-token (MTP) head as drafter
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
// n_seq: sequences the context holds (KV shared across them when kv_unified: n_ctx is the total).
void * rl_load_multi(const char * path, int n_gpu_layers, int n_ctx, int n_batch, int n_rs_seq, int flash,
                     int n_seq, int kv_unified, int load_mtp) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.load_mtp     = load_mtp != 0;  // the nextn (MTP) layers, for rl_mtp_enable
    mp.progress_callback = [](float, void *) { return true; };  // no progress dots
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        return nullptr;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) n_batch;
    cp.n_ubatch        = (uint32_t) n_batch;
    cp.n_seq_max       = (uint32_t) (n_seq < 1 ? 1 : n_seq);
    cp.kv_unified      = kv_unified != 0;
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

void * rl_load(const char * path, int n_gpu_layers, int n_ctx, int n_batch, int n_rs_seq, int flash) {
    return rl_load_multi(path, n_gpu_layers, n_ctx, n_batch, n_rs_seq, flash, 1, 0, 0);
}

void rl_free(void * p) {
    auto * h = static_cast<Handle *>(p);
    if (!h) {
        return;
    }
    if (h->mtp) {
        llama_batch_ext_free(h->mtp->batch);
        llama_free(h->mtp->ctx);
        delete h->mtp;
        h->mtp = nullptr;
    }
    llama_batch_free(h->batch);
    llama_free(h->ctx);
    llama_model_free(h->model);
    delete h;
}

int rl_n_vocab(void * p) { return static_cast<Handle *>(p)->n_vocab; }

// 0 = attention only (KV truncates freely), 1 = recurrent, 2 = hybrid (recurrent + attention):
// recurrent state rolls back only within the n_rs_seq snapshots of the last ubatch.
int rl_model_kind(void * p) {
    auto * h = static_cast<Handle *>(p);
    if (llama_model_is_hybrid(h->model))    return 2;
    if (llama_model_is_recurrent(h->model)) return 1;
    return 0;
}

// The GGUF's chat template (tokenizer.chat_template), or nullptr.
const char * rl_chat_template(void * p) {
    return llama_model_chat_template(static_cast<Handle *>(p)->model, nullptr);
}

int rl_n_rs_seq(void * p) { return (int) llama_n_rs_seq(static_cast<Handle *>(p)->ctx); }

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

// After a target decode of batch b: feed the MTP context the same tokens, each with the
// target's hidden state of the previous position (the previous call's last row for a
// sequence's first token here); keep each sequence's rows for rollback. Batches must group
// a sequence's tokens contiguously (all ours do). Returns llama_process's code.
static int mtp_process(Handle * h, const llama_batch & b) {
    Mtp * m = h->mtp;
    if (!m || b.n_tokens <= 0) {
        return 0;
    }
    const int n = b.n_tokens, E = m->n_embd;
    const float * tgt = llama_get_embeddings_nextn(h->ctx);
    if (!tgt) {
        return -201;
    }
    m->hbuf.resize((size_t) n * E);
    llama_batch_ext_clear(m->batch);
    for (int k = 0; k < n; ++k) {
        const int seq = b.seq_id[k][0];
        const bool first = k == 0 || b.seq_id[k - 1][0] != seq;
        const float * src = first ? m->pending[seq].data() : tgt + (size_t) (k - 1) * E;
        std::memcpy(m->hbuf.data() + (size_t) k * E, src, (size_t) E * sizeof(float));
    }
    for (int k = 0; k < n; ++k) {
        const int32_t idx = llama_batch_ext_add_token(m->batch, b.seq_id[k][0], b.token[k]);
        if (idx < 0) {
            return -202;
        }
        const llama_pos pos = b.pos[k];
        llama_batch_ext_set_pos(m->batch, idx, &pos);
        llama_batch_ext_set_embd_state(m->batch, idx, { m->hbuf.data() + (size_t) k * E, 1, (size_t) E });
    }
    const int rc = llama_process(m->ctx, LLAMA_PROCESS_TYPE_DECODE, m->batch);
    if (rc != 0) {
        return rc;
    }
    for (int k = 0; k < n; ) {  // per sequence: its rows in this batch, the last one pending
        const int seq = b.seq_id[k][0];
        int e = k;
        while (e + 1 < n && b.seq_id[e + 1][0] == seq) {
            ++e;
        }
        const int rows = e - k + 1;
        m->verify[seq].assign(tgt + (size_t) k * E, tgt + (size_t) (e + 1) * E);
        m->verify_pos0[seq] = b.pos[k];
        m->verify_rows[seq] = rows;
        m->pending[seq].assign(tgt + (size_t) e * E, tgt + (size_t) (e + 1) * E);
        k = e + 1;
    }
    return 0;
}

// Append tokens at positions pos0.. to seq 0. all_logits: every row of the LAST chunk gets
// logits (then n must be <= n_batch), else only the last token. Returns llama_decode's code
// (0 ok), or -100 for a bad request.
int rl_eval_seq(void * p, int seq, const int * toks, int n, int pos0, int all_logits) {
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
            b.seq_id[i][0] = seq;
            b.logits[i]    = last_chunk && (all_logits || i == m - 1);
        }
        const int rc = llama_decode(h->ctx, b);
        if (rc != 0) {
            return rc;
        }
        if (const int rm = mtp_process(h, b)) {
            return rm;
        }
        h->n_out = last_chunk ? (all_logits ? m : 1) : 0;
        h->out0  = all_logits ? 0 : m - 1;  // llama_get_logits_ith takes the batch index
    }
    return 0;
}

int rl_eval(void * p, const int * toks, int n, int pos0, int all_logits) {
    return rl_eval_seq(p, 0, toks, n, pos0, all_logits);
}

// One decode step for several sequences: token i at position pos[i] of sequence seqs[i],
// every row with logits (row i = entry i). n <= n_batch. Returns llama_decode's code.
int rl_decode_multi(void * p, int n, const int * toks, const int * pos, const int * seqs) {
    auto * h = static_cast<Handle *>(p);
    if (n <= 0 || n > h->n_batch) {
        return -100;
    }
    llama_batch & b = h->batch;
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos[i];
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seqs[i];
        b.logits[i]    = 1;
    }
    int rc = llama_decode(h->ctx, b);
    if (rc == 0) {
        rc = mtp_process(h, b);
    }
    if (rc == 0) {
        h->n_out = n;
        h->out0  = 0;
    }
    return rc;
}

// A mixed batch: entry i is token toks[i] at position pos[i] of sequence seqs[i], with
// logits where want[i] != 0 (several entries may belong to one sequence: a prefill chunk;
// decode tokens of other sequences ride along). n <= n_batch. Read rows with rl_argmax_at /
// rl_logits_at by batch index.
int rl_decode_batch(void * p, int n, const int * toks, const int * pos, const int * seqs, const int * want) {
    auto * h = static_cast<Handle *>(p);
    if (n <= 0 || n > h->n_batch) {
        return -100;
    }
    llama_batch & b = h->batch;
    b.n_tokens = n;
    int n_out = 0;
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos[i];
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seqs[i];
        b.logits[i]    = want[i] != 0;
        n_out += want[i] != 0;
    }
    int rc = llama_decode(h->ctx, b);
    if (rc == 0) {
        rc = mtp_process(h, b);
    }
    h->n_out = 0;  // rows are read by batch index, not as a contiguous run
    return rc == 0 ? n_out : rc;
}

// Greedy readout of batch indices idx[0..n): argmax and top1-top2 gap each.
int rl_argmax_at(void * p, int n, const int * idx, int * toks, float * gaps) {
    auto * h = static_cast<Handle *>(p);
    for (int r = 0; r < n; ++r) {
        const float * l = llama_get_logits_ith(h->ctx, idx[r]);
        if (!l) {
            return -1;
        }
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
    return n;
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

// ---- per-sequence variants -------------------------------------------------------------

// MTP: a sequence cut to n_keep tokens continues from the hidden state of token n_keep-1,
// a row of the last target batch when the cut falls inside it (a rejected draft), else
// unknown (zeros: drafts degrade until the next batch, output stays exact).
static void mtp_truncate(Handle * h, int seq, int n_keep) {
    Mtp * m = h->mtp;
    if (!m) {
        return;
    }
    llama_memory_seq_rm(llama_get_memory(m->ctx), seq, n_keep < 0 ? -1 : n_keep, -1);
    const int r = n_keep - 1 - m->verify_pos0[seq];
    if (n_keep > 0 && r >= 0 && r < m->verify_rows[seq]) {
        const float * row = m->verify[seq].data() + (size_t) r * m->n_embd;
        m->pending[seq].assign(row, row + m->n_embd);
    } else if (n_keep <= 0 || r < 0) {  // before the last batch: hidden state unknown
        std::fill(m->pending[seq].begin(), m->pending[seq].end(), 0.0f);
        m->verify_rows[seq] = 0;
    }  // else at or past the last batch's end: pending already holds the right row
}

int rl_seq_truncate(void * p, int seq, int n_keep) {
    auto * h = static_cast<Handle *>(p);
    const bool ok = llama_memory_seq_rm(llama_get_memory(h->ctx), seq, n_keep, -1);
    if (ok) {
        mtp_truncate(h, seq, n_keep);
    }
    return ok ? 1 : 0;
}

// Enable MTP drafting on a context loaded with rl_load_multi (the GGUF must carry nextn
// layers). Returns 0, or -1 (no MTP head / context failed).
int rl_mtp_enable(void * p) {
    auto * h = static_cast<Handle *>(p);
    if (h->mtp) {
        return 0;
    }
    if (llama_model_n_layer_nextn(h->model) < 1) {
        return -1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = llama_n_ctx(h->ctx);
    cp.n_batch         = (uint32_t) h->n_batch;
    cp.n_ubatch        = (uint32_t) h->n_batch;
    cp.n_seq_max       = llama_n_seq_max(h->ctx);
    cp.kv_unified      = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf         = true;
    cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cp.ctx_other       = h->ctx;
    llama_context * ctx = llama_init_from_model(h->model, cp);
    if (!ctx) {
        return -1;
    }
    auto * m   = new Mtp;
    m->ctx     = ctx;
    m->batch   = llama_batch_ext_init(ctx);
    m->n_embd  = llama_model_n_embd_out(h->model);
    const int ns = (int) llama_n_seq_max(h->ctx);
    m->pending.assign(ns, std::vector<float>(m->n_embd, 0.0f));
    m->verify.assign(ns, {});
    m->verify_pos0.assign(ns, 0);
    m->verify_rows.assign(ns, 0);
    llama_set_embeddings_nextn(h->ctx, true, /*masked*/ false);
    llama_set_embeddings_nextn(ctx, true, /*masked*/ true);
    h->mtp = m;
    return 0;
}

// Greedy MTP drafts for n sequences: sequence seqs[j] continues from token last[j] at
// position pos0[j] (the token the target has not evaluated yet) and drafts ks[j] tokens
// (<= kmax) into out[j*kmax ..]. Steps are batched across sequences. The MTP context's
// draft positions are removed afterwards (the target's verify batch rewrites them).
// Returns 0 or a llama_process code.
int rl_mtp_draft(void * p, int n, const int * seqs, const int * last, const int * pos0, const int * ks,
                 int * out, int kmax) {
    auto * h = static_cast<Handle *>(p);
    Mtp * m = h->mtp;
    if (!m || n <= 0) {
        return -100;
    }
    const int E = m->n_embd;
    std::vector<std::vector<float>> cur(n);
    std::vector<int> tok(last, last + n), idx(n, -1);
    int steps = 0;
    for (int j = 0; j < n; ++j) {
        cur[j] = m->pending[seqs[j]];
        steps = ks[j] > steps ? ks[j] : steps;
    }
    int rc = 0;
    for (int i = 0; i < steps && rc == 0; ++i) {
        llama_batch_ext_clear(m->batch);
        for (int j = 0; j < n; ++j) {
            idx[j] = -1;
            if (ks[j] <= i) {
                continue;
            }
            idx[j] = llama_batch_ext_add_token(m->batch, seqs[j], tok[j]);
            const llama_pos pos = pos0[j] + i;
            llama_batch_ext_set_pos(m->batch, idx[j], &pos);
            llama_batch_ext_set_embd_state(m->batch, idx[j], { cur[j].data(), 1, (size_t) E });
            llama_batch_ext_set_output_logits(m->batch, idx[j], true);
        }
        rc = llama_process(m->ctx, LLAMA_PROCESS_TYPE_DECODE, m->batch);
        if (rc != 0) {
            break;
        }
        for (int j = 0; j < n; ++j) {
            if (idx[j] < 0) {
                continue;
            }
            const float * l = llama_get_logits_ith(m->ctx, idx[j]);
            int best = 0;
            for (int v = 1; v < h->n_vocab; ++v) {
                if (l[v] > l[best]) {
                    best = v;
                }
            }
            out[j * kmax + i] = best;
            tok[j] = best;
            const float * hr = llama_get_embeddings_nextn_ith(m->ctx, idx[j]);
            cur[j].assign(hr, hr + E);
        }
    }
    for (int j = 0; j < n; ++j) {
        if (ks[j] > 0) {
            llama_memory_seq_rm(llama_get_memory(m->ctx), seqs[j], pos0[j], -1);
        }
    }
    return rc;
}

int rl_seq_pos_max(void * p, int seq) {
    return llama_memory_seq_pos_max(llama_get_memory(static_cast<Handle *>(p)->ctx), seq);
}

size_t rl_state_seq_size(void * p, int seq) {
    return llama_state_seq_get_size(static_cast<Handle *>(p)->ctx, seq);
}

size_t rl_state_seq_get(void * p, int seq, uint8_t * dst, size_t size) {
    return llama_state_seq_get_data(static_cast<Handle *>(p)->ctx, dst, size, seq);
}

size_t rl_state_seq_set(void * p, int seq, const uint8_t * src, size_t size) {
    auto * h = static_cast<Handle *>(p);
    llama_memory_seq_rm(llama_get_memory(h->ctx), seq, -1, -1);
    mtp_truncate(h, seq, 0);  // restored sequences start without MTP context (TODO: snapshot it)
    return llama_state_seq_set_data(h->ctx, src, size, seq);
}

}  // extern "C"
