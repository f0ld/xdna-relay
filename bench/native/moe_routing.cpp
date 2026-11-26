// T-046: routing skew of a MoE GGUF on real text. Evaluates a prompt file in ubatches and
// records every layer's top-k expert choices (cb_eval on "ffn_moe_argsort-<il>", the first k=8 of each row; "ffn_moe_topk" is a view and is never evaluated); prints, per
// window of T tokens (T = 512, 1024, 2048, 4096), the max tokens any expert gets in any layer
// (mean over windows) and the fraction of (window, layer) pairs whose max exceeds 128 / 256.
// Usage: moe_routing MODEL.gguf PROMPT.txt
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Cap { std::map<int, std::vector<int32_t>> sel; int k = 0; };  // layer -> [token * k]

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    if (getenv("MOE_NAMES") && ask && strstr(t->name, "-0") && strstr(t->name, "moe")) fprintf(stderr, "NAME %s %s\n", t->name, ggml_op_desc(t));
    if (strncmp(t->name, "ffn_moe_argsort-", 16) != 0) return ask ? false : true;
    if (ask) return true;
    auto * c = (Cap *) ud;
    const int il = atoi(t->name + 16), k = 8, n = (int) t->ne[1];  // argsort row: top-k first
    c->k = k;
    if (!t->buffer || t->type != GGML_TYPE_I32) { fprintf(stderr, "skip %s\n", t->name); return true; }
    std::vector<int32_t> buf((size_t) k * n);
    for (int i = 0; i < n; ++i)  // a view: the first k of each argsort row
        ggml_backend_tensor_get(t, buf.data() + (size_t) i * k, (size_t) i * t->nb[1], (size_t) k * 4);
    auto & v = c->sel[il];
    v.insert(v.end(), buf.begin(), buf.end());
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model prompt\n", argv[0]); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = getenv("MOE_CPU") ? 0 : 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); std::string text = ss.str();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
    toks.resize(std::min(n, 8192));
    Cap cap;
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192 + 16; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.cb_eval = cb; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    for (size_t b = 0; b < toks.size(); b += 512) {
        int m = (int) std::min<size_t>(512, toks.size() - b);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + b, m))) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    const int k = cap.k, ntok = (int) toks.size();
    if (cap.sel.empty() || k == 0) { fprintf(stderr, "no ffn_moe_topk captured\n"); return 1; }
    printf("tokens %d, layers %zu, k %d\n", ntok, cap.sel.size(), k);
    {   // sanity: distinct experts per row; layer 0's busiest experts over the first 2048 tokens
        auto & v = cap.sel.begin()->second;
        int distinct = 0, T = std::min(ntok, 2048);
        std::vector<int> cnt(256, 0);
        for (int t = 0; t < T; ++t) {
            std::vector<int> r(v.begin() + (size_t) t * k, v.begin() + (size_t) (t + 1) * k);
            for (int e : r) cnt[e]++;
            std::sort(r.begin(), r.end());
            distinct += std::unique(r.begin(), r.end()) == r.end();
        }
        std::vector<int> idx(256); for (int i = 0; i < 256; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        printf("layer0: rows with %d distinct experts %d/%d; busiest:", k, distinct, T);
        for (int i = 0; i < 6; ++i) printf(" e%d=%d", idx[i], cnt[idx[i]]);
        printf("; row0:"); for (int j = 0; j < k; ++j) printf(" %d", v[j]); printf("\n");
    }
    if (const char * dump = getenv("MOE_DUMP")) {  // per-layer counts of the first window
        FILE * fo = fopen(dump, "w");
        for (int T : {1024, 2048, 4096, 8192}) {
            if (T > ntok) break;
            for (auto & [il, v] : cap.sel) {
                if (v.size() < (size_t) ntok * k) continue;
                std::vector<int> cnt(256, 0);
                for (int t = 0; t < T; ++t) for (int j = 0; j < k; ++j) cnt[v[(size_t) t * k + j]]++;
                fprintf(fo, "%d,%d", T, il);
                for (int c : cnt) fprintf(fo, ",%d", c);
                fprintf(fo, "\n");
            }
        }
        fclose(fo);
    }
    // the largest count outside the top H experts (what the tail's capacity must hold)
    printf("T,H,mean_rest_max,worst_rest_max,frac_rest_over_128,frac_rest_over_256,top_H_rows_mean\n");
    for (int T : {1024, 2048, 4096}) {
        if (T > ntok) break;
        for (int H : {4, 8, 12, 16}) {
            double rm = 0, top = 0; int worst = 0, pairs = 0, o128 = 0, o256 = 0;
            for (auto & [il, v] : cap.sel) {
                if (v.size() < (size_t) ntok * k) continue;
                for (int w = 0; w + T <= ntok; w += T) {
                    std::vector<int> cnt(256, 0);
                    for (int t = w; t < w + T; ++t) for (int j = 0; j < k; ++j) cnt[v[(size_t) t * k + j]]++;
                    std::sort(cnt.rbegin(), cnt.rend());
                    int r = cnt[H]; rm += r; worst = std::max(worst, r); ++pairs; o128 += r > 128; o256 += r > 256;
                    for (int i = 0; i < H; ++i) top += cnt[i];
                }
            }
            printf("%d,%d,%.1f,%d,%.3f,%.3f,%.0f\n", T, H, rm / pairs, worst, (double) o128 / pairs, (double) o256 / pairs, top / pairs);
        }
    }
    printf("T,windows,mean_max,worst_max,frac_over_64,frac_over_128,frac_over_256,mean_count,used_experts,units64,units128,units256,top8_share\n");
    for (int T : {512, 1024, 2048, 4096, 8192}) {
        if (T > ntok) break;
        double sum_max = 0, used = 0, u64 = 0, u128 = 0, u256 = 0, top8 = 0; int worst = 0, pairs = 0, o64 = 0, o128 = 0, o256 = 0;
        for (auto & [il, v] : cap.sel) {
            if (v.size() < (size_t) ntok * k) continue;  // the last layer runs on output rows only
            for (int w = 0; w + T <= ntok; w += T) {
                std::vector<int> cnt(256, 0);
                for (int t = w; t < w + T; ++t) for (int j = 0; j < k; ++j) cnt[v[(size_t) t * k + j]]++;
                int mx = *std::max_element(cnt.begin(), cnt.end());
                std::vector<int> sc(cnt); std::sort(sc.rbegin(), sc.rend());
                for (int i = 0; i < 8; ++i) top8 += (double) sc[i] / ((double) T * k);
                for (int c : cnt) if (c) { used += 1; u64 += (c + 63) / 64; u128 += (c + 127) / 128; u256 += (c + 255) / 256; }
                sum_max += mx; worst = std::max(worst, mx); ++pairs;
                o64 += mx > 64; o128 += mx > 128; o256 += mx > 256;
            }
        }
        printf("%d,%d,%.1f,%d,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f\n", T, ntok / T, sum_max / pairs, worst,
               (double) o64 / pairs, (double) o128 / pairs, (double) o256 / pairs, (double) T * k / 256,
               used / pairs, u64 / pairs, u128 / pairs, u256 / pairs, top8 / pairs);
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}
