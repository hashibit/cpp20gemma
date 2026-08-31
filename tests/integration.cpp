// Integration test: end-to-end greedy decode vs the Python reference.
// 64 tokens must match e2e_tokens.bin exactly; final logits must be within
// tolerance of e2e_last_logits.bin. On a token mismatch the near-tie margin
// (logit[ref] - logit[own]) is printed — a |margin| < 2e-4 is a benign
// float32 accumulation-order near-tie, anything larger is a real bug.
//
// Usage: ./build/integration weights/tiny_weights.bin weights/tiny_tokenizer.bin \
//                tests/golden/e2e_tokens.bin tests/golden/e2e_last_logits.bin
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "config.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "weights.h"

using namespace gemma;

static const char* kPrompt = "The meaning of life";
static const uint32_t kN = 64;

static std::vector<uint32_t> read_u32(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "integration: cannot open %s\n", path);
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v((size_t)n / 4);
    std::fread(v.data(), 1, (size_t)n, f);
    std::fclose(f);
    return v;
}

static std::vector<float> read_f32(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "integration: cannot open %s\n", path);
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t)n / 4);
    std::fread(v.data(), 1, (size_t)n, f);
    std::fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <weights> <tokenizer> <e2e_tokens.bin> <e2e_last_logits.bin>\n", argv[0]);
        return 1;
    }

    Weights w;
    if (!w.load(argv[1])) return 1;
    Tokenizer tok;
    if (!tok.load(argv[2])) return 1;
    auto ref_tokens = read_u32(argv[3]);
    auto ref_logits = read_f32(argv[4]);
    if (ref_tokens.size() != kN) {
        std::fprintf(stderr, "integration: expected %u tokens in %s, got %zu\n", kN, argv[3], ref_tokens.size());
        return 1;
    }

    Model m(w, w.cfg.max_seq_len);
    Sampler sampler(0);

    auto ptoks = tok.encode(kPrompt);
    std::printf("prompt: \"%s\" -> %zu tokens\n", kPrompt, ptoks.size());
    for (uint32_t pos = 0; pos < ptoks.size(); pos++) m.forward_token(ptoks[pos], pos);

    int fails = 0;
    for (uint32_t step = 0; step < kN; step++) {
        auto logits = m.logits_span();
        uint32_t next = sampler.greedy(logits);
        if (next != ref_tokens[step]) {
            float own = logits[next], ref = logits[ref_tokens[step]];
            std::printf("MISMATCH step %u: cpp=%u ref=%u margin=%g%s\n",
                        step, next, ref_tokens[step], ref - own,
                        std::fabs(ref - own) < 2e-4f ? " (near-tie, benign)" : "");
            fails++;
        }
        m.forward_token(next, (uint32_t)ptoks.size() + step);
    }

    // Final logits (after the last generated token), tolerance check.
    auto logits = m.logits_span();
    if (logits.size() != ref_logits.size()) {
        std::fprintf(stderr, "integration: logits size mismatch (%zu vs %zu)\n", logits.size(), ref_logits.size());
        return 1;
    }
    size_t bad = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        float d = std::fabs(logits[i] - ref_logits[i]);
        if (d > worst) worst = d;
        if (d > 2e-3f + 1e-3f * std::fabs(ref_logits[i])) bad++;
    }
    std::printf("final logits: worst |diff| = %g (%zu/%zu outside tolerance)\n",
                worst, bad, logits.size());
    if (bad > 0) fails++;

    if (fails == 0) {
        std::printf("INTEGRATION PASSED: %u greedy tokens match the Python reference exactly\n", kN);
        return 0;
    }
    std::printf("INTEGRATION FAILED\n");
    return 1;
}
