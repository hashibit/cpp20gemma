// gemma — single-threaded, dependency-free C++20 inference engine for a
// Gemma-3-style model (INT8-quantized weights, NEON/AVX2 SIMD kernels).
//
// CLI flags:
//   ./build/gemma --weights_path W --model_size 270m --n_dec 250 \
//       --minp 0.1 --temp 0.7 --prompt "What is a transformer?" \
//       --terminate_on_eos 1 --chat_format 1
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include "config.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "weights.h"

using namespace gemma;

namespace {

struct Args {
    std::string weights_path;
    std::string tok_path;         // tokenizer file; defaults to weights_path + ".tok"
    std::string model_size;       // optional preset name for validation ("270m"/"1B"/"tiny")
    std::string prompt = "What is a transformer?";
    uint32_t n_dec          = 250;    // max tokens to generate
    double   temp           = 0.7;    // <= 0 -> greedy
    double   min_p          = -1.0;   // < 0 -> greedy (absent --minp means greedy)
    uint32_t max_cache_len  = 8192;   // KV cache cap (positions)
    uint64_t seed           = 0;      // 0 -> seed from clock
    bool     terminate_on_eos = true;
    bool     chat_format      = false;
    bool     dump             = false;  // print config + tensor layout, then exit
};

void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "  --weights_path <path>     model weights file (GMW1 format)\n"
        "  --tok_path <path>         tokenizer file (default: weights_path + \".tok\")\n"
        "  --model_size <name>       preset name for validation: 270m | 1B | tiny\n"
        "  --prompt <text>           input prompt (default: \"What is a transformer?\")\n"
        "  --n_dec <n>               max tokens to generate (default 250)\n"
        "  --temp <t>                temperature; <= 0 selects greedy (default 0.7)\n"
        "  --minp <p>                Min-P threshold; omit for greedy\n"
        "  --max_cache_len <n>       KV cache size cap in positions (default 8192)\n"
        "  --seed <n>                RNG seed for sampling (default: from clock)\n"
        "  --terminate_on_eos <0|1>  stop at EOS token (default 1)\n"
        "  --chat_format <0|1>       wrap prompt in the Gemma chat template (default 0)\n"
        "  --dump                    print model config and tensor layout, then exit\n"
        "  --help                    show this help\n",
        argv0);
}

bool parse_args(int argc, char** argv, Args& a) {
    auto next = [&](int& i, const char* flag) -> const char* {
        if (i + 1 >= argc) { std::fprintf(stderr, "gemma: %s needs a value\n", flag); return nullptr; }
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (arg == "--weights_path")  { const char* v = next(i, arg.c_str()); if (!v) return false; a.weights_path = v; }
        else if (arg == "--tok_path")      { const char* v = next(i, arg.c_str()); if (!v) return false; a.tok_path = v; }
        else if (arg == "--model_size")    { const char* v = next(i, arg.c_str()); if (!v) return false; a.model_size = v; }
        else if (arg == "--prompt")        { const char* v = next(i, arg.c_str()); if (!v) return false; a.prompt = v; }
        else if (arg == "--n_dec")         { const char* v = next(i, arg.c_str()); if (!v) return false; a.n_dec = (uint32_t)std::strtoul(v, nullptr, 10); }
        else if (arg == "--temp")          { const char* v = next(i, arg.c_str()); if (!v) return false; a.temp = std::strtod(v, nullptr); }
        else if (arg == "--minp")          { const char* v = next(i, arg.c_str()); if (!v) return false; a.min_p = std::strtod(v, nullptr); }
        else if (arg == "--max_cache_len") { const char* v = next(i, arg.c_str()); if (!v) return false; a.max_cache_len = (uint32_t)std::strtoul(v, nullptr, 10); }
        else if (arg == "--seed")          { const char* v = next(i, arg.c_str()); if (!v) return false; a.seed = std::strtoull(v, nullptr, 10); }
        else if (arg == "--terminate_on_eos") { const char* v = next(i, arg.c_str()); if (!v) return false; a.terminate_on_eos = std::atoi(v) != 0; }
        else if (arg == "--chat_format")   { const char* v = next(i, arg.c_str()); if (!v) return false; a.chat_format = std::atoi(v) != 0; }
        else if (arg == "--dump")          { a.dump = true; }
        else { std::fprintf(stderr, "gemma: unknown flag %s\n", arg.c_str()); return false; }
    }
    if (a.weights_path.empty()) {
        std::fprintf(stderr, "gemma: --weights_path is required\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// Gemma chat template (mirrors gemma-3n's <start_of_turn> format). The control
// tokens are literal pieces in the vocab, so encoding the wrapped string finds
// them by longest match.
std::string wrap_chat(const std::string& prompt) {
    return "<bos><start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 1;

    Weights weights;
    if (!weights.load(a.weights_path.c_str())) return 1;
    const ModelConfig& cfg = weights.cfg;

    if (!a.model_size.empty()) {
        const ModelPreset& p = preset_for(a.model_size);
        if (p.cfg.n_layers == 0) {
            std::fprintf(stderr, "gemma: unknown --model_size \"%s\" (use 270m, 1B, or tiny)\n", a.model_size.c_str());
            return 1;
        }
        if (p.cfg.n_layers != cfg.n_layers || p.cfg.dim != cfg.dim || p.cfg.n_heads != cfg.n_heads ||
            p.cfg.n_kv_heads != cfg.n_kv_heads || p.cfg.mlp_dim != cfg.mlp_dim) {
            std::fprintf(stderr, "gemma: weights file does not match the \"%s\" preset (file: L=%u D=%u H=%u KV=%u MLP=%u)\n",
                         a.model_size.c_str(), cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_dim);
            return 1;
        }
    }

    if (a.dump) {
        weights.dump(stdout);
        return 0;
    }

    Tokenizer tok;
    std::string tok_path = a.tok_path.empty() ? a.weights_path + ".tok" : a.tok_path;
    if (!tok.load(tok_path.c_str())) {
        std::fprintf(stderr, "gemma: no tokenizer at %s (convert one with py/convert_tokenizer.py or py/make_test_weights.py)\n",
                     tok_path.c_str());
        return 1;
    }

    std::string text = a.chat_format ? wrap_chat(a.prompt) : a.prompt;
    std::vector<uint32_t> prompt_toks = tok.encode(text);

    uint64_t seed = a.seed ? a.seed : (uint64_t)std::time(nullptr) ^ (uint64_t)std::clock();
    Sampler sampler(seed);

    Model model(weights, std::min(cfg.max_seq_len, a.max_cache_len));
    const uint32_t n_ctx = std::min(cfg.max_seq_len, a.max_cache_len);
    if (prompt_toks.size() > n_ctx) {
        std::fprintf(stderr, "gemma: prompt is %zu tokens but the cache cap is %u\n", prompt_toks.size(), n_ctx);
        return 1;
    }
    // Generation positions are prompt + step; never let them pass the cache cap
    // (KVCache::write has no bounds check — going past max_len is an OOB write).
    const uint32_t remaining = n_ctx - (uint32_t)prompt_toks.size();   // >= 0: checked above
    if (a.n_dec > remaining) {
        std::fprintf(stderr, "gemma: warning: cache cap %u leaves room for %u more tokens after the prompt; "
                             "clamping n_dec from %u to %u\n",
                     n_ctx, remaining, a.n_dec, remaining);
        a.n_dec = remaining;
    }

    // Prefill the prompt.
    for (uint32_t pos = 0; pos < prompt_toks.size(); pos++)
        model.forward_token(prompt_toks[pos], pos);

    // Generate.
    std::string output;
    const auto t0 = std::clock();
    uint32_t n_gen = 0;
    for (uint32_t step = 0; step < a.n_dec; step++) {
        auto logits = model.logits_span();
        uint32_t next;
        if (a.min_p < 0.0 || a.temp <= 0.0)
            next = sampler.greedy(logits);
        else
            next = sampler.min_p(logits, a.temp, a.min_p);

        if (next == tok.eos_id && a.terminate_on_eos) break;
        output += tok.decode_id(next);
        model.forward_token(next, (uint32_t)prompt_toks.size() + step);
        n_gen++;
    }
    const double secs = (double)(std::clock() - t0) / (double)CLOCKS_PER_SEC;
    std::fprintf(stderr, "generated %u tokens in %.2fs (%.1f tok/s)\n", n_gen, secs, n_gen / secs);

    std::printf("%s", output.c_str());
    if (!output.empty() && output.back() != '\n') std::printf("\n");
    return 0;
}
