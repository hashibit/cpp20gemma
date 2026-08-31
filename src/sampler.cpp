#include "sampler.h"

#include <cmath>

namespace gemma {

uint32_t Sampler::greedy(std::span<const float> logits) const {
    uint32_t best = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (logits[i] > logits[best]) best = (uint32_t)i;
    return best;
}

uint32_t Sampler::min_p(std::span<const float> logits, double temp, double min_p) {
    const size_t n = logits.size();
    scratch.resize(n);

    // Softmax over logits / temp, all in float32 (mirrors NumPy reference).
    const float t = (float)temp;
    float mx = logits[0] / t;
    for (size_t i = 1; i < n; i++) {
        float v = logits[i] / t;
        if (v > mx) mx = v;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = std::exp(logits[i] / t - mx);
        scratch[i] = e;
        sum += e;
    }
    for (size_t i = 0; i < n; i++) scratch[i] /= sum;

    // Min-P threshold, candidate set, and renormalization.
    const float thr = (float)min_p * [&] {
        float m = scratch[0];
        for (size_t i = 1; i < n; i++)
            if (scratch[i] > m) m = scratch[i];
        return m;
    }();

    std::vector<uint32_t> cands;
    cands.reserve(n);
    for (size_t i = 0; i < n; i++)
        if (scratch[i] >= thr) cands.push_back((uint32_t)i);
    if (cands.empty()) cands.push_back(greedy(logits));

    // Double-precision cumulative walk, same accumulation order as Python.
    const double u = rng.uniform();
    double cum = 0.0;
    for (uint32_t c : cands) {
        cum += (double)scratch[c];
        if (u < cum) return c;
    }
    return cands.back();
}

}  // namespace gemma
