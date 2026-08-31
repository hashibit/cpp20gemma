// Sampling: splitmix64 RNG, greedy argmax, Min-P with temperature.
// The RNG and the Min-P walk are mirrored bit-for-bit in py/reference_model.py
// so golden draws compare exactly.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace gemma {

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}

    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform double in [0, 1), 53 bits of entropy.
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
};

struct Sampler {
    SplitMix64 rng;
    std::vector<float> scratch;  // probabilities workspace

    explicit Sampler(uint64_t seed) : rng(seed) {}

    // Argmax; first index wins ties (matches np.argmax).
    uint32_t greedy(std::span<const float> logits) const;

    // softmax(logits / temp) -> filter probs >= max_prob * min_p ->
    // renormalize -> weighted draw. Mirrors the Python reference exactly:
    // float32 softmax, double cumulative walk.
    uint32_t min_p(std::span<const float> logits, double temp, double min_p);
};

}  // namespace gemma
