// Minimal span-based tensor helpers and an aligned buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

namespace gemma {

// 64-byte-aligned heap buffer. Owns its memory; move-only.
struct AlignedBuffer {
    void*  ptr  = nullptr;
    size_t size = 0;

    AlignedBuffer() = default;
    ~AlignedBuffer() { std::free(ptr); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr = nullptr;
        o.size = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            std::free(ptr);
            ptr = o.ptr;
            size = o.size;
            o.ptr = nullptr;
            o.size = 0;
        }
        return *this;
    }

    // macOS aligned_alloc requires size % alignment == 0, so round up.
    bool alloc(size_t n, size_t align = 64) {
        std::free(ptr);
        ptr = nullptr;
        size = 0;
        if (n == 0) return true;
        size_t rounded = (n + align - 1) / align * align;
        void* p = std::aligned_alloc(align, rounded);
        if (!p) return false;
        ptr = p;
        size = rounded;
        return true;
    }
};

// Descriptor of one quantized weight tensor inside the weights file.
struct QTensor {
    const int8_t* data = nullptr;  // [rows][cols] row-major, cols contiguous
    float scale  = 0.0f;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

inline std::span<const float> as_span(const float* p, size_t n) { return {p, n}; }
inline std::span<float> as_span(float* p, size_t n) { return {p, n}; }

}  // namespace gemma
