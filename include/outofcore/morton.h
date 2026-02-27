/**
 * @file morton.h
 * @brief Morton (Z-order) code encoding for 3D coordinates.
 *
 * Provides both CPU and CUDA device functions for computing 30-bit and
 * 63-bit Morton codes from normalised 3D coordinates, preserving
 * spatial locality when sorted.
 */

#ifndef OOC_MORTON_H
#define OOC_MORTON_H

#include <cstdint>

#ifdef __CUDACC__
#define OOC_HOST_DEVICE __host__ __device__
#else
#define OOC_HOST_DEVICE
#endif

/**
 * Expand a 10-bit integer so that each bit occupies every 3rd position.
 * E.g. 0b1111111111 → 0b001001001001001001001001001001.
 */
OOC_HOST_DEVICE inline uint32_t expandBits10(uint32_t v) {
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v <<  8u)) & 0x0300F00Fu;
    v = (v | (v <<  4u)) & 0x030C30C3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}

/**
 * Compute a 30-bit Morton code from three 10-bit integers.
 */
OOC_HOST_DEVICE inline uint32_t encodeMorton3D_30(uint32_t x, uint32_t y, uint32_t z) {
    return (expandBits10(x) << 2u) | (expandBits10(y) << 1u) | expandBits10(z);
}

/**
 * Compute a 30-bit Morton code from normalised [0,1] coordinates.
 * Coordinates are quantised to 10 bits each (1024 levels).
 */
OOC_HOST_DEVICE inline uint32_t mortonFromNormalized(float nx, float ny, float nz) {
    auto clamp01 = [](float v) -> float {
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    };
    uint32_t ix = static_cast<uint32_t>(clamp01(nx) * 1023.0f);
    uint32_t iy = static_cast<uint32_t>(clamp01(ny) * 1023.0f);
    uint32_t iz = static_cast<uint32_t>(clamp01(nz) * 1023.0f);
    return encodeMorton3D_30(ix, iy, iz);
}

#undef OOC_HOST_DEVICE

#endif // OOC_MORTON_H
