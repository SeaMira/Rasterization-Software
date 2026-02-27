/**
 * @file outofcore_types.h
 * @brief Shared types for the out-of-core octree streaming pipeline.
 *
 * Defines data structures used across CPU preprocessing, GPU kernels,
 * and the runtime streaming manager. Based on the hierarchical view
 * frustum culling (Atomsviewer, Sharma et al. 2004) combined with
 * ray-guided streaming ideas (GigaVoxels, Crassin et al. 2009).
 */

#ifndef OUTOFCORE_TYPES_H
#define OUTOFCORE_TYPES_H

#include <cstdint>
#include <glm/glm.hpp>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

static constexpr int OOC_ATOMS_PER_BLOCK       = 512;
static constexpr int OOC_MAX_OCTREE_DEPTH       = 10;
static constexpr int OOC_BLOCKS_PER_LEAF        = 4;
static constexpr int OOC_MAX_BLOCK_POOL_SLOTS   = 2048;
static constexpr int OOC_MAX_REQUESTS_PER_FRAME = 256;
static constexpr float OOC_VISIBILITY_THRESHOLD = 0.01f;

/**
 * Magic number for the binary block file header ("OOCBLKDT").
 */
static constexpr uint64_t OOC_BLOCK_FILE_MAGIC = 0x4F4F43424C4B4454ULL;

// ─────────────────── Block metadata ───────────────────

struct OocBlockMetadata {
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    glm::vec3 center;
    uint32_t  atomCount;
    uint64_t  fileOffset;
    uint32_t  blockId;
};

// ─────────────────── Octree node (GPU-resident) ───────────────────

/**
 * Reduced octree node stored in a linear GPU buffer. Interior nodes
 * contain a childBaseIndex pointing to the first of up to 8 contiguous
 * children. Leaf nodes store a range of block IDs.
 */
struct OocOctreeNode {
    glm::vec3 aabbMin;
    float     _pad0;
    glm::vec3 aabbMax;
    float     _pad1;

    int32_t  childBaseIndex;   ///< Index of first child (-1 = leaf)
    uint8_t  childMask;        ///< Bitmask of which of the 8 octants exist
    uint8_t  _reserved[3];

    int32_t  blockRangeStart;  ///< First block ID in this leaf (valid when leaf)
    int32_t  blockRangeEnd;    ///< Past-the-end block ID in this leaf
};

// ─────────────────── GPU constant buffer ───────────────────

struct OocConstants {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::vec4 frustumPlanes[6];
    glm::vec3 cameraPos;
    int       screenWidth;
    int       screenHeight;
    float     fov;
    int       totalBlocks;
    int       octreeNodeCount;
    float     visibilityThreshold;
    int       maxPoolSlots;

    /** Pre-computed screen ray casting (camera space). Same as HybridConstants.
     *  ray(px,py) = rayStart + px*dx + py*dy. rd = normalize(ray.x*right + ray.y*up - ray.z*front). */
    glm::vec3 rayStart;
    glm::vec3 dx;
    glm::vec3 dy;
    glm::vec3 right;
    glm::vec3 up;
    glm::vec3 front;

    /** Raster color constants (atoms). */
    glm::vec3 atomsColor;
    float     diffuse;
};

// ─────────────────── Block pool slot (CPU side) ───────────────────

struct OocBlockSlot {
    uint32_t blockId        = UINT32_MAX;
    uint64_t lastUsedFrame  = 0;
    bool     valid          = false;
};

// ─────────────────── Block file header ───────────────────

struct OocBlockFileHeader {
    uint64_t  magic;
    uint32_t  numBlocks;
    uint32_t  atomsPerBlock;
    glm::vec3 sceneMin;
    float     _pad0;
    glm::vec3 sceneMax;
    float     _pad1;
};

// ─────────────────── Depth-sorted block (GPU side) ───────────────────

struct OocBlockDepthInfo {
    uint32_t blockId;
    float    depth;
    float    projectedArea;
    float    _pad;
};

#endif // OUTOFCORE_TYPES_H
