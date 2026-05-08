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
#include <string>
#include <glm/glm.hpp>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// Default values (used when no JSON config is loaded)
static constexpr int OOC_DEFAULT_ATOMS_PER_BLOCK       = 512;
static constexpr int OOC_DEFAULT_MAX_OCTREE_DEPTH      = 10;
static constexpr int OOC_DEFAULT_BLOCKS_PER_LEAF       = 4;
static constexpr int OOC_DEFAULT_MAX_BLOCK_POOL_SLOTS  = 2048;
static constexpr int OOC_DEFAULT_MAX_REQUESTS_PER_FRAME = 256;
static constexpr float OOC_DEFAULT_VISIBILITY_THRESHOLD = 0.01f;
static constexpr int   OOC_DEFAULT_LOD_MAX_ATOMS_PER_NODE = 64;
static constexpr float OOC_DEFAULT_LOD_AREA_THRESHOLD = 0.005f; ///< Projected area fraction below which LOD is used

/**
 * Magic number for the binary block file header ("OOCBLKDT").
 */
static constexpr uint64_t OOC_BLOCK_FILE_MAGIC = 0x4F4F43424C4B4454ULL;

// ─────────────────── Occlusion method enum ───────────────────

enum class OocOcclusionMethod : int {
    NONE                  = 0,
    PROBABILISTIC         = 1,
    PROBABILISTIC_OVERLAP = 2,
    HIZ                   = 3,
    HIZ_PROBABILISTIC     = 4
};

inline OocOcclusionMethod parseOcclusionMethod(const std::string& s) {
    if (s == "probabilistic")         return OocOcclusionMethod::PROBABILISTIC;
    if (s == "probabilistic_overlap") return OocOcclusionMethod::PROBABILISTIC_OVERLAP;
    if (s == "hiz")                   return OocOcclusionMethod::HIZ;
    if (s == "hiz_probabilistic")     return OocOcclusionMethod::HIZ_PROBABILISTIC;
    return OocOcclusionMethod::NONE;
}

// ─────────────────── Runtime OOC config (from JSON) ──────────────────

struct OocConfig {
    int   atomsPerBlock       = OOC_DEFAULT_ATOMS_PER_BLOCK;
    int   maxOctreeDepth      = OOC_DEFAULT_MAX_OCTREE_DEPTH;
    int   blocksPerLeaf       = OOC_DEFAULT_BLOCKS_PER_LEAF;
    int   maxBlockPoolSlots   = OOC_DEFAULT_MAX_BLOCK_POOL_SLOTS;
    int   maxRequestsPerFrame = OOC_DEFAULT_MAX_REQUESTS_PER_FRAME;
    float visibilityThreshold = OOC_DEFAULT_VISIBILITY_THRESHOLD;
    OocOcclusionMethod occlusionMethod = OocOcclusionMethod::NONE;
};

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
    int32_t   lodOffset;       ///< Offset into LOD atom buffer (-1 = no LOD)

    int32_t  childBaseIndex;   ///< Index of first child (-1 = leaf)
    uint8_t  childMask;        ///< Bitmask of which of the 8 octants exist
    uint8_t  _reserved[1];
    uint16_t lodCount;         ///< Number of LOD atoms for this node (0 = none)

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
    int       atomsPerBlock;
    int       occlusionMethod;

    /** Pre-computed screen ray casting (camera space). */
    glm::vec3 rayStart;
    glm::vec3 dx;
    glm::vec3 dy;
    glm::vec3 right;
    glm::vec3 up;
    glm::vec3 front;

    /** Raster color constants (atoms). */
    glm::vec3 atomsColor;
    float     diffuse;

    float     nearPlane;
    float     farPlane;

    /** LOD rendering: projected area fraction threshold for using LOD atoms. */
    float     lodAreaThreshold;
    int       lodTotalAtoms;     ///< Total atoms in the LOD buffer
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
    float    screenMinX;
    float    screenMinY;
    float    screenMaxX;
    float    screenMaxY;
    uint32_t atomCount;
    float    minNdcDepth;    ///< Closest NDC depth of the AABB (for HiZ test)
    float    screenMinU;     ///< UV-space min X (0..1) for HiZ sampling
    float    screenMinV;     ///< UV-space min Y (0..1) for HiZ sampling
    float    screenMaxU;     ///< UV-space max X (0..1) for HiZ sampling
    float    screenMaxV;     ///< UV-space max Y (0..1) for HiZ sampling
};

#endif // OUTOFCORE_TYPES_H
