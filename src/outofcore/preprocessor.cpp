/**
 * @file preprocessor.cpp
 * @brief Full preprocessing pipeline: Morton sort → blocks → octree.
 *
 * Optimised for very large datasets (500 M – 1 B+ atoms):
 *   • Packed (morton, index) uint64_t avoids a separate index array.
 *   • 4-pass LSB radix sort — O(N) instead of O(N log N).
 *   • Streaming block-file write: never materialises a full sorted copy
 *     of the atom array (saves 16 GB at 1 B atoms).
 *   • LOD generation uses block centres (bottom-up merge), so it never
 *     accesses atom data and runs in O(numNodes × maxLodPerNode).
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "outofcore/preprocessor.h"
#include "outofcore/morton.h"
#include "outofcore/block_file_io.h"
#include "outofcore/octree_builder.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <vector>

namespace ooc {

// ═════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════

namespace {

double elapsedMs(std::chrono::steady_clock::time_point t0,
                 std::chrono::steady_clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ─────────────── 4-pass LSB radix sort on upper 32 bits ───────────────
// Sorts an array of uint64_t values by their upper 32 bits (the morton
// code).  The lower 32 bits carry the original atom index and come along
// for the ride.  4 passes × 8-bit digits = 32-bit key.
// After an even number of passes the result is back in `data`.
//
// Memory: requires a temporary buffer of the same size as `data`.

void radixSortByUpper32(uint64_t* data, uint64_t* temp, size_t n)
{
    uint64_t* src = data;
    uint64_t* dst = temp;

    for (int pass = 0; pass < 4; pass++)
    {
        const int shift = pass * 8 + 32;   // +32 → upper half

        // Histogram
        size_t count[256] = {};
        for (size_t i = 0; i < n; i++)
            count[(src[i] >> shift) & 0xFF]++;

        // Prefix sum → write offsets
        size_t offset[256];
        offset[0] = 0;
        for (int b = 1; b < 256; b++)
            offset[b] = offset[b - 1] + count[b - 1];

        // Scatter
        for (size_t i = 0; i < n; i++) {
            uint8_t byte = static_cast<uint8_t>((src[i] >> shift) & 0xFF);
            dst[offset[byte]++] = src[i];
        }

        std::swap(src, dst);
    }
    // After 4 (even) passes: src == data, result is in data. ✓
}

// ─── Streaming block-file writer (avoids full sortedAtoms copy) ───
//
// Gathers each block's atoms on the fly from the *original* (unsorted)
// atom array using the permutation encoded in `sortedKeys`.

bool writeBlockFileStreaming(
    const std::string& path,
    const glm::vec4*   atoms,        // original unsorted atoms
    const uint64_t*    sortedKeys,   // packed (morton|index), sorted
    uint32_t           numAtoms,
    glm::vec3          sceneMin,
    glm::vec3          sceneMax,
    int                atomsPerBlock,
    std::vector<OocBlockMetadata>& blocks)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "[OOC] Failed to open block file for writing: " << path << std::endl;
        return false;
    }

    const uint32_t apb = static_cast<uint32_t>(atomsPerBlock);
    const uint32_t numBlocks = (numAtoms + apb - 1) / apb;
    blocks.resize(numBlocks);

    OocBlockFileHeader header{};
    header.magic         = OOC_BLOCK_FILE_MAGIC;
    header.numBlocks     = numBlocks;
    header.atomsPerBlock = apb;
    header.sceneMin      = sceneMin;
    header.sceneMax      = sceneMax;
    fwrite(&header, sizeof(header), 1, f);

    // Small reusable gather buffer (atomsPerBlock × 16 B ≈ 8 KB)
    std::vector<glm::vec4> buf(apb);
    uint64_t fileOffset = sizeof(header);

    const uint32_t logInterval = std::max(numBlocks / 20u, 1u);

    for (uint32_t b = 0; b < numBlocks; b++)
    {
        const size_t start = static_cast<size_t>(b) * apb;
        const uint32_t count = std::min(apb, numAtoms - static_cast<uint32_t>(start));

        // Gather atoms from original array via permutation
        glm::vec3 bboxMin(1e18f);
        glm::vec3 bboxMax(-1e18f);
        glm::vec3 centerSum(0.0f);

        for (uint32_t i = 0; i < count; i++) {
            uint32_t origIdx = static_cast<uint32_t>(sortedKeys[start + i] & 0xFFFFFFFFu);
            glm::vec4 atom = atoms[origIdx];
            buf[i] = atom;

            glm::vec3 p(atom);
            float r = atom.w;
            bboxMin = glm::min(bboxMin, p - glm::vec3(r));
            bboxMax = glm::max(bboxMax, p + glm::vec3(r));
            centerSum += p;
        }

        blocks[b].aabbMin    = bboxMin;
        blocks[b].aabbMax    = bboxMax;
        blocks[b].center     = centerSum / static_cast<float>(count);
        blocks[b].atomCount  = count;
        blocks[b].fileOffset = fileOffset;
        blocks[b].blockId    = b;

        fwrite(&bboxMin, sizeof(glm::vec3), 1, f);
        fwrite(&bboxMax, sizeof(glm::vec3), 1, f);
        fwrite(&count,   sizeof(uint32_t),  1, f);
        fwrite(buf.data(), sizeof(glm::vec4), count, f);

        fileOffset += 2 * sizeof(glm::vec3) + sizeof(uint32_t)
                    + static_cast<uint64_t>(count) * sizeof(glm::vec4);

        if (b % logInterval == 0)
            std::cout << "\r[OOC]   Writing blocks... "
                      << (b * 100 / numBlocks) << "%" << std::flush;
    }

    fclose(f);
    std::cout << "\r[OOC] Wrote " << numBlocks << " blocks to " << path
              << "                    " << std::endl;
    return true;
}

} // anonymous namespace

// ═════════════════════════════════════════════════════════
// Public API
// ═════════════════════════════════════════════════════════

PreprocessResult preprocess(const glm::vec4* atoms,
                            uint32_t numAtoms,
                            const std::string& outputDir,
                            const OocConfig& config,
                            bool verbose)
{
    PreprocessResult result{};
    std::cout << "[OOC] Preprocessing " << numAtoms << " atoms ("
              << (static_cast<double>(numAtoms) * sizeof(glm::vec4) / (1024.0 * 1024.0))
              << " MiB)..." << std::endl;

    // ── 1. Compute scene AABB + average atom radius ──────────────

    auto tMorton0 = std::chrono::steady_clock::now();

    glm::vec3 sceneMin(1e18f);
    glm::vec3 sceneMax(-1e18f);
    double radiusSum = 0.0;

    for (uint32_t i = 0; i < numAtoms; i++) {
        glm::vec3 p(atoms[i]);
        float r = atoms[i].w;
        sceneMin = glm::min(sceneMin, p - glm::vec3(r));
        sceneMax = glm::max(sceneMax, p + glm::vec3(r));
        radiusSum += static_cast<double>(r);
    }
    float avgAtomRadius = (numAtoms > 0)
        ? static_cast<float>(radiusSum / static_cast<double>(numAtoms))
        : 1.0f;

    glm::vec3 sceneExtent = sceneMax - sceneMin;
    glm::vec3 safeExtent(
        sceneExtent.x > 1e-6f ? sceneExtent.x : 1.0f,
        sceneExtent.y > 1e-6f ? sceneExtent.y : 1.0f,
        sceneExtent.z > 1e-6f ? sceneExtent.z : 1.0f);

    result.sceneMin = sceneMin;
    result.sceneMax = sceneMax;

    std::cout << "[OOC] AABB done.  avgRadius=" << avgAtomRadius << std::endl;

    // ── 2. Compute packed (morton | index) keys ──────────────────
    // Upper 32 bits = morton code, lower 32 bits = original atom index.
    // Single allocation: 8 bytes × N (vs 8 bytes for two separate arrays).

    std::cout << "[OOC] Computing Morton codes..." << std::endl;

    std::vector<uint64_t> keys(numAtoms);
    {
        const float invX = 1.0f / safeExtent.x;
        const float invY = 1.0f / safeExtent.y;
        const float invZ = 1.0f / safeExtent.z;
        for (uint32_t i = 0; i < numAtoms; i++) {
            glm::vec3 p(atoms[i]);
            float nx = (p.x - sceneMin.x) * invX;
            float ny = (p.y - sceneMin.y) * invY;
            float nz = (p.z - sceneMin.z) * invZ;
            uint32_t morton = mortonFromNormalized(nx, ny, nz);
            keys[i] = (static_cast<uint64_t>(morton) << 32) | static_cast<uint64_t>(i);
        }
    }

    // ── 3. Radix sort (O(N), 4 passes on 32-bit key) ────────────
    std::cout << "[OOC] Radix sort (" << numAtoms << " keys)..." << std::endl;

    {
        std::vector<uint64_t> temp(numAtoms);   // temporary for radix sort
        radixSortByUpper32(keys.data(), temp.data(), numAtoms);
    }   // `temp` freed here — saves 8 GB at 1 B atoms

    std::cout << "[OOC] Sort complete." << std::endl;

    auto tMorton1 = std::chrono::steady_clock::now();
    result.metrics.msMortonPipeline = elapsedMs(tMorton0, tMorton1);

    // ── 4. Streaming block-file write ────────────────────────────
    // Gathers each block from the original atom array using the
    // permutation in `keys`.  No full sortedAtoms copy is created.

    std::filesystem::create_directories(outputDir);
    result.blockFilePath = outputDir + "/block_data.bin";

    auto tBlocks0 = std::chrono::steady_clock::now();
    if (!writeBlockFileStreaming(result.blockFilePath, atoms, keys.data(),
                                numAtoms, sceneMin, sceneMax,
                                config.atomsPerBlock, result.blocks))
    {
        std::cerr << "[OOC] Failed to write block file!" << std::endl;
        return result;
    }
    auto tBlocks1 = std::chrono::steady_clock::now();
    result.metrics.msBlocksAndFile = elapsedMs(tBlocks0, tBlocks1);

    // Free the sort keys — no longer needed
    { std::vector<uint64_t>().swap(keys); }

    if (std::filesystem::exists(result.blockFilePath)) {
        std::error_code ec;
        auto sz = std::filesystem::file_size(result.blockFilePath, ec);
        if (!ec)
            result.metrics.blockFileBytes = static_cast<size_t>(sz);
    }

    // ── 5. Build reduced octree ──────────────────────────────────
    auto tOct0 = std::chrono::steady_clock::now();
    result.octreeNodes = buildReducedOctree(
        result.blocks.data(),
        static_cast<uint32_t>(result.blocks.size()),
        sceneMin, sceneMax, config.maxOctreeDepth, config.blocksPerLeaf,
        result.blockIndexBuffer,
        verbose);
    auto tOct1 = std::chrono::steady_clock::now();
    result.metrics.msOctreeBuild = elapsedMs(tOct0, tOct1);

    // ── 6. LOD generation (block-centre based, no atom access) ───
    // Bottom-up: leaves build LOD from their block centres; internal
    // nodes merge children's LOD and subsample to maxLodPerNode.
    // Total work: O(numNodes × maxLodPerNode).  Memory: a few MB.
    {
        const int maxLodPerNode = OOC_DEFAULT_LOD_MAX_ATOMS_PER_NODE;
        auto& nodes  = result.octreeNodes;
        auto& lodBuf = result.lodAtoms;
        lodBuf.clear();

        const size_t numNodes = nodes.size();
        // Per-node LOD candidates (small vectors, max 64 elements each)
        std::vector<std::vector<glm::vec4>> nodeLod(numNodes);

        // Process in reverse BFS order → children before parents
        for (int ni = static_cast<int>(numNodes) - 1; ni >= 0; ni--)
        {
            auto& node = nodes[ni];
            node.lodOffset = -1;
            node.lodCount  = 0;

            if (node.childBaseIndex < 0)
            {
                // Leaf: create one representative atom per block (centre)
                for (int32_t b = node.blockRangeStart; b < node.blockRangeEnd; b++) {
                    if (b < 0 || b >= static_cast<int32_t>(result.blockIndexBuffer.size()))
                        continue;
                    uint32_t bid = result.blockIndexBuffer[b];
                    if (bid >= result.blocks.size()) continue;
                    const auto& blk = result.blocks[bid];
                    nodeLod[ni].push_back(glm::vec4(blk.center, avgAtomRadius));
                }
                // Subsample if more blocks than maxLodPerNode
                if (static_cast<int>(nodeLod[ni].size()) > maxLodPerNode) {
                    std::vector<glm::vec4> sub;
                    sub.reserve(maxLodPerNode);
                    float step = static_cast<float>(nodeLod[ni].size())
                               / static_cast<float>(maxLodPerNode);
                    for (int i = 0; i < maxLodPerNode; i++) {
                        int idx = static_cast<int>(step * i);
                        sub.push_back(nodeLod[ni][idx]);
                    }
                    nodeLod[ni] = std::move(sub);
                }
            }
            else
            {
                // Internal: merge children's LOD sets
                std::vector<glm::vec4> merged;
                int childIdx = 0;
                for (int oct = 0; oct < 8; oct++) {
                    if (node.childMask & (1u << oct)) {
                        int ci = node.childBaseIndex + childIdx;
                        childIdx++;
                        if (ci < 0 || ci >= static_cast<int>(numNodes)) continue;
                        merged.insert(merged.end(),
                                      nodeLod[ci].begin(), nodeLod[ci].end());
                        // Free child's LOD to reduce peak memory
                        { std::vector<glm::vec4>().swap(nodeLod[ci]); }
                    }
                }
                // Subsample merged set
                if (static_cast<int>(merged.size()) > maxLodPerNode) {
                    std::vector<glm::vec4> sub;
                    sub.reserve(maxLodPerNode);
                    float step = static_cast<float>(merged.size())
                               / static_cast<float>(maxLodPerNode);
                    for (int i = 0; i < maxLodPerNode; i++) {
                        int idx = static_cast<int>(step * i);
                        sub.push_back(merged[idx]);
                    }
                    nodeLod[ni] = std::move(sub);
                } else {
                    nodeLod[ni] = std::move(merged);
                }
            }

            // Append to flat LOD buffer
            if (!nodeLod[ni].empty()) {
                node.lodOffset = static_cast<int32_t>(lodBuf.size());
                node.lodCount  = static_cast<uint16_t>(
                    std::min(static_cast<int>(nodeLod[ni].size()), 65535));
                lodBuf.insert(lodBuf.end(),
                              nodeLod[ni].begin(),
                              nodeLod[ni].begin() + node.lodCount);
            }
        }

        std::cout << "[OOC] LOD generated: " << lodBuf.size() << " atoms for "
                  << numNodes << " nodes." << std::endl;
    }

    // ── Metrics ──────────────────────────────────────────────────
    result.metrics.blockCount      = static_cast<int>(result.blocks.size());
    result.metrics.octreeNodeCount = static_cast<int>(result.octreeNodes.size());
    result.metrics.hostStructuresBytes =
        result.blocks.size()      * sizeof(OocBlockMetadata) +
        result.octreeNodes.size() * sizeof(OocOctreeNode) +
        result.blockIndexBuffer.size() * sizeof(uint32_t) +
        result.lodAtoms.size()    * sizeof(glm::vec4);

    std::cout << "[OOC] Octree built: " << result.octreeNodes.size()
              << " nodes, " << result.blocks.size() << " blocks." << std::endl;
    std::cout << "[OOC] Preprocess timing: morton+sort="
              << result.metrics.msMortonPipeline << " ms, "
              << "blocks+file=" << result.metrics.msBlocksAndFile << " ms, "
              << "octree=" << result.metrics.msOctreeBuild << " ms" << std::endl;
    std::cout << "[OOC] Host structures (blocks+octree+index+lod): "
              << (result.metrics.hostStructuresBytes / (1024.0 * 1024.0)) << " MiB, "
              << "block file: "
              << (result.metrics.blockFileBytes / (1024.0 * 1024.0)) << " MiB" << std::endl;

    return result;
}

} // namespace ooc
