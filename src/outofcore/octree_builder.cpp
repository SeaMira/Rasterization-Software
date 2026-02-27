/**
 * @file octree_builder.cpp
 * @brief CPU-side reduced octree construction over spatially sorted blocks.
 *
 * Partitions blocks by octant at each level so that child nodes receive
 * the correct subset of blocks (bucketStart/bucketEnd correspond to
 * actual octant membership).
 */

#include "outofcore/octree_builder.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace ooc {

namespace {

int classifyOctant(glm::vec3 point, glm::vec3 mid) {
    int oct = 0;
    if (point.x >= mid.x) oct |= 1;
    if (point.y >= mid.y) oct |= 2;
    if (point.z >= mid.z) oct |= 4;
    return oct;
}

glm::vec3 octantMin(int oct, glm::vec3 regionMin, glm::vec3 mid) {
    return {
        (oct & 1) ? mid.x : regionMin.x,
        (oct & 2) ? mid.y : regionMin.y,
        (oct & 4) ? mid.z : regionMin.z
    };
}

glm::vec3 octantMax(int oct, glm::vec3 regionMin, glm::vec3 mid, glm::vec3 regionMax) {
    return {
        (oct & 1) ? regionMax.x : mid.x,
        (oct & 2) ? regionMax.y : mid.y,
        (oct & 4) ? regionMax.z : mid.z
    };
}

/**
 * Recursively build octree. Partitions indices by octant at each level.
 * indices[start..end-1] are block IDs for this node; we partition them
 * in place so children receive the correct blocks.
 */
void buildRecursive(std::vector<OocOctreeNode>& nodes,
                    std::vector<uint32_t>& indices,
                    const OocBlockMetadata* blocks,
                    int idxStart, int idxEnd,
                    glm::vec3 regionMin, glm::vec3 regionMax,
                    int depth, int maxDepth, bool verbose,
                    int parentNodeIdx, int childOctant)
{
    int blockCount = idxEnd - idxStart;
    int nodeIdx = static_cast<int>(nodes.size());
    nodes.push_back({});

    if (verbose) {
        std::string indent(depth * 2, ' ');
        std::cout << indent << "[Nodo " << nodeIdx;
        if (depth > 0) std::cout << " | hijo octante " << childOctant << " de " << parentNodeIdx;
        std::cout << "] Profundidad " << depth << ", bloques " << blockCount << std::endl;
        std::cout << indent << "  Region espacial: min=(" << std::fixed << std::setprecision(2)
                  << regionMin.x << "," << regionMin.y << "," << regionMin.z << ") max=("
                  << regionMax.x << "," << regionMax.y << "," << regionMax.z << ")";
    }

    glm::vec3 aabbMin(1e18f);
    glm::vec3 aabbMax(-1e18f);
    for (int i = idxStart; i < idxEnd; i++) 
    {
        uint32_t bid = indices[i];
        aabbMin = glm::min(aabbMin, blocks[bid].aabbMin);
        aabbMax = glm::max(aabbMax, blocks[bid].aabbMax);
    }

    nodes[nodeIdx].aabbMin = aabbMin;
    nodes[nodeIdx].aabbMax = aabbMax;
    nodes[nodeIdx]._pad0 = 0.0f;
    nodes[nodeIdx]._pad1 = 0.0f;
    std::memset(nodes[nodeIdx]._reserved, 0, sizeof(nodes[nodeIdx]._reserved));

    if (verbose) {
        std::cout << std::endl << std::string(depth * 2, ' ')
                  << "  AABB bloques:    min=(" << std::fixed << std::setprecision(2)
                  << aabbMin.x << "," << aabbMin.y << "," << aabbMin.z << ") max=("
                  << aabbMax.x << "," << aabbMax.y << "," << aabbMax.z << ")";
    }

    if (blockCount <= OOC_BLOCKS_PER_LEAF || depth >= maxDepth) 
    {
        nodes[nodeIdx].childBaseIndex  = -1;
        nodes[nodeIdx].childMask       = 0;
        nodes[nodeIdx].blockRangeStart = idxStart;
        nodes[nodeIdx].blockRangeEnd   = idxEnd;
        if (verbose) std::cout << " -> HOJA" << std::endl;
        return;
    }

    if (verbose) std::cout << " -> INTERIOR" << std::endl;

    glm::vec3 mid = (regionMin + regionMax) * 0.5f;

    // Count blocks per octant
    int bucketCount[8] = {};
    for (int i = idxStart; i < idxEnd; i++) 
    {
        int oct = classifyOctant(blocks[indices[i]].center, mid);
        bucketCount[oct]++;
    }

    // Prefix sum for bucket boundaries
    int bucketStart[8], bucketEnd[8];
    int cursor = idxStart;
    for (int oct = 0; oct < 8; oct++) 
    {
        bucketStart[oct] = cursor;
        cursor += bucketCount[oct];
        bucketEnd[oct] = cursor;
    }

    // Partition: place each block ID into its octant bucket
    std::vector<uint32_t> temp(blockCount);
    int bucketCur[8];
    for (int i = 0; i < 8; i++) bucketCur[i] = bucketStart[i];

    for (int i = idxStart; i < idxEnd; i++) 
    {
        uint32_t bid = indices[i];
        int oct = classifyOctant(blocks[bid].center, mid);
        temp[bucketCur[oct] - idxStart] = bid;
        bucketCur[oct]++;
    }

    for (int i = 0; i < blockCount; i++) {
        indices[idxStart + i] = temp[i];
    }

    nodes[nodeIdx].childBaseIndex  = -1;
    nodes[nodeIdx].childMask       = 0;
    nodes[nodeIdx].blockRangeStart = -1;
    nodes[nodeIdx].blockRangeEnd   = -1;

    uint8_t mask = 0;
    for (int oct = 0; oct < 8; oct++) 
    {
        if (bucketCount[oct] > 0) mask |= (1u << oct);
    }
    nodes[nodeIdx].childMask = mask;

    int childBase = static_cast<int>(nodes.size());
    nodes[nodeIdx].childBaseIndex = childBase;

    if (verbose) {
        std::string indent(depth * 2, ' ');
        std::cout << indent << "  Particion: ";
        for (int oct = 0; oct < 8; oct++) {
            if (bucketCount[oct] > 0)
                std::cout << "oct" << oct << "=" << bucketCount[oct] << " ";
        }
        std::cout << std::endl;
    }

    for (int oct = 0; oct < 8; oct++) 
    {
        if (bucketCount[oct] == 0) continue;
        buildRecursive(nodes, indices, blocks,
                       bucketStart[oct], bucketEnd[oct],
                       octantMin(oct, regionMin, mid),
                       octantMax(oct, regionMin, mid, regionMax),
                       depth + 1, maxDepth, verbose,
                       nodeIdx, oct);
    }
}

} // anonymous namespace

std::vector<OocOctreeNode> buildReducedOctree(
    const OocBlockMetadata* blocks,
    uint32_t numBlocks,
    glm::vec3 sceneMin,
    glm::vec3 sceneMax,
    int maxDepth,
    std::vector<uint32_t>& indexBuffer)
{
    std::vector<OocOctreeNode> nodes;
    nodes.reserve(numBlocks * 2);

    if (numBlocks == 0) return nodes;

    indexBuffer.resize(numBlocks);
    for (uint32_t i = 0; i < numBlocks; i++) indexBuffer[i] = i;

    if (verbose) {
        std::cout << "\n[OOC] === Construccion del octree (paso a paso) ===" << std::endl;
        std::cout << "[OOC] Bloques totales: " << numBlocks
                  << ", profundidad maxima: " << maxDepth
                  << ", bloques por hoja: " << OOC_BLOCKS_PER_LEAF << "\n" << std::endl;
    }

    buildRecursive(nodes, indexBuffer, blocks, 0, static_cast<int>(numBlocks),
                   sceneMin, sceneMax, 0, maxDepth, verbose, -1, -1);

    if (verbose) {
        std::cout << "\n[OOC] === Fin construccion: " << nodes.size() << " nodos ===" << std::endl;
    }

    return nodes;
}

} // namespace ooc
