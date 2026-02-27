/**
 * @file octree_builder.cpp
 * @brief CPU-side reduced octree construction over spatially sorted blocks.
 */

#include "outofcore/octree_builder.h"
#include <algorithm>
#include <cstring>

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

void buildRecursive(std::vector<OocOctreeNode>& nodes,
                    const OocBlockMetadata* blocks,
                    int blockStart, int blockEnd,
                    glm::vec3 regionMin, glm::vec3 regionMax,
                    int depth, int maxDepth)
{
    int blockCount = blockEnd - blockStart;
    int nodeIdx = static_cast<int>(nodes.size());
    nodes.push_back({});

    glm::vec3 aabbMin(1e18f);
    glm::vec3 aabbMax(-1e18f);
    for (int i = blockStart; i < blockEnd; i++) 
    {
        aabbMin = glm::min(aabbMin, blocks[i].aabbMin);
        aabbMax = glm::max(aabbMax, blocks[i].aabbMax);
    }

    nodes[nodeIdx].aabbMin = aabbMin;
    nodes[nodeIdx].aabbMax = aabbMax;
    nodes[nodeIdx]._pad0 = 0.0f;
    nodes[nodeIdx]._pad1 = 0.0f;
    std::memset(nodes[nodeIdx]._reserved, 0, sizeof(nodes[nodeIdx]._reserved));

    if (blockCount <= OOC_BLOCKS_PER_LEAF || depth >= maxDepth) 
    {
        nodes[nodeIdx].childBaseIndex  = -1;
        nodes[nodeIdx].childMask       = 0;
        nodes[nodeIdx].blockRangeStart = blockStart;
        nodes[nodeIdx].blockRangeEnd   = blockEnd;
        return;
    }

    glm::vec3 mid = (regionMin + regionMax) * 0.5f;

    int bucketCount[8] = {};
    for (int i = blockStart; i < blockEnd; i++) 
    {
        int oct = classifyOctant(blocks[i].center, mid);
        bucketCount[oct]++;
    }

    int bucketStart[8], bucketEnd[8];
    int cursor = blockStart;
    for (int oct = 0; oct < 8; oct++) 
    {
        bucketStart[oct] = cursor;
        cursor += bucketCount[oct];
        bucketEnd[oct] = cursor;
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

    for (int oct = 0; oct < 8; oct++) 
    {
        if (bucketCount[oct] == 0) continue;
        buildRecursive(nodes, blocks,
                       bucketStart[oct], bucketEnd[oct],
                       octantMin(oct, regionMin, mid),
                       octantMax(oct, regionMin, mid, regionMax),
                       depth + 1, maxDepth);
    }
}

} // anonymous namespace

std::vector<OocOctreeNode> buildReducedOctree(
    const OocBlockMetadata* blocks,
    uint32_t numBlocks,
    glm::vec3 sceneMin,
    glm::vec3 sceneMax,
    int maxDepth)
{
    std::vector<OocOctreeNode> nodes;
    nodes.reserve(numBlocks * 2);

    if (numBlocks == 0) return nodes;

    buildRecursive(nodes, blocks, 0, static_cast<int>(numBlocks),
                   sceneMin, sceneMax, 0, maxDepth);

    return nodes;
}

} // namespace ooc
