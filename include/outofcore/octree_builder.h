/**
 * @file octree_builder.h
 * @brief CPU-side octree construction from sorted block metadata.
 *
 * Builds a reduced octree whose leaves reference block ranges. The tree
 * is shallow (typically 4-8 levels) so that frustum culling on the GPU
 * rejects large portions of space quickly.
 */

#ifndef OOC_OCTREE_BUILDER_H
#define OOC_OCTREE_BUILDER_H

#include <vector>
#include <glm/glm.hpp>
#include "outofcore/outofcore_types.h"

namespace ooc {

/**
 * Build a reduced octree over the given blocks.
 * Partitions blocks by octant at each level; produces nodes and an index buffer.
 *
 * @param blocks       Block metadata array (must be sorted by Morton code).
 * @param numBlocks    Number of blocks.
 * @param sceneMin     Scene AABB minimum.
 * @param sceneMax     Scene AABB maximum.
 * @param maxDepth     Maximum octree depth.
 * @param[out] indexBuffer For leaves, blockRangeStart/End index into this buffer.
 *                         Contains block IDs in octant order. Must be non-null.
 * @param verbose         If true, prints step-by-step construction to stdout.
 * @return Linear array of OocOctreeNode ready for GPU upload.
 */
std::vector<OocOctreeNode> buildReducedOctree(
    const OocBlockMetadata* blocks,
    uint32_t numBlocks,
    glm::vec3 sceneMin,
    glm::vec3 sceneMax,
    int maxDepth,
    int blocksPerLeaf,
    std::vector<uint32_t>& indexBuffer,
    bool verbose = false);

} // namespace ooc

#endif // OOC_OCTREE_BUILDER_H
