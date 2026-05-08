/**
 * @file preprocessor.h
 * @brief One-shot preprocessing: Morton sort + block file + octree.
 *
 * For static scenes this runs once before the render loop begins.
 */

#ifndef OOC_PREPROCESSOR_H
#define OOC_PREPROCESSOR_H

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "outofcore/outofcore_types.h"

namespace ooc {

/** Timing and size snapshot for the one-shot preprocess (CPU + block file on disk). */
struct PreprocessMetrics {
    /** AABB, Morton codes, sort by Morton, reorder atoms into `sortedAtoms`. */
    double   msMortonPipeline = 0.0;
    /** Partition into blocks, per-block AABB, write `block_data.bin`. */
    double   msBlocksAndFile  = 0.0;
    /** `buildReducedOctree` (octree + block index buffer). */
    double   msOctreeBuild    = 0.0;
    /** RAM of `blocks`, `octreeNodes`, `blockIndexBuffer` in `PreprocessResult` (after preprocess). */
    size_t   hostStructuresBytes = 0;
    /** Size of `block_data.bin` on disk (after successful write). */
    size_t   blockFileBytes   = 0;
    int      blockCount       = 0;
    int      octreeNodeCount  = 0;
};

struct PreprocessResult {
    std::string                  blockFilePath;
    std::vector<OocBlockMetadata> blocks;
    std::vector<OocOctreeNode>    octreeNodes;
    std::vector<uint32_t>        blockIndexBuffer;  ///< For leaves, blockRangeStart/End index here
    std::vector<glm::vec4>       lodAtoms;          ///< LOD atom buffer (referenced by octreeNodes[i].lodOffset)
    glm::vec3                    sceneMin;
    glm::vec3                    sceneMax;
    PreprocessMetrics            metrics;
};

/**
 * Run the full preprocessing pipeline:
 *   1. Compute scene AABB.
 *   2. Morton-code sort atoms.
 *   3. Partition into blocks and compute per-block AABB.
 *   4. Write binary block file to disk.
 *   5. Build reduced octree over blocks.
 *
 * @param atoms       Raw atom positions (xyz + radius in w).
 * @param numAtoms    Total atom count.
 * @param outputDir   Directory where the block file is written.
 * @param verbose     If true, prints step-by-step octree construction to stdout.
 * @return PreprocessResult with all metadata needed at runtime.
 */
PreprocessResult preprocess(const glm::vec4* atoms,
                            uint32_t numAtoms,
                            const std::string& outputDir,
                            const OocConfig& config,
                            bool verbose = false);

} // namespace ooc

#endif // OOC_PREPROCESSOR_H
