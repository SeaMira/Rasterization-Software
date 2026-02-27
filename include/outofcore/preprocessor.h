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

struct PreprocessResult {
    std::string                  blockFilePath;
    std::vector<OocBlockMetadata> blocks;
    std::vector<OocOctreeNode>    octreeNodes;
    glm::vec3                    sceneMin;
    glm::vec3                    sceneMax;
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
 * @return PreprocessResult with all metadata needed at runtime.
 */
PreprocessResult preprocess(const glm::vec4* atoms,
                            uint32_t numAtoms,
                            const std::string& outputDir);

} // namespace ooc

#endif // OOC_PREPROCESSOR_H
