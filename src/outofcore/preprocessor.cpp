/**
 * @file preprocessor.cpp
 * @brief Full preprocessing pipeline: Morton sort → blocks → octree.
 */

#include "outofcore/preprocessor.h"
#include "outofcore/morton.h"
#include "outofcore/block_file_io.h"
#include "outofcore/octree_builder.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <filesystem>

namespace ooc {

PreprocessResult preprocess(const glm::vec4* atoms,
                            uint32_t numAtoms,
                            const std::string& outputDir,
                            bool verbose)
{
    PreprocessResult result{};
    std::cout << "[OOC] Preprocessing " << numAtoms << " atoms..." << std::endl;

    // 1. Compute scene AABB
    glm::vec3 sceneMin(1e18f);
    glm::vec3 sceneMax(-1e18f);
    for (uint32_t i = 0; i < numAtoms; i++) {
        glm::vec3 p(atoms[i]);
        float r = atoms[i].w;
        sceneMin = glm::min(sceneMin, p - glm::vec3(r));
        sceneMax = glm::max(sceneMax, p + glm::vec3(r));
    }
    glm::vec3 sceneExtent = sceneMax - sceneMin;
    float maxExtent = std::max({sceneExtent.x, sceneExtent.y, sceneExtent.z});
    if (maxExtent < 1e-6f) maxExtent = 1.0f;

    result.sceneMin = sceneMin;
    result.sceneMax = sceneMax;

    // 2. Compute Morton codes
    std::vector<uint32_t> mortonCodes(numAtoms);
    for (uint32_t i = 0; i < numAtoms; i++) {
        glm::vec3 p(atoms[i]);
        float nx = (p.x - sceneMin.x) / maxExtent;
        float ny = (p.y - sceneMin.y) / maxExtent;
        float nz = (p.z - sceneMin.z) / maxExtent;
        mortonCodes[i] = mortonFromNormalized(nx, ny, nz);
    }

    // 3. Sort atoms by Morton code
    std::vector<uint32_t> indices(numAtoms);
    std::iota(indices.begin(), indices.end(), 0u);
    std::sort(indices.begin(), indices.end(),
              [&mortonCodes](uint32_t a, uint32_t b) {
                  return mortonCodes[a] < mortonCodes[b];
              });

    std::vector<glm::vec4> sortedAtoms(numAtoms);
    for (uint32_t i = 0; i < numAtoms; i++) {
        sortedAtoms[i] = atoms[indices[i]];
    }

    std::cout << "[OOC] Morton sort complete." << std::endl;

    // 4. Write block file
    std::filesystem::create_directories(outputDir);
    result.blockFilePath = outputDir + "/block_data.bin";

    if (!writeBlockFile(result.blockFilePath, sortedAtoms.data(), numAtoms,
                        sceneMin, sceneMax, result.blocks)) {
        std::cerr << "[OOC] Failed to write block file!" << std::endl;
        return result;
    }

    // 5. Build reduced octree (partitions blocks by octant; produces index buffer)
    result.octreeNodes = buildReducedOctree(
        result.blocks.data(),
        static_cast<uint32_t>(result.blocks.size()),
        sceneMin, sceneMax, OOC_MAX_OCTREE_DEPTH,
        result.blockIndexBuffer,
        verbose);

    std::cout << "[OOC] Octree built: " << result.octreeNodes.size()
              << " nodes, " << result.blocks.size() << " blocks." << std::endl;

    return result;
}

} // namespace ooc
