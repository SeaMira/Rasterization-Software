/**
 * @file block_file_io.h
 * @brief Read/write routines for the binary block file format.
 *
 * The block file stores atoms partitioned into spatially-sorted blocks
 * so that the runtime can seek and load individual blocks on demand.
 *
 * Layout:
 *   [OocBlockFileHeader]
 *   [Block 0: aabbMin, aabbMax, atomCount, atoms...]
 *   [Block 1: ...]
 *   ...
 */

#ifndef OOC_BLOCK_FILE_IO_H
#define OOC_BLOCK_FILE_IO_H

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "outofcore/outofcore_types.h"

namespace ooc {

/**
 * Write sorted atoms to a block file and populate block metadata.
 *
 * @param path       Output file path.
 * @param atoms      Sorted atom array (x,y,z,radius).
 * @param numAtoms   Total atom count.
 * @param sceneMin   Scene AABB minimum.
 * @param sceneMax   Scene AABB maximum.
 * @param[out] blocks Vector filled with metadata for every block written.
 * @return true on success.
 */
bool writeBlockFile(const std::string& path,
                    const glm::vec4* atoms,
                    uint32_t numAtoms,
                    glm::vec3 sceneMin,
                    glm::vec3 sceneMax,
                    std::vector<OocBlockMetadata>& blocks);

/**
 * Read a single block's atom data from a previously written file.
 *
 * @param path   Block file path.
 * @param meta   Metadata for the block to read.
 * @param[out] outAtoms Atom data.
 * @return true on success.
 */
bool readBlock(const std::string& path,
               const OocBlockMetadata& meta,
               std::vector<glm::vec4>& outAtoms);

} // namespace ooc

#endif // OOC_BLOCK_FILE_IO_H
