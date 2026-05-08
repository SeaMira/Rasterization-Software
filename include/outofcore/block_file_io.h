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

#include <cstdio>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "outofcore/outofcore_types.h"

namespace ooc {

/**
 * Write sorted atoms to a block file and populate block metadata.
 */
bool writeBlockFile(const std::string& path,
                    const glm::vec4* atoms,
                    uint32_t numAtoms,
                    glm::vec3 sceneMin,
                    glm::vec3 sceneMax,
                    int atomsPerBlock,
                    std::vector<OocBlockMetadata>& blocks);

/**
 * Read a single block's atom data from a previously written file.
 * (Legacy — opens/closes file per call. Prefer BlockFileReader.)
 */
bool readBlock(const std::string& path,
               const OocBlockMetadata& meta,
               std::vector<glm::vec4>& outAtoms);

// ─────────────────────────────────────────────────────────
// Phase 5: Persistent file reader — keeps file handle open
// across all block reads within a session, eliminating
// fopen/fclose overhead (~256 calls/frame → 0).
// ─────────────────────────────────────────────────────────

class BlockFileReader {
public:
    BlockFileReader() = default;
    ~BlockFileReader();

    BlockFileReader(const BlockFileReader&) = delete;
    BlockFileReader& operator=(const BlockFileReader&) = delete;

    bool open(const std::string& path);
    void close();
    bool isOpen() const { return m_file != nullptr; }

    /**
     * Read a block's atoms directly into a caller-owned buffer.
     * No heap allocation — writes directly to outBuffer.
     * @param meta      Block metadata (file offset, atom count).
     * @param outBuffer Pre-allocated buffer (must hold at least meta.atomCount vec4s).
     * @return Atom count actually read, or 0 on failure.
     */
    uint32_t readBlockDirect(const OocBlockMetadata& meta, glm::vec4* outBuffer);

private:
    FILE* m_file = nullptr;
};

} // namespace ooc

#endif // OOC_BLOCK_FILE_IO_H
