/**
 * @file block_file_io.cpp
 * @brief Binary block file read/write implementation.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "outofcore/block_file_io.h"
#include <cstdio>
#include <algorithm>
#include <iostream>

namespace ooc {

bool writeBlockFile(const std::string& path,
                    const glm::vec4* atoms,
                    uint32_t numAtoms,
                    glm::vec3 sceneMin,
                    glm::vec3 sceneMax,
                    int atomsPerBlock,
                    std::vector<OocBlockMetadata>& blocks)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "[OOC] Failed to open block file for writing: " << path << std::endl;
        return false;
    }

    uint32_t apb = static_cast<uint32_t>(atomsPerBlock);
    uint32_t numBlocks = (numAtoms + apb - 1) / apb;
    blocks.resize(numBlocks);

    OocBlockFileHeader header{};
    header.magic        = OOC_BLOCK_FILE_MAGIC;
    header.numBlocks    = numBlocks;
    header.atomsPerBlock = apb;
    header.sceneMin     = sceneMin;
    header.sceneMax     = sceneMax;
    fwrite(&header, sizeof(header), 1, f);

    uint64_t offset = sizeof(header);

    for (uint32_t b = 0; b < numBlocks; b++) {
        uint32_t start = b * apb;
        uint32_t count = std::min(apb, numAtoms - start);

        glm::vec3 bboxMin(1e18f);
        glm::vec3 bboxMax(-1e18f);
        glm::vec3 centerSum(0.0f);

        for (uint32_t i = start; i < start + count; i++) {
            glm::vec3 p(atoms[i]);
            float r = atoms[i].w;
            bboxMin = glm::min(bboxMin, p - glm::vec3(r));
            bboxMax = glm::max(bboxMax, p + glm::vec3(r));
            centerSum += p;
        }

        blocks[b].aabbMin    = bboxMin;
        blocks[b].aabbMax    = bboxMax;
        blocks[b].center     = centerSum / static_cast<float>(count);
        blocks[b].atomCount  = count;
        blocks[b].fileOffset = offset;
        blocks[b].blockId    = b;

        fwrite(&bboxMin,  sizeof(glm::vec3), 1, f);
        fwrite(&bboxMax,  sizeof(glm::vec3), 1, f);
        fwrite(&count,    sizeof(uint32_t),  1, f);
        fwrite(atoms + start, sizeof(glm::vec4), count, f);

        offset += 2 * sizeof(glm::vec3) + sizeof(uint32_t) + count * sizeof(glm::vec4);
    }

    fclose(f);
    std::cout << "[OOC] Wrote " << numBlocks << " blocks to " << path << std::endl;
    return true;
}

bool readBlock(const std::string& path,
               const OocBlockMetadata& meta,
               std::vector<glm::vec4>& outAtoms)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

#ifdef _MSC_VER
    _fseeki64(f, static_cast<__int64>(meta.fileOffset), SEEK_SET);
#else
    fseeko(f, static_cast<off_t>(meta.fileOffset), SEEK_SET);
#endif

    glm::vec3 aabbMin, aabbMax;
    uint32_t count;
    fread(&aabbMin, sizeof(glm::vec3), 1, f);
    fread(&aabbMax, sizeof(glm::vec3), 1, f);
    fread(&count,   sizeof(uint32_t),  1, f);

    outAtoms.resize(count);
    fread(outAtoms.data(), sizeof(glm::vec4), count, f);

    fclose(f);
    return true;
}

// ─────────────────── Phase 5: Persistent BlockFileReader ───────────────────

BlockFileReader::~BlockFileReader() { close(); }

bool BlockFileReader::open(const std::string& path) {
    close();
    m_file = fopen(path.c_str(), "rb");
    if (!m_file) {
        std::cerr << "[OOC] BlockFileReader: failed to open " << path << std::endl;
        return false;
    }
    return true;
}

void BlockFileReader::close() {
    if (m_file) { fclose(m_file); m_file = nullptr; }
}

uint32_t BlockFileReader::readBlockDirect(const OocBlockMetadata& meta,
                                           glm::vec4* outBuffer)
{
    if (!m_file) return 0;

#ifdef _MSC_VER
    _fseeki64(m_file, static_cast<__int64>(meta.fileOffset), SEEK_SET);
#else
    fseeko(m_file, static_cast<off_t>(meta.fileOffset), SEEK_SET);
#endif

    glm::vec3 aabbMin, aabbMax;
    uint32_t count;
    if (fread(&aabbMin, sizeof(glm::vec3), 1, m_file) != 1) return 0;
    if (fread(&aabbMax, sizeof(glm::vec3), 1, m_file) != 1) return 0;
    if (fread(&count,   sizeof(uint32_t),  1, m_file) != 1) return 0;

    if (fread(outBuffer, sizeof(glm::vec4), count, m_file) != count) return 0;
    return count;
}

} // namespace ooc
