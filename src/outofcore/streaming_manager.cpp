/**
 * @file streaming_manager.cpp
 * @brief CPU-side LRU block pool manager with double-buffered GPU upload.
 *
 * Uses batched staging buffer to reduce many cudaMemcpyAsync calls to
 * a single transfer + scatter kernel.
 */

#include "outofcore/streaming_manager.h"
#include "outofcore/block_file_io.h"

#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

#define OOC_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess)                                                  \
            std::cerr << "[OOC CUDA] " << cudaGetErrorString(e)                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
    } while (0)

extern "C" void launchScatterStagingToPool(
    const glm::vec4* d_staging, glm::vec4* d_atomPool,
    const unsigned int* d_slotOffsets, const int32_t* d_slotIds,
    const unsigned int* d_atomCounts, unsigned int numUploads,
    cudaStream_t stream);

extern "C" void launchApplySlotMapUpdates(
    int32_t* d_blockSlotMap, const uint32_t* d_blockIds,
    const int32_t* d_slots, unsigned int numUpdates,
    cudaStream_t stream);

namespace ooc {

StreamingManager::~StreamingManager()
{
    destroy();
}

void StreamingManager::initialize(int numSlots,
                                  int totalBlocks,
                                  const std::string& blockFilePath,
                                  const std::vector<OocBlockMetadata>& blockMeta)
{
    m_pool.numSlots   = numSlots;
    m_totalBlocks     = totalBlocks;
    m_blockFilePath   = blockFilePath;
    m_blockMeta       = blockMeta;

    size_t atomPoolBytes = static_cast<size_t>(numSlots) * OOC_ATOMS_PER_BLOCK * sizeof(glm::vec4);
    size_t slotMapBytes  = static_cast<size_t>(totalBlocks) * sizeof(int32_t);

    for (int b = 0; b < 2; b++)
    {
        OOC_CUDA_CHECK(cudaMalloc(&m_pool.d_atomPool[b], atomPoolBytes));
        OOC_CUDA_CHECK(cudaMalloc(&m_pool.d_blockSlotMap[b], slotMapBytes));
        OOC_CUDA_CHECK(cudaMemset(m_pool.d_atomPool[b], 0, atomPoolBytes));

        std::vector<int32_t> invalidMap(totalBlocks, -1);
        OOC_CUDA_CHECK(cudaMemcpy(m_pool.d_blockSlotMap[b], invalidMap.data(),
                                  slotMapBytes, cudaMemcpyHostToDevice));

        m_cpuSlots[b].resize(numSlots);
    }

    m_stagingCapacity = static_cast<size_t>(OOC_MAX_REQUESTS_PER_FRAME) *
                        OOC_ATOMS_PER_BLOCK * sizeof(glm::vec4);
    OOC_CUDA_CHECK(cudaHostAlloc(&m_h_stagingBuffer, m_stagingCapacity, cudaHostAllocDefault));
    OOC_CUDA_CHECK(cudaMalloc(&m_d_stagingBuffer, m_stagingCapacity));

    OOC_CUDA_CHECK(cudaMalloc(&m_d_slotOffsets, OOC_MAX_REQUESTS_PER_FRAME * sizeof(unsigned int)));
    OOC_CUDA_CHECK(cudaMalloc(&m_d_slotIds,     OOC_MAX_REQUESTS_PER_FRAME * sizeof(int32_t)));
    OOC_CUDA_CHECK(cudaMalloc(&m_d_atomCounts,  OOC_MAX_REQUESTS_PER_FRAME * sizeof(unsigned int)));
    OOC_CUDA_CHECK(cudaMalloc(&m_d_slotMapBlockIds, OOC_MAX_REQUESTS_PER_FRAME * 2 * sizeof(uint32_t)));
    OOC_CUDA_CHECK(cudaMalloc(&m_d_slotMapSlots,    OOC_MAX_REQUESTS_PER_FRAME * 2 * sizeof(int32_t)));

    std::cout << "[OOC] StreamingManager: " << numSlots << " slots, "
              << totalBlocks << " blocks, pool = "
              << (atomPoolBytes / (1024 * 1024)) << " MB per buffer." << std::endl;
}

void StreamingManager::destroy()
{
    for (int b = 0; b < 2; b++) {
        if (m_pool.d_atomPool[b])     { cudaFree(m_pool.d_atomPool[b]);     m_pool.d_atomPool[b] = nullptr; }
        if (m_pool.d_blockSlotMap[b]) { cudaFree(m_pool.d_blockSlotMap[b]); m_pool.d_blockSlotMap[b] = nullptr; }
    }
    if (m_h_stagingBuffer)     { cudaFreeHost(m_h_stagingBuffer);     m_h_stagingBuffer = nullptr; }
    if (m_d_stagingBuffer)     { cudaFree(m_d_stagingBuffer);         m_d_stagingBuffer = nullptr; }
    if (m_d_slotOffsets)       { cudaFree(m_d_slotOffsets);           m_d_slotOffsets = nullptr; }
    if (m_d_slotIds)           { cudaFree(m_d_slotIds);               m_d_slotIds = nullptr; }
    if (m_d_atomCounts)        { cudaFree(m_d_atomCounts);            m_d_atomCounts = nullptr; }
    if (m_d_slotMapBlockIds)   { cudaFree(m_d_slotMapBlockIds);       m_d_slotMapBlockIds = nullptr; }
    if (m_d_slotMapSlots)      { cudaFree(m_d_slotMapSlots);          m_d_slotMapSlots = nullptr; }
}

int StreamingManager::findEvictionSlot(int bufIdx, uint64_t currentFrame) const {
    int best = -1;
    uint64_t oldest = currentFrame + 1;

    for (int s = 0; s < m_pool.numSlots; s++)
    {
        if (!m_cpuSlots[bufIdx][s].valid)
            return s;
        if (m_cpuSlots[bufIdx][s].lastUsedFrame < oldest) 
        {
            oldest = m_cpuSlots[bufIdx][s].lastUsedFrame;
            best = s;
        }
    }
    return best;
}

void StreamingManager::processRequests(const uint32_t* h_requestBuffer,
                                       uint32_t requestCount,
                                       const uint32_t* h_usedBlockIds,
                                       uint32_t usedCount,
                                       uint64_t currentFrame,
                                       cudaStream_t uploadStream)
{
    int writeBuf = 1 - m_pool.activeBuffer;
    auto& slots  = m_cpuSlots[writeBuf];
    int readBuf  = m_pool.activeBuffer;

    slots = m_cpuSlots[readBuf];
    OOC_CUDA_CHECK(cudaMemcpyAsync(
        m_pool.d_blockSlotMap[writeBuf],
        m_pool.d_blockSlotMap[readBuf],
        static_cast<size_t>(m_totalBlocks) * sizeof(int32_t),
        cudaMemcpyDeviceToDevice, uploadStream));
    OOC_CUDA_CHECK(cudaMemcpyAsync(
        m_pool.d_atomPool[writeBuf],
        m_pool.d_atomPool[readBuf],
        static_cast<size_t>(m_pool.numSlots) * OOC_ATOMS_PER_BLOCK * sizeof(glm::vec4),
        cudaMemcpyDeviceToDevice, uploadStream));

    for (uint32_t i = 0; i < usedCount; i++) {
        uint32_t bid = h_usedBlockIds[i];
        for (int s = 0; s < m_pool.numSlots; s++) {
            if (slots[s].valid && slots[s].blockId == bid) {
                slots[s].lastUsedFrame = currentFrame;
                break;
            }
        }
    }

    // Phase 1: gather all blocks into staging buffer, build metadata
    std::vector<unsigned int> slotOffsets;
    std::vector<int32_t>      slotIds;
    std::vector<unsigned int> atomCounts;
    std::vector<uint32_t>     slotMapBlockIds;
    std::vector<int32_t>      slotMapSlots;

    slotOffsets.reserve(OOC_MAX_REQUESTS_PER_FRAME);
    slotIds.reserve(OOC_MAX_REQUESTS_PER_FRAME);
    atomCounts.reserve(OOC_MAX_REQUESTS_PER_FRAME);
    slotMapBlockIds.reserve(OOC_MAX_REQUESTS_PER_FRAME * 2);
    slotMapSlots.reserve(OOC_MAX_REQUESTS_PER_FRAME * 2);

    unsigned int stagingOffset = 0;

    for (uint32_t r = 0; r < requestCount && slotOffsets.size() < static_cast<size_t>(OOC_MAX_REQUESTS_PER_FRAME); r++) {
        uint32_t blockId = h_requestBuffer[r];
        if (blockId >= static_cast<uint32_t>(m_totalBlocks)) continue;

        bool alreadyLoaded = false;
        for (int s = 0; s < m_pool.numSlots; s++) {
            if (slots[s].valid && slots[s].blockId == blockId) {
                alreadyLoaded = true;
                slots[s].lastUsedFrame = currentFrame;
                break;
            }
        }
        if (alreadyLoaded) continue;

        int slot = findEvictionSlot(writeBuf, currentFrame);
        if (slot < 0) break;

        if (slots[slot].valid) {
            slotMapBlockIds.push_back(slots[slot].blockId);
            slotMapSlots.push_back(-1);
        }

        std::vector<glm::vec4> atomData;
        if (!readBlock(m_blockFilePath, m_blockMeta[blockId], atomData))
            continue;

        size_t count = atomData.size();
        if (stagingOffset + count > (m_stagingCapacity / sizeof(glm::vec4)))
            break;

        std::memcpy(m_h_stagingBuffer + stagingOffset, atomData.data(),
                    count * sizeof(glm::vec4));

        slotOffsets.push_back(static_cast<unsigned int>(stagingOffset));
        slotIds.push_back(slot);
        atomCounts.push_back(static_cast<unsigned int>(count));
        slotMapBlockIds.push_back(blockId);
        slotMapSlots.push_back(slot);

        stagingOffset += count;
        slots[slot].blockId       = blockId;
        slots[slot].lastUsedFrame = currentFrame;
        slots[slot].valid         = true;
    }

    uint32_t numUploads = static_cast<uint32_t>(slotOffsets.size());
    if (numUploads == 0) return;

    // Phase 2: single cudaMemcpyAsync for all atom data
    size_t totalAtoms = stagingOffset;
    OOC_CUDA_CHECK(cudaMemcpyAsync(
        m_d_stagingBuffer,
        m_h_stagingBuffer,
        totalAtoms * sizeof(glm::vec4),
        cudaMemcpyHostToDevice, uploadStream));

    // Phase 3: scatter kernel (uses persistent buffers)
    OOC_CUDA_CHECK(cudaMemcpyAsync(m_d_slotOffsets, slotOffsets.data(),
                                    numUploads * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice, uploadStream));
    OOC_CUDA_CHECK(cudaMemcpyAsync(m_d_slotIds, slotIds.data(),
                                    numUploads * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, uploadStream));
    OOC_CUDA_CHECK(cudaMemcpyAsync(m_d_atomCounts, atomCounts.data(),
                                    numUploads * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice, uploadStream));

    launchScatterStagingToPool(
        m_d_stagingBuffer, m_pool.d_atomPool[writeBuf],
        m_d_slotOffsets, m_d_slotIds, m_d_atomCounts, numUploads, uploadStream);

    // Phase 4: single cudaMemcpyAsync + kernel for slot map updates
    uint32_t numSlotMapUpdates = static_cast<uint32_t>(slotMapBlockIds.size());
    if (numSlotMapUpdates > 0) {
        OOC_CUDA_CHECK(cudaMemcpyAsync(m_d_slotMapBlockIds, slotMapBlockIds.data(),
                                        numSlotMapUpdates * sizeof(uint32_t),
                                        cudaMemcpyHostToDevice, uploadStream));
        OOC_CUDA_CHECK(cudaMemcpyAsync(m_d_slotMapSlots, slotMapSlots.data(),
                                        numSlotMapUpdates * sizeof(int32_t),
                                        cudaMemcpyHostToDevice, uploadStream));

        launchApplySlotMapUpdates(
            m_pool.d_blockSlotMap[writeBuf], m_d_slotMapBlockIds, m_d_slotMapSlots,
            numSlotMapUpdates, uploadStream);
    }
}

void StreamingManager::swapBuffers() 
{
    m_pool.activeBuffer = 1 - m_pool.activeBuffer;
}

glm::vec4* StreamingManager::getReadAtomPool()  const { return m_pool.d_atomPool[m_pool.activeBuffer]; }
int32_t*   StreamingManager::getReadSlotMap()   const { return m_pool.d_blockSlotMap[m_pool.activeBuffer]; }
glm::vec4* StreamingManager::getWriteAtomPool() const { return m_pool.d_atomPool[1 - m_pool.activeBuffer]; }
int32_t*   StreamingManager::getWriteSlotMap()  const { return m_pool.d_blockSlotMap[1 - m_pool.activeBuffer]; }

} // namespace ooc
