/**
 * @file streaming_manager.h
 * @brief CPU-side LRU block pool manager with double-buffered GPU upload.
 *
 * Coordinates GPU request readback, LRU eviction, disk I/O, and async
 * cudaMemcpy uploads. Inspired by the LRU brick cache in GigaVoxels
 * (Crassin et al. 2009).
 */

#ifndef OOC_STREAMING_MANAGER_H
#define OOC_STREAMING_MANAGER_H

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "outofcore/outofcore_types.h"

namespace ooc {

/**
 * Double-buffered atom pool on the GPU. One buffer is read during
 * rendering while the other receives new uploads.
 */
struct DoubleBufferedPool {
    glm::vec4* d_atomPool[2]         = {};
    int32_t*   d_blockSlotMap[2]     = {};
    int        activeBuffer          = 0;
    int        numSlots              = 0;
};

class StreamingManager {
public:
    StreamingManager() = default;
    ~StreamingManager();

    /**
     * Allocate GPU pool buffers and CPU bookkeeping arrays.
     *
     * @param numSlots       Number of block slots in the pool.
     * @param totalBlocks    Total number of blocks in the dataset.
     * @param blockFilePath  Path to the binary block file on disk.
     * @param blockMeta      Per-block metadata from preprocessing.
     */
    void initialize(int numSlots,
                    int totalBlocks,
                    const std::string& blockFilePath,
                    const std::vector<OocBlockMetadata>& blockMeta);

    void destroy();

    /**
     * Process GPU request buffer: evict LRU slots and upload new blocks.
     *
     * @param h_requestBuffer  Host-side array of requested block IDs.
     * @param requestCount     Number of unique requests.
     * @param h_usedBlockIds   Host-side array of block IDs used this frame
     *                         (already in the pool — to refresh timestamps).
     * @param usedCount        Number of used block IDs.
     * @param currentFrame     Current frame number for LRU timestamps.
     * @param uploadStream     CUDA stream used for async memcpy.
     */
    void processRequests(const uint32_t* h_requestBuffer,
                         uint32_t requestCount,
                         const uint32_t* h_usedBlockIds,
                         uint32_t usedCount,
                         uint64_t currentFrame,
                         cudaStream_t uploadStream);

    /**
     * Swap the active read/write buffers at the end of the frame.
     * Must be called after processRequests and before next render.
     */
    void swapBuffers();

    /** @return pointer to the currently-readable atom pool (read buffer). */
    glm::vec4* getReadAtomPool()   const;
    /** @return pointer to the currently-readable slot map (read buffer). */
    int32_t*   getReadSlotMap()    const;
    /** @return pointer to the write atom pool (upload target). */
    glm::vec4* getWriteAtomPool()  const;
    /** @return pointer to the write slot map (upload target). */
    int32_t*   getWriteSlotMap()   const;

    int getNumSlots() const { return m_pool.numSlots; }

private:
    DoubleBufferedPool              m_pool;
    std::vector<OocBlockSlot>       m_cpuSlots[2];
    std::vector<OocBlockMetadata>   m_blockMeta;
    std::string                     m_blockFilePath;
    int                             m_totalBlocks = 0;

    // Batched upload: pinned staging buffer + GPU staging
    glm::vec4*  m_h_stagingBuffer = nullptr;
    glm::vec4*  m_d_stagingBuffer = nullptr;
    size_t      m_stagingCapacity  = 0;

    // Persistent GPU buffers for scatter metadata (avoid per-frame alloc)
    unsigned int* m_d_slotOffsets = nullptr;
    int32_t*      m_d_slotIds     = nullptr;
    unsigned int* m_d_atomCounts  = nullptr;
    uint32_t*     m_d_slotMapBlockIds = nullptr;
    int32_t*      m_d_slotMapSlots    = nullptr;

    int findEvictionSlot(int bufIdx, uint64_t currentFrame) const;
};

} // namespace ooc

#endif // OOC_STREAMING_MANAGER_H
