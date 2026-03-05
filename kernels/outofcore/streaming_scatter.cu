/**
 * @file streaming_scatter.cu
 * @brief Kernel to scatter batched block data from staging buffer to pool slots.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "outofcore/outofcore_types.h"

static __constant__ int scatterAtomsPerBlock;

__global__ void scatterStagingToPoolKernel(
    const glm::vec4* __restrict__ d_staging,
    glm::vec4*       __restrict__ d_atomPool,
    const unsigned int* __restrict__ d_slotOffsets,
    const int32_t*   __restrict__ d_slotIds,
    const unsigned int* __restrict__ d_atomCounts,
    unsigned int     numUploads)
{
    unsigned int uploadIdx = blockIdx.x;
    if (uploadIdx >= numUploads) return;

    int32_t slot      = d_slotIds[uploadIdx];
    unsigned int cnt  = d_atomCounts[uploadIdx];
    unsigned int soff = d_slotOffsets[uploadIdx];

    const glm::vec4* src = d_staging + soff;
    glm::vec4* dst       = d_atomPool + static_cast<size_t>(slot) * scatterAtomsPerBlock;

    for (unsigned int i = threadIdx.x; i < cnt; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void launchScatterStagingToPool(
    const glm::vec4*    d_staging,
    glm::vec4*          d_atomPool,
    const unsigned int* d_slotOffsets,
    const int32_t*      d_slotIds,
    const unsigned int* d_atomCounts,
    unsigned int        numUploads,
    int                 atomsPerBlock,
    cudaStream_t        stream)
{
    if (numUploads == 0) return;
    cudaMemcpyToSymbolAsync(scatterAtomsPerBlock, &atomsPerBlock, sizeof(int), 0,
                            cudaMemcpyHostToDevice, stream);
    scatterStagingToPoolKernel<<<numUploads, 256, 0, stream>>>(
        d_staging, d_atomPool, d_slotOffsets, d_slotIds, d_atomCounts, numUploads);
}

__global__ void applySlotMapUpdatesKernel(
    int32_t*       __restrict__ d_blockSlotMap,
    const uint32_t* __restrict__ d_blockIds,
    const int32_t* __restrict__ d_slots,
    unsigned int   numUpdates)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numUpdates) return;

    uint32_t bid = d_blockIds[idx];
    int32_t slot = d_slots[idx];
    d_blockSlotMap[bid] = slot;
}

extern "C" void launchApplySlotMapUpdates(
    int32_t*       d_blockSlotMap,
    const uint32_t* d_blockIds,
    const int32_t*  d_slots,
    unsigned int    numUpdates,
    cudaStream_t    stream)
{
    if (numUpdates == 0) return;
    dim3 block(256);
    dim3 grid((numUpdates + block.x - 1) / block.x);
    applySlotMapUpdatesKernel<<<grid, block, 0, stream>>>(
        d_blockSlotMap, d_blockIds, d_slots, numUpdates);
}
