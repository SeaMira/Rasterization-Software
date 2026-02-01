/**
 * @file generate_pairs.cu
 * @brief Generate (TileID, EntityID) pairs for binning-by-sort.
 * Each large entity emits one pair per overlapping tile; atomic counter for output index.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "hybrid_binning_types.h"

extern __constant__ HybridConstants hybridCst;

__global__ void generateSpherePairsKernel(
    const SphereBillboard* __restrict__ billboards,
    unsigned int billboardCount,
    unsigned int* __restrict__ d_keys,    // TileID
    unsigned int* __restrict__ d_values,  // EntityID (original index)
    unsigned int* __restrict__ d_pairCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= billboardCount) return;
    
    const SphereBillboard& bb = billboards[idx];
    unsigned int entityId = bb.originalIndex;
    
    for (unsigned int ty = bb.tileMinY; ty <= bb.tileMaxY; ++ty) {
        for (unsigned int tx = bb.tileMinX; tx <= bb.tileMaxX; ++tx) {
            unsigned int tileId = ty * hybridCst.tilesX + tx;
            unsigned int outIdx = atomicAdd(d_pairCount, 1u);
            d_keys[outIdx] = tileId;
            d_values[outIdx] = entityId;
        }
    }
}

__global__ void generateCylinderPairsKernel(
    const CylinderBillboard* __restrict__ billboards,
    unsigned int billboardCount,
    unsigned int* __restrict__ d_keys,
    unsigned int* __restrict__ d_values,
    unsigned int* __restrict__ d_pairCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= billboardCount) return;
    
    const CylinderBillboard& bb = billboards[idx];
    unsigned int entityId = bb.originalIndex;
    
    for (unsigned int ty = bb.tileMinY; ty <= bb.tileMaxY; ++ty) {
        for (unsigned int tx = bb.tileMinX; tx <= bb.tileMaxX; ++tx) {
            unsigned int tileId = ty * hybridCst.tilesX + tx;
            unsigned int outIdx = atomicAdd(d_pairCount, 1u);
            d_keys[outIdx] = tileId;
            d_values[outIdx] = entityId;
        }
    }
}

extern "C" void launchGenerateSpherePairs(
    const SphereBillboard* d_billboards,
    unsigned int billboardCount,
    unsigned int* d_keys,
    unsigned int* d_values,
    unsigned int* d_pairCount,
    cudaStream_t stream)
{
    if (billboardCount == 0) return;
    cudaMemsetAsync(d_pairCount, 0, sizeof(unsigned int), stream);
    
    dim3 block(256);
    dim3 grid((billboardCount + block.x - 1) / block.x);
    generateSpherePairsKernel<<<grid, block, 0, stream>>>(
        d_billboards, billboardCount, d_keys, d_values, d_pairCount);
}

extern "C" void launchGenerateCylinderPairs(
    const CylinderBillboard* d_billboards,
    unsigned int billboardCount,
    unsigned int* d_keys,
    unsigned int* d_values,
    unsigned int* d_pairCount,
    cudaStream_t stream)
{
    if (billboardCount == 0) return;
    cudaMemsetAsync(d_pairCount, 0, sizeof(unsigned int), stream);
    
    dim3 block(256);
    dim3 grid((billboardCount + block.x - 1) / block.x);
    generateCylinderPairsKernel<<<grid, block, 0, stream>>>(
        d_billboards, billboardCount, d_keys, d_values, d_pairCount);
}
