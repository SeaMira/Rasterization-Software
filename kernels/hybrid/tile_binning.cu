/**
 * @file tile_binning.cu
 * @brief Kernel 3: Tile binning for large billboards
 * 
 * Assigns sorted billboards to tiles for efficient per-tile processing.
 * Uses atomic operations to build per-tile entity lists.
 */

#include <cuda_runtime.h>
#include <cuda.h>  // Required for CUDA_VERSION definition before GLM
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "hybrid_types.h"

// Access to constants from frustum_bbox_classify.cu
extern __constant__ HybridConstants hybridCst;

// ============================================================================
// Tile Binning Kernels
// ============================================================================

/**
 * @brief Count how many entities each tile will receive
 * First pass: count entities per tile
 */
__global__ void countSpheresPerTileKernel(
    const SphereBillboard* __restrict__ billboards,
    unsigned int* __restrict__ tileCounts,  // [tilesX * tilesY]
    unsigned int billboardCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= billboardCount) return;
    
    const SphereBillboard& bb = billboards[idx];
    
    // Iterate over tiles this billboard overlaps
    for (unsigned int ty = bb.tileMinY; ty <= bb.tileMaxY; ++ty) {
        for (unsigned int tx = bb.tileMinX; tx <= bb.tileMaxX; ++tx) {
            unsigned int tileIdx = ty * hybridCst.tilesX + tx;
            atomicAdd(&tileCounts[tileIdx], 1u);
        }
    }
}

__global__ void countCylindersPerTileKernel(
    const CylinderBillboard* __restrict__ billboards,
    unsigned int* __restrict__ tileCounts,  // [tilesX * tilesY]
    unsigned int billboardCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= billboardCount) return;
    
    const CylinderBillboard& bb = billboards[idx];
    
    for (unsigned int ty = bb.tileMinY; ty <= bb.tileMaxY; ++ty) {
        for (unsigned int tx = bb.tileMinX; tx <= bb.tileMaxX; ++tx) {
            unsigned int tileIdx = ty * hybridCst.tilesX + tx;
            atomicAdd(&tileCounts[tileIdx], 1u);
        }
    }
}

/**
 * @brief Compute prefix sum of tile counts to get offsets
 * Uses simple serial prefix sum on GPU (good enough for small tile counts)
 */
__global__ void computeTileOffsetsKernel(
    const unsigned int* __restrict__ tileCounts,
    unsigned int* __restrict__ tileOffsets,
    unsigned int* __restrict__ totalCount,
    unsigned int tileCount)
{
    // Single thread kernel for simplicity (could use parallel scan for many tiles)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    unsigned int offset = 0;
    for (unsigned int i = 0; i < tileCount; ++i) {
        tileOffsets[i] = offset;
        offset += tileCounts[i];
    }
    *totalCount = offset;
}

/**
 * @brief Bin spheres into tiles
 * Second pass: write entity indices to per-tile lists
 */
__global__ void binSpheresToTilesKernel(
    const SphereBillboard* __restrict__ billboards,
    const unsigned int* __restrict__ tileOffsets,
    unsigned int* __restrict__ tileCurrentCounts,  // Atomic counters per tile
    unsigned int* __restrict__ tileEntityIndices,  // Flat array of entity indices
    unsigned int billboardCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= billboardCount) return;
    
    const SphereBillboard& bb = billboards[idx];
    
    for (unsigned int ty = bb.tileMinY; ty <= bb.tileMaxY; ++ty) {
        for (unsigned int tx = bb.tileMinX; tx <= bb.tileMaxX; ++tx) {
            unsigned int tileIdx = ty * hybridCst.tilesX + tx;
            unsigned int offset = tileOffsets[tileIdx];
            unsigned int localIdx = atomicAdd(&tileCurrentCounts[tileIdx], 1u);
            tileEntityIndices[offset + localIdx] = idx;
        }
    }
}

__global__ void binCylindersToTilesKernel(
    const CylinderBillboard* __restrict__ billboards,
    const unsigned int* __restrict__ tileOffsets,
    unsigned int* __restrict__ tileCurrentCounts,
    unsigned int* __restrict__ tileEntityIndices,
    unsigned int billboardCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= billboardCount) return;
    
    const CylinderBillboard& bb = billboards[idx];
    
    for (unsigned int ty = bb.tileMinY; ty <= bb.tileMaxY; ++ty) {
        for (unsigned int tx = bb.tileMinX; tx <= bb.tileMaxX; ++tx) {
            unsigned int tileIdx = ty * hybridCst.tilesX + tx;
            unsigned int offset = tileOffsets[tileIdx];
            unsigned int localIdx = atomicAdd(&tileCurrentCounts[tileIdx], 1u);
            tileEntityIndices[offset + localIdx] = idx;
        }
    }
}

// ============================================================================
// Tile Binning Data Structure
// ============================================================================

struct TileBinningData {
    unsigned int* d_sphereTileCounts;
    unsigned int* d_sphereTileOffsets;
    unsigned int* d_sphereTileCurrentCounts;
    unsigned int* d_sphereTileEntityIndices;
    unsigned int* d_sphereTotalEntries;
    
    unsigned int* d_cylinderTileCounts;
    unsigned int* d_cylinderTileOffsets;
    unsigned int* d_cylinderTileCurrentCounts;
    unsigned int* d_cylinderTileEntityIndices;
    unsigned int* d_cylinderTotalEntries;
    
    unsigned int tileCount;
    unsigned int maxEntriesPerType;
};

// Helper macro for CUDA error checking
#define CUDA_CHECK_BINNING(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %d [%s, %d]: %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

extern "C" void allocateTileBinningData(
    TileBinningData** data,
    unsigned int tilesX,
    unsigned int tilesY,
    unsigned int maxEntities,
    cudaStream_t stream)
{
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    TileBinningData* d = new TileBinningData();
    d->tileCount = tilesX * tilesY;
    // Each entity can span multiple tiles, estimate max entries
    d->maxEntriesPerType = maxEntities * 16; // Assume average 16 tiles per large entity
    
    // Sphere binning
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_sphereTileCounts, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_sphereTileOffsets, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_sphereTileCurrentCounts, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_sphereTileEntityIndices, d->maxEntriesPerType * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_sphereTotalEntries, sizeof(unsigned int)));
    
    // Cylinder binning
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_cylinderTileCounts, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_cylinderTileOffsets, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_cylinderTileCurrentCounts, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_cylinderTileEntityIndices, d->maxEntriesPerType * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMalloc(&d->d_cylinderTotalEntries, sizeof(unsigned int)));
    
    // Initialize to zero
    CUDA_CHECK_BINNING(cudaMemset(d->d_sphereTileCounts, 0, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMemset(d->d_sphereTileOffsets, 0, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMemset(d->d_cylinderTileCounts, 0, d->tileCount * sizeof(unsigned int)));
    CUDA_CHECK_BINNING(cudaMemset(d->d_cylinderTileOffsets, 0, d->tileCount * sizeof(unsigned int)));
    
    *data = d;
}

extern "C" void freeTileBinningData(TileBinningData* data) {
    if (data) {
        cudaFree(data->d_sphereTileCounts);
        cudaFree(data->d_sphereTileOffsets);
        cudaFree(data->d_sphereTileCurrentCounts);
        cudaFree(data->d_sphereTileEntityIndices);
        cudaFree(data->d_sphereTotalEntries);
        
        cudaFree(data->d_cylinderTileCounts);
        cudaFree(data->d_cylinderTileOffsets);
        cudaFree(data->d_cylinderTileCurrentCounts);
        cudaFree(data->d_cylinderTileEntityIndices);
        cudaFree(data->d_cylinderTotalEntries);
        
        delete data;
    }
}

extern "C" void launchSphereTileBinning(
    const SphereBillboard* d_billboards,
    unsigned int billboardCount,
    TileBinningData* binningData,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    if (billboardCount == 0) return;
    
    unsigned int tileCount = tilesX * tilesY;
    
    // Reset counters
    cudaMemsetAsync(binningData->d_sphereTileCounts, 0, tileCount * sizeof(unsigned int), stream);
    cudaMemsetAsync(binningData->d_sphereTileCurrentCounts, 0, tileCount * sizeof(unsigned int), stream);
    
    dim3 block(256);
    dim3 grid((billboardCount + block.x - 1) / block.x);
    
    // Pass 1: Count entities per tile
    countSpheresPerTileKernel<<<grid, block, 0, stream>>>(
        d_billboards, binningData->d_sphereTileCounts, billboardCount);
    
    // Compute offsets (serial, single block)
    computeTileOffsetsKernel<<<1, 1, 0, stream>>>(
        binningData->d_sphereTileCounts,
        binningData->d_sphereTileOffsets,
        binningData->d_sphereTotalEntries,
        tileCount);
    
    // Pass 2: Bin entities to tiles
    binSpheresToTilesKernel<<<grid, block, 0, stream>>>(
        d_billboards,
        binningData->d_sphereTileOffsets,
        binningData->d_sphereTileCurrentCounts,
        binningData->d_sphereTileEntityIndices,
        billboardCount);
}

extern "C" void launchCylinderTileBinning(
    const CylinderBillboard* d_billboards,
    unsigned int billboardCount,
    TileBinningData* binningData,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    if (billboardCount == 0) return;
    
    unsigned int tileCount = tilesX * tilesY;
    
    // Reset counters
    cudaMemsetAsync(binningData->d_cylinderTileCounts, 0, tileCount * sizeof(unsigned int), stream);
    cudaMemsetAsync(binningData->d_cylinderTileCurrentCounts, 0, tileCount * sizeof(unsigned int), stream);
    
    dim3 block(256);
    dim3 grid((billboardCount + block.x - 1) / block.x);
    
    // Pass 1: Count entities per tile
    countCylindersPerTileKernel<<<grid, block, 0, stream>>>(
        d_billboards, binningData->d_cylinderTileCounts, billboardCount);
    
    // Compute offsets
    computeTileOffsetsKernel<<<1, 1, 0, stream>>>(
        binningData->d_cylinderTileCounts,
        binningData->d_cylinderTileOffsets,
        binningData->d_cylinderTotalEntries,
        tileCount);
    
    // Pass 2: Bin entities to tiles
    binCylindersToTilesKernel<<<grid, block, 0, stream>>>(
        d_billboards,
        binningData->d_cylinderTileOffsets,
        binningData->d_cylinderTileCurrentCounts,
        binningData->d_cylinderTileEntityIndices,
        billboardCount);
}
