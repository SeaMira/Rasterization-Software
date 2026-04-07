/**
 * @file cylinder_pipeline_binning.cu
 * @brief Cylinder pipeline: Classify -> Generate pairs -> Sort -> RLE -> tileOffsets -> Small/Tiled raster.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cub/cub.cuh>
#include <nvtx3/nvToolsExt.h>
#include "cylinder_pipeline_binning.cuh"
#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

extern "C" void launchCylinderFrustumBBoxClassify(
    const Cylinder* d_cylinders,
    const glm::vec4* d_spheres,
    unsigned int* d_smallCylinderIndices,
    unsigned int* d_smallCylinderCount,
    unsigned long long* d_tile_entity_pairs,
    unsigned int* d_pair_count,
    unsigned int* d_frustumPassedCount,
    int cylinderCount,
    int maxPairs,
    cudaStream_t stream);
extern "C" void sortPairs64AndBuildTileOffsets(
    unsigned long long* d_pairs_in,
    unsigned long long* d_pairs_out,
    unsigned int num_pairs,
    unsigned int total_tiles,
    unsigned int* d_tile_offsets,
    unsigned int* d_unique_out,
    unsigned int* d_counts_out,
    unsigned int* d_run_offsets,
    unsigned int* d_num_runs,
    void* d_temp_sort64,
    size_t temp_sort64_bytes,
    void* d_temp_rle,
    size_t temp_rle_bytes,
    void* d_temp_scan,
    size_t temp_scan_bytes,
    cudaStream_t stream);
extern "C" void getRleTempStorageBytesForPairs64(size_t* out_bytes, int maxPairs);
extern "C" void launchSmallCylinderRaster(
    const Cylinder* d_cylinders, const glm::vec4* d_spheres,
    const unsigned int* d_smallCylinderIndices,
    unsigned int smallCylinderCount, unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage, cudaStream_t stream);
extern "C" void launchExpandWorkGroups(
    const unsigned int* d_unique_out,
    const unsigned int* d_counts_out,
    const unsigned int* d_run_offsets,
    unsigned int numRuns,
    unsigned int* d_wg_tileId,
    unsigned int* d_wg_entityStart,
    unsigned int* d_wg_entityCount,
    unsigned int* d_wg_totalCount,
    cudaStream_t stream);
extern "C" void launchTiledCylinderRasterWG(
    const Cylinder* d_cylinders,
    const glm::vec4* d_spheres,
    const unsigned long long* d_tile_entity_pairs_sorted,
    const unsigned int* d_wg_tileId,
    const unsigned int* d_wg_entityStart,
    const unsigned int* d_wg_entityCount,
    unsigned int totalWorkGroups,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream);

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) printf("CUDA error %d: %s\n", e, cudaGetErrorString(e)); } while(0)

void initCylinderBinningResources(CylinderBinningResources* r, int maxCylinders, int tilesX, int tilesY, int totalTiles, int avgEntitiesPerTile) {
    cudaGetLastError();
    r->maxCylinders = maxCylinders;
    r->tilesX = tilesX;
    r->tilesY = tilesY;
    r->totalTiles = totalTiles;
    int avgPerTile = (avgEntitiesPerTile > 0) ? avgEntitiesPerTile : AVG_ENTITIES_PER_TILE;
    r->maxPairs = avgPerTile * totalTiles;
    if (r->maxPairs < totalTiles * 4) r->maxPairs = totalTiles * 4;

    CUDA_CHECK(cudaMalloc(&r->d_smallIndices, maxCylinders * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_smallCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_frustumPassedCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(&r->h_smallCount, sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&r->h_pairCount, sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&r->h_frustumPassedCount, sizeof(unsigned int), cudaHostAllocDefault));

    CUDA_CHECK(cudaMalloc(&r->d_pairCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_tile_entity_pairs, r->maxPairs * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&r->d_tile_entity_pairs_sorted, r->maxPairs * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&r->d_tile_offsets, (totalTiles + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_unique_out, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_counts_out, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_run_offsets, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_num_runs, sizeof(unsigned int)));

    r->d_temp_sort64 = nullptr;
    r->temp_sort64_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, r->temp_sort64_bytes,
        (unsigned long long*)nullptr, (unsigned long long*)nullptr, (int)r->maxPairs, 0, 64);
    r->d_temp_rle = nullptr;
    r->d_temp_scan = nullptr;
    r->temp_rle_bytes = 0;
    r->temp_scan_bytes = 0;
    getRleTempStorageBytesForPairs64(&r->temp_rle_bytes, (int)r->maxPairs);
    cub::DeviceScan::ExclusiveSum(nullptr, r->temp_scan_bytes,
        (unsigned int*)nullptr, (unsigned int*)nullptr, (int)r->maxPairs);
    if (r->temp_sort64_bytes > 0) CUDA_CHECK(cudaMalloc(&r->d_temp_sort64, r->temp_sort64_bytes));
    if (r->temp_rle_bytes > 0) CUDA_CHECK(cudaMalloc(&r->d_temp_rle, r->temp_rle_bytes));
    if (r->temp_scan_bytes > 0) CUDA_CHECK(cudaMalloc(&r->d_temp_scan, r->temp_scan_bytes));

    // Work-group expansion buffers
    r->maxWorkGroups = r->maxPairs / SHARED_BATCH_SIZE + totalTiles;
    CUDA_CHECK(cudaMalloc(&r->d_wg_tileId, r->maxWorkGroups * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_wg_entityStart, r->maxWorkGroups * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_wg_entityCount, r->maxWorkGroups * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_wg_totalCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(&r->h_wg_totalCount, sizeof(unsigned int), cudaHostAllocDefault));
}

void freeCylinderBinningResources(CylinderBinningResources* r) {
    if (!r) return;
    cudaFree(r->d_smallIndices);
    cudaFree(r->d_smallCount);
    cudaFree(r->d_frustumPassedCount);
    cudaFreeHost(r->h_smallCount);
    cudaFreeHost(r->h_pairCount);
    cudaFreeHost(r->h_frustumPassedCount);
    cudaFree(r->d_pairCount);
    cudaFree(r->d_tile_entity_pairs);
    cudaFree(r->d_tile_entity_pairs_sorted);
    cudaFree(r->d_tile_offsets);
    cudaFree(r->d_unique_out);
    cudaFree(r->d_counts_out);
    cudaFree(r->d_run_offsets);
    cudaFree(r->d_num_runs);
    if (r->d_temp_sort64) cudaFree(r->d_temp_sort64);
    if (r->d_temp_rle) cudaFree(r->d_temp_rle);
    if (r->d_temp_scan) cudaFree(r->d_temp_scan);
    cudaFree(r->d_wg_tileId);
    cudaFree(r->d_wg_entityStart);
    cudaFree(r->d_wg_entityCount);
    cudaFree(r->d_wg_totalCount);
    cudaFreeHost(r->h_wg_totalCount);
}

void resetCylinderBinningCounters(CylinderBinningResources* r, cudaStream_t stream) {
    cudaMemsetAsync(r->d_smallCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(r->d_frustumPassedCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(r->d_pairCount, 0, sizeof(unsigned int), stream);
}

void executeCylinderPipelineBinning(
    const Cylinder* d_cylinders,
    const glm::vec4* d_spheres,
    int cylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    CylinderBinningResources* r,
    cudaStream_t stream)
{
    if (cylinderCount == 0) return;

    nvtxRangePushA("Cylinder: Classify");
    launchCylinderFrustumBBoxClassify(
        d_cylinders, d_spheres, r->d_smallIndices, r->d_smallCount,
        r->d_tile_entity_pairs, r->d_pairCount, r->d_frustumPassedCount,
        cylinderCount, r->maxPairs, stream);

    cudaMemcpyAsync(r->h_smallCount, r->d_smallCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(r->h_pairCount, r->d_pairCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(r->h_frustumPassedCount, r->d_frustumPassedCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    nvtxRangePop(); // Cylinder: Classify

    unsigned int smallCount = *r->h_smallCount;
    unsigned int numPairs = *r->h_pairCount;

    if (numPairs > 0) {
        if (numPairs > (unsigned int)r->maxPairs) numPairs = r->maxPairs;

        nvtxRangePushA("Cylinder: Sort+RLE+TileOffsets");
        sortPairs64AndBuildTileOffsets(
            r->d_tile_entity_pairs, r->d_tile_entity_pairs_sorted,
            numPairs, r->totalTiles,
            r->d_tile_offsets,
            r->d_unique_out, r->d_counts_out, r->d_run_offsets, r->d_num_runs,
            r->d_temp_sort64, r->temp_sort64_bytes,
            r->d_temp_rle, r->temp_rle_bytes,
            r->d_temp_scan, r->temp_scan_bytes, stream);
        unsigned int numRuns;
        cudaMemcpyAsync(&numRuns, r->d_num_runs, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (numRuns > numPairs) numRuns = numPairs;

        cub::DeviceScan::ExclusiveSum(
            r->d_temp_scan, r->temp_scan_bytes,
            r->d_counts_out, r->d_run_offsets,
            numRuns, stream);
        nvtxRangePop(); // Cylinder: Sort+RLE+TileOffsets

        nvtxRangePushA("Cylinder: Expand WorkGroups");
        cudaMemsetAsync(r->d_wg_totalCount, 0, sizeof(unsigned int), stream);
        launchExpandWorkGroups(
            r->d_unique_out, r->d_counts_out, r->d_run_offsets, numRuns,
            r->d_wg_tileId, r->d_wg_entityStart, r->d_wg_entityCount,
            r->d_wg_totalCount, stream);

        cudaMemcpyAsync(r->h_wg_totalCount, r->d_wg_totalCount,
            sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        unsigned int totalWorkGroups = *r->h_wg_totalCount;
        nvtxRangePop(); // Cylinder: Expand WorkGroups

        if (totalWorkGroups > 0) {
            nvtxRangePushA("Cylinder: Tiled Raster WG");
            launchTiledCylinderRasterWG(
                d_cylinders, d_spheres, r->d_tile_entity_pairs_sorted,
                r->d_wg_tileId, r->d_wg_entityStart, r->d_wg_entityCount,
                totalWorkGroups, d_depthBuffer, outputImage, stream);
            nvtxRangePop();
        }
    }

    if (smallCount > 0) {
        nvtxRangePushA("Cylinder: Small Raster");
        launchSmallCylinderRaster(d_cylinders, d_spheres, r->d_smallIndices, smallCount, d_depthBuffer, outputImage, stream);
        nvtxRangePop();
    }
}

void getCylinderBinningStats(CylinderBinningResources* r,
    unsigned int* outFrustumPassed, unsigned int* outSmallCount, unsigned int* outLargeCount)
{
    if (outFrustumPassed) *outFrustumPassed = *r->h_frustumPassedCount;
    if (outSmallCount) *outSmallCount = *r->h_smallCount;
    if (outLargeCount) *outLargeCount = *r->h_pairCount;
}
