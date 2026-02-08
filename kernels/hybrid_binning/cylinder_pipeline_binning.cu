/**
 * @file cylinder_pipeline_binning.cu
 * @brief Cylinder pipeline: Classify -> Generate pairs -> Sort -> RLE -> tileOffsets -> Small/Tiled raster.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cub/cub.cuh>
#include "cylinder_pipeline_binning.cuh"
#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

extern "C" void launchCylinderFrustumBBoxClassify(
    const Cylinder* d_cylinders,
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
extern "C" void buildTileOffsetsFromRLE(
    const unsigned int* d_unique_out, const unsigned int* d_counts_out,
    unsigned int num_runs, unsigned int num_pairs, unsigned int total_tiles,
    unsigned int* d_run_offsets, unsigned int* d_tile_offsets,
    void* d_temp_scan, size_t temp_scan_bytes, cudaStream_t stream);
extern "C" void launchScatterAndFillTileOffsets(
    const unsigned int* d_unique_out, const unsigned int* d_run_offsets,
    unsigned int num_runs, unsigned int num_pairs, unsigned int total_tiles,
    unsigned int* d_tile_offsets, cudaStream_t stream);
extern "C" void launchSmallCylinderRaster(
    const Cylinder* d_cylinders, const unsigned int* d_smallCylinderIndices,
    unsigned int smallCylinderCount, unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage, cudaStream_t stream);
extern "C" void launchTiledCylinderRasterBinning(
    const Cylinder* d_cylinders, const unsigned int* d_tile_offsets,
    const unsigned long long* d_tile_entity_pairs_sorted,
    unsigned int* d_depthBuffer, cudaSurfaceObject_t outputImage,
    unsigned int tilesX, unsigned int tilesY, cudaStream_t stream);

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
}

void resetCylinderBinningCounters(CylinderBinningResources* r, cudaStream_t stream) {
    cudaMemsetAsync(r->d_smallCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(r->d_frustumPassedCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(r->d_pairCount, 0, sizeof(unsigned int), stream);
}

void executeCylinderPipelineBinning(
    const Cylinder* d_cylinders,
    int cylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    CylinderBinningResources* r,
    cudaStream_t stream)
{
    if (cylinderCount == 0) return;

    launchCylinderFrustumBBoxClassify(
        d_cylinders, r->d_smallIndices, r->d_smallCount,
        r->d_tile_entity_pairs, r->d_pairCount, r->d_frustumPassedCount,
        cylinderCount, r->maxPairs, stream);

    cudaMemcpyAsync(r->h_smallCount, r->d_smallCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(r->h_pairCount, r->d_pairCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(r->h_frustumPassedCount, r->d_frustumPassedCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    // cudaStreamSynchronize(stream);

    unsigned int smallCount = *r->h_smallCount;
    unsigned int numPairs = *r->h_pairCount;

    if (numPairs > 0) {
        
        // cudaStreamSynchronize(stream);
        if (numPairs > (unsigned int)r->maxPairs) numPairs = r->maxPairs;

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

        buildTileOffsetsFromRLE(
            r->d_unique_out, r->d_counts_out, numRuns, numPairs, r->totalTiles,
            r->d_run_offsets, r->d_tile_offsets,
            r->d_temp_scan, r->temp_scan_bytes, stream);
        launchScatterAndFillTileOffsets(
            r->d_unique_out, r->d_run_offsets, numRuns, numPairs, r->totalTiles,
            r->d_tile_offsets, stream);
    } else {
        cudaMemsetAsync(r->d_tile_offsets, 0, (r->totalTiles + 1) * sizeof(unsigned int), stream);
    }

    if (smallCount > 0) {
        launchSmallCylinderRaster(d_cylinders, r->d_smallIndices, smallCount, d_depthBuffer, outputImage, stream);
    }

    if (numPairs > 0) {
        launchTiledCylinderRasterBinning(
            d_cylinders, r->d_tile_offsets, r->d_tile_entity_pairs_sorted,
            d_depthBuffer, outputImage, r->tilesX, r->tilesY, stream);
    }
}

void getCylinderBinningStats(CylinderBinningResources* r,
    unsigned int* outFrustumPassed, unsigned int* outSmallCount, unsigned int* outLargeCount)
{
    if (outFrustumPassed) *outFrustumPassed = *r->h_frustumPassedCount;
    if (outSmallCount) *outSmallCount = *r->h_smallCount;
    if (outLargeCount) *outLargeCount = *r->h_pairCount;
}
