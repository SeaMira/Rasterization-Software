/**
 * @file cylinder_pipeline_binning.cu
 * @brief Cylinder pipeline: Classify -> Generate pairs -> Sort -> RLE -> tileOffsets -> Small/Tiled raster.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cub/cub.cuh>
#include "cylinder_pipeline_binning.cuh"
#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

extern "C" void launchCylinderFrustumBBoxClassify(
    const Cylinder* d_cylinders,
    unsigned int* d_smallCylinderIndices,
    unsigned int* d_smallCylinderCount,
    CylinderBillboard* d_largeBillboards,
    unsigned int* d_largeCylinderCount,
    unsigned int* d_frustumPassedCount,
    int cylinderCount,
    cudaStream_t stream);
extern "C" void launchGenerateCylinderPairs(
    const CylinderBillboard* d_billboards,
    unsigned int billboardCount,
    unsigned int* d_keys,
    unsigned int* d_values,
    unsigned int* d_pairCount,
    cudaStream_t stream);
extern "C" void sortPairsAndBuildTileOffsets(
    unsigned int* d_keys_in, unsigned int* d_values_in,
    unsigned int* d_keys_out, unsigned int* d_values_out,
    unsigned int num_pairs, unsigned int total_tiles,
    unsigned int* d_tile_offsets,
    unsigned int* d_unique_out, unsigned int* d_counts_out,
    unsigned int* d_run_offsets, unsigned int* d_num_runs,
    void* d_temp_sort, size_t temp_sort_bytes_sort,
    void* d_temp_rle, size_t temp_rle_bytes,
    void* d_temp_scan, size_t temp_scan_bytes,
    cudaStream_t stream);
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
    const unsigned int* d_sorted_entity_indices,
    unsigned int* d_depthBuffer, cudaSurfaceObject_t outputImage,
    unsigned int tilesX, unsigned int tilesY, cudaStream_t stream);

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) printf("CUDA error %d: %s\n", e, cudaGetErrorString(e)); } while(0)

void initCylinderBinningResources(CylinderBinningResources* r, int maxCylinders, int tilesX, int tilesY, int totalTiles) {
    cudaGetLastError();
    r->maxCylinders = maxCylinders;
    r->tilesX = tilesX;
    r->tilesY = tilesY;
    r->totalTiles = totalTiles;
    r->maxPairs = maxCylinders * 64;
    if (r->maxPairs < totalTiles * 4) r->maxPairs = totalTiles * 4;

    CUDA_CHECK(cudaMalloc(&r->d_smallIndices, maxCylinders * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_smallCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_largeCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_largeBillboards, maxCylinders * sizeof(CylinderBillboard)));
    CUDA_CHECK(cudaMalloc(&r->d_frustumPassedCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(&r->h_smallCount, sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&r->h_largeCount, sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&r->h_frustumPassedCount, sizeof(unsigned int), cudaHostAllocDefault));

    CUDA_CHECK(cudaMalloc(&r->d_pairCount, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_keys_in, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_values_in, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_keys_out, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_values_out, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_tile_offsets, (totalTiles + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_unique_out, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_counts_out, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_run_offsets, r->maxPairs * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&r->d_num_runs, sizeof(unsigned int)));

    r->d_temp_sort = nullptr;
    r->d_temp_rle = nullptr;
    r->d_temp_scan = nullptr;
    r->temp_sort_bytes = 0;
    r->temp_rle_bytes = 0;
    r->temp_scan_bytes = 0;
    sortPairsAndBuildTileOffsets(
        r->d_keys_in, r->d_values_in, r->d_keys_out, r->d_values_out,
        1, totalTiles, r->d_tile_offsets, r->d_unique_out, r->d_counts_out,
        r->d_run_offsets, r->d_num_runs,
        nullptr, r->temp_sort_bytes, nullptr, r->temp_rle_bytes,
        nullptr, r->temp_scan_bytes, nullptr);
    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (unsigned int*)nullptr, (unsigned int*)nullptr, (int)r->maxPairs);
    if (scan_bytes > r->temp_scan_bytes) r->temp_scan_bytes = scan_bytes;
    if (r->temp_sort_bytes > 0) CUDA_CHECK(cudaMalloc(&r->d_temp_sort, r->temp_sort_bytes));
    if (r->temp_rle_bytes > 0) CUDA_CHECK(cudaMalloc(&r->d_temp_rle, r->temp_rle_bytes));
    if (r->temp_scan_bytes > 0) CUDA_CHECK(cudaMalloc(&r->d_temp_scan, r->temp_scan_bytes));
}

void freeCylinderBinningResources(CylinderBinningResources* r) {
    if (!r) return;
    cudaFree(r->d_smallIndices);
    cudaFree(r->d_smallCount);
    cudaFree(r->d_largeCount);
    cudaFree(r->d_largeBillboards);
    cudaFree(r->d_frustumPassedCount);
    cudaFreeHost(r->h_smallCount);
    cudaFreeHost(r->h_largeCount);
    cudaFreeHost(r->h_frustumPassedCount);
    cudaFree(r->d_pairCount);
    cudaFree(r->d_keys_in);
    cudaFree(r->d_values_in);
    cudaFree(r->d_keys_out);
    cudaFree(r->d_values_out);
    cudaFree(r->d_tile_offsets);
    cudaFree(r->d_unique_out);
    cudaFree(r->d_counts_out);
    cudaFree(r->d_run_offsets);
    cudaFree(r->d_num_runs);
    if (r->d_temp_sort) cudaFree(r->d_temp_sort);
    if (r->d_temp_rle) cudaFree(r->d_temp_rle);
    if (r->d_temp_scan) cudaFree(r->d_temp_scan);
}

void resetCylinderBinningCounters(CylinderBinningResources* r, cudaStream_t stream) {
    cudaMemsetAsync(r->d_smallCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(r->d_largeCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(r->d_frustumPassedCount, 0, sizeof(unsigned int), stream);
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
        r->d_largeBillboards, r->d_largeCount, r->d_frustumPassedCount,
        cylinderCount, stream);

    cudaMemcpyAsync(r->h_smallCount, r->d_smallCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(r->h_largeCount, r->d_largeCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(r->h_frustumPassedCount, r->d_frustumPassedCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    unsigned int smallCount = *r->h_smallCount;
    unsigned int largeCount = *r->h_largeCount;

    if (largeCount > 0) {
        launchGenerateCylinderPairs(
            r->d_largeBillboards, largeCount,
            r->d_keys_in, r->d_values_in, r->d_pairCount, stream);
        unsigned int numPairs;
        cudaMemcpyAsync(&numPairs, r->d_pairCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (numPairs > (unsigned int)r->maxPairs) numPairs = r->maxPairs;

        sortPairsAndBuildTileOffsets(
            r->d_keys_in, r->d_values_in, r->d_keys_out, r->d_values_out,
            numPairs, r->totalTiles, r->d_tile_offsets,
            r->d_unique_out, r->d_counts_out, r->d_run_offsets, r->d_num_runs,
            r->d_temp_sort, r->temp_sort_bytes,
            r->d_temp_rle, r->temp_rle_bytes,
            r->d_temp_scan, r->temp_scan_bytes, stream);
        unsigned int numRuns;
        cudaMemcpyAsync(&numRuns, r->d_num_runs, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

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

    if (largeCount > 0) {
        launchTiledCylinderRasterBinning(
            d_cylinders, r->d_tile_offsets, r->d_values_out,
            d_depthBuffer, outputImage, r->tilesX, r->tilesY, stream);
    }
}

void getCylinderBinningStats(CylinderBinningResources* r,
    unsigned int* outFrustumPassed, unsigned int* outSmallCount, unsigned int* outLargeCount)
{
    if (outFrustumPassed) *outFrustumPassed = *r->h_frustumPassedCount;
    if (outSmallCount) *outSmallCount = *r->h_smallCount;
    if (outLargeCount) *outLargeCount = *r->h_largeCount;
}
