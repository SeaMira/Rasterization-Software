/**
 * @file sphere_pipeline_binning.cuh
 * @brief Sphere pipeline for binning-by-sort (Generate -> Sort -> RLE -> tileOffsets -> tiled raster).
 */

#ifndef SPHERE_PIPELINE_BINNING_CUH
#define SPHERE_PIPELINE_BINNING_CUH

#include <cuda_runtime.h>
#include "hybrid_binning_types.h"

struct SphereBinningResources {
    unsigned int* d_smallIndices;
    unsigned int* d_smallCount;
    unsigned int* d_frustumPassedCount;
    unsigned int* h_smallCount;
    unsigned int* h_pairCount;
    unsigned int* h_frustumPassedCount;
    /** 64-bit buffer: (tile_id << 32) | entity_id, written by classifier with atomics */
    unsigned long long* d_tile_entity_pairs;
    /** Sorted 64-bit pairs (output of radix sort) */
    unsigned long long* d_tile_entity_pairs_sorted;
    unsigned int* d_pairCount;
    /** Temp: tile_id per element for RLE input (extracted from 64-bit pairs) */
    unsigned int* d_keys_extract;
    unsigned int* d_tile_offsets;
    unsigned int* d_unique_out;
    unsigned int* d_counts_out;
    unsigned int* d_run_offsets;
    unsigned int* d_num_runs;
    void* d_temp_sort64;
    size_t temp_sort64_bytes;
    void* d_temp_rle;
    size_t temp_rle_bytes;
    void* d_temp_scan;
    size_t temp_scan_bytes;
    int maxSpheres;
    int maxPairs;
    int tilesX;
    int tilesY;
    int totalTiles;
};

void initSphereBinningResources(SphereBinningResources* r, int maxSpheres, int tilesX, int tilesY, int totalTiles, int avgEntitiesPerTile = 0);
void freeSphereBinningResources(SphereBinningResources* r);
void resetSphereBinningCounters(SphereBinningResources* r, cudaStream_t stream);

void executeSpherePipelineBinning(
    const glm::vec4* d_spheres,
    int sphereCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    SphereBinningResources* r,
    cudaStream_t stream);

void getSphereBinningStats(SphereBinningResources* r,
    unsigned int* outFrustumPassed, unsigned int* outSmallCount, unsigned int* outLargeCount);

#endif
