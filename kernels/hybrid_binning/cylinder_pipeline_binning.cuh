/**
 * @file cylinder_pipeline_binning.cuh
 * @brief Cylinder pipeline for binning-by-sort.
 */

#ifndef CYLINDER_PIPELINE_BINNING_CUH
#define CYLINDER_PIPELINE_BINNING_CUH

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

struct CylinderBinningResources {
    unsigned int* d_smallIndices;
    unsigned int* d_smallCount;
    unsigned int* d_frustumPassedCount;
    unsigned int* h_smallCount;
    unsigned int* h_pairCount;
    unsigned int* h_frustumPassedCount;
    unsigned long long* d_tile_entity_pairs;
    unsigned long long* d_tile_entity_pairs_sorted;
    unsigned int* d_pairCount;

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
    int maxCylinders;
    int maxPairs;
    int tilesX;
    int tilesY;
    int totalTiles;
    // Work-group expansion buffers
    unsigned int* d_wg_tileId;       ///< tile index for each work group
    unsigned int* d_wg_entityStart;  ///< start offset in sorted pairs for each WG
    unsigned int* d_wg_entityCount;  ///< entity count per WG (<= SHARED_BATCH_SIZE)
    unsigned int* d_wg_totalCount;   ///< single uint: atomicAdd counter for total WGs
    unsigned int* h_wg_totalCount;   ///< pinned host copy of total WG count
    int maxWorkGroups;               ///< allocated size of WG arrays
};

void initCylinderBinningResources(CylinderBinningResources* r, int maxCylinders, int tilesX, int tilesY, int totalTiles, int avgEntitiesPerTile = 0);
void freeCylinderBinningResources(CylinderBinningResources* r);
void resetCylinderBinningCounters(CylinderBinningResources* r, cudaStream_t stream);

void executeCylinderPipelineBinning(
    const Cylinder* d_cylinders,
    const glm::vec4* d_spheres,
    int cylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    CylinderBinningResources* r,
    cudaStream_t stream);

void getCylinderBinningStats(CylinderBinningResources* r,
    unsigned int* outFrustumPassed, unsigned int* outSmallCount, unsigned int* outLargeCount);

#endif
