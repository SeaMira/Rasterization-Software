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
    unsigned int* d_largeCount;
    CylinderBillboard* d_largeBillboards;
    unsigned int* d_frustumPassedCount;
    unsigned int* h_smallCount;
    unsigned int* h_largeCount;
    unsigned int* h_frustumPassedCount;
    unsigned int* d_pairCount;
    unsigned int* d_keys_in;
    unsigned int* d_values_in;
    unsigned int* d_keys_out;
    unsigned int* d_values_out;
    unsigned int* d_tile_offsets;
    unsigned int* d_unique_out;
    unsigned int* d_counts_out;
    unsigned int* d_run_offsets;
    unsigned int* d_num_runs;
    void* d_temp_sort;
    size_t temp_sort_bytes;
    void* d_temp_rle;
    size_t temp_rle_bytes;
    void* d_temp_scan;
    size_t temp_scan_bytes;
    int maxCylinders;
    int maxPairs;
    int tilesX;
    int tilesY;
    int totalTiles;
};

void initCylinderBinningResources(CylinderBinningResources* r, int maxCylinders, int tilesX, int tilesY, int totalTiles);
void freeCylinderBinningResources(CylinderBinningResources* r);
void resetCylinderBinningCounters(CylinderBinningResources* r, cudaStream_t stream);

void executeCylinderPipelineBinning(
    const Cylinder* d_cylinders,
    int cylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    CylinderBinningResources* r,
    cudaStream_t stream);

void getCylinderBinningStats(CylinderBinningResources* r,
    unsigned int* outFrustumPassed, unsigned int* outSmallCount, unsigned int* outLargeCount);

#endif
