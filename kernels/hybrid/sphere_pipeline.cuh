/**
 * @file sphere_pipeline.cuh
 * @brief Complete sphere processing pipeline - from culling to rasterization
 * 
 * Encapsulates all sphere-specific processing:
 * 1. Frustum culling + BBox + Classification
 * 2. Radix sort for large spheres
 * 3. Tile binning for large spheres
 * 4. Rasterization (small direct + large tiled)
 */

#ifndef SPHERE_PIPELINE_CUH
#define SPHERE_PIPELINE_CUH

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "hybrid_types.h"

// Forward declarations
struct RadixSortTempStorage;
struct TileBinningData;

/**
 * @brief Resources for sphere pipeline
 */
struct SpherePipelineResources {
    // Classification output
    unsigned int* d_smallIndices;
    unsigned int* d_smallCount;
    unsigned int* d_largeCount;
    SphereBillboard* d_largeBillboards;
    SphereBillboard* d_largeSorted;
    
    // Statistics
    unsigned int* d_frustumPassedCount;
    
    // Host pinned memory for readback
    unsigned int* h_smallCount;
    unsigned int* h_largeCount;
    unsigned int* h_frustumPassedCount;
    
    // Sort and binning storage
    RadixSortTempStorage* sortStorage;
    TileBinningData* tileBinning;
    
    // Configuration
    int maxSpheres;
    int tilesX;
    int tilesY;
};

/**
 * @brief Initialize sphere pipeline resources
 */
void initSpherePipelineResources(
    SpherePipelineResources* resources,
    int maxSpheres,
    int tilesX,
    int tilesY);

/**
 * @brief Free sphere pipeline resources
 */
void freeSpherePipelineResources(SpherePipelineResources* resources);

/**
 * @brief Reset counters before processing
 */
void resetSpherePipelineCounters(SpherePipelineResources* resources, cudaStream_t stream);

/**
 * @brief Execute complete sphere pipeline
 * 
 * Processes all spheres through:
 * 1. Frustum culling + BBox extraction + Size classification
 * 2. Radix sort of large billboards (front-to-back)
 * 3. Tile binning for large billboards
 * 4. Small sphere direct rasterization
 * 5. Large sphere tiled rasterization
 */
void executeSpherePipeline(
    const glm::vec4* d_spheres,
    int sphereCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    SpherePipelineResources* resources,
    cudaStream_t stream);

/**
 * @brief Get statistics from last execution
 */
void getSpherePipelineStats(
    SpherePipelineResources* resources,
    unsigned int* outFrustumPassed,
    unsigned int* outSmallCount,
    unsigned int* outLargeCount);

#endif // SPHERE_PIPELINE_CUH
