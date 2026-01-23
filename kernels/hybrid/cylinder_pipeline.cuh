/**
 * @file cylinder_pipeline.cuh
 * @brief Complete cylinder processing pipeline - from culling to rasterization
 * 
 * Encapsulates all cylinder-specific processing:
 * 1. Frustum culling + BBox + Classification
 * 2. Radix sort for large cylinders
 * 3. Tile binning for large cylinders
 * 4. Rasterization (small direct + large tiled)
 */

#ifndef CYLINDER_PIPELINE_CUH
#define CYLINDER_PIPELINE_CUH

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "hybrid_types.h"
#include "geometry/cylinder/cylinder.h"

// Forward declarations
struct RadixSortTempStorage;
struct TileBinningData;

/**
 * @brief Resources for cylinder pipeline
 */
struct CylinderPipelineResources {
    // Classification output
    unsigned int* d_smallIndices;
    unsigned int* d_smallCount;
    unsigned int* d_largeCount;
    CylinderBillboard* d_largeBillboards;
    CylinderBillboard* d_largeSorted;
    
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
    int maxCylinders;
    int tilesX;
    int tilesY;
};

/**
 * @brief Initialize cylinder pipeline resources
 */
void initCylinderPipelineResources(
    CylinderPipelineResources* resources,
    int maxCylinders,
    int tilesX,
    int tilesY);

/**
 * @brief Free cylinder pipeline resources
 */
void freeCylinderPipelineResources(CylinderPipelineResources* resources);

/**
 * @brief Reset counters before processing
 */
void resetCylinderPipelineCounters(CylinderPipelineResources* resources, cudaStream_t stream);

/**
 * @brief Execute complete cylinder pipeline
 * 
 * Processes all cylinders through:
 * 1. Frustum culling + BBox extraction + Size classification
 * 2. Radix sort of large billboards (front-to-back)
 * 3. Tile binning for large billboards
 * 4. Small cylinder direct rasterization
 * 5. Large cylinder tiled rasterization
 */
void executeCylinderPipeline(
    const Cylinder* d_cylinders,
    int cylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    CylinderPipelineResources* resources,
    cudaStream_t stream);

/**
 * @brief Get statistics from last execution
 */
void getCylinderPipelineStats(
    CylinderPipelineResources* resources,
    unsigned int* outFrustumPassed,
    unsigned int* outSmallCount,
    unsigned int* outLargeCount);

#endif // CYLINDER_PIPELINE_CUH
