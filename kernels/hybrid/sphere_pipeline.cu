/**
 * @file sphere_pipeline.cu
 * @brief Complete sphere processing pipeline implementation
 */

#include <cuda_runtime.h>
#include <cuda.h>  // Required for CUDA_VERSION definition before GLM
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include <glm/glm.hpp>

#include "sphere_pipeline.cuh"
#include "hybrid_types.h"

// ============================================================================
// External function declarations (from other kernel files)
// ============================================================================

// From frustum_bbox_classify.cu
extern "C" void uploadHybridConstants(const HybridConstants& constants, cudaStream_t stream);

extern "C" void launchSphereFrustumBBoxClassify(
    const glm::vec4* d_spheres,
    unsigned int* d_smallSphereIndices,
    unsigned int* d_smallSphereCount,
    SphereBillboard* d_largeBillboards,
    unsigned int* d_largeSphereCount,
    unsigned int* d_frustumPassedCount,
    int sphereCount,
    cudaStream_t stream);

// From radix_sort.cu
extern "C" void allocateRadixSortTempStorage(
    RadixSortTempStorage** storage,
    size_t maxElements,
    cudaStream_t stream);

extern "C" void freeRadixSortTempStorage(RadixSortTempStorage* storage);

extern "C" void launchSphereBillboardSort(
    SphereBillboard* d_billboards_in,
    SphereBillboard* d_billboards_out,
    unsigned int count,
    RadixSortTempStorage* storage,
    cudaStream_t stream);

// From tile_binning.cu
extern "C" void allocateTileBinningData(
    TileBinningData** data,
    unsigned int tilesX,
    unsigned int tilesY,
    unsigned int maxEntities,
    cudaStream_t stream);

extern "C" void freeTileBinningData(TileBinningData* data);

extern "C" void launchSphereTileBinning(
    const SphereBillboard* d_billboards,
    unsigned int billboardCount,
    TileBinningData* binningData,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream);

// From small_entity_raster.cu
extern "C" void launchSmallSphereRaster(
    const glm::vec4* d_spheres,
    const unsigned int* d_smallSphereIndices,
    unsigned int smallSphereCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream);

// From tiled_raster.cu
extern "C" void launchTiledSphereRaster(
    const SphereBillboard* d_billboards,
    const unsigned int* d_tileOffsets,
    const unsigned int* d_tileCounts,
    const unsigned int* d_tileEntityIndices,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream);

// ============================================================================
// TileBinningData accessor (need to expose internals)
// ============================================================================

// Forward declare the full structure from tile_binning.cu
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

// ============================================================================
// Pipeline Resource Management
// ============================================================================

// Helper macro for CUDA error checking
#define CUDA_CHECK_PIPELINE(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %d [%s, %d]: %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

void initSpherePipelineResources(
    SpherePipelineResources* resources,
    int maxSpheres,
    int tilesX,
    int tilesY)
{
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    resources->maxSpheres = maxSpheres;
    resources->tilesX = tilesX;
    resources->tilesY = tilesY;
    
    // Classification output buffers
    CUDA_CHECK_PIPELINE(cudaMalloc(&resources->d_smallIndices, maxSpheres * sizeof(unsigned int)));
    CUDA_CHECK_PIPELINE(cudaMalloc(&resources->d_smallCount, sizeof(unsigned int)));
    CUDA_CHECK_PIPELINE(cudaMalloc(&resources->d_largeCount, sizeof(unsigned int)));
    CUDA_CHECK_PIPELINE(cudaMalloc(&resources->d_largeBillboards, maxSpheres * sizeof(SphereBillboard)));
    CUDA_CHECK_PIPELINE(cudaMalloc(&resources->d_largeSorted, maxSpheres * sizeof(SphereBillboard)));
    
    // Statistics
    CUDA_CHECK_PIPELINE(cudaMalloc(&resources->d_frustumPassedCount, sizeof(unsigned int)));
    
    // Host pinned memory for readback
    CUDA_CHECK_PIPELINE(cudaHostAlloc(&resources->h_smallCount, sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK_PIPELINE(cudaHostAlloc(&resources->h_largeCount, sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK_PIPELINE(cudaHostAlloc(&resources->h_frustumPassedCount, sizeof(unsigned int), cudaHostAllocDefault));
    
    // Initialize counters to zero
    CUDA_CHECK_PIPELINE(cudaMemset(resources->d_smallCount, 0, sizeof(unsigned int)));
    CUDA_CHECK_PIPELINE(cudaMemset(resources->d_largeCount, 0, sizeof(unsigned int)));
    CUDA_CHECK_PIPELINE(cudaMemset(resources->d_frustumPassedCount, 0, sizeof(unsigned int)));
    
    // Radix sort storage
    allocateRadixSortTempStorage(&resources->sortStorage, maxSpheres, nullptr);
    
    // Tile binning data
    allocateTileBinningData(&resources->tileBinning, tilesX, tilesY, maxSpheres, nullptr);
}

void freeSpherePipelineResources(SpherePipelineResources* resources)
{
    if (!resources) return;
    
    cudaFree(resources->d_smallIndices);
    cudaFree(resources->d_smallCount);
    cudaFree(resources->d_largeCount);
    cudaFree(resources->d_largeBillboards);
    cudaFree(resources->d_largeSorted);
    cudaFree(resources->d_frustumPassedCount);
    
    cudaFreeHost(resources->h_smallCount);
    cudaFreeHost(resources->h_largeCount);
    cudaFreeHost(resources->h_frustumPassedCount);
    
    freeRadixSortTempStorage(resources->sortStorage);
    freeTileBinningData(resources->tileBinning);
}

void resetSpherePipelineCounters(SpherePipelineResources* resources, cudaStream_t stream)
{
    cudaMemsetAsync(resources->d_smallCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(resources->d_largeCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(resources->d_frustumPassedCount, 0, sizeof(unsigned int), stream);
}

// ============================================================================
// Main Pipeline Execution
// ============================================================================

void executeSpherePipeline(
    const glm::vec4* d_spheres,
    int sphereCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    SpherePipelineResources* resources,
    cudaStream_t stream)
{
    if (sphereCount == 0) return;
    
    nvtxRangePushA("Sphere Pipeline");
    
    // =========================================================================
    // Stage 1: Frustum Culling + BBox Extraction + Size Classification
    // =========================================================================
    nvtxRangePushA("Sphere: Classify");
    
    launchSphereFrustumBBoxClassify(
        d_spheres,
        resources->d_smallIndices,
        resources->d_smallCount,
        resources->d_largeBillboards,
        resources->d_largeCount,
        resources->d_frustumPassedCount,
        sphereCount,
        stream);
    
    // Readback counts (synchronous - needed for conditional execution)
    cudaMemcpyAsync(resources->h_smallCount, resources->d_smallCount, 
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(resources->h_largeCount, resources->d_largeCount,
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(resources->h_frustumPassedCount, resources->d_frustumPassedCount,
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    unsigned int smallCount = *resources->h_smallCount;
    unsigned int largeCount = *resources->h_largeCount;
    
    nvtxRangePop(); // Classify
    
    // =========================================================================
    // Stage 2: Radix Sort for Large Spheres (front-to-back by depth)
    // =========================================================================
    if (largeCount > 0) {
        nvtxRangePushA("Sphere: Sort");
        
        launchSphereBillboardSort(
            resources->d_largeBillboards,
            resources->d_largeSorted,
            largeCount,
            resources->sortStorage,
            stream);
        
        nvtxRangePop(); // Sort
    }
    
    // =========================================================================
    // Stage 3: Tile Binning for Large Spheres
    // =========================================================================
    if (largeCount > 0) {
        nvtxRangePushA("Sphere: Tile Binning");
        
        launchSphereTileBinning(
            resources->d_largeSorted,
            largeCount,
            resources->tileBinning,
            resources->tilesX,
            resources->tilesY,
            stream);
        
        nvtxRangePop(); // Tile Binning
    }
    
    // =========================================================================
    // Stage 4a: Small Sphere Direct Rasterization
    // =========================================================================
    if (smallCount > 0) {
        nvtxRangePushA("Sphere: Small Raster");
        
        launchSmallSphereRaster(
            d_spheres,
            resources->d_smallIndices,
            smallCount,
            d_depthBuffer,
            outputImage,
            stream);
        
        nvtxRangePop(); // Small Raster
    }
    
    // =========================================================================
    // Stage 4b: Large Sphere Tiled Rasterization
    // =========================================================================
    if (largeCount > 0) {
        nvtxRangePushA("Sphere: Tiled Raster");
        
        // Access tile binning data internals
        TileBinningData* binning = resources->tileBinning;
        
        launchTiledSphereRaster(
            resources->d_largeSorted,
            binning->d_sphereTileOffsets,
            binning->d_sphereTileCounts,
            binning->d_sphereTileEntityIndices,
            d_depthBuffer,
            outputImage,
            resources->tilesX,
            resources->tilesY,
            stream);
        
        nvtxRangePop(); // Tiled Raster
    }
    
    nvtxRangePop(); // Sphere Pipeline
}

void getSpherePipelineStats(
    SpherePipelineResources* resources,
    unsigned int* outFrustumPassed,
    unsigned int* outSmallCount,
    unsigned int* outLargeCount)
{
    if (outFrustumPassed) *outFrustumPassed = *resources->h_frustumPassedCount;
    if (outSmallCount) *outSmallCount = *resources->h_smallCount;
    if (outLargeCount) *outLargeCount = *resources->h_largeCount;
}
