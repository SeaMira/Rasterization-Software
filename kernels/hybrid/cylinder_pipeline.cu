/**
 * @file cylinder_pipeline.cu
 * @brief Complete cylinder processing pipeline implementation
 */

#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_INLINE

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include <glm/glm.hpp>

#include "cylinder_pipeline.cuh"
#include "hybrid_types.h"
#include "geometry/cylinder/cylinder.h"

// ============================================================================
// External function declarations (from other kernel files)
// ============================================================================

// From frustum_bbox_classify.cu
extern "C" void launchCylinderFrustumBBoxClassify(
    const Cylinder* d_cylinders,
    unsigned int* d_smallCylinderIndices,
    unsigned int* d_smallCylinderCount,
    CylinderBillboard* d_largeBillboards,
    unsigned int* d_largeCylinderCount,
    unsigned int* d_frustumPassedCount,
    int cylinderCount,
    cudaStream_t stream);

// From radix_sort.cu
extern "C" void allocateRadixSortTempStorage(
    RadixSortTempStorage** storage,
    size_t maxElements,
    cudaStream_t stream);

extern "C" void freeRadixSortTempStorage(RadixSortTempStorage* storage);

extern "C" void launchCylinderBillboardSort(
    CylinderBillboard* d_billboards_in,
    CylinderBillboard* d_billboards_out,
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

extern "C" void launchCylinderTileBinning(
    const CylinderBillboard* d_billboards,
    unsigned int billboardCount,
    TileBinningData* binningData,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream);

// From small_entity_raster.cu
extern "C" void launchSmallCylinderRaster(
    const Cylinder* d_cylinders,
    const unsigned int* d_smallCylinderIndices,
    unsigned int smallCylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream);

// From tiled_raster.cu
extern "C" void launchTiledCylinderRaster(
    const CylinderBillboard* d_billboards,
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

void initCylinderPipelineResources(
    CylinderPipelineResources* resources,
    int maxCylinders,
    int tilesX,
    int tilesY)
{
    resources->maxCylinders = maxCylinders;
    resources->tilesX = tilesX;
    resources->tilesY = tilesY;
    
    // Classification output buffers
    cudaMalloc(&resources->d_smallIndices, maxCylinders * sizeof(unsigned int));
    cudaMalloc(&resources->d_smallCount, sizeof(unsigned int));
    cudaMalloc(&resources->d_largeCount, sizeof(unsigned int));
    cudaMalloc(&resources->d_largeBillboards, maxCylinders * sizeof(CylinderBillboard));
    cudaMalloc(&resources->d_largeSorted, maxCylinders * sizeof(CylinderBillboard));
    
    // Statistics
    cudaMalloc(&resources->d_frustumPassedCount, sizeof(unsigned int));
    
    // Host pinned memory for readback
    cudaHostAlloc(&resources->h_smallCount, sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&resources->h_largeCount, sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&resources->h_frustumPassedCount, sizeof(unsigned int), cudaHostAllocDefault);
    
    // Radix sort storage
    allocateRadixSortTempStorage(&resources->sortStorage, maxCylinders, nullptr);
    
    // Tile binning data
    allocateTileBinningData(&resources->tileBinning, tilesX, tilesY, maxCylinders, nullptr);
}

void freeCylinderPipelineResources(CylinderPipelineResources* resources)
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

void resetCylinderPipelineCounters(CylinderPipelineResources* resources, cudaStream_t stream)
{
    cudaMemsetAsync(resources->d_smallCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(resources->d_largeCount, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(resources->d_frustumPassedCount, 0, sizeof(unsigned int), stream);
}

// ============================================================================
// Main Pipeline Execution
// ============================================================================

void executeCylinderPipeline(
    const Cylinder* d_cylinders,
    int cylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    CylinderPipelineResources* resources,
    cudaStream_t stream)
{
    if (cylinderCount == 0) return;
    
    nvtxRangePushA("Cylinder Pipeline");
    
    // =========================================================================
    // Stage 1: Frustum Culling + BBox Extraction + Size Classification
    // =========================================================================
    nvtxRangePushA("Cylinder: Classify");
    
    launchCylinderFrustumBBoxClassify(
        d_cylinders,
        resources->d_smallIndices,
        resources->d_smallCount,
        resources->d_largeBillboards,
        resources->d_largeCount,
        resources->d_frustumPassedCount,
        cylinderCount,
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
    // Stage 2: Radix Sort for Large Cylinders (front-to-back by depth)
    // =========================================================================
    if (largeCount > 0) {
        nvtxRangePushA("Cylinder: Sort");
        
        launchCylinderBillboardSort(
            resources->d_largeBillboards,
            resources->d_largeSorted,
            largeCount,
            resources->sortStorage,
            stream);
        
        nvtxRangePop(); // Sort
    }
    
    // =========================================================================
    // Stage 3: Tile Binning for Large Cylinders
    // =========================================================================
    if (largeCount > 0) {
        nvtxRangePushA("Cylinder: Tile Binning");
        
        launchCylinderTileBinning(
            resources->d_largeSorted,
            largeCount,
            resources->tileBinning,
            resources->tilesX,
            resources->tilesY,
            stream);
        
        nvtxRangePop(); // Tile Binning
    }
    
    // =========================================================================
    // Stage 4a: Small Cylinder Direct Rasterization
    // =========================================================================
    if (smallCount > 0) {
        nvtxRangePushA("Cylinder: Small Raster");
        
        launchSmallCylinderRaster(
            d_cylinders,
            resources->d_smallIndices,
            smallCount,
            d_depthBuffer,
            outputImage,
            stream);
        
        nvtxRangePop(); // Small Raster
    }
    
    // =========================================================================
    // Stage 4b: Large Cylinder Tiled Rasterization
    // =========================================================================
    if (largeCount > 0) {
        nvtxRangePushA("Cylinder: Tiled Raster");
        
        // Access tile binning data internals
        TileBinningData* binning = resources->tileBinning;
        
        launchTiledCylinderRaster(
            resources->d_largeSorted,
            binning->d_cylinderTileOffsets,
            binning->d_cylinderTileCounts,
            binning->d_cylinderTileEntityIndices,
            d_depthBuffer,
            outputImage,
            resources->tilesX,
            resources->tilesY,
            stream);
        
        nvtxRangePop(); // Tiled Raster
    }
    
    nvtxRangePop(); // Cylinder Pipeline
}

void getCylinderPipelineStats(
    CylinderPipelineResources* resources,
    unsigned int* outFrustumPassed,
    unsigned int* outSmallCount,
    unsigned int* outLargeCount)
{
    if (outFrustumPassed) *outFrustumPassed = *resources->h_frustumPassedCount;
    if (outSmallCount) *outSmallCount = *resources->h_smallCount;
    if (outLargeCount) *outLargeCount = *resources->h_largeCount;
}
