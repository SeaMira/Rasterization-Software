/**
 * @file radix_sort.cu
 * @brief Kernel 2: Radix sort for large billboards by depth (front-to-back)
 * 
 * Uses CUB library for efficient parallel radix sort.
 * Sorts billboards by center depth to enable early-Z rejection during tiled rendering.
 */

#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_INLINE

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <glm/glm.hpp>

#include "hybrid_types.h"

// ============================================================================
// Key extraction kernels
// ============================================================================

__global__ void extractSphereSortKeysKernel(
    const SphereBillboard* __restrict__ billboards,
    unsigned int* __restrict__ keys,
    unsigned int* __restrict__ values,
    unsigned int count)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Convert depth to uint for sorting (IEEE 754 floats sort correctly as uints for positive values)
    float depth = billboards[idx].centerDepth;
    // Flip sign bit for correct ordering (smaller depth = closer = should come first)
    unsigned int key = __float_as_uint(depth);
    // Handle negative depths (behind camera) - push to end
    if (depth < 0.0f) {
        key = 0xFFFFFFFF;
    }
    
    keys[idx] = key;
    values[idx] = idx;
}

__global__ void extractCylinderSortKeysKernel(
    const CylinderBillboard* __restrict__ billboards,
    unsigned int* __restrict__ keys,
    unsigned int* __restrict__ values,
    unsigned int count)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    float depth = billboards[idx].centerDepth;
    unsigned int key = __float_as_uint(depth);
    if (depth < 0.0f) {
        key = 0xFFFFFFFF;
    }
    
    keys[idx] = key;
    values[idx] = idx;
}

// ============================================================================
// Reorder billboards by sorted indices
// ============================================================================

__global__ void reorderSphereBillboardsKernel(
    const SphereBillboard* __restrict__ input,
    const unsigned int* __restrict__ sortedIndices,
    SphereBillboard* __restrict__ output,
    unsigned int count)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    unsigned int srcIdx = sortedIndices[idx];
    output[idx] = input[srcIdx];
}

__global__ void reorderCylinderBillboardsKernel(
    const CylinderBillboard* __restrict__ input,
    const unsigned int* __restrict__ sortedIndices,
    CylinderBillboard* __restrict__ output,
    unsigned int count)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    unsigned int srcIdx = sortedIndices[idx];
    output[idx] = input[srcIdx];
}

// ============================================================================
// Host wrapper for radix sort
// ============================================================================

// Temporary storage structure
struct RadixSortTempStorage {
    void* d_temp_storage;
    size_t temp_storage_bytes;
    unsigned int* d_keys_in;
    unsigned int* d_keys_out;
    unsigned int* d_values_in;
    unsigned int* d_values_out;
    size_t max_elements;
};

extern "C" void allocateRadixSortTempStorage(
    RadixSortTempStorage** storage,
    size_t maxElements,
    cudaStream_t stream)
{
    RadixSortTempStorage* s = new RadixSortTempStorage();
    s->max_elements = maxElements;
    
    // Allocate key/value buffers
    cudaMalloc(&s->d_keys_in, maxElements * sizeof(unsigned int));
    cudaMalloc(&s->d_keys_out, maxElements * sizeof(unsigned int));
    cudaMalloc(&s->d_values_in, maxElements * sizeof(unsigned int));
    cudaMalloc(&s->d_values_out, maxElements * sizeof(unsigned int));
    
    // Determine temporary storage size
    s->d_temp_storage = nullptr;
    s->temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        s->d_temp_storage, s->temp_storage_bytes,
        s->d_keys_in, s->d_keys_out,
        s->d_values_in, s->d_values_out,
        maxElements);
    
    // Allocate temporary storage
    cudaMalloc(&s->d_temp_storage, s->temp_storage_bytes);
    
    *storage = s;
}

extern "C" void freeRadixSortTempStorage(RadixSortTempStorage* storage) {
    if (storage) {
        cudaFree(storage->d_temp_storage);
        cudaFree(storage->d_keys_in);
        cudaFree(storage->d_keys_out);
        cudaFree(storage->d_values_in);
        cudaFree(storage->d_values_out);
        delete storage;
    }
}

extern "C" void launchSphereBillboardSort(
    SphereBillboard* d_billboards_in,
    SphereBillboard* d_billboards_out,
    unsigned int count,
    RadixSortTempStorage* storage,
    cudaStream_t stream)
{
    if (count == 0) return;
    
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    
    // Extract keys
    extractSphereSortKeysKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_keys_in, storage->d_values_in, count);
    
    // Sort
    cub::DeviceRadixSort::SortPairs(
        storage->d_temp_storage, storage->temp_storage_bytes,
        storage->d_keys_in, storage->d_keys_out,
        storage->d_values_in, storage->d_values_out,
        count, 0, 32, stream);
    
    // Reorder billboards
    reorderSphereBillboardsKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_values_out, d_billboards_out, count);
}

extern "C" void launchCylinderBillboardSort(
    CylinderBillboard* d_billboards_in,
    CylinderBillboard* d_billboards_out,
    unsigned int count,
    RadixSortTempStorage* storage,
    cudaStream_t stream)
{
    if (count == 0) return;
    
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    
    // Extract keys
    extractCylinderSortKeysKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_keys_in, storage->d_values_in, count);
    
    // Sort
    cub::DeviceRadixSort::SortPairs(
        storage->d_temp_storage, storage->temp_storage_bytes,
        storage->d_keys_in, storage->d_keys_out,
        storage->d_values_in, storage->d_values_out,
        count, 0, 32, stream);
    
    // Reorder billboards
    reorderCylinderBillboardsKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_values_out, d_billboards_out, count);
}
