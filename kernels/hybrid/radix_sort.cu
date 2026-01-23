/**
 * @file radix_sort.cu
 * @brief Kernel 2: Radix sort for large billboards by depth (front-to-back)
 * 
 * Uses CUB library for efficient parallel radix sort.
 * Sorts billboards by center depth to enable early-Z rejection during tiled rendering.
 */

#include <cuda_runtime.h>
#include <cuda.h>  // Required for CUDA_VERSION definition before GLM
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

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %d [%s, %d]: %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

extern "C" void allocateRadixSortTempStorage(
    RadixSortTempStorage** storage,
    size_t maxElements,
    cudaStream_t stream)
{
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    RadixSortTempStorage* s = new RadixSortTempStorage();
    s->max_elements = maxElements;
    
    // Allocate key/value buffers
    CUDA_CHECK(cudaMalloc(&s->d_keys_in, maxElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&s->d_keys_out, maxElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&s->d_values_in, maxElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&s->d_values_out, maxElements * sizeof(unsigned int)));
    
    // Initialize buffers to zero to avoid uninitialized memory issues
    CUDA_CHECK(cudaMemset(s->d_keys_in, 0, maxElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(s->d_keys_out, 0, maxElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(s->d_values_in, 0, maxElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(s->d_values_out, 0, maxElements * sizeof(unsigned int)));
    
    // Determine temporary storage size
    s->d_temp_storage = nullptr;
    s->temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        s->d_temp_storage, s->temp_storage_bytes,
        s->d_keys_in, s->d_keys_out,
        s->d_values_in, s->d_values_out,
        (int)maxElements));
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&s->d_temp_storage, s->temp_storage_bytes));
    
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
    
    // Ensure count doesn't exceed allocated storage
    if (count > storage->max_elements) {
        printf("Warning: Sort count %u exceeds max elements %zu, clamping\n", 
               count, storage->max_elements);
        count = (unsigned int)storage->max_elements;
    }
    
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    
    // Extract keys
    extractSphereSortKeysKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_keys_in, storage->d_values_in, count);
    CUDA_CHECK(cudaGetLastError());
    
    // Sort
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        storage->d_temp_storage, storage->temp_storage_bytes,
        storage->d_keys_in, storage->d_keys_out,
        storage->d_values_in, storage->d_values_out,
        (int)count, 0, 32, stream));
    
    // Reorder billboards
    reorderSphereBillboardsKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_values_out, d_billboards_out, count);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchCylinderBillboardSort(
    CylinderBillboard* d_billboards_in,
    CylinderBillboard* d_billboards_out,
    unsigned int count,
    RadixSortTempStorage* storage,
    cudaStream_t stream)
{
    if (count == 0) return;
    
    // Ensure count doesn't exceed allocated storage
    if (count > storage->max_elements) {
        printf("Warning: Sort count %u exceeds max elements %zu, clamping\n", 
               count, storage->max_elements);
        count = (unsigned int)storage->max_elements;
    }
    
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    
    // Extract keys
    extractCylinderSortKeysKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_keys_in, storage->d_values_in, count);
    CUDA_CHECK(cudaGetLastError());
    
    // Sort
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        storage->d_temp_storage, storage->temp_storage_bytes,
        storage->d_keys_in, storage->d_keys_out,
        storage->d_values_in, storage->d_values_out,
        (int)count, 0, 32, stream));
    
    // Reorder billboards
    reorderCylinderBillboardsKernel<<<grid, block, 0, stream>>>(
        d_billboards_in, storage->d_values_out, d_billboards_out, count);
    CUDA_CHECK(cudaGetLastError());
}
