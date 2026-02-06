/**
 * @file sort_and_rle.cu
 * @brief CUB DeviceRadixSort::SortPairs + DeviceRunLengthEncode to build tile offsets.
 * Sort pairs by TileID; RLE on sorted keys; build tileOffsets[0..totalTiles] from RLE output.
 * Also supports 64-bit (tile_id, entity_id) pairs: sort by tile_id (high 32 bits), RLE, extract entity indices.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

// ---- 64-bit pair path: d_pairs = (tile_id << 32) | entity_id ----

/** Functor: extracts tile_id (high 32 bits) from a 64-bit (tile_id, entity_id) pair.
 *  Used with cub::TransformInputIterator to feed RLE without a separate extraction kernel. */
struct ExtractTileId {
    __host__ __device__ __forceinline__
    unsigned int operator()(const unsigned long long& pair) const {
        return static_cast<unsigned int>(pair >> 32);
    }
};

using TileIdIterator = thrust::transform_iterator<ExtractTileId, const unsigned long long*, unsigned int>;

/** Sort 64-bit pairs (tile_id in high 32 bits), run RLE on tile_id, build tile offsets.
 *  Entity index = low 32 bits, extracted in tiled raster.
 *  Uses TransformInputIterator to extract tile_id on-the-fly, avoiding a separate kernel + buffer. */
extern "C" void sortPairs64AndBuildTileOffsets(
    unsigned long long* d_pairs_in,
    unsigned long long* d_pairs_out,
    unsigned int num_pairs,
    unsigned int total_tiles,
    unsigned int* d_tile_offsets,
    unsigned int* d_unique_out,
    unsigned int* d_counts_out,
    unsigned int* d_run_offsets,
    unsigned int* d_num_runs,
    void* d_temp_storage_sort64,
    size_t temp_storage_bytes_sort64,
    void* d_temp_storage_rle,
    size_t temp_storage_bytes_rle,
    void* d_temp_storage_scan,
    size_t temp_storage_bytes_scan,
    cudaStream_t stream)
{
    if (num_pairs == 0) {
        cudaMemsetAsync(d_tile_offsets, 0, (total_tiles + 1) * sizeof(unsigned int), stream);
        return;
    }

    // 1) Radix sort 64-bit pairs (sorts by full key: tile_id in high 32 bits, then entity_id)
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage_sort64,
        temp_storage_bytes_sort64,
        d_pairs_in,
        d_pairs_out,
        num_pairs,
        0, 64,
        stream);

    // 2) Run-length encode: TransformInputIterator extracts tile_id (high 32 bits) lazily from sorted pairs
    TileIdIterator tile_id_iter(d_pairs_out, ExtractTileId{});
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage_rle,
        temp_storage_bytes_rle,
        tile_id_iter,
        d_unique_out,
        d_counts_out,
        d_num_runs,
        num_pairs,
        stream);

    // Caller must: cudaStreamSynchronize, cudaMemcpy d_num_runs to get num_runs, then:
    // buildTileOffsetsFromRLE(..., num_runs, ...)  // does ExclusiveSum on d_counts_out -> d_run_offsets
    // launchScatterAndFillTileOffsets(...)
    // Tiled raster reads d_pairs_out and extracts entity_id = pair & 0xFFFFFFFFu per element.
}

// Sort: keys/values in -> keys/values out (separate buffers)
extern "C" void sortPairsAndBuildTileOffsets(
    unsigned int* d_keys_in,
    unsigned int* d_values_in,
    unsigned int* d_keys_out,
    unsigned int* d_values_out,
    unsigned int num_pairs,
    unsigned int total_tiles,
    unsigned int* d_tile_offsets,
    unsigned int* d_unique_out,
    unsigned int* d_counts_out,
    unsigned int* d_run_offsets,
    unsigned int* d_num_runs,
    void* d_temp_storage_sort,
    size_t temp_storage_bytes_sort,
    void* d_temp_storage_rle,
    size_t temp_storage_bytes_rle,
    void* d_temp_storage_scan,
    size_t temp_storage_bytes_scan,
    cudaStream_t stream)
{
    if (num_pairs == 0) {
        cudaMemsetAsync(d_tile_offsets, 0, (total_tiles + 1) * sizeof(unsigned int), stream);
        return;
    }

    // 1) Sort pairs by key (TileID): in -> out
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage_sort,
        temp_storage_bytes_sort,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_pairs,
        0, 32,
        stream);

    // 2) Run-length encode on sorted keys
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage_rle,
        temp_storage_bytes_rle,
        d_keys_out,
        d_unique_out,
        d_counts_out,
        d_num_runs,
        num_pairs,
        stream);

    // Caller must: cudaStreamSynchronize(stream), then cudaMemcpy from d_num_runs to get num_runs,
    // then call buildTileOffsetsFromRLE and launchScatterAndFillTileOffsets.
}

/** Query CUB for the temp storage bytes needed by RLE with TransformInputIterator on 64-bit pairs.
 *  Callers use this during resource init instead of computing it directly (avoids exposing the iterator type). */
extern "C" void getRleTempStorageBytesForPairs64(size_t* out_bytes, int maxPairs) {
    *out_bytes = 0;
    TileIdIterator dummy_iter(nullptr, ExtractTileId{});
    cub::DeviceRunLengthEncode::Encode(
        nullptr, *out_bytes,
        dummy_iter,
        (unsigned int*)nullptr, (unsigned int*)nullptr,
        (unsigned int*)nullptr, maxPairs);
}

extern "C" void buildTileOffsetsFromRLE(
    const unsigned int* d_unique_out,
    const unsigned int* d_counts_out,
    unsigned int num_runs,
    unsigned int num_pairs,
    unsigned int total_tiles,
    unsigned int* d_run_offsets,     // exclusive sum of d_counts_out, computed here
    unsigned int* d_tile_offsets,
    void* d_temp_storage_scan,
    size_t temp_storage_bytes_scan,
    cudaStream_t stream)
{
    if (num_pairs == 0) {
        cudaMemsetAsync(d_tile_offsets, 0, (total_tiles + 1) * sizeof(unsigned int), stream);
        return;
    }

    // Exclusive sum of run lengths -> run_offsets
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage_scan,
        temp_storage_bytes_scan,
        d_counts_out,
        d_run_offsets,
        num_runs,
        stream);

    // Scatter: tileOffsets[unique_out[i]] = run_offsets[i], and tileOffsets[total_tiles] = num_pairs
    // Then fill gaps. We do this in a kernel.
    cudaMemsetAsync(d_tile_offsets, 0xFF, (total_tiles + 1) * sizeof(unsigned int), stream);
}

// Kernel: scatter run_offsets into tileOffsets, set tileOffsets[total_tiles]=num_pairs
__global__ void scatterTileOffsetsKernel(
    const unsigned int* __restrict__ d_unique_tile_ids,
    const unsigned int* __restrict__ d_run_offsets,
    unsigned int num_runs,
    unsigned int num_pairs,
    unsigned int total_tiles,
    unsigned int* __restrict__ tileOffsets)
{
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_runs; i += blockDim.x * gridDim.x) {
        unsigned int t = d_unique_tile_ids[i];
        if (t < total_tiles)
            tileOffsets[t] = d_run_offsets[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tileOffsets[total_tiles] = num_pairs;
    }
}

// Kernel: fill gaps (tiles with no entities).
// Must propagate BACKWARD so empty tiles get the offset of the NEXT occupied tile.
// Forward propagation is wrong: it copies the START of the previous tile, stealing its entities.
__global__ void fillTileOffsetsGapsKernel(
    unsigned int total_tiles,
    unsigned int* __restrict__ tileOffsets)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // tileOffsets[total_tiles] = num_pairs (set by scatter kernel) acts as sentinel.
    for (int t = (int)total_tiles - 1; t >= 0; --t) {
        if (tileOffsets[t] == 0xFFFFFFFFu)
            tileOffsets[t] = tileOffsets[t + 1];
    }
}

extern "C" void launchScatterAndFillTileOffsets(
    const unsigned int* d_unique_out,
    const unsigned int* d_run_offsets,
    unsigned int num_runs,
    unsigned int num_pairs,
    unsigned int total_tiles,
    unsigned int* d_tile_offsets,
    cudaStream_t stream)
{
    scatterTileOffsetsKernel<<<(num_runs + 255) / 256, 256, 0, stream>>>(
        d_unique_out, d_run_offsets, num_runs, num_pairs, total_tiles, d_tile_offsets);
    fillTileOffsetsGapsKernel<<<1, 1, 0, stream>>>(total_tiles, d_tile_offsets);
}
