/**
 * @file sort_and_rle.cu
 * @brief CUB DeviceRadixSort::SortPairs + DeviceRunLengthEncode to build tile offsets.
 * Sort pairs by TileID; RLE on sorted keys; build tileOffsets[0..totalTiles] from RLE output.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

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
        tileOffsets[t] = d_run_offsets[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tileOffsets[total_tiles] = num_pairs;
    }
}

// Kernel: fill gaps (tiles with no entities). Single-thread sequential pass so propagation is correct.
__global__ void fillTileOffsetsGapsKernel(
    unsigned int total_tiles,
    unsigned int* __restrict__ tileOffsets)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int last = 0u;
    for (unsigned int t = 0; t < total_tiles; ++t) {
        if (tileOffsets[t] != 0xFFFFFFFFu)
            last = tileOffsets[t];
        else
            tileOffsets[t] = last;
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
