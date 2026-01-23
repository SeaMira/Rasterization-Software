/**
 * @file hybrid_types.h
 * @brief Shared types and constants for the hybrid rasterization pipeline.
 * 
 * This header defines structures used across all kernels in the hybrid pipeline:
 * - Frustum culling + BBox extraction
 * - Small entity rasterization (1 thread per entity)
 * - Large entity tile binning + tiled rasterization
 */

#ifndef HYBRID_TYPES_H
#define HYBRID_TYPES_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

// ============================================================================
// Configuration Constants
// ============================================================================

// Threshold for small vs large entities (in pixels squared)
// Entities with bbox area <= this threshold use direct rasterization
// Entities with bbox area > this threshold use tiled rasterization
#define SMALL_ENTITY_THRESHOLD 256  // 16x16 pixels

// Tile configuration
#define TILE_SIZE 16
#define TILE_SIZE_SHIFT 4  // log2(TILE_SIZE)
#define MAX_ENTITIES_PER_TILE 256

// Work group sizes
#define CULL_BLOCK_SIZE 256
#define SMALL_RASTER_BLOCK_SIZE 256
#define TILE_BLOCK_SIZE_X 16
#define TILE_BLOCK_SIZE_Y 16

// ============================================================================
// Sphere Billboard Structure (for large entities)
// ============================================================================

struct SphereBillboard {
    glm::vec4 positionRadius;   // xyz = world position, w = radius
    glm::vec2 screenMin;        // Screen-space min corner
    glm::vec2 screenMax;        // Screen-space max corner
    float centerDepth;          // Depth at center for sorting
    unsigned int originalIndex; // Index in original sphere buffer
    unsigned int tileMinX;      // First tile X coordinate
    unsigned int tileMinY;      // First tile Y coordinate
    unsigned int tileMaxX;      // Last tile X coordinate
    unsigned int tileMaxY;      // Last tile Y coordinate
};

struct CylinderBillboard {
    glm::vec4 pa_r;             // Point A + radius
    glm::vec4 pb_r;             // Point B + radius (w unused, but for alignment)
    glm::vec2 screenMin;        // Screen-space min corner
    glm::vec2 screenMax;        // Screen-space max corner
    float centerDepth;          // Depth at center for sorting
    unsigned int originalIndex; // Index in original cylinder buffer
    unsigned int tileMinX;      // First tile X coordinate
    unsigned int tileMinY;      // First tile Y coordinate
    unsigned int tileMaxX;      // Last tile X coordinate
    unsigned int tileMaxY;      // Last tile Y coordinate
};

// ============================================================================
// Small Entity Reference (for direct rasterization)
// ============================================================================

struct SmallEntityRef {
    unsigned int originalIndex; // Index in original buffer
    unsigned int entityType;    // 0 = sphere, 1 = cylinder
};

// ============================================================================
// Tile Data Structure
// ============================================================================

struct TileHeader {
    unsigned int sphereCount;
    unsigned int cylinderCount;
    unsigned int sphereOffset;   // Offset into tile entity buffer
    unsigned int cylinderOffset; // Offset into tile entity buffer
};

// ============================================================================
// Hybrid Pipeline Constants (uploaded to constant memory)
// ============================================================================

struct HybridConstants {
    // Matrices
    glm::mat4 view;
    glm::mat4 proj;
    
    // Frustum planes (normal.xyz, distance.w)
    glm::vec4 frustumPlanes[6];
    
    // Camera vectors
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 cameraPos;
    
    // Screen dimensions
    int screenWidth;
    int screenHeight;
    float fov;
    
    // Entity counts
    int sphereCount;
    int cylinderCount;
    
    // Tile grid dimensions
    int tilesX;
    int tilesY;
    int totalTiles;
    
    // Threshold for small/large entity split
    int smallEntityThreshold;
    
    // Benchmark flag
    int benchmark;
};

// ============================================================================
// Dispatch Parameters (for indirect dispatch)
// ============================================================================

struct DispatchParams {
    unsigned int groupsX;
    unsigned int groupsY;
    unsigned int groupsZ;
};

// ============================================================================
// Statistics Counters
// ============================================================================

struct HybridStats {
    unsigned int frustumPassedSpheres;
    unsigned int frustumPassedCylinders;
    unsigned int smallSphereCount;
    unsigned int largeSphereCount;
    unsigned int smallCylinderCount;
    unsigned int largeCylinderCount;
    unsigned int tilesProcessed;
    unsigned int pixelsShaded;
};

// ============================================================================
// Radix Sort Key-Value Pair
// ============================================================================

struct SortKeyValue {
    unsigned int key;   // Depth as uint bits (for sorting front-to-back)
    unsigned int value; // Index into billboard array
};

// ============================================================================
// Color and shading constants
// ============================================================================

__device__ __constant__ static const float3 atomsColor = {0.8f, 0.1f, 0.1f};
__device__ __constant__ static const float diffuse = 0.9f;

#endif // HYBRID_TYPES_H
