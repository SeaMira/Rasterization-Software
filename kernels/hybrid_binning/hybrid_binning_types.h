/**
 * @file hybrid_binning_types.h
 * @brief Shared types for hybrid binning-by-sort pipeline (no linked lists).
 * Uses CUB DeviceRadixSort + DeviceRunLengthEncode for tile binning.
 */

#ifndef HYBRID_BINNING_TYPES_H
#define HYBRID_BINNING_TYPES_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define SMALL_ENTITY_THRESHOLD 256
#define TILE_SIZE 16
#define TILE_SIZE_SHIFT 4
#define CULL_BLOCK_SIZE 256
#define SMALL_RASTER_BLOCK_SIZE 256
#define TILE_BLOCK_SIZE_X 16
#define TILE_BLOCK_SIZE_Y 16
#define SHARED_BATCH_SIZE 64

struct SphereBillboard {
    glm::vec4 positionRadius;
    glm::vec2 screenMin;
    glm::vec2 screenMax;
    float centerDepth;
    unsigned int originalIndex;
    unsigned int tileMinX;
    unsigned int tileMinY;
    unsigned int tileMaxX;
    unsigned int tileMaxY;
};

struct CylinderBillboard {
    glm::vec4 pa_r;
    glm::vec4 pb_r;
    glm::vec2 screenMin;
    glm::vec2 screenMax;
    float centerDepth;
    unsigned int originalIndex;
    unsigned int tileMinX;
    unsigned int tileMinY;
    unsigned int tileMaxX;
    unsigned int tileMaxY;
};

struct BBox3D
{
    glm::vec3 mMin;
    glm::vec3 mMax;
};

struct HybridConstants {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 frustumPlanes[6];
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 cameraPos;
    int screenWidth;
    int screenHeight;
    float fov;
    int sphereCount;
    int cylinderCount;
    int tilesX;
    int tilesY;
    int totalTiles;
    int smallEntityThreshold;
    int benchmark;
};

#endif
