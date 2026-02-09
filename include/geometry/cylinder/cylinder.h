#ifndef CYLINDER_H
#define CYLINDER_H

#include <cstdint>
#include <glm/glm.hpp>


/**
 * @struct CylinderIndex
 * 
 * @brief Represents a cylinder in 3D space via indices into the sphere buffer.
 * 
 * The cylinder is defined by two sphere indices (its endpoints) and a radius.
 * The actual 3D positions are resolved from the sphere buffer at runtime.
 * Total size: 16 bytes (GPU-aligned).
 */
struct CylinderIndex {
    uint32_t sphereIndexA;  ///< Index of sphere at end A
    uint32_t sphereIndexB;  ///< Index of sphere at end B
    float radius;           ///< Cylinder radius
    uint32_t _padding;      ///< Padding for alignment

    CylinderIndex(uint32_t a, uint32_t b, float r)
        : sphereIndexA(a), sphereIndexB(b), radius(r), _padding(0) {}
    CylinderIndex() : sphereIndexA(0), sphereIndexB(0), radius(0.0f), _padding(0) {}
};

/// Backward-compatible alias
using Cylinder = CylinderIndex;

#endif // CYLINDER_H