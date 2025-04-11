#ifndef _AUX_SEQUENTIAL_COMMON_H_
#define _AUX_SEQUENTIAL_COMMON_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using Sphere = glm::vec4; ///< Sphere type defined as a 4D vector (x, y, z, radius).

struct BBox3D
{
    glm::vec3 mMin;
    glm::vec3 mMax;
};

struct Cylinder
{
    glm::vec3 pa; // extreme A
    glm::vec3 pb; // extreme B
    float radius;
};

const glm::vec3 lightColor(0.01f, 1.0f, 0.05f);
const float diffuseI = 0.9f;

struct ScreenRayCasting
{
    glm::vec3 rayStart; ///< Starting caster ray in world coordinates.
    glm::vec3 dx; ///< Per pixel delta direction vector for the x-axis.
    glm::vec3 dy; ///< Per pixel delta direction vector for the y-axis.

    ScreenRayCasting(const float& fov, const float& aspectRatio, 
        const int& SCR_WIDTH, const int& SCR_HEIGHT,
        const glm::vec3& right, const glm::vec3& up, const glm::vec3& front,
        const glm::mat4& view)
    {
        float fovRad = glm::radians(fov);
        float fovTan = tan(fovRad * 0.5f);
        float halfFovTan = fovTan * aspectRatio;

        // View-space ray directions for screen corners
        glm::vec3 corner00 = glm::normalize((-1.0f * halfFovTan) * right + (-1.0f * fovTan) * up + front); // bottom-left
        glm::vec3 corner10 = glm::normalize(( 1.0f * halfFovTan) * right + (-1.0f * fovTan) * up + front); // bottom-right
        glm::vec3 corner01 = glm::normalize((-1.0f * halfFovTan) * right + ( 1.0f * fovTan) * up + front); // top-left

        // Transform to world space only once
        glm::vec3 wCorner00 = glm::mat3(view) * corner00;
        glm::vec3 wCorner10 = glm::mat3(view) * corner10;
        glm::vec3 wCorner01 = glm::mat3(view) * corner01;

        // Delta per pixel
        dx = (wCorner10 - wCorner00) / float(SCR_WIDTH);
        dy = (wCorner01 - wCorner00) / float(SCR_HEIGHT);

        rayStart = wCorner00;
    }
};

/**
 * @brief Takes a color represented in three integers and packages it in an unsigned int.
 * 
 * Thought for obtaining a basic shading color for evey sphere pixel.
 * 
 * @param lambertCos
 */
uint32_t vecToColor(glm::vec3 lambertCos);

#endif // _AUX_SEQUENTIAL_COMMON_H_