#ifndef _AUX_SEQUENTIAL_H_
#define _AUX_SEQUENTIAL_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>
#include <omp.h>

#include "algorithms/occlusion_cull.h"

const glm::vec3 lightColor(0.01f, 1.0f, 0.05f);
const float diffuseI = 0.9f;

/**
 * @struct bboxCorners
 * @brief Represents the corners of a bounding box in view and screen space.
 * 
 * The `bboxCorners` struct represents the corners of a bounding box in view space and screen space, storing vectors and coordinates.
 * Also they may store corners depth values to avoid extra matrix calculations on the future.
 */
struct bboxCorners
{
    // view space
    glm::vec3 upRightCorner; ///< Upper right corner of the bounding box in view space.
    float upRightCornerndcZ; ///< Depth of the upper right corner of the bounding box in projection space.
    glm::vec3 upLeftCorner; ///< Upper left corner of the bounding box in view space.
    float upLeftCornerndcZ; ///< Depth of the upper left corner of the bounding box in projection space.
    glm::vec3 downRightCorner; ///< Down right corner of the bounding box in view space.
    float downRightCornerndcZ; ///< Depth of the down right corner of the bounding box in projection space.
    glm::vec3 downLeftCorner; ///< Down left corner of the bounding box in view space.
    float downLeftCornerndcZ; ///< Depth of the down left corner of the bounding box in projection space.

    // screen space
    glm::vec2 minCorner; ///< Down left corner of the bounding box in screen space.
    glm::vec2 maxCorner; ///< Upper right corner of the bounding box in screen space.


};

/**
 * @brief Computes the distance of impact from a ray origin (camera origin) to a sphere surface.
 * 
 * Tells if a ray casted from an origin impacts a sphere and returns the distance of impact. It is a signed 
 * distance: if its lower than zero then there is no impact.
 * 
 * @param ro Ray origin (camera origin).
 * @param rd Ray direction.
 * @param sph Sphere center.
 * @param radius Sphere radius.
 * 
 * @return The distance of impact from the ray origin to the sphere surface. If negative, the ray doesn't intersect the sphere.
 */
float iSphere(glm::vec3 ro, glm::vec3 rd, glm::vec3 sph, float radius );

/**
 * @brief Computes the billboard of a sphere in view and screen space.
 * 
 * Uses camera info. (such as caits position, projection matrix, orientation vectors, etc) 
 * to project the boundaries of a sphere in a billboard.abort
 * 
 * @param cameraSpaceSphere Sphere center in camera space.
 * @param camImposPos Camera space position of the billboard center.
 * @param normCamSpaceSphere Normalized sphere center in camera space.
 * @param proj Projection matrix.
 * @param camPos Camera position
 * @param front Front orientation camera vector.
 * @param up Up orientation camera vector
 * @param sphRadius Sphere radius.
 * @param hizPyramid Hierarchival Z buffer
 * 
 * @return bboxCorners struct: info on camera space and projection space boundaries. 
 */
bboxCorners getSphereBbox(const glm::vec3& cameraSpaceSphere, const glm::vec3& camImposPos, 
    const glm::vec3& normCamSpaceSphere, const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up,const float sphRadius);


/**
 * @brief Takes a color represented in three integers and packages it in an unsigned int.
 * 
 * Thought for obtaining a basic shading color for evey sphere pixel.
 * 
 * @param lambertCos
 */
uint32_t vecToColor(glm::vec3 lambertCos);

/**
 * @brief Projects and draws a sphere onto the viewport.
 * 
 * Takes a sphere on world coordinates and projects it onto the viewport in the framebuffer 
 * and taking into account the values of the depth buffer for it. If not visible it gets culled.
 * 
 * @param proj Projection Matrix.
 * @param view View Matrix.
 * @param up Camera Up orientation vector.
 * @param front Camera Front orientation vector.
 * @param camPos Camera position.
 * @param SCR_WIDTH Viewport width. 
 * @param SCR_HEIGHT Viewport height. 
 * @param sphere Sphere represented as a position and radius. 
 * @param framebuffer Framebuffer containing color info of every pixel.
 * @param depthBuffer Depth buffer containing info of every pixel's projection depth. 
 * @param hizPyramid Hierarchical Z buffer with mipmaps for occlusion culling.
 * 
 * @return True if any pixels from the sphere was drawn, false otherwise.
 */
bool drawSphere(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer, HierarchicalZBuffer& hizPyramid);

/**
 * @brief Projects and draws a sphere billboard onto the viewport.abort
 * 
 * Takes a sphere on world coordinates and projects its billboard onto the viewport in the framebuffer 
 * and taking into account the values of the depth buffer for it. If not visible it gets culled.
 * 
 * @param proj Projection Matrix.
 * @param view View Matrix.
 * @param up Camera Up orientation vector.
 * @param front Camera Front orientation vector.
 * @param camPos Camera position.
 * @param SCR_WIDTH Viewport width. 
 * @param SCR_HEIGHT Viewport height. 
 * @param sphere Sphere represented as a position and radius. 
 * @param framebuffer Framebuffer containing color info of every pixel.
 * @param depthBuffer Depth buffer containing info of every pixel's projection depth. 
 * @param hizPyramid Hierarchical Z buffer with mipmaps for occlusion culling.
 * 
 * @return True if any pixels from the billboard was drawn, false otherwise.
 */
bool drawBillboard(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer, HierarchicalZBuffer& hizPyramid);

#endif // _AUX_SEQUENTIAL_H_