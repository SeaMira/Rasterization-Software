#ifndef _AUX_SEQUENTIAL_CYLINDER_H_
#define _AUX_SEQUENTIAL_CYLINDER_H_

#include <glm/glm.hpp>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>
#include <omp.h>

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
 * @brief Computes the billboard of a cylinder in view and screen space.
 * 
 * Uses camera info. (such as camera position, projection matrix, orientation vectors, etc) 
 * to project the boundaries of a cylinder in a billboard.
 * 
 * @param pa Cylinder extreme A.
 * @param pb Cylinder extreme B.
 * @param center Cylinder centre.
 * @param proj Projection matrix.
 * @param camPos Camera position
 * @param front Front orientation camera vector.
 * @param up Up orientation camera vector
 * @param cylRadius Cylinder radius.
 * 
 * @return bboxCorners struct: info on camera space and projection cylinder boundaries. 
 */
bboxCorners getCylinderBbox(const glm::vec3& pa, const glm::vec3& pb, const glm::vec3& center, 
    const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up, const float cylRadius);


/**
 * @brief Projects and draws a cylinder onto the viewport.
 * 
 * Takes a cylinder on world coordinates and projects it onto the viewport in the framebuffer 
 * and taking into account the values of the depth buffer for it. If not visible it gets culled.
 * 
 * @param proj Projection Matrix.
 * @param view View Matrix.
 * @param up Camera Up orientation vector.
 * @param front Camera Front orientation vector.
 * @param front Camera Right orientation vector.
 * @param camPos Camera position.
 * @param SCR_WIDTH Viewport width. 
 * @param SCR_HEIGHT Viewport height. 
 * @param pa Extreme A of the cylinder. 
 * @param pb Extreme B of the cylinder. 
 * @param cylRadius Radius of the cylinder. 
 * @param framebuffer Framebuffer containing color info of every pixel.
 * @param depthBuffer Depth buffer containing info of every pixel's projection depth. 
 * 
 * @return True if any pixels from the sphere was drawn, false otherwise.
 */
bool drawCylinder(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& right, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec3& pa, const glm::vec3& pb, const float& cylRadius, const float& fov,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer);

#endif // _AUX_SEQUENTIAL_CYLINDER_H_