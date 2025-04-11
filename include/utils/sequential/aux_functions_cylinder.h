#ifndef _AUX_SEQUENTIAL_CYLINDER_H_
#define _AUX_SEQUENTIAL_CYLINDER_H_

#include "utils/sequential/common.h"

/**
 * @struct bboxCorners
 * @brief Represents the corners of a bounding box in view and screen space.
 * 
 * The `bboxCorners` struct represents the corners of a bounding box 2D in screen space.
 * 
 */
struct BBoxCorners
{
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
BBoxCorners getCylinderBbox(const glm::vec3& pa, const glm::vec3& pb, const glm::vec3& center, 
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
 * @param right Camera Right orientation vector.
 * @param camPos Camera position.
 * @param SCR_WIDTH Viewport width. 
 * @param SCR_HEIGHT Viewport height. 
 * @param pa Extreme A of the cylinder. 
 * @param pb Extreme B of the cylinder. 
 * @param cylRadius Radius of the cylinder. 
 * @param fov Field of view of the camera. 
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


    std::vector<glm::vec2> getCylinderBbox2ndAttempt(const glm::vec3& pa, const glm::vec3& pb, const glm::vec3& center, 
    const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up, const float cylRadius);

bool drawCylinder2ndAttempt(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& right, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec3& pa, const glm::vec3& pb, const float& cylRadius, const float& fov,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer);

#endif // _AUX_SEQUENTIAL_CYLINDER_H_