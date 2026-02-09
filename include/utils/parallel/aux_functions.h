#ifndef _AUX_PARALLEL_H_
#define _AUX_PARALLEL_H_

#define CPU_FRUSTUM_CULLING 0

#include "algorithms/frustum_cull.h"
#include "geometry/cylinder/cylinder.h"
#include "vis/compute_shader_program.h"
#include "vis/shader_program.h"
#include "ux/camera.h"
using Sphere = glm::vec4;

/**
 * @struct SphereContainer
 * @brief Represents a sphere in 3D space.
 * 
 * The sphere is defined by its position (center) and radius. But it also stores
 * the index of the sphere in the original vector and a padding integers array
 * to be used on the SSBO.
 */
struct SphereContainer
{
    glm::vec4 positionr;
    int index;
    int wasDrawn[3];
};

/**
 * @struct CylinderContainer
 * @brief Represents a cylinder in 3D space.
 * 
 * The cylinder is defined by its two sphere indices, a radius, and bookkeeping data
 * for the SSBO.
 */
struct CylinderContainer
{
    uint32_t sphereIndexA;
    uint32_t sphereIndexB;
    float radius;
    int index;
    int wasDrawn[3];
    uint32_t _padding;
};

/**
 * @brief Filters spheres that are actually on the camera frustum.
 * 
 * Takes a list of spheres and filters the ones that are actually on the camera frustum, storing them in another sphere vector. It also stores how many spheres are in this second vector.
 * 
 * @param spheres The list of spheres to be filtered.
 * @param visibleSpheres The list of spheres that are actually on the camera frustum.
 * @param frustum The camera frustum.
 * @param visibleSpheresCount The number of spheres that are on the camera frustum.
 * @param indexOffset The offset to be added to the index of the spheres in the visibleSpheres vector.
 */
void cullSimpleSpheres(std::vector<Sphere>& spheres, std::vector<Sphere>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount, int indexOffset = 0);

/**
 * @brief Filters spheres that are actually on the camera frustum.
 * 
 * Takes a list of spheres and filters the ones that are actually on the camera frustum, storing them in another sphere container (sphere related info, like index and multipurpose padding) vector. It also stores how many spheres are in this second vector.
 * 
 * @param spheres The list of spheres to be filtered.
 * @param visibleSpheres The list of spheres and its info that are actually on the camera frustum.
 * @param frustum The camera frustum.
 * @param visibleSpheresCount The number of spheres that are on the camera frustum.
 * @param indexOffset The offset to be added to the index of the spheres in the visibleSpheres vector.
 */
void cullSpheres(std::vector<Sphere>& spheres, std::vector<SphereContainer>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount, int indexOffset = 0);

/**
 * @brief Completes a spheres vector with index and visibility bool.
 * 
 * Completes a spheres vector with index, visibility bool and padding for the SSBO from
 * the complete spheres vector.
 * 
 * @param spheres The list of spheres to be filtered.
 * @param visibleSpheres The list of spheres with index, visibility bool and padding (for SSBO).
 * @param indexOffset The offset to be added to the index of the spheres in the visibleSpheres vector.
 */
void fillSpheresData(std::vector<glm::vec4>& spheres, std::vector<SphereContainer>& visibleSpheres, int indexOffset = 0);

/**
 * @brief Filters cylinders that are actually on the camera frustum.
 * 
 * Takes a list of cylinders and filters the ones that are actually on the camera frustum, storing them in another cylinders vector. It also stores how many cylinders are in this second vector.
 * 
 * @param cylinders The list of cylinders to be filtered.
 * @param visibleCylinders The list of cylinders that are actually on the camera frustum.
 * @param frustum The camera frustum.
 * @param visibleCylindersCount The number of cylinders that are on the camera frustum.
 * @param indexOffset The offset to be added to the index of the cylinders in the visibleCylinders vector.
 */
void cullSimpleCylinders(std::vector<CylinderIndex>& cylinders, std::vector<CylinderIndex>& visibleCylinders, const std::vector<glm::vec4>& spheres, Frustum& frustum, int& visibleCylindersCount, int indexOffset = 0);

/**
 * @brief Filters cylinders that are actually on the camera frustum.
 * 
 * Takes a list of cylinders and filters the ones that are actually on the camera frustum, storing them in another cylinders  container (cylinder related info, like index and multipurpose padding) vector. It also stores how many cylinders are in this second vector.
 * 
 * @param cylinders The list of cylinders to be filtered.
 * @param visibleCylinders The list of cylinders and its info that are actually on the camera frustum.
 * @param frustum The camera frustum.
 * @param visibleCylindersCount The number of cylinders that are on the camera frustum.
 * @param indexOffset The offset to be added to the index of the cylinders in the visibleCylinders vector.
 */
void cullCylinders(std::vector<CylinderIndex>& cylinders, std::vector<CylinderContainer>& visibleCylinders, const std::vector<glm::vec4>& spheres, Frustum& frustum, int& visibleCylindersCount, int indexOffset = 0);

/**
 * @brief Completes a cylinders vector with index and visibility bool.
 * 
 * Completes a cylinders vector with index, visibility bool and padding for the SSBO from
 * the complete cylinders vector.
 * 
 * @param cylinders The list of cylinders to be filtered.
 * @param visibleCylinders The list of cylinders with index, visibility bool and padding (for SSBO).
 * @param indexOffset The offset to be added to the index of the cylinders in the visibleCylinders vector.
 */
void fillCylindersData(std::vector<CylinderIndex>& cylinders, std::vector<CylinderContainer>& visibleCylinders, int indexOffset = 0);

/**
 * @brief Sets the camera uniforms on the shader.
 * 
 * Sets the camera uniforms on the shader, including projection and view matrices, up, front and right vectors, camera position and camera fov.
 * 
 * @param shader The shader to set the uniforms on.
 * @param camera The camera to get the uniforms from.
 */
void setCameraUniforms(ComputeShader& shader, Camera& camera);

/**
 * @brief Sets the camera uniforms on the shader.
 * 
 * Sets the camera uniforms on the shader, including projection and view matrices, up, front and right vectors, camera position and camera fov.
 * 
 * @param shader The shader to set the uniforms on.
 * @param camera The camera to get the uniforms from.
 */
void setCameraUniforms(ShaderProgram& shader, Camera& camera);

/**
 * @brief Sets the frustum uniforms on the shader.
 * 
 * Sets the frustum uniforms on the shader, including the six planes of the frustum.
 * 
 * @param shader The shader to set the uniforms on.
 * @param frustum The frustum to get the uniforms from.
 */
void setFrustumUniforms(ComputeShader& shader, Frustum& frustum);


/**
 * @brief Dispatches a compute shader with label.
 * 
 * Dispatches the active compute shader with a label for debugging purposes, indicating 
 * the work group size.
 * 
 * @param numGroupsX Number of work groups in X dimension.
 * @param numGroupsY Number of work groups in Y dimension.
 * @param label The label to be used for debugging.
 * 
 */
void dispatchComputeShaderWithLabel(GLuint numGroupsX, GLuint numGroupsY, const std::string& label, GLbitfield barrier = 0);

/**
 * @brief Launches a shaders pipeline with label.
 * 
 * Launches a draw call of a shaders pipeline with a label for debugging purposes..
 * 
 * @param mode Specifies what kind of primitives to render. See khronos specifications.
 * @param first Specifies the starting index in the enabled arrays.
 * @param count Specifies the number of elements to be rendered.
 * @param label The label to be used for debugging.
 * 
 */
void drawArraysWithLabel(GLenum mode, GLint first, GLsizei count, const std::string& label);
#endif // _AUX_PARALLEL_H_