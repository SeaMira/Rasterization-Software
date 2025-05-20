#ifndef _AUX_PARALLEL_H_
#define _AUX_PARALLEL_H_

#define CPU_FRUSTUM_CULLING 0

#include "algorithms/frustum_cull.h"
#include "geometry/cylinder/cylinder.h"
#include "vis/compute_shader_program.h"
#include "vis/shader_program.h"
#include "ux/camera.h"
using Sphere = glm::vec4;

struct SphereContainer
{
    glm::vec4 positionr;
    int index;
    int wasDrawn[3];
};

struct CylinderContainer
{
    glm::vec4 pa_r;
    glm::vec4 pb_r;
    int index;
    int wasDrawn[3];
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
void cullSimpleCylinders(std::vector<Cylinder>& cylinders, std::vector<Cylinder>& visibleCylinders, Frustum& frustum, int& visibleCylindersCount, int indexOffset = 0);

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
void cullCylinders(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders, Frustum& frustum, int& visibleCylindersCount, int indexOffset = 0);

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
void fillCylindersData(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders, int indexOffset = 0);

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

#endif // _AUX_PARALLEL_H_