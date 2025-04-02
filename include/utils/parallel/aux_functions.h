#ifndef _AUX_PARALLEL_H_
#define _AUX_PARALLEL_H_

#define CPU_FRUSTUM_CULLING 0

#include "algorithms/frustum_cull.h"

struct SphereContainer
{
    glm::vec4 positionr;
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
 */
void cullSpheres(std::vector<glm::vec4>& spheres, std::vector<SphereContainer>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount);

/**
 * @brief Completes a spheres vector with index and visibility bool.
 * 
 * Completes a spheres vector with index, visibility bool and padding for the SSBO from
 * the complete spheres vector.
 * 
 * @param spheres The list of spheres to be filtered.
 * @param visibleSpheres The list of spheres with index, visibility bool and padding (for SSBO).
 */
void fillSpheresData(std::vector<glm::vec4>& spheres, std::vector<SphereContainer>& visibleSpheres);

#endif // _AUX_PARALLEL_H_