#include "utils/parallel/aux_functions.h"

void cullSpheres(std::vector<glm::vec4>& spheres, std::vector<SphereContainer>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount)
{
    visibleSpheresCount = 0;
    for (int i = 0 ; i < spheres.size(); i++)
        if (frustum.isSphereInside(spheres[i]))
            visibleSpheres[visibleSpheresCount++] = {spheres[i], i, {0, 0, 0}};    
}

void fillSpheresData(std::vector<glm::vec4>& spheres, std::vector<SphereContainer>& visibleSpheres)
{
    for (int i = 0 ; i < spheres.size(); i++)
        visibleSpheres[i] = {spheres[i], i, {0, 0, 0}};    
}