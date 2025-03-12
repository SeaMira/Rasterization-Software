#include "utils/parallel/aux_functions.h"

void cullSpheres(std::vector<glm::vec4>& spheres, std::vector<glm::vec4>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount)
{
    visibleSpheresCount = 0;
    for (auto& sphere : spheres)
        if (frustum.isSphereInside(sphere))
        {
            visibleSpheres[visibleSpheresCount] = sphere;    
            visibleSpheresCount++;
        }
}