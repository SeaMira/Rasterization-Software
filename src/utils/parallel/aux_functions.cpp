#include "utils/parallel/aux_functions.h"

void cullSpheres(std::vector<Sphere>& spheres, std::vector<SphereContainer>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount, int indexOffset)
{
    visibleSpheresCount = 0;
    for (int i = 0 ; i < spheres.size(); i++)
        if (frustum.isSphereInside(spheres[i]))
            visibleSpheres[visibleSpheresCount++] = {spheres[i], i + indexOffset, {0, 0, 0}};    
}

void fillSpheresData(std::vector<Sphere>& spheres, std::vector<SphereContainer>& visibleSpheres, int indexOffset)
{
    for (int i = 0 ; i < spheres.size(); i++)
        visibleSpheres[i] = {spheres[i], i + indexOffset, {0, 0, 0}};    
}


void cullCylinders(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders, Frustum& frustum, int& visibleCylindersCount, int indexOffset)
{
    visibleCylindersCount = 0;
    for (int i = 0 ; i < cylinders.size(); i++)
        if (frustum.isCylinderInside(cylinders[i]))
        visibleCylinders[visibleCylindersCount++] = {
            glm::vec4(cylinders[i].pa, cylinders[i].radius), 
            glm::vec4(cylinders[i].pb, cylinders[i].radius),
            i + indexOffset, 
            {0, 0, 0}
        };    
}

void fillCylindersData(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders, int indexOffset)
{
    for (int i = 0 ; i < cylinders.size(); i++)
        visibleCylinders[i] = {
            glm::vec4(cylinders[i].pa, cylinders[i].radius), 
            glm::vec4(cylinders[i].pb, cylinders[i].radius),
            i + indexOffset, 
            {0, 0, 0}
        };  
}