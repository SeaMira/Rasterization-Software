#include "utils/parallel/aux_functions.h"

void cullSpheres(std::vector<Sphere>& spheres, std::vector<SphereContainer>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount)
{
    visibleSpheresCount = 0;
    for (int i = 0 ; i < spheres.size(); i++)
        if (frustum.isSphereInside(spheres[i]))
            visibleSpheres[visibleSpheresCount++] = {spheres[i], i, {0, 0, 0}};    
}

void fillSpheresData(std::vector<Sphere>& spheres, std::vector<SphereContainer>& visibleSpheres)
{
    for (int i = 0 ; i < spheres.size(); i++)
        visibleSpheres[i] = {spheres[i], i, {0, 0, 0}};    
}


void cullCylinders(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders, Frustum& frustum, int& visibleCylindersCount)
{
    visibleCylindersCount = 0;
    for (int i = 0 ; i < cylinders.size(); i++)
        if (frustum.isCylinderInside(cylinders[i]))
        visibleCylinders[visibleCylindersCount++] = {
            glm::vec4(cylinders[i].pa, cylinders[i].radius), 
            glm::vec4(cylinders[i].pb, cylinders[i].radius),
            i, 
            {0, 0, 0}
        };    
}

void fillCylindersData(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders)
{
    for (int i = 0 ; i < cylinders.size(); i++)
        visibleCylinders[i] = {
            glm::vec4(cylinders[i].pa, cylinders[i].radius), 
            glm::vec4(cylinders[i].pb, cylinders[i].radius),
            i, 
            {0, 0, 0}
        };  
}