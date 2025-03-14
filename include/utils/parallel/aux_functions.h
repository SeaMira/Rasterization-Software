#ifndef _AUX_PARALLEL_H_
#define _AUX_PARALLEL_H_

#define CPU_FRUSTUM_CULLING 1

#include "algorithms/frustum_cull.h"

void cullSpheres(std::vector<glm::vec4>& spheres, std::vector<glm::vec4>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount);


#endif // _AUX_PARALLEL_H_