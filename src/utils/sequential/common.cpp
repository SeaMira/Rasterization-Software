#include "utils/sequential/common.h"

uint32_t vecToColor(glm::vec3 lambertCos) 
{
    const uint8_t R = lambertCos.x;
    const uint8_t G = lambertCos.y;
    const uint8_t B = lambertCos.z;

    return (0xFF000000) | (R << 16) | (G << 8) | B;
}