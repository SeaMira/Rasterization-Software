#include "utils/sequential/aux_functions_cylinder.h"

bboxCorners getCylinderBbox(glm::vec3& pa, glm::vec3& pb, glm::vec3& center, 
    const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up, const float cylRadius)
{

    // Cylinder axis
    const glm::vec3 z = glm::normalize(pb - pa);

    // Find orthonormal x,y axes orthogonal to cylinder axis
    glm::vec3 arbitrary = glm::abs(z.x) < 0.99f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 x = glm::normalize(glm::cross(arbitrary, z));
    glm::vec3 y = glm::normalize(glm::cross(z, x)); // make full basis

    // Generate 8 points around caps (pa and pb)
    std::vector<glm::vec3> points = {
        pa + cylRadius * x + cylRadius * y,
        pa + cylRadius * x - cylRadius * y,
        pa - cylRadius * x + cylRadius * y,
        pa - cylRadius * x - cylRadius * y,
        pb + cylRadius * x + cylRadius * y,
        pb + cylRadius * x - cylRadius * y,
        pb - cylRadius * x + cylRadius * y,
        pb - cylRadius * x - cylRadius * y
    };

    glm::vec2 minCorner(FLT_MAX), maxCorner(-FLT_MAX);

    for (const glm::vec3& p : points)
    {
        glm::vec4 clip = proj * glm::vec4(p, 1.0f);
        if (clip.w != 0.0f)
        {
            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            minCorner = glm::min(minCorner, glm::vec2(ndc));
            maxCorner = glm::max(maxCorner, glm::vec2(ndc));
        }
    }
    
    
    return {arbitrary, arbitrary.z, arbitrary, arbitrary.z, arbitrary, arbitrary.z, arbitrary, arbitrary.z, minCorner, maxCorner};
}

glm::vec4 iCylinder(glm::vec3 ro, glm::vec3 rd, glm::vec3 pa, glm::vec3 pb, float ra) 
{
    glm::vec3  ba = pb - pa;
    glm::vec3  oc = ro - pa;

    float baba = glm::dot(ba,ba);
    float bard = glm::dot(ba,rd);
    float baoc = glm::dot(ba,oc);
    
    float k2 = baba - bard*bard;
    float k1 = baba*glm::dot(oc,rd) - baoc*bard;
    float k0 = baba*glm::dot(oc,oc) - baoc*baoc - ra*ra*baba;
    
    
    float h = k1*k1 - k2*k0;
    if( h<0.0 ) return glm::vec4(-1.0);
    h = sqrt(h);
    float t = (-k1-h)/k2;

    // body
    float y = baoc + t*bard;
    if( y>0.0 && y<baba ) return glm::vec4( t, oc+t*rd - ba*y/baba );
    
    // caps
    t = ( ((y<0.0) ? 0.0 : baba) - baoc)/bard;
    if( abs(k1+k2*t)<h ) return glm::vec4( t, ba*glm::sign(y)/sqrt(baba) );

    return glm::vec4(-1.0);
}

uint32_t vecToColor(glm::vec3 lambertCos) 
{
    const uint8_t R = lambertCos.x;
    const uint8_t G = lambertCos.y;
    const uint8_t B = lambertCos.z;

    return (0xFF000000) | (R << 16) | (G << 8) | B;
}

bool drawCylinder(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& right, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec3& pa, const glm::vec3& pb, const float& cylRadius, const float& fov,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer)
{
    glm::vec3 camSpaceCylA = glm::vec3(view * glm::vec4(pa.x, pa.y, pa.z, 1.0f));
    glm::vec3 camSpaceCylB = glm::vec3(view * glm::vec4(pb.x, pb.y, pb.z, 1.0f));

    glm::vec3 camImpPosA, camImpPosB;
    if ( camSpaceCylA.z < camSpaceCylB.z )
	{
		camImpPosA = camSpaceCylB;
		camImpPosB = camSpaceCylA;
	}
	else
	{
		camImpPosA = camSpaceCylA;
		camImpPosB = camSpaceCylB;
	}
    glm::vec3 center = normalize( ( camImpPosA + camImpPosB ) * 0.5f );
    
    bboxCorners cylBbox = getCylinderBbox(camImpPosA, camImpPosB, center,
        proj, camPos, front, up, cylRadius);
        
    float aspectRatio = (float)SCR_WIDTH / (float)SCR_HEIGHT;
    glm::ivec2 screenMin, screenMax;

    screenMin.x = ((cylBbox.minCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMin.y = (cylBbox.minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
    screenMax.x = ((cylBbox.maxCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMax.y = (cylBbox.maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;

    const float difx = screenMax.x - screenMin.x;
    const float dify = screenMax.y - screenMin.y;
    if (difx*dify <= 2) return false;
    
    float fovRad = glm::radians(fov);
    float fovTan = tan(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    // View-space ray directions for screen corners
    glm::vec3 corner00 = glm::normalize((-1.0f * halfFovTan) * right + (-1.0f * fovTan) * up + front); // bottom-left
    glm::vec3 corner10 = glm::normalize(( 1.0f * halfFovTan) * right + (-1.0f * fovTan) * up + front); // bottom-right
    glm::vec3 corner01 = glm::normalize((-1.0f * halfFovTan) * right + ( 1.0f * fovTan) * up + front); // top-left

    // Transform to world space only once
    glm::vec3 wCorner00 = glm::mat3(view) * corner00;
    glm::vec3 wCorner10 = glm::mat3(view) * corner10;
    glm::vec3 wCorner01 = glm::mat3(view) * corner01;

    // Delta per pixel
    glm::vec3 dx = (wCorner10 - wCorner00) / float(SCR_WIDTH);
    glm::vec3 dy = (wCorner01 - wCorner00) / float(SCR_HEIGHT);

    // Start ray origin in world
    glm::vec3 rayStart = wCorner00;
    #pragma omp parallel for collapse(2)
    for (int px = std::max(0, screenMin.x); px < std::min(screenMax.x, SCR_WIDTH); px++)
    {
        glm::vec3 rayColStart = rayStart + float(px) * dx;
        bool finishedLine = false;
        for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
        {
            const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;

            glm::vec2 p = (-glm::vec2(SCR_WIDTH, SCR_HEIGHT) + 2.0f*glm::vec2(px, py))/glm::vec2(SCR_WIDTH, SCR_HEIGHT);

            // create view ray
            glm::vec3 rd = glm::normalize( p.x*right* halfFovTan + p.y*up * fovTan + front );
            
            glm::vec4 tnor = iCylinder( camPos, rd, pa, pb, cylRadius );
            // const bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
            // if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
            if (tnor.x > 0.0f) 
            {
                if (!finishedLine) finishedLine = true;
                glm::vec3 col;
                float t = tnor.x;
                glm::vec3  hit = (rayColStart + float(py) * dy) * t;
                const float depth = (hit.z * proj[2].z + proj[3].z) / -hit.z ;

                glm::vec3  nor = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                const float lambertCos = glm::dot(nor, -glm::normalize(t*rd));
                if (depth < depthBuffer[index])
                {
                    depthBuffer[index] = depth;
                    framebuffer[index] = vecToColor(255 * lambertCos * lightColor * diffuseI);
                }
            } else if (finishedLine)
            {
                break;
            }
        }
    }    
    return true;
}