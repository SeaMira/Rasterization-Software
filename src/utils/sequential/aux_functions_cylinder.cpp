#include <algorithm>
#include "utils/sequential/aux_functions_cylinder.h"

// BBoxCorners getCylinderBbox(glm::vec3& pa, glm::vec3& pb, glm::vec3& center, 
//     const glm::mat4& proj, const glm::vec3& camPos, 
//     const glm::vec3& front, const glm::vec3& up, const float cylRadius)
// {

//     // Cylinder axis
//     const glm::vec3 z = glm::normalize(pb - pa);

//     // Find orthonormal x,y axes orthogonal to cylinder axis
//     glm::vec3 arbitrary = glm::abs(z.x) < 0.99f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
//     glm::vec3 x = glm::normalize(glm::cross(arbitrary, z));
//     glm::vec3 y = glm::normalize(glm::cross(z, x)); // make full basis

//     // Generate 8 points around caps (pa and pb)
//     std::vector<glm::vec3> points = {
//         pa + cylRadius * x + cylRadius * y,
//         pa + cylRadius * x - cylRadius * y,
//         pa - cylRadius * x + cylRadius * y,
//         pa - cylRadius * x - cylRadius * y,
//         pb + cylRadius * x + cylRadius * y,
//         pb + cylRadius * x - cylRadius * y,
//         pb - cylRadius * x + cylRadius * y,
//         pb - cylRadius * x - cylRadius * y
//     };

//     glm::vec2 minCorner(FLT_MAX), maxCorner(-FLT_MAX);

//     for (const glm::vec3& p : points)
//     {
//         glm::vec4 clip = proj * glm::vec4(p, 1.0f);
//         if (clip.w != 0.0f)
//         {
//             glm::vec3 ndc = glm::vec3(clip) / clip.w;
//             minCorner = glm::min(minCorner, glm::vec2(ndc));
//             maxCorner = glm::max(maxCorner, glm::vec2(ndc));
//         }
//     }
    
    
//     return {minCorner, maxCorner};
// }

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
    // t = ( ((y<0.0) ? 0.0 : baba) - baoc)/bard;
    // if( abs(k1+k2*t)<h ) return glm::vec4( t, ba*glm::sign(y)/sqrt(baba) );

    return glm::vec4(-1.0);
}


// bool drawCylinder(const glm::mat4& proj, const glm::mat4& view, 
//     const glm::vec3& up, const glm::vec3& front, const glm::vec3& right, const glm::vec3& camPos, 
//     const int SCR_WIDTH, const int SCR_HEIGHT, 
//     const glm::vec3& pa, const glm::vec3& pb, const float& cylRadius, const float& fov,
//     std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer)
// {
//     glm::vec3 camSpaceCylA = glm::vec3(view * glm::vec4(pa.x, pa.y, pa.z, 1.0f));
//     glm::vec3 camSpaceCylB = glm::vec3(view * glm::vec4(pb.x, pb.y, pb.z, 1.0f));

//     glm::vec3 camImpPosA, camImpPosB;
//     if ( camSpaceCylA.z < camSpaceCylB.z )
// 	{
// 		camImpPosA = camSpaceCylB;
// 		camImpPosB = camSpaceCylA;
// 	}
// 	else
// 	{
// 		camImpPosA = camSpaceCylA;
// 		camImpPosB = camSpaceCylB;
// 	}
//     glm::vec3 center = normalize( ( camImpPosA + camImpPosB ) * 0.5f );
    
//     BBoxCorners cylBbox = getCylinderBbox(camImpPosA, camImpPosB, center,
//         proj, camPos, front, up, cylRadius);
        
//     float aspectRatio = (float)SCR_WIDTH / (float)SCR_HEIGHT;
//     glm::ivec2 screenMin, screenMax;

//     screenMin.x = ((cylBbox.minCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
//     screenMin.y = (cylBbox.minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
//     screenMax.x = ((cylBbox.maxCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
//     screenMax.y = (cylBbox.maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;

//     const float difx = screenMax.x - screenMin.x;
//     const float dify = screenMax.y - screenMin.y;
//     if (difx*dify <= 2) return false;
    
//     float fovRad = glm::radians(fov);
//     float fovTan = tan(fovRad * 0.5f);
//     float halfFovTan = fovTan * aspectRatio;

//     // View-space ray directions for screen corners
//     ScreenRayCasting screenRayCasting(fov, aspectRatio, SCR_WIDTH, SCR_HEIGHT, right, up, front, view);

//     // Delta per pixel
//     glm::vec3 dx = screenRayCasting.dx;
//     glm::vec3 dy = screenRayCasting.dy;

//     // Start ray origin in world
//     glm::vec3 rayStart = screenRayCasting.rayStart;
//     #pragma omp parallel for collapse(2)
//     for (int px = std::max(0, screenMin.x); px < std::min(screenMax.x, SCR_WIDTH); px++)
//     {
//         glm::vec3 rayColStart = rayStart + float(px) * dx;
//         bool finishedLine = false;
//         for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
//         {
//             const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;

//             glm::vec2 p = (-glm::vec2(SCR_WIDTH, SCR_HEIGHT) + 2.0f*glm::vec2(px, py))/glm::vec2(SCR_WIDTH, SCR_HEIGHT);

//             // create view ray
//             glm::vec3 rd = glm::normalize( p.x*right* halfFovTan + p.y*up * fovTan + front );
            
//             glm::vec4 tnor = iCylinder( camPos, rd, pa, pb, cylRadius );
//             // const bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
//             // if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
//             if (tnor.x > 0.0f) 
//             {
//                 if (!finishedLine) finishedLine = true;
//                 float t = tnor.x;
//                 const glm::vec3  hit = (rayColStart + float(py) * dy) * t;
//                 const float depth = (hit.z * proj[2].z + proj[3].z) / -hit.z ;

//                 if (depth < depthBuffer[index])
//                 {
//                     glm::vec3  normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
//                     const float lambertCos = glm::dot(normal, -glm::normalize(t*rd));
//                     depthBuffer[index] = depth;
//                     framebuffer[index] = vecToColor(255 * lambertCos * lightColor * diffuseI);
//                 }
//             } else if (finishedLine)
//             {
//                 break;
//             }
//         }
//     }    
//     return true;
// }


std::vector<glm::vec2> getCylinderBbox(glm::vec3& pa, glm::vec3& pb, glm::vec3& center, 
    const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up, const float cylRadius)
{

    // Cylinder axis
    const glm::vec3 z = glm::normalize(pb - pa);

    // Find orthonormal x,y axes orthogonal to cylinder axis
    glm::vec3 x = glm::normalize(glm::cross(center, z));
    glm::vec3 y = glm::normalize(glm::cross(x, z)); // make full basis

    // Compute impostor construction vectors.
    const float dV0 = glm::length( pa );
    const float dV1 = glm::length( pb );

    const float sinAngle = cylRadius / dV0;
    float		angle	 = asin( sinAngle );
    const glm::vec3	y1		 = y * cylRadius;
    const glm::vec3	x2		 = x * cylRadius * cos( angle );
    const glm::vec3	y2		 = y1 * sinAngle;
    angle				 = asin( cylRadius / dV1 );
    const glm::vec3 x3		 = x * ( dV1 - cylRadius ) * tan( angle );

    // Compute impostors vertices.
    const glm::vec3 v1 = pa - x2 + y2;
    const glm::vec3 v2 = pa + x2 + y2;
    const glm::vec3 v3 = pb - x3 + y1;
    const glm::vec3 v4 = pb + x3 + y1;

    const glm::vec4 v1Proj = proj * glm::vec4(v1, 1.0f);
    const glm::vec4 v2Proj = proj * glm::vec4(v2, 1.0f);
    const glm::vec4 v3Proj = proj * glm::vec4(v3, 1.0f);
    const glm::vec4 v4Proj = proj * glm::vec4(v4, 1.0f);

    glm::vec3 ndcv1Proj = glm::vec3(v1Proj.x / v1Proj.w, v1Proj.y / v1Proj.w, v1Proj.z / v1Proj.w);
    glm::vec3 ndcv2Proj = glm::vec3(v2Proj.x / v2Proj.w, v2Proj.y / v2Proj.w, v2Proj.z / v2Proj.w);
    glm::vec3 ndcv3Proj = glm::vec3(v3Proj.x / v3Proj.w, v3Proj.y / v3Proj.w, v3Proj.z / v3Proj.w);
    glm::vec3 ndcv4Proj = glm::vec3(v4Proj.x / v4Proj.w, v4Proj.y / v4Proj.w, v4Proj.z / v4Proj.w);

    std::vector<glm::vec2> projectedPoints = {
        glm::vec2(ndcv1Proj),
        glm::vec2(ndcv2Proj),
        glm::vec2(ndcv3Proj),
        glm::vec2(ndcv4Proj)
    };
    
    return projectedPoints;
}

float intersectX(const glm::vec2& a, const glm::vec2& b, float y) 
{
    if (a.y == b.y) return a.x; // horizontal line
    float t = (y - a.y) / (b.y - a.y);
    return a.x + t * (b.x - a.x);
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
    std::vector<glm::vec2> projectedPoints = getCylinderBbox(camImpPosA, camImpPosB, center, proj, camPos, front, up, cylRadius);

    int minY = SCR_HEIGHT;
    int maxY = 0;
    for (auto& point : projectedPoints) 
    {
        point.x = (int)((point.x * 0.5f + 0.5f) * SCR_WIDTH);
        point.y = (int)((point.y * 0.5f + 0.5f) * SCR_HEIGHT);

        minY = std::min(minY, (int)point.y);
        maxY = std::max(maxY, (int)point.y);
    }

    float aspectRatio = (float)SCR_WIDTH / (float)SCR_HEIGHT;
    float fovRad = glm::radians(fov);
    float fovTan = tan(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    // View-space ray directions for screen corners
    ScreenRayCasting screenRayCasting(fov, aspectRatio, SCR_WIDTH, SCR_HEIGHT, right, up, front, view);

    // Delta per pixel
    glm::vec3 dx = screenRayCasting.dx;
    glm::vec3 dy = screenRayCasting.dy;
    // Start ray origin in world
    glm::vec3 rayStart = screenRayCasting.rayStart;
    
    #pragma omp parallel for collapse(2)
    for (int py = std::min(SCR_HEIGHT-1, maxY); py >= std::max(0, minY); --py)
    {
        std::vector<float> xIntersections;

        for (int i = 0; i < 4; ++i) 
        {
            const glm::vec2& a = projectedPoints[i];
            const glm::vec2& b = projectedPoints[(i + 1) % 4];

            if ((py >= a.y && py <= b.y) || (py >= b.y && py <= a.y)) 
            {
                float x = intersectX(a, b, py);
                xIntersections.push_back(x);
            }
        }
        if (xIntersections.size() >= 2) 
        {
            float xMin = xIntersections[0];
            float xMax = xIntersections[0];
            for (size_t i = 1; i < xIntersections.size(); ++i) 
            {
                xMin = std::min(xMin, xIntersections[i]);
                xMax = std::max(xMax, xIntersections[i]);
            }
            glm::vec3 rayColStart = rayStart + float(py) * dy;
            for (int px = std::max(0, (int)std::ceil(xMin)); px <= std::min((int)std::floor(xMax), SCR_WIDTH -1); ++px) 
            {

                glm::vec2 p = (-glm::vec2(SCR_WIDTH, SCR_HEIGHT) + 2.0f*glm::vec2(px, py))/glm::vec2(SCR_WIDTH, SCR_HEIGHT);

                // create view ray
                glm::vec3 rd = glm::normalize( p.x*right* halfFovTan + p.y*up * fovTan + front );

                glm::vec4 tnor = iCylinder( camPos, rd, pa, pb, cylRadius );
                int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
                // const bool showBbox = (px == std::min((int)std::floor(xMax), SCR_WIDTH -1) || 
                //     py == std::min(SCR_HEIGHT-1, (int)projectedPoints[0].y) || 
                //     px == std::max(0, (int)std::ceil(xMin)) || 
                //     py == std::max(0, (int)projectedPoints[2].y));
                // if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
                if (tnor.x > 0.0f) 
                {
                    float t = tnor.x;
                    const glm::vec3  hit = (rayColStart + float(px) * dx) * t;
                    const float depth = (hit.z * proj[2].z + proj[3].z) / -hit.z ;

                    if (depth < depthBuffer[index])
                    {
                        glm::vec3  normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                        const float lambertCos = glm::dot(normal, -glm::normalize(t*rd));
                        depthBuffer[index] = depth;
                        framebuffer[index] = vecToColor(255 * lambertCos * lightColor * diffuseI);
                    }
                }
            }
        }
    }

    return true;

}

// std::vector<glm::vec2> getCylinderBbox3rdAttempt(const glm::vec3& pa, const glm::vec3& pb, const glm::vec3& center, 
//     const glm::mat4& proj, const glm::vec3& camPos, 
//     const glm::vec3& front, const glm::vec3& up, const float cylRadius)
// {
//     // Cylinder axis
//     const glm::vec3 z = glm::normalize(pb - pa);

//     // Find orthonormal x,y axes orthogonal to cylinder axis
//     glm::vec3 x = glm::normalize(glm::cross(center, z));
//     glm::vec3 y = glm::normalize(glm::cross(x, z)); // make full basis

//     // Compute impostor construction vectors.
//     const float dV0 = glm::length( pa );
//     const float dV1 = glm::length( pb );

//     const float sinAngle = cylRadius / dV0;
//     float		angle	 = asin( sinAngle );
//     const glm::vec3	y1		 = y * cylRadius;
//     const glm::vec3	x2		 = x * cylRadius * cos( angle );
//     const glm::vec3	y2		 = y1 * sinAngle;
//     angle				 = asin( cylRadius / dV1 );
//     const glm::vec3 x3		 = x * ( dV1 - cylRadius ) * tan( angle );

//     // Compute impostors vertices.
//     const glm::vec3 v1 = pa - x2 + y2;
//     const glm::vec3 v2 = pa + x2 + y2;
//     const glm::vec3 v3 = pb - x3 + y1;
//     const glm::vec3 v4 = pb + x3 + y1;

//     const glm::vec4 v1Proj = proj * glm::vec4(v1, 1.0f);
//     const glm::vec4 v2Proj = proj * glm::vec4(v2, 1.0f);
//     const glm::vec4 v3Proj = proj * glm::vec4(v3, 1.0f);
//     const glm::vec4 v4Proj = proj * glm::vec4(v4, 1.0f);

//     glm::vec3 ndcv1Proj = glm::vec3(v1Proj.x / v1Proj.w, v1Proj.y / v1Proj.w, v1Proj.z / v1Proj.w);
//     glm::vec3 ndcv2Proj = glm::vec3(v2Proj.x / v2Proj.w, v2Proj.y / v2Proj.w, v2Proj.z / v2Proj.w);
//     glm::vec3 ndcv3Proj = glm::vec3(v3Proj.x / v3Proj.w, v3Proj.y / v3Proj.w, v3Proj.z / v3Proj.w);
//     glm::vec3 ndcv4Proj = glm::vec3(v4Proj.x / v4Proj.w, v4Proj.y / v4Proj.w, v4Proj.z / v4Proj.w);

//     std::vector<glm::vec2> projectedPoints = {
//         glm::vec2(ndcv1Proj),
//         glm::vec2(ndcv2Proj),
//         glm::vec2(ndcv3Proj),
//         glm::vec2(ndcv4Proj)
//     };

    
//     return projectedPoints;
// }

// bool drawCylinder3rdAttempt(const glm::mat4& proj, const glm::mat4& view, 
//     const glm::vec3& up, const glm::vec3& front, const glm::vec3& right, const glm::vec3& camPos, 
//     const int SCR_WIDTH, const int SCR_HEIGHT, 
//     const glm::vec3& pa, const glm::vec3& pb, const float& cylRadius, const float& fov,
//     std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer)
// {
//     glm::vec3 camSpaceCylA = glm::vec3(view * glm::vec4(pa.x, pa.y, pa.z, 1.0f));
//     glm::vec3 camSpaceCylB = glm::vec3(view * glm::vec4(pb.x, pb.y, pb.z, 1.0f));

//     glm::vec3 camImpPosA, camImpPosB;
//     if ( camSpaceCylA.z < camSpaceCylB.z )
// 	{
// 		camImpPosA = camSpaceCylB;
// 		camImpPosB = camSpaceCylA;
// 	}
// 	else
// 	{
// 		camImpPosA = camSpaceCylA;
// 		camImpPosB = camSpaceCylB;
// 	}
//     glm::vec3 center = normalize( ( camImpPosA + camImpPosB ) * 0.5f );
//     std::vector<glm::vec2> projectedPoints = getCylinderBbox3rdAttempt(camImpPosA, camImpPosB, center, proj, camPos, front, up, cylRadius);

//     int minY = SCR_HEIGHT;
//     int maxY = 0;
//     int minX = SCR_WIDTH;
//     int maxX = 0;
//     for (auto& point : projectedPoints) 
//     {
//         point.x = (int)((point.x * 0.5f + 0.5f) * SCR_WIDTH);
//         point.y = (int)((point.y * 0.5f + 0.5f) * SCR_HEIGHT);

//         minY = std::min(minY, (int)point.y);
//         maxY = std::max(maxY, (int)point.y);
//         minX = std::min(minX, (int)point.x);
//         maxX = std::max(maxX, (int)point.x);
//     }

//     float aspectRatio = (float)SCR_WIDTH / (float)SCR_HEIGHT;
//     float fovRad = glm::radians(fov);
//     float fovTan = tan(fovRad * 0.5f);
//     float halfFovTan = fovTan * aspectRatio;

//     // View-space ray directions for screen corners
//     ScreenRayCasting screenRayCasting(fov, aspectRatio, SCR_WIDTH, SCR_HEIGHT, right, up, front, view);

//     // Delta per pixel
//     glm::vec3 dx = screenRayCasting.dx;
//     glm::vec3 dy = screenRayCasting.dy;
//     // Start ray origin in world
//     glm::vec3 rayStart = screenRayCasting.rayStart;

//     #pragma omp parallel for collapse(2)
//     for (int px = std::max(0, minX); px < std::min(SCR_WIDTH, maxX); ++px)
//     {
//         glm::vec3 rayColStart = rayStart + float(px) * dx;
//         for (int py = std::max(0, minY); py < std::min(SCR_HEIGHT, maxY); ++py)
//         {
//             glm::vec2 p = (-glm::vec2(SCR_WIDTH, SCR_HEIGHT) + 2.0f*glm::vec2(px, py))/glm::vec2(SCR_WIDTH, SCR_HEIGHT);

//             // create view ray
//             glm::vec3 rd = glm::normalize( p.x*right* halfFovTan + p.y*up * fovTan + front );

//             glm::vec4 tnor = iCylinder( camPos, rd, pa, pb, cylRadius );
//             int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
//             const bool showBbox = (px == std::min(maxX, SCR_WIDTH)-1 || 
//                 py == std::min(SCR_HEIGHT, maxY)-1 || 
//                 px == std::max(0, minX) || 
//                 py == std::max(0, minY));
//             if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
//             if (tnor.x > 0.0f) 
//             {
//                 float t = tnor.x;
//                 const glm::vec3  hit = (rayColStart + float(py) * dy) * t;
//                 const float depth = (hit.z * proj[2].z + proj[3].z) / -hit.z ;

//                 if (depth < depthBuffer[index])
//                 {
//                     glm::vec3  normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
//                     const float lambertCos = glm::dot(normal, -glm::normalize(t*rd));
//                     depthBuffer[index] = depth;
//                     framebuffer[index] = vecToColor(255 * lambertCos * lightColor * diffuseI);
//                 }
//             }
//         }
//     }

// }