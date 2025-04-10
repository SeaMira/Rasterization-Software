#include "utils/sequential/aux_functions_cylinder.h"

bboxCorners getCylinderBbox(glm::vec3& pa, glm::vec3& pb, glm::vec3& center, 
    const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up, const float cylRadius)
{

    // Compute cylinder coordinates system with 'x' orthogonal to 'view'.
    const glm::vec3 z = glm::normalize( pb - pa );
    const glm::vec3 x = glm::normalize( cross( center, z ) );
    const glm::vec3 y = cross( x, z ); // no need to normalize

    // Compute impostor construction vectors.
    // const float dVa = glm::length( pa );
    // const float dVb = glm::length( pb );

    // const float sinAngle = cylRadius / dVa;
    // float angle	 = asin( sinAngle );
    // const glm::vec3	y1 = y * cylRadius;
    // const glm::vec3	x2 = x * cylRadius * cos( angle );
    // const glm::vec3	y2 = y1 * sinAngle;
    // angle = asin( cylRadius / dVb );
    // const glm::vec3 x3 = x * ( dVb - cylRadius ) * tan( angle );

    glm::vec3 maxMinY1 = pa + x*cylRadius + y*cylRadius;
    glm::vec3 maxMinY2 = pa - x*cylRadius - y*cylRadius;
    glm::vec3 maxMinY3 = pb + x*cylRadius + y*cylRadius;
    glm::vec3 maxMinY4 = pb - x*cylRadius - y*cylRadius;
    glm::vec3 maxMinX1 = pa - x*cylRadius + y*cylRadius;
    glm::vec3 maxMinX2 = pa + x*cylRadius - y*cylRadius;
    glm::vec3 maxMinX3 = pb - x*cylRadius + y*cylRadius;
    glm::vec3 maxMinX4 = pb + x*cylRadius - y*cylRadius;

    const glm::vec4 maxMinY1Proj = proj * glm::vec4(maxMinY1, 1.0f);
    const glm::vec4 maxMinY2Proj = proj * glm::vec4(maxMinY2, 1.0f);
    const glm::vec4 maxMinY3Proj = proj * glm::vec4(maxMinY3, 1.0f);
    const glm::vec4 maxMinY4Proj = proj * glm::vec4(maxMinY4, 1.0f);
    
    glm::vec3 ndcMaxMinY1 = glm::vec3(maxMinY1Proj.x / maxMinY1Proj.w, maxMinY1Proj.y / maxMinY1Proj.w, maxMinY1Proj.z / maxMinY1Proj.w);
    glm::vec3 ndcmaxMinY2 = glm::vec3(maxMinY2Proj.x / maxMinY2Proj.w, maxMinY2Proj.y / maxMinY2Proj.w, maxMinY2Proj.z / maxMinY2Proj.w);
    glm::vec3 ndcmaxMinY3 = glm::vec3(maxMinY3Proj.x / maxMinY3Proj.w, maxMinY3Proj.y / maxMinY3Proj.w, maxMinY3Proj.z / maxMinY3Proj.w);
    glm::vec3 ndcmaxMinY4 = glm::vec3(maxMinY4Proj.x / maxMinY4Proj.w, maxMinY4Proj.y / maxMinY4Proj.w, maxMinY4Proj.z / maxMinY4Proj.w);
    
    // float minY = std::min({maxMinY1.y, maxMinY2.y, maxMinY3.y, maxMinY4.y});
    // float maxY = std::max({maxMinY1.y, maxMinY2.y, maxMinY3.y, maxMinY4.y});
    
    
    const glm::vec4 maxMinX1Proj = proj * glm::vec4(maxMinX1, 1.0f);
    const glm::vec4 maxMinX2Proj = proj * glm::vec4(maxMinX2, 1.0f);
    const glm::vec4 maxMinX3Proj = proj * glm::vec4(maxMinX3, 1.0f);
    const glm::vec4 maxMinX4Proj = proj * glm::vec4(maxMinX4, 1.0f);
    
    glm::vec3 ndcMaxMinX1 = glm::vec3(maxMinX1Proj.x / maxMinX1Proj.w, maxMinX1Proj.y / maxMinX1Proj.w, maxMinX1Proj.z / maxMinX1Proj.w);
    glm::vec3 ndcmaxMinX2 = glm::vec3(maxMinX2Proj.x / maxMinX2Proj.w, maxMinX2Proj.y / maxMinX2Proj.w, maxMinX2Proj.z / maxMinX2Proj.w);
    glm::vec3 ndcmaxMinX3 = glm::vec3(maxMinX3Proj.x / maxMinX3Proj.w, maxMinX3Proj.y / maxMinX3Proj.w, maxMinX3Proj.z / maxMinX3Proj.w);
    glm::vec3 ndcmaxMinX4 = glm::vec3(maxMinX4Proj.x / maxMinX4Proj.w, maxMinX4Proj.y / maxMinX4Proj.w, maxMinX4Proj.z / maxMinX4Proj.w);

    glm::vec2 minCorner = glm::min(glm::min(glm::min(ndcMaxMinX1, ndcmaxMinX2), 
        glm::min(ndcmaxMinX3, ndcmaxMinX4)), glm::min(glm::min(ndcMaxMinY1, ndcmaxMinY2), 
        glm::min(ndcmaxMinY3, ndcmaxMinY4)));

    glm::vec2 maxCorner = glm::max(glm::max(glm::max(ndcMaxMinX1, ndcmaxMinX2), 
        glm::max(ndcmaxMinX3, ndcmaxMinX4)), glm::max(glm::max(ndcMaxMinY1, ndcmaxMinY2), 
        glm::max(ndcmaxMinY3, ndcmaxMinY4)));
    
    // float minX = std::min({maxMinY1.x, maxMinY2.x, maxMinY3.x, maxMinY4.x});
    // float maxX = std::max({maxMinY1.x, maxMinY2.x, maxMinY3.x, maxMinY4.x});

    // glm::vec3 minCorner = glm::vec3(minX, minY, 0.0f);
    // glm::vec3 maxCorner = glm::vec3(maxX, maxY, 0.0f);
    

    // Compute impostors vertices.
    // const glm::vec3 upLeftCorner = pa - x2 + y2;
    // const glm::vec3 downLeftCorner = pa + x2 + y2;
    // const glm::vec3 upRightCorner = pb - x3 + y1;
    // const glm::vec3 downRightCorner = pb + x3 + y1;

    // const glm::vec4 upRight = proj * glm::vec4(upRightCorner, 1.0f);
    // const glm::vec4 upLeft = proj * glm::vec4(upLeftCorner, 1.0f);
    // const glm::vec4 downRight = proj * glm::vec4(downRightCorner, 1.0f);
    // const glm::vec4 downLeft = proj * glm::vec4(downLeftCorner, 1.0f);
       
    // glm::vec3 ndcUpRight = glm::vec3(upRight.x / upRight.w, upRight.y / upRight.w, upRight.z / upRight.w);
    // glm::vec3 ndcUpLeft = glm::vec3(upLeft.x / upLeft.w, upLeft.y / upLeft.w, upLeft.z / upLeft.w);
    // glm::vec3 ndcDownRight = glm::vec3(downRight.x / downRight.w, downRight.y / downRight.w, downRight.z / downRight.w);
    // glm::vec3 ndcDownLeft = glm::vec3(downLeft.x / downLeft.w, downLeft.y / downLeft.w, downLeft.z / downLeft.w);

    // glm::vec2 minCorner = glm::min(glm::min(ndcUpRight, ndcUpLeft), glm::min(ndcDownRight, ndcDownLeft));
    // glm::vec2 maxCorner = glm::max(glm::max(ndcUpRight, ndcUpLeft), glm::max(ndcDownRight, ndcDownLeft));

    // return {upRightCorner, ndcUpRight.z, upLeftCorner, ndcUpLeft.z, downRightCorner, ndcDownRight.z, downLeftCorner, ndcDownLeft.z, minCorner, maxCorner};
    return {maxMinY1, maxMinY1.z, maxMinY1, maxMinY1.z, maxMinY1, maxMinY1.z, maxMinY1, maxMinY1.z, minCorner, maxCorner};
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

glm::vec3 pattern(glm::vec2 uv)
{
    glm::vec3 col = glm::vec3(0.6);
    col += 0.4*glm::smoothstep(-0.01,0.01,cos(uv.x*0.5)*cos(uv.y*0.5)); 
    col *= glm::smoothstep(-1.0f,-0.98f,cos(uv.x))* glm::smoothstep(-1.0f,-0.98f,cos(uv.y));
    return col;
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

    // const glm::vec4 upRight = proj * glm::vec4(cylBbox.upRightCorner, 1.0f);
    // const glm::vec4 upLeft = proj * glm::vec4(cylBbox.upLeftCorner, 1.0f);
    // const glm::vec4 downRight = proj * glm::vec4(cylBbox.downRightCorner, 1.0f);
    // const glm::vec4 downLeft = proj * glm::vec4(cylBbox.downLeftCorner, 1.0f);
       
    // glm::vec3 ndcUpRight = glm::vec3(upRight.x / upRight.w, upRight.y / upRight.w, upRight.z / upRight.w);
    // glm::vec3 ndcUpLeft = glm::vec3(upLeft.x / upLeft.w, upLeft.y / upLeft.w, upLeft.z / upLeft.w);
    // glm::vec3 ndcDownRight = glm::vec3(downRight.x / downRight.w, downRight.y / downRight.w, downRight.z / downRight.w);
    // glm::vec3 ndcDownLeft = glm::vec3(downLeft.x / downLeft.w, downLeft.y / downLeft.w, downLeft.z / downLeft.w);

    // glm::ivec2 green = glm::ivec2((ndcUpRight.x* 0.5f + 0.5f) * SCR_WIDTH, (ndcUpRight.y * 0.5f + 0.5f) * SCR_HEIGHT);
    // glm::ivec2 red = glm::ivec2((ndcUpLeft.x* 0.5f + 0.5f) * SCR_WIDTH, (ndcUpLeft.y * 0.5f + 0.5f) * SCR_HEIGHT);
    // glm::ivec2 white = glm::ivec2((ndcDownRight.x* 0.5f + 0.5f) * SCR_WIDTH, (ndcDownRight.y * 0.5f + 0.5f) * SCR_HEIGHT);
    // glm::ivec2 black = glm::ivec2((ndcDownLeft.x* 0.5f + 0.5f) * SCR_WIDTH, (ndcDownLeft.y * 0.5f + 0.5f) * SCR_HEIGHT);
    // if ( green.y < SCR_HEIGHT - 1 && green.y >= 1 && green.x < SCR_WIDTH - 1 && green.x >= 1 )
    // {
    //     std::cout << "green: " << green.x << ", " << green.y << std::endl;
    //     framebuffer[(SCR_HEIGHT - (int)green.y - 1) * SCR_WIDTH + (int)green.x] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y ) * SCR_WIDTH + (int)green.x+1] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y - 2) * SCR_WIDTH + (int)green.x+1] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y ) * SCR_WIDTH + (int)green.x-1] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y - 2) * SCR_WIDTH + (int)green.x-1] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y - 1) * SCR_WIDTH + (int)green.x +1] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y - 1) * SCR_WIDTH + (int)green.x - 1] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y) * SCR_WIDTH + (int)green.x] = vecToColor(glm::vec3(0, 255, 0));
    //     framebuffer[(SCR_HEIGHT - (int)green.y - 2) * SCR_WIDTH + (int)green.x] = vecToColor(glm::vec3(0, 255, 0));
    // }
    // if ( red.y < SCR_HEIGHT - 1 && red.y >= 1 && red.x < SCR_WIDTH - 1 && red.x >= 1 )
    // {
    //     std::cout << "red: " << red.x << ", " << red.y << std::endl;
    //     framebuffer[(SCR_HEIGHT - (int)red.y - 1) * SCR_WIDTH + (int)red.x] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y ) * SCR_WIDTH + (int)red.x+1] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y - 2) * SCR_WIDTH + (int)red.x+1] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y ) * SCR_WIDTH + (int)red.x-1] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y - 2) * SCR_WIDTH + (int)red.x-1] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y - 1) * SCR_WIDTH + (int)red.x +1] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y - 1) * SCR_WIDTH + (int)red.x - 1] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y) * SCR_WIDTH + (int)red.x] = vecToColor(glm::vec3(255, 0, 0));
    //     framebuffer[(SCR_HEIGHT - (int)red.y - 2) * SCR_WIDTH + (int)red.x] = vecToColor(glm::vec3(255, 0, 0));
    // }
    // if ( white.y < SCR_HEIGHT - 1 && white.y >= 1 && white.x < SCR_WIDTH - 1 && white.x >= 1 )
    // {
    //     std::cout << "white: " << white.x << ", " << white.y << std::endl;
    //     framebuffer[(SCR_HEIGHT - (int)white.y - 1) * SCR_WIDTH + (int)white.x] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y ) * SCR_WIDTH + (int)white.x+1] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y - 2) * SCR_WIDTH + (int)white.x+1] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y ) * SCR_WIDTH + (int)white.x-1] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y - 2) * SCR_WIDTH + (int)white.x-1] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y - 1) * SCR_WIDTH + (int)white.x +1] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y - 1) * SCR_WIDTH + (int)white.x - 1] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y) * SCR_WIDTH + (int)white.x] = vecToColor(glm::vec3(255));
    //     framebuffer[(SCR_HEIGHT - (int)white.y - 2) * SCR_WIDTH + (int)white.x] = vecToColor(glm::vec3(255));
    // }
    // if ( black.y < SCR_HEIGHT - 1 && black.y >= 1 && black.x < SCR_WIDTH - 1 && black.x >= 1 )
    // {
    //     std::cout << "black: " << black.x << ", " << black.y << std::endl;
    //     framebuffer[(SCR_HEIGHT - (int)black.y - 1) * SCR_WIDTH + (int)black.x] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y ) * SCR_WIDTH + (int)black.x+1] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y - 2) * SCR_WIDTH + (int)black.x+1] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y ) * SCR_WIDTH + (int)black.x-1] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y - 2) * SCR_WIDTH + (int)black.x-1] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y - 1) * SCR_WIDTH + (int)black.x +1] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y - 1) * SCR_WIDTH + (int)black.x - 1] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y) * SCR_WIDTH + (int)black.x] = vecToColor(glm::vec3(0));
    //     framebuffer[(SCR_HEIGHT - (int)black.y - 2) * SCR_WIDTH + (int)black.x] = vecToColor(glm::vec3(0));
    // }
    // framebuffer[(SCR_HEIGHT - (int)red.y - 1) * SCR_WIDTH + (int)red.x] = vecToColor(glm::vec3(255, 0, 0));
    // framebuffer[(SCR_HEIGHT - (int)white.y - 1) * SCR_WIDTH + (int)white.x] = vecToColor(glm::vec3(255));
    // framebuffer[(SCR_HEIGHT - (int)black.y - 1) * SCR_WIDTH + (int)black.x] = vecToColor(glm::vec3(0));

    float fovRad = glm::radians(fov);
    float fovTan = tan(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;
    for (int px = std::max(0, screenMin.x); px < std::min(screenMax.x, SCR_WIDTH); px++)
    {
        bool finishedLine = false;
        for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
        {
            const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
            // const float u = (float)(px - screenMin.x) / difx;
            // const float v = (float)(py - screenMin.y) / dify;
            // const glm::vec3 A = glm::mix(cylBbox.upLeftCorner, cylBbox.upRightCorner, u);
            // const glm::vec3 B = glm::mix(cylBbox.downLeftCorner, cylBbox.downRightCorner, u);
            // const glm::vec3 viewImpPos = glm::mix(A, B, v);
            // const glm::vec3 rd = glm::normalize(viewImpPos);
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
                glm::vec3  hit = glm::vec3(view*glm::vec4(rd, 1.0f))*t;
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