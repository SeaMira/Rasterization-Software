#version 460

struct CylinderIndex
{
    uint sphereIndexA;
    uint sphereIndexB;
    float radius;
    uint _padding;
};

layout(std430, binding = 6) buffer SphereBuffer
{
    vec4 spheres[];
};

layout(std430, binding = 2) buffer CylinderBuffer 
{
    CylinderIndex cylinderIndices[];
};

uniform mat4 view;

flat out vec3  pa; // Extreme 1 position
flat out vec3  pb; // Extreme 2 position
flat out vec3  paCamPos; // Sphere position in view space.
flat out vec3  pbCamPos; // Sphere position in view space.
flat out float cylinderRad;
flat out uint  id;

void main()
{
    id = gl_VertexID;
    CylinderIndex cylinder = cylinderIndices[gl_VertexID];
    pa = spheres[cylinder.sphereIndexA].xyz;
    pb = spheres[cylinder.sphereIndexB].xyz;
    cylinderRad = cylinder.radius;
    paCamPos = vec3( view * vec4( pa, 1. ) );
    pbCamPos = vec3( view * vec4( pb, 1. ) );

    gl_Position = vec4( (paCamPos + pbCamPos) * 0.5f, 1.f );
}