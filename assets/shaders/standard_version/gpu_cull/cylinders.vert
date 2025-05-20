#version 460

struct Cylinder 
{
    vec4 pa_r;
    vec4 pb_r;
};

layout(std430, binding = 4) buffer CylinderBuffer 
{
    Cylinder cylinders[];
};

layout(std430, binding = 5) buffer VisibleCylindersIndexes 
{
    uint visibilityIndex[];
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
    id = visibilityIndex[gl_VertexID];
    Cylinder cylinder = cylinders[id];
    pa = cylinder.pa_r.xyz;
    pb = cylinder.pb_r.xyz;
    cylinderRad = cylinder.pa_r.w;
    paCamPos = vec3( view * vec4( pa, 1. ) );
    pbCamPos = vec3( view * vec4( pb, 1. ) );

    gl_Position = vec4( (paCamPos + pbCamPos) * 0.5f, 1.f );
}