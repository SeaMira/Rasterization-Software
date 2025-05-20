#version 460

struct Sphere
{
    vec3  pos;
    float rad;
};

layout(std430, binding = 1) buffer SphereBuffer 
{
    Sphere spheres[];
};


uniform mat4 view;

uniform vec3 up;

flat out vec3  spherePos; // Sphere position
flat out vec3  sphereCamPos; // Sphere position in view space.
flat out float sphereRad;
flat out vec3  impU; 
flat out vec3  impV;
flat out uint  id;

void main()
{
    id = gl_VertexID;
    Sphere sphere = spheres[gl_VertexID];
    spherePos = sphere.pos;
    sphereRad = sphere.rad;
    sphereCamPos = vec3( view * vec4( sphere.pos, 1. ) );

    vec3 normSphViewPos = normalize(sphereCamPos);
    vec3 camImposPos = sphereCamPos - sphereRad*normSphViewPos;

    // get sphere bbox
    const float sinAngle = sphereRad / (length(sphereCamPos) + 1e-6f);
    const float tanAngle = tan(asin(sinAngle));
    const float quadScale = tanAngle * length(camImposPos);

    impU = normalize(cross(normSphViewPos, up));
    impV = cross(impU, normSphViewPos) * quadScale;
    impU *= quadScale;

    gl_Position = vec4( camImposPos, 1.f );
}