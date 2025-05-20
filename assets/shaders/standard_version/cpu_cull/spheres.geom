#version 460

layout( points ) in;
layout(triangle_strip, max_vertices = 4) out;

flat in vec3  spherePos[]; // Sphere position
flat in vec3  sphereCamPos[]; // Sphere position in view space.
flat in float sphereRad[];
flat in vec3  impU[]; 
flat in vec3  impV[];
flat in uint  id[];


flat out vec3	f_spherePos; // Sphere position.
flat out vec3	f_sphereCamPos; // Sphere position in view space.
smooth out vec3 f_impCamPos;	   // Impostor position in view space.
flat out float	f_sphereRad;
flat out uint   f_id;

uniform mat4 proj;

void main()
{
    f_sphereCamPos = sphereCamPos[0];
    f_spherePos = spherePos[0]; 
    f_sphereRad = sphereRad[0];
    f_id = id[0];

    // Compute impostors vertices.
	const vec3 v1 = gl_in[ 0 ].gl_Position.xyz - impU[0] - impV[0];
    f_impCamPos = v1;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

	const vec3 v2 = gl_in[ 0 ].gl_Position.xyz + impU[0] - impV[0];
    f_impCamPos = v2;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

	const vec3 v3 = gl_in[ 0 ].gl_Position.xyz - impU[0] + impV[0];
    f_impCamPos = v3;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

	const vec3 v4 = gl_in[ 0 ].gl_Position.xyz + impU[0] + impV[0];
    f_impCamPos = v4;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

    EndPrimitive();
}