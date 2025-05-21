#version 460

layout( points ) in;
layout(triangle_strip, max_vertices = 4) out;

flat in vec3  pa[]; // Extreme 1 position
flat in vec3  pb[]; // Extreme 2 position
flat in vec3  paCamPos[]; // Sphere position in view space.
flat in vec3  pbCamPos[]; // Sphere position in view space.
flat in float cylinderRad[];
flat in uint  id[];


flat out vec3	f_paCamPos; // Extreme 1 position.
flat out vec3	f_pbCamPos; // Extreme 2 position.
smooth out vec3 f_impCamPos;	   // Impostor position in view space.
flat out float	f_cylinderRad;
flat out uint   f_id;

uniform mat4 proj;

void main()
{
    f_cylinderRad = cylinderRad[0];
    f_id = id[0];

    if ( paCamPos[0].z < pbCamPos[0].z )
	{
		f_paCamPos = pbCamPos[0];
		f_pbCamPos = paCamPos[0];
	}
	else
	{
		f_paCamPos = paCamPos[0];
		f_pbCamPos = pbCamPos[0];
	}

    vec3 center = gl_in[ 0 ].gl_Position.xyz;

    const vec3 z = normalize( f_pbCamPos - f_paCamPos );
    const vec3 x = normalize( cross( center, z ) );
    const vec3 y = cross( x, z ); 

    const float dV0 = length( f_paCamPos );
    const float dV1 = length( f_pbCamPos );

    const float sinAngle = f_cylinderRad / dV0;
    float		angle	 = asin( sinAngle );
    const vec3	y1		 = y * f_cylinderRad;
    const vec3	x2		 = x * f_cylinderRad * cos( angle );
    const vec3	y2		 = y1 * sinAngle;
    angle				 = asin( f_cylinderRad / dV1 );
    const vec3 x3		 = x * ( dV1 - f_cylinderRad ) * tan( angle );

    // Compute impostors vertices.
    const vec3 v1 = f_paCamPos - x2 + y2;
    const vec3 v2 = f_paCamPos + x2 + y2;
    const vec3 v3 = f_pbCamPos - x3 + y1;
    const vec3 v4 = f_pbCamPos + x3 + y1;

    f_impCamPos = v1;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

    f_impCamPos = v2;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

    f_impCamPos = v3;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

    f_impCamPos = v4;
    gl_Position = proj * vec4( f_impCamPos, 1.f );
	EmitVertex();

    EndPrimitive();
}