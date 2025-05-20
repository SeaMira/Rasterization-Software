#version 460

layout (depth_less) out float gl_FragDepth;

flat in vec3	f_spherePos; // Sphere position.
flat in vec3	f_sphereCamPos; // Sphere position in view space.
smooth in vec3 f_impCamPos;	   // Impostor position in view space.
flat in float	f_sphereRad;
flat in uint   f_id;

out vec4 FragColor;

uniform mat4 proj;

vec3 lightColor = vec3(0.01, 1.0, 0.05);
float diffuseI = 0.9;

void main()
{
    float a, b, delta;

    a = dot( f_impCamPos, f_impCamPos );
    b = dot( f_impCamPos, f_sphereCamPos );
    const float c = dot(f_sphereCamPos, f_sphereCamPos) - f_sphereRad * f_sphereRad;
    delta = b * b - a * c;

    if ( delta >= 0.f )
	{
        const float t = ( b - sqrt( delta ) ) / a;

        const vec3 hit	  = f_impCamPos * t;
		const vec3 normal = normalize( hit - f_sphereCamPos );

        gl_FragDepth = (hit.z * proj[2].z + proj[3].z) / -hit.z ;

        const float lambertCos = dot(normal, -normalize(hit));
        FragColor = vec4(lambertCos * lightColor * diffuseI, 1.0);
    } else discard;
}