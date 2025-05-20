#version 460

layout (depth_less) out float gl_FragDepth;

flat in vec3	f_paCamPos; // Extreme 1 position.
flat in vec3	f_pbCamPos; // Extreme 2 position.
smooth in vec3 f_impCamPos;	   // Impostor position in view space.
flat in float	f_cylinderRad;
flat in uint   f_id;

out vec4 FragColor;

uniform mat4 proj;

vec3 lightColor = vec3(0.01, 1.0, 0.05);
float diffuseI = 0.9;

void main()
{
    // Only consider cylinder body.
	const vec3 v1v0	  = f_pbCamPos - f_paCamPos;
	const vec3 v0	  =  -f_paCamPos;
	const vec3 rayDir = normalize( f_impCamPos );

	const float d0 = dot( v1v0, v1v0 );
	const float d1 = dot( v1v0, rayDir );
	const float d2 = dot( v1v0, v0 );

	const float a = d0 - d1 * d1;
	const float b = d0 * dot( v0, rayDir ) - d2 * d1;
	const float c = d0 * dot( v0, v0 ) - d2 * d2 - f_cylinderRad * f_cylinderRad * d0;

	const float h = b * b - a * c;

    if ( h >= 0.f )
	{
        const float t = ( -b - sqrt( h ) ) / a;

		const float y = d2 + t * d1;

        if ( !(y < 0.f || y > d0) )
        {
            vec3 hit	 = rayDir * t;
			vec3 normal = normalize( v0 + hit - v1v0 * y / d0 );

            gl_FragDepth = (hit.z * proj[2].z + proj[3].z) / -hit.z ;
            const float lambertCos = dot(normal, -normalize(hit));
            FragColor = vec4(lambertCos * lightColor * diffuseI, 1.0);

        } else discard;

    } else discard;
}