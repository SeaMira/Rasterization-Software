#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D outputImage;

uniform vec4 sphere; // [x, y, z, radius]
uniform mat4 viewMatrix;

uniform vec3 front;
uniform vec3 up;
uniform vec3 right;
uniform vec3 cameraPos;

uniform vec2 screenResolution;

uniform float iTime;
uniform float FOV;

uniform bool show_grid;
uniform bool show_axis;

struct ProjectionResult
{
    float area;      // probably all we care about is the area
    vec2  center;    // but i'm outputing all the information 
    vec2  axisA;     // for debugging and illustration purposes
    vec2  axisB;
    // implicit ellipse f(x,y) = a·x² + b·x·y + c·y² + d·x + e·y + f = 0 */
    float a, b, c, d, e, f; 
};

ProjectionResult projectSphere( /* sphere        */ in vec4 sph, 
                        /* camera matrix */ in mat4 cam,
                        /* projection    */ in float fle )
{
    // transform to camera space	
    vec3  o = (cam*vec4(sph.xyz,1.0)).xyz;
    
    float r2 = sph.w*sph.w;
    float z2 = o.z*o.z;	
    float l2 = dot(o,o);
    
    float area = -3.141593*fle*fle*r2*sqrt(abs((l2-r2)/(r2-z2)))/(r2-z2);
    
    //return area;
    
    
    //-- debug stuff ---

    
    // axis
    vec2 axa = fle*sqrt(-r2*(r2-l2)/((l2-z2)*(r2-z2)*(r2-z2)))*vec2( o.x,o.y);
    vec2 axb = fle*sqrt(-r2*(r2-l2)/((l2-z2)*(r2-z2)*(r2-l2)))*vec2(-o.y,o.x);

    //area = length(axa)*length(axb)*3.141593;	
    
    // center
    vec2  cen = fle*o.z*o.xy/(z2-r2);
    

    return ProjectionResult( area, cen, axa, axb, 
                    /* a */ r2 - o.y*o.y - z2,
                    /* b */ 2.0*o.x*o.y,
                    /* c */ r2 - o.x*o.x - z2,
                    /* d */ -2.0*o.x*o.z*fle,
                    /* e */ -2.0*o.y*o.z*fle,
                    /* f */ (r2-l2+z2)*fle*fle );
    
}

float SampleDigit(const in float n, const in vec2 vUV)
{		
    if(vUV.x  < 0.0) return 0.0;
    if(vUV.y  < 0.0) return 0.0;
    if(vUV.x >= 1.0) return 0.0;
    if(vUV.y >= 1.0) return 0.0;
    
    float data = 0.0;
    
        if(n < 0.5) data = 7.0 + 5.0*16.0 + 5.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
    else if(n < 1.5) data = 2.0 + 2.0*16.0 + 2.0*256.0 + 2.0*4096.0 + 2.0*65536.0;
    else if(n < 2.5) data = 7.0 + 1.0*16.0 + 7.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
    else if(n < 3.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
    else if(n < 4.5) data = 4.0 + 7.0*16.0 + 5.0*256.0 + 1.0*4096.0 + 1.0*65536.0;
    else if(n < 5.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 1.0*4096.0 + 7.0*65536.0;
    else if(n < 6.5) data = 7.0 + 5.0*16.0 + 7.0*256.0 + 1.0*4096.0 + 7.0*65536.0;
    else if(n < 7.5) data = 4.0 + 4.0*16.0 + 4.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
    else if(n < 8.5) data = 7.0 + 5.0*16.0 + 7.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
    else if(n < 9.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
    
    vec2 vPixel = floor(vUV * vec2(4.0, 5.0));
    float fIndex = vPixel.x + (vPixel.y * 4.0);
    
    return mod(floor(data / pow(2.0, fIndex)), 2.0);
}

float PrintInt(const in vec2 uv, const in float value )
{
    float res = 0.0;
    float maxDigits = 1.0+ceil(log2(value)/log2(10.0));
    float digitID = floor(uv.x);
    if( digitID>0.0 && digitID<maxDigits )
    {
        float digitVa = mod( floor( value/pow(10.0,maxDigits-1.0-digitID) ), 10.0 );
        res = SampleDigit( digitVa, vec2(fract(uv.x), uv.y) );
    }

    return res;	
}

float iSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    if( h<0.0 ) return -1.0;
    return -b - sqrt( h );
}

float ssSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
    vec3 oc = sph.xyz - ro;
    float b = dot( oc, rd );
    
    float res = 1.0;
    if( b>0.0 )
    {
        float h = dot(oc,oc) - b*b - sph.w*sph.w;
        res = smoothstep( 0.0, 1.0, 12.0*h/b );
    }
    return res;
}

float sdSegment( vec2 p, vec2 a, vec2 b )
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

// Para calcular la derivada, puedes hacerlo manualmente.
vec2 calcGrad(vec2 p, vec2 a, vec2 b, float epsilon) {
    // Derivada en el eje X (usamos un pequeño valor epsilon)
    float dx = sdSegment(p + vec2(epsilon, 0.0), a, b) - sdSegment(p - vec2(epsilon, 0.0), a, b);
    
    // Derivada en el eje Y
    float dy = sdSegment(p + vec2(0.0, epsilon), a, b) - sdSegment(p - vec2(0.0, epsilon), a, b);

    return vec2(dx, dy);
}

float gridTextureGradBox( in vec2 p, in vec2 x, in vec2 y )
{
    const float N = 10.0;
    vec2 grad = calcGrad(p, x, y, 0.01);
    vec2 w = abs(grad) + 0.01;
    vec2 a = p + 0.5*w;
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
            floor(b)-min(fract(b)*N,1.0))/(N*w);
    return (1.0-i.x)*(1.0-i.y);
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

    // Normalizar las coordenadas de la textura a [-1, 1]
    vec2 uv = vec2(coords) / screenResolution * 2.0 - 1.0;

    float fov = FOV/90.0;

    vec3 ro = cameraPos;
    vec3 rd = normalize( uv.x * right + uv.y * up + fov * front );

    const vec4 sph[3] = vec4[3]( vec4(-2.0, 1.0,0.0,1.1),
                                vec4( 3.0, 1.5,1.0,1.2),
                                vec4( 1.0,-1.0,1.0,1.3) );

    float tmin = 10000.0;
    vec3  nor = vec3(0.0);
    vec3  pos = vec3(0.0);

    vec3 sur = vec3(1.0);

    for( int i=0; i<3; i++ )
    {
        float h = iSphere( ro, rd, sph[i] );
        if( h>0.0 && h<tmin ) 
        { 
            tmin = h; 
            pos = ro + h*rd;
            nor = normalize(pos-sph[i].xyz); 
            sur = 0.5 + 0.5*cos(float(i)*2.0+vec3(0.0,2.0,4.0));              
            sur *= 0.4;
            sur *= smoothstep(-0.6,-0.2,sin(20.0*(pos.x-sph[i].x)));
        }
    }

    float h = (-2.0-ro.z)/rd.z;
    if( h>0.0 && h<tmin && show_grid) 
    { 
        tmin = h; 
        pos = ro + h*rd;
        nor = vec3(0.0,0.0,1.0); 
        sur = vec3(1.0)*gridTextureGradBox( pos.xy, pos.xy, pos.xy );
    }

    vec3 col = vec3(0.0);

    if( tmin<100.0 )
    {
        pos = ro + tmin*rd;
        col = vec3(1.0);
        
        vec3 lig = normalize( vec3(2.0,1.4,-1.0) );
        float sha = 1.0;
        for( int i=0; i<3; i++ )
        {
            sha *= ssSphere( pos, lig, sph[i] );
        }

        float ndl = clamp( dot(nor,lig), 0.0, 1.0 );
        col = (0.5+0.5*nor.y)*vec3(0.2,0.3,0.4) + sha*vec3(1.0,0.9,0.8)*ndl + sha*vec3(1.5)*ndl*pow( clamp(dot(normalize(-rd+lig),nor),0.0,1.0), 16.0 );
        col *= sur;
        
        col *= exp( -0.25*(max(0.0,tmin-3.0)) );

    }

    col = pow( col, vec3(0.45) );

    //-------------------------------------------------------
    for( int i=0; i<3; i++ )
    {
        ProjectionResult res = projectSphere( sph[i], viewMatrix, fov );
        res.area *= screenResolution.y*screenResolution.y*0.25;
        if( res.area>0.0 )
        {
            float f = res.a*uv.x*uv.x + res.b*uv.x*uv.y + res.c*uv.y*uv.y + res.d*uv.x + res.e*uv.y + res.f;
            vec2  g = 2.0*vec2(res.a,res.c)*uv.xy + res.b*uv.yx + vec2(res.d,res.e);                   
            float d = abs(f)/length(g);
            if (show_axis) {
                float showMaths = smoothstep( -0.5, 0.5, cos(0.5*6.2831*iTime) );
                col = mix( col, vec3(1.0,0.0,0.0), showMaths*(1.0-smoothstep(0.00,0.01, d)));
                col = mix( col, vec3(1.0,1.0,0.0), showMaths*(1.0-smoothstep(0.00,0.01, sdSegment( uv, -res.center-res.axisA, -res.center+res.axisA )) ));
                col = mix( col, vec3(1.0,1.0,0.0), showMaths*(1.0-smoothstep(0.00,0.01, sdSegment( uv, -res.center-res.axisB, -res.center+res.axisB )) ));
                col = mix( col, vec3(1.0,0.0,0.0), showMaths*(1.0-smoothstep(0.03,0.04, length(uv+res.center))));
            }
            vec2 pp = -res.center + 0.5 * max(max(res.axisA, -res.axisA), max(res.axisB, -res.axisB));
            col = mix( col, vec3(1.0), PrintInt( (uv-pp)/0.07, floor(res.area) ) );
        }
    }

    // Proyectar la esfera
    vec4 fragColor = vec4(col, 1.0);
    imageStore(outputImage, coords, fragColor);
}