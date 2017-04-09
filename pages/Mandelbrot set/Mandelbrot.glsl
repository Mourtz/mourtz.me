#ifdef GL_ES
precision mediump float;
#endif

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

#define zoom 3.0
#define MAX_ITERATIONS 128
#define MAX_DISTANCE 2.0

void main( void ) {
	vec2 uv = gl_FragCoord.xy / u_resolution;
    uv *= zoom;

	// Calculate starting values for Mandelbrot set for the pixel
	vec2 c = vec2(-zoom/2.0 - 1.0,-zoom/2.0) + uv;

	highp vec3 shade = vec3(0.0);

	// Mandelbrot loop.
	vec2 z = vec2(0.0);
	for (int i = 0; i < MAX_ITERATIONS ; i++) {
		// The famous Mandelbrot equation (z = z^2+c)
		z = vec2(z.x*z.x - z.y*z.y, 2.*z.x*z.y) + c;

		// break if point becomes greater than 2.0
		if (distance(z.x, z.y) > MAX_DISTANCE) break;

		shade += vec3(1.0-dot(z,z)*0.125,0.5,1.0);
	}

	shade = 2e-2*(shade - log2(log2(dot(z,z)))); 
	gl_FragColor = vec4(shade, 1.0);
}
