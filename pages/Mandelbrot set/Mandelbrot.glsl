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

	// Calculate starting values for Mandelbrot set for the pixel (complex plane)
	vec2 c = vec2(-zoom/2.0 - 1.0,-zoom/2.0) + uv;

    float shade = 0.0;

	// Mandelbrot loop.
	vec2 z = vec2(0.0);
	for (int i = 0; i < MAX_ITERATIONS ; i++) {
		// z -> z^2+c
		z = vec2(z.x*z.x - z.y*z.y, 2.*z.x*z.y) + c;

		// break if point becomes greater than 2.0
		if (length(z) > MAX_DISTANCE) break;

		++shade;
	}

	shade = 2e-2*(shade - log2(dot(z,z)));
	gl_FragColor = vec4(vec3(shade), 1.0);
}
