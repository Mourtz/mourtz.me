
class CyberpunkRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2', { alpha: false, antialias: false });
        this.isWebGL2 = !!this.gl;

        if (!this.isWebGL2) {
            this.gl = canvas.getContext('webgl', { alpha: false, antialias: false }) ||
                canvas.getContext('experimental-webgl', { alpha: false, antialias: false });
        }

        if (!this.gl) {
            throw new Error('WebGL not supported');
        }

        this.ext = null;
        this.oesDerivatives = null;
        this.linearFloat = null;
        this.colorBufferFloat = null;

        if (this.isWebGL2) {
            this.colorBufferFloat = this.gl.getExtension('EXT_color_buffer_float');
            this.linearFloat = this.gl.getExtension('OES_texture_float_linear');
        } else {
            this.ext = this.gl.getExtension('ANGLE_instanced_arrays');
            this.oesDerivatives = this.gl.getExtension('OES_standard_derivatives');

            this.gl.getExtension('OES_texture_float');
            this.linearFloat = this.gl.getExtension('OES_texture_float_linear');
            this.gl.getExtension('OES_texture_half_float');
            this.gl.getExtension('OES_texture_half_float_linear');

            if (!this.ext) {
                throw new Error('WebGL 1 found but ANGLE_instanced_arrays not supported');
            }
        }

        this.initShaders();
        this.initBuffers();
        this.initFramebuffers();

        this.resize();
        this.resizeHandler = () => this.resize();
        window.addEventListener('resize', this.resizeHandler);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const finalDpr = isMobile ? Math.min(dpr, 1.5) : dpr;
        this.dpr = finalDpr;

        const displayWidth = Math.floor(this.canvas.clientWidth * finalDpr);
        const displayHeight = Math.floor(this.canvas.clientHeight * finalDpr);

        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            this.initFramebuffers();
        }
    }

    createShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source.trim());
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
            console.log('Shader Source Start:', source.trim().substring(0, 100));
            this.gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    createProgram(vsSource, fsSource) {
        const vs = this.createShader(this.gl.VERTEX_SHADER, vsSource);
        const fs = this.createShader(this.gl.FRAGMENT_SHADER, fsSource);
        if (!vs || !fs) return null;

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program link error:', this.gl.getProgramInfoLog(program));
            return null;
        }
        return program;
    }

    getShaderHeader(isFrag) {
        if (this.isWebGL2) {
            return `#version 300 es
precision highp float;
`;
        }
        let header = '';
        if (isFrag && this.oesDerivatives) {
            header += `#extension GL_OES_standard_derivatives : enable
`;
        }
        header += `precision highp float;
`;
        return header;
    }

    initFramebuffers() {
        if (this.fboScene) this.gl.deleteFramebuffer(this.fboScene);
        if (this.texScene) this.gl.deleteTexture(this.texScene);
        if (this.fboBlur1) this.gl.deleteFramebuffer(this.fboBlur1);
        if (this.texBlur1) this.gl.deleteTexture(this.texBlur1);
        if (this.fboBlur2) this.gl.deleteFramebuffer(this.fboBlur2);
        if (this.texBlur2) this.gl.deleteTexture(this.texBlur2);

        const w = this.canvas.width;
        const h = this.canvas.height;

        let internalFormat, format, type;
        let useFloat = this.isWebGL2 && this.colorBufferFloat && this.linearFloat;

        if (useFloat) {
            internalFormat = this.gl.RGBA16F;
            format = this.gl.RGBA;
            type = this.gl.HALF_FLOAT;
        } else {
            internalFormat = this.gl.RGBA;
            format = this.gl.RGBA;
            type = this.gl.UNSIGNED_BYTE;
        }

        const createFBO = (width, height) => {
            const tex = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);

            const fbo = this.gl.createFramebuffer();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, tex, 0);

            return { fbo, tex };
        };

        let scene = createFBO(w, h);
        if (useFloat && this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER) !== this.gl.FRAMEBUFFER_COMPLETE) {
            console.warn("FBO with RGBA16F not complete, falling back to RGBA/UNSIGNED_BYTE");
            this.gl.deleteFramebuffer(scene.fbo);
            this.gl.deleteTexture(scene.tex);

            internalFormat = this.gl.RGBA;
            format = this.gl.RGBA;
            type = this.gl.UNSIGNED_BYTE;
            scene = createFBO(w, h);
        }

        this.fboScene = scene.fbo;
        this.texScene = scene.tex;
        const bloomW = Math.max(1, Math.floor(w * 0.5));
        const bloomH = Math.max(1, Math.floor(h * 0.5));

        const blur1 = createFBO(bloomW, bloomH);
        this.fboBlur1 = blur1.fbo;
        this.texBlur1 = blur1.tex;

        const blur2 = createFBO(bloomW, bloomH);
        this.fboBlur2 = blur2.fbo;
        this.texBlur2 = blur2.tex;

        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    }

    initShaders() {
        const _in = this.isWebGL2 ? 'in' : 'attribute';
        const _out = this.isWebGL2 ? 'out' : 'varying';
        const _varIn = this.isWebGL2 ? 'in' : 'varying';
        const _fragColor = this.isWebGL2 ? 'outColor' : 'gl_FragColor';
        const _fragOutDecl = this.isWebGL2 ? 'out vec4 outColor;' : '';
        const _texture = this.isWebGL2 ? 'texture' : 'texture2D';

        // --- BACKGROUND GRID SHADER ---

        const gridVs = `
${this.getShaderHeader(false)}
${_in} vec2 a_position;
${_out} vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

        const useDerivs = (this.isWebGL2 || this.oesDerivatives);

        const gridFs = `
${this.getShaderHeader(true)}

uniform float u_time;
uniform vec2 u_resolution;
uniform float u_tearStrength;
uniform float u_tearPhase;
uniform vec2 u_hoverPos;
uniform float u_hoverStrength;

${_varIn} vec2 v_uv;
${_fragOutDecl}

float hash(float n) { return fract(sin(n) * 43758.5453123); }
float noise(float p) {
    float fl = floor(p);
    float fc = fract(p);
    return mix(hash(fl), hash(fl + 1.0), fc);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    
    // Tearing
    float tearOffset = 0.0;
    if (u_tearStrength > 0.001) {
        float yBlock = floor(gl_FragCoord.y / 8.0);
        float n = noise(yBlock * 0.1 + u_tearPhase * 5.0 + u_time * 2.0);
        if (n > 0.6) {
            tearOffset = (n - 0.5) * 0.2 * u_tearStrength;
        }
    }

    vec2 px = gl_FragCoord.xy;
    px.x += tearOffset * u_resolution.x;

    float gridSize = 120.0;
    if (u_resolution.x < 800.0) gridSize = 60.0;

    vec2 p = px;
    float t = u_time * 2.0;
    float distort = sin(t * 0.5 + (p.x + p.y) * 0.002) * 16.0;
    vec2 gridPos = p + vec2(distort);

    // Grid lines
    #if ${useDerivs ? 1 : 0}
        vec2 grid = abs(fract(gridPos / gridSize - 0.5) - 0.5) / fwidth(gridPos / gridSize);
    #else
        vec2 grid = vec2(1.0); // Fallback
    #endif
    
    // Glow
    float gx = mod(gridPos.x, gridSize);
    float gy = mod(gridPos.y, gridSize);
    float blur = 1.5;
    float alphaV = smoothstep(1.5 + blur, 1.5 - blur, gx);
    float alphaH = smoothstep(1.5 + blur, 1.5 - blur, gy);
    
    vec3 color = vec3(0.0);
    // boost intensity for HDR
    color += vec3(0.0, 3.0, 4.0) * alphaV; // Super Bright Cyan
    color += vec3(3.0, 3.0, 3.0) * alphaH; // Super Bright White
    
    // Bloom / Spot Light
    if (u_hoverStrength > 0.0) {
        float dist = distance(gl_FragCoord.xy, u_hoverPos);
        float radius = min(u_resolution.x, u_resolution.y) * 0.25 + 120.0;
        float bloom = smoothstep(radius, 0.0, dist);
        color += vec3(0.4, 0.4, 3.0) * bloom * u_hoverStrength * 0.5;
    }

    ${_fragColor} = vec4(color, 1.0);
}
`;

        this.gridProgram = this.createProgram(gridVs, gridFs);

        // --- TRIANGLE SHADER ---

        const triVs = `
${this.getShaderHeader(false)}

${_in} vec2 a_modelPos;
${_in} vec2 a_offset;
${_in} float a_rotation;
${_in} vec4 a_color;
${_in} float a_scale;

uniform vec2 u_resolution;

${_out} vec4 v_color;

void main() {
    v_color = a_color;
    
    float c = cos(a_rotation);
    float s = sin(a_rotation);
    mat2 rot = mat2(c, -s, s, c);
    
    vec2 pos = rot * (a_modelPos * a_scale);
    pos += a_offset;
    
    vec2 clip = (pos / u_resolution) * 2.0 - 1.0;
    
    gl_Position = vec4(clip.x, 1.0 - (pos.y / u_resolution.y) * 2.0, 0.0, 1.0);
}
`;

        const triFs = `
${this.getShaderHeader(true)}
${_varIn} vec4 v_color;
${_fragOutDecl}

void main() {
    // HDR Boost
    vec4 col = v_color;
    col.rgb *= 4.0; // Stronger boost for intense neon
    ${_fragColor} = col;
}
`;

        this.triProgram = this.createProgram(triVs, triFs);

        // --- BLUR SHADER ---

        const blurVs = `
${this.getShaderHeader(false)}
${_in} vec2 a_position;
${_out} vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

        const blurFs = `
${this.getShaderHeader(true)}
uniform sampler2D u_tex;
uniform vec2 u_resolution; // Resolution of TEXTURE (not screen)
uniform vec2 u_dir; // (1,0) or (0,1)

${_varIn} vec2 v_uv;
${_fragOutDecl}

void main() {
    vec2 off = u_dir / u_resolution;
    vec4 sum = vec4(0.0);
    
    // Gaussian-ish weights (5 tap)
    sum += ${_texture}(u_tex, v_uv - off * 2.0) * 0.06136;
    sum += ${_texture}(u_tex, v_uv - off * 1.0) * 0.24477;
    sum += ${_texture}(u_tex, v_uv)             * 0.38774;
    sum += ${_texture}(u_tex, v_uv + off * 1.0) * 0.24477;
    sum += ${_texture}(u_tex, v_uv + off * 2.0) * 0.06136;
    
    ${_fragColor} = sum;
}
`;
        this.blurProgram = this.createProgram(blurVs, blurFs);

        // --- COMPOSITE SHADER ---

        const compVs = `
${this.getShaderHeader(false)}
${_in} vec2 a_position;
${_out} vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

        const compFs = `
${this.getShaderHeader(true)}
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloomStrength;
uniform float u_aberration;
uniform float u_time;

${_varIn} vec2 v_uv;
${_fragOutDecl}

// Simple dither noise
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

void main() {
    // Chromatic Aberration
    vec2 dist = v_uv - 0.5;
    vec2 offset = dist * u_aberration * 0.02;
    
    float r = ${_texture}(u_scene, v_uv - offset).r;
    float g = ${_texture}(u_scene, v_uv).g;
    float b = ${_texture}(u_scene, v_uv + offset).b;
    vec3 scene = vec3(r, g, b);
    
    // Bloom
    vec3 bloom = ${_texture}(u_bloom, v_uv).rgb;
    
    // Additive mix
    vec3 color = scene + bloom * u_bloomStrength;
    
    // Tone Mapping
    color = color / (color + vec3(1.0));
    
    // Vignette
    float len = length(dist);
    float vignette = smoothstep(0.8, 0.2, len * 0.8);
    color *= vignette;
    
    // Film Grain
    float grain = random(v_uv + u_time) * 0.05 - 0.025;
    color += grain;
    
    // Gamma Correct
    color = pow(color, vec3(1.0/2.2));
    
    ${_fragColor} = vec4(color, 1.0);
}
`;
        this.compProgram = this.createProgram(compVs, compFs);

        if (!this.gridProgram || !this.triProgram || !this.blurProgram || !this.compProgram) {
            throw new Error('Failed to compile or link WebGL shader programs');
        }
    }

    initBuffers() {
        this.quadBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 1, -1, -1, 1,
            -1, 1, 1, -1, 1, 1,
        ]), this.gl.STATIC_DRAW);

        const triVerts = new Float32Array([
            0.0, -0.5,
            0.5, 0.5,
            -0.5, 0.5,
            0.0, -0.5
        ]);

        this.triGeoBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.triGeoBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, triVerts, this.gl.STATIC_DRAW);

        this.instanceBuffer = this.gl.createBuffer();
    }

    render(time, state) {
        if (!this.gl || !this.fboScene) return;

        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fboScene);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0.005, 0.005, 0.008, 1.0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        // -- Render Grid --
        if (this.gridProgram) {
            this.gl.useProgram(this.gridProgram);
            this.gl.uniform2f(this.gl.getUniformLocation(this.gridProgram, 'u_resolution'), this.canvas.width, this.canvas.height);
            this.gl.uniform1f(this.gl.getUniformLocation(this.gridProgram, 'u_time'), time);
            this.gl.uniform1f(this.gl.getUniformLocation(this.gridProgram, 'u_tearStrength'), state.tearStrength || 0);
            this.gl.uniform1f(this.gl.getUniformLocation(this.gridProgram, 'u_tearPhase'), state.tearPhase || 0);

            if (state.hoverPos) {
                this.gl.uniform2f(this.gl.getUniformLocation(this.gridProgram, 'u_hoverPos'),
                    state.hoverPos.x * (this.dpr || 1),
                    (window.innerHeight - state.hoverPos.y) * (this.dpr || 1));
                this.gl.uniform1f(this.gl.getUniformLocation(this.gridProgram, 'u_hoverStrength'), 1.0);
            } else {
                this.gl.uniform1f(this.gl.getUniformLocation(this.gridProgram, 'u_hoverStrength'), 0.0);
            }

            const aPos = this.gl.getAttribLocation(this.gridProgram, 'a_position');
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
            this.gl.enableVertexAttribArray(aPos);
            this.gl.vertexAttribPointer(aPos, 2, this.gl.FLOAT, false, 0, 0);
            this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
        }

        // -- Render Triangles --
        if (state.triangles && state.triangles.length > 0 && this.triProgram) {
            this.gl.useProgram(this.triProgram);
            this.gl.uniform2f(this.gl.getUniformLocation(this.triProgram, 'u_resolution'), this.canvas.width, this.canvas.height);

            const allTriangles = state.triangles.concat(state.centerTriangles || []);
            const instanceCount = allTriangles.length;
            const floatsPerInstance = 8;
            const data = new Float32Array(instanceCount * floatsPerInstance);

            let ptr = 0;
            const parseColor = (str) => {
                if (!str) return [1, 1, 1, 1];
                if (str.startsWith('#')) {
                    let hex = str.slice(1);
                    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
                    const bigInt = parseInt(hex, 16);
                    return [((bigInt >> 16) & 255) / 255, ((bigInt >> 8) & 255) / 255, (bigInt & 255) / 255, 1];
                }
                return [1, 1, 1, 1];
            };

            for (let i = 0; i < instanceCount; i++) {
                const tri = allTriangles[i];
                data[ptr++] = tri.x * (this.dpr || 1);
                data[ptr++] = tri.y * (this.dpr || 1);
                data[ptr++] = tri.angle + Math.sin(time * 1.0 + i) * 0.1;
                data[ptr++] = (tri.isCenter) ? 64.0 : 48.0;
                const col = parseColor(tri.color);
                data[ptr++] = col[0]; data[ptr++] = col[1]; data[ptr++] = col[2]; data[ptr++] = 1.0;
            }

            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.instanceBuffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.DYNAMIC_DRAW);

            const aModelPos = this.gl.getAttribLocation(this.triProgram, 'a_modelPos');
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.triGeoBuffer);
            this.gl.enableVertexAttribArray(aModelPos);
            this.gl.vertexAttribPointer(aModelPos, 2, this.gl.FLOAT, false, 0, 0);

            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.instanceBuffer);
            const stride = floatsPerInstance * 4;

            const aOffset = this.gl.getAttribLocation(this.triProgram, 'a_offset');
            this.gl.enableVertexAttribArray(aOffset);
            this.gl.vertexAttribPointer(aOffset, 2, this.gl.FLOAT, false, stride, 0);

            const aRot = this.gl.getAttribLocation(this.triProgram, 'a_rotation');
            this.gl.enableVertexAttribArray(aRot);
            this.gl.vertexAttribPointer(aRot, 1, this.gl.FLOAT, false, stride, 8);

            const aScale = this.gl.getAttribLocation(this.triProgram, 'a_scale');
            this.gl.enableVertexAttribArray(aScale);
            this.gl.vertexAttribPointer(aScale, 1, this.gl.FLOAT, false, stride, 12);

            const aColor = this.gl.getAttribLocation(this.triProgram, 'a_color');
            this.gl.enableVertexAttribArray(aColor);
            this.gl.vertexAttribPointer(aColor, 4, this.gl.FLOAT, false, stride, 16);

            if (this.isWebGL2) {
                this.gl.vertexAttribDivisor(aModelPos, 0);
                this.gl.vertexAttribDivisor(aOffset, 1);
                this.gl.vertexAttribDivisor(aRot, 1);
                this.gl.vertexAttribDivisor(aScale, 1);
                this.gl.vertexAttribDivisor(aColor, 1);
                this.gl.drawArraysInstanced(this.gl.LINE_LOOP, 0, 4, instanceCount);

                this.gl.vertexAttribDivisor(aOffset, 0);
                this.gl.vertexAttribDivisor(aRot, 0);
                this.gl.vertexAttribDivisor(aScale, 0);
                this.gl.vertexAttribDivisor(aColor, 0);
            } else {
                this.ext.vertexAttribDivisorANGLE(aModelPos, 0);
                this.ext.vertexAttribDivisorANGLE(aOffset, 1);
                this.ext.vertexAttribDivisorANGLE(aRot, 1);
                this.ext.vertexAttribDivisorANGLE(aScale, 1);
                this.ext.vertexAttribDivisorANGLE(aColor, 1);
                this.ext.drawArraysInstancedANGLE(this.gl.LINE_LOOP, 0, 4, instanceCount);

                this.ext.vertexAttribDivisorANGLE(aOffset, 0);
                this.ext.vertexAttribDivisorANGLE(aRot, 0);
                this.ext.vertexAttribDivisorANGLE(aScale, 0);
                this.ext.vertexAttribDivisorANGLE(aColor, 0);
            }

            this.gl.disableVertexAttribArray(aModelPos);
            this.gl.disableVertexAttribArray(aOffset);
            this.gl.disableVertexAttribArray(aRot);
            this.gl.disableVertexAttribArray(aScale);
            this.gl.disableVertexAttribArray(aColor);
        }

        // 2. Blur Passes (Post-Processing)
        if (this.blurProgram) {
            const bloomW = this.canvas.width * 0.5;
            const bloomH = this.canvas.height * 0.5;

            this.gl.useProgram(this.blurProgram);
            this.gl.uniform2f(this.gl.getUniformLocation(this.blurProgram, 'u_resolution'), bloomW, bloomH);
            const aPos = this.gl.getAttribLocation(this.blurProgram, 'a_position');
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
            this.gl.enableVertexAttribArray(aPos);
            this.gl.vertexAttribPointer(aPos, 2, this.gl.FLOAT, false, 0, 0);

            // Pass 1: Scene -> Blur1 (Horizontal)
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fboBlur1);
            this.gl.viewport(0, 0, bloomW, bloomH);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.texScene);
            this.gl.uniform1i(this.gl.getUniformLocation(this.blurProgram, 'u_tex'), 0);
            this.gl.uniform2f(this.gl.getUniformLocation(this.blurProgram, 'u_dir'), 1.0, 0.0);
            this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fboBlur2);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindTexture(this.gl.TEXTURE_2D, this.texBlur1);
            this.gl.uniform2f(this.gl.getUniformLocation(this.blurProgram, 'u_dir'), 0.0, 1.0);
            this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
        }

        // 3. Composite to Screen
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        if (this.compProgram) {
            this.gl.useProgram(this.compProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.compProgram, 'u_bloomStrength'), 3.0); // Stronger bloom

            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.texScene);
            this.gl.uniform1i(this.gl.getUniformLocation(this.compProgram, 'u_scene'), 0);

            this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.texBlur2);
            this.gl.uniform1i(this.gl.getUniformLocation(this.compProgram, 'u_bloom'), 1);

            const baseAberration = 0.5 + (state.tearStrength || 0.0) * 5.0;
            this.gl.uniform1f(this.gl.getUniformLocation(this.compProgram, 'u_aberration'), baseAberration);
            this.gl.uniform1f(this.gl.getUniformLocation(this.compProgram, 'u_time'), time);

            const aPos = this.gl.getAttribLocation(this.compProgram, 'a_position');
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
            this.gl.enableVertexAttribArray(aPos);
            this.gl.vertexAttribPointer(aPos, 2, this.gl.FLOAT, false, 0, 0);

            this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
        }
    }

    destroy() {
        if (this.resizeHandler) {
            window.removeEventListener('resize', this.resizeHandler);
        }

        if (this.gl) {
            if (this.fboScene) this.gl.deleteFramebuffer(this.fboScene);
            if (this.texScene) this.gl.deleteTexture(this.texScene);
            if (this.fboBlur1) this.gl.deleteFramebuffer(this.fboBlur1);
            if (this.texBlur1) this.gl.deleteTexture(this.texBlur1);
            if (this.fboBlur2) this.gl.deleteFramebuffer(this.fboBlur2);
            if (this.texBlur2) this.gl.deleteTexture(this.texBlur2);

            if (this.quadBuffer) this.gl.deleteBuffer(this.quadBuffer);
            if (this.triGeoBuffer) this.gl.deleteBuffer(this.triGeoBuffer);
            if (this.instanceBuffer) this.gl.deleteBuffer(this.instanceBuffer);

            if (this.gridProgram) this.gl.deleteProgram(this.gridProgram);
            if (this.triProgram) this.gl.deleteProgram(this.triProgram);
            if (this.blurProgram) this.gl.deleteProgram(this.blurProgram);
            if (this.compProgram) this.gl.deleteProgram(this.compProgram);
        }
    }
}
