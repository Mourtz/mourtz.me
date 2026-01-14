
class WebGPURenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;
        this.format = navigator.gpu ? navigator.gpu.getPreferredCanvasFormat() : 'bgra8unorm';
        
        this.pipelines = {};
        this.buffers = {};
        this.textures = {};
        
        // Default params (overwritten in init)
        this.params = { maxTile: 128, tileSize: 16, curveCount: 1024 };
        this.initialized = false;
        this.dpr = window.devicePixelRatio || 1;

        this.init().catch(err => {
            console.error("WebGPU Init Failed:", err);
        });

        this.resize();
        window.addEventListener('resize', () => { setTimeout(()=>this.resize(), 10); });
    }

    async init() {
        if (!navigator.gpu) throw new Error("No WebGPU");
        const adapter = await navigator.gpu.requestAdapter();
        if(!adapter) throw new Error("No Adapter");
        
        const limits = adapter.limits;
        const maxInvo = limits.maxComputeInvocationsPerWorkgroup || 256;
        
        // Check for mobile (using the helper from index.html or UA fallback)
        const isMobile = (window.mobilecheck && window.mobilecheck()) || /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

        if (maxInvo >= 1024) {
            // High-end GPU
            const count = isMobile ? 2048 : 8192;
            this.params = { maxTile: 1024, tileSize: 32, curveCount: count };
            this.device = await adapter.requestDevice({
                requiredLimits: { maxComputeInvocationsPerWorkgroup: maxInvo }
            });
        } else {
            // Standard/Low-end GPU
            const count = isMobile ? 1024 : 4096;
            this.params = { maxTile: 128, tileSize: 16, curveCount: count };
            this.device = await adapter.requestDevice();
        }
        
        console.log(`WebGPU Initialized: TileSize=${this.params.tileSize}, MaxTile=${this.params.maxTile}, Curves=${this.params.curveCount}`);

        this.context = this.canvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: this.format, alphaMode: 'premultiplied' });

        await this.initShaders();
        this.initBuffers();
        this.initialized = true;
    }

    resize() {
        this.dpr = window.devicePixelRatio || 1;
        const width = Math.floor(this.canvas.clientWidth * this.dpr);
        const height = Math.floor(this.canvas.clientHeight * this.dpr);
        
        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
            if(this.initialized) this.resizeAssets();
        }
    }

    resizeAssets() {
        if(this.textures.output) this.textures.output.destroy();
        this.textures.output = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        // Bloom Texture (1/4 Resolution)
        const bloomW = Math.max(1, Math.floor(this.canvas.width / 4));
        const bloomH = Math.max(1, Math.floor(this.canvas.height / 4));
        
        if(this.textures.bloom) this.textures.bloom.destroy();
        this.textures.bloom = this.device.createTexture({
            size: [bloomW, bloomH],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        const tileDimX = Math.ceil(this.canvas.width / this.params.tileSize);
        const tileDimY = Math.ceil(this.canvas.height / this.params.tileSize);
        const totalTiles = tileDimX * tileDimY;

        if(this.buffers.tileCounts) this.buffers.tileCounts.destroy();
        this.buffers.tileCounts = this.device.createBuffer({ size: totalTiles * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

        if(this.buffers.tileIndices) this.buffers.tileIndices.destroy();
        const idxSize = totalTiles * this.params.maxTile * 4;
        this.buffers.tileIndices = this.device.createBuffer({ size: idxSize, usage: GPUBufferUsage.STORAGE });

        this.updateBindGroups();
    }

    initBuffers() {
        this.buffers.gridUni = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.buffers.velloUni = this.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); 
        
        this.initCurveAssets();
        this.resizeAssets();
    }

    initCurveAssets() {
        const cData = new Float32Array(this.params.curveCount * 16);
        for(let i=0; i<this.params.curveCount; i++) {
            const o = i*16;
            cData[o+0] = Math.random(); 
            cData[o+1] = Math.random();
            cData[o+12] = i / this.params.curveCount; // Gradient ID
        }

        this.buffers.curves = this.device.createBuffer({
            size: cData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.buffers.curves, 0, cData);

        // Segment buffer
        this.segmentsPerCurve = 24;
        const totalSegs = this.params.curveCount * this.segmentsPerCurve;
        
        if(this.buffers.segments) this.buffers.segments.destroy();
        this.buffers.segments = this.device.createBuffer({
            size: totalSegs * 32, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
    }

    async initShaders() {
        const GRID_SHADER = `
            struct Uniforms {
                resolution: vec2f,
                time: f32,
                tearStrength: f32,
                tearPhase: f32,
                hoverPos: vec2f,
                hoverStrength: f32
            };
            @group(0) @binding(0) var<uniform> global: Uniforms;

            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @location(0) uv: vec2f
            };

            @vertex
            fn vs(@builtin(vertex_index) i: u32) -> VertexOutput {
                var pos = array<vec2f, 6>(
                    vec2f(-1, -1), vec2f( 1, -1), vec2f(-1,  1),
                    vec2f(-1,  1), vec2f( 1, -1), vec2f( 1,  1)
                );
                var out: VertexOutput;
                out.pos = vec4f(pos[i], 0.0, 1.0);
                out.uv = pos[i] * 0.5 + 0.5;
                return out;
            }

            @fragment
            fn fs(in: VertexOutput) -> @location(0) vec4f {
                let time = global.time; 
                let col1 = vec3f(0.005, 0.0, 0.02); 
                let col2 = vec3f(0.02, 0.01, 0.05);
                let bg = mix(col1, col2, in.uv.y + sin(time*0.1)*0.1);
                return vec4f(bg, 1.0);
            }
        `;

        const COMMON = `
            struct Curve { p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, col: vec4f, param: vec4f };
            struct Segment { p0: vec2f, p1: vec2f, color: vec4f };
            struct Uniforms {
                screen: vec2f,
                cam_pos: vec2f,
                zoom: f32,
                time: f32,
                tile_dim_x: u32, tile_dim_y: u32,
                tile_size: u32, max_per_tile: u32
            };
            const SEG_PER_CURVE = 24u;
        `;

        const SHADER_SIM = `
            ${COMMON}
            @group(0) @binding(0) var<storage, read_write> curves: array<Curve>;
            @group(0) @binding(1) var<storage, read_write> segments: array<Segment>;
            @group(0) @binding(2) var<uniform> global: Uniforms;

            fn world_to_screen(p: vec2f, u: Uniforms) -> vec2f {
                 let center = u.screen * 0.5;
                 return center + (p - u.cam_pos) * u.zoom;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&curves)) { return; }
                
                var c = curves[i];
                let t = global.time * 0.15;
                let seed = c.param.x * 200.0;
                
                let min_dim = min(global.screen.x, global.screen.y);
                let max_dim = max(global.screen.x, global.screen.y);
                
                // Sweep across the screen
                let r1 = min_dim * 0.2 + sin(t + seed) * (min_dim * 0.1);
                let r2 = max_dim * 0.9 + cos(t * 0.8 + seed) * (min_dim * 0.3);
                
                let a1 = t * 0.5 + c.param.x * 6.28;
                let a2 = t * 1.0 + c.param.x * 12.56;
                
                let p0 = vec2f(cos(a1)*r1, sin(a1)*r1);
                let p3 = vec2f(cos(a2)*r2, sin(a2)*r2);
                
                // Handles
                let h_len = min_dim * 0.8;
                let p1 = p0 + vec2f(cos(a1+1.5), sin(a1+1.5)) * h_len;
                let p2 = p3 + vec2f(cos(a2-1.5), sin(a2-1.5)) * h_len;
                
                // Colors - "Aurora Borealis" Palette
                let hue = c.param.x + t * 0.1;
                let r = 0.5 + 0.5*cos(6.28*hue);
                let g = 0.5 + 0.5*cos(6.28*hue + 2.0);
                let b = 0.5 + 0.5*cos(6.28*hue + 4.0);
                // FILL EFFECT: Lower alpha to create "washes" of color
                var col = vec4f(r, g, b, 0.03); 
                
                let seg_step = 1.0 / f32(SEG_PER_CURVE);
                var tt = 0.0;
                var prev = world_to_screen(p0, global);
                let base_idx = i * SEG_PER_CURVE;
                
                for(var j=0u; j<SEG_PER_CURVE; j++) {
                    tt += seg_step;
                    let mt = 1.0 - tt;
                    let pt = mt*mt*mt*p0 + 3.0*mt*mt*tt*p1 + 3.0*mt*tt*tt*p2 + tt*tt*tt*p3;
                    let curr = world_to_screen(pt, global);
                    segments[base_idx + j] = Segment(prev, curr, col);
                    prev = curr;
                }
            }
        `;

        const SHADER_BIN = `
            ${COMMON}
            @group(0) @binding(0) var<storage, read> segments: array<Segment>;
            @group(0) @binding(1) var<storage, read_write> tile_counts: array<atomic<u32>>;
            @group(0) @binding(2) var<storage, read_write> tile_indices: array<u32>;
            @group(0) @binding(3) var<uniform> global: Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&segments)) { return; }
                let s = segments[i];

                let ts = f32(global.tile_size);
                // Larger bounding box for "Glow" effect
                let min_v = min(s.p0, s.p1) - 4.0;
                let max_v = max(s.p0, s.p1) + 4.0;
                let t_min = vec2i(floor(min_v / ts));
                let t_max = vec2i(floor(max_v / ts));

                let start_x = max(0, t_min.x);
                let end_x = min(i32(global.tile_dim_x) - 1, t_max.x);
                let start_y = max(0, t_min.y);
                let end_y = min(i32(global.tile_dim_y) - 1, t_max.y);

                if (start_x > end_x || start_y > end_y) { return; }

                for (var y = start_y; y <= end_y; y++) {
                    for (var x = start_x; x <= end_x; x++) {
                        let tile_idx = u32(y) * global.tile_dim_x + u32(x);
                        let cnt = atomicAdd(&tile_counts[tile_idx], 1u);
                        if (cnt < global.max_per_tile) {
                            tile_indices[tile_idx * global.max_per_tile + cnt] = i;
                        }
                    }
                }
            }
        `;

        // RASTERIZER: Modified for "Filled" effect
        const SHADER_RAS = `
            ${COMMON}
            @group(0) @binding(0) var<storage, read> segments: array<Segment>;
            @group(0) @binding(1) var<storage, read> tile_counts: array<u32>;
            @group(0) @binding(2) var<storage, read> tile_indices: array<u32>;
            @group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;
            @group(0) @binding(4) var<uniform> global: Uniforms;

            fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
                let pa = p - a; let ba = b - a;
                let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                return length(pa - ba * h);
            }

            @compute @workgroup_size(${this.params.tileSize}, ${this.params.tileSize})
            fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u) {
                let tx = wid.x; let ty = wid.y;
                if (tx >= global.tile_dim_x || ty >= global.tile_dim_y) { return; }
                let tile_idx = ty * global.tile_dim_x + tx;

                let count = min(tile_counts[tile_idx], global.max_per_tile);
                var col = vec4f(0.0);
                let pix = vec2f(f32(gid.x)+0.5, f32(gid.y)+0.5);
                let base = tile_idx * global.max_per_tile;

                for(var i=0u; i<count; i++) {
                    let s = segments[tile_indices[base + i]];
                    
                    // OPTIMIZATION: Fast AABB Check
                    // Only compute distance if pixel is within the segment's bounding box (+ radius)
                    let min_p = min(s.p0, s.p1) - 4.5; // 4.0 radius + 0.5 margin
                    let max_p = max(s.p0, s.p1) + 4.5;
                    
                    if (pix.x >= min_p.x && pix.x <= max_p.x && pix.y >= min_p.y && pix.y <= max_p.y) {
                        let d = sdSegment(pix, s.p0, s.p1);
                        let a = 1.0 - smoothstep(0.0, 4.0, d);
                        if(a > 0.0) {
                            let sa = s.color.a * a;
                            let src = s.color.rgb * sa;
                            col = vec4f(col.rgb + src, col.a + sa); 
                            
                            // OPTIMIZATION: Early Exit on Saturation
                            // If pixel is fully white, adding more won't change display
                            if(col.r >= 1.0 && col.g >= 1.0 && col.b >= 1.0) {
                                break;
                            }
                        }
                    }
                }
                textureStore(output_tex, vec2i(gid.xy), col);
            }
        `;

        const SHADER_CLEAR = `
            @group(0) @binding(0) var output_tex: texture_storage_2d<rgba16float, write>;
            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) id: vec3u) { textureStore(output_tex, vec2i(id.xy), vec4f(0.0)); }
        `;

        // BLOOM SHADER: Downsample + Threshold + Blur
        const SHADER_BLOOM = `
            @group(0) @binding(0) var input_tex: texture_2d<f32>;
            @group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
            
            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let size = textureDimensions(output_tex);
                if (id.x >= size.x || id.y >= size.y) { return; }
                
                // Map to input coordinates (4x upscale)
                let base_uv = vec2i(id.xy) * 4;
                
                var col = vec3f(0.0);
                var count = 0.0;
                
                // Simple Box Sample (4x4 area from source)
                for(var y=0; y<4; y++) {
                    for(var x=0; x<4; x++) {
                        let c = textureLoad(input_tex, base_uv + vec2i(x,y), 0).rgb;
                        // Threshold check (High threshold = only hotspots glow)
                        let bright = max(c - vec3f(0.6), vec3f(0.0)); 
                        col += bright;
                        count += 1.0;
                    }
                }
                col = col / count; 
                col *= 1.5; // Subtle Bloom Intensity
                
                textureStore(output_tex, vec2i(id.xy), vec4f(col, 1.0));
            }
        `;

        // Update BLIT to composite with Tone Mapping
        const BLIT_SHADER = `
            @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
                var p=array<vec2f,6>( vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1), vec2f(-1,1),vec2f(1,-1),vec2f(1,1) ); 
                return vec4f(p[i],0.,1.);
            }
            @group(0) @binding(0) var t_scene: texture_2d<f32>;
            @group(0) @binding(1) var t_bloom: texture_2d<f32>;
            @group(0) @binding(2) var s_lin: sampler;
            
            @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f { 
                let uv = p.xy / vec2f(textureDimensions(t_scene));
                let scene = textureLoad(t_scene, vec2i(p.xy), 0);
                let bloom = textureSampleLevel(t_bloom, s_lin, uv, 0.0);
                
                // Comp: Sharp Scene + Soft Glow
                let hdr = scene.rgb + bloom.rgb * 0.6;
                
                // Classic Reinhard Tone Mapping
                let mapped = hdr / (hdr + vec3f(1.0));
                
                return vec4f(mapped, 1.0);
            }
        `;

        const mSim = this.device.createShaderModule({ code: SHADER_SIM });
        this.pipelines.sim = this.device.createComputePipeline({ layout: 'auto', compute: { module: mSim, entryPoint: 'main' } });
        const mBin = this.device.createShaderModule({ code: SHADER_BIN });
        this.pipelines.bin = this.device.createComputePipeline({ layout: 'auto', compute: { module: mBin, entryPoint: 'main' } });
        const mRas = this.device.createShaderModule({ code: SHADER_RAS });
        this.pipelines.ras = this.device.createComputePipeline({ layout: 'auto', compute: { module: mRas, entryPoint: 'main' } });
        const mClr = this.device.createShaderModule({ code: SHADER_CLEAR });
        this.pipelines.clear = this.device.createComputePipeline({ layout: 'auto', compute: { module: mClr, entryPoint: 'main' } });
        const mBloom = this.device.createShaderModule({ code: SHADER_BLOOM });
        this.pipelines.bloom = this.device.createComputePipeline({ layout: 'auto', compute: { module: mBloom, entryPoint: 'main' } });
        const mGrid = this.device.createShaderModule({ code: GRID_SHADER });
        this.pipelines.grid = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: mGrid, entryPoint: 'vs' },
            fragment: { module: mGrid, entryPoint: 'fs', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-list' }
        });
        this.pipelines.blit = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: this.device.createShaderModule({ code: BLIT_SHADER }), entryPoint: 'vs' },
            fragment: { 
                module: this.device.createShaderModule({ code: BLIT_SHADER }), entryPoint: 'fs', 
                targets: [{ format: this.format, blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] 
            },
            primitive: { topology: 'triangle-list' }
        });
    }

    updateBindGroups() {
        if(!this.textures.output) return;

        this.pipelines.bgGrid = this.device.createBindGroup({
            layout: this.pipelines.grid.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.buffers.gridUni } }]
        });
        
        // Sampler for Bloom Upscale
        const linearSampler = this.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
        
        this.pipelines.bgBlit = this.device.createBindGroup({
            layout: this.pipelines.blit.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.textures.output.createView() },
                { binding: 1, resource: this.textures.bloom.createView() },
                { binding: 2, resource: linearSampler }
            ]
        });
        this.pipelines.bgClear = this.device.createBindGroup({
            layout: this.pipelines.clear.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: this.textures.output.createView() }]
        });
        
        this.pipelines.bgBloom = this.device.createBindGroup({
            layout: this.pipelines.bloom.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.textures.output.createView() },
                { binding: 1, resource: this.textures.bloom.createView() }
            ]
        });

        this.pipelines.bgSim = this.device.createBindGroup({
            layout: this.pipelines.sim.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.curves } },
                { binding: 1, resource: { buffer: this.buffers.segments } },
                { binding: 2, resource: { buffer: this.buffers.velloUni } }
            ]
        });
        this.pipelines.bgBin = this.device.createBindGroup({
            layout: this.pipelines.bin.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.segments } },
                { binding: 1, resource: { buffer: this.buffers.tileCounts } },
                { binding: 2, resource: { buffer: this.buffers.tileIndices } },
                { binding: 3, resource: { buffer: this.buffers.velloUni } }
            ]
        });
        this.pipelines.bgRas = this.device.createBindGroup({
            layout: this.pipelines.ras.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.segments } },
                { binding: 1, resource: { buffer: this.buffers.tileCounts } },
                { binding: 2, resource: { buffer: this.buffers.tileIndices } },
                { binding: 3, resource: this.textures.output.createView() },
                { binding: 4, resource: { buffer: this.buffers.velloUni } }
            ]
        });
    }

    render(time, state) {
        if(!this.initialized || !this.device) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        const gUni = new Float32Array(12);
        gUni[0] = width; gUni[1] = height;
        gUni[2] = time;
        this.device.queue.writeBuffer(this.buffers.gridUni, 0, gUni);

        const vUni = new ArrayBuffer(80);
        const vf = new Float32Array(vUni); const vu = new Uint32Array(vUni);
        vf[0] = width; vf[1] = height;
        vf[2] = Math.sin(time*0.1) * 200.0; vf[3] = Math.cos(time*0.13) * 100.0; // Slow pan
        vf[4] = 0.8 + Math.pow(Math.sin(time*0.2)*0.5 + 0.5, 2.0) * 3.0; // Zoom 0.8x to 3.8x
        vf[5] = time;
        vu[6] = Math.ceil(width/this.params.tileSize);
        vu[7] = Math.ceil(height/this.params.tileSize);
        vu[8] = this.params.tileSize;
        vu[9] = this.params.maxTile;
        this.device.queue.writeBuffer(this.buffers.velloUni, 0, vUni);

        const enc = this.device.createCommandEncoder();

        // 1. Render Grid Background (writes to Screen)
        const passGrid = enc.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(), // Screen (cleared)
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear', storeOp: 'store'
            }]
        });
        passGrid.setPipeline(this.pipelines.grid);
        passGrid.setBindGroup(0, this.pipelines.bgGrid);
        passGrid.draw(6); 
        passGrid.end();
        
        // 1.5 Clear Output Texture (for scene rendering)
        const gridW = Math.ceil(width/16), gridH = Math.ceil(height/16);
        const passClr = enc.beginComputePass();
        passClr.setPipeline(this.pipelines.clear);
        passClr.setBindGroup(0, this.pipelines.bgClear);
        passClr.dispatchWorkgroups(gridW, gridH);
        passClr.end();

        // 2. Simulate Curves
        const groups = Math.ceil(this.params.curveCount / 64);
        const passSim = enc.beginComputePass();
        passSim.setPipeline(this.pipelines.sim);
        passSim.setBindGroup(0, this.pipelines.bgSim);
        passSim.dispatchWorkgroups(groups);
        passSim.end();

        // 3. Bin Segments into Tiles
        const totalSegs = this.params.curveCount * this.segmentsPerCurve;
        enc.clearBuffer(this.buffers.tileCounts);
        const passBin = enc.beginComputePass();
        passBin.setPipeline(this.pipelines.bin);
        passBin.setBindGroup(0, this.pipelines.bgBin);
        passBin.dispatchWorkgroups(Math.ceil(totalSegs/64)); 
        passBin.end();

        // 4. Rasterize Tiles to Output Texture
        const passRas = enc.beginComputePass();
        passRas.setPipeline(this.pipelines.ras);
        passRas.setBindGroup(0, this.pipelines.bgRas);
        passRas.dispatchWorkgroups(vu[6], vu[7]);
        passRas.end();
        
        // 5. BLOOM Pass (Output -> BloomTex)
        const bloomW = Math.max(1, Math.floor(width / 4));
        const bloomH = Math.max(1, Math.floor(height / 4));
        const passBloom = enc.beginComputePass();
        passBloom.setPipeline(this.pipelines.bloom);
        passBloom.setBindGroup(0, this.pipelines.bgBloom);
        passBloom.dispatchWorkgroups(Math.ceil(bloomW/8), Math.ceil(bloomH/8));
        passBloom.end();

        // 6. Blit (Output + Bloom -> Screen)
        const passBlit = enc.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'load', storeOp: 'store' // Load Grid background
            }]
        });
        passBlit.setPipeline(this.pipelines.blit);
        passBlit.setBindGroup(0, this.pipelines.bgBlit);
        passBlit.draw(6);
        passBlit.end();

        this.device.queue.submit([enc.finish()]);

        // // Performance Monitoring
        // if(!this.frameCount) this.frameCount = 0;
        // this.frameCount++;
        // if(!this.lastTime) this.lastTime = performance.now();
        
        // if(this.frameCount % 60 === 0) {
        //     const now = performance.now();
        //     const fps = 1000 / ((now - this.lastTime) / 60);
        //     console.log(`FPS: ${fps.toFixed(1)}`);
        //     this.lastTime = now;
        // }
    }
}
window.WebGPURenderer = WebGPURenderer;

