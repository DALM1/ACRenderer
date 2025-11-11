#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <float.h>
#import <math.h>
#import "../include/MMRenderer.h"

// Forward declarations for math helpers used inside methods
static void mat4_perspective(float* m, float fovRadians, float aspect, float znear, float zfar);
static void mat4_identity(float* m);
static void mat4_mul(float* out,const float* a,const float* b);
static void mat4_translate(float* m,float x,float y,float z);
static void mat4_rotateY(float* m,float a);
static void mat4_scale(float* m,float x,float y,float z);
static void mat4_lookAt(float* m, float eyeX,float eyeY,float eyeZ, float centerX,float centerY,float centerZ, float upX,float upY,float upZ);

@implementation MMRenderer {
    BOOL _running;
    CAMetalLayer *_layer;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLRenderPipelineState> _pipeline;
    id<MTLRenderPipelineState> _skyPipeline;
    id<MTLDepthStencilState> _skyDepthState;
    id<MTLBuffer> _skyUniformBuffer;
    id<MTLBuffer> _skyIndexBuffer;
    NSUInteger _skyIndexCount;
    id<MTLRenderPipelineState> _groundPipeline;
    id<MTLBuffer> _groundVertexBuffer;
    id<MTLBuffer> _groundDomeVertexBuffer;
    id<MTLBuffer> _groundDomeIndexBuffer;
    NSUInteger _groundDomeIndexCount;
    id<MTLBuffer> _groundUniformBuffer;
    id<MTLBuffer> _vertexBuffer;
    id<MTLBuffer> _indexBuffer;
    NSUInteger _indexCount;
    NSUInteger _vertexCount;
    id<MTLTexture> _texture;
    id<MTLBuffer> _uniformBuffer; // MVP + extras
    id<MTLBuffer> _skyVertexBuffer;
    float _modelCenter[3];
    float _modelScale;
    // Model bounds for ground placement
    float _modelMinY;
    float _modelMaxY;
    float _modelExtentY;
    id<MTLDepthStencilState> _depthState;
    id<MTLTexture> _depthTexture;
    CGSize _depthSize;
    id<MTLTexture> _fallbackTexture;

    // Orbit camera state
    float _camYaw;    // radians
    float _camPitch;  // radians
    float _camDist;   // units
    float _camTarget[3];

    // Storm/lighting state
    float _time;
    float _exposure;
    float _stormIntensity;

    // Ground collision constraints
    float _groundY;
    float _groundClearance;
}

static NSString* kMSLTextured = @
    "using namespace metal;\n"
    "struct VSIn { float3 position [[attribute(0)]]; float2 uv [[attribute(1)]]; };\n"
    "struct VSOut { float4 position [[position]]; float2 uv; };\n"
    "struct Uniforms { float4x4 mvp; float time; float exposure; float storm; };\n"
    "vertex VSOut vertex_main(VSIn in [[stage_in]],\n"
    "                        constant Uniforms& U [[ buffer(1) ]]) {\n"
    "  VSOut o;\n"
    "  o.position = U.mvp * float4(in.position, 1.0);\n"
    "  o.uv = in.uv;\n"
    "  return o;\n"
    "}\n"
    "fragment float4 fragment_main(VSOut in [[stage_in]],\n"
    "                             texture2d<float> tex [[ texture(0) ]],\n"
    "                             sampler s [[ sampler(0) ]],\n"
    "                             constant Uniforms& U [[ buffer(1) ]]) {\n"
    "  float4 c = tex.sample(s, in.uv);\n"
    "  // Simple lightning pulse: occasional flashes brighten the scene\n"
    "  float pulse = smoothstep(0.98, 1.0, sin(U.time * 20.0));\n"
    "  float exp = max(0.5, U.exposure + pulse * U.storm);\n"
    "  c.rgb *= exp;\n"
    "  return c;\n"
    "}\n";

// Procedural stormy sky (full-screen quad rendered in world behind the model)
static NSString* kMSLSky = @
    "using namespace metal;\n"
    "struct SkyVSIn { float3 position [[attribute(0)]]; float2 uv [[attribute(1)]]; };\n"
    "struct SkyVSOut { float4 position [[position]]; float3 dir; float2 uv; };\n"
    "struct SkyUniforms { float4x4 mvp; float time; float exposure; float storm; };\n"
    "inline float hash31(float3 p){ p = fract(p*0.3183099 + float3(0.1,0.2,0.3)); p += dot(p, p.yzx + 19.19); return fract((p.x + p.y) * p.z); }\n"
    "inline float noise3(float3 p){ float3 i = floor(p); float3 f = fract(p); float n = mix(mix(mix(hash31(i+float3(0,0,0)), hash31(i+float3(1,0,0)), f.x), mix(hash31(i+float3(0,1,0)), hash31(i+float3(1,1,0)), f.x), f.y), mix(mix(hash31(i+float3(0,0,1)), hash31(i+float3(1,0,1)), f.x), mix(hash31(i+float3(0,1,1)), hash31(i+float3(1,1,1)), f.x), f.y), f.z); return n; }\n"
    "inline float fbm3(float3 p){ float a=0.0, w=0.5; float3 q=p; for(int i=0;i<5;i++){ a+=w*noise3(q); q*=2.02; w*=0.55; } return a; }\n"
    "vertex SkyVSOut sky_vertex(SkyVSIn in [[stage_in]], constant SkyUniforms& U [[ buffer(1) ]]) {\n"
    "  SkyVSOut o; o.position = U.mvp * float4(in.position, 1.0); o.dir = normalize(in.position); o.uv = in.uv; return o;\n"
    "}\n"
    "fragment float4 sky_fragment(SkyVSOut in [[stage_in]], constant SkyUniforms& U [[ buffer(1) ]]) {\n"
    "  float3 d = in.dir; float t = U.time;\n"
    "  float3 p = d*0.8 + float3(0.0, t*0.03, 0.0);\n"
    "  float c = fbm3(p) * 0.9 + fbm3(p*1.9)*0.5;\n"
    "  float grad = smoothstep(-0.1, 0.6, d.y);\n"
    "  float skyBase = mix(0.05, 0.20, grad);\n"
    "  float clouds = smoothstep(0.35, 0.75, c);\n"
    "  float brightness = skyBase + clouds*0.55;\n"
    "  float flash = U.storm>0.2 ? smoothstep(0.0, 0.5, sin(t*20.0)*0.5+0.5) * U.exposure*0.4 : 0.0;\n"
    "  float3 col = float3(0.15,0.18,0.22)*brightness + float3(0.8,0.85,1.0)*flash;\n"
    "  return float4(clamp(col,0.0,1.0), 1.0);\n"
    "}\n";

// Infinite-looking checkerboard floor (large plane with world-anchored checker)
static NSString* kMSLGround = @
    "using namespace metal;\n"
    "struct GroundVSIn { float3 position [[attribute(0)]]; float2 uv [[attribute(1)]]; };\n"
    "struct GroundVSOut { float4 position [[position]]; float3 local; };\n"
    "struct GroundUniforms { float4x4 mvp; float tile; float exposure; float storm; float offsetX; float offsetZ; float fogStart; float fogEnd; float3 horizonColor; };\n"
    "vertex GroundVSOut ground_vertex(GroundVSIn in [[stage_in]], constant GroundUniforms& U [[ buffer(1) ]]) {\n"
    "  GroundVSOut o; o.position = U.mvp * float4(in.position, 1.0); o.local = in.position; return o;\n"
    "}\n"
    "fragment float4 ground_fragment(GroundVSOut in [[stage_in]], constant GroundUniforms& U [[ buffer(1) ]]) {\n"
    "  float2 wp = (in.local.xz + float2(U.offsetX, U.offsetZ)) / max(U.tile, 1e-3);\n"
    "  // Checker via fract/XOR (robust, avoids int/fmod issues)\n"
    "  float2 f = fract(wp);\n"
    "  float a = step(0.5, f.x);\n"
    "  float b = step(0.5, f.y);\n"
    "  float m = abs(a - b);\n"
    "  float3 colA = float3(0.0,0.0,0.0);\n"
    "  float3 colB = float3(1.0,1.0,1.0);\n"
    "  float3 base = mix(colA, colB, m);\n"
    "  base *= 1.0;\n"
    "  // Keep fog disabled in test (values set very far in CPU)\n"
    "  float ndcY = in.position.y / max(in.position.w, 1e-4);\n"
    "  float h = smoothstep(-0.02, 0.08, ndcY);\n"
    "  float fog = h;\n"
    "  float3 col = mix(base, U.horizonColor, fog);\n"
    "  return float4(col, 1.0);\n"
    "}\n";

typedef struct { float x,y,z; } Vec3;
typedef struct { float u,v; } Vec2;
typedef struct { Vec3 pos; Vec2 uv; } VertexPU;

// OBJ parser minimal: v, vt, f with triangulated faces. Builds unique vertices per (v/vt) pair.
static BOOL parseOBJ(NSString* path, NSMutableData* vertexData, NSMutableData* indexData, NSError** error) {
    NSError* readErr = nil;
    NSString* content = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&readErr];
    if (!content) { if (error) *error = readErr; return NO; }
    NSMutableArray<NSValue*>* positions = [NSMutableArray array];
    NSMutableArray<NSValue*>* uvs = [NSMutableArray array];
    NSMutableDictionary<NSString*, NSNumber*>* uniq = [NSMutableDictionary dictionary];
    NSMutableArray<NSNumber*>* indices = [NSMutableArray array];

    NSArray<NSString*>* lines = [content componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    for (NSString* line in lines) {
        if (line.length == 0) continue;
        if ([line hasPrefix:@"v "]) {
            NSArray* parts = [line componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            if (parts.count >= 4) {
                Vec3 v = { [parts[1] floatValue], [parts[2] floatValue], [parts[3] floatValue] };
                [positions addObject:[NSValue valueWithBytes:&v objCType:@encode(Vec3)]];
            }
        } else if ([line hasPrefix:@"vt "]) {
            NSArray* parts = [line componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            if (parts.count >= 3) {
                Vec2 t = { [parts[1] floatValue], [parts[2] floatValue] };
                [uvs addObject:[NSValue valueWithBytes:&t objCType:@encode(Vec2)]];
            }
        } else if ([line hasPrefix:@"f "]) {
            NSString* facesStr = [line substringFromIndex:2];
            NSArray* faceParts = [facesStr componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            if (faceParts.count < 3) continue;
            int vids[faceParts.count];
            for (NSUInteger i=0;i<faceParts.count;i++) {
                NSString* vert = faceParts[i];
                if (vert.length==0) { vids[i] = -1; continue; }
                NSArray* comps = [vert componentsSeparatedByString:@"/"];
                NSInteger vi = comps.count>0 ? [comps[0] integerValue] : 0;
                NSInteger vti = (comps.count>1 && [comps[1] length]>0) ? [comps[1] integerValue] : 0;
                NSInteger vi0 = vi - 1;
                NSInteger vt0 = vti - 1;
                NSString* key = [NSString stringWithFormat:@"%ld/%ld", (long)vi0, (long)vt0];
                NSNumber* idx = uniq[key];
                if (!idx) {
                    Vec3 vp; [positions[vi0] getValue:&vp];
                    Vec2 vt = {0,0}; if (vt0>=0 && vt0<(NSInteger)uvs.count) { [uvs[vt0] getValue:&vt]; }
                    VertexPU vpu = { vp, vt };
                    [vertexData appendBytes:&vpu length:sizeof(VertexPU)];
                    idx = @( (int)(vertexData.length/sizeof(VertexPU) - 1) );
                    uniq[key] = idx;
                }
                vids[i] = idx.intValue;
            }
            for (NSUInteger i=1;i+1<faceParts.count;i++) {
                [indices addObject:@(vids[0])];
                [indices addObject:@(vids[i])];
                [indices addObject:@(vids[i+1])];
            }
        }
    }
    for (NSNumber* n in indices) {
        uint32_t idx = (uint32_t)n.unsignedIntValue;
        [indexData appendBytes:&idx length:sizeof(uint32_t)];
    }
    return YES;
}

// Minimal GLB (glTF 2.0 binary) loader supporting:
// - Single buffer with JSON and BIN chunks
// - One mesh primitive with POSITION (float32x3), TEXCOORD_0 (float32x2), indices (uint32)
// - Embedded JPEG texture via images[0] with bufferView
static BOOL loadGLB(NSString* path,
                    NSMutableData* vertexData,
                    NSMutableData* indexData,
                    id<MTLDevice> device,
                    id<MTLTexture>* outTexture,
                    float outCenter[3],
                    float* outScale,
                    NSError** error) {
    NSData* file = [NSData dataWithContentsOfFile:path options:0 error:error];
    if (!file) return NO;
    const uint8_t* bytes = (const uint8_t*)file.bytes;
    NSUInteger len = file.length;
    if (len < 20) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:1 userInfo:@{NSLocalizedDescriptionKey:@"GLB too small"}]; return NO; }
    uint32_t magic = *(const uint32_t*)(bytes+0);
    uint32_t version = *(const uint32_t*)(bytes+4);
    uint32_t totalLen = *(const uint32_t*)(bytes+8);
    if (magic != 0x46546C67 || version != 2 || totalLen > len) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:2 userInfo:@{NSLocalizedDescriptionKey:@"Invalid GLB header"}]; return NO; }
    NSUInteger cursor = 12;
    NSData* jsonChunk = nil; NSData* binChunk = nil;
    while (cursor + 8 <= len) {
        uint32_t chunkLen = *(const uint32_t*)(bytes+cursor); cursor += 4;
        uint32_t chunkType = *(const uint32_t*)(bytes+cursor); cursor += 4;
        if (cursor + chunkLen > len) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:3 userInfo:@{NSLocalizedDescriptionKey:@"GLB chunk overflow"}]; return NO; }
        NSData* chunkData = [file subdataWithRange:NSMakeRange(cursor, chunkLen)];
        cursor += chunkLen;
        if (chunkType == 0x4E4F534A) { jsonChunk = chunkData; }
        else if (chunkType == 0x004E4942) { binChunk = chunkData; }
    }
    if (!jsonChunk || !binChunk) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:4 userInfo:@{NSLocalizedDescriptionKey:@"Missing JSON or BIN chunk"}]; return NO; }

    NSError* jerr = nil;
    NSDictionary* gltf = [NSJSONSerialization JSONObjectWithData:jsonChunk options:0 error:&jerr];
    if (!gltf) { if (error) *error=jerr; return NO; }

    NSArray* bufferViews = gltf[@"bufferViews"] ?: @[];
    NSArray* accessors = gltf[@"accessors"] ?: @[];
    NSArray* meshes = gltf[@"meshes"] ?: @[];
    NSArray* images = gltf[@"images"] ?: @[];

    if (meshes.count == 0) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:5 userInfo:@{NSLocalizedDescriptionKey:@"No meshes in GLB"}]; return NO; }
    NSDictionary* mesh0 = meshes[0];
    NSArray* prims = mesh0[@"primitives"] ?: @[];
    if (prims.count == 0) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:6 userInfo:@{NSLocalizedDescriptionKey:@"No primitives in mesh"}]; return NO; }
    NSDictionary* prim = prims[0];
    NSDictionary* attrs = prim[@"attributes"] ?: @{};
    NSNumber* posAccessorIndex = attrs[@"POSITION"];
    NSNumber* uvAccessorIndex = attrs[@"TEXCOORD_0"];
    NSNumber* idxAccessorIndex = prim[@"indices"];
    if (!posAccessorIndex || !idxAccessorIndex) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:7 userInfo:@{NSLocalizedDescriptionKey:@"Missing POSITION or indices"}]; return NO; }

    NSDictionary* posAcc = accessors[posAccessorIndex.integerValue];
    NSDictionary* uvAcc = uvAccessorIndex ? accessors[uvAccessorIndex.integerValue] : nil;
    NSDictionary* idxAcc = accessors[idxAccessorIndex.integerValue];

    NSUInteger posCount = [posAcc[@"count"] unsignedIntegerValue];
    NSUInteger uvCount = uvAcc ? [uvAcc[@"count"] unsignedIntegerValue] : 0;
    NSUInteger idxCount = [idxAcc[@"count"] unsignedIntegerValue];

    NSDictionary* posBV = bufferViews[[posAcc[@"bufferView"] integerValue]];
    NSDictionary* uvBV = uvAcc ? bufferViews[[uvAcc[@"bufferView"] integerValue]] : nil;
    NSDictionary* idxBV = bufferViews[[idxAcc[@"bufferView"] integerValue]];

    NSUInteger posOffset = [posBV[@"byteOffset"] unsignedIntegerValue] + [posAcc[@"byteOffset"] unsignedIntegerValue];
    NSUInteger uvOffset = uvBV ? [uvBV[@"byteOffset"] unsignedIntegerValue] + [uvAcc[@"byteOffset"] unsignedIntegerValue] : 0;
    NSUInteger idxOffset = [idxBV[@"byteOffset"] unsignedIntegerValue] + [idxAcc[@"byteOffset"] unsignedIntegerValue];

    const uint8_t* bin = (const uint8_t*)binChunk.bytes;
    if (posOffset + posCount* sizeof(float)*3 > binChunk.length) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:8 userInfo:@{NSLocalizedDescriptionKey:@"POSITION overflow"}]; return NO; }
    if (uvAcc && (uvOffset + uvCount* sizeof(float)*2 > binChunk.length)) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:9 userInfo:@{NSLocalizedDescriptionKey:@"UV overflow"}]; return NO; }
    // indices are uint16/uint32 but here expect uint32 (componentType=5125)
    BOOL idxIs32 = ([idxAcc[@"componentType"] integerValue] == 5125);
    BOOL idxIs16 = ([idxAcc[@"componentType"] integerValue] == 5123);
    if (!idxIs32 && !idxIs16) { if(error) *error=[NSError errorWithDomain:@"MMRenderer" code:10 userInfo:@{NSLocalizedDescriptionKey:@"Unsupported index type"}]; return NO; }

    // Build vertices
    vertexData.length = 0; indexData.length = 0;
    const float* posPtr = (const float*)(bin + posOffset);
    const float* uvPtr = uvAcc ? (const float*)(bin + uvOffset) : NULL;
    for (NSUInteger i=0;i<posCount;i++) {
        VertexPU v;
        v.pos.x = posPtr[i*3+0]; v.pos.y = posPtr[i*3+1]; v.pos.z = posPtr[i*3+2];
        if (uvPtr && i<uvCount) { v.uv.u = uvPtr[i*2+0]; v.uv.v = uvPtr[i*2+1]; }
        else { v.uv.u = 0; v.uv.v = 0; }
        [vertexData appendBytes:&v length:sizeof(VertexPU)];
    }

    // Build indices
    if (idxIs32) {
        const uint32_t* idPtr = (const uint32_t*)(bin + idxOffset);
        [indexData appendBytes:idPtr length:idxCount*sizeof(uint32_t)];
    } else {
        const uint16_t* idPtr = (const uint16_t*)(bin + idxOffset);
        for (NSUInteger i=0;i<idxCount;i++) { uint32_t v = idPtr[i]; [indexData appendBytes:&v length:sizeof(uint32_t)]; }
    }

    // Compute center/scale (use accessor min/max if available)
    NSArray* posMin = posAcc[@"min"]; NSArray* posMax = posAcc[@"max"];
    Vec3 minv = { FLT_MAX, FLT_MAX, FLT_MAX };
    Vec3 maxv = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    if (posMin && posMax && posMin.count==3 && posMax.count==3) {
        minv.x = [posMin[0] floatValue]; minv.y = [posMin[1] floatValue]; minv.z = [posMin[2] floatValue];
        maxv.x = [posMax[0] floatValue]; maxv.y = [posMax[1] floatValue]; maxv.z = [posMax[2] floatValue];
    } else {
        const VertexPU* verts = (const VertexPU*)vertexData.bytes; NSUInteger vcount = vertexData.length/sizeof(VertexPU);
        for (NSUInteger i=0;i<vcount;i++) {
            Vec3 p = verts[i].pos;
            if (p.x < minv.x) minv.x = p.x; if (p.y < minv.y) minv.y = p.y; if (p.z < minv.z) minv.z = p.z;
            if (p.x > maxv.x) maxv.x = p.x; if (p.y > maxv.y) maxv.y = p.y; if (p.z > maxv.z) maxv.z = p.z;
        }
    }
    outCenter[0] = (minv.x+maxv.x)*0.5f;
    outCenter[1] = (minv.y+maxv.y)*0.5f;
    outCenter[2] = (minv.z+maxv.z)*0.5f;
    float ex = maxv.x - minv.x; float ey = maxv.y - minv.y; float ez = maxv.z - minv.z;
    float maxExtent = fmaxf(ex, fmaxf(ey, ez));
    *outScale = (maxExtent > 0.0f) ? (2.0f / maxExtent) : 1.0f;

    // Texture from embedded image if present
    *outTexture = nil;
    if (images.count > 0) {
        NSDictionary* img0 = images[0];
        NSNumber* bvIdx = img0[@"bufferView"];
        if (bvIdx) {
            NSDictionary* bv = bufferViews[bvIdx.integerValue];
            NSUInteger imgOffset = [bv[@"byteOffset"] unsignedIntegerValue];
            NSUInteger imgLen = [bv[@"byteLength"] unsignedIntegerValue];
            if (imgOffset + imgLen <= binChunk.length) {
                NSData* imgData = [binChunk subdataWithRange:NSMakeRange(imgOffset, imgLen)];
                MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:device];
                NSDictionary* opts = @{ MTKTextureLoaderOptionSRGB:@(YES) };
                NSError* terr = nil;
                id<MTLTexture> tex = [loader newTextureWithData:imgData options:opts error:&terr];
                if (!tex) { NSLog(@"Embedded texture load failed: %@", terr); }
                else { *outTexture = tex; }
            }
        }
    }
    return YES;
}

- (void)setDrawableLayer:(CAMetalLayer *)layer {
    _layer = layer;
    if (!_device) {
        _device = MTLCreateSystemDefaultDevice();
    }
    _layer.device = _device;
    _layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    _layer.framebufferOnly = YES;

    _queue = [_device newCommandQueue];

    NSError *err = nil;
    NSString* mslSrc = [[[kMSLTextured stringByAppendingString:kMSLSky] stringByAppendingString:kMSLGround] copy];
    id<MTLLibrary> lib = [_device newLibraryWithSource:mslSrc options:nil error:&err];
    if (!lib) { NSLog(@"Metal library compilation failed: %@", err); return; }
    id<MTLFunction> vs = [lib newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fs = [lib newFunctionWithName:@"fragment_main"];

    MTLVertexDescriptor *vd = [[MTLVertexDescriptor alloc] init];
    vd.attributes[0].format = MTLVertexFormatFloat3;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.attributes[1].format = MTLVertexFormatFloat2;
    vd.attributes[1].offset = sizeof(float)*3;
    vd.attributes[1].bufferIndex = 0;
    vd.layouts[0].stride = sizeof(VertexPU);
    vd.layouts[0].stepRate = 1;
    vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.vertexDescriptor = vd;
    desc.colorAttachments[0].pixelFormat = _layer.pixelFormat;
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    _pipeline = [_device newRenderPipelineStateWithDescriptor:desc error:&err];
    if (!_pipeline) { NSLog(@"Pipeline creation failed: %@", err); return; }

    // Sky pipeline (opaque dome)
    id<MTLFunction> skyVS = [lib newFunctionWithName:@"sky_vertex"];
    id<MTLFunction> skyFS = [lib newFunctionWithName:@"sky_fragment"];
    MTLVertexDescriptor *skyVD = [[MTLVertexDescriptor alloc] init];
    skyVD.attributes[0].format = MTLVertexFormatFloat3; skyVD.attributes[0].offset = 0; skyVD.attributes[0].bufferIndex = 0;
    skyVD.attributes[1].format = MTLVertexFormatFloat2; skyVD.attributes[1].offset = sizeof(float)*3; skyVD.attributes[1].bufferIndex = 0;
    skyVD.layouts[0].stride = sizeof(VertexPU);
    skyVD.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex; skyVD.layouts[0].stepRate = 1;
    MTLRenderPipelineDescriptor *skyDesc = [[MTLRenderPipelineDescriptor alloc] init];
    skyDesc.vertexFunction = skyVS; skyDesc.fragmentFunction = skyFS; skyDesc.vertexDescriptor = skyVD;
    skyDesc.colorAttachments[0].pixelFormat = _layer.pixelFormat; skyDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    _skyPipeline = [_device newRenderPipelineStateWithDescriptor:skyDesc error:&err];
    if (!_skyPipeline) { NSLog(@"Sky pipeline creation failed: %@", err); }

    // Ground pipeline (checker plane)
    id<MTLFunction> groundVS = [lib newFunctionWithName:@"ground_vertex"];
    id<MTLFunction> groundFS = [lib newFunctionWithName:@"ground_fragment"];
    MTLVertexDescriptor *groundVD = [[MTLVertexDescriptor alloc] init];
    groundVD.attributes[0].format = MTLVertexFormatFloat3; groundVD.attributes[0].offset = 0; groundVD.attributes[0].bufferIndex = 0;
    groundVD.attributes[1].format = MTLVertexFormatFloat2; groundVD.attributes[1].offset = sizeof(float)*3; groundVD.attributes[1].bufferIndex = 0;
    groundVD.layouts[0].stride = sizeof(VertexPU);
    groundVD.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex; groundVD.layouts[0].stepRate = 1;
    MTLRenderPipelineDescriptor *groundDesc = [[MTLRenderPipelineDescriptor alloc] init];
    groundDesc.vertexFunction = groundVS; groundDesc.fragmentFunction = groundFS; groundDesc.vertexDescriptor = groundVD;
    groundDesc.colorAttachments[0].pixelFormat = _layer.pixelFormat; groundDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    _groundPipeline = [_device newRenderPipelineStateWithDescriptor:groundDesc error:&err];
    if (!_groundPipeline) { NSLog(@"Ground pipeline creation failed: %@", err); }

    // Larger uniform buffer to carry MVP + time/exposure/storm (aligned)
    _uniformBuffer = [_device newBufferWithLength:sizeof(float)*32 options:MTLResourceStorageModeShared];
    _skyUniformBuffer = [_device newBufferWithLength:sizeof(float)*32 options:MTLResourceStorageModeShared];
    _groundUniformBuffer = [_device newBufferWithLength:sizeof(float)*32 options:MTLResourceStorageModeShared];

    // Depth state for sky (no depth write, compare always)
    MTLDepthStencilDescriptor *skyDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    skyDepthDesc.depthCompareFunction = MTLCompareFunctionAlways;
    skyDepthDesc.depthWriteEnabled = NO;
    _skyDepthState = [_device newDepthStencilStateWithDescriptor:skyDepthDesc];

    // Build a hemisphere mesh around the camera (radius ~ 30)
    const int stacks = 16, slices = 32;
    const float radius = 30.0f;
    const float phiStart = -0.15f; // extend slightly below horizon to avoid band
    NSMutableData* domeVerts = [NSMutableData dataWithLength:sizeof(VertexPU)*(stacks+1)*(slices+1)];
    NSMutableData* domeIdx = [NSMutableData data];
    VertexPU* vptr = (VertexPU*)domeVerts.mutableBytes;
    int vidx = 0;
    for(int i=0;i<=stacks;i++){
        float phi = phiStart + (float)i/(float)stacks * ((float)M_PI_2 - phiStart); // slightly under horizon .. pi/2
        float y = sinf(phi);
        float r = cosf(phi);
        for(int j=0;j<=slices;j++){
            float theta = (float)j/(float)slices * (float)(2.0*M_PI);
            float x = r * cosf(theta);
            float z = r * sinf(theta);
            vptr[vidx].pos = (Vec3){ x*radius, y*radius, z*radius };
            vptr[vidx].uv  = (Vec2){ (float)j/(float)slices, (float)i/(float)stacks };
            vidx++;
        }
    }
    for(int i=0;i<stacks;i++){
        for(int j=0;j<slices;j++){
            uint32_t a = i*(slices+1)+j;
            uint32_t b = a + slices + 1;
            uint32_t c = a + 1;
            uint32_t d = b + 1;
            uint32_t tri[6] = { a,b,c, c,b,d };
            [domeIdx appendBytes:tri length:sizeof(tri)];
        }
    }
    _skyVertexBuffer = [_device newBufferWithBytes:domeVerts.bytes length:domeVerts.length options:MTLResourceStorageModeShared];
    _skyIndexBuffer = [_device newBufferWithBytes:domeIdx.bytes length:domeIdx.length options:MTLResourceStorageModeShared];
    _skyIndexCount = domeIdx.length/sizeof(uint32_t);

    // Large ground plane centered at origin (y≈-1)
    typedef struct { float x,y,z,u,v; } GroundV;
    float gy = -1.05f; float half = 200.0f;
    GroundV gquad[4] = {
        { -half, gy, -half, -half, -half },
        {  half, gy, -half,  half, -half },
        { -half, gy,  half, -half,  half },
        {  half, gy,  half,  half,  half }
    };
    _groundVertexBuffer = [_device newBufferWithBytes:gquad length:sizeof(gquad) options:MTLResourceStorageModeShared];
    _groundY = gy; _groundClearance = 0.2f;

    // Build a curved ground dome to close horizon gap
    const int gStacks = 10, gSlices = 32; const float gRadius = 30.0f;
    const float gPhiStart = -0.6f; const float gPhiEnd = 0.05f; // overlap slightly into near-horizon
    NSMutableData* gDomeVerts = [NSMutableData dataWithLength:sizeof(VertexPU)*(gStacks+1)*(gSlices+1)];
    NSMutableData* gDomeIdx = [NSMutableData data];
    VertexPU* gv = (VertexPU*)gDomeVerts.mutableBytes; int gvidx=0;
    for(int i=0;i<=gStacks;i++){
        float phi = gPhiStart + (float)i/(float)gStacks * (gPhiEnd - gPhiStart);
        float y = sinf(phi);
        float r = cosf(phi);
        for(int j=0;j<=gSlices;j++){
            float theta = (float)j/(float)gSlices * (float)(2.0*M_PI);
            float x = r * cosf(theta);
            float z = r * sinf(theta);
            gv[gvidx].pos = (Vec3){ x*gRadius, y*gRadius, z*gRadius };
            gv[gvidx].uv  = (Vec2){ (float)j/(float)gSlices, (float)i/(float)gStacks };
            gvidx++;
        }
    }
    for(int i=0;i<gStacks;i++){
        for(int j=0;j<gSlices;j++){
            uint32_t a = i*(gSlices+1)+j;
            uint32_t b = a + gSlices + 1;
            uint32_t c = a + 1;
            uint32_t d = b + 1;
            uint32_t tri[6] = { a,b,c, c,b,d };
            [gDomeIdx appendBytes:tri length:sizeof(tri)];
        }
    }
    _groundDomeVertexBuffer = [_device newBufferWithBytes:gDomeVerts.bytes length:gDomeVerts.length options:MTLResourceStorageModeShared];
    _groundDomeIndexBuffer = [_device newBufferWithBytes:gDomeIdx.bytes length:gDomeIdx.length options:MTLResourceStorageModeShared];
    _groundDomeIndexCount = gDomeIdx.length/sizeof(uint32_t);

    MTLDepthStencilDescriptor *dsDesc = [[MTLDepthStencilDescriptor alloc] init];
    dsDesc.depthCompareFunction = MTLCompareFunctionLess;
    dsDesc.depthWriteEnabled = YES;
    _depthState = [_device newDepthStencilStateWithDescriptor:dsDesc];

    // Create a 1x1 white fallback texture for rendering when no material is bound
    MTLTextureDescriptor *wdesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                    width:1 height:1 mipmapped:NO];
    wdesc.usage = MTLTextureUsageShaderRead;
    wdesc.storageMode = MTLStorageModeManaged;
    _fallbackTexture = [_device newTextureWithDescriptor:wdesc];
    uint8_t white[4] = {255,255,255,255};
    MTLRegion r = { {0,0,0}, {1,1,1} };
    [_fallbackTexture replaceRegion:r mipmapLevel:0 withBytes:white bytesPerRow:4];
}

 - (void)start {
     _running = YES;
     // Only apply a generic default orbit if no model is loaded yet
     if (_vertexCount == 0) {
         _camYaw = 0.0f; _camPitch = -0.3f; _camDist = 5.0f;
         _camTarget[0] = 0.0f; _camTarget[1] = 0.0f; _camTarget[2] = 0.0f;
         // Sane defaults when no model is loaded
         _modelCenter[0] = 0.0f; _modelCenter[1] = 0.0f; _modelCenter[2] = 0.0f;
         _modelScale = 5.0f;
     }
     // Storm state defaults
     _time = 0.0f; _exposure = 1.0f; _stormIntensity = 0.8f;
     NSLog(@"MMRenderer start");
  }

- (void)renderFrame {
    if (!_running || !_layer) { return; }
    // Advance time (approximate; tied to frame rate)
    _time += 1.0f/60.0f;
    [self updateUniforms];
    id<CAMetalDrawable> drawable = [_layer nextDrawable];
    if (!drawable) { return; }
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.15, 1.0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;

    CGSize ds = _layer.drawableSize;
    if (!_depthTexture || !CGSizeEqualToSize(_depthSize, ds)) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                    width:(NSUInteger)ds.width
                                                                                   height:(NSUInteger)ds.height
                                                                                mipmapped:NO];
        td.storageMode = MTLStorageModePrivate;
        td.usage = MTLTextureUsageRenderTarget;
        _depthTexture = [_device newTextureWithDescriptor:td];
        _depthSize = ds;
    }
    rp.depthAttachment.texture = _depthTexture;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.storeAction = MTLStoreActionDontCare;
    rp.depthAttachment.clearDepth = 1.0;

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    // 1) Draw stormy sky dome (infinite depth): view translation removed in uniforms
    if (_skyPipeline && _skyVertexBuffer) {
        [enc setRenderPipelineState:_skyPipeline];
        [enc setDepthStencilState:_skyDepthState];
        [enc setCullMode:MTLCullModeNone];
        [enc setFrontFacingWinding:MTLWindingCounterClockwise];
        [enc setTriangleFillMode:MTLTriangleFillModeFill];
        [enc setViewport:(MTLViewport){0,0, ds.width, ds.height, 0.0, 1.0}];
        [enc setVertexBuffer:_skyVertexBuffer offset:0 atIndex:0];
        [enc setVertexBuffer:_skyUniformBuffer offset:0 atIndex:1];
        [enc setFragmentBuffer:_skyUniformBuffer offset:0 atIndex:1];
        if (_skyIndexBuffer && _skyIndexCount>0) {
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:_skyIndexCount indexType:MTLIndexTypeUInt32 indexBuffer:_skyIndexBuffer indexBufferOffset:0];
        } else {
            // fallback: draw vertex array if indices missing
            NSUInteger vcount = _skyVertexBuffer.length/sizeof(VertexPU);
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:vcount];
        }
    }
    // 1.5) Draw ground checker plane
    if (_groundPipeline && _groundVertexBuffer) {
        [enc setRenderPipelineState:_groundPipeline];
        [enc setDepthStencilState:_depthState];
        [enc setCullMode:MTLCullModeNone];
        [enc setFrontFacingWinding:MTLWindingCounterClockwise];
        [enc setTriangleFillMode:MTLTriangleFillModeFill];
        [enc setViewport:(MTLViewport){0,0, ds.width, ds.height, 0.0, 1.0}];
        [enc setVertexBuffer:_groundVertexBuffer offset:0 atIndex:0];
        [enc setVertexBuffer:_groundUniformBuffer offset:0 atIndex:1];
        [enc setFragmentBuffer:_groundUniformBuffer offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    }
    // 1.6) Draw curved ground dome to ensure seamless horizon closure
    if (_groundPipeline && _groundDomeVertexBuffer && _groundDomeIndexBuffer && _groundDomeIndexCount>0) {
        [enc setRenderPipelineState:_groundPipeline];
        [enc setDepthStencilState:_depthState];
        [enc setCullMode:MTLCullModeNone];
        [enc setFrontFacingWinding:MTLWindingCounterClockwise];
        [enc setTriangleFillMode:MTLTriangleFillModeFill];
        [enc setViewport:(MTLViewport){0,0, ds.width, ds.height, 0.0, 1.0}];
        [enc setVertexBuffer:_groundDomeVertexBuffer offset:0 atIndex:0];
        [enc setVertexBuffer:_groundUniformBuffer offset:0 atIndex:1];
        [enc setFragmentBuffer:_groundUniformBuffer offset:0 atIndex:1];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:_groundDomeIndexCount indexType:MTLIndexTypeUInt32 indexBuffer:_groundDomeIndexBuffer indexBufferOffset:0];
    }
    // 2) Draw model
    [enc setRenderPipelineState:_pipeline];
    [enc setDepthStencilState:_depthState];
    [enc setCullMode:MTLCullModeNone];
    [enc setFrontFacingWinding:MTLWindingCounterClockwise];
    [enc setTriangleFillMode:MTLTriangleFillModeFill];
    [enc setViewport:(MTLViewport){0,0, ds.width, ds.height, 0.0, 1.0}];
    [enc setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
    [enc setVertexBuffer:_uniformBuffer offset:0 atIndex:1];
    [enc setFragmentBuffer:_uniformBuffer offset:0 atIndex:1];
    if (_texture) {
        static id<MTLSamplerState> sstate;
        if (!sstate) {
            MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
            sd.minFilter = MTLSamplerMinMagFilterLinear;
            sd.magFilter = MTLSamplerMinMagFilterLinear;
            sd.sAddressMode = MTLSamplerAddressModeRepeat;
            sd.tAddressMode = MTLSamplerAddressModeRepeat;
            sstate = [_device newSamplerStateWithDescriptor:sd];
        }
        [enc setFragmentTexture:_texture atIndex:0];
        [enc setFragmentSamplerState:sstate atIndex:0];
    } else if (_fallbackTexture) {
        static id<MTLSamplerState> sstate2;
        if (!sstate2) {
            MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
            sd.minFilter = MTLSamplerMinMagFilterNearest;
            sd.magFilter = MTLSamplerMinMagFilterNearest;
            sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
            sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
            sstate2 = [_device newSamplerStateWithDescriptor:sd];
        }
        [enc setFragmentTexture:_fallbackTexture atIndex:0];
        [enc setFragmentSamplerState:sstate2 atIndex:0];
    }
    if (_indexBuffer && _indexCount>0) {
        // Debug: log counts once
        static BOOL logged = NO; if (!logged) { NSLog(@"Draw indexed: vtx=%lu idx=%lu", (unsigned long)_vertexCount, (unsigned long)_indexCount); logged = YES; }
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:_indexCount indexType:MTLIndexTypeUInt32 indexBuffer:_indexBuffer indexBufferOffset:0];
        // When a model is loaded, do NOT draw the fallback triangle.
    } else {
        if (_vertexCount>0 && _vertexBuffer) {
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:_vertexCount];
        } else {
            // Render a visible fallback triangle when no model is loaded
            typedef struct { float x,y,z,u,v; } FallbackVertex;
            FallbackVertex tri[3] = {
                { -0.8f, -0.8f, 0.0f, 0.0f, 0.0f },
                {  0.8f, -0.8f, 0.0f, 1.0f, 0.0f },
                {  0.0f,  0.8f, 0.0f, 0.5f, 1.0f }
            };
            // Bind the temporary vertices directly
            [enc setVertexBytes:tri length:sizeof(tri) atIndex:0];
            static BOOL fbLogged = NO;
            if (!fbLogged) {
                NSLog(@"Drawing fallback triangle (vtx=%lu idx=%lu vbuf=%@ ibuf=%@)",
                      (unsigned long)_vertexCount,
                      (unsigned long)_indexCount,
                      _vertexBuffer?@"YES":@"NO",
                      _indexBuffer?@"YES":@"NO");
                fbLogged = YES;
            }
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
    }
    [enc endEncoding];
    [cb presentDrawable:drawable];
    [cb commit];
}

- (BOOL)loadOBJAtPath:(NSString *)objPath texturePath:(NSString *)texturePath error:(NSError **)error {
    NSMutableData* vdata = [NSMutableData data];
    NSMutableData* idata = [NSMutableData data];
    NSError* perr = nil;
    if (!parseOBJ(objPath, vdata, idata, &perr)) { if (error) *error = perr; return NO; }

    _vertexCount = vdata.length / sizeof(VertexPU);
    Vec3 minv = { FLT_MAX, FLT_MAX, FLT_MAX };
    Vec3 maxv = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    const VertexPU* verts = (const VertexPU*)vdata.bytes;
    for (NSUInteger i=0;i<_vertexCount;i++) {
        Vec3 p = verts[i].pos;
        if (p.x < minv.x) minv.x = p.x; if (p.y < minv.y) minv.y = p.y; if (p.z < minv.z) minv.z = p.z;
        if (p.x > maxv.x) maxv.x = p.x; if (p.y > maxv.y) maxv.y = p.y; if (p.z > maxv.z) maxv.z = p.z;
    }
    _modelCenter[0] = (minv.x+maxv.x)*0.5f;
    _modelCenter[1] = (minv.y+maxv.y)*0.5f;
    _modelCenter[2] = (minv.z+maxv.z)*0.5f;
    _modelMinY = minv.y; _modelMaxY = maxv.y; _modelExtentY = maxv.y - minv.y;
    float ex = maxv.x - minv.x; float ey = maxv.y - minv.y; float ez = maxv.z - minv.z;
    float maxExtent = fmaxf(ex, fmaxf(ey, ez));
    _modelScale = (maxExtent > 0.0f) ? (2.0f / maxExtent) : 1.0f;

    _vertexBuffer = [_device newBufferWithBytes:vdata.bytes length:vdata.length options:MTLResourceStorageModeShared];
    _indexBuffer = idata.length>0 ? [_device newBufferWithBytes:idata.bytes length:idata.length options:MTLResourceStorageModeShared] : nil;
    _indexCount = idata.length/sizeof(uint32_t);

    if (texturePath) {
        MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:_device];
        NSDictionary* opts = @{ MTKTextureLoaderOptionSRGB:@(YES) };
        NSError* terr = nil;
        NSURL* url = [NSURL fileURLWithPath:texturePath];
        _texture = [loader newTextureWithContentsOfURL:url options:opts error:&terr];
        if (!_texture) { NSLog(@"Texture load failed: %@", terr); }
    }

    NSLog(@"Loaded OBJ: vertices=%lu, indices=%lu, center=(%.2f,%.2f,%.2f), scale=%.3f",
          (unsigned long)_vertexCount,
          (unsigned long)_indexCount,
          _modelCenter[0], _modelCenter[1], _modelCenter[2], _modelScale);
    return YES;
}

- (BOOL)loadGLBAtPath:(NSString *)glbPath error:(NSError **)error {
    NSMutableData* vdata = [NSMutableData data];
    NSMutableData* idata = [NSMutableData data];
    id<MTLTexture> tex = nil;
    float center[3]; float scale = 1.0f;
    NSError* lerr = nil;
    if (!loadGLB(glbPath, vdata, idata, _device, &tex, center, &scale, &lerr)) { if (error) *error = lerr; return NO; }

    _vertexCount = vdata.length / sizeof(VertexPU);
    _modelCenter[0] = center[0]; _modelCenter[1] = center[1]; _modelCenter[2] = center[2];
    _modelScale = scale;
    _vertexBuffer = [_device newBufferWithBytes:vdata.bytes length:vdata.length options:MTLResourceStorageModeShared];
    _indexBuffer = idata.length>0 ? [_device newBufferWithBytes:idata.bytes length:idata.length options:MTLResourceStorageModeShared] : nil;
    _indexCount = idata.length/sizeof(uint32_t);
    _texture = tex;
    // Compute Y bounds for ground placement
    Vec3 minv = { FLT_MAX, FLT_MAX, FLT_MAX };
    Vec3 maxv = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    const VertexPU* verts = (const VertexPU*)vdata.bytes;
    for (NSUInteger i=0;i<_vertexCount;i++) {
        Vec3 p = verts[i].pos;
        if (p.y < minv.y) minv.y = p.y;
        if (p.y > maxv.y) maxv.y = p.y;
    }
    _modelMinY = minv.y; _modelMaxY = maxv.y; _modelExtentY = maxv.y - minv.y;
    // Auto-fit camera: the model matrix translates center to origin,
    // so aim the camera at the world origin (0,0,0).
    _camTarget[0] = 0.0f;
    _camTarget[1] = 0.0f;
    _camTarget[2] = 0.0f;
    // Centre visuel: vue légèrement plongeante et distance plus confortable
    _camYaw = 0.0f;
    _camPitch = 0.15f;
    // With FOV=60°, radius≈1 after our normalization, keep a small margin
    float radius = 1.0f; // since scale maps max extent to ~2 units
    float fov = (float)M_PI/3.0f;
    _camDist = (radius / sinf(fov*0.5f)) + 1.0f; // un peu plus loin

    NSLog(@"Loaded GLB: vertices=%lu, indices=%lu, center=(%.2f,%.2f,%.2f), scale=%.3f",
          (unsigned long)_vertexCount,
          (unsigned long)_indexCount,
          _modelCenter[0], _modelCenter[1], _modelCenter[2], _modelScale);
    return YES;
}

- (void)setOrbitYaw:(float)yaw pitch:(float)pitch distance:(float)dist {
    _camYaw = yaw;
    // Clamp pitch to avoid gimbal lock
    if (pitch > 1.5f) pitch = 1.5f; if (pitch < -1.5f) pitch = -1.5f;
    _camPitch = pitch;
    if (dist < 0.5f) dist = 0.5f; if (dist > 100.0f) dist = 100.0f;
    _camDist = dist;
}

- (void)setOrbitYawPitch:(float)yaw pitch:(float)pitch {
    _camYaw = yaw;
    if (pitch > 1.5f) pitch = 1.5f; if (pitch < -1.5f) pitch = -1.5f;
    _camPitch = pitch;
    // keep _camDist unchanged
}

- (void)setOrbitTargetX:(float)x y:(float)y z:(float)z {
    _camTarget[0] = x; _camTarget[1] = y; _camTarget[2] = z;
}

- (void)getOrbitYaw:(float*)yaw pitch:(float*)pitch distance:(float*)dist {
    if (yaw) *yaw = _camYaw;
    if (pitch) *pitch = _camPitch;
    if (dist) *dist = _camDist;
}

- (void)updateUniforms {
     CGSize ds = _layer.drawableSize;
     float aspect = ds.width>0 ? (ds.width/ds.height) : 1.3333f;
    float P[16], V[16], S[16], Tc[16], M[16], PV[16], MVP[16];
     mat4_perspective(P, (float)M_PI/3.0f, aspect, 0.1f, 100.0f);
     // Orbit camera position around target
    // Clamp pitch to prevent camera from entering the floor
    float minPitch = -1.5f;
    if (_camDist > 0.001f) {
        float minSin = (_groundY + _groundClearance - _camTarget[1]) / _camDist;
        // ensure in [-1,1]
        if (minSin > 1.0f) minSin = 1.0f; if (minSin < -1.0f) minSin = -1.0f;
        float mp = asinf(minSin);
        if (mp > minPitch) minPitch = mp;
    }
    float usePitch = _camPitch;
    if (usePitch < minPitch) { usePitch = minPitch; _camPitch = usePitch; }

    float cx = _camTarget[0] + _camDist * cosf(usePitch) * sinf(_camYaw);
    float cy = _camTarget[1] + _camDist * sinf(usePitch);
    float cz = _camTarget[2] + _camDist * cosf(usePitch) * cosf(_camYaw);
     // Build lookAt view matrix
     mat4_lookAt(V, cx, cy, cz, _camTarget[0], _camTarget[1], _camTarget[2], 0.0f, 1.0f, 0.0f);
     // When no model is loaded, avoid collapsing geometry by using identity
    if (_vertexCount == 0) {
        // No model loaded: use identity model matrix so test geometry is visible
        mat4_identity(M);
    } else {
        float s = (_modelScale > 0.0f) ? _modelScale : 1.0f;
        mat4_scale(S, s, s, s);
        mat4_translate(Tc, -_modelCenter[0], -_modelCenter[1], -_modelCenter[2]);
        // Place the model so its minimum Y (after centering and scaling) sits on groundY
        float baseLocal = (_modelMinY - _modelCenter[1]); // exact offset from center to minY
        float lift = _groundY - 1e-3f - (baseLocal * s);  // sink a hair to avoid hover
        float Ty[16]; mat4_translate(Ty, 0.0f, lift, 0.0f);
        float ST[16]; mat4_mul(ST, S, Tc);
        mat4_mul(M, Ty, ST);
    }
     mat4_mul(PV, P, V);
     mat4_mul(MVP, PV, M);
    // Pack uniforms: MVP (16 floats) + time + exposure + storm
    float U[32];
    memcpy(U, MVP, sizeof(MVP));
    U[16] = _time;
    U[17] = _exposure;
    U[18] = _stormIntensity;
    // pad remaining to keep alignment
    for (int i=19;i<32;i++) U[i]=0.0f;
    memcpy(_uniformBuffer.contents, U, sizeof(U));

    // Sky uniforms: use same projection, but remove translation from view to simulate infinite distance
    float Vsky[16]; memcpy(Vsky, V, sizeof(V)); Vsky[12]=0.0f; Vsky[13]=0.0f; Vsky[14]=0.0f;
    float Msky[16]; mat4_scale(Msky, 1.0f, 1.0f, 1.0f); // dome already large
    float PVsky[16], MVPsky[16];
    mat4_mul(PVsky, P, Vsky);
    mat4_mul(MVPsky, PVsky, Msky);
    float Us[32]; memcpy(Us, MVPsky, sizeof(MVPsky)); Us[16]=_time; Us[17]=_exposure; Us[18]=_stormIntensity; for(int i=19;i<32;i++) Us[i]=0.0f;
    memcpy(_skyUniformBuffer.contents, Us, sizeof(Us));

    // Ground uniforms: translate ground to follow camera XZ to look infinite
    float Mg[16]; mat4_translate(Mg, cx, _groundY, cz);
    float MVPg[16]; mat4_mul(MVPg, PV, Mg);
    float Ug[32]; memcpy(Ug, MVPg, sizeof(MVPg));
    Ug[16] = 2.0f; // tile size (units per square) — larger checks for visibility
    Ug[17] = _exposure;
    Ug[18] = _stormIntensity;
    Ug[19] = cx; // offsetX to anchor pattern in world
    Ug[20] = cz; // offsetZ
    // Horizon fog: fine grayscale (near-black) nuance, narrow blend band
    Ug[21] = 1000.0f;  // fogStart — effectively disables fog
    Ug[22] = 2000.0f;  // fogEnd — keep far to avoid blend
    Ug[23] = 0.0f;     // horizonColor — black (no tint)
    Ug[24] = 0.0f;
    Ug[25] = 0.0f;
    for (int i=26;i<32;i++) Ug[i]=0.0f;
    memcpy(_groundUniformBuffer.contents, Ug, sizeof(Ug));
}

- (void)stop { _running = NO; NSLog(@"MMRenderer stop"); }

@end

// Simple math helpers
static void mat4_perspective(float* m, float fovRadians, float aspect, float znear, float zfar) {
    float f = 1.0f / tanf(fovRadians * 0.5f);
    m[0]=f/aspect; m[1]=0; m[2]=0; m[3]=0;
    m[4]=0; m[5]=f; m[6]=0; m[7]=0;
    m[8]=0; m[9]=0; m[10]=(zfar+znear)/(znear-zfar); m[11]=-1;
    m[12]=0; m[13]=0; m[14]=(2*zfar*znear)/(znear-zfar); m[15]=0;
}
static void mat4_identity(float* m){ memset(m,0,sizeof(float)*16); m[0]=m[5]=m[10]=m[15]=1; }
// Column-major matrix multiply: out = a * b (Metal/OpenGL style)
static void mat4_mul(float* out,const float* a,const float* b){
    for (int r=0; r<4; ++r) {
        for (int c=0; c<4; ++c) {
            out[c*4 + r] = a[0*4 + r]*b[c*4 + 0]
                          + a[1*4 + r]*b[c*4 + 1]
                          + a[2*4 + r]*b[c*4 + 2]
                          + a[3*4 + r]*b[c*4 + 3];
        }
    }
}
static void mat4_translate(float* m,float x,float y,float z){ mat4_identity(m); m[12]=x; m[13]=y; m[14]=z; }
static void mat4_rotateY(float* m,float a){ mat4_identity(m); m[0]=cosf(a); m[2]=sinf(a); m[8]=-sinf(a); m[10]=cosf(a); }
static void mat4_scale(float* m,float x,float y,float z){ mat4_identity(m); m[0]=x; m[5]=y; m[10]=z; }

// Build a lookAt matrix
static void mat4_lookAt(float* m, float eyeX,float eyeY,float eyeZ, float centerX,float centerY,float centerZ, float upX,float upY,float upZ) {
    // Compute forward (z), right (x), up (y)
    float fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
    float fl = sqrtf(fx*fx + fy*fy + fz*fz); if (fl==0) fl = 1; fx/=fl; fy/=fl; fz/=fl;
    float rlX = fy*upZ - fz*upY; // cross(f, up)
    float rlY = fz*upX - fx*upZ;
    float rlZ = fx*upY - fy*upX;
    float rl = sqrtf(rlX*rlX + rlY*rlY + rlZ*rlZ); if (rl==0) rl=1; rlX/=rl; rlY/=rl; rlZ/=rl;
    float up2X = rlY*fz - rlZ*fy; // cross(r, f)
    float up2Y = rlZ*fx - rlX*fz;
    float up2Z = rlX*fy - rlY*fx;

    m[0]=rlX; m[1]=up2X; m[2]=-fx;  m[3]=0;
    m[4]=rlY; m[5]=up2Y; m[6]=-fy;  m[7]=0;
    m[8]=rlZ; m[9]=up2Z; m[10]=-fz; m[11]=0;
    m[12]=-(rlX*eyeX + rlY*eyeY + rlZ*eyeZ);
    m[13]=-(up2X*eyeX + up2Y*eyeY + up2Z*eyeZ);
    m[14]=fx*eyeX + fy*eyeY + fz*eyeZ; // since we stored -f in basis
    m[15]=1;
}
