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
    id<MTLBuffer> _vertexBuffer;
    id<MTLBuffer> _indexBuffer;
    NSUInteger _indexCount;
    NSUInteger _vertexCount;
    id<MTLTexture> _texture;
    id<MTLBuffer> _uniformBuffer; // MVP
    float _modelCenter[3];
    float _modelScale;
    id<MTLDepthStencilState> _depthState;
    id<MTLTexture> _depthTexture;
    CGSize _depthSize;
    id<MTLTexture> _fallbackTexture;

    // Orbit camera state
    float _camYaw;    // radians
    float _camPitch;  // radians
    float _camDist;   // units
    float _camTarget[3];
}

static NSString* kMSLTextured = @
    "using namespace metal;\n"
    "struct VSIn { float3 position [[attribute(0)]]; float2 uv [[attribute(1)]]; };\n"
    "struct VSOut { float4 position [[position]]; float2 uv; };\n"
    "struct Uniforms { float4x4 mvp; };\n"
    "vertex VSOut vertex_main(VSIn in [[stage_in]],\n"
    "                        constant Uniforms& U [[ buffer(1) ]]) {\n"
    "  VSOut o;\n"
    "  o.position = U.mvp * float4(in.position, 1.0);\n"
    "  o.uv = in.uv;\n"
    "  return o;\n"
    "}\n"
    "fragment float4 fragment_main(VSOut in [[stage_in]],\n"
    "                             texture2d<float> tex [[ texture(0) ]],\n"
    "                             sampler s [[ sampler(0) ]]) {\n"
    "  return tex.sample(s, in.uv);\n"
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
    id<MTLLibrary> lib = [_device newLibraryWithSource:kMSLTextured options:nil error:&err];
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

    _uniformBuffer = [_device newBufferWithLength:sizeof(float)*16 options:MTLResourceStorageModeShared];

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
     NSLog(@"MMRenderer start");
  }

- (void)renderFrame {
    if (!_running || !_layer) { return; }
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
    [enc setRenderPipelineState:_pipeline];
    [enc setDepthStencilState:_depthState];
    [enc setCullMode:MTLCullModeNone];
    [enc setFrontFacingWinding:MTLWindingCounterClockwise];
    [enc setTriangleFillMode:MTLTriangleFillModeFill];
    [enc setViewport:(MTLViewport){0,0, ds.width, ds.height, 0.0, 1.0}];
    [enc setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
    [enc setVertexBuffer:_uniformBuffer offset:0 atIndex:1];
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
     float cx = _camTarget[0] + _camDist * cosf(_camPitch) * sinf(_camYaw);
     float cy = _camTarget[1] + _camDist * sinf(_camPitch);
     float cz = _camTarget[2] + _camDist * cosf(_camPitch) * cosf(_camYaw);
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
        mat4_mul(M, S, Tc);
    }
     mat4_mul(PV, P, V);
     mat4_mul(MVP, PV, M);
    memcpy(_uniformBuffer.contents, MVP, sizeof(MVP));
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
