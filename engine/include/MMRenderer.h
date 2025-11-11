// Include guard to avoid multiple inclusion and make header safe for C/C++ indexers
#ifndef MMRENDERER_H
#define MMRENDERER_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <QuartzCore/CAMetalLayer.h>

@interface MMRenderer : NSObject
- (void)setDrawableLayer:(CAMetalLayer *)layer;
- (BOOL)loadOBJAtPath:(NSString *)objPath texturePath:(NSString *)texturePath error:(NSError **)error;
- (BOOL)loadGLBAtPath:(NSString *)glbPath error:(NSError **)error;
- (void)start;
- (void)renderFrame;
- (void)stop;

// Orbit camera controls
- (void)setOrbitYaw:(float)yaw pitch:(float)pitch distance:(float)dist;
- (void)setOrbitYawPitch:(float)yaw pitch:(float)pitch; // preserve current distance
- (void)setOrbitTargetX:(float)x y:(float)y z:(float)z;
// Get current orbit parameters
- (void)getOrbitYaw:(float*)yaw pitch:(float*)pitch distance:(float*)dist;
@end

#else
// Forward declaration for non-Objective-C translation units to prevent parsing errors
typedef struct MMRenderer MMRenderer;
#endif

#endif /* MMRENDERER_H */
