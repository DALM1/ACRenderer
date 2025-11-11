#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import "../engine/include/MMRenderer.h"

@interface OrbitView : NSView { MMRenderer* _renderer; NSPoint _last; BOOL _dragging; float _yaw; float _pitch; float _dist; }
- (instancetype)initWithFrame:(NSRect)frame renderer:(MMRenderer*)renderer;
// Sync local orbit state from renderer to avoid jumps
- (void)syncOrbitFromRenderer;
@end

@implementation OrbitView
- (BOOL)acceptsFirstResponder { return YES; }
- (instancetype)initWithFrame:(NSRect)frame renderer:(MMRenderer*)renderer { if (self=[super initWithFrame:frame]) { _renderer=renderer; _yaw=0.0f; _pitch=0.15f; _dist=5.0f; /* Ne pas forcer la caméra ici: laisser le renderer régler yaw/pitch/dist lors du chargement du modèle */ } return self; }
- (void)syncOrbitFromRenderer { float yaw=0, pitch=0, dist=0; [_renderer getOrbitYaw:&yaw pitch:&pitch distance:&dist]; _yaw=yaw; _pitch=pitch; _dist=dist; }
- (void)mouseDown:(NSEvent*)e { _dragging=YES; _last=[e locationInWindow]; }
- (void)mouseUp:(NSEvent*)e { _dragging=NO; }
- (void)mouseDragged:(NSEvent*)e { if(!_dragging) return; NSPoint p=[e locationInWindow]; CGFloat dx=p.x-_last.x; CGFloat dy=p.y-_last.y; _yaw += dx*0.01f; _pitch += dy*0.01f; if(_pitch>1.5f)_pitch=1.5f; if(_pitch<-1.5f)_pitch=-1.5f; [_renderer setOrbitYawPitch:_yaw pitch:_pitch]; _last=p; }
- (void)scrollWheel:(NSEvent*)e { _dist -= e.deltaY*0.1f; if(_dist<0.5f)_dist=0.5f; if(_dist>100.0f)_dist=100.0f; [_renderer setOrbitYaw:_yaw pitch:_pitch distance:_dist]; }
- (void)keyDown:(NSEvent*)e { if([[e charactersIgnoringModifiers] isEqualToString:@"r"]) { _yaw=0; _pitch=0.15f; /* garder la distance actuelle */ [_renderer setOrbitYawPitch:_yaw pitch:_pitch]; } else { [super keyDown:e]; } }
@end

int main(int argc, char* argv[]) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect frame = NSMakeRect(0, 0, 1200, 700);
        NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window setTitle:@"ACRenderer"];
        [window center];

        NSView *contentView = [window contentView];
        [contentView setWantsLayer:YES];
        CAMetalLayer *metalLayer = [CAMetalLayer layer];
        metalLayer.frame = contentView.bounds;
        metalLayer.contentsScale = NSScreen.mainScreen.backingScaleFactor;
        MMRenderer *renderer = [[MMRenderer alloc] init];
        [renderer setDrawableLayer:metalLayer];
        // Remplacer le contentView par une OrbitView pour capter les entrées, liée au renderer
        OrbitView *orbitView = [[OrbitView alloc] initWithFrame:contentView.bounds renderer:renderer];
        orbitView.wantsLayer = YES;
        orbitView.layer = metalLayer;
        [window setContentView:orbitView];

        // Resolve asset path robustly: compile-time ASSETS_DIR, then fallbacks
        NSString* primaryAssets = nil;
#ifdef ASSETS_DIR
        primaryAssets = @ASSETS_DIR;
#else
        // Default to CWD/assets when ASSETS_DIR is not defined
        primaryAssets = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:@"assets"];
#endif
        // Default to the restored museum model (file name with spaces)
        NSString* glbPath = [primaryAssets stringByAppendingPathComponent:@"models/Leda e il cigno- VA Museum.glb"];
        NSFileManager* fm = [NSFileManager defaultManager];
        // Allow CLI override: pass a path as argv[1]
        if (argc > 1 && argv[1]) {
            NSString* cliPath = [NSString stringWithUTF8String:argv[1]];
            if (cliPath.length > 0) { glbPath = cliPath; }
        }
        if (![fm fileExistsAtPath:glbPath]) {
            // Fallback: search for any .glb under assets/models
            NSArray<NSString*>* candidatesRoots = @[ [fm currentDirectoryPath], [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent] ];
            BOOL found = NO;
            for (NSString* root in candidatesRoots) {
                NSString* modelsDir = [root stringByAppendingPathComponent:@"assets/models"];
                BOOL isDir = NO;
                if ([fm fileExistsAtPath:modelsDir isDirectory:&isDir] && isDir) {
                    NSArray<NSString*>* contents = [fm contentsOfDirectoryAtPath:modelsDir error:nil] ?: @[];
                    for (NSString* name in contents) {
                        if ([[name.pathExtension lowercaseString] isEqualToString:@"glb"]) {
                            NSString* p = [modelsDir stringByAppendingPathComponent:name];
                            if ([fm fileExistsAtPath:p]) { glbPath = p; found = YES; break; }
                        }
                    }
                }
                if (found) break;
            }
        }
        NSLog(@"GLB path resolved: %@ exists=%@", glbPath, [fm fileExistsAtPath:glbPath]?@"YES":@"NO");
        NSError* loadErr = nil;
        if (![renderer loadGLBAtPath:glbPath error:&loadErr]) {
            NSLog(@"GLB load failed: %@", loadErr);
        }
        [renderer start];
        // Synchroniser l'état de l'UI avec la caméra réelle
        [orbitView syncOrbitFromRenderer];

        NSTimer *timer = [NSTimer scheduledTimerWithTimeInterval:(1.0/60.0)
                                                          repeats:YES
                                                            block:^(NSTimer * _Nonnull t) {
            CGSize bounds = window.contentView.bounds.size;
            CGFloat scale = metalLayer.contentsScale;
            metalLayer.drawableSize = CGSizeMake(bounds.width * scale, bounds.height * scale);
            [renderer renderFrame];
        }];
        (void)timer;

        [window makeKeyAndOrderFront:nil];
        [window orderFrontRegardless];
        [window setIsVisible:YES];
        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];

        [renderer stop];
    }
    return 0;
}