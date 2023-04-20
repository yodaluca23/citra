#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>

@interface Emulator : NSObject
@property(atomic, assign) BOOL useJIT;
@property(atomic, copy) NSURL* executableURL;

+ (BOOL)checkJITIsAvailable;
- (nonnull id)initWithMetalLayer:(nonnull CAMetalLayer*)metalLayer viewController:(nonnull UIViewController*)viewController;
- (void)startEmulator;
- (void)layerWasResized;
@end
