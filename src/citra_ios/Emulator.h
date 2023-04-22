#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>

typedef NS_ENUM(NSInteger, InstallCIAResult) {
    InstallCIAResultSuccess,
    InstallCIAResultErrorInvalid,
    InstallCIAResultErrorEncrypted,
    InstallCIAResultErrorUnknown,
};

@interface Emulator : NSObject
@property(atomic, assign) BOOL useJIT;
@property(atomic, copy) NSURL* executableURL;

+ (BOOL)checkJITIsAvailable;
+ (InstallCIAResult)installCIA:(nonnull NSURL*)ciaURL;
+ (NSData*)getSMDH:(NSURL*)appURL;
+ (UIImage*)getIcon:(nonnull NSData*)smdh large:(BOOL)large;
- (nonnull id)initWithMetalLayer:(nonnull CAMetalLayer*)metalLayer
                  viewController:(nonnull UIViewController*)viewController;
- (void)startEmulator;
- (void)layerWasResized;
@end
