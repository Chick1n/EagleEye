#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenCVBridge : NSObject

+ (NSArray<NSValue *> *)detectCoinRectsInCGImage:(CGImageRef)image;

@end

NS_ASSUME_NONNULL_END
