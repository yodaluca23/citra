#import "InputBridge.h"

@implementation ButtonInputBridge {
    InputBridge<bool>* _cppBridge;
}
- (nonnull id)init {
    if (self = [super init]) {
        _cppBridge = new InputBridge<bool>(false);
    }
    return self;
}
- (void)valueChangedHandler:(nonnull GCControllerButtonInput *)input value:(float)value pressed:(BOOL)pressed {
    NSLog(@"%@ %f %d", input, value, pressed);
    _cppBridge->current_value = pressed;
}
- (InputBridge<bool>*)getCppBridge {
    return _cppBridge;
}
@end
