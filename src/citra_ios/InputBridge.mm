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

@implementation StickInputBridge {
    AnalogInputBridge* _cppBridge;
}
- (nonnull id)init {
    if (self = [super init]) {
        _cppBridge = new AnalogInputBridge(Float2D{0, 0});
    }
    return self;
}
- (void)valueChangedHandler:(nonnull GCControllerDirectionPad*)input x:(float)xValue y:(float)yValue {
    _cppBridge->current_value.exchange(Float2D{xValue, yValue});
}
- (AnalogInputBridge*)getCppBridge {
    return _cppBridge;
}
@end
