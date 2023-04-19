struct Float2D {
    float x;
    float y;
};

#ifdef __cplusplus
#include "core/frontend/input.h"

template <typename StatusType>
class InputBridge : public Input::InputDevice<StatusType> {
public:
    std::atomic<StatusType> current_value;

    InputBridge(StatusType initial_value) {
        current_value = initial_value;
    }

    StatusType GetStatus() const {
        return current_value;
    }
};

class AnalogInputBridge : public Input::InputDevice<std::tuple<float, float>> {
public:
    std::atomic<Float2D> current_value;

    AnalogInputBridge(Float2D initial_value) {
        current_value = initial_value;
    }

    std::tuple<float, float> GetStatus() const {
        Float2D cv = current_value.load();
        return { cv.x, cv.y };
    }
};
#endif

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <GameController/GameController.h>

@interface ButtonInputBridge: NSObject
-(nonnull id)init;
-(void)valueChangedHandler:(nonnull GCControllerButtonInput*)input value:(float)value pressed:(BOOL)pressed;
#ifdef __cplusplus
-(InputBridge<bool>*)getCppBridge;
#endif
@end


@interface StickInputBridge: NSObject
-(nonnull id)init;
-(void)valueChangedHandler:(nonnull GCControllerDirectionPad*)input x:(float)xValue y:(float)yValue;
#ifdef __cplusplus
-(AnalogInputBridge*)getCppBridge;
#endif
@end

#endif
