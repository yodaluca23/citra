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
#endif