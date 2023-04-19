#import "Emulator.h"
#import "citra_ios-Swift.h"
#import <Foundation/Foundation.h>
#import "InputBridge.h"

#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "common/logging/backend.h"
#include "common/logging/log.h"

#define CS_OPS_STATUS 0
#define CS_DEBUGGED 0x10000000

extern "C" int csops(pid_t pid, unsigned int ops, void* useraddr, size_t usersize);

class EmuWindow_IOS : public Frontend::EmuWindow {
public:
    int width;
    int height;
    void InitWithMetalLayer(CAMetalLayer* metalLayer) {
        // TODO: iOS is not macOS so we should use like "AppleOS" or "Metal"
        // but it works though
        window_info.type = Frontend::WindowSystemType::MacOS;
        window_info.render_surface = (void*)CFBridgingRetain(metalLayer);
        // window_info.render_surface_scale = 2.0f; // it seems anyone dont care it
    }
    ~EmuWindow_IOS() {
        if (window_info.render_surface != nullptr) {
            CFBridgingRelease(window_info.render_surface);
        }
    }
protected:
    void PollEvents() {
        // please dont spam it
        // printf("TODO: PollEvents %d %d\n", width, height);
        UpdateCurrentFramebufferLayout(width, height, true);
    }
};

class EmuButtonFactory : public Input::Factory<Input::ButtonDevice> {
    std::unique_ptr<Input::ButtonDevice> Create(const Common::ParamPackage& params) override {
        int button_id = params.Get("code", 0);
        printf("GET BUTTON ID %d\n", button_id);
        ButtonInputBridge* emuInput = nullptr;
        switch ((Settings::NativeButton::Values)button_id) {
            case Settings::NativeButton::A:
                emuInput = EmulatorInput.buttonA;
                break;
            case Settings::NativeButton::B:
                emuInput = EmulatorInput.buttonB;
                break;
            case Settings::NativeButton::X:
                emuInput = EmulatorInput.buttonX;
                break;
            case Settings::NativeButton::Y:
                emuInput = EmulatorInput.buttonY;
                break;
            case Settings::NativeButton::Up:
                emuInput = EmulatorInput.dpadUp;
                break;
            case Settings::NativeButton::Down:
                emuInput = EmulatorInput.dpadDown;
                break;
            case Settings::NativeButton::Left:
                emuInput = EmulatorInput.dpadLeft;
                break;
            case Settings::NativeButton::Right:
                emuInput = EmulatorInput.dpadRight;
                break;
            case Settings::NativeButton::L:
                emuInput = EmulatorInput.buttonL;
                break;
            case Settings::NativeButton::R:
                emuInput = EmulatorInput.buttonR;
                break;
            case Settings::NativeButton::Start:
                emuInput = EmulatorInput.buttonStart;
                break;
            case Settings::NativeButton::Select:
                emuInput = EmulatorInput.buttonSelect;
                break;
            case Settings::NativeButton::ZL:
                emuInput = EmulatorInput.buttonZL;
                break;
            case Settings::NativeButton::ZR:
                emuInput = EmulatorInput.buttonZR;
                break;
            case Settings::NativeButton::Debug:
            case Settings::NativeButton::Gpio14:
            case Settings::NativeButton::Home:
            case Settings::NativeButton::NumButtons:
                emuInput = EmulatorInput._buttonDummy;
        }
        if (emuInput == nullptr) {
            return {};
        }
        InputBridge<bool>* ib = [emuInput getCppBridge];
        return std::unique_ptr<InputBridge<bool>>(ib);
    };
};

class EmuAnalogFactory : public Input::Factory<Input::AnalogDevice> {
    std::unique_ptr<Input::AnalogDevice> Create(const Common::ParamPackage& params) override {
        int button_id = params.Get("code", 0);
        printf("GET ANALOG ID %d\n", button_id);
        StickInputBridge* emuInput = nullptr;
        switch ((Settings::NativeAnalog::Values)button_id) {
            case Settings::NativeAnalog::CirclePad:
                emuInput = EmulatorInput.circlePad;
                break;
            case Settings::NativeAnalog::CStick:
                emuInput = EmulatorInput.circlePadPro;
                break;
            case Settings::NativeAnalog::NumAnalogs:
                UNREACHABLE();
                break;
        }
        if (emuInput == nullptr) {
            return {};
        }
        AnalogInputBridge* ib = [emuInput getCppBridge];
        return std::unique_ptr<AnalogInputBridge>(ib);
    };
};

@implementation Emulator {
    __weak CAMetalLayer* _metalLayer;
    EmuWindow_IOS* _emuWindow;
    NSThread* _thread;
}

+ (BOOL)checkJITIsAvailable {
    // taken from dolphin-ios, licensed under GPL-2.0-or-later
    // https://github.com/OatmealDome/dolphin-ios/blob/a2485b9309db3cf14f8c5683a3f4d64476883eda/Source/iOS/App/Common/Jit/JitManager%2BDebugger.m#L6
    int flags;
    if (csops(getpid(), CS_OPS_STATUS, &flags, sizeof(flags) != 0)) {
      return false;
    }

    return flags & CS_DEBUGGED;
}

- (nonnull id)initWithMetalLayer:(nonnull CAMetalLayer *)metalLayer {
    if (self = [super init]) {
        _metalLayer = metalLayer;
        _emuWindow = new EmuWindow_IOS();
        _emuWindow->InitWithMetalLayer(metalLayer);
        _thread = [NSThread.alloc initWithTarget:self selector:@selector(_startEmulator) object:nil];
        [_thread setName: @"citra.emulator"];
        [_thread setQualityOfService: NSQualityOfServiceUserInteractive];
    }

    return self;
}

- (void)startEmulator {
    [self layerWasResized];
    [_thread start];
}

- (void)_startEmulator {
    NSLog(@"Emulator starting...");
    std::string filepath([_executableURL.path UTF8String]);
    Log::Filter log_filter(Log::Level::Info);
//    log_filter.ParseFilterString("*:Debug");
    Log::SetGlobalFilter(log_filter);
   Log::AddBackend(std::make_unique<Log::ConsoleBackend>());

    // set some settings
    Settings::values.graphics_api.SetValue(Settings::GraphicsAPI::Vulkan);
    Settings::values.use_cpu_jit.SetValue(_useJIT);
    for (const auto& service_module : Service::service_module_map) {
        Settings::values.lle_modules.emplace(service_module.name, false /* Always use HLE */);
    }
    // controls
    for (int i = 0; i < Settings::NativeButton::NumButtons; i++) {
        Common::ParamPackage param{
            {"engine", "ios_gamepad"},
            {"code", std::to_string(i)},
        };
        Settings::values.current_input_profile.buttons[i] = param.Serialize();
    }
    for (int i = 0; i < Settings::NativeAnalog::NumAnalogs; i++) {
        Common::ParamPackage param{
            {"engine", "ios_gamepad"},
            {"code", std::to_string(i)},
        };
        Settings::values.current_input_profile.analogs[i] = param.Serialize();
    }
    Settings::Apply();
    Settings::LogSettings();

    Input::RegisterFactory<Input::ButtonDevice>("ios_gamepad", std::make_shared<EmuButtonFactory>());
    Input::RegisterFactory<Input::AnalogDevice>("ios_gamepad", std::make_shared<EmuAnalogFactory>());

    Core::System& system{Core::System::GetInstance()};

    const auto load_result = system.Load(*_emuWindow, filepath, nullptr);

    if (load_result != Core::System::ResultStatus::Success) {
        NSLog(@"Failed to load");
        return;
    }

    NSLog(@"loaded");

    while (true) {
        const auto result = system.RunLoop();
        if (result == Core::System::ResultStatus::Success) {
        } else if (result == Core::System::ResultStatus::ShutdownRequested) {
            break;
        } else {
            LOG_ERROR(Frontend, "Error in main run loop: {}", result, system.GetStatusDetails());
//            NSLog(@"???");
            break;
        }
    }
}

- (void)layerWasResized {
    _emuWindow->width = _metalLayer.bounds.size.width * UIScreen.mainScreen.scale;
    _emuWindow->height = _metalLayer.bounds.size.height * UIScreen.mainScreen.scale;
}

@end
