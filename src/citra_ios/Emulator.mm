#import "Emulator.h"

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
        printf("TODO: PollEvents %d %d\n", width, height);
        UpdateCurrentFramebufferLayout(width, height, true);
    }
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
    Settings::Apply();
    Settings::LogSettings();

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
//            break;
        }
    }
}

- (void)layerWasResized {
    _emuWindow->width = _metalLayer.bounds.size.width * UIScreen.mainScreen.scale;
    _emuWindow->height = _metalLayer.bounds.size.height * UIScreen.mainScreen.scale;
}

@end
