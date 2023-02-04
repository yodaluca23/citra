// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include "core/frontend/framebuffer_layout.h"
#include "video_core/rasterizer_interface.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {

struct RendererSettings {
    // Screenshot
    std::atomic_bool screenshot_requested{false};
    void* screenshot_bits{};
    std::function<void()> screenshot_complete_callback;
    Layout::FramebufferLayout screenshot_framebuffer_layout;
    // Present
    std::atomic_bool bg_color_update_requested{true};
    std::atomic_bool sampler_update_requested{true};
    std::atomic_bool shader_update_requested{true};
};

class RendererBase : NonCopyable {
public:
    explicit RendererBase(Frontend::EmuWindow& window, Frontend::EmuWindow* secondary_window);
    virtual ~RendererBase();

    /// Returns the rasterizer owned by the renderer
    virtual VideoCore::RasterizerInterface* Rasterizer() = 0;

    /// Finalize rendering the guest frame and draw into the presentation texture
    virtual void SwapBuffers() = 0;

    /// Draws the latest frame to the window waiting timeout_ms for a frame to arrive (Renderer
    /// specific implementation)
    virtual void TryPresent(int timeout_ms, bool is_secondary) = 0;
    virtual void TryPresent(int timeout_ms) {
        TryPresent(timeout_ms, false);
    }

    /// This is called to notify the rendering backend of a surface change
    virtual void NotifySurfaceChanged() {}

    /// Prepares for video dumping (e.g. create necessary buffers, etc)
    virtual void PrepareVideoDumping() = 0;

    /// Cleans up after video dumping is ended
    virtual void CleanupVideoDumping() = 0;

    /// Synchronizes fixed function renderer state
    virtual void Sync() = 0;

    /// Updates the framebuffer layout of the contained render window handle.
    void UpdateCurrentFramebufferLayout(bool is_portrait_mode = {});

    // Getter/setter functions:
    // ------------------------

    f32 GetCurrentFPS() const {
        return m_current_fps;
    }

    int GetCurrentFrame() const {
        return m_current_frame;
    }

    Frontend::EmuWindow& GetRenderWindow() {
        return render_window;
    }

    const Frontend::EmuWindow& GetRenderWindow() const {
        return render_window;
    }

    [[nodiscard]] RendererSettings& Settings() {
        return settings;
    }

    [[nodiscard]] const RendererSettings& Settings() const {
        return settings;
    }

    /// Refreshes the settings common to all renderers
    void RefreshBaseSettings();

    /// Returns true if a screenshot is being processed
    bool IsScreenshotPending() const;

    /// Request a screenshot of the next frame
    void RequestScreenshot(void* data, std::function<void()> callback,
                           const Layout::FramebufferLayout& layout);

protected:
    RendererSettings settings;
    Frontend::EmuWindow& render_window;    ///< Reference to the render window handle.
    Frontend::EmuWindow* secondary_window; ///< Reference to the secondary render window handle.
    f32 m_current_fps = 0.0f;              ///< Current framerate, should be set by the renderer
    int m_current_frame = 0;               ///< Current frame, should be set by the renderer
};

} // namespace VideoCore
