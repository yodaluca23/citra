// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <condition_variable>
#include <mutex>
#include <queue>
#include "core/frontend/emu_window.h"
#include "video_core/renderer_vulkan/vk_common.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Frontend {

struct Frame {
    u32 width{};
    u32 height{};
    VmaAllocation allocation{};
    vk::Framebuffer framebuffer{};
    vk::Image image{};
    vk::ImageView image_view{};
    vk::Semaphore render_ready{};
    vk::Fence present_done{};
    std::mutex fence_mutex{};
    vk::CommandBuffer cmdbuf{};
};

} // namespace Frontend

namespace Vulkan {

class Instance;
class Swapchain;
class RenderpassCache;

class TextureMailbox final : public Frontend::TextureMailbox {
    static constexpr std::size_t SWAP_CHAIN_SIZE = 8;

public:
    TextureMailbox(const Instance& instance, const Swapchain& swapchain,
                   const RenderpassCache& renderpass_cache);
    ~TextureMailbox() override;

    void ReloadRenderFrame(Frontend::Frame* frame, u32 width, u32 height) override;

    Frontend::Frame* GetRenderFrame() override;
    Frontend::Frame* TryGetPresentFrame(int timeout_ms) override;

    void ReleaseRenderFrame(Frontend::Frame* frame) override;
    void ReleasePresentFrame(Frontend::Frame* frame) override;

private:
    const Instance& instance;
    const Swapchain& swapchain;
    const RenderpassCache& renderpass_cache;
    vk::CommandPool command_pool;
    std::mutex free_mutex;
    std::mutex present_mutex;
    std::condition_variable free_cv;
    std::condition_variable present_cv;
    std::array<Frontend::Frame, SWAP_CHAIN_SIZE> swap_chain{};
    std::queue<Frontend::Frame*> free_queue{};
    std::queue<Frontend::Frame*> present_queue{};
};

} // namespace Vulkan
