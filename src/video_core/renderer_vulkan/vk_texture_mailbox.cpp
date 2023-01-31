// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_texture_mailbox.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

TextureMailbox::TextureMailbox(const Instance& instance_, const Swapchain& swapchain_,
                               const RenderpassCache& renderpass_cache_)
    : instance{instance_}, swapchain{swapchain_}, renderpass_cache{renderpass_cache_} {

    const vk::Device device = instance.GetDevice();
    const vk::CommandPoolCreateInfo pool_info = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                 vk::CommandPoolCreateFlagBits::eTransient,
        .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
    };
    command_pool = device.createCommandPool(pool_info);

    const vk::CommandBufferAllocateInfo alloc_info = {
        .commandPool = command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = SWAP_CHAIN_SIZE,
    };
    const std::vector command_buffers = device.allocateCommandBuffers(alloc_info);

    for (u32 i = 0; i < SWAP_CHAIN_SIZE; i++) {
        Frontend::Frame& frame = swap_chain[i];
        frame.cmdbuf = command_buffers[i];
        frame.render_ready = device.createSemaphore({});
        frame.present_done = device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled});
        free_queue.push(&frame);
    }
}

TextureMailbox::~TextureMailbox() {
    std::scoped_lock lock{present_mutex, free_mutex};
    free_queue = {};
    present_queue = {};
    present_cv.notify_all();
    free_cv.notify_all();

    const vk::Device device = instance.GetDevice();
    device.destroyCommandPool(command_pool);
    for (auto& frame : swap_chain) {
        device.destroyImageView(frame.image_view);
        device.destroyFramebuffer(frame.framebuffer);
        device.destroySemaphore(frame.render_ready);
        device.destroyFence(frame.present_done);
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    }
}

void TextureMailbox::ReloadRenderFrame(Frontend::Frame* frame, u32 width, u32 height) {
    vk::Device device = instance.GetDevice();
    if (frame->framebuffer) {
        device.destroyFramebuffer(frame->framebuffer);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    frame->image_view = device.createImageView(view_info);

    const vk::FramebufferCreateInfo framebuffer_info = {
        .renderPass = renderpass_cache.GetPresentRenderpass(),
        .attachmentCount = 1,
        .pAttachments = &frame->image_view,
        .width = width,
        .height = height,
        .layers = 1,
    };
    frame->framebuffer = instance.GetDevice().createFramebuffer(framebuffer_info);

    frame->width = width;
    frame->height = height;
}

Frontend::Frame* TextureMailbox::GetRenderFrame() {
    std::unique_lock lock{free_mutex};

    if (free_queue.empty()) {
        free_cv.wait(lock, [&] { return !free_queue.empty(); });
    }

    Frontend::Frame* frame = free_queue.front();
    free_queue.pop();
    return frame;
}

void TextureMailbox::ReleaseRenderFrame(Frontend::Frame* frame) {
    std::unique_lock lock{present_mutex};
    present_queue.push(frame);
    present_cv.notify_one();
}

void TextureMailbox::ReleasePresentFrame(Frontend::Frame* frame) {
    std::unique_lock lock{free_mutex};
    free_queue.push(frame);
    free_cv.notify_one();
}

Frontend::Frame* TextureMailbox::TryGetPresentFrame(int timeout_ms) {
    std::unique_lock lock{present_mutex};
    // Wait for new entries in the present_queue
    present_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return !present_queue.empty(); });
    if (present_queue.empty()) {
        LOG_DEBUG(Render_Vulkan, "Timed out waiting present frame");
        return nullptr;
    }

    Frontend::Frame* frame = present_queue.front();
    present_queue.pop();
    return frame;
}

} // namespace Vulkan
