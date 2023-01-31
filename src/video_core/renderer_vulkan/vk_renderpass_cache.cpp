// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <limits>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Vulkan {

RenderpassCache::RenderpassCache(const Instance& instance, Scheduler& scheduler)
    : instance{instance}, scheduler{scheduler}, dynamic_rendering{
                                                    instance.IsDynamicRenderingSupported()} {}

RenderpassCache::~RenderpassCache() {
    vk::Device device = instance.GetDevice();
    for (u32 color = 0; color <= MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth <= MAX_DEPTH_FORMATS; depth++) {
            if (vk::RenderPass load_pass = cached_renderpasses[color][depth][0]; load_pass) {
                device.destroyRenderPass(load_pass);
            }

            if (vk::RenderPass clear_pass = cached_renderpasses[color][depth][1]; clear_pass) {
                device.destroyRenderPass(clear_pass);
            }
        }
    }

    for (auto& [key, framebuffer] : framebuffers) {
        device.destroyFramebuffer(framebuffer);
    }

    device.destroyRenderPass(present_renderpass);
}

void RenderpassCache::EnterRenderpass(Surface* const color, Surface* const depth_stencil,
                                      vk::Rect2D render_area, bool do_clear, vk::ClearValue clear) {
    ASSERT(color || depth_stencil);

    if (dynamic_rendering) {
        return BeginRendering(color, depth_stencil, render_area, do_clear, clear);
    }

    u32 width = UINT32_MAX;
    u32 height = UINT32_MAX;
    u32 cursor = 0;
    std::array<VideoCore::PixelFormat, 2> formats{};
    std::array<vk::ImageView, 2> views{};

    const auto Prepare = [&](Surface* const surface) {
        if (!surface) {
            formats[cursor++] = VideoCore::PixelFormat::Invalid;
            return;
        }

        width = std::min(width, surface->GetScaledWidth());
        height = std::min(height, surface->GetScaledHeight());
        formats[cursor] = surface->pixel_format;
        views[cursor++] = surface->GetFramebufferView();
    };

    Prepare(color);
    Prepare(depth_stencil);

    const vk::RenderPass renderpass = GetRenderpass(formats[0], formats[1], do_clear);

    const FramebufferInfo framebuffer_info = {
        .color = views[0],
        .depth = views[1],
        .width = width,
        .height = height,
    };

    auto [it, new_framebuffer] = framebuffers.try_emplace(framebuffer_info);
    if (new_framebuffer) {
        it->second = CreateFramebuffer(framebuffer_info, renderpass);
    }

    const RenderpassState new_state = {
        .renderpass = renderpass,
        .framebuffer = it->second,
        .render_area = render_area,
        .clear = clear,
    };

    const u64 new_state_hash = Common::ComputeStructHash64(new_state);
    const bool is_dirty = scheduler.IsStateDirty(StateFlags::Renderpass);
    if (state_hash == new_state_hash && rendering && !is_dirty) {
        cmd_count++;
        return;
    }

    if (rendering) {
        ExitRenderpass();
    }

    scheduler.Record([new_state](vk::CommandBuffer cmdbuf) {
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = new_state.renderpass,
            .framebuffer = new_state.framebuffer,
            .renderArea = new_state.render_area,
            .clearValueCount = 1,
            .pClearValues = &new_state.clear,
        };

        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
    });

    scheduler.MarkStateNonDirty(StateFlags::Renderpass);
    state_hash = new_state_hash;
    rendering = true;
}

void RenderpassCache::ExitRenderpass() {
    if (!rendering) {
        return;
    }

    rendering = false;
    scheduler.Record([dynamic_rendering = dynamic_rendering](vk::CommandBuffer cmdbuf) {
        if (dynamic_rendering) {
            cmdbuf.endRenderingKHR();
        } else {
            cmdbuf.endRenderPass();
        }
    });

    // The Mali guide recommends flushing at the end of each major renderpass
    // Testing has shown this has a significant effect on rendering performance
    if (cmd_count > 20 && instance.IsMaliGpu()) {
        scheduler.Flush();
        cmd_count = 0;
    }
}

void RenderpassCache::BeginRendering(Surface* const color, Surface* const depth_stencil,
                                     vk::Rect2D render_area, bool do_clear, vk::ClearValue clear) {
    RenderingState new_state = {
        .render_area = render_area,
        .clear = clear,
        .do_clear = do_clear,
    };

    if (color) {
        new_state.color_view = color->GetFramebufferView();
    }
    if (depth_stencil) {
        new_state.depth_view = depth_stencil->GetFramebufferView();
    }

    const u64 new_state_hash = Common::ComputeStructHash64(new_state);
    const bool is_dirty = scheduler.IsStateDirty(StateFlags::Renderpass);
    if (state_hash == new_state_hash && rendering && !is_dirty) {
        cmd_count++;
        return;
    }

    if (rendering) {
        ExitRenderpass();
    }

    const bool has_stencil =
        depth_stencil && depth_stencil->type == VideoCore::SurfaceType::DepthStencil;
    scheduler.Record([new_state, has_stencil](vk::CommandBuffer cmdbuf) {
        u32 cursor = 0;
        std::array<vk::RenderingAttachmentInfoKHR, 2> infos{};

        const auto Prepare = [&](vk::ImageView image_view) {
            if (!image_view) {
                cursor++;
                return;
            }

            infos[cursor++] = vk::RenderingAttachmentInfoKHR{
                .imageView = image_view,
                .imageLayout = vk::ImageLayout::eGeneral,
                .loadOp =
                    new_state.do_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = new_state.clear,
            };
        };

        Prepare(new_state.color_view);
        Prepare(new_state.depth_view);

        const u32 color_attachment_count = new_state.color_view ? 1u : 0u;
        const vk::RenderingAttachmentInfoKHR* depth_info =
            new_state.depth_view ? &infos[1] : nullptr;
        const vk::RenderingAttachmentInfoKHR* stencil_info = has_stencil ? &infos[1] : nullptr;
        const vk::RenderingInfoKHR rendering_info = {
            .renderArea = new_state.render_area,
            .layerCount = 1,
            .colorAttachmentCount = color_attachment_count,
            .pColorAttachments = &infos[0],
            .pDepthAttachment = depth_info,
            .pStencilAttachment = stencil_info,
        };

        cmdbuf.beginRenderingKHR(rendering_info);
    });

    scheduler.MarkStateNonDirty(StateFlags::Renderpass);
    state_hash = new_state_hash;
    rendering = true;
}

void RenderpassCache::CreatePresentRenderpass(vk::Format format) {
    if (!present_renderpass) {
        present_renderpass =
            CreateRenderPass(format, vk::Format::eUndefined, vk::AttachmentLoadOp::eClear,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
    }
}

vk::RenderPass RenderpassCache::GetRenderpass(VideoCore::PixelFormat color,
                                              VideoCore::PixelFormat depth, bool is_clear) {
    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? MAX_COLOR_FORMATS : static_cast<u32>(color);
    const u32 depth_index = depth == VideoCore::PixelFormat::Invalid
                                ? MAX_DEPTH_FORMATS
                                : (static_cast<u32>(depth) - 14);

    ASSERT_MSG(color_index <= MAX_COLOR_FORMATS && depth_index <= MAX_DEPTH_FORMATS &&
                   (color_index != MAX_COLOR_FORMATS || depth_index != MAX_DEPTH_FORMATS),
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    vk::RenderPass& renderpass = cached_renderpasses[color_index][depth_index][is_clear];
    if (!renderpass) {
        const vk::Format color_format = instance.GetTraits(color).native;
        const vk::Format depth_format = instance.GetTraits(depth).native;
        const vk::AttachmentLoadOp load_op =
            is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
        renderpass = CreateRenderPass(color_format, depth_format, load_op,
                                      vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
    }

    return renderpass;
}

vk::RenderPass RenderpassCache::CreateRenderPass(vk::Format color, vk::Format depth,
                                                 vk::AttachmentLoadOp load_op,
                                                 vk::ImageLayout initial_layout,
                                                 vk::ImageLayout final_layout) const {
    u32 attachment_count = 0;
    std::array<vk::AttachmentDescription, 2> attachments;

    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = color,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = initial_layout,
            .finalLayout = final_layout,
        };

        color_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eGeneral,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = depth,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = load_op,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        };

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eGeneral,
        };

        use_depth = true;
    }

    // We also require only one subpass
    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = 0,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr,
    };

    return instance.GetDevice().createRenderPass(renderpass_info);
}

vk::Framebuffer RenderpassCache::CreateFramebuffer(const FramebufferInfo& info,
                                                   vk::RenderPass renderpass) {
    u32 attachment_count = 0;
    std::array<vk::ImageView, 2> attachments;

    if (info.color) {
        attachments[attachment_count++] = info.color;
    }

    if (info.depth) {
        attachments[attachment_count++] = info.depth;
    }

    const vk::FramebufferCreateInfo framebuffer_info = {
        .renderPass = renderpass,
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .width = info.width,
        .height = info.height,
        .layers = 1,
    };

    return instance.GetDevice().createFramebuffer(framebuffer_info);
}

} // namespace Vulkan
