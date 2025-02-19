add_subdirectory(host_shaders)

add_library(video_core STATIC
    command_processor.cpp
    command_processor.h
    debug_utils/debug_utils.cpp
    debug_utils/debug_utils.h
    geometry_pipeline.cpp
    geometry_pipeline.h
    gpu_debugger.h
    pica.cpp
    pica.h
    pica_state.h
    pica_types.h
    precompiled_headers.h
    primitive_assembly.cpp
    primitive_assembly.h
    rasterizer_accelerated.cpp
    rasterizer_accelerated.h
    rasterizer_interface.h
    regs.cpp
    regs.h
    regs_framebuffer.h
    regs_lighting.h
    regs_pipeline.h
    regs_rasterizer.h
    regs_shader.h
    regs_texturing.h
    renderer_base.cpp
    renderer_base.h
    rasterizer_cache/custom_tex_manager.cpp
    rasterizer_cache/custom_tex_manager.h
    rasterizer_cache/framebuffer_base.cpp
    rasterizer_cache/framebuffer_base.h
    rasterizer_cache/pixel_format.cpp
    rasterizer_cache/pixel_format.h
    rasterizer_cache/rasterizer_cache.cpp
    rasterizer_cache/rasterizer_cache.h
    rasterizer_cache/rasterizer_cache_base.h
    rasterizer_cache/sampler_params.h
    rasterizer_cache/slot_vector.h
    rasterizer_cache/surface_base.cpp
    rasterizer_cache/surface_base.h
    rasterizer_cache/texture_codec.h
    rasterizer_cache/utils.cpp
    rasterizer_cache/utils.h
    rasterizer_cache/surface_params.cpp
    rasterizer_cache/surface_params.h
    renderer_opengl/frame_dumper_opengl.cpp
    renderer_opengl/frame_dumper_opengl.h
    renderer_opengl/gl_driver.cpp
    renderer_opengl/gl_driver.h
    renderer_opengl/gl_format_reinterpreter.cpp
    renderer_opengl/gl_format_reinterpreter.h
    renderer_opengl/gl_rasterizer.cpp
    renderer_opengl/gl_rasterizer.h
    renderer_opengl/gl_rasterizer_cache.cpp
    renderer_opengl/gl_resource_manager.cpp
    renderer_opengl/gl_resource_manager.h
    renderer_opengl/gl_shader_decompiler.cpp
    renderer_opengl/gl_shader_decompiler.h
    renderer_opengl/gl_shader_disk_cache.cpp
    renderer_opengl/gl_shader_disk_cache.h
    renderer_opengl/gl_shader_gen.cpp
    renderer_opengl/gl_shader_gen.h
    renderer_opengl/gl_shader_manager.cpp
    renderer_opengl/gl_shader_manager.h
    renderer_opengl/gl_shader_util.cpp
    renderer_opengl/gl_shader_util.h
    renderer_opengl/gl_state.cpp
    renderer_opengl/gl_state.h
    renderer_opengl/gl_stream_buffer.cpp
    renderer_opengl/gl_stream_buffer.h
    renderer_opengl/gl_texture_runtime.cpp
    renderer_opengl/gl_texture_runtime.h
    renderer_opengl/gl_vars.cpp
    renderer_opengl/gl_vars.h
    renderer_opengl/pica_to_gl.h
    renderer_opengl/post_processing_opengl.cpp
    renderer_opengl/post_processing_opengl.h
    renderer_opengl/renderer_opengl.cpp
    renderer_opengl/renderer_opengl.h
    renderer_opengl/texture_filters/anime4k/anime4k_ultrafast.cpp
    renderer_opengl/texture_filters/anime4k/anime4k_ultrafast.h
    renderer_opengl/texture_filters/bicubic/bicubic.cpp
    renderer_opengl/texture_filters/bicubic/bicubic.h
    renderer_opengl/texture_filters/nearest_neighbor/nearest_neighbor.cpp
    renderer_opengl/texture_filters/nearest_neighbor/nearest_neighbor.h
    renderer_opengl/texture_filters/scale_force/scale_force.cpp
    renderer_opengl/texture_filters/scale_force/scale_force.h
    renderer_opengl/texture_filters/texture_filter_base.h
    renderer_opengl/texture_filters/texture_filterer.cpp
    renderer_opengl/texture_filters/texture_filterer.h
    renderer_opengl/texture_filters/xbrz/xbrz_freescale.cpp
    renderer_opengl/texture_filters/xbrz/xbrz_freescale.h
    renderer_vulkan/pica_to_vk.h
    renderer_vulkan/renderer_vulkan.cpp
    renderer_vulkan/renderer_vulkan.h
    renderer_vulkan/vk_blit_helper.cpp
    renderer_vulkan/vk_blit_helper.h
    renderer_vulkan/vk_common.cpp
    renderer_vulkan/vk_common.h
    renderer_vulkan/vk_descriptor_manager.cpp
    renderer_vulkan/vk_descriptor_manager.h
    renderer_vulkan/vk_format_reinterpreter.cpp
    renderer_vulkan/vk_format_reinterpreter.h
    renderer_vulkan/vk_master_semaphore.cpp
    renderer_vulkan/vk_master_semaphore.h
    renderer_vulkan/vk_rasterizer.cpp
    renderer_vulkan/vk_rasterizer.h
    renderer_vulkan/vk_rasterizer_cache.cpp
    renderer_vulkan/vk_scheduler.cpp
    renderer_vulkan/vk_scheduler.h
    renderer_vulkan/vk_resource_pool.cpp
    renderer_vulkan/vk_resource_pool.h
    renderer_vulkan/vk_instance.cpp
    renderer_vulkan/vk_instance.h
    renderer_vulkan/vk_pipeline_cache.cpp
    renderer_vulkan/vk_pipeline_cache.h
    renderer_vulkan/vk_platform.cpp
    renderer_vulkan/vk_platform.h
    renderer_vulkan/vk_renderpass_cache.cpp
    renderer_vulkan/vk_renderpass_cache.h
    renderer_vulkan/vk_shader_gen.cpp
    renderer_vulkan/vk_shader_gen.h
    renderer_vulkan/vk_shader_gen_spv.cpp
    renderer_vulkan/vk_shader_gen_spv.h
    renderer_vulkan/vk_shader_util.cpp
    renderer_vulkan/vk_shader_util.h
    renderer_vulkan/vk_stream_buffer.cpp
    renderer_vulkan/vk_stream_buffer.h
    renderer_vulkan/vk_swapchain.cpp
    renderer_vulkan/vk_swapchain.h
    renderer_vulkan/vk_texture_mailbox.cpp
    renderer_vulkan/vk_texture_mailbox.h
    renderer_vulkan/vk_texture_runtime.cpp
    renderer_vulkan/vk_texture_runtime.h
    shader/debug_data.h
    shader/shader.cpp
    shader/shader.h
    shader/shader_interpreter.cpp
    shader/shader_interpreter.h
    shader/shader_jit_x64.cpp
    shader/shader_jit_x64_compiler.cpp
    shader/shader_jit_x64.h
    shader/shader_jit_x64_compiler.h
    shader/shader_uniforms.cpp
    shader/shader_uniforms.h
    swrasterizer/clipper.cpp
    swrasterizer/clipper.h
    swrasterizer/framebuffer.cpp
    swrasterizer/framebuffer.h
    swrasterizer/lighting.cpp
    swrasterizer/lighting.h
    swrasterizer/proctex.cpp
    swrasterizer/proctex.h
    swrasterizer/rasterizer.cpp
    swrasterizer/rasterizer.h
    swrasterizer/swrasterizer.cpp
    swrasterizer/swrasterizer.h
    swrasterizer/texturing.cpp
    swrasterizer/texturing.h
    texture/etc1.cpp
    texture/etc1.h
    texture/texture_decode.cpp
    texture/texture_decode.h
    utils.h
    vertex_loader.cpp
    vertex_loader.h
    video_core.cpp
    video_core.h
)

target_include_directories(video_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

create_target_directory_groups(video_core)

# Ignore nullability warnings generated from VMA
if (NOT MSVC)
    target_compile_options(vma INTERFACE -Wno-unused-variable -Wno-nullability-completeness)
endif()

target_link_libraries(video_core PUBLIC common core)
set_target_properties(video_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${ENABLE_LTO})

add_dependencies(video_core host_shaders)
target_include_directories(video_core PRIVATE ${HOST_SHADERS_INCLUDE})
target_link_libraries(video_core PRIVATE vulkan-headers vma sirit SPIRV glslang glad)
target_link_libraries(video_core PRIVATE nihstro-headers Boost::serialization glm::glm)

if ("x86_64" IN_LIST ARCHITECTURE)
    target_link_libraries(video_core PUBLIC xbyak)
endif()

if (CITRA_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(video_core PRIVATE precompiled_headers.h)
endif()
