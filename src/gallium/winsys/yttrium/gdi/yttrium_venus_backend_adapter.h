/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_VENUS_BACKEND_NAME
#error "YTTRIUM_VENUS_BACKEND_NAME must be defined before including this file"
#endif

static void *
backend_create(struct gdikmt_device *device)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create)(device);
}

static struct gdikmt_context *
backend_get_kmt_context(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(get_kmt_context)(ctx);
}

static void
backend_destroy(void *ctx)
{
   YTTRIUM_VENUS_BACKEND_SYM(destroy)(ctx);
}

static bool
backend_create_shader_module(void *ctx,
                             struct yttrium_venus_object *obj,
                             VkShaderModule *shader,
                             const uint32_t *code,
                             size_t code_size,
                             const char *label,
                             VkResult *out_result)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_shader_module)(
      ctx, obj, shader, code, code_size, label, out_result);
}

static VkFormat
backend_pipe_format(enum pipe_format format)
{
   return YTTRIUM_VENUS_BACKEND_SYM(pipe_format)(format);
}

static void
backend_destroy_shader_module(void *ctx,
                              struct yttrium_venus_object *obj,
                              VkShaderModule shader)
{
   YTTRIUM_VENUS_BACKEND_SYM(destroy_shader_module)(ctx, obj, shader);
}

static bool
backend_pipeline_init(void *ctx,
                      struct yttrium_pipeline *pipeline,
                      struct yttrium_venus_resource *resource,
                      uint32_t resource_id,
                      struct yttrium_venus_resource **color_resources,
                      const uint32_t *color_resource_ids,
                      uint32_t color_resource_count,
                      struct yttrium_venus_resource *depth_resource,
                      uint32_t depth_resource_id,
                      VkShaderModule vertex_shader,
                      VkShaderModule tess_ctrl_shader,
                      VkShaderModule tess_eval_shader,
                      VkShaderModule geometry_shader,
                      VkShaderModule fragment_shader,
                      const VkVertexInputBindingDescription *bindings,
                      uint32_t binding_count,
                      const uint32_t *binding_divisors,
                      const VkVertexInputAttributeDescription *attribs,
                      uint32_t attrib_count,
                      const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
                      uint32_t ubo_binding_count,
                      const struct yttrium_venus_sampled_binding_layout *sampled_bindings,
                      uint32_t sampled_binding_count,
                      uint64_t storage_image_mask,
                      uint64_t storage_buffer_mask,
                      VkShaderStageFlags storage_stage_flags,
                      const struct yttrium_venus_draw_state *draw_state)
{
   return YTTRIUM_VENUS_BACKEND_SYM(pipeline_init)(
      ctx, pipeline, resource, resource_id, color_resources,
      color_resource_ids, color_resource_count, depth_resource,
      depth_resource_id, vertex_shader, tess_ctrl_shader, tess_eval_shader,
      geometry_shader, fragment_shader,
      bindings, binding_count, binding_divisors, attribs, attrib_count,
      ubo_bindings, ubo_binding_count, sampled_bindings,
      sampled_binding_count, storage_image_mask, storage_buffer_mask,
      storage_stage_flags, draw_state);
}

static bool
backend_supports_load_store_op_none(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(supports_load_store_op_none)(ctx);
}

static bool
backend_supports_multisampled_render_to_single_sampled(void *ctx,
                                                       uint32_t sample_count)
{
   return YTTRIUM_VENUS_BACKEND_SYM(
      supports_multisampled_render_to_single_sampled)(ctx, sample_count);
}

static bool
backend_supports_forced_sample_interlock(void *ctx, uint32_t sample_count)
{
   return YTTRIUM_VENUS_BACKEND_SYM(
      supports_forced_sample_interlock)(ctx, sample_count);
}

#ifdef YTTRIUM_VENUS_BACKEND_HAS_FRAMEBUFFER_COLOR_SAMPLE_COUNTS
static VkSampleCountFlags
backend_framebuffer_color_sample_counts(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(framebuffer_color_sample_counts)(ctx);
}
#endif

#ifdef YTTRIUM_VENUS_BACKEND_HAS_SAMPLED_TEXTURE_FORMAT_CAPS
static bool
backend_sampled_texture_format_supported(void *ctx,
                                         enum pipe_format format,
                                         enum pipe_texture_target target)
{
   return YTTRIUM_VENUS_BACKEND_SYM(sampled_texture_format_supported)(
      ctx, format, target);
}
#endif

#ifdef YTTRIUM_VENUS_BACKEND_HAS_STORAGE_IMAGE_FORMATLESS_CAPS
static bool
backend_storage_image_without_format_supported(void *ctx,
                                               bool read, bool write)
{
   return YTTRIUM_VENUS_BACKEND_SYM(storage_image_without_format_supported)(
      ctx, read, write);
}

static bool
backend_storage_image_formatless_format_supported(void *ctx,
                                                   enum pipe_format format,
                                                   bool read, bool write)
{
   return YTTRIUM_VENUS_BACKEND_SYM(storage_image_formatless_format_supported)(
      ctx, format, read, write);
}
#endif

static bool
backend_compute_pipeline_init(
   void *ctx,
   struct yttrium_pipeline *pipeline,
   VkShaderModule compute_shader,
   const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
   uint32_t ubo_binding_count,
   const struct yttrium_venus_sampled_binding_layout *sampled_bindings,
   uint32_t sampled_binding_count,
   uint64_t storage_image_mask,
   uint64_t storage_buffer_mask)
{
   return YTTRIUM_VENUS_BACKEND_SYM(compute_pipeline_init)(
      ctx, pipeline, compute_shader, ubo_bindings, ubo_binding_count,
      sampled_bindings, sampled_binding_count, storage_image_mask,
      storage_buffer_mask);
}

static void
backend_pipeline_fini(void *ctx, struct yttrium_pipeline *pipeline)
{
   YTTRIUM_VENUS_BACKEND_SYM(pipeline_fini)(ctx, pipeline);
}

static bool
backend_draw_pipeline(void *ctx,
                      struct yttrium_venus_resource *resource,
                      uint32_t resource_id,
                      struct yttrium_venus_resource **color_resources,
                      const uint32_t *color_resource_ids,
                      uint32_t color_resource_count,
                      struct yttrium_venus_resource *depth_resource,
                      uint32_t depth_resource_id,
                      struct yttrium_pipeline *pipeline,
                      const struct yttrium_venus_sampled_image *sampled_images,
                      uint32_t sampled_image_count,
                      const struct yttrium_venus_storage_image *storage_images,
                      uint32_t storage_image_count,
                      const struct yttrium_venus_vertex_upload *vertex_uploads,
                      uint32_t vertex_upload_count,
                      uint32_t vertex_count,
                      uint32_t instance_count,
                      const void *index_data,
                      size_t index_data_size,
                      struct yttrium_venus_resource *index_resource,
                      uint32_t index_resource_id,
                      VkDeviceSize index_buffer_offset,
                      uint32_t index_count,
                      VkIndexType index_type,
                      bool index_host_write_pending,
                      int32_t vertex_offset,
                      const struct yttrium_venus_ubo_upload *ubo_uploads,
                      uint32_t ubo_upload_count,
                      struct yttrium_venus_stream_output_target *so_targets,
                      uint32_t so_target_count,
                      const struct yttrium_venus_stream_output_target *draw_auto_target,
                      uint32_t draw_auto_stride,
                      const struct yttrium_venus_draw_state *draw_state)
{
   return YTTRIUM_VENUS_BACKEND_SYM(draw_pipeline)(
      ctx, resource, resource_id, color_resources, color_resource_ids,
      color_resource_count, depth_resource, depth_resource_id, pipeline,
      sampled_images, sampled_image_count, storage_images, storage_image_count,
      vertex_uploads, vertex_upload_count, vertex_count, instance_count,
      index_data, index_data_size, index_resource, index_resource_id,
      index_buffer_offset, index_count, index_type, index_host_write_pending,
      vertex_offset,
      ubo_uploads, ubo_upload_count, so_targets, so_target_count,
      draw_auto_target, draw_auto_stride, draw_state);
}

static bool
backend_dispatch_compute(void *ctx,
                         struct yttrium_pipeline *pipeline,
                         const struct yttrium_venus_sampled_image *sampled_images,
                         uint32_t sampled_image_count,
                         const struct yttrium_venus_storage_image *storage_images,
                         uint32_t storage_image_count,
                         const struct yttrium_venus_ubo_upload *ubo_uploads,
                         uint32_t ubo_upload_count,
                         uint32_t grid_x,
                         uint32_t grid_y,
                         uint32_t grid_z)
{
   return YTTRIUM_VENUS_BACKEND_SYM(dispatch_compute)(
      ctx, pipeline, sampled_images, sampled_image_count, storage_images,
      storage_image_count, ubo_uploads, ubo_upload_count, grid_x, grid_y, grid_z);
}

static bool
backend_clear_display(void *ctx,
                      struct gdikmt_context *kmt_ctx,
                      struct yttrium_venus_resource *resource,
                      uint32_t resource_id,
                      uint32_t width,
                      uint32_t height,
                      enum pipe_format pipe_format,
                      uint64_t allocation_size,
                      const union pipe_color_union *color,
                      uint32_t level,
                      uint32_t first_layer,
                      uint32_t layer_count)
{
   return YTTRIUM_VENUS_BACKEND_SYM(clear_display)(
      ctx, kmt_ctx, resource, resource_id, width, height, pipe_format,
      allocation_size, color, level, first_layer, layer_count);
}

#ifdef YTTRIUM_VENUS_BACKEND_HAS_CLEAR_DISPLAY_RECT
static bool
backend_clear_display_rect(void *ctx,
                           struct gdikmt_context *kmt_ctx,
                           struct yttrium_venus_resource *resource,
                           uint32_t resource_id,
                           enum pipe_format pipe_format,
                           const union pipe_color_union *color,
                           uint32_t level,
                           uint32_t first_layer,
                           uint32_t layer_count,
                           uint32_t dstx,
                           uint32_t dsty,
                           uint32_t width,
                           uint32_t height)
{
   return YTTRIUM_VENUS_BACKEND_SYM(clear_display_rect)(
      ctx, kmt_ctx, resource, resource_id, pipe_format, color, level,
      first_layer, layer_count, dstx, dsty, width, height);
}
#endif

static bool
backend_draw_vertex_buffer_vertices(
   void *ctx,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   const struct yttrium_venus_triangle_vertex *vertices,
   uint32_t vertex_count,
   const struct yttrium_venus_draw_state *draw_state)
{
   return YTTRIUM_VENUS_BACKEND_SYM(draw_vertex_buffer_vertices)(
      ctx, resource, resource_id, vertices, vertex_count, draw_state);
}

static bool
backend_draw_textured_vertices(
   void *ctx,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   struct yttrium_venus_resource *sampled,
   uint32_t sampled_resource_id,
   const struct yttrium_venus_textured_vertex *vertices,
   uint32_t vertex_count,
   const struct yttrium_venus_draw_state *draw_state)
{
   return YTTRIUM_VENUS_BACKEND_SYM(draw_textured_vertices)(
      ctx, resource, resource_id, sampled, sampled_resource_id, vertices,
      vertex_count, draw_state);
}

static bool
backend_update_buffer(void *ctx,
                      struct yttrium_venus_resource *resource,
                      uint64_t offset,
                      uint64_t size,
                      const void *data)
{
   return YTTRIUM_VENUS_BACKEND_SYM(update_buffer)(
      ctx, resource, offset, size, data);
}

static bool
backend_clear_buffer(void *ctx,
                     struct yttrium_venus_resource *resource,
                     uint64_t offset,
                     uint64_t size,
                     uint32_t value)
{
   return YTTRIUM_VENUS_BACKEND_SYM(clear_buffer)(
      ctx, resource, offset, size, value);
}

static bool
backend_copy_image_to_display_buffer(void *ctx,
                                     struct yttrium_venus_resource *render,
                                     struct yttrium_venus_resource *scanout,
                                     uint32_t render_resource_id,
                                     uint32_t scanout_resource_id,
                                     uint32_t width,
                                     uint32_t height,
                                     enum pipe_format pipe_format,
                                     uint32_t scanout_stride)
{
   return YTTRIUM_VENUS_BACKEND_SYM(copy_image_to_display_buffer)(
      ctx, render, scanout, render_resource_id, scanout_resource_id,
      width, height, pipe_format, scanout_stride);
}

static bool
backend_copy_image_region_to_display_buffer(
   void *ctx,
   struct yttrium_venus_resource *render,
   struct yttrium_venus_resource *scanout,
   uint32_t render_resource_id,
   uint32_t scanout_resource_id,
   uint32_t src_level,
   uint32_t src_layer,
   uint32_t src_x,
   uint32_t src_y,
   uint32_t dst_x,
   uint32_t dst_y,
   uint32_t width,
   uint32_t height,
   enum pipe_format pipe_format,
   uint32_t scanout_stride)
{
   return YTTRIUM_VENUS_BACKEND_SYM(copy_image_region_to_display_buffer)(
      ctx, render, scanout, render_resource_id, scanout_resource_id,
      src_level, src_layer, src_x, src_y, dst_x, dst_y, width, height,
      pipe_format, scanout_stride);
}

static bool
backend_copy_image_region_aspect_to_display_buffer(
   void *ctx,
   struct yttrium_venus_resource *render,
   struct yttrium_venus_resource *scanout,
   uint32_t render_resource_id,
   uint32_t scanout_resource_id,
   uint32_t src_level,
   uint32_t src_layer,
   uint32_t src_x,
   uint32_t src_y,
   uint32_t dst_x,
   uint32_t dst_y,
   uint32_t width,
   uint32_t height,
   enum pipe_format pipe_format,
   uint32_t scanout_stride,
   VkImageAspectFlags copy_aspect)
{
   return YTTRIUM_VENUS_BACKEND_SYM(copy_image_region_aspect_to_display_buffer)(
      ctx, render, scanout, render_resource_id, scanout_resource_id,
      src_level, src_layer, src_x, src_y, dst_x, dst_y, width, height,
      pipe_format, scanout_stride, copy_aspect);
}

static bool
backend_copy_display_image(void *ctx,
                           struct yttrium_venus_resource *src,
                           struct yttrium_venus_resource *dst,
                           uint32_t src_resource_id,
                           uint32_t dst_resource_id,
                           uint32_t src_level,
                           uint32_t src_x,
                           uint32_t src_y,
                           uint32_t src_layer,
                           uint32_t dst_level,
                           uint32_t dst_x,
                           uint32_t dst_y,
                           uint32_t dst_layer,
                           uint32_t width,
                           uint32_t height)
{
   return YTTRIUM_VENUS_BACKEND_SYM(copy_display_image)(
      ctx, src, dst, src_resource_id, dst_resource_id, src_level, src_x,
      src_y, src_layer, dst_level, dst_x, dst_y, dst_layer, width, height);
}

static bool
backend_copy_depth_stencil_image(void *ctx,
                                 struct yttrium_venus_resource *src,
                                 struct yttrium_venus_resource *dst,
                                 uint32_t src_resource_id,
                                 uint32_t dst_resource_id,
                                 uint32_t src_level,
                                 uint32_t src_x,
                                 uint32_t src_y,
                                 uint32_t src_layer,
                                 uint32_t dst_level,
                                 uint32_t dst_x,
                                 uint32_t dst_y,
                                 uint32_t dst_layer,
                                 uint32_t width,
                                 uint32_t height,
                                 VkImageAspectFlags aspect_mask)
{
   return YTTRIUM_VENUS_BACKEND_SYM(copy_depth_stencil_image)(
      ctx, src, dst, src_resource_id, dst_resource_id, src_level, src_x,
      src_y, src_layer, dst_level, dst_x, dst_y, dst_layer, width, height,
      aspect_mask);
}

static bool
backend_flush(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(flush)(ctx);
}

static bool
backend_flush_async(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(flush_async)(ctx);
}

static bool
backend_flush_async_present_publish(
   void *ctx,
   const struct yttrium_venus_present_publication *publication)
{
   return YTTRIUM_VENUS_BACKEND_SYM(flush_async_present_publish)(ctx,
                                                                 publication);
}

static uint64_t
backend_last_submit_order(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(last_submit_order)(ctx);
}

static bool
backend_submit_order_complete(void *ctx, uint64_t submit_order)
{
   return YTTRIUM_VENUS_BACKEND_SYM(submit_order_complete)(ctx,
                                                            submit_order);
}

static bool
backend_wait_resource(void *ctx,
                      struct yttrium_venus_resource *resource,
                      const char *label)
{
   return YTTRIUM_VENUS_BACKEND_SYM(wait_resource)(ctx, resource, label);
}

static bool
backend_blit_display_image(void *ctx,
                           struct yttrium_venus_resource *src,
                           struct yttrium_venus_resource *dst,
                           uint32_t src_resource_id,
                           uint32_t dst_resource_id,
                           uint32_t src_level,
                           uint32_t src_x,
                           uint32_t src_y,
                           uint32_t src_layer,
                           uint32_t src_width,
                           uint32_t src_height,
                           uint32_t src_depth,
                           uint32_t dst_level,
                           uint32_t dst_x,
                           uint32_t dst_y,
                           uint32_t dst_layer,
                           uint32_t dst_width,
                           uint32_t dst_height,
                           uint32_t dst_depth,
                           bool linear_filter)
{
   return YTTRIUM_VENUS_BACKEND_SYM(blit_display_image)(
      ctx, src, dst, src_resource_id, dst_resource_id, src_level, src_x,
      src_y, src_layer, src_width, src_height, src_depth, dst_level, dst_x,
      dst_y, dst_layer, dst_width, dst_height, dst_depth, linear_filter);
}

static bool
backend_resolve_display_image(void *ctx,
                              struct yttrium_venus_resource *src,
                              struct yttrium_venus_resource *dst,
                              uint32_t src_resource_id,
                              uint32_t dst_resource_id,
                              uint32_t src_level,
                              uint32_t src_x,
                              uint32_t src_y,
                              uint32_t src_layer,
                              uint32_t dst_level,
                              uint32_t dst_x,
                              uint32_t dst_y,
                              uint32_t dst_layer,
                              enum pipe_format resolve_format,
                              uint32_t width,
                              uint32_t height)
{
   return YTTRIUM_VENUS_BACKEND_SYM(resolve_display_image)(
      ctx, src, dst, src_resource_id, dst_resource_id, src_level, src_x,
      src_y, src_layer, dst_level, dst_x, dst_y, dst_layer, resolve_format,
      width, height);
}

static bool
backend_copy_buffer_to_display_image(void *ctx,
                                     struct yttrium_venus_resource *src,
                                     struct yttrium_venus_resource *dst,
                                     uint32_t src_resource_id,
                                     uint32_t dst_resource_id,
                                     VkDeviceSize src_offset,
                                     uint32_t src_stride,
                                     uint32_t src_layer_stride,
                                     uint32_t dst_level,
                                     uint32_t dst_x,
                                     uint32_t dst_y,
                                     uint32_t dst_layer,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t depth,
                                     enum pipe_format pipe_format)
{
   return YTTRIUM_VENUS_BACKEND_SYM(copy_buffer_to_display_image)(
      ctx, src, dst, src_resource_id, dst_resource_id, src_offset,
      src_stride, src_layer_stride, dst_level, dst_x, dst_y, dst_layer,
      width, height, depth, pipe_format);
}

static bool
backend_copy_buffer_to_buffer(void *ctx,
                              struct yttrium_venus_resource *src,
                              struct yttrium_venus_resource *dst,
                              VkDeviceSize src_offset,
                              VkDeviceSize dst_offset,
                              VkDeviceSize size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(copy_buffer_to_buffer)(
      ctx, src, dst, src_offset, dst_offset, size);
}

static bool
backend_create_display_buffer(void *ctx,
                              struct yttrium_venus_resource *resource,
                              uint64_t allocation_size,
                              uint64_t *out_memory_id)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_display_buffer)(
      ctx, resource, allocation_size, out_memory_id);
}

static bool
backend_create_stream_output_buffer(void *ctx,
                                    struct yttrium_venus_resource *resource,
                                    uint64_t allocation_size,
                                    uint64_t *out_memory_id)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_stream_output_buffer)(
      ctx, resource, allocation_size, out_memory_id);
}

static bool
backend_create_bind_buffer(void *ctx,
                           struct yttrium_venus_resource *resource,
                           uint64_t allocation_size,
                           VkBufferUsageFlags usage,
                           uint64_t *out_memory_id)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_bind_buffer)(
      ctx, resource, allocation_size, usage, out_memory_id);
}

static bool
backend_transform_feedback_enabled(const void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(transform_feedback_enabled)(ctx);
}

static bool
backend_transform_feedback_draw_enabled(const void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(transform_feedback_draw_enabled)(ctx);
}

static bool
backend_vertex_attribute_divisor_supported(void *ctx, uint32_t divisor)
{
   return YTTRIUM_VENUS_BACKEND_SYM(vertex_attribute_divisor_supported)(
      ctx, divisor);
}

static bool
backend_depth_clamp_enabled(const void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(depth_clamp_enabled)(ctx);
}

static bool
backend_logic_op_enabled(const void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(logic_op_enabled)(ctx);
}

static uint32_t
backend_max_dual_source_render_targets(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(max_dual_source_render_targets)(ctx);
}

static float
backend_max_sampler_anisotropy(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(max_sampler_anisotropy)(ctx);
}

static float
backend_max_sampler_lod_bias(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(max_sampler_lod_bias)(ctx);
}

static uint32_t
backend_max_transform_feedback_stride(const void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(max_transform_feedback_stride)(ctx);
}

static uint32_t
backend_max_viewports(void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(max_viewports)(ctx);
}

static uint32_t
backend_mipmap_precision_bits(const void *ctx)
{
   return YTTRIUM_VENUS_BACKEND_SYM(mipmap_precision_bits)(ctx);
}

static bool
backend_create_sampled_buffer(void *ctx,
                              struct yttrium_venus_resource *resource,
                              uint64_t allocation_size,
                              enum pipe_format pipe_format,
                              uint64_t *out_memory_id)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_sampled_buffer)(
      ctx, resource, allocation_size, pipe_format, out_memory_id);
}

static bool
backend_ensure_null_sampled_buffer(void *ctx,
                                   enum pipe_format pipe_format,
                                   uint64_t min_size,
                                   struct yttrium_venus_resource **out_resource,
                                   uint32_t *out_resource_id)
{
   return YTTRIUM_VENUS_BACKEND_SYM(ensure_null_sampled_buffer)(
      ctx, pipe_format, min_size, out_resource, out_resource_id);
}

static bool
backend_create_display_image(void *ctx,
                             struct yttrium_venus_resource *resource,
                             uint32_t width,
                             uint32_t height,
                             enum pipe_format pipe_format,
                             uint64_t min_allocation_size,
                             uint64_t *out_memory_id,
                             uint64_t *out_allocation_size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_display_image)(
      ctx, resource, width, height, pipe_format, min_allocation_size,
      out_memory_id, out_allocation_size);
}

static bool
backend_create_color_attachment_image(void *ctx,
                                      struct yttrium_venus_resource *resource,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t levels,
                                      uint32_t layers,
                                      enum pipe_format pipe_format,
                                      uint64_t *out_allocation_size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_color_attachment_image)(
      ctx, resource, width, height, levels, layers, pipe_format,
      out_allocation_size);
}

static bool
backend_create_sampled_texture_image(void *ctx,
                                     struct yttrium_venus_resource *resource,
                                     enum pipe_texture_target target,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t depth,
                                     uint32_t levels,
                                     uint32_t layers,
                                     enum pipe_format pipe_format,
                                     uint64_t *out_allocation_size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_sampled_texture_image)(
      ctx, resource, target, width, height, depth, levels, layers,
      pipe_format, out_allocation_size);
}

static bool
backend_create_texture_image_for_bind(void *ctx,
                                      struct yttrium_venus_resource *resource,
                                      enum pipe_texture_target target,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t depth,
                                      uint32_t levels,
                                      uint32_t layers,
                                      enum pipe_format pipe_format,
                                      unsigned sample_count,
                                      unsigned bind,
                                      uint64_t *out_allocation_size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_texture_image_for_bind)(
      ctx, resource, target, width, height, depth, levels, layers,
      pipe_format, sample_count, bind, out_allocation_size);
}

static bool
backend_create_depth_stencil_image(void *ctx,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t width,
                                   uint32_t height,
                                   enum pipe_format pipe_format,
                                   uint64_t *out_allocation_size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(create_depth_stencil_image)(
      ctx, resource, width, height, pipe_format, out_allocation_size);
}

static bool
backend_clear_depth_stencil(void *ctx,
                            struct yttrium_venus_resource *resource,
                            uint32_t resource_id,
                            unsigned clear_flags,
                            double depth,
                            unsigned stencil,
                            uint32_t level,
                            uint32_t first_layer,
                            uint32_t layer_count,
                            uint32_t dstx,
                            uint32_t dsty,
                            uint32_t width,
                            uint32_t height)
{
   return YTTRIUM_VENUS_BACKEND_SYM(clear_depth_stencil)(
      ctx, resource, resource_id, clear_flags, depth, stencil, level,
      first_layer, layer_count, dstx, dsty, width, height);
}

static bool
backend_import_display_image(void *ctx,
                             struct yttrium_venus_resource *resource,
                             uint32_t resource_id,
                             uint32_t width,
                             uint32_t height,
                             enum pipe_format pipe_format,
                             uint64_t min_allocation_size,
                             uint64_t *out_memory_id,
                             uint64_t *out_allocation_size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(import_display_image)(
      ctx, resource, resource_id, width, height, pipe_format,
      min_allocation_size, out_memory_id, out_allocation_size);
}

static bool
backend_import_texture_image_for_bind(void *ctx,
                                      struct yttrium_venus_resource *resource,
                                      uint32_t resource_id,
                                      enum pipe_texture_target target,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t depth,
                                      uint32_t levels,
                                      uint32_t layers,
                                      enum pipe_format pipe_format,
                                      unsigned sample_count,
                                      unsigned bind,
                                      uint64_t min_allocation_size,
                                      uint64_t *out_memory_id,
                                      uint64_t *out_allocation_size)
{
   return YTTRIUM_VENUS_BACKEND_SYM(import_texture_image_for_bind)(
      ctx, resource, resource_id, target, width, height, depth, levels,
      layers, pipe_format, sample_count, bind, min_allocation_size,
      out_memory_id, out_allocation_size);
}

static bool
backend_resource_fini(void *ctx,
                      struct gdikmt_context *kmt_ctx,
                      struct yttrium_venus_resource *resource,
                      const struct yttrium_venus_allocation_snapshot *allocation)
{
   return YTTRIUM_VENUS_BACKEND_SYM(resource_fini)(
      ctx, kmt_ctx, resource, allocation);
}

const struct yttrium_venus_backend YTTRIUM_VENUS_BACKEND_SYM(backend) = {
   .name = YTTRIUM_VENUS_BACKEND_NAME,
   .create = backend_create,
   .get_kmt_context = backend_get_kmt_context,
   .destroy = backend_destroy,
   .create_shader_module = backend_create_shader_module,
   .pipe_format = backend_pipe_format,
   .destroy_shader_module = backend_destroy_shader_module,
   .pipeline_init = backend_pipeline_init,
   .supports_load_store_op_none = backend_supports_load_store_op_none,
   .supports_multisampled_render_to_single_sampled =
      backend_supports_multisampled_render_to_single_sampled,
   .supports_forced_sample_interlock =
      backend_supports_forced_sample_interlock,
#ifdef YTTRIUM_VENUS_BACKEND_HAS_FRAMEBUFFER_COLOR_SAMPLE_COUNTS
   .framebuffer_color_sample_counts = backend_framebuffer_color_sample_counts,
#endif
#ifdef YTTRIUM_VENUS_BACKEND_HAS_SAMPLED_TEXTURE_FORMAT_CAPS
   .sampled_texture_format_supported = backend_sampled_texture_format_supported,
#endif
#ifdef YTTRIUM_VENUS_BACKEND_HAS_STORAGE_IMAGE_FORMATLESS_CAPS
   .storage_image_without_format_supported =
      backend_storage_image_without_format_supported,
   .storage_image_formatless_format_supported =
      backend_storage_image_formatless_format_supported,
#endif
   .compute_pipeline_init = backend_compute_pipeline_init,
   .pipeline_fini = backend_pipeline_fini,
   .draw_pipeline = backend_draw_pipeline,
   .dispatch_compute = backend_dispatch_compute,
   .clear_display = backend_clear_display,
#ifdef YTTRIUM_VENUS_BACKEND_HAS_CLEAR_DISPLAY_RECT
   .clear_display_rect = backend_clear_display_rect,
#endif
   .draw_vertex_buffer_vertices = backend_draw_vertex_buffer_vertices,
   .draw_textured_vertices = backend_draw_textured_vertices,
   .update_buffer = backend_update_buffer,
   .clear_buffer = backend_clear_buffer,
   .copy_image_to_display_buffer = backend_copy_image_to_display_buffer,
   .copy_image_region_to_display_buffer =
      backend_copy_image_region_to_display_buffer,
   .copy_image_region_aspect_to_display_buffer =
      backend_copy_image_region_aspect_to_display_buffer,
   .copy_display_image = backend_copy_display_image,
   .copy_depth_stencil_image = backend_copy_depth_stencil_image,
   .flush = backend_flush,
   .flush_async = backend_flush_async,
   .flush_async_present_publish = backend_flush_async_present_publish,
   .last_submit_order = backend_last_submit_order,
   .submit_order_complete = backend_submit_order_complete,
   .wait_resource = backend_wait_resource,
   .blit_display_image = backend_blit_display_image,
   .resolve_display_image = backend_resolve_display_image,
   .copy_buffer_to_display_image = backend_copy_buffer_to_display_image,
   .copy_buffer_to_buffer = backend_copy_buffer_to_buffer,
   .create_display_buffer = backend_create_display_buffer,
   .create_bind_buffer = backend_create_bind_buffer,
   .create_stream_output_buffer = backend_create_stream_output_buffer,
   .transform_feedback_enabled = backend_transform_feedback_enabled,
   .transform_feedback_draw_enabled = backend_transform_feedback_draw_enabled,
   .vertex_attribute_divisor_supported =
      backend_vertex_attribute_divisor_supported,
   .depth_clamp_enabled = backend_depth_clamp_enabled,
   .logic_op_enabled = backend_logic_op_enabled,
   .max_dual_source_render_targets = backend_max_dual_source_render_targets,
   .max_sampler_anisotropy = backend_max_sampler_anisotropy,
   .max_sampler_lod_bias = backend_max_sampler_lod_bias,
   .max_transform_feedback_stride = backend_max_transform_feedback_stride,
   .max_viewports = backend_max_viewports,
   .mipmap_precision_bits = backend_mipmap_precision_bits,
   .create_sampled_buffer = backend_create_sampled_buffer,
   .ensure_null_sampled_buffer = backend_ensure_null_sampled_buffer,
   .create_display_image = backend_create_display_image,
   .create_color_attachment_image = backend_create_color_attachment_image,
   .create_sampled_texture_image = backend_create_sampled_texture_image,
   .create_texture_image_for_bind = backend_create_texture_image_for_bind,
   .create_depth_stencil_image = backend_create_depth_stencil_image,
   .clear_depth_stencil = backend_clear_depth_stencil,
   .import_display_image = backend_import_display_image,
   .import_texture_image_for_bind = backend_import_texture_image_for_bind,
   .resource_fini = backend_resource_fini,
};
