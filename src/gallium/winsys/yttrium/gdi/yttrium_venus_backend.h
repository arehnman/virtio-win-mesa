/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_VENUS_BACKEND_H
#define YTTRIUM_VENUS_BACKEND_H

#include "yttrium_venus.h"

struct yttrium_venus_backend {
   const char *name;

   void *(*create)(struct gdikmt_device *device);
   struct gdikmt_context *(*get_kmt_context)(void *ctx);
   void (*destroy)(void *ctx);

   bool (*create_shader_module)(void *ctx,
                                struct yttrium_venus_object *obj,
                                VkShaderModule *shader,
                                const uint32_t *code,
                                size_t code_size,
                                const char *label,
                                VkResult *out_result);
   VkFormat (*pipe_format)(enum pipe_format format);
   void (*destroy_shader_module)(void *ctx,
                                 struct yttrium_venus_object *obj,
                                 VkShaderModule shader);

   bool (*pipeline_init)(void *ctx,
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
                         const struct yttrium_venus_draw_state *draw_state);
   bool (*supports_load_store_op_none)(void *ctx);
   bool (*supports_multisampled_render_to_single_sampled)(void *ctx,
                                                          uint32_t sample_count);
   bool (*supports_forced_sample_interlock)(void *ctx,
                                            uint32_t sample_count);
   VkSampleCountFlags (*framebuffer_color_sample_counts)(void *ctx);
   bool (*sampled_texture_format_supported)(void *ctx,
                                             enum pipe_format format,
                                             enum pipe_texture_target target);
   bool (*storage_image_without_format_supported)(void *ctx,
                                                  bool read, bool write);
   bool (*storage_image_formatless_format_supported)(void *ctx,
                                                      enum pipe_format format,
                                                      bool read, bool write);
   bool (*compute_pipeline_init)(
      void *ctx,
      struct yttrium_pipeline *pipeline,
      VkShaderModule compute_shader,
      const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
      uint32_t ubo_binding_count,
      const struct yttrium_venus_sampled_binding_layout *sampled_bindings,
      uint32_t sampled_binding_count,
      uint64_t storage_image_mask,
      uint64_t storage_buffer_mask);
   void (*pipeline_fini)(void *ctx, struct yttrium_pipeline *pipeline);

   bool (*draw_pipeline)(void *ctx,
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
                         const struct yttrium_venus_draw_state *draw_state);
   bool (*dispatch_compute)(void *ctx,
                            struct yttrium_pipeline *pipeline,
                            const struct yttrium_venus_sampled_image *sampled_images,
                            uint32_t sampled_image_count,
                            const struct yttrium_venus_storage_image *storage_images,
                            uint32_t storage_image_count,
                            const struct yttrium_venus_ubo_upload *ubo_uploads,
                            uint32_t ubo_upload_count,
                            uint32_t grid_x,
                            uint32_t grid_y,
                            uint32_t grid_z);

   bool (*clear_display)(void *ctx,
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
                         uint32_t layer_count);
   bool (*clear_display_rect)(void *ctx,
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
                              uint32_t height);
   bool (*draw_vertex_buffer_vertices)(
      void *ctx,
      struct yttrium_venus_resource *resource,
      uint32_t resource_id,
      const struct yttrium_venus_triangle_vertex *vertices,
      uint32_t vertex_count,
      const struct yttrium_venus_draw_state *draw_state);
   bool (*draw_textured_vertices)(
      void *ctx,
      struct yttrium_venus_resource *resource,
      uint32_t resource_id,
      struct yttrium_venus_resource *sampled,
      uint32_t sampled_resource_id,
      const struct yttrium_venus_textured_vertex *vertices,
      uint32_t vertex_count,
      const struct yttrium_venus_draw_state *draw_state);
   bool (*update_buffer)(void *ctx,
                         struct yttrium_venus_resource *resource,
                         uint64_t offset,
                         uint64_t size,
                         const void *data);
   bool (*clear_buffer)(void *ctx,
                        struct yttrium_venus_resource *resource,
                        uint64_t offset,
                        uint64_t size,
                        uint32_t value);

   bool (*copy_image_to_display_buffer)(void *ctx,
                                        struct yttrium_venus_resource *render,
                                        struct yttrium_venus_resource *scanout,
                                        uint32_t render_resource_id,
                                        uint32_t scanout_resource_id,
                                        uint32_t width,
                                        uint32_t height,
                                        enum pipe_format pipe_format,
                                        uint32_t scanout_stride);
   bool (*copy_image_region_to_display_buffer)(
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
      uint32_t scanout_stride);
   bool (*copy_image_region_aspect_to_display_buffer)(
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
      VkImageAspectFlags copy_aspect);
   bool (*copy_display_image)(void *ctx,
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
                              uint32_t height);
   bool (*copy_depth_stencil_image)(void *ctx,
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
                                    VkImageAspectFlags aspect_mask);
   bool (*flush)(void *ctx);
   bool (*flush_async)(void *ctx);
   bool (*flush_async_present_publish)(
      void *ctx,
      const struct yttrium_venus_present_publication *publication);
   uint64_t (*last_submit_order)(void *ctx);
   bool (*submit_order_complete)(void *ctx, uint64_t submit_order);
   bool (*wait_resource)(void *ctx,
                         struct yttrium_venus_resource *resource,
                         const char *label);
   bool (*blit_display_image)(void *ctx,
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
                              bool linear_filter);
   bool (*resolve_display_image)(void *ctx,
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
                                 uint32_t height);
   bool (*copy_buffer_to_display_image)(void *ctx,
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
                                        enum pipe_format pipe_format);
   bool (*copy_buffer_to_buffer)(void *ctx,
                                 struct yttrium_venus_resource *src,
                                 struct yttrium_venus_resource *dst,
                                 VkDeviceSize src_offset,
                                 VkDeviceSize dst_offset,
                                 VkDeviceSize size);

   bool (*create_display_buffer)(void *ctx,
                                 struct yttrium_venus_resource *resource,
                                 uint64_t allocation_size,
                                 uint64_t *out_memory_id);
   bool (*create_bind_buffer)(void *ctx,
                              struct yttrium_venus_resource *resource,
                              uint64_t allocation_size,
                              VkBufferUsageFlags usage,
                              uint64_t *out_memory_id);
   bool (*create_stream_output_buffer)(void *ctx,
                                       struct yttrium_venus_resource *resource,
                                       uint64_t allocation_size,
                                       uint64_t *out_memory_id);
   bool (*transform_feedback_enabled)(const void *ctx);
   bool (*transform_feedback_draw_enabled)(const void *ctx);
   bool (*vertex_attribute_divisor_supported)(void *ctx, uint32_t divisor);
   bool (*depth_clamp_enabled)(const void *ctx);
   bool (*logic_op_enabled)(const void *ctx);
   uint32_t (*max_dual_source_render_targets)(void *ctx);
   float (*max_sampler_anisotropy)(void *ctx);
   float (*max_sampler_lod_bias)(void *ctx);
   uint32_t (*max_transform_feedback_stride)(const void *ctx);
   uint32_t (*max_viewports)(void *ctx);
   uint32_t (*mipmap_precision_bits)(const void *ctx);

   bool (*create_sampled_buffer)(void *ctx,
                                 struct yttrium_venus_resource *resource,
                                 uint64_t allocation_size,
                                 enum pipe_format pipe_format,
                                 uint64_t *out_memory_id);
   bool (*ensure_null_sampled_buffer)(void *ctx,
                                      enum pipe_format pipe_format,
                                      uint64_t min_size,
                                      struct yttrium_venus_resource **out_resource,
                                      uint32_t *out_resource_id);
   bool (*create_display_image)(void *ctx,
                                struct yttrium_venus_resource *resource,
                                uint32_t width,
                                uint32_t height,
                                enum pipe_format pipe_format,
                                uint64_t min_allocation_size,
                                uint64_t *out_memory_id,
                                uint64_t *out_allocation_size);
   bool (*create_color_attachment_image)(void *ctx,
                                         struct yttrium_venus_resource *resource,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t levels,
                                         uint32_t layers,
                                         enum pipe_format pipe_format,
                                         uint64_t *out_allocation_size);
   bool (*create_sampled_texture_image)(void *ctx,
                                        struct yttrium_venus_resource *resource,
                                        enum pipe_texture_target target,
                                        uint32_t width,
                                        uint32_t height,
                                        uint32_t depth,
                                        uint32_t levels,
                                        uint32_t layers,
                                        enum pipe_format pipe_format,
                                        uint64_t *out_allocation_size);
   bool (*create_texture_image_for_bind)(void *ctx,
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
                                         uint64_t *out_allocation_size);
   bool (*create_depth_stencil_image)(void *ctx,
                                      struct yttrium_venus_resource *resource,
                                      uint32_t width,
                                      uint32_t height,
                                      enum pipe_format pipe_format,
                                      uint64_t *out_allocation_size);
   bool (*clear_depth_stencil)(void *ctx,
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
                               uint32_t height);
   bool (*import_display_image)(void *ctx,
                                struct yttrium_venus_resource *resource,
                                uint32_t resource_id,
                                uint32_t width,
                                uint32_t height,
                                enum pipe_format pipe_format,
                                uint64_t min_allocation_size,
                                uint64_t *out_memory_id,
                                uint64_t *out_allocation_size);
   bool (*import_texture_image_for_bind)(void *ctx,
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
                                         uint64_t *out_allocation_size);
   bool (*resource_fini)(void *ctx,
                         struct gdikmt_context *kmt_ctx,
                         struct yttrium_venus_resource *resource,
                         const struct yttrium_venus_allocation_snapshot *allocation);
};

extern const struct yttrium_venus_backend yttrium_venus2_backend;

#endif /* YTTRIUM_VENUS_BACKEND_H */
