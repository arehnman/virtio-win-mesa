/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>

#include "util/u_memory.h"

#include "yttrium_gdi_public.h"
#include "yttrium_trace.h"
#include "yttrium_options.h"
#include "yttrium_venus_backend.h"

#define YTTRIUM_VENUS_BACKEND_ENV "D3D10UMD_YTTRIUM_VENUS_BACKEND"

struct yttrium_venus {
   const struct yttrium_venus_backend *backend;
   void *ctx;
};

static volatile LONG yttrium_venus_backend_selection_state;
static const struct yttrium_venus_backend *yttrium_venus_selected_backend;
static _Thread_local const char *yttrium_venus_tls_flush_label;

static const struct yttrium_venus_backend *
yttrium_venus_select_backend_uncached(void)
{
   const char *backend_name =
      yttrium_gdi_debug_get_option(YTTRIUM_VENUS_BACKEND_ENV, NULL);
   if (backend_name && backend_name[0] && strcmp(backend_name, "venus2") != 0) {
      YTTRIUM_WARN("yttrium: ERROR: unsupported Venus backend owner=venus-router env=%s value=%s supported=venus2 action=fail-device-creation\n",
                   YTTRIUM_VENUS_BACKEND_ENV, backend_name);
      return NULL;
   }

   return &yttrium_venus2_backend;
}

static const struct yttrium_venus_backend *
yttrium_venus_select_backend(void)
{
   LONG state = InterlockedCompareExchange(
      &yttrium_venus_backend_selection_state, 1, 0);
   if (state == 0) {
      yttrium_venus_selected_backend =
         yttrium_venus_select_backend_uncached();
      InterlockedExchange(&yttrium_venus_backend_selection_state, 2);
      return yttrium_venus_selected_backend;
   }

   while (yttrium_venus_backend_selection_state != 2)
      Sleep(0);

   return yttrium_venus_selected_backend;
}

static const struct yttrium_venus_backend *
yttrium_venus_backend_or_null(struct yttrium_venus *venus)
{
   return venus ? venus->backend : NULL;
}

static const char *
yttrium_venus_current_process_name(char *buffer, size_t size)
{
   DWORD len;
   const char *slash;
   const char *backslash;
   const char *base;

   if (!buffer || !size)
      return "(unknown)";

   len = GetModuleFileNameA(NULL, buffer, (DWORD)size);
   if (!len || len >= size) {
      buffer[0] = '\0';
      return "(unknown)";
   }

   slash = strrchr(buffer, '/');
   backslash = strrchr(buffer, '\\');
   if (slash && backslash)
      base = slash > backslash ? slash : backslash;
   else
      base = slash ? slash : backslash;
   return base ? base + 1 : buffer;
}

static void
yttrium_venus_log_backend_selection(const struct yttrium_venus_backend *backend)
{
   char process_path[MAX_PATH];
   const char *process =
      yttrium_venus_current_process_name(process_path, sizeof(process_path));
   const char *backend_option =
      yttrium_gdi_debug_get_option(YTTRIUM_VENUS_BACKEND_ENV, NULL);
   bool config_loaded = false;
   bool config_found = false;
   const char *config_path = NULL;
   unsigned config_entries = 0;

   yttrium_gdi_debug_get_config_status(&config_loaded, &config_found,
                                       &config_path, &config_entries);

   yttrium_gdi_user_logf(
      "yttrium: Venus router selected backend=%s process=%s "
      "backend_option=%s etw=%u verbose_etw_text=%u "
      "config_loaded=%u config_found=%u config_entries=%u config_path=%s\n",
      backend->name,
      process,
      backend_option ? backend_option : "(null)",
      yttrium_trace_is_enabled() ? 1 : 0,
      yttrium_trace_verbose_etw_text_enabled() ? 1 : 0,
      config_loaded ? 1 : 0,
      config_found ? 1 : 0,
      config_entries,
      config_path ? config_path : "(null)");

   YTTRIUM_LOG(
      "yttrium: Venus router selected backend=%s process=%s "
      "backend_option=%s etw=%u verbose_etw_text=%u\n",
      backend->name,
      process,
      backend_option ? backend_option : "(null)",
      yttrium_trace_is_enabled() ? 1 : 0,
      yttrium_trace_verbose_etw_text_enabled() ? 1 : 0);
}

struct yttrium_venus *
yttrium_venus_create(struct gdikmt_device *device)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_select_backend();
   if (!backend)
      return NULL;

   yttrium_venus_log_backend_selection(backend);

   void *ctx = backend->create(device);
   if (!ctx) {
      YTTRIUM_WARN("yttrium: Venus router backend create failed backend=%s\n",
                   backend->name);
      return NULL;
   }

   struct yttrium_venus *venus = CALLOC_STRUCT(yttrium_venus);
   if (!venus) {
      backend->destroy(ctx);
      return NULL;
   }

   venus->backend = backend;
   venus->ctx = ctx;
   return venus;
}

struct gdikmt_context *
yttrium_venus_get_kmt_context(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend ? backend->get_kmt_context(venus->ctx) : NULL;
}

void
yttrium_venus_destroy(struct yttrium_venus *venus)
{
   if (!venus)
      return;
   if (venus->backend && venus->ctx)
      venus->backend->destroy(venus->ctx);
   FREE(venus);
}

bool
yttrium_venus_create_shader_module(struct yttrium_venus *venus,
                                   struct yttrium_venus_object *obj,
                                   VkShaderModule *shader,
                                   const uint32_t *code,
                                   size_t code_size,
                                   const char *label,
                                   VkResult *out_result)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_shader_module(
      venus->ctx, obj, shader, code, code_size, label, out_result);
}

VkFormat
yttrium_venus_pipe_format(enum pipe_format format)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_select_backend();
   return backend ? backend->pipe_format(format) : VK_FORMAT_UNDEFINED;
}

void
yttrium_venus_destroy_shader_module(struct yttrium_venus *venus,
                                    struct yttrium_venus_object *obj,
                                    VkShaderModule shader)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   if (backend)
      backend->destroy_shader_module(venus->ctx, obj, shader);
}

bool
yttrium_venus_pipeline_init(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->pipeline_init(
      venus->ctx, pipeline, resource, resource_id, color_resources,
      color_resource_ids, color_resource_count, depth_resource,
      depth_resource_id, vertex_shader, tess_ctrl_shader, tess_eval_shader,
      geometry_shader, fragment_shader,
      bindings, binding_count, binding_divisors, attribs, attrib_count,
      ubo_bindings, ubo_binding_count, sampled_bindings,
      sampled_binding_count, storage_image_mask, storage_buffer_mask,
      storage_stage_flags, draw_state);
}

bool
yttrium_venus_supports_load_store_op_none(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->supports_load_store_op_none &&
      backend->supports_load_store_op_none(venus->ctx);
}

bool
yttrium_venus_supports_multisampled_render_to_single_sampled(
   struct yttrium_venus *venus,
   uint32_t sample_count)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend &&
      backend->supports_multisampled_render_to_single_sampled &&
      backend->supports_multisampled_render_to_single_sampled(venus->ctx,
                                                              sample_count);
}

bool
yttrium_venus_supports_forced_sample_interlock(
   struct yttrium_venus *venus,
   uint32_t sample_count)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->supports_forced_sample_interlock &&
      backend->supports_forced_sample_interlock(venus->ctx, sample_count);
}

VkSampleCountFlags
yttrium_venus_framebuffer_color_sample_counts(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->framebuffer_color_sample_counts ?
      backend->framebuffer_color_sample_counts(venus->ctx) : 0;
}

bool
yttrium_venus_sampled_texture_format_supported(
   struct yttrium_venus *venus,
   enum pipe_format format,
   enum pipe_texture_target target)
{
   static volatile LONG missing_capability_warned;
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);

   if (!backend || !backend->sampled_texture_format_supported) {
      if (InterlockedCompareExchange(&missing_capability_warned, 1, 0) == 0) {
         YTTRIUM_WARN("yttrium: ERROR: sampled texture format capability unavailable owner=venus-router reason=%s action=fail-closed\n",
                      backend ? "backend-callback-missing" :
                                "backend-unavailable");
      }
      return false;
   }

   return backend->sampled_texture_format_supported(venus->ctx, format,
                                                     target);
}

bool
yttrium_venus_storage_image_without_format_supported(
   struct yttrium_venus *venus, bool read, bool write)
{
   static volatile LONG missing_capability_warned;
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);

   if (!backend || !backend->storage_image_without_format_supported) {
      if (InterlockedCompareExchange(&missing_capability_warned, 1, 0) == 0) {
         YTTRIUM_WARN("yttrium: ERROR: formatless storage image capability unavailable owner=venus-router reason=%s action=fail-closed\n",
                      backend ? "backend-callback-missing" :
                                "backend-unavailable");
      }
      return false;
   }

   return backend->storage_image_without_format_supported(venus->ctx,
                                                           read, write);
}

bool
yttrium_venus_storage_image_formatless_format_supported(
   struct yttrium_venus *venus, enum pipe_format format,
   bool read, bool write)
{
   static volatile LONG missing_capability_warned;
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);

   if (!backend || !backend->storage_image_formatless_format_supported) {
      if (InterlockedCompareExchange(&missing_capability_warned, 1, 0) == 0) {
         YTTRIUM_WARN("yttrium: ERROR: formatless storage image format capability unavailable owner=venus-router reason=%s action=fail-closed\n",
                      backend ? "backend-callback-missing" :
                                "backend-unavailable");
      }
      return false;
   }

   return backend->storage_image_formatless_format_supported(venus->ctx,
                                                              format,
                                                              read, write);
}

bool
yttrium_venus_compute_pipeline_init(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   VkShaderModule compute_shader,
   const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
   uint32_t ubo_binding_count,
   const struct yttrium_venus_sampled_binding_layout *sampled_bindings,
   uint32_t sampled_binding_count,
   uint64_t storage_image_mask,
   uint64_t storage_buffer_mask)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->compute_pipeline_init(
      venus->ctx, pipeline, compute_shader, ubo_bindings, ubo_binding_count,
      sampled_bindings, sampled_binding_count, storage_image_mask,
      storage_buffer_mask);
}

void
yttrium_venus_pipeline_fini(struct yttrium_venus *venus,
                            struct yttrium_pipeline *pipeline)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   if (backend)
      backend->pipeline_fini(venus->ctx, pipeline);
}

bool
yttrium_venus_draw_pipeline(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->draw_pipeline(
      venus->ctx, resource, resource_id, color_resources, color_resource_ids,
      color_resource_count, depth_resource, depth_resource_id, pipeline,
      sampled_images, sampled_image_count, storage_images, storage_image_count,
      vertex_uploads, vertex_upload_count, vertex_count, instance_count,
      index_data, index_data_size, index_resource, index_resource_id,
      index_buffer_offset, index_count, index_type, index_host_write_pending,
      vertex_offset,
      ubo_uploads, ubo_upload_count, so_targets, so_target_count,
      draw_auto_target, draw_auto_stride, draw_state);
}

bool
yttrium_venus_dispatch_compute(
   struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->dispatch_compute(
      venus->ctx, pipeline, sampled_images, sampled_image_count, storage_images,
      storage_image_count, ubo_uploads, ubo_upload_count, grid_x, grid_y, grid_z);
}

bool
yttrium_venus_clear_display(struct yttrium_venus *venus,
                            struct gdikmt_context *ctx,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->clear_display(
      venus->ctx, ctx, resource, resource_id, width, height, pipe_format,
      allocation_size, color, level, first_layer, layer_count);
}

bool
yttrium_venus_clear_display_rect(struct yttrium_venus *venus,
                                 struct gdikmt_context *ctx,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->clear_display_rect &&
      backend->clear_display_rect(venus->ctx, ctx, resource, resource_id,
                                  pipe_format, color, level, first_layer,
                                  layer_count, dstx, dsty, width, height);
}

bool
yttrium_venus_draw_vertex_buffer_vertices(struct yttrium_venus *venus,
                                          struct yttrium_venus_resource *resource,
                                          uint32_t resource_id,
                                          const struct yttrium_venus_triangle_vertex *vertices,
                                          uint32_t vertex_count,
                                          const struct yttrium_venus_draw_state *draw_state)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->draw_vertex_buffer_vertices(
      venus->ctx, resource, resource_id, vertices, vertex_count, draw_state);
}

bool
yttrium_venus_draw_textured_vertices(struct yttrium_venus *venus,
                                     struct yttrium_venus_resource *resource,
                                     uint32_t resource_id,
                                     struct yttrium_venus_resource *sampled,
                                     uint32_t sampled_resource_id,
                                     const struct yttrium_venus_textured_vertex *vertices,
                                     uint32_t vertex_count,
                                     const struct yttrium_venus_draw_state *draw_state)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->draw_textured_vertices(
      venus->ctx, resource, resource_id, sampled, sampled_resource_id,
      vertices, vertex_count, draw_state);
}

bool
yttrium_venus_update_buffer(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                            uint64_t offset,
                            uint64_t size,
                            const void *data)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->update_buffer(
      venus->ctx, resource, offset, size, data);
}

bool
yttrium_venus_clear_buffer(struct yttrium_venus *venus,
                           struct yttrium_venus_resource *resource,
                           uint64_t offset,
                           uint64_t size,
                           uint32_t value)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->clear_buffer(
      venus->ctx, resource, offset, size, value);
}

bool
yttrium_venus_copy_image_to_display_buffer(struct yttrium_venus *venus,
                                           struct yttrium_venus_resource *render,
                                           struct yttrium_venus_resource *scanout,
                                           uint32_t render_resource_id,
                                           uint32_t scanout_resource_id,
                                           uint32_t width,
                                           uint32_t height,
                                           enum pipe_format pipe_format,
                                           uint32_t scanout_stride)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->copy_image_to_display_buffer(
      venus->ctx, render, scanout, render_resource_id, scanout_resource_id,
      width, height, pipe_format, scanout_stride);
}

bool
yttrium_venus_copy_image_region_to_display_buffer(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->copy_image_region_to_display_buffer(
      venus->ctx, render, scanout, render_resource_id, scanout_resource_id,
      src_level, src_layer, src_x, src_y, dst_x, dst_y, width, height,
      pipe_format, scanout_stride);
}

bool
yttrium_venus_copy_image_region_aspect_to_display_buffer(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->copy_image_region_aspect_to_display_buffer(
      venus->ctx, render, scanout, render_resource_id, scanout_resource_id,
      src_level, src_layer, src_x, src_y, dst_x, dst_y, width, height,
      pipe_format, scanout_stride, copy_aspect);
}

bool
yttrium_venus_copy_display_image(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->copy_display_image(
      venus->ctx, src, dst, src_resource_id, dst_resource_id, src_level,
      src_x, src_y, src_layer, dst_level, dst_x, dst_y, dst_layer,
      width, height);
}

bool
yttrium_venus_copy_depth_stencil_image(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->copy_depth_stencil_image(
      venus->ctx, src, dst, src_resource_id, dst_resource_id, src_level,
      src_x, src_y, src_layer, dst_level, dst_x, dst_y, dst_layer,
      width, height, aspect_mask);
}

bool
yttrium_venus_flush_labeled(struct yttrium_venus *venus, const char *label)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   if (!backend)
      return false;

   const char *old_label = yttrium_venus_tls_flush_label;
   yttrium_venus_tls_flush_label = label;
   const bool ok = backend->flush(venus->ctx);
   yttrium_venus_tls_flush_label = old_label;
   return ok;
}

bool
yttrium_venus_flush_async_labeled(struct yttrium_venus *venus,
                                  const char *label)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   if (!backend)
      return false;

   const char *old_label = yttrium_venus_tls_flush_label;
   yttrium_venus_tls_flush_label = label;
   const bool ok = backend->flush_async(venus->ctx);
   yttrium_venus_tls_flush_label = old_label;
   return ok;
}

bool
yttrium_venus_flush_async_present_publish_labeled(
   struct yttrium_venus *venus,
   const char *label,
   const struct yttrium_venus_present_publication *publication)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   if (!backend || !backend->flush_async_present_publish)
      return false;

   const char *old_label = yttrium_venus_tls_flush_label;
   yttrium_venus_tls_flush_label = label;
   const bool ok =
      backend->flush_async_present_publish(venus->ctx, publication);
   yttrium_venus_tls_flush_label = old_label;
   return ok;
}

bool
yttrium_venus_flush(struct yttrium_venus *venus)
{
   return yttrium_venus_flush_labeled(venus, "pipe context flush");
}

uint64_t
yttrium_venus_last_submit_order(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->last_submit_order ?
      backend->last_submit_order(venus->ctx) : 0;
}

bool
yttrium_venus_submit_order_complete(struct yttrium_venus *venus,
                                    uint64_t submit_order)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->submit_order_complete &&
      backend->submit_order_complete(venus->ctx, submit_order);
}

const char *
yttrium_venus_current_flush_label(void)
{
   return yttrium_venus_tls_flush_label;
}

bool
yttrium_venus_wait_resource(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                            const char *label)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->wait_resource(venus->ctx, resource, label);
}

bool
yttrium_venus_blit_display_image(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->blit_display_image(
      venus->ctx, src, dst, src_resource_id, dst_resource_id, src_level,
      src_x, src_y, src_layer, src_width, src_height, src_depth, dst_level,
      dst_x, dst_y, dst_layer, dst_width, dst_height, dst_depth,
      linear_filter);
}

bool
yttrium_venus_resolve_display_image(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->resolve_display_image(
      venus->ctx, src, dst, src_resource_id, dst_resource_id, src_level,
      src_x, src_y, src_layer, dst_level, dst_x, dst_y, dst_layer,
      resolve_format, width, height);
}

bool
yttrium_venus_copy_buffer_to_display_image(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->copy_buffer_to_display_image(
      venus->ctx, src, dst, src_resource_id, dst_resource_id, src_offset,
      src_stride, src_layer_stride, dst_level, dst_x, dst_y, dst_layer,
      width, height, depth, pipe_format);
}

bool
yttrium_venus_copy_buffer_to_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *src,
                                    struct yttrium_venus_resource *dst,
                                    VkDeviceSize src_offset,
                                    VkDeviceSize dst_offset,
                                    VkDeviceSize size)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->copy_buffer_to_buffer(
      venus->ctx, src, dst, src_offset, dst_offset, size);
}

bool
yttrium_venus_create_display_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint64_t allocation_size,
                                    uint64_t *out_memory_id)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_display_buffer(
      venus->ctx, resource, allocation_size, out_memory_id);
}

bool
yttrium_venus_create_bind_buffer(struct yttrium_venus *venus,
                                 struct yttrium_venus_resource *resource,
                                 uint64_t allocation_size,
                                 VkBufferUsageFlags usage,
                                 uint64_t *out_memory_id)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_bind_buffer(
      venus->ctx, resource, allocation_size, usage, out_memory_id);
}

bool
yttrium_venus_create_stream_output_buffer(struct yttrium_venus *venus,
                                          struct yttrium_venus_resource *resource,
                                          uint64_t allocation_size,
                                          uint64_t *out_memory_id)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_stream_output_buffer(
      venus->ctx, resource, allocation_size, out_memory_id);
}

bool
yttrium_venus_transform_feedback_enabled(const struct yttrium_venus *venus)
{
   return venus && venus->backend &&
      venus->backend->transform_feedback_enabled(venus->ctx);
}

bool
yttrium_venus_transform_feedback_draw_enabled(const struct yttrium_venus *venus)
{
   return venus && venus->backend &&
      venus->backend->transform_feedback_draw_enabled(venus->ctx);
}

bool
yttrium_venus_vertex_attribute_divisor_supported(struct yttrium_venus *venus,
                                                 uint32_t divisor)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend &&
      backend->vertex_attribute_divisor_supported(venus->ctx, divisor);
}

bool
yttrium_venus_depth_clamp_enabled(const struct yttrium_venus *venus)
{
   return venus && venus->backend &&
      venus->backend->depth_clamp_enabled(venus->ctx);
}

bool
yttrium_venus_logic_op_enabled(const struct yttrium_venus *venus)
{
   return venus && venus->backend &&
      venus->backend->logic_op_enabled(venus->ctx);
}

uint32_t
yttrium_venus_max_dual_source_render_targets(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend ? backend->max_dual_source_render_targets(venus->ctx) : 0;
}

float
yttrium_venus_max_sampler_anisotropy(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->max_sampler_anisotropy ?
      MAX2(backend->max_sampler_anisotropy(venus->ctx), 1.0f) : 1.0f;
}

float
yttrium_venus_max_sampler_lod_bias(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->max_sampler_lod_bias ?
      MAX2(backend->max_sampler_lod_bias(venus->ctx), 0.0f) : 0.0f;
}

uint32_t
yttrium_venus_max_transform_feedback_stride(const struct yttrium_venus *venus)
{
   return venus && venus->backend ?
      venus->backend->max_transform_feedback_stride(venus->ctx) : 0;
}

uint32_t
yttrium_venus_max_viewports(struct yttrium_venus *venus)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend ? backend->max_viewports(venus->ctx) : 1;
}

uint32_t
yttrium_venus_mipmap_precision_bits(const struct yttrium_venus *venus)
{
   return venus && venus->backend && venus->backend->mipmap_precision_bits ?
      venus->backend->mipmap_precision_bits(venus->ctx) : 0;
}

bool
yttrium_venus_create_sampled_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint64_t allocation_size,
                                    enum pipe_format pipe_format,
                                    uint64_t *out_memory_id)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_sampled_buffer(
      venus->ctx, resource, allocation_size, pipe_format, out_memory_id);
}

bool
yttrium_venus_ensure_null_sampled_buffer(struct yttrium_venus *venus,
                                         enum pipe_format pipe_format,
                                         uint64_t min_size,
                                         struct yttrium_venus_resource **out_resource,
                                         uint32_t *out_resource_id)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->ensure_null_sampled_buffer(
      venus->ctx, pipe_format, min_size, out_resource, out_resource_id);
}

bool
yttrium_venus_create_display_image(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t width,
                                   uint32_t height,
                                   enum pipe_format pipe_format,
                                   uint64_t min_allocation_size,
                                   uint64_t *out_memory_id,
                                   uint64_t *out_allocation_size)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_display_image(
      venus->ctx, resource, width, height, pipe_format, min_allocation_size,
      out_memory_id, out_allocation_size);
}

bool
yttrium_venus_create_color_attachment_image(struct yttrium_venus *venus,
                                            struct yttrium_venus_resource *resource,
                                            uint32_t width,
                                            uint32_t height,
                                            uint32_t levels,
                                            uint32_t layers,
                                            enum pipe_format pipe_format,
                                            uint64_t *out_allocation_size)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_color_attachment_image(
      venus->ctx, resource, width, height, levels, layers, pipe_format,
      out_allocation_size);
}

bool
yttrium_venus_create_sampled_texture_image(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_sampled_texture_image(
      venus->ctx, resource, target, width, height, depth, levels, layers,
      pipe_format, out_allocation_size);
}

bool
yttrium_venus_create_texture_image_for_bind(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_texture_image_for_bind(
      venus->ctx, resource, target, width, height, depth, levels, layers,
      pipe_format, sample_count, bind, out_allocation_size);
}

bool
yttrium_venus_create_depth_stencil_image(struct yttrium_venus *venus,
                                         struct yttrium_venus_resource *resource,
                                         uint32_t width,
                                         uint32_t height,
                                         enum pipe_format pipe_format,
                                         uint64_t *out_allocation_size)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->create_depth_stencil_image(
      venus->ctx, resource, width, height, pipe_format, out_allocation_size);
}

bool
yttrium_venus_clear_depth_stencil(struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->clear_depth_stencil(
      venus->ctx, resource, resource_id, clear_flags, depth, stencil,
      level, first_layer, layer_count, dstx, dsty, width, height);
}

bool
yttrium_venus_import_display_image(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t resource_id,
                                   uint32_t width,
                                   uint32_t height,
                                   enum pipe_format pipe_format,
                                   uint64_t min_allocation_size,
                                   uint64_t *out_memory_id,
                                   uint64_t *out_allocation_size)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->import_display_image(
      venus->ctx, resource, resource_id, width, height, pipe_format,
      min_allocation_size, out_memory_id, out_allocation_size);
}

bool
yttrium_venus_import_texture_image_for_bind(
   struct yttrium_venus *venus,
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
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   return backend && backend->import_texture_image_for_bind(
      venus->ctx, resource, resource_id, target, width, height, depth, levels,
      layers, pipe_format, sample_count, bind, min_allocation_size,
      out_memory_id, out_allocation_size);
}

bool
yttrium_venus_resource_fini(struct yttrium_venus *venus,
                            struct gdikmt_context *ctx,
                            struct yttrium_venus_resource *resource,
                            const struct yttrium_venus_allocation_snapshot *allocation)
{
   const struct yttrium_venus_backend *backend =
      yttrium_venus_backend_or_null(venus);
   if (backend)
      return backend->resource_fini(venus->ctx, ctx, resource, allocation);
   return false;
}
