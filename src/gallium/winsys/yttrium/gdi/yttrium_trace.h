/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_TRACE_H
#define YTTRIUM_TRACE_H

#include <stdbool.h>
#include <stdint.h>
#include <windows.h>

#include "util/u_debug.h"

#ifdef __cplusplus
extern "C" {
#endif

enum yttrium_trace_copy_path {
   YTTRIUM_TRACE_COPY_CPU_TO_CPU = 1,
   YTTRIUM_TRACE_COPY_CPU_TO_DISPLAY_BUFFER = 2,
   YTTRIUM_TRACE_COPY_CPU_TO_DISPLAY_IMAGE = 3,
   YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_DISPLAY_IMAGE = 4,
   YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_CPU = 5,
   YTTRIUM_TRACE_COPY_BUFFER_TO_CPU = 6,
};

enum yttrium_trace_event_id {
   YTTRIUM_EVENT_SCREEN_CREATE = 1,
   YTTRIUM_EVENT_PRESENT = 2,
   YTTRIUM_EVENT_PRESENT_MUTED = 3,
   YTTRIUM_EVENT_SCANOUT_SKIP = 4,
   YTTRIUM_EVENT_SCANOUT_DRY_RUN = 5,
   YTTRIUM_EVENT_SCANOUT_VALIDATE_ONLY = 6,
   YTTRIUM_EVENT_PRESENT_SKIP = 7,
   YTTRIUM_EVENT_PRESENT_SUBMIT = 8,
   YTTRIUM_EVENT_SET_SCANOUT_BLOB = 9,
   YTTRIUM_EVENT_RESOURCE_COPY = 10,
   YTTRIUM_EVENT_RESOURCE_COPY_UNSUPPORTED = 11,
   YTTRIUM_EVENT_DRAW_VBO = 12,
   YTTRIUM_EVENT_DRAW_SKIP = 13,
   YTTRIUM_EVENT_TEXTURED_SAMPLER = 14,
   YTTRIUM_EVENT_VENUS_DRAW = 15,
   YTTRIUM_EVENT_VENUS_TEXTURED_DRAW = 16,
   YTTRIUM_EVENT_MESSAGE = 17,
   YTTRIUM_EVENT_TIMING = 18,
   YTTRIUM_EVENT_SCANOUT_REFRESH = 19,
   YTTRIUM_EVENT_RESOURCE_COPY_TARGET = 20,
   YTTRIUM_EVENT_VENUS_ACTIVITY = 21,
   YTTRIUM_EVENT_RESOURCE_LIFETIME = 22,
   YTTRIUM_EVENT_DXGI_PRESENT = 23,
   YTTRIUM_EVENT_PFN_PRESENT_CB = 24,
   YTTRIUM_EVENT_RESOURCE_IMPORT = 25,
   YTTRIUM_EVENT_RESOURCE_DESTROY = 26,
   YTTRIUM_EVENT_SYNC_WAIT = 27,
   YTTRIUM_EVENT_NATIVE_DRAW_BATCH_DECISION = 28,
   YTTRIUM_EVENT_CMD_BATCH_SUBMIT = 29,
   YTTRIUM_EVENT_VENUS_UPLOAD = 30,
};

enum yttrium_trace_severity {
   YTTRIUM_TRACE_DEBUG = 1,
   YTTRIUM_TRACE_WARNING = 2,
   YTTRIUM_TRACE_ERROR = 3,
};

enum yttrium_trace_timing_point {
   YTTRIUM_TRACE_TIMING_VENUS_RAW_SUBMIT = 1,
   YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE = 2,
   YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SEQNO = 3,
   YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT = 4,
   YTTRIUM_TRACE_TIMING_VENUS_RESET_FENCES = 5,
   YTTRIUM_TRACE_TIMING_VENUS_QUEUE_SUBMIT = 6,
   YTTRIUM_TRACE_TIMING_VENUS_WAIT_FENCES = 7,
   YTTRIUM_TRACE_TIMING_VENUS_SUBMIT_WAIT_TOTAL = 8,
   YTTRIUM_TRACE_TIMING_KMT_BUSY_WAIT = 9,
   YTTRIUM_TRACE_TIMING_KMT_NOP_RENDER = 10,
   YTTRIUM_TRACE_TIMING_SET_SCANOUT_BLOB = 11,
   YTTRIUM_TRACE_TIMING_PFN_PRESENT_CB = 12,
   YTTRIUM_TRACE_TIMING_SCANOUT_FULL_COPY = 13,
   YTTRIUM_TRACE_TIMING_SCANOUT_DIRTY_COPY = 14,
   YTTRIUM_TRACE_TIMING_PIPELINE_GET = 15,
   YTTRIUM_TRACE_TIMING_PIPELINE_DRAW_UPLOAD = 16,
   YTTRIUM_TRACE_TIMING_PIPELINE_UBO_UPLOADS = 17,
   YTTRIUM_TRACE_TIMING_PIPELINE_SAMPLED_TEXTURES = 18,
   YTTRIUM_TRACE_TIMING_PIPELINE_SO_TARGETS = 19,
   YTTRIUM_TRACE_TIMING_PIPELINE_DRAW_TOTAL = 20,
   YTTRIUM_TRACE_TIMING_VENUS_NATIVE_DRAW_PREP = 21,
   YTTRIUM_TRACE_TIMING_VENUS_NATIVE_DRAW_RECORD = 22,
   YTTRIUM_TRACE_TIMING_VENUS_NATIVE_DRAW_TOTAL = 23,
};

enum yttrium_trace_scanout_refresh_reason {
   YTTRIUM_TRACE_SCANOUT_REFRESH_SUBMITTED_DIRTY = 1,
   YTTRIUM_TRACE_SCANOUT_REFRESH_SUBMITTED_FULL = 2,
   YTTRIUM_TRACE_SCANOUT_REFRESH_SKIP_NOT_PRIMARY = 3,
   YTTRIUM_TRACE_SCANOUT_REFRESH_SKIP_DISABLED = 4,
   YTTRIUM_TRACE_SCANOUT_REFRESH_SKIP_DRY_OR_VALIDATE = 5,
};

enum yttrium_trace_resource_copy_stage {
   YTTRIUM_TRACE_RESOURCE_COPY_BEGIN = 1,
   YTTRIUM_TRACE_RESOURCE_COPY_COPIED = 2,
   YTTRIUM_TRACE_RESOURCE_COPY_UNSUPPORTED_STAGE = 3,
};

enum yttrium_trace_dxgi_present_stage {
   YTTRIUM_TRACE_DXGI_PRESENT_BEGIN = 1,
   YTTRIUM_TRACE_DXGI_PRESENT_FLUSH_FRONTBUFFER = 2,
   YTTRIUM_TRACE_DXGI_PRESENT_END = 3,
};

enum yttrium_trace_pfn_present_stage {
   YTTRIUM_TRACE_PFN_PRESENT_BEFORE = 1,
   YTTRIUM_TRACE_PFN_PRESENT_AFTER = 2,
};

enum yttrium_trace_resource_import_stage {
   YTTRIUM_TRACE_RESOURCE_IMPORT_BEGIN = 1,
   YTTRIUM_TRACE_RESOURCE_IMPORT_QUERY_FAILED = 2,
   YTTRIUM_TRACE_RESOURCE_IMPORT_OPEN_FAILED = 3,
   YTTRIUM_TRACE_RESOURCE_IMPORT_PRIVATE_TOO_SMALL = 4,
   YTTRIUM_TRACE_RESOURCE_IMPORT_RESINFO_FAILED = 5,
   YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_SKIPPED = 6,
   YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_BEGIN = 7,
   YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_OK = 8,
   YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_FAILED = 9,
   YTTRIUM_TRACE_RESOURCE_IMPORT_OPENED = 10,
};

enum yttrium_trace_resource_destroy_stage {
   YTTRIUM_TRACE_RESOURCE_DESTROY_BEGIN = 1,
   YTTRIUM_TRACE_RESOURCE_DESTROY_ALLOCATION_CLOSED = 2,
   YTTRIUM_TRACE_RESOURCE_DESTROY_END = 3,
};

enum yttrium_trace_venus_upload_kind {
   YTTRIUM_TRACE_VENUS_UPLOAD_UPDATE_BUFFER = 1,
   YTTRIUM_TRACE_VENUS_UPLOAD_BATCH_UPDATE_BUFFER = 2,
   YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_VERTEX = 3,
   YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_INDEX = 4,
   YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_UBO = 5,
   YTTRIUM_TRACE_VENUS_UPLOAD_CPU_IMAGE_STAGING = 6,
   YTTRIUM_TRACE_VENUS_UPLOAD_BUFFER_TO_IMAGE = 7,
   YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_TO_BUFFER = 8,
   YTTRIUM_TRACE_VENUS_UPLOAD_BUFFER_COPY = 9,
   YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_COPY = 10,
   YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_BLIT = 11,
   YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_RESOLVE = 12,
};

void
yttrium_trace_init(bool etw_enabled, bool text_enabled);

void
yttrium_trace_shutdown(void);

bool
yttrium_trace_is_enabled(void);

/* True while the provider can record the bounded wait stream.  This remains
 * true in wait-stats-only mode while the general trace predicate is false.
 */
bool
yttrium_trace_sync_wait_is_enabled(void);

bool
yttrium_trace_text_enabled(void);

uint64_t
yttrium_trace_now_us(void);

/* Base name of the running executable, for logs shared by every process that
 * loads the UMD.  Never NULL.
 */
const char *
yttrium_trace_process_name(void);

void
yttrium_trace_logf(uint32_t severity, const char *format, ...);

void
yttrium_trace_debug_stringf(const char *format, ...);

void
yttrium_trace_resource_event(uint32_t kind,
                             const char *kind_name,
                             uint64_t handle,
                             uint64_t object,
                             uint64_t pipe_resource,
                             int32_t refcount,
                             uint32_t a,
                             uint32_t b,
                             uint64_t c);

void
yttrium_trace_dxgi_present(uint32_t stage,
                           uint64_t src_resource,
                           uint64_t dst_resource,
                           uint64_t src_allocation,
                           uint64_t dst_allocation,
                           uint64_t dxgi_context,
                           uint64_t hwnd,
                           uint64_t source,
                           uint64_t destination,
                           uint32_t src_subresource,
                           uint32_t dst_subresource,
                           uint32_t flags,
                           uint32_t flip_interval,
                           uint32_t present_count,
                           uint32_t gdi_readback,
                           uint32_t readback_presented);

void
yttrium_trace_present_callback(uint32_t stage,
                               long status,
                               uint64_t src_allocation,
                               uint64_t dst_allocation,
                               uint64_t context,
                               uint64_t dxgi_context,
                               uint32_t private_size,
                               uint32_t optimize_for_composition,
                               uint32_t broadcast_count);

void
yttrium_trace_resource_import(uint32_t stage,
                              long status,
                              uint64_t global_handle,
                              uint64_t resource_handle,
                              uint64_t allocation,
                              uint32_t resource_id,
                              uint64_t memory_id,
                              uint32_t width,
                              uint32_t height,
                              uint32_t format,
                              uint32_t bind,
                              uint32_t flags,
                              uint32_t usage,
                              uint32_t display,
                              uint32_t classic,
                              uint32_t owns,
                              uint32_t venus_initialized,
                              uint32_t import_enabled,
                              uint64_t image_id,
                              uint64_t buffer_id,
                              uint64_t size,
                              uint32_t stride);

void
yttrium_trace_resource_destroy(uint32_t stage,
                               long status,
                               uint64_t resource,
                               uint64_t resource_handle,
                               uint64_t allocation,
                               uint32_t resource_id,
                               uint64_t memory_id,
                               uint32_t display,
                               uint32_t primary,
                               uint32_t classic,
                               uint32_t owns,
                               uint32_t venus_initialized,
                               uint32_t venus_buffer_backed,
                               uint64_t image_id,
                               uint64_t buffer_id,
                               uint64_t venus_memory_id,
                               uint64_t scanout_allocation,
                               uint32_t scanout_resource_id,
                               uint32_t scanout_initialized,
                               uint64_t size,
                               uint32_t stride);

/*
 * label names the call site that spent the time, and is emitted verbatim - a
 * hash here meant every analysis had to invert it against a corpus of every
 * string literal in the tree just to learn which wait was firing.  Pass a
 * literal, or NULL where the point alone identifies the site.
 */
void
yttrium_trace_timing(uint32_t point,
                     uint32_t status,
                     uint64_t elapsed_us,
                     const char *label,
                     uint64_t a,
                     uint64_t b,
                     uint32_t c,
                     uint32_t d);

void
yttrium_trace_scanout_refresh(uint32_t reason,
                              uint32_t copy_id,
                              uint64_t allocation,
                              uint32_t resource_id,
                              uint32_t x,
                              uint32_t y,
                              uint32_t width,
                              uint32_t height,
                              uint32_t display,
                              uint32_t primary,
                              uint32_t classic,
                              uint32_t venus,
                              uint32_t scanout_present);

void
yttrium_trace_resource_copy_target(uint32_t copy_id,
                                   uint32_t stage,
                                   uint32_t path,
                                   uint64_t src_allocation,
                                   uint32_t src_resource_id,
                                   uint64_t dst_allocation,
                                   uint32_t dst_resource_id,
                                   uint32_t dst_target,
                                   uint32_t dst_format,
                                   uint32_t dst_bind,
                                   uint32_t dst_width,
                                   uint32_t dst_height,
                                   uint32_t x,
                                   uint32_t y,
                                   uint32_t width,
                                   uint32_t height,
                                   uint32_t display,
                                   uint32_t primary,
                                   uint32_t classic,
                                   uint32_t venus,
                                   uint32_t buffer_backed,
                                   uint32_t image,
                                   uint32_t data,
                                   uint32_t scanout_present);

void
yttrium_trace_venus_activity(uint64_t window_us,
                             uint64_t total_bytes,
                             uint64_t max_elapsed_us,
                             uint32_t total_count,
                             uint32_t raw_count,
                             uint32_t ring_count,
                             uint32_t wait_space_count,
                             uint32_t wait_seqno_count,
                             uint32_t reset_count,
                             uint32_t queue_submit_count,
                             uint32_t wait_fences_count,
                             uint32_t submit_wait_count,
                             uint32_t failure_count,
                             uint32_t last_point,
                             uint32_t last_status,
                             uint32_t last_c,
                             uint32_t last_d);

void
yttrium_trace_sync_wait(uint32_t kind,
                        const char *kind_name,
                        uint32_t status,
                        uint64_t elapsed_us,
                        uint32_t command_type,
                        const char *command_name,
                        uint32_t command_size,
                        uint32_t reply_size,
                        uint64_t backlog_command_count,
                        uint64_t backlog_command_bytes,
                        uint64_t backlog_queue_submit_count,
                        const char *label,
                        uint32_t a,
                        uint32_t b);

void
yttrium_trace_ring_kick(uint64_t elapsed_us,
                        uint32_t seqno,
                        uint32_t status,
                        uint32_t path,
                        const char *path_name,
                        uint32_t command_size,
                        uint32_t blocking,
                        const char *label);

bool
yttrium_trace_verbose_etw_text_enabled(void);

void
yttrium_trace_sync_wait_summary(const char *message);

void
yttrium_trace_screen_create(uint32_t supports_3d,
                            uint32_t has_resource_blob,
                            uint32_t has_host_visible,
                            uint32_t scanout_present,
                            uint32_t scanout_dry_run,
                            uint32_t mute_kmd_present,
                            uint32_t no_present,
                            uint32_t no_primary_present);

void
yttrium_trace_present(uint64_t allocation,
                      uint32_t resource_id,
                      uint64_t memory_id,
                      uint32_t display,
                      uint32_t primary,
                      uint32_t classic,
                      const void *context_private,
                      const void *window,
                      uint32_t boxes,
                      uint32_t scanout_present,
                      uint32_t scanout_dry_run,
                      uint32_t no_present,
                      uint32_t no_primary_present);

void
yttrium_trace_present_state(uint16_t event_id,
                            uint64_t allocation,
                            uint32_t resource_id,
                            uint32_t a,
                            uint32_t b,
                            uint32_t c);

void
yttrium_trace_present_submit(uint64_t allocation,
                             uint32_t resource_id,
                             const void *context_private,
                             long status,
                             uint64_t submitted_context,
                             uint64_t dxgi_context,
                             uint64_t dxgi_window,
                             uint64_t dxgi_source,
                             uint64_t dxgi_destination,
                             uint32_t dxgi_flags,
                             uint32_t dxgi_flip_interval,
                             uint32_t dxgi_present_count);

void
yttrium_trace_scanout_blob(uint64_t allocation,
                           uint32_t resource_id,
                           uint32_t scanout_id,
                           uint32_t width,
                           uint32_t height,
                           uint32_t x,
                           uint32_t y,
                           uint32_t format,
                           uint32_t stride,
                           uint32_t offset,
                           uint64_t size,
                           uint32_t classic,
                           long status);

void
yttrium_trace_resource_copy(uint32_t path,
                            uint64_t src_allocation,
                            uint32_t src_resource_id,
                            uint64_t dst_allocation,
                            uint32_t dst_resource_id,
                            uint32_t src_x,
                            uint32_t src_y,
                            uint32_t src_layer,
                            uint32_t dst_x,
                            uint32_t dst_y,
                            uint32_t dst_layer,
                            uint32_t width,
                            uint32_t height,
                            uint32_t src_stride,
                            uint32_t dst_stride);

void
yttrium_trace_resource_copy_unsupported(uint64_t src_allocation,
                                        uint32_t src_resource_id,
                                        uint64_t dst_allocation,
                                        uint32_t dst_resource_id,
                                        int32_t width,
                                        int32_t height);

void
yttrium_trace_draw_vbo(uint32_t mode,
                       uint32_t num_draws,
                       uint32_t count,
                       uint32_t instances,
                       uint32_t start_instance,
                       const void *cbuf,
                       uint32_t have_vertices,
                       uint32_t vertex_count,
                       uint32_t vs_inputs,
                       uint32_t vs_outputs,
                       uint32_t fs_inputs,
                       uint32_t fs_outputs,
                       uint32_t fs_srvs,
                       uint32_t fs_samplers);

void
yttrium_trace_draw_skip(uint32_t reason,
                        uint64_t allocation,
                        uint32_t resource_id,
                        uint32_t a,
                        uint32_t b,
                        uint32_t c);

enum yttrium_trace_draw_skip_reason {
   YTTRIUM_TRACE_DRAW_SKIP_UNSUPPORTED_FIXED_COLOR = 1,
   YTTRIUM_TRACE_DRAW_SKIP_IGNORED = 2,
};

void
yttrium_trace_textured_sampler(uint32_t slot,
                               const void *texture,
                               uint64_t allocation,
                               uint32_t resource_id,
                               uint64_t image_id,
                               uint32_t width,
                               uint32_t height,
                               uint32_t format);

void
yttrium_trace_venus_draw(uint32_t event_id,
                         uint32_t dst_resource_id,
                         uint64_t dst_image_id,
                         uint32_t src_resource_id,
                         uint64_t src_image_id,
                         uint64_t pipeline_id,
                         uint32_t vertex_count,
                         float viewport_x,
                         float viewport_y,
                         float viewport_width,
                         float viewport_height,
                         int32_t scissor_x,
                         int32_t scissor_y,
                         uint32_t scissor_width,
                         uint32_t scissor_height);

enum yttrium_trace_native_draw_batch_reject {
   YTTRIUM_TRACE_NATIVE_DRAW_BATCH_REJECT_DRAW_BATCH_DISABLED = 1 << 0,
   YTTRIUM_TRACE_NATIVE_DRAW_BATCH_REJECT_CPU_VERTEX_DISABLED = 1 << 1,
   YTTRIUM_TRACE_NATIVE_DRAW_BATCH_REJECT_SAMPLED_WITHOUT_PUSH = 1 << 2,
   YTTRIUM_TRACE_NATIVE_DRAW_BATCH_REJECT_PUSH_BATCH_DISABLED = 1 << 3,
   YTTRIUM_TRACE_NATIVE_DRAW_BATCH_REJECT_PUSH_LAYOUT_MISSING = 1 << 4,
   YTTRIUM_TRACE_NATIVE_DRAW_BATCH_REJECT_PUSH_PIPELINE_MISSING = 1 << 5,
   YTTRIUM_TRACE_NATIVE_DRAW_BATCH_REJECT_SAMPLED_CPU_VERTEX_DISABLED = 1 << 6,
};

void
yttrium_trace_native_draw_batch_decision(uint32_t candidate,
                                         uint32_t reject_mask,
                                         uint32_t native_draw_batch_enabled,
                                         uint32_t cpu_vertex_batch_allowed,
                                         uint32_t cpu_vertex_batch_enabled,
                                         uint32_t has_cpu_vertex_upload,
                                         uint32_t has_sampled_descriptor,
                                         uint32_t has_ubo_descriptor,
                                         uint32_t push_descriptor_batch_enabled,
                                         uint32_t push_descriptors_available,
                                         uint32_t pipeline_has_push_layout,
                                         uint32_t pipeline_has_push_pipeline,
                                         uint32_t use_push_descriptors,
                                         uint32_t mode,
                                         uint32_t vertex_count,
                                         uint32_t index_count,
                                         uint64_t pipeline_id);

void
yttrium_trace_cmd_batch_submit(uint32_t async_submit,
                               uint32_t native_draw_only,
                               uint32_t op_count,
                               uint32_t resource_count,
                               uint32_t pipeline_count,
                               uint32_t transient_count,
                               uint32_t allocated_batch_count,
                               uint32_t live_batch_count,
                               uint32_t peak_live_batch_count,
                               uint32_t pending_submit_count,
                               uint32_t group_submit_size,
                               uint64_t batch_pool_bytes,
                               uint64_t ubo_arena_bytes,
                               uint64_t peak_ubo_arena_bytes,
                               uint64_t draw_backing_pool_bytes,
                               uint64_t peak_draw_backing_pool_bytes,
                               const char *label);

void
yttrium_trace_venus_upload(uint32_t kind,
                           uint32_t flags,
                           uint64_t bytes,
                           uint64_t src_object_id,
                           uint64_t dst_object_id,
                           uint32_t src_resource_id,
                           uint32_t dst_resource_id,
                           uint32_t width,
                           uint32_t height,
                           uint32_t depth,
                           uint32_t row_stride,
                           uint32_t layer_stride);

/* Per-emission diagnostic totals; no descriptor payloads or per-draw events. */
struct yttrium_trace_deferred_draw_stats {
   uint32_t draws;
   uint32_t render_passes;
   uint32_t pipeline_binds;
   uint32_t pipeline_binds_skipped;
   uint32_t descriptor_binds;
   uint32_t descriptor_binds_skipped;
   uint32_t push_calls;
   uint32_t push_writes;
   uint32_t repeated_push_layouts;
   uint32_t requested_push_calls;
   uint32_t requested_push_writes;
   uint32_t push_comparisons;
   uint32_t push_calls_skipped;
   uint32_t push_writes_skipped;
};

void
yttrium_trace_deferred_draw_state(
   const struct yttrium_trace_deferred_draw_stats *stats);

#define YTTRIUM_LOG(...)                                                \
   do {                                                                \
      if (yttrium_trace_verbose_etw_text_enabled() &&                    \
          yttrium_trace_is_enabled())                                  \
         yttrium_trace_logf(YTTRIUM_TRACE_DEBUG, __VA_ARGS__);            \
   } while (0)

#define YTTRIUM_WARN(...)                                               \
   yttrium_trace_logf(YTTRIUM_TRACE_WARNING, __VA_ARGS__)

#define YTTRIUM_ERROR(...)                                              \
   yttrium_trace_logf(YTTRIUM_TRACE_ERROR, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* YTTRIUM_TRACE_H */
