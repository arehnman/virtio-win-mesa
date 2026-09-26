/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_RESOURCE_H
#define YTTRIUM_RESOURCE_H

#include <stdbool.h>
#include <stdint.h>

#include "yttrium_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

struct pipe_context;
struct pipe_resource;
struct winsys_handle;

void
yttrium_resource_init_screen(struct yttrium_screen *screen);

void
yttrium_resource_cleanup_screen(struct yttrium_screen *screen);

uint64_t
yttrium_resource_size(const struct pipe_resource *templ,
                      unsigned *out_stride, unsigned *out_layer_stride);

uint32_t
yttrium_pipe_to_resource_format(enum pipe_format format);

enum pipe_format
yttrium_resource_to_pipe_format(uint32_t format);

bool
yttrium_map_allocation(struct pipe_screen *pscreen,
                       D3DKMT_HANDLE hAllocation,
                       uint64_t size,
                       const char *label,
                       void **out_map,
                       uint32_t *out_map_info,
                       bool *out_map_is_blob);

bool
yttrium_create_venus_memory_mapping(struct pipe_screen *pscreen,
                                    uint64_t venus_mem_id,
                                    uint64_t size,
                                    unsigned stride,
                                    struct yttrium_readback_mapping *mapping);

void
yttrium_destroy_readback_mapping(struct pipe_screen *pscreen,
                                 struct yttrium_readback_mapping *mapping);

bool
yttrium_map_display_allocation(struct pipe_screen *pscreen,
                               struct yttrium_resource *res);

struct pipe_resource *
yttrium_resource_create(struct pipe_screen *pscreen,
                        const struct pipe_resource *templ);

void
yttrium_resource_destroy(struct pipe_screen *pscreen,
                         struct pipe_resource *resource);

bool
yttrium_resource_get_handle(struct pipe_screen *pscreen,
                            struct pipe_context *ctx,
                            struct pipe_resource *resource,
                            struct winsys_handle *whandle,
                            unsigned usage);

struct pipe_resource *
yttrium_resource_from_handle(struct pipe_screen *pscreen,
                             const struct pipe_resource *templ,
                             struct winsys_handle *whandle,
                             unsigned usage);

void
yttrium_destroy_context_upload_staging(struct pipe_screen *pscreen,
                                       struct yttrium_context *yctx);

void
yttrium_destroy_context_image_copy_staging(struct pipe_screen *pscreen,
                                           struct yttrium_context *yctx);

void *
yttrium_transfer_map(struct pipe_context *ctx,
                     struct pipe_resource *resource,
                     unsigned level,
                     unsigned usage,
                     const struct pipe_box *box,
                     struct pipe_transfer **ptransfer);

void
yttrium_transfer_unmap(struct pipe_context *ctx,
                       struct pipe_transfer *transfer);

void
yttrium_transfer_flush_region(struct pipe_context *ctx,
                              struct pipe_transfer *transfer,
                              const struct pipe_box *box);

bool
yttrium_resource_publish_private_draw_buffer(struct pipe_context *ctx,
                                             struct yttrium_resource *res);

void
yttrium_buffer_subdata(struct pipe_context *ctx,
                       struct pipe_resource *resource,
                       unsigned usage,
                       unsigned offset,
                       unsigned size,
                       const void *data);

void
yttrium_replace_buffer_storage(struct pipe_context *ctx,
                               struct pipe_resource *dst,
                               struct pipe_resource *src,
                               unsigned minimum_num_rebinds,
                               uint32_t rebind_mask,
                               uint32_t delete_buffer_id);

void
yttrium_texture_subdata(struct pipe_context *ctx,
                        struct pipe_resource *resource,
                        unsigned level,
                        unsigned usage,
                        const struct pipe_box *box,
                        const void *data,
                        unsigned stride,
                        uintptr_t layer_stride);

void
yttrium_clear_buffer(struct pipe_context *ctx,
                     struct pipe_resource *resource,
                     unsigned offset,
                     unsigned size,
                     const void *clear_value,
                     int clear_value_size);

void
yttrium_clear_render_target(struct pipe_context *ctx,
                            struct pipe_surface *dst,
                            const union pipe_color_union *color,
                            unsigned dstx,
                            unsigned dsty,
                            unsigned width,
                            unsigned height,
                            bool render_condition_enabled);

void
yttrium_clear_depth_stencil(struct pipe_context *ctx,
                            struct pipe_surface *dst,
                            unsigned clear_flags,
                            double depth,
                            unsigned stencil,
                            unsigned dstx,
                            unsigned dsty,
                            unsigned width,
                            unsigned height,
                            bool render_condition_enabled);

void
yttrium_clear(struct pipe_context *ctx,
              unsigned buffers,
              uint32_t color_clear_mask,
              uint8_t stencil_clear_mask,
              const struct pipe_scissor_state *scissor_state,
              const union pipe_color_union *color,
              double depth,
              unsigned stencil);

bool
yttrium_copy_cpu_to_venus_image(struct pipe_context *ctx,
                                struct yttrium_resource *ysrc,
                                struct yttrium_resource *ydst,
                                uint32_t src_level,
                                uint32_t dst_level,
                                uint32_t src_x,
                                uint32_t src_y,
                                uint32_t src_layer,
                                uint32_t dstx,
                                uint32_t dsty,
                                uint32_t dstz,
                                uint32_t width,
                                uint32_t height,
                                uint32_t depth);

void
yttrium_resource_copy_region(struct pipe_context *ctx,
                             struct pipe_resource *dst,
                             unsigned dst_level,
                             unsigned dstx,
                             unsigned dsty,
                             unsigned dstz,
                             struct pipe_resource *src,
                             unsigned src_level,
                             const struct pipe_box *src_box);

void
yttrium_blit(struct pipe_context *ctx, const struct pipe_blit_info *info);

bool
yttrium_is_format_supported(struct pipe_screen *pscreen,
                            enum pipe_format format,
                            enum pipe_texture_target target,
                            unsigned sample_count,
                            unsigned storage_sample_count,
                            unsigned bindings);

void
yttrium_flush_resource(struct pipe_context *ctx, struct pipe_resource *resource);

#ifdef __cplusplus
}
#endif

#endif /* YTTRIUM_RESOURCE_H */
