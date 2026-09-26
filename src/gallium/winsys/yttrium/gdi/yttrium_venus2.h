/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_VENUS2_H
#define YTTRIUM_VENUS2_H

#define YTTRIUM_VENUS_BACKEND_PREFIX yttrium_venus2_
#include "yttrium_venus_backend_prefix.h"
#include "yttrium_venus.h"

#undef yttrium_venus_create
#undef yttrium_venus_get_kmt_context
#undef yttrium_venus_destroy
#undef yttrium_venus_create_shader_module
#undef yttrium_venus_pipe_format
#undef yttrium_venus_destroy_shader_module
#undef yttrium_venus_pipeline_init
#undef yttrium_venus_supports_load_store_op_none
#undef yttrium_venus_supports_multisampled_render_to_single_sampled
#undef yttrium_venus_supports_forced_sample_interlock
#undef yttrium_venus_framebuffer_color_sample_counts
#undef yttrium_venus_sampled_texture_format_supported
#undef yttrium_venus_storage_image_without_format_supported
#undef yttrium_venus_storage_image_formatless_format_supported
#undef yttrium_venus_compute_pipeline_init
#undef yttrium_venus_pipeline_fini
#undef yttrium_venus_draw_pipeline
#undef yttrium_venus_dispatch_compute
#undef yttrium_venus_clear_display
#undef yttrium_venus_clear_display_rect
#undef yttrium_venus_draw_vertex_buffer_vertices
#undef yttrium_venus_draw_textured_vertices
#undef yttrium_venus_update_buffer
#undef yttrium_venus_clear_buffer
#undef yttrium_venus_copy_image_to_display_buffer
#undef yttrium_venus_copy_image_region_to_display_buffer
#undef yttrium_venus_copy_image_region_aspect_to_display_buffer
#undef yttrium_venus_copy_display_image
#undef yttrium_venus_copy_depth_stencil_image
#undef yttrium_venus_flush
#undef yttrium_venus_flush_async
#undef yttrium_venus_flush_async_present_publish
#undef yttrium_venus_last_submit_order
#undef yttrium_venus_submit_order_complete
#undef yttrium_venus_wait_resource
#undef yttrium_venus_blit_display_image
#undef yttrium_venus_resolve_display_image
#undef yttrium_venus_copy_buffer_to_display_image
#undef yttrium_venus_copy_buffer_to_buffer
#undef yttrium_venus_create_display_buffer
#undef yttrium_venus_create_bind_buffer
#undef yttrium_venus_create_stream_output_buffer
#undef yttrium_venus_transform_feedback_enabled
#undef yttrium_venus_transform_feedback_draw_enabled
#undef yttrium_venus_vertex_attribute_divisor_supported
#undef yttrium_venus_depth_clamp_enabled
#undef yttrium_venus_max_dual_source_render_targets
#undef yttrium_venus_max_sampler_anisotropy
#undef yttrium_venus_max_sampler_lod_bias
#undef yttrium_venus_max_transform_feedback_stride
#undef yttrium_venus_max_viewports
#undef yttrium_venus_mipmap_precision_bits
#undef yttrium_venus_create_sampled_buffer
#undef yttrium_venus_ensure_null_sampled_buffer
#undef yttrium_venus_create_display_image
#undef yttrium_venus_create_color_attachment_image
#undef yttrium_venus_create_sampled_texture_image
#undef yttrium_venus_create_texture_image_for_bind
#undef yttrium_venus_create_depth_stencil_image
#undef yttrium_venus_clear_depth_stencil
#undef yttrium_venus_import_display_image
#undef yttrium_venus_resource_fini
#undef yttrium_vn_ring_submit_command
#undef yttrium_vn_ring_get_command_reply
#undef yttrium_vn_ring_free_command_reply
#undef YTTRIUM_VENUS_BACKEND_SYM
#undef YTTRIUM_VENUS_CONCAT
#undef YTTRIUM_VENUS_CONCAT2
#undef YTTRIUM_VENUS_BACKEND_PREFIX

struct vn_ring;
struct vn_ring_submit_command;
struct vn_cs_decoder;

bool
yttrium_venus2_vn_ring_submit_command(struct vn_ring *vn_ring,
                                      struct vn_ring_submit_command *submit);

struct vn_cs_decoder *
yttrium_venus2_vn_ring_get_command_reply(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit);

void
yttrium_venus2_vn_ring_free_command_reply(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit);

#endif /* YTTRIUM_VENUS2_H */
