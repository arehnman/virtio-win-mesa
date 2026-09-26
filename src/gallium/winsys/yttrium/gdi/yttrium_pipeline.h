/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_PIPELINE_H
#define YTTRIUM_PIPELINE_H

#include <stdbool.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include "yttrium_venus.h"

#ifdef __cplusplus
extern "C" {
#endif

struct pipe_draw_info;
struct pipe_draw_indirect_info;
struct pipe_draw_start_count_bias;
struct yttrium_context;
struct yttrium_resource;
struct yttrium_shader_state;

struct yttrium_pipeline_key {
   uint32_t vs_id;
   uint32_t tcs_id;
   uint32_t tes_id;
   uint32_t gs_id;
   uint32_t fs_id;
   uint64_t vs_hash;
   uint64_t tcs_hash;
   uint64_t tes_hash;
   uint64_t gs_hash;
   uint64_t fs_hash;
   uint32_t rt_count;
   uint64_t dst_image_id[PIPE_MAX_COLOR_BUFS];
   uint64_t zs_image_id;
   VkFormat rt_format[PIPE_MAX_COLOR_BUFS];
   uint32_t a8_rt_mask;
   uint32_t x8_rt_mask;
   uint32_t dual_source_blend;
   VkFormat zs_format;
   uint32_t width;
   uint32_t height;
   /*
    * Per colour target: Vulkan puts baseMipLevel and baseArrayLayer on the
    * image view rather than the framebuffer, so attachments may name different
    * subresources.  layerCount is shared, so rt_layers stays single.  The
    * pipeline owns the views, so these have to be in the key or a pipeline gets
    * reused with views onto the wrong mip or slice.
    */
   uint32_t rt_level[PIPE_MAX_COLOR_BUFS];
   uint32_t rt_layer[PIPE_MAX_COLOR_BUFS];
   uint32_t rt_layers;
   uint32_t zs_level;
   uint32_t zs_layer;
   uint32_t zs_layers;
   uint32_t viewport_count;
   VkPrimitiveTopology topology;
   uint8_t patch_vertices;
   VkBool32 primitive_restart_enable;
   VkBool32 rasterizer_discard_enable;
   VkBool32 targetless_stream_output;
   VkCullModeFlags cull_mode;
   VkFrontFace front_face;
   VkBool32 depth_bias_enable;
   VkBool32 depth_clamp_enable;
   float depth_bias_constant_factor;
   float depth_bias_clamp;
   float depth_bias_slope_factor;
   VkSampleMask sample_mask;
   uint32_t rasterization_samples;
   uint32_t forced_sample_count;
   uint32_t forced_sample_expand;
   VkBool32 forced_sample_interlock;
   VkBool32 sample_shading_enable;
   VkBool32 alpha_to_coverage_enable;
   VkBool32 logic_op_enable;
   VkLogicOp logic_op;
   VkBool32 blend_enable[PIPE_MAX_COLOR_BUFS];
   VkColorComponentFlags color_write_mask[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor src_color_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor dst_color_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendOp color_blend_op[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor src_alpha_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor dst_alpha_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendOp alpha_blend_op[PIPE_MAX_COLOR_BUFS];
   VkBool32 depth_test_enable;
   VkBool32 depth_write_enable;
   VkCompareOp depth_compare_op;
   VkBool32 alpha_test_enable;
   uint32_t alpha_func;
   float alpha_ref_value;
   VkBool32 stencil_test_enable;
   VkStencilOpState stencil_front;
   VkStencilOpState stencil_back;
   uint32_t vs_ubo_used_mask;
   uint32_t tcs_ubo_used_mask;
   uint32_t tes_ubo_used_mask;
   uint32_t fs_ubo_used_mask;
   uint32_t gs_ubo_used_mask;
   uint32_t sampled_stage_mask;
   uint32_t sampled_binding_count;
   struct yttrium_venus_sampled_binding_layout
      sampled_bindings[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint32_t color_feedback_loop_mask;
   VkBool32 depth_feedback_loop;
   /* Whole-image, depth-only sampled attachment with no effective writes.
    * Mutually exclusive with depth_feedback_loop; color feedback is separate.
    */
   VkBool32 depth_read_only;
   uint64_t storage_image_mask;
   uint64_t storage_buffer_mask;
   uint32_t storage_stage_mask;
   uint8_t vs_ubo_default;
   uint8_t tcs_ubo_default;
   uint8_t tes_ubo_default;
   uint8_t fs_ubo_default;
   uint8_t gs_ubo_default;
   uint8_t vs_ubo_first;
   uint8_t tcs_ubo_first;
   uint8_t tes_ubo_first;
   uint8_t fs_ubo_first;
   uint8_t gs_ubo_first;
   uint8_t vs_ubo_count;
   uint8_t tcs_ubo_count;
   uint8_t tes_ubo_count;
   uint8_t fs_ubo_count;
   uint8_t gs_ubo_count;
   uint32_t num_bindings;
   uint32_t num_attribs;
   VkVertexInputBindingDescription bindings[PIPE_MAX_ATTRIBS];
   uint32_t binding_divisors[PIPE_MAX_ATTRIBS];
   VkVertexInputAttributeDescription attribs[PIPE_MAX_ATTRIBS];
};

struct yttrium_pipeline {
   struct yttrium_pipeline_key key;
   uint32_t key_hash;
   void *render_target_cache_entry;
   struct pipe_resource *rt_resources[PIPE_MAX_COLOR_BUFS];
   struct pipe_resource *zs_resource;
   struct yttrium_venus_object image_view_objs[PIPE_MAX_COLOR_BUFS];
   struct yttrium_venus_object depth_image_view_obj;
   struct yttrium_venus_object render_pass_obj;
   struct yttrium_venus_object framebuffer_obj;
   struct yttrium_venus_object descriptor_set_layout_obj;
   struct yttrium_venus_object descriptor_pool_obj;
   struct yttrium_venus_object descriptor_set_obj;
   struct yttrium_venus_object
      sampler_objs[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   struct yttrium_venus_object pipeline_layout_obj;
   struct yttrium_venus_object pipeline_obj;
   struct yttrium_venus_object push_descriptor_set_layout_obj;
   struct yttrium_venus_object push_descriptor_set_layout_alt_obj;
   struct yttrium_venus_object push_pipeline_layout_obj;
   struct yttrium_venus_object push_pipeline_layout_alt_obj;
   struct yttrium_venus_object push_pipeline_obj;
   VkImageView image_views[PIPE_MAX_COLOR_BUFS];
   uint32_t color_attachment_count;
   VkSampleCountFlagBits render_samples;
   bool use_mrss;
   VkRenderPass render_pass;
   VkFramebuffer framebuffer;
   VkDescriptorSetLayout descriptor_set_layout;
   VkDescriptorPool descriptor_pool;
   VkDescriptorSet descriptor_set;
   VkSampler samplers[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint32_t custom_border_color_sampler_count;
   VkPipelineLayout pipeline_layout;
   VkPipeline pipeline;
   VkDescriptorSetLayout push_descriptor_set_layout;
   VkDescriptorSetLayout push_descriptor_set_layout_alt;
   VkPipelineLayout push_pipeline_layout;
   VkPipelineLayout push_pipeline_layout_alt;
   VkPipeline push_pipeline;
   VkImageView depth_image_view;
   struct yttrium_venus_sampled_image
      sampled_descriptor_cache[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   struct pipe_resource
      *sampled_descriptor_cache_resources
         [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint64_t sampled_descriptor_cache_object_ids
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint32_t sampled_descriptor_cache_count;
   bool sampled_descriptor_cache_initialized;
   struct yttrium_venus_ubo_slot ubos[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   uint32_t ubo_count;
   uint32_t ubo_descriptor_count;
   uint32_t sampled_image_descriptor_count;
   uint32_t sampled_buffer_descriptor_count;
   uint32_t storage_image_descriptor_count;
   uint32_t storage_buffer_descriptor_count;
   uint64_t storage_image_mask;
   uint64_t storage_buffer_mask;
   bool has_sampled_image;
   bool has_sampled_buffer;
   bool has_storage_image;
   bool has_storage_buffer;
   struct yttrium_shader_state *generated_vs;
   struct yttrium_shader_state *generated_gs;
   struct yttrium_shader_state *generated_fs;
};

void
yttrium_pipeline_destroy(struct yttrium_venus *venus,
                         struct yttrium_pipeline *pipeline);

bool
yttrium_pipeline_cache_init(struct yttrium_context *yctx);

void
yttrium_pipeline_cache_fini(struct yttrium_context *yctx);

void
yttrium_compute_pipeline_cache_invalidate(
   struct yttrium_context *yctx, const struct yttrium_shader_state *shader);

void
yttrium_pipeline_state_changed(struct yttrium_context *yctx);

void
yttrium_pipeline_fast_state_changed(struct yttrium_context *yctx);

enum yttrium_pipeline_draw_result {
   YTTRIUM_PIPELINE_DRAW_UNSUPPORTED = 0,
   YTTRIUM_PIPELINE_DRAW_EMITTED = 1,
   YTTRIUM_PIPELINE_DRAW_EMIT_FAILED = 2,
};

enum yttrium_pipeline_draw_result
yttrium_pipeline_try_draw(struct pipe_context *ctx,
                          struct yttrium_resource *dst,
                          const struct pipe_draw_info *info,
                          const struct pipe_draw_indirect_info *indirect,
                          const struct pipe_draw_start_count_bias *draws,
                          unsigned num_draws,
                          const struct yttrium_venus_draw_state *draw_state);

void
yttrium_launch_grid(struct pipe_context *ctx,
                    const struct pipe_grid_info *info);

#ifdef __cplusplus
}
#endif

#endif /* YTTRIUM_PIPELINE_H */
