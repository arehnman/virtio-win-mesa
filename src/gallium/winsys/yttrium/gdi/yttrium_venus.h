/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_VENUS_H
#define YTTRIUM_VENUS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

#include "compiler/shader_enums.h"
#include "pipe/p_state.h"
#include "util/format/u_formats.h"

#define YTTRIUM_SHADER_PUSH_CONSTANT_BYTES 256
#define YTTRIUM_SHADER_VS_PUSH_CONSTANT_OFFSET 0
#define YTTRIUM_SHADER_VS_PUSH_CONSTANT_BYTES 96
#define YTTRIUM_SHADER_FS_PUSH_CONSTANT_OFFSET 96
#define YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES 160

struct gdikmt_context;
struct gdikmt_device;
struct yttrium_pipeline;

struct yttrium_venus_object {
   uint64_t id;
};

struct yttrium_venus_present_publication {
   bool requested;
   uint32_t allocation;
   uint32_t scanout_id;
};

#define YTTRIUM_VENUS_MAX_DRAW_VERTICES 2048
#define YTTRIUM_VENUS_MAX_PIPELINE_UBO_BYTES (64 * 1024)
#define YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS 32
#define YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS \
   YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS
/* Sampled-resource slots are local to each shader stage.  Keep the active
 * descriptor limit compact, but reserve a disjoint binding bank for every GL
 * shader stage so (for example) VS s0 and FS s0 cannot alias.
 */
#define YTTRIUM_VENUS_SAMPLED_STAGE_COUNT MESA_SHADER_STAGES
#define YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES PIPE_MAX_SAMPLERS
#define YTTRIUM_VENUS_PIPELINE_SAMPLED_IMAGE_MASK UINT32_MAX
#define YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES PIPE_MAX_SHADER_IMAGES
#define YTTRIUM_VENUS_PIPELINE_STORAGE_IMAGE_MASK \
   UINT64_MAX
#define YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS 32
#define YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS 4
#define YTTRIUM_VENUS_MAX_DS_CLEAR_HISTORY 32
#define YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS 3
#define YTTRIUM_VENUS_SAMPLE_SWIZZLE_MASK \
   ((1u << YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS) - 1)

static inline enum pipe_swizzle
yttrium_venus_sample_swizzle_normalize(unsigned swizzle,
                                       enum pipe_swizzle fallback)
{
   switch (swizzle) {
   case PIPE_SWIZZLE_X:
   case PIPE_SWIZZLE_Y:
   case PIPE_SWIZZLE_Z:
   case PIPE_SWIZZLE_W:
   case PIPE_SWIZZLE_0:
   case PIPE_SWIZZLE_1:
      return (enum pipe_swizzle)swizzle;
   default:
      return fallback;
   }
}

static inline uint32_t
yttrium_venus_sample_swizzle_key(unsigned r,
                                 unsigned g,
                                 unsigned b,
                                 unsigned a)
{
   const enum pipe_swizzle sr =
      yttrium_venus_sample_swizzle_normalize(r, PIPE_SWIZZLE_X);
   const enum pipe_swizzle sg =
      yttrium_venus_sample_swizzle_normalize(g, PIPE_SWIZZLE_Y);
   const enum pipe_swizzle sb =
      yttrium_venus_sample_swizzle_normalize(b, PIPE_SWIZZLE_Z);
   const enum pipe_swizzle sa =
      yttrium_venus_sample_swizzle_normalize(a, PIPE_SWIZZLE_W);

   return ((uint32_t)sr << (0 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)) |
          ((uint32_t)sg << (1 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)) |
          ((uint32_t)sb << (2 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)) |
          ((uint32_t)sa << (3 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS));
}

#define YTTRIUM_VENUS_SAMPLE_SWIZZLE_IDENTITY \
   (((uint32_t)PIPE_SWIZZLE_X << (0 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)) | \
    ((uint32_t)PIPE_SWIZZLE_Y << (1 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)) | \
    ((uint32_t)PIPE_SWIZZLE_Z << (2 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)) | \
    ((uint32_t)PIPE_SWIZZLE_W << (3 * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)))

struct yttrium_venus_ubo_binding_layout {
   uint32_t binding;
   uint32_t descriptor_count;
   VkShaderStageFlags stage_flags;
};

#define YTTRIUM_VENUS_UBO_VERSION_CACHE_SIZE 4

struct yttrium_venus_ubo_version {
   uint64_t arena_generation;
   VkDeviceSize arena_offset;
   uint32_t contents_serial;
   uint32_t source_offset;
   uint32_t source_size;
   bool valid;
};

struct yttrium_venus_ubo_version_cache {
   struct yttrium_venus_ubo_version
      entries[YTTRIUM_VENUS_UBO_VERSION_CACHE_SIZE];
   uint32_t next;
};

struct yttrium_venus_ubo_upload {
   uint32_t binding;
   uint32_t array_element;
   const void *data;
   size_t size;
   struct yttrium_venus_resource *direct_resource;
   VkDeviceSize direct_offset;
   uint32_t source_contents_serial;
   uint32_t source_offset;
   struct yttrium_venus_ubo_version_cache *source_version_cache;
};

struct yttrium_venus_ubo_slot {
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceSize size;
   VkDeviceSize offset;
   uint32_t binding;
   uint32_t array_element;
   bool upload_reused;
   bool resource_version_cacheable;
};

struct yttrium_venus_sampler_state {
   VkFilter min_filter;
   VkFilter mag_filter;
   VkSamplerMipmapMode mipmap_mode;
   VkSamplerAddressMode address_mode_u;
   VkSamplerAddressMode address_mode_v;
   VkSamplerAddressMode address_mode_w;
   VkBool32 anisotropy_enable;
   VkBool32 compare_enable;
   VkCompareOp compare_op;
   VkBorderColor border_color;
   VkClearColorValue custom_border_color;
   float mip_lod_bias;
   float min_lod;
   float max_lod;
   float max_anisotropy;
};

struct yttrium_venus_sampled_binding_layout {
   uint32_t binding;
   uint32_t stage;
   uint32_t raw_slot;
   VkShaderStageFlags stage_flags;
   VkBool32 buffer;
   struct yttrium_venus_sampler_state sampler;
};

struct yttrium_venus_sampled_image {
   struct yttrium_venus_resource *resource;
   struct pipe_resource *pipe_resource;
   const void *buffer_data;
   size_t buffer_size;
   VkDeviceSize buffer_offset;
   VkDeviceSize buffer_range;
   uint32_t resource_id;
   uint32_t binding;
   uint32_t swizzle_key;
   VkImageViewType view_type;
   VkImageAspectFlags aspect_mask;
   uint32_t first_level;
   uint32_t level_count;
   uint32_t first_layer;
   uint32_t layer_count;
   enum pipe_format format;
   bool buffer;
};

struct yttrium_venus_storage_image {
   struct yttrium_venus_resource *resource;
   const void *buffer_data;
   size_t buffer_size;
   VkDeviceSize buffer_offset;
   VkDeviceSize buffer_range;
   uint32_t resource_id;
   uint32_t binding;
   VkImageViewType view_type;
   VkImageAspectFlags aspect_mask;
   uint32_t first_level;
   uint32_t level_count;
   uint32_t first_layer;
   uint32_t layer_count;
   enum pipe_format format;
   bool buffer;
};

struct yttrium_venus_vertex_upload {
   const void *data;
   struct yttrium_venus_resource *resource;
   uint32_t resource_id;
   size_t size;
   VkDeviceSize buffer_offset;
   bool host_write_pending;
};

struct yttrium_venus_stream_output_target {
   struct yttrium_venus_resource *resource;
   uint32_t resource_id;
   VkDeviceSize buffer_offset;
   VkDeviceSize buffer_size;
   struct yttrium_venus_resource *counter_resource;
   bool counter_buffer_valid;
};

struct yttrium_venus_sample_image_view {
   struct yttrium_venus_sample_image_view *next;
   struct yttrium_venus_object obj;
   VkImageView view;
   VkFormat vk_format;
   uint32_t swizzle_key;
   VkImageViewType view_type;
   VkImageAspectFlags aspect_mask;
   uint32_t first_level;
   uint32_t level_count;
   uint32_t first_layer;
   uint32_t layer_count;
};

struct yttrium_venus_sample_buffer_view {
   struct yttrium_venus_sample_buffer_view *next;
   struct yttrium_venus_object obj;
   VkBufferView view;
   VkFormat vk_format;
   VkDeviceSize offset;
   VkDeviceSize range;
};

struct yttrium_venus_ds_clear_record {
   VkImageAspectFlags aspects;
   float depth;
   uint32_t stencil;
   uint32_t level;
   uint32_t layer;
   uint32_t x;
   uint32_t y;
   uint32_t width;
   uint32_t height;
};

struct yttrium_venus_memory_mapping {
   uint32_t hAllocation;
   void *hResource;
   uint64_t size;
   void *map;
   bool map_is_blob;
};

struct yttrium_venus_resource {
   struct pipe_resource *owner;
   struct yttrium_venus_object image_obj;
   struct yttrium_venus_object buffer_obj;
   struct yttrium_venus_object memory_obj;
   struct yttrium_venus_object image_view_obj;
   struct yttrium_venus_object render_pass_obj;
   struct yttrium_venus_object framebuffer_obj;
   struct yttrium_venus_object pipeline_layout_obj;
   struct yttrium_venus_object descriptor_set_layout_obj;
   struct yttrium_venus_object descriptor_pool_obj;
   struct yttrium_venus_object descriptor_set_obj;
   struct yttrium_venus_object sampler_obj;
   struct yttrium_venus_object vertex_shader_obj;
   struct yttrium_venus_object fragment_shader_obj;
   struct yttrium_venus_object pipeline_obj;
   struct yttrium_venus_object draw_vertex_buffer_obj;
   struct yttrium_venus_object draw_vertex_memory_obj;
   struct yttrium_venus_object draw_index_buffer_obj;
   struct yttrium_venus_object draw_index_memory_obj;
   struct yttrium_venus_object device_local_draw_buffer_obj;
   struct yttrium_venus_object device_local_draw_memory_obj;
   VkImage image;
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkImageView image_view;
   struct yttrium_venus_sample_image_view *sample_image_views;
   struct yttrium_venus_sample_buffer_view *sample_buffer_views;
   VkRenderPass render_pass;
   VkFramebuffer framebuffer;
   VkPipelineLayout pipeline_layout;
   VkDescriptorSetLayout descriptor_set_layout;
   VkDescriptorPool descriptor_pool;
   VkDescriptorSet descriptor_set;
   VkSampler sampler;
   VkShaderModule vertex_shader;
   VkShaderModule fragment_shader;
   VkPipeline pipeline;
   VkBuffer draw_vertex_buffer;
   VkDeviceMemory draw_vertex_memory;
   VkBuffer draw_index_buffer;
   VkDeviceMemory draw_index_memory;
   VkBuffer device_local_draw_buffer;
   VkDeviceMemory device_local_draw_memory;
   struct yttrium_venus_memory_mapping draw_vertex_mapping;
   struct yttrium_venus_memory_mapping draw_index_mapping;
   VkImageLayout layout;
   VkImageUsageFlags image_usage;
   VkImageAspectFlags initialized_aspects;
   VkBufferUsageFlags buffer_usage;
   VkFormat vk_format;
   uint64_t allocation_size;
   uint64_t draw_vertex_buffer_size;

   /*
    * How far this resource's vertex arena has been written.  Uploads roll
    * forward from here rather than restarting at 0 each batch, so a batch
    * writes bytes no in-flight batch is reading and needs no wait.  Reset to 0
    * when the roll wraps, and whenever the arena itself is reallocated.
    */
   uint64_t draw_vertex_roll_base;
   uint64_t draw_index_buffer_size;

   /*
    * How far this resource's index arena has been written.  Uploads roll
    * forward from here rather than restarting at 0 each batch, so a batch
    * writes bytes no in-flight batch is reading and needs no wait.  Reset to 0
    * when the roll wraps, and whenever the arena itself is reallocated.
    */
   uint64_t draw_index_roll_base;
   uint64_t draw_vertex_buffer_generation;
   uint64_t draw_index_buffer_generation;
   uint64_t device_local_draw_buffer_size;
   uint32_t draw_source_contents_serial;
   uint32_t device_local_draw_contents_serial;
   uint32_t device_local_draw_pending_serial;
   uint64_t image_offset;
   uint64_t image_size;
   uint64_t image_row_pitch;
   uint64_t image_array_pitch;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t levels;
   uint32_t layers;
   VkSampleCountFlagBits samples;
   bool cpu_readback;
   bool initialized;
   bool buffer_backed;
   bool contents_initialized;
   bool graphics_ready;
   uint32_t graphics_mode;
   uint32_t graphics_level;
   uint32_t graphics_layer;
   uint32_t graphics_layers;
   VkPrimitiveTopology graphics_topology;
   uint32_t graphics_cull_mode;
   uint32_t graphics_front_face;
   VkBool32 graphics_blend_enable;
   VkSampleMask graphics_sample_mask;
   VkBool32 graphics_alpha_to_coverage_enable;
   VkColorComponentFlags graphics_color_write_mask;
   VkBlendFactor graphics_src_color_blend_factor;
   VkBlendFactor graphics_dst_color_blend_factor;
   VkBlendOp graphics_color_blend_op;
   VkBlendFactor graphics_src_alpha_blend_factor;
   VkBlendFactor graphics_dst_alpha_blend_factor;
   VkBlendOp graphics_alpha_blend_op;
   struct yttrium_venus_ds_clear_record
      ds_clear_history[YTTRIUM_VENUS_MAX_DS_CLEAR_HISTORY];
   uint32_t ds_clear_history_count;
   bool ds_clear_history_valid;
   bool device_local_draw_contents_valid;
   bool device_local_draw_pending_valid;
};

struct yttrium_venus_triangle_vertex {
   float position[4];
   float color[4];
};

struct yttrium_venus_textured_vertex {
   float position[4];
   float color[4];
   float texcoord[2];
};

struct yttrium_venus_draw_state {
   uint32_t viewport_count;
   VkViewport viewports[PIPE_MAX_VIEWPORTS];
   VkRect2D scissors[PIPE_MAX_VIEWPORTS];
   VkPrimitiveTopology topology;
   VkBool32 primitive_restart_enable;
   VkBool32 rasterizer_discard_enable;
   /*
    * The draw has live stream-output state but no colour or depth target.
    * resource remains a lifetime/submission anchor; it is not an attachment.
    */
   VkBool32 targetless_stream_output;
   VkCullModeFlags cull_mode;
   VkFrontFace front_face;
   VkBool32 depth_bias_enable;
   VkBool32 depth_clamp_enable;
   float depth_bias_constant_factor;
   float depth_bias_clamp;
   float depth_bias_slope_factor;
   VkBool32 blend_enable;
   VkSampleMask sample_mask;
   uint32_t rasterization_samples;
   uint32_t forced_sample_count;
   VkBool32 forced_sample_interlock;
   VkBool32 alpha_to_coverage_enable;
   VkBool32 logic_op_enable;
   VkLogicOp logic_op;
   VkColorComponentFlags color_write_mask;
   VkBlendFactor src_color_blend_factor;
   VkBlendFactor dst_color_blend_factor;
   VkBlendOp color_blend_op;
   VkBlendFactor src_alpha_blend_factor;
   VkBlendFactor dst_alpha_blend_factor;
   VkBlendOp alpha_blend_op;
   uint32_t rt_count;
   VkBool32 rt_blend_enable[PIPE_MAX_COLOR_BUFS];
   VkColorComponentFlags rt_color_write_mask[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_src_color_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_dst_color_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendOp rt_color_blend_op[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_src_alpha_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_dst_alpha_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendOp rt_alpha_blend_op[PIPE_MAX_COLOR_BUFS];
   VkBool32 depth_test_enable;
   VkBool32 depth_write_enable;
   VkCompareOp depth_compare_op;
   VkBool32 alpha_test_enable;
   uint32_t alpha_func;
   float alpha_ref_value;
   VkBool32 stencil_test_enable;
   VkStencilOpState stencil_front;
   VkStencilOpState stencil_back;
   float blend_constants[4];
   uint32_t render_level;
   /*
    * render_level/render_layer describe the first colour target and remain the
    * common render-extent anchor.  Each attachment still needs its own mip and
    * slice because Vulkan selects both on the image view.
    */
   uint32_t render_layer;
   uint32_t rt_level[PIPE_MAX_COLOR_BUFS];
   uint32_t rt_layer[PIPE_MAX_COLOR_BUFS];
   uint32_t render_layers;
   uint32_t render_width;
   uint32_t render_height;
   uint32_t depth_level;
   uint32_t depth_layer;
   uint32_t depth_layers;
   uint16_t push_constant_vs_size;
   uint16_t push_constant_fs_size;
   uint8_t push_constant_data[YTTRIUM_SHADER_PUSH_CONSTANT_BYTES];
   /* Per-draw arguments, never part of the immutable pipeline key. */
   struct yttrium_venus_resource *indirect_resource;
   VkDeviceSize indirect_offset;
};

struct yttrium_venus;

struct yttrium_venus *
yttrium_venus_create(struct gdikmt_device *device);

struct gdikmt_context *
yttrium_venus_get_kmt_context(struct yttrium_venus *venus);

void
yttrium_venus_destroy(struct yttrium_venus *venus);

bool
yttrium_venus_create_shader_module(struct yttrium_venus *venus,
                                   struct yttrium_venus_object *obj,
                                   VkShaderModule *shader,
                                   const uint32_t *code,
                                   size_t code_size,
                                   const char *label,
                                   VkResult *out_result);

VkFormat
yttrium_venus_pipe_format(enum pipe_format format);

void
yttrium_venus_destroy_shader_module(struct yttrium_venus *venus,
                                    struct yttrium_venus_object *obj,
                                    VkShaderModule shader);

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
                            const struct yttrium_venus_draw_state *draw_state);

bool
yttrium_venus_supports_load_store_op_none(struct yttrium_venus *venus);

bool
yttrium_venus_supports_multisampled_render_to_single_sampled(
   struct yttrium_venus *venus,
   uint32_t sample_count);

bool
yttrium_venus_supports_forced_sample_interlock(
   struct yttrium_venus *venus,
   uint32_t sample_count);

VkSampleCountFlags
yttrium_venus_framebuffer_color_sample_counts(struct yttrium_venus *venus);

bool
yttrium_venus_sampled_texture_format_supported(
   struct yttrium_venus *venus,
   enum pipe_format format,
   enum pipe_texture_target target);

bool
yttrium_venus_storage_image_without_format_supported(
   struct yttrium_venus *venus, bool read, bool write);

bool
yttrium_venus_storage_image_formatless_format_supported(
   struct yttrium_venus *venus, enum pipe_format format,
   bool read, bool write);

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
   uint64_t storage_buffer_mask);

void
yttrium_venus_pipeline_fini(struct yttrium_venus *venus,
                            struct yttrium_pipeline *pipeline);

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
                            const struct yttrium_venus_draw_state *draw_state);

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
   uint32_t grid_z);

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
                            uint32_t layer_count);

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
                                 uint32_t height);

bool
yttrium_venus_draw_vertex_buffer_vertices(struct yttrium_venus *venus,
                                          struct yttrium_venus_resource *resource,
                                          uint32_t resource_id,
                                          const struct yttrium_venus_triangle_vertex *vertices,
                                          uint32_t vertex_count,
                                          const struct yttrium_venus_draw_state *draw_state);

bool
yttrium_venus_draw_textured_vertices(struct yttrium_venus *venus,
                                     struct yttrium_venus_resource *resource,
                                     uint32_t resource_id,
                                     struct yttrium_venus_resource *sampled,
                                     uint32_t sampled_resource_id,
                                     const struct yttrium_venus_textured_vertex *vertices,
                                     uint32_t vertex_count,
                                     const struct yttrium_venus_draw_state *draw_state);

bool
yttrium_venus_update_buffer(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                            uint64_t offset,
                            uint64_t size,
                            const void *data);

bool
yttrium_venus_clear_buffer(struct yttrium_venus *venus,
                           struct yttrium_venus_resource *resource,
                           uint64_t offset,
                           uint64_t size,
                           uint32_t value);

bool
yttrium_venus_copy_image_to_display_buffer(struct yttrium_venus *venus,
                                           struct yttrium_venus_resource *render,
                                           struct yttrium_venus_resource *scanout,
                                           uint32_t render_resource_id,
                                           uint32_t scanout_resource_id,
                                           uint32_t width,
                                           uint32_t height,
                                           enum pipe_format pipe_format,
                                           uint32_t scanout_stride);

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
                                                  uint32_t scanout_stride);

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
                                                         VkImageAspectFlags copy_aspect);

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
                                 uint32_t height);

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
                                       VkImageAspectFlags aspect_mask);

bool
yttrium_venus_flush(struct yttrium_venus *venus);

bool
yttrium_venus_flush_async(struct yttrium_venus *venus);

bool
yttrium_venus_flush_async_present_publish(
   struct yttrium_venus *venus,
   const struct yttrium_venus_present_publication *publication);

bool
yttrium_venus_flush_labeled(struct yttrium_venus *venus, const char *label);

bool
yttrium_venus_flush_async_labeled(struct yttrium_venus *venus,
                                  const char *label);

bool
yttrium_venus_flush_async_present_publish_labeled(
   struct yttrium_venus *venus,
   const char *label,
   const struct yttrium_venus_present_publication *publication);

uint64_t
yttrium_venus_last_submit_order(struct yttrium_venus *venus);

bool
yttrium_venus_submit_order_complete(struct yttrium_venus *venus,
                                    uint64_t submit_order);

const char *
yttrium_venus_current_flush_label(void);

bool
yttrium_venus_wait_resource(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                            const char *label);

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
                                 bool linear_filter);

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
                                    uint32_t height);

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
                                           enum pipe_format pipe_format);

bool
yttrium_venus_copy_buffer_to_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *src,
                                    struct yttrium_venus_resource *dst,
                                    VkDeviceSize src_offset,
                                    VkDeviceSize dst_offset,
                                    VkDeviceSize size);

bool
yttrium_venus_create_display_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint64_t allocation_size,
                                    uint64_t *out_memory_id);

bool
yttrium_venus_create_bind_buffer(struct yttrium_venus *venus,
                                 struct yttrium_venus_resource *resource,
                                 uint64_t allocation_size,
                                 VkBufferUsageFlags usage,
                                 uint64_t *out_memory_id);

bool
yttrium_venus_create_stream_output_buffer(struct yttrium_venus *venus,
                                          struct yttrium_venus_resource *resource,
                                          uint64_t allocation_size,
                                          uint64_t *out_memory_id);

bool
yttrium_venus_transform_feedback_enabled(const struct yttrium_venus *venus);

bool
yttrium_venus_transform_feedback_draw_enabled(const struct yttrium_venus *venus);

bool
yttrium_venus_vertex_attribute_divisor_supported(struct yttrium_venus *venus,
                                                 uint32_t divisor);

bool
yttrium_venus_depth_clamp_enabled(const struct yttrium_venus *venus);

bool
yttrium_venus_logic_op_enabled(const struct yttrium_venus *venus);

uint32_t
yttrium_venus_max_dual_source_render_targets(struct yttrium_venus *venus);

float
yttrium_venus_max_sampler_anisotropy(struct yttrium_venus *venus);

float
yttrium_venus_max_sampler_lod_bias(struct yttrium_venus *venus);

uint32_t
yttrium_venus_max_transform_feedback_stride(const struct yttrium_venus *venus);

uint32_t
yttrium_venus_max_viewports(struct yttrium_venus *venus);

uint32_t
yttrium_venus_mipmap_precision_bits(const struct yttrium_venus *venus);

bool
yttrium_venus_create_sampled_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint64_t allocation_size,
                                    enum pipe_format pipe_format,
                                    uint64_t *out_memory_id);

bool
yttrium_venus_ensure_null_sampled_buffer(struct yttrium_venus *venus,
                                         enum pipe_format pipe_format,
                                         uint64_t min_size,
                                         struct yttrium_venus_resource **out_resource,
                                         uint32_t *out_resource_id);

bool
yttrium_venus_create_display_image(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t width,
                                   uint32_t height,
                                   enum pipe_format pipe_format,
                                   uint64_t min_allocation_size,
                                   uint64_t *out_memory_id,
                                   uint64_t *out_allocation_size);

bool
yttrium_venus_create_color_attachment_image(struct yttrium_venus *venus,
                                            struct yttrium_venus_resource *resource,
                                            uint32_t width,
                                            uint32_t height,
                                            uint32_t levels,
                                            uint32_t layers,
                                            enum pipe_format pipe_format,
                                            uint64_t *out_allocation_size);

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
                                           uint64_t *out_allocation_size);

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
                                            uint64_t *out_allocation_size);

bool
yttrium_venus_create_depth_stencil_image(struct yttrium_venus *venus,
                                         struct yttrium_venus_resource *resource,
                                         uint32_t width,
                                         uint32_t height,
                                         enum pipe_format pipe_format,
                                         uint64_t *out_allocation_size);

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
                                  uint32_t height);

bool
yttrium_venus_import_display_image(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t resource_id,
                                   uint32_t width,
                                   uint32_t height,
                                   enum pipe_format pipe_format,
                                   uint64_t min_allocation_size,
                                   uint64_t *out_memory_id,
                                   uint64_t *out_allocation_size);

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
   uint64_t *out_allocation_size);

struct yttrium_venus_allocation_snapshot {
   uint32_t hAllocation;
   void *hResource;
   void *hAllocationResource;
   bool hResourceIsD3D9Runtime;
   uint64_t size;
   void *map;
   uint32_t map_info;
   bool map_is_blob;
   bool owns_allocation;
   bool allocation_destroyed_by_runtime;
};

bool
yttrium_venus_resource_fini(struct yttrium_venus *venus,
                            struct gdikmt_context *ctx,
                            struct yttrium_venus_resource *resource,
                            const struct yttrium_venus_allocation_snapshot *allocation);

#endif /* YTTRIUM_VENUS_H */
