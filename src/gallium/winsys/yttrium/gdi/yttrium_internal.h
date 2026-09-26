/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_INTERNAL_H
#define YTTRIUM_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>
#include <windows.h>

#include "gdikmt/gdikmt.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "compiler/nir/nir_shader_compiler_options.h"
#include "util/simple_mtx.h"
#include "util/slab.h"
#include "util/u_idalloc.h"
#include "util/u_threaded_context.h"
#include "util/u_inlines.h"
#include "virtio/wddm/viogpu_wddm_driver.h"

#include "yttrium_shader.h"
#include "yttrium_venus.h"

struct yttrium_pipeline;
struct yttrium_compute_pipeline_cache_entry;
struct yttrium_buffer_storage;

#define YTTRIUM_COMPUTE_PIPELINE_CACHE_SIZE 64
#define YTTRIUM_ORDERED_UPLOAD_POOL_MAX_ENTRIES 64
#define YTTRIUM_ORDERED_UPLOAD_POOL_MAX_BYTES (64ull * 1024ull * 1024ull)
#define YTTRIUM_BUFFER_REPLACEMENT_POOL_MAX_ENTRIES 256
#define YTTRIUM_BUFFER_REPLACEMENT_POOL_MAX_BYTES (64ull * 1024ull * 1024ull)

struct yttrium_ordered_upload_pool_entry {
   void *data;
   uint64_t capacity;
   bool direct_backing;
   D3DKMT_HANDLE hAllocation;
   HANDLE hResource;
   HANDLE hAllocationResource;
   bool hResourceIsD3D9Runtime;
   uint32_t venus_res_id;
   uint64_t venus_mem_id;
   uint32_t map_info;
   bool map_is_blob;
   bool owns_allocation;
   bool allocation_destroyed_by_runtime;
   struct yttrium_venus_resource venus;
};

struct yttrium_screen {
   struct pipe_screen base;
   struct slab_parent_pool transfer_pool;
   bool transfer_pool_initialized;
   struct util_idalloc_mt buffer_ids;
   bool buffer_ids_initialized;
   simple_mtx_t ordered_upload_pool_lock;
   bool ordered_upload_pool_initialized;
   struct yttrium_ordered_upload_pool_entry
      ordered_upload_pool[YTTRIUM_ORDERED_UPLOAD_POOL_MAX_ENTRIES];
   unsigned ordered_upload_pool_count;
   uint64_t ordered_upload_pool_bytes;
   simple_mtx_t buffer_replacement_pool_lock;
   bool buffer_replacement_pool_initialized;
   bool buffer_replacement_pool_enabled;
   struct yttrium_buffer_storage *
      buffer_replacement_pool[YTTRIUM_BUFFER_REPLACEMENT_POOL_MAX_ENTRIES];
   unsigned buffer_replacement_pool_count;
   uint64_t buffer_replacement_pool_bytes;
   struct gdikmt_device *device;
   VIOGPU_ADAPTERINFO adapter_info;
   struct yttrium_venus *venus;
   char *capture_textured_draw_path;
   unsigned capture_textured_draw_limit;
   unsigned capture_textured_draw_count;
   char *capture_skipped_draw_path;
   unsigned capture_skipped_draw_limit;
   unsigned capture_skipped_draw_count;
   struct nir_shader_compiler_options nir_options;
   bool glsl_type_singleton_ref;
};

struct yttrium_constant_buffer {
   struct pipe_resource *buffer;
   unsigned buffer_offset;
   unsigned buffer_size;
   const void *user_buffer;
};

struct yttrium_stream_output_target {
   struct pipe_stream_output_target base;
   struct yttrium_venus_resource counter;
   bool counter_buffer_valid;
   unsigned output_buffer;

   /*
    * Where writing starts, from set_stream_output_targets.  Separate from
    * base.buffer_offset on purpose: that one, with base.buffer_size, is the
    * window the target was created over and does not move.  Writing the
    * append offset into it left the window running past the end of the
    * buffer - 0x17de0 + 0x4c600 in a 0x4d000 resource - and the draw was
    * rejected as a bad target.
    */
   unsigned append_offset;
};

struct yttrium_sampler_state {
   struct pipe_sampler_state state;
};

struct yttrium_dsa_state {
   struct pipe_depth_stencil_alpha_state state;
};

struct yttrium_readback_mapping {
   D3DKMT_HANDLE hAllocation;
   HANDLE hResource;
   uint32_t venus_res_id;
   void *map;
   uint32_t map_info;
   bool map_is_blob;
};

struct yttrium_vertex_elements_state {
   /* Original shader-location span, including PIPE_FORMAT_NONE holes. */
   unsigned num_elements;
   struct pipe_vertex_element elements[PIPE_MAX_ATTRIBS];
   unsigned num_bindings;
   uint8_t binding_map[PIPE_MAX_ATTRIBS];
   uint32_t binding_divisor[PIPE_MAX_ATTRIBS];
   VkVertexInputBindingDescription bindings[PIPE_MAX_ATTRIBS];
   /* Compact Vulkan attributes; location indexes the original elements. */
   unsigned num_attribs;
   VkVertexInputAttributeDescription attribs[PIPE_MAX_ATTRIBS];
   bool vk_vertex_input_valid;
};

struct yttrium_rasterizer_state {
   struct pipe_rasterizer_state state;
};

struct yttrium_blend_state {
   struct pipe_blend_state state;
   /* Immutable translations; all slots are initialized, including unused RTs. */
   VkLogicOp logic_op;
   VkBool32 rt_blend_enable[PIPE_MAX_COLOR_BUFS];
   VkColorComponentFlags rt_color_write_mask[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_src_color_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_dst_color_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendOp rt_color_blend_op[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_src_alpha_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendFactor rt_dst_alpha_blend_factor[PIPE_MAX_COLOR_BUFS];
   VkBlendOp rt_alpha_blend_op[PIPE_MAX_COLOR_BUFS];
};

/*
 * One cached index bounds result.  Keyed by the resource's cache_id rather
 * than its address, because a freed resource can be replaced at the same
 * address and would otherwise alias a stale entry into a live draw.
 */
struct yttrium_index_bounds_entry {
   uint64_t cache_id;
   uint64_t offset;
   uint32_t serial;
   uint32_t count;
   uint32_t type;
   uint32_t min;
   uint32_t max;
   bool restart;
   bool valid;
};

/*
 * Four ways retain colliding static ranges without making lookup expensive.
 * Every entry is recomputable, so a full set can use round-robin replacement.
 */
#define YTTRIUM_INDEX_BOUNDS_CACHE_SET_COUNT 2048
#define YTTRIUM_INDEX_BOUNDS_CACHE_WAYS 4

struct yttrium_gdi_present_ticket;
struct yttrium_gdi_flush_issuance;

struct yttrium_context {
   struct pipe_context base;
   struct threaded_context *threaded;
   volatile LONG draw_failure;
   char pending_flush_label[96];
   bool pending_flush_label_valid;
   struct yttrium_gdi_present_ticket *pending_present_ticket;
   struct yttrium_gdi_flush_issuance *pending_flush_issuance;
   struct yttrium_index_bounds_entry
      index_bounds_cache[YTTRIUM_INDEX_BOUNDS_CACHE_SET_COUNT]
                        [YTTRIUM_INDEX_BOUNDS_CACHE_WAYS];
   uint8_t index_bounds_cache_next[YTTRIUM_INDEX_BOUNDS_CACHE_SET_COUNT];
   struct gdikmt_context *kmt_ctx;
   struct yttrium_query *queries;
   struct yttrium_query *active_queries;
   struct pipe_framebuffer_state fb;
   /* Diagnostic state for RT_STATS_EVERY; see yttrium_report_rt_stats. */
   unsigned rt_stats_draw_index;
   unsigned rt_stats_prev_frame_draws;
   unsigned rt_stats_frame_seen;
   uint32_t rt_stats_fb_sig;
   bool rt_stats_sampling;
   struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS];
   unsigned num_vertex_buffers;
   struct yttrium_vertex_elements_state *vertex_elements;
   struct yttrium_rasterizer_state *rasterizer;
   struct yttrium_blend_state *blend;
   struct yttrium_dsa_state *dsa;
   struct yttrium_shader_state *shaders[MESA_SHADER_STAGES];
   struct yttrium_shader_state *base_vs;
   struct yttrium_shader_state *vs_stream_output_gs;
   struct yttrium_shader_state *vs_stream_output_source_gs;
   struct yttrium_constant_buffer constant_buffers[MESA_SHADER_STAGES]
                                                  [PIPE_MAX_CONSTANT_BUFFERS];
   struct yttrium_sampler_state *sampler_states[MESA_SHADER_STAGES]
                                                [PIPE_MAX_SAMPLERS];
   struct pipe_sampler_view *sampler_views[MESA_SHADER_STAGES]
                                          [PIPE_MAX_SHADER_SAMPLER_VIEWS];
   struct pipe_image_view shader_images[MESA_SHADER_STAGES]
                                      [PIPE_MAX_SHADER_IMAGES];
   unsigned num_shader_images[MESA_SHADER_STAGES];
   struct pipe_stream_output_target *so_targets[PIPE_MAX_SO_BUFFERS];
   unsigned num_so_targets;
   struct pipe_resource *so_dummy_target;
   struct pipe_resource *so_dummy_buffer;
   /* Exact-size UAV-only scratch targets, most recently used first. */
   struct pipe_resource *uav_only_dummy_targets[4];
   struct pipe_blend_color blend_color;
   struct pipe_stencil_ref stencil_ref;
   unsigned sample_mask;
   struct pipe_viewport_state viewports[PIPE_MAX_VIEWPORTS];
   struct pipe_scissor_state scissors[PIPE_MAX_VIEWPORTS];
   unsigned num_viewports;
   unsigned num_scissors;
   uint8_t patch_vertices;
   struct yttrium_pipeline *pipeline;
   struct yttrium_pipeline **pipeline_cache;
   uint32_t *pipeline_cache_hash_heads;
   uint32_t *pipeline_cache_hash_next;
   uint32_t pipeline_cache_hash_size;
   uint32_t pipeline_cache_size;
   uint32_t pipeline_cache_count;
   uint32_t pipeline_cache_next;
   struct yttrium_compute_pipeline_cache_entry
      *compute_pipeline_cache[YTTRIUM_COMPUTE_PIPELINE_CACHE_SIZE];
   uint32_t compute_pipeline_cache_next;

   /*
    * Ordinary state setters advance pipeline_state_serial and clear the
    * current pointer.  These draw-local discriminants cover the remaining
    * inputs needed to reuse that pointer before rebuilding the complete key.
    */
   uint64_t pipeline_state_serial;
   uint64_t current_pipeline_state_serial;
   uint64_t current_pipeline_dst_cache_id;
   uint64_t current_pipeline_dst_image_id;
   uint64_t current_pipeline_zs_cache_id;
   uint64_t current_pipeline_zs_image_id;
   VkPrimitiveTopology current_pipeline_topology;
   VkBool32 current_pipeline_primitive_restart_enable;
   VkBool32 current_pipeline_targetless_stream_output;
   uint32_t current_pipeline_viewport_count;
   uint32_t current_pipeline_rasterization_samples;
   uint32_t current_pipeline_forced_sample_count;
   VkBool32 current_pipeline_forced_sample_interlock;
   uint32_t current_pipeline_rt_count;
   uint32_t current_pipeline_render_width;
   uint32_t current_pipeline_render_height;
   uint32_t current_pipeline_render_layers;
   uint32_t current_pipeline_depth_level;
   uint32_t current_pipeline_depth_layer;
   uint32_t current_pipeline_depth_layers;

   struct yttrium_venus_resource upload_staging;
   uint64_t upload_staging_mem_id;
   uint64_t upload_staging_size;
   uint64_t upload_staging_offset;
   struct yttrium_readback_mapping upload_staging_mapping;

   /*
    * Format-compatible depth-to-color copies use an image -> buffer -> image
    * sequence.  Keep that GPU scratch buffer alive across copies so the draw
    * thread does not synchronously create and bind a new buffer for every
    * copy.
    */
   struct yttrium_venus_resource image_copy_staging;
   uint64_t image_copy_staging_mem_id;
   uint64_t image_copy_staging_size;
};

struct yttrium_resource {
   union {
      struct pipe_resource base;
      struct threaded_resource threaded;
   };

   /*
    * Bumped wherever resource contents change.  Every pipe entry point that
    * can write a resource must bump this, or a cached index bounds result goes
    * stale and the draw uploads the wrong vertex range.
    */
   uint32_t contents_serial;

   /*
    * Process-unique, assigned once at create.  Index bounds cache entries key
    * on this instead of the resource address, which can be recycled.
    */
   uint64_t cache_id;
   struct yttrium_venus_ubo_version_cache ubo_version_cache;

   D3DKMT_HANDLE hAllocation;
   HANDLE hResource;
   HANDLE hAllocationResource;
   bool hResourceIsD3D9Runtime;
   uint32_t venus_res_id;
   uint64_t venus_mem_id;
   struct yttrium_venus_resource venus;

   uint64_t size;
   unsigned stride;
   unsigned layer_stride;
   bool display_target;
   bool primary_target;
   bool classic_display;
   bool owns_allocation;
   bool allocation_destroyed_by_runtime;

   void *map;
   uint32_t map_info;
   bool map_is_blob;

   void *data;
   uint64_t data_capacity;
   bool owns_data;
   bool data_dirty;
   /* Counter/indirect buffer writes make GPU storage authoritative.  Detached
    * CPU storage is refreshed only for an explicit CPU map, never for a draw. */
   bool gpu_buffer_written;
   bool direct_bind_unsafe;
   bool private_immutable_draw_buffer;
   bool ordered_worker_upload_buffer;
   bool ordered_worker_upload_direct_backing;
   struct yttrium_buffer_storage *replacement_storage;
   struct yttrium_resource *replacement_owner;
};

struct yttrium_query {
   struct threaded_query threaded;
   struct pipe_reference reference;
   struct yttrium_query *next;
   struct yttrium_query *active_next;
   enum pipe_query_type type;
   unsigned index;
   uint64_t completion_order;
   bool active;
   bool ready;
   union pipe_query_result result;
};

static inline struct yttrium_screen *
yttrium_screen(struct pipe_screen *screen)
{
   return (struct yttrium_screen *)screen;
}

static inline struct yttrium_context *
yttrium_context(struct pipe_context *ctx)
{
   return (struct yttrium_context *)ctx;
}

static inline struct yttrium_resource *
yttrium_resource(struct pipe_resource *resource)
{
   return (struct yttrium_resource *)resource;
}

static inline bool
yttrium_resource_is_venus_backed_display(const struct yttrium_resource *res)
{
   return res && res->display_target && !res->classic_display &&
          res->venus.initialized;
}

static inline bool
yttrium_resource_is_venus_color_attachment(const struct yttrium_resource *res)
{
   return res && !res->classic_display && res->venus.initialized &&
          !res->venus.buffer_backed && res->venus.image &&
          (res->venus.image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
}

#endif /* YTTRIUM_INTERNAL_H */
