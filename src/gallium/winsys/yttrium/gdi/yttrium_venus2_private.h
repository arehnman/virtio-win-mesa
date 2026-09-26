/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_VENUS2_PRIVATE_H
#define YTTRIUM_VENUS2_PRIVATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <windows.h>
#include <winternl.h>

#include <d3dkmthk.h>

#include "gdikmt/gdikmt.h"
#include "pipe/p_defines.h"
#include "vn_ring.h"
#include "yttrium_pipeline.h"
#include "yttrium_venus2.h"
#include "yttrium_venus2_ring_seqno.h"

struct hash_table_u64;

#ifndef VIRTGPU_BLOB_MEM_HOST3D
#define VIRTGPU_BLOB_MEM_HOST3D 0x0002
#endif
#ifndef VIRTGPU_BLOB_FLAG_USE_MAPPABLE
#define VIRTGPU_BLOB_FLAG_USE_MAPPABLE 0x0001
#endif
#ifndef VIRTGPU_BLOB_FLAG_USE_SHAREABLE
#define VIRTGPU_BLOB_FLAG_USE_SHAREABLE 0x0002
#endif
#ifndef VIRTGPU_DRM_CAPSET_VENUS
#define VIRTGPU_DRM_CAPSET_VENUS 4
#endif

/*
 * A single command has to fit here whole.  128 KB was enough until a real
 * D3D11 workload turned up: Superposition compiles a shader whose
 * vkCreateShaderModule encodes to 137120 bytes, the ring write failed, and
 * because the create path reports success regardless the driver went on to
 * build a pipeline against a shader module the host had never been told
 * about - "failed to look up object N of type 15" thousands of commands
 * later, then a wait-seqno timeout.
 *
 * SPIR-V has no size limit, so no fixed ring is correct in general.  A fixed
 * size is still not the right answer - 1 MB per device is more than an ordinary
 * device encodes - and two designs would remove the trade-off: keep the ring
 * small and give oversized commands their own shmem, as upstream Venus does, or
 * grow the ring on demand.  Until then the size is a knob.
 *
 * Raising this to 1 MB briefly looked like it cost the d3d9 visual suite: 146249
 * tests went from 4 failures to a crash on a null surface after
 * CreateRenderTarget returned E_OUTOFMEMORY, and dropping back to 128 KB
 * restored the baseline.  That was the ring size amplifying a different bug,
 * not causing one - allocations were failing to destroy, so any per-device
 * allocation accumulated until host memory ran out, and 1 MB simply got there
 * 8x sooner.  With that ownership bug fixed, all four suites pass at 1 MB.
 * Worth remembering when a size or a cache looks guilty: churn exposes leaks.
 */
#define YTTRIUM_VENUS_RING_BUFFER_SIZE_DEFAULT (1024 * 1024)
#define YTTRIUM_VENUS_RING_BUFFER_SIZE_MAX (8 * 1024 * 1024)
#define YTTRIUM_VENUS_RING_BUFFER_SIZE_ENV \
   "D3D10UMD_YTTRIUM_RING_BUFFER_SIZE"
#define YTTRIUM_VENUS_REPLY_SIZE (1024 * 1024)
#define YTTRIUM_VENUS_RING_IDLE_TIMEOUT_NS (1000 * 1000)
#define YTTRIUM_VENUS_FENCE_WAIT_MS 5000u
#define YTTRIUM_VENUS_RING_WAIT_HEADROOM_MS 10000u
#define YTTRIUM_VENUS_RING_WAIT_MS \
   (YTTRIUM_VENUS_FENCE_WAIT_MS + YTTRIUM_VENUS_RING_WAIT_HEADROOM_MS)
#define YTTRIUM_VENUS_FENCE_WAIT_NS \
   (YTTRIUM_VENUS_FENCE_WAIT_MS * 1000ull * 1000ull)
#define YTTRIUM_VENUS_TIMING_SLOW_US 1000
#define YTTRIUM_VENUS_SYNC_WAIT_DIAG_WINDOW_US 1000000
#define YTTRIUM_VENUS_SYNC_WAIT_DIAG_MAX_COUNT 1024
#define YTTRIUM_VENUS_SYNC_WAIT_DIAG_CMD_SLOTS 8
#define YTTRIUM_VENUS_MAX_PHYSICAL_DEVICES 16
#define YTTRIUM_VENUS_ASYNC_BATCH_ENV "D3D10UMD_YTTRIUM_ASYNC_BATCH"
#define YTTRIUM_VENUS_GROUP_QUEUE_SUBMITS_ENV \
   "D3D10UMD_YTTRIUM_GROUP_QUEUE_SUBMITS"
#define YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_OP_THRESHOLD_ENV \
   "D3D10UMD_YTTRIUM_GROUP_QUEUE_SUBMIT_OP_THRESHOLD"
#define YTTRIUM_VENUS_BATCH_FENCE_FEEDBACK_ENV \
   "D3D10UMD_YTTRIUM_BATCH_FENCE_FEEDBACK"
#define YTTRIUM_VENUS_NATIVE_DRAW_BATCH_ENV "D3D10UMD_YTTRIUM_DRAW_BATCH"
#define YTTRIUM_VENUS_NATIVE_DRAW_BATCH_LIMIT_ENV \
   "D3D10UMD_YTTRIUM_DRAW_BATCH_LIMIT"
#define YTTRIUM_VENUS_CPU_VERTEX_BATCH_ENV \
   "D3D10UMD_YTTRIUM_CPU_VERTEX_BATCH"
#define YTTRIUM_VENUS_SAMPLED_CPU_VERTEX_BATCH_ENV \
   "D3D10UMD_YTTRIUM_SAMPLED_CPU_VERTEX_BATCH"
#define YTTRIUM_VENUS_SAMPLED_CPU_VERTEX_RENDER_PASS_BATCH_ENV \
   "D3D10UMD_YTTRIUM_SAMPLED_CPU_VERTEX_RENDER_PASS_BATCH"
#define YTTRIUM_VENUS_CHECKED_DESCRIPTOR_ALLOC_ENV \
   "D3D10UMD_YTTRIUM_CHECKED_DESCRIPTOR_ALLOC"
#define YTTRIUM_VENUS_PUSH_DESCRIPTOR_BATCH_ENV \
   "D3D10UMD_YTTRIUM_PUSH_DESCRIPTOR_BATCH"
#define YTTRIUM_VENUS_PUSH_DESCRIPTOR_LAYOUT_ROTATION_ENV \
   "D3D10UMD_YTTRIUM_PUSH_DESCRIPTOR_LAYOUT_ROTATION"
#define YTTRIUM_VENUS_RENDER_PASS_BATCH_ENV \
   "D3D10UMD_YTTRIUM_RENDER_PASS_BATCH"
#define YTTRIUM_VENUS_MIXED_DRAW_TRANSFER_BATCH_ENV \
   "D3D10UMD_YTTRIUM_MIXED_DRAW_TRANSFER_BATCH"
#define YTTRIUM_VENUS_DIRECT_CPU_VERTEX_UPLOAD_ENV \
   "D3D10UMD_YTTRIUM_DIRECT_CPU_VERTEX_UPLOAD"
#define YTTRIUM_VENUS_DEVICE_LOCAL_STATIC_DRAW_BUFFERS_ENV \
   "D3D10UMD_YTTRIUM_DEVICE_LOCAL_STATIC_DRAW_BUFFERS"
#define YTTRIUM_VENUS_TEST_FAIL_AFTER_DRAW_MIRROR_COPY_ONCE_ENV \
   "D3D10UMD_YTTRIUM_TEST_FAIL_AFTER_DRAW_MIRROR_COPY_ONCE"
#define YTTRIUM_VENUS_DRAW_ARENA_BAR_ENV \
   "D3D10UMD_YTTRIUM_DRAW_ARENA_BAR"
#define YTTRIUM_VENUS_DIRECT_UBO_UPLOAD_ENV \
   "D3D10UMD_YTTRIUM_DIRECT_UBO_UPLOAD"
#define YTTRIUM_VENUS_DRAW_BACKING_POOL_ENV \
   "D3D10UMD_YTTRIUM_DRAW_BACKING_POOL"
#define YTTRIUM_VENUS_COMPACT_IMAGE_BARRIERS_ENV \
   "D3D10UMD_YTTRIUM_COMPACT_IMAGE_BARRIERS"
#define YTTRIUM_VENUS_COMPACT_DRAW_PACKETS_ENV \
   "D3D10UMD_YTTRIUM_COMPACT_DRAW_PACKETS"

#if VK_USE_64_BIT_PTR_DEFINES
#define YTTRIUM_VENUS_HANDLE_TO_U64(handle) ((uint64_t)(uintptr_t)(handle))
#else
#define YTTRIUM_VENUS_HANDLE_TO_U64(handle) ((uint64_t)(handle))
#endif
#define YTTRIUM_VENUS_HANDLE(type, obj) ((type)(uintptr_t)(obj))

#if defined(_MSC_VER) || (defined(__clang__) && defined(_WIN64))
#define YTTRIUM_VENUS_HAS_NATIVE_SEH 1
#endif

#if defined(YTTRIUM_VENUS_HAS_NATIVE_SEH)
#define YTTRIUM_VENUS_SEH_TRY __try
#define YTTRIUM_VENUS_SEH_EXCEPT(filter) __except (filter)
#else
#if defined(__GNUC__) || defined(__clang__)
#warning "Yttrium Venus ring fault handling: native SEH is unavailable; guarded accesses will execute but faults cannot be recovered"
#else
#pragma message("warning: Yttrium Venus ring fault handling: native SEH is unavailable; guarded accesses will execute but faults cannot be recovered")
#endif
#define YTTRIUM_VENUS_SEH_TRY if (1)
#define YTTRIUM_VENUS_SEH_EXCEPT(filter) else
#endif

enum yttrium_venus_graphics_mode {
   YTTRIUM_VENUS_GRAPHICS_VERTEX_BUFFER = 0,
   YTTRIUM_VENUS_GRAPHICS_TEXTURED_VERTEX_BUFFER = 1,
};

enum yttrium_venus_sync_wait_kind {
   YTTRIUM_VENUS_SYNC_WAIT_RAW_SUBMIT = 0,
   YTTRIUM_VENUS_SYNC_WAIT_RING_SPACE = 1,
   YTTRIUM_VENUS_SYNC_WAIT_RING_SEQNO = 2,
   YTTRIUM_VENUS_SYNC_WAIT_BATCH_COMPLETION = 3,
};

struct yttrium_venus_sync_wait_diag {
   struct {
      uint32_t command_type;
      uint32_t count;
      uint32_t command_size;
      uint32_t reply_size;
      uint64_t wait_us;
      uint64_t max_wait_us;
   } commands[YTTRIUM_VENUS_SYNC_WAIT_DIAG_CMD_SLOTS];
   uint64_t window_start_us;
   uint64_t total_wait_us;
   uint64_t max_wait_us;
   uint32_t total_count;
   uint32_t raw_submit_count;
   uint32_t ring_space_count;
   uint32_t ring_seqno_count;
   uint32_t timeout_count;
   uint32_t last_kind;
   uint32_t last_status;
   const char *last_label;
   uint32_t last_a;
   uint32_t last_b;
   uint32_t last_command_type;
   uint32_t last_command_size;
   uint32_t last_reply_size;
};

struct yttrium_venus_bo {
   D3DKMT_HANDLE hAllocation;
   HANDLE hResource;
   uint32_t res_id;
   uint64_t size;
   void *map;
   uint32_t map_info;
};

struct yttrium_venus_ring {
   struct yttrium_venus_bo bo;
   uint64_t id;

   volatile uint32_t *head;
   volatile uint32_t *tail;
   volatile uint32_t *status;
   uint8_t *buffer;
   uint32_t buffer_size;
   uint32_t buffer_mask;
   uint32_t cur;
   uint32_t published_cur;

   size_t head_offset;
   size_t tail_offset;
   size_t status_offset;
   size_t buffer_offset;
   size_t extra_offset;
   size_t extra_size;

   uint64_t protocol_command_count;
   uint64_t protocol_command_bytes;
   uint64_t protocol_queue_submit_count;
   uint64_t wait_baseline_command_count;
   uint64_t wait_baseline_command_bytes;
   uint64_t wait_baseline_queue_submit_count;
   uint64_t current_wait_command_count;
   uint64_t current_wait_command_bytes;
   uint64_t current_wait_queue_submit_count;
};

#define YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT 8192
#define YTTRIUM_VENUS_CMD_BATCH_PENDING_PIPELINE_REF_LIMIT 4096
/*
 * D3D10UMD_YTTRIUM_BATCH_COUNT is the initial pool size, not a synchronous
 * pipeline-depth limit.  Completed prefixes are polled through mapped feedback
 * and retired opportunistically.  If every allocated slot is still live, the
 * pool grows lazily up to this hard bounded-memory ceiling.  A slot owns large
 * resource/pipeline reference arrays (~165 KB), so allocate slots individually
 * and keep their addresses stable rather than reserving the whole ceiling for
 * every context.
 */
#define YTTRIUM_VENUS_BATCH_COUNT_MAX 256
#define YTTRIUM_VENUS_BATCH_COUNT_DEFAULT 16
#define YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_MAX 64
/* Grouping and initial allocation are independent: grow toward the submit
 * target only when a workload actually queues that many command buffers. */
#define YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_DEFAULT 64
#define YTTRIUM_VENUS_CMD_BATCH_TRANSIENT_LIMIT 128
#define YTTRIUM_VENUS_RENDER_TARGET_CACHE_MAX 1024
#define YTTRIUM_VENUS_DRAW_BACKING_POOL_MAX_ENTRIES 128
#define YTTRIUM_VENUS_DRAW_BACKING_POOL_MAX_BYTES \
   (128ull * 1024ull * 1024ull)
#define YTTRIUM_VENUS_DRAW_BACKING_POOL_MAX_ENTRY_BYTES \
   (16ull * 1024ull * 1024ull)

struct yttrium_venus_render_target_key {
   uint64_t color_image_ids[PIPE_MAX_COLOR_BUFS];
   uint64_t depth_image_id;
   VkFormat color_formats[PIPE_MAX_COLOR_BUFS];
   VkFormat depth_format;
   VkImageViewType color_view_types[PIPE_MAX_COLOR_BUFS];
   VkImageViewType depth_view_type;
   uint32_t color_levels[PIPE_MAX_COLOR_BUFS];
   uint32_t color_layers[PIPE_MAX_COLOR_BUFS];
   uint32_t color_attachment_count;
   uint32_t width;
   uint32_t height;
   uint32_t layers;
   uint32_t depth_level;
   uint32_t depth_layer;
   uint32_t depth_layers;
   VkSampleCountFlagBits attachment_samples;
   VkSampleCountFlagBits render_samples;
   VkBool32 use_mrss;
   uint32_t color_feedback_loop_mask;
   VkBool32 depth_feedback_loop;
   VkBool32 depth_read_only;
};

struct yttrium_venus_render_target {
   struct yttrium_venus_render_target_key key;
   struct yttrium_venus_object image_view_objs[PIPE_MAX_COLOR_BUFS];
   struct yttrium_venus_object depth_image_view_obj;
   struct yttrium_venus_object render_pass_obj;
   struct yttrium_venus_object framebuffer_obj;
   VkImageView image_views[PIPE_MAX_COLOR_BUFS];
   VkImageView depth_image_view;
   VkRenderPass render_pass;
   VkFramebuffer framebuffer;
   uint32_t refcount;
   bool cached;
};

struct yttrium_venus_cmd_batch_descriptor_pool {
   struct yttrium_venus_object *pool_obj;
   struct yttrium_venus_object *set_objs;
   VkDescriptorPool pool;
   VkDescriptorSet *sets;
   uint32_t set_capacity;
   uint32_t set_count;
   uint32_t ubo_descriptor_count;
   uint32_t sampled_image_descriptor_count;
   uint32_t sampled_buffer_descriptor_count;
   uint32_t storage_image_descriptor_count;
   uint32_t storage_buffer_descriptor_count;
   bool pool_created;
};

struct yttrium_venus_cmd_batch_transient {
   struct yttrium_venus_object view_obj;
   struct yttrium_venus_object render_pass_obj;
   struct yttrium_venus_object framebuffer_obj;
   VkImageView view;
   VkRenderPass render_pass;
   VkFramebuffer framebuffer;
   bool view_created;
   bool render_pass_created;
   bool framebuffer_created;
};

struct yttrium_venus_ubo_arena {
   struct yttrium_venus_ubo_arena *next;
   struct yttrium_venus_object buffer_obj;
   struct yttrium_venus_object memory_obj;
   VkBuffer buffer;
   VkDeviceMemory memory;
   struct yttrium_venus_memory_mapping mapping;
   VkDeviceSize size;
   VkDeviceSize roll_base;
   uint64_t generation;
   uint32_t batch_refcount;
   bool retired;
};

enum yttrium_venus_draw_backing_kind {
   YTTRIUM_VENUS_DRAW_BACKING_NONE = 0,
   YTTRIUM_VENUS_DRAW_BACKING_VERTEX,
   YTTRIUM_VENUS_DRAW_BACKING_INDEX,
};

struct yttrium_venus_retired_resource {
   struct yttrium_venus_retired_resource *next;
   enum yttrium_venus_draw_backing_kind draw_backing_kind;
   VkDeviceSize draw_backing_size;
   D3DKMT_HANDLE hAllocation;
   HANDLE hResource;
   uint64_t allocation_size;
   void *map;
   uint32_t map_info;
   bool map_is_blob;
   bool owns_allocation;
   bool allocation_destroyed_by_runtime;
   struct yttrium_venus_object image_obj;
   struct yttrium_venus_object buffer_obj;
   struct yttrium_venus_object memory_obj;
   struct yttrium_venus_object image_view_obj;
   struct yttrium_venus_object render_pass_obj;
   struct yttrium_venus_object framebuffer_obj;
   struct yttrium_venus_object pipeline_layout_obj;
   struct yttrium_venus_object descriptor_set_layout_obj;
   struct yttrium_venus_object descriptor_pool_obj;
   struct yttrium_venus_object sampler_obj;
   struct yttrium_venus_object vertex_shader_obj;
   struct yttrium_venus_object fragment_shader_obj;
   struct yttrium_venus_object pipeline_obj;
   struct yttrium_venus_object push_descriptor_set_layout_obj;
   struct yttrium_venus_object push_descriptor_set_layout_alt_obj;
   struct yttrium_venus_object push_pipeline_layout_obj;
   struct yttrium_venus_object push_pipeline_layout_alt_obj;
   struct yttrium_venus_object push_pipeline_obj;
   struct yttrium_venus_object
      pipeline_image_view_objs[PIPE_MAX_COLOR_BUFS];
   struct yttrium_venus_object pipeline_depth_image_view_obj;
   struct yttrium_venus_object
      pipeline_sampler_objs[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
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
   VkSampler sampler;
   VkShaderModule vertex_shader;
   VkShaderModule fragment_shader;
   VkPipeline pipeline;
   VkDescriptorSetLayout push_descriptor_set_layout;
   VkDescriptorSetLayout push_descriptor_set_layout_alt;
   VkPipelineLayout push_pipeline_layout;
   VkPipelineLayout push_pipeline_layout_alt;
   VkPipeline push_pipeline;
   VkImageView pipeline_image_views[PIPE_MAX_COLOR_BUFS];
   VkImageView pipeline_depth_image_view;
   VkSampler pipeline_samplers[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint32_t custom_border_color_sampler_count;
   VkBuffer draw_vertex_buffer;
   VkDeviceMemory draw_vertex_memory;
   VkBuffer draw_index_buffer;
   VkDeviceMemory draw_index_memory;
   VkBuffer device_local_draw_buffer;
   VkDeviceMemory device_local_draw_memory;
   struct yttrium_venus_memory_mapping draw_vertex_mapping;
   struct yttrium_venus_memory_mapping draw_index_mapping;
   struct yttrium_venus_render_target *render_target_cache_entry;
};

/*
 * A device-local vertex/index mirror update is visible to later command
 * recording before its containing VkCommandBuffer is submitted.  Keep the
 * previous provisional state so an aborted command batch can restore it, and
 * carry the final source serial with the batch until vkQueueSubmit is encoded.
 */
struct yttrium_venus_draw_mirror_update {
   struct yttrium_venus_resource *resource;
   uint32_t source_serial;
   uint32_t previous_pending_serial;
   bool previous_pending_valid;
};

struct yttrium_venus_batch {
   struct yttrium_venus_object command_buffer_obj;
   struct yttrium_venus_object feedback_command_buffer_obj;
   struct yttrium_venus_object fence_obj;
   VkCommandBuffer command_buffer;
   VkCommandBuffer feedback_command_buffer;
   VkFence fence;
   bool initialized;
   bool busy;
   bool pending_submit;
   struct yttrium_venus_ring_dependency host_retirement;
   uint64_t submit_order;
   uint64_t completion_order;
   VkFence completion_fence;
   uint32_t feedback_index;
   uint32_t completion_feedback_index;
   struct yttrium_venus_ubo_arena *ubo_arena;
   struct yttrium_venus_retired_resource *retired_resources;
   struct yttrium_venus_resource
      *resources[YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT];
   struct pipe_resource
      *resource_refs[YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT];
   uint32_t resource_count;
   struct yttrium_pipeline
      *pipelines[YTTRIUM_VENUS_CMD_BATCH_PENDING_PIPELINE_REF_LIMIT];
   uint32_t pipeline_count;
   struct yttrium_venus_cmd_batch_transient
      transients[YTTRIUM_VENUS_CMD_BATCH_TRANSIENT_LIMIT];
   uint32_t transient_count;
   struct yttrium_venus_cmd_batch_descriptor_pool descriptor_pool;
   struct yttrium_venus_draw_mirror_update *draw_mirror_updates;
   uint32_t draw_mirror_update_count;
   uint32_t draw_mirror_update_capacity;
};

struct yttrium_venus_cmd_batch_buffer_footprint {
   VkBuffer buffer;
   VkDeviceSize offset;
   VkDeviceSize size;
   uint64_t generation;
};

struct yttrium_venus_cmd_batch_buffer_index {
   VkBuffer buffer;
   VkDeviceSize end;
   uint64_t generation;
};

struct yttrium_venus_cmd_batch_footprint {
   VkDescriptorSet descriptor_set;
   struct yttrium_venus_cmd_batch_buffer_footprint ubo;
   struct yttrium_venus_cmd_batch_buffer_footprint vertex;
   struct yttrium_venus_cmd_batch_buffer_footprint index;
   uint64_t sampled_image_ids[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint64_t attachment_image_ids[PIPE_MAX_COLOR_BUFS + 1];
   uint64_t read_only_depth_image_id;
   uint64_t pipeline_id;
   uint32_t resource_id;
   uint32_t sampled_image_count;
   uint32_t attachment_image_count;
};

struct yttrium_venus_cmd_batch_upload {
   VkBuffer buffer;
   VkDeviceSize offset;
   VkDeviceSize size;
   VkDeviceSize capacity;
   uint8_t *data;
};

#define YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT \
   (YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS + \
    YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES)

/*
 * Two consecutive deferred draws with equal keys reuse the same render pass and
 * framebuffer, so every subresource the framebuffer views select has to be in
 * here.  The scalar render level/layer only carries the first colour surface,
 * while the views are created per target from draw_state->rt_level[] and
 * draw_state->rt_layer[] and the depth view from
 * depth_level/depth_layer/depth_layers.  Draws differing only in a second
 * target's slice, or only in the depth level, layer, or layer count, would
 * compare equal and silently render through the previous framebuffer - into
 * the wrong slice.
 *
 * Latent rather than live today only because deferral is rarely taken (a render
 * pass per draw is what we measure); it becomes reachable the moment render-pass
 * batching starts working, which is why it is fixed ahead of that work.
 */
struct yttrium_venus_render_pass_group_key {
   uint64_t color_image_ids[PIPE_MAX_COLOR_BUFS];
   uint64_t depth_image_id;
   VkFormat color_formats[PIPE_MAX_COLOR_BUFS];
   VkFormat depth_format;
   VkSampleCountFlagBits color_samples[PIPE_MAX_COLOR_BUFS];
   VkSampleCountFlagBits depth_samples;
   VkSampleCountFlagBits render_samples;
   VkBool32 use_mrss;
   uint32_t color_levels[PIPE_MAX_COLOR_BUFS];
   uint32_t color_layers[PIPE_MAX_COLOR_BUFS];
   uint32_t color_attachment_count;
   uint32_t render_width;
   uint32_t render_height;
   uint32_t render_layers;
   uint32_t depth_level;
   uint32_t depth_layer;
   uint32_t depth_layers;
   uint32_t color_feedback_loop_mask;
   VkBool32 depth_feedback_loop;
   VkBool32 depth_read_only;
};

struct yttrium_venus_deferred_draw {
   struct yttrium_venus_object render_pass_obj;
   struct yttrium_venus_object framebuffer_obj;
   struct yttrium_venus_object pipeline_obj;
   struct yttrium_venus_object pipeline_layout_obj;
   struct yttrium_venus_object descriptor_set_obj;
   struct yttrium_venus_object
      sampled_sampler_objs[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkRenderPass render_pass;
   VkFramebuffer framebuffer;
   VkPipeline pipeline;
   VkPipelineLayout pipeline_layout;
   VkDescriptorSet descriptor_set;
   VkBuffer vertex_buffers[YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   VkDeviceSize vertex_offsets[YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   VkBuffer index_buffer;
   VkDeviceSize index_offset;
   VkIndexType index_type;
   VkViewport viewports[PIPE_MAX_VIEWPORTS];
   VkRect2D scissors[PIPE_MAX_VIEWPORTS];
   float blend_constants[4];
   VkDescriptorBufferInfo ubo_infos[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   VkDescriptorImageInfo
      sampled_image_infos[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkBufferView
      sampled_buffer_views[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkWriteDescriptorSet
      push_writes[YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT];
   struct yttrium_venus_render_pass_group_key render_key;
   uint8_t push_constant_data[YTTRIUM_SHADER_PUSH_CONSTANT_BYTES];
   uint32_t render_width;
   uint32_t render_height;
   uint32_t viewport_count;
   uint32_t vertex_buffer_count;
   uint32_t ubo_push_write_count;
   uint32_t sampled_push_write_count;
   uint32_t push_write_count;
   uint32_t declared_push_descriptor_count;
   uint32_t vertex_count;
   uint32_t index_count;
   uint32_t instance_count;
   int32_t vertex_offset;
   uint16_t push_constant_vs_size;
   uint16_t push_constant_fs_size;
   bool indexed;
   bool use_push_descriptors;
};

enum yttrium_venus_compact_descriptor_kind {
   YTTRIUM_VENUS_COMPACT_DESCRIPTOR_BUFFER = 1,
   YTTRIUM_VENUS_COMPACT_DESCRIPTOR_IMAGE = 2,
   YTTRIUM_VENUS_COMPACT_DESCRIPTOR_BUFFER_VIEW = 3,
};

struct yttrium_venus_compact_descriptor_write {
   uint32_t dst_binding;
   uint32_t dst_array_element;
   uint32_t descriptor_count;
   VkDescriptorType descriptor_type;
   enum yttrium_venus_compact_descriptor_kind kind;
   struct yttrium_venus_object sampler_obj;
   union {
      VkDescriptorBufferInfo buffer;
      VkDescriptorImageInfo image;
      VkBufferView buffer_view;
   } info;
};

/*
 * Variable-size replacement for yttrium_venus_deferred_draw.  Live arrays
 * immediately follow this header in viewport, scissor, vertex-buffer,
 * vertex-offset, and compact-descriptor order.  No pointer targets live in
 * the packet, so growing the byte stream does not require an O(draws) fixup.
 */
struct yttrium_venus_compact_draw_packet {
   uint32_t packet_size;
   uint32_t viewport_count;
   uint32_t vertex_buffer_count;
   uint32_t push_write_count;
   uint32_t declared_push_descriptor_count;
   struct yttrium_venus_object render_pass_obj;
   struct yttrium_venus_object framebuffer_obj;
   struct yttrium_venus_object pipeline_obj;
   struct yttrium_venus_object pipeline_layout_obj;
   struct yttrium_venus_object descriptor_set_obj;
   VkDescriptorSet descriptor_set;
   struct yttrium_venus_render_pass_group_key render_key;
   VkBuffer index_buffer;
   VkDeviceSize index_offset;
   VkIndexType index_type;
   uint32_t render_width;
   uint32_t render_height;
   uint32_t vertex_count;
   uint32_t index_count;
   uint32_t instance_count;
   int32_t vertex_offset;
   uint16_t push_constant_vs_size;
   uint16_t push_constant_fs_size;
   float blend_constants[4];
   bool indexed;
   bool use_push_descriptors;
};

struct yttrium_venus {
   struct gdikmt_device *device;
   struct gdikmt_context *kmt_ctx;

   uint64_t next_id;
   struct vn_ring vn_ring;
   struct yttrium_venus_ring ring;
   struct yttrium_venus_bo reply_bo;

   struct yttrium_venus_object instance_obj;
   struct yttrium_venus_object
      physical_device_objs[YTTRIUM_VENUS_MAX_PHYSICAL_DEVICES];
   struct yttrium_venus_object device_obj;
   struct yttrium_venus_object queue_obj;
   struct yttrium_venus_object command_pool_obj;
   struct yttrium_venus_object batch_feedback_buffer_obj;
   struct yttrium_venus_object batch_feedback_memory_obj;

   VkInstance instance;
   VkPhysicalDevice physical_device;
   VkDevice device_handle;
   VkQueue queue;
   VkCommandPool command_pool;
   VkCommandBuffer command_buffer;
   struct yttrium_venus_ubo_arena *ubo_upload_arena;
   struct yttrium_venus_ubo_arena *retired_ubo_arenas;
   struct yttrium_venus_ubo_arena *free_ubo_arenas;
   struct yttrium_venus_retired_resource *draw_backing_pool;
   VkDeviceSize draw_backing_pool_bytes;
   uint32_t draw_backing_pool_count;
   VkBuffer ubo_upload_buffer;
   VkDeviceMemory ubo_upload_memory;
   VkDeviceSize ubo_upload_buffer_size;
   VkBuffer batch_feedback_buffer;
   VkDeviceMemory batch_feedback_memory;
   struct yttrium_venus_memory_mapping batch_feedback_mapping;
   VkDeviceSize batch_feedback_buffer_size;
   bool batch_feedback_initialized;

   /*
    * How far the UBO arena has been written.  Uploads roll forward from here
    * rather than restarting at offset 0 every batch, so a batch writes bytes
    * no in-flight batch is reading and needs no drain.  Reset to 0 when the
    * roll wraps, and whenever the arena itself is reallocated.
    */
   VkDeviceSize ubo_upload_roll_base;
   uint64_t ubo_upload_buffer_generation;
   VkDeviceSize ubo_arena_bytes;
   VkDeviceSize uniform_buffer_offset_alignment;
   VkPhysicalDeviceMemoryProperties memory_props;
   VkSampleCountFlags framebuffer_sample_counts;
   VkSampleCountFlags framebuffer_no_attachments_sample_counts;
   float max_sampler_anisotropy;
   float max_sampler_lod_bias;
   uint32_t max_custom_border_color_samplers;
   /* Includes samplers retained by in-flight retired pipelines. */
   uint32_t custom_border_color_sampler_count;
   uint32_t mipmap_precision_bits;
   uint32_t instance_version;
   uint32_t queue_family_index;
   struct yttrium_venus_sync_wait_diag sync_wait_diag;
   struct yttrium_venus_batch **batches;
   struct yttrium_venus_batch **pending_submit_batches;
   uint32_t batch_count;
   uint32_t batch_capacity;
   uint32_t live_batch_count;
   uint32_t peak_live_batch_count;
   uint32_t pending_submit_count;
   uint32_t group_queue_submit_size;
   uint64_t pending_submit_op_count;
   uint64_t group_queue_submit_op_threshold;
   VkDeviceSize peak_ubo_arena_bytes;
   VkDeviceSize peak_draw_backing_pool_bytes;
   bool group_queue_submits;
   struct yttrium_venus_resource null_sampled_image;
   struct yttrium_venus_resource null_sampled_buffer;
   VkPhysicalDeviceTransformFeedbackPropertiesEXT transform_feedback_props;
   bool depth_bias_control;
   bool depth_clamp;
   bool dual_src_blend;
   bool independent_blend;
   bool logic_op;
   bool sample_rate_shading;
   bool tessellation_shader;
   bool attachment_feedback_loop_layout;
   bool load_store_op_none;
   bool fragment_shader_pixel_interlock;
   bool fragment_stores_and_atomics;
   bool draw_indirect_first_instance;
   bool vertex_divisor_nonzero_first_instance;
   bool storage_image_read_without_format;
   bool storage_image_write_without_format;
   /* Immutable per-device format support: 0 unknown, 1 unsupported, 2 supported. */
   uint8_t storage_image_formatless_formats[PIPE_FORMAT_COUNT];
   bool multisampled_render_to_single_sampled;
   bool transform_feedback;
   bool vertex_attribute_instance_rate_divisor;
   bool vertex_attribute_instance_rate_zero_divisor;
   bool push_descriptor;
   bool custom_border_colors;
   bool custom_border_color_without_format;
   bool multi_viewport;
   bool shader_output_viewport_index;
   uint32_t max_viewports;
   uint32_t max_dual_source_render_targets;
   uint32_t max_tessellation_patch_size;
   uint32_t max_vertex_attrib_divisor;
   uint32_t max_push_descriptors;
   bool display_copy_batch_recording;
   bool compact_image_barriers;
   bool device_local_static_draw_buffers;
   bool draw_arena_bar;
   bool cmd_batch_native_draw_only;
   bool cmd_batch_async_submit;
   bool cmd_batch_has_transfer_ops;
   bool ring_notify_pending;
   bool ring_transaction_active;
   uint32_t ring_transaction_thread_id;
   uint32_t display_copy_batch_count;
   uint64_t next_batch_submit_order;
   uint64_t last_completed_submit_order;
   uint32_t push_descriptor_layout_index;
   uint32_t compute_push_descriptor_layout_index;
   /*
    * The frame the ordered worker owes the display, held one Present deep so
    * that the wait for its rendering runs while the next frame's work is
    * already on the ring.  Worker-owned, so no lock.  The batch is identified
    * by both pointer and submit order: batches are recycled, and a mismatch
    * means this frame's batch already retired, which is completion.
    */
   bool present_publish_pending;
   uint32_t present_publish_allocation;
   uint32_t present_publish_scanout_id;
   struct yttrium_venus_batch *present_publish_batch;
   uint64_t present_publish_submit_order;
   struct yttrium_venus_batch *cmd_batch;
   struct yttrium_venus_ubo_arena *cmd_batch_ubo_arena;
   struct yttrium_venus_resource
      *cmd_batch_pending_resources[YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT];
   struct pipe_resource
      *cmd_batch_pending_resource_refs[YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT];
   struct yttrium_pipeline
      *cmd_batch_pending_pipelines[YTTRIUM_VENUS_CMD_BATCH_PENDING_PIPELINE_REF_LIMIT];
   uint32_t cmd_batch_pending_resource_count;
   uint32_t cmd_batch_pending_pipeline_count;
   struct yttrium_venus_draw_mirror_update *cmd_batch_draw_mirror_updates;
   uint32_t cmd_batch_draw_mirror_update_count;
   uint32_t cmd_batch_draw_mirror_update_capacity;
   /* Transient objects referenced by the deferred command buffer. */
   struct yttrium_venus_cmd_batch_transient
      cmd_batch_transients[YTTRIUM_VENUS_CMD_BATCH_TRANSIENT_LIMIT];
   uint32_t cmd_batch_transient_count;
   struct yttrium_venus_cmd_batch_descriptor_pool cmd_batch_descriptor_pool;
   VkDeviceSize cmd_batch_ubo_watermark;
   VkDeviceSize cmd_batch_vertex_watermark;
   VkDeviceSize cmd_batch_index_watermark;
   VkImageMemoryBarrier *cmd_batch_image_barriers;
   uint32_t cmd_batch_image_barrier_count;
   uint32_t cmd_batch_image_barrier_capacity;
   VkPipelineStageFlags cmd_batch_image_src_stages;
   VkPipelineStageFlags cmd_batch_image_dst_stages;
   VkDependencyFlags cmd_batch_image_dependency_flags;
   VkBufferMemoryBarrier *cmd_batch_upload_barriers;
   uint32_t cmd_batch_upload_barrier_count;
   uint32_t cmd_batch_upload_barrier_capacity;
   /* Candidate index + 1, revalidated against the live count and merge key.
    * Indices survive realloc; these entries own no barriers or resources. */
   uint32_t cmd_batch_upload_barrier_candidates[64];
   VkPipelineStageFlags cmd_batch_upload_src_stages;
   VkPipelineStageFlags cmd_batch_upload_dst_stages;
   struct yttrium_venus_cmd_batch_upload *cmd_batch_uploads;
   uint32_t cmd_batch_upload_count;
   uint32_t cmd_batch_upload_capacity;
   struct yttrium_venus_cmd_batch_footprint *cmd_batch_footprints;
   uint32_t cmd_batch_footprint_count;
   uint32_t cmd_batch_footprint_capacity;
   struct hash_table_u64 *cmd_batch_sampled_image_roles;
   struct hash_table_u64 *cmd_batch_attachment_image_roles;
   struct hash_table_u64 *cmd_batch_descriptor_sets;
   struct yttrium_venus_cmd_batch_buffer_index cmd_batch_ubo_footprint;
   struct yttrium_venus_cmd_batch_buffer_index cmd_batch_vertex_footprint;
   struct yttrium_venus_cmd_batch_buffer_index cmd_batch_index_footprint;
   struct yttrium_venus_deferred_draw *cmd_batch_deferred_draws;
   uint32_t cmd_batch_deferred_draw_count;
   uint32_t cmd_batch_deferred_draw_capacity;
   uint8_t *cmd_batch_compact_draw_packets;
   size_t cmd_batch_compact_draw_packet_size;
   size_t cmd_batch_compact_draw_packet_capacity;
   uint32_t next_batch;
   uint32_t ring_notify_seqno;
   int64_t ring_last_notify;
   int64_t ring_next_notify;
   struct yttrium_venus_render_target **render_target_cache;
   uint32_t render_target_cache_capacity;

   bool transport_initialized;
   bool instance_initialized;
   bool physical_device_initialized;
   bool initialized;
   bool failed;
   bool test_fail_after_draw_mirror_copy_once;
   bool test_fail_after_draw_mirror_copy_consumed;
};

#define YTTRIUM_VENUS_DISPLAY_COPY_BATCH_LIMIT 64
#define YTTRIUM_VENUS_NATIVE_DRAW_BATCH_LIMIT_DEFAULT 512
#define YTTRIUM_VENUS_NATIVE_DRAW_BATCH_LIMIT_MAX 4096

struct yttrium_venus_native_draw_batch_state {
   uint32_t limit;
   uint32_t reject_mask;
   bool candidate;
   bool use_push_descriptors;
   bool native_draw_batch_enabled;
   bool cpu_vertex_batch_enabled;
   bool cpu_vertex_batch_allowed;
   bool push_descriptor_batch_enabled;
   bool push_descriptors_available;
   bool pipeline_has_push_layout;
   bool pipeline_has_push_pipeline;
};

/* Ring size in bytes, power of two, from D3D10UMD_YTTRIUM_RING_BUFFER_SIZE. */
uint32_t
yttrium_venus_ring_buffer_size(void);

/* Live submission depth: D3D10UMD_YTTRIUM_BATCH_COUNT, clamped to
 * [1, YTTRIUM_VENUS_BATCH_COUNT_MAX].
 */
uint32_t
yttrium_venus_batch_count(void);

bool
yttrium_venus_async_batch_enabled(void);

bool
yttrium_venus_batch_epoch_wait_enabled(void);

bool
yttrium_venus_batch_fence_feedback_enabled(void);

bool
yttrium_venus_native_draw_batch_enabled(void);

uint32_t
yttrium_venus_native_draw_batch_limit(void);

bool
yttrium_venus_checked_descriptor_alloc_enabled(void);

bool
yttrium_venus_push_descriptor_batch_enabled(void);

bool
yttrium_venus_push_descriptor_layout_rotation_enabled(void);

bool
yttrium_venus_render_pass_batch_enabled(void);

bool
yttrium_venus_mixed_draw_transfer_batch_enabled(void);

bool
yttrium_venus_direct_cpu_vertex_upload_enabled(void);

bool
yttrium_venus_sampled_cpu_vertex_render_pass_batch_enabled(void);

bool
yttrium_venus_direct_ubo_upload_enabled(void);

bool
yttrium_venus_draw_backing_pool_enabled(void);

bool
yttrium_venus_compact_draw_packets_enabled(void);

struct yttrium_venus_native_draw_batch_state
yttrium_venus_get_native_draw_batch_state(
   const struct yttrium_pipeline *pipeline,
   bool has_cpu_vertex_upload,
   bool has_sampled_descriptor,
   bool has_ubo_descriptor);

bool
yttrium_venus2_wait_resource(struct yttrium_venus *venus,
                             struct yttrium_venus_resource *resource,
                             const char *label);

VkFormat
yttrium_venus2_pipe_format(enum pipe_format format);

VkFormat
yttrium_venus_pipe_format_for_bind(enum pipe_format format, unsigned bind);

VkImageAspectFlags
yttrium_venus_format_aspects(VkFormat format);

bool
yttrium_venus_format_has_depth(VkFormat format);

bool
yttrium_venus_format_has_stencil(VkFormat format);

VkImageAspectFlags
yttrium_venus_initialized_aspects(
   const struct yttrium_venus_resource *resource);

void
yttrium_venus_mark_aspects_initialized(struct yttrium_venus_resource *resource,
                                       VkImageAspectFlags aspects);

bool
yttrium_venus_clear_pattern(enum pipe_format format,
                            const union pipe_color_union *color,
                            uint32_t *pattern);

bool
yttrium_venus2_create_display_buffer(struct yttrium_venus *venus,
                                     struct yttrium_venus_resource *resource,
                                     uint64_t allocation_size,
                                     uint64_t *out_memory_id);

bool
yttrium_venus2_create_bind_buffer(struct yttrium_venus *venus,
                                  struct yttrium_venus_resource *resource,
                                  uint64_t allocation_size,
                                  VkBufferUsageFlags usage,
                                  uint64_t *out_memory_id);

bool
yttrium_venus2_create_stream_output_buffer(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint64_t allocation_size,
   uint64_t *out_memory_id);

bool
yttrium_venus2_transform_feedback_enabled(
   const struct yttrium_venus *venus);

bool
yttrium_venus2_supports_load_store_op_none(struct yttrium_venus *venus);

bool
yttrium_venus2_supports_multisampled_render_to_single_sampled(
   struct yttrium_venus *venus,
   uint32_t sample_count);

bool
yttrium_venus2_supports_forced_sample_interlock(
   struct yttrium_venus *venus,
   uint32_t sample_count);

VkSampleCountFlags
yttrium_venus2_framebuffer_color_sample_counts(struct yttrium_venus *venus);

bool
yttrium_venus2_sampled_texture_format_supported(
   struct yttrium_venus *venus,
   enum pipe_format pipe_format,
   enum pipe_texture_target target);

bool
yttrium_venus2_storage_image_without_format_supported(
   struct yttrium_venus *venus, bool read, bool write);

bool
yttrium_venus2_storage_image_formatless_format_supported(
   struct yttrium_venus *venus, enum pipe_format format,
   bool read, bool write);

bool
yttrium_venus2_transform_feedback_draw_enabled(
   const struct yttrium_venus *venus);

bool
yttrium_venus2_vertex_attribute_divisor_supported(
   struct yttrium_venus *venus,
   uint32_t divisor);

bool
yttrium_venus2_depth_clamp_enabled(const struct yttrium_venus *venus);

bool
yttrium_venus2_logic_op_enabled(const struct yttrium_venus *venus);

uint32_t
yttrium_venus2_max_dual_source_render_targets(
   struct yttrium_venus *venus);

float
yttrium_venus2_max_sampler_anisotropy(struct yttrium_venus *venus);

float
yttrium_venus2_max_sampler_lod_bias(struct yttrium_venus *venus);

uint32_t
yttrium_venus2_max_transform_feedback_stride(
   const struct yttrium_venus *venus);

uint32_t
yttrium_venus2_max_viewports(struct yttrium_venus *venus);

uint32_t
yttrium_venus2_mipmap_precision_bits(const struct yttrium_venus *venus);

bool
yttrium_venus2_create_sampled_buffer(struct yttrium_venus *venus,
                                     struct yttrium_venus_resource *resource,
                                     uint64_t allocation_size,
                                     enum pipe_format pipe_format,
                                     uint64_t *out_memory_id);

bool
yttrium_venus2_ensure_null_sampled_buffer(
   struct yttrium_venus *venus,
   enum pipe_format pipe_format,
   uint64_t min_size,
   struct yttrium_venus_resource **out_resource,
   uint32_t *out_resource_id);

bool
yttrium_venus2_create_display_image(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint32_t width,
                                    uint32_t height,
                                    enum pipe_format pipe_format,
                                    uint64_t min_allocation_size,
                                    uint64_t *out_memory_id,
                                    uint64_t *out_allocation_size);

bool
yttrium_venus2_create_color_attachment_image(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t width,
   uint32_t height,
   uint32_t levels,
   uint32_t layers,
   enum pipe_format pipe_format,
   uint64_t *out_allocation_size);

bool
yttrium_venus2_create_sampled_texture_image(
   struct yttrium_venus *venus,
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
yttrium_venus2_create_texture_image_for_bind(
   struct yttrium_venus *venus,
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
yttrium_venus2_create_depth_stencil_image(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t width,
   uint32_t height,
   enum pipe_format pipe_format,
   uint64_t *out_allocation_size);

bool
yttrium_venus2_import_display_image(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint32_t resource_id,
                                    uint32_t width,
                                    uint32_t height,
                                    enum pipe_format pipe_format,
                                    uint64_t min_allocation_size,
                                    uint64_t *out_memory_id,
                                    uint64_t *out_allocation_size);

bool
yttrium_venus2_import_texture_image_for_bind(
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

bool
yttrium_venus2_resource_fini(
   struct yttrium_venus *venus,
   struct gdikmt_context *ctx,
   struct yttrium_venus_resource *resource,
   const struct yttrium_venus_allocation_snapshot *allocation);

uint32_t
yttrium_venus_subresource_width(const struct yttrium_venus_resource *resource,
                                uint32_t level);

uint32_t
yttrium_venus_subresource_height(const struct yttrium_venus_resource *resource,
                                 uint32_t level);

uint32_t
yttrium_venus_subresource_depth(const struct yttrium_venus_resource *resource,
                                uint32_t level);

bool
yttrium_venus_resource_is_3d(
   const struct yttrium_venus_resource *resource);

bool
yttrium_venus_valid_image_subresource(
   const struct yttrium_venus_resource *resource,
   uint32_t level, uint32_t first_layer, uint32_t layer_count);

void
yttrium_venus_ds_clear_history_reset(struct yttrium_venus_resource *resource);

void
yttrium_venus_ds_clear_history_invalidate(
   struct yttrium_venus_resource *resource);

void
yttrium_venus_ds_clear_history_drop_aspects(
   struct yttrium_venus_resource *resource,
   VkImageAspectFlags aspects);

void
yttrium_venus_ds_clear_history_note(
   struct yttrium_venus_resource *resource,
   VkImageAspectFlags aspects, double depth, unsigned stencil,
   uint32_t level, uint32_t layer, uint32_t x, uint32_t y,
   uint32_t width, uint32_t height);

bool
yttrium_venus_valid_render_subresource(
   const struct yttrium_venus_resource *resource,
   uint32_t level, uint32_t first_layer, uint32_t layer_count);

VkImageViewType
yttrium_venus_render_view_type(const struct yttrium_venus_resource *resource,
                               uint32_t layer_count);

VkImageSubresourceRange
yttrium_venus_render_barrier_range(
   const struct yttrium_venus_resource *resource,
   VkImageAspectFlags aspect_mask,
   uint32_t level, uint32_t first_layer, uint32_t layer_count);

bool
yttrium_venus_warn_encoder_overflow(const char *label,
                                    const struct vn_cs_encoder *enc,
                                    size_t reply_size);

bool
yttrium_venus_async_submit_succeeded(
   struct yttrium_venus *venus,
   const struct vn_ring_submit_command *submit,
   const char *operation,
   uint64_t object_id);

void
yttrium_venus_trace_timing(uint32_t point,
                           uint32_t status,
                           uint64_t start_us,
                           const char *label,
                           uint64_t a,
                           uint64_t b,
                           uint32_t c,
                           uint32_t d);

const char *
yttrium_venus_command_type_name(uint32_t command_type);

void
yttrium_venus_debug_sync_wait(struct yttrium_venus *venus,
                              enum yttrium_venus_sync_wait_kind kind,
                              uint64_t wait_us,
                              uint32_t status,
                              const char *label,
                              uint32_t a,
                              uint32_t b,
                              uint32_t command_type,
                              uint32_t command_size,
                              uint32_t reply_size);

uint64_t
yttrium_venus_next_id(struct yttrium_venus *venus);

void
yttrium_venus_init_object(struct yttrium_venus *venus,
                          struct yttrium_venus_object *obj);

void
yttrium_venus_unmap_memory(struct yttrium_venus *venus,
                           struct yttrium_venus_memory_mapping *mapping);

bool
yttrium_venus_map_memory(struct yttrium_venus *venus,
                         uint64_t venus_memory_id,
                         uint64_t size,
                         struct yttrium_venus_memory_mapping *mapping);

uint32_t
yttrium_venus_choose_memory_type(struct yttrium_venus *venus,
                                 uint32_t bits,
                                 VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred);

bool
yttrium_venus_ensure_initialized(struct yttrium_venus *venus);

bool
yttrium_venus_ensure_physical_device(struct yttrium_venus *venus);

bool
yttrium_venus_begin_command_buffer(
   struct yttrium_venus *venus,
   const char *label,
   const VkCommandBufferBeginInfo *begin_info);

bool
yttrium_venus_end_command_buffer(struct yttrium_venus *venus,
                                 const char *label);

bool
yttrium_venus_drain_batches(struct yttrium_venus *venus, const char *label);

bool
yttrium_venus_cmd_batch_track_resource(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource);

bool
yttrium_venus_cmd_batch_record_draw_mirror_update(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t source_serial);

bool
yttrium_venus_cmd_batch_track_pipeline(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline);

bool
yttrium_venus_cmd_batch_track_draw_refs(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   const struct yttrium_venus_vertex_upload *vertex_uploads,
   uint32_t vertex_upload_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   struct yttrium_pipeline *pipeline);

bool
yttrium_venus_cmd_batch_track_stream_output_refs(
   struct yttrium_venus *venus,
   const struct yttrium_venus_stream_output_target *targets,
   uint32_t target_count,
   const struct yttrium_venus_stream_output_target *draw_auto_target);

bool
yttrium_venus_ensure_null_sampled_image(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource **out_resource,
   uint32_t *out_resource_id);

bool
yttrium_venus_cmd_batch_track_sampled_refs(
   struct yttrium_venus *venus,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count);

bool
yttrium_venus_cmd_batch_track_storage_refs(
   struct yttrium_venus *venus,
   const struct yttrium_venus_storage_image *storage_images,
   uint32_t storage_image_count);

void
yttrium_venus_cmd_batch_clear_upload_barriers(struct yttrium_venus *venus);

void
yttrium_venus_cmd_batch_clear_uploads(struct yttrium_venus *venus);

void
yttrium_venus_cmd_batch_clear_image_barriers(struct yttrium_venus *venus);

void
yttrium_venus_cmd_batch_destroy_descriptor_pool(
   struct yttrium_venus *venus,
   struct yttrium_venus_cmd_batch_descriptor_pool *pool);

struct yttrium_venus_retired_resource *
yttrium_venus_retired_resource_create(
   struct yttrium_venus_resource *resource);

struct yttrium_venus_retired_resource *
yttrium_venus_retired_graphics_objects_take(
   struct yttrium_venus_resource *resource);

struct yttrium_venus_retired_resource *
yttrium_venus_retired_draw_vertex_backing_take(
   struct yttrium_venus_resource *resource);

struct yttrium_venus_retired_resource *
yttrium_venus_retired_draw_index_backing_take(
   struct yttrium_venus_resource *resource);

void
yttrium_venus_recycle_draw_backing(
   struct yttrium_venus *venus,
   struct yttrium_venus_retired_resource *retired);

bool
yttrium_venus_take_draw_backing(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   enum yttrium_venus_draw_backing_kind kind,
   VkDeviceSize minimum_size);

struct yttrium_venus_retired_resource *
yttrium_venus_retired_pipeline_create(struct yttrium_pipeline *pipeline);

void
yttrium_venus_render_target_release(
   struct yttrium_venus *venus,
   struct yttrium_venus_render_target *target);

void
yttrium_venus_retired_resource_adopt_allocation(
   struct yttrium_venus_retired_resource *retired,
   const struct yttrium_venus_allocation_snapshot *allocation);

void
yttrium_venus_destroy_retired_resource(
   struct yttrium_venus *venus,
   struct yttrium_venus_retired_resource *retired);

void
yttrium_venus_cmd_batch_clear_transient_upload_state(
   struct yttrium_venus *venus);

void
yttrium_venus_abort_command_batch(struct yttrium_venus *venus,
                                  const char *label);

void
yttrium_venus_cancel_command_batch_setup_failure(
   struct yttrium_venus *venus,
   const char *label);

bool
yttrium_venus_flush_command_batch(struct yttrium_venus *venus,
                                  const char *label);

bool
yttrium_venus_pipeline_ubo_footprint(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_ubo_upload *uploads,
   uint32_t upload_count,
   struct yttrium_venus_cmd_batch_buffer_footprint *footprint);

bool
yttrium_venus_cmd_batch_record_footprint(
   struct yttrium_venus *venus,
   const struct yttrium_venus_cmd_batch_footprint *footprint);

void
yttrium_venus_cmd_batch_destroy_footprint_index(
   struct yttrium_venus *venus);

void
yttrium_venus_cmd_batch_collect_image_roles(
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   bool depth_read_only,
   struct yttrium_venus_cmd_batch_footprint *footprint);

bool
yttrium_venus_cmd_batch_deferred_image_role_conflict(
   struct yttrium_venus *venus,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   bool depth_read_only);

bool
yttrium_venus_cmd_batch_emit_deferred_draws(struct yttrium_venus *venus,
                                            const char *label);

void
yttrium_venus_cmd_batch_clear_state(struct yttrium_venus *venus);

bool
yttrium_venus_cmd_batch_alloc_transient(struct yttrium_venus *venus,
                                        uint32_t *out_index);

void
yttrium_venus_cmd_batch_release_transient(struct yttrium_venus *venus,
                                          uint32_t index);

bool
yttrium_venus_cmd_batch_after_record(struct yttrium_venus *venus,
                                     const char *label);

bool
yttrium_venus_cmd_batch_after_native_draw_record(struct yttrium_venus *venus,
                                                const char *label);

bool
yttrium_venus_cmd_batch_submit_after_record(struct yttrium_venus *venus,
                                            const char *label);

bool
yttrium_venus2_flush(struct yttrium_venus *venus);

bool
yttrium_venus2_flush_async(struct yttrium_venus *venus);

bool
yttrium_venus2_flush_async_present_publish(
   struct yttrium_venus *venus,
   const struct yttrium_venus_present_publication *publication);

NTSTATUS
yttrium_venus_escape_publish_display(struct yttrium_venus *venus,
                                     uint32_t allocation,
                                     uint32_t scanout_id);

uint64_t
yttrium_venus2_last_submit_order(struct yttrium_venus *venus);

bool
yttrium_venus2_submit_order_complete(struct yttrium_venus *venus,
                                     uint64_t submit_order);

struct yttrium_venus_batch *
yttrium_venus_find_latest_resource_batch(
   struct yttrium_venus *venus,
   const struct yttrium_venus_resource *resource);

void
yttrium_venus_batch_retire_resource(
   struct yttrium_venus_batch *batch,
   struct yttrium_venus_retired_resource *retired);

struct yttrium_venus_batch *
yttrium_venus_find_latest_pipeline_batch(
   struct yttrium_venus *venus,
   const struct yttrium_pipeline *pipeline);

bool
yttrium_venus_wait_resource_batches(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    const char *label);

bool
yttrium_venus_wait_pipeline_batches(struct yttrium_venus *venus,
                                    struct yttrium_pipeline *pipeline,
                                    const char *label);

void
yttrium_venus_destroy_batches(struct yttrium_venus *venus);

void
yttrium_venus_destroy_ubo_arenas(struct yttrium_venus *venus);

void
yttrium_venus_retire_ubo_arena(
   struct yttrium_venus *venus,
   struct yttrium_venus_ubo_arena *arena);

bool
yttrium_venus_begin_command_batch(struct yttrium_venus *venus,
                                  const char *label,
                                  bool native_draw_batch,
                                  bool async_submit);

bool
yttrium_venus_begin_display_copy_batch(struct yttrium_venus *venus);

bool
yttrium_venus_begin_transfer_batch(struct yttrium_venus *venus);

bool
yttrium_venus_raw_submit(struct yttrium_venus *venus,
                         const void *data,
                         size_t size);

bool
yttrium_venus_raw_submit_ring_kick(struct yttrium_venus *venus,
                                   const void *data,
                                   size_t size,
                                   uint32_t seqno,
                                   bool blocking,
                                   const char *label);

bool
yttrium_venus_raw_submit_sync(struct yttrium_venus *venus,
                              const void *data,
                              size_t size,
                              const char *label);

bool
yttrium_venus_bo_create(struct yttrium_venus *venus,
                        struct yttrium_venus_bo *bo,
                        uint64_t size);

void
yttrium_venus_bo_destroy(struct yttrium_venus *venus,
                         struct yttrium_venus_bo *bo);

void
yttrium_venus_bo_forget_at_device_teardown(const char *label,
                                           struct yttrium_venus_bo *bo);

#endif /* YTTRIUM_VENUS2_PRIVATE_H */
