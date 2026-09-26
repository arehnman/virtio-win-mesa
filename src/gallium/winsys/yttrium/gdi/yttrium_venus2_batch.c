/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>

#include "util/u_math.h"
#include "util/u_inlines.h"
#include "util/hash_table.h"

#include "yttrium_venus2_ring.h"
#include "yttrium_venus2_private.h"
#include "yttrium_trace.h"
#include "virtio/wddm/viogpu_wddm_driver.h"

#include "venus-protocol/vn_protocol_driver_buffer.h"
#include "venus-protocol/vn_protocol_driver_buffer_view.h"
#include "venus-protocol/vn_protocol_driver_command_buffer.h"
#include "venus-protocol/vn_protocol_driver_command_pool.h"
#include "venus-protocol/vn_protocol_driver_descriptor_pool.h"
#include "venus-protocol/vn_protocol_driver_descriptor_set.h"
#include "venus-protocol/vn_protocol_driver_descriptor_set_layout.h"
#include "venus-protocol/vn_protocol_driver_device.h"
#include "venus-protocol/vn_protocol_driver_device_memory.h"
#include "venus-protocol/vn_protocol_driver_fence.h"
#include "venus-protocol/vn_protocol_driver_framebuffer.h"
#include "venus-protocol/vn_protocol_driver_image.h"
#include "venus-protocol/vn_protocol_driver_image_view.h"
#include "venus-protocol/vn_protocol_driver_pipeline.h"
#include "venus-protocol/vn_protocol_driver_pipeline_layout.h"
#include "venus-protocol/vn_protocol_driver_queue.h"
#include "venus-protocol/vn_protocol_driver_render_pass.h"
#include "venus-protocol/vn_protocol_driver_sampler.h"
#include "venus-protocol/vn_protocol_driver_semaphore.h"
#include "venus-protocol/vn_protocol_driver_shader_module.h"
#include "venus-protocol/vn_protocol_driver_transport.h"

static bool
yttrium_venus_queue_submit(struct yttrium_venus *venus,
                          const VkSubmitInfo *info, VkFence fence)
{
   if (!venus->kmt_ctx->scheduled_present_mode) {
      vn_async_vkQueueSubmit(&venus->vn_ring, venus->queue, 1, info, fence);
      return !venus->failed;
   }

   /* Recording stays on the shared ring. Windows owns execution of QueueSubmit.
    * The host context dispatch waits for recording before touching the handles. */
   yttrium_venus2_ring_lock();
   uint32_t seqno;
   bool ok = yttrium_venus_ring_publish_for_submit(venus, &seqno);
   size_t size = vn_sizeof_vkWaitRingSeqnoMESA(venus->ring.id, 0) +
                 vn_sizeof_vkQueueSubmit(venus->queue, 1, info, fence);
   void *data = ok ? MALLOC(size) : NULL;
   if (!data) {
      yttrium_venus2_ring_unlock();
      return false;
   }
   struct vn_cs_encoder enc = VN_CS_ENCODER_INITIALIZER_LOCAL(data, size);
   vn_encode_vkWaitRingSeqnoMESA(&enc, 0, venus->ring.id, seqno);
   vn_encode_vkQueueSubmit(&enc, 0, venus->queue, 1, info, fence);
   ok = !enc.fatal_error && yttrium_venus_submit_gpu(venus, data, vn_cs_encoder_get_len(&enc));
   FREE(data);
   yttrium_venus2_ring_unlock();
   return ok;
}

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

static void
yttrium_venus_batch_clear_refs(struct yttrium_venus_batch *batch);

static void
yttrium_venus_batch_release_ubo_arena(
   struct yttrium_venus *venus,
   struct yttrium_venus_batch *batch);

static bool
yttrium_venus_submit_batch_async(struct yttrium_venus *venus,
                                 struct yttrium_venus_batch *batch,
                                 uint32_t op_count,
                                 const char *label);

static bool
yttrium_venus_flush_pending_submits(struct yttrium_venus *venus,
                                    const char *label);

static struct yttrium_venus_batch *
yttrium_venus_latest_busy_batch(struct yttrium_venus *venus);

static bool
yttrium_venus_wait_batch(struct yttrium_venus *venus,
                         struct yttrium_venus_batch *batch,
                         const char *label);

static bool
yttrium_venus_wait_batch_for_present(struct yttrium_venus *venus,
                                     struct yttrium_venus_batch *batch,
                                     const char *label);

#define YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE 4u
#define YTTRIUM_VENUS_BATCH_FEEDBACK_PAUSE_ITERS 4096u
#define YTTRIUM_VENUS_BATCH_FEEDBACK_SLEEP0_ITERS 65536u

void
yttrium_venus_cmd_batch_destroy_descriptor_pool(
   struct yttrium_venus *venus,
   struct yttrium_venus_cmd_batch_descriptor_pool *pool)
{
   if (!venus || !pool)
      return;

   if (pool->pool_created)
      vn_async_vkDestroyDescriptorPool(&venus->vn_ring,
                                       venus->device_handle,
                                       pool->pool, NULL);
   FREE(pool->pool_obj);
   FREE(pool->set_objs);
   FREE(pool->sets);
   memset(pool, 0, sizeof(*pool));
}

static void
yttrium_venus_destroy_cmd_batch_transients(
   struct yttrium_venus *venus,
   struct yttrium_venus_cmd_batch_transient *transients,
   uint32_t *count)
{
   if (!venus || !transients || !count)
      return;

   for (uint32_t i = 0; i < *count; i++) {
      if (transients[i].framebuffer_created)
         vn_async_vkDestroyFramebuffer(&venus->vn_ring, venus->device_handle,
                                       transients[i].framebuffer, NULL);
      if (transients[i].render_pass_created)
         vn_async_vkDestroyRenderPass(&venus->vn_ring, venus->device_handle,
                                      transients[i].render_pass, NULL);
      if (transients[i].view_created)
         vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                     transients[i].view, NULL);
   }

   memset(transients, 0,
          sizeof(struct yttrium_venus_cmd_batch_transient) * *count);
   *count = 0;
}

static void
yttrium_venus_cmd_batch_destroy_transients(struct yttrium_venus *venus)
{
   yttrium_venus_destroy_cmd_batch_transients(
      venus, venus->cmd_batch_transients, &venus->cmd_batch_transient_count);
}

static void
yttrium_venus_retired_resource_rebuild_handles(
   struct yttrium_venus_retired_resource *retired)
{
   if (retired->image)
      retired->image = YTTRIUM_VENUS_HANDLE(VkImage, &retired->image_obj);
   if (retired->buffer)
      retired->buffer = YTTRIUM_VENUS_HANDLE(VkBuffer, &retired->buffer_obj);
   if (retired->memory)
      retired->memory =
         YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &retired->memory_obj);
   if (retired->image_view)
      retired->image_view =
         YTTRIUM_VENUS_HANDLE(VkImageView, &retired->image_view_obj);
   if (retired->render_pass)
      retired->render_pass =
         YTTRIUM_VENUS_HANDLE(VkRenderPass, &retired->render_pass_obj);
   if (retired->framebuffer)
      retired->framebuffer =
         YTTRIUM_VENUS_HANDLE(VkFramebuffer, &retired->framebuffer_obj);
   if (retired->pipeline_layout)
      retired->pipeline_layout =
         YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                              &retired->pipeline_layout_obj);
   if (retired->descriptor_set_layout)
      retired->descriptor_set_layout =
         YTTRIUM_VENUS_HANDLE(VkDescriptorSetLayout,
                              &retired->descriptor_set_layout_obj);
   if (retired->descriptor_pool)
      retired->descriptor_pool =
         YTTRIUM_VENUS_HANDLE(VkDescriptorPool,
                              &retired->descriptor_pool_obj);
   if (retired->sampler)
      retired->sampler = YTTRIUM_VENUS_HANDLE(VkSampler,
                                              &retired->sampler_obj);
   if (retired->vertex_shader)
      retired->vertex_shader =
         YTTRIUM_VENUS_HANDLE(VkShaderModule,
                              &retired->vertex_shader_obj);
   if (retired->fragment_shader)
      retired->fragment_shader =
         YTTRIUM_VENUS_HANDLE(VkShaderModule,
                              &retired->fragment_shader_obj);
   if (retired->pipeline)
      retired->pipeline =
         YTTRIUM_VENUS_HANDLE(VkPipeline, &retired->pipeline_obj);
   if (retired->push_descriptor_set_layout)
      retired->push_descriptor_set_layout =
         YTTRIUM_VENUS_HANDLE(VkDescriptorSetLayout,
                              &retired->push_descriptor_set_layout_obj);
   if (retired->push_descriptor_set_layout_alt)
      retired->push_descriptor_set_layout_alt =
         YTTRIUM_VENUS_HANDLE(
            VkDescriptorSetLayout,
            &retired->push_descriptor_set_layout_alt_obj);
   if (retired->push_pipeline_layout)
      retired->push_pipeline_layout =
         YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                              &retired->push_pipeline_layout_obj);
   if (retired->push_pipeline_layout_alt)
      retired->push_pipeline_layout_alt =
         YTTRIUM_VENUS_HANDLE(
            VkPipelineLayout, &retired->push_pipeline_layout_alt_obj);
   if (retired->push_pipeline)
      retired->push_pipeline =
         YTTRIUM_VENUS_HANDLE(VkPipeline, &retired->push_pipeline_obj);
   if (retired->pipeline_depth_image_view)
      retired->pipeline_depth_image_view =
         YTTRIUM_VENUS_HANDLE(VkImageView,
                              &retired->pipeline_depth_image_view_obj);
   for (unsigned i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
      if (retired->pipeline_image_views[i])
         retired->pipeline_image_views[i] =
            YTTRIUM_VENUS_HANDLE(VkImageView,
                                 &retired->pipeline_image_view_objs[i]);
   }
   for (unsigned i = 0; i < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; i++) {
      if (retired->pipeline_samplers[i])
         retired->pipeline_samplers[i] =
            YTTRIUM_VENUS_HANDLE(VkSampler,
                                 &retired->pipeline_sampler_objs[i]);
   }
   if (retired->draw_vertex_buffer)
      retired->draw_vertex_buffer =
         YTTRIUM_VENUS_HANDLE(VkBuffer,
                              &retired->draw_vertex_buffer_obj);
   if (retired->draw_vertex_memory)
      retired->draw_vertex_memory =
         YTTRIUM_VENUS_HANDLE(VkDeviceMemory,
                              &retired->draw_vertex_memory_obj);
   if (retired->draw_index_buffer)
      retired->draw_index_buffer =
         YTTRIUM_VENUS_HANDLE(VkBuffer,
                              &retired->draw_index_buffer_obj);
   if (retired->draw_index_memory)
      retired->draw_index_memory =
         YTTRIUM_VENUS_HANDLE(VkDeviceMemory,
                              &retired->draw_index_memory_obj);
   if (retired->device_local_draw_buffer)
      retired->device_local_draw_buffer =
         YTTRIUM_VENUS_HANDLE(
            VkBuffer, &retired->device_local_draw_buffer_obj);
   if (retired->device_local_draw_memory)
      retired->device_local_draw_memory =
         YTTRIUM_VENUS_HANDLE(
            VkDeviceMemory, &retired->device_local_draw_memory_obj);
}

struct yttrium_venus_retired_resource *
yttrium_venus_retired_resource_create(
   struct yttrium_venus_resource *resource)
{
   struct yttrium_venus_retired_resource *retired =
      CALLOC_STRUCT(yttrium_venus_retired_resource);
   if (!retired)
      return NULL;

   retired->image_obj = resource->image_obj;
   retired->buffer_obj = resource->buffer_obj;
   retired->memory_obj = resource->memory_obj;
   retired->image_view_obj = resource->image_view_obj;
   retired->render_pass_obj = resource->render_pass_obj;
   retired->framebuffer_obj = resource->framebuffer_obj;
   retired->pipeline_layout_obj = resource->pipeline_layout_obj;
   retired->descriptor_set_layout_obj = resource->descriptor_set_layout_obj;
   retired->descriptor_pool_obj = resource->descriptor_pool_obj;
   retired->sampler_obj = resource->sampler_obj;
   retired->vertex_shader_obj = resource->vertex_shader_obj;
   retired->fragment_shader_obj = resource->fragment_shader_obj;
   retired->pipeline_obj = resource->pipeline_obj;
   retired->draw_vertex_buffer_obj = resource->draw_vertex_buffer_obj;
   retired->draw_vertex_memory_obj = resource->draw_vertex_memory_obj;
   retired->draw_index_buffer_obj = resource->draw_index_buffer_obj;
   retired->draw_index_memory_obj = resource->draw_index_memory_obj;
   retired->device_local_draw_buffer_obj =
      resource->device_local_draw_buffer_obj;
   retired->device_local_draw_memory_obj =
      resource->device_local_draw_memory_obj;
   retired->image = resource->image;
   retired->buffer = resource->buffer;
   retired->memory = resource->memory;
   retired->image_view = resource->image_view;
   retired->sample_image_views = resource->sample_image_views;
   resource->sample_image_views = NULL;
   retired->sample_buffer_views = resource->sample_buffer_views;
   resource->sample_buffer_views = NULL;
   retired->render_pass = resource->render_pass;
   retired->framebuffer = resource->framebuffer;
   retired->pipeline_layout = resource->pipeline_layout;
   retired->descriptor_set_layout = resource->descriptor_set_layout;
   retired->descriptor_pool = resource->descriptor_pool;
   retired->sampler = resource->sampler;
   retired->vertex_shader = resource->vertex_shader;
   retired->fragment_shader = resource->fragment_shader;
   retired->pipeline = resource->pipeline;
   retired->draw_vertex_buffer = resource->draw_vertex_buffer;
   retired->draw_vertex_memory = resource->draw_vertex_memory;
   retired->draw_index_buffer = resource->draw_index_buffer;
   retired->draw_index_memory = resource->draw_index_memory;
   retired->device_local_draw_buffer =
      resource->device_local_draw_buffer;
   retired->device_local_draw_memory =
      resource->device_local_draw_memory;
   retired->draw_vertex_mapping = resource->draw_vertex_mapping;
   retired->draw_index_mapping = resource->draw_index_mapping;

   yttrium_venus_retired_resource_rebuild_handles(retired);
   return retired;
}

struct yttrium_venus_retired_resource *
yttrium_venus_retired_graphics_objects_take(
   struct yttrium_venus_resource *resource)
{
   if (!resource ||
       (!resource->image_view && !resource->render_pass &&
        !resource->framebuffer && !resource->pipeline_layout &&
        !resource->descriptor_set_layout && !resource->descriptor_pool &&
        !resource->sampler && !resource->vertex_shader &&
        !resource->fragment_shader && !resource->pipeline))
      return NULL;

   struct yttrium_venus_retired_resource *retired =
      CALLOC_STRUCT(yttrium_venus_retired_resource);
   if (!retired)
      return NULL;

   retired->image_view_obj = resource->image_view_obj;
   retired->render_pass_obj = resource->render_pass_obj;
   retired->framebuffer_obj = resource->framebuffer_obj;
   retired->pipeline_layout_obj = resource->pipeline_layout_obj;
   retired->descriptor_set_layout_obj = resource->descriptor_set_layout_obj;
   retired->descriptor_pool_obj = resource->descriptor_pool_obj;
   retired->sampler_obj = resource->sampler_obj;
   retired->vertex_shader_obj = resource->vertex_shader_obj;
   retired->fragment_shader_obj = resource->fragment_shader_obj;
   retired->pipeline_obj = resource->pipeline_obj;
   retired->image_view = resource->image_view;
   retired->render_pass = resource->render_pass;
   retired->framebuffer = resource->framebuffer;
   retired->pipeline_layout = resource->pipeline_layout;
   retired->descriptor_set_layout = resource->descriptor_set_layout;
   retired->descriptor_pool = resource->descriptor_pool;
   retired->sampler = resource->sampler;
   retired->vertex_shader = resource->vertex_shader;
   retired->fragment_shader = resource->fragment_shader;
   retired->pipeline = resource->pipeline;
   yttrium_venus_retired_resource_rebuild_handles(retired);

   memset(&resource->image_view_obj, 0, sizeof(resource->image_view_obj));
   memset(&resource->render_pass_obj, 0, sizeof(resource->render_pass_obj));
   memset(&resource->framebuffer_obj, 0, sizeof(resource->framebuffer_obj));
   memset(&resource->pipeline_layout_obj, 0,
          sizeof(resource->pipeline_layout_obj));
   memset(&resource->descriptor_set_layout_obj, 0,
          sizeof(resource->descriptor_set_layout_obj));
   memset(&resource->descriptor_pool_obj, 0,
          sizeof(resource->descriptor_pool_obj));
   memset(&resource->descriptor_set_obj, 0,
          sizeof(resource->descriptor_set_obj));
   memset(&resource->sampler_obj, 0, sizeof(resource->sampler_obj));
   memset(&resource->vertex_shader_obj, 0,
          sizeof(resource->vertex_shader_obj));
   memset(&resource->fragment_shader_obj, 0,
          sizeof(resource->fragment_shader_obj));
   memset(&resource->pipeline_obj, 0, sizeof(resource->pipeline_obj));

   resource->image_view = VK_NULL_HANDLE;
   resource->render_pass = VK_NULL_HANDLE;
   resource->framebuffer = VK_NULL_HANDLE;
   resource->pipeline_layout = VK_NULL_HANDLE;
   resource->descriptor_set_layout = VK_NULL_HANDLE;
   resource->descriptor_pool = VK_NULL_HANDLE;
   resource->descriptor_set = VK_NULL_HANDLE;
   resource->sampler = VK_NULL_HANDLE;
   resource->vertex_shader = VK_NULL_HANDLE;
   resource->fragment_shader = VK_NULL_HANDLE;
   resource->pipeline = VK_NULL_HANDLE;
   return retired;
}

struct yttrium_venus_retired_resource *
yttrium_venus_retired_draw_vertex_backing_take(
   struct yttrium_venus_resource *resource)
{
   if (!resource || (!resource->draw_vertex_buffer &&
                     !resource->draw_vertex_memory))
      return NULL;

   struct yttrium_venus_retired_resource *retired =
      CALLOC_STRUCT(yttrium_venus_retired_resource);
   if (!retired)
      return NULL;

   retired->draw_vertex_buffer_obj = resource->draw_vertex_buffer_obj;
   retired->draw_vertex_memory_obj = resource->draw_vertex_memory_obj;
   retired->draw_vertex_buffer = resource->draw_vertex_buffer;
   retired->draw_vertex_memory = resource->draw_vertex_memory;
   retired->draw_vertex_mapping = resource->draw_vertex_mapping;
   retired->draw_backing_kind = YTTRIUM_VENUS_DRAW_BACKING_VERTEX;
   retired->draw_backing_size = resource->draw_vertex_buffer_size;
   yttrium_venus_retired_resource_rebuild_handles(retired);

   memset(&resource->draw_vertex_buffer_obj, 0,
          sizeof(resource->draw_vertex_buffer_obj));
   memset(&resource->draw_vertex_memory_obj, 0,
          sizeof(resource->draw_vertex_memory_obj));
   resource->draw_vertex_buffer = VK_NULL_HANDLE;
   resource->draw_vertex_memory = VK_NULL_HANDLE;
   memset(&resource->draw_vertex_mapping, 0,
          sizeof(resource->draw_vertex_mapping));
   resource->draw_vertex_buffer_size = 0;
   resource->draw_vertex_roll_base = 0;
   return retired;
}

struct yttrium_venus_retired_resource *
yttrium_venus_retired_draw_index_backing_take(
   struct yttrium_venus_resource *resource)
{
   if (!resource || (!resource->draw_index_buffer &&
                     !resource->draw_index_memory))
      return NULL;

   struct yttrium_venus_retired_resource *retired =
      CALLOC_STRUCT(yttrium_venus_retired_resource);
   if (!retired)
      return NULL;

   retired->draw_index_buffer_obj = resource->draw_index_buffer_obj;
   retired->draw_index_memory_obj = resource->draw_index_memory_obj;
   retired->draw_index_buffer = resource->draw_index_buffer;
   retired->draw_index_memory = resource->draw_index_memory;
   retired->draw_index_mapping = resource->draw_index_mapping;
   retired->draw_backing_kind = YTTRIUM_VENUS_DRAW_BACKING_INDEX;
   retired->draw_backing_size = resource->draw_index_buffer_size;
   yttrium_venus_retired_resource_rebuild_handles(retired);

   memset(&resource->draw_index_buffer_obj, 0,
          sizeof(resource->draw_index_buffer_obj));
   memset(&resource->draw_index_memory_obj, 0,
          sizeof(resource->draw_index_memory_obj));
   resource->draw_index_buffer = VK_NULL_HANDLE;
   resource->draw_index_memory = VK_NULL_HANDLE;
   memset(&resource->draw_index_mapping, 0,
          sizeof(resource->draw_index_mapping));
   resource->draw_index_buffer_size = 0;
   resource->draw_index_roll_base = 0;
   return retired;
}

struct yttrium_venus_retired_resource *
yttrium_venus_retired_pipeline_create(struct yttrium_pipeline *pipeline)
{
   struct yttrium_venus_retired_resource *retired =
      CALLOC_STRUCT(yttrium_venus_retired_resource);
   if (!retired)
      return NULL;

   retired->render_target_cache_entry =
      pipeline->render_target_cache_entry;
   if (!retired->render_target_cache_entry) {
      retired->render_pass_obj = pipeline->render_pass_obj;
      retired->framebuffer_obj = pipeline->framebuffer_obj;
      memcpy(retired->pipeline_image_view_objs, pipeline->image_view_objs,
             sizeof(retired->pipeline_image_view_objs));
      retired->pipeline_depth_image_view_obj =
         pipeline->depth_image_view_obj;
      retired->render_pass = pipeline->render_pass;
      retired->framebuffer = pipeline->framebuffer;
      memcpy(retired->pipeline_image_views, pipeline->image_views,
             sizeof(retired->pipeline_image_views));
      retired->pipeline_depth_image_view = pipeline->depth_image_view;
   }
   retired->pipeline_layout_obj = pipeline->pipeline_layout_obj;
   retired->descriptor_set_layout_obj = pipeline->descriptor_set_layout_obj;
   retired->descriptor_pool_obj = pipeline->descriptor_pool_obj;
   retired->pipeline_obj = pipeline->pipeline_obj;
   retired->push_descriptor_set_layout_obj =
      pipeline->push_descriptor_set_layout_obj;
   retired->push_descriptor_set_layout_alt_obj =
      pipeline->push_descriptor_set_layout_alt_obj;
   retired->push_pipeline_layout_obj = pipeline->push_pipeline_layout_obj;
   retired->push_pipeline_layout_alt_obj =
      pipeline->push_pipeline_layout_alt_obj;
   retired->push_pipeline_obj = pipeline->push_pipeline_obj;
   memcpy(retired->pipeline_sampler_objs, pipeline->sampler_objs,
          sizeof(retired->pipeline_sampler_objs));

   retired->pipeline_layout = pipeline->pipeline_layout;
   retired->descriptor_set_layout = pipeline->descriptor_set_layout;
   retired->descriptor_pool = pipeline->descriptor_pool;
   retired->pipeline = pipeline->pipeline;
   retired->push_descriptor_set_layout =
      pipeline->push_descriptor_set_layout;
   retired->push_descriptor_set_layout_alt =
      pipeline->push_descriptor_set_layout_alt;
   retired->push_pipeline_layout = pipeline->push_pipeline_layout;
   retired->push_pipeline_layout_alt = pipeline->push_pipeline_layout_alt;
   retired->push_pipeline = pipeline->push_pipeline;
   memcpy(retired->pipeline_samplers, pipeline->samplers,
          sizeof(retired->pipeline_samplers));
   retired->custom_border_color_sampler_count =
      pipeline->custom_border_color_sampler_count;

   yttrium_venus_retired_resource_rebuild_handles(retired);
   return retired;
}

void
yttrium_venus_retired_resource_adopt_allocation(
   struct yttrium_venus_retired_resource *retired,
   const struct yttrium_venus_allocation_snapshot *allocation)
{
   if (!retired || !allocation || !allocation->hAllocation)
      return;

   retired->hAllocation = (D3DKMT_HANDLE)allocation->hAllocation;
   /*
    * D3DDDICB_DEALLOCATE::hResource is a runtime object, never the KM resource
    * handle returned by pfnAllocateCb.  Deferred retirement can outlive that
    * runtime object, so resource-backed allocations with a KM token must use
    * the allocation HandleList path.  resource_fini drains allocations which
    * have only a runtime handle before adopting them, making that one pointer
    * safe for the immediate destroy below.
    */
   if (!allocation->hResourceIsD3D9Runtime && allocation->hResource &&
       !allocation->hAllocationResource)
      retired->hResource = allocation->hResource;
   else
      retired->hResource = NULL;
   retired->allocation_size = allocation->size;
   retired->map = allocation->map;
   retired->map_info = allocation->map_info;
   retired->map_is_blob = allocation->map_is_blob;
   retired->owns_allocation = allocation->owns_allocation;
   retired->allocation_destroyed_by_runtime =
      allocation->allocation_destroyed_by_runtime;
}

static void
yttrium_venus_destroy_retired_allocation(
   struct yttrium_venus *venus,
   struct yttrium_venus_retired_resource *retired)
{
   if (!venus || !venus->device || !retired || !retired->hAllocation)
      return;

   if (retired->allocation_destroyed_by_runtime) {
      YTTRIUM_LOG("yttrium: Venus release runtime-owned allocation without "
                  "DeallocateCb hAllocation=0x%lx size=0x%llx\n",
                  (unsigned long)retired->hAllocation,
                  (unsigned long long)retired->allocation_size);
      retired->hAllocation = 0;
      retired->hResource = NULL;
      retired->map = NULL;
      retired->map_info = 0;
      retired->map_is_blob = false;
      return;
   }

   if (retired->map && retired->map_is_blob) {
      VIOGPU_ESCAPE unmap;
      memset(&unmap, 0, sizeof(unmap));
      unmap.Type = VIOGPU_RES_UNMAP_BLOB;
      unmap.DataLength = sizeof(VIOGPU_RES_UNMAP_BLOB_REQ);
      unmap.ResourceUnmapBlob.ResHandle = retired->hAllocation;

      NTSTATUS status = venus->device->escape(venus->device, &unmap,
                                              sizeof(unmap));
      if (!NT_SUCCESS(status)) {
         YTTRIUM_LOG("yttrium: Venus retired allocation unmap blob failed status=0x%lx hAllocation=0x%lx size=0x%llx\n",
                     (unsigned long)status,
                     (unsigned long)retired->hAllocation,
                     (unsigned long long)retired->allocation_size);
      }
   }

   NTSTATUS status =
      venus->device->destroyAllocation(venus->device, retired->hResource,
                                       retired->hAllocation);
   YTTRIUM_LOG("yttrium: Venus retired allocation %s hAllocation=0x%lx hResource=%p status=0x%lx size=0x%llx map=%p map_blob=%u\n",
               retired->owns_allocation ? "destroy" : "close",
               (unsigned long)retired->hAllocation,
               retired->hResource,
               (unsigned long)status,
               (unsigned long long)retired->allocation_size,
               retired->map,
               retired->map_is_blob);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: Venus retired allocation destroy failed owner=venus2 status=0x%lx hAllocation=0x%lx hResource=%p owns=%u runtime_destroyed=%u size=0x%llx map=%p map_blob=%u image_id=%llu buffer_id=%llu memory_id=%llu pid=%lu tid=%lu\n",
                   status, (unsigned long)retired->hAllocation,
                   retired->hResource, retired->owns_allocation,
                   retired->allocation_destroyed_by_runtime,
                   (unsigned long long)retired->allocation_size,
                   retired->map, retired->map_is_blob,
                   (unsigned long long)retired->image_obj.id,
                   (unsigned long long)retired->buffer_obj.id,
                   (unsigned long long)retired->memory_obj.id,
                   (unsigned long)GetCurrentProcessId(),
                   (unsigned long)GetCurrentThreadId());
   }

   retired->hAllocation = 0;
   retired->hResource = NULL;
   retired->map = NULL;
   retired->map_info = 0;
   retired->map_is_blob = false;
}

void
yttrium_venus_destroy_retired_resource(
   struct yttrium_venus *venus,
   struct yttrium_venus_retired_resource *retired)
{
   if (!venus || !retired)
      return;

   if (retired->pipeline)
      vn_async_vkDestroyPipeline(&venus->vn_ring, venus->device_handle,
                                 retired->pipeline, NULL);
   if (retired->push_pipeline)
      vn_async_vkDestroyPipeline(&venus->vn_ring, venus->device_handle,
                                 retired->push_pipeline, NULL);
   if (retired->vertex_shader)
      vn_async_vkDestroyShaderModule(&venus->vn_ring, venus->device_handle,
                                     retired->vertex_shader, NULL);
   if (retired->fragment_shader)
      vn_async_vkDestroyShaderModule(&venus->vn_ring, venus->device_handle,
                                     retired->fragment_shader, NULL);
   if (retired->pipeline_layout)
      vn_async_vkDestroyPipelineLayout(&venus->vn_ring, venus->device_handle,
                                       retired->pipeline_layout, NULL);
   if (retired->push_pipeline_layout)
      vn_async_vkDestroyPipelineLayout(&venus->vn_ring, venus->device_handle,
                                       retired->push_pipeline_layout, NULL);
   if (retired->push_pipeline_layout_alt)
      vn_async_vkDestroyPipelineLayout(
         &venus->vn_ring, venus->device_handle,
         retired->push_pipeline_layout_alt, NULL);
   if (retired->descriptor_pool)
      vn_async_vkDestroyDescriptorPool(&venus->vn_ring, venus->device_handle,
                                       retired->descriptor_pool, NULL);
   if (retired->descriptor_set_layout)
      vn_async_vkDestroyDescriptorSetLayout(
         &venus->vn_ring, venus->device_handle,
         retired->descriptor_set_layout, NULL);
   if (retired->push_descriptor_set_layout)
      vn_async_vkDestroyDescriptorSetLayout(
         &venus->vn_ring, venus->device_handle,
         retired->push_descriptor_set_layout, NULL);
   if (retired->push_descriptor_set_layout_alt)
      vn_async_vkDestroyDescriptorSetLayout(
         &venus->vn_ring, venus->device_handle,
         retired->push_descriptor_set_layout_alt, NULL);
   if (retired->sampler)
      vn_async_vkDestroySampler(&venus->vn_ring, venus->device_handle,
                                retired->sampler, NULL);
   if (retired->framebuffer)
      vn_async_vkDestroyFramebuffer(&venus->vn_ring, venus->device_handle,
                                    retired->framebuffer, NULL);
   if (retired->render_pass)
      vn_async_vkDestroyRenderPass(&venus->vn_ring, venus->device_handle,
                                   retired->render_pass, NULL);
   if (retired->image_view)
      vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                  retired->image_view, NULL);
   if (retired->pipeline_depth_image_view)
      vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                  retired->pipeline_depth_image_view, NULL);
   for (uint32_t i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
      if (retired->pipeline_image_views[i])
         vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                     retired->pipeline_image_views[i], NULL);
   }
   for (uint32_t i = 0; i < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; i++) {
      if (retired->pipeline_samplers[i])
         vn_async_vkDestroySampler(&venus->vn_ring, venus->device_handle,
                                   retired->pipeline_samplers[i], NULL);
   }
   assert(venus->custom_border_color_sampler_count >=
          retired->custom_border_color_sampler_count);
   venus->custom_border_color_sampler_count -=
      retired->custom_border_color_sampler_count;

   struct yttrium_venus_sample_image_view *image_view =
      retired->sample_image_views;
   while (image_view) {
      struct yttrium_venus_sample_image_view *next = image_view->next;
      if (image_view->view)
         vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                     image_view->view, NULL);
      FREE(image_view);
      image_view = next;
   }
   retired->sample_image_views = NULL;

   struct yttrium_venus_sample_buffer_view *buffer_view =
      retired->sample_buffer_views;
   while (buffer_view) {
      struct yttrium_venus_sample_buffer_view *next = buffer_view->next;
      if (buffer_view->view)
         vn_async_vkDestroyBufferView(&venus->vn_ring, venus->device_handle,
                                      buffer_view->view, NULL);
      FREE(buffer_view);
      buffer_view = next;
   }

   yttrium_venus_unmap_memory(venus, &retired->draw_vertex_mapping);
   yttrium_venus_unmap_memory(venus, &retired->draw_index_mapping);

   if (retired->draw_vertex_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               retired->draw_vertex_buffer, NULL);
   if (retired->draw_vertex_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            retired->draw_vertex_memory, NULL);
   if (retired->draw_index_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               retired->draw_index_buffer, NULL);
   if (retired->draw_index_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            retired->draw_index_memory, NULL);
   if (retired->device_local_draw_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               retired->device_local_draw_buffer, NULL);
   if (retired->device_local_draw_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            retired->device_local_draw_memory, NULL);
   if (retired->buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               retired->buffer, NULL);
   if (retired->image)
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              retired->image, NULL);
   if (retired->memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            retired->memory, NULL);

   if (retired->hAllocation) {
      yttrium_venus_drain_ring(venus, "retired resource destroy");
      yttrium_venus_destroy_retired_allocation(venus, retired);
   }

   if (retired->render_target_cache_entry)
      yttrium_venus_render_target_release(
         venus, retired->render_target_cache_entry);

   FREE(retired);
}

static void
yttrium_venus_destroy_retired_resources(
   struct yttrium_venus *venus,
   struct yttrium_venus_retired_resource **list)
{
   if (!list)
      return;

   struct yttrium_venus_retired_resource *retired = *list;
   *list = NULL;
   while (retired) {
      struct yttrium_venus_retired_resource *next = retired->next;
      yttrium_venus_destroy_retired_resource(venus, retired);
      retired = next;
   }
}

void
yttrium_venus_recycle_draw_backing(
   struct yttrium_venus *venus,
   struct yttrium_venus_retired_resource *retired)
{
   if (!venus || !retired)
      return;

   const bool valid_vertex =
      retired->draw_backing_kind == YTTRIUM_VENUS_DRAW_BACKING_VERTEX &&
      retired->draw_vertex_buffer && retired->draw_vertex_memory &&
      retired->draw_vertex_mapping.map;
   const bool valid_index =
      retired->draw_backing_kind == YTTRIUM_VENUS_DRAW_BACKING_INDEX &&
      retired->draw_index_buffer && retired->draw_index_memory &&
      retired->draw_index_mapping.map;
   const bool within_limits =
      retired->draw_backing_size &&
      retired->draw_backing_size <=
         YTTRIUM_VENUS_DRAW_BACKING_POOL_MAX_ENTRY_BYTES &&
      venus->draw_backing_pool_count <
         YTTRIUM_VENUS_DRAW_BACKING_POOL_MAX_ENTRIES &&
      venus->draw_backing_pool_bytes + retired->draw_backing_size <=
         YTTRIUM_VENUS_DRAW_BACKING_POOL_MAX_BYTES;

   if (!yttrium_venus_draw_backing_pool_enabled() ||
       (!valid_vertex && !valid_index) || !within_limits) {
      yttrium_venus_destroy_retired_resource(venus, retired);
      return;
   }

   retired->next = venus->draw_backing_pool;
   venus->draw_backing_pool = retired;
   venus->draw_backing_pool_count++;
   venus->draw_backing_pool_bytes += retired->draw_backing_size;
   venus->peak_draw_backing_pool_bytes =
      MAX2(venus->peak_draw_backing_pool_bytes,
           venus->draw_backing_pool_bytes);
}

bool
yttrium_venus_take_draw_backing(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   enum yttrium_venus_draw_backing_kind kind,
   VkDeviceSize minimum_size)
{
   if (!venus || !resource || !minimum_size ||
       !yttrium_venus_draw_backing_pool_enabled())
      return false;

   struct yttrium_venus_retired_resource **best_link = NULL;
   VkDeviceSize best_size = UINT64_MAX;
   for (struct yttrium_venus_retired_resource **link =
           &venus->draw_backing_pool;
        *link; link = &(*link)->next) {
      struct yttrium_venus_retired_resource *candidate = *link;
      if (candidate->draw_backing_kind != kind ||
          candidate->draw_backing_size < minimum_size ||
          candidate->draw_backing_size >= best_size)
         continue;
      best_link = link;
      best_size = candidate->draw_backing_size;
   }

   if (!best_link) {
      return false;
   }

   struct yttrium_venus_retired_resource *backing = *best_link;
   *best_link = backing->next;
   backing->next = NULL;
   venus->draw_backing_pool_count--;
   venus->draw_backing_pool_bytes -= backing->draw_backing_size;

   if (kind == YTTRIUM_VENUS_DRAW_BACKING_VERTEX) {
      resource->draw_vertex_buffer_obj = backing->draw_vertex_buffer_obj;
      resource->draw_vertex_memory_obj = backing->draw_vertex_memory_obj;
      resource->draw_vertex_buffer =
         YTTRIUM_VENUS_HANDLE(VkBuffer,
                              &resource->draw_vertex_buffer_obj);
      resource->draw_vertex_memory =
         YTTRIUM_VENUS_HANDLE(VkDeviceMemory,
                              &resource->draw_vertex_memory_obj);
      resource->draw_vertex_mapping = backing->draw_vertex_mapping;
      resource->draw_vertex_buffer_size = backing->draw_backing_size;
      resource->draw_vertex_roll_base = 0;
      resource->draw_vertex_buffer_generation++;
   } else {
      resource->draw_index_buffer_obj = backing->draw_index_buffer_obj;
      resource->draw_index_memory_obj = backing->draw_index_memory_obj;
      resource->draw_index_buffer =
         YTTRIUM_VENUS_HANDLE(VkBuffer,
                              &resource->draw_index_buffer_obj);
      resource->draw_index_memory =
         YTTRIUM_VENUS_HANDLE(VkDeviceMemory,
                              &resource->draw_index_memory_obj);
      resource->draw_index_mapping = backing->draw_index_mapping;
      resource->draw_index_buffer_size = backing->draw_backing_size;
      resource->draw_index_roll_base = 0;
      resource->draw_index_buffer_generation++;
   }

   FREE(backing);
   return true;
}

static void
yttrium_venus_retire_resources_after_host_wait(
   struct yttrium_venus *venus,
   struct yttrium_venus_retired_resource **list)
{
   if (!list)
      return;

   struct yttrium_venus_retired_resource *retired = *list;
   *list = NULL;
   while (retired) {
      struct yttrium_venus_retired_resource *next = retired->next;
      retired->next = NULL;
      if (retired->draw_backing_kind !=
          YTTRIUM_VENUS_DRAW_BACKING_NONE)
         yttrium_venus_recycle_draw_backing(venus, retired);
      else
         yttrium_venus_destroy_retired_resource(venus, retired);
      retired = next;
   }
}

static void
yttrium_venus_destroy_draw_backing_pool(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   venus->draw_backing_pool_count = 0;
   venus->draw_backing_pool_bytes = 0;
   yttrium_venus_destroy_retired_resources(
      venus, &venus->draw_backing_pool);
}

bool
yttrium_venus_cmd_batch_track_resource(struct yttrium_venus *venus,
                                       struct yttrium_venus_resource *resource)
{
   if (!resource)
      return true;

   for (uint32_t i = 0; i < venus->cmd_batch_pending_resource_count; i++) {
      if (venus->cmd_batch_pending_resources[i] == resource)
         return true;
   }

   if (venus->cmd_batch_pending_resource_count >=
       ARRAY_SIZE(venus->cmd_batch_pending_resources)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch resource pending list overflow\n");
      return false;
   }

   const uint32_t index = venus->cmd_batch_pending_resource_count++;
   venus->cmd_batch_pending_resources[index] = resource;
   if (resource->owner)
      pipe_resource_reference(&venus->cmd_batch_pending_resource_refs[index],
                              resource->owner);
   return true;
}

static bool
yttrium_venus_cmd_batch_ensure_draw_mirror_update_capacity(
   struct yttrium_venus *venus)
{
   if (venus->cmd_batch_draw_mirror_update_count <
       venus->cmd_batch_draw_mirror_update_capacity)
      return true;

   uint32_t new_capacity = venus->cmd_batch_draw_mirror_update_capacity ?
      venus->cmd_batch_draw_mirror_update_capacity * 2 : 16;
   if (new_capacity > YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT)
      new_capacity = YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT;
   if (new_capacity <= venus->cmd_batch_draw_mirror_update_capacity) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 draw-mirror transaction overflow owner=venus2 count=%u limit=%u\n",
                   venus->cmd_batch_draw_mirror_update_count,
                   YTTRIUM_VENUS_CMD_BATCH_PENDING_RESOURCE_REF_LIMIT);
      return false;
   }

   struct yttrium_venus_draw_mirror_update *updates =
      realloc(venus->cmd_batch_draw_mirror_updates,
              new_capacity * sizeof(*updates));
   if (!updates) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 draw-mirror transaction allocation failed owner=venus2 count=%u capacity=%u\n",
                   venus->cmd_batch_draw_mirror_update_count, new_capacity);
      return false;
   }
   venus->cmd_batch_draw_mirror_updates = updates;
   venus->cmd_batch_draw_mirror_update_capacity = new_capacity;
   return true;
}

bool
yttrium_venus_cmd_batch_record_draw_mirror_update(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t source_serial)
{
   if (!venus || !resource || !venus->display_copy_batch_recording)
      return false;

   bool tracked = false;
   for (uint32_t i = 0; i < venus->cmd_batch_pending_resource_count; i++) {
      if (venus->cmd_batch_pending_resources[i] == resource) {
         tracked = true;
         break;
      }
   }
   if (!tracked) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 draw-mirror update rejected owner=venus2 reason=resource_not_batch_owned\n");
      return false;
   }

   for (uint32_t i = 0;
        i < venus->cmd_batch_draw_mirror_update_count; i++) {
      struct yttrium_venus_draw_mirror_update *update =
         &venus->cmd_batch_draw_mirror_updates[i];
      if (update->resource != resource)
         continue;

      update->source_serial = source_serial;
      resource->device_local_draw_pending_serial = source_serial;
      resource->device_local_draw_pending_valid = true;
      return true;
   }

   if (!yttrium_venus_cmd_batch_ensure_draw_mirror_update_capacity(venus))
      return false;

   struct yttrium_venus_draw_mirror_update *update =
      &venus->cmd_batch_draw_mirror_updates
         [venus->cmd_batch_draw_mirror_update_count++];
   update->resource = resource;
   update->source_serial = source_serial;
   update->previous_pending_serial =
      resource->device_local_draw_pending_serial;
   update->previous_pending_valid =
      resource->device_local_draw_pending_valid;
   resource->device_local_draw_pending_serial = source_serial;
   resource->device_local_draw_pending_valid = true;
   return true;
}

static void
yttrium_venus_restore_draw_mirror_updates(
   struct yttrium_venus_draw_mirror_update *updates,
   uint32_t *count)
{
   if (!updates || !count)
      return;

   while (*count) {
      struct yttrium_venus_draw_mirror_update *update =
         &updates[--*count];
      if (!update->resource)
         continue;
      update->resource->device_local_draw_pending_serial =
         update->previous_pending_serial;
      update->resource->device_local_draw_pending_valid =
         update->previous_pending_valid;
   }
}

static void
yttrium_venus_apply_draw_mirror_updates(
   struct yttrium_venus_batch *batch)
{
   if (!batch)
      return;
   for (uint32_t i = 0; i < batch->draw_mirror_update_count; i++) {
      struct yttrium_venus_draw_mirror_update *update =
         &batch->draw_mirror_updates[i];
      if (!update->resource)
         continue;
      update->resource->device_local_draw_contents_serial =
         update->source_serial;
      update->resource->device_local_draw_contents_valid = true;
   }
}

static void
yttrium_venus_clear_draw_mirror_pending_state(
   struct yttrium_venus_batch *batch)
{
   if (!batch)
      return;
   for (uint32_t i = 0; i < batch->draw_mirror_update_count; i++) {
      struct yttrium_venus_resource *resource =
         batch->draw_mirror_updates[i].resource;
      if (resource)
         resource->device_local_draw_pending_valid = false;
   }
   batch->draw_mirror_update_count = 0;
}

static void
yttrium_venus_commit_draw_mirror_updates(
   struct yttrium_venus_batch *batch)
{
   yttrium_venus_apply_draw_mirror_updates(batch);
   yttrium_venus_clear_draw_mirror_pending_state(batch);
}

static bool
yttrium_venus_move_draw_mirror_updates_to_batch(
   struct yttrium_venus *venus,
   struct yttrium_venus_batch *batch)
{
   if (!venus || !batch || batch->draw_mirror_update_count) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 draw-mirror transaction handoff failed owner=venus2 reason=%s\n",
                   !venus || !batch ? "invalid_batch" :
                                      "destination_not_empty");
      return false;
   }
   if (!venus->cmd_batch_draw_mirror_update_count)
      return true;

   struct yttrium_venus_draw_mirror_update *updates =
      batch->draw_mirror_updates;
   const uint32_t capacity = batch->draw_mirror_update_capacity;
   batch->draw_mirror_updates = venus->cmd_batch_draw_mirror_updates;
   batch->draw_mirror_update_count =
      venus->cmd_batch_draw_mirror_update_count;
   batch->draw_mirror_update_capacity =
      venus->cmd_batch_draw_mirror_update_capacity;
   venus->cmd_batch_draw_mirror_updates = updates;
   venus->cmd_batch_draw_mirror_update_count = 0;
   venus->cmd_batch_draw_mirror_update_capacity = capacity;
   return true;
}

bool
yttrium_venus_cmd_batch_track_pipeline(struct yttrium_venus *venus,
                                       struct yttrium_pipeline *pipeline)
{
   if (!pipeline)
      return true;

   for (uint32_t i = 0; i < venus->cmd_batch_pending_pipeline_count; i++) {
      if (venus->cmd_batch_pending_pipelines[i] == pipeline)
         return true;
   }

   if (venus->cmd_batch_pending_pipeline_count >=
       ARRAY_SIZE(venus->cmd_batch_pending_pipelines)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch pipeline pending list overflow\n");
      return false;
   }

   venus->cmd_batch_pending_pipelines
      [venus->cmd_batch_pending_pipeline_count++] = pipeline;
   return true;
}

bool
yttrium_venus_cmd_batch_track_draw_refs(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   const struct yttrium_venus_vertex_upload *vertex_uploads,
   uint32_t vertex_upload_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   struct yttrium_pipeline *pipeline)
{
   if (!yttrium_venus_cmd_batch_track_resource(venus, resource))
      return false;
   for (uint32_t i = 0; i < vertex_upload_count; i++) {
      if (!yttrium_venus_cmd_batch_track_resource(
             venus, vertex_uploads[i].resource))
         return false;
   }
   for (uint32_t i = 0; i < color_resource_count; i++) {
      if (!yttrium_venus_cmd_batch_track_resource(venus, color_resources[i]))
         return false;
   }
   if (!yttrium_venus_cmd_batch_track_resource(venus, depth_resource))
      return false;
   return yttrium_venus_cmd_batch_track_pipeline(venus, pipeline);
}

bool
yttrium_venus_cmd_batch_track_stream_output_refs(
   struct yttrium_venus *venus,
   const struct yttrium_venus_stream_output_target *targets,
   uint32_t target_count,
   const struct yttrium_venus_stream_output_target *draw_auto_target)
{
   for (uint32_t i = 0; i < target_count; i++) {
      const struct yttrium_venus_stream_output_target *target = &targets[i];

      if (!yttrium_venus_cmd_batch_track_resource(venus, target->resource))
         return false;
      if (!yttrium_venus_cmd_batch_track_resource(venus,
                                                  target->counter_resource))
         return false;
   }

   if (draw_auto_target &&
       !yttrium_venus_cmd_batch_track_resource(
          venus, draw_auto_target->counter_resource))
      return false;

   return true;
}

bool
yttrium_venus_cmd_batch_track_sampled_refs(
   struct yttrium_venus *venus,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count)
{
   for (uint32_t i = 0; i < sampled_image_count; i++) {
      const struct yttrium_venus_sampled_image *sampled = &sampled_images[i];
      struct yttrium_venus_resource *resource = sampled->resource;

      if (!resource && !sampled->buffer) {
         uint32_t null_resource_id = 0;
         if (!yttrium_venus_ensure_null_sampled_image(
                venus, &resource, &null_resource_id))
            return false;
      }

      if (!yttrium_venus_cmd_batch_track_resource(venus, resource))
         return false;
   }

   return true;
}

bool
yttrium_venus_cmd_batch_track_storage_refs(
   struct yttrium_venus *venus,
   const struct yttrium_venus_storage_image *storage_images,
   uint32_t storage_image_count)
{
   for (uint32_t i = 0; i < storage_image_count; i++) {
      if (!yttrium_venus_cmd_batch_track_resource(
             venus, storage_images[i].resource))
         return false;
   }

   return true;
}

static void
yttrium_venus_cmd_batch_clear_footprints(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   venus->cmd_batch_footprint_count = 0;
   memset(&venus->cmd_batch_ubo_footprint, 0,
          sizeof(venus->cmd_batch_ubo_footprint));
   memset(&venus->cmd_batch_vertex_footprint, 0,
          sizeof(venus->cmd_batch_vertex_footprint));
   memset(&venus->cmd_batch_index_footprint, 0,
          sizeof(venus->cmd_batch_index_footprint));
   if (venus->cmd_batch_sampled_image_roles)
      _mesa_hash_table_u64_clear(venus->cmd_batch_sampled_image_roles);
   if (venus->cmd_batch_attachment_image_roles)
      _mesa_hash_table_u64_clear(venus->cmd_batch_attachment_image_roles);
   if (venus->cmd_batch_descriptor_sets)
      _mesa_hash_table_u64_clear(venus->cmd_batch_descriptor_sets);
}

void
yttrium_venus_cmd_batch_destroy_footprint_index(
   struct yttrium_venus *venus)
{
   if (!venus)
      return;

   _mesa_hash_table_u64_destroy(venus->cmd_batch_sampled_image_roles);
   _mesa_hash_table_u64_destroy(venus->cmd_batch_attachment_image_roles);
   _mesa_hash_table_u64_destroy(venus->cmd_batch_descriptor_sets);
   venus->cmd_batch_sampled_image_roles = NULL;
   venus->cmd_batch_attachment_image_roles = NULL;
   venus->cmd_batch_descriptor_sets = NULL;
}

static bool
yttrium_venus_cmd_batch_ensure_footprint_index(
   struct yttrium_venus *venus)
{
   if (!venus->cmd_batch_sampled_image_roles)
      venus->cmd_batch_sampled_image_roles =
         _mesa_hash_table_u64_create(NULL);
   if (!venus->cmd_batch_attachment_image_roles)
      venus->cmd_batch_attachment_image_roles =
         _mesa_hash_table_u64_create(NULL);
   if (!venus->cmd_batch_descriptor_sets)
      venus->cmd_batch_descriptor_sets =
         _mesa_hash_table_u64_create(NULL);

   return venus->cmd_batch_sampled_image_roles &&
          venus->cmd_batch_attachment_image_roles &&
          venus->cmd_batch_descriptor_sets;
}

static void
yttrium_venus_cmd_batch_clear_deferred_draws(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   venus->cmd_batch_deferred_draw_count = 0;
   venus->cmd_batch_compact_draw_packet_size = 0;
}

static bool
yttrium_venus_cmd_batch_buffer_footprint_overlaps(
   const struct yttrium_venus_cmd_batch_buffer_footprint *a,
   const struct yttrium_venus_cmd_batch_buffer_footprint *b)
{
   if (!a || !b || !a->buffer || !b->buffer || !a->size || !b->size)
      return false;

   if (a->buffer != b->buffer || a->generation != b->generation)
      return false;

   const VkDeviceSize a_end = a->offset + a->size;
   const VkDeviceSize b_end = b->offset + b->size;
   if (a_end < a->offset || b_end < b->offset)
      return true;

   return a->offset < b_end && b->offset < a_end;
}

static bool
yttrium_venus_cmd_batch_ensure_footprint_capacity(
   struct yttrium_venus *venus)
{
   if (venus->cmd_batch_footprint_count < venus->cmd_batch_footprint_capacity)
      return true;

   uint32_t new_capacity = venus->cmd_batch_footprint_capacity ?
      venus->cmd_batch_footprint_capacity * 2 : 64;
   const uint32_t limit = yttrium_venus_native_draw_batch_limit();
   if (new_capacity < venus->cmd_batch_footprint_count + 1)
      new_capacity = venus->cmd_batch_footprint_count + 1;
   if (new_capacity > limit)
      new_capacity = limit;
   if (new_capacity <= venus->cmd_batch_footprint_capacity) {
      YTTRIUM_LOG("yttrium: Venus cmd batch footprint overflow count=%u limit=%u\n",
                  venus->cmd_batch_footprint_count, limit);
      return false;
   }

   struct yttrium_venus_cmd_batch_footprint *footprints =
      realloc(venus->cmd_batch_footprints,
              new_capacity * sizeof(*venus->cmd_batch_footprints));
   if (!footprints) {
      YTTRIUM_LOG("yttrium: Venus cmd batch footprint allocation failed count=%u capacity=%u\n",
                  venus->cmd_batch_footprint_count, new_capacity);
      return false;
   }

   memset(footprints + venus->cmd_batch_footprint_capacity, 0,
          (new_capacity - venus->cmd_batch_footprint_capacity) *
          sizeof(*footprints));
   venus->cmd_batch_footprints = footprints;
   venus->cmd_batch_footprint_capacity = new_capacity;
   return true;
}

static bool
yttrium_venus_cmd_batch_id_in_list(uint64_t id, const uint64_t *ids,
                                   uint32_t count)
{
   if (!id || !ids)
      return false;

   for (uint32_t i = 0; i < count; i++) {
      if (ids[i] == id)
         return true;
   }
   return false;
}

void
yttrium_venus_cmd_batch_collect_image_roles(
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   bool depth_read_only,
   struct yttrium_venus_cmd_batch_footprint *footprint)
{
   uint32_t sampled_count = 0;
   uint32_t attachment_count = 0;
   uint64_t *sampled_ids = footprint->sampled_image_ids;
   uint64_t *attachment_ids = footprint->attachment_image_ids;

   /* Preserve the descriptor and buffer footprint populated by the caller. */
   footprint->read_only_depth_image_id = 0;
   for (uint32_t i = 0;
        sampled_images && i < sampled_image_count &&
        sampled_count < ARRAY_SIZE(footprint->sampled_image_ids);
        i++) {
      if (sampled_images[i].buffer || !sampled_images[i].resource)
         continue;

      const uint64_t id = sampled_images[i].resource->image_obj.id;
      if (!yttrium_venus_cmd_batch_id_in_list(id, sampled_ids,
                                              sampled_count))
         sampled_ids[sampled_count++] = id;
   }
   footprint->sampled_image_count = sampled_count;

   for (uint32_t i = 0;
        color_resources && i < color_resource_count &&
        attachment_count < ARRAY_SIZE(footprint->attachment_image_ids);
        i++) {
      if (!color_resources[i])
         continue;

      const uint64_t id = color_resources[i]->image_obj.id;
      if (!yttrium_venus_cmd_batch_id_in_list(id, attachment_ids,
                                              attachment_count))
         attachment_ids[attachment_count++] = id;
   }
   if (depth_resource) {
      if (depth_read_only) {
         /* This bit also certifies the caller's read-only layout path.  Keep
          * the depth read even if it is not listed among sampled resources,
          * so a later attachment write cannot pass the read-role index.
          */
         footprint->read_only_depth_image_id = depth_resource->image_obj.id;
      } else if (attachment_count <
                 ARRAY_SIZE(footprint->attachment_image_ids)) {
         const uint64_t id = depth_resource->image_obj.id;
         if (!yttrium_venus_cmd_batch_id_in_list(id, attachment_ids,
                                                 attachment_count))
            attachment_ids[attachment_count++] = id;
      }
   }
   footprint->attachment_image_count = attachment_count;
}

static bool
yttrium_venus_cmd_batch_index_lookup(
   const struct yttrium_venus *venus,
   struct hash_table_u64 *index,
   uint64_t key,
   const struct yttrium_venus_cmd_batch_footprint **seen)
{
   if (seen)
      *seen = NULL;
   if (!venus || !index || !key)
      return false;

   void *data = _mesa_hash_table_u64_search(index, key);
   if (!data)
      return false;

   const uintptr_t encoded = (uintptr_t)data;
   if (seen && encoded && encoded - 1 < venus->cmd_batch_footprint_count)
      *seen = &venus->cmd_batch_footprints[encoded - 1];
   return true;
}

static bool
yttrium_venus_cmd_batch_indexed_image_role_conflict(
   const struct yttrium_venus *venus,
   const struct yttrium_venus_cmd_batch_footprint *footprint,
   const struct yttrium_venus_cmd_batch_footprint **seen)
{
   if (seen)
      *seen = NULL;
   if (!venus || !footprint)
      return false;

   for (uint32_t i = 0; i < footprint->sampled_image_count; i++) {
      if (yttrium_venus_cmd_batch_index_lookup(
             venus, venus->cmd_batch_attachment_image_roles,
             footprint->sampled_image_ids[i], seen))
         return true;
   }
   if (yttrium_venus_cmd_batch_index_lookup(
          venus, venus->cmd_batch_attachment_image_roles,
          footprint->read_only_depth_image_id, seen))
      return true;
   for (uint32_t i = 0; i < footprint->attachment_image_count; i++) {
      if (yttrium_venus_cmd_batch_index_lookup(
             venus, venus->cmd_batch_sampled_image_roles,
             footprint->attachment_image_ids[i], seen))
         return true;
   }

   return false;
}

static bool
yttrium_venus_cmd_batch_index_footprint(
   struct yttrium_venus *venus,
   const struct yttrium_venus_cmd_batch_footprint *footprint)
{
   const uintptr_t encoded = (uintptr_t)venus->cmd_batch_footprint_count + 1;
   void *value = (void *)encoded;

   for (uint32_t i = 0; i < footprint->sampled_image_count; i++) {
      const uint64_t id = footprint->sampled_image_ids[i];
      if (!id)
         continue;
      _mesa_hash_table_u64_insert(venus->cmd_batch_sampled_image_roles,
                                  id, value);
      if (!_mesa_hash_table_u64_search(
             venus->cmd_batch_sampled_image_roles, id))
         return false;
   }
   /* The sampled-image table is the read-role index.  Read-only depth joins
    * it without consuming a sampled binding or an attachment-write role.
    */
   if (footprint->read_only_depth_image_id) {
      const uint64_t id = footprint->read_only_depth_image_id;
      _mesa_hash_table_u64_insert(venus->cmd_batch_sampled_image_roles,
                                  id, value);
      if (!_mesa_hash_table_u64_search(
             venus->cmd_batch_sampled_image_roles, id))
         return false;
   }
   for (uint32_t i = 0; i < footprint->attachment_image_count; i++) {
      const uint64_t id = footprint->attachment_image_ids[i];
      if (!id)
         continue;
      _mesa_hash_table_u64_insert(venus->cmd_batch_attachment_image_roles,
                                  id, value);
      if (!_mesa_hash_table_u64_search(
             venus->cmd_batch_attachment_image_roles, id))
         return false;
   }
   if (footprint->descriptor_set) {
      const uint64_t id = YTTRIUM_VENUS_HANDLE_TO_U64(
         footprint->descriptor_set);
      _mesa_hash_table_u64_insert(venus->cmd_batch_descriptor_sets,
                                  id, value);
      if (!_mesa_hash_table_u64_search(venus->cmd_batch_descriptor_sets, id))
         return false;
   }

   return true;
}

enum yttrium_venus_cmd_batch_buffer_index_kind {
   YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_UBO,
   YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_VERTEX,
   YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_INDEX,
};

static const struct yttrium_venus_cmd_batch_buffer_footprint *
yttrium_venus_cmd_batch_buffer_footprint(
   const struct yttrium_venus_cmd_batch_footprint *footprint,
   enum yttrium_venus_cmd_batch_buffer_index_kind kind)
{
   switch (kind) {
   case YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_UBO:
      return &footprint->ubo;
   case YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_VERTEX:
      return &footprint->vertex;
   case YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_INDEX:
      return &footprint->index;
   default:
      return NULL;
   }
}

static bool
yttrium_venus_cmd_batch_buffer_index_conflict(
   const struct yttrium_venus *venus,
   const struct yttrium_venus_cmd_batch_buffer_index *index,
   const struct yttrium_venus_cmd_batch_buffer_footprint *footprint,
   enum yttrium_venus_cmd_batch_buffer_index_kind kind,
   const struct yttrium_venus_cmd_batch_footprint **seen)
{
   if (seen)
      *seen = NULL;
   if (!footprint->buffer || !footprint->size)
      return false;

   const VkDeviceSize end = footprint->offset + footprint->size;
   if (end < footprint->offset)
      return true;

   if (index->buffer == footprint->buffer &&
       index->generation == footprint->generation &&
       footprint->offset >= index->end)
      return false;

   for (uint32_t i = 0; i < venus->cmd_batch_footprint_count; i++) {
      const struct yttrium_venus_cmd_batch_footprint *old =
         &venus->cmd_batch_footprints[i];
      const struct yttrium_venus_cmd_batch_buffer_footprint *old_footprint =
         yttrium_venus_cmd_batch_buffer_footprint(old, kind);
      if (!old_footprint)
         return true;
      if (yttrium_venus_cmd_batch_buffer_footprint_overlaps(
             footprint, old_footprint)) {
         if (seen)
            *seen = old;
         return true;
      }
   }

   return false;
}

static void
yttrium_venus_cmd_batch_update_buffer_index(
   struct yttrium_venus_cmd_batch_buffer_index *index,
   const struct yttrium_venus_cmd_batch_buffer_footprint *footprint)
{
   if (!footprint->buffer || !footprint->size)
      return;

   const VkDeviceSize end = footprint->offset + footprint->size;
   if (!index->buffer) {
      index->buffer = footprint->buffer;
      index->generation = footprint->generation;
      index->end = end;
   } else if (index->buffer == footprint->buffer &&
              index->generation == footprint->generation) {
      index->end = MAX2(index->end, end);
   }
}

bool
yttrium_venus_cmd_batch_deferred_image_role_conflict(
   struct yttrium_venus *venus,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   bool depth_read_only)
{
   struct yttrium_venus_cmd_batch_footprint new_footprint;

   if (!venus || !venus->display_copy_batch_recording ||
       !venus->cmd_batch_deferred_draw_count)
      return false;

   memset(&new_footprint, 0, sizeof(new_footprint));
   yttrium_venus_cmd_batch_collect_image_roles(
      sampled_images, sampled_image_count, color_resources,
      color_resource_count, depth_resource, depth_read_only, &new_footprint);

   if (!yttrium_venus_cmd_batch_ensure_footprint_index(venus))
      return true;
   return yttrium_venus_cmd_batch_indexed_image_role_conflict(
      venus, &new_footprint, NULL);
}

bool
yttrium_venus_pipeline_ubo_footprint(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_ubo_upload *uploads,
   uint32_t upload_count,
   struct yttrium_venus_cmd_batch_buffer_footprint *footprint)
{
   memset(footprint, 0, sizeof(*footprint));
   if (!upload_count)
      return true;

   VkDeviceSize begin = UINT64_MAX;
   VkDeviceSize end = 0;
   for (uint32_t i = 0; i < upload_count; i++) {
      struct yttrium_venus_ubo_slot *slot = NULL;
      for (uint32_t j = 0; j < pipeline->ubo_count; j++) {
         if (pipeline->ubos[j].binding == uploads[i].binding &&
             pipeline->ubos[j].array_element == uploads[i].array_element) {
            slot = &pipeline->ubos[j];
            break;
         }
      }
      if (!slot || !slot->buffer || !slot->size)
         return false;
      if (slot->upload_reused)
         continue;

      const VkDeviceSize slot_end = slot->offset + slot->size;
      if (slot_end < slot->offset)
         return false;

      begin = MIN2(begin, slot->offset);
      end = MAX2(end, slot_end);
   }

   if (begin == UINT64_MAX || end <= begin)
      return true;

   footprint->buffer = venus->ubo_upload_buffer;
   footprint->offset = begin;
   footprint->size = end - begin;
   footprint->generation = venus->ubo_upload_buffer_generation;
   return true;
}

bool
yttrium_venus_cmd_batch_record_footprint(
   struct yttrium_venus *venus,
   const struct yttrium_venus_cmd_batch_footprint *footprint)
{
   if (!venus || !venus->display_copy_batch_recording ||
       !venus->cmd_batch_async_submit || !footprint)
      return true;

   if (!yttrium_venus_cmd_batch_ensure_footprint_index(venus)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch footprint index allocation failed\n");
      return false;
   }

   const struct yttrium_venus_cmd_batch_footprint *seen = NULL;
   if (footprint->descriptor_set &&
       yttrium_venus_cmd_batch_index_lookup(
          venus, venus->cmd_batch_descriptor_sets,
          YTTRIUM_VENUS_HANDLE_TO_U64(footprint->descriptor_set), &seen)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch transient descriptor conflict set=0x%llx new_pipeline=%llu old_pipeline=%llu new_res=%u old_res=%u\n",
                  (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                     footprint->descriptor_set),
                  (unsigned long long)footprint->pipeline_id,
                  (unsigned long long)(seen ? seen->pipeline_id : 0),
                  footprint->resource_id,
                  seen ? seen->resource_id : 0);
      return false;
   }

   if (yttrium_venus_cmd_batch_indexed_image_role_conflict(
          venus, footprint, &seen)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch deferred image role conflict new_pipeline=%llu old_pipeline=%llu new_res=%u old_res=%u\n",
                  (unsigned long long)footprint->pipeline_id,
                  (unsigned long long)(seen ? seen->pipeline_id : 0),
                  footprint->resource_id,
                  seen ? seen->resource_id : 0);
      return false;
   }

   if (yttrium_venus_cmd_batch_buffer_index_conflict(
          venus, &venus->cmd_batch_ubo_footprint, &footprint->ubo,
          YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_UBO, &seen)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch transient ubo range conflict buffer=0x%llx gen=%llu new=0x%llx+0x%llx old=0x%llx+0x%llx new_pipeline=%llu old_pipeline=%llu\n",
                  (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                     footprint->ubo.buffer),
                  (unsigned long long)footprint->ubo.generation,
                  (unsigned long long)footprint->ubo.offset,
                  (unsigned long long)footprint->ubo.size,
                  (unsigned long long)(seen ? seen->ubo.offset : 0),
                  (unsigned long long)(seen ? seen->ubo.size : 0),
                  (unsigned long long)footprint->pipeline_id,
                  (unsigned long long)(seen ? seen->pipeline_id : 0));
      return false;
   }

   if (yttrium_venus_cmd_batch_buffer_index_conflict(
          venus, &venus->cmd_batch_vertex_footprint, &footprint->vertex,
          YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_VERTEX, &seen)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch transient vertex range conflict buffer=0x%llx gen=%llu new=0x%llx+0x%llx old=0x%llx+0x%llx new_pipeline=%llu old_pipeline=%llu\n",
                  (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                     footprint->vertex.buffer),
                  (unsigned long long)footprint->vertex.generation,
                  (unsigned long long)footprint->vertex.offset,
                  (unsigned long long)footprint->vertex.size,
                  (unsigned long long)(seen ? seen->vertex.offset : 0),
                  (unsigned long long)(seen ? seen->vertex.size : 0),
                  (unsigned long long)footprint->pipeline_id,
                  (unsigned long long)(seen ? seen->pipeline_id : 0));
      return false;
   }

   if (yttrium_venus_cmd_batch_buffer_index_conflict(
          venus, &venus->cmd_batch_index_footprint, &footprint->index,
          YTTRIUM_VENUS_CMD_BATCH_BUFFER_INDEX_INDEX, &seen)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch transient index range conflict buffer=0x%llx gen=%llu new=0x%llx+0x%llx old=0x%llx+0x%llx new_pipeline=%llu old_pipeline=%llu\n",
                  (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                     footprint->index.buffer),
                  (unsigned long long)footprint->index.generation,
                  (unsigned long long)footprint->index.offset,
                  (unsigned long long)footprint->index.size,
                  (unsigned long long)(seen ? seen->index.offset : 0),
                  (unsigned long long)(seen ? seen->index.size : 0),
                  (unsigned long long)footprint->pipeline_id,
                  (unsigned long long)(seen ? seen->pipeline_id : 0));
      return false;
   }

   if (!yttrium_venus_cmd_batch_ensure_footprint_capacity(venus))
      return false;

   if (!yttrium_venus_cmd_batch_index_footprint(venus, footprint)) {
      YTTRIUM_LOG("yttrium: Venus cmd batch footprint index update failed count=%u\n",
                  venus->cmd_batch_footprint_count);
      return false;
   }

   venus->cmd_batch_footprints[venus->cmd_batch_footprint_count] = *footprint;
   yttrium_venus_cmd_batch_update_buffer_index(
      &venus->cmd_batch_ubo_footprint, &footprint->ubo);
   yttrium_venus_cmd_batch_update_buffer_index(
      &venus->cmd_batch_vertex_footprint, &footprint->vertex);
   yttrium_venus_cmd_batch_update_buffer_index(
      &venus->cmd_batch_index_footprint, &footprint->index);
   venus->cmd_batch_footprint_count++;
   return true;
}

static void
yttrium_venus_cmd_batch_clear_pending_refs(struct yttrium_venus *venus)
{
   const uint32_t resource_count =
      venus->cmd_batch_pending_resource_count;
   const uint32_t pipeline_count =
      venus->cmd_batch_pending_pipeline_count;

   for (uint32_t i = 0; i < resource_count; i++)
      pipe_resource_reference(&venus->cmd_batch_pending_resource_refs[i],
                              NULL);
   memset(venus->cmd_batch_pending_resources, 0,
          resource_count * sizeof(venus->cmd_batch_pending_resources[0]));
   memset(venus->cmd_batch_pending_resource_refs, 0,
          resource_count * sizeof(venus->cmd_batch_pending_resource_refs[0]));
   memset(venus->cmd_batch_pending_pipelines, 0,
          pipeline_count * sizeof(venus->cmd_batch_pending_pipelines[0]));
   venus->cmd_batch_pending_resource_count = 0;
   venus->cmd_batch_pending_pipeline_count = 0;
}

void
yttrium_venus_cmd_batch_clear_transient_upload_state(
   struct yttrium_venus *venus)
{
   venus->cmd_batch_ubo_watermark = 0;
   venus->cmd_batch_vertex_watermark = 0;
   venus->cmd_batch_index_watermark = 0;
   yttrium_venus_cmd_batch_clear_uploads(venus);
   yttrium_venus_cmd_batch_clear_upload_barriers(venus);
   yttrium_venus_cmd_batch_clear_image_barriers(venus);
}

void
yttrium_venus_cmd_batch_clear_state(struct yttrium_venus *venus)
{
   venus->display_copy_batch_recording = false;
   venus->cmd_batch_native_draw_only = false;
   venus->cmd_batch_async_submit = false;
   venus->cmd_batch_has_transfer_ops = false;
   venus->display_copy_batch_count = 0;
   venus->cmd_batch = NULL;
   venus->cmd_batch_ubo_arena = NULL;
   venus->command_buffer = VK_NULL_HANDLE;
   yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
   yttrium_venus_cmd_batch_clear_pending_refs(venus);
   yttrium_venus_cmd_batch_clear_footprints(venus);
   yttrium_venus_cmd_batch_clear_deferred_draws(venus);
   yttrium_venus_cmd_batch_destroy_descriptor_pool(
      venus, &venus->cmd_batch_descriptor_pool);
   yttrium_venus_cmd_batch_destroy_transients(venus);
}

void
yttrium_venus_abort_command_batch(struct yttrium_venus *venus,
                                  const char *label)
{
   if (!venus || !venus->display_copy_batch_recording)
      return;

   YTTRIUM_LOG("yttrium: Venus aborting deferred command batch label=%s ops=%u\n",
               label ? label : "<unknown>",
               venus->display_copy_batch_count);
   if (venus->cmd_batch)
      venus->command_buffer = venus->cmd_batch->command_buffer;
   vn_async_vkResetCommandBuffer(&venus->vn_ring, venus->command_buffer, 0);
   if (venus->cmd_batch && !venus->cmd_batch->busy)
      yttrium_venus_destroy_retired_resources(
         venus, &venus->cmd_batch->retired_resources);
   yttrium_venus_restore_draw_mirror_updates(
      venus->cmd_batch_draw_mirror_updates,
      &venus->cmd_batch_draw_mirror_update_count);
   yttrium_venus_cmd_batch_clear_state(venus);
}

bool
yttrium_venus_flush_command_batch(struct yttrium_venus *venus,
                                  const char *label);

void
yttrium_venus_cancel_command_batch_setup_failure(struct yttrium_venus *venus,
                                                 const char *label)
{
   if (!venus || !venus->display_copy_batch_recording)
      return;

   if (venus->display_copy_batch_count) {
      yttrium_venus_flush_command_batch(venus, label);
      return;
   }

   yttrium_venus_abort_command_batch(venus, label);
}

/* Every Venus2 command batch owns a slot until mapped completion retires it.
 * Batching policy controls how many operations are coalesced, never whether
 * command-buffer ownership is asynchronous.  The state was historically
 * named for display copies, hence the remaining field names. */
bool
yttrium_venus_flush_command_batch(struct yttrium_venus *venus,
                                  const char *label)
{
   if (!venus || !venus->display_copy_batch_recording)
      return true;

   const bool async_submit = venus->cmd_batch_async_submit;
   const uint32_t op_count = venus->display_copy_batch_count;
   const uint32_t resource_count = venus->cmd_batch_pending_resource_count;
   const uint32_t pipeline_count = venus->cmd_batch_pending_pipeline_count;
   const uint32_t transient_count = venus->cmd_batch_transient_count;
   struct yttrium_venus_batch *batch = venus->cmd_batch;
   if (batch)
      venus->command_buffer = batch->command_buffer;
   bool ok = yttrium_venus_cmd_batch_emit_deferred_draws(venus, label);
   if (ok)
      ok = yttrium_venus_end_command_buffer(venus, label);
   /* A failed emission may leave the buffer recording.  Reset it before
    * discarding the batch; unlike completed recordings, it cannot rely on
    * the implicit reset when a free slot is begun again.
    */
   if (!ok)
      vn_async_vkResetCommandBuffer(&venus->vn_ring, venus->command_buffer, 0);
   if (ok) {
      const uint32_t submitted_live_batch_count =
         venus->live_batch_count +
         (async_submit && batch && !batch->busy ? 1 : 0);
      yttrium_trace_cmd_batch_submit(async_submit ? 1 : 0,
                                     venus->cmd_batch_native_draw_only ? 1 : 0,
                                     op_count, resource_count, pipeline_count,
                                     transient_count, venus->batch_count,
                                     submitted_live_batch_count,
                                     MAX2(venus->peak_live_batch_count,
                                          submitted_live_batch_count),
                                     venus->pending_submit_count,
                                     venus->group_queue_submit_size,
                                     (uint64_t)venus->batch_count *
                                        sizeof(*batch) +
                                        (uint64_t)venus->batch_capacity *
                                        sizeof(*venus->batches) +
                                        (venus->pending_submit_batches ?
                                           (uint64_t)
                                              venus->group_queue_submit_size *
                                              sizeof(*venus->pending_submit_batches) :
                                           0),
                                     venus->ubo_arena_bytes,
                                     venus->peak_ubo_arena_bytes,
                                     venus->draw_backing_pool_bytes,
                                     venus->peak_draw_backing_pool_bytes,
                                     label);
   }
   if (ok && async_submit) {
      if (!batch) {
         YTTRIUM_LOG("yttrium: Venus async command batch missing in-flight batch label=%s\n",
                     label ? label : "<unknown>");
         ok = false;
      } else {
         memcpy(batch->resources, venus->cmd_batch_pending_resources,
                venus->cmd_batch_pending_resource_count *
                sizeof(batch->resources[0]));
         memcpy(batch->resource_refs, venus->cmd_batch_pending_resource_refs,
                venus->cmd_batch_pending_resource_count *
                sizeof(batch->resource_refs[0]));
         batch->resource_count = venus->cmd_batch_pending_resource_count;
         memset(venus->cmd_batch_pending_resources, 0,
                venus->cmd_batch_pending_resource_count *
                sizeof(venus->cmd_batch_pending_resources[0]));
         memset(venus->cmd_batch_pending_resource_refs, 0,
                venus->cmd_batch_pending_resource_count *
                sizeof(venus->cmd_batch_pending_resource_refs[0]));
         venus->cmd_batch_pending_resource_count = 0;
         memcpy(batch->pipelines, venus->cmd_batch_pending_pipelines,
                venus->cmd_batch_pending_pipeline_count *
                sizeof(batch->pipelines[0]));
         batch->pipeline_count = venus->cmd_batch_pending_pipeline_count;
         memset(venus->cmd_batch_pending_pipelines, 0,
                venus->cmd_batch_pending_pipeline_count *
                sizeof(venus->cmd_batch_pending_pipelines[0]));
         venus->cmd_batch_pending_pipeline_count = 0;
         memcpy(batch->transients, venus->cmd_batch_transients,
                venus->cmd_batch_transient_count *
                sizeof(batch->transients[0]));
         batch->transient_count = venus->cmd_batch_transient_count;
         memset(venus->cmd_batch_transients, 0,
                venus->cmd_batch_transient_count *
                sizeof(venus->cmd_batch_transients[0]));
         venus->cmd_batch_transient_count = 0;
         batch->descriptor_pool = venus->cmd_batch_descriptor_pool;
         memset(&venus->cmd_batch_descriptor_pool, 0,
                sizeof(venus->cmd_batch_descriptor_pool));
         batch->ubo_arena = venus->cmd_batch_ubo_arena;
         if (batch->ubo_arena)
            batch->ubo_arena->batch_refcount++;
         ok = yttrium_venus_move_draw_mirror_updates_to_batch(venus, batch);
         if (ok)
            ok = yttrium_venus_submit_batch_async(venus, batch, op_count,
                                                   label);
         /* A failed ring notify can be reported after the command buffer was
          * submitted and ownership moved to the busy batch.  Keep all batch
          * references alive until that batch is retired or destroyed. */
         if (!ok && !batch->busy) {
            yttrium_venus_restore_draw_mirror_updates(
               batch->draw_mirror_updates,
               &batch->draw_mirror_update_count);
            yttrium_venus_batch_release_ubo_arena(venus, batch);
            yttrium_venus_cmd_batch_destroy_descriptor_pool(
               venus, &batch->descriptor_pool);
            yttrium_venus_batch_clear_refs(batch);
         }
      }
   } else if (ok) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 command batch rejected owner=venus2 reason=synchronous_submission_invariant label=%s\n",
                   label ? label : "<unknown>");
      ok = false;
   }

   if (!ok && venus->cmd_batch_draw_mirror_update_count) {
      yttrium_venus_restore_draw_mirror_updates(
         venus->cmd_batch_draw_mirror_updates,
         &venus->cmd_batch_draw_mirror_update_count);
   }

   yttrium_venus_cmd_batch_clear_state(venus);
   return ok;
}

/* Allocate stable backing storage for a batched op's transient handles. */
bool
yttrium_venus_cmd_batch_alloc_transient(struct yttrium_venus *venus,
                                        uint32_t *out_index)
{
   if (out_index)
      *out_index = UINT32_MAX;

   if (!venus || !out_index)
      return false;

   if (venus->cmd_batch_transient_count >=
       ARRAY_SIZE(venus->cmd_batch_transients)) {
      /* Transient-allocating display/transfer ops still use the small
       * display-copy batch limit.  Native draws may extend a mixed batch to
       * the draw limit, but they do not allocate these transient handles. */
      YTTRIUM_LOG("yttrium: Venus cmd batch transient overflow\n");
      return false;
   }

   const uint32_t i = venus->cmd_batch_transient_count++;
   memset(&venus->cmd_batch_transients[i], 0,
          sizeof(venus->cmd_batch_transients[i]));
   yttrium_venus_init_object(venus,
                             &venus->cmd_batch_transients[i].view_obj);
   yttrium_venus_init_object(venus,
                             &venus->cmd_batch_transients[i].render_pass_obj);
   yttrium_venus_init_object(venus,
                             &venus->cmd_batch_transients[i].framebuffer_obj);
   venus->cmd_batch_transients[i].view =
      YTTRIUM_VENUS_HANDLE(VkImageView,
                            &venus->cmd_batch_transients[i].view_obj);
   venus->cmd_batch_transients[i].render_pass =
      YTTRIUM_VENUS_HANDLE(VkRenderPass,
                            &venus->cmd_batch_transients[i].render_pass_obj);
   venus->cmd_batch_transients[i].framebuffer =
      YTTRIUM_VENUS_HANDLE(VkFramebuffer,
                            &venus->cmd_batch_transients[i].framebuffer_obj);
   *out_index = i;
   return true;
}

void
yttrium_venus_cmd_batch_release_transient(struct yttrium_venus *venus,
                                          uint32_t index)
{
   if (!venus || index == UINT32_MAX ||
       index >= venus->cmd_batch_transient_count)
      return;

   if (venus->cmd_batch_transients[index].framebuffer_created)
      vn_async_vkDestroyFramebuffer(&venus->vn_ring, venus->device_handle,
                                    venus->cmd_batch_transients[index].framebuffer,
                                    NULL);
   if (venus->cmd_batch_transients[index].render_pass_created)
      vn_async_vkDestroyRenderPass(&venus->vn_ring, venus->device_handle,
                                   venus->cmd_batch_transients[index].render_pass,
                                   NULL);
   if (venus->cmd_batch_transients[index].view_created)
      vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                  venus->cmd_batch_transients[index].view, NULL);
   memset(&venus->cmd_batch_transients[index], 0,
          sizeof(venus->cmd_batch_transients[index]));
   if (index + 1 == venus->cmd_batch_transient_count)
      venus->cmd_batch_transient_count--;
}

/* Called by a batched op after it finishes recording (in place of its own
 * end/submit).  Bounds the batch so the command buffer and transient list stay
 * small by flushing once the op count reaches the limit. */
bool
yttrium_venus_cmd_batch_after_record(struct yttrium_venus *venus,
                                     const char *label)
{
   const uint32_t limit =
      (venus->cmd_batch_native_draw_only ||
       (venus->cmd_batch_async_submit && yttrium_venus_async_batch_enabled())) ?
      yttrium_venus_native_draw_batch_limit() :
      YTTRIUM_VENUS_DISPLAY_COPY_BATCH_LIMIT;
   const uint32_t count = ++venus->display_copy_batch_count;
   if (count >= limit)
      return yttrium_venus_flush_command_batch(venus, label);
   return true;
}

bool
yttrium_venus_cmd_batch_after_native_draw_record(struct yttrium_venus *venus,
                                                const char *label)
{
   const uint32_t count = ++venus->display_copy_batch_count;
   if (count >= yttrium_venus_native_draw_batch_limit())
      return yttrium_venus_flush_command_batch(venus, label);
   return true;
}

bool
yttrium_venus_cmd_batch_submit_after_record(struct yttrium_venus *venus,
                                            const char *label)
{
   if (!yttrium_venus_cmd_batch_after_record(venus, label))
      return false;
   if (venus && venus->display_copy_batch_recording &&
       venus->cmd_batch_async_submit)
      return yttrium_venus_flush_command_batch(venus, label);
   return true;
}

bool
yttrium_venus2_flush(struct yttrium_venus *venus)
{
   const char *label = yttrium_venus_current_flush_label();
   if (!label)
      label = "explicit Venus flush";

   bool ok = yttrium_venus_flush_command_batch(venus, label);
   if (!yttrium_venus_drain_batches(venus, label))
      ok = false;
   return ok;
}

bool
yttrium_venus2_flush_async(struct yttrium_venus *venus)
{
   const char *label = yttrium_venus_current_flush_label();
   if (!label)
      label = "explicit Venus async flush";

   if (!yttrium_venus_flush_command_batch(venus, label))
      return false;
   return yttrium_venus_flush_pending_submits(venus, label);
}

static bool
yttrium_venus_publish_pending_display(struct yttrium_venus *venus,
                                      const char *label)
{
   if (!venus || !venus->present_publish_pending)
      return true;

   struct yttrium_venus_batch *batch = venus->present_publish_batch;
   const uint32_t allocation = venus->present_publish_allocation;
   const uint32_t scanout_id = venus->present_publish_scanout_id;
   const uint64_t submit_order = venus->present_publish_submit_order;
   bool ok = true;

   if (batch && batch->busy && batch->submit_order == submit_order &&
       !yttrium_venus_wait_batch_for_present(venus, batch, label)) {
      YTTRIUM_WARN("yttrium: ERROR: display publication failed owner=venus2 reason=frame-completion-wait-failed allocation=0x%lx submit_order=%llu label=%s\n",
                   (unsigned long)allocation,
                   (unsigned long long)submit_order, label);
      ok = false;
   } else {
      const NTSTATUS status = yttrium_venus_escape_publish_display(
         venus, allocation, scanout_id);
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: ERROR: display publication failed owner=venus2 reason=set-scanout-blob-escape-rejected status=0x%lx allocation=0x%lx scanout=%u submit_order=%llu label=%s\n",
                      (unsigned long)status, (unsigned long)allocation,
                      scanout_id, (unsigned long long)submit_order, label);
         ok = false;
      }
   }

   venus->present_publish_pending = false;
   venus->present_publish_allocation = 0;
   venus->present_publish_scanout_id = 0;
   venus->present_publish_batch = NULL;
   venus->present_publish_submit_order = 0;
   return ok;
}

bool
yttrium_venus2_flush_async_present_publish(
   struct yttrium_venus *venus,
   const struct yttrium_venus_present_publication *publication)
{
   const char *label = yttrium_venus_current_flush_label();
   if (!label)
      label = "fullscreen Present publication";

   if (!venus || !publication || !publication->requested ||
       !publication->allocation) {
      YTTRIUM_WARN("yttrium: ERROR: display publication failed owner=venus2 reason=invalid-request label=%s\n",
                   label);
      return false;
   }

   if (!yttrium_venus_flush_command_batch(venus, label) ||
       !yttrium_venus_flush_pending_submits(venus, label))
      return false;

   /*
    * Publish the previous frame after its exact batch completes, then retain
    * this frame.  Flushing this frame before waiting keeps useful GPU work in
    * flight instead of serializing the producer against the backend tail.
    * A recycled batch (pointer matches but submit order does not) has already
    * retired and therefore needs no further wait.
    */
   const bool previous_ok =
      yttrium_venus_publish_pending_display(venus, label);
   struct yttrium_venus_batch *batch =
      yttrium_venus_latest_busy_batch(venus);

   venus->present_publish_allocation = publication->allocation;
   venus->present_publish_scanout_id = publication->scanout_id;
   venus->present_publish_batch = batch;
   venus->present_publish_submit_order = batch ? batch->submit_order : 0;
   venus->present_publish_pending = true;
   return previous_ok;
}

static void
yttrium_venus_batch_clear_refs(struct yttrium_venus_batch *batch)
{
   if (!batch)
      return;

   const uint32_t resource_count = batch->resource_count;
   const uint32_t pipeline_count = batch->pipeline_count;

   for (uint32_t i = 0; i < resource_count; i++)
      pipe_resource_reference(&batch->resource_refs[i], NULL);
   memset(batch->resources, 0,
          resource_count * sizeof(batch->resources[0]));
   memset(batch->resource_refs, 0,
          resource_count * sizeof(batch->resource_refs[0]));
   batch->resource_count = 0;
   memset(batch->pipelines, 0,
          pipeline_count * sizeof(batch->pipelines[0]));
   batch->pipeline_count = 0;
}

static void
yttrium_venus_recycle_ubo_arena(struct yttrium_venus *venus,
                                struct yttrium_venus_ubo_arena *arena)
{
   if (!venus || !arena)
      return;

   struct yttrium_venus_ubo_arena **link = &venus->retired_ubo_arenas;
   while (*link) {
      if (*link == arena) {
         *link = arena->next;
         break;
      }
      link = &(*link)->next;
   }
   if (venus->ubo_upload_arena == arena) {
      venus->ubo_upload_arena = NULL;
      venus->ubo_upload_buffer = VK_NULL_HANDLE;
      venus->ubo_upload_memory = VK_NULL_HANDLE;
      venus->ubo_upload_buffer_size = 0;
      venus->ubo_upload_roll_base = 0;
   }

   arena->retired = false;
   arena->roll_base = 0;
   arena->next = venus->free_ubo_arenas;
   venus->free_ubo_arenas = arena;
}

static void
yttrium_venus_batch_release_ubo_arena(
   struct yttrium_venus *venus,
   struct yttrium_venus_batch *batch)
{
   struct yttrium_venus_ubo_arena *arena = batch ? batch->ubo_arena : NULL;
   if (!arena)
      return;

   batch->ubo_arena = NULL;
   if (!arena->batch_refcount) {
      YTTRIUM_WARN("yttrium: ERROR: UBO arena batch ownership underflow owner=venus2 generation=%llu\n",
                   (unsigned long long)arena->generation);
      return;
   }
   arena->batch_refcount--;
   if (arena->retired && !arena->batch_refcount)
      yttrium_venus_recycle_ubo_arena(venus, arena);
}

void
yttrium_venus_retire_ubo_arena(
   struct yttrium_venus *venus,
   struct yttrium_venus_ubo_arena *arena)
{
   if (!venus || !arena || arena->retired)
      return;

   if (venus->ubo_upload_arena == arena) {
      venus->ubo_upload_arena = NULL;
      venus->ubo_upload_buffer = VK_NULL_HANDLE;
      venus->ubo_upload_memory = VK_NULL_HANDLE;
      venus->ubo_upload_buffer_size = 0;
      venus->ubo_upload_roll_base = 0;
   }
   arena->retired = true;
   arena->next = venus->retired_ubo_arenas;
   venus->retired_ubo_arenas = arena;
   if (!arena->batch_refcount)
      yttrium_venus_recycle_ubo_arena(venus, arena);
}

void
yttrium_venus_destroy_ubo_arenas(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   struct yttrium_venus_ubo_arena *arena = venus->retired_ubo_arenas;
   while (arena) {
      struct yttrium_venus_ubo_arena *next = arena->next;
      arena->next = NULL;
      yttrium_venus_unmap_memory(venus, &arena->mapping);
      if (arena->buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  arena->buffer, NULL);
      if (arena->memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               arena->memory, NULL);
      FREE(arena);
      arena = next;
   }
   venus->retired_ubo_arenas = NULL;

   arena = venus->free_ubo_arenas;
   while (arena) {
      struct yttrium_venus_ubo_arena *next = arena->next;
      yttrium_venus_unmap_memory(venus, &arena->mapping);
      if (arena->buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  arena->buffer, NULL);
      if (arena->memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               arena->memory, NULL);
      FREE(arena);
      arena = next;
   }
   venus->free_ubo_arenas = NULL;

   arena = venus->ubo_upload_arena;
   if (arena) {
      yttrium_venus_unmap_memory(venus, &arena->mapping);
      if (arena->buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  arena->buffer, NULL);
      if (arena->memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               arena->memory, NULL);
      FREE(arena);
   }
   venus->ubo_upload_arena = NULL;
   venus->ubo_upload_buffer = VK_NULL_HANDLE;
   venus->ubo_upload_memory = VK_NULL_HANDLE;
   venus->ubo_upload_buffer_size = 0;
   venus->ubo_upload_roll_base = 0;
   venus->ubo_arena_bytes = 0;
}

static volatile LONG *
yttrium_venus_batch_feedback_slot(struct yttrium_venus *venus,
                                  uint32_t index)
{
   if (!venus || !venus->batch_feedback_initialized ||
       !venus->batch_feedback_mapping.map ||
       index >= venus->batch_capacity)
      return NULL;

   const VkDeviceSize offset =
      (VkDeviceSize)index * YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE;
   if (offset + YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE >
       venus->batch_feedback_buffer_size)
      return NULL;

   return (volatile LONG *)((uint8_t *)venus->batch_feedback_mapping.map +
                            offset);
}

static bool
yttrium_venus_batch_feedback_store(struct yttrium_venus *venus,
                                   volatile LONG *slot,
                                   VkResult value,
                                   const char *reason)
{
   if (!venus || !slot || venus->failed ||
       gdikmt_device_get_reset_status(venus->device) != PIPE_NO_RESET ||
       !yttrium_venus_ring_probe(venus, reason))
      return false;

   *slot = (LONG)value;
   MemoryBarrier();
   return true;
}

static void
yttrium_venus_destroy_batch_feedback_buffer(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   yttrium_venus_unmap_memory(venus, &venus->batch_feedback_mapping);
   if (venus->batch_feedback_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               venus->batch_feedback_buffer, NULL);
   if (venus->batch_feedback_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            venus->batch_feedback_memory, NULL);

   memset(&venus->batch_feedback_buffer_obj, 0,
          sizeof(venus->batch_feedback_buffer_obj));
   memset(&venus->batch_feedback_memory_obj, 0,
          sizeof(venus->batch_feedback_memory_obj));
   venus->batch_feedback_buffer = VK_NULL_HANDLE;
   venus->batch_feedback_memory = VK_NULL_HANDLE;
   venus->batch_feedback_buffer_size = 0;
   venus->batch_feedback_initialized = false;
}

static bool
yttrium_venus_ensure_batch_feedback_buffer(struct yttrium_venus *venus)
{
   if (!yttrium_venus_batch_fence_feedback_enabled())
      return true;
   if (venus && venus->batch_feedback_initialized)
      return true;
   if (!venus || !venus->batch_capacity ||
       venus->batch_capacity > YTTRIUM_VENUS_BATCH_COUNT_MAX) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=invalid_batch_capacity capacity=%u\n",
                   venus ? venus->batch_capacity : 0);
      return false;
   }

   const VkDeviceSize buffer_size =
      (VkDeviceSize)venus->batch_capacity *
      YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE;
   yttrium_venus_init_object(venus, &venus->batch_feedback_buffer_obj);
   venus->batch_feedback_buffer =
      YTTRIUM_VENUS_HANDLE(VkBuffer, &venus->batch_feedback_buffer_obj);
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkResult result =
      vn_call_vkCreateBuffer(&venus->vn_ring, venus->device_handle,
                             &buffer_info, NULL,
                             &venus->batch_feedback_buffer);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=vkCreateBuffer result=%d size=%llu\n",
                   result, (unsigned long long)buffer_size);
      memset(&venus->batch_feedback_buffer_obj, 0,
             sizeof(venus->batch_feedback_buffer_obj));
      venus->batch_feedback_buffer = VK_NULL_HANDLE;
      return false;
   }

   VkMemoryRequirements requirements;
   memset(&requirements, 0, sizeof(requirements));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         venus->batch_feedback_buffer,
                                         &requirements);
   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(
         venus, requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=no_host_visible_coherent_memory bits=0x%x size=%llu\n",
                   requirements.memoryTypeBits,
                   (unsigned long long)buffer_size);
      yttrium_venus_destroy_batch_feedback_buffer(venus);
      return false;
   }

   yttrium_venus_init_object(venus, &venus->batch_feedback_memory_obj);
   venus->batch_feedback_memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory,
                           &venus->batch_feedback_memory_obj);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = MAX2(requirements.size, buffer_size),
      .memoryTypeIndex = memory_type_index,
   };
   result = vn_call_vkAllocateMemory(&venus->vn_ring,
                                     venus->device_handle,
                                     &memory_info, NULL,
                                     &venus->batch_feedback_memory);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=vkAllocateMemory result=%d type=%u size=%llu\n",
                   result, memory_type_index,
                   (unsigned long long)memory_info.allocationSize);
      memset(&venus->batch_feedback_memory_obj, 0,
             sizeof(venus->batch_feedback_memory_obj));
      venus->batch_feedback_memory = VK_NULL_HANDLE;
      yttrium_venus_destroy_batch_feedback_buffer(venus);
      return false;
   }

   result = vn_call_vkBindBufferMemory(&venus->vn_ring,
                                       venus->device_handle,
                                       venus->batch_feedback_buffer,
                                       venus->batch_feedback_memory, 0);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=vkBindBufferMemory result=%d memory_id=%llu\n",
                   result,
                   (unsigned long long)
                      venus->batch_feedback_memory_obj.id);
      yttrium_venus_destroy_batch_feedback_buffer(venus);
      return false;
   }

   if (!yttrium_venus_map_memory(
          venus, venus->batch_feedback_memory_obj.id,
          memory_info.allocationSize, &venus->batch_feedback_mapping)) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=blob_map_failed memory_id=%llu size=%llu\n",
                   (unsigned long long)
                      venus->batch_feedback_memory_obj.id,
                   (unsigned long long)memory_info.allocationSize);
      yttrium_venus_destroy_batch_feedback_buffer(venus);
      return false;
   }

   venus->batch_feedback_buffer_size = buffer_size;
   venus->batch_feedback_initialized = true;
   for (uint32_t i = 0; i < venus->batch_capacity; i++) {
      volatile LONG *slot = yttrium_venus_batch_feedback_slot(venus, i);
      if (!slot) {
         YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=invalid_mapped_slot index=%u capacity=%u\n",
                      i, venus->batch_capacity);
         yttrium_venus_destroy_batch_feedback_buffer(venus);
         return false;
      }
      if (!yttrium_venus_batch_feedback_store(
             venus, slot, VK_NOT_READY, "batch-feedback-init-fault")) {
         yttrium_venus_destroy_batch_feedback_buffer(venus);
         return false;
      }
   }

   return true;
}

static bool
yttrium_venus_record_batch_feedback_command(
   struct yttrium_venus *venus,
   struct yttrium_venus_batch *batch,
   uint32_t index)
{
   if (!yttrium_venus_batch_fence_feedback_enabled())
      return true;
   if (batch && batch->feedback_command_buffer)
      return true;
   if (!venus || !batch || !venus->batch_feedback_initialized ||
       index >= venus->batch_capacity) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback setup failed owner=venus2 reason=invalid_command_slot slot=%u capacity=%u\n",
                   index, venus ? venus->batch_capacity : 0);
      return false;
   }

   yttrium_venus_init_object(venus,
                             &batch->feedback_command_buffer_obj);
   batch->feedback_command_buffer =
      YTTRIUM_VENUS_HANDLE(VkCommandBuffer,
                           &batch->feedback_command_buffer_obj);
   const VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = venus->command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateCommandBuffers(&venus->vn_ring, 0,
                                      venus->device_handle,
                                      &alloc_info,
                                      &batch->feedback_command_buffer,
                                      &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateCommandBuffers-feedback",
          batch->feedback_command_buffer_obj.id)) {
      memset(&batch->feedback_command_buffer_obj, 0,
             sizeof(batch->feedback_command_buffer_obj));
      batch->feedback_command_buffer = VK_NULL_HANDLE;
      return false;
   }

   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   vn_async_vkBeginCommandBuffer(&venus->vn_ring,
                                 batch->feedback_command_buffer,
                                 &begin_info);

   const VkDeviceSize offset =
      (VkDeviceSize)index * YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE;
   const VkMemoryBarrier memory_before = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = 0,
   };
   const VkBufferMemoryBarrier buffer_before = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = venus->batch_feedback_buffer,
      .offset = offset,
      .size = YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE,
   };
   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, batch->feedback_command_buffer,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      1, &memory_before, 1, &buffer_before, 0, NULL);
   vn_async_vkCmdFillBuffer(&venus->vn_ring,
                            batch->feedback_command_buffer,
                            venus->batch_feedback_buffer, offset,
                            YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE,
                            (uint32_t)VK_SUCCESS);
   const VkBufferMemoryBarrier buffer_after = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT |
                       VK_ACCESS_HOST_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = venus->batch_feedback_buffer,
      .offset = offset,
      .size = YTTRIUM_VENUS_BATCH_FEEDBACK_SLOT_SIZE,
   };
   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, batch->feedback_command_buffer,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
      0, NULL, 1, &buffer_after, 0, NULL);
   vn_async_vkEndCommandBuffer(&venus->vn_ring,
                               batch->feedback_command_buffer);

   batch->feedback_index = index;
   batch->completion_feedback_index = UINT32_MAX;
   return true;
}

static bool
yttrium_venus_batch_feedback_reset(struct yttrium_venus *venus,
                                   uint32_t index,
                                   const char *label)
{
   volatile LONG *slot = yttrium_venus_batch_feedback_slot(venus, index);
   if (!slot) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback reset failed owner=venus2 reason=invalid_slot index=%u label=%s\n",
                   index, label ? label : "<unknown>");
      return false;
   }

   return yttrium_venus_batch_feedback_store(
      venus, slot, VK_NOT_READY, "batch-feedback-reset-fault");
}

static VkResult
yttrium_venus_batch_feedback_load(struct yttrium_venus *venus,
                                  volatile LONG *slot)
{
   if (!venus || venus->failed ||
       gdikmt_device_get_reset_status(venus->device) != PIPE_NO_RESET ||
       !yttrium_venus_ring_probe(venus,
                                 "batch-feedback-ring-probe-fault"))
      return VK_ERROR_DEVICE_LOST;

   MemoryBarrier();
   return (VkResult)*slot;
}

static void
yttrium_venus_batch_feedback_backoff(uint32_t *iterations)
{
   const uint32_t iteration = (*iterations)++;

   if (iteration < YTTRIUM_VENUS_BATCH_FEEDBACK_PAUSE_ITERS)
      YieldProcessor();
   else if (iteration < YTTRIUM_VENUS_BATCH_FEEDBACK_SLEEP0_ITERS)
      Sleep(0);
   else
      Sleep(1);
}

static VkResult
yttrium_venus_wait_batch_feedback(struct yttrium_venus *venus,
                                  uint32_t index,
                                  VkFence fence,
                                  const char *label)
{
   volatile LONG *slot = yttrium_venus_batch_feedback_slot(venus, index);
   if (!slot) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback wait failed owner=venus2 reason=invalid_slot index=%u label=%s\n",
                   index, label ? label : "<unknown>");
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   VkResult status = yttrium_venus_batch_feedback_load(venus, slot);
   if (status == VK_SUCCESS)
      return VK_SUCCESS;
   if (status != VK_NOT_READY) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback wait failed owner=venus2 reason=invalid_status status=%d index=%u label=%s\n",
                   status, index, label ? label : "<unknown>");
      return status;
   }

   const uint64_t start_ms = GetTickCount64();
   uint32_t iterations = 0;
   do {
      if (GetTickCount64() - start_ms > YTTRIUM_VENUS_FENCE_WAIT_MS) {
         const VkResult host_fence_status =
            vn_call_vkGetFenceStatus(&venus->vn_ring,
                                     venus->device_handle, fence);
         const VkResult feedback_status_after_host_query =
            yttrium_venus_batch_feedback_load(venus, slot);
         if (host_fence_status == VK_SUCCESS ||
             feedback_status_after_host_query == VK_SUCCESS)
            return VK_SUCCESS;
         YTTRIUM_WARN("yttrium: FATAL: mapped batch-fence feedback timeout owner=venus2 wait_ms=%u index=%u iterations=%u host_fence_status=%d feedback_after_host_query=%d label=%s\n",
                      YTTRIUM_VENUS_FENCE_WAIT_MS, index, iterations,
                      host_fence_status, feedback_status_after_host_query,
                      label ? label : "<unknown>");
         return VK_TIMEOUT;
      }
      yttrium_venus_batch_feedback_backoff(&iterations);
      status = yttrium_venus_batch_feedback_load(venus, slot);
   } while (status == VK_NOT_READY);

   if (status != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback wait failed owner=venus2 reason=terminal_status status=%d index=%u iterations=%u label=%s\n",
                   status, index, iterations,
                   label ? label : "<unknown>");
   }
   return status;
}

static bool
yttrium_venus_queue_host_fence_wait(struct yttrium_venus *venus,
                                    VkFence fence,
                                    struct yttrium_venus_ring_dependency *dependency)
{
   if (dependency)
      *dependency = yttrium_venus_ring_dependency_none();
   if (!venus || !fence || !dependency)
      return false;

   struct vn_ring_submit_command submit;
   vn_submit_vkWaitForFences(&venus->vn_ring, 0, venus->device_handle,
                             1, &fence, VK_TRUE, UINT64_MAX, &submit);
   if (!submit.ring_seqno_valid) {
      YTTRIUM_WARN("yttrium: ERROR: asynchronous host fence retirement failed owner=venus2 reason=invalid_ring_seqno fence_id=%llu\n",
                   (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(fence));
      return false;
   }

   *dependency =
      yttrium_venus_ring_dependency_create(submit.ring_seqno);
   return true;
}

/* Release a batch's guest-side state.  The GPU feedback and its queued host
 * fence retirement have completed, so mapped backings are safe to reuse.
 */
static void
yttrium_venus_mark_batch_live(struct yttrium_venus *venus,
                              struct yttrium_venus_batch *batch)
{
   if (!venus || !batch || batch->busy)
      return;

   batch->busy = true;
   venus->live_batch_count++;
   venus->peak_live_batch_count =
      MAX2(venus->peak_live_batch_count, venus->live_batch_count);
}

static void
yttrium_venus_retire_batch(struct yttrium_venus *venus,
                           struct yttrium_venus_batch *batch)
{
   if (batch->busy && venus->live_batch_count)
      venus->live_batch_count--;
   batch->busy = false;
   batch->pending_submit = false;
   batch->host_retirement = yttrium_venus_ring_dependency_none();
   batch->submit_order = 0;
   batch->completion_order = 0;
   batch->completion_fence = VK_NULL_HANDLE;
   batch->completion_feedback_index = UINT32_MAX;

   yttrium_venus_batch_release_ubo_arena(venus, batch);
   yttrium_venus_retire_resources_after_host_wait(
      venus, &batch->retired_resources);
   yttrium_venus_destroy_cmd_batch_transients(
      venus, batch->transients, &batch->transient_count);
   yttrium_venus_cmd_batch_destroy_descriptor_pool(
      venus, &batch->descriptor_pool);
   yttrium_venus_batch_clear_refs(batch);
}

/*
 * Completion feedback is written by the GPU as the last command in a submit.
 * Reading it is a poll, never a wait.  Since all batches use one Vulkan queue,
 * completion of an order also completes every earlier order.  Queue an
 * asynchronous fence wait so the renderer validates/reset-orders the real
 * fence before reuse.  Keep the completed guest ownership prefix attached to
 * its batches until the ring head proves that the queued wait has completed.
 */
static bool
yttrium_venus_retire_completed_batches(struct yttrium_venus *venus,
                                       const char *label)
{
   uint64_t completed_order = 0;
   VkFence completed_fence = VK_NULL_HANDLE;

   if (!venus)
      return false;

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (batch && batch->busy && batch->host_retirement.valid &&
          yttrium_venus_ring_seqno_complete(
             venus, &batch->host_retirement))
         yttrium_venus_retire_batch(venus, batch);
   }

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch || !batch->busy || batch->pending_submit ||
          batch->host_retirement.valid || !batch->submit_order)
         continue;

      const uint64_t order = batch->completion_order ?
         batch->completion_order : batch->submit_order;
      const VkFence fence = batch->completion_fence ?
         batch->completion_fence : batch->fence;
      const uint32_t feedback_index =
         batch->completion_feedback_index;
      volatile LONG *slot =
         yttrium_venus_batch_feedback_slot(venus, feedback_index);
      if (!order || !fence || !slot) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 completion poll rejected owner=venus2 reason=invalid_completion_state slot=%u submit_order=%llu completion_order=%llu label=%s\n",
                      feedback_index,
                      (unsigned long long)batch->submit_order,
                      (unsigned long long)batch->completion_order,
                      label ? label : "<unknown>");
         return false;
      }

      const VkResult status =
         yttrium_venus_batch_feedback_load(venus, slot);
      if (status == VK_NOT_READY)
         continue;
      if (status != VK_SUCCESS) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 completion poll failed owner=venus2 reason=terminal_feedback status=%d slot=%u submit_order=%llu label=%s\n",
                      status, feedback_index,
                      (unsigned long long)batch->submit_order,
                      label ? label : "<unknown>");
         return false;
      }
      if (order > completed_order) {
         completed_order = order;
         completed_fence = fence;
      }
   }

   if (!completed_order)
      return true;

   struct yttrium_venus_ring_dependency host_retirement =
      yttrium_venus_ring_dependency_none();
   if (!yttrium_venus_queue_host_fence_wait(
          venus, completed_fence, &host_retirement))
      return false;
   if (completed_order > venus->last_completed_submit_order)
      venus->last_completed_submit_order = completed_order;
   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (batch && batch->busy && !batch->host_retirement.valid &&
          batch->submit_order && batch->submit_order <= completed_order) {
         batch->host_retirement = host_retirement;
      }
   }

   return true;
}

static struct yttrium_venus_batch *
yttrium_venus_latest_busy_batch(struct yttrium_venus *venus)
{
   struct yttrium_venus_batch *latest = NULL;

   if (!venus)
      return NULL;

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch)
         continue;
      if (!batch->busy || !batch->submit_order)
         continue;
      if (!latest || batch->submit_order > latest->submit_order)
         latest = batch;
   }

   return latest;
}

static bool
yttrium_venus_wait_batch_internal(struct yttrium_venus *venus,
                                  struct yttrium_venus_batch *batch,
                                  const char *label,
                                  bool wait_for_host_retirement)
{
   bool pending_flush_ok = true;

   if (!venus || !batch || !batch->busy)
      return true;

   if (batch->host_retirement.valid) {
      if (!wait_for_host_retirement)
         return true;
      if (!yttrium_venus_ring_wait_for_seqno(
             venus, &batch->host_retirement, label))
         return false;
      yttrium_venus_retire_batch(venus, batch);
      return true;
   }

   if (batch->pending_submit &&
       !yttrium_venus_flush_pending_submits(venus, label))
      pending_flush_ok = false;
   if (batch->pending_submit || !batch->submit_order) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 batch wait rejected owner=venus2 reason=unsubmitted_batch label=%s\n",
                   label ? label : "<unknown>");
      return false;
   }

   VkFence completion_fence = batch->fence;
   uint64_t completed_order = batch->submit_order;
   if (venus->group_queue_submits) {
      if (!batch->completion_fence || !batch->completion_order) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 batch wait rejected owner=venus2 reason=missing_group_tail_completion submit_order=%llu label=%s\n",
                      (unsigned long long)batch->submit_order,
                      label ? label : "<unknown>");
         return false;
      }
      completion_fence = batch->completion_fence;
      completed_order = batch->completion_order;
   }

   const uint64_t start_us =
      yttrium_trace_sync_wait_is_enabled() ? yttrium_trace_now_us() : 0;
   const bool feedback_enabled =
      yttrium_venus_batch_fence_feedback_enabled();
   VkResult result;
   struct yttrium_venus_ring_dependency host_retirement =
      yttrium_venus_ring_dependency_none();
   if (feedback_enabled) {
      if (!venus->batch_feedback_initialized ||
          batch->completion_feedback_index == UINT32_MAX) {
         YTTRIUM_WARN("yttrium: ERROR: mapped batch-fence feedback wait rejected owner=venus2 reason=missing_completion_feedback label=%s submit_order=%llu completion_order=%llu\n",
                      label ? label : "<unknown>",
                      (unsigned long long)batch->submit_order,
                      (unsigned long long)completed_order);
         return false;
      }

      result = yttrium_venus_wait_batch_feedback(
         venus, batch->completion_feedback_index, completion_fence, label);
      if (result == VK_SUCCESS) {
         /*
          * The mapped write precedes the real fence signal.  Keep host fence
          * retirement and validation ordered before a later reset/reuse,
          * without waiting for another Venus reply here.
          */
         if (!yttrium_venus_queue_host_fence_wait(
                venus, completion_fence, &host_retirement))
            return false;
      }
   } else {
      result =
         vn_call_vkWaitForFences(&venus->vn_ring, venus->device_handle, 1,
                                 &completion_fence, VK_TRUE,
                                 YTTRIUM_VENUS_FENCE_WAIT_NS);
   }
   if (feedback_enabled && start_us) {
      yttrium_venus_debug_sync_wait(
         venus, YTTRIUM_VENUS_SYNC_WAIT_BATCH_COMPLETION,
         yttrium_trace_now_us() - start_us,
         result == VK_SUCCESS ? 0 : (uint32_t)result,
         label,
         batch->completion_feedback_index,
         completed_order > UINT32_MAX ? UINT32_MAX :
                                         (uint32_t)completed_order,
         VK_COMMAND_TYPE_vkWaitForFences_EXT, 0, 0);
   }
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_WAIT_FENCES,
                              result == VK_SUCCESS ? 0 : (uint32_t)result,
                              start_us, label, 0,
                              YTTRIUM_VENUS_FENCE_WAIT_NS, 2, 0);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: Venus batch fence wait failed owner=venus2 path=%s label=%s result=%d timeout_ns=%llu\n",
                   feedback_enabled ?
                      "mapped_feedback" : "synchronous_reply",
                   label ? label : "<unknown>", result,
                   (unsigned long long)YTTRIUM_VENUS_FENCE_WAIT_NS);
      return false;
   }

   if (completed_order) {
      /* All asynchronous batches use the same Vulkan queue.  Completion of
       * this fence therefore proves completion of every earlier submit.
       */
      if (completed_order > venus->last_completed_submit_order)
         venus->last_completed_submit_order = completed_order;
      if (feedback_enabled) {
         for (uint32_t i = 0; i < venus->batch_count; i++) {
            struct yttrium_venus_batch *completed = venus->batches[i];
            if (completed && completed->busy &&
                !completed->host_retirement.valid &&
                completed->submit_order &&
                completed->submit_order <= completed_order)
               completed->host_retirement = host_retirement;
         }
         if (wait_for_host_retirement) {
            if (!yttrium_venus_ring_wait_for_seqno(
                   venus, &host_retirement, label))
               return false;
            for (uint32_t i = 0; i < venus->batch_count; i++) {
               struct yttrium_venus_batch *completed = venus->batches[i];
               if (completed && completed->busy &&
                   completed->host_retirement.valid &&
                   yttrium_venus_ring_seqno_complete(
                      venus, &completed->host_retirement))
                  yttrium_venus_retire_batch(venus, completed);
            }
         }
      } else {
         for (uint32_t i = 0; i < venus->batch_count; i++) {
            struct yttrium_venus_batch *completed = venus->batches[i];
            if (completed && completed->busy &&
                completed->submit_order &&
                completed->submit_order <= completed_order)
               yttrium_venus_retire_batch(venus, completed);
         }
      }
   } else {
      yttrium_venus_retire_batch(venus, batch);
   }
   return pending_flush_ok;
}

static bool
yttrium_venus_wait_batch(struct yttrium_venus *venus,
                         struct yttrium_venus_batch *batch,
                         const char *label)
{
   return yttrium_venus_wait_batch_internal(venus, batch, label, true);
}

static bool
yttrium_venus_wait_batch_for_present(struct yttrium_venus *venus,
                                     struct yttrium_venus_batch *batch,
                                     const char *label)
{
   /* Scanout needs proof that rendering completed, but batch reset/reuse does
    * not lie on this edge.  Keep the queued host-fence retirement attached to
    * the batch and let ordinary completion polling retire it later. */
   return yttrium_venus_wait_batch_internal(venus, batch, label, false);
}

uint64_t
yttrium_venus2_last_submit_order(struct yttrium_venus *venus)
{
   return venus ? venus->next_batch_submit_order : 0;
}

bool
yttrium_venus2_submit_order_complete(struct yttrium_venus *venus,
                                     uint64_t submit_order)
{
   if (!venus)
      return false;
   if (!submit_order || submit_order <= venus->last_completed_submit_order)
      return true;

   if (!yttrium_venus_retire_completed_batches(
          venus, "GPU finished query completion poll"))
      return false;

   return submit_order <= venus->last_completed_submit_order;
}

bool
yttrium_venus_drain_batches(struct yttrium_venus *venus, const char *label)
{
   bool ok = true;

   if (!venus)
      return false;

   if (!yttrium_venus_flush_pending_submits(venus, label))
      ok = false;

   if (yttrium_venus_batch_epoch_wait_enabled()) {
      struct yttrium_venus_batch *latest =
         yttrium_venus_latest_busy_batch(venus);
      if (latest && !yttrium_venus_wait_batch(venus, latest, label))
         ok = false;
   }

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      if (!yttrium_venus_wait_batch(venus, venus->batches[i], label))
         ok = false;
   }

   return ok;
}

static bool
yttrium_venus_initialize_batch(struct yttrium_venus *venus,
                               struct yttrium_venus_batch *batch,
                               uint32_t index)
{
   if (!venus || !batch || index >= venus->batch_capacity)
      return false;

   if (!yttrium_venus_record_batch_feedback_command(venus, batch, index))
      return false;
   if (batch->initialized)
      return true;

   if (!batch->command_buffer) {
      yttrium_venus_init_object(venus, &batch->command_buffer_obj);
      batch->command_buffer =
         YTTRIUM_VENUS_HANDLE(VkCommandBuffer,
                              &batch->command_buffer_obj);
      const VkCommandBufferAllocateInfo alloc_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = venus->command_pool,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      struct vn_ring_submit_command allocate_submit;
      vn_submit_vkAllocateCommandBuffers(&venus->vn_ring, 0,
                                         venus->device_handle,
                                         &alloc_info,
                                         &batch->command_buffer,
                                         &allocate_submit);
      if (!yttrium_venus_async_submit_succeeded(
             venus, &allocate_submit, "vkAllocateCommandBuffers-batch",
             batch->command_buffer_obj.id)) {
         memset(&batch->command_buffer_obj, 0,
                sizeof(batch->command_buffer_obj));
         batch->command_buffer = VK_NULL_HANDLE;
         return false;
      }
   }

   if (!batch->fence) {
      yttrium_venus_init_object(venus, &batch->fence_obj);
      batch->fence = YTTRIUM_VENUS_HANDLE(VkFence, &batch->fence_obj);
      const VkFenceCreateInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      };
      struct vn_ring_submit_command fence_submit;
      vn_submit_vkCreateFence(&venus->vn_ring, 0, venus->device_handle,
                              &fence_info, NULL, &batch->fence,
                              &fence_submit);
      if (!yttrium_venus_async_submit_succeeded(
             venus, &fence_submit, "vkCreateFence-batch",
             batch->fence_obj.id)) {
         memset(&batch->fence_obj, 0, sizeof(batch->fence_obj));
         batch->fence = VK_NULL_HANDLE;
         return false;
      }
   }

   batch->initialized = true;
   batch->busy = false;
   batch->feedback_index = index;
   batch->completion_feedback_index = UINT32_MAX;
   return true;
}

static void
yttrium_venus_discard_uninitialized_batch(
   struct yttrium_venus *venus,
   struct yttrium_venus_batch *batch)
{
   if (!venus || !batch)
      return;

   /* Initialization can fail after an earlier asynchronous allocation was
    * accepted.  Release that accepted prefix before discarding a dynamically
    * grown slot; the object whose submission failed has already been cleared.
    */
   if (batch->fence)
      vn_async_vkDestroyFence(&venus->vn_ring, venus->device_handle,
                              batch->fence, NULL);
   if (batch->command_buffer)
      vn_async_vkFreeCommandBuffers(&venus->vn_ring,
                                    venus->device_handle,
                                    venus->command_pool, 1,
                                    &batch->command_buffer);
   if (batch->feedback_command_buffer)
      vn_async_vkFreeCommandBuffers(&venus->vn_ring,
                                    venus->device_handle,
                                    venus->command_pool, 1,
                                    &batch->feedback_command_buffer);
   memset(batch, 0, sizeof(*batch));
}

static bool
yttrium_venus_ensure_batches(struct yttrium_venus *venus)
{
   if (!venus || !venus->batches || !venus->batch_count ||
       venus->batch_count > venus->batch_capacity ||
       !venus->device_handle || !venus->command_pool)
      return false;
   if (!yttrium_venus_ensure_batch_feedback_buffer(venus))
      return false;

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch)
         return false;
      if (!batch->initialized &&
          !yttrium_venus_initialize_batch(venus, batch, i))
         return false;
   }

   return true;
}

static struct yttrium_venus_batch *
yttrium_venus_find_free_batch(struct yttrium_venus *venus,
                              uint32_t *out_index)
{
   if (!venus || !venus->batch_count)
      return NULL;

   for (uint32_t n = 0; n < venus->batch_count; n++) {
      const uint32_t index =
         (venus->next_batch + n) % venus->batch_count;
      struct yttrium_venus_batch *batch = venus->batches[index];
      if (batch && !batch->busy) {
         if (out_index)
            *out_index = index;
         return batch;
      }
   }
   return NULL;
}

static struct yttrium_venus_batch *
yttrium_venus_grow_batch_pool(struct yttrium_venus *venus,
                              uint32_t *out_index)
{
   if (!venus || venus->batch_count >= venus->batch_capacity)
      return NULL;

   const uint32_t index = venus->batch_count;
   struct yttrium_venus_batch *batch =
      CALLOC_STRUCT(yttrium_venus_batch);
   if (!batch) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 batch pool growth failed owner=venus2 reason=out_of_memory slot=%u capacity=%u bytes=%llu\n",
                   index, venus->batch_capacity,
                   (unsigned long long)sizeof(*batch));
      return NULL;
   }
   venus->batches[index] = batch;
   venus->batch_count++;
   if (!yttrium_venus_initialize_batch(venus, batch, index)) {
      venus->batch_count--;
      venus->batches[index] = NULL;
      yttrium_venus_discard_uninitialized_batch(venus, batch);
      FREE(batch);
      return NULL;
   }

   if (out_index)
      *out_index = index;
   return batch;
}

static void
yttrium_venus_grow_batch_pool_for_pressure(struct yttrium_venus *venus)
{
   if (!venus || !venus->batch_count ||
       venus->batch_count >= venus->batch_capacity)
      return;

   /*
    * Growing only after the last free slot is consumed is not sufficient.
    * The completion poll on each acquisition can retire one slot and leave
    * a heavy workload oscillating at the initial pool size indefinitely.  In
    * that state producer and renderer concurrency is capped even though the
    * bounded pool still has ample room.
    *
    * Keep a quarter of the current pool as producer headroom.  Sustained
    * pressure grows geometrically to the hard bound, while light contexts
    * retain the small initial allocation.
    */
   const uint32_t pressure_threshold =
      venus->batch_count - venus->batch_count / 4;
   if (venus->live_batch_count < pressure_threshold)
      return;

   const uint32_t target =
      MIN2(venus->batch_capacity, venus->batch_count * 2);
   while (venus->batch_count < target) {
      if (!yttrium_venus_grow_batch_pool(venus, NULL))
         break;
   }
}

static struct yttrium_venus_batch *
yttrium_venus_acquire_batch(struct yttrium_venus *venus, const char *label)
{
   if (!yttrium_venus_ensure_batches(venus))
      return NULL;

   yttrium_venus_grow_batch_pool_for_pressure(venus);

   if (!yttrium_venus_retire_completed_batches(venus, label))
      return NULL;

   uint32_t index = 0;
   struct yttrium_venus_batch *batch =
      yttrium_venus_find_free_batch(venus, &index);
   if (!batch)
      batch = yttrium_venus_grow_batch_pool(venus, &index);

   if (!batch) {
      /*
       * This is the sole ordinary-acquisition wait: every slot at the hard
       * memory ceiling is live.  Flush an incomplete submit group, retry the
       * mapped poll, then block on the oldest order only if the ceiling is
       * genuinely still exhausted.
       */
      if (!yttrium_venus_flush_pending_submits(
             venus, "batch pool bounded-memory exhaustion") ||
          !yttrium_venus_retire_completed_batches(
             venus, "batch pool bounded-memory exhaustion"))
         return NULL;
      batch = yttrium_venus_find_free_batch(venus, &index);
      if (!batch) {
         struct yttrium_venus_batch *oldest = NULL;
         for (uint32_t i = 0; i < venus->batch_count; i++) {
            struct yttrium_venus_batch *candidate = venus->batches[i];
            if (!candidate || !candidate->busy ||
                candidate->pending_submit || !candidate->submit_order)
               continue;
            if (!oldest || candidate->submit_order < oldest->submit_order)
               oldest = candidate;
         }
         if (!oldest || !yttrium_venus_wait_batch(
               venus, oldest, "batch pool bounded-memory exhaustion"))
            return NULL;
         batch = yttrium_venus_find_free_batch(venus, &index);
      }
   }
   if (!batch)
      return NULL;

   venus->next_batch = (index + 1) % venus->batch_count;
   yttrium_venus_batch_clear_refs(batch);
   return batch;
}

static bool
yttrium_venus_flush_pending_submits(struct yttrium_venus *venus,
                                    const char *label)
{
   if (!venus)
      return false;

   const uint32_t count = venus->pending_submit_count;
   if (!count)
      return true;
   if (!venus->group_queue_submits ||
       !venus->pending_submit_batches ||
       !venus->group_queue_submit_size) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submit failed owner=venus2 reason=invalid_group_state count=%u label=%s\n",
                   count, label ? label : "<unknown>");
      return false;
   }
   if (count > venus->group_queue_submit_size ||
       count > YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_MAX) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submit failed owner=venus2 reason=pending_count_overflow count=%u limit=%u label=%s\n",
                   count, venus->group_queue_submit_size,
                   label ? label : "<unknown>");
      return false;
   }

   VkCommandBuffer
      command_buffers[YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_MAX + 1];
   for (uint32_t i = 0; i < count; i++) {
      struct yttrium_venus_batch *batch =
         venus->pending_submit_batches[i];
      if (!batch || !batch->initialized || !batch->command_buffer ||
          !batch->fence || !batch->busy || !batch->pending_submit ||
          !batch->submit_order || batch->completion_order ||
          batch->completion_fence) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submit failed owner=venus2 reason=invalid_pending_batch index=%u count=%u label=%s\n",
                      i, count, label ? label : "<unknown>");
         return false;
      }
      command_buffers[i] = batch->command_buffer;
   }

   struct yttrium_venus_batch *tail =
      venus->pending_submit_batches[count - 1];
   const bool feedback_enabled =
      yttrium_venus_batch_fence_feedback_enabled();
   uint32_t submit_command_buffer_count = count;
   if (feedback_enabled) {
      if (!tail->feedback_command_buffer ||
          tail->feedback_index >= venus->batch_capacity) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submit failed owner=venus2 reason=missing_fence_feedback count=%u slot=%u label=%s\n",
                      count, tail->feedback_index,
                      label ? label : "<unknown>");
         return false;
      }
      if (!yttrium_venus_batch_feedback_reset(
             venus, tail->feedback_index, label))
         return false;
      command_buffers[submit_command_buffer_count++] =
         tail->feedback_command_buffer;
   }
   uint64_t start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   vn_async_vkResetFences(&venus->vn_ring, venus->device_handle, 1,
                          &tail->fence);
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RESET_FENCES,
                              0, start_us, label, 0, 1, 2, 0);

   const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = submit_command_buffer_count,
      .pCommandBuffers = command_buffers,
   };
   start_us = yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   if (!yttrium_venus_queue_submit(venus, &submit_info, tail->fence))
      return false;
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_QUEUE_SUBMIT,
                              0, start_us, label, 0, count, 2, 0);

   for (uint32_t i = 0; i < count; i++)
      yttrium_venus_apply_draw_mirror_updates(
         venus->pending_submit_batches[i]);
   for (uint32_t i = 0; i < count; i++)
      yttrium_venus_clear_draw_mirror_pending_state(
         venus->pending_submit_batches[i]);

   const uint64_t completion_order = tail->submit_order;
   for (uint32_t i = 0; i < count; i++) {
      struct yttrium_venus_batch *batch =
         venus->pending_submit_batches[i];
      batch->pending_submit = false;
      batch->completion_fence = tail->fence;
      batch->completion_order = completion_order;
      batch->completion_feedback_index =
         feedback_enabled ? tail->feedback_index : UINT32_MAX;
      venus->pending_submit_batches[i] = NULL;
   }
   venus->pending_submit_count = 0;
   venus->pending_submit_op_count = 0;

   if (!yttrium_venus_ring_flush_notify(venus, label, false)) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submit ring notify failed owner=venus2 count=%u label=%s\n",
                   count, label ? label : "<unknown>");
      return false;
   }

   return true;
}

static bool
yttrium_venus_submit_batch_async(struct yttrium_venus *venus,
                                 struct yttrium_venus_batch *batch,
                                 uint32_t op_count,
                                 const char *label)
{
   if (!venus || !batch || !batch->initialized || batch->busy)
      return false;

   if (venus->group_queue_submits) {
      if (!venus->pending_submit_batches ||
          !venus->group_queue_submit_size ||
          venus->group_queue_submit_size >
             YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_MAX) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submit rejected owner=venus2 reason=invalid_group_state size=%u label=%s\n",
                      venus->group_queue_submit_size,
                      label ? label : "<unknown>");
         return false;
      }
      if (venus->pending_submit_count >=
          venus->group_queue_submit_size) {
         if (!yttrium_venus_flush_pending_submits(venus, label))
            return false;
      }

      yttrium_venus_mark_batch_live(venus, batch);
      batch->pending_submit = true;
      batch->submit_order = ++venus->next_batch_submit_order;
      batch->completion_order = 0;
      batch->completion_fence = VK_NULL_HANDLE;
      batch->completion_feedback_index = UINT32_MAX;
      venus->pending_submit_batches[venus->pending_submit_count++] = batch;
      venus->pending_submit_op_count += op_count;

      if (venus->pending_submit_count >= venus->group_queue_submit_size ||
          (venus->group_queue_submit_op_threshold &&
           venus->pending_submit_op_count >=
              venus->group_queue_submit_op_threshold))
         return yttrium_venus_flush_pending_submits(venus, label);
      return true;
   }

   uint64_t start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;

   const bool feedback_enabled =
      yttrium_venus_batch_fence_feedback_enabled();
   if (feedback_enabled) {
      if (!batch->feedback_command_buffer ||
          batch->feedback_index >= venus->batch_capacity) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 batch queue submit failed owner=venus2 reason=missing_fence_feedback slot=%u label=%s\n",
                      batch->feedback_index,
                      label ? label : "<unknown>");
         return false;
      }
      if (!yttrium_venus_batch_feedback_reset(
             venus, batch->feedback_index, label))
         return false;
   }

   vn_async_vkResetFences(&venus->vn_ring, venus->device_handle, 1,
                          &batch->fence);
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RESET_FENCES,
                              0, start_us, label, 0, 1, 2, 0);

   const VkCommandBuffer command_buffers[2] = {
      batch->command_buffer,
      batch->feedback_command_buffer,
   };
   const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = feedback_enabled ? 2 : 1,
      .pCommandBuffers = command_buffers,
   };
   start_us = yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   if (!yttrium_venus_queue_submit(venus, &submit_info, batch->fence))
      return false;
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_QUEUE_SUBMIT,
                              0, start_us, label, 0, 1, 2, 0);

   yttrium_venus_commit_draw_mirror_updates(batch);

   yttrium_venus_mark_batch_live(venus, batch);
   batch->submit_order = ++venus->next_batch_submit_order;
   batch->completion_feedback_index =
      feedback_enabled ? batch->feedback_index : UINT32_MAX;
   if (!yttrium_venus_ring_flush_notify(venus, label, false)) {
      YTTRIUM_LOG("yttrium: Venus batch ring notify failed label=%s\n",
                  label ? label : "<unknown>");
      return false;
   }

   return true;
}

static bool
yttrium_venus_batch_uses_resource(const struct yttrium_venus_batch *batch,
                                  const struct yttrium_venus_resource *resource)
{
   if (!batch || !resource)
      return false;
   for (uint32_t i = 0; i < batch->resource_count; i++) {
      if (batch->resources[i] == resource)
         return true;
   }
   return false;
}

struct yttrium_venus_batch *
yttrium_venus_find_latest_resource_batch(
   struct yttrium_venus *venus,
   const struct yttrium_venus_resource *resource)
{
   struct yttrium_venus_batch *latest = NULL;

   if (!venus || !resource)
      return NULL;

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch)
         continue;
      if (!batch->busy || !yttrium_venus_batch_uses_resource(batch, resource))
         continue;
      if (!latest || batch->submit_order > latest->submit_order)
         latest = batch;
   }

   return latest;
}

void
yttrium_venus_batch_retire_resource(
   struct yttrium_venus_batch *batch,
   struct yttrium_venus_retired_resource *retired)
{
   if (!batch || !retired)
      return;

   retired->next = batch->retired_resources;
   batch->retired_resources = retired;
}

static bool
yttrium_venus_batch_uses_pipeline(const struct yttrium_venus_batch *batch,
                                  const struct yttrium_pipeline *pipeline)
{
   if (!batch || !pipeline)
      return false;
   for (uint32_t i = 0; i < batch->pipeline_count; i++) {
      if (batch->pipelines[i] == pipeline)
         return true;
   }
   return false;
}

struct yttrium_venus_batch *
yttrium_venus_find_latest_pipeline_batch(
   struct yttrium_venus *venus,
   const struct yttrium_pipeline *pipeline)
{
   struct yttrium_venus_batch *latest = NULL;

   if (!venus || !pipeline)
      return NULL;

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch)
         continue;
      if (!batch->busy || !yttrium_venus_batch_uses_pipeline(batch, pipeline))
         continue;
      if (!latest || batch->submit_order > latest->submit_order)
         latest = batch;
   }

   return latest;
}

bool
yttrium_venus_wait_resource_batches(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    const char *label)
{
   bool ok = true;

   if (!venus || !resource)
      return true;

   if (!yttrium_venus_flush_pending_submits(venus, label))
      ok = false;

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch)
         continue;
      if (!batch->busy || !yttrium_venus_batch_uses_resource(batch, resource))
         continue;
      if (!yttrium_venus_wait_batch(venus, batch, label))
         ok = false;
   }

   return ok;
}

bool
yttrium_venus_wait_pipeline_batches(struct yttrium_venus *venus,
                                    struct yttrium_pipeline *pipeline,
                                    const char *label)
{
   bool ok = true;

   if (!venus || !pipeline)
      return true;

   if (!yttrium_venus_flush_pending_submits(venus, label))
      ok = false;

   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch)
         continue;
      if (!batch->busy || !yttrium_venus_batch_uses_pipeline(batch, pipeline))
         continue;
      if (!yttrium_venus_wait_batch(venus, batch, label))
         ok = false;
   }

   return ok;
}

void
yttrium_venus_destroy_batches(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   const bool drained =
      yttrium_venus_drain_batches(venus, "batch queue destroy");
   for (uint32_t i = 0; i < venus->batch_count; i++) {
      struct yttrium_venus_batch *batch = venus->batches[i];
      if (!batch)
         continue;
      if (!batch->initialized && !batch->fence &&
          !batch->command_buffer && !batch->feedback_command_buffer)
         continue;
      if (batch->retired_resources) {
         if (batch->busy && !drained) {
            YTTRIUM_LOG("yttrium: Venus destroying busy batch with retired resources slot=%u label=batch queue destroy\n",
                        i);
         } else {
            yttrium_venus_destroy_retired_resources(
               venus, &batch->retired_resources);
         }
      }
      yttrium_venus_destroy_cmd_batch_transients(
         venus, batch->transients, &batch->transient_count);
      yttrium_venus_cmd_batch_destroy_descriptor_pool(
         venus, &batch->descriptor_pool);
      if (batch->fence)
         vn_async_vkDestroyFence(&venus->vn_ring, venus->device_handle,
                                 batch->fence, NULL);
      if (batch->command_buffer)
         vn_async_vkFreeCommandBuffers(&venus->vn_ring,
                                       venus->device_handle,
                                       venus->command_pool, 1,
                                       &batch->command_buffer);
      if (batch->feedback_command_buffer)
         vn_async_vkFreeCommandBuffers(&venus->vn_ring,
                                       venus->device_handle,
                                       venus->command_pool, 1,
                                       &batch->feedback_command_buffer);
      memset(batch, 0, sizeof(*batch));
   }
   /*
    * A pooled mapping can still be protected by an asynchronous host fence
    * wait.  Teardown is an explicit synchronization edge: let the renderer
    * pass every queued wait before unmapping the guest allocation.
    */
   if (venus->draw_backing_pool)
      yttrium_venus_drain_ring(venus, "draw backing pool destroy");
   yttrium_venus_destroy_draw_backing_pool(venus);
   yttrium_venus_destroy_batch_feedback_buffer(venus);
}

bool
yttrium_venus_begin_command_buffer(
   struct yttrium_venus *venus,
   const char *label,
   const VkCommandBufferBeginInfo *begin_info)
{
   (void)label;
   vn_async_vkBeginCommandBuffer(&venus->vn_ring, venus->command_buffer,
                                 begin_info);
   return true;
}

bool
yttrium_venus_end_command_buffer(struct yttrium_venus *venus,
                                 const char *label)
{
   (void)label;
   vn_async_vkEndCommandBuffer(&venus->vn_ring, venus->command_buffer);
   return true;
}

static void
yttrium_venus_cmd_batch_barrier_deferred_draws_to_transfer(
   struct yttrium_venus *venus)
{
   const VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT,
   };

   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, venus->command_buffer,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
}

static void
yttrium_venus_cmd_batch_barrier_transfer_to_deferred_draws(
   struct yttrium_venus *venus)
{
   const VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                       VK_ACCESS_INDEX_READ_BIT |
                       VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                       VK_ACCESS_UNIFORM_READ_BIT |
                       VK_ACCESS_SHADER_READ_BIT |
                       VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
                       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
   };

   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0,
                                 1, &barrier, 0, NULL, 0, NULL);
}

bool
yttrium_venus_begin_command_batch(struct yttrium_venus *venus,
                                  const char *label,
                                  bool native_draw_batch,
                                  bool async_submit)
{
   if (!venus)
      return false;

   /*
    * A Venus2 command batch always owns a pool slot.  The caller's batching
    * policy may control how many operations are coalesced into that slot, but
    * it must not fall back to the shared command buffer and wait before reset.
    */
   (void)async_submit;
   async_submit = true;

   if (venus->display_copy_batch_recording) {
      if (venus->cmd_batch_async_submit && venus->cmd_batch)
         venus->command_buffer = venus->cmd_batch->command_buffer;

      if (venus->cmd_batch_deferred_draw_count && !native_draw_batch) {
         if (venus->cmd_batch_async_submit != async_submit ||
             !yttrium_venus_mixed_draw_transfer_batch_enabled()) {
            if (!yttrium_venus_flush_command_batch(venus, label))
               return false;
         } else if (!yttrium_venus_cmd_batch_emit_deferred_draws(venus,
                                                                 label)) {
            return false;
         } else {
            yttrium_venus_cmd_batch_barrier_deferred_draws_to_transfer(venus);
            venus->cmd_batch_native_draw_only = false;
            venus->cmd_batch_has_transfer_ops = true;
            return true;
         }
      } else if (venus->cmd_batch_async_submit != async_submit ||
                 (!native_draw_batch && venus->cmd_batch_native_draw_only)) {
         if (!yttrium_venus_flush_command_batch(venus, label))
            return false;
      } else {
         if (native_draw_batch && venus->cmd_batch_has_transfer_ops) {
            yttrium_venus_cmd_batch_barrier_transfer_to_deferred_draws(venus);
            venus->cmd_batch_has_transfer_ops = false;
         } else if (!native_draw_batch) {
            venus->cmd_batch_has_transfer_ops = true;
         }
         venus->cmd_batch_native_draw_only =
            venus->cmd_batch_native_draw_only && native_draw_batch;
         return true;
      }
   }

   struct yttrium_venus_batch *batch =
      yttrium_venus_acquire_batch(venus, label);
   if (!batch)
      return false;

   venus->cmd_batch = batch;
   venus->command_buffer = batch->command_buffer;

   /* The pool allows individual resets, and acquisition only returns slots
    * that are no longer pending.  Begin implicitly resets the old recording;
    * an explicit reset here would duplicate the host driver's reset work.
    */
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (!yttrium_venus_begin_command_buffer(venus, label, &begin_info))
      return false;

   venus->display_copy_batch_recording = true;
   venus->display_copy_batch_count = 0;
   venus->cmd_batch_native_draw_only = native_draw_batch;
   venus->cmd_batch_async_submit = async_submit;
   venus->cmd_batch_has_transfer_ops = !native_draw_batch;
   return true;
}

bool
yttrium_venus_begin_display_copy_batch(struct yttrium_venus *venus)
{
   return yttrium_venus_begin_command_batch(venus,
                                           "display image copy batch",
                                           false, true);
}

bool
yttrium_venus_begin_transfer_batch(struct yttrium_venus *venus)
{
   return yttrium_venus_begin_command_batch(venus, "transfer batch",
                                           false, true);
}
