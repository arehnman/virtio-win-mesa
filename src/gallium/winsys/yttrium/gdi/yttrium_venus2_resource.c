/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "util/u_math.h"

#include "yttrium_options.h"
#include "yttrium_venus2_private.h"
#include "yttrium_trace.h"

#include "venus-protocol/vn_protocol_driver_buffer.h"
#include "venus-protocol/vn_protocol_driver_device.h"
#include "venus-protocol/vn_protocol_driver_device_memory.h"
#include "venus-protocol/vn_protocol_driver_image.h"

static bool
yttrium_venus_sample_count(unsigned sample_count,
                           VkSampleCountFlagBits *out_samples)
{
   switch (sample_count) {
   case 0:
   case 1:
      *out_samples = VK_SAMPLE_COUNT_1_BIT;
      return true;
   case 2:
      *out_samples = VK_SAMPLE_COUNT_2_BIT;
      return true;
   case 4:
      *out_samples = VK_SAMPLE_COUNT_4_BIT;
      return true;
   case 8:
      *out_samples = VK_SAMPLE_COUNT_8_BIT;
      return true;
   case 16:
      *out_samples = VK_SAMPLE_COUNT_16_BIT;
      return true;
   default:
      *out_samples = VK_SAMPLE_COUNT_1_BIT;
      return false;
   }
}

static VkImageUsageFlags
yttrium_venus_feedback_loop_usage(const struct yttrium_venus *venus,
                                  VkImageUsageFlags usage)
{
   const VkImageUsageFlags attachment_usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

   if (venus && venus->attachment_feedback_loop_layout &&
       (usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
       (usage & attachment_usage))
      usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   return usage;
}

static void
yttrium_venus_discard_accepted_image(struct yttrium_venus *venus,
                                     struct yttrium_venus_resource *resource,
                                     bool memory_accepted)
{
   if (memory_accepted) {
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->memory, NULL);
   }
   vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                           resource->image, NULL);
   memset(resource, 0, sizeof(*resource));
}

static void
yttrium_venus_discard_accepted_buffer(struct yttrium_venus *venus,
                                      struct yttrium_venus_resource *resource,
                                      bool memory_accepted)
{
   if (memory_accepted) {
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->memory, NULL);
   }
   vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                            resource->buffer, NULL);
   memset(resource, 0, sizeof(*resource));
}

static void
yttrium_venus_discard_accepted_bound_image(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource)
{
   /* The bind was accepted, so destroy the image before freeing its memory. */
   vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                           resource->image, NULL);
   vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                         resource->memory, NULL);
   memset(resource, 0, sizeof(*resource));
}

static bool
yttrium_venus_get_image_subresource_layout_checked(
   struct yttrium_venus *venus,
   const struct yttrium_venus_resource *resource,
   const VkImageSubresource *subresource,
   VkSubresourceLayout *layout,
   const char *owner)
{
   struct vn_ring_submit_command submit;
   vn_submit_vkGetImageSubresourceLayout(
      &venus->vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT,
      venus->device_handle, resource->image, subresource, layout, &submit);

   struct vn_cs_decoder *reply =
      submit.ring_seqno_valid ?
         vn_ring_get_command_reply(&venus->vn_ring, &submit) : NULL;
   if (!reply) {
      if (submit.ring_seqno_valid)
         vn_ring_free_command_reply(&venus->vn_ring, &submit);
      YTTRIUM_WARN("yttrium: ERROR: Venus image subresource layout query failed owner=venus2 operation=vkGetImageSubresourceLayout resource_owner=%s image_id=%llu reason=%s action=fail-resource-create\n",
                   owner ? owner : "<unknown>",
                   (unsigned long long)resource->image_obj.id,
                   submit.ring_seqno_valid ? "missing-reply" :
                                             "local-ring-enqueue-failed");
      return false;
   }

   vn_decode_vkGetImageSubresourceLayout_reply(
      reply, venus->device_handle, resource->image, subresource, layout);
   vn_ring_free_command_reply(&venus->vn_ring, &submit);
   return true;
}

static bool
yttrium_venus_create_image_tiled(struct yttrium_venus *venus,
                                 struct yttrium_venus_resource *resource,
                                 VkImageType image_type,
                                 VkFormat vk_format,
                                 uint32_t width,
                                 uint32_t height,
                                 uint32_t depth,
                                 uint32_t levels,
                                 uint32_t layers,
                                 VkImageCreateFlags extra_flags,
                                 VkImageUsageFlags usage,
                                 VkSampleCountFlagBits samples,
                                 VkImageTiling tiling,
                                 VkImageLayout initial_layout)
{
   yttrium_venus_init_object(venus, &resource->image_obj);
   resource->image = YTTRIUM_VENUS_HANDLE(VkImage, &resource->image_obj);
   VkImageCreateFlags image_flags =
      extra_flags |
      ((usage & (VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_STORAGE_BIT |
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) ?
          VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT :
          0);

   if (venus->multisampled_render_to_single_sampled &&
       samples == VK_SAMPLE_COUNT_1_BIT &&
       tiling == VK_IMAGE_TILING_OPTIMAL &&
       (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)))
      image_flags |=
         VK_IMAGE_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT;

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = image_flags,
      .imageType = image_type,
      .format = vk_format,
      .extent = { width, height, depth },
      .mipLevels = MAX2(levels, 1),
      .arrayLayers = MAX2(layers, 1),
      .samples = samples,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = initial_layout,
   };

   struct vn_ring_submit_command submit;
   vn_submit_vkCreateImage(&venus->vn_ring, 0, venus->device_handle,
                           &image_info, NULL, &resource->image, &submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &submit, "vkCreateImage", resource->image_obj.id)) {
      memset(&resource->image_obj, 0, sizeof(resource->image_obj));
      resource->image = VK_NULL_HANDLE;
      return false;
   }

   resource->image_usage = usage;
   resource->vk_format = vk_format;
   resource->width = width;
   resource->height = height;
   resource->depth = MAX2(depth, 1);
   resource->levels = MAX2(levels, 1);
   resource->layers = MAX2(layers, 1);
   resource->samples = samples;
   resource->ds_clear_history_count = 0;
   resource->ds_clear_history_valid =
      yttrium_venus_format_has_depth(vk_format);
   return true;
}

static bool
yttrium_venus_create_image(struct yttrium_venus *venus,
                           struct yttrium_venus_resource *resource,
                           VkFormat vk_format,
                           uint32_t width,
                           uint32_t height,
                           VkImageUsageFlags usage)
{
   return yttrium_venus_create_image_tiled(venus, resource, VK_IMAGE_TYPE_2D,
                                           vk_format, width, height, 1,
                                           1, 1, 0, usage,
                                           VK_SAMPLE_COUNT_1_BIT,
                                           VK_IMAGE_TILING_LINEAR,
                                           VK_IMAGE_LAYOUT_UNDEFINED);
}

uint32_t
yttrium_venus_subresource_width(const struct yttrium_venus_resource *resource,
                                uint32_t level)
{
   if (!resource || !resource->width)
      return 0;
   return MAX2(resource->width >> MIN2(level, 31u), 1);
}

uint32_t
yttrium_venus_subresource_height(const struct yttrium_venus_resource *resource,
                                 uint32_t level)
{
   if (!resource || !resource->height)
      return 0;
   return MAX2(resource->height >> MIN2(level, 31u), 1);
}

uint32_t
yttrium_venus_subresource_depth(const struct yttrium_venus_resource *resource,
                                uint32_t level)
{
   if (!resource || !resource->depth)
      return 0;
   return MAX2(resource->depth >> MIN2(level, 31u), 1);
}

bool
yttrium_venus_resource_is_3d(
   const struct yttrium_venus_resource *resource)
{
   return resource && resource->depth > 1 && MAX2(resource->layers, 1) == 1;
}

bool
yttrium_venus_valid_image_subresource(
   const struct yttrium_venus_resource *resource,
   uint32_t level, uint32_t first_layer, uint32_t layer_count)
{
   const uint32_t levels = resource ? MAX2(resource->levels, 1) : 0;
   const uint32_t layers = resource ? MAX2(resource->layers, 1) : 0;

   return resource && level < levels && first_layer < layers &&
          layer_count && layer_count <= layers - first_layer;
}

void
yttrium_venus_ds_clear_history_reset(struct yttrium_venus_resource *resource)
{
   if (!resource)
      return;

   resource->ds_clear_history_count = 0;
   resource->ds_clear_history_valid = true;
}

void
yttrium_venus_ds_clear_history_invalidate(
   struct yttrium_venus_resource *resource)
{
   if (!resource || !yttrium_venus_format_has_depth(resource->vk_format))
      return;

   resource->ds_clear_history_count = 0;
   resource->ds_clear_history_valid = false;
}

void
yttrium_venus_ds_clear_history_drop_aspects(
   struct yttrium_venus_resource *resource,
   VkImageAspectFlags aspects)
{
   if (!resource || !resource->ds_clear_history_valid || !aspects)
      return;

   uint32_t out = 0;
   for (uint32_t i = 0; i < resource->ds_clear_history_count; i++) {
      struct yttrium_venus_ds_clear_record record =
         resource->ds_clear_history[i];
      record.aspects &= ~aspects;
      if (record.aspects)
         resource->ds_clear_history[out++] = record;
   }
   resource->ds_clear_history_count = out;
}

static bool
yttrium_venus_ds_clear_covers_subresource(
   const struct yttrium_venus_resource *resource,
   uint32_t level, uint32_t layer, uint32_t x, uint32_t y,
   uint32_t width, uint32_t height)
{
   return resource && level == 0 && layer == 0 && x == 0 && y == 0 &&
          width == yttrium_venus_subresource_width(resource, level) &&
          height == yttrium_venus_subresource_height(resource, level) &&
          MAX2(resource->levels, 1) == 1 && MAX2(resource->layers, 1) == 1;
}

void
yttrium_venus_ds_clear_history_note(
   struct yttrium_venus_resource *resource,
   VkImageAspectFlags aspects, double depth, unsigned stencil,
   uint32_t level, uint32_t layer, uint32_t x, uint32_t y,
   uint32_t width, uint32_t height)
{
   if (!resource || !yttrium_venus_format_has_depth(resource->vk_format))
      return;

   aspects &= yttrium_venus_format_aspects(resource->vk_format) &
              (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   if (!aspects)
      return;

   const bool full_resource_clear =
      yttrium_venus_ds_clear_covers_subresource(resource, level, layer, x, y,
                                                width, height);
   if (!resource->ds_clear_history_valid) {
      if (!full_resource_clear)
         return;
      yttrium_venus_ds_clear_history_reset(resource);
   }

   if (full_resource_clear)
      yttrium_venus_ds_clear_history_drop_aspects(resource, aspects);

   if (resource->ds_clear_history_count >=
       YTTRIUM_VENUS_MAX_DS_CLEAR_HISTORY) {
      yttrium_venus_ds_clear_history_invalidate(resource);
      return;
   }

   struct yttrium_venus_ds_clear_record *record =
      &resource->ds_clear_history[resource->ds_clear_history_count++];
   *record = (struct yttrium_venus_ds_clear_record) {
      .aspects = aspects,
      .depth = (float)depth,
      .stencil = stencil & 0xff,
      .level = level,
      .layer = layer,
      .x = x,
      .y = y,
      .width = width,
      .height = height,
   };
}

bool
yttrium_venus_valid_render_subresource(
   const struct yttrium_venus_resource *resource,
   uint32_t level, uint32_t first_layer, uint32_t layer_count)
{
   if (!resource || level >= MAX2(resource->levels, 1) || !layer_count)
      return false;

   const uint32_t layers = yttrium_venus_resource_is_3d(resource) ?
      yttrium_venus_subresource_depth(resource, level) :
      MAX2(resource->layers, 1);
   return first_layer < layers && layer_count <= layers - first_layer;
}

VkImageViewType
yttrium_venus_render_view_type(const struct yttrium_venus_resource *resource,
                               uint32_t layer_count)
{
   if (yttrium_venus_resource_is_3d(resource) || layer_count > 1)
      return layer_count > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY :
                               VK_IMAGE_VIEW_TYPE_2D;

   return VK_IMAGE_VIEW_TYPE_2D;
}

VkImageSubresourceRange
yttrium_venus_render_barrier_range(
   const struct yttrium_venus_resource *resource,
   VkImageAspectFlags aspect_mask,
   uint32_t level, uint32_t first_layer, uint32_t layer_count)
{
   VkImageSubresourceRange range = {
      .aspectMask = aspect_mask,
      .baseMipLevel = level,
      .levelCount = 1,
      .baseArrayLayer = first_layer,
      .layerCount = layer_count,
   };

   if (yttrium_venus_resource_is_3d(resource)) {
      range.baseArrayLayer = 0;
      range.layerCount = 1;
   }

   return range;
}

/* A buffer bound to memory allocated with the export info below has to
 * declare the same handle type when it is created, or the binding is
 * invalid: the host driver is entitled to have chosen a layout the
 * export cannot satisfy.
 */
static VkExternalMemoryBufferCreateInfo
yttrium_venus_external_buffer_info(void)
{
   return (VkExternalMemoryBufferCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
}

static VkExportMemoryAllocateInfo
yttrium_venus_export_memory_info(void)
{
   return (VkExportMemoryAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
}

bool
yttrium_venus2_create_display_image(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t width,
                                   uint32_t height,
                                   enum pipe_format pipe_format,
                                   uint64_t min_allocation_size,
                                   uint64_t *out_memory_id,
                                   uint64_t *out_allocation_size)
{
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);

   if (out_memory_id)
      *out_memory_id = 0;
   if (out_allocation_size)
      *out_allocation_size = 0;

   if (!resource || vk_format == VK_FORMAT_UNDEFINED)
      return false;

   if (resource->initialized) {
      if (resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_memory_id)
         *out_memory_id = resource->memory_obj.id;
      if (out_allocation_size)
         *out_allocation_size = resource->allocation_size;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   usage = yttrium_venus_feedback_loop_usage(venus, usage);
   if (!yttrium_venus_create_image(venus, resource, vk_format, width, height,
                                    usage)) {
      usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      if (!yttrium_venus_create_image(venus, resource, vk_format, width,
                                      height, usage))
         return false;
   }

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetImageMemoryRequirements(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->image, &reqs);

   /*
    * Display images remain CPU-mappable, but prefer coherent device-local
    * memory because the host display path reads them with the GPU on Present.
    * A legacy BAR can satisfy this preference while it has space; otherwise
    * the chooser falls back to any host-visible type.
    */
   const VkMemoryPropertyFlags required_memory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      (resource->cpu_readback ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
   const VkMemoryPropertyFlags preferred_memory =
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      (resource->cpu_readback ? 0 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, reqs.memoryTypeBits,
                                       required_memory, preferred_memory);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no compatible memory type for display image bits=0x%x %ux%u format=%u req_size=0x%llx cpu_readback=%u required=0x%x preferred=0x%x\n",
                   reqs.memoryTypeBits, width, height, pipe_format,
                   (unsigned long long)reqs.size, resource->cpu_readback,
                   required_memory, preferred_memory);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   VkExportMemoryAllocateInfo export_info =
      yttrium_venus_export_memory_info();
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export_info,
      .allocationSize = MAX2(reqs.size, (VkDeviceSize)min_allocation_size),
      .memoryTypeIndex = memory_type_index,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateMemory(&venus->vn_ring, 0, venus->device_handle,
                              &memory_info, NULL, &resource->memory,
                              &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateMemory(display-image)",
          resource->memory_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, false);
      return false;
   }

   struct vn_ring_submit_command bind_submit;
   vn_submit_vkBindImageMemory(&venus->vn_ring, 0, venus->device_handle,
                               resource->image, resource->memory, 0,
                               &bind_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &bind_submit, "vkBindImageMemory(display-image)",
          resource->image_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, true);
      return false;
   }

   const VkImageSubresource subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .arrayLayer = 0,
   };
   VkSubresourceLayout layout;
   memset(&layout, 0, sizeof(layout));
   if (!yttrium_venus_get_image_subresource_layout_checked(
          venus, resource, &subresource, &layout, "display-image")) {
      yttrium_venus_discard_accepted_bound_image(venus, resource);
      return false;
   }

   resource->layout = VK_IMAGE_LAYOUT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_offset = layout.offset;
   resource->image_size = layout.size;
   resource->image_row_pitch = layout.rowPitch;
   resource->image_array_pitch = layout.arrayPitch;
   resource->initialized = true;
   resource->buffer_backed = false;

   if (out_memory_id)
      *out_memory_id = resource->memory_obj.id;
   if (out_allocation_size)
      *out_allocation_size = memory_info.allocationSize;

   YTTRIUM_LOG("yttrium: Venus display image memory_id=%llu image_id=%llu size=0x%llx req_size=0x%llx type=%u flags=0x%x bits=0x%x usage=0x%x layout_offset=0x%llx layout_size=0x%llx row_pitch=%llu array_pitch=%llu depth_pitch=%llu\n",
                (unsigned long long)resource->memory_obj.id,
                (unsigned long long)resource->image_obj.id,
                (unsigned long long)memory_info.allocationSize,
                (unsigned long long)reqs.size,
                memory_type_index,
                venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
                reqs.memoryTypeBits, usage,
                (unsigned long long)layout.offset,
                (unsigned long long)layout.size,
                (unsigned long long)layout.rowPitch,
                (unsigned long long)layout.arrayPitch,
                (unsigned long long)layout.depthPitch);
   return true;
}

bool
yttrium_venus2_create_color_attachment_image(struct yttrium_venus *venus,
                                            struct yttrium_venus_resource *resource,
                                            uint32_t width,
                                            uint32_t height,
                                            uint32_t levels,
                                            uint32_t layers,
                                            enum pipe_format pipe_format,
                                            uint64_t *out_allocation_size)
{
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);

   if (out_allocation_size)
      *out_allocation_size = 0;

   if (!resource || vk_format == VK_FORMAT_UNDEFINED ||
       yttrium_venus_format_has_depth(vk_format))
      return false;

   if (resource->initialized) {
      if (resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_allocation_size)
         *out_allocation_size = resource->allocation_size;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   usage = yttrium_venus_feedback_loop_usage(venus, usage);
   if (!yttrium_venus_create_image_tiled(venus, resource, VK_IMAGE_TYPE_2D,
                                         vk_format, width, height, 1,
                                         levels, layers, 0, usage,
                                         VK_SAMPLE_COUNT_1_BIT,
                                         VK_IMAGE_TILING_OPTIMAL,
                                         VK_IMAGE_LAYOUT_UNDEFINED))
      return false;

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetImageMemoryRequirements(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->image, &reqs);

   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, reqs.memoryTypeBits, 0,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for color attachment image bits=0x%x %ux%u format=%u req_size=0x%llx\n",
                  reqs.memoryTypeBits, width, height, pipe_format,
                  (unsigned long long)reqs.size);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = memory_type_index,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateMemory(&venus->vn_ring, 0, venus->device_handle,
                              &memory_info, NULL, &resource->memory,
                              &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateMemory(color-attachment)",
          resource->memory_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, false);
      return false;
   }

   struct vn_ring_submit_command bind_submit;
   vn_submit_vkBindImageMemory(&venus->vn_ring, 0, venus->device_handle,
                               resource->image, resource->memory, 0,
                               &bind_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &bind_submit, "vkBindImageMemory(color-attachment)",
          resource->image_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, true);
      return false;
   }

   resource->layout = VK_IMAGE_LAYOUT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_offset = 0;
   resource->image_size = reqs.size;
   resource->image_row_pitch = 0;
   resource->image_array_pitch = 0;
   resource->initialized = true;
   resource->buffer_backed = false;

   if (out_allocation_size)
      *out_allocation_size = memory_info.allocationSize;

   YTTRIUM_LOG("yttrium: Venus color attachment image memory_id=%llu image_id=%llu size=0x%llx req_size=0x%llx type=%u flags=0x%x bits=0x%x usage=0x%x extent=%ux%u levels=%u layers=%u format=%u vk_format=%u\n",
               (unsigned long long)resource->memory_obj.id,
               (unsigned long long)resource->image_obj.id,
               (unsigned long long)memory_info.allocationSize,
               (unsigned long long)reqs.size,
               memory_type_index,
               venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
               reqs.memoryTypeBits, usage, width, height,
               resource->levels, resource->layers, pipe_format, vk_format);
   return true;
}

static bool
yttrium_venus2_sampled_image_supported(struct yttrium_venus *venus,
                                       VkFormat vk_format,
                                       VkImageType image_type,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t depth,
                                       uint32_t levels,
                                       uint32_t layers,
                                       VkSampleCountFlagBits samples,
                                       VkImageCreateFlags image_flags,
                                       VkImageUsageFlags usage,
                                       VkImageTiling tiling);

bool
yttrium_venus2_create_sampled_texture_image(struct yttrium_venus *venus,
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
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);

   if (out_allocation_size)
      *out_allocation_size = 0;

   if (!resource || vk_format == VK_FORMAT_UNDEFINED ||
       yttrium_venus_format_has_depth(vk_format))
      return false;

   if (resource->initialized) {
      if (resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_allocation_size)
         *out_allocation_size = resource->allocation_size;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   VkImageType image_type;
   VkImageCreateFlags image_flags = 0;
   uint32_t image_height = MAX2(height, 1);
   uint32_t image_depth = 1;
   uint32_t image_layers = MAX2(layers, 1);
   switch (target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      image_type = VK_IMAGE_TYPE_1D;
      image_height = 1;
      break;
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
      image_type = VK_IMAGE_TYPE_2D;
      if (target == PIPE_TEXTURE_2D_ARRAY &&
          width == image_height && image_layers >= 6 && image_layers % 6 == 0)
         image_flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      break;
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      if (width != height || image_layers < 6 || image_layers % 6)
         return false;
      image_type = VK_IMAGE_TYPE_2D;
      image_flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      break;
   case PIPE_TEXTURE_3D:
      image_type = VK_IMAGE_TYPE_3D;
      image_depth = MAX2(depth, 1);
      image_layers = 1;
      break;
   default:
      return false;
   }

   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT;
   VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
   if (!yttrium_venus2_sampled_image_supported(venus, vk_format, image_type,
                                                width, image_height, image_depth,
                                                levels, image_layers,
                                                VK_SAMPLE_COUNT_1_BIT,
                                                image_flags, usage, tiling)) {
      /* Some native formats, notably RGB32 on RADV, are sampleable only as
       * linear images.  Preserve the exact format and keep the resource on
       * the GPU instead of rejecting the draw or introducing a CPU path.
       */
      tiling = VK_IMAGE_TILING_LINEAR;
      if (!yttrium_venus2_sampled_image_supported(
             venus, vk_format, image_type, width, image_height, image_depth,
             levels, image_layers, VK_SAMPLE_COUNT_1_BIT, image_flags, usage,
             tiling))
         return false;

      YTTRIUM_LOG("yttrium: Venus sampled texture uses linear tiling "
                  "target=%u extent=%ux%ux%u levels=%u layers=%u "
                  "format=%u vk_format=%u\n",
                  target, width, image_height, image_depth, levels,
                  image_layers, pipe_format, vk_format);
   }
   if (!yttrium_venus_create_image_tiled(
          venus, resource, image_type, vk_format, width, image_height,
          image_depth, levels, image_layers, image_flags, usage,
          VK_SAMPLE_COUNT_1_BIT,
          tiling, VK_IMAGE_LAYOUT_UNDEFINED))
      return false;

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetImageMemoryRequirements(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->image, &reqs);

   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, reqs.memoryTypeBits, 0,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for sampled texture image bits=0x%x target=%u %ux%ux%u format=%u req_size=0x%llx\n",
                  reqs.memoryTypeBits, target, width, image_height,
                  image_depth, pipe_format, (unsigned long long)reqs.size);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = memory_type_index,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateMemory(&venus->vn_ring, 0, venus->device_handle,
                              &memory_info, NULL, &resource->memory,
                              &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateMemory(sampled-texture)",
          resource->memory_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, false);
      return false;
   }

   struct vn_ring_submit_command bind_submit;
   vn_submit_vkBindImageMemory(&venus->vn_ring, 0, venus->device_handle,
                               resource->image, resource->memory, 0,
                               &bind_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &bind_submit, "vkBindImageMemory(sampled-texture)",
          resource->image_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, true);
      return false;
   }

   resource->layout = VK_IMAGE_LAYOUT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_offset = 0;
   resource->image_size = reqs.size;
   resource->image_row_pitch = 0;
   resource->image_array_pitch = 0;
   resource->initialized = true;
   resource->buffer_backed = false;

   if (out_allocation_size)
      *out_allocation_size = memory_info.allocationSize;

   return true;
}

static bool
yttrium_venus_texture_target_params(enum pipe_texture_target target,
                                    uint32_t height,
                                    uint32_t depth,
                                    uint32_t layers,
                                    VkImageType *image_type,
                                    uint32_t *image_height,
                                    uint32_t *image_depth,
                                    uint32_t *image_layers)
{
   switch (target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      *image_type = VK_IMAGE_TYPE_1D;
      *image_height = 1;
      *image_depth = 1;
      *image_layers = MAX2(layers, 1);
      return true;
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_RECT:
      *image_type = VK_IMAGE_TYPE_2D;
      *image_height = MAX2(height, 1);
      *image_depth = 1;
      *image_layers = MAX2(layers, 1);
      return true;
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      *image_type = VK_IMAGE_TYPE_2D;
      *image_height = MAX2(height, 1);
      *image_depth = 1;
      *image_layers = MAX2(layers, 1);
      return true;
   case PIPE_TEXTURE_3D:
      *image_type = VK_IMAGE_TYPE_3D;
      *image_height = MAX2(height, 1);
      *image_depth = MAX2(depth, 1);
      *image_layers = 1;
      return true;
   default:
      return false;
   }
}

static VkImageCreateFlags
yttrium_venus_image_flags_for_target(enum pipe_texture_target target,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t layers)
{
   switch (target) {
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      if (width != height || layers < 6 || layers % 6)
         return 0;
      return VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   case PIPE_TEXTURE_2D_ARRAY:
      if (width == height && layers >= 6 && layers % 6 == 0)
         return VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      return 0;
   case PIPE_TEXTURE_3D:
      return VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
   default:
      return 0;
   }
}

static VkImageUsageFlags
yttrium_venus_image_usage_for_bind(const struct yttrium_venus *venus,
                                   VkFormat vk_format, unsigned bind)
{
   const bool depth_format = yttrium_venus_format_has_depth(vk_format);
   VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT;

   if (bind & PIPE_BIND_SAMPLER_VIEW)
      usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

   if (bind & PIPE_BIND_SHADER_IMAGE)
      usage |= VK_IMAGE_USAGE_STORAGE_BIT;

   if (vk_format == VK_FORMAT_R16_UINT &&
       (bind & PIPE_BIND_RENDER_TARGET))
      usage |= VK_IMAGE_USAGE_STORAGE_BIT;

   if (bind & PIPE_BIND_DEPTH_STENCIL) {
      if (!depth_format)
         return 0;
      usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   }

   if (bind & PIPE_BIND_RENDER_TARGET) {
      if (depth_format)
         return 0;
      usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   }

   return yttrium_venus_feedback_loop_usage(venus, usage);
}

static bool
yttrium_venus2_sampled_image_supported(struct yttrium_venus *venus,
                                       VkFormat vk_format,
                                       VkImageType image_type,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t depth,
                                       uint32_t levels,
                                       uint32_t layers,
                                       VkSampleCountFlagBits samples,
                                       VkImageCreateFlags image_flags,
                                       VkImageUsageFlags usage,
                                       VkImageTiling tiling)
{
   const VkFormatFeatureFlags required_features =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   VkFormatProperties2 format_props = {
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
   };
   const VkPhysicalDeviceImageFormatInfo2 image_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = vk_format,
      .type = image_type,
      .tiling = tiling,
      .usage = usage,
      .flags = image_flags | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
   };
   VkImageFormatProperties2 image_props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
   };

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   vn_call_vkGetPhysicalDeviceFormatProperties2(
      &venus->vn_ring, venus->physical_device, vk_format, &format_props);
   const VkFormatFeatureFlags tiling_features =
      tiling == VK_IMAGE_TILING_LINEAR
         ? format_props.formatProperties.linearTilingFeatures
         : format_props.formatProperties.optimalTilingFeatures;
   if ((tiling_features & required_features) != required_features)
      return false;

   if (vn_call_vkGetPhysicalDeviceImageFormatProperties2(
          &venus->vn_ring, venus->physical_device, &image_info,
          &image_props) != VK_SUCCESS)
      return false;

   const VkImageFormatProperties *props = &image_props.imageFormatProperties;
   return width && height && depth && levels && layers &&
      width <= props->maxExtent.width &&
      height <= props->maxExtent.height &&
      depth <= props->maxExtent.depth &&
      levels <= props->maxMipLevels &&
      layers <= props->maxArrayLayers &&
      (props->sampleCounts & samples);
}

bool
yttrium_venus2_sampled_texture_format_supported(
   struct yttrium_venus *venus,
   enum pipe_format pipe_format,
   enum pipe_texture_target target)
{
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);
   VkImageType image_type;
   uint32_t image_height;
   uint32_t image_depth;
   uint32_t image_layers;
   VkImageCreateFlags image_flags = 0;

   if (vk_format == VK_FORMAT_UNDEFINED ||
       yttrium_venus_format_has_depth(vk_format) ||
       !yttrium_venus_texture_target_params(target, 1, 1, 1,
                                            &image_type, &image_height,
                                            &image_depth, &image_layers) ||
       !image_height || !image_depth || !image_layers)
      return false;

   if (target == PIPE_TEXTURE_CUBE || target == PIPE_TEXTURE_CUBE_ARRAY)
      image_flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

   return yttrium_venus2_sampled_image_supported(
      venus, vk_format, image_type, 1, image_height,
      target == PIPE_TEXTURE_3D ? 2 : image_depth,
      1, image_layers, VK_SAMPLE_COUNT_1_BIT, image_flags,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TILING_OPTIMAL);
}

bool
yttrium_venus2_create_texture_image_for_bind(struct yttrium_venus *venus,
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
   const VkFormat vk_format =
      yttrium_venus_pipe_format_for_bind(pipe_format, bind);

   if (out_allocation_size)
      *out_allocation_size = 0;

   if (!resource || vk_format == VK_FORMAT_UNDEFINED)
      return false;

   if (resource->initialized) {
      if (resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_allocation_size)
         *out_allocation_size = resource->allocation_size;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   VkImageType image_type;
   uint32_t image_height;
   uint32_t image_depth;
   uint32_t image_layers;
   if (!yttrium_venus_texture_target_params(target, height, depth, layers,
                                            &image_type, &image_height,
                                            &image_depth, &image_layers))
      return false;

   const VkImageUsageFlags usage =
      yttrium_venus_image_usage_for_bind(venus, vk_format, bind);
   if (!usage)
      return false;

   VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
   if (!yttrium_venus_sample_count(sample_count, &samples))
      return false;
   if (samples != VK_SAMPLE_COUNT_1_BIT &&
       !(venus->framebuffer_sample_counts & samples))
      return false;

   VkImageCreateFlags image_flags =
      yttrium_venus_image_flags_for_target(target, width, image_height,
                                           image_layers);
   if ((target == PIPE_TEXTURE_CUBE || target == PIPE_TEXTURE_CUBE_ARRAY) &&
       !image_flags)
      return false;

   /* Shared single-sample colour images use linear tiling and ALIAS so
    * yttrium_venus2_import_texture_image_for_bind and the win32 WSI
    * DXGI_SHARED path can reproduce their memory layout.  Multisampled and
    * depth images retain optimal tiling.
    */
   const bool shared_colour = (bind & PIPE_BIND_SHARED) &&
                              !yttrium_venus_format_has_depth(vk_format) &&
                              samples == VK_SAMPLE_COUNT_1_BIT;

   const VkImageTiling tiling = shared_colour ? VK_IMAGE_TILING_LINEAR
                                             : VK_IMAGE_TILING_OPTIMAL;
   if (shared_colour)
      image_flags |= VK_IMAGE_CREATE_ALIAS_BIT;

   if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
       !yttrium_venus2_sampled_image_supported(
          venus, vk_format, image_type, width, image_height, image_depth,
          levels, image_layers, samples, image_flags, usage, tiling))
      return false;

   if (!yttrium_venus_create_image_tiled(
          venus, resource, image_type, vk_format, width, image_height,
          image_depth, levels, image_layers, image_flags, usage, samples,
          tiling, VK_IMAGE_LAYOUT_UNDEFINED))
      return false;

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetImageMemoryRequirements(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->image, &reqs);
   /* Never allocate or bind zero-sized backing.  Changing tiling here would
    * break the shared image layout agreement with the importer.
    */
   if (!reqs.size) {
      YTTRIUM_WARN("yttrium: Venus zero memory requirement for image %ux%ux%u "
                   "format=%u vk_format=%u tiling=%u usage=0x%x "
                   "flags=0x%x samples=0x%x bind=0x%x; refusing allocation\n",
                   width, image_height, image_depth, pipe_format, vk_format,
                   tiling, usage, image_flags, samples, bind);

      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, reqs.memoryTypeBits, 0,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for bind texture image bits=0x%x target=%u %ux%ux%u levels=%u layers=%u format=%u bind=0x%x req_size=0x%llx usage=0x%x\n",
                  reqs.memoryTypeBits, target, width, image_height,
                  image_depth, MAX2(levels, 1), image_layers, pipe_format,
                  bind, (unsigned long long)reqs.size, usage);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   VkExportMemoryAllocateInfo export_info =
      yttrium_venus_export_memory_info();
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = (bind & PIPE_BIND_SHARED) ? &export_info : NULL,
      .allocationSize = reqs.size,
      .memoryTypeIndex = memory_type_index,
   };
   if (bind & PIPE_BIND_SHARED) {
      /* KMD may open the exported blob immediately after this returns.  Keep
       * allocate/bind synchronous so the host renderer has materialized it.
       */
      VkResult result =
         vn_call_vkAllocateMemory(&venus->vn_ring, venus->device_handle,
                                  &memory_info, NULL, &resource->memory);
      if (result != VK_SUCCESS) {
         YTTRIUM_WARN("yttrium: Venus vkAllocateMemory shared bind texture image failed result=%d alloc_size=0x%llx target=%u type=%u bits=0x%x flags=0x%x usage=0x%x bind=0x%x\n",
                     result,
                     (unsigned long long)memory_info.allocationSize,
                     target, image_type, reqs.memoryTypeBits,
                     venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
                     usage, bind);
         vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                                 resource->image, NULL);
         memset(resource, 0, sizeof(*resource));
         return false;
      }

      result = vn_call_vkBindImageMemory(&venus->vn_ring,
                                         venus->device_handle,
                                         resource->image,
                                         resource->memory, 0);
      if (result != VK_SUCCESS) {
         YTTRIUM_WARN("yttrium: Venus vkBindImageMemory shared bind texture image failed result=%d memory_id=%llu image_id=%llu target=%u bind=0x%x\n",
                     result,
                     (unsigned long long)resource->memory_obj.id,
                     (unsigned long long)resource->image_obj.id,
                     target, bind);
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               resource->memory, NULL);
         vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                                 resource->image, NULL);
         memset(resource, 0, sizeof(*resource));
         return false;
      }
   } else {
      struct vn_ring_submit_command allocate_submit;
      vn_submit_vkAllocateMemory(&venus->vn_ring, 0,
                                 venus->device_handle, &memory_info, NULL,
                                 &resource->memory, &allocate_submit);
      if (!yttrium_venus_async_submit_succeeded(
             venus, &allocate_submit,
             "vkAllocateMemory(bind-texture-image)",
             resource->memory_obj.id)) {
         yttrium_venus_discard_accepted_image(venus, resource, false);
         return false;
      }

      struct vn_ring_submit_command bind_submit;
      vn_submit_vkBindImageMemory(&venus->vn_ring, 0,
                                  venus->device_handle, resource->image,
                                  resource->memory, 0, &bind_submit);
      if (!yttrium_venus_async_submit_succeeded(
             venus, &bind_submit, "vkBindImageMemory(bind-texture-image)",
             resource->image_obj.id)) {
         yttrium_venus_discard_accepted_image(venus, resource, true);
         return false;
      }
   }

   resource->layout = VK_IMAGE_LAYOUT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_offset = 0;
   resource->image_size = reqs.size;
   resource->image_row_pitch = 0;
   resource->image_array_pitch = 0;
   resource->initialized = true;
   resource->buffer_backed = false;

   if (out_allocation_size)
      *out_allocation_size = memory_info.allocationSize;

   YTTRIUM_LOG("yttrium: Venus bind texture image memory_id=%llu image_id=%llu size=0x%llx req_size=0x%llx target=%u type=%u flags=0x%x bits=0x%x usage=0x%x bind=0x%x extent=%ux%ux%u levels=%u layers=%u format=%u vk_format=%u\n",
               (unsigned long long)resource->memory_obj.id,
               (unsigned long long)resource->image_obj.id,
               (unsigned long long)memory_info.allocationSize,
               (unsigned long long)reqs.size,
               target, image_type,
               venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
               reqs.memoryTypeBits, usage, bind, width, image_height,
               image_depth, resource->levels, resource->layers,
               pipe_format, vk_format);
   return true;
}

bool
yttrium_venus2_create_depth_stencil_image(struct yttrium_venus *venus,
                                         struct yttrium_venus_resource *resource,
                                         uint32_t width,
                                         uint32_t height,
                                         enum pipe_format pipe_format,
                                         uint64_t *out_allocation_size)
{
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);

   if (out_allocation_size)
      *out_allocation_size = 0;

   if (!resource || vk_format == VK_FORMAT_UNDEFINED ||
       !yttrium_venus_format_has_depth(vk_format))
      return false;

   if (resource->initialized) {
      if (resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_allocation_size)
         *out_allocation_size = resource->allocation_size;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   if (!yttrium_venus_create_image_tiled(venus, resource, VK_IMAGE_TYPE_2D,
                                         vk_format, width, height, 1,
                                         1, 1, 0, usage,
                                         VK_SAMPLE_COUNT_1_BIT,
                                         VK_IMAGE_TILING_OPTIMAL,
                                         VK_IMAGE_LAYOUT_UNDEFINED))
      return false;

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetImageMemoryRequirements(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->image, &reqs);

   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, reqs.memoryTypeBits, 0,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for depth image bits=0x%x %ux%u format=%u req_size=0x%llx\n",
                  reqs.memoryTypeBits, width, height, pipe_format,
                  (unsigned long long)reqs.size);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = memory_type_index,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateMemory(&venus->vn_ring, 0, venus->device_handle,
                              &memory_info, NULL, &resource->memory,
                              &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateMemory(depth-stencil)",
          resource->memory_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, false);
      return false;
   }

   struct vn_ring_submit_command bind_submit;
   vn_submit_vkBindImageMemory(&venus->vn_ring, 0, venus->device_handle,
                               resource->image, resource->memory, 0,
                               &bind_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &bind_submit, "vkBindImageMemory(depth-stencil)",
          resource->image_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, true);
      return false;
   }

   resource->layout = VK_IMAGE_LAYOUT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->initialized = true;
   resource->buffer_backed = false;

   if (out_allocation_size)
      *out_allocation_size = memory_info.allocationSize;

   YTTRIUM_LOG("yttrium: Venus depth image memory_id=%llu image_id=%llu size=0x%llx req_size=0x%llx type=%u flags=0x%x bits=0x%x usage=0x%x extent=%ux%u format=%u vk_format=%u\n",
               (unsigned long long)resource->memory_obj.id,
               (unsigned long long)resource->image_obj.id,
               (unsigned long long)memory_info.allocationSize,
               (unsigned long long)reqs.size,
               memory_type_index,
               venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
               reqs.memoryTypeBits, usage, width, height, pipe_format,
               vk_format);
   return true;
}

bool
yttrium_venus2_import_display_image(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t resource_id,
                                   uint32_t width,
                                   uint32_t height,
                                   enum pipe_format pipe_format,
                                   uint64_t min_allocation_size,
                                   uint64_t *out_memory_id,
                                   uint64_t *out_allocation_size)
{
   VkResult result;
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);

   if (out_memory_id)
      *out_memory_id = 0;
   if (out_allocation_size)
      *out_allocation_size = 0;

   if (!resource || !resource_id || vk_format == VK_FORMAT_UNDEFINED)
      return false;

   if (resource->initialized) {
      if (resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_memory_id)
         *out_memory_id = resource->memory_obj.id;
      if (out_allocation_size)
         *out_allocation_size = resource->allocation_size;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   usage = yttrium_venus_feedback_loop_usage(venus, usage);
   if (!yttrium_venus_create_image_tiled(
          venus, resource, VK_IMAGE_TYPE_2D, vk_format, width, height, 1,
          1, 1, 0, usage,
          VK_SAMPLE_COUNT_1_BIT,
          VK_IMAGE_TILING_LINEAR, VK_IMAGE_LAYOUT_PREINITIALIZED)) {
      usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      if (!yttrium_venus_create_image_tiled(
             venus, resource, VK_IMAGE_TYPE_2D, vk_format, width, height, 1,
             1, 1, 0, usage,
             VK_SAMPLE_COUNT_1_BIT,
             VK_IMAGE_TILING_LINEAR, VK_IMAGE_LAYOUT_PREINITIALIZED))
         return false;
   }

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetImageMemoryRequirements(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->image, &reqs);

   VkMemoryResourcePropertiesMESA resource_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_RESOURCE_PROPERTIES_MESA,
   };
   result = vn_call_vkGetMemoryResourcePropertiesMESA(&venus->vn_ring,
                                                      venus->device_handle,
                                                      resource_id,
                                                      &resource_props);
   if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus vkGetMemoryResourcePropertiesMESA failed result=%d res_id=%u\n",
                   result, resource_id);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   const uint32_t memory_type_bits =
      reqs.memoryTypeBits & resource_props.memoryTypeBits;
   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, memory_type_bits,
                                       0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no compatible memory type for imported display image res_id=%u image_bits=0x%x resource_bits=0x%x %ux%u format=%u req_size=0x%llx\n",
                   resource_id, reqs.memoryTypeBits,
                   resource_props.memoryTypeBits, width, height, pipe_format,
                   (unsigned long long)reqs.size);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   VkImportMemoryResourceInfoMESA import_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA,
      .resourceId = resource_id,
   };
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import_info,
      .allocationSize = MAX2(reqs.size, (VkDeviceSize)min_allocation_size),
      .memoryTypeIndex = memory_type_index,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateMemory(&venus->vn_ring, 0, venus->device_handle,
                              &memory_info, NULL, &resource->memory,
                              &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateMemory(import-display-image)",
          resource->memory_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, false);
      return false;
   }

   struct vn_ring_submit_command bind_submit;
   vn_submit_vkBindImageMemory(&venus->vn_ring, 0, venus->device_handle,
                               resource->image, resource->memory, 0,
                               &bind_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &bind_submit, "vkBindImageMemory(import-display-image)",
          resource->image_obj.id)) {
      yttrium_venus_discard_accepted_image(venus, resource, true);
      return false;
   }

   const VkImageSubresource subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .arrayLayer = 0,
   };
   VkSubresourceLayout layout;
   memset(&layout, 0, sizeof(layout));
   if (!yttrium_venus_get_image_subresource_layout_checked(
          venus, resource, &subresource, &layout,
          "import-display-image")) {
      yttrium_venus_discard_accepted_bound_image(venus, resource);
      return false;
   }

   resource->layout = VK_IMAGE_LAYOUT_PREINITIALIZED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_offset = layout.offset;
   resource->image_size = layout.size;
   resource->image_row_pitch = layout.rowPitch;
   resource->image_array_pitch = layout.arrayPitch;
   resource->initialized = true;
   resource->buffer_backed = false;
   /* This image aliases an existing presented allocation.  DWM may import it
    * after the producer has already rendered the frame, so treating it as
    * uninitialized would clear the producer's contents before sampling.
    */
   resource->contents_initialized = true;

   if (out_memory_id)
      *out_memory_id = resource->memory_obj.id;
   if (out_allocation_size)
      *out_allocation_size = memory_info.allocationSize;

   YTTRIUM_LOG("yttrium: Venus imported display image res_id=%u memory_id=%llu image_id=%llu size=0x%llx req_size=0x%llx type=%u flags=0x%x image_bits=0x%x resource_bits=0x%x usage=0x%x layout_offset=0x%llx layout_size=0x%llx row_pitch=%llu\n",
                resource_id,
                (unsigned long long)resource->memory_obj.id,
                (unsigned long long)resource->image_obj.id,
                (unsigned long long)memory_info.allocationSize,
                (unsigned long long)reqs.size,
                memory_type_index,
                venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
                reqs.memoryTypeBits, resource_props.memoryTypeBits, usage,
                (unsigned long long)layout.offset,
                (unsigned long long)layout.size,
                (unsigned long long)layout.rowPitch);
   return true;
}

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
   uint64_t *out_allocation_size)
{
   VkResult result;
   const VkFormat vk_format =
      yttrium_venus_pipe_format_for_bind(pipe_format, bind);

   if (out_memory_id)
      *out_memory_id = 0;
   if (out_allocation_size)
      *out_allocation_size = 0;

   if (!resource || !resource_id || vk_format == VK_FORMAT_UNDEFINED)
      return false;

   if (resource->initialized) {
      if (resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_memory_id)
         *out_memory_id = resource->memory_obj.id;
      if (out_allocation_size)
         *out_allocation_size = resource->allocation_size;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   VkImageType image_type;
   uint32_t image_height;
   uint32_t image_depth;
   uint32_t image_layers;
   if (!yttrium_venus_texture_target_params(target, height, depth, layers,
                                            &image_type, &image_height,
                                            &image_depth, &image_layers))
      return false;

   const VkImageUsageFlags usage =
      yttrium_venus_image_usage_for_bind(venus, vk_format, bind);
   if (!usage)
      return false;

   VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
   if (!yttrium_venus_sample_count(sample_count, &samples))
      return false;
   if (samples != VK_SAMPLE_COUNT_1_BIT &&
       !(venus->framebuffer_sample_counts & samples))
      return false;

   VkImageCreateFlags image_flags =
      yttrium_venus_image_flags_for_target(target, width, image_height,
                                           image_layers);
   if ((target == PIPE_TEXTURE_CUBE || target == PIPE_TEXTURE_CUBE_ARRAY) &&
       !image_flags)
      return false;

   /* Match the shared colour image created by
    * yttrium_venus2_create_texture_image_for_bind.  Importing its linear
    * backing through an optimal-tiled image changes texel addressing.
    */
   const bool shared_colour = (bind & PIPE_BIND_SHARED) &&
                              !yttrium_venus_format_has_depth(vk_format) &&
                              samples == VK_SAMPLE_COUNT_1_BIT;
   const VkImageTiling tiling = shared_colour ? VK_IMAGE_TILING_LINEAR
                                             : VK_IMAGE_TILING_OPTIMAL;
   if (shared_colour)
      image_flags |= VK_IMAGE_CREATE_ALIAS_BIT;

   if (!yttrium_venus_create_image_tiled(
          venus, resource, image_type, vk_format, width, image_height,
          image_depth, levels, image_layers, image_flags, usage, samples,
          tiling, VK_IMAGE_LAYOUT_UNDEFINED))
      return false;

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetImageMemoryRequirements(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->image, &reqs);

   VkMemoryResourcePropertiesMESA resource_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_RESOURCE_PROPERTIES_MESA,
   };
   result = vn_call_vkGetMemoryResourcePropertiesMESA(&venus->vn_ring,
                                                      venus->device_handle,
                                                      resource_id,
                                                      &resource_props);
   if (result != VK_SUCCESS) {
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   const uint32_t memory_type_bits =
      reqs.memoryTypeBits & resource_props.memoryTypeBits;
   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, memory_type_bits,
                                       0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   VkImportMemoryResourceInfoMESA import_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA,
      .resourceId = resource_id,
   };
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import_info,
      .allocationSize = MAX2(reqs.size, (VkDeviceSize)min_allocation_size),
      .memoryTypeIndex = memory_type_index,
   };
   result = vn_call_vkAllocateMemory(&venus->vn_ring, venus->device_handle,
                                     &memory_info, NULL, &resource->memory);
   if (result != VK_SUCCESS) {
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   result = vn_call_vkBindImageMemory(&venus->vn_ring, venus->device_handle,
                                      resource->image, resource->memory, 0);
   if (result != VK_SUCCESS) {
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->memory, NULL);
      vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                              resource->image, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   resource->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_offset = 0;
   resource->image_size = reqs.size;
   resource->image_row_pitch = 0;
   resource->image_array_pitch = 0;
   resource->initialized = true;
   resource->buffer_backed = false;
   resource->contents_initialized = true;

   if (out_memory_id)
      *out_memory_id = resource->memory_obj.id;
   if (out_allocation_size)
      *out_allocation_size = memory_info.allocationSize;

   return true;
}

bool
yttrium_venus2_create_display_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint64_t allocation_size,
                                    uint64_t *out_memory_id)
{
   if (out_memory_id)
      *out_memory_id = 0;

   if (!resource)
      return false;

   if (resource->initialized) {
      if (!resource->buffer_backed || !resource->memory_obj.id)
         return false;
      if (out_memory_id)
         *out_memory_id = resource->memory_obj.id;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   yttrium_venus_init_object(venus, &resource->buffer_obj);
   resource->buffer = YTTRIUM_VENUS_HANDLE(VkBuffer, &resource->buffer_obj);

   const VkDeviceSize buffer_size = MAX2((VkDeviceSize)allocation_size,
                                         (VkDeviceSize)1);
   const VkExternalMemoryBufferCreateInfo external_buffer_info =
      yttrium_venus_external_buffer_info();
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = &external_buffer_info,
      .size = buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };

   struct vn_ring_submit_command create_submit;
   vn_submit_vkCreateBuffer(&venus->vn_ring, 0, venus->device_handle,
                            &buffer_info, NULL, &resource->buffer,
                            &create_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &create_submit, "vkCreateBuffer(display-buffer)",
          resource->buffer_obj.id)) {
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         resource->buffer, &reqs);

   uint32_t memory_type_index = UINT32_MAX;
   if (resource->cpu_readback) {
      /*
       * Readback buffers are consumed by the CPU after the GPU copy.  A
       * merely coherent exported allocation is reported by virglrenderer as
       * write-combined, making sequential CPU reads pathologically slow.
       * Requiring CACHED as well as COHERENT gives the KMD a legitimate
       * write-back mapping and avoids any explicit invalidate requirement.
       */
      memory_type_index = yttrium_venus_choose_memory_type(
         venus, reqs.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
         0);
      if (memory_type_index == UINT32_MAX) {
         YTTRIUM_WARN("yttrium: WARNING: readback cache fallback owner=venus2 reason=no_host_cached_coherent_memory fallback=host_visible_coherent bits=0x%x size=0x%llx\n",
                      reqs.memoryTypeBits,
                      (unsigned long long)buffer_size);
      }
   }
   if (memory_type_index == UINT32_MAX) {
      memory_type_index =
         yttrium_venus_choose_memory_type(
            venus, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   }
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no host-visible memory type for display buffer bits=0x%x size=0x%llx\n",
                   reqs.memoryTypeBits, (unsigned long long)buffer_size);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   VkExportMemoryAllocateInfo export_info =
      yttrium_venus_export_memory_info();
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export_info,
      .allocationSize = MAX2(reqs.size, buffer_size),
      .memoryTypeIndex = memory_type_index,
   };
   /* KMD may create/open the blob mapping immediately after this returns.  Keep
    * allocate/bind synchronous so Venus has exported a valid host object.
    */
   VkResult result =
      vn_call_vkAllocateMemory(&venus->vn_ring, venus->device_handle,
                               &memory_info, NULL, &resource->memory);
   if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus vkAllocateMemory display buffer failed result=%d alloc_size=0x%llx type=%u bits=0x%x flags=0x%x\n",
                  result,
                  (unsigned long long)memory_info.allocationSize,
                  memory_type_index, reqs.memoryTypeBits,
                  venus->memory_props.memoryTypes[memory_type_index].propertyFlags);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   result = vn_call_vkBindBufferMemory(&venus->vn_ring,
                                       venus->device_handle,
                                       resource->buffer,
                                       resource->memory, 0);
   if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus vkBindBufferMemory display failed result=%d memory_id=%llu\n",
                  result,
                  (unsigned long long)resource->memory_obj.id);
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->memory, NULL);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   resource->initialized = true;
   resource->buffer_backed = true;
   resource->buffer_usage = buffer_info.usage;
   resource->vk_format = VK_FORMAT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_size = buffer_size;

   if (out_memory_id)
      *out_memory_id = resource->memory_obj.id;

   YTTRIUM_LOG("yttrium: Venus display buffer memory_id=%llu buffer_id=%llu size=0x%llx req_size=0x%llx type=%u flags=0x%x bits=0x%x cpu_readback=%u\n",
                (unsigned long long)resource->memory_obj.id,
                (unsigned long long)resource->buffer_obj.id,
                (unsigned long long)buffer_size,
                (unsigned long long)reqs.size,
                memory_type_index,
                venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
                reqs.memoryTypeBits, resource->cpu_readback);
   return true;
}

bool
yttrium_venus2_transform_feedback_enabled(const struct yttrium_venus *venus)
{
   return venus && venus->initialized && venus->transform_feedback;
}

bool
yttrium_venus2_supports_multisampled_render_to_single_sampled(
   struct yttrium_venus *venus,
   uint32_t sample_count)
{
   VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

   if (!yttrium_venus_ensure_initialized(venus))
      return false;
   if (!yttrium_venus_sample_count(sample_count, &samples))
      return false;

   return venus->multisampled_render_to_single_sampled &&
      samples != VK_SAMPLE_COUNT_1_BIT &&
      (venus->framebuffer_sample_counts & samples);
}

bool
yttrium_venus2_supports_forced_sample_interlock(
   struct yttrium_venus *venus,
   uint32_t sample_count)
{
   VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

   if (!yttrium_venus_ensure_initialized(venus))
      return false;
   if (!yttrium_venus_sample_count(sample_count, &samples))
      return false;

   return venus->fragment_shader_pixel_interlock &&
      venus->fragment_stores_and_atomics &&
      samples != VK_SAMPLE_COUNT_1_BIT &&
      (venus->framebuffer_no_attachments_sample_counts & samples);
}

VkSampleCountFlags
yttrium_venus2_framebuffer_color_sample_counts(struct yttrium_venus *venus)
{
   return yttrium_venus_ensure_physical_device(venus) ?
      venus->framebuffer_sample_counts : 0;
}

static bool
yttrium_venus2_create_device_local_draw_mirror(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   VkDeviceSize buffer_size,
   VkBufferUsageFlags source_usage)
{
   const VkBufferUsageFlags draw_usage =
      source_usage & (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
   if (!draw_usage)
      return true;

   yttrium_venus_init_object(venus,
                              &resource->device_local_draw_buffer_obj);
   resource->device_local_draw_buffer =
      YTTRIUM_VENUS_HANDLE(
         VkBuffer, &resource->device_local_draw_buffer_obj);
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | draw_usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   struct vn_ring_submit_command create_submit;
   vn_submit_vkCreateBuffer(&venus->vn_ring, 0, venus->device_handle,
                            &buffer_info, NULL,
                            &resource->device_local_draw_buffer,
                            &create_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &create_submit, "vkCreateBuffer(device-local-draw-mirror)",
          resource->device_local_draw_buffer_obj.id)) {
      memset(&resource->device_local_draw_buffer_obj, 0,
             sizeof(resource->device_local_draw_buffer_obj));
      resource->device_local_draw_buffer = VK_NULL_HANDLE;
      return false;
   }

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(
      &venus->vn_ring, venus->device_handle,
      resource->device_local_draw_buffer, &reqs);

   /* Prefer real local VRAM over a host-visible BAR memory type. */
   uint32_t memory_type_index = UINT32_MAX;
   for (uint32_t i = 0; i < venus->memory_props.memoryTypeCount; i++) {
      if (!(reqs.memoryTypeBits & (1u << i)))
         continue;

      const VkMemoryPropertyFlags flags =
         venus->memory_props.memoryTypes[i].propertyFlags;
      if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
          !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
         memory_type_index = i;
         break;
      }
   }
   if (memory_type_index == UINT32_MAX) {
      memory_type_index = yttrium_venus_choose_memory_type(
         venus, reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
   }
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_WARN("yttrium: WARNING: device-local draw mirror unavailable owner=venus2 reason=no-device-local-memory-type action=use-host-visible-source bits=0x%x size=0x%llx usage=0x%x\n",
                   reqs.memoryTypeBits,
                   (unsigned long long)buffer_size, draw_usage);
      goto fail_buffer;
   }

   yttrium_venus_init_object(venus,
                              &resource->device_local_draw_memory_obj);
   resource->device_local_draw_memory =
      YTTRIUM_VENUS_HANDLE(
         VkDeviceMemory, &resource->device_local_draw_memory_obj);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = MAX2(reqs.size, buffer_size),
      .memoryTypeIndex = memory_type_index,
   };
   VkResult result = vn_call_vkAllocateMemory(
      &venus->vn_ring, venus->device_handle, &memory_info, NULL,
      &resource->device_local_draw_memory);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: WARNING: device-local draw mirror unavailable owner=venus2 reason=vkAllocateMemory-failed result=%d action=use-host-visible-source type=%u flags=0x%x bits=0x%x size=0x%llx usage=0x%x\n",
                   result, memory_type_index,
                   venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
                   reqs.memoryTypeBits,
                   (unsigned long long)memory_info.allocationSize, draw_usage);
      goto fail_buffer;
   }

   result = vn_call_vkBindBufferMemory(
      &venus->vn_ring, venus->device_handle,
      resource->device_local_draw_buffer,
      resource->device_local_draw_memory, 0);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: WARNING: device-local draw mirror unavailable owner=venus2 reason=vkBindBufferMemory-failed result=%d action=use-host-visible-source type=%u size=0x%llx usage=0x%x\n",
                   result, memory_type_index,
                   (unsigned long long)memory_info.allocationSize, draw_usage);
      goto fail_memory;
   }

   resource->device_local_draw_buffer_size = buffer_size;
   return true;

fail_memory:
   vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                         resource->device_local_draw_memory, NULL);
fail_buffer:
   vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                            resource->device_local_draw_buffer, NULL);
   memset(&resource->device_local_draw_buffer_obj, 0,
          sizeof(resource->device_local_draw_buffer_obj));
   memset(&resource->device_local_draw_memory_obj, 0,
          sizeof(resource->device_local_draw_memory_obj));
   resource->device_local_draw_buffer = VK_NULL_HANDLE;
   resource->device_local_draw_memory = VK_NULL_HANDLE;
   resource->device_local_draw_buffer_size = 0;
   return false;
}

bool
yttrium_venus2_create_bind_buffer(struct yttrium_venus *venus,
                                  struct yttrium_venus_resource *resource,
                                  uint64_t allocation_size,
                                  VkBufferUsageFlags usage,
                                  uint64_t *out_memory_id)
{
   const bool exportable = out_memory_id != NULL;

   if (out_memory_id)
      *out_memory_id = 0;

   if (!resource)
      return false;

   usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;

   if (resource->initialized) {
      if (!resource->buffer_backed || !resource->memory_obj.id ||
          !resource->buffer || (resource->buffer_usage & usage) != usage)
         return false;
      if (out_memory_id)
         *out_memory_id = resource->memory_obj.id;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   yttrium_venus_init_object(venus, &resource->buffer_obj);
   resource->buffer = YTTRIUM_VENUS_HANDLE(VkBuffer, &resource->buffer_obj);

   const VkDeviceSize buffer_size = MAX2((VkDeviceSize)allocation_size,
                                         (VkDeviceSize)1);
   const VkExternalMemoryBufferCreateInfo external_buffer_info =
      yttrium_venus_external_buffer_info();
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = &external_buffer_info,
      .size = buffer_size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };

   struct vn_ring_submit_command create_submit;
   vn_submit_vkCreateBuffer(&venus->vn_ring, 0, venus->device_handle,
                            &buffer_info, NULL, &resource->buffer,
                            &create_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &create_submit, "vkCreateBuffer(bind-buffer)",
          resource->buffer_obj.id)) {
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         resource->buffer, &reqs);

   /*
    * For exported buffers, prefer HOST_CACHED as well as HOST_COHERENT.
    * virglrenderer derives the
    * blob's map_info from these very bits - `(coherent && cached) ? CACHED :
    * WC` in vkr_device_memory.c - and the guest maps the blob with whatever it
    * reports.  Without HOST_CACHED we are handed write-combined memory, where
    * CPU reads are uncached: the draw-time index scan measured ~142 ns per
    * element against well under a nanosecond from cached memory.  Asking for a
    * cacheable memory type gets a write-back mapping legitimately, with the
    * host using a cached coherent view of the same pages - unlike overriding
    * the reported type in the KMD, which mismatches the guest and host
    * attributes and corrupts everything the host reads (tried 2026-08-01).
    * Preferred, not required, so this degrades to the previous behaviour when
    * no such memory type exists.
    */
   uint32_t memory_type_index = UINT32_MAX;
   if (exportable) {
      memory_type_index =
         yttrium_venus_choose_memory_type(
            venus, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
   } else {
      /*
       * A private immutable draw buffer is populated through Vulkan transfer
       * commands.  Prefer real VRAM and deliberately omit export metadata so
       * RADV does not place it on the global external-memory BO list.
       */
      for (uint32_t i = 0; i < venus->memory_props.memoryTypeCount; i++) {
         if (!(reqs.memoryTypeBits & (1u << i)))
            continue;

         const VkMemoryPropertyFlags flags =
            venus->memory_props.memoryTypes[i].propertyFlags;
         if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
             !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memory_type_index = i;
            break;
         }
      }
      if (memory_type_index == UINT32_MAX) {
         memory_type_index = yttrium_venus_choose_memory_type(
            venus, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
      }
   }
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_WARN("yttrium: Venus no compatible memory type for bind buffer bits=0x%x size=0x%llx usage=0x%x\n",
                   reqs.memoryTypeBits, (unsigned long long)buffer_size,
                   usage);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   VkExportMemoryAllocateInfo export_info =
      yttrium_venus_export_memory_info();
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = exportable ? &export_info : NULL,
      .allocationSize = MAX2(reqs.size, buffer_size),
      .memoryTypeIndex = memory_type_index,
   };
   VkResult result =
      vn_call_vkAllocateMemory(&venus->vn_ring, venus->device_handle,
                               &memory_info, NULL, &resource->memory);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: Venus vkAllocateMemory bind buffer failed result=%d alloc_size=0x%llx type=%u bits=0x%x flags=0x%x usage=0x%x\n",
                   result,
                   (unsigned long long)memory_info.allocationSize,
                   memory_type_index, reqs.memoryTypeBits,
                   venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
                   usage);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   result = vn_call_vkBindBufferMemory(&venus->vn_ring,
                                       venus->device_handle,
                                       resource->buffer,
                                       resource->memory, 0);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: Venus vkBindBufferMemory bind buffer failed result=%d memory_id=%llu usage=0x%x\n",
                   result,
                   (unsigned long long)resource->memory_obj.id,
                   usage);
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->memory, NULL);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   resource->initialized = true;
   resource->buffer_backed = true;
   resource->buffer_usage = usage;
   resource->vk_format = VK_FORMAT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_size = buffer_size;

   if (exportable && venus->device_local_static_draw_buffers &&
       (usage & (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT))) {
      yttrium_venus2_create_device_local_draw_mirror(
         venus, resource, buffer_size, usage);
   }

   if (out_memory_id)
      *out_memory_id = resource->memory_obj.id;

   return true;
}

bool
yttrium_venus2_transform_feedback_draw_enabled(const struct yttrium_venus *venus)
{
   return yttrium_venus2_transform_feedback_enabled(venus) &&
          venus->transform_feedback_props.transformFeedbackDraw;
}

bool
yttrium_venus2_vertex_attribute_divisor_supported(struct yttrium_venus *venus,
                                                 uint32_t divisor)
{
   if (divisor == 1)
      return true;

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   if (divisor == UINT32_MAX)
      return venus->vertex_attribute_instance_rate_zero_divisor;

   return venus->vertex_attribute_instance_rate_divisor &&
          divisor <= venus->max_vertex_attrib_divisor;
}

bool
yttrium_venus2_depth_clamp_enabled(const struct yttrium_venus *venus)
{
   return venus && venus->initialized && venus->depth_clamp;
}

bool
yttrium_venus2_logic_op_enabled(const struct yttrium_venus *venus)
{
   return venus && venus->initialized && venus->logic_op;
}

uint32_t
yttrium_venus2_max_dual_source_render_targets(struct yttrium_venus *venus)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return 0;

   return venus->max_dual_source_render_targets;
}

float
yttrium_venus2_max_sampler_anisotropy(struct yttrium_venus *venus)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return 1.0f;

   return MAX2(venus->max_sampler_anisotropy, 1.0f);
}

float
yttrium_venus2_max_sampler_lod_bias(struct yttrium_venus *venus)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return 0.0f;

   return MAX2(venus->max_sampler_lod_bias, 0.0f);
}

uint32_t
yttrium_venus2_max_transform_feedback_stride(const struct yttrium_venus *venus)
{
   return venus ?
          venus->transform_feedback_props.maxTransformFeedbackBufferDataStride :
          0;
}

uint32_t
yttrium_venus2_max_viewports(struct yttrium_venus *venus)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return 1;

   return MAX2(venus->max_viewports, 1u);
}

uint32_t
yttrium_venus2_mipmap_precision_bits(const struct yttrium_venus *venus)
{
   return venus ? venus->mipmap_precision_bits : 0;
}

bool
yttrium_venus2_create_stream_output_buffer(struct yttrium_venus *venus,
                                          struct yttrium_venus_resource *resource,
                                          uint64_t allocation_size,
                                          uint64_t *out_memory_id)
{
   if (out_memory_id)
      *out_memory_id = 0;

   if (!resource)
      return false;

   if (resource->initialized) {
      const VkBufferUsageFlags required =
         VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
         VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
      if (!resource->buffer_backed || !resource->memory_obj.id ||
          !resource->buffer ||
          (resource->buffer_usage & required) != required)
         return false;
      if (out_memory_id)
         *out_memory_id = resource->memory_obj.id;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus) || !venus->transform_feedback)
      return false;

   yttrium_venus_init_object(venus, &resource->buffer_obj);
   resource->buffer = YTTRIUM_VENUS_HANDLE(VkBuffer, &resource->buffer_obj);

   const VkDeviceSize buffer_size = MAX2((VkDeviceSize)allocation_size,
                                         (VkDeviceSize)1);
   const VkBufferUsageFlags usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
      VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = buffer_size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };

   struct vn_ring_submit_command create_submit;
   vn_submit_vkCreateBuffer(&venus->vn_ring, 0, venus->device_handle,
                            &buffer_info, NULL, &resource->buffer,
                            &create_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &create_submit, "vkCreateBuffer(stream-output)",
          resource->buffer_obj.id)) {
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         resource->buffer, &reqs);

   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, reqs.memoryTypeBits,
                                       0,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for stream output buffer bits=0x%x size=0x%llx\n",
                  reqs.memoryTypeBits, (unsigned long long)buffer_size);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = MAX2(reqs.size, buffer_size),
      .memoryTypeIndex = memory_type_index,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateMemory(&venus->vn_ring, 0, venus->device_handle,
                              &memory_info, NULL, &resource->memory,
                              &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateMemory(stream-output)",
          resource->memory_obj.id)) {
      yttrium_venus_discard_accepted_buffer(venus, resource, false);
      return false;
   }

   struct vn_ring_submit_command bind_submit;
   vn_submit_vkBindBufferMemory(&venus->vn_ring, 0, venus->device_handle,
                                resource->buffer, resource->memory, 0,
                                &bind_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &bind_submit, "vkBindBufferMemory(stream-output)",
          resource->buffer_obj.id)) {
      yttrium_venus_discard_accepted_buffer(venus, resource, true);
      return false;
   }

   resource->initialized = true;
   resource->buffer_backed = true;
   resource->buffer_usage = usage;
   resource->vk_format = VK_FORMAT_UNDEFINED;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_size = buffer_size;

   if (out_memory_id)
      *out_memory_id = resource->memory_obj.id;

   return true;
}

bool
yttrium_venus2_create_sampled_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *resource,
                                    uint64_t allocation_size,
                                    enum pipe_format pipe_format,
                                    uint64_t *out_memory_id)
{
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);

   if (out_memory_id)
      *out_memory_id = 0;

   if (!resource || vk_format == VK_FORMAT_UNDEFINED)
      return false;

   if (resource->initialized) {
      if (!resource->buffer_backed || !resource->memory_obj.id ||
          !resource->buffer ||
          !(resource->buffer_usage &
            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT))
         return false;
      resource->vk_format = vk_format;
      if (out_memory_id)
         *out_memory_id = resource->memory_obj.id;
      return true;
   }

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   yttrium_venus_init_object(venus, &resource->buffer_obj);
   resource->buffer = YTTRIUM_VENUS_HANDLE(VkBuffer, &resource->buffer_obj);

   const VkDeviceSize buffer_size = MAX2((VkDeviceSize)allocation_size,
                                         (VkDeviceSize)1);
   const VkBufferUsageFlags usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
      VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = buffer_size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };

   struct vn_ring_submit_command create_submit;
   vn_submit_vkCreateBuffer(&venus->vn_ring, 0, venus->device_handle,
                            &buffer_info, NULL, &resource->buffer,
                            &create_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &create_submit, "vkCreateBuffer(sampled-buffer)",
          resource->buffer_obj.id)) {
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         resource->buffer, &reqs);

   const uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(venus, reqs.memoryTypeBits, 0,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for sampled buffer bits=0x%x size=0x%llx format=%u vk_format=%u\n",
                  reqs.memoryTypeBits, (unsigned long long)buffer_size,
                  pipe_format, vk_format);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->buffer, NULL);
      memset(resource, 0, sizeof(*resource));
      return false;
   }

   yttrium_venus_init_object(venus, &resource->memory_obj);
   resource->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &resource->memory_obj);

   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = MAX2(reqs.size, buffer_size),
      .memoryTypeIndex = memory_type_index,
   };
   struct vn_ring_submit_command allocate_submit;
   vn_submit_vkAllocateMemory(&venus->vn_ring, 0, venus->device_handle,
                              &memory_info, NULL, &resource->memory,
                              &allocate_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &allocate_submit, "vkAllocateMemory(sampled-buffer)",
          resource->memory_obj.id)) {
      yttrium_venus_discard_accepted_buffer(venus, resource, false);
      return false;
   }

   struct vn_ring_submit_command bind_submit;
   vn_submit_vkBindBufferMemory(&venus->vn_ring, 0, venus->device_handle,
                                resource->buffer, resource->memory, 0,
                                &bind_submit);
   if (!yttrium_venus_async_submit_succeeded(
          venus, &bind_submit, "vkBindBufferMemory(sampled-buffer)",
          resource->buffer_obj.id)) {
      yttrium_venus_discard_accepted_buffer(venus, resource, true);
      return false;
   }

   resource->initialized = true;
   resource->buffer_backed = true;
   resource->buffer_usage = usage;
   resource->vk_format = vk_format;
   resource->allocation_size = memory_info.allocationSize;
   resource->image_size = buffer_size;

   if (out_memory_id)
      *out_memory_id = resource->memory_obj.id;

   YTTRIUM_LOG("yttrium: Venus sampled buffer memory_id=%llu buffer_id=%llu size=0x%llx req_size=0x%llx type=%u flags=0x%x bits=0x%x format=%u vk_format=%u\n",
               (unsigned long long)resource->memory_obj.id,
               (unsigned long long)resource->buffer_obj.id,
               (unsigned long long)buffer_size,
               (unsigned long long)reqs.size,
               memory_type_index,
               venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
               reqs.memoryTypeBits, pipe_format, vk_format);
   return true;
}

bool
yttrium_venus2_ensure_null_sampled_buffer(struct yttrium_venus *venus,
                                         enum pipe_format pipe_format,
                                         uint64_t min_size,
                                         struct yttrium_venus_resource **out_resource,
                                         uint32_t *out_resource_id)
{
   const uint64_t null_size = MAX2(min_size, 4096ull);

   if (!venus || !out_resource || !out_resource_id ||
       yttrium_venus2_pipe_format(pipe_format) == VK_FORMAT_UNDEFINED)
      return false;

   if (!venus->null_sampled_buffer.initialized ||
       venus->null_sampled_buffer.image_size < min_size) {
      if (venus->null_sampled_buffer.initialized)
         yttrium_venus2_resource_fini(venus, NULL,
                                     &venus->null_sampled_buffer, NULL);

      uint64_t memory_id = 0;
      if (!yttrium_venus2_create_sampled_buffer(
             venus, &venus->null_sampled_buffer, null_size, pipe_format,
             &memory_id)) {
         YTTRIUM_LOG("yttrium: Venus null sampled buffer create failed size=0x%llx format=%u\n",
                     (unsigned long long)null_size, pipe_format);
         return false;
      }
   } else {
      venus->null_sampled_buffer.vk_format =
         yttrium_venus2_pipe_format(pipe_format);
   }

   *out_resource = &venus->null_sampled_buffer;
   *out_resource_id = 0;
   return true;
}
