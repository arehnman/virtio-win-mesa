/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include "yttrium_resource.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <d3dkmthk.h>

#include "frontend/winsys_handle.h"
#include "util/format_srgb.h"
#include "util/format/u_format.h"
#include "util/helpers.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "virtio-gpu/virgl_hw.h"

#include "util/u_atomic.h"

#include "yttrium_gdi_public.h"
#include "yttrium_internal.h"
#include "yttrium_options.h"
#include "yttrium_present.h"
#include "yttrium_trace.h"

static bool
yttrium_copy_venus_buffer_to_cpu(struct pipe_context *ctx,
                                 struct yttrium_resource *ysrc,
                                 struct yttrium_resource *ydst,
                                 uint32_t src_offset, uint32_t dst_offset,
                                 uint32_t size);

static bool
yttrium_prepare_gpu_copy_buffer(struct pipe_context *ctx,
                                 struct yttrium_resource *res);

/*
 * Process-unique id stamped on every resource, so index bounds cache entries
 * cannot be aliased by a later resource landing on a recycled address.
 */
static uint64_t
yttrium_next_cache_id(void)
{
   static uint32_t next;

   return (uint64_t)p_atomic_inc_return(&next);
}

struct yttrium_buffer_storage {
   int reference_count;
   struct yttrium_screen *screen;
   D3DKMT_HANDLE hAllocation;
   HANDLE hResource;
   HANDLE hAllocationResource;
   bool hResourceIsD3D9Runtime;
   uint32_t venus_res_id;
   uint64_t size;
   void *map;
   uint32_t map_info;
   bool map_is_blob;
   bool owns_allocation;
   bool allocation_destroyed_by_runtime;
};

#define YTTRIUM_TEST_FAIL_BUFFER_REPLACEMENT_METADATA_AT_ENV \
   "D3D10UMD_YTTRIUM_TEST_FAIL_BUFFER_REPLACEMENT_METADATA_AT"

static bool
yttrium_take_buffer_replacement_metadata_failpoint(void)
{
   static volatile LONG configured_at = -1;
   static volatile LONG occurrence;
   LONG fail_at = InterlockedCompareExchange(&configured_at, -1, -1);

   if (fail_at < 0) {
      const char *value =
         getenv(YTTRIUM_TEST_FAIL_BUFFER_REPLACEMENT_METADATA_AT_ENV);
      char *end = NULL;
      const unsigned long parsed = value ? strtoul(value, &end, 10) : 0;
      const LONG requested = value && end && end != value && *end == '\0' &&
                             parsed > 0 && parsed <= LONG_MAX ?
         (LONG)parsed : 0;
      InterlockedCompareExchange(&configured_at, requested, -1);
      fail_at = InterlockedCompareExchange(&configured_at, -1, -1);
   }

   if (fail_at <= 0)
      return false;

   return InterlockedIncrement(&occurrence) == fail_at;
}

static struct yttrium_buffer_storage *
yttrium_buffer_storage_pool_acquire(struct yttrium_screen *screen,
                                    uint64_t size)
{
   struct yttrium_buffer_storage *storage = NULL;
   unsigned best;

   if (!screen || !screen->buffer_replacement_pool_initialized ||
       !screen->buffer_replacement_pool_enabled)
      return NULL;

   simple_mtx_lock(&screen->buffer_replacement_pool_lock);
   best = screen->buffer_replacement_pool_count;
   for (unsigned i = 0; i < screen->buffer_replacement_pool_count; i++) {
      const struct yttrium_buffer_storage *candidate =
         screen->buffer_replacement_pool[i];
      if (candidate && candidate->size >= size &&
          (best == screen->buffer_replacement_pool_count ||
           candidate->size < screen->buffer_replacement_pool[best]->size))
         best = i;
   }

   if (best != screen->buffer_replacement_pool_count) {
      storage = screen->buffer_replacement_pool[best];
      screen->buffer_replacement_pool_bytes -= storage->size;
      screen->buffer_replacement_pool_count--;
      screen->buffer_replacement_pool[best] =
         screen->buffer_replacement_pool[screen->buffer_replacement_pool_count];
      screen->buffer_replacement_pool[screen->buffer_replacement_pool_count] =
         NULL;
   }
   simple_mtx_unlock(&screen->buffer_replacement_pool_lock);

   return storage;
}

static bool
yttrium_buffer_storage_pool_return(struct yttrium_buffer_storage *storage)
{
   struct yttrium_screen *screen;
   bool pooled = false;

   if (!storage || !storage->screen || !storage->map ||
       !storage->hAllocation || !storage->owns_allocation ||
       storage->allocation_destroyed_by_runtime || !storage->size)
      return false;

   screen = storage->screen;
   if (!screen->buffer_replacement_pool_initialized ||
       !screen->buffer_replacement_pool_enabled)
      return false;

   simple_mtx_lock(&screen->buffer_replacement_pool_lock);
   if (screen->buffer_replacement_pool_initialized &&
       screen->buffer_replacement_pool_count <
          YTTRIUM_BUFFER_REPLACEMENT_POOL_MAX_ENTRIES &&
       storage->size <= YTTRIUM_BUFFER_REPLACEMENT_POOL_MAX_BYTES -
                           screen->buffer_replacement_pool_bytes) {
      storage->reference_count = 1;
      screen->buffer_replacement_pool[
         screen->buffer_replacement_pool_count++] = storage;
      screen->buffer_replacement_pool_bytes += storage->size;
      pooled = true;
   }
   simple_mtx_unlock(&screen->buffer_replacement_pool_lock);

   return pooled;
}

static void
yttrium_ordered_upload_rehome_venus_handles(
   struct yttrium_venus_resource *venus)
{
   if (!venus)
      return;

   if (venus->buffer_obj.id)
      venus->buffer = (VkBuffer)(uintptr_t)&venus->buffer_obj;
   if (venus->memory_obj.id)
      venus->memory = (VkDeviceMemory)(uintptr_t)&venus->memory_obj;
}

static bool
yttrium_ordered_upload_pool_acquire(
   struct yttrium_screen *screen, uint64_t size, bool direct_backing,
   struct yttrium_ordered_upload_pool_entry *out)
{
   unsigned best;
   bool found = false;

   if (!screen || !out || !screen->ordered_upload_pool_initialized)
      return false;

   memset(out, 0, sizeof(*out));
   simple_mtx_lock(&screen->ordered_upload_pool_lock);
   best = screen->ordered_upload_pool_count;
   for (unsigned i = 0; i < screen->ordered_upload_pool_count; i++) {
      const struct yttrium_ordered_upload_pool_entry *candidate =
         &screen->ordered_upload_pool[i];
      if (candidate->direct_backing == direct_backing &&
          candidate->capacity >= size &&
          (best == screen->ordered_upload_pool_count ||
           candidate->capacity <
              screen->ordered_upload_pool[best].capacity))
         best = i;
   }

   if (best != screen->ordered_upload_pool_count) {
      found = true;
      *out = screen->ordered_upload_pool[best];
      screen->ordered_upload_pool_bytes -= out->capacity;
      screen->ordered_upload_pool_count--;
      screen->ordered_upload_pool[best] =
         screen->ordered_upload_pool[screen->ordered_upload_pool_count];
      memset(&screen->ordered_upload_pool[screen->ordered_upload_pool_count],
             0, sizeof(screen->ordered_upload_pool[0]));
   }
   simple_mtx_unlock(&screen->ordered_upload_pool_lock);

   return found;
}

static bool
yttrium_ordered_upload_pool_return(struct yttrium_screen *screen,
                                   struct yttrium_resource *res)
{
   bool pooled = false;
   const uint64_t capacity =
      res && res->data_capacity ? res->data_capacity : (res ? res->size : 0);

   if (!screen || !res || !screen->ordered_upload_pool_initialized)
      return false;

   simple_mtx_lock(&screen->ordered_upload_pool_lock);
   if (res->data && capacity &&
       screen->ordered_upload_pool_count <
          YTTRIUM_ORDERED_UPLOAD_POOL_MAX_ENTRIES &&
       capacity <= YTTRIUM_ORDERED_UPLOAD_POOL_MAX_BYTES -
                      screen->ordered_upload_pool_bytes) {
      struct yttrium_ordered_upload_pool_entry *entry =
         &screen->ordered_upload_pool[screen->ordered_upload_pool_count++];
      memset(entry, 0, sizeof(*entry));
      entry->data = res->data;
      entry->capacity = capacity;
      entry->direct_backing = res->ordered_worker_upload_direct_backing;
      if (entry->direct_backing) {
         entry->hAllocation = res->hAllocation;
         entry->hResource = res->hResource;
         entry->hAllocationResource = res->hAllocationResource;
         entry->hResourceIsD3D9Runtime = res->hResourceIsD3D9Runtime;
         entry->venus_res_id = res->venus_res_id;
         entry->venus_mem_id = res->venus_mem_id;
         entry->map_info = res->map_info;
         entry->map_is_blob = res->map_is_blob;
         entry->owns_allocation = res->owns_allocation;
         entry->allocation_destroyed_by_runtime =
            res->allocation_destroyed_by_runtime;
         entry->venus = res->venus;
         entry->venus.owner = NULL;
         yttrium_ordered_upload_rehome_venus_handles(&entry->venus);

         memset(&res->venus, 0, sizeof(res->venus));
         res->hAllocation = 0;
         res->hResource = NULL;
         res->hAllocationResource = NULL;
         res->hResourceIsD3D9Runtime = false;
         res->venus_res_id = 0;
         res->venus_mem_id = 0;
         res->map = NULL;
         res->map_info = 0;
         res->map_is_blob = false;
         res->owns_allocation = false;
         res->allocation_destroyed_by_runtime = false;
      }
      res->data = NULL;
      res->data_capacity = 0;
      res->owns_data = false;
      screen->ordered_upload_pool_bytes += capacity;
      pooled = true;
   }
   simple_mtx_unlock(&screen->ordered_upload_pool_lock);

   return pooled;
}

static void
yttrium_resource_init_threaded(struct yttrium_resource *res)
{
   threaded_resource_init(&res->base, false);

   if (res->base.target == PIPE_BUFFER) {
      struct yttrium_screen *screen = yttrium_screen(res->base.screen);
      assert(screen->buffer_ids_initialized);
      threaded_resource(&res->base)->buffer_id_unique =
         util_idalloc_mt_alloc(&screen->buffer_ids);
   }

   /* Only resources with reference-counted private backing are replaceable. */
   threaded_resource(&res->base)->is_shared = true;
}

static void
yttrium_resource_free(struct yttrium_resource *res)
{
   if (!res)
      return;

   if (res->base.target == PIPE_BUFFER) {
      struct yttrium_screen *screen = yttrium_screen(res->base.screen);
      util_idalloc_mt_free(&screen->buffer_ids,
                           threaded_resource(&res->base)->buffer_id_unique);
      threaded_resource(&res->base)->buffer_id_unique = 0;
   }

   threaded_resource_deinit(&res->base);
   FREE(res);
}

#define VIRTGPU_BLOB_MEM_HOST3D 0x0002
#define VIRTGPU_BLOB_FLAG_USE_MAPPABLE 0x0001
#define VIRTGPU_BLOB_FLAG_USE_SHAREABLE 0x0002
#define YTTRIUM_DISPLAY_STAGING_ARENA_DEFAULT_SIZE (16ull * 1024ull * 1024ull)

void
yttrium_gdi_resource_set_primary_target(struct pipe_resource *resource,
                                        bool primary_target)
{
   struct yttrium_resource *res;

   if (!resource || resource->target == PIPE_BUFFER)
      return;

   res = yttrium_resource(resource);
   if (res->primary_target == primary_target)
      return;

   YTTRIUM_LOG("yttrium: resource primary identity hAllocation=0x%lx res_id=%u primary=%u->%u\n",
               (unsigned long)res->hAllocation,
               res->venus_res_id,
               res->primary_target,
               primary_target);
   res->primary_target = primary_target;
}

void
yttrium_gdi_resource_set_allocation_ownership(struct pipe_resource *resource,
                                              bool owns_allocation)
{
   struct yttrium_resource *res;

   if (!resource)
      return;

   res = yttrium_resource(resource);
   if (res->owns_allocation == owns_allocation)
      return;

   YTTRIUM_LOG("yttrium: resource allocation ownership hAllocation=0x%lx "
               "res_id=%u owns=%u->%u\n",
               (unsigned long)res->hAllocation, res->venus_res_id,
               res->owns_allocation, owns_allocation);
   res->owns_allocation = owns_allocation;
   res->allocation_destroyed_by_runtime = !owns_allocation;
   if (res->replacement_storage) {
      res->replacement_storage->owns_allocation = owns_allocation;
      res->replacement_storage->allocation_destroyed_by_runtime =
         !owns_allocation;
   }
}

bool
yttrium_gdi_resource_has_runtime_allocation(struct pipe_resource *resource)
{
   const struct yttrium_resource *res;
   const char *name;

   if (!resource || !resource->screen || !resource->screen->get_name)
      return false;

   name = resource->screen->get_name(resource->screen);
   if (!name || strcmp(name, "yttrium") != 0)
      return false;

   res = yttrium_resource(resource);
   return res->hAllocation && res->hResource &&
          (!res->hAllocationResource ||
           res->hResource != res->hAllocationResource);
}

bool
yttrium_gdi_resource_rotate_runtime_handles(
   struct pipe_resource *const *resources, unsigned count)
{
   unsigned runtime_handle_count = 0;

   if (!resources || count <= 1)
      return true;

   /* DXGI rotates the kernel allocation identities while each runtime
    * resource handle remains attached to its original frontend resource.
    * The frontend rotates whole pipe_resource pointers, so rotate only the
    * runtime handles in the opposite direction before that pointer rotation.
    */
   for (unsigned i = 0; i < count; i++) {
      struct pipe_resource *resource = resources[i];
      const char *name;

      if (!resource || !resource->screen || !resource->screen->get_name) {
         YTTRIUM_WARN("yttrium: ERROR: DXGI runtime-handle rotation rejected owner=yttrium-resource component=RotateResourceIdentities reason=invalid-resource action=abort index=%u count=%u\n",
                      i, count);
         return false;
      }

      name = resource->screen->get_name(resource->screen);
      if (!name || strcmp(name, "yttrium") != 0) {
         YTTRIUM_WARN("yttrium: ERROR: DXGI runtime-handle rotation rejected owner=yttrium-resource component=RotateResourceIdentities reason=foreign-resource action=abort index=%u count=%u screen=%s\n",
                      i, count, name ? name : "<unknown>");
         return false;
      }

      struct yttrium_resource *res = yttrium_resource(resource);
      const bool has_runtime_handle =
         res->hResource &&
         (!res->hAllocationResource ||
          res->hResource != res->hAllocationResource);

      if (has_runtime_handle) {
         if (!res->hAllocation || res->replacement_storage) {
            YTTRIUM_WARN("yttrium: ERROR: DXGI runtime-handle rotation rejected owner=yttrium-resource component=RotateResourceIdentities reason=invalid-runtime-allocation action=abort index=%u count=%u hAllocation=0x%lx hResource=%p replacement=%p\n",
                         i, count, (unsigned long)res->hAllocation,
                         res->hResource, (void *)res->replacement_storage);
            return false;
         }

         for (unsigned j = 0; j < i; j++) {
            const struct yttrium_resource *previous =
               yttrium_resource(resources[j]);
            if (previous->hResource == res->hResource) {
               YTTRIUM_WARN("yttrium: ERROR: DXGI runtime-handle rotation rejected owner=yttrium-resource component=RotateResourceIdentities reason=duplicate-runtime-handle action=abort index=%u previous=%u count=%u hResource=%p\n",
                            i, j, count, res->hResource);
               return false;
            }
         }
         runtime_handle_count++;
      }
   }

   if (!runtime_handle_count)
      return true;

   if (runtime_handle_count != count) {
      YTTRIUM_WARN("yttrium: ERROR: DXGI runtime-handle rotation rejected owner=yttrium-resource component=RotateResourceIdentities reason=mixed-runtime-ownership action=abort runtime_handles=%u count=%u\n",
                   runtime_handle_count, count);
      return false;
   }

   struct yttrium_resource *last = yttrium_resource(resources[count - 1]);
   HANDLE last_handle = last->hResource;
   bool last_is_d3d9_runtime = last->hResourceIsD3D9Runtime;

   for (unsigned i = count - 1; i > 0; i--) {
      struct yttrium_resource *dst = yttrium_resource(resources[i]);
      const struct yttrium_resource *src = yttrium_resource(resources[i - 1]);
      dst->hResource = src->hResource;
      dst->hResourceIsD3D9Runtime = src->hResourceIsD3D9Runtime;
   }

   struct yttrium_resource *first = yttrium_resource(resources[0]);
   first->hResource = last_handle;
   first->hResourceIsD3D9Runtime = last_is_d3d9_runtime;
   return true;
}

void
yttrium_gdi_resource_debug_log(struct pipe_resource *resource,
                               const char *label)
{
   const struct yttrium_resource *res;
   const char *name;

   if (!resource || !resource->screen || !resource->screen->get_name)
      return;

   name = resource->screen->get_name(resource->screen);
   if (!name || strcmp(name, "yttrium") != 0)
      return;

   res = yttrium_resource(resource);
   YTTRIUM_LOG("yttrium: dxgi %s ownership hAllocation=0x%lx res_id=%u mem_id=0x%llx display=%u primary=%u classic=%u owns=%u size=0x%llx stride=%u map_blob=%u venus=(init=%u buffer=%u image_id=%llu buffer_id=%llu memory_id=%llu alloc=0x%llx extent=%ux%u format=%u image_offset=0x%llx image_size=0x%llx row_pitch=%llu array_pitch=%llu)\n",
                label ? label : "<unknown>",
                (unsigned long)res->hAllocation,
                res->venus_res_id,
                (unsigned long long)res->venus_mem_id,
                res->display_target,
                res->primary_target,
                res->classic_display,
                res->owns_allocation,
                (unsigned long long)res->size,
                res->stride,
                res->map_is_blob,
                res->venus.initialized,
                res->venus.buffer_backed,
                (unsigned long long)res->venus.image_obj.id,
                (unsigned long long)res->venus.buffer_obj.id,
                (unsigned long long)res->venus.memory_obj.id,
                (unsigned long long)res->venus.allocation_size,
                res->venus.width,
                res->venus.height,
                res->venus.vk_format,
                (unsigned long long)res->venus.image_offset,
                (unsigned long long)res->venus.image_size,
                (unsigned long long)res->venus.image_row_pitch,
                (unsigned long long)res->venus.image_array_pitch);
}

static unsigned
yttrium_resource_level_width(const struct pipe_resource *resource,
                             unsigned level);
static unsigned
yttrium_resource_level_height(const struct pipe_resource *resource,
                              unsigned level);
static unsigned
yttrium_resource_level_depth(const struct pipe_resource *resource,
                             unsigned level);
static unsigned
yttrium_resource_level_stride(const struct pipe_resource *resource,
                              unsigned level);
static uint64_t
yttrium_resource_level_slice_size(const struct pipe_resource *resource,
                                  unsigned level);
static uint64_t
yttrium_resource_array_stride(const struct pipe_resource *resource);
static bool
yttrium_blit_ensure_image(struct pipe_context *ctx,
                          struct yttrium_resource *res);

uint64_t
yttrium_resource_size(const struct pipe_resource *templ,
                      unsigned *out_stride, unsigned *out_layer_stride)
{
   uint64_t size;
   unsigned stride = 0;
   unsigned layer_stride = 0;

   if (templ->target == PIPE_BUFFER) {
      stride = templ->width0;
      layer_stride = templ->width0;
      size = templ->width0;
   } else {
      const unsigned array_size = MAX2(templ->array_size, 1);

      stride = util_format_get_stride(templ->format, templ->width0);
      if (templ->bind & PIPE_BIND_DISPLAY_TARGET)
         stride = (unsigned)align64(stride, 256);
      if (templ->last_level == 0) {
         const unsigned height = MAX2(templ->height0, 1);
         const unsigned depth = MAX2(templ->depth0, 1);
         layer_stride = (unsigned)MIN2((uint64_t)UINT_MAX,
                                       util_format_get_2d_size(templ->format,
                                                               stride,
                                                               height));
         size = (uint64_t)layer_stride * depth * array_size;
      } else {
         const uint64_t array_stride = yttrium_resource_array_stride(templ);
         layer_stride = (unsigned)MIN2((uint64_t)UINT_MAX, array_stride);
         size = array_stride * array_size;
      }
   }

   if (out_stride)
      *out_stride = stride;
   if (out_layer_stride)
      *out_layer_stride = layer_stride;

   return align64(MAX2(size, 1), 4096);
}

static uint64_t
yttrium_display_staging_arena_size(uint64_t required_size)
{
   int64_t opt =
      yttrium_gdi_debug_get_num_option(
         "D3D10UMD_YTTRIUM_DISPLAY_STAGING_ARENA_SIZE",
         (int64_t)YTTRIUM_DISPLAY_STAGING_ARENA_DEFAULT_SIZE);
   uint64_t arena_size = opt > 0 ? (uint64_t)opt : required_size;

   return align64(MAX2(required_size, arena_size), 4096);
}

static unsigned
yttrium_resource_level_width(const struct pipe_resource *resource,
                             unsigned level)
{
   return MAX2(resource->width0 >> MIN2(level, 31u), 1);
}

static unsigned
yttrium_resource_level_height(const struct pipe_resource *resource,
                              unsigned level)
{
   return MAX2(resource->height0 >> MIN2(level, 31u), 1);
}

static unsigned
yttrium_resource_level_depth(const struct pipe_resource *resource,
                             unsigned level)
{
   if (resource->target != PIPE_TEXTURE_3D)
      return 1;
   return MAX2(resource->depth0 >> MIN2(level, 31u), 1);
}

static unsigned
yttrium_resource_level_stride(const struct pipe_resource *resource,
                              unsigned level)
{
   return util_format_get_stride(resource->format,
                                 yttrium_resource_level_width(resource,
                                                               level));
}

static uint64_t
yttrium_resource_level_slice_size(const struct pipe_resource *resource,
                                  unsigned level)
{
   const unsigned stride = yttrium_resource_level_stride(resource, level);
   return util_format_get_2d_size(resource->format, stride,
                                  yttrium_resource_level_height(resource,
                                                                 level));
}

static uint64_t
yttrium_resource_array_stride(const struct pipe_resource *resource)
{
   uint64_t stride = 0;

   for (unsigned level = 0; level <= resource->last_level; level++)
      stride += yttrium_resource_level_slice_size(resource, level) *
                yttrium_resource_level_depth(resource, level);

   return stride;
}

static uint64_t
yttrium_resource_level_base_offset(const struct pipe_resource *resource,
                                   unsigned level)
{
   uint64_t offset = 0;

   for (unsigned l = 0; l < level && l <= resource->last_level; l++)
      offset += yttrium_resource_level_slice_size(resource, l) *
                yttrium_resource_level_depth(resource, l);

   return offset;
}

uint32_t
yttrium_pipe_to_resource_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_B8G8R8A8_UNORM:
      return VIRGL_FORMAT_B8G8R8A8_UNORM;
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return VIRGL_FORMAT_B8G8R8X8_UNORM;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
      return VIRGL_FORMAT_R8G8B8A8_UNORM;
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return VIRGL_FORMAT_R8G8B8X8_UNORM;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return VIRGL_FORMAT_R10G10B10A2_UNORM;
   case PIPE_FORMAT_B8G8R8A8_SRGB:
      return VIRGL_FORMAT_B8G8R8A8_SRGB;
   case PIPE_FORMAT_R8G8B8A8_SRGB:
      return VIRGL_FORMAT_R8G8B8A8_SRGB;
   default:
      return format;
   }
}

enum pipe_format
yttrium_resource_to_pipe_format(uint32_t format)
{
   switch (format) {
   case VIRGL_FORMAT_B8G8R8A8_UNORM:
      return PIPE_FORMAT_B8G8R8A8_UNORM;
   case VIRGL_FORMAT_B8G8R8X8_UNORM:
      return PIPE_FORMAT_B8G8R8X8_UNORM;
   case VIRGL_FORMAT_R8G8B8A8_UNORM:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case VIRGL_FORMAT_R8G8B8X8_UNORM:
      return PIPE_FORMAT_R8G8B8X8_UNORM;
   case VIRGL_FORMAT_R10G10B10A2_UNORM:
      return PIPE_FORMAT_R10G10B10A2_UNORM;
   case VIRGL_FORMAT_B8G8R8A8_SRGB:
      return PIPE_FORMAT_B8G8R8A8_SRGB;
   case VIRGL_FORMAT_R8G8B8A8_SRGB:
      return PIPE_FORMAT_R8G8B8A8_SRGB;
   default:
      return (enum pipe_format)format;
   }
}

bool
yttrium_is_format_supported(struct pipe_screen *pscreen,
                            enum pipe_format format,
                            enum pipe_texture_target target,
                            unsigned sample_count,
                            unsigned storage_sample_count,
                            unsigned bindings)
{
   if (target == PIPE_BUFFER)
      return true;

   if (sample_count > 1 || storage_sample_count > 1) {
      const bool depth_stencil = util_format_is_depth_or_stencil(format);
      const VkSampleCountFlags framebuffer_sample_counts =
         yttrium_venus_framebuffer_color_sample_counts(
            yttrium_screen(pscreen)->venus);

      if (sample_count != storage_sample_count)
         return false;
      if (sample_count != 2 && sample_count != 4 && sample_count != 8)
         return false;
      if (target != PIPE_TEXTURE_2D && target != PIPE_TEXTURE_2D_ARRAY)
         return false;
      if (!(framebuffer_sample_counts & sample_count))
         return false;
      if (bindings & PIPE_BIND_SHADER_IMAGE)
         return false;
      if (depth_stencil) {
         if (!(bindings & PIPE_BIND_DEPTH_STENCIL))
            return false;
      } else if (!(bindings & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW))) {
         return false;
      }
   }

   if (util_format_is_compressed(format)) {
      if (bindings & ~PIPE_BIND_SAMPLER_VIEW)
         return false;
      if ((bindings & PIPE_BIND_SAMPLER_VIEW) &&
          !yttrium_venus_sampled_texture_format_supported(
             yttrium_screen(pscreen)->venus, format, target))
         return false;
   }

   return util_format_get_blocksize(format) != 0;
}

static uint64_t
yttrium_resource_offset(const struct pipe_resource *resource,
                        unsigned level, unsigned x, unsigned y, unsigned z)
{
   const unsigned stride = yttrium_resource_level_stride(resource, level);
   const uint64_t layer_stride = yttrium_resource_array_stride(resource);
   const uint64_t level_offset =
      yttrium_resource_level_base_offset(resource, level);
   const uint64_t slice_size =
      yttrium_resource_level_slice_size(resource, level);
   const bool array_texture = resource->target != PIPE_TEXTURE_3D;
   const unsigned blocksize = util_format_get_blocksize(resource->format);
   const unsigned block_x = x / util_format_get_blockwidth(resource->format);
   const unsigned block_y = y / util_format_get_blockheight(resource->format);

   return (array_texture ? (uint64_t)z * layer_stride :
                           (uint64_t)z * slice_size) +
          level_offset +
          (uint64_t)block_y * stride +
          (uint64_t)block_x * blocksize;
}

bool
yttrium_map_allocation(struct pipe_screen *pscreen,
                       D3DKMT_HANDLE hAllocation,
                       uint64_t size,
                       const char *label,
                       void **out_map,
                       uint32_t *out_map_info,
                       bool *out_map_is_blob)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   VIOGPU_ESCAPE map;

   if (!hAllocation || !out_map || !out_map_info || !out_map_is_blob)
      return false;

   memset(&map, 0, sizeof(map));
   map.Type = VIOGPU_RES_MAP_BLOB;
   map.DataLength = sizeof(VIOGPU_RES_MAP_BLOB_REQ);
   map.ResourceMapBlob.ResHandle = hAllocation;
   map.ResourceMapBlob.Size = size;

   NTSTATUS status = screen->device->escape(screen->device, &map,
                                             sizeof(map));
   if (!NT_SUCCESS(status) || !map.ResourceMapBlob.UserVa) {
      YTTRIUM_WARN("yttrium: allocation map failed label=%s status=0x%lx hAllocation=0x%lx size=0x%llx user_va=0x%llx\n",
                   label, (unsigned long)status,
                   (unsigned long)hAllocation, (unsigned long long)size,
                   (unsigned long long)map.ResourceMapBlob.UserVa);
      return false;
   }

   *out_map = (void *)(uintptr_t)map.ResourceMapBlob.UserVa;
   *out_map_info = map.ResourceMapBlob.MapInfo;
   *out_map_is_blob = true;

   YTTRIUM_LOG("yttrium: mapped %s blob hAllocation=0x%lx va=%p size=0x%llx map_info=0x%x\n",
               label, (unsigned long)hAllocation, *out_map,
               (unsigned long long)map.ResourceMapBlob.Size,
               *out_map_info);
   return true;
}

bool
yttrium_create_venus_memory_mapping(struct pipe_screen *pscreen,
                                    uint64_t venus_mem_id,
                                    uint64_t size,
                                    unsigned stride,
                                    struct yttrium_readback_mapping *mapping)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   VIOGPU_CREATE_ALLOCATION_EXCHANGE alloc_exchange;
   VIOGPU_CREATE_RESOURCE_EXCHANGE res_exchange;
   struct gdikmt_createallocation create;
   D3DDDI_ALLOCATIONINFO alloc_info;

   if (!venus_mem_id || !size || !mapping)
      return false;

   memset(mapping, 0, sizeof(*mapping));
   memset(&alloc_exchange, 0, sizeof(alloc_exchange));
   memset(&res_exchange, 0, sizeof(res_exchange));
   memset(&create, 0, sizeof(create));
   memset(&alloc_info, 0, sizeof(alloc_info));

   alloc_exchange.ResourceOptions.target = PIPE_BUFFER;
   alloc_exchange.ResourceOptions.width =
      (ULONG)MIN2(size, (uint64_t)UINT_MAX);
   alloc_exchange.ResourceOptions.height = 1;
   alloc_exchange.ResourceOptions.depth = 1;
   alloc_exchange.ResourceOptions.array_size = 1;
   alloc_exchange.Size = size;
   alloc_exchange.BlobId = venus_mem_id;
   alloc_exchange.BlobMem = VIRTGPU_BLOB_MEM_HOST3D;
   alloc_exchange.BlobFlags = VIRTGPU_BLOB_FLAG_USE_MAPPABLE |
                              VIRTGPU_BLOB_FLAG_USE_SHAREABLE;
   alloc_exchange.Stride = stride;

   create.NumAllocations = 1;
   create.pAllocationInfo = &alloc_info;
   create.pPrivateDriverData = &res_exchange;
   create.PrivateDriverDataSize = sizeof(res_exchange);
   create.force_allocation_handle = true;

   alloc_info.pPrivateDriverData = &alloc_exchange;
   alloc_info.PrivateDriverDataSize = sizeof(alloc_exchange);

   NTSTATUS status = screen->device->createAllocation(screen->device,
                                                       &create);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation create failed owner=present-readback-create status=0x%lx mem_id=0x%llx size=0x%llx stride=%u\n",
                   (unsigned long)status, (unsigned long long)venus_mem_id,
                   (unsigned long long)size, stride);
      return false;
   }

   mapping->hResource = create.hResource;
   mapping->hAllocation = alloc_info.hAllocation;

   VIOGPU_ESCAPE resinfo;
   memset(&resinfo, 0, sizeof(resinfo));
   resinfo.Type = VIOGPU_RES_INFO;
   resinfo.DataLength = sizeof(VIOGPU_RES_INFO_REQ);
   resinfo.ResourceInfo.ResHandle = mapping->hAllocation;

   status = screen->device->escape(screen->device, &resinfo,
                                   sizeof(resinfo));
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation query failed owner=present-readback-res-info status=0x%lx hAllocation=0x%lx hResource=%p\n",
                   (unsigned long)status,
                   (unsigned long)mapping->hAllocation,
                   mapping->hResource);
      status = screen->device->destroyAllocation(
         screen->device, NULL, mapping->hAllocation);
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: internal allocation destroy failed owner=present-readback-res-info-cleanup status=0x%lx hAllocation=0x%lx hResource=%p\n",
                      (unsigned long)status,
                      (unsigned long)mapping->hAllocation,
                      mapping->hResource);
      }
      memset(mapping, 0, sizeof(*mapping));
      return false;
   }

   mapping->venus_res_id = resinfo.ResourceInfo.Id;

   if (!yttrium_map_allocation(pscreen, mapping->hAllocation, size,
                               "present readback", &mapping->map,
                               &mapping->map_info,
                               &mapping->map_is_blob)) {
      YTTRIUM_WARN("yttrium: internal allocation map failed owner=present-readback-map hAllocation=0x%lx hResource=%p mem_id=0x%llx size=0x%llx\n",
                   (unsigned long)mapping->hAllocation,
                   mapping->hResource,
                   (unsigned long long)venus_mem_id,
                   (unsigned long long)size);
      status = screen->device->destroyAllocation(
         screen->device, NULL, mapping->hAllocation);
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: internal allocation destroy failed owner=present-readback-map-cleanup status=0x%lx hAllocation=0x%lx hResource=%p\n",
                      (unsigned long)status,
                      (unsigned long)mapping->hAllocation,
                      mapping->hResource);
      }
      memset(mapping, 0, sizeof(*mapping));
      return false;
   }

   YTTRIUM_LOG("yttrium: present readback allocation hAllocation=0x%lx res_id=%u mem_id=0x%llx size=0x%llx stride=%u map=%p map_blob=%u\n",
                (unsigned long)mapping->hAllocation,
                mapping->venus_res_id,
                (unsigned long long)venus_mem_id,
                (unsigned long long)size, stride, mapping->map,
                mapping->map_is_blob);
   return true;
}

void
yttrium_destroy_readback_mapping(struct pipe_screen *pscreen,
                                 struct yttrium_readback_mapping *mapping)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   if (!mapping || !mapping->hAllocation)
      return;

   if (mapping->map && mapping->map_is_blob) {
      VIOGPU_ESCAPE unmap;
      memset(&unmap, 0, sizeof(unmap));
      unmap.Type = VIOGPU_RES_UNMAP_BLOB;
      unmap.DataLength = sizeof(VIOGPU_RES_UNMAP_BLOB_REQ);
      unmap.ResourceUnmapBlob.ResHandle = mapping->hAllocation;
      NTSTATUS unmap_status =
         screen->device->escape(screen->device, &unmap, sizeof(unmap));
      if (!NT_SUCCESS(unmap_status)) {
         YTTRIUM_WARN("yttrium: internal allocation unmap failed owner=present-readback-unmap status=0x%lx hAllocation=0x%lx hResource=%p res_id=%u\n",
                      (unsigned long)unmap_status,
                      (unsigned long)mapping->hAllocation,
                      mapping->hResource, mapping->venus_res_id);
      }
   }

   NTSTATUS status = screen->device->destroyAllocation(
      screen->device, NULL, mapping->hAllocation);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation destroy failed owner=present-readback-destroy status=0x%lx hAllocation=0x%lx hResource=%p res_id=%u\n",
                   (unsigned long)status,
                   (unsigned long)mapping->hAllocation,
                   mapping->hResource, mapping->venus_res_id);
   }
   memset(mapping, 0, sizeof(*mapping));
}

bool
yttrium_map_display_allocation(struct pipe_screen *pscreen,
                               struct yttrium_resource *res)
{
   return yttrium_map_allocation(pscreen, res->hAllocation, res->size,
                                 "display", &res->map, &res->map_info,
                                 &res->map_is_blob);
}

static void
yttrium_unmap_display_allocation(struct pipe_screen *pscreen,
                                 struct yttrium_resource *res)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   VIOGPU_ESCAPE unmap;

   if (!res->map)
      return;

   if (res->map_is_blob) {
      memset(&unmap, 0, sizeof(unmap));
      unmap.Type = VIOGPU_RES_UNMAP_BLOB;
      unmap.DataLength = sizeof(VIOGPU_RES_UNMAP_BLOB_REQ);
      unmap.ResourceUnmapBlob.ResHandle = res->hAllocation;

      NTSTATUS status = screen->device->escape(screen->device, &unmap,
                                               sizeof(unmap));
      if (!NT_SUCCESS(status)) {
         YTTRIUM_LOG("yttrium: VIOGPU_RES_UNMAP_BLOB failed status=0x%lx hAllocation=0x%lx\n",
                      status, (unsigned long)res->hAllocation);
      }
   }

   res->map = NULL;
   res->map_info = 0;
   res->map_is_blob = false;
}

static void
yttrium_buffer_storage_destroy(struct yttrium_buffer_storage *storage)
{
   struct yttrium_screen *screen = storage->screen;

   if (storage->map && storage->map_is_blob && storage->hAllocation) {
      VIOGPU_ESCAPE unmap;
      memset(&unmap, 0, sizeof(unmap));
      unmap.Type = VIOGPU_RES_UNMAP_BLOB;
      unmap.DataLength = sizeof(VIOGPU_RES_UNMAP_BLOB_REQ);
      unmap.ResourceUnmapBlob.ResHandle = storage->hAllocation;

      NTSTATUS status = screen->device->escape(screen->device, &unmap,
                                               sizeof(unmap));
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: WARNING: buffer replacement backing unmap failed owner=threaded-context-storage status=0x%lx hAllocation=0x%lx size=0x%llx\n",
                      (unsigned long)status,
                      (unsigned long)storage->hAllocation,
                      (unsigned long long)storage->size);
      }
   }

   if (storage->hAllocation && !storage->allocation_destroyed_by_runtime) {
      NTSTATUS status = screen->device->destroyAllocation(
         screen->device, storage->hResource, storage->hAllocation);
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: WARNING: buffer replacement backing destroy failed owner=threaded-context-storage status=0x%lx hAllocation=0x%lx hResource=%p size=0x%llx owns=%u\n",
                      (unsigned long)status,
                      (unsigned long)storage->hAllocation,
                      storage->hResource,
                      (unsigned long long)storage->size,
                      storage->owns_allocation ? 1u : 0u);
      }
   }

   FREE(storage);
}

static void
yttrium_buffer_storage_reference(struct yttrium_buffer_storage **dst,
                                 struct yttrium_buffer_storage *src)
{
   struct yttrium_buffer_storage *old = *dst;

   if (old == src)
      return;
   if (src)
      p_atomic_inc(&src->reference_count);
   *dst = src;
   if (old && p_atomic_dec_zero(&old->reference_count) &&
       !yttrium_buffer_storage_pool_return(old))
      yttrium_buffer_storage_destroy(old);
}

static bool
yttrium_buffer_storage_attach(struct yttrium_resource *res)
{
   assert(res->base.target == PIPE_BUFFER);
   assert(!res->display_target);
   assert(!res->venus.initialized);
   assert(res->owns_allocation);
   assert(!res->allocation_destroyed_by_runtime);
   assert(res->hAllocation);
   assert(res->map && res->data == res->map);
   assert(!res->owns_data);

   if (yttrium_take_buffer_replacement_metadata_failpoint()) {
      YTTRIUM_WARN("yttrium: TEST ONLY: injected buffer replacement metadata allocation failure owner=threaded-context-storage reason=deterministic-test-fault variable=%s resource=%p size=0x%llx bind=0x%x; replacement remains conservative/shared\n",
                   YTTRIUM_TEST_FAIL_BUFFER_REPLACEMENT_METADATA_AT_ENV,
                   (void *)res, (unsigned long long)res->size,
                   res->base.bind);
      return false;
   }

   struct yttrium_buffer_storage *storage =
      CALLOC_STRUCT(yttrium_buffer_storage);
   if (!storage) {
      YTTRIUM_WARN("yttrium: WARNING: buffer replacement disabled for resource owner=threaded-context-storage reason=out-of-memory fallback=conservative-shared resource=%p size=0x%llx bind=0x%x\n",
                   (void *)res, (unsigned long long)res->size,
                   res->base.bind);
      return false;
   }

   storage->reference_count = 1;
   storage->screen = yttrium_screen(res->base.screen);
   storage->hAllocation = res->hAllocation;
   storage->hResource = res->hResource;
   storage->hAllocationResource = res->hAllocationResource;
   storage->hResourceIsD3D9Runtime = res->hResourceIsD3D9Runtime;
   storage->venus_res_id = res->venus_res_id;
   storage->size = res->size;
   storage->map = res->map;
   storage->map_info = res->map_info;
   storage->map_is_blob = res->map_is_blob;
   storage->owns_allocation = res->owns_allocation;
   storage->allocation_destroyed_by_runtime =
      res->allocation_destroyed_by_runtime;
   res->replacement_storage = storage;
   return true;
}

static void
yttrium_resource_apply_buffer_storage(
   struct yttrium_resource *res,
   const struct yttrium_buffer_storage *storage)
{
   res->hAllocation = storage->hAllocation;
   res->hResource = storage->hResource;
   res->hAllocationResource = storage->hAllocationResource;
   res->hResourceIsD3D9Runtime = storage->hResourceIsD3D9Runtime;
   res->venus_res_id = storage->venus_res_id;
   res->map = storage->map;
   res->map_info = storage->map_info;
   res->map_is_blob = storage->map_is_blob;
   res->owns_allocation = storage->owns_allocation;
   res->allocation_destroyed_by_runtime =
      storage->allocation_destroyed_by_runtime;
   res->data = storage->map;
   res->data_capacity = storage->size;
   res->owns_data = false;
}

static void
yttrium_resource_detach_buffer_storage(struct yttrium_resource *res)
{
   yttrium_buffer_storage_reference(&res->replacement_storage, NULL);
   res->replacement_owner = NULL;
   res->hAllocation = 0;
   res->hResource = NULL;
   res->hAllocationResource = NULL;
   res->hResourceIsD3D9Runtime = false;
   res->venus_res_id = 0;
   res->map = NULL;
   res->map_info = 0;
   res->map_is_blob = false;
   res->owns_allocation = false;
   res->allocation_destroyed_by_runtime = false;
   res->data = NULL;
   res->data_capacity = 0;
   res->owns_data = false;
}

static bool
yttrium_resource_uses_mapped_venus_buffer(const struct yttrium_resource *res)
{
   return res && res->base.target == PIPE_BUFFER &&
          res->venus.initialized && res->venus.buffer_backed &&
          res->venus.buffer && res->venus.memory_obj.id &&
          res->venus_mem_id == res->venus.memory_obj.id &&
          res->data && res->data == res->map && res->map_is_blob;
}

bool
yttrium_resource_publish_private_draw_buffer(struct pipe_context *ctx,
                                             struct yttrium_resource *res)
{
   if (!res || !res->private_immutable_draw_buffer || !res->data_dirty)
      return true;

   struct yttrium_screen *screen =
      ctx && ctx->screen ? yttrium_screen(ctx->screen) : NULL;
   if (!screen || !screen->venus || !res->data || !res->size ||
       !res->venus.initialized || !res->venus.buffer_backed ||
       !res->venus.buffer) {
      YTTRIUM_WARN("yttrium: ERROR: private immutable draw buffer publication failed owner=yttrium_resource component=immutable-draw-buffer reason=invalid-publication-state action=fail-draw resource=%p data=%p size=0x%llx initialized=%u buffer_backed=%u buffer_id=%llu\n",
                   (void *)res, res->data,
                   (unsigned long long)res->size,
                   res->venus.initialized, res->venus.buffer_backed,
                   (unsigned long long)res->venus.buffer_obj.id);
      return false;
   }

   /*
    * Immutable buffers are initialized once.  Publish the complete CPU
    * backing rather than only the reported map range so bytes not explicitly
    * touched by a partial initialization retain the zeroed guest contents.
    */
   if (!yttrium_venus_update_buffer(screen->venus, &res->venus, 0,
                                    res->size, res->data)) {
      YTTRIUM_WARN("yttrium: ERROR: private immutable draw buffer publication failed owner=yttrium_resource component=immutable-draw-buffer reason=venus-buffer-update-failed action=fail-draw resource=%p memory_id=%llu buffer_id=%llu size=0x%llx\n",
                   (void *)res,
                   (unsigned long long)res->venus.memory_obj.id,
                   (unsigned long long)res->venus.buffer_obj.id,
                   (unsigned long long)res->size);
      return false;
   }

   res->data_dirty = false;
   return true;
}

static bool
yttrium_create_guest_allocation(struct pipe_screen *pscreen,
                                struct yttrium_resource *res,
                                const char *label,
                                bool mappable)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   struct pipe_resource *base = &res->base;
   VIOGPU_CREATE_ALLOCATION_EXCHANGE alloc_exchange;
   VIOGPU_CREATE_RESOURCE_EXCHANGE res_exchange;
   struct gdikmt_createallocation create;
   D3DDDI_ALLOCATIONINFO alloc_info;

   memset(&alloc_exchange, 0, sizeof(alloc_exchange));
   memset(&res_exchange, 0, sizeof(res_exchange));
   memset(&create, 0, sizeof(create));
   memset(&alloc_info, 0, sizeof(alloc_info));

   if (res->classic_display) {
      YTTRIUM_WARN("yttrium: refusing to create classic %s allocation target=%u format=%u bind=0x%x size=%ux%u\n",
                   label, base->target, base->format, base->bind,
                   base->width0, base->height0);
      return false;
   }

   alloc_exchange.ResourceOptions.target = base->target;
   alloc_exchange.ResourceOptions.format =
      yttrium_pipe_to_resource_format(base->format);
   alloc_exchange.ResourceOptions.bind = base->bind;
   alloc_exchange.ResourceOptions.width = base->width0;
   alloc_exchange.ResourceOptions.height = base->height0;
   alloc_exchange.ResourceOptions.depth = base->depth0;
   alloc_exchange.ResourceOptions.array_size = base->array_size;
   alloc_exchange.ResourceOptions.last_level = base->last_level;
   alloc_exchange.ResourceOptions.nr_samples = base->nr_samples;
   alloc_exchange.ResourceOptions.flags = base->flags;
   alloc_exchange.Size = res->size;
   alloc_exchange.Stride = res->stride;
   alloc_exchange.BlobId = res->venus_mem_id;
   if (res->venus.initialized && !res->venus.buffer_backed &&
       res->venus.image_offset <= UINT32_MAX) {
      alloc_exchange.ScanoutOffset = (ULONG)res->venus.image_offset;
   }
   alloc_exchange.BlobMem = VIRTGPU_BLOB_MEM_HOST3D;
   alloc_exchange.BlobFlags = 0;
   if (mappable)
      alloc_exchange.BlobFlags |= VIRTGPU_BLOB_FLAG_USE_MAPPABLE;
   if (res->venus_mem_id) {
      alloc_exchange.BlobFlags |= VIRTGPU_BLOB_FLAG_USE_SHAREABLE;
   }

   create.NumAllocations = 1;
   create.pAllocationInfo = &alloc_info;
   create.pPrivateDriverData = &res_exchange;
   create.PrivateDriverDataSize = sizeof(res_exchange);
   create.force_allocation_handle =
      !res->display_target && !res->primary_target &&
      !(base->bind & PIPE_BIND_SHARED);

   alloc_info.pPrivateDriverData = &alloc_exchange;
   alloc_info.PrivateDriverDataSize = sizeof(alloc_exchange);

   NTSTATUS status = screen->device->createAllocation(screen->device, &create);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: %s allocation failed status=0x%lx %ux%u format=%u bind=0x%x size=0x%llx classic=%u mem_id=0x%llx\n",
                   label, status, base->width0, base->height0,
                   base->format, base->bind,
                   (unsigned long long)res->size, res->classic_display,
                   (unsigned long long)res->venus_mem_id);
      return false;
   }

   res->hResource = create.hResource;
   res->hAllocationResource = create.hAllocationResource;
   res->hResourceIsD3D9Runtime = create.hResourceIsD3D9Runtime;
   res->hAllocation = alloc_info.hAllocation;

   VIOGPU_ESCAPE resinfo;
   memset(&resinfo, 0, sizeof(resinfo));
   resinfo.Type = VIOGPU_RES_INFO;
   resinfo.DataLength = sizeof(VIOGPU_RES_INFO_REQ);
   resinfo.ResourceInfo.ResHandle = res->hAllocation;

   status = screen->device->escape(screen->device, &resinfo, sizeof(resinfo));
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: VIOGPU_RES_INFO failed status=0x%lx hAllocation=0x%lx\n",
                   status, (unsigned long)res->hAllocation);
      screen->device->destroyAllocation(screen->device, res->hResource,
                                        res->hAllocation);
      res->hResource = NULL;
      res->hAllocationResource = NULL;
      res->hResourceIsD3D9Runtime = false;
      res->hAllocation = 0;
      return false;
   }

   res->venus_res_id = resinfo.ResourceInfo.Id;

   YTTRIUM_LOG("yttrium: %s allocation hAllocation=0x%lx res_id=%u mem_id=0x%llx blob_mem=0x%lx blob_flags=0x%lx size=0x%llx stride=%u primary=%u classic=%u\n",
               label, (unsigned long)res->hAllocation, res->venus_res_id,
               (unsigned long long)res->venus_mem_id,
               (unsigned long)alloc_exchange.BlobMem,
               (unsigned long)alloc_exchange.BlobFlags,
               (unsigned long long)res->size, res->stride,
               res->primary_target, res->classic_display);
   return true;
}

struct pipe_resource *
yttrium_resource_create(struct pipe_screen *pscreen,
                        const struct pipe_resource *templ)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   struct yttrium_resource *res = CALLOC_STRUCT(yttrium_resource);
   if (!res)
      return NULL;

   res->cache_id = yttrium_next_cache_id();
   res->base = *templ;
   res->base.screen = pscreen;
   res->primary_target = (templ->flags & PIPE_RESOURCE_FLAG_FRONTEND_PRIV) != 0;
   res->venus.cpu_readback =
      (templ->flags & YTTRIUM_GDI_RESOURCE_FLAG_CPU_READBACK) != 0;
   res->base.flags &= ~(PIPE_RESOURCE_FLAG_FRONTEND_PRIV |
                        YTTRIUM_GDI_RESOURCE_FLAG_CPU_READBACK);
   pipe_reference_init(&res->base.reference, 1);
   yttrium_resource_init_threaded(res);
   res->venus.owner = &res->base;

   res->size = yttrium_resource_size(templ, &res->stride, &res->layer_stride);
   res->display_target = templ->bind & PIPE_BIND_DISPLAY_TARGET;
   res->owns_allocation = true;
   const bool shared_target = (templ->bind & PIPE_BIND_SHARED) != 0;
   const bool shared_display_target = shared_target && res->display_target;
   const bool depth_stencil_target =
      (templ->bind & PIPE_BIND_DEPTH_STENCIL) != 0 &&
      templ->target != PIPE_BUFFER;
   const bool offscreen_color_target =
      (templ->bind & PIPE_BIND_RENDER_TARGET) != 0 &&
      !res->display_target && !depth_stencil_target &&
      templ->target != PIPE_BUFFER;
   const bool shader_image_target =
      (templ->bind & PIPE_BIND_SHADER_IMAGE) != 0 &&
      templ->target != PIPE_BUFFER;
   const bool stream_output_buffer =
      templ->target == PIPE_BUFFER &&
      (templ->bind & PIPE_BIND_STREAM_OUTPUT) != 0;
   const bool ordered_worker_enabled =
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_WORKER", true);
   const bool ordered_worker_upload_buffer =
      templ->target == PIPE_BUFFER &&
      (templ->flags & PIPE_RESOURCE_FLAG_SINGLE_THREAD_USE) != 0 &&
      ordered_worker_enabled;
   const bool ordered_worker_upload_direct_backing =
      ordered_worker_upload_buffer &&
      (templ->bind & PIPE_BIND_CONSTANT_BUFFER) != 0 &&
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_CONSTANT_BUFFER_PUBLICATION", true);
   const bool buffer_replacement_enabled =
      ordered_worker_enabled &&
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_BUFFER_REPLACEMENT", true);
   res->ordered_worker_upload_buffer = ordered_worker_upload_buffer;
   res->ordered_worker_upload_direct_backing =
      ordered_worker_upload_direct_backing;
   /* Stream-output buffers may also be rebound as vertex buffers.  The
    * Venus SO buffer is created with vertex-buffer usage, so do not allocate
    * a second mappable guest buffer for the PIPE_BIND_VERTEX_BUFFER bit.
    * u_threaded_context upload buffers are private CPU staging and likewise
    * do not need a KMD allocation.
    */
   const bool allocation_backed_buffer =
      templ->target == PIPE_BUFFER &&
      !stream_output_buffer && !ordered_worker_upload_buffer &&
      (templ->bind & (PIPE_BIND_VERTEX_BUFFER |
                      PIPE_BIND_INDEX_BUFFER |
                      PIPE_BIND_COMMAND_ARGS_BUFFER)) != 0;
   const bool replacement_candidate =
      buffer_replacement_enabled && allocation_backed_buffer &&
      !shared_target &&
      (templ->usage == PIPE_USAGE_DYNAMIC ||
       templ->usage == PIPE_USAGE_STREAM);

   if (replacement_candidate) {
      struct yttrium_buffer_storage *pooled =
         yttrium_buffer_storage_pool_acquire(screen, res->size);
      if (pooled) {
         res->replacement_storage = pooled;
         yttrium_resource_apply_buffer_storage(res, pooled);
         threaded_resource(&res->base)->is_shared = false;
         memset(res->data, 0, res->size);
         YTTRIUM_LOG("yttrium: buffer replacement backing reused resource=%p hAllocation=0x%lx res_id=%u data=%p size=0x%llx capacity=0x%llx bind=0x%x\n",
                     (void *)res, (unsigned long)res->hAllocation,
                     res->venus_res_id, res->data,
                     (unsigned long long)res->size,
                     (unsigned long long)res->data_capacity,
                     res->base.bind);
      }
   }

   YTTRIUM_LOG("yttrium: resource_create target=%u %ux%ux%u array=%u levels=%u format=%u bind=0x%x usage=%u flags=0x%x size=0x%llx stride=%u layer_stride=%u display=%u primary=%u shared=%u shared_display=%u\n",
                templ->target,
                templ->width0,
                templ->height0,
                templ->depth0,
                templ->array_size,
                templ->last_level + 1,
                templ->format,
                templ->bind,
                templ->usage,
                templ->flags,
                (unsigned long long)res->size,
                res->stride,
                res->layer_stride,
                res->display_target,
                res->primary_target,
                shared_target,
                shared_display_target);

   if (res->display_target) {
      const uint64_t display_size = res->size;
      uint64_t venus_allocation_size = 0;
      bool venus_backed = false;
      const char *display_path = "unset";
      const bool enable_display_image =
         yttrium_gdi_debug_get_bool_option(
            "D3D10UMD_YTTRIUM_ENABLE_DISPLAY_IMAGE", true);
      const bool force_display_buffer =
         yttrium_gdi_debug_get_bool_option(
            "D3D10UMD_YTTRIUM_FORCE_DISPLAY_BUFFER", false);
      const bool use_display_buffer =
         force_display_buffer || !enable_display_image;

      YTTRIUM_LOG("yttrium: create display target %ux%u format=%u bind=0x%x primary=%u shared=%u enable_image=%u force_buffer=%u export_sensitive=%u cpu_readback=%u\n",
                   templ->width0, templ->height0, templ->format, templ->bind,
                   res->primary_target, shared_target, enable_display_image,
                   force_display_buffer, shared_display_target,
                   res->venus.cpu_readback);

      if (use_display_buffer) {
         YTTRIUM_LOG("yttrium: using Venus display buffer allocation path=%s\n",
                     force_display_buffer ? "forced" :
                        "display-image-disabled");
         res->size = display_size;
         if (yttrium_venus_create_display_buffer(screen->venus, &res->venus,
                                                  res->size,
                                                  &res->venus_mem_id)) {
            venus_backed = true;
            display_path = force_display_buffer ? "forced-buffer" :
               "display-image-disabled-buffer";
         } else {
            YTTRIUM_WARN("yttrium: Venus display buffer allocation failed for display target %ux%u format=%u bind=0x%x; refusing classic fallback\n",
                         templ->width0, templ->height0, templ->format,
                         templ->bind);
            yttrium_resource_free(res);
            return NULL;
         }
      } else {
         if (yttrium_venus_create_display_image(screen->venus, &res->venus,
                                                templ->width0, templ->height0,
                                                templ->format, display_size,
                                                &res->venus_mem_id,
                                                &venus_allocation_size)) {
            const unsigned cpp = util_format_get_blocksize(templ->format);
            const uint64_t min_row = (uint64_t)templ->width0 * cpp;
            if (cpp && res->venus.image_row_pitch >= min_row &&
                res->venus.image_row_pitch <= UINT32_MAX) {
               res->stride = (unsigned)res->venus.image_row_pitch;
               res->layer_stride =
                  (unsigned)MIN2((uint64_t)UINT32_MAX,
                                 res->venus.image_array_pitch ?
                                    res->venus.image_array_pitch :
                                    res->venus.image_row_pitch *
                                       MAX2(templ->height0, 1));
            }
            const uint64_t venus_layout_size =
               res->venus.image_offset + res->venus.image_size;
            res->size = MAX2(display_size,
                              MAX2(venus_allocation_size,
                                   venus_layout_size));
            venus_backed = true;
            display_path = "image";
         } else {
            YTTRIUM_WARN("yttrium: Venus display image allocation failed for display target %ux%u format=%u bind=0x%x; refusing implicit buffer fallback\n",
                         templ->width0, templ->height0, templ->format,
                         templ->bind);
            yttrium_resource_free(res);
            return NULL;
         }
      }

      if (!yttrium_create_guest_allocation(pscreen, res, "display", true)) {
         if (venus_backed) {
            YTTRIUM_WARN("yttrium: Venus-backed display allocation failed for display target %ux%u format=%u bind=0x%x path=%s buffer_backed=%u; refusing implicit fallback\n",
                         templ->width0, templ->height0, templ->format,
                         templ->bind, display_path,
                         res->venus.buffer_backed);
            yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                        NULL);
         }

         yttrium_resource_free(res);
         return NULL;
      }

      YTTRIUM_LOG("yttrium: display backing selected path=%s hAllocation=0x%lx res_id=%u shared_display=%u classic=%u venus_initialized=%u buffer_backed=%u image_id=%llu buffer_id=%llu mem_id=0x%llx size=0x%llx stride=%u\n",
                   display_path,
                   (unsigned long)res->hAllocation,
                   res->venus_res_id,
                   shared_display_target,
                   res->classic_display,
                   res->venus.initialized,
                   res->venus.buffer_backed,
                   (unsigned long long)res->venus.image_obj.id,
                   (unsigned long long)res->venus.buffer_obj.id,
                   (unsigned long long)res->venus_mem_id,
                   (unsigned long long)res->size,
                   res->stride);

   } else {
      if (ordered_worker_upload_direct_backing) {
         struct yttrium_ordered_upload_pool_entry pooled;
         if (yttrium_ordered_upload_pool_acquire(
                screen, res->size, true, &pooled)) {
            res->data = pooled.data;
            res->data_capacity = pooled.capacity;
            res->hAllocation = pooled.hAllocation;
            res->hResource = pooled.hResource;
            res->hAllocationResource = pooled.hAllocationResource;
            res->hResourceIsD3D9Runtime = pooled.hResourceIsD3D9Runtime;
            res->venus_res_id = pooled.venus_res_id;
            res->venus_mem_id = pooled.venus_mem_id;
            res->map = pooled.data;
            res->map_info = pooled.map_info;
            res->map_is_blob = pooled.map_is_blob;
            res->owns_allocation = pooled.owns_allocation;
            res->allocation_destroyed_by_runtime =
               pooled.allocation_destroyed_by_runtime;
            res->venus = pooled.venus;
            res->venus.owner = &res->base;
            yttrium_ordered_upload_rehome_venus_handles(&res->venus);
            YTTRIUM_LOG("yttrium: ordered worker direct upload backing reused resource=%p hAllocation=0x%lx res_id=%u memory_id=%llu buffer_id=%llu data=%p size=0x%llx capacity=0x%llx\n",
                        (void *)res, (unsigned long)res->hAllocation,
                        res->venus_res_id,
                        (unsigned long long)res->venus.memory_obj.id,
                        (unsigned long long)res->venus.buffer_obj.id,
                        res->data, (unsigned long long)res->size,
                        (unsigned long long)res->data_capacity);
         } else {
            if (!yttrium_venus_create_bind_buffer(
                   screen->venus, &res->venus, res->size,
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   &res->venus_mem_id)) {
               YTTRIUM_WARN("yttrium: ordered worker direct upload backing create failed owner=yttrium_resource reason=venus_uniform_buffer_create_failed resource=%p size=0x%llx\n",
                            (void *)res, (unsigned long long)res->size);
               yttrium_resource_free(res);
               return NULL;
            }
            if (!yttrium_create_guest_allocation(
                   pscreen, res, "ordered direct upload", true)) {
               YTTRIUM_WARN("yttrium: ordered worker direct upload backing create failed owner=yttrium_resource reason=guest_allocation_create_failed resource=%p size=0x%llx memory_id=%llu\n",
                            (void *)res, (unsigned long long)res->size,
                            (unsigned long long)res->venus.memory_obj.id);
               yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                           NULL);
               yttrium_resource_free(res);
               return NULL;
            }
            if (!yttrium_map_allocation(
                   pscreen, res->hAllocation, res->size,
                   "ordered direct upload", &res->map, &res->map_info,
                   &res->map_is_blob)) {
               YTTRIUM_WARN("yttrium: ordered worker direct upload backing create failed owner=yttrium_resource reason=guest_allocation_map_failed resource=%p hAllocation=0x%lx size=0x%llx\n",
                            (void *)res, (unsigned long)res->hAllocation,
                            (unsigned long long)res->size);
               screen->device->destroyAllocation(screen->device,
                                                 res->hResource,
                                                 res->hAllocation);
               yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                           NULL);
               yttrium_resource_free(res);
               return NULL;
            }
            res->data = res->map;
            res->data_capacity = res->size;
            YTTRIUM_LOG("yttrium: ordered worker direct upload backing created resource=%p hAllocation=0x%lx res_id=%u memory_id=%llu buffer_id=%llu data=%p size=0x%llx\n",
                        (void *)res, (unsigned long)res->hAllocation,
                        res->venus_res_id,
                        (unsigned long long)res->venus.memory_obj.id,
                        (unsigned long long)res->venus.buffer_obj.id,
                        res->data, (unsigned long long)res->size);
         }
      }

      if (stream_output_buffer) {
         if (yttrium_venus_create_stream_output_buffer(screen->venus,
                                                       &res->venus,
                                                       res->size,
                                                       &res->venus_mem_id)) {
            res->venus_res_id = (uint32_t)res->venus.memory_obj.id;
         } else {
            YTTRIUM_WARN("yttrium: Venus stream output buffer allocation unavailable resource=%p size=0x%llx bind=0x%x; refusing CPU/mappable-blob fallback\n",
                         (void *)res,
                         (unsigned long long)res->size,
                         templ->bind);
            yttrium_resource_free(res);
            return NULL;
         }
      }

      if (allocation_backed_buffer && !res->replacement_storage) {
         const bool command_args_buffer =
            (templ->bind & PIPE_BIND_COMMAND_ARGS_BUFFER) != 0;
         const bool venus_bind_buffer =
            yttrium_gdi_debug_get_bool_option(
               "D3D10UMD_YTTRIUM_BIND_VERTEX_BUFFER", true);
         const bool bind_buffer_dynamic =
            templ->usage == PIPE_USAGE_DYNAMIC ||
            templ->usage == PIPE_USAGE_STREAM;
         if (command_args_buffer || (venus_bind_buffer && !bind_buffer_dynamic)) {
            VkBufferUsageFlags bind_usage = 0;
            if (templ->bind & PIPE_BIND_VERTEX_BUFFER)
               bind_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            if (templ->bind & PIPE_BIND_INDEX_BUFFER)
               bind_usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            if (templ->bind & PIPE_BIND_SAMPLER_VIEW)
               bind_usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
            if (templ->bind & PIPE_BIND_SHADER_IMAGE)
               bind_usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
            if (templ->bind & PIPE_BIND_SHADER_BUFFER)
               bind_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            if (templ->bind & PIPE_BIND_COMMAND_ARGS_BUFFER)
               bind_usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

            const unsigned draw_bind_mask =
               PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_INDEX_BUFFER;
            const bool private_immutable_draw_buffer =
               templ->usage == PIPE_USAGE_IMMUTABLE &&
               (templ->bind & draw_bind_mask) != 0 &&
               (templ->bind & ~draw_bind_mask) == 0;
            uint64_t *out_memory_id = private_immutable_draw_buffer ?
               NULL : &res->venus_mem_id;

            if (bind_usage &&
                !yttrium_venus_create_bind_buffer(screen->venus, &res->venus,
                                                  res->size, bind_usage,
                                                  out_memory_id)) {
               if (command_args_buffer) {
                  YTTRIUM_WARN("yttrium: command buffer allocation failed owner=yttrium_resource reason=native-indirect-buffer-create-failed action=abort-resource\n");
                  yttrium_resource_free(res);
                  return NULL;
               }
               if (private_immutable_draw_buffer) {
                  YTTRIUM_WARN("yttrium: ERROR: private immutable draw buffer create failed owner=resource_create component=immutable-draw-buffer reason=venus_bind_buffer_create_failed action=abort-resource resource=%p size=0x%llx bind=0x%x usage=0x%x\n",
                               (void *)res,
                               (unsigned long long)res->size,
                               templ->bind, bind_usage);
                  yttrium_resource_free(res);
                  return NULL;
               }
               YTTRIUM_WARN("yttrium: WARNING: allocation-backed buffer owner=resource_create direct_bind=create_failed path=per_draw_upload reason=venus_bind_buffer_create_failed resource=%p size=0x%llx bind=0x%x usage=0x%x\n",
                            (void *)res,
                            (unsigned long long)res->size,
                            templ->bind, bind_usage);
            } else if (private_immutable_draw_buffer) {
               res->private_immutable_draw_buffer = true;
            }
         }

         if (!yttrium_create_guest_allocation(pscreen, res, "buffer", true)) {
            yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                        NULL);
            yttrium_resource_free(res);
            return NULL;
         }

         if (!yttrium_map_allocation(pscreen, res->hAllocation, res->size,
                                     "buffer", &res->map, &res->map_info,
                                     &res->map_is_blob)) {
            YTTRIUM_WARN("yttrium: buffer allocation map failed hAllocation=0x%lx size=0x%llx bind=0x%x\n",
                         (unsigned long)res->hAllocation,
                         (unsigned long long)res->size,
                         templ->bind);
            screen->device->destroyAllocation(screen->device, res->hResource,
                                              res->hAllocation);
            yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                        NULL);
            yttrium_resource_free(res);
            return NULL;
         }

         res->data = res->map;
         memset(res->data, 0, res->size);
         if (res->private_immutable_draw_buffer) {
            res->data_dirty = true;
            res->contents_serial++;
         }
         YTTRIUM_LOG("yttrium: allocation-backed buffer resource=%p hAllocation=0x%lx res_id=%u data=%p size=0x%llx bind=0x%x\n",
                     (void *)res,
                     (unsigned long)res->hAllocation,
                     res->venus_res_id,
                     res->data,
                     (unsigned long long)res->size,
                     res->base.bind);
      }

      if (offscreen_color_target || depth_stencil_target ||
          shader_image_target) {
         uint64_t venus_allocation_size = 0;
         const bool shader_resource =
            (templ->bind & PIPE_BIND_SAMPLER_VIEW) != 0;
         const bool legacy_shape_fallback =
            (templ->target != PIPE_TEXTURE_2D &&
             templ->target != PIPE_TEXTURE_2D_ARRAY) ||
            (depth_stencil_target &&
             (templ->last_level != 0 ||
              templ->array_size > 1 ||
              templ->depth0 != 1));
         const bool crash_containment_fallback =
            legacy_shape_fallback ||
            (depth_stencil_target && shader_resource);

         if (yttrium_venus_create_texture_image_for_bind(
                   screen->venus, &res->venus, templ->target,
                   templ->width0, templ->height0, templ->depth0,
                   templ->last_level + 1, templ->array_size,
                   templ->format, templ->nr_samples, templ->bind,
                   &venus_allocation_size)) {
            res->size = MAX2(res->size, venus_allocation_size);
            res->venus_res_id = (uint32_t)res->venus.memory_obj.id;
            if (shared_target)
               res->venus_mem_id = res->venus.memory_obj.id;
            YTTRIUM_LOG("yttrium: Venus-backed bind texture target created resource=%p image_id=%llu memory_id=%llu res_id=%u size=0x%llx target=%u format=%u extent=%ux%ux%u levels=%u array=%u bind=0x%x usage=0x%x\n",
                        (void *)res,
                        (unsigned long long)res->venus.image_obj.id,
                        (unsigned long long)res->venus.memory_obj.id,
                        res->venus_res_id,
                        (unsigned long long)res->size,
                        templ->target, templ->format, templ->width0,
                        templ->height0, templ->depth0,
                        templ->last_level + 1, templ->array_size,
                        templ->bind, res->venus.image_usage);
         } else if (crash_containment_fallback) {
            YTTRIUM_WARN("yttrium: software fallback native bind texture image unavailable; keeping temporary CPU storage resource=%p target=%u format=%u bind=0x%x extent=%ux%ux%u levels=%u array=%u depth_stencil=%u sampler=%u legacy_shape=%u\n",
                         (void *)res, templ->target, templ->format,
                         templ->bind, templ->width0, templ->height0,
                         templ->depth0, templ->last_level + 1,
                         templ->array_size, depth_stencil_target,
                         shader_resource, legacy_shape_fallback);
         } else {
            YTTRIUM_WARN("yttrium: Venus bind texture target allocation failed resource=%p target=%u format=%u bind=0x%x extent=%ux%ux%u levels=%u array=%u; refusing CPU backing\n",
                         (void *)res, templ->target, templ->format,
                         templ->bind, templ->width0, templ->height0,
                         templ->depth0, templ->last_level + 1,
                         templ->array_size);
            yttrium_resource_free(res);
            return NULL;
         }
      }

      if ((offscreen_color_target || depth_stencil_target ||
           shader_image_target) &&
          res->venus.initialized && !res->venus.buffer_backed &&
          res->venus.image) {
         if (shared_target &&
             !yttrium_create_guest_allocation(pscreen, res, "shared", false)) {
            YTTRIUM_WARN("yttrium: shared bind texture allocation failed resource=%p target=%u format=%u bind=0x%x extent=%ux%ux%u levels=%u array=%u; refusing unexportable shared resource\n",
                         (void *)res, templ->target, templ->format,
                         templ->bind, templ->width0, templ->height0,
                         templ->depth0, templ->last_level + 1,
                         templ->array_size);
            yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                        NULL);
            yttrium_resource_free(res);
            return NULL;
         }
         YTTRIUM_LOG("yttrium: render/depth target uses Venus-only backing resource=%p size=0x%llx bind=0x%x image_id=%llu format=%u\n",
                     (void *)res,
                     (unsigned long long)res->size,
                     res->base.bind,
                     (unsigned long long)res->venus.image_obj.id,
                     res->base.format);
      } else {
         if (!res->data) {
            if (res->size > SIZE_MAX) {
               yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                           NULL);
               yttrium_resource_free(res);
               return NULL;
            }

            if (ordered_worker_upload_buffer) {
               struct yttrium_ordered_upload_pool_entry pooled;
               if (yttrium_ordered_upload_pool_acquire(
                      screen, res->size, false, &pooled)) {
                  res->data = pooled.data;
                  res->data_capacity = pooled.capacity;
               }
               if (!res->data) {
                  res->data = MALLOC((size_t)res->size);
                  res->data_capacity = res->size;
               }
            } else {
               res->data = CALLOC(1, (size_t)res->size);
               res->data_capacity = res->size;
            }
            res->owns_data = true;
         }
         if (!res->data) {
            yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                        NULL);
            yttrium_resource_free(res);
            return NULL;
         }
         YTTRIUM_LOG("yttrium: CPU-backed resource created resource=%p data=%p size=0x%llx bind=0x%x venus_initialized=%u image_id=%llu\n",
                     (void *)res,
                     res->data,
                     (unsigned long long)res->size,
                     res->base.bind,
                     res->venus.initialized,
                     (unsigned long long)res->venus.image_obj.id);
         /* A newly allocated stream-output shadow does not initialize its
          * separate device-local buffer.  Only a real CPU write should make
          * that shadow authoritative; transfer_unmap publishes such writes.
          */
         if (!stream_output_buffer) {
            res->data_dirty = true;
            res->contents_serial++;
         }
         if (ordered_worker_upload_buffer &&
             !ordered_worker_upload_direct_backing)
             res->owns_allocation = false;
      }
   }

   const bool replacement_eligible =
      replacement_candidate && !res->venus.initialized &&
      res->owns_allocation && !res->allocation_destroyed_by_runtime &&
      res->hAllocation && res->map && res->data == res->map &&
      !res->owns_data;
   if (replacement_eligible &&
       (res->replacement_storage || yttrium_buffer_storage_attach(res)))
      threaded_resource(&res->base)->is_shared = false;

   return &res->base;
}

static void
yttrium_trace_resource_destroy_state(uint32_t stage,
                                     NTSTATUS status,
                                     const struct yttrium_resource *res)
{
   if (!res)
      return;

   yttrium_trace_resource_destroy(
      stage, status, (uint64_t)(uintptr_t)res,
      (uint64_t)(uintptr_t)res->hResource, (uint64_t)res->hAllocation,
      res->venus_res_id, res->venus_mem_id, res->display_target,
      res->primary_target, res->classic_display, res->owns_allocation,
      res->venus.initialized, res->venus.buffer_backed,
      res->venus.image_obj.id, res->venus.buffer_obj.id,
      res->venus.memory_obj.id, 0, 0, 0,
      res->size, res->stride);
}

void
yttrium_resource_destroy(struct pipe_screen *pscreen,
                         struct pipe_resource *resource)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   struct yttrium_resource *res = yttrium_resource(resource);

   yttrium_trace_resource_destroy_state(
      YTTRIUM_TRACE_RESOURCE_DESTROY_BEGIN, 0, res);

   YTTRIUM_LOG("yttrium: resource_destroy begin resource=%p hAllocation=0x%lx hResource=%p res_id=%u mem_id=0x%llx display=%u primary=%u classic=%u owns=%u size=0x%llx stride=%u venus=(init=%u buffer=%u image_id=%llu buffer_id=%llu memory_id=%llu)\n",
                (void *)res,
                (unsigned long)res->hAllocation,
                res->hResource,
                res->venus_res_id,
                (unsigned long long)res->venus_mem_id,
                res->display_target,
                res->primary_target,
                res->classic_display,
                res->owns_allocation,
                (unsigned long long)res->size,
                res->stride,
                res->venus.initialized,
                res->venus.buffer_backed,
                (unsigned long long)res->venus.image_obj.id,
                (unsigned long long)res->venus.buffer_obj.id,
                (unsigned long long)res->venus.memory_obj.id);

   if (res->replacement_storage) {
      assert(res->base.target == PIPE_BUFFER);
      assert(!res->venus.initialized);
      yttrium_resource_detach_buffer_storage(res);
   }

   /* The submitted batch owns the pipe-resource reference, so reaching
    * resource_destroy proves that no GPU work can still read this backing. */
   if (res->ordered_worker_upload_direct_backing &&
       yttrium_ordered_upload_pool_return(screen, res)) {
      YTTRIUM_LOG("yttrium: ordered worker direct upload backing pooled resource=%p size=0x%llx\n",
                  (void *)res, (unsigned long long)res->size);
      yttrium_trace_resource_destroy_state(
         YTTRIUM_TRACE_RESOURCE_DESTROY_END, 0, res);
      yttrium_resource_free(res);
      return;
   }

   struct yttrium_venus_allocation_snapshot allocation_snapshot;
   struct yttrium_venus_allocation_snapshot *allocation_snapshot_ptr = NULL;
   memset(&allocation_snapshot, 0, sizeof(allocation_snapshot));
   if (res->hAllocation) {
      allocation_snapshot.hAllocation = res->hAllocation;
      allocation_snapshot.hResource = res->hResource;
      allocation_snapshot.hAllocationResource = res->hAllocationResource;
      allocation_snapshot.hResourceIsD3D9Runtime =
         res->hResourceIsD3D9Runtime;
      allocation_snapshot.size =
         res->ordered_worker_upload_direct_backing && res->data_capacity ?
            res->data_capacity : res->size;
      allocation_snapshot.map = res->map;
      allocation_snapshot.map_info = res->map_info;
      allocation_snapshot.map_is_blob = res->map_is_blob;
      allocation_snapshot.owns_allocation = res->owns_allocation;
      allocation_snapshot.allocation_destroyed_by_runtime =
         res->allocation_destroyed_by_runtime;
      allocation_snapshot_ptr = &allocation_snapshot;
   }

   bool allocation_adopted =
      yttrium_venus_resource_fini(screen->venus, NULL, &res->venus,
                                  allocation_snapshot_ptr);

   if (allocation_adopted) {
      YTTRIUM_LOG("yttrium: resource_destroy deferred allocation to Venus batch hAllocation=0x%lx hResource=%p size=0x%llx map=%p map_blob=%u owns=%u\n",
                  (unsigned long)allocation_snapshot.hAllocation,
                  allocation_snapshot.hResource,
                  (unsigned long long)allocation_snapshot.size,
                  allocation_snapshot.map,
                  allocation_snapshot.map_is_blob,
                  allocation_snapshot.owns_allocation);
      if (allocation_snapshot.map && res->data == allocation_snapshot.map)
         res->data = NULL;
      res->hAllocation = 0;
      res->hResource = NULL;
      res->hAllocationResource = NULL;
      res->hResourceIsD3D9Runtime = false;
      res->map = NULL;
      res->map_info = 0;
      res->map_is_blob = false;
      res->owns_allocation = false;
      res->allocation_destroyed_by_runtime = false;
   }

   if (res->hAllocation) {
      if (res->allocation_destroyed_by_runtime) {
         YTTRIUM_LOG("yttrium: release runtime-owned allocation without "
                     "DeallocateCb hAllocation=0x%lx res_id=%u\n",
                     (unsigned long)res->hAllocation, res->venus_res_id);
         yttrium_trace_resource_destroy_state(
            YTTRIUM_TRACE_RESOURCE_DESTROY_ALLOCATION_CLOSED, 0, res);
      } else {
         yttrium_unmap_display_allocation(pscreen, res);

         if (res->owns_allocation) {
            NTSTATUS status =
               screen->device->destroyAllocation(screen->device,
                                                 res->hResource,
                                                 res->hAllocation);
            YTTRIUM_LOG("yttrium: destroy allocation hAllocation=0x%lx hResource=%p status=0x%lx\n",
                        (unsigned long)res->hAllocation,
                        res->hResource,
                        (unsigned long)status);
            yttrium_trace_resource_destroy_state(
               YTTRIUM_TRACE_RESOURCE_DESTROY_ALLOCATION_CLOSED, status, res);
            if (!NT_SUCCESS(status)) {
               YTTRIUM_WARN("yttrium: destroy allocation failed status=0x%lx hAllocation=0x%lx hResource=%p resource=%p target=%u format=%u bind=0x%x width=%u height=%u size=0x%llx stride=%u display=%u primary=%u classic=%u owns=%u map=%p map_blob=%u venus_init=%u venus_buffer=%u venus_image=%llu venus_buffer_id=%llu res_id=%u mem_id=0x%llx\n",
                            status, (unsigned long)res->hAllocation,
                            res->hResource, (void *)res, res->base.target,
                            res->base.format, res->base.bind,
                            res->base.width0, res->base.height0,
                            (unsigned long long)res->size, res->stride,
                            res->display_target, res->primary_target,
                            res->classic_display, res->owns_allocation,
                            res->map, res->map_is_blob,
                            res->venus.initialized, res->venus.buffer_backed,
                            (unsigned long long)res->venus.image_obj.id,
                            (unsigned long long)res->venus.buffer_obj.id,
                            res->venus_res_id,
                            (unsigned long long)res->venus_mem_id);
            }
         } else {
            NTSTATUS status =
               screen->device->destroyAllocation(screen->device,
                                                 res->hResource,
                                                 res->hAllocation);
            YTTRIUM_LOG("yttrium: close opened allocation hAllocation=0x%lx hResource=%p status=0x%lx\n",
                        (unsigned long)res->hAllocation,
                        res->hResource,
                        (unsigned long)status);
            yttrium_trace_resource_destroy_state(
               YTTRIUM_TRACE_RESOURCE_DESTROY_ALLOCATION_CLOSED, status, res);
            if (!NT_SUCCESS(status)) {
               YTTRIUM_WARN("yttrium: close opened allocation failed status=0x%lx hAllocation=0x%lx hResource=%p resource=%p target=%u format=%u bind=0x%x width=%u height=%u size=0x%llx stride=%u display=%u primary=%u classic=%u owns=%u map=%p map_blob=%u venus_init=%u venus_buffer=%u venus_image=%llu venus_buffer_id=%llu res_id=%u mem_id=0x%llx\n",
                            status, (unsigned long)res->hAllocation,
                            res->hResource, (void *)res, res->base.target,
                            res->base.format, res->base.bind,
                            res->base.width0, res->base.height0,
                            (unsigned long long)res->size, res->stride,
                            res->display_target, res->primary_target,
                            res->classic_display, res->owns_allocation,
                            res->map, res->map_is_blob,
                            res->venus.initialized, res->venus.buffer_backed,
                            (unsigned long long)res->venus.image_obj.id,
                            (unsigned long long)res->venus.buffer_obj.id,
                            res->venus_res_id,
                            (unsigned long long)res->venus_mem_id);
            }
         }
      }
   }

   YTTRIUM_LOG("yttrium: resource_destroy end resource=%p hAllocation=0x%lx hResource=%p res_id=%u mem_id=0x%llx owns=%u\n",
                (void *)res,
                (unsigned long)res->hAllocation,
                res->hResource,
                res->venus_res_id,
                (unsigned long long)res->venus_mem_id,
                res->owns_allocation);
   yttrium_trace_resource_destroy_state(
      YTTRIUM_TRACE_RESOURCE_DESTROY_END, 0, res);

   if (res->ordered_worker_upload_buffer && res->owns_data && res->data)
      yttrium_ordered_upload_pool_return(screen, res);

   if (res->owns_data)
      FREE(res->data);
   yttrium_resource_free(res);
}

bool
yttrium_resource_get_handle(struct pipe_screen *pscreen,
                            struct pipe_context *ctx,
                            struct pipe_resource *resource,
                            struct winsys_handle *whandle,
                            unsigned usage)
{
   struct yttrium_resource *res = yttrium_resource(resource);

   if (!res || whandle->type != WINSYS_HANDLE_TYPE_D3DKMT_ALLOC ||
       !res->hAllocation)
      return false;

   whandle->handle = (HANDLE)(uintptr_t)res->hAllocation;
   whandle->stride = res->stride;
   whandle->array_stride = res->layer_stride;
   whandle->size = res->size;
   YTTRIUM_LOG("yttrium: get_handle type=%u usage=0x%x handle=0x%lx hAllocation=0x%lx res_id=%u %ux%u format=%u bind=0x%x flags=0x%x primary=%u classic=%u stride=%u size=0x%llx\n",
                whandle->type, usage,
                (unsigned long)(uintptr_t)whandle->handle,
                (unsigned long)res->hAllocation,
                res->venus_res_id,
                res->base.width0,
                res->base.height0,
                res->base.format,
                res->base.bind,
                res->base.flags,
                res->primary_target,
                res->classic_display,
                res->stride,
                (unsigned long long)res->size);
   return true;
}

static void
yttrium_trace_resource_import_state(uint32_t stage,
                                    NTSTATUS status,
                                    const struct gdikmt_openallocation *open,
                                    const struct pipe_resource *templ,
                                    const struct yttrium_resource *res,
                                    uint32_t import_enabled)
{
   const struct pipe_resource *base = res ? &res->base : templ;
   const uint64_t global_handle = open ? open->hGlobalHandle : 0;
   const uint64_t resource_handle =
      res ? (uint64_t)(uintptr_t)res->hResource :
      open ? (uint64_t)(uintptr_t)open->hResource : 0;
   const uint64_t allocation = res ? res->hAllocation : 0;
   const uint32_t display =
      res ? res->display_target :
      base ? ((base->bind & PIPE_BIND_DISPLAY_TARGET) != 0) : 0;

   yttrium_trace_resource_import(
      stage, status, global_handle, resource_handle, allocation,
      res ? res->venus_res_id : 0, res ? res->venus_mem_id : 0,
      base ? base->width0 : 0, base ? base->height0 : 0,
      base ? base->format : PIPE_FORMAT_NONE, base ? base->bind : 0,
      base ? base->flags : 0, base ? base->usage : 0, display,
      res ? res->classic_display : 0, res ? res->owns_allocation : 0,
      res ? res->venus.initialized : 0, import_enabled,
      res ? res->venus.image_obj.id : 0, res ? res->venus.buffer_obj.id : 0,
      res ? res->size : 0, res ? res->stride : 0);
}

struct pipe_resource *
yttrium_resource_from_handle(struct pipe_screen *pscreen,
                             const struct pipe_resource *templ,
                             struct winsys_handle *whandle,
                             unsigned usage)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   struct gdikmt_openallocation open;
   D3DDDI_OPENALLOCATIONINFO *open_info = NULL;
   struct yttrium_resource *res = NULL;
   NTSTATUS status;

   if (!whandle || whandle->type != WINSYS_HANDLE_TYPE_WIN32_HANDLE ||
       !whandle->handle) {
      YTTRIUM_LOG("yttrium: unsupported resource_from_handle type=%u handle=%p\n",
                   whandle ? whandle->type : 0,
                   whandle ? whandle->handle : NULL);
      return NULL;
   }

   YTTRIUM_LOG("yttrium: resource_from_handle type=%u handle=0x%lx usage=0x%x templ=%ux%u format=%u bind=0x%x flags=0x%x\n",
                whandle->type,
                (unsigned long)(uintptr_t)whandle->handle,
                usage,
                templ ? templ->width0 : 0,
                templ ? templ->height0 : 0,
                templ ? templ->format : PIPE_FORMAT_NONE,
                templ ? templ->bind : 0,
                templ ? templ->flags : 0);

   memset(&open, 0, sizeof(open));
   open.hGlobalHandle = (D3DKMT_HANDLE)(uintptr_t)whandle->handle;
   yttrium_trace_resource_import_state(
      YTTRIUM_TRACE_RESOURCE_IMPORT_BEGIN, 0, &open, templ, NULL,
      0xffffffffu);

   status = screen->device->queryAllocation(screen->device, &open);
   if (!NT_SUCCESS(status) || open.NumAllocations != 1) {
      yttrium_trace_resource_import_state(
         YTTRIUM_TRACE_RESOURCE_IMPORT_QUERY_FAILED, status, &open, templ,
         NULL, 0xffffffffu);
      YTTRIUM_WARN("yttrium: guest import query failed status=0x%lx handle=0x%lx allocations=%u private=%u\n",
                   status, (unsigned long)open.hGlobalHandle,
                   open.NumAllocations, open.PrivateDriverDataSize);
      return NULL;
   }

   open_info = CALLOC(open.NumAllocations, sizeof(*open_info));
   open.pPrivateDriverData = MALLOC(open.PrivateDriverDataSize);
   open.pTotalBuffer = MALLOC(open.TotalBufferSize);
   if (!open_info || !open.pPrivateDriverData || !open.pTotalBuffer)
      goto out;

   open.pOpenAllocation = open_info;

   status = screen->device->openAllocation(screen->device, &open);
   if (!NT_SUCCESS(status)) {
      yttrium_trace_resource_import_state(
         YTTRIUM_TRACE_RESOURCE_IMPORT_OPEN_FAILED, status, &open, templ,
         NULL, 0xffffffffu);
      YTTRIUM_WARN("yttrium: guest import open failed status=0x%lx handle=0x%lx\n",
                   status, (unsigned long)open.hGlobalHandle);
      goto out;
   }

   const VIOGPU_CREATE_ALLOCATION_EXCHANGE *alloc =
      (const VIOGPU_CREATE_ALLOCATION_EXCHANGE *)open_info[0].pPrivateDriverData;
   if (!alloc ||
       open_info[0].PrivateDriverDataSize <
          sizeof(VIOGPU_CREATE_ALLOCATION_EXCHANGE)) {
      yttrium_trace_resource_import_state(
         YTTRIUM_TRACE_RESOURCE_IMPORT_PRIVATE_TOO_SMALL, 0, &open, templ,
         NULL, 0xffffffffu);
      YTTRIUM_WARN("yttrium: guest import private data too small handle=0x%lx hAllocation=0x%lx private=%u resource_private=%u\n",
                   (unsigned long)open.hGlobalHandle,
                   (unsigned long)open_info[0].hAllocation,
                   open_info[0].PrivateDriverDataSize,
                   open.PrivateDriverDataSize);
      goto out;
   }

   res = CALLOC_STRUCT(yttrium_resource);
   if (!res)
      goto out;

   res->cache_id = yttrium_next_cache_id();
   pipe_reference_init(&res->base.reference, 1);
   res->base.screen = pscreen;
   res->venus.owner = &res->base;
   res->base.target = (enum pipe_texture_target)alloc->ResourceOptions.target;
   res->base.format =
      yttrium_resource_to_pipe_format(alloc->ResourceOptions.format);
   res->base.bind = alloc->ResourceOptions.bind;
   res->base.width0 = alloc->ResourceOptions.width;
   res->base.height0 = alloc->ResourceOptions.height;
   res->base.depth0 = alloc->ResourceOptions.depth;
   res->base.array_size = alloc->ResourceOptions.array_size;
   res->base.last_level = alloc->ResourceOptions.last_level;
   res->base.nr_samples = alloc->ResourceOptions.nr_samples;
   res->base.nr_storage_samples = alloc->ResourceOptions.nr_samples;
   res->base.usage = templ ? templ->usage : PIPE_USAGE_DEFAULT;
   res->base.flags = alloc->ResourceOptions.flags;
   yttrium_resource_init_threaded(res);

   res->hAllocation = open_info[0].hAllocation;
   res->hResource = open.hResource ?
      open.hResource : (HANDLE)(uintptr_t)open.hGlobalHandle;
   res->hAllocationResource = open.hAllocationResource ?
      open.hAllocationResource : res->hResource;
   res->hResourceIsD3D9Runtime = open.hResourceIsD3D9Runtime;
   res->venus_mem_id = alloc->BlobId;
   res->size = alloc->Size;
   res->display_target = res->base.bind & PIPE_BIND_DISPLAY_TARGET;
   res->classic_display = alloc->BlobMem == 0;
   res->owns_allocation = false;
   yttrium_resource_size(&res->base, &res->stride, &res->layer_stride);
   if (alloc->Stride) {
      res->stride = alloc->Stride;
      res->layer_stride =
         (unsigned)MIN2((uint64_t)UINT_MAX,
                        util_format_get_2d_size(res->base.format,
                                                res->stride,
                                                MAX2(res->base.height0, 1)));
   }

   VIOGPU_ESCAPE resinfo;
   memset(&resinfo, 0, sizeof(resinfo));
   resinfo.Type = VIOGPU_RES_INFO;
   resinfo.DataLength = sizeof(VIOGPU_RES_INFO_REQ);
   resinfo.ResourceInfo.ResHandle = res->hAllocation;

   status = screen->device->escape(screen->device, &resinfo, sizeof(resinfo));
   if (!NT_SUCCESS(status)) {
      yttrium_trace_resource_import_state(
         YTTRIUM_TRACE_RESOURCE_IMPORT_RESINFO_FAILED, status, &open, templ,
         res, 0xffffffffu);
      YTTRIUM_WARN("yttrium: guest import resinfo failed status=0x%lx hAllocation=0x%lx\n",
                   status, (unsigned long)res->hAllocation);
      yttrium_resource_free(res);
      res = NULL;
      goto out;
   }

   res->venus_res_id = resinfo.ResourceInfo.Id;

   if (res->display_target && res->classic_display) {
      yttrium_trace_resource_import_state(
         YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_FAILED, 0, &open, templ, res,
         0xffffffffu);
      YTTRIUM_WARN("yttrium: refusing imported classic display allocation hAllocation=0x%lx res_id=%u %ux%u format=%u bind=0x%x blob_mem=0x%lx size=0x%llx owner=resource_import reason=classic_display_not_supported_for_yttrium_display_path\n",
                   (unsigned long)res->hAllocation,
                   res->venus_res_id,
                   res->base.width0,
                   res->base.height0,
                   res->base.format,
                   res->base.bind,
                   (unsigned long)alloc->BlobMem,
                   (unsigned long long)res->size);
      yttrium_resource_free(res);
      res = NULL;
      goto out;
   }

   if (res->display_target && !res->classic_display && res->venus_res_id) {
      /* Opened display allocations may name resources owned by another Venus
       * context.  The attach-wait escape makes the cross-context import path
       * safe enough for normal use, but keep the option as an emergency
       * opt-out while the display ownership model is still settling.
       */
      const bool import_opened_display =
         yttrium_gdi_debug_get_bool_option(
            "D3D10UMD_YTTRIUM_IMPORT_OPENED_DISPLAY", true);
      if (!res->owns_allocation && !import_opened_display) {
         yttrium_trace_resource_import_state(
            YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_SKIPPED, 0, &open, templ,
            res, import_opened_display);
         YTTRIUM_LOG("yttrium: opened display allocation Venus import skipped res_id=%u hAllocation=0x%lx reason=debug_option_disabled\n",
                      res->venus_res_id,
                      (unsigned long)res->hAllocation);
         goto skip_opened_display_import;
      }
      if (!res->owns_allocation) {
         YTTRIUM_LOG("yttrium: opened display allocation Venus import enabled res_id=%u hAllocation=0x%lx reason=cross_context_display\n",
                     res->venus_res_id,
                     (unsigned long)res->hAllocation);

         /* The attach-wait escape must run after the Venus render context
          * exists, because the following Venus import query is context-scoped.
          * With async batches enabled, issuing vkGetMemoryResourcePropertiesMESA
          * against a resource that is not attached to the importing Venus
          * context can put the host decoder into fatal state.
          */
         if (!yttrium_venus_get_kmt_context(screen->venus)) {
            yttrium_trace_resource_import_state(
               YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_FAILED, 0, &open,
               templ, res, import_opened_display);
            YTTRIUM_WARN("yttrium: opened display allocation Venus import skipped; missing Venus KMT context res_id=%u hAllocation=0x%lx\n",
                         res->venus_res_id,
                         (unsigned long)res->hAllocation);
            goto skip_opened_display_import;
         }
         if (!yttrium_venus_flush_labeled(screen->venus,
                                          "opened display import attach flush")) {
            yttrium_trace_resource_import_state(
               YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_FAILED, 0, &open,
               templ, res, import_opened_display);
            YTTRIUM_WARN("yttrium: opened display allocation Venus import skipped; Venus flush failed before attach res_id=%u hAllocation=0x%lx\n",
                         res->venus_res_id,
                         (unsigned long)res->hAllocation);
            goto skip_opened_display_import;
         }

         VIOGPU_ESCAPE attach_wait;
         memset(&attach_wait, 0, sizeof(attach_wait));
         attach_wait.Type = VIOGPU_RES_ATTACH_WAIT;
         attach_wait.DataLength = sizeof(VIOGPU_RES_ATTACH_WAIT_REQ);
         attach_wait.ResourceAttachWait.ResHandle = res->hAllocation;
         status = screen->device->escape(screen->device, &attach_wait,
                                         sizeof(attach_wait));
         if (!NT_SUCCESS(status)) {
            yttrium_trace_resource_import_state(
               YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_FAILED, status, &open,
               templ, res, import_opened_display);
            YTTRIUM_WARN("yttrium: opened display allocation attach-wait failed status=0x%lx res_id=%u hAllocation=0x%lx\n",
                         status, res->venus_res_id,
                         (unsigned long)res->hAllocation);
            goto skip_opened_display_import;
         }
         if (attach_wait.ResourceAttachWait.Id &&
             attach_wait.ResourceAttachWait.Id != res->venus_res_id) {
            YTTRIUM_WARN("yttrium: opened display allocation attach-wait id mismatch res_id=%u attach_id=%u hAllocation=0x%lx\n",
                         res->venus_res_id, attach_wait.ResourceAttachWait.Id,
                         (unsigned long)res->hAllocation);
         }
         YTTRIUM_LOG("yttrium: opened display allocation attach-wait ok res_id=%u hAllocation=0x%lx\n",
                     res->venus_res_id, (unsigned long)res->hAllocation);
      }
      yttrium_trace_resource_import_state(
         YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_BEGIN, 0, &open, templ, res,
         import_opened_display);

      uint64_t memory_id = 0;
      uint64_t allocation_size = 0;
      if (yttrium_venus_import_display_image(screen->venus, &res->venus,
                                             res->venus_res_id,
                                             res->base.width0,
                                             res->base.height0,
                                             res->base.format,
                                             res->size,
                                             &memory_id,
                                             &allocation_size)) {
         const unsigned cpp = util_format_get_blocksize(res->base.format);
         const uint64_t min_row = (uint64_t)res->base.width0 * cpp;
         if (cpp && res->venus.image_row_pitch >= min_row &&
             res->venus.image_row_pitch <= UINT32_MAX) {
            res->stride = (unsigned)res->venus.image_row_pitch;
            res->layer_stride =
               (unsigned)MIN2((uint64_t)UINT32_MAX,
                              res->venus.image_array_pitch ?
                                 res->venus.image_array_pitch :
                                 res->venus.image_row_pitch *
                                    MAX2(res->base.height0, 1));
         }
         if (allocation_size)
            res->size = MAX2(res->size, allocation_size);
         YTTRIUM_LOG("yttrium: imported opened display allocation into Venus res_id=%u image_id=%llu memory_id=%llu size=0x%llx stride=%u\n",
                      res->venus_res_id,
                      (unsigned long long)res->venus.image_obj.id,
                      (unsigned long long)memory_id,
                      (unsigned long long)res->size,
                      res->stride);
         yttrium_trace_resource_import_state(
            YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_OK, 0, &open, templ, res,
            import_opened_display);
      } else {
         yttrium_trace_resource_import_state(
            YTTRIUM_TRACE_RESOURCE_IMPORT_VENUS_FAILED, 0, &open, templ, res,
            import_opened_display);
         YTTRIUM_WARN("yttrium: opened display allocation Venus import failed res_id=%u hAllocation=0x%lx\n",
                      res->venus_res_id,
                      (unsigned long)res->hAllocation);
      }
   }

   if (!res->display_target && !res->classic_display && res->venus_res_id &&
       res->venus_mem_id && (res->base.bind & PIPE_BIND_SHARED) &&
       (res->base.bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW |
                          PIPE_BIND_SHADER_IMAGE | PIPE_BIND_DEPTH_STENCIL))) {
      if (!yttrium_venus_get_kmt_context(screen->venus)) {
         YTTRIUM_WARN("yttrium: opened shared bind allocation Venus import failed; missing Venus KMT context res_id=%u hAllocation=0x%lx bind=0x%x owner=resource_import\n",
                      res->venus_res_id,
                      (unsigned long)res->hAllocation,
                      res->base.bind);
         yttrium_resource_free(res);
         res = NULL;
         goto out;
      }
      if (!yttrium_venus_flush_labeled(screen->venus,
                                       "opened shared bind import attach flush")) {
         YTTRIUM_WARN("yttrium: opened shared bind allocation Venus import failed; Venus flush failed before attach res_id=%u hAllocation=0x%lx bind=0x%x owner=resource_import\n",
                      res->venus_res_id,
                      (unsigned long)res->hAllocation,
                      res->base.bind);
         yttrium_resource_free(res);
         res = NULL;
         goto out;
      }

      VIOGPU_ESCAPE attach_wait;
      memset(&attach_wait, 0, sizeof(attach_wait));
      attach_wait.Type = VIOGPU_RES_ATTACH_WAIT;
      attach_wait.DataLength = sizeof(VIOGPU_RES_ATTACH_WAIT_REQ);
      attach_wait.ResourceAttachWait.ResHandle = res->hAllocation;
      status = screen->device->escape(screen->device, &attach_wait,
                                      sizeof(attach_wait));
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: opened shared bind allocation attach-wait failed status=0x%lx res_id=%u hAllocation=0x%lx bind=0x%x owner=resource_import\n",
                      status, res->venus_res_id,
                      (unsigned long)res->hAllocation,
                      res->base.bind);
         yttrium_resource_free(res);
         res = NULL;
         goto out;
      }
      if (attach_wait.ResourceAttachWait.Id &&
          attach_wait.ResourceAttachWait.Id != res->venus_res_id) {
         YTTRIUM_WARN("yttrium: opened shared bind allocation attach-wait id mismatch res_id=%u attach_id=%u hAllocation=0x%lx bind=0x%x owner=resource_import\n",
                      res->venus_res_id, attach_wait.ResourceAttachWait.Id,
                      (unsigned long)res->hAllocation,
                      res->base.bind);
      }

      uint64_t allocation_size = 0;
      if (yttrium_venus_import_texture_image_for_bind(
             screen->venus, &res->venus, res->venus_res_id,
             res->base.target, res->base.width0, res->base.height0,
             res->base.depth0, res->base.last_level + 1,
             res->base.array_size, res->base.format, res->base.nr_samples,
             res->base.bind, res->size, NULL, &allocation_size)) {
         if (allocation_size)
            res->size = MAX2(res->size, allocation_size);
      } else {
         YTTRIUM_WARN("yttrium: opened shared bind allocation Venus import failed res_id=%u hAllocation=0x%lx bind=0x%x target=%u %ux%ux%u levels=%u array=%u format=%u owner=resource_import reason=no_venus_import\n",
                      res->venus_res_id,
                      (unsigned long)res->hAllocation,
                      res->base.bind,
                      res->base.target,
                      res->base.width0,
                      res->base.height0,
                      res->base.depth0,
                      res->base.last_level + 1,
                      res->base.array_size,
                      res->base.format);
         yttrium_resource_free(res);
         res = NULL;
         goto out;
      }
   }

skip_opened_display_import:
   YTTRIUM_LOG("yttrium: opened resource handle=0x%lx hResource=%p hAllocation=0x%lx res_id=%u %ux%u format=%u bind=0x%x display=%u shared=%u mem_id=0x%llx blob_mem=0x%lx size=0x%llx stride=%u classic=%u venus_initialized=%u buffer_backed=%u image_id=%llu buffer_id=%llu\n",
                (unsigned long)open.hGlobalHandle,
                res->hResource,
                (unsigned long)res->hAllocation, res->venus_res_id,
                res->base.width0, res->base.height0, res->base.format,
                res->base.bind, res->display_target,
                (res->base.bind & PIPE_BIND_SHARED) != 0,
                (unsigned long long)res->venus_mem_id,
                (unsigned long)alloc->BlobMem,
                (unsigned long long)res->size, res->stride,
                res->classic_display,
                res->venus.initialized,
                res->venus.buffer_backed,
                (unsigned long long)res->venus.image_obj.id,
                (unsigned long long)res->venus.buffer_obj.id);
   yttrium_trace_resource_import_state(
      YTTRIUM_TRACE_RESOURCE_IMPORT_OPENED, 0, &open, templ, res,
      0xffffffffu);

out:
   FREE(open_info);
   FREE(open.pPrivateDriverData);
   FREE(open.pTotalBuffer);
   return res ? &res->base : NULL;
}


static volatile LONG yttrium_resource_copy_sequence;

struct yttrium_transfer {
   union {
      struct pipe_transfer base;
      struct threaded_transfer threaded;
   };
   void *staging;
   bool upload_to_venus;
   bool writes_cpu_data;

   /*
    * Read maps of a Venus image copy the region into a host-visible buffer and
    * hand that mapping back directly, so these have to outlive the map call
    * and be torn down on unmap.
    */
   struct yttrium_venus_resource readback;
   struct yttrium_readback_mapping readback_mapping;
   bool reads_venus_image;
};

void
yttrium_replace_buffer_storage(struct pipe_context *ctx,
                               struct pipe_resource *dst_resource,
                               struct pipe_resource *src_resource,
                               unsigned minimum_num_rebinds,
                               uint32_t rebind_mask,
                               uint32_t delete_buffer_id)
{
   struct yttrium_resource *dst = yttrium_resource(dst_resource);
   struct yttrium_resource *src = yttrium_resource(src_resource);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   (void)minimum_num_rebinds;
   (void)rebind_mask;

   if (!dst || !src || dst_resource->target != PIPE_BUFFER ||
       src_resource->target != PIPE_BUFFER ||
       dst_resource->width0 != src_resource->width0 ||
       !dst->replacement_storage || !src->replacement_storage) {
      YTTRIUM_WARN("yttrium: ERROR: ordered buffer replacement invariant failed owner=threaded-context-storage dst=%p src=%p dst_storage=%p src_storage=%p dst_target=%u src_target=%u dst_size=%u src_size=%u\n",
                   (void *)dst, (void *)src,
                   dst ? (void *)dst->replacement_storage : NULL,
                   src ? (void *)src->replacement_storage : NULL,
                   dst_resource ? dst_resource->target : 0,
                   src_resource ? src_resource->target : 0,
                   dst_resource ? dst_resource->width0 : 0,
                   src_resource ? src_resource->width0 : 0);
      return;
   }

   assert(!dst->venus.initialized);
   assert(!src->venus.initialized);
   assert(dst->data == dst->replacement_storage->map);
   assert(src->data == src->replacement_storage->map);

   yttrium_buffer_storage_reference(&dst->replacement_storage,
                                    src->replacement_storage);
   yttrium_resource_apply_buffer_storage(dst, dst->replacement_storage);
   src->replacement_owner = dst;

   dst->data_dirty = true;
   dst->contents_serial++;
   util_idalloc_mt_free(&screen->buffer_ids, delete_buffer_id);
}

void
yttrium_resource_init_screen(struct yttrium_screen *screen)
{
   util_idalloc_mt_init_tc(&screen->buffer_ids);
   screen->buffer_ids_initialized = true;
   slab_create_parent(&screen->transfer_pool,
                      sizeof(struct yttrium_transfer), 16);
   screen->transfer_pool_initialized = true;
   simple_mtx_init(&screen->ordered_upload_pool_lock, mtx_plain);
   screen->ordered_upload_pool_initialized = true;
   simple_mtx_init(&screen->buffer_replacement_pool_lock, mtx_plain);
   screen->buffer_replacement_pool_initialized = true;
   screen->buffer_replacement_pool_enabled =
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_WORKER", true) &&
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_BUFFER_REPLACEMENT", true) &&
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_BUFFER_REPLACEMENT_POOL", true);
}

void
yttrium_resource_cleanup_screen(struct yttrium_screen *screen)
{
   if (!screen)
      return;

   if (screen->buffer_ids_initialized) {
      util_idalloc_mt_fini(&screen->buffer_ids);
      screen->buffer_ids_initialized = false;
   }

   if (screen->ordered_upload_pool_initialized) {
      for (unsigned i = 0; i < screen->ordered_upload_pool_count; i++) {
         struct yttrium_ordered_upload_pool_entry *entry =
            &screen->ordered_upload_pool[i];
         if (!entry->direct_backing) {
            FREE(entry->data);
            continue;
         }

         struct yttrium_venus_allocation_snapshot snapshot = {
            .hAllocation = entry->hAllocation,
            .hResource = entry->hResource,
            .hAllocationResource = entry->hAllocationResource,
            .hResourceIsD3D9Runtime = entry->hResourceIsD3D9Runtime,
            .size = entry->capacity,
            .map = entry->data,
            .map_info = entry->map_info,
            .map_is_blob = entry->map_is_blob,
            .owns_allocation = entry->owns_allocation,
            .allocation_destroyed_by_runtime =
               entry->allocation_destroyed_by_runtime,
         };
         const bool allocation_adopted =
            yttrium_venus_resource_fini(screen->venus, NULL, &entry->venus,
                                        &snapshot);
         if (!allocation_adopted && entry->hAllocation) {
            struct yttrium_resource temp;
            memset(&temp, 0, sizeof(temp));
            temp.hAllocation = entry->hAllocation;
            temp.map = entry->data;
            temp.map_info = entry->map_info;
            temp.map_is_blob = entry->map_is_blob;
            yttrium_unmap_display_allocation(&screen->base, &temp);
            NTSTATUS status = screen->device->destroyAllocation(
               screen->device, entry->hResource, entry->hAllocation);
            if (!NT_SUCCESS(status)) {
               YTTRIUM_WARN("yttrium: ordered direct upload pool allocation destroy failed owner=screen-cleanup status=0x%lx hAllocation=0x%lx hResource=%p memory_id=%llu buffer_id=%llu\n",
                            (unsigned long)status,
                            (unsigned long)entry->hAllocation,
                            entry->hResource,
                            (unsigned long long)entry->venus.memory_obj.id,
                            (unsigned long long)entry->venus.buffer_obj.id);
            }
         }
         memset(entry, 0, sizeof(*entry));
      }
      screen->ordered_upload_pool_count = 0;
      screen->ordered_upload_pool_bytes = 0;
      simple_mtx_destroy(&screen->ordered_upload_pool_lock);
      screen->ordered_upload_pool_initialized = false;
   }

   if (screen->buffer_replacement_pool_initialized) {
      struct yttrium_buffer_storage *pooled[
         YTTRIUM_BUFFER_REPLACEMENT_POOL_MAX_ENTRIES];
      unsigned count;

      simple_mtx_lock(&screen->buffer_replacement_pool_lock);
      count = screen->buffer_replacement_pool_count;
      memcpy(pooled, screen->buffer_replacement_pool,
             count * sizeof(pooled[0]));
      memset(screen->buffer_replacement_pool, 0,
             sizeof(screen->buffer_replacement_pool));
      screen->buffer_replacement_pool_count = 0;
      screen->buffer_replacement_pool_bytes = 0;
      screen->buffer_replacement_pool_initialized = false;
      screen->buffer_replacement_pool_enabled = false;
      simple_mtx_unlock(&screen->buffer_replacement_pool_lock);
      simple_mtx_destroy(&screen->buffer_replacement_pool_lock);

      for (unsigned i = 0; i < count; i++)
         yttrium_buffer_storage_destroy(pooled[i]);
   }

   if (screen->transfer_pool_initialized) {
      slab_destroy_parent(&screen->transfer_pool);
      screen->transfer_pool_initialized = false;
   }
}

void
yttrium_destroy_context_upload_staging(struct pipe_screen *pscreen,
                                       struct yttrium_context *yctx)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);

   if (!yctx)
      return;

   yttrium_destroy_readback_mapping(pscreen, &yctx->upload_staging_mapping);
   yttrium_venus_resource_fini(screen->venus, NULL, &yctx->upload_staging,
                               NULL);
   yctx->upload_staging_mem_id = 0;
   yctx->upload_staging_size = 0;
   yctx->upload_staging_offset = 0;
}

void
yttrium_destroy_context_image_copy_staging(struct pipe_screen *pscreen,
                                           struct yttrium_context *yctx)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);

   if (!yctx)
      return;

   yttrium_venus_resource_fini(screen->venus, NULL,
                               &yctx->image_copy_staging, NULL);
   yctx->image_copy_staging_mem_id = 0;
   yctx->image_copy_staging_size = 0;
}

static bool
yttrium_ensure_context_image_copy_staging(struct pipe_context *ctx,
                                          uint64_t required_size)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   const VkBufferUsageFlags staging_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;

   if (!required_size)
      return false;

   if (yctx->image_copy_staging.initialized &&
       yctx->image_copy_staging.buffer_backed &&
       yctx->image_copy_staging.buffer &&
       yctx->image_copy_staging.memory &&
       yctx->image_copy_staging.memory_obj.id &&
       yctx->image_copy_staging_mem_id ==
          yctx->image_copy_staging.memory_obj.id &&
       (yctx->image_copy_staging.buffer_usage & staging_usage) ==
          staging_usage &&
       yctx->image_copy_staging_size >= required_size)
      return true;

   /* resource_fini flushes the current batch and retires the old allocation
    * against its latest user, so growing this scratch buffer cannot free an
    * in-flight transfer.
    */
   yttrium_destroy_context_image_copy_staging(ctx->screen, yctx);

   /* This scratch is consumed only by Vulkan transfer commands.  Keep it
    * private and device-local: exporting host-visible memory puts every
    * image-buffer-image round trip through a CPU-mappable heap even though
    * neither the CPU nor the KMD maps the allocation.
    */
   if (!yttrium_venus_create_bind_buffer(
          screen->venus, &yctx->image_copy_staging, required_size,
          staging_usage, NULL)) {
      YTTRIUM_WARN("yttrium: resource_copy_region image compatible copy unavailable owner=yttrium_resource_copy_region reason=create-private-device-local-staging-failed action=return-to-resource-copy-dispatch size=0x%llx\n",
                   (unsigned long long)required_size);
      return false;
   }

   yctx->image_copy_staging_mem_id =
      yctx->image_copy_staging.memory_obj.id;
   yctx->image_copy_staging_size =
      yctx->image_copy_staging.image_size;
   YTTRIUM_LOG("yttrium: private device-local image copy staging buffer res_id=%u mem_id=0x%llx buffer_id=0x%llx size=0x%llx usage=0x%x\n",
               (uint32_t)yctx->image_copy_staging_mem_id,
               (unsigned long long)yctx->image_copy_staging_mem_id,
               (unsigned long long)yctx->image_copy_staging.buffer_obj.id,
               (unsigned long long)yctx->image_copy_staging_size,
               yctx->image_copy_staging.buffer_usage);
   return true;
}

static bool
yttrium_ensure_context_upload_staging(struct pipe_context *ctx,
                                      uint64_t required_size,
                                      unsigned stride)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (!required_size || !stride)
      return false;

   if (yctx->upload_staging_mapping.hAllocation &&
       yctx->upload_staging.initialized &&
       yctx->upload_staging.buffer_backed &&
       yctx->upload_staging_size >= required_size)
      return true;

   yttrium_destroy_context_upload_staging(ctx->screen, yctx);

   if (!yttrium_venus_create_display_buffer(screen->venus,
                                            &yctx->upload_staging,
                                            required_size,
                                            &yctx->upload_staging_mem_id)) {
      YTTRIUM_LOG("yttrium: upload staging failed to create Venus buffer size=0x%llx stride=%u\n",
                  (unsigned long long)required_size,
                  stride);
      return false;
   }

   if (!yttrium_create_venus_memory_mapping(ctx->screen,
                                            yctx->upload_staging_mem_id,
                                            yctx->upload_staging.allocation_size,
                                            stride,
                                            &yctx->upload_staging_mapping)) {
      YTTRIUM_LOG("yttrium: upload staging failed to create KMD wrapper mem_id=0x%llx size=0x%llx stride=%u\n",
                  (unsigned long long)yctx->upload_staging_mem_id,
                  (unsigned long long)yctx->upload_staging.allocation_size,
                  stride);
      yttrium_venus_resource_fini(screen->venus, NULL,
                                  &yctx->upload_staging, NULL);
      yctx->upload_staging_mem_id = 0;
      return false;
   }

   yctx->upload_staging_size = yctx->upload_staging.allocation_size;
   yctx->upload_staging_offset = 0;
   YTTRIUM_LOG("yttrium: upload staging buffer hAllocation=0x%lx res_id=%u mem_id=0x%llx size=0x%llx stride=%u\n",
               (unsigned long)yctx->upload_staging_mapping.hAllocation,
               yctx->upload_staging_mapping.venus_res_id,
               (unsigned long long)yctx->upload_staging_mem_id,
               (unsigned long long)yctx->upload_staging_size,
               stride);
   return true;
}

void *
yttrium_transfer_map(struct pipe_context *ctx,
                     struct pipe_resource *resource,
                     unsigned level,
                     unsigned usage,
                     const struct pipe_box *box,
                     struct pipe_transfer **ptransfer)
{
   struct yttrium_resource *res = yttrium_resource(resource);
   struct yttrium_transfer *ytransfer;
   struct pipe_transfer *transfer;
   uint8_t *ptr;

   if (res->display_target) {
      if (level != 0) {
         YTTRIUM_WARN("yttrium: unsupported display target CPU map level=%u\n",
                      level);
         return NULL;
      }

      const bool read_only =
         (usage & PIPE_MAP_READ) && !(usage & PIPE_MAP_WRITE);
      const bool write_only =
         (usage & PIPE_MAP_WRITE) && !(usage & PIPE_MAP_READ);
      if (!read_only && !write_only) {
         YTTRIUM_WARN("yttrium: unsupported display target CPU map usage=0x%x\n",
                     usage);
         return NULL;
      }

      if (write_only && !res->classic_display &&
          res->venus.initialized && !res->venus.buffer_backed &&
          res->venus.image) {
         const unsigned cpp = util_format_get_blocksize(resource->format);
         const uint64_t row_bytes =
            util_format_get_stride(resource->format, box->width);
         const uint64_t layer_bytes =
            util_format_get_2d_size(resource->format, (unsigned)row_bytes,
                                    box->height);
         const uint64_t staging_size = layer_bytes * box->depth;
         if (!cpp || !row_bytes || !layer_bytes || !staging_size ||
             row_bytes > UINT32_MAX || layer_bytes > UINTPTR_MAX) {
            YTTRIUM_WARN("yttrium: unsupported Venus display CPU write map dimensions resource=%p format=%u box=%dx%dx%d cpp=%u\n",
                        (void *)res, resource->format,
                        box->width, box->height, box->depth, cpp);
            return NULL;
         }

         ytransfer = CALLOC_STRUCT(yttrium_transfer);
         if (!ytransfer)
            return NULL;
         transfer = &ytransfer->base;

         ytransfer->staging = CALLOC(1, (size_t)staging_size);
         if (!ytransfer->staging) {
            FREE(ytransfer);
            return NULL;
         }

         pipe_resource_reference(&transfer->resource, resource);
         transfer->level = level;
         transfer->usage = usage;
         transfer->box = *box;
         transfer->stride = (unsigned)row_bytes;
         transfer->layer_stride = (uintptr_t)layer_bytes;
         ytransfer->upload_to_venus = true;
         *ptransfer = transfer;

         YTTRIUM_LOG("yttrium: Venus display CPU write map staging resource=%p hAllocation=0x%lx image_id=%llu res_id=%u box=%d,%d,%d %dx%dx%d stride=%u layer_stride=%llu usage=0x%x\n",
                     (void *)res,
                     (unsigned long)res->hAllocation,
                     (unsigned long long)res->venus.image_obj.id,
                     res->venus_res_id,
                     box->x, box->y, box->z,
                     box->width, box->height, box->depth,
                     transfer->stride,
                     (unsigned long long)transfer->layer_stride,
                     usage);
         return ytransfer->staging;
      }

      if (!yttrium_ensure_display_mapped(ctx, res))
         return NULL;

      ytransfer = CALLOC_STRUCT(yttrium_transfer);
      if (!ytransfer)
         return NULL;
      transfer = &ytransfer->base;

      pipe_resource_reference(&transfer->resource, resource);
      transfer->level = level;
      transfer->usage = usage;
      transfer->box = *box;
      transfer->stride = yttrium_resource_level_stride(resource, level);
      transfer->layer_stride =
         (uintptr_t)yttrium_resource_level_slice_size(resource, level);
      *ptransfer = transfer;

      ptr = (uint8_t *)res->map +
            yttrium_resource_offset(resource, level, box->x, box->y,
                                    box->z);
      return ptr;
   }

   ytransfer = CALLOC_STRUCT(yttrium_transfer);
   if (!ytransfer)
      return NULL;
   transfer = &ytransfer->base;

   pipe_resource_reference(&transfer->resource, resource);
   transfer->level = level;
   transfer->usage = usage;
   transfer->box = *box;
   transfer->stride = resource->target == PIPE_BUFFER ? res->stride :
      yttrium_resource_level_stride(resource, level);
   transfer->layer_stride = resource->target == PIPE_BUFFER ?
      res->layer_stride :
      (uintptr_t)yttrium_resource_level_slice_size(resource, level);
   *ptransfer = transfer;

   if (!res->data && resource->target != PIPE_BUFFER) {
      /*
       * Read map of a Venus image: copy the region into a host-visible buffer
       * and return that mapping.  Refusing this made every CPU readback of a
       * texture fail - util_resource_copy_region's fallback could not map its
       * source ("mapping src-texture failed"), and a RenderDoc capture came
       * back with texture contents missing.
       *
       * Read-only for now.  A read-write map would have to copy back on
       * unmap, and nothing observed needs it; a write-only map already has its
       * own upload path below.
       */
      if ((usage & PIPE_MAP_READ) && !(usage & PIPE_MAP_WRITE) &&
          level <= resource->last_level && res->venus.initialized &&
          !res->venus.buffer_backed && res->venus.image) {
         struct yttrium_screen *rscreen = yttrium_screen(ctx->screen);
         const unsigned rb_cpp = util_format_get_blocksize(resource->format);
         const uint32_t rb_stride =
            util_format_get_stride(resource->format, box->width);
         const uint32_t rb_rows =
            util_format_get_nblocksy(resource->format, box->height);
         const uint64_t rb_layer = (uint64_t)rb_stride * rb_rows;
         const uint64_t rb_size = rb_layer * MAX2(box->depth, 1);
         uint64_t rb_mem_id = 0;

         if (!rb_cpp || !rb_stride || !rb_rows || !rb_size) {
            YTTRIUM_WARN("yttrium: Venus image CPU read map bad dimensions resource=%p format=%u box=%dx%dx%d cpp=%u\n",
                         (void *)res, resource->format, box->width,
                         box->height, box->depth, rb_cpp);
            goto read_map_failed;
         }

         ytransfer->readback.cpu_readback = true;
         if (!yttrium_venus_create_display_buffer(rscreen->venus,
                                                 &ytransfer->readback,
                                                 rb_size, &rb_mem_id)) {
            YTTRIUM_WARN("yttrium: Venus image CPU read map staging buffer failed resource=%p size=0x%llx\n",
                         (void *)res, (unsigned long long)rb_size);
            goto read_map_failed;
         }

         if (!yttrium_create_venus_memory_mapping(ctx->screen, rb_mem_id,
                                                 rb_size, rb_stride,
                                                 &ytransfer->readback_mapping))
            goto read_map_failed;

         /*
          * One copy per layer, each landing a layer further into the buffer.
          * The destination offset is dst_y * stride + dst_x * cpp, so stepping
          * dst_y by the block rows puts layer N at N * layer_stride - passing
          * dst_y = 0 for every layer would stack them all on top of each other
          * and return the last one in every slot.
          */
         for (int layer = 0; layer < MAX2(box->depth, 1); layer++) {
            if (!yttrium_venus_copy_image_region_to_display_buffer(
                   rscreen->venus, &res->venus, &ytransfer->readback,
                   res->venus_res_id,
                   ytransfer->readback_mapping.venus_res_id,
                   level, box->z + layer,
                   box->x, box->y, 0, layer * rb_rows,
                   box->width, box->height,
                   resource->format, rb_stride))
               goto read_map_failed;
         }

         /*
          * Unlike a write map, this has to wait: the caller reads the bytes as
          * soon as we return, so the copy must have completed.
          */
         if (!yttrium_venus_flush_labeled(rscreen->venus,
                                          "venus image cpu read map")) {
            YTTRIUM_WARN("yttrium: Venus image CPU read map wait failed resource=%p res_id=%u %dx%d\n",
                         (void *)res, res->venus_res_id, box->width,
                         box->height);
            goto read_map_failed;
         }

         ytransfer->reads_venus_image = true;
         transfer->stride = rb_stride;
         transfer->layer_stride = rb_layer;
         return ytransfer->readback_mapping.map;

      read_map_failed:
         yttrium_destroy_readback_mapping(ctx->screen,
                                          &ytransfer->readback_mapping);
         yttrium_venus_resource_fini(rscreen->venus, NULL,
                                     &ytransfer->readback, NULL);
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(ytransfer);
         *ptransfer = NULL;
         return NULL;
      }

      if ((usage & PIPE_MAP_READ) || !(usage & PIPE_MAP_WRITE) ||
          level > resource->last_level || !res->venus.initialized ||
          res->venus.buffer_backed || !res->venus.image) {
         YTTRIUM_WARN("yttrium: unsupported Venus image CPU map resource=%p usage=0x%x level=%u data=%p initialized=%u buffer_backed=%u image=%p\n",
                     (void *)res, usage, level, res->data,
                     res->venus.initialized, res->venus.buffer_backed,
                     res->venus.image);
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(ytransfer);
         *ptransfer = NULL;
         return NULL;
      }

      const unsigned cpp = util_format_get_blocksize(resource->format);
      const uint64_t row_bytes =
         util_format_get_stride(resource->format, box->width);
      const uint64_t layer_bytes =
         util_format_get_2d_size(resource->format, (unsigned)row_bytes,
                                 box->height);
      const uint64_t staging_size = layer_bytes * box->depth;
      if (!cpp || !row_bytes || !layer_bytes || !staging_size ||
          row_bytes > UINT32_MAX || layer_bytes > UINTPTR_MAX) {
         YTTRIUM_WARN("yttrium: unsupported Venus image CPU map dimensions resource=%p format=%u box=%dx%dx%d cpp=%u\n",
                     (void *)res, resource->format,
                     box->width, box->height, box->depth, cpp);
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(ytransfer);
         *ptransfer = NULL;
         return NULL;
      }

      ytransfer->staging = CALLOC(1, (size_t)staging_size);
      if (!ytransfer->staging) {
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(ytransfer);
         *ptransfer = NULL;
         return NULL;
      }

      ytransfer->upload_to_venus = true;
      transfer->stride = (unsigned)row_bytes;
      transfer->layer_stride = (uintptr_t)layer_bytes;
      YTTRIUM_LOG("yttrium: Venus image CPU write map staging resource=%p image_id=%llu res_id=%u box=%d,%d,%d %dx%dx%d stride=%u layer_stride=%llu usage=0x%x\n",
                  (void *)res,
                  (unsigned long long)res->venus.image_obj.id,
                  res->venus_res_id,
                  box->x, box->y, box->z,
                  box->width, box->height, box->depth,
                  transfer->stride,
                  (unsigned long long)transfer->layer_stride,
                  usage);
      return ytransfer->staging;
   }

   if (resource->target == PIPE_BUFFER && res->gpu_buffer_written &&
       !yttrium_resource_uses_mapped_venus_buffer(res)) {
      /* Map is an explicit CPU access.  A detached shadow cannot answer it
       * after a GPU counter write, nor preserve neighbours of a partial write. */
      const bool full_discard = (usage & PIPE_MAP_WRITE) &&
         !(usage & PIPE_MAP_READ) && box->x == 0 &&
         (unsigned)box->width == resource->width0 &&
         (usage & PIPE_MAP_DISCARD_WHOLE_RESOURCE);
      if (!full_discard &&
          !yttrium_copy_venus_buffer_to_cpu(ctx, res, res, 0, 0,
                                            resource->width0)) {
         YTTRIUM_WARN("yttrium: buffer CPU map failed owner=yttrium_resource reason=gpu-authoritative-readback-failed action=fail-map\n");
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(ytransfer);
         *ptransfer = NULL;
         return NULL;
      }
      res->data_dirty = false;
   }

   if (resource->target == PIPE_BUFFER &&
       ((usage & PIPE_MAP_WRITE) || res->gpu_buffer_written) &&
       yttrium_resource_uses_mapped_venus_buffer(res) &&
       ctx && ctx->screen) {
      /*
       * A direct-bound buffer is read by the GPU where it lies, so a discard
       * map has to be answered either by stalling here or by giving up direct
       * bind and copying the buffer into the per-draw arena instead.  We take
       * the second, which is why a discarding app ends up paying an upload per
       * draw for the life of the resource.
       *
       * PIPE_MAP_UNSYNCHRONIZED is different: the caller is guaranteeing it
       * will not write anything the GPU is still reading, which is exactly the
       * case where direct bind needs neither the stall nor the copy.  Treating
       * it as a hazard cost us both - the resource was demoted permanently,
       * and D3D10/11 streams dynamic buffers with NO_OVERWRITE constantly.
       */
      const bool direct_bind_map_hazard =
         (usage & (PIPE_MAP_DISCARD_RANGE |
                   PIPE_MAP_DISCARD_WHOLE_RESOURCE)) != 0 &&
         !(resource->bind & PIPE_BIND_COMMAND_ARGS_BUFFER);
      if (direct_bind_map_hazard && !res->direct_bind_unsafe)
         res->direct_bind_unsafe = true;

      struct yttrium_screen *screen = yttrium_screen(ctx->screen);
      if (screen && screen->venus && !(usage & PIPE_MAP_UNSYNCHRONIZED) &&
          !yttrium_venus_wait_resource(screen->venus, &res->venus,
                                        "mapped buffer transfer")) {
         YTTRIUM_WARN("yttrium: buffer CPU map failed owner=yttrium_resource reason=native-buffer-wait-failed action=fail-map\n");
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(ytransfer);
         *ptransfer = NULL;
         return NULL;
      }
   }

   if (resource->target == PIPE_BUFFER) {
      ptr = (uint8_t *)res->data + box->x;
   } else {
      ptr = (uint8_t *)res->data +
            yttrium_resource_offset(resource, level, box->x, box->y,
                                    box->z);
   }
   ytransfer->writes_cpu_data = (usage & PIPE_MAP_WRITE) != 0;

   return ptr;
}

void
yttrium_transfer_unmap(struct pipe_context *ctx, struct pipe_transfer *transfer)
{
   struct yttrium_transfer *ytransfer = (struct yttrium_transfer *)transfer;

   /*
    * Read map of a Venus image: the caller was handed the staging mapping
    * directly, so there is nothing to copy back - just release it.  Read-only
    * by construction, see the map side.
    */
   if (ytransfer->reads_venus_image) {
      struct yttrium_screen *screen = yttrium_screen(ctx->screen);

      yttrium_destroy_readback_mapping(ctx->screen,
                                       &ytransfer->readback_mapping);
      yttrium_venus_resource_fini(screen->venus, NULL, &ytransfer->readback,
                                  NULL);
      pipe_resource_reference(&transfer->resource, NULL);
      FREE(ytransfer);
      return;
   }

   if (ytransfer->upload_to_venus && ytransfer->staging) {
      struct yttrium_resource *dst = yttrium_resource(transfer->resource);
      struct yttrium_resource src;
      bool ok;

      memset(&src, 0, sizeof(src));
      src.base.target = transfer->resource->target;
      src.base.format = dst->base.format;
      src.base.width0 = transfer->box.width;
      src.base.height0 = transfer->box.height;
      src.base.depth0 = transfer->box.depth;
      src.base.array_size = transfer->box.depth;
      src.base.last_level = 0;
      src.data = ytransfer->staging;
      src.stride = transfer->stride;
      src.layer_stride = (unsigned)transfer->layer_stride;

      ok = yttrium_copy_cpu_to_venus_image(ctx, &src, dst,
                                           0, transfer->level,
                                           0, 0, 0,
                                           transfer->box.x,
                                           transfer->box.y,
                                           transfer->box.z,
                                           transfer->box.width,
                                           transfer->box.height,
                                           transfer->box.depth);
      if (!ok) {
         YTTRIUM_WARN("yttrium: Venus image CPU write unmap upload failed resource=%p res_id=%u box=%d,%d,%d %dx%dx%d\n",
                      (void *)dst, dst->venus_res_id,
                      transfer->box.x, transfer->box.y, transfer->box.z,
                      transfer->box.width, transfer->box.height,
                      transfer->box.depth);
      }
   }

   if (ytransfer->writes_cpu_data && transfer->resource) {
      struct yttrium_resource *res = yttrium_resource(transfer->resource);
      if (res && res->data) {
         struct yttrium_resource *owner =
            res->replacement_owner ? res->replacement_owner : res;
         assert(owner->data == res->data);
         owner->data_dirty = true;
         /* A mapped write publishes a new resource-owned UBO version. */
         owner->contents_serial++;

         /*
          * Stream-output and GPU-written counter destinations can have a
          * device-local Venus buffer and a detached CPU shadow.  A HOST_WRITE
          * dependency cannot make writes
          * to that shadow visible to the device-local buffer, so publish the
          * exact mapped range while the CPU bytes are authoritative.  This is
          * also the path used by D3D10 CreateResource initial data.
          *
          * Do not defer this as a whole-buffer upload until a later draw:
          * after transform feedback has written the resource, the detached
          * shadow is stale and must never overwrite the GPU-produced bytes.
          */
         const bool detached_stream_output =
            owner->base.target == PIPE_BUFFER &&
            ((owner->base.bind & PIPE_BIND_STREAM_OUTPUT) ||
             owner->gpu_buffer_written) &&
            !yttrium_resource_uses_mapped_venus_buffer(owner);
         if (detached_stream_output) {
            struct yttrium_screen *screen =
               ctx && ctx->screen ? yttrium_screen(ctx->screen) : NULL;
            const bool nonnegative_range =
               transfer->box.x >= 0 && transfer->box.width > 0;
            const uint64_t write_offset =
               nonnegative_range ? (uint64_t)transfer->box.x : 0;
            const uint64_t write_size =
               nonnegative_range ? (uint64_t)transfer->box.width : 0;
            const bool range_valid =
               nonnegative_range && write_offset <= owner->size &&
               write_size <= owner->size - write_offset;
            const bool update_aligned =
               range_valid && !(write_offset & 3) &&
               (!(write_size & 3) || write_offset + write_size == owner->size);
            const bool publication_state_valid =
               screen && screen->venus && owner->venus.initialized &&
               owner->venus.buffer_backed && owner->venus.buffer &&
               (owner->venus.buffer_usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const bool published =
               publication_state_valid && update_aligned &&
               yttrium_venus_update_buffer(
                  screen->venus, &owner->venus, write_offset, write_size,
                  (const uint8_t *)owner->data + write_offset);

            if (published) {
               owner->data_dirty = false;
               owner->venus.contents_initialized = true;
            } else {
               YTTRIUM_WARN("yttrium: ERROR: native buffer CPU write publication failed owner=yttrium_resource component=native-buffer reason=venus-range-update-failed action=fail-next-draw resource=%p data=%p offset=0x%llx size=0x%llx resource_size=0x%llx initialized=%u buffer_backed=%u buffer_id=%llu usage=0x%x range_valid=%u aligned=%u\n",
                            (void *)owner, owner->data,
                            (unsigned long long)write_offset,
                            (unsigned long long)write_size,
                            (unsigned long long)owner->size,
                            owner->venus.initialized,
                            owner->venus.buffer_backed,
                            (unsigned long long)owner->venus.buffer_obj.id,
                            owner->venus.buffer_usage, range_valid,
                            update_aligned);
               if (ctx)
                  InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
            }
         }
      }
   }

   FREE(ytransfer->staging);
   pipe_resource_reference(&transfer->resource, NULL);
   FREE(ytransfer);
}

void
yttrium_transfer_flush_region(struct pipe_context *ctx,
                              struct pipe_transfer *transfer,
                              const struct pipe_box *box)
{
}

void
yttrium_buffer_subdata(struct pipe_context *ctx,
                       struct pipe_resource *resource,
                       unsigned usage, unsigned offset,
                       unsigned size, const void *data)
{
   struct yttrium_resource *res = yttrium_resource(resource);
   struct yttrium_screen *screen =
      ctx && ctx->screen ? yttrium_screen(ctx->screen) : NULL;
   (void)usage;

   if (!res || !data || offset > res->size || size > res->size - offset)
      return;

   const bool mapped_venus_buffer =
      yttrium_resource_uses_mapped_venus_buffer(res);
   if (mapped_venus_buffer && screen && screen->venus &&
       !yttrium_venus_wait_resource(screen->venus, &res->venus,
                                     "mapped buffer subdata")) {
      YTTRIUM_WARN("yttrium: buffer CPU update failed owner=yttrium_resource reason=native-buffer-wait-failed action=fail-draw\n");
      InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
      return;
   }

   if (res->data) {
      memcpy((uint8_t *)res->data + offset, data, size);
      struct yttrium_resource *owner =
         res->replacement_owner ? res->replacement_owner : res;
      assert(owner->data == res->data);
      owner->data_dirty = true;
      owner->contents_serial++;
   }

   if (mapped_venus_buffer)
      return;

   if (!res->private_immutable_draw_buffer && screen &&
       res->venus.initialized && res->venus.buffer_backed) {
      if (!yttrium_venus_update_buffer(screen->venus, &res->venus, offset,
                                        size, data)) {
         YTTRIUM_WARN("yttrium: buffer upload failed owner=yttrium_resource reason=partial-buffer-update-failed action=fail-draw\n");
         InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
      } else if (res->gpu_buffer_written) {
         /* Only this range of the detached shadow is current. */
         res->data_dirty = false;
      }
   }
}

void
yttrium_texture_subdata(struct pipe_context *ctx,
                        struct pipe_resource *resource,
                        unsigned level,
                        unsigned usage,
                        const struct pipe_box *box,
                        const void *data,
                        unsigned stride,
                        uintptr_t layer_stride)
{
   struct yttrium_resource *dst = yttrium_resource(resource);

   (void)usage;

   if (!dst || !box || !data || level > resource->last_level ||
       box->x < 0 || box->y < 0 || box->z < 0 ||
       box->width <= 0 || box->height <= 0 || box->depth <= 0)
      return;

   const uint32_t level_width = yttrium_resource_level_width(resource, level);
   const uint32_t level_height =
      yttrium_resource_level_height(resource, level);
   if ((uint32_t)box->x > level_width ||
       (uint32_t)box->y > level_height ||
       (uint32_t)box->width > level_width - (uint32_t)box->x ||
       (uint32_t)box->height > level_height - (uint32_t)box->y)
      return;

   if (!dst->data && dst->venus.initialized && !dst->venus.buffer_backed &&
       dst->venus.image && !dst->classic_display) {
      struct yttrium_resource src;
      memset(&src, 0, sizeof(src));
      src.base.target = resource->target;
      src.base.format = resource->format;
      src.base.width0 = box->width;
      src.base.height0 = box->height;
      src.base.depth0 = box->depth;
      src.base.array_size = box->depth;
      src.base.last_level = 0;
      src.data = (void *)data;
      src.stride = stride;
      src.layer_stride = (unsigned)layer_stride;
      if (!src.layer_stride)
         src.layer_stride =
            util_format_get_2d_size(resource->format, stride, box->height);

      bool ok = yttrium_copy_cpu_to_venus_image(ctx, &src, dst, 0, level,
                                                0, 0, 0,
                                                (uint32_t)box->x,
                                                (uint32_t)box->y,
                                                (uint32_t)box->z,
                                                (uint32_t)box->width,
                                                (uint32_t)box->height,
                                                (uint32_t)box->depth);
      if (!ok) {
         YTTRIUM_WARN("yttrium: texture_subdata Venus upload failed resource=%p res_id=%u level=%u box=%d,%d,%d %dx%dx%d\n",
                      (void *)dst, dst->venus_res_id, level,
                      box->x, box->y, box->z,
                      box->width, box->height, box->depth);
      }
      return;
   }

   if (!dst->data)
      return;

   const uint8_t *src = (const uint8_t *)data;
   const uintptr_t effective_layer_stride = layer_stride ? layer_stride :
      (uintptr_t)util_format_get_2d_size(resource->format, stride,
                                         box->height);
   for (int z = 0; z < box->depth; z++) {
      uint8_t *dst_ptr = (uint8_t *)dst->data +
         yttrium_resource_offset(resource, level, (uint32_t)box->x,
                                 (uint32_t)box->y,
                                 (uint32_t)box->z + z);
      const uint32_t row_bytes =
         util_format_get_stride(resource->format, box->width);
      const uint32_t rows =
         util_format_get_nblocksy(resource->format, box->height);
      const unsigned dst_stride =
         yttrium_resource_level_stride(resource, level);

      for (uint32_t y = 0; y < rows; y++) {
         memcpy(dst_ptr + (uint64_t)y * dst_stride,
                src + (uint64_t)z * effective_layer_stride +
                   (uint64_t)y * stride,
                row_bytes);
      }
   }
   dst->data_dirty = true;
   dst->contents_serial++;
}

void
yttrium_clear_buffer(struct pipe_context *ctx,
                     struct pipe_resource *resource,
                     unsigned offset,
                     unsigned size,
                     const void *clear_value,
                     int clear_value_size)
{
   struct yttrium_resource *res = resource ? yttrium_resource(resource) : NULL;
   struct yttrium_screen *screen =
      ctx && ctx->screen ? yttrium_screen(ctx->screen) : NULL;
   uint32_t dword_value = 0;
   int lowered_clear_size = clear_value_size;
   const void *lowered_clear_value = clear_value;

   if (!res || !clear_value || clear_value_size <= 0 ||
       offset > resource->width0 || size > resource->width0 - offset)
      return;

   if (util_lower_clearsize_to_dword(clear_value, &lowered_clear_size,
                                     &dword_value))
      lowered_clear_value = &dword_value;

   const bool mapped_venus_buffer =
      yttrium_resource_uses_mapped_venus_buffer(res);
   if (mapped_venus_buffer && screen && screen->venus &&
       !yttrium_venus_wait_resource(screen->venus, &res->venus,
                                     "mapped buffer clear")) {
      YTTRIUM_WARN("yttrium: buffer clear failed owner=yttrium_resource reason=native-buffer-wait-failed action=fail-draw\n");
      InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
      return;
   }

   if (res->data) {
      uint8_t *dst = (uint8_t *)res->data + offset;
      for (unsigned off = 0; off < size; off += clear_value_size)
         memcpy(dst + off, clear_value, MIN2((unsigned)clear_value_size,
                                             size - off));
      res->data_dirty = true;
      res->contents_serial++;
   }

   if (!screen || !res->venus.initialized || !res->venus.buffer_backed)
      return;

   if (mapped_venus_buffer)
      return;

   if (lowered_clear_size == 4 && !(offset & 3) && !(size & 3)) {
      if (yttrium_venus_clear_buffer(screen->venus, &res->venus, offset, size,
                                     *(const uint32_t *)lowered_clear_value)) {
         if (res->gpu_buffer_written)
            res->data_dirty = false;
         return;
      }
   }

   if ((offset & 3) || (size & 3) || !size)
      return;

   uint8_t stack_data[256];
   uint8_t *upload = size <= sizeof(stack_data) ? stack_data : MALLOC(size);
   if (!upload)
      return;

   for (unsigned off = 0; off < size; off += clear_value_size)
      memcpy(upload + off, clear_value, MIN2((unsigned)clear_value_size,
                                             size - off));

   if (yttrium_venus_update_buffer(screen->venus, &res->venus, offset, size,
                                     upload) && res->gpu_buffer_written)
      res->data_dirty = false;

   if (upload != stack_data)
      FREE(upload);
}

void
yttrium_clear_render_target(struct pipe_context *ctx,
                            struct pipe_surface *dst,
                            const union pipe_color_union *color,
                            unsigned dstx, unsigned dsty,
                            unsigned width, unsigned height,
                            bool render_condition_enabled)
{
   struct yttrium_resource *res =
      dst && dst->texture ? yttrium_resource(dst->texture) : NULL;

   if (!res)
      return;

   if (res->display_target || yttrium_resource_is_venus_color_attachment(res)) {
      struct yttrium_screen *screen = yttrium_screen(ctx->screen);
      const unsigned level = dst ? dst->level : 0;
      const unsigned first_layer = dst ? dst->first_layer : 0;
      const enum pipe_format view_format =
         dst ? dst->format : res->base.format;
      const unsigned layer_count =
         dst && dst->last_layer >= dst->first_layer ?
         dst->last_layer - dst->first_layer + 1 : 1;
      const unsigned level_width =
         yttrium_resource_level_width(&res->base, level);
      const unsigned level_height =
         yttrium_resource_level_height(&res->base, level);

      if (dstx >= level_width || dsty >= level_height)
         return;

      width = MIN2(width, level_width - dstx);
      height = MIN2(height, level_height - dsty);
      if (!width || !height)
         return;

      YTTRIUM_LOG("yttrium: clear color target hAllocation=0x%lx res_id=%u display=%u level=%u layers=%u+%u color_f=%f,%f,%f,%f color_ui=%u,%u,%u,%u color_i=%d,%d,%d,%d\n",
                   (unsigned long)res->hAllocation, res->venus_res_id,
                   res->display_target, level, first_layer, layer_count,
                   color->f[0], color->f[1], color->f[2], color->f[3],
                   color->ui[0], color->ui[1], color->ui[2], color->ui[3],
                   color->i[0], color->i[1], color->i[2], color->i[3]);

      union pipe_color_union clear_color = *color;
      if (util_format_is_srgb(view_format) &&
          (!util_format_is_srgb(res->base.format) ||
           res->venus.buffer_backed)) {
         clear_color.f[0] =
            util_format_linear_float_to_srgb_8unorm(color->f[0]) / 255.0f;
         clear_color.f[1] =
            util_format_linear_float_to_srgb_8unorm(color->f[1]) / 255.0f;
         clear_color.f[2] =
            util_format_linear_float_to_srgb_8unorm(color->f[2]) / 255.0f;
      }

      if (!res->classic_display) {
         const bool full_clear =
            dstx == 0 && dsty == 0 &&
            width == level_width && height == level_height;
         const bool cleared = full_clear ?
            yttrium_venus_clear_display(screen->venus,
                                        yttrium_context(ctx)->kmt_ctx,
                                        &res->venus,
                                        res->venus_res_id,
                                        res->base.width0, res->base.height0,
                                        view_format,
                                        res->size,
                                        &clear_color,
                                        level, first_layer, layer_count) :
            yttrium_venus_clear_display_rect(screen->venus,
                                             yttrium_context(ctx)->kmt_ctx,
                                             &res->venus,
                                             res->venus_res_id,
                                             view_format,
                                             &clear_color,
                                             level, first_layer, layer_count,
                                             dstx, dsty, width, height);
         if (cleared)
            return;
      }

      if (res->base.flags & YTTRIUM_GDI_RESOURCE_FLAG_UAV_COUNTER) {
         YTTRIUM_WARN("yttrium: counter reset failed owner=yttrium_resource reason=native-image-clear-failed action=fail-draw\n");
         InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
         return;
      }

      if (!res->display_target)
         return;

      YTTRIUM_WARN("yttrium: display clear skipped without native Venus clear hAllocation=0x%lx res_id=%u classic=%u display=%u level=%u layers=%u+%u box=%u,%u %ux%u\n",
                   (unsigned long)res->hAllocation,
                   res->venus_res_id,
                   res->classic_display,
                   res->display_target,
                   level, first_layer, layer_count,
                   dstx, dsty, width, height);
      return;
   }

   if (!res->data)
      return;
}

void
yttrium_clear_depth_stencil(struct pipe_context *ctx,
                            struct pipe_surface *dst,
                            unsigned clear_flags,
                            double depth,
                            unsigned stencil,
                            unsigned dstx, unsigned dsty,
                            unsigned width, unsigned height,
                            bool render_condition_enabled)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_resource *res =
      dst && dst->texture ? yttrium_resource(dst->texture) : NULL;

   if (!res)
      return;

   if (render_condition_enabled) {
      YTTRIUM_LOG("yttrium: clear_depth_stencil ignoring render condition\n");
   }

   if ((!res->venus.initialized || res->venus.buffer_backed ||
         !res->venus.image) &&
       !yttrium_blit_ensure_image(ctx, res)) {
      YTTRIUM_WARN("yttrium: clear_depth_stencil failed to create Venus image resource=%p flags=0x%x depth=%f stencil=%u\n",
                   (void *)res, clear_flags, (float)depth, stencil);
      return;
   }

   if (res->venus.initialized && !res->venus.buffer_backed &&
       res->venus.image) {
      const uint32_t level = dst->level;
      const uint32_t first_layer = dst->first_layer;
      const uint32_t layer_count =
         dst->last_layer >= dst->first_layer ?
         dst->last_layer - dst->first_layer + 1 : 1;
      if (!yttrium_venus_clear_depth_stencil(screen->venus, &res->venus,
                                              res->venus_res_id,
                                              clear_flags, depth, stencil,
                                              level, first_layer,
                                              layer_count,
                                              dstx, dsty, width, height)) {
         YTTRIUM_WARN("yttrium: clear_depth_stencil Venus clear failed resource=%p image_id=%llu flags=0x%x depth=%f stencil=%u\n",
                       (void *)res,
                      (unsigned long long)res->venus.image_obj.id,
                      clear_flags, (float)depth, stencil);
      }
      return;
   }

   YTTRIUM_WARN("yttrium: clear_depth_stencil unsupported CPU-only resource=%p flags=0x%x depth=%f stencil=%u\n",
                (void *)res, clear_flags, (float)depth, stencil);
}

void
yttrium_clear(struct pipe_context *ctx, unsigned buffers,
              uint32_t color_clear_mask, uint8_t stencil_clear_mask,
              const struct pipe_scissor_state *scissor_state,
              const union pipe_color_union *color,
              double depth, unsigned stencil)
{
}

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
                                uint32_t depth)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   bool ok = false;

   if (!ysrc || !ysrc->data || !ydst || ydst->classic_display ||
       !ydst->venus.initialized || ydst->venus.buffer_backed ||
       !ydst->venus.image)
      return false;

   const unsigned cpp = util_format_get_blocksize(ydst->base.format);
   if (!cpp)
      return false;

   const uint32_t row_bytes = util_format_get_stride(ydst->base.format, width);
   const uint32_t staging_stride = row_bytes;
   const uint32_t rows = util_format_get_nblocksy(ydst->base.format, height);
   const uint64_t staging_layer_stride64 = (uint64_t)staging_stride * rows;
   const uint64_t staging_size = staging_layer_stride64 * depth;
   if (!staging_size || staging_layer_stride64 > UINT32_MAX)
      return false;
   const uint32_t staging_layer_stride = (uint32_t)staging_layer_stride64;
   const uint64_t upload_alignment = MAX2((uint64_t)cpp, 4ull);
   const uint64_t staging_arena_size =
      yttrium_display_staging_arena_size(staging_size);

   if (yctx->upload_staging.initialized &&
       yctx->upload_staging_size < staging_arena_size) {
      if (!yttrium_venus_flush_labeled(screen->venus,
                                       "cpu to display staging arena grow flush")) {
         YTTRIUM_WARN("yttrium: resource_copy_region CPU to display image staging grow wait failed dst hAllocation=0x%lx res_id=%u staging_res_id=%u old_size=0x%llx new_size=0x%llx\n",
                      (unsigned long)ydst->hAllocation,
                      ydst->venus_res_id,
                      yctx->upload_staging_mapping.venus_res_id,
                      (unsigned long long)yctx->upload_staging_size,
                      (unsigned long long)staging_arena_size);
         return false;
      }
      yctx->upload_staging_offset = 0;
   }

   if (!yttrium_ensure_context_upload_staging(ctx, staging_arena_size,
                                              staging_stride)) {
      YTTRIUM_LOG("yttrium: resource_copy_region CPU to display image failed to get staging buffer dst hAllocation=0x%lx res_id=%u size=0x%llx stride=%u\n",
                  (unsigned long)ydst->hAllocation, ydst->venus_res_id,
                  (unsigned long long)staging_arena_size, staging_stride);
      return false;
   }

   uint64_t staging_offset = align64(yctx->upload_staging_offset,
                                     upload_alignment);
   if (staging_offset > yctx->upload_staging_size ||
       staging_size > yctx->upload_staging_size - staging_offset) {
      if (!yttrium_venus_flush_labeled(screen->venus,
                                       "cpu to display staging arena wrap flush")) {
         YTTRIUM_WARN("yttrium: resource_copy_region CPU to display image staging wrap wait failed dst hAllocation=0x%lx res_id=%u staging_res_id=%u offset=0x%llx size=0x%llx arena=0x%llx\n",
                      (unsigned long)ydst->hAllocation,
                      ydst->venus_res_id,
                      yctx->upload_staging_mapping.venus_res_id,
                      (unsigned long long)staging_offset,
                      (unsigned long long)staging_size,
                      (unsigned long long)yctx->upload_staging_size);
         return false;
      }
      yctx->upload_staging_offset = 0;
      staging_offset = 0;
   }
   if (staging_size > yctx->upload_staging_size - staging_offset)
      return false;

   const unsigned src_stride =
      src_level == 0 && ysrc->stride ?
      ysrc->stride : yttrium_resource_level_stride(&ysrc->base, src_level);
   const uint64_t src_layer_stride =
      ysrc->base.target == PIPE_TEXTURE_3D ?
      util_format_get_2d_size(ysrc->base.format, src_stride,
                              yttrium_resource_level_height(&ysrc->base,
                                                             src_level)) :
      (ysrc->layer_stride ?
         ysrc->layer_stride : yttrium_resource_array_stride(&ysrc->base));
   const unsigned src_blocksize = util_format_get_blocksize(ysrc->base.format);
   const unsigned src_block_x =
      src_x / util_format_get_blockwidth(ysrc->base.format);
   const unsigned src_block_y =
      src_y / util_format_get_blockheight(ysrc->base.format);
   const uint8_t *src_ptr =
      (const uint8_t *)ysrc->data +
      (uint64_t)src_layer * src_layer_stride +
      yttrium_resource_level_base_offset(&ysrc->base, src_level) +
      (uint64_t)src_block_y * src_stride +
      (uint64_t)src_block_x * src_blocksize;
   uint8_t *dst_ptr =
      (uint8_t *)yctx->upload_staging_mapping.map + staging_offset;

   for (uint32_t z = 0; z < depth; z++) {
      for (uint32_t y = 0; y < rows; y++) {
         memcpy(dst_ptr + (uint64_t)z * staging_layer_stride +
                   (uint64_t)y * staging_stride,
                src_ptr + (uint64_t)z * src_layer_stride +
                   (uint64_t)y * src_stride,
                row_bytes);
      }
   }
   MemoryBarrier();
   yctx->upload_staging_offset =
      align64(staging_offset + staging_size, upload_alignment);
   yttrium_trace_venus_upload(
      YTTRIUM_TRACE_VENUS_UPLOAD_CPU_IMAGE_STAGING, 0, staging_size,
      0, ydst->venus.image_obj.id, 0, ydst->venus_res_id,
      width, height, depth, staging_stride, staging_layer_stride);

   ok = yttrium_venus_copy_buffer_to_display_image(screen->venus,
                                                   &yctx->upload_staging,
                                                   &ydst->venus,
                                                   yctx->upload_staging_mapping.venus_res_id,
                                                   ydst->venus_res_id,
                                                   staging_offset,
                                                   staging_stride,
                                                   staging_layer_stride,
                                                   dst_level,
                                                   dstx, dsty, dstz,
                                                   width, height, depth,
                                                   ydst->base.format);
   if (ok) {
      yttrium_trace_resource_copy(YTTRIUM_TRACE_COPY_CPU_TO_DISPLAY_IMAGE,
                                  0, 0, ydst->hAllocation,
                                  ydst->venus_res_id,
                                  src_x, src_y, src_layer,
                                  dstx, dsty, dstz,
                                  width, height,
                                  src_stride, staging_stride);
      YTTRIUM_LOG("yttrium: resource_copy_region copied CPU to display image src=%p dst hAllocation=0x%lx res_id=%u src=%u,%u layer=%u dst_level=%u dst=%u,%u layer=%u %ux%ux%u stride=%u/%u staging_res_id=%u\n",
                  ysrc->data,
                  (unsigned long)ydst->hAllocation,
                  ydst->venus_res_id,
                  src_x, src_y, src_layer,
                  dst_level, dstx, dsty, dstz,
                  width, height, depth,
                  src_stride, staging_stride,
                  yctx->upload_staging_mapping.venus_res_id);
   } else {
      yttrium_trace_resource_copy_unsupported(0, 0, ydst->hAllocation,
                                              ydst->venus_res_id,
                                              width, height);
      YTTRIUM_WARN("yttrium: resource_copy_region CPU to display image GPU copy failed dst hAllocation=0x%lx res_id=%u src=%u,%u layer=%u dst_level=%u dst=%u,%u layer=%u %ux%ux%u\n",
                  (unsigned long)ydst->hAllocation,
                  ydst->venus_res_id,
                  src_x, src_y, src_layer,
                  dst_level, dstx, dsty, dstz,
                  width, height, depth);
   }

   return ok;
}

static bool
yttrium_copy_venus_image_to_cpu(struct pipe_context *ctx,
                                struct yttrium_resource *ysrc,
                                struct yttrium_resource *ydst,
                                uint32_t src_level,
                                uint32_t src_x,
                                uint32_t src_y,
                                uint32_t src_layer,
                                uint32_t dst_level,
                                uint32_t dstx,
                                uint32_t dsty,
                                uint32_t dstz,
                                uint32_t width,
                                uint32_t height)
{
   /* GPU readback path used when a Venus image is copied into CPU storage. */
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_venus_resource readback;
   struct yttrium_venus_resource stencil_readback;
   struct yttrium_readback_mapping mapping;
   struct yttrium_readback_mapping stencil_mapping;
   uint64_t readback_mem_id = 0;
   uint64_t stencil_readback_mem_id = 0;
   bool ok = false;

   memset(&readback, 0, sizeof(readback));
   memset(&stencil_readback, 0, sizeof(stencil_readback));
   memset(&mapping, 0, sizeof(mapping));
   memset(&stencil_mapping, 0, sizeof(stencil_mapping));

   if (!ysrc || ysrc->classic_display || !ysrc->venus.initialized ||
       ysrc->venus.buffer_backed || !ysrc->venus.image || !ydst ||
       !ydst->data)
      return false;

   const unsigned cpp = util_format_get_blocksize(ydst->base.format);
   if (!cpp || !width || !height)
      return false;

   const uint32_t readback_stride =
      util_format_get_stride(ydst->base.format, width);
   const uint32_t rows = util_format_get_nblocksy(ydst->base.format, height);
   const uint64_t readback_size = (uint64_t)readback_stride * rows;
   const bool d24s8_readback =
      ysrc->base.format == PIPE_FORMAT_Z24_UNORM_S8_UINT ||
      ysrc->base.format == PIPE_FORMAT_S8_UINT_Z24_UNORM;
   const uint32_t stencil_readback_stride = d24s8_readback ? width : 0;
   const uint64_t stencil_readback_size =
      d24s8_readback ? (uint64_t)stencil_readback_stride * rows : 0;
   if (!readback_size)
      return false;

   readback.cpu_readback = true;
   if (!yttrium_venus_create_display_buffer(screen->venus, &readback,
                                            readback_size,
                                            &readback_mem_id)) {
      YTTRIUM_WARN("yttrium: resource_copy_region image to CPU failed to create readback buffer src hAllocation=0x%lx res_id=%u size=0x%llx\n",
                  (unsigned long)ysrc->hAllocation,
                  ysrc->venus_res_id,
                  (unsigned long long)readback_size);
      return false;
   }

   if (!yttrium_create_venus_memory_mapping(ctx->screen, readback_mem_id,
                                            readback.allocation_size,
                                            readback_stride,
                                            &mapping)) {
      YTTRIUM_WARN("yttrium: resource_copy_region image to CPU failed to map readback buffer src hAllocation=0x%lx res_id=%u mem_id=0x%llx size=0x%llx\n",
                  (unsigned long)ysrc->hAllocation,
                  ysrc->venus_res_id,
                  (unsigned long long)readback_mem_id,
                  (unsigned long long)readback.allocation_size);
      goto out;
   }

   if (!yttrium_venus_copy_image_region_to_display_buffer(
          screen->venus, &ysrc->venus, &readback,
          ysrc->venus_res_id, mapping.venus_res_id,
          src_level, src_layer, src_x, src_y, 0, 0, width, height,
          ysrc->base.format, readback_stride)) {
      YTTRIUM_WARN("yttrium: resource_copy_region image to CPU GPU copy failed src hAllocation=0x%lx res_id=%u dst=%p %ux%u\n",
                  (unsigned long)ysrc->hAllocation,
                  ysrc->venus_res_id,
                  ydst->data,
                  width, height);
      goto out;
   }

   if (d24s8_readback) {
      stencil_readback.cpu_readback = true;
      if (!yttrium_venus_create_display_buffer(screen->venus,
                                               &stencil_readback,
                                               stencil_readback_size,
                                               &stencil_readback_mem_id)) {
         YTTRIUM_WARN("yttrium: resource_copy_region image to CPU failed to create stencil readback buffer src hAllocation=0x%lx res_id=%u size=0x%llx\n",
                     (unsigned long)ysrc->hAllocation,
                     ysrc->venus_res_id,
                     (unsigned long long)stencil_readback_size);
         goto out;
      }

      if (!yttrium_create_venus_memory_mapping(
             ctx->screen, stencil_readback_mem_id,
             stencil_readback.allocation_size, stencil_readback_stride,
             &stencil_mapping)) {
         YTTRIUM_WARN("yttrium: resource_copy_region image to CPU failed to map stencil readback buffer src hAllocation=0x%lx res_id=%u mem_id=0x%llx size=0x%llx\n",
                     (unsigned long)ysrc->hAllocation,
                     ysrc->venus_res_id,
                     (unsigned long long)stencil_readback_mem_id,
                     (unsigned long long)stencil_readback.allocation_size);
         goto out;
      }

      if (!yttrium_venus_copy_image_region_aspect_to_display_buffer(
             screen->venus, &ysrc->venus, &stencil_readback,
             ysrc->venus_res_id, stencil_mapping.venus_res_id,
             src_level, src_layer, src_x, src_y, 0, 0, width, height,
             PIPE_FORMAT_S8_UINT, stencil_readback_stride,
             VK_IMAGE_ASPECT_STENCIL_BIT)) {
         YTTRIUM_WARN("yttrium: resource_copy_region image to CPU stencil GPU copy failed src hAllocation=0x%lx res_id=%u dst=%p %ux%u\n",
                     (unsigned long)ysrc->hAllocation,
                     ysrc->venus_res_id,
                     ydst->data,
                     width, height);
         goto out;
      }
   }

   if (!yttrium_venus_flush_labeled(screen->venus,
                                    "image to cpu readback flush")) {
      YTTRIUM_WARN("yttrium: resource_copy_region image to CPU wait failed src hAllocation=0x%lx res_id=%u dst=%p %ux%u\n",
                  (unsigned long)ysrc->hAllocation,
                  ysrc->venus_res_id,
                  ydst->data,
                  width, height);
      goto out;
   }

   const uint8_t *src_ptr = (const uint8_t *)mapping.map;
   const uint8_t *stencil_src_ptr =
      d24s8_readback ? (const uint8_t *)stencil_mapping.map : NULL;
   uint8_t *dst_ptr =
      (uint8_t *)ydst->data +
      yttrium_resource_offset(&ydst->base, dst_level, dstx, dsty, dstz);
   const unsigned dst_stride =
      yttrium_resource_level_stride(&ydst->base, dst_level);
   if (ysrc->base.format == PIPE_FORMAT_Z24_UNORM_S8_UINT ||
       ysrc->base.format == PIPE_FORMAT_S8_UINT_Z24_UNORM) {
      for (uint32_t y = 0; y < rows; y++) {
         const uint8_t *src_row =
            src_ptr + (uint64_t)y * readback_stride;
         uint32_t *dst_row =
            (uint32_t *)(void *)(dst_ptr + (uint64_t)y * dst_stride);
         const uint8_t *stencil_row =
            stencil_src_ptr + (uint64_t)y * stencil_readback_stride;

         for (uint32_t x = 0; x < width; x++) {
            const uint32_t depth_bits =
               ((const uint32_t *)(const void *)src_row)[x];
            uint32_t z24;
            /* Depth copies return zero-extended Z24 or floats. */
            if (!(depth_bits & 0xff000000u)) {
               z24 = depth_bits;
            } else {
               float depth;
               memcpy(&depth, &depth_bits, sizeof(depth));
               if (!(depth >= 0.0f))
                  depth = 0.0f;
               if (depth >= 1.0f) {
                  z24 = 0x00ffffffu;
               } else {
                  const double scaled = (double)depth * 16777215.0 + 0.5;
                  z24 = (uint32_t)scaled;
                  if (z24 > 0x00ffffffu)
                     z24 = 0x00ffffffu;
               }
            }
            const uint32_t stencil = stencil_row[x];
            dst_row[x] =
               ysrc->base.format == PIPE_FORMAT_S8_UINT_Z24_UNORM ?
               (z24 << 8) | stencil : z24 | (stencil << 24);
         }
      }
   } else {
      for (uint32_t y = 0; y < rows; y++) {
         memcpy(dst_ptr + (uint64_t)y * dst_stride,
                src_ptr + (uint64_t)y * readback_stride,
                readback_stride);
      }
   }
   MemoryBarrier();

   yttrium_trace_resource_copy(YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_CPU,
                               ysrc->hAllocation, ysrc->venus_res_id,
                               0, 0,
                               src_x, src_y, src_layer,
                               dstx, dsty, dstz,
                               width, height,
                               readback_stride, dst_stride);
   YTTRIUM_LOG("yttrium: resource_copy_region copied display image to CPU src hAllocation=0x%lx res_id=%u src_level=%u src=%u,%u layer=%u dst=%p dst_level=%u dst=%u,%u layer=%u %ux%u stride=%u/%u readback_res_id=%u\n",
               (unsigned long)ysrc->hAllocation,
               ysrc->venus_res_id,
               src_level, src_x, src_y, src_layer,
               ydst->data,
               dst_level, dstx, dsty, dstz,
               width, height,
               readback_stride, dst_stride,
               mapping.venus_res_id);
   ok = true;

out:
   yttrium_destroy_readback_mapping(ctx->screen, &stencil_mapping);
   yttrium_destroy_readback_mapping(ctx->screen, &mapping);
   yttrium_venus_resource_fini(screen->venus, NULL, &stencil_readback, NULL);
   yttrium_venus_resource_fini(screen->venus, NULL, &readback, NULL);
   return ok;
}

static bool
yttrium_copy_venus_buffer_to_cpu(struct pipe_context *ctx,
                                 struct yttrium_resource *ysrc,
                                 struct yttrium_resource *ydst,
                                 uint32_t src_offset,
                                 uint32_t dst_offset,
                                 uint32_t size)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_venus_resource readback;
   struct yttrium_readback_mapping mapping;
   uint64_t readback_mem_id = 0;
   bool ok = false;

   memset(&readback, 0, sizeof(readback));
   memset(&mapping, 0, sizeof(mapping));

   if (!ysrc || !ydst || ysrc->base.target != PIPE_BUFFER ||
       ydst->base.target != PIPE_BUFFER || !ysrc->venus.initialized ||
       !ysrc->venus.buffer_backed || !ysrc->venus.buffer || !ydst->data ||
       !size)
      return false;
   if (src_offset > ysrc->size || size > ysrc->size - src_offset ||
       dst_offset > ydst->size || size > ydst->size - dst_offset)
      return false;

   /* A newly initialized CPU-backed buffer may not have been bound for a
    * draw yet. Publish its current bytes before reading the native buffer;
    * allocation alone does not make the native contents authoritative.
    */
   if (!yttrium_prepare_gpu_copy_buffer(ctx, ysrc)) {
      YTTRIUM_WARN("yttrium: buffer readback failed owner=yttrium_resource "
                   "reason=source-publication-failed action=fail-draw\n");
      InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
      return false;
   }

   readback.cpu_readback = true;
   if (!yttrium_venus_create_display_buffer(screen->venus, &readback, size,
                                            &readback_mem_id)) {
      YTTRIUM_WARN("yttrium: resource_copy_region buffer to CPU failed to create readback buffer src_res_id=%u size=0x%x\n",
                   ysrc->venus_res_id, size);
      return false;
   }

   if (!yttrium_create_venus_memory_mapping(ctx->screen, readback_mem_id,
                                            readback.allocation_size,
                                            size,
                                            &mapping)) {
      YTTRIUM_WARN("yttrium: resource_copy_region buffer to CPU failed to map readback buffer src_res_id=%u mem_id=0x%llx size=0x%llx\n",
                   ysrc->venus_res_id,
                   (unsigned long long)readback_mem_id,
                   (unsigned long long)readback.allocation_size);
      goto out;
   }

   if (!yttrium_venus_copy_buffer_to_buffer(screen->venus, &ysrc->venus,
                                            &readback, src_offset, 0, size)) {
      YTTRIUM_WARN("yttrium: resource_copy_region buffer to CPU GPU copy failed src_res_id=%u dst=%p src_offset=0x%x size=0x%x\n",
                   ysrc->venus_res_id, ydst->data, src_offset, size);
      goto out;
   }
   if (!yttrium_venus_flush_labeled(screen->venus,
                                    "buffer to cpu readback flush")) {
      YTTRIUM_WARN("yttrium: resource_copy_region buffer to CPU wait failed src_res_id=%u dst=%p src_offset=0x%x size=0x%x\n",
                   ysrc->venus_res_id, ydst->data, src_offset, size);
      goto out;
   }

   memcpy((uint8_t *)ydst->data + dst_offset, mapping.map, size);
   MemoryBarrier();

   yttrium_trace_resource_copy(YTTRIUM_TRACE_COPY_BUFFER_TO_CPU,
                               0, ysrc->venus_res_id, 0, 0,
                               src_offset, 0, 0,
                               dst_offset, 0, 0,
                               size, 1, size, size);
   ok = true;

out:
   yttrium_destroy_readback_mapping(ctx->screen, &mapping);
   yttrium_venus_resource_fini(screen->venus, NULL, &readback, NULL);
   return ok;
}

static bool
yttrium_resource_formats_compatible(enum pipe_format src_format,
                                    enum pipe_format dst_format)
{
   if (src_format == dst_format)
      return true;

   if ((src_format == PIPE_FORMAT_Z16_UNORM &&
        dst_format == PIPE_FORMAT_R16_UNORM) ||
       (src_format == PIPE_FORMAT_Z32_FLOAT &&
        dst_format == PIPE_FORMAT_R32_FLOAT))
      return true;

   const enum pipe_format src_linear = util_format_linear(src_format);
   const enum pipe_format dst_linear = util_format_linear(dst_format);
   if (src_linear == dst_linear)
      return true;

   const struct util_format_description *src_desc =
      util_format_description(src_format);
   const struct util_format_description *dst_desc =
      util_format_description(dst_format);
   if (!src_desc || !dst_desc ||
       src_desc->block.width != dst_desc->block.width ||
       src_desc->block.height != dst_desc->block.height ||
       src_desc->block.bits != dst_desc->block.bits)
      return false;

   const struct util_format_description *src_linear_desc =
      util_format_description(src_linear);
   const struct util_format_description *dst_linear_desc =
      util_format_description(dst_linear);
   if (src_linear_desc && dst_linear_desc &&
       util_is_format_compatible(src_linear_desc, dst_linear_desc))
      return true;

   return util_is_format_compatible(src_desc, dst_desc);
}

static bool
yttrium_copy_venus_image_via_buffer(struct pipe_context *ctx,
                                    struct yttrium_resource *ysrc,
                                    struct yttrium_resource *ydst,
                                    uint32_t src_level,
                                    uint32_t src_x,
                                    uint32_t src_y,
                                    uint32_t src_layer,
                                    uint32_t dst_level,
                                    uint32_t dstx,
                                    uint32_t dsty,
                                    uint32_t dstz,
                                    uint32_t width,
                                    uint32_t height)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (!ysrc || !ydst || ysrc->classic_display || ydst->classic_display ||
       !ysrc->venus.initialized || ysrc->venus.buffer_backed ||
       !ysrc->venus.image || !ydst->venus.initialized ||
       ydst->venus.buffer_backed || !ydst->venus.image)
      return false;
   if (!yttrium_resource_formats_compatible(ysrc->base.format,
                                            ydst->base.format))
      return false;
   if (ysrc->base.nr_samples > 1 || ydst->base.nr_samples > 1 ||
       ysrc->venus.samples != VK_SAMPLE_COUNT_1_BIT ||
       ydst->venus.samples != VK_SAMPLE_COUNT_1_BIT) {
      YTTRIUM_WARN("yttrium: resource_copy_region image compatible copy unsupported owner=yttrium_resource_copy_region reason=multisampled_image_to_buffer src_res_id=%u src_format=%u src_samples=%u src_vk_samples=0x%x dst_res_id=%u dst_format=%u dst_samples=%u dst_vk_samples=0x%x\n",
                   ysrc->venus_res_id, ysrc->base.format,
                   ysrc->base.nr_samples, ysrc->venus.samples,
                   ydst->venus_res_id, ydst->base.format,
                   ydst->base.nr_samples, ydst->venus.samples);
      return false;
   }

   const unsigned cpp = util_format_get_blocksize(ydst->base.format);
   const uint32_t staging_stride =
      util_format_get_stride(ydst->base.format, width);
   const uint32_t rows = util_format_get_nblocksy(ydst->base.format, height);
   const uint64_t staging_size = (uint64_t)staging_stride * rows;
   if (!cpp || !staging_stride || !rows || !staging_size)
      return false;

   if (!yttrium_ensure_context_image_copy_staging(ctx, staging_size)) {
      YTTRIUM_WARN("yttrium: resource_copy_region image compatible copy failed to get context staging buffer src_res_id=%u dst_res_id=%u size=0x%llx\n",
                   ysrc->venus_res_id, ydst->venus_res_id,
                   (unsigned long long)staging_size);
      return false;
   }

   struct yttrium_venus_resource *staging = &yctx->image_copy_staging;
   const uint32_t staging_res_id =
      (uint32_t)yctx->image_copy_staging_mem_id;

   if (!yttrium_venus_copy_image_region_to_display_buffer(
          screen->venus, &ysrc->venus, staging,
          ysrc->venus_res_id, staging_res_id,
          src_level, src_layer, src_x, src_y, 0, 0, width, height,
          ysrc->base.format, staging_stride)) {
      YTTRIUM_WARN("yttrium: resource_copy_region image compatible copy failed image-to-buffer src_res_id=%u dst_res_id=%u\n",
                   ysrc->venus_res_id, ydst->venus_res_id);
      return false;
   }

   if (!yttrium_venus_copy_buffer_to_display_image(
          screen->venus, staging, &ydst->venus,
          staging_res_id, ydst->venus_res_id,
          0, staging_stride, 0, dst_level, dstx, dsty, dstz, width, height,
          1, ydst->base.format)) {
      YTTRIUM_WARN("yttrium: resource_copy_region image compatible copy failed buffer-to-image src_res_id=%u dst_res_id=%u\n",
                   ysrc->venus_res_id, ydst->venus_res_id);
      return false;
   }

   return true;
}

static void
yttrium_trace_copy_target(struct pipe_context *ctx,
                          uint32_t copy_id,
                          uint32_t stage,
                          uint32_t path,
                          const struct yttrium_resource *ysrc,
                          const struct yttrium_resource *ydst,
                          uint32_t dstx,
                          uint32_t dsty,
                          uint32_t width,
                          uint32_t height)
{
   if (!yttrium_trace_is_enabled())
      return;

   yttrium_trace_resource_copy_target(
      copy_id, stage, path,
      ysrc ? ysrc->hAllocation : 0,
      ysrc ? ysrc->venus_res_id : 0,
      ydst ? ydst->hAllocation : 0,
      ydst ? ydst->venus_res_id : 0,
      ydst ? ydst->base.target : 0,
      ydst ? ydst->base.format : 0,
      ydst ? ydst->base.bind : 0,
      ydst ? ydst->base.width0 : 0,
      ydst ? ydst->base.height0 : 0,
      dstx, dsty, width, height,
      ydst ? ydst->display_target : 0,
      ydst ? ydst->primary_target : 0,
      ydst ? ydst->classic_display : 0,
      ydst ? ydst->venus.initialized : 0,
      ydst ? ydst->venus.buffer_backed : 0,
      ydst && ydst->venus.image ? 1 : 0,
      ydst && ydst->data ? 1 : 0,
      0);
}

/* Publish initial CPU bytes before a partial GPU write.  In particular, the
 * instance count adjacent to CopyStructureCount's four-byte destination must
 * survive.  A buffer already written by the GPU must not be reinitialized from
 * its detached CPU shadow. */
static bool
yttrium_prepare_gpu_copy_buffer(struct pipe_context *ctx,
                                 struct yttrium_resource *res)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   if (!res || res->base.target != PIPE_BUFFER || !res->size)
      return false;

   if (!res->venus.initialized) {
      const VkBufferUsageFlags usage =
         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
         VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
         VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      if (!yttrium_venus_create_bind_buffer(screen->venus, &res->venus,
                                             res->size, usage, NULL))
         return false;
      res->venus.owner = &res->base;
      res->venus_res_id = (uint32_t)res->venus.memory_obj.id;
   }
   if (!res->venus.buffer_backed || !res->venus.buffer ||
       !(res->venus.buffer_usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT))
      return false;

   if (res->data_dirty) {
      if (!res->data)
         return false;
      if (!yttrium_resource_uses_mapped_venus_buffer(res) &&
          !yttrium_venus_update_buffer(screen->venus, &res->venus, 0,
                                        res->base.width0, res->data))
         return false;
      res->data_dirty = false;
      res->venus.contents_initialized = true;
   }
   return true;
}

static void
yttrium_copy_uav_counter(struct pipe_context *ctx,
                          struct yttrium_resource *dst, unsigned dst_level,
                          unsigned dstx, unsigned dsty, unsigned dstz,
                          struct yttrium_resource *src, unsigned src_level,
                          const struct pipe_box *box)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   if (!dst || !src || !box || dst->base.target != PIPE_BUFFER ||
       src->base.target != PIPE_TEXTURE_2D ||
       src->base.format != PIPE_FORMAT_R32_UINT ||
       src->base.width0 != 1 || src->base.height0 != 1 ||
       src->base.array_size != 1 || src->base.last_level != 0 ||
       dst_level || src_level || dsty || dstz || (dstx & 3) ||
       dstx > dst->base.width0 || 4 > dst->base.width0 - dstx ||
       box->x || box->y || box->z || box->width != 1 ||
       box->height != 1 || box->depth != 1 ||
       !yttrium_prepare_gpu_copy_buffer(ctx, dst) ||
       !yttrium_venus_copy_image_region_to_display_buffer(
          screen->venus, &src->venus, &dst->venus,
          src->venus_res_id, dst->venus_res_id,
          0, 0, 0, 0, dstx / 4, 0, 1, 1, PIPE_FORMAT_R32_UINT, 4)) {
      YTTRIUM_WARN("yttrium: counter copy failed owner=yttrium_resource reason=native-four-byte-image-copy-failed offset=%u action=fail-draw\n", dstx);
      InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
      return;
   }
   dst->gpu_buffer_written = true;
   dst->data_dirty = false;
   dst->venus.contents_initialized = true;
   dst->contents_serial++;
}

void
yttrium_resource_copy_region(struct pipe_context *ctx,
                             struct pipe_resource *dst,
                             unsigned dst_level,
                             unsigned dstx, unsigned dsty, unsigned dstz,
                             struct pipe_resource *src,
                             unsigned src_level,
                             const struct pipe_box *src_box)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_resource *ydst = yttrium_resource(dst);
   struct yttrium_resource *ysrc = yttrium_resource(src);
   uint32_t copy_id = 0;
   uint32_t copy_path = 0;

   /* This is an internal CopyStructureCount operation, not a texture-format
    * reinterpretation or a CPU readback.  Buffer x is a byte offset. */
   if (ysrc && (ysrc->base.flags & YTTRIUM_GDI_RESOURCE_FLAG_UAV_COUNTER) &&
       dst && dst->target == PIPE_BUFFER) {
      yttrium_copy_uav_counter(ctx, ydst, dst_level, dstx, dsty, dstz,
                                 ysrc, src_level, src_box);
      return;
   }

   const bool buffer_copy =
      src && dst && src->target == PIPE_BUFFER && dst->target == PIPE_BUFFER;

   if (!ydst || !ysrc || dst_level > ydst->base.last_level ||
       src_level > ysrc->base.last_level || !src_box ||
       src_box->x < 0 || src_box->y < 0 || src_box->z < 0 ||
       src_box->width <= 0 || src_box->height <= 0 ||
       src_box->depth <= 0 ||
       (!buffer_copy &&
        !yttrium_resource_formats_compatible(ysrc->base.format,
                                             ydst->base.format)))
      return;

   uint32_t src_x = (uint32_t)src_box->x;
   uint32_t src_y = (uint32_t)src_box->y;
   uint32_t src_layer = (uint32_t)src_box->z;
   uint32_t depth = (uint32_t)src_box->depth;
   uint32_t width = (uint32_t)src_box->width;
   uint32_t height = (uint32_t)src_box->height;
   const uint32_t src_level_width =
      yttrium_resource_level_width(&ysrc->base, src_level);
   const uint32_t src_level_height =
      yttrium_resource_level_height(&ysrc->base, src_level);
   const uint32_t dst_level_width =
      yttrium_resource_level_width(&ydst->base, dst_level);
   const uint32_t dst_level_height =
      yttrium_resource_level_height(&ydst->base, dst_level);
   const uint32_t src_depth_limit =
      src->target == PIPE_TEXTURE_3D ?
      yttrium_resource_level_depth(src, src_level) :
      MAX2(ysrc->base.array_size, 1);
   const uint32_t dst_depth_limit =
      dst->target == PIPE_TEXTURE_3D ?
      yttrium_resource_level_depth(dst, dst_level) :
      MAX2(ydst->base.array_size, 1);

   if (src_layer > src_depth_limit || dstz > dst_depth_limit ||
       depth > src_depth_limit - src_layer ||
       depth > dst_depth_limit - dstz ||
       src_x > src_level_width || src_y > src_level_height ||
       dstx > dst_level_width || dsty > dst_level_height ||
       width > src_level_width - src_x ||
       height > src_level_height - src_y ||
       width > dst_level_width - dstx ||
       height > dst_level_height - dsty)
      return;

   if (depth > 1) {
      struct pipe_box slice = *src_box;

      slice.depth = 1;
      for (uint32_t z = 0; z < depth; z++) {
         slice.z = src_box->z + (int)z;
         yttrium_resource_copy_region(ctx, dst, dst_level,
                                      dstx, dsty, dstz + z,
                                      src, src_level, &slice);
      }
      return;
   }

   copy_id = (uint32_t)InterlockedIncrement(&yttrium_resource_copy_sequence);
   yttrium_trace_copy_target(ctx, copy_id, YTTRIUM_TRACE_RESOURCE_COPY_BEGIN,
                             0, ysrc, ydst, dstx, dsty, width, height);

   /* Counter results copied onward to another GPU buffer remain GPU-owned.
    * Explicit staging/readback destinations retain the normal CPU-map path. */
   if (buffer_copy && ydst->base.usage != PIPE_USAGE_STAGING &&
       (ysrc->gpu_buffer_written || ydst->gpu_buffer_written) &&
       !(ysrc->base.flags & PIPE_RESOURCE_FLAG_SINGLE_THREAD_USE)) {
      if (!yttrium_prepare_gpu_copy_buffer(ctx, ysrc) ||
          !yttrium_prepare_gpu_copy_buffer(ctx, ydst) ||
          !yttrium_venus_copy_buffer_to_buffer(screen->venus,
                                                &ysrc->venus, &ydst->venus,
                                                src_x, dstx, width)) {
         YTTRIUM_WARN("yttrium: GPU buffer copy failed owner=yttrium_resource reason=counter-buffer-copy-failed action=fail-draw\n");
         InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
         return;
      }
      ydst->gpu_buffer_written = true;
      ydst->data_dirty = false;
      ydst->venus.contents_initialized = true;
      ydst->contents_serial++;
      return;
   }

   /*
    * u_threaded_context's private uploader remains mapped across allocations,
    * so its CPU bytes are authoritative before transfer-unmap.  Copy those
    * bytes directly instead of reading stale Venus contents back from the
    * host and turning the upload into a synchronous round trip.
    */
   if (buffer_copy &&
       (ysrc->base.flags & PIPE_RESOURCE_FLAG_SINGLE_THREAD_USE) &&
       ysrc->data && ydst->data &&
       src_x <= ysrc->size && width <= ysrc->size - src_x &&
       dstx <= ydst->size && width <= ydst->size - dstx) {
      if (ydst->gpu_buffer_written) {
         /* The uploader's host bytes are authoritative, but the destination
          * may still be read by an earlier indirect draw.  Record a GPU upload
          * rather than writing its mapped pages ahead of that draw. */
         if (!yttrium_venus_update_buffer(
                screen->venus, &ydst->venus, dstx, width,
                (const uint8_t *)ysrc->data + src_x)) {
            YTTRIUM_WARN("yttrium: ordered GPU buffer upload failed owner=yttrium_resource reason=counter-buffer-upload-failed action=fail-draw\n");
            InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
            return;
         }
         ydst->data_dirty = false;
         ydst->contents_serial++;
         return;
      }
      memmove((uint8_t *)ydst->data + dstx,
              (const uint8_t *)ysrc->data + src_x, width);
      yttrium_trace_resource_copy(YTTRIUM_TRACE_COPY_CPU_TO_CPU,
                                  0, ysrc->venus_res_id,
                                  0, ydst->venus_res_id,
                                  src_x, src_y, src_layer,
                                  dstx, dsty, dstz,
                                  width, height, 1, 1);
      copy_path = YTTRIUM_TRACE_COPY_CPU_TO_CPU;
      goto copied;
   }

   if (buffer_copy &&
       yttrium_copy_venus_buffer_to_cpu(ctx, ysrc, ydst, src_x, dstx,
                                        width)) {
      copy_path = YTTRIUM_TRACE_COPY_BUFFER_TO_CPU;
      goto copied;
   }

   /*
    * D3D permits a depth resource to be copied into a format-compatible
    * shader resource.  Sampler-only textures begin life with CPU storage so
    * their initial contents can be uploaded when an image is created lazily.
    * The copy itself must create that image when it is the first GPU use;
    * waiting for a sampling draw routes the copy through CPU storage and
    * turns every full-size depth copy into a synchronous GPU readback.
    *
    * A complete overwrite lets the destination image become authoritative.
    * Drop only driver-owned heap storage here: merely clearing data_dirty
    * would leave a stale, non-NULL shadow that transfer_map and later CPU
    * copies would incorrectly treat as current.  A destination whose shadow
    * has already been retired stays on this GPU path on subsequent copies.
    */
   const bool full_depth_to_sampled_image_copy =
      ysrc != ydst &&
      ysrc->base.target == PIPE_TEXTURE_2D &&
      ydst->base.target == PIPE_TEXTURE_2D &&
      ysrc->base.format == PIPE_FORMAT_Z32_FLOAT &&
      ydst->base.format == PIPE_FORMAT_R32_FLOAT &&
      src_level == 0 && dst_level == 0 &&
      src_x == 0 && src_y == 0 && src_layer == 0 &&
      dstx == 0 && dsty == 0 && dstz == 0 &&
      width == ysrc->base.width0 && height == ysrc->base.height0 &&
      width == ydst->base.width0 && height == ydst->base.height0 &&
      ysrc->base.last_level == 0 && ydst->base.last_level == 0 &&
      ysrc->base.array_size == 1 && ydst->base.array_size == 1 &&
      ysrc->base.depth0 == 1 && ydst->base.depth0 == 1 &&
      ysrc->base.nr_samples <= 1 && ydst->base.nr_samples <= 1 &&
      ysrc->base.nr_storage_samples <= 1 &&
      ydst->base.nr_storage_samples <= 1 &&
      ysrc->venus.samples == VK_SAMPLE_COUNT_1_BIT &&
      ysrc->venus.initialized && !ysrc->venus.buffer_backed &&
      ysrc->venus.image && ysrc->venus.contents_initialized &&
      !ysrc->classic_display && !ysrc->data_dirty &&
      (!ydst->venus.initialized ||
       (!ydst->venus.buffer_backed && ydst->venus.image &&
        ydst->venus.samples == VK_SAMPLE_COUNT_1_BIT)) &&
      !ydst->classic_display && !ydst->display_target &&
      !ydst->primary_target &&
      ydst->base.bind == PIPE_BIND_SAMPLER_VIEW &&
      ydst->base.usage == PIPE_USAGE_DEFAULT &&
      (!ydst->data ||
       (ydst->data_capacity >= ydst->size && ydst->owns_data)) &&
      !ydst->map && !ydst->hAllocation && !ydst->hResource &&
      !ydst->hAllocationResource && !ydst->hResourceIsD3D9Runtime &&
      !ydst->replacement_storage && !ydst->replacement_owner;

   if (full_depth_to_sampled_image_copy) {
      if (!ydst->venus.initialized) {
         uint64_t allocation_size = 0;
         if (!yttrium_venus_create_sampled_texture_image(
                screen->venus, &ydst->venus, ydst->base.target,
                width, height, 1, 1, 1, ydst->base.format,
                &allocation_size)) {
            YTTRIUM_WARN("yttrium: resource_copy_region failed owner=yttrium_resource_copy_region reason=depth-copy-destination-image-create-failed dst=%p extent=%ux%u\n",
                         (void *)ydst, width, height);
            goto unsupported;
         }
         ydst->venus_mem_id = ydst->venus.memory_obj.id;
         ydst->venus_res_id = (uint32_t)ydst->venus.memory_obj.id;
      }

      if (!yttrium_copy_venus_image_via_buffer(ctx, ysrc, ydst, src_level,
                                               src_x, src_y, src_layer,
                                               dst_level, dstx, dsty, dstz,
                                               width, height)) {
         YTTRIUM_WARN("yttrium: resource_copy_region failed owner=yttrium_resource_copy_region reason=depth-to-sampled-gpu-copy-failed src_res_id=%u dst_res_id=%u extent=%ux%u\n",
                      ysrc->venus_res_id, ydst->venus_res_id, width, height);
         goto unsupported;
      }

      if (ydst->data)
         YTTRIUM_LOG("yttrium: resource_copy_region retired CPU shadow after full GPU depth-to-sampled copy dst=%p res_id=%u data=%p capacity=0x%llx extent=%ux%u\n",
                     (void *)ydst, ydst->venus_res_id, ydst->data,
                     (unsigned long long)ydst->data_capacity, width, height);
      FREE(ydst->data);
      ydst->data = NULL;
      ydst->data_capacity = 0;
      ydst->owns_data = false;
      ydst->data_dirty = false;
      copy_path = YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_DISPLAY_IMAGE;
      goto copied;
   }

   if (!ysrc->classic_display &&
       ysrc->venus.initialized && !ysrc->venus.buffer_backed &&
       ysrc->venus.image && ydst->data &&
       yttrium_copy_venus_image_to_cpu(ctx, ysrc, ydst, src_level,
                                       src_x, src_y, src_layer,
                                       dst_level,
                                       dstx, dsty, dstz,
                                       width, height)) {
      copy_path = YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_CPU;
      goto copied;
   }

   if (ysrc->data && ydst->data) {
      const unsigned cpp = util_format_get_blocksize(ydst->base.format);
      if (cpp) {
         const uint8_t *src_ptr =
            (const uint8_t *)ysrc->data +
            yttrium_resource_offset(&ysrc->base, src_level, src_x, src_y,
                                    src_layer);
         uint8_t *dst_ptr =
            (uint8_t *)ydst->data +
            yttrium_resource_offset(&ydst->base, dst_level, dstx, dsty,
                                    dstz);
         const unsigned src_stride =
            yttrium_resource_level_stride(&ysrc->base, src_level);
         const unsigned dst_stride =
            yttrium_resource_level_stride(&ydst->base, dst_level);
         const uint32_t row_bytes =
            util_format_get_stride(ydst->base.format, width);
         const uint32_t rows =
            util_format_get_nblocksy(ydst->base.format, height);

         for (uint32_t y = 0; y < rows; y++) {
            memcpy(dst_ptr + (uint64_t)y * dst_stride,
                   src_ptr + (uint64_t)y * src_stride,
                   row_bytes);
         }
         yttrium_trace_resource_copy(YTTRIUM_TRACE_COPY_CPU_TO_CPU,
                                     0, 0, 0, 0,
                                     src_x, src_y, src_layer,
                                     dstx, dsty, dstz,
                                     width, height,
                                     src_stride, dst_stride);
         YTTRIUM_LOG("yttrium: resource_copy_region copied CPU to CPU src=%p dst=%p src_level=%u src=%u,%u layer=%u dst_level=%u dst=%u,%u layer=%u %ux%u stride=%u/%u\n",
                     ysrc->data,
                     ydst->data,
                     src_level, src_x, src_y, src_layer,
                     dst_level, dstx, dsty, dstz,
                     width, height,
                     src_stride, dst_stride);
         copy_path = YTTRIUM_TRACE_COPY_CPU_TO_CPU;
         goto copied;
      }
   }

   if (ysrc->data && ydst->display_target && !ydst->classic_display &&
       ydst->venus.buffer_backed) {
      const unsigned cpp = util_format_get_blocksize(ydst->base.format);
      if (cpp && yttrium_ensure_display_mapped(ctx, ydst)) {
         const uint8_t *src_ptr =
            (const uint8_t *)ysrc->data +
            yttrium_resource_offset(&ysrc->base, src_level, src_x, src_y,
                                    src_layer);
         uint8_t *dst_ptr =
            (uint8_t *)ydst->map +
            yttrium_resource_offset(&ydst->base, dst_level, dstx, dsty,
                                    dstz);
         const unsigned src_stride =
            yttrium_resource_level_stride(&ysrc->base, src_level);
         const unsigned dst_stride =
            yttrium_resource_level_stride(&ydst->base, dst_level);
         const uint32_t row_bytes =
            util_format_get_stride(ydst->base.format, width);
         const uint32_t rows =
            util_format_get_nblocksy(ydst->base.format, height);

         for (uint32_t y = 0; y < rows; y++) {
            memcpy(dst_ptr + (uint64_t)y * dst_stride,
                   src_ptr + (uint64_t)y * src_stride,
                   row_bytes);
         }
         MemoryBarrier();
         yttrium_trace_resource_copy(YTTRIUM_TRACE_COPY_CPU_TO_DISPLAY_BUFFER,
                                     0, 0, ydst->hAllocation,
                                     ydst->venus_res_id,
                                     src_x, src_y, src_layer,
                                     dstx, dsty, dstz,
                                     width, height,
                                     src_stride, dst_stride);
         YTTRIUM_LOG("yttrium: resource_copy_region copied CPU to display buffer src=%p dst hAllocation=0x%lx res_id=%u src_level=%u src=%u,%u layer=%u dst_level=%u dst=%u,%u layer=%u %ux%u stride=%u/%u\n",
                     ysrc->data,
                     (unsigned long)ydst->hAllocation,
                     ydst->venus_res_id,
                     src_level, src_x, src_y, src_layer,
                     dst_level, dstx, dsty, dstz,
                     width, height,
                     src_stride, dst_stride);
         copy_path = YTTRIUM_TRACE_COPY_CPU_TO_DISPLAY_BUFFER;
         goto copied;
      }
   }

   if (yttrium_copy_cpu_to_venus_image(ctx, ysrc, ydst, src_level, dst_level,
                                       src_x, src_y, src_layer,
                                       dstx, dsty, dstz,
                                       width, height, 1)) {
      copy_path = YTTRIUM_TRACE_COPY_CPU_TO_DISPLAY_IMAGE;
      goto copied;
   }

   if (!ydst->classic_display && !ysrc->classic_display &&
       ydst->venus.initialized && ysrc->venus.initialized &&
       !ydst->venus.buffer_backed && !ysrc->venus.buffer_backed &&
       ydst->base.format != ysrc->base.format &&
       yttrium_copy_venus_image_via_buffer(ctx, ysrc, ydst, src_level,
                                           src_x, src_y, src_layer,
                                           dst_level, dstx, dsty, dstz,
                                           width, height)) {
      copy_path = YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_DISPLAY_IMAGE;
      goto copied;
   }

   /*
    * Depth included: yttrium_venus_copy_display_image copies whichever aspects
    * the formats share.  The depth-stencil helper next door is not usable here
    * - it replays clear history rather than copying, so it only applies to a
    * source holding nothing but clears.
    */
   if (!ydst->classic_display && !ysrc->classic_display &&
       ydst->venus.initialized && ysrc->venus.initialized &&
       !ydst->venus.buffer_backed && !ysrc->venus.buffer_backed &&
       ydst->base.format == ysrc->base.format &&
       yttrium_venus_copy_display_image(screen->venus,
                                        &ysrc->venus, &ydst->venus,
                                        ysrc->venus_res_id,
                                       ydst->venus_res_id,
                                       src_level,
                                       src_x, src_y, src_layer,
                                       dst_level,
                                       dstx, dsty, dstz,
                                        width, height)) {
      yttrium_trace_resource_copy(
         YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_DISPLAY_IMAGE,
         ysrc->hAllocation, ysrc->venus_res_id,
         ydst->hAllocation, ydst->venus_res_id,
         src_x, src_y, src_layer,
         dstx, dsty, dstz,
         width, height,
         ysrc->stride, ydst->stride);
      YTTRIUM_LOG("yttrium: resource_copy_region copied display image src hAllocation=0x%lx res_id=%u %u,%u layer=%u dst hAllocation=0x%lx res_id=%u %u,%u layer=%u %ux%u\n",
                  (unsigned long)ysrc->hAllocation,
                  ysrc->venus_res_id,
                  src_x, src_y, src_layer,
                  (unsigned long)ydst->hAllocation,
                  ydst->venus_res_id,
                  dstx, dsty, dstz,
                  width, height);
      copy_path = YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_DISPLAY_IMAGE;
      goto copied;
   }

unsupported:
   yttrium_trace_copy_target(ctx, copy_id,
                             YTTRIUM_TRACE_RESOURCE_COPY_UNSUPPORTED_STAGE,
                             0, ysrc, ydst, dstx, dsty, width, height);
   yttrium_trace_resource_copy_unsupported(
      ysrc ? ysrc->hAllocation : 0,
      ysrc ? ysrc->venus_res_id : 0,
      ydst ? ydst->hAllocation : 0,
      ydst ? ydst->venus_res_id : 0,
      src_box ? src_box->width : 0,
      src_box ? src_box->height : 0);
   /*
    * A refused copy leaves the destination holding whatever it held before,
    * which reaches the screen as a wrong or empty image with no other symptom.
    * This was debug-level for a depth copy Superposition depends on, so the
    * driver reported nothing at all while rendering black.  Warn instead, with
    * the formats, since those are what decide which path was eligible.
    */
   YTTRIUM_WARN("yttrium: resource_copy_region unsupported box=%dx%d "
                "src res_id=%u format=%u bind=0x%x venus_init=%u buffer_backed=%u image=%u usage=0x%x data=%u "
                "dst res_id=%u format=%u bind=0x%x venus_init=%u buffer_backed=%u image=%u usage=0x%x data=%u\n",
                src_box ? src_box->width : 0,
                src_box ? src_box->height : 0,
                ysrc ? ysrc->venus_res_id : 0,
                ysrc ? ysrc->base.format : 0,
                ysrc ? ysrc->base.bind : 0,
                ysrc ? ysrc->venus.initialized : 0,
                ysrc ? ysrc->venus.buffer_backed : 0,
                ysrc && ysrc->venus.image ? 1 : 0,
                ysrc ? ysrc->venus.image_usage : 0,
                ysrc && ysrc->data ? 1 : 0,
                ydst ? ydst->venus_res_id : 0,
                ydst ? ydst->base.format : 0,
                ydst ? ydst->base.bind : 0,
                ydst ? ydst->venus.initialized : 0,
                ydst ? ydst->venus.buffer_backed : 0,
                ydst && ydst->venus.image ? 1 : 0,
                ydst ? ydst->venus.image_usage : 0,
                ydst && ydst->data ? 1 : 0);
   return;

copied:
   if (buffer_copy && ydst->gpu_buffer_written && ydst->data &&
       (copy_path == YTTRIUM_TRACE_COPY_CPU_TO_CPU ||
        copy_path == YTTRIUM_TRACE_COPY_BUFFER_TO_CPU)) {
      /* Ordered uploader writes carry only this byte range.  Do not publish
       * untouched shadow bytes over an adjacent GPU-written counter. */
      if (!yttrium_resource_uses_mapped_venus_buffer(ydst) &&
          !yttrium_venus_update_buffer(screen->venus, &ydst->venus, dstx,
                                        width, (uint8_t *)ydst->data + dstx)) {
         YTTRIUM_WARN("yttrium: buffer range publication failed owner=yttrium_resource reason=counter-buffer-range-upload-failed action=fail-draw\n");
         InterlockedExchange(&yttrium_context(ctx)->draw_failure, 1);
      }
      ydst->data_dirty = false;
      ydst->contents_serial++;
      return;
   }
   if (ydst && ydst->data &&
       (copy_path == YTTRIUM_TRACE_COPY_CPU_TO_CPU ||
        copy_path == YTTRIUM_TRACE_COPY_DISPLAY_IMAGE_TO_CPU ||
        copy_path == YTTRIUM_TRACE_COPY_BUFFER_TO_CPU))
      ydst->data_dirty = true;

   /* Every path that got here wrote the destination, so the serial always
    * moves - it just has to be its own statement, not the second line of the
    * unbraced if above. */
   ydst->contents_serial++;

   yttrium_trace_copy_target(ctx, copy_id, YTTRIUM_TRACE_RESOURCE_COPY_COPIED,
                             copy_path, ysrc, ydst, dstx, dsty, width, height);
}

static bool
yttrium_blit_target_supported(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_3D:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return true;
   default:
      return false;
   }
}

static bool
yttrium_blit_ensure_image(struct pipe_context *ctx,
                          struct yttrium_resource *res)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   uint64_t allocation_size = 0;

   if (!screen || !res || res->classic_display ||
       !yttrium_blit_target_supported(res->base.target))
      return false;

   if (res->venus.initialized)
      return !res->venus.buffer_backed && res->venus.image;

   const bool depth_stencil =
      (res->base.bind & PIPE_BIND_DEPTH_STENCIL) != 0 ||
      util_format_is_depth_or_stencil(res->base.format);
   const unsigned bind = depth_stencil ?
      res->base.bind : (res->base.bind | PIPE_BIND_SAMPLER_VIEW);
   if (!yttrium_venus_create_texture_image_for_bind(
          screen->venus, &res->venus, res->base.target,
          res->base.width0, res->base.height0, res->base.depth0,
          res->base.last_level + 1, res->base.array_size,
          res->base.format, res->base.nr_samples, bind,
          &allocation_size))
      return false;

   res->size = MAX2(res->size, allocation_size);
   if (!res->venus_res_id)
      res->venus_res_id = (uint32_t)res->venus.memory_obj.id;
   if (!res->venus_mem_id)
      res->venus_mem_id = res->venus.memory_obj.id;

   YTTRIUM_LOG("yttrium: blit promoted texture to Venus image resource=%p "
               "res_id=%u image_id=%llu memory_id=%llu target=%u "
               "%ux%ux%u levels=%u array=%u format=%u bind=0x%x\n",
               (void *)res, res->venus_res_id,
               (unsigned long long)res->venus.image_obj.id,
               (unsigned long long)res->venus.memory_obj.id,
               res->base.target, res->base.width0, res->base.height0,
               res->base.depth0, res->base.last_level + 1,
               res->base.array_size, res->base.format, bind);
   return true;
}

static bool
yttrium_blit_upload_cpu_contents(struct pipe_context *ctx,
                                 struct yttrium_resource *res)
{
   if (!res || !res->data ||
       (res->venus.contents_initialized && !res->data_dirty))
      return true;

   bool upload_ok = true;
   for (unsigned level = 0; level <= res->base.last_level; level++) {
      const unsigned width = yttrium_resource_level_width(&res->base, level);
      const unsigned height = yttrium_resource_level_height(&res->base, level);

      if (res->base.target == PIPE_TEXTURE_3D) {
         const unsigned depth =
            yttrium_resource_level_depth(&res->base, level);
         upload_ok =
            yttrium_copy_cpu_to_venus_image(ctx, res, res, level, level,
                                            0, 0, 0, 0, 0, 0,
                                            width, height, depth) &&
            upload_ok;
      } else {
         const unsigned layers = MAX2(res->base.array_size, 1);
         for (unsigned layer = 0; layer < layers; layer++) {
            upload_ok =
               yttrium_copy_cpu_to_venus_image(ctx, res, res, level, level,
                                               0, 0, layer, 0, 0, layer,
                                               width, height, 1) &&
               upload_ok;
         }
      }
   }

   if (upload_ok)
      res->data_dirty = false;
   return upload_ok;
}

void
yttrium_blit(struct pipe_context *ctx, const struct pipe_blit_info *info)
{
   if (ctx && info && info->src.resource && info->dst.resource) {
      struct yttrium_screen *screen = yttrium_screen(ctx->screen);
      struct yttrium_resource *src = yttrium_resource(info->src.resource);
      struct yttrium_resource *dst = yttrium_resource(info->dst.resource);
      const struct pipe_box *src_box = &info->src.box;
      const struct pipe_box *dst_box = &info->dst.box;
      const bool src_3d = src->base.target == PIPE_TEXTURE_3D;
      const bool dst_3d = dst->base.target == PIPE_TEXTURE_3D;
      const bool supported_targets =
         yttrium_blit_target_supported(src->base.target) &&
         yttrium_blit_target_supported(dst->base.target) &&
         src_3d == dst_3d;
      const bool depth_stencil_copy =
         (info->mask & PIPE_MASK_ZS) != 0 &&
         (info->mask & ~PIPE_MASK_ZS) == 0 &&
         src->base.format == dst->base.format &&
         info->src.format == src->base.format &&
         info->dst.format == dst->base.format &&
         info->filter == PIPE_TEX_FILTER_NEAREST &&
         src_box->width == dst_box->width &&
         src_box->height == dst_box->height &&
         src_box->depth == dst_box->depth &&
         src->base.nr_samples == dst->base.nr_samples &&
         src->base.nr_samples <= 1;
      const bool supported_blit_shape =
         screen && src && dst &&
         !src->classic_display && !dst->classic_display &&
         supported_targets &&
         yttrium_resource_formats_compatible(info->src.format,
                                             src->base.format) &&
         yttrium_resource_formats_compatible(info->dst.format,
                                             dst->base.format) &&
         (info->mask & PIPE_MASK_RGBA) == info->mask &&
         info->mask != 0 &&
         !info->scissor_enable &&
         info->src.level <= src->base.last_level &&
         info->dst.level <= dst->base.last_level &&
         src_box->x >= 0 && src_box->y >= 0 && src_box->z >= 0 &&
         dst_box->x >= 0 && dst_box->y >= 0 && dst_box->z >= 0 &&
         src_box->width > 0 && src_box->height > 0 &&
         dst_box->width > 0 && dst_box->height > 0 &&
         src_box->depth > 0 && dst_box->depth > 0 &&
         (src_3d || src_box->depth == dst_box->depth);

      if (screen && depth_stencil_copy &&
          !src->classic_display && !dst->classic_display &&
          supported_targets &&
          src_box->x >= 0 && src_box->y >= 0 && src_box->z >= 0 &&
          dst_box->x >= 0 && dst_box->y >= 0 && dst_box->z >= 0 &&
          src_box->width > 0 && src_box->height > 0 &&
          dst_box->width > 0 && dst_box->height > 0 &&
          src_box->depth > 0 && dst_box->depth > 0 &&
          (src_3d || src_box->depth == dst_box->depth) &&
          yttrium_blit_ensure_image(ctx, src) &&
          yttrium_blit_ensure_image(ctx, dst) &&
          yttrium_blit_upload_cpu_contents(ctx, src) &&
          yttrium_blit_upload_cpu_contents(ctx, dst)) {
         bool ok = true;
         VkImageAspectFlags aspect_mask = 0;

         if (info->mask & PIPE_MASK_Z)
            aspect_mask |= VK_IMAGE_ASPECT_DEPTH_BIT;
         if (info->mask & PIPE_MASK_S)
            aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;

         for (int z = 0; z < src_box->depth && ok; z++) {
            ok = yttrium_venus_copy_depth_stencil_image(
               screen->venus, &src->venus, &dst->venus,
               src->venus_res_id, dst->venus_res_id,
               info->src.level,
               (uint32_t)src_box->x,
               (uint32_t)src_box->y,
               (uint32_t)src_box->z + (uint32_t)z,
               info->dst.level,
               (uint32_t)dst_box->x,
               (uint32_t)dst_box->y,
               (uint32_t)dst_box->z + (uint32_t)z,
               (uint32_t)src_box->width,
               (uint32_t)src_box->height,
               aspect_mask);
         }

         if (ok)
            return;
      }

      if (supported_blit_shape &&
          (!yttrium_blit_ensure_image(ctx, src) ||
           !yttrium_blit_ensure_image(ctx, dst) ||
           !yttrium_blit_upload_cpu_contents(ctx, src) ||
           !yttrium_blit_upload_cpu_contents(ctx, dst)))
         goto unsupported;

      if (supported_blit_shape &&
          src->venus.initialized && dst->venus.initialized &&
          !src->venus.buffer_backed && !dst->venus.buffer_backed &&
          src->venus.image && dst->venus.image &&
          supported_targets) {
         bool ok = true;

         const bool msaa_resolve =
            src->venus.samples != VK_SAMPLE_COUNT_1_BIT &&
            dst->venus.samples == VK_SAMPLE_COUNT_1_BIT &&
            src_box->width == dst_box->width &&
            src_box->height == dst_box->height &&
            src_box->depth == dst_box->depth;

         if (msaa_resolve) {
            if (src_3d || src_box->depth != 1) {
               ok = false;
            } else if (!yttrium_venus_resolve_display_image(
                          screen->venus, &src->venus, &dst->venus,
                          src->venus_res_id, dst->venus_res_id,
                          info->src.level,
                          (uint32_t)src_box->x,
                          (uint32_t)src_box->y,
                          (uint32_t)src_box->z,
                          info->dst.level,
                          (uint32_t)dst_box->x,
                          (uint32_t)dst_box->y,
                          (uint32_t)dst_box->z,
                          info->dst.format,
                          (uint32_t)src_box->width,
                          (uint32_t)src_box->height)) {
               ok = false;
            }
         } else if (src_3d) {
            if (!yttrium_venus_blit_display_image(
                   screen->venus, &src->venus, &dst->venus,
                   src->venus_res_id, dst->venus_res_id,
                   info->src.level,
                   (uint32_t)src_box->x,
                   (uint32_t)src_box->y,
                   (uint32_t)src_box->z,
                   (uint32_t)src_box->width,
                   (uint32_t)src_box->height,
                   (uint32_t)src_box->depth,
                   info->dst.level,
                   (uint32_t)dst_box->x,
                   (uint32_t)dst_box->y,
                   (uint32_t)dst_box->z,
                   (uint32_t)dst_box->width,
                   (uint32_t)dst_box->height,
                   (uint32_t)dst_box->depth,
                   info->filter == PIPE_TEX_FILTER_LINEAR)) {
               ok = false;
            }
         } else {
            for (int z = 0; z < src_box->depth; z++) {
               if (!yttrium_venus_blit_display_image(
                      screen->venus, &src->venus, &dst->venus,
                      src->venus_res_id, dst->venus_res_id,
                      info->src.level,
                      (uint32_t)src_box->x,
                      (uint32_t)src_box->y,
                      (uint32_t)src_box->z + (uint32_t)z,
                      (uint32_t)src_box->width,
                      (uint32_t)src_box->height,
                      1,
                      info->dst.level,
                      (uint32_t)dst_box->x,
                      (uint32_t)dst_box->y,
                      (uint32_t)dst_box->z + (uint32_t)z,
                      (uint32_t)dst_box->width,
                      (uint32_t)dst_box->height,
                      1,
                      info->filter == PIPE_TEX_FILTER_LINEAR)) {
                  ok = false;
                  break;
               }
            }
         }

         if (ok)
            return;
      }
   }

unsupported:
   YTTRIUM_WARN("yttrium: blit unsupported src=%p dst=%p src_level=%u dst_level=%u src_box=%d,%d,%d %dx%dx%d dst_box=%d,%d,%d %dx%dx%d mask=0x%x filter=%u\n",
                info ? (void *)info->src.resource : NULL,
                info ? (void *)info->dst.resource : NULL,
                info ? info->src.level : 0,
                info ? info->dst.level : 0,
                info ? info->src.box.x : 0,
                info ? info->src.box.y : 0,
                info ? info->src.box.z : 0,
                info ? info->src.box.width : 0,
                info ? info->src.box.height : 0,
                info ? info->src.box.depth : 0,
                info ? info->dst.box.x : 0,
                info ? info->dst.box.y : 0,
                info ? info->dst.box.z : 0,
                info ? info->dst.box.width : 0,
                info ? info->dst.box.height : 0,
                info ? info->dst.box.depth : 0,
                info ? info->mask : 0,
                info ? info->filter : 0);
}

void
yttrium_flush_resource(struct pipe_context *ctx, struct pipe_resource *resource)
{
   struct yttrium_screen *screen;
   struct yttrium_resource *res;

   if (!ctx || !resource)
      return;

   screen = yttrium_screen(ctx->screen);
   res = yttrium_resource(resource);
   if (!screen || !screen->venus || !res->venus.initialized)
      return;

   if (!yttrium_venus_flush_async_labeled(screen->venus,
                                           "pipe resource flush")) {
      YTTRIUM_WARN("yttrium: resource flush submit failed owner=yttrium_resource reason=pending Venus batch submission failed resource=%p res_id=%u\n",
                   (void *)res, res->venus_res_id);
      return;
   }

   if (!yttrium_venus_wait_resource(screen->venus, &res->venus,
                                     "pipe resource flush")) {
      YTTRIUM_WARN("yttrium: resource flush wait failed owner=yttrium_resource reason=Venus batch wait failed resource=%p res_id=%u\n",
                   (void *)res, res->venus_res_id);
   }
}
