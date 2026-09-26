/**************************************************************************
 *
 * Copyright 2012-2021 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE COPYRIGHT HOLDERS, AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 **************************************************************************/

/*
 * Resource.cpp --
 *    Functions that manipulate GPU resources.
 */


#include "Resource.h"
#include "Format.h"
#include "State.h"
#include "Query.h"

#include "Debug.h"

#include "pipe/p_defines.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "util/u_rect.h"
#include "util/u_surface.h"
#include "util/u_atomic.h"
#include "util/u_threaded_context.h"
#include "util/u_upload_mgr.h"

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include <emmintrin.h>
#endif
#include <string.h>


#define CONSTANT_PUBLICATION_NONTEMPORAL_MIN_SIZE (4u * 1024u)

static void
CopyConstantPublication(void *dst, const void *src, size_t size)
{
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
   if (size >= CONSTANT_PUBLICATION_NONTEMPORAL_MIN_SIZE &&
       !(reinterpret_cast<uintptr_t>(dst) & 15u) && !(size & 15u)) {
      __m128i *dst_vec = reinterpret_cast<__m128i *>(dst);
      const __m128i *src_vec = reinterpret_cast<const __m128i *>(src);
      const size_t vector_count = size / sizeof(__m128i);

      size_t i = 0;
      for (; i + 4 <= vector_count; i += 4) {
         const __m128i value0 = _mm_loadu_si128(src_vec + i + 0);
         const __m128i value1 = _mm_loadu_si128(src_vec + i + 1);
         const __m128i value2 = _mm_loadu_si128(src_vec + i + 2);
         const __m128i value3 = _mm_loadu_si128(src_vec + i + 3);
         _mm_stream_si128(dst_vec + i + 0, value0);
         _mm_stream_si128(dst_vec + i + 1, value1);
         _mm_stream_si128(dst_vec + i + 2, value2);
         _mm_stream_si128(dst_vec + i + 3, value3);
      }
      for (; i < vector_count; i++)
         _mm_stream_si128(dst_vec + i, _mm_loadu_si128(src_vec + i));

      /* Make the streamed bytes visible before the ordered worker can issue
       * a host read from this directly published resource. */
      _mm_sfence();
      return;
   }
#endif

   memcpy(dst, src, size);
}

static void
RegisterResource(Device *device, Resource *resource, const char *reason,
                 D3D10DDI_HRESOURCE hResource, D3D10DDI_HRTRESOURCE hRTResource)
{
   if (!device || !resource || resource->listed)
      return;

   list_addtail(&resource->list, &device->resources);
   resource->listed = true;

   ResourceEvent(RESOURCE_EVENT_REGISTER,
                 (uint64_t)(uintptr_t)hResource.pDrvPrivate,
                 resource,
                 resource->resource,
                 resource->resource ?
                    p_atomic_read(&resource->resource->reference.count) : 0,
                 resource->yttrium_primary ? 1 : 0,
                 resource->buffer ? 1 : 0,
                 resource->NumSubResources);
}

static void
UnregisterResource(Device *device, Resource *resource, const char *reason,
                   D3D10DDI_HRESOURCE hResource)
{
   if (!device || !resource || !resource->listed)
      return;

   ResourceEvent(RESOURCE_EVENT_UNREGISTER,
                 (uint64_t)(uintptr_t)hResource.pDrvPrivate,
                 resource,
                 resource->resource,
                 resource->resource ?
                    p_atomic_read(&resource->resource->reference.count) : 0,
                 resource->yttrium_primary ? 1 : 0,
                 resource->buffer ? 1 : 0,
                 resource->NumSubResources);

   list_delinit(&resource->list);
   resource->listed = false;
}

static void
UpdateBufferShadow(Resource *resource,
                   unsigned offset,
                   unsigned size,
                   const void *data)
{
   if (!resource || !resource->buffer || !size || !data)
      return;

   if (offset > UINT_MAX - size)
      return;

   const unsigned required_size = offset + size;
   if (required_size > resource->buffer_shadow_size) {
      void *shadow = realloc(resource->buffer_shadow, required_size);
      if (!shadow)
         return;

      memset((uint8_t *)shadow + resource->buffer_shadow_size, 0,
             required_size - resource->buffer_shadow_size);
      resource->buffer_shadow = shadow;
      resource->buffer_shadow_size = required_size;
   }

   memcpy((uint8_t *)resource->buffer_shadow + offset, data, size);
}

static bool
EnsureBufferShadow(Resource *resource, unsigned size)
{
   if (!resource || !resource->buffer)
      return false;

   if (resource->buffer_shadow_size >= size)
      return true;

   void *shadow = realloc(resource->buffer_shadow, size);
   if (!shadow)
      return false;

   memset((uint8_t *)shadow + resource->buffer_shadow_size, 0,
          size - resource->buffer_shadow_size);
   resource->buffer_shadow = shadow;
   resource->buffer_shadow_size = size;
   return true;
}

static void
ReleasePipeResourceOrdered(struct pipe_context *pipe,
                           struct pipe_resource *resource)
{
   if (!resource)
      return;

   if (pipe && pipe->resource_release)
      pipe_resource_release(pipe, resource);
   else
      pipe_resource_reference(&resource, NULL);
}

static void
ReleasePublishedConstantBuffer(struct pipe_context *pipe, Resource *resource)
{
   if (!resource)
      return;

   struct pipe_resource *published = resource->constant_published_buffer;
   resource->constant_published_buffer = NULL;
   resource->constant_published_cpu = NULL;
   resource->constant_published_offset = 0;
   resource->constant_published_size = 0;
   ReleasePipeResourceOrdered(pipe, published);
}

static bool
ConstantBufferPublicationCandidate(Device *device, Resource *resource)
{
   return device && device->constant_publication_enabled && device->pipe &&
          device->pipe->const_uploader && resource && resource->buffer &&
          resource->resource && resource->resource->target == PIPE_BUFFER &&
          resource->resource->bind == PIPE_BIND_CONSTANT_BUFFER &&
          resource->resource->usage != PIPE_USAGE_DYNAMIC &&
          resource->resource->usage != PIPE_USAGE_STAGING &&
          resource->resource->width0 > 0;
}

static bool
ConstantBufferPublicationEligible(Device *device, Resource *resource)
{
   return ConstantBufferPublicationCandidate(device, resource);
}

static bool
FillPublishedConstantBuffer(Device *device, Resource *resource,
                            unsigned resource_offset,
                            unsigned buffer_size,
                            struct pipe_constant_buffer *cb)
{
   if (!ConstantBufferPublicationEligible(device, resource) || !cb ||
       !buffer_size || !resource->constant_published_buffer ||
       resource_offset > resource->constant_published_size ||
       buffer_size > resource->constant_published_size - resource_offset ||
       resource->constant_published_offset > UINT_MAX - resource_offset)
      return false;

   cb->buffer = resource->constant_published_buffer;
   cb->user_buffer = NULL;
   cb->buffer_offset = resource->constant_published_offset + resource_offset;
   cb->buffer_size = buffer_size;
   return true;
}

static void
RebindConstantBufferPublication(Device *device, Resource *resource,
                                bool published)
{
   if (!device || !resource || !resource->resource)
      return;

   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; stage++) {
      uint32_t bindings = resource->constant_buffer_bindings[stage];
      while (bindings) {
         const unsigned slot = u_bit_scan(&bindings);
         if (device->constant_buffer_binding_resources[stage][slot] !=
             resource ||
             !device->constant_buffer_sizes[stage][slot])
            continue;

         struct pipe_constant_buffer cb = {};
         if (published) {
            if (!FillPublishedConstantBuffer(
                   device, resource,
                   device->constant_buffer_offsets[stage][slot],
                   device->constant_buffer_sizes[stage][slot], &cb))
               continue;
         } else {
            if (!device->constant_buffer_published[stage][slot])
               continue;
            cb.buffer = resource->resource;
            cb.buffer_offset = device->constant_buffer_offsets[stage][slot];
            cb.buffer_size = device->constant_buffer_sizes[stage][slot];
         }

         device->pipe->set_constant_buffer(
            device->pipe, (mesa_shader_stage)stage, slot, &cb);
         device->constant_buffer_published[stage][slot] = published;
      }
   }
}

static void
WarnConstantBufferPublicationFallback(Device *device,
                                      Resource *resource,
                                      const char *reason)
{
   if (!device || !resource || resource->constant_publication_fallback_warned)
      return;

   resource->constant_publication_fallback_warned = true;
   yttrium_gdi_trace_warnf(
      "yttrium: WARNING: direct constant-buffer publication fallback owner=d3d10umd resource=%p pipe_resource=%p reason=%s action=restore-and-bind-original-resource\n",
      (void *)resource, (void *)resource->resource,
      reason ? reason : "unknown");
}

static bool
ConstantBufferPublicationUpdateHasCompleteContents(
   Resource *resource, unsigned update_offset, unsigned update_size)
{
   if (!resource || !resource->resource ||
       update_offset > resource->resource->width0 ||
       update_size > resource->resource->width0 - update_offset)
      return false;

   const unsigned size = resource->resource->width0;
   return (update_offset == 0 && update_size == size) ||
          (resource->constant_published_cpu &&
           resource->constant_published_size >= size) ||
          (resource->constant_shadow_valid && resource->buffer_shadow &&
           resource->buffer_shadow_size >= size);
}

static bool
PublishConstantBufferUpdate(Device *device, Resource *resource,
                            unsigned update_offset, unsigned update_size,
                            const void *data)
{
   if (!ConstantBufferPublicationEligible(device, resource) || !data ||
       !ConstantBufferPublicationUpdateHasCompleteContents(
          resource, update_offset, update_size))
      return false;

   const unsigned size = resource->resource->width0;
   const unsigned alignment =
      MAX2(64u, device->screen->caps.constant_buffer_offset_alignment);
   unsigned published_offset = 0;
   struct pipe_resource *published = NULL;
   struct pipe_resource *uploader_release = NULL;
   void *published_cpu = NULL;

   u_upload_alloc(device->pipe->const_uploader, 0, size, alignment,
                  &published_offset, &published, &uploader_release,
                  &published_cpu);
   if (!published || !published_cpu) {
      ReleasePipeResourceOrdered(device->pipe, uploader_release);
      return false;
   }

   uint8_t *dst = (uint8_t *)published_cpu;
   const uint8_t *old =
      resource->constant_published_size >= size ?
         (const uint8_t *)resource->constant_published_cpu : NULL;
   const uint8_t *shadow =
      resource->constant_shadow_valid &&
      resource->buffer_shadow_size >= size ?
         (const uint8_t *)resource->buffer_shadow : NULL;

   if (!update_offset) {
      if (update_size == size)
         CopyConstantPublication(dst, data, update_size);
      else
         memcpy(dst, data, update_size);
      if (update_size < size) {
         if (old)
            memcpy(dst + update_size, old + update_size, size - update_size);
         else {
            assert(shadow);
            memcpy(dst + update_size, shadow + update_size,
                   size - update_size);
         }
      }
   } else {
      if (old)
         memcpy(dst, old, size);
      else {
         assert(shadow);
         memcpy(dst, shadow, size);
      }
      memcpy(dst + update_offset, data, update_size);
   }

   if (published != resource->constant_published_buffer) {
      struct pipe_resource *held = NULL;
      struct pipe_resource *old_resource =
         resource->constant_published_buffer;
      pipe_resource_reference(&held, published);
      resource->constant_published_buffer = held;
      ReleasePipeResourceOrdered(device->pipe, old_resource);
   }
   ReleasePipeResourceOrdered(device->pipe, uploader_release);

   resource->constant_published_cpu = published_cpu;
   resource->constant_published_offset = published_offset;
   resource->constant_published_size = size;
   resource->constant_shadow_valid = false;
   resource->constant_original_stale = true;
   RebindConstantBufferPublication(device, resource, true);
   return true;
}

static bool
RestoreConstantBufferOriginal(Device *device, Resource *resource,
                              const char *reason)
{
   if (!resource || !resource->constant_original_stale)
      return true;

   const unsigned size = resource->resource ? resource->resource->width0 : 0;
   if (!device || !device->pipe || !device->pipe->buffer_subdata || !size ||
       !resource->constant_published_cpu ||
       resource->constant_published_size < size) {
      WarnConstantBufferPublicationFallback(device, resource, reason);
      return false;
   }

   device->pipe->buffer_subdata(
      device->pipe, resource->resource, PIPE_MAP_DISCARD_WHOLE_RESOURCE,
      0, size, resource->constant_published_cpu);
   resource->constant_original_stale = false;
   return true;
}

static bool
SuspendConstantBufferPublication(Device *device, Resource *resource,
                                 bool preserve_contents,
                                 const char *reason)
{
   if (!ConstantBufferPublicationCandidate(device, resource))
      return true;

   if (preserve_contents &&
       !RestoreConstantBufferOriginal(device, resource, reason))
      return false;

   RebindConstantBufferPublication(device, resource, false);
   ReleasePublishedConstantBuffer(device->pipe, resource);
   resource->constant_original_stale = false;
   resource->constant_shadow_valid = false;
   return true;
}

bool
PreparePublishedConstantBuffer(Device *device, Resource *resource,
                               unsigned resource_offset,
                               unsigned buffer_size,
                               struct pipe_constant_buffer *cb)
{
   if (!ConstantBufferPublicationEligible(device, resource))
      return false;

   if (!resource->constant_published_buffer &&
       resource->constant_shadow_valid && resource->buffer_shadow &&
       resource->buffer_shadow_size >= resource->resource->width0 &&
       !PublishConstantBufferUpdate(
          device, resource, 0, resource->resource->width0,
          resource->buffer_shadow)) {
      WarnConstantBufferPublicationFallback(
         device, resource, "immutable uploader allocation failed");
      return false;
   }

   return FillPublishedConstantBuffer(device, resource, resource_offset,
                                      buffer_size, cb);
}

static void
ReleaseResourceContents(struct pipe_context *pipe, Resource *resource)
{
   if (!resource)
      return;

   if (resource->so_target)
      pipe_so_target_reference(&resource->so_target, NULL);

   if (resource->transfers) {
      for (UINT SubResource = 0; SubResource < resource->NumSubResources;
           ++SubResource) {
         if (resource->transfers[SubResource]) {
            if (resource->buffer) {
               pipe_buffer_unmap(pipe, resource->transfers[SubResource]);
            } else {
               pipe_texture_unmap(pipe, resource->transfers[SubResource]);
            }
            resource->transfers[SubResource] = NULL;
         }
      }
      free(resource->transfers);
      resource->transfers = NULL;
   }

   resource->constant_shadow_valid = false;
   resource->constant_original_stale = false;
   ReleasePublishedConstantBuffer(pipe, resource);
   pipe_resource_reference(&resource->resource, NULL);
   free(resource->buffer_shadow);
   resource->buffer_shadow = NULL;
   resource->buffer_shadow_size = 0;
}

static void
ReleaseResourceBindings(Device *device, Resource *resource)
{
   if (!device || !resource || !resource->resource)
      return;

   struct pipe_resource *pipe_resource = resource->resource;
   bool vertex_buffers_changed = false;

   /*
    * The D3D runtime owns the Resource object lifetime.  Our IA and constant
    * buffer shadows must not keep the Gallium resource (and its allocation)
    * alive after DestroyResource returns.  In particular, device teardown
    * invalidates device-owned allocation handles before DestroyDevice calls
    * us, so releasing a stale shadow reference there makes pfnDeallocateCb
    * reject the otherwise valid allocation with E_INVALIDARG.
    */
   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; i++) {
      struct pipe_vertex_buffer *vb = &device->vertex_buffers[i];
      if (vb->is_user_buffer || vb->buffer.resource != pipe_resource)
         continue;

      pipe_resource_reference(&vb->buffer.resource, NULL);
      device->vertex_strides[i] = 0;
      vb->buffer_offset = 0;
      vb->is_user_buffer = true;
      vb->buffer.user = NULL;
      vertex_buffers_changed = true;
   }

   if (vertex_buffers_changed) {
      device->velems_changed = true;
      device->vbuffers_changed = true;
   }

   if (device->index_buffer == pipe_resource) {
      pipe_resource_reference(&device->index_buffer, NULL);
      device->ib_offset = 0;
      device->index_size = 0;
      device->restart_index = 0;
   }

   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; stage++) {
      for (unsigned slot = 0; slot < PIPE_MAX_CONSTANT_BUFFERS; slot++) {
         if (device->constant_buffers[stage][slot] == pipe_resource)
            pipe_resource_reference(
               &device->constant_buffers[stage][slot], NULL);
         if (device->constant_buffer_resources[stage][slot] == resource)
            device->constant_buffer_resources[stage][slot] = NULL;
         if (device->constant_buffer_resources[stage][slot] == NULL) {
            device->constant_buffer_offsets[stage][slot] = 0;
            device->constant_buffer_sizes[stage][slot] = 0;
            device->constant_buffer_published[stage][slot] = false;
         }
         if (device->constant_buffer_binding_resources[stage][slot] ==
             resource) {
            device->constant_buffer_binding_resources[stage][slot] = NULL;
            device->constant_buffer_offsets[stage][slot] = 0;
            device->constant_buffer_sizes[stage][slot] = 0;
            device->constant_buffer_published[stage][slot] = false;
         }
      }
      resource->constant_buffer_bindings[stage] = 0;
   }
}

void
DestroyDeviceResourceDiagnostics(D3D10DDI_HDEVICE hDevice)
{
   Device *device = CastDevice(hDevice);

   if (!device)
      return;

   unsigned count = 0;
   list_for_each_entry(Resource, resource, &device->resources, list) {
      ResourceEvent(RESOURCE_EVENT_DEVICE_LIVE_RESOURCE,
                    (uint64_t)(uintptr_t)device,
                    resource,
                    resource->resource,
                    resource->resource ?
                       p_atomic_read(&resource->resource->reference.count) : 0,
                    count,
                    resource->buffer ? 1 : 0,
                    resource->NumSubResources);
      count++;
   }

   ResourceEvent(RESOURCE_EVENT_DEVICE_LIVE_COUNT,
                 (uint64_t)(uintptr_t)device,
                 NULL, NULL, 0, count, 0, 0);
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateResourceSize --
 *
 *    The CalcPrivateResourceSize function determines the size of
 *    the user-mode display driver's private region of memory
 *    (that is, the size of internal driver structures, not the
 *    size of the resource video memory).
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateResourceSize(D3D10DDI_HDEVICE hDevice,                                // IN
                        __in const D3D10DDIARG_CREATERESOURCE *pCreateResource)  // IN
{
   LOG_ENTRYPOINT();
   return sizeof(Resource);
}

SIZE_T APIENTRY
CalcPrivateResourceSize11(D3D10DDI_HDEVICE hDevice,
                          __in const D3D11DDIARG_CREATERESOURCE *pCreateResource)
{
   LOG_ENTRYPOINT();
   return sizeof(Resource);
}

static UINT
sanitize_d3d11_resource_bind_flags(UINT flags)
{
   return flags;
}

static UINT
sanitize_d3d11_resource_misc_flags(UINT flags)
{
   const UINT d3d11_shared_keyedmutex = 0x00000100;
   const UINT d3d11_shared_nthandle = 0x00000800;
   const UINT supported =
      D3D10_DDI_RESOURCE_MISC_SHARED |
      D3D10_DDI_RESOURCE_MISC_DISCARD_ON_PRESENT |
      D3D10_DDI_RESOURCE_MISC_REMOTE |
      D3D11_DDI_RESOURCE_MISC_DRAWINDIRECT_ARGS;

   UINT sanitized = flags & supported;
   if (flags & (d3d11_shared_keyedmutex | d3d11_shared_nthandle))
      sanitized |= D3D10_DDI_RESOURCE_MISC_SHARED;

   return sanitized;
}

static UINT
translate_d3d11_resource_array_size(const D3D11DDIARG_CREATERESOURCE *create)
{
   if (create->ResourceDimension == D3D10DDIRESOURCE_TEXTURECUBE) {
      if (create->ArraySize < 6)
         return create->ArraySize * 6;
      return ((create->ArraySize + 5) / 6) * 6;
   }

   return create->ArraySize;
}


static pipe_resource_usage
translate_resource_usage(UINT usage)
{
   pipe_resource_usage resource_usage = PIPE_USAGE_DEFAULT;

   switch (usage) {
   case D3D10_DDI_USAGE_DEFAULT:
      resource_usage = PIPE_USAGE_DEFAULT;
      break;
   case D3D10_DDI_USAGE_IMMUTABLE:
      resource_usage = PIPE_USAGE_IMMUTABLE;
      break;
   case D3D10_DDI_USAGE_DYNAMIC:
      resource_usage = PIPE_USAGE_DYNAMIC;
      break;
   case D3D10_DDI_USAGE_STAGING:
      resource_usage = PIPE_USAGE_STAGING;
      break;
   default:
      assert(0);
      break;
   }

   return resource_usage;
}


static unsigned
translate_resource_flags(UINT flags)
{
   unsigned bind = 0;

   if (flags & D3D10_DDI_BIND_VERTEX_BUFFER)
      bind |= PIPE_BIND_VERTEX_BUFFER;

   if (flags & D3D10_DDI_BIND_INDEX_BUFFER)
      bind |= PIPE_BIND_INDEX_BUFFER;

   if (flags & D3D10_DDI_BIND_CONSTANT_BUFFER)
      bind |= PIPE_BIND_CONSTANT_BUFFER;

   if (flags & D3D10_DDI_BIND_SHADER_RESOURCE)
      bind |= PIPE_BIND_SAMPLER_VIEW;

   if (flags & D3D10_DDI_BIND_RENDER_TARGET)
      bind |= PIPE_BIND_RENDER_TARGET;

   if (flags & D3D10_DDI_BIND_DEPTH_STENCIL)
      bind |= PIPE_BIND_DEPTH_STENCIL;

   if (flags & D3D10_DDI_BIND_STREAM_OUTPUT)
      bind |= PIPE_BIND_STREAM_OUTPUT;

   if (flags & D3D10_DDI_BIND_PRESENT)
      bind |= PIPE_BIND_DISPLAY_TARGET;

   return bind;
}


static enum pipe_texture_target
translate_texture_target( D3D10DDIRESOURCE_TYPE ResourceDimension,
                             UINT ArraySize)
{
   assert(ArraySize >= 1);
   switch(ResourceDimension) {
   case D3D10DDIRESOURCE_BUFFER:
      assert(ArraySize == 1);
      return PIPE_BUFFER;
   case D3D10DDIRESOURCE_TEXTURE1D:
      return ArraySize > 1 ? PIPE_TEXTURE_1D_ARRAY : PIPE_TEXTURE_1D;
   case D3D10DDIRESOURCE_TEXTURE2D:
      return ArraySize > 1 ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D;
   case D3D10DDIRESOURCE_TEXTURE3D:
      assert(ArraySize == 1);
      return PIPE_TEXTURE_3D;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      assert(ArraySize % 6 == 0);
      return ArraySize > 6 ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_CUBE;
   default:
      assert(0);
      return PIPE_TEXTURE_1D;
   }
}

static bool
validate_resource_dimension(const D3D10DDIARG_CREATERESOURCE *create)
{
   if (!create->ArraySize)
      return false;

   switch (create->ResourceDimension) {
   case D3D10DDIRESOURCE_BUFFER:
   case D3D10DDIRESOURCE_TEXTURE3D:
      return create->ArraySize == 1;
   case D3D10DDIRESOURCE_TEXTURE1D:
   case D3D10DDIRESOURCE_TEXTURE2D:
      return true;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      return create->ArraySize % 6 == 0;
   default:
      return false;
   }
}


static bool
is_resource_format_supported(struct pipe_screen *screen,
                             const struct pipe_resource *templat)
{
   return screen->is_format_supported(screen,
                                      templat->format,
                                      templat->target,
                                      templat->nr_samples,
                                      templat->nr_storage_samples,
                                      templat->bind);
}

static bool
is_yttrium_screen(struct pipe_screen *screen)
{
   if (!screen || !screen->get_name)
      return false;

   const char *name = screen->get_name(screen);
   return name && strcmp(name, "yttrium") == 0;
}

static bool
try_resource_format_fallback(struct pipe_screen *screen,
                             struct pipe_resource *templat)
{
   enum pipe_format fallback = FormatFallback(templat->format);
   if (fallback == PIPE_FORMAT_NONE)
      return false;

   enum pipe_format original = templat->format;
   templat->format = fallback;

   if (is_resource_format_supported(screen, templat))
      return true;

   templat->format = original;
   return false;
}


static void
subResourceBox(struct pipe_resource *resource, // IN
                 UINT SubResource,  // IN
                 unsigned *pLevel, // OUT
                 struct pipe_box *pBox)   // OUT
{
   UINT MipLevels = resource->last_level + 1;
   unsigned layer;
   unsigned width;
   unsigned height;
   unsigned depth;

   *pLevel = SubResource % MipLevels;
   layer = SubResource / MipLevels;

   width  = u_minify(resource->width0,  *pLevel);
   height = u_minify(resource->height0, *pLevel);
   depth  = u_minify(resource->depth0,  *pLevel);

   pBox->x = 0;
   pBox->y = 0;
   pBox->z = 0 + layer;
   pBox->width  = width;
   pBox->height = height;
   pBox->depth  = depth;
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateResource --
 *
 *    The CreateResource function creates a resource.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateResource(D3D10DDI_HDEVICE hDevice,                                // IN
               __in const D3D10DDIARG_CREATERESOURCE *pCreateResource,  // IN
               D3D10DDI_HRESOURCE hResource,                            // IN
               D3D10DDI_HRTRESOURCE hRTResource)                        // IN
{
   LOG_ENTRYPOINT();

   Device *device = CastDevice(hDevice);
   struct pipe_context *pipe = CastPipeContext(hDevice);
   struct pipe_screen *screen = pipe->screen;
   const bool yttrium = is_yttrium_screen(screen);
   const bool has_primary_desc = pCreateResource->pPrimaryDesc != NULL;

   if (!validate_resource_dimension(pCreateResource)) {
      SetError(hDevice, DXGI_DDI_ERR_UNSUPPORTED);
      return;
   }

   if ((pCreateResource->MiscFlags & D3D10_DDI_RESOURCE_MISC_SHARED) ||
       has_primary_desc) {

      DebugPrintf("%s(%dx%dx%d hResource=%p)\n",
	       __func__,
	       pCreateResource->pMipInfoList[0].TexelWidth,
	       pCreateResource->pMipInfoList[0].TexelHeight,
	       pCreateResource->pMipInfoList[0].TexelDepth,
	       hResource.pDrvPrivate);
      DebugPrintf("  ResourceDimension = %u\n",
	       pCreateResource->ResourceDimension);
      DebugPrintf("  Usage = %u\n",
	       pCreateResource->Usage);
      DebugPrintf("  BindFlags = 0x%x\n",
	       pCreateResource->BindFlags);
      DebugPrintf("  MapFlags = 0x%x\n",
	       pCreateResource->MapFlags);
      DebugPrintf("  MiscFlags = 0x%x\n",
	       pCreateResource->MiscFlags);
      DebugPrintf("  Format = %s\n",
	       FormatToName(pCreateResource->Format));
      DebugPrintf("  SampleDesc.Count = %u\n", pCreateResource->SampleDesc.Count);
      DebugPrintf("  SampleDesc.Quality = %u\n", pCreateResource->SampleDesc.Quality);
      DebugPrintf("  MipLevels = %u\n", pCreateResource->MipLevels);
      DebugPrintf("  ArraySize = %u\n", pCreateResource->ArraySize);
      DebugPrintf("  pPrimaryDesc = %p\n", pCreateResource->pPrimaryDesc);
      if (pCreateResource->pPrimaryDesc) {
	 DebugPrintf("    Flags = 0x%x\n",
		  pCreateResource->pPrimaryDesc->Flags);
	 DebugPrintf("    VidPnSourceId = %u\n", pCreateResource->pPrimaryDesc->VidPnSourceId);
	 DebugPrintf("    ModeDesc.Width = %u\n", pCreateResource->pPrimaryDesc->ModeDesc.Width);
	 DebugPrintf("    ModeDesc.Height = %u\n", pCreateResource->pPrimaryDesc->ModeDesc.Height);
	 DebugPrintf("    ModeDesc.Format = %u)\n",
		  pCreateResource->pPrimaryDesc->ModeDesc.Format);
	 DebugPrintf("    ModeDesc.RefreshRate.Numerator = %u\n", pCreateResource->pPrimaryDesc->ModeDesc.RefreshRate.Numerator);
	 DebugPrintf("    ModeDesc.RefreshRate.Denominator = %u\n", pCreateResource->pPrimaryDesc->ModeDesc.RefreshRate.Denominator);
	 DebugPrintf("    ModeDesc.ScanlineOrdering = %u\n",
		  pCreateResource->pPrimaryDesc->ModeDesc.ScanlineOrdering);
	 DebugPrintf("    ModeDesc.Rotation = %u\n",
		  pCreateResource->pPrimaryDesc->ModeDesc.Rotation);
	 DebugPrintf("    ModeDesc.Scaling = %u\n",
		  pCreateResource->pPrimaryDesc->ModeDesc.Scaling);
	 DebugPrintf("    DriverFlags = 0x%x\n",
		  pCreateResource->pPrimaryDesc->DriverFlags);
      }

   }

   if (yttrium && has_primary_desc) {
      pCreateResource->pPrimaryDesc->DriverFlags &=
         ~DXGI_DDI_PRIMARY_DRIVER_FLAG_NO_SCANOUT;
   }

   mtx_lock(&device->CreateResourceMtx);
   device->device.allocationVidPn =
      (yttrium && has_primary_desc) ? pCreateResource->pPrimaryDesc->VidPnSourceId : 0;
   device->device.isPrimary = pCreateResource->pPrimaryDesc != NULL;
   device->device.hRTResource = hRTResource.handle;
   device->device.hRTResourceIsD3D9 = false;

   Resource *pResource = CastResource(hResource);

   memset(pResource, 0, sizeof *pResource);
   pResource->yttrium_primary = yttrium && has_primary_desc;

#if 0
   if (pCreateResource->pPrimaryDesc) {
      pCreateResource->pPrimaryDesc->DriverFlags = DXGI_DDI_PRIMARY_DRIVER_FLAG_NO_SCANOUT;
      if (!(pCreateResource->pPrimaryDesc->DriverFlags & DXGI_DDI_PRIMARY_OPTIONAL)) {
         // http://msdn.microsoft.com/en-us/library/windows/hardware/ff568846.aspx
         SetError(hDevice, DXGI_DDI_ERR_UNSUPPORTED);
         return;
      }
   }
#endif

   pResource->Format = pCreateResource->Format;
   pResource->MiscFlags = pCreateResource->MiscFlags;
   pResource->ByteStride = 0;
   pResource->MipLevels = pCreateResource->MipLevels;
   pResource->SampleCount = pCreateResource->SampleDesc.Count;

   struct pipe_resource templat;

   memset(&templat, 0, sizeof templat);

   templat.target     = translate_texture_target( pCreateResource->ResourceDimension,
                                                  pCreateResource->ArraySize );
   pResource->buffer = templat.target == PIPE_BUFFER;

   if (pCreateResource->Format == DXGI_FORMAT_UNKNOWN) {
      assert(pCreateResource->ResourceDimension == D3D10DDIRESOURCE_BUFFER);
      templat.format = PIPE_FORMAT_R8_UINT;
   } else {
      BOOL bindDepthStencil = !!(pCreateResource->BindFlags & D3D10_DDI_BIND_DEPTH_STENCIL);
      templat.format = FormatTranslate(pCreateResource->Format, bindDepthStencil);
   }

   templat.width0     = pCreateResource->pMipInfoList[0].TexelWidth;
   templat.height0    = pCreateResource->pMipInfoList[0].TexelHeight;
   templat.depth0     = pCreateResource->pMipInfoList[0].TexelDepth;
   templat.array_size = pCreateResource->ArraySize;
   templat.last_level = pCreateResource->MipLevels - 1;
   templat.nr_samples = pCreateResource->SampleDesc.Count;
   templat.nr_storage_samples = pCreateResource->SampleDesc.Count;
   templat.bind       = translate_resource_flags(pCreateResource->BindFlags);
#if SUPPORT_D3D11
   if (pCreateResource->MiscFlags & D3D11_DDI_RESOURCE_MISC_DRAWINDIRECT_ARGS)
      templat.bind |= PIPE_BIND_COMMAND_ARGS_BUFFER;
   if (pCreateResource->BindFlags & D3D11_DDI_BIND_UNORDERED_ACCESS) {
      /* UAV accesses lower to TGSI images, including buffer UAVs.  Native
       * argument/VB/IB allocations must therefore support texel-buffer views
       * from creation, not only the shader-buffer binding below.
       */
      templat.bind |= PIPE_BIND_SHADER_IMAGE;
      if (pCreateResource->ResourceDimension == D3D10DDIRESOURCE_BUFFER)
         templat.bind |= PIPE_BIND_SHADER_BUFFER;
   }
#endif
   templat.usage      = translate_resource_usage(pCreateResource->Usage);
   if (yttrium &&
       (pCreateResource->BindFlags & D3D10_DDI_BIND_PRESENT) &&
       !(pCreateResource->MiscFlags & D3D10_DDI_RESOURCE_MISC_SHARED) &&
       !has_primary_desc)
      templat.flags |= YTTRIUM_GDI_RESOURCE_FLAG_CPU_READBACK;
   if (yttrium && has_primary_desc)
      templat.flags |= PIPE_RESOURCE_FLAG_FRONTEND_PRIV;

   if (templat.bind == 0 && templat.usage != PIPE_USAGE_STAGING) {
      templat.bind = PIPE_BIND_SAMPLER_VIEW;
   }
   if ((pCreateResource->MiscFlags & D3D10_DDI_RESOURCE_MISC_SHARED)) {
      templat.bind |= PIPE_BIND_SHARED;
   }

   if (templat.target != PIPE_BUFFER) {
      if (!is_resource_format_supported(screen, &templat) &&
          !try_resource_format_fallback(screen, &templat)) {
         debug_printf("%s: unsupported format %s\n",
                     __func__, util_format_name(templat.format));
         SetError(hDevice, E_OUTOFMEMORY);
         goto unlock;
      }
   }

   pResource->resource = screen->resource_create(screen, &templat);
   if (!pResource->resource) {
      DebugPrintf("%s: failed to create resource\n", __func__);
      SetError(hDevice, E_OUTOFMEMORY);
      goto unlock;
   }
   ResourceEvent(RESOURCE_EVENT_CREATE,
                 (uint64_t)(uintptr_t)hResource.pDrvPrivate,
                 (const void *)(uintptr_t)hRTResource.handle,
                 pResource->resource,
                 p_atomic_read(&pResource->resource->reference.count),
                 templat.bind,
                 pCreateResource->BindFlags,
                 pCreateResource->MiscFlags);

   pResource->NumSubResources = pCreateResource->MipLevels * pCreateResource->ArraySize;
   pResource->transfers = (struct pipe_transfer **)calloc(pResource->NumSubResources,
                                                          sizeof *pResource->transfers);
   RegisterResource(device, pResource, "create", hResource, hRTResource);

   if (pCreateResource->pInitialDataUP) {
      if (pResource->buffer) {
         assert(pResource->NumSubResources == 1);
         const D3D10_DDIARG_SUBRESOURCE_UP* pInitialDataUP =
               &pCreateResource->pInitialDataUP[0];

         unsigned level;
         struct pipe_box box;
         subResourceBox(pResource->resource, 0, &level, &box);

         struct pipe_transfer *transfer;
         void *map;
         map = pipe->buffer_map(pipe,
                                pResource->resource,
                                level,
                                PIPE_MAP_WRITE |
                                PIPE_MAP_UNSYNCHRONIZED,
                                &box,
                                &transfer);
         assert(map);
         if (map) {
            memcpy(map, pInitialDataUP->pSysMem, box.width);
            UpdateBufferShadow(pResource, box.x, box.width,
                               pInitialDataUP->pSysMem);
            pResource->constant_shadow_valid =
               box.x == 0 && box.width == pResource->resource->width0;
            pipe_buffer_unmap(pipe, transfer);
         }
      } else {
         for (UINT SubResource = 0; SubResource < pResource->NumSubResources; ++SubResource) {
            const D3D10_DDIARG_SUBRESOURCE_UP* pInitialDataUP =
                  &pCreateResource->pInitialDataUP[SubResource];

            unsigned level;
            struct pipe_box box;
            subResourceBox(pResource->resource, SubResource, &level, &box);

            struct pipe_transfer *transfer;
            void *map;
            map = pipe->texture_map(pipe,
                                    pResource->resource,
                                    level,
                                    PIPE_MAP_WRITE |
                                    PIPE_MAP_UNSYNCHRONIZED,
                                    &box,
                                    &transfer);
            assert(map);
            if (map) {
               for (int z = 0; z < box.depth; ++z) {
                  uint8_t *dst = (uint8_t*)map + z*transfer->layer_stride;
                  const uint8_t *src = (const uint8_t*)pInitialDataUP->pSysMem + z*pInitialDataUP->SysMemSlicePitch;
                  util_copy_rect(dst,
                                 templat.format,
                                 transfer->stride,
                                 0, 0, box.width, box.height,
                                 src,
                                 pInitialDataUP->SysMemPitch,
                                 0, 0);
               }
               pipe_texture_unmap(pipe, transfer);
            }
         }
      }
   }

unlock:
   device->device.allocationVidPn = 0;
   device->device.isPrimary = false;
   device->device.hRTResource = NULL;
   device->device.hRTResourceIsD3D9 = false;
   mtx_unlock(&device->CreateResourceMtx);
}

void APIENTRY
CreateResource11(D3D10DDI_HDEVICE hDevice,
                 __in const D3D11DDIARG_CREATERESOURCE *pCreateResource,
                 D3D10DDI_HRESOURCE hResource,
                 D3D10DDI_HRTRESOURCE hRTResource)
{
   D3D10DDIARG_CREATERESOURCE create10 = {};

   create10.pMipInfoList = pCreateResource->pMipInfoList;
   create10.pInitialDataUP = pCreateResource->pInitialDataUP;
   create10.ResourceDimension = pCreateResource->ResourceDimension;
   create10.Usage = pCreateResource->Usage;
   create10.BindFlags =
      sanitize_d3d11_resource_bind_flags(pCreateResource->BindFlags);
   create10.MapFlags = pCreateResource->MapFlags;
   create10.MiscFlags =
      sanitize_d3d11_resource_misc_flags(pCreateResource->MiscFlags);
   create10.Format = pCreateResource->Format;
   create10.SampleDesc = pCreateResource->SampleDesc;
   create10.MipLevels = pCreateResource->MipLevels;
   create10.ArraySize = translate_d3d11_resource_array_size(pCreateResource);
   create10.pPrimaryDesc = pCreateResource->pPrimaryDesc;

   CreateResource(hDevice, &create10, hResource, hRTResource);

   Resource *resource = CastResource(hResource);
   if (resource) {
      resource->MiscFlags = pCreateResource->MiscFlags;
      resource->ByteStride = pCreateResource->ByteStride;
   }
}

void APIENTRY
SetResourceMinLOD(D3D10DDI_HDEVICE hDevice,
                  D3D10DDI_HRESOURCE hResource,
                  FLOAT MinLOD)
{
   LOG_ENTRYPOINT();
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateOpenedResourceSize --
 *
 *    The CalcPrivateOpenedResourceSize function determines the size
 *    of the user-mode display driver's private shared region of memory
 *    (that is, the size of internal driver structures, not the size
 *    of the resource video memory) for an opened resource.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateOpenedResourceSize(D3D10DDI_HDEVICE hDevice,                             // IN
                              __in const D3D10DDIARG_OPENRESOURCE *pOpenResource)   // IN
{
   return sizeof(Resource);
}


/*
 * ----------------------------------------------------------------------
 *
 * OpenResource --
 *
 *    The OpenResource function opens a shared resource.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
OpenResource(D3D10DDI_HDEVICE hDevice,                            // IN
             __in const D3D10DDIARG_OPENRESOURCE *pOpenResource,  // IN
             D3D10DDI_HRESOURCE hResource,                        // IN
             D3D10DDI_HRTRESOURCE hRTResource)                    // IN
{
   LOG_ENTRYPOINT();
   
   Device *pDevice = CastDevice(hDevice);
   if (!pDevice->screen->resource_from_handle) {
      SetError(hDevice, E_OUTOFMEMORY);
      return;
   }
   
   mtx_lock(&pDevice->CreateResourceMtx);
   pDevice->device.hRTResource = hRTResource.handle;
   pDevice->device.hRTResourceIsD3D9 = false;
   pDevice->device.pOpenResource = pOpenResource;
   
   Resource *pResource = CastResource(hResource);
   memset(pResource, 0, sizeof *pResource);
   
   struct winsys_handle whandle;
   memset(&whandle, 0, sizeof(whandle));
   whandle.type = WINSYS_HANDLE_TYPE_WIN32_HANDLE;
   whandle.handle = (HANDLE)(uintptr_t)pOpenResource->hKMResource.handle;
   
   pResource->resource =
      pDevice->screen->resource_from_handle(pDevice->screen, NULL, &whandle, 0);
   
   if (!pResource->resource) {
      SetError(hDevice, E_OUTOFMEMORY);
      goto unlock;
   }
   
   pResource->buffer = false;
   pResource->Format = FormatDeTranslate(pResource->resource->format);
   pResource->MipLevels = pResource->resource->last_level + 1;
   pResource->NumSubResources =
      pResource->MipLevels * pResource->resource->array_size;
   pResource->transfers = (struct pipe_transfer **)calloc(
      pResource->NumSubResources, sizeof *pResource->transfers);
   ResourceEvent(RESOURCE_EVENT_OPEN,
                 (uint64_t)(uintptr_t)hResource.pDrvPrivate,
                 (const void *)(uintptr_t)hRTResource.handle,
                 pResource->resource,
                 p_atomic_read(&pResource->resource->reference.count),
                 pResource->resource->bind,
                 pResource->resource->flags,
                 (uint64_t)pOpenResource->hKMResource.handle);
   RegisterResource(pDevice, pResource, "open", hResource, hRTResource);

unlock:
   pDevice->device.pOpenResource = NULL;
   pDevice->device.hRTResource = NULL;
   pDevice->device.hRTResourceIsD3D9 = false;
   mtx_unlock(&pDevice->CreateResourceMtx);
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyResource --
 *
 *    The DestroyResource function destroys the specified resource
 *    object. The resource object can be destoyed only if it is not
 *    currently bound to a display device, and if all views that
 *    refer to the resource are also destroyed.
 *
 * ----------------------------------------------------------------------
 */


void APIENTRY
DestroyResource(D3D10DDI_HDEVICE hDevice,       // IN
                D3D10DDI_HRESOURCE hResource)   // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   Device *device = CastDevice(hDevice);
   Resource *pResource = CastResource(hResource);
   struct pipe_resource *resource = pResource ? pResource->resource : NULL;

   ResourceEvent(RESOURCE_EVENT_DESTROY_BEGIN,
                 (uint64_t)(uintptr_t)hResource.pDrvPrivate,
                 pResource,
                 resource,
                 resource ? p_atomic_read(&resource->reference.count) : 0,
                 pResource && pResource->buffer ? 1 : 0,
                 pResource ? pResource->NumSubResources : 0,
                 0);

   UnregisterResource(device, pResource, "destroy", hResource);
   ReleaseResourceBindings(device, pResource);

   /*
    * Resource-associated allocations are deallocated through the runtime
    * resource handle.  Submit pending work and retire only batches which
    * reference this resource before its final frontend reference is released.
    * Cached pipelines also own render-target references, so evict matching
    * entries at this explicit API lifetime edge before releasing the
    * frontend's final reference.
    */
   const bool runtime_allocation =
      yttrium_gdi_resource_has_runtime_allocation(resource);
   if (pipe && pipe->flush_resource && resource) {
      pipe->flush_resource(pipe, resource);
      /*
       * threaded_context queues flush_resource and retains the resource.
       * Synchronize this explicit runtime lifetime edge so the
       * resource-specific flush runs before cache eviction and the frontend
       * drops its final reference.  Standalone allocations stay fully
       * asynchronous.
       */
      if (runtime_allocation) {
         struct pipe_context *driver_pipe =
            threaded_context_unwrap_sync(pipe);
         yttrium_gdi_pipeline_invalidate_resource(driver_pipe, resource);
      }
   }

   ReleaseResourceContents(pipe, pResource);
   ResourceEvent(RESOURCE_EVENT_DESTROY_END,
                 (uint64_t)(uintptr_t)hResource.pDrvPrivate,
                 NULL, resource, 0, 0, 0, 0);
}


/*
 * ----------------------------------------------------------------------
 *
 * ResourceMap --
 *
 *    The ResourceMap function maps a subresource of a resource.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ResourceMap(D3D10DDI_HDEVICE hDevice,                                // IN
            D3D10DDI_HRESOURCE hResource,                            // IN
            UINT SubResource,                                        // IN
            D3D10_DDI_MAP DDIMap,                                    // IN
            UINT Flags,                                              // IN
            __out D3D10DDI_MAPPED_SUBRESOURCE *pMappedSubResource)   // OUT
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Resource *pResource = CastResource(hResource);
   struct pipe_resource *resource = pResource->resource;

   unsigned usage;
   bool preserve_constant_buffer_contents = true;
   switch (DDIMap) {
   case D3D10_DDI_MAP_READ:
      usage = PIPE_MAP_READ;
      break;
   case D3D10_DDI_MAP_READWRITE:
      usage = PIPE_MAP_READ | PIPE_MAP_WRITE;
      break;
   case D3D10_DDI_MAP_WRITE:
      usage = PIPE_MAP_WRITE;
      break;
   case D3D10_DDI_MAP_WRITE_DISCARD:
      usage = PIPE_MAP_WRITE;
      preserve_constant_buffer_contents = false;
      if (resource->last_level == 0 && resource->array_size == 1) {
         usage |= PIPE_MAP_DISCARD_WHOLE_RESOURCE;
      } else {
         usage |= PIPE_MAP_DISCARD_RANGE;
      }
      break;
   case D3D10_DDI_MAP_WRITE_NOOVERWRITE:
      usage = PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED;
      break;
   default:
      assert(0);
      return;
   }

   if (DDIMap == D3D10_DDI_MAP_READ) {
      if (!RestoreConstantBufferOriginal(
             pDevice, pResource,
             "CPU read map requires original resource contents")) {
         SetError(hDevice, E_FAIL);
         return;
      }
   } else if (!SuspendConstantBufferPublication(
                 pDevice, pResource, preserve_constant_buffer_contents,
                 "CPU write map requires original resource contents")) {
      SetError(hDevice, E_FAIL);
      return;
   }

   assert(SubResource < pResource->NumSubResources);

   unsigned level;
   struct pipe_box box;
   subResourceBox(resource, SubResource, &level, &box);

   assert(!pResource->transfers[SubResource]);

   void *map;
   if (pResource->buffer) {
      map = pipe->buffer_map(pipe,
                             resource,
                             level,
                             usage,
                             &box,
                             &pResource->transfers[SubResource]);
   } else {
      map = pipe->texture_map(pipe,
                              resource,
                              level,
                              usage,
                              &box,
                              &pResource->transfers[SubResource]);
   }
   if (!map) {
      DebugPrintf("%s: failed to map resource\n", __func__);
      SetError(hDevice, E_FAIL);
      return;
   }

   pMappedSubResource->pData = map;
   pMappedSubResource->RowPitch = pResource->transfers[SubResource]->stride;
   pMappedSubResource->DepthPitch = pResource->transfers[SubResource]->layer_stride;
}


/*
 * ----------------------------------------------------------------------
 *
 * ResourceUnmap --
 *
 *    The ResourceUnmap function unmaps a subresource of a resource.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ResourceUnmap(D3D10DDI_HDEVICE hDevice,      // IN
              D3D10DDI_HRESOURCE hResource,  // IN
              UINT SubResource)              // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   Resource *pResource = CastResource(hResource);

   assert(SubResource < pResource->NumSubResources);

   if (pResource->transfers[SubResource]) {
      if (pResource->buffer) {
         pipe_buffer_unmap(pipe, pResource->transfers[SubResource]);
      } else {
         pipe_texture_unmap(pipe, pResource->transfers[SubResource]);
      }
      pResource->transfers[SubResource] = NULL;
   }
}


/*
 *----------------------------------------------------------------------
 *
 * areResourcesCompatible --
 *
 *      Check whether two resources can be safely passed to
 *      pipe_context::resource_copy_region method.
 *
 * Results:
 *      As above.
 *
 * Side effects:
 *      None.
 *
 *----------------------------------------------------------------------
 */

static bool
areResourcesCompatible(const struct pipe_resource *src_resource, // IN
                       const struct pipe_resource *dst_resource) // IN
{
   if (src_resource->format == dst_resource->format) {
      /*
       * Trivial.
       */

      return true;
   } else if (src_resource->target == PIPE_BUFFER &&
              dst_resource->target == PIPE_BUFFER) {
      /*
       * Buffer resources are merely a collection of bytes.
       */

      return true;
   } else {
      /*
       * Check whether the formats are supported by
       * the resource_copy_region method.
       */

      const struct util_format_description *src_format_desc;
      const struct util_format_description *dst_format_desc;

      src_format_desc = util_format_description(src_resource->format);
      dst_format_desc = util_format_description(dst_resource->format);

      if (!src_format_desc || !dst_format_desc ||
          src_format_desc->block.width  != dst_format_desc->block.width ||
          src_format_desc->block.height != dst_format_desc->block.height ||
          src_format_desc->block.bits   != dst_format_desc->block.bits)
         return false;

      return util_is_format_compatible(src_format_desc, dst_format_desc);
   }
}

static bool
areResourcesFallbackCopyCompatible(const struct pipe_resource *src_resource,
                                   const struct pipe_resource *dst_resource)
{
   const struct util_format_description *src_format_desc;
   const struct util_format_description *dst_format_desc;

   if (src_resource->target == PIPE_BUFFER &&
       dst_resource->target == PIPE_BUFFER)
      return true;

   if (src_resource->target == PIPE_BUFFER ||
       dst_resource->target == PIPE_BUFFER)
      return false;

   src_format_desc = util_format_description(src_resource->format);
   dst_format_desc = util_format_description(dst_resource->format);
   if (!src_format_desc || !dst_format_desc)
      return false;

   return src_format_desc->block.bits == dst_format_desc->block.bits;
}

static bool
areYttriumDepthReadbackFormatsCompatible(const struct pipe_resource *src_resource,
                                         const struct pipe_resource *dst_resource)
{
   return (src_resource->format == PIPE_FORMAT_Z16_UNORM &&
           dst_resource->format == PIPE_FORMAT_R16_UNORM) ||
          (src_resource->format == PIPE_FORMAT_Z32_FLOAT &&
           dst_resource->format == PIPE_FORMAT_R32_FLOAT);
}

/*
 * ----------------------------------------------------------------------
 *
 * ResourceCopy --
 *
 *    The ResourceCopy function copies an entire source
 *    resource to a destination resource.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ResourceCopy(D3D10DDI_HDEVICE hDevice,          // IN
             D3D10DDI_HRESOURCE hDstResource,   // IN
             D3D10DDI_HRESOURCE hSrcResource)   // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   if (!CheckPredicate(pDevice)) {
      return;
   }

   struct pipe_context *pipe = pDevice->pipe;
   Resource *pDstResource = CastResource(hDstResource);
   Resource *pSrcResource = CastResource(hSrcResource);

   if (!RestoreConstantBufferOriginal(
          pDevice, pSrcResource, "resource copy reads original storage") ||
       !SuspendConstantBufferPublication(
          pDevice, pDstResource, false,
          "resource copy writes original storage")) {
      SetError(hDevice, E_FAIL);
      return;
   }

   struct pipe_resource *dst_resource = pDstResource->resource;
   struct pipe_resource *src_resource = pSrcResource->resource;
   bool compatible;
   bool fallback_compatible;

   assert(dst_resource->target == src_resource->target);
   assert(dst_resource->last_level == src_resource->last_level);
   assert(dst_resource->array_size == src_resource->array_size);

   compatible = areResourcesCompatible(src_resource, dst_resource) ||
                (is_yttrium_screen(pipe->screen) &&
                 areYttriumDepthReadbackFormatsCompatible(src_resource,
                                                          dst_resource));
   fallback_compatible = !compatible &&
                         areResourcesFallbackCopyCompatible(src_resource,
                                                            dst_resource);

   if (compatible) {
      assert(dst_resource->width0 == src_resource->width0);
      assert(dst_resource->height0 == src_resource->height0);
      assert(dst_resource->depth0 == src_resource->depth0);
   } else if (!fallback_compatible) {
      return;
   }

   if (dst_resource->target == PIPE_BUFFER) {
      struct pipe_box box;
      box.x = 0;
      box.y = 0;
      box.z = 0;
      box.width = MIN2(dst_resource->width0, src_resource->width0);
      box.height = 1;
      box.depth = 1;

      if (box.width) {
         pipe->resource_copy_region(pipe,
                                    dst_resource, 0,
                                    0, 0, 0,
                                    src_resource, 0,
                                    &box);
      }
      return;
   }

   /* could also use one 3d copy for arrays */
   for (unsigned layer = 0; layer < dst_resource->array_size; ++layer) {
      for (unsigned level = 0; level <= dst_resource->last_level; ++level) {
         struct pipe_box box;
         box.x = 0;
         box.y = 0;
         box.z = 0 + layer;
         box.width  = u_minify(src_resource->width0,  level);
         box.height = u_minify(src_resource->height0, level);
         box.depth  = u_minify(src_resource->depth0,  level);

	 if (compatible) {
            pipe->resource_copy_region(pipe,
                                       dst_resource, level,
                                       0, 0, layer,
                                       src_resource, level,
                                       &box);
         } else {
            util_resource_copy_region(pipe,
                                      dst_resource, level,
                                      0, 0, layer,
                                      src_resource, level,
                                      &box);
         }
      }
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * ResourceCopyRegion --
 *
 *    The ResourceCopyRegion function copies a source subresource
 *    region to a location on a destination subresource.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ResourceCopyRegion(D3D10DDI_HDEVICE hDevice,                // IN
                   D3D10DDI_HRESOURCE hDstResource,         // IN
                   UINT DstSubResource,                     // IN
                   UINT DstX,                               // IN
                   UINT DstY,                               // IN
                   UINT DstZ,                               // IN
                   D3D10DDI_HRESOURCE hSrcResource,         // IN
                   UINT SrcSubResource,                     // IN
                   __in_opt const D3D10_DDI_BOX *pSrcBox)   // IN (optional)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   if (!CheckPredicate(pDevice)) {
      return;
   }

   struct pipe_context *pipe = pDevice->pipe;
   Resource *pDstResource = CastResource(hDstResource);
   Resource *pSrcResource = CastResource(hSrcResource);

   if (!RestoreConstantBufferOriginal(
          pDevice, pSrcResource,
          "resource copy-region reads original storage") ||
       !SuspendConstantBufferPublication(
          pDevice, pDstResource, true,
          "resource copy-region writes original storage")) {
      SetError(hDevice, E_FAIL);
      return;
   }

   struct pipe_resource *dst_resource = pDstResource->resource;
   struct pipe_resource *src_resource = pSrcResource->resource;

   unsigned dst_level = DstSubResource % (dst_resource->last_level + 1);
   unsigned dst_layer = DstSubResource / (dst_resource->last_level + 1);
   unsigned src_level = SrcSubResource % (src_resource->last_level + 1);
   unsigned src_layer = SrcSubResource / (src_resource->last_level + 1);

   struct pipe_box src_box;
   if (pSrcBox) {
      src_box.x = pSrcBox->left;
      src_box.y = pSrcBox->top;
      src_box.z = pSrcBox->front + src_layer;
      src_box.width  = pSrcBox->right  - pSrcBox->left;
      src_box.height = pSrcBox->bottom - pSrcBox->top;
      src_box.depth  = pSrcBox->back   - pSrcBox->front;
   } else {
      src_box.x = 0;
      src_box.y = 0;
      src_box.z = 0 + src_layer;
      src_box.width  = u_minify(src_resource->width0,  src_level);
      src_box.height = u_minify(src_resource->height0, src_level);
      src_box.depth  = u_minify(src_resource->depth0,  src_level);
   }

   if (areResourcesCompatible(src_resource, dst_resource) ||
       (is_yttrium_screen(pipe->screen) &&
        areYttriumDepthReadbackFormatsCompatible(src_resource,
                                                 dst_resource))) {
      pipe->resource_copy_region(pipe,
                                 dst_resource, dst_level,
                                 DstX, DstY, DstZ + dst_layer,
                                 src_resource, src_level,
                                 &src_box);
   } else if (areResourcesFallbackCopyCompatible(src_resource, dst_resource)) {
      util_resource_copy_region(pipe,
                                dst_resource, dst_level,
                                DstX, DstY, DstZ + dst_layer,
                                src_resource, src_level,
                                &src_box);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * ResourceResolveSubResource --
 *
 *    The ResourceResolveSubResource function resolves
 *    multiple samples to one pixel.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ResourceResolveSubResource(D3D10DDI_HDEVICE hDevice,        // IN
                           D3D10DDI_HRESOURCE hDstResource, // IN
                           UINT DstSubResource,             // IN
                           D3D10DDI_HRESOURCE hSrcResource, // IN
                           UINT SrcSubResource,             // IN
                           DXGI_FORMAT ResolveFormat)       // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   struct pipe_resource *dst_resource = CastPipeResource(hDstResource);
   struct pipe_resource *src_resource = CastPipeResource(hSrcResource);
   if (pipe && dst_resource && src_resource) {
      unsigned dst_level = DstSubResource % (dst_resource->last_level + 1);
      unsigned dst_layer = DstSubResource / (dst_resource->last_level + 1);
      unsigned src_level = SrcSubResource % (src_resource->last_level + 1);
      unsigned src_layer = SrcSubResource / (src_resource->last_level + 1);
      enum pipe_format resolve_format = FormatTranslate(ResolveFormat, false);
      if (pipe->blit &&
          src_resource->nr_samples > 1 &&
          dst_resource->nr_samples <= 1 &&
          resolve_format != PIPE_FORMAT_NONE) {
         struct pipe_blit_info info = {};
         info.dst.resource = dst_resource;
         info.dst.level = dst_level;
         info.dst.format = resolve_format;
         info.dst.box.width = MIN2(dst_resource->width0, src_resource->width0);
         info.dst.box.height = MIN2(dst_resource->height0, src_resource->height0);
         info.dst.box.depth = 1;
         info.dst.box.z = dst_layer;
         info.src.resource = src_resource;
         info.src.level = src_level;
         info.src.format = resolve_format;
         info.src.box.width = info.dst.box.width;
         info.src.box.height = info.dst.box.height;
         info.src.box.depth = 1;
         info.src.box.z = src_layer;
         info.mask = util_format_get_mask(resolve_format);
         info.filter = PIPE_TEX_FILTER_NEAREST;
         pipe->blit(pipe, &info);
         return;
      }

      struct pipe_box src_box = {};
      src_box.width = MIN2(dst_resource->width0, src_resource->width0);
      src_box.height = MIN2(dst_resource->height0, src_resource->height0);
      src_box.depth = 1;

      if (pipe->resource_copy_region) {
         pipe->resource_copy_region(pipe, dst_resource, DstSubResource, 0, 0, 0,
                                    src_resource, SrcSubResource, &src_box);
      } else if (areResourcesFallbackCopyCompatible(src_resource, dst_resource)) {
         util_resource_copy_region(pipe, dst_resource, DstSubResource, 0, 0, 0,
                                   src_resource, SrcSubResource, &src_box);
      }
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * ResourceIsStagingBusy --
 *
 *    The ResourceIsStagingBusy function determines whether a
 *    resource is currently being used by the graphics pipeline.
 *
 * ----------------------------------------------------------------------
 */

BOOL APIENTRY
ResourceIsStagingBusy(D3D10DDI_HDEVICE hDevice,       // IN
                      D3D10DDI_HRESOURCE hResource)   // IN
{
   LOG_ENTRYPOINT();

   /* ignore */

   return false;
}


/*
 * ----------------------------------------------------------------------
 *
 * ResourceReadAfterWriteHazard --
 *
 *    The ResourceReadAfterWriteHazard function informs the user-mode
 *    display driver that the specified resource was used as an output
 *    from the graphics processing unit (GPU) and that the resource
 *    will be used as an input to the GPU.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ResourceReadAfterWriteHazard(D3D10DDI_HDEVICE hDevice,      // IN
                             D3D10DDI_HRESOURCE hResource)  // IN
{
   LOG_ENTRYPOINT();

   /* Not actually necessary */
}


/*
 * ----------------------------------------------------------------------
 *
 * ResourceUpdateSubResourceUP --
 *
 *    The ResourceUpdateSubresourceUP function updates a
 *    destination subresource region from a source
 *    system memory region.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ResourceUpdateSubResourceUP(D3D10DDI_HDEVICE hDevice,                // IN
                            D3D10DDI_HRESOURCE hDstResource,         // IN
                            UINT DstSubResource,                     // IN
                            __in_opt const D3D10_DDI_BOX *pDstBox,   // IN
                            __in const void *pSysMemUP,              // IN
                            UINT RowPitch,                           // IN
                            UINT DepthPitch)                         // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   if (!CheckPredicate(pDevice)) {
      return;
   }

   struct pipe_context *pipe = pDevice->pipe;
   Resource *pDstResource = CastResource(hDstResource);
   struct pipe_resource *dst_resource = pDstResource->resource;

   unsigned level;
   struct pipe_box box;

   if (pDstBox) {
      UINT DstMipLevels = dst_resource->last_level + 1;
      level = DstSubResource % DstMipLevels;
      unsigned dst_layer = DstSubResource / DstMipLevels;

      if (pDstBox->right <= pDstBox->left ||
          pDstBox->bottom <= pDstBox->top ||
          pDstBox->back <= pDstBox->front)
         return;

      box.x = pDstBox->left;
      box.y = pDstBox->top;
      box.z = pDstBox->front + dst_layer;
      box.width  = pDstBox->right  - pDstBox->left;
      box.height = pDstBox->bottom - pDstBox->top;
      box.depth  = pDstBox->back   - pDstBox->front;
   } else {
      subResourceBox(dst_resource, DstSubResource, &level, &box);
   }

   if (pDstResource->buffer &&
       ConstantBufferPublicationEligible(pDevice, pDstResource)) {
      if (ConstantBufferPublicationUpdateHasCompleteContents(
             pDstResource, box.x, box.width)) {
         if (PublishConstantBufferUpdate(
                pDevice, pDstResource, box.x, box.width, pSysMemUP))
            return;

         WarnConstantBufferPublicationFallback(
            pDevice, pDstResource, "immutable uploader allocation failed");
         if (!SuspendConstantBufferPublication(
                pDevice, pDstResource, true,
                "immutable uploader allocation failed")) {
            SetError(hDevice, E_FAIL);
            return;
         }
      }
   }

   if (pDstResource->buffer && pipe->buffer_subdata) {
      const bool shadow_was_valid =
         pDstResource->constant_shadow_valid &&
         pDstResource->buffer_shadow_size >= dst_resource->width0;
      pipe->buffer_subdata(pipe,
                           dst_resource,
                           PIPE_MAP_DISCARD_RANGE,
                           box.x,
                           box.width,
                           pSysMemUP);
      UpdateBufferShadow(pDstResource, box.x, box.width, pSysMemUP);
      if (ConstantBufferPublicationCandidate(pDevice, pDstResource) &&
          EnsureBufferShadow(pDstResource, dst_resource->width0)) {
         pDstResource->constant_shadow_valid =
            (box.x == 0 && box.width == dst_resource->width0) ||
            shadow_was_valid;
      }
      return;
   }

   if (!pDstResource->buffer && pipe->texture_subdata) {
      pipe->texture_subdata(pipe,
                            dst_resource,
                            level,
                            PIPE_MAP_DISCARD_RANGE,
                            &box,
                            pSysMemUP,
                            RowPitch,
                            DepthPitch);
      return;
   }

   struct pipe_transfer *transfer;
   void *map;
   if (pDstResource->buffer) {
      map = pipe->buffer_map(pipe,
                              dst_resource,
                              level,
                              PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE,
                              &box,
                              &transfer);
   } else {
      map = pipe->texture_map(pipe,
                              dst_resource,
                              level,
                              PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE,
                              &box,
                              &transfer);
   }
   assert(map);
   if (map) {
      for (int z = 0; z < box.depth; ++z) {
         uint8_t *dst = (uint8_t*)map + z*transfer->layer_stride;
         const uint8_t *src = (const uint8_t*)pSysMemUP + z*DepthPitch;
         util_copy_rect(dst,
                        dst_resource->format,
                        transfer->stride,
                        0, 0, box.width, box.height,
                        src,
                        RowPitch,
                        0, 0);
      }
      if (pDstResource->buffer) {
         UpdateBufferShadow(pDstResource, box.x, box.width, pSysMemUP);
         pipe_buffer_unmap(pipe, transfer);
      } else {
         pipe_texture_unmap(pipe, transfer);
      }
   }
}
