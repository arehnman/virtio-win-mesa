/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <mutex>

#include "util/cnd_monotonic.h"
#include "util/log.h"
#include "util/timespec.h"
#include "util/u_thread.h"
#include "vk_format.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_util.h"
#include "wsi_common_entrypoints.h"
#include "wsi_common_private.h"

#define D3D12_IGNORE_SDK_LAYERS
#include <dxgi1_5.h>
#if defined(HAVE_YTTRIUM)
#include <d3d11.h>
#endif
#include <directx/d3d12.h>
#include <dxguids/dxguids.h>

#include <dcomp.h>

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"      // warning: cast to pointer from integer of different size
#endif

struct wsi_win32;

struct wsi_win32 {
   struct wsi_interface                     base;

   struct wsi_device *wsi;

   const VkAllocationCallbacks *alloc;
   VkPhysicalDevice physical_device;
   struct {
      IDXGIFactory4 *factory;
      IDCompositionDevice *dcomp;
      bool supports_tearing;
   } dxgi;
};

enum wsi_win32_image_state {
   WSI_IMAGE_IDLE,
   WSI_IMAGE_DRAWING,
   WSI_IMAGE_QUEUED,
};

struct wsi_win32_image {
   struct wsi_image base;
   enum wsi_win32_image_state state;
   struct wsi_win32_swapchain *chain;
   struct {
      ID3D12Resource *swapchain_res;
   } dxgi;
#if defined(HAVE_YTTRIUM)
   struct {
      ID3D11Texture2D *texture;
   } d3d11;
#endif
   struct {
      HDC dc;
      HBITMAP bmp;
      int bmp_row_pitch;
      void *ppvBits;
   } sw;
};

struct wsi_win32_surface {
   VkIcdSurfaceWin32 base;

   /* The first time a DXGI swapchain is presented against this surface, a
    * DComp target/visual is created and that swapchain is bound.  A GDI
    * successor clears the visual before blitting so it is not hidden under
    * stale DComp content.  current_swapchain records which path most recently
    * completed that serialized takeover.
    */
   IDCompositionTarget *target;
   IDCompositionVisual *visual;
   IDCompositionDevice *dcomp_owner;
   mtx_t mutex;
   bool creating_swapchain;
   struct wsi_win32_swapchain *active_swapchain;
   struct wsi_win32_swapchain *current_swapchain;
};

enum wsi_win32_present_path {
   WSI_WIN32_PRESENT_GDI,
   WSI_WIN32_PRESENT_DXGI_D3D12,
#if defined(HAVE_YTTRIUM)
   WSI_WIN32_PRESENT_DXGI_SHARED,
#endif
};

struct wsi_win32_swapchain {
   struct wsi_swapchain         base;
#if defined(HAVE_YTTRIUM)
   HMODULE                    d3d11_mod;
   ID3D11Device              *d3d11_device;
   ID3D11DeviceContext       *d3d11_context;
#endif
   IDXGISwapChain3            *dxgi;
   enum wsi_win32_present_path present_path;
   struct wsi_win32           *wsi;
   wsi_win32_surface          *surface;
   mtx_t                      acquire_mutex;
   struct u_cnd_monotonic     acquire_cond;
   uint64_t                     flip_sequence;
   VkResult                     status;
   bool                         retired;
   VkExtent2D                 extent;
   HWND wnd;
   HDC chain_dc;
   struct wsi_win32_image     images[0];
};

/* DirectComposition transactions are device-global.  Serialize mutations
 * across all Win32 surfaces so one surface cannot commit another surface's
 * pending visual changes.
 */
static std::mutex wsi_win32_dcomp_mutex;

static VkResult
wsi_win32_hresult_to_result(HRESULT hr)
{
   if (hr == DXGI_ERROR_DEVICE_REMOVED)
      return VK_ERROR_DEVICE_LOST;
   if (hr == E_OUTOFMEMORY)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   return VK_ERROR_SURFACE_LOST_KHR;
}

/* Must be called with surface->mutex and wsi_win32_dcomp_mutex held. */
static void
wsi_win32_surface_drop_dcomp_locked(struct wsi_win32_surface *surface)
{
   if (surface->target && surface->dcomp_owner) {
      HRESULT root_hr = surface->target->SetRoot(NULL);
      HRESULT commit_hr = surface->dcomp_owner->Commit();
      if (FAILED(root_hr) || FAILED(commit_hr)) {
         mesa_logw("wsi/win32: DirectComposition reset failed "
                   "(SetRoot HRESULT 0x%08lx, Commit HRESULT 0x%08lx)",
                   (unsigned long)root_hr, (unsigned long)commit_hr);
      }
   }

   if (surface->visual) {
      surface->visual->Release();
      surface->visual = NULL;
   }
   if (surface->target) {
      surface->target->Release();
      surface->target = NULL;
   }
   if (surface->dcomp_owner) {
      surface->dcomp_owner->Release();
      surface->dcomp_owner = NULL;
   }
   surface->current_swapchain = NULL;
}

/* Must be called with surface->mutex held.  DirectComposition batches
 * SetContent and Commit as one transaction, so do not publish the new
 * current_swapchain until both operations have succeeded.
 */
static VkResult
wsi_win32_surface_set_content_locked(struct wsi_win32_surface *surface,
                                     struct wsi_win32 *wsi,
                                     struct wsi_win32_swapchain *chain)
{
   if (surface->current_swapchain == chain)
      return VK_SUCCESS;

   /* A present of an image acquired before oldSwapchain was retired remains
    * valid only while that chain is still the visual content.  Once a
    * replacement has taken over, reject the late present without disturbing
    * the replacement's DirectComposition graph.
    */
   if (chain && chain->retired)
      return VK_ERROR_OUT_OF_DATE_KHR;

   std::lock_guard<std::mutex> dcomp_lock(wsi_win32_dcomp_mutex);

   if (surface->dcomp_owner && surface->dcomp_owner != wsi->dxgi.dcomp)
      wsi_win32_surface_drop_dcomp_locked(surface);

   if (!chain) {
      if (!surface->visual) {
         surface->current_swapchain = NULL;
         return VK_SUCCESS;
      }

      HRESULT hr = surface->visual->SetContent(NULL);
      if (SUCCEEDED(hr))
         hr = surface->dcomp_owner->Commit();
      if (FAILED(hr)) {
         mesa_logw("wsi/win32: DirectComposition detach failed "
                   "(HRESULT 0x%08lx)", (unsigned long)hr);
         wsi_win32_surface_drop_dcomp_locked(surface);
         return wsi_win32_hresult_to_result(hr);
      }

      surface->current_swapchain = NULL;
      return VK_SUCCESS;
   }

   IDCompositionTarget *target = surface->target;
   IDCompositionVisual *visual = surface->visual;
   bool created_objects = false;
   HRESULT hr = S_OK;

   if (!target || !visual) {
      if (target || visual)
         wsi_win32_surface_drop_dcomp_locked(surface);

      target = NULL;
      visual = NULL;
      hr = wsi->dxgi.dcomp->CreateTargetForHwnd(
         surface->base.hwnd, false, &target);
      if (SUCCEEDED(hr))
         hr = wsi->dxgi.dcomp->CreateVisual(&visual);
      if (SUCCEEDED(hr))
         hr = target->SetRoot(visual);
      created_objects = true;
   }

   if (SUCCEEDED(hr))
      hr = visual->SetContent(chain->dxgi);
   if (SUCCEEDED(hr))
      hr = wsi->dxgi.dcomp->Commit();

   if (FAILED(hr)) {
      mesa_logw("wsi/win32: DirectComposition attach failed "
                "(HRESULT 0x%08lx)", (unsigned long)hr);
      if (created_objects) {
         HRESULT root_hr = target ? target->SetRoot(NULL) : S_OK;
         HRESULT commit_hr = wsi->dxgi.dcomp ?
            wsi->dxgi.dcomp->Commit() : E_FAIL;
         if (FAILED(root_hr) || FAILED(commit_hr)) {
            mesa_logw("wsi/win32: DirectComposition cleanup failed "
                      "(SetRoot HRESULT 0x%08lx, Commit HRESULT 0x%08lx)",
                      (unsigned long)root_hr, (unsigned long)commit_hr);
         }
         if (visual)
            visual->Release();
         if (target)
            target->Release();
         surface->current_swapchain = NULL;
      } else {
         wsi_win32_surface_drop_dcomp_locked(surface);
      }
      return wsi_win32_hresult_to_result(hr);
   }

   if (created_objects) {
      surface->target = target;
      surface->visual = visual;
      wsi->dxgi.dcomp->AddRef();
      surface->dcomp_owner = wsi->dxgi.dcomp;
   }
   surface->current_swapchain = chain;
   return VK_SUCCESS;
}

static VkResult
wsi_win32_surface_set_content(struct wsi_win32_swapchain *chain)
{
   struct wsi_win32_surface *surface = chain->surface;
   mtx_lock(&surface->mutex);
   VkResult result =
      wsi_win32_surface_set_content_locked(surface, chain->wsi, chain);
   mtx_unlock(&surface->mutex);
   return result;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
wsi_GetPhysicalDeviceWin32PresentationSupportKHR(VkPhysicalDevice physicalDevice,
                                                 uint32_t queueFamilyIndex)
{
   VK_FROM_HANDLE(vk_physical_device, pdevice, physicalDevice);
   struct wsi_device *wsi_device = pdevice->wsi_device;
   return (wsi_device->queue_supports_blit & BITFIELD64_BIT(queueFamilyIndex)) != 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_CreateWin32SurfaceKHR(VkInstance _instance,
                          const VkWin32SurfaceCreateInfoKHR *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   wsi_win32_surface *surface;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR);

   surface = (wsi_win32_surface *)vk_zalloc2(&instance->alloc, pAllocator, sizeof(*surface), 8,
                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (surface == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   if (mtx_init(&surface->mutex, mtx_plain) != thrd_success) {
      vk_free2(&instance->alloc, pAllocator, surface);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   surface->base.base.platform = VK_ICD_WSI_PLATFORM_WIN32;

   surface->base.hinstance = pCreateInfo->hinstance;
   surface->base.hwnd = pCreateInfo->hwnd;

   *pSurface = VkIcdSurfaceBase_to_handle(&surface->base.base);

   return VK_SUCCESS;
}

void
wsi_win32_surface_destroy(VkIcdSurfaceBase *icd_surface, VkInstance _instance,
                          const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   wsi_win32_surface *surface = (wsi_win32_surface *)icd_surface;
   mtx_lock(&surface->mutex);
   {
      std::lock_guard<std::mutex> dcomp_lock(wsi_win32_dcomp_mutex);
      wsi_win32_surface_drop_dcomp_locked(surface);
   }
   mtx_unlock(&surface->mutex);
   mtx_destroy(&surface->mutex);
   vk_free2(&instance->alloc, pAllocator, icd_surface);
}

static VkResult
wsi_win32_surface_get_support(VkIcdSurfaceBase *surface,
                              struct wsi_device *wsi_device,
                              uint32_t queueFamilyIndex,
                              VkBool32* pSupported)
{
   *pSupported = true;

   return VK_SUCCESS;
}

static VkResult
wsi_win32_surface_get_capabilities(VkIcdSurfaceBase *surf,
                                   struct wsi_device *wsi_device,
                                   VkSurfaceCapabilitiesKHR* caps)
{
   VkIcdSurfaceWin32 *surface = (VkIcdSurfaceWin32 *)surf;
   struct wsi_win32 *wsi =
      (struct wsi_win32 *)wsi_device->wsi[VK_ICD_WSI_PLATFORM_WIN32];

   RECT win_rect;
   if (!GetClientRect(surface->hwnd, &win_rect))
      return VK_ERROR_SURFACE_LOST_KHR;

   caps->minImageCount = 1;

   if (!wsi_device->sw && wsi && wsi->dxgi.factory && wsi->dxgi.dcomp &&
       (wsi_device->win32.get_d3d12_command_queue
#if defined(HAVE_YTTRIUM)
        || wsi_device->win32.create_image_memory_from_win32_handle
#endif
       )) {
      /* DXGI doesn't support random presenting order (images need to
       * be presented in the order they were acquired), so we can't
       * expose more than two image per swapchain.
       */
      caps->minImageCount = caps->maxImageCount = 2;
   } else {
      caps->minImageCount = 1;
      /* Software callbacke, there is no real maximum */
      caps->maxImageCount = 0;
   }

   caps->currentExtent = {
      (uint32_t)win_rect.right - (uint32_t)win_rect.left,
      (uint32_t)win_rect.bottom - (uint32_t)win_rect.top
   };
   caps->minImageExtent = { 1u, 1u };
   caps->maxImageExtent = {
      wsi_device->maxImageDimension2D,
      wsi_device->maxImageDimension2D,
   };

   caps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->maxImageArrayLayers = 1;

   caps->supportedCompositeAlpha =
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR |
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR |
      VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;

   caps->supportedUsageFlags = wsi_caps_get_image_usage();

   VK_FROM_HANDLE(vk_physical_device, pdevice, wsi_device->pdevice);
   if (pdevice->supported_extensions.EXT_attachment_feedback_loop_layout)
      caps->supportedUsageFlags |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   return VK_SUCCESS;
}

static VkResult
wsi_win32_surface_get_capabilities2(VkIcdSurfaceBase *surface,
                                    struct wsi_device *wsi_device,
                                    const void *info_next,
                                    VkSurfaceCapabilities2KHR* caps)
{
   assert(caps->sType == VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR);

   const VkSurfacePresentModeKHR *present_mode =
      (const VkSurfacePresentModeKHR *)vk_find_struct_const(info_next, SURFACE_PRESENT_MODE_KHR);

   VkResult result =
      wsi_win32_surface_get_capabilities(surface, wsi_device,
                                      &caps->surfaceCapabilities);

   vk_foreach_struct(ext, caps->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR: {
         VkSurfaceProtectedCapabilitiesKHR *protected_cap = (VkSurfaceProtectedCapabilitiesKHR *)ext;
         protected_cap->supportsProtected = VK_FALSE;
         break;
      }

      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_KHR: {
         /* Unsupported. */
         VkSurfacePresentScalingCapabilitiesEXT *scaling =
            (VkSurfacePresentScalingCapabilitiesEXT *)ext;
         scaling->supportedPresentScaling = 0;
         scaling->supportedPresentGravityX = 0;
         scaling->supportedPresentGravityY = 0;
         scaling->minScaledImageExtent = caps->surfaceCapabilities.minImageExtent;
         scaling->maxScaledImageExtent = caps->surfaceCapabilities.maxImageExtent;
         break;
      }

      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_KHR: {
         /* Unsupported, just report the input present mode. */
         VkSurfacePresentModeCompatibilityKHR *compat =
            (VkSurfacePresentModeCompatibilityKHR *)ext;
         if (compat->pPresentModes) {
            if (compat->presentModeCount) {
               assert(present_mode);
               compat->pPresentModes[0] = present_mode->presentMode;
               compat->presentModeCount = 1;
            }
         } else {
            if (!present_mode)
               wsi_common_vk_warn_once("Use of VkSurfacePresentModeCompatibilityKHR "
                                       "without a VkSurfacePresentModeKHR set. This is an "
                                       "application bug.\n");
            compat->presentModeCount = 1;
         }
         break;
      }

      case VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT: {
         VkPresentTimingSurfaceCapabilitiesEXT *wait = (VkPresentTimingSurfaceCapabilitiesEXT *)ext;

         wait->presentStageQueries = 0;
         wait->presentTimingSupported = VK_FALSE;
         wait->presentAtAbsoluteTimeSupported = VK_FALSE;
         wait->presentAtRelativeTimeSupported = VK_FALSE;
         break;
      }

      default:
         /* Ignored */
         break;
      }
   }

   return result;
}


static const struct {
   VkFormat     format;
} available_surface_formats[] = {
   { VK_FORMAT_B8G8R8A8_SRGB },
   { VK_FORMAT_B8G8R8A8_UNORM },
};


static void
get_sorted_vk_formats(struct wsi_device *wsi_device, VkFormat *sorted_formats)
{
   for (unsigned i = 0; i < ARRAY_SIZE(available_surface_formats); i++)
      sorted_formats[i] = available_surface_formats[i].format;

   if (wsi_device->force_bgra8_unorm_first) {
      for (unsigned i = 0; i < ARRAY_SIZE(available_surface_formats); i++) {
         if (sorted_formats[i] == VK_FORMAT_B8G8R8A8_UNORM) {
            sorted_formats[i] = sorted_formats[0];
            sorted_formats[0] = VK_FORMAT_B8G8R8A8_UNORM;
            break;
         }
      }
   }
}

static VkResult
wsi_win32_surface_get_formats(VkIcdSurfaceBase *icd_surface,
                              struct wsi_device *wsi_device,
                              uint32_t* pSurfaceFormatCount,
                              VkSurfaceFormatKHR* pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats, pSurfaceFormatCount);

   VkFormat sorted_formats[ARRAY_SIZE(available_surface_formats)];
   get_sorted_vk_formats(wsi_device, sorted_formats);

   for (unsigned i = 0; i < ARRAY_SIZE(sorted_formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormatKHR, &out, f) {
         f->format = sorted_formats[i];
         f->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }

   return vk_outarray_status(&out);
}

static VkResult
wsi_win32_surface_get_formats2(VkIcdSurfaceBase *icd_surface,
                               struct wsi_device *wsi_device,
                               const void *info_next,
                               uint32_t* pSurfaceFormatCount,
                               VkSurfaceFormat2KHR* pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, pSurfaceFormats, pSurfaceFormatCount);

   VkFormat sorted_formats[ARRAY_SIZE(available_surface_formats)];
   get_sorted_vk_formats(wsi_device, sorted_formats);

   for (unsigned i = 0; i < ARRAY_SIZE(sorted_formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, f) {
         assert(f->sType == VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR);
         f->surfaceFormat.format = sorted_formats[i];
         f->surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }

   return vk_outarray_status(&out);
}

static const VkPresentModeKHR present_modes_gdi[] = {
   VK_PRESENT_MODE_FIFO_KHR,
};
static const VkPresentModeKHR present_modes_dxgi[] = {
   VK_PRESENT_MODE_IMMEDIATE_KHR,
   VK_PRESENT_MODE_MAILBOX_KHR,
   VK_PRESENT_MODE_FIFO_KHR,
};
static const VkPresentModeKHR present_modes_dxgi_no_tearing[] = {
   VK_PRESENT_MODE_MAILBOX_KHR,
   VK_PRESENT_MODE_FIFO_KHR,
};

static VkResult
wsi_win32_surface_get_present_modes(VkIcdSurfaceBase *surface,
                                    struct wsi_device *wsi_device,
                                    uint32_t* pPresentModeCount,
                                    VkPresentModeKHR* pPresentModes)
{
   const VkPresentModeKHR *array;
   size_t array_size;
   struct wsi_win32 *wsi =
      (struct wsi_win32 *)wsi_device->wsi[VK_ICD_WSI_PLATFORM_WIN32];

   if (wsi_device->sw || !wsi || !wsi->dxgi.factory || !wsi->dxgi.dcomp ||
       (!wsi_device->win32.get_d3d12_command_queue
#if defined(HAVE_YTTRIUM)
        && !wsi_device->win32.create_image_memory_from_win32_handle
#endif
       )) {
      array = present_modes_gdi;
      array_size = ARRAY_SIZE(present_modes_gdi);
   } else if (wsi->dxgi.supports_tearing) {
      array = present_modes_dxgi;
      array_size = ARRAY_SIZE(present_modes_dxgi);
   } else {
      array = present_modes_dxgi_no_tearing;
      array_size = ARRAY_SIZE(present_modes_dxgi_no_tearing);
   }

   if (pPresentModes == NULL) {
      *pPresentModeCount = array_size;
      return VK_SUCCESS;
   }

   *pPresentModeCount = MIN2(*pPresentModeCount, array_size);
   typed_memcpy(pPresentModes, array, *pPresentModeCount);

   if (*pPresentModeCount < array_size)
      return VK_INCOMPLETE;
   else
      return VK_SUCCESS;
}

static VkResult
wsi_win32_surface_get_present_rectangles(VkIcdSurfaceBase *surface,
                                      struct wsi_device *wsi_device,
                                      uint32_t* pRectCount,
                                      VkRect2D* pRects)
{
   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, pRects, pRectCount);

   vk_outarray_append_typed(VkRect2D, &out, rect) {
      /* We don't know a size so just return the usual "I don't know." */
      *rect = {
         { 0, 0 },
         { UINT32_MAX, UINT32_MAX },
      };
   }

   return vk_outarray_status(&out);
}

static VkResult
wsi_create_dxgi_image_mem(const struct wsi_swapchain *drv_chain,
                          const struct wsi_image_info *info,
                          struct wsi_image *image)
{
   struct wsi_win32_swapchain *chain = (struct wsi_win32_swapchain *)drv_chain;
   const struct wsi_device *wsi = chain->base.wsi;

   assert(chain->base.blit.type != WSI_SWAPCHAIN_BUFFER_BLIT);

   struct wsi_win32_image *win32_image =
      container_of(image, struct wsi_win32_image, base);
   uint32_t image_idx =
      ((uintptr_t)win32_image - (uintptr_t)chain->images) /
      sizeof(*win32_image);
   if (FAILED(chain->dxgi->GetBuffer(image_idx,
                                     IID_PPV_ARGS(&win32_image->dxgi.swapchain_res))))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   VkResult result =
      wsi->win32.create_image_memory(chain->base.device,
                                     win32_image->dxgi.swapchain_res,
                                     &chain->base.alloc,
                                     chain->base.blit.type == WSI_SWAPCHAIN_NO_BLIT ?
                                     &image->memory : &image->blit.memory);
   if (result != VK_SUCCESS)
      return result;

   if (chain->base.blit.type == WSI_SWAPCHAIN_NO_BLIT)
      return VK_SUCCESS;

   VkImageCreateInfo create = info->create;

   create.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
   create.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

   result = wsi->CreateImage(chain->base.device, &create,
                             &chain->base.alloc, &image->blit.image);
   if (result != VK_SUCCESS)
      return result;

   result = wsi->BindImageMemory(chain->base.device, image->blit.image,
                                 image->blit.memory, 0);
   if (result != VK_SUCCESS)
      return result;

   VkMemoryRequirements reqs;
   wsi->GetImageMemoryRequirements(chain->base.device, image->image, &reqs);

   const VkMemoryDedicatedAllocateInfo memory_dedicated_info = {
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      nullptr,
      image->blit.image,
      VK_NULL_HANDLE,
   };
   const VkMemoryAllocateInfo memory_info = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      &memory_dedicated_info,
      reqs.size,
      info->select_image_memory_type(wsi, reqs.memoryTypeBits),
   };

   return wsi->AllocateMemory(chain->base.device, &memory_info,
                              &chain->base.alloc, &image->memory);
}

#if defined(HAVE_YTTRIUM)
static DXGI_FORMAT
wsi_dxgi_format_from_vk(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      return DXGI_FORMAT_B8G8R8A8_UNORM;
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
   default:
      return DXGI_FORMAT_UNKNOWN;
   }
}

static VkResult
wsi_create_dxgi_shared_image_mem(const struct wsi_swapchain *drv_chain,
                                 const struct wsi_image_info *info,
                                 struct wsi_image *image)
{
   struct wsi_win32_swapchain *chain = (struct wsi_win32_swapchain *)drv_chain;
   const struct wsi_device *wsi = chain->base.wsi;
   struct wsi_win32_image *win32_image =
      container_of(image, struct wsi_win32_image, base);

   if (!chain->d3d11_device ||
       !wsi->win32.create_image_memory_from_win32_handle)
      return VK_ERROR_INITIALIZATION_FAILED;

   const DXGI_FORMAT format = wsi_dxgi_format_from_vk(info->create.format);
   if (format == DXGI_FORMAT_UNKNOWN)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   D3D11_TEXTURE2D_DESC desc = {};
   desc.Width = info->create.extent.width;
   desc.Height = info->create.extent.height;
   desc.MipLevels = 1;
   desc.ArraySize = 1;
   desc.Format = format;
   desc.SampleDesc.Count = 1;
   desc.Usage = D3D11_USAGE_DEFAULT;
   desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
   desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

   HRESULT hr = chain->d3d11_device->CreateTexture2D(
      &desc, NULL, &win32_image->d3d11.texture);
   if (FAILED(hr)) {
      mesa_logw("wsi/win32: DXGI_SHARED CreateTexture2D failed "
                "(HRESULT 0x%08lx)", (unsigned long)hr);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   IDXGIResource *resource = NULL;
   hr = win32_image->d3d11.texture->QueryInterface(IID_PPV_ARGS(&resource));
   if (FAILED(hr)) {
      mesa_logw("wsi/win32: DXGI_SHARED IDXGIResource query failed "
                "(HRESULT 0x%08lx)", (unsigned long)hr);
      win32_image->d3d11.texture->Release();
      win32_image->d3d11.texture = NULL;
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   HANDLE handle = NULL;
   hr = resource->GetSharedHandle(&handle);
   resource->Release();
   if (FAILED(hr) || !handle) {
      mesa_logw("wsi/win32: DXGI_SHARED GetSharedHandle failed "
                "(HRESULT 0x%08lx handle=%p)", (unsigned long)hr, handle);
      win32_image->d3d11.texture->Release();
      win32_image->d3d11.texture = NULL;
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   VkMemoryRequirements reqs;
   wsi->GetImageMemoryRequirements(chain->base.device, image->image, &reqs);

   VkResult result = wsi->win32.create_image_memory_from_win32_handle(
      chain->base.device, image->image, handle, reqs.size, &chain->base.alloc,
      &image->memory);
   if (result != VK_SUCCESS)
      mesa_logw("wsi/win32: DXGI_SHARED shared-memory import failed "
                "(VkResult %d)", (int)result);
   return result;
}
#endif

enum wsi_swapchain_blit_type
wsi_dxgi_image_needs_blit(const struct wsi_device *wsi,
                          const struct wsi_dxgi_image_params *params,
                          VkDevice device)
{
   if (wsi->win32.requires_blits && wsi->win32.requires_blits(device))
      return WSI_SWAPCHAIN_IMAGE_BLIT;
   else if (params->storage_image)
      return WSI_SWAPCHAIN_IMAGE_BLIT;
   return WSI_SWAPCHAIN_NO_BLIT;
}

VkResult
wsi_dxgi_configure_image(const struct wsi_swapchain *chain,
                         const VkSwapchainCreateInfoKHR *pCreateInfo,
                         const struct wsi_dxgi_image_params *params,
                         struct wsi_image_info *info)
{
   VkResult result =
      wsi_configure_image(chain, pCreateInfo, 0, info);
   if (result != VK_SUCCESS)
      return result;

   info->create_mem = wsi_create_dxgi_image_mem;

   if (chain->blit.type != WSI_SWAPCHAIN_NO_BLIT) {
      wsi_configure_image_blit_image(chain, info);
      info->select_image_memory_type = wsi_select_device_memory_type;
      info->select_blit_dst_memory_type = wsi_select_device_memory_type;
   }

   return VK_SUCCESS;
}

#if defined(HAVE_YTTRIUM)
VkResult
wsi_dxgi_shared_configure_image(const struct wsi_swapchain *chain,
                                const VkSwapchainCreateInfoKHR *pCreateInfo,
                                const struct wsi_dxgi_shared_image_params *params,
                                struct wsi_image_info *info)
{
   assert(params->base.image_type == WSI_IMAGE_TYPE_DXGI_SHARED);

   VkResult result = wsi_configure_image(chain, pCreateInfo, 0, info);
   if (result != VK_SUCCESS)
      return result;

   info->create_mem = wsi_create_dxgi_shared_image_mem;

   /* This image aliases the memory of the D3D11 texture created in
    * wsi_create_dxgi_shared_image_mem, so the two must agree on the layout of
    * that one allocation.  Vulkan only defines aliasing between images created
    * with VK_IMAGE_CREATE_ALIAS_BIT and otherwise identical parameters, so
    * match what yttrium uses for a PIPE_BIND_SHARED texture.
    *
    * Tiling must be LINEAR on both sides.  With OPTIMAL, each driver picks a
    * swizzle variant from its own heuristics; RADV chose different ones for
    * the two images even with identical format, extent, usage and flags, and
    * the sizes matched exactly (589824 is both a linear 384x384x4 surface and
    * nine 64KB tiles) so the disagreement was invisible in the memory
    * requirements.  Only ~78% of the two surfaces overlapped, and the
    * presented image was scrambled at tile granularity.
    */
   info->create.tiling = VK_IMAGE_TILING_LINEAR;
   info->create.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   info->create.flags &= ~(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                           VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
   info->create.flags |= VK_IMAGE_CREATE_ALIAS_BIT;

   return VK_SUCCESS;
}
#endif

static VkResult
wsi_win32_image_init(VkDevice device_h,
                     struct wsi_win32_swapchain *chain,
                     const VkSwapchainCreateInfoKHR *create_info,
                     const VkAllocationCallbacks *allocator,
                     struct wsi_win32_image *image)
{
   VkResult result = wsi_create_image(&chain->base, &chain->base.image_info,
                                      &image->base);
   if (result != VK_SUCCESS) {
      /* create_mem may already have created the native half of an imported
       * image before Vulkan image creation or binding failed.  This slot is
       * not counted in base.image_count, so swapchain destruction will not
       * visit it.
       */
      if (image->dxgi.swapchain_res) {
         image->dxgi.swapchain_res->Release();
         image->dxgi.swapchain_res = NULL;
      }
#if defined(HAVE_YTTRIUM)
      if (image->d3d11.texture) {
         image->d3d11.texture->Release();
         image->d3d11.texture = NULL;
      }
#endif
      return result;
   }

   VkIcdSurfaceWin32 *win32_surface = (VkIcdSurfaceWin32 *)create_info->surface;
   chain->wnd = win32_surface->hwnd;
   image->chain = chain;

   if (chain->present_path != WSI_WIN32_PRESENT_GDI)
      return VK_SUCCESS;

   chain->chain_dc = GetDC(chain->wnd);
   image->sw.dc = CreateCompatibleDC(chain->chain_dc);
   HBITMAP bmp = NULL;

   BITMAPINFO info = { 0 };
   info.bmiHeader.biSize = sizeof(BITMAPINFO);
   info.bmiHeader.biWidth = create_info->imageExtent.width;
   info.bmiHeader.biHeight = -create_info->imageExtent.height;
   info.bmiHeader.biPlanes = 1;
   info.bmiHeader.biBitCount = 32;
   info.bmiHeader.biCompression = BI_RGB;

   bmp = CreateDIBSection(image->sw.dc, &info, DIB_RGB_COLORS, &image->sw.ppvBits, NULL, 0);
   assert(bmp && image->sw.ppvBits);

   SelectObject(image->sw.dc, bmp);

   BITMAP header;
   int status = GetObject(bmp, sizeof(BITMAP), &header);
   (void)status;
   image->sw.bmp_row_pitch = header.bmWidthBytes;
   image->sw.bmp = bmp;

   return VK_SUCCESS;
}

static void
wsi_win32_image_finish(struct wsi_win32_swapchain *chain,
                       const VkAllocationCallbacks *allocator,
                       struct wsi_win32_image *image)
{
   if (image->dxgi.swapchain_res)
      image->dxgi.swapchain_res->Release();
#if defined(HAVE_YTTRIUM)
   if (image->d3d11.texture)
      image->d3d11.texture->Release();
#endif

   if (image->sw.dc)
      DeleteDC(image->sw.dc);
   if(image->sw.bmp)
      DeleteObject(image->sw.bmp);
   wsi_destroy_image(&chain->base, &image->base);
}

static VkResult
wsi_win32_swapchain_destroy(struct wsi_swapchain *drv_chain,
                            const VkAllocationCallbacks *allocator)
{
   struct wsi_win32_swapchain *chain =
      (struct wsi_win32_swapchain *) drv_chain;

   mtx_lock(&chain->surface->mutex);
   if (chain->surface->active_swapchain == chain)
      chain->surface->active_swapchain = NULL;
   if (chain->surface->current_swapchain == chain) {
      if (chain->present_path == WSI_WIN32_PRESENT_GDI)
         chain->surface->current_swapchain = NULL;
      else
         wsi_win32_surface_set_content_locked(chain->surface, chain->wsi,
                                              NULL);
   }
   mtx_unlock(&chain->surface->mutex);

   for (uint32_t i = 0; i < chain->base.image_count; i++)
      wsi_win32_image_finish(chain, allocator, &chain->images[i]);

   DeleteDC(chain->chain_dc);

   if (chain->dxgi)
      chain->dxgi->Release();
#if defined(HAVE_YTTRIUM)
   if (chain->d3d11_context)
      chain->d3d11_context->Release();
   if (chain->d3d11_device)
      chain->d3d11_device->Release();
   if (chain->d3d11_mod)
      FreeLibrary(chain->d3d11_mod);
#endif

   wsi_swapchain_finish(&chain->base);

   u_cnd_monotonic_destroy(&chain->acquire_cond);
   mtx_destroy(&chain->acquire_mutex);

   vk_free(allocator, chain);
   return VK_SUCCESS;
}

static struct wsi_image *
wsi_win32_get_wsi_image(struct wsi_swapchain *drv_chain,
                        uint32_t image_index)
{
   struct wsi_win32_swapchain *chain =
      (struct wsi_win32_swapchain *) drv_chain;

   return &chain->images[image_index].base;
}

static void
wsi_win32_set_image_idle(struct wsi_win32_swapchain *chain,
                         struct wsi_win32_image *image)
{
   if (chain->present_path != WSI_WIN32_PRESENT_DXGI_D3D12)
      mtx_lock(&chain->acquire_mutex);

   image->state = WSI_IMAGE_IDLE;

   if (chain->present_path != WSI_WIN32_PRESENT_DXGI_D3D12) {
      u_cnd_monotonic_broadcast(&chain->acquire_cond);
      mtx_unlock(&chain->acquire_mutex);
   }
}

static VkResult
wsi_win32_release_images(struct wsi_swapchain *drv_chain,
                         uint32_t count, const uint32_t *indices)
{
   struct wsi_win32_swapchain *chain =
      (struct wsi_win32_swapchain *)drv_chain;

   if (chain->status == VK_ERROR_SURFACE_LOST_KHR)
      return chain->status;

   for (uint32_t i = 0; i < count; i++) {
      uint32_t index = indices[i];
      assert(index < chain->base.image_count);
      assert(chain->images[index].state == WSI_IMAGE_DRAWING);
      wsi_win32_set_image_idle(chain, &chain->images[index]);
   }

   return VK_SUCCESS;
}

static bool
wsi_win32_find_idle_image(struct wsi_win32_swapchain *chain,
                          uint32_t *out_image_index)
{
   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      if (chain->images[i].state == WSI_IMAGE_IDLE) {
         *out_image_index = i;
         chain->images[i].state = WSI_IMAGE_DRAWING;
         return true;
      }
   }
   return false;
}

static VkResult
wsi_win32_acquire_idle_cpu_image_locked(struct wsi_win32_swapchain *chain,
                                        const VkAcquireNextImageInfoKHR *info,
                                        uint32_t *out_image_index)
{
   if (wsi_win32_find_idle_image(chain, out_image_index))
      return VK_SUCCESS;

   if (info->timeout == 0)
      return VK_NOT_READY;

   const uint64_t abs_timeout = os_time_get_absolute_timeout(info->timeout);
   struct timespec abs_timespec;
   timespec_from_nsec(&abs_timespec, abs_timeout);
   do {
      int ret = u_cnd_monotonic_timedwait(
         &chain->acquire_cond, &chain->acquire_mutex, &abs_timespec);
      if (ret == thrd_timedout)
         return VK_TIMEOUT;
      else if (ret != thrd_success)
         return VK_ERROR_OUT_OF_DATE_KHR;
   } while (!wsi_win32_find_idle_image(chain, out_image_index));

   return VK_SUCCESS;
}

static inline VkResult
wsi_win32_acquire_idle_cpu_image(struct wsi_win32_swapchain *chain,
                                 const VkAcquireNextImageInfoKHR *info,
                                 uint32_t *out_image_index)
{
   mtx_lock(&chain->acquire_mutex);
   VkResult result = wsi_win32_acquire_idle_cpu_image_locked(chain, info,
                                                             out_image_index);
   mtx_unlock(&chain->acquire_mutex);
   return result;
}

static VkResult
wsi_win32_acquire_next_image(struct wsi_swapchain *drv_chain,
                             const VkAcquireNextImageInfoKHR *info,
                             uint32_t *image_index)
{
   struct wsi_win32_swapchain *chain =
      (struct wsi_win32_swapchain *)drv_chain;

   /* Bail early if the swapchain is broken */
   if (chain->status != VK_SUCCESS)
      return chain->status;

   if (chain->retired)
      return VK_ERROR_OUT_OF_DATE_KHR;

   /* acquire timeout has to be explicitly handled for sw wsi */
   if (chain->present_path != WSI_WIN32_PRESENT_DXGI_D3D12)
      return wsi_win32_acquire_idle_cpu_image(chain, info, image_index);

   if (wsi_win32_find_idle_image(chain, image_index))
      return VK_SUCCESS;

   assert(chain->dxgi);
   uint32_t index = chain->dxgi->GetCurrentBackBufferIndex();
   if (chain->images[index].state == WSI_IMAGE_DRAWING) {
      index = (index + 1) % chain->base.image_count;
      assert(chain->images[index].state == WSI_IMAGE_QUEUED);
   }
   if (chain->wsi->wsi->WaitForFences(chain->base.device, 1,
                                      &chain->base.fences[index],
                                      false, info->timeout) != VK_SUCCESS)
      return VK_TIMEOUT;

   *image_index = index;
   chain->images[index].state = WSI_IMAGE_DRAWING;
   return VK_SUCCESS;
}

static VkResult
wsi_win32_queue_present_dxgi(struct wsi_win32_swapchain *chain,
                             struct wsi_win32_image *image,
                             const VkPresentRegionKHR *damage)
{
   uint32_t rect_count = damage ? damage->rectangleCount : 0;
   STACK_ARRAY(RECT, rects, rect_count);

   for (uint32_t r = 0; r < rect_count; r++) {
      rects[r].left = damage->pRectangles[r].offset.x;
      rects[r].top = damage->pRectangles[r].offset.y;
      rects[r].right = damage->pRectangles[r].offset.x + damage->pRectangles[r].extent.width;
      rects[r].bottom = damage->pRectangles[r].offset.y + damage->pRectangles[r].extent.height;
   }

   DXGI_PRESENT_PARAMETERS params = {
      rect_count,
      rects,
   };

   image->state = WSI_IMAGE_QUEUED;
   UINT sync_interval = chain->base.present_mode == VK_PRESENT_MODE_FIFO_KHR ? 1 : 0;
   UINT present_flags = chain->wsi->dxgi.supports_tearing &&
                        chain->base.present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR ?
      DXGI_PRESENT_ALLOW_TEARING : 0;

   HRESULT hres = chain->dxgi->Present1(sync_interval, present_flags, &params);
   switch (hres) {
   case DXGI_ERROR_DEVICE_REMOVED: return VK_ERROR_DEVICE_LOST;
   case E_OUTOFMEMORY: return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   default:
      if (FAILED(hres))
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      break;
   }

   VkResult result = wsi_win32_surface_set_content(chain);
   if (result != VK_SUCCESS) {
      chain->status = result;
      return result;
   }

   return VK_SUCCESS;
}

#if defined(HAVE_YTTRIUM)
static VkResult
wsi_win32_shared_present_fail(struct wsi_win32_swapchain *chain,
                              struct wsi_win32_image *image,
                              VkResult result)
{
   wsi_win32_set_image_idle(chain, image);
   if (result < VK_SUCCESS)
      chain->status = result;
   return result;
}

static VkResult
wsi_win32_queue_present_dxgi_shared(struct wsi_win32_swapchain *chain,
                                    struct wsi_win32_image *image,
                                    uint32_t image_index,
                                    const VkPresentRegionKHR *damage)
{
   (void)damage;

   if (!chain->dxgi || !chain->d3d11_context || !image->d3d11.texture)
      return wsi_win32_shared_present_fail(
         chain, image, VK_ERROR_SURFACE_LOST_KHR);

   /* The shared D3D11 context cannot consume Venus synchronization objects.
    * Wait for the WSI submission which consumed the application's present
    * semaphores before copying the Vulkan image through DXGI_SHARED.
    */
   VkResult result = chain->wsi->wsi->WaitForFences(
      chain->base.device, 1, &chain->base.fences[image_index], true,
      UINT64_MAX);
   if (result != VK_SUCCESS)
      return wsi_win32_shared_present_fail(chain, image, result);

   ID3D11Texture2D *buffer = NULL;
   /* D3D10/11 rotates buffer 0 to the current render target for flip-model
    * swapchains.  GetCurrentBackBufferIndex is a D3D12 indexing convention
    * and can select the wrong buffer here.
    */
   HRESULT hres = chain->dxgi->GetBuffer(0, IID_PPV_ARGS(&buffer));
   if (FAILED(hres) || !buffer)
      return wsi_win32_shared_present_fail(
         chain, image, VK_ERROR_OUT_OF_DATE_KHR);

   chain->d3d11_context->CopyResource(buffer, image->d3d11.texture);
   buffer->Release();

   /* The whole image was copied, so there is no benefit in forwarding dirty
    * rectangles even though FLIP_SEQUENTIAL preserves the other buffers.
    */
   DXGI_PRESENT_PARAMETERS params = {};
   UINT sync_interval =
      chain->base.present_mode == VK_PRESENT_MODE_FIFO_KHR ? 1 : 0;
   UINT present_flags =
      chain->wsi->dxgi.supports_tearing &&
      chain->base.present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR ?
      DXGI_PRESENT_ALLOW_TEARING : 0;

   hres = chain->dxgi->Present1(sync_interval, present_flags, &params);
   switch (hres) {
   case DXGI_ERROR_DEVICE_REMOVED:
      return wsi_win32_shared_present_fail(
         chain, image, VK_ERROR_DEVICE_LOST);
   case E_OUTOFMEMORY:
      return wsi_win32_shared_present_fail(
         chain, image, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   default:
      if (FAILED(hres))
         return wsi_win32_shared_present_fail(
            chain, image, VK_ERROR_OUT_OF_HOST_MEMORY);
      break;
   }

   result = wsi_win32_surface_set_content(chain);
   if (result != VK_SUCCESS) {
      return wsi_win32_shared_present_fail(chain, image, result);
   }

   wsi_win32_set_image_idle(chain, image);
   return VK_SUCCESS;
}
#endif

static VkResult
wsi_win32_queue_present(struct wsi_swapchain *drv_chain,
                        uint32_t image_index,
                        uint64_t present_id,
                        const VkPresentRegionKHR *damage)
{
   struct wsi_win32_swapchain *chain = (struct wsi_win32_swapchain *) drv_chain;
   assert(image_index < chain->base.image_count);
   struct wsi_win32_image *image = &chain->images[image_index];

   assert(image->state == WSI_IMAGE_DRAWING);

   if (chain->present_path == WSI_WIN32_PRESENT_DXGI_D3D12)
      return wsi_win32_queue_present_dxgi(chain, image, damage);
#if defined(HAVE_YTTRIUM)
   if (chain->present_path == WSI_WIN32_PRESENT_DXGI_SHARED)
      return wsi_win32_queue_present_dxgi_shared(chain, image, image_index,
                                                  damage);
#endif

   char *ptr = (char *)image->base.cpu_map;
   char *dptr = (char *)image->sw.ppvBits;

   for (unsigned h = 0; h < chain->extent.height; h++) {
      memcpy(dptr, ptr, chain->extent.width * 4);
      dptr += image->sw.bmp_row_pitch;
      ptr += image->base.row_pitches[0];
   }
   VkResult result = VK_SUCCESS;
   mtx_lock(&chain->surface->mutex);

   if (chain->surface->current_swapchain != chain) {
      if (chain->retired) {
         result = VK_ERROR_OUT_OF_DATE_KHR;
      } else {
         /* DirectComposition content is above the HWND's GDI layer.  Clear
          * it before the first GDI blit from a replacement swapchain.
          */
         result = wsi_win32_surface_set_content_locked(chain->surface,
                                                        chain->wsi, NULL);
      }
   }

   if (result == VK_SUCCESS) {
      if (!StretchBlt(chain->chain_dc, 0, 0, chain->extent.width,
                      chain->extent.height, image->sw.dc, 0, 0,
                      chain->extent.width, chain->extent.height, SRCCOPY)) {
         result = VK_ERROR_MEMORY_MAP_FAILED;
      } else {
         chain->surface->current_swapchain = chain;
      }
   }

   mtx_unlock(&chain->surface->mutex);

   if (result < VK_SUCCESS)
      chain->status = result;

   wsi_win32_set_image_idle(chain, image);

   return chain->status;
}

static DXGI_ALPHA_MODE
wsi_win32_dxgi_alpha_mode(VkCompositeAlphaFlagBitsKHR composite_alpha)
{
   switch (composite_alpha) {
   case VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR:
      return DXGI_ALPHA_MODE_IGNORE;
   case VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR:
      return DXGI_ALPHA_MODE_PREMULTIPLIED;
   case VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR:
      return DXGI_ALPHA_MODE_STRAIGHT;
   default:
      return DXGI_ALPHA_MODE_UNSPECIFIED;
   }
}

static VkResult
wsi_win32_surface_create_swapchain_dxgi(
   wsi_win32_surface *surface,
   VkDevice device,
   struct wsi_win32 *wsi,
   const VkSwapchainCreateInfoKHR *create_info,
   struct wsi_win32_swapchain *chain)
{
   (void)surface;
   IDXGIFactory4 *factory = wsi->dxgi.factory;
   ID3D12CommandQueue *queue =
      (ID3D12CommandQueue *)wsi->wsi->win32.get_d3d12_command_queue(device);

   DXGI_ALPHA_MODE alpha_mode =
      wsi_win32_dxgi_alpha_mode(create_info->compositeAlpha);

   DXGI_SWAP_CHAIN_DESC1 desc = {
      create_info->imageExtent.width,
      create_info->imageExtent.height,
      DXGI_FORMAT_B8G8R8A8_UNORM,
      create_info->imageArrayLayers > 1,  // Stereo
      { 1 },                              // SampleDesc
      0,                                  // Usage (filled in below)
      create_info->minImageCount,
      DXGI_SCALING_STRETCH,
      DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
      alpha_mode,
      wsi->dxgi.supports_tearing &&
      chain->base.present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR ?
         DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0u
   };

   if (create_info->imageUsage &
       (VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))
      desc.BufferUsage |= DXGI_USAGE_SHADER_INPUT;

   if (create_info->imageUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
      desc.BufferUsage |= DXGI_USAGE_RENDER_TARGET_OUTPUT;

   IDXGISwapChain1 *swapchain1 = NULL;
   HRESULT hr = factory->CreateSwapChainForComposition(
      queue, &desc, NULL, &swapchain1);
   if (FAILED(hr))
      return VK_ERROR_INITIALIZATION_FAILED;

   hr = swapchain1->QueryInterface(&chain->dxgi);
   swapchain1->Release();
   if (FAILED(hr))
      return VK_ERROR_INITIALIZATION_FAILED;

   /* Attach the swapchain after its first successful Present1.  This keeps
    * replacement-chain creation transactional and avoids displaying an
    * uninitialized back buffer.
    */
   return VK_SUCCESS;
}

static HMODULE
wsi_win32_load_system_module(const WCHAR *module_name)
{
   WCHAR module_path[MAX_PATH];
   UINT system_dir_len =
      GetSystemDirectoryW(module_path, ARRAY_SIZE(module_path));
   if (!system_dir_len || system_dir_len >= ARRAY_SIZE(module_path))
      return NULL;

   size_t module_name_len = wcslen(module_name);
   if (module_name_len >= ARRAY_SIZE(module_path) - system_dir_len - 1)
      return NULL;

   module_path[system_dir_len++] = L'\\';
   memcpy(module_path + system_dir_len, module_name,
          (module_name_len + 1) * sizeof(*module_name));

   /* Applications may provide their own DXGI or D3D implementation, notably
    * DXVK.  WSI needs the native Windows modules and must not re-enter an
    * application's translation layer while its Vulkan instance is being
    * initialized.
    */
   return LoadLibraryExW(module_path, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
}

#if defined(HAVE_YTTRIUM)
static VkResult
wsi_win32_surface_create_swapchain_dxgi_shared(
   wsi_win32_surface *surface,
   VkDevice device,
   struct wsi_win32 *wsi,
   const VkSwapchainCreateInfoKHR *create_info,
   struct wsi_win32_swapchain *chain)
{
   (void)surface;
   (void)device;
   chain->d3d11_mod = wsi_win32_load_system_module(L"d3d11.dll");
   if (!chain->d3d11_mod) {
      mesa_logw("wsi/win32: DXGI_SHARED failed to load system d3d11.dll "
                "(error 0x%08lx)", (unsigned long)GetLastError());
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   PFN_D3D11_CREATE_DEVICE create_device =
      (PFN_D3D11_CREATE_DEVICE)GetProcAddress(chain->d3d11_mod,
                                              "D3D11CreateDevice");
   if (!create_device) {
      mesa_logw("wsi/win32: DXGI_SHARED could not resolve D3D11CreateDevice "
                "(error 0x%08lx)", (unsigned long)GetLastError());
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   static const D3D_FEATURE_LEVEL feature_levels[] = {
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_10_0,
   };

   /* Pair the device with the adapter the swapchain factory will use, rather
    * than whatever the default happens to be.
    */
   IDXGIAdapter1 *adapter = NULL;
   HRESULT hr = wsi->dxgi.factory->EnumAdapters1(0, &adapter);
   if (FAILED(hr) || !adapter) {
      mesa_logw("wsi/win32: DXGI_SHARED EnumAdapters1 failed "
                "(HRESULT 0x%08lx)", (unsigned long)hr);
      if (adapter)
         adapter->Release();
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   hr = create_device(adapter, D3D_DRIVER_TYPE_UNKNOWN, NULL,
                      D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                      feature_levels, ARRAY_SIZE(feature_levels),
                      D3D11_SDK_VERSION, &chain->d3d11_device,
                      NULL, &chain->d3d11_context);
   adapter->Release();
   if (FAILED(hr)) {
      mesa_logw("wsi/win32: DXGI_SHARED D3D11CreateDevice failed "
                "(HRESULT 0x%08lx)", (unsigned long)hr);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   DXGI_SWAP_CHAIN_DESC1 desc = {};
   desc.Width = create_info->imageExtent.width;
   desc.Height = create_info->imageExtent.height;
   desc.Format = wsi_dxgi_format_from_vk(create_info->imageFormat);
   desc.SampleDesc.Count = 1;
   desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
   desc.BufferCount = create_info->minImageCount;
   desc.Scaling = DXGI_SCALING_STRETCH;
   desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
   desc.AlphaMode = wsi_win32_dxgi_alpha_mode(create_info->compositeAlpha);
   desc.Flags = wsi->dxgi.supports_tearing &&
                chain->base.present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR ?
      DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0u;

   if (desc.Format == DXGI_FORMAT_UNKNOWN)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   IDXGISwapChain1 *swapchain = NULL;
   hr = wsi->dxgi.factory->CreateSwapChainForComposition(
      chain->d3d11_device, &desc, NULL, &swapchain);
   if (FAILED(hr)) {
      mesa_logw("wsi/win32: DXGI_SHARED CreateSwapChainForComposition failed "
                "(HRESULT 0x%08lx)", (unsigned long)hr);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   hr = swapchain->QueryInterface(IID_PPV_ARGS(&chain->dxgi));
   swapchain->Release();
   if (FAILED(hr)) {
      mesa_logw("wsi/win32: DXGI_SHARED IDXGISwapChain3 query failed "
                "(HRESULT 0x%08lx)", (unsigned long)hr);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   return VK_SUCCESS;
}
#endif

static VkResult
wsi_win32_surface_create_swapchain_internal(
   VkIcdSurfaceBase *icd_surface,
   VkDevice device,
   struct wsi_device *wsi_device,
   const VkSwapchainCreateInfoKHR *create_info,
   const VkAllocationCallbacks *allocator,
   struct wsi_swapchain **swapchain_out)
{
   wsi_win32_surface *surface = (wsi_win32_surface *)icd_surface;
   struct wsi_win32 *wsi =
      (struct wsi_win32 *) wsi_device->wsi[VK_ICD_WSI_PLATFORM_WIN32];

   assert(create_info->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);

   bool has_hardware_present =
      !wsi_device->sw &&
      (wsi_device->win32.get_d3d12_command_queue
#if defined(HAVE_YTTRIUM)
       || wsi_device->win32.create_image_memory_from_win32_handle
#endif
      );
   if (has_hardware_present && (!wsi->dxgi.factory || !wsi->dxgi.dcomp)) {
      mesa_loge("wsi/win32: hardware DXGI presentation is unavailable; "
                "refusing the GDI CPU fallback");
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   const unsigned num_images = create_info->minImageCount;
   struct wsi_win32_swapchain *chain;
   size_t size = sizeof(*chain) + num_images * sizeof(chain->images[0]);

   chain = (wsi_win32_swapchain *)vk_zalloc(allocator, size,
                     8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (chain == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   int ret = mtx_init(&chain->acquire_mutex, mtx_plain);
   if (ret != thrd_success) {
      vk_free(allocator, chain);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   ret = u_cnd_monotonic_init(&chain->acquire_cond);
   if (ret != thrd_success) {
      mtx_destroy(&chain->acquire_mutex);
      vk_free(allocator, chain);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   struct wsi_dxgi_image_params dxgi_image_params = {
      { WSI_IMAGE_TYPE_DXGI },
   };
   dxgi_image_params.storage_image = (create_info->imageUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;

#if defined(HAVE_YTTRIUM)
   struct wsi_dxgi_shared_image_params dxgi_shared_image_params = {
      { WSI_IMAGE_TYPE_DXGI_SHARED },
   };
#endif

   struct wsi_cpu_image_params cpu_image_params = {
      { WSI_IMAGE_TYPE_CPU },
   };

   bool supports_dxgi_d3d12 = wsi->dxgi.factory &&
                              wsi->dxgi.dcomp &&
                              wsi->wsi->win32.get_d3d12_command_queue;
#if defined(HAVE_YTTRIUM)
   bool supports_dxgi_shared =
      wsi->dxgi.factory &&
      wsi->dxgi.dcomp &&
      wsi->wsi->win32.create_image_memory_from_win32_handle;
#endif
   struct wsi_base_image_params *image_params =
      supports_dxgi_d3d12 ? &dxgi_image_params.base :
#if defined(HAVE_YTTRIUM)
      supports_dxgi_shared ? &dxgi_shared_image_params.base :
#endif
      &cpu_image_params.base;

   VkResult result = wsi_swapchain_init(wsi_device, &chain->base, device,
                                        create_info, image_params,
                                        allocator);
   if (result != VK_SUCCESS) {
      u_cnd_monotonic_destroy(&chain->acquire_cond);
      mtx_destroy(&chain->acquire_mutex);
      vk_free(allocator, chain);
      return result;
   }

   chain->base.destroy = wsi_win32_swapchain_destroy;
   chain->base.get_wsi_image = wsi_win32_get_wsi_image;
   chain->base.acquire_next_image = wsi_win32_acquire_next_image;
   chain->base.release_images = wsi_win32_release_images;
   chain->base.queue_present = wsi_win32_queue_present;
   chain->base.present_mode = wsi_swapchain_get_present_mode(wsi_device, create_info);
   chain->extent = create_info->imageExtent;

   chain->wsi = wsi;
   chain->status = VK_SUCCESS;

   chain->surface = surface;

   if (image_params->image_type == WSI_IMAGE_TYPE_DXGI) {
      chain->present_path = WSI_WIN32_PRESENT_DXGI_D3D12;
      result = wsi_win32_surface_create_swapchain_dxgi(surface, device, wsi, create_info, chain);
      if (result != VK_SUCCESS)
         goto fail;
#if defined(HAVE_YTTRIUM)
   } else if (image_params->image_type == WSI_IMAGE_TYPE_DXGI_SHARED) {
      chain->present_path = WSI_WIN32_PRESENT_DXGI_SHARED;
      result = wsi_win32_surface_create_swapchain_dxgi_shared(
         surface, device, wsi, create_info, chain);
      if (result != VK_SUCCESS)
         goto fail;
#endif
   } else {
      chain->present_path = WSI_WIN32_PRESENT_GDI;
   }

   for (uint32_t image = 0; image < num_images; image++) {
      result = wsi_win32_image_init(device, chain,
                                    create_info, allocator,
                                    &chain->images[image]);
      if (result != VK_SUCCESS)
         goto fail;

      chain->base.image_count++;
   }

   *swapchain_out = &chain->base;

   return VK_SUCCESS;

fail:
   wsi_win32_swapchain_destroy(&chain->base, allocator);
   return result;
}

static VkResult
wsi_win32_surface_create_swapchain(
   VkIcdSurfaceBase *icd_surface,
   VkDevice device,
   struct wsi_device *wsi_device,
   const VkSwapchainCreateInfoKHR *create_info,
   const VkAllocationCallbacks *allocator,
   struct wsi_swapchain **swapchain_out)
{
   struct wsi_win32_surface *surface =
      (struct wsi_win32_surface *)icd_surface;
   VkResult result = VK_SUCCESS;

   mtx_lock(&surface->mutex);
   if (surface->creating_swapchain) {
      result = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   } else if (create_info->oldSwapchain) {
      VK_FROM_HANDLE(wsi_swapchain, old_base, create_info->oldSwapchain);
      struct wsi_win32_swapchain *old_chain =
         (struct wsi_win32_swapchain *)old_base;

      if (old_chain->surface != surface ||
          surface->active_swapchain != old_chain) {
         result = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
      } else {
         /* Vulkan retires oldSwapchain even when replacement creation later
          * fails.  Already-acquired images may still be presented, but no
          * further acquisitions are allowed.
          */
         old_chain->retired = true;
         surface->active_swapchain = NULL;
      }
   } else if (surface->active_swapchain) {
      result = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   }

   if (result == VK_SUCCESS)
      surface->creating_swapchain = true;
   mtx_unlock(&surface->mutex);

   if (result != VK_SUCCESS)
      return result;

   result = wsi_win32_surface_create_swapchain_internal(
      icd_surface, device, wsi_device, create_info, allocator, swapchain_out);

   mtx_lock(&surface->mutex);
   surface->creating_swapchain = false;
   if (result == VK_SUCCESS)
      surface->active_swapchain =
         (struct wsi_win32_swapchain *)*swapchain_out;
   mtx_unlock(&surface->mutex);

   return result;
}

static IDXGIFactory4 *
dxgi_get_factory(bool debug)
{
   HMODULE dxgi_mod = wsi_win32_load_system_module(L"DXGI.DLL");
   if (!dxgi_mod) {
      return NULL;
   }

   typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY2)(UINT flags, REFIID riid, void **ppFactory);
   PFN_CREATE_DXGI_FACTORY2 CreateDXGIFactory2;

   CreateDXGIFactory2 = (PFN_CREATE_DXGI_FACTORY2)GetProcAddress(dxgi_mod, "CreateDXGIFactory2");
   if (!CreateDXGIFactory2) {
      return NULL;
   }

   UINT flags = 0;
   if (debug)
      flags |= DXGI_CREATE_FACTORY_DEBUG;

   IDXGIFactory4 *factory;
   HRESULT hr = CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory));
   if (FAILED(hr)) {
      return NULL;
   }

   return factory;
}

static IDCompositionDevice *
dcomp_get_device()
{
   HMODULE dcomp_mod = wsi_win32_load_system_module(L"DComp.DLL");
   if (!dcomp_mod) {
      return NULL;
   }

   typedef HRESULT (STDAPICALLTYPE *PFN_DCOMP_CREATE_DEVICE)(IDXGIDevice *, REFIID, void **);
   PFN_DCOMP_CREATE_DEVICE DCompositionCreateDevice;

   DCompositionCreateDevice = (PFN_DCOMP_CREATE_DEVICE)GetProcAddress(dcomp_mod, "DCompositionCreateDevice");
   if (!DCompositionCreateDevice) {
      return NULL;
   }

   IDCompositionDevice *device;
   HRESULT hr = DCompositionCreateDevice(NULL, IID_PPV_ARGS(&device));
   if (FAILED(hr)) {
      return NULL;
   }

   return device;
}

VkResult
wsi_win32_init_wsi(struct wsi_device *wsi_device,
                   const VkAllocationCallbacks *alloc,
                   VkPhysicalDevice physical_device)
{
   struct wsi_win32 *wsi;
   VkResult result;

   wsi = (wsi_win32 *)vk_zalloc(alloc, sizeof(*wsi), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!wsi) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }

   wsi->physical_device = physical_device;
   wsi->alloc = alloc;
   wsi->wsi = wsi_device;

   if (!wsi_device->sw) {
      wsi->dxgi.factory = dxgi_get_factory(WSI_DEBUG & WSI_DEBUG_DXGI);
      if (!wsi->dxgi.factory) {
         vk_free(alloc, wsi);
         result = VK_ERROR_INITIALIZATION_FAILED;
         goto fail;
      }

      IDXGIFactory5 *factory5 = NULL;
      BOOL supports_tearing = FALSE;
      HRESULT hr = wsi->dxgi.factory->QueryInterface(IID_PPV_ARGS(&factory5));
      if (SUCCEEDED(hr)) {
         hr = factory5->CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING, &supports_tearing,
            sizeof(supports_tearing));
         factory5->Release();
      }
      wsi->dxgi.supports_tearing = SUCCEEDED(hr) && supports_tearing;

      /* DirectComposition is required by both hardware DXGI paths.  Keep WSI
       * initialization alive for a true software device, but hardware
       * swapchain creation will fail rather than fall back to GDI.
       */
      wsi->dxgi.dcomp = dcomp_get_device();
      if (!wsi->dxgi.dcomp)
         mesa_logw("wsi/win32: no DirectComposition device; "
                   "the DXGI presentation paths are unavailable");
   }

   wsi->base.get_support = wsi_win32_surface_get_support;
   wsi->base.get_capabilities2 = wsi_win32_surface_get_capabilities2;
   wsi->base.get_formats = wsi_win32_surface_get_formats;
   wsi->base.get_formats2 = wsi_win32_surface_get_formats2;
   wsi->base.get_present_modes = wsi_win32_surface_get_present_modes;
   wsi->base.get_present_rectangles = wsi_win32_surface_get_present_rectangles;
   wsi->base.create_swapchain = wsi_win32_surface_create_swapchain;

   wsi_device->wsi[VK_ICD_WSI_PLATFORM_WIN32] = &wsi->base;

   return VK_SUCCESS;

fail:
   wsi_device->wsi[VK_ICD_WSI_PLATFORM_WIN32] = NULL;

   return result;
}

void
wsi_win32_finish_wsi(struct wsi_device *wsi_device,
                  const VkAllocationCallbacks *alloc)
{
   struct wsi_win32 *wsi =
      (struct wsi_win32 *)wsi_device->wsi[VK_ICD_WSI_PLATFORM_WIN32];
   if (!wsi)
      return;

   if (wsi->dxgi.factory)
      wsi->dxgi.factory->Release();
   if (wsi->dxgi.dcomp)
      wsi->dxgi.dcomp->Release();

   vk_free(alloc, wsi);
}
