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
 * OutputMerger.cpp --
 *    Functions that manipulate the output merger state.
 */


#include "OutputMerger.h"
#include "Shader.h"
#include "State.h"

#include "Debug.h"
#include "Format.h"

#include "util/u_framebuffer.h"
#include "util/format/u_format.h"
#include "util/blend.h"
#include "util/u_debug.h"
#include "util/u_math.h"

#include <stdlib.h>
#include <string.h>

extern "C" void
yttrium_gdi_resource_debug_log(struct pipe_resource *resource,
                               const char *label);

/*
 * debug_get_bool_option() calls getenv() every time, and this knob is read
 * from GetPipeRenderTargetView(), which runs per render target per
 * SetRenderTargets and on every clear.  It showed up in the profile as a
 * per-draw walk of the environment block to re-answer a constant.
 */
DEBUG_GET_ONCE_BOOL_OPTION(rt_trace, "D3D10UMD_DEBUG_RT_TRACE", false)

static inline struct pipe_surface *
GetPipeRenderTargetView(D3D10DDI_HRENDERTARGETVIEW hRenderTargetView)
{
   RenderTargetView *pRenderTargetView = CastRenderTargetView(hRenderTargetView);
   if (!pRenderTargetView) return NULL;

   // Validate that surface texture matches resource 
   // If it is not (resource rotated) then recreate surface
   struct pipe_surface *currentSurface = &pRenderTargetView->surface;
   Resource *res = pRenderTargetView->resource;
   if (currentSurface->texture != res->resource) {
      if (debug_get_option_rt_trace()) {
         debug_printf("d3d10umd: rtv update view=%p old=%p new=%p resource=%p\n",
                      pRenderTargetView,
                      currentSurface->texture,
                      res->resource,
                      res);
         yttrium_gdi_resource_debug_log(res->resource, "RTV update");
      }
      pipe_resource_reference(&currentSurface->texture, res->resource);
   }
   return &pRenderTargetView->surface;
}

static bool
IsYttriumScreen(struct pipe_screen *screen)
{
   if (!screen || !screen->get_name)
      return false;

   const char *name = screen->get_name(screen);
   return name && strcmp(name, "yttrium") == 0;
}

static unsigned
GetForcedSampleCount(Device *pDevice)
{
   RasterizerState *rasterizer =
      pDevice ? pDevice->rasterizer_state : NULL;

   if (!rasterizer || rasterizer->forced_sample_count <= 1)
      return 0;

   return rasterizer->forced_sample_count;
}

static void
ApplyForcedSampleCount(struct pipe_surface *surface,
                       unsigned forced_sample_count)
{
   if (!surface || !surface->texture)
      return;

   surface->nr_samples =
      forced_sample_count && surface->texture->nr_samples <= 1 ?
      forced_sample_count : 0;
}

void
UpdateFramebufferForcedSampleCount(Device *pDevice)
{
   if (!pDevice)
      return;

   const unsigned forced_sample_count = GetForcedSampleCount(pDevice);

   for (unsigned i = 0; i < pDevice->fb.nr_cbufs; ++i)
      ApplyForcedSampleCount(&pDevice->fb.cbufs[i], forced_sample_count);
   ApplyForcedSampleCount(&pDevice->fb.zsbuf, forced_sample_count);

   pDevice->fb.samples = forced_sample_count ? forced_sample_count : 1;
}

static void
ConvertClearColor(enum pipe_format format,
                  const FLOAT color[4],
                  union pipe_color_union *clear_color)
{
   /*
    * DX10 always uses float clear color but gallium does not.
    * Conversion should just be ordinary conversion. Actual clamping will
    * be done later but need to make sure values exceeding int/uint range
    * are handled correctly.
    */
   if (util_format_is_pure_integer(format)) {
      if (util_format_is_pure_sint(format)) {
         unsigned i;
         int min_int32 = 0x80000000;
         int max_int32 = 0x7fffffff;
         for (i = 0; i < 4; i++) {
            float value = color[i];
            if (util_is_nan(value)) {
               clear_color->i[i] = 0;
            } else if (value <= (float)min_int32) {
               clear_color->i[i] = min_int32;
            } else if (value >= (float)max_int32) {
               clear_color->i[i] = max_int32;
            } else {
               clear_color->i[i] = value;
            }
         }
      } else {
         assert(util_format_is_pure_uint(format));
         unsigned i;
         unsigned max_uint32 = 0xffffffffU;
         for (i = 0; i < 4; i++) {
            float value = color[i];
            if (!(value >= 0.0f)) {
               clear_color->ui[i] = 0;
            } else if (value >= (float)max_uint32) {
               clear_color->ui[i] = max_uint32;
            } else {
               clear_color->ui[i] = value;
            }
         }
      }
   } else {
      clear_color->f[0] = color[0];
      clear_color->f[1] = color[1];
      clear_color->f[2] = color[2];
      clear_color->f[3] = color[3];
   }
}

static bool
ClearRenderTargetViewRectUpload(struct pipe_context *pipe,
                                struct pipe_surface *surface,
                                const union pipe_color_union *clear_color,
                                unsigned x,
                                unsigned y,
                                unsigned width,
                                unsigned height)
{
   if (!pipe || !pipe->texture_subdata || !surface || !surface->texture ||
       !clear_color || !width || !height)
      return false;

   const enum pipe_format format = surface->format;
   const struct util_format_pack_description *pack =
      util_format_pack_description(format);
   if (!pack)
      return false;

   const bool pure_uint = util_format_is_pure_uint(format);
   const bool pure_sint = util_format_is_pure_sint(format);
   if ((pure_uint && !pack->pack_rgba_uint) ||
       (pure_sint && !pack->pack_rgba_sint) ||
       (!pure_uint && !pure_sint && !pack->pack_rgba_float))
      return false;

   const unsigned stride = util_format_get_stride(format, width);
   const uintptr_t layer_stride =
      (uintptr_t)util_format_get_2d_size(format, stride, height);
   const unsigned first_layer = surface->first_layer;
   const unsigned layer_count =
      surface->last_layer >= surface->first_layer ?
      surface->last_layer - surface->first_layer + 1 : 1;

   if (!stride || !layer_stride || layer_stride > SIZE_MAX / layer_count)
      return false;

   uint8_t *data = (uint8_t *)malloc((size_t)layer_stride * layer_count);
   if (!data)
      return false;

   uint8_t *row = (uint8_t *)malloc(stride);
   void *src = NULL;
   if (!row)
      goto fail;

   if (pure_uint) {
      uint32_t *rgba = (uint32_t *)malloc(width * 4 * sizeof(*rgba));
      if (!rgba)
         goto fail;
      for (unsigned i = 0; i < width; ++i) {
         rgba[i * 4 + 0] = clear_color->ui[0];
         rgba[i * 4 + 1] = clear_color->ui[1];
         rgba[i * 4 + 2] = clear_color->ui[2];
         rgba[i * 4 + 3] = clear_color->ui[3];
      }
      pack->pack_rgba_uint(row, 0, rgba, 0, width, 1);
      src = rgba;
   } else if (pure_sint) {
      int32_t *rgba = (int32_t *)malloc(width * 4 * sizeof(*rgba));
      if (!rgba)
         goto fail;
      for (unsigned i = 0; i < width; ++i) {
         rgba[i * 4 + 0] = clear_color->i[0];
         rgba[i * 4 + 1] = clear_color->i[1];
         rgba[i * 4 + 2] = clear_color->i[2];
         rgba[i * 4 + 3] = clear_color->i[3];
      }
      pack->pack_rgba_sint(row, 0, rgba, 0, width, 1);
      src = rgba;
   } else {
      float *rgba = (float *)malloc(width * 4 * sizeof(*rgba));
      if (!rgba)
         goto fail;
      for (unsigned i = 0; i < width; ++i) {
         rgba[i * 4 + 0] = clear_color->f[0];
         rgba[i * 4 + 1] = clear_color->f[1];
         rgba[i * 4 + 2] = clear_color->f[2];
         rgba[i * 4 + 3] = clear_color->f[3];
      }
      pack->pack_rgba_float(row, 0, rgba, 0, width, 1);
      src = rgba;
   }

   for (unsigned layer = 0; layer < layer_count; ++layer) {
      uint8_t *layer_data = data + (size_t)layer_stride * layer;
      for (unsigned row_index = 0; row_index < height; ++row_index)
         memcpy(layer_data + (size_t)stride * row_index, row, stride);
   }

   struct pipe_box box;
   box.x = x;
   box.y = y;
   box.z = first_layer;
   box.width = width;
   box.height = height;
   box.depth = layer_count;
   pipe->texture_subdata(pipe, surface->texture, surface->level, 0, &box,
                         data, stride, layer_stride);

   free(src);
   free(row);
   free(data);
   return true;

fail:
   free(src);
   free(row);
   free(data);
   return false;
}

/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateRenderTargetViewSize --
 *
 *    The CalcPrivateRenderTargetViewSize function determines the size
 *    of the user-mode display driver's private region of memory
 *    (that is, the size of internal driver structures, not the size
 *    of the resource video memory) for a render target view.
 *
 * ----------------------------------------------------------------------
 */


SIZE_T APIENTRY
CalcPrivateRenderTargetViewSize(
   D3D10DDI_HDEVICE hDevice,                                               // IN
   __in const D3D10DDIARG_CREATERENDERTARGETVIEW *pCreateRenderTargetView) // IN
{
   return sizeof(RenderTargetView);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateRenderTargetView --
 *
 *    The CreateRenderTargetView function creates a render target view.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateRenderTargetView(
   D3D10DDI_HDEVICE hDevice,                                               // IN
   __in const D3D10DDIARG_CREATERENDERTARGETVIEW *pCreateRenderTargetView, // IN
   D3D10DDI_HRENDERTARGETVIEW hRenderTargetView,                           // IN
   D3D10DDI_HRTRENDERTARGETVIEW hRTRenderTargetView)                       // IN
{
   LOG_ENTRYPOINT();

   struct pipe_resource *resource = CastPipeResource(pCreateRenderTargetView->hDrvResource);
   RenderTargetView *pRTView = CastRenderTargetView(hRenderTargetView);
   pRTView->resource = CastResource(pCreateRenderTargetView->hDrvResource);

   if (debug_get_option_rt_trace()) {
      debug_printf("d3d10umd: CreateRenderTargetView view=%p hDrvResource=%p resource=%p pipe=%p format=%u dim=%u\n",
                   pRTView,
                   (void *)pCreateRenderTargetView->hDrvResource.pDrvPrivate,
                   pRTView->resource,
                   resource,
                   pCreateRenderTargetView->Format,
                   pCreateRenderTargetView->ResourceDimension);
      yttrium_gdi_resource_debug_log(resource, "CreateRenderTargetView");
   }

   struct pipe_surface desc;

   memset(&desc, 0, sizeof desc);
   pipe_resource_reference(&desc.texture, resource);
   desc.format = FormatTranslate(pCreateRenderTargetView->Format, false);

   switch (pCreateRenderTargetView->ResourceDimension) {
   case D3D10DDIRESOURCE_BUFFER:
      desc.level = 0;
      desc.first_layer = 0;
      desc.last_layer = 0;
      break;
   case D3D10DDIRESOURCE_TEXTURE1D:
      ASSERT(pCreateRenderTargetView->Tex1D.ArraySize != (UINT)-1);
      desc.level = pCreateRenderTargetView->Tex1D.MipSlice;
      desc.first_layer = pCreateRenderTargetView->Tex1D.FirstArraySlice;
      desc.last_layer = pCreateRenderTargetView->Tex1D.ArraySize - 1 +
                                 desc.first_layer;
      break;
   case D3D10DDIRESOURCE_TEXTURE2D:
      ASSERT(pCreateRenderTargetView->Tex2D.ArraySize != (UINT)-1);
      desc.level = pCreateRenderTargetView->Tex2D.MipSlice;
      desc.first_layer = pCreateRenderTargetView->Tex2D.FirstArraySlice;
      desc.last_layer = pCreateRenderTargetView->Tex2D.ArraySize - 1 +
                                 desc.first_layer;
      break;
   case D3D10DDIRESOURCE_TEXTURE3D:
      desc.level = pCreateRenderTargetView->Tex3D.MipSlice;
      desc.first_layer = pCreateRenderTargetView->Tex3D.FirstW;
      if (pCreateRenderTargetView->Tex3D.WSize == (UINT)-1) {
         const unsigned depth = resource ?
            u_minify(resource->depth0, desc.level) : 1;
         desc.last_layer = depth - 1;
      } else {
         desc.last_layer = pCreateRenderTargetView->Tex3D.WSize - 1 +
                                    desc.first_layer;
      }
      break;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      ASSERT(pCreateRenderTargetView->TexCube.ArraySize != (UINT)-1);
      desc.level = pCreateRenderTargetView->TexCube.MipSlice;
      desc.first_layer = pCreateRenderTargetView->TexCube.FirstArraySlice;
      desc.last_layer = pCreateRenderTargetView->TexCube.ArraySize - 1 +
                                 desc.first_layer;
      break;
   default:
      ASSERT(0);
      return;
   }

   pRTView->surface = desc;
   ResourceEvent(RESOURCE_EVENT_RTV_CREATE,
                 (uint64_t)hRTRenderTargetView.handle,
                 pRTView,
                 pRTView->surface.texture,
                 PipeResourceRefCount(pRTView->surface.texture),
                 pCreateRenderTargetView->Format,
                 pCreateRenderTargetView->ResourceDimension,
                 (uint64_t)(uintptr_t)pRTView->resource);
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyRenderTargetView --
 *
 *    The DestroyRenderTargetView function destroys the specified
 *    render target view object. The render target view object can
 *    be destoyed only if it is not currently bound to a display device.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroyRenderTargetView(D3D10DDI_HDEVICE hDevice,                       // IN
                        D3D10DDI_HRENDERTARGETVIEW hRenderTargetView)   // IN
{
   LOG_ENTRYPOINT();

   RenderTargetView *pRTView = CastRenderTargetView(hRenderTargetView);

   ResourceEvent(RESOURCE_EVENT_RTV_DESTROY,
                 0,
                 pRTView,
                 pRTView ? pRTView->surface.texture : NULL,
                 pRTView ? PipeResourceRefCount(pRTView->surface.texture) : 0,
                 0, 0,
                 (uint64_t)(uintptr_t)(pRTView ? pRTView->resource : NULL));
   pipe_resource_reference(&pRTView->surface.texture, NULL);
}


/*
 * ----------------------------------------------------------------------
 *
 * ClearRenderTargetView --
 *
 *    The ClearRenderTargetView function clears the specified
 *    render target view by setting it to a constant value.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ClearRenderTargetView(D3D10DDI_HDEVICE hDevice,                      // IN
                      D3D10DDI_HRENDERTARGETVIEW hRenderTargetView,  // IN
                      FLOAT pColorRGBA[4])                           // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   struct pipe_surface *surface = GetPipeRenderTargetView(hRenderTargetView);
   union pipe_color_union clear_color;

   ConvertClearColor(surface->format, pColorRGBA, &clear_color);

   pipe->clear_render_target(pipe,
                             surface,
                             &clear_color,
                             0, 0,
                             pipe_surface_width(surface),
                             pipe_surface_height(surface),
                             true);
}

void APIENTRY
ClearView(D3D10DDI_HDEVICE hDevice,
          D3D11DDI_HANDLETYPE ViewType,
          void *hView,
          const FLOAT Color[4],
          const D3D10_DDI_RECT *pRects,
          UINT NumRects)
{
   LOG_ENTRYPOINT();

   if (ViewType != D3D10DDI_HT_RENDERTARGETVIEW || !hView || !Color)
      return;

   D3D10DDI_HRENDERTARGETVIEW hRenderTargetView;
   hRenderTargetView.pDrvPrivate = hView;

   struct pipe_context *pipe = CastPipeContext(hDevice);
   struct pipe_surface *surface = GetPipeRenderTargetView(hRenderTargetView);
   if (!pipe || !surface)
      return;

   union pipe_color_union clear_color;
   ConvertClearColor(surface->format, Color, &clear_color);

   unsigned surface_width = pipe_surface_width(surface);
   unsigned surface_height = pipe_surface_height(surface);
   const bool yttrium = IsYttriumScreen(pipe->screen);
   if (!pRects || !NumRects) {
      pipe->clear_render_target(pipe, surface, &clear_color, 0, 0,
                                surface_width, surface_height, true);
      return;
   }

   for (UINT i = 0; i < NumRects; ++i) {
      int left = MAX2(pRects[i].left, 0);
      int top = MAX2(pRects[i].top, 0);
      int right = MIN2(pRects[i].right, (int)surface_width);
      int bottom = MIN2(pRects[i].bottom, (int)surface_height);

      if (right <= left || bottom <= top)
         continue;

      unsigned width = (unsigned)(right - left);
      unsigned height = (unsigned)(bottom - top);
      if (yttrium) {
         pipe->clear_render_target(pipe, surface, &clear_color,
                                   (unsigned)left, (unsigned)top,
                                   width, height, true);
         continue;
      }

      if (!ClearRenderTargetViewRectUpload(pipe, surface, &clear_color,
                                           (unsigned)left, (unsigned)top,
                                           width, height)) {
         pipe->clear_render_target(pipe, surface, &clear_color,
                                   (unsigned)left, (unsigned)top,
                                   width, height, true);
      }
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateDepthStencilViewSize --
 *
 *    The CalcPrivateDepthStencilViewSize function determines the size
 *    of the user-mode display driver's private region of memory
 *    (that is, the size of internal driver structures, not the size
 *    of the resource video memory) for a depth stencil view.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateDepthStencilViewSize(
   D3D10DDI_HDEVICE hDevice,                                               // IN
   __in const D3D10DDIARG_CREATEDEPTHSTENCILVIEW *pCreateDepthStencilView) // IN
{
   return sizeof(DepthStencilView);
}

SIZE_T APIENTRY
CalcPrivateDepthStencilViewSize11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEDEPTHSTENCILVIEW *pCreateDepthStencilView)
{
   return sizeof(DepthStencilView);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateDepthStencilView --
 *
 *    The CreateDepthStencilView function creates a depth stencil view.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateDepthStencilView(
   D3D10DDI_HDEVICE hDevice,                                               // IN
   __in const D3D10DDIARG_CREATEDEPTHSTENCILVIEW *pCreateDepthStencilView, // IN
   D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView,                           // IN
   D3D10DDI_HRTDEPTHSTENCILVIEW hRTDepthStencilView)                       // IN
{
   LOG_ENTRYPOINT();

   struct pipe_resource *resource = CastPipeResource(pCreateDepthStencilView->hDrvResource);
   DepthStencilView *pDSView = CastDepthStencilView(hDepthStencilView);

   struct pipe_surface desc;

   memset(&desc, 0, sizeof desc);
   pipe_resource_reference(&desc.texture, resource);
   desc.format = FormatTranslate(pCreateDepthStencilView->Format, true);
   if (FormatFallback(desc.format) == resource->format)
      desc.format = resource->format;

   switch (pCreateDepthStencilView->ResourceDimension) {
   case D3D10DDIRESOURCE_TEXTURE1D:
      ASSERT(pCreateDepthStencilView->Tex1D.ArraySize != (UINT)-1);
      desc.level = pCreateDepthStencilView->Tex1D.MipSlice;
      desc.first_layer = pCreateDepthStencilView->Tex1D.FirstArraySlice;
      desc.last_layer = pCreateDepthStencilView->Tex1D.ArraySize - 1 +
                                 desc.first_layer;
      break;
   case D3D10DDIRESOURCE_TEXTURE2D:
      ASSERT(pCreateDepthStencilView->Tex2D.ArraySize != (UINT)-1);
      desc.level = pCreateDepthStencilView->Tex2D.MipSlice;
      desc.first_layer = pCreateDepthStencilView->Tex2D.FirstArraySlice;
      desc.last_layer = pCreateDepthStencilView->Tex2D.ArraySize - 1 +
                                 desc.first_layer;
      break;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      ASSERT(pCreateDepthStencilView->TexCube.ArraySize != (UINT)-1);
      desc.level = pCreateDepthStencilView->TexCube.MipSlice;
      desc.first_layer = pCreateDepthStencilView->TexCube.FirstArraySlice;
      desc.last_layer = pCreateDepthStencilView->TexCube.ArraySize - 1 +
                                 desc.first_layer;
      break;
   default:
      ASSERT(0);
      return;
   }

   pDSView->surface = desc;
   ResourceEvent(RESOURCE_EVENT_DSV_CREATE,
                 (uint64_t)hRTDepthStencilView.handle,
                 pDSView,
                 pDSView->surface.texture,
                 PipeResourceRefCount(pDSView->surface.texture),
                 pCreateDepthStencilView->Format,
                 pCreateDepthStencilView->ResourceDimension,
                 0);
}

void APIENTRY
CreateDepthStencilView11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEDEPTHSTENCILVIEW *pCreateDepthStencilView,
   D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView,
   D3D10DDI_HRTDEPTHSTENCILVIEW hRTDepthStencilView)
{
   D3D10DDIARG_CREATEDEPTHSTENCILVIEW create10 = {};

   create10.hDrvResource = pCreateDepthStencilView->hDrvResource;
   create10.Format = pCreateDepthStencilView->Format;
   create10.ResourceDimension = pCreateDepthStencilView->ResourceDimension;
   switch (pCreateDepthStencilView->ResourceDimension) {
   case D3D10DDIRESOURCE_TEXTURE1D:
      create10.Tex1D = pCreateDepthStencilView->Tex1D;
      break;
   case D3D10DDIRESOURCE_TEXTURE2D:
      create10.Tex2D = pCreateDepthStencilView->Tex2D;
      break;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      create10.TexCube = pCreateDepthStencilView->TexCube;
      break;
   default:
      break;
   }

   CreateDepthStencilView(hDevice, &create10, hDepthStencilView,
                          hRTDepthStencilView);
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyDepthStencilView --
 *
 *    The DestroyDepthStencilView function destroys the specified
 *    depth stencil view object. The depth stencil view object can
 *    be destoyed only if it is not currently bound to a display device.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroyDepthStencilView(D3D10DDI_HDEVICE hDevice,                       // IN
                        D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView)   // IN
{
   LOG_ENTRYPOINT();

   DepthStencilView *pDSView = CastDepthStencilView(hDepthStencilView);

   ResourceEvent(RESOURCE_EVENT_DSV_DESTROY,
                 0,
                 pDSView,
                 pDSView ? pDSView->surface.texture : NULL,
                 pDSView ? PipeResourceRefCount(pDSView->surface.texture) : 0,
                 0, 0, 0);
   pipe_resource_reference(&pDSView->surface.texture, NULL);
}


/*
 * ----------------------------------------------------------------------
 *
 * ClearDepthStencilView --
 *
 *    The ClearDepthStencilView function clears the specified
 *    currently bound depth-stencil view.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ClearDepthStencilView(D3D10DDI_HDEVICE hDevice,                      // IN
                      D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView,  // IN
                      UINT Flags,                                    // IN
                      FLOAT Depth,                                   // IN
                      UINT8 Stencil)                                 // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   struct pipe_surface *surface = CastPipeDepthStencilView(hDepthStencilView);

   unsigned flags = 0;
   if (Flags & D3D10_DDI_CLEAR_DEPTH) {
      flags |= PIPE_CLEAR_DEPTH;
   }
   if (Flags & D3D10_DDI_CLEAR_STENCIL) {
      flags |= PIPE_CLEAR_STENCIL;
   }

   const struct util_format_description *format_desc =
      surface ? util_format_description(surface->format) : NULL;
   if (format_desc) {
      if (!util_format_has_depth(format_desc)) {
         flags &= ~PIPE_CLEAR_DEPTH;
      }
      if (!util_format_has_stencil(format_desc)) {
         flags &= ~PIPE_CLEAR_STENCIL;
      }
   }

   pipe->clear_depth_stencil(pipe,
                             surface,
                             flags,
                             Depth,
                             Stencil,
                             0, 0,
                             pipe_surface_width(surface),
                             pipe_surface_height(surface),
                             true);
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateBlendStateSize --
 *
 *    The CalcPrivateBlendStateSize function determines the size of
 *    the user-mode display driver's private region of memory (that
 *    is, the size of internal driver structures, not the size of
 *    the resource video memory) for a blend state.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateBlendStateSize(D3D10DDI_HDEVICE hDevice,                     // IN
                          __in const D3D10_DDI_BLEND_DESC *pBlendDesc)  // IN
{
   return sizeof(BlendState);
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateBlendStateSize1 --
 *
 *    The CalcPrivateBlendStateSize1 function determines the size of
 *    the user-mode display driver's private region of memory (that
 *    is, the size of internal driver structures, not the size of
 *    the resource video memory) for a blend state.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateBlendStateSize1(D3D10DDI_HDEVICE hDevice,                     // IN
                           __in const D3D10_1_DDI_BLEND_DESC *pBlendDesc)  // IN
{
   return sizeof(BlendState);
}


/*
 * ----------------------------------------------------------------------
 *
 * translateBlendOp --
 *
 *   Translate blend function from svga3d to gallium representation.
 *
 * ----------------------------------------------------------------------
 */
static uint
translateBlendOp(D3D10_DDI_BLEND_OP op)
{
   switch (op) {
   case D3D10_DDI_BLEND_OP_ADD:
      return PIPE_BLEND_ADD;
   case D3D10_DDI_BLEND_OP_SUBTRACT:
      return PIPE_BLEND_SUBTRACT;
   case D3D10_DDI_BLEND_OP_REV_SUBTRACT:
      return PIPE_BLEND_REVERSE_SUBTRACT;
   case D3D10_DDI_BLEND_OP_MIN:
      return PIPE_BLEND_MIN;
   case D3D10_DDI_BLEND_OP_MAX:
      return PIPE_BLEND_MAX;
   default:
      assert(0);
      return PIPE_BLEND_ADD;
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * translateBlend --
 *
 *   Translate blend factor from svga3d to gallium representation.
 *
 * ----------------------------------------------------------------------
 */
static uint
translateBlend(Device *pDevice,
               D3D10_DDI_BLEND blend)
{
   if (!pDevice->max_dual_source_render_targets) {
      switch (blend) {
      case D3D10_DDI_BLEND_SRC1_COLOR:
      case D3D10_DDI_BLEND_SRC1_ALPHA:
         LOG_UNSUPPORTED(true);
         return D3D10_DDI_BLEND_ZERO;
      case D3D10_DDI_BLEND_INV_SRC1_COLOR:
      case D3D10_DDI_BLEND_INV_SRC1_ALPHA:
         LOG_UNSUPPORTED(true);
         return D3D10_DDI_BLEND_ONE;
      default:
         break;
      }
   }

   switch (blend) {
   case D3D10_DDI_BLEND_ZERO:
      return PIPE_BLENDFACTOR_ZERO;
   case D3D10_DDI_BLEND_ONE:
      return PIPE_BLENDFACTOR_ONE;
   case D3D10_DDI_BLEND_SRC_COLOR:
      return PIPE_BLENDFACTOR_SRC_COLOR;
   case D3D10_DDI_BLEND_INV_SRC_COLOR:
      return PIPE_BLENDFACTOR_INV_SRC_COLOR;
   case D3D10_DDI_BLEND_SRC_ALPHA:
      return PIPE_BLENDFACTOR_SRC_ALPHA;
   case D3D10_DDI_BLEND_INV_SRC_ALPHA:
      return PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   case D3D10_DDI_BLEND_DEST_ALPHA:
      return PIPE_BLENDFACTOR_DST_ALPHA;
   case D3D10_DDI_BLEND_INV_DEST_ALPHA:
      return PIPE_BLENDFACTOR_INV_DST_ALPHA;
   case D3D10_DDI_BLEND_DEST_COLOR:
      return PIPE_BLENDFACTOR_DST_COLOR;
   case D3D10_DDI_BLEND_INV_DEST_COLOR:
      return PIPE_BLENDFACTOR_INV_DST_COLOR;
   case D3D10_DDI_BLEND_SRC_ALPHASAT:
      return PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE;
   case D3D10_DDI_BLEND_BLEND_FACTOR:
      return PIPE_BLENDFACTOR_CONST_COLOR;
   case D3D10_DDI_BLEND_INVBLEND_FACTOR:
      return PIPE_BLENDFACTOR_INV_CONST_COLOR;
   case D3D10_DDI_BLEND_SRC1_COLOR:
      return PIPE_BLENDFACTOR_SRC1_COLOR;
   case D3D10_DDI_BLEND_INV_SRC1_COLOR:
      return PIPE_BLENDFACTOR_INV_SRC1_COLOR;
   case D3D10_DDI_BLEND_SRC1_ALPHA:
      return PIPE_BLENDFACTOR_SRC1_ALPHA;
   case D3D10_DDI_BLEND_INV_SRC1_ALPHA:
      return PIPE_BLENDFACTOR_INV_SRC1_ALPHA;
   default:
      assert(0);
      return PIPE_BLENDFACTOR_ONE;
   }
}

static unsigned
translateLogicOp(D3D11_1_DDI_LOGIC_OP logic_op)
{
   switch (logic_op) {
   case D3D11_1_DDI_LOGIC_OP_CLEAR:
      return PIPE_LOGICOP_CLEAR;
   case D3D11_1_DDI_LOGIC_OP_SET:
      return PIPE_LOGICOP_SET;
   case D3D11_1_DDI_LOGIC_OP_COPY:
      return PIPE_LOGICOP_COPY;
   case D3D11_1_DDI_LOGIC_OP_COPY_INVERTED:
      return PIPE_LOGICOP_COPY_INVERTED;
   case D3D11_1_DDI_LOGIC_OP_NOOP:
      return PIPE_LOGICOP_NOOP;
   case D3D11_1_DDI_LOGIC_OP_INVERT:
      return PIPE_LOGICOP_INVERT;
   case D3D11_1_DDI_LOGIC_OP_AND:
      return PIPE_LOGICOP_AND;
   case D3D11_1_DDI_LOGIC_OP_NAND:
      return PIPE_LOGICOP_NAND;
   case D3D11_1_DDI_LOGIC_OP_OR:
      return PIPE_LOGICOP_OR;
   case D3D11_1_DDI_LOGIC_OP_NOR:
      return PIPE_LOGICOP_NOR;
   case D3D11_1_DDI_LOGIC_OP_XOR:
      return PIPE_LOGICOP_XOR;
   case D3D11_1_DDI_LOGIC_OP_EQUIV:
      return PIPE_LOGICOP_EQUIV;
   case D3D11_1_DDI_LOGIC_OP_AND_REVERSE:
      return PIPE_LOGICOP_AND_REVERSE;
   case D3D11_1_DDI_LOGIC_OP_AND_INVERTED:
      return PIPE_LOGICOP_AND_INVERTED;
   case D3D11_1_DDI_LOGIC_OP_OR_REVERSE:
      return PIPE_LOGICOP_OR_REVERSE;
   case D3D11_1_DDI_LOGIC_OP_OR_INVERTED:
      return PIPE_LOGICOP_OR_INVERTED;
   default:
      assert(0);
      return PIPE_LOGICOP_COPY;
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateBlendState --
 *
 *    The CreateBlendState function creates a blend state.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateBlendState(D3D10DDI_HDEVICE hDevice,                     // IN
                 __in const D3D10_DDI_BLEND_DESC *pBlendDesc,  // IN
                 D3D10DDI_HBLENDSTATE hBlendState,             // IN
                 D3D10DDI_HRTBLENDSTATE hRTBlendState)         // IN
{
   unsigned i;

   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   BlendState *pBlendState = CastBlendState(hBlendState);

   struct pipe_blend_state state;
   memset(&state, 0, sizeof state);

   for (i = 0; i < MIN2(PIPE_MAX_COLOR_BUFS, D3D10_DDI_SIMULTANEOUS_RENDER_TARGET_COUNT); ++i) {
      state.rt[i].blend_enable = pBlendDesc->BlendEnable[i];
      state.rt[i].colormask = pBlendDesc->RenderTargetWriteMask[i];

      if (pBlendDesc->BlendEnable[0] != pBlendDesc->BlendEnable[i] ||
          pBlendDesc->RenderTargetWriteMask[0] != pBlendDesc->RenderTargetWriteMask[i]) {
         state.independent_blend_enable = 1;
      }
   }

   state.rt[0].rgb_func = translateBlendOp(pBlendDesc->BlendOp);
   if (pBlendDesc->BlendOp == D3D10_DDI_BLEND_OP_MIN ||
       pBlendDesc->BlendOp == D3D10_DDI_BLEND_OP_MAX) {
      state.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
      state.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_ONE;
   } else {
      state.rt[0].rgb_src_factor = translateBlend(pDevice, pBlendDesc->SrcBlend);
      state.rt[0].rgb_dst_factor = translateBlend(pDevice, pBlendDesc->DestBlend);
   }

   state.rt[0].alpha_func = translateBlendOp(pBlendDesc->BlendOpAlpha);
   if (pBlendDesc->BlendOpAlpha == D3D10_DDI_BLEND_OP_MIN ||
       pBlendDesc->BlendOpAlpha == D3D10_DDI_BLEND_OP_MAX) {
      state.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
      state.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_ONE;
   } else {
      state.rt[0].alpha_src_factor = translateBlend(pDevice, pBlendDesc->SrcBlendAlpha);
      state.rt[0].alpha_dst_factor = translateBlend(pDevice, pBlendDesc->DestBlendAlpha);
   }

   /*
    * Propagate to all the other rendertargets
    */
   for (i = 1; i < MIN2(PIPE_MAX_COLOR_BUFS, D3D10_DDI_SIMULTANEOUS_RENDER_TARGET_COUNT); ++i) {
      state.rt[i].rgb_func = state.rt[0].rgb_func;
      state.rt[i].rgb_src_factor = state.rt[0].rgb_src_factor;
      state.rt[i].rgb_dst_factor = state.rt[0].rgb_dst_factor;
      state.rt[i].alpha_func = state.rt[0].alpha_func;
      state.rt[i].alpha_src_factor = state.rt[0].alpha_src_factor;
      state.rt[i].alpha_dst_factor = state.rt[0].alpha_dst_factor;
   }

   state.alpha_to_coverage = pBlendDesc->AlphaToCoverageEnable;

   pBlendState->handle = pipe->create_blend_state(pipe, &state);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateBlendState1 --
 *
 *    The CreateBlendState1 function creates a blend state.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateBlendState1(D3D10DDI_HDEVICE hDevice,                     // IN
                  __in const D3D10_1_DDI_BLEND_DESC *pBlendDesc,  // IN
                  D3D10DDI_HBLENDSTATE hBlendState,             // IN
                  D3D10DDI_HRTBLENDSTATE hRTBlendState)         // IN
{
   unsigned i;

   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   BlendState *pBlendState = CastBlendState(hBlendState);

   struct pipe_blend_state state;
   memset(&state, 0, sizeof state);

   state.alpha_to_coverage = pBlendDesc->AlphaToCoverageEnable;
   state.independent_blend_enable = pBlendDesc->IndependentBlendEnable;

   for (i = 0; i < MIN2(PIPE_MAX_COLOR_BUFS, D3D10_DDI_SIMULTANEOUS_RENDER_TARGET_COUNT); ++i) {
      state.rt[i].blend_enable = pBlendDesc->RenderTarget[i].BlendEnable;
      state.rt[i].colormask = pBlendDesc->RenderTarget[i].RenderTargetWriteMask;

      state.rt[i].rgb_func = translateBlendOp(pBlendDesc->RenderTarget[i].BlendOp);
      if (pBlendDesc->RenderTarget[i].BlendOp == D3D10_DDI_BLEND_OP_MIN ||
          pBlendDesc->RenderTarget[i].BlendOp == D3D10_DDI_BLEND_OP_MAX) {
         state.rt[i].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
         state.rt[i].rgb_dst_factor = PIPE_BLENDFACTOR_ONE;
      } else {
         state.rt[i].rgb_src_factor = translateBlend(pDevice, pBlendDesc->RenderTarget[i].SrcBlend);
         state.rt[i].rgb_dst_factor = translateBlend(pDevice, pBlendDesc->RenderTarget[i].DestBlend);
      }

      state.rt[i].alpha_func = translateBlendOp(pBlendDesc->RenderTarget[i].BlendOpAlpha);
      if (pBlendDesc->RenderTarget[i].BlendOpAlpha == D3D10_DDI_BLEND_OP_MIN ||
          pBlendDesc->RenderTarget[i].BlendOpAlpha == D3D10_DDI_BLEND_OP_MAX) {
         state.rt[i].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
         state.rt[i].alpha_dst_factor = PIPE_BLENDFACTOR_ONE;
      } else {
         state.rt[i].alpha_src_factor = translateBlend(pDevice, pBlendDesc->RenderTarget[i].SrcBlendAlpha);
         state.rt[i].alpha_dst_factor = translateBlend(pDevice, pBlendDesc->RenderTarget[i].DestBlendAlpha);
      }
   }

   pBlendState->handle = pipe->create_blend_state(pipe, &state);
}

void APIENTRY
CreateBlendState11_1Impl(D3D10DDI_HDEVICE hDevice,
                         __in const D3D11_1_DDI_BLEND_DESC *pBlendDesc,
                         D3D10DDI_HBLENDSTATE hBlendState,
                         D3D10DDI_HRTBLENDSTATE hRTBlendState)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   BlendState *pBlendState = CastBlendState(hBlendState);

   struct pipe_blend_state state;
   memset(&state, 0, sizeof state);

   state.alpha_to_coverage = pBlendDesc->AlphaToCoverageEnable;
   state.independent_blend_enable = pBlendDesc->IndependentBlendEnable;

   for (unsigned i = 0;
        i < MIN2(PIPE_MAX_COLOR_BUFS, D3D10_DDI_SIMULTANEOUS_RENDER_TARGET_COUNT);
        ++i) {
      const D3D11_1_DDI_RENDER_TARGET_BLEND_DESC *rt =
         &pBlendDesc->RenderTarget[i];

      if (rt->LogicOpEnable) {
         state.logicop_enable = 1;
         state.logicop_func = translateLogicOp(rt->LogicOp);
      }

      state.rt[i].blend_enable = rt->BlendEnable && !rt->LogicOpEnable;
      state.rt[i].colormask = rt->RenderTargetWriteMask;

      state.rt[i].rgb_func = translateBlendOp(rt->BlendOp);
      if (rt->BlendOp == D3D10_DDI_BLEND_OP_MIN ||
          rt->BlendOp == D3D10_DDI_BLEND_OP_MAX) {
         state.rt[i].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
         state.rt[i].rgb_dst_factor = PIPE_BLENDFACTOR_ONE;
      } else {
         state.rt[i].rgb_src_factor = translateBlend(pDevice, rt->SrcBlend);
         state.rt[i].rgb_dst_factor = translateBlend(pDevice, rt->DestBlend);
      }

      state.rt[i].alpha_func = translateBlendOp(rt->BlendOpAlpha);
      if (rt->BlendOpAlpha == D3D10_DDI_BLEND_OP_MIN ||
          rt->BlendOpAlpha == D3D10_DDI_BLEND_OP_MAX) {
         state.rt[i].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
         state.rt[i].alpha_dst_factor = PIPE_BLENDFACTOR_ONE;
      } else {
         state.rt[i].alpha_src_factor =
            translateBlend(pDevice, rt->SrcBlendAlpha);
         state.rt[i].alpha_dst_factor =
            translateBlend(pDevice, rt->DestBlendAlpha);
      }
   }

   pBlendState->handle = pipe->create_blend_state(pipe, &state);
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyBlendState --
 *
 *    The DestroyBlendState function destroys the specified blend
 *    state object. The blend state object can be destoyed only if
 *    it is not currently bound to a display device.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroyBlendState(D3D10DDI_HDEVICE hDevice,           // IN
                  D3D10DDI_HBLENDSTATE hBlendState)   // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   BlendState *pBlendState = CastBlendState(hBlendState);

   pipe->delete_blend_state(pipe, pBlendState->handle);
}


/*
 * ----------------------------------------------------------------------
 *
 * SetBlendState --
 *
 *    The SetBlendState function sets a blend state.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
SetBlendState(D3D10DDI_HDEVICE hDevice,      // IN
              D3D10DDI_HBLENDSTATE hState,   // IN
              const FLOAT pBlendFactor[4],   // IN
              UINT SampleMask)               // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   void *state = CastPipeBlendState(hState);

   pipe->bind_blend_state(pipe, state);

   struct pipe_blend_color color;
   color.color[0] = pBlendFactor[0];
   color.color[1] = pBlendFactor[1];
   color.color[2] = pBlendFactor[2];
   color.color[3] = pBlendFactor[3];

   pipe->set_blend_color(pipe, &color);

   pipe->set_sample_mask(pipe, SampleMask);
}


/*
 * ----------------------------------------------------------------------
 *
 * SetRenderTargets --
 *
 *    Set the rendertargets.
 *
 * ----------------------------------------------------------------------
 */

static const mesa_shader_stage graphics_uav_stages[] = {
   MESA_SHADER_VERTEX,
   MESA_SHADER_GEOMETRY,
   MESA_SHADER_FRAGMENT,
};

static void
ClearGraphicsUnorderedAccessViews(Device *pDevice)
{
   bool already_clear = true;
   for (unsigned i = 0; i < ARRAY_SIZE(graphics_uav_stages); i++) {
      const mesa_shader_stage stage = graphics_uav_stages[i];

      if (!pDevice->bufinfo_constants_bound[stage] ||
          pDevice->bufinfo_constants_dirty[stage]) {
         already_clear = false;
         break;
      }

      for (unsigned slot = 0; slot < PIPE_MAX_SHADER_IMAGES; slot++) {
         if (pDevice->unordered_access_views[stage][slot]) {
            already_clear = false;
            break;
         }
      }
      if (!already_clear)
         break;
   }
   if (already_clear)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(graphics_uav_stages); i++) {
      const mesa_shader_stage stage = graphics_uav_stages[i];

      memset(pDevice->unordered_access_views[stage], 0,
             sizeof(pDevice->unordered_access_views[stage]));
      memset(pDevice->shader_images[stage], 0,
             sizeof(pDevice->shader_images[stage]));

      pDevice->pipe->set_shader_images(pDevice->pipe, stage,
                                       0, 0, PIPE_MAX_SHADER_IMAGES, NULL);
      pDevice->counter_image_bound_mask[stage] = 0;
      UpdateBufferInfoUavConstants(pDevice, stage, 0,
                                   PIPE_MAX_SHADER_IMAGES);
      UpdateBufferInfoConstants(pDevice, stage);
   }
}

static void
SetRenderTargetsImpl(D3D10DDI_HDEVICE hDevice,                              // IN
                     __in_ecount (NumViews)
                     const D3D10DDI_HRENDERTARGETVIEW *phRenderTargetView,  // IN
                     UINT RTargets,                                         // IN
                     UINT ClearTargets,                                     // IN
                     D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView,          // IN
                     bool clear_fragment_uavs)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);

   struct pipe_context *pipe = pDevice->pipe;

   if (clear_fragment_uavs)
      ClearGraphicsUnorderedAccessViews(pDevice);

   pDevice->fb.nr_cbufs = 0;

   for (unsigned i = 0; i < RTargets; ++i) {
      struct pipe_surface *psurf = GetPipeRenderTargetView(phRenderTargetView[i]);
      pipe_resource_reference(&pDevice->fb.cbufs[i].texture,
                              psurf && psurf->texture ? psurf->texture : NULL);
      ResourceEvent(RESOURCE_EVENT_SET_RENDER_TARGET,
                    (uint64_t)(uintptr_t)(phRenderTargetView
                                             ? phRenderTargetView[i].pDrvPrivate
                                             : NULL),
                    NULL,
                    psurf ? psurf->texture : NULL,
                    psurf ? PipeResourceRefCount(psurf->texture) : 0,
                    i,
                    RTargets,
                    ClearTargets);
      if (psurf && psurf->texture) {
         if (debug_get_option_rt_trace()) {
            debug_printf("d3d10umd: SetRenderTargets[%u] view=%p surface_tex=%p rtv_resource=%p\n",
                         i,
                         phRenderTargetView[i].pDrvPrivate,
                         psurf->texture,
                         CastRenderTargetView(phRenderTargetView[i]) ?
                            CastRenderTargetView(phRenderTargetView[i])->resource :
                            NULL);
            yttrium_gdi_resource_debug_log(psurf->texture,
                                           "SetRenderTargets cbuf");
         }
         pDevice->fb.nr_cbufs = i + 1;
         pDevice->fb.cbufs[i] = *psurf;
      }
   }

   for (unsigned i = RTargets; i < PIPE_MAX_COLOR_BUFS; ++i) {
      pipe_resource_reference(&pDevice->fb.cbufs[i].texture, NULL);
   }

   struct pipe_surface *zsbuf = CastPipeDepthStencilView(hDepthStencilView);
   pipe_resource_reference(&pDevice->fb.zsbuf.texture, zsbuf && zsbuf->texture ? zsbuf->texture : NULL);
   ResourceEvent(RESOURCE_EVENT_SET_DEPTH_STENCIL,
                 (uint64_t)(uintptr_t)hDepthStencilView.pDrvPrivate,
                 NULL,
                 zsbuf ? zsbuf->texture : NULL,
                 zsbuf ? PipeResourceRefCount(zsbuf->texture) : 0,
                 0, 0, 0);
   if(zsbuf && zsbuf->texture) {
      pDevice->fb.zsbuf = *zsbuf;
   }

   /*
    * Calculate the width/height fields for this framebuffer.  D3D10
    * actually specifies that they be identical for all bound views.
    */
   unsigned width, height;
   util_framebuffer_min_size(&pDevice->fb, &width, &height);
   if (!width && !height && pDevice->viewport_fb_width &&
       pDevice->viewport_fb_height) {
      width = pDevice->viewport_fb_width;
      height = pDevice->viewport_fb_height;
   }
   pDevice->fb.width = width;
   pDevice->fb.height = height;

   UpdateFramebufferForcedSampleCount(pDevice);
   pipe->set_framebuffer_state(pipe, &pDevice->fb);
   UpdateBufferInfoSampleConstants(pDevice, MESA_SHADER_FRAGMENT);
   UpdateBufferInfoConstants(pDevice, MESA_SHADER_FRAGMENT);
}

void APIENTRY
SetRenderTargets(D3D10DDI_HDEVICE hDevice,                              // IN
                 __in_ecount (NumViews)
                  const D3D10DDI_HRENDERTARGETVIEW *phRenderTargetView, // IN
                 UINT RTargets,                                         // IN
                 UINT ClearTargets,                                     // IN
                 D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView)          // IN
{
   SetRenderTargetsImpl(hDevice, phRenderTargetView, RTargets, ClearTargets,
                        hDepthStencilView, true);
}

void APIENTRY
SetRenderTargets11(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (NumRTVs) const D3D10DDI_HRENDERTARGETVIEW *phRenderTargetView,
   UINT RTargets, UINT ClearTargets, D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView,
   __in_ecount (NumUAVs) const D3D11DDI_HUNORDEREDACCESSVIEW *phUnorderedAccessView,
   __in_ecount (NumUAVs) const UINT *pUAVInitialCounts,
   UINT UAVStartSlot, UINT NumUAVs, UINT UAVRangeStart, UINT UAVRangeSize)
{
   Device *pDevice = CastDevice(hDevice);

   SetRenderTargetsImpl(hDevice, phRenderTargetView, RTargets, ClearTargets,
                        hDepthStencilView, false);

   if (!NumUAVs) {
      ClearGraphicsUnorderedAccessViews(pDevice);
      return;
   }

   assert(UAVStartSlot + NumUAVs <= PIPE_MAX_SHADER_IMAGES);
   for (UINT i = 0; i < NumUAVs; i++) {
      UnorderedAccessView *uav =
         CastUnorderedAccessView(phUnorderedAccessView[i]);
      if (uav) {
         if (pUAVInitialCounts &&
             pUAVInitialCounts[i] != ~0u &&
             (uav->buffer_counter || uav->buffer_append))
            ResetUAVCounter(pDevice, uav, pUAVInitialCounts[i]);
         for (unsigned stage_idx = 0;
              stage_idx < ARRAY_SIZE(graphics_uav_stages);
              stage_idx++) {
            const mesa_shader_stage stage = graphics_uav_stages[stage_idx];
            pDevice->unordered_access_views[stage][UAVStartSlot + i] =
               uav;
            pDevice->shader_images[stage][UAVStartSlot + i] =
               uav->image;
         }
      } else {
         for (unsigned stage_idx = 0;
              stage_idx < ARRAY_SIZE(graphics_uav_stages);
              stage_idx++) {
            const mesa_shader_stage stage = graphics_uav_stages[stage_idx];
            pDevice->unordered_access_views[stage][UAVStartSlot + i] =
               NULL;
            memset(&pDevice->shader_images[stage][UAVStartSlot + i],
                   0, sizeof(pDevice->shader_images[stage][0]));
         }
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(graphics_uav_stages); i++) {
      const mesa_shader_stage stage = graphics_uav_stages[i];
      BindShaderImages(pDevice, stage, UAVStartSlot, NumUAVs);
      UpdateBufferInfoUavConstants(pDevice, stage, UAVStartSlot, NumUAVs);
      UpdateBufferInfoConstants(pDevice, stage);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateDepthStencilStateSize --
 *
 *    The CalcPrivateDepthStencilStateSize function determines the size
 *    of the user-mode display driver's private region of memory (that
 *    is, the size of internal driver structures, not the size of the
 *    resource video memory) for a depth stencil state.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateDepthStencilStateSize(
   D3D10DDI_HDEVICE hDevice,                                   // IN
   __in const D3D10_DDI_DEPTH_STENCIL_DESC *pDepthStencilDesc) // IN
{
   return sizeof(DepthStencilState);
}


/*
 * ----------------------------------------------------------------------
 *
 * translateComparison --
 *
 *   Translate comparison function from DX10 to gallium representation.
 *
 * ----------------------------------------------------------------------
 */
static uint
translateComparison(D3D10_DDI_COMPARISON_FUNC Func)
{
   switch (Func) {
   case D3D10_DDI_COMPARISON_NEVER:
      return PIPE_FUNC_NEVER;
   case D3D10_DDI_COMPARISON_LESS:
      return PIPE_FUNC_LESS;
   case D3D10_DDI_COMPARISON_EQUAL:
      return PIPE_FUNC_EQUAL;
   case D3D10_DDI_COMPARISON_LESS_EQUAL:
      return PIPE_FUNC_LEQUAL;
   case D3D10_DDI_COMPARISON_GREATER:
      return PIPE_FUNC_GREATER;
   case D3D10_DDI_COMPARISON_NOT_EQUAL:
      return PIPE_FUNC_NOTEQUAL;
   case D3D10_DDI_COMPARISON_GREATER_EQUAL:
      return PIPE_FUNC_GEQUAL;
   case D3D10_DDI_COMPARISON_ALWAYS:
      return PIPE_FUNC_ALWAYS;
   default:
      assert(0);
      return PIPE_FUNC_ALWAYS;
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * translateStencilOp --
 *
 *   Translate stencil op from DX10 to gallium representation.
 *
 * ----------------------------------------------------------------------
 */
static uint
translateStencilOp(D3D10_DDI_STENCIL_OP StencilOp)
{
   switch (StencilOp) {
   case D3D10_DDI_STENCIL_OP_KEEP:
      return PIPE_STENCIL_OP_KEEP;
   case D3D10_DDI_STENCIL_OP_ZERO:
      return PIPE_STENCIL_OP_ZERO;
   case D3D10_DDI_STENCIL_OP_REPLACE:
      return PIPE_STENCIL_OP_REPLACE;
   case D3D10_DDI_STENCIL_OP_INCR_SAT:
      return PIPE_STENCIL_OP_INCR;
   case D3D10_DDI_STENCIL_OP_DECR_SAT:
      return PIPE_STENCIL_OP_DECR;
   case D3D10_DDI_STENCIL_OP_INVERT:
      return PIPE_STENCIL_OP_INVERT;
   case D3D10_DDI_STENCIL_OP_INCR:
      return PIPE_STENCIL_OP_INCR_WRAP;
   case D3D10_DDI_STENCIL_OP_DECR:
      return PIPE_STENCIL_OP_DECR_WRAP;
   default:
      assert(0);
      return PIPE_STENCIL_OP_KEEP;
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateDepthStencilState --
 *
 *    The CreateDepthStencilState function creates a depth stencil state.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateDepthStencilState(
   D3D10DDI_HDEVICE hDevice,                                   // IN
   __in const D3D10_DDI_DEPTH_STENCIL_DESC *pDepthStencilDesc, // IN
   D3D10DDI_HDEPTHSTENCILSTATE hDepthStencilState,             // IN
   D3D10DDI_HRTDEPTHSTENCILSTATE hRTDepthStencilState)         // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   DepthStencilState *pDepthStencilState = CastDepthStencilState(hDepthStencilState);

   struct pipe_depth_stencil_alpha_state state;
   memset(&state, 0, sizeof state);

   /* Depth. */
   state.depth_enabled = (pDepthStencilDesc->DepthEnable ? 1 : 0);
   state.depth_writemask = (pDepthStencilDesc->DepthWriteMask ? 1 : 0);
   state.depth_func = translateComparison(pDepthStencilDesc->DepthFunc);

   /* Stencil. */
   if (pDepthStencilDesc->StencilEnable) {
      struct pipe_stencil_state *face0 = &state.stencil[0];
      struct pipe_stencil_state *face1 = &state.stencil[1];

      face0->enabled   = 1;
      face0->func      = translateComparison(pDepthStencilDesc->FrontFace.StencilFunc);
      face0->fail_op   = translateStencilOp(pDepthStencilDesc->FrontFace.StencilFailOp);
      face0->zpass_op  = translateStencilOp(pDepthStencilDesc->FrontFace.StencilPassOp);
      face0->zfail_op  = translateStencilOp(pDepthStencilDesc->FrontFace.StencilDepthFailOp);
      face0->valuemask = pDepthStencilDesc->StencilReadMask;
      face0->writemask = pDepthStencilDesc->StencilWriteMask;

      face1->enabled   = 1;
      face1->func      = translateComparison(pDepthStencilDesc->BackFace.StencilFunc);
      face1->fail_op   = translateStencilOp(pDepthStencilDesc->BackFace.StencilFailOp);
      face1->zpass_op  = translateStencilOp(pDepthStencilDesc->BackFace.StencilPassOp);
      face1->zfail_op  = translateStencilOp(pDepthStencilDesc->BackFace.StencilDepthFailOp);
      face1->valuemask = pDepthStencilDesc->StencilReadMask;
      face1->writemask = pDepthStencilDesc->StencilWriteMask;
#if MESA_DEBUG
      if (!pDepthStencilDesc->FrontEnable) {
         ASSERT(face0->func == PIPE_FUNC_ALWAYS);
         ASSERT(face0->fail_op == PIPE_STENCIL_OP_KEEP);
         ASSERT(face0->zpass_op == PIPE_STENCIL_OP_KEEP);
         ASSERT(face0->zfail_op == PIPE_STENCIL_OP_KEEP);
      }

      if (!pDepthStencilDesc->BackEnable) {
         ASSERT(face1->func == PIPE_FUNC_ALWAYS);
         ASSERT(face1->fail_op == PIPE_STENCIL_OP_KEEP);
         ASSERT(face1->zpass_op == PIPE_STENCIL_OP_KEEP);
         ASSERT(face1->zfail_op == PIPE_STENCIL_OP_KEEP);
      }
#endif
   }

   pDepthStencilState->handle =
      pipe->create_depth_stencil_alpha_state(pipe, &state);
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyDepthStencilState --
 *
 *    The DestroyDepthStencilState function destroy a depth stencil state.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroyDepthStencilState(D3D10DDI_HDEVICE hDevice,                         // IN
                         D3D10DDI_HDEPTHSTENCILSTATE hDepthStencilState)   // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   DepthStencilState *pDepthStencilState = CastDepthStencilState(hDepthStencilState);

   pipe->delete_depth_stencil_alpha_state(pipe, pDepthStencilState->handle);
}


/*
 * ----------------------------------------------------------------------
 *
 * SetDepthStencilState --
 *
 *    The SetDepthStencilState function sets a depth-stencil state.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
SetDepthStencilState(D3D10DDI_HDEVICE hDevice,           // IN
                     D3D10DDI_HDEPTHSTENCILSTATE hState, // IN
                     UINT StencilRef)                    // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   void *state = hState.pDrvPrivate ?
      CastPipeDepthStencilState(hState) : pDevice->default_depth_stencil_state;
   struct pipe_stencil_ref psr;

   psr.ref_value[0] = StencilRef;
   psr.ref_value[1] = StencilRef;

   pipe->bind_depth_stencil_alpha_state(pipe, state);
   pipe->set_stencil_ref(pipe, psr);
}
