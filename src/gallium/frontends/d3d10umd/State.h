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
 * State.h --
 *    State declarations.
 */


#include "DriverIncludes.h"
#include "ShaderParse.h"
#include "pipe/p_screen.h"
#include "util/u_hash_table.h"
#include "util/list.h"
#include "util/u_atomic.h"
#include "cso_cache/cso_context.h"
#include "util/u_thread.h"

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

#include <d3dukmdt.h>
#include "gdikmt_d3dddi.h"

#define SUPPORT_MSAA 0
#define SUPPORT_D3D10_1 1
#define SUPPORT_D3D11 1
#ifndef SUPPORT_D3D11_1
#define SUPPORT_D3D11_1 1
#endif
#ifndef SUPPORT_D3D_WDDM1_3
#define SUPPORT_D3D_WDDM1_3 (SUPPORT_D3D11_1 && D3D11DDI_MINOR_HEADER_VERSION >= 4)
#endif


struct Adapter
{
   struct pipe_screen *screen;
   HANDLE hRTAdapter;
   D3DDDI_ADAPTERCALLBACKS AdapterCallbacks;
};


static inline Adapter *
CastAdapter(D3D10DDI_HADAPTER hAdapter)
{
   return static_cast<Adapter *>(hAdapter.pDrvPrivate);
}

struct Shader
{
   void *handle;
   uint type;
   struct pipe_shader_state state;
   struct pipe_compute_state compute_state;
   struct pipe_shader_sampler_binding_map sampler_binding_map;
   /* D3D UAV index -> hidden counter image index + 1 (zero means unused). */
   uint8_t counter_image_slots[PIPE_MAX_SHADER_IMAGES];
   unsigned thread_group_size[3];
   unsigned compute_emulation;
   unsigned compute_store_imm[4];
   unsigned tess_output_primitive;
   struct Shader_tessellation_properties tessellation_properties;
   struct Shader_tessellation_properties tessellation_compiled_properties;
   unsigned *tessellation_code;
   unsigned tessellation_code_size;
   D3D10_SB_NAME tessellation_input_system_values[32];
   D3D10_SB_NAME tessellation_output_system_values[32];
   unsigned char tessellation_input_masks[32];
   unsigned char tessellation_output_masks[32];
   unsigned tessellation_input_signature_count;
   unsigned tessellation_output_signature_count;
   bool tessellation_has_signatures;
   bool tessellation_properties_valid;
   bool tessellation_compiled_properties_valid;
   /* Original D3D output register -> translated TGSI output register. */
   unsigned output_mapping[PIPE_MAX_SHADER_OUTPUTS];
   struct pipe_stream_output_info stream_output_template;
   uint8_t stream_output_d3d_registers[PIPE_MAX_SO_OUTPUTS];
   uint8_t stream_output_signature_masks[PIPE_MAX_SHADER_OUTPUTS];
   bool no_rasterized_stream;
};

struct Query;
struct ElementLayout;
struct Resource;
struct RasterizerState;
struct UnorderedAccessView;

struct Device
{
   struct pipe_screen *screen;
   struct pipe_context *pipe;

   struct gdikmt_device_d3dddi device;

   struct cso_context *cso;
   unsigned feature_level;

   struct pipe_framebuffer_state fb;
   struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS];
   unsigned vertex_strides[PIPE_MAX_ATTRIBS];
   struct pipe_resource *index_buffer;
   unsigned restart_index;
   unsigned index_size;
   unsigned ib_offset;
   unsigned viewport_fb_width;
   unsigned viewport_fb_height;
   void *samplers[MESA_SHADER_STAGES][PIPE_MAX_SAMPLERS];
   D3D10DDI_HSHADERRESOURCEVIEW shader_resource_views[MESA_SHADER_STAGES][PIPE_MAX_SHADER_SAMPLER_VIEWS];
   struct pipe_sampler_view *sampler_views[MESA_SHADER_STAGES][PIPE_MAX_SHADER_SAMPLER_VIEWS];
   struct UnorderedAccessView *unordered_access_views[MESA_SHADER_STAGES][PIPE_MAX_SHADER_IMAGES];
   struct pipe_resource *constant_buffers[MESA_SHADER_STAGES][PIPE_MAX_CONSTANT_BUFFERS];
   Resource *constant_buffer_resources[MESA_SHADER_STAGES][PIPE_MAX_CONSTANT_BUFFERS];
   Resource *constant_buffer_binding_resources[MESA_SHADER_STAGES][PIPE_MAX_CONSTANT_BUFFERS];
   unsigned constant_buffer_offsets[MESA_SHADER_STAGES][PIPE_MAX_CONSTANT_BUFFERS];
   unsigned constant_buffer_sizes[MESA_SHADER_STAGES][PIPE_MAX_CONSTANT_BUFFERS];
   bool constant_buffer_published[MESA_SHADER_STAGES][PIPE_MAX_CONSTANT_BUFFERS];
   bool constant_publication_enabled;
   bool yttrium_sampler_binding_map_enabled;
   uint32_t bufinfo_constants[MESA_SHADER_STAGES][D3D10UMD_BUFINFO_CB_DWORDS];
   bool bufinfo_constants_dirty[MESA_SHADER_STAGES];
   bool bufinfo_constants_bound[MESA_SHADER_STAGES];
   bool shader_resource_views_dirty;
   struct pipe_image_view shader_images[MESA_SHADER_STAGES][PIPE_MAX_SHADER_IMAGES];
   uint64_t counter_image_bound_mask[MESA_SHADER_STAGES];

   void *empty_fs;
   void *empty_vs;
   void *default_depth_stencil_state;

   enum mesa_prim primitive;

   struct pipe_stream_output_target *so_targets[PIPE_MAX_SO_BUFFERS];
   struct pipe_stream_output_target *draw_so_target;
   Shader *bound_empty_gs;
   Shader *bound_gs;
   Shader *bound_vs;
   Shader *bound_ps;
   Shader *bound_cs;
   Shader *bound_hs;
   Shader *bound_ds;
   unsigned patch_vertices;
   RasterizerState *rasterizer_state;
   void *default_rasterizer_discard_state;

   unsigned max_dual_source_render_targets;
   
   D3D10DDI_HRTCORELAYER  hRTCoreLayer;
   D3D10DDI_CORELAYER_DEVICECALLBACKS UMCallbacks;

   INT LastEmittedQuerySeqNo;
   INT LastFinishedQuerySeqNo;

   Query *pPredicate;
   BOOL PredicateValue;

   ElementLayout *element_layout;
   BOOL velems_changed;
   BOOL vbuffers_changed;
   
   mtx_t CreateResourceMtx;
   struct list_head resources;
};


static inline Device *
CastDevice(D3D10DDI_HDEVICE hDevice)
{
   return static_cast<Device *>(hDevice.pDrvPrivate);
}


static inline struct pipe_context *
CastPipeContext(D3D10DDI_HDEVICE hDevice)
{
   Device *pDevice = CastDevice(hDevice);
   return pDevice ? pDevice->pipe : NULL;
}

static inline Device *
CastDevice(DXGI_DDI_HDEVICE hDevice)
{
   return reinterpret_cast<Device *>(hDevice);
}


static inline struct pipe_context *
CastPipeDevice(DXGI_DDI_HDEVICE hDevice)
{
   Device *pDevice = CastDevice(hDevice);
   return pDevice ? pDevice->pipe : NULL;
}


static inline void
SetErrorAt(D3D10DDI_HDEVICE hDevice, HRESULT hr,
           const char *function, unsigned line)
{
   if (FAILED(hr)) {
      Device *pDevice = CastDevice(hDevice);
      if (hr != DXGI_DDI_ERR_WASSTILLDRAWING) {
         yttrium_gdi_trace_errorf(
            "d3d10umd: pfnSetErrorCb hr=0x%08x caller=%s:%u device=%p",
            (unsigned)hr, function, line, pDevice);
      }
      pDevice->UMCallbacks.pfnSetErrorCb(pDevice->hRTCoreLayer, hr);
   }
}

#define SetError(hDevice, hr) \
   SetErrorAt((hDevice), (hr), __func__, __LINE__)


struct Resource
{
   struct list_head list;
   bool listed;
   DXGI_FORMAT Format;
   UINT MiscFlags;
   UINT ByteStride;
   UINT MipLevels;
   UINT NumSubResources;
   UINT SampleCount;
   bool buffer;
   bool yttrium_primary;
   struct pipe_resource *resource;
   void *buffer_shadow;
   unsigned buffer_shadow_size;
   struct pipe_resource *constant_published_buffer;
   void *constant_published_cpu;
   unsigned constant_published_offset;
   unsigned constant_published_size;
   bool constant_shadow_valid;
   bool constant_original_stale;
   bool constant_publication_fallback_warned;
   uint32_t constant_buffer_bindings[MESA_SHADER_STAGES];
   struct pipe_transfer **transfers;
   struct pipe_stream_output_target *so_target;
};


static inline Resource *
CastResource(D3D10DDI_HRESOURCE hResource)
{
   return static_cast<Resource *>(hResource.pDrvPrivate);
}


static inline Resource *
CastResource(DXGI_DDI_HRESOURCE hResource)
{
   return reinterpret_cast<Resource *>(hResource);
}


static inline struct pipe_resource *
CastPipeResource(D3D10DDI_HRESOURCE hResource)
{
   Resource *pResource = CastResource(hResource);
   return pResource ? pResource->resource : NULL;
}


static inline struct pipe_resource *
CastPipeResource(DXGI_DDI_HRESOURCE hResource)
{
   Resource *pResource = CastResource(hResource);
   return pResource ? pResource->resource : NULL;
}


static inline struct pipe_resource *
CastPipeBuffer(D3D10DDI_HRESOURCE hResource)
{
   Resource *pResource = CastResource(hResource);
   if (!pResource) {
      return NULL;
   }
   return static_cast<struct pipe_resource *>(pResource->resource);
}


static inline int
PipeResourceRefCount(struct pipe_resource *resource)
{
   return resource ? p_atomic_read(&resource->reference.count) : 0;
}


struct RenderTargetView
{
   struct pipe_surface surface;
   Resource* resource; //akre
   D3D10DDI_HRTRENDERTARGETVIEW hRTRenderTargetView;
};


static inline RenderTargetView *
CastRenderTargetView(D3D10DDI_HRENDERTARGETVIEW hRenderTargetView)
{
   return static_cast<RenderTargetView *>(hRenderTargetView.pDrvPrivate);
}


static inline struct pipe_surface *
CastPipeRenderTargetView(D3D10DDI_HRENDERTARGETVIEW hRenderTargetView)
{
   RenderTargetView *pRenderTargetView = CastRenderTargetView(hRenderTargetView);
   return pRenderTargetView ? &pRenderTargetView->surface : NULL;
}


struct DepthStencilView
{
   struct pipe_surface surface;
   D3D10DDI_HRTDEPTHSTENCILVIEW hRTDepthStencilView;
};


static inline DepthStencilView *
CastDepthStencilView(D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView)
{
   return static_cast<DepthStencilView *>(hDepthStencilView.pDrvPrivate);
}


static inline struct pipe_surface *
CastPipeDepthStencilView(D3D10DDI_HDEPTHSTENCILVIEW hDepthStencilView)
{
   DepthStencilView *pDepthStencilView = CastDepthStencilView(hDepthStencilView);
   return pDepthStencilView ? &pDepthStencilView->surface : NULL;
}


struct BlendState
{
   void *handle;
};


static inline BlendState *
CastBlendState(D3D10DDI_HBLENDSTATE hBlendState)
{
   return static_cast<BlendState *>(hBlendState.pDrvPrivate);
}


static inline void *
CastPipeBlendState(D3D10DDI_HBLENDSTATE hBlendState)
{
   BlendState *pBlendState = CastBlendState(hBlendState);
   return pBlendState ? pBlendState->handle : NULL;
}


struct DepthStencilState
{
   void *handle;
};


static inline DepthStencilState *
CastDepthStencilState(D3D10DDI_HDEPTHSTENCILSTATE hDepthStencilState)
{
   return static_cast<DepthStencilState *>(hDepthStencilState.pDrvPrivate);
}


static inline void *
CastPipeDepthStencilState(D3D10DDI_HDEPTHSTENCILSTATE hDepthStencilState)
{
   DepthStencilState *pDepthStencilState = CastDepthStencilState(hDepthStencilState);
   return pDepthStencilState ? pDepthStencilState->handle : NULL;
}


struct RasterizerState
{
   void *handle;
   void *discard_handle;
   struct pipe_rasterizer_state state;
   unsigned forced_sample_count;
};


static inline RasterizerState *
CastRasterizerState(D3D10DDI_HRASTERIZERSTATE hRasterizerState)
{
   return static_cast<RasterizerState *>(hRasterizerState.pDrvPrivate);
}


static inline void *
CastPipeRasterizerState(D3D10DDI_HRASTERIZERSTATE hRasterizerState)
{
   RasterizerState *pRasterizerState = CastRasterizerState(hRasterizerState);
   return pRasterizerState ? pRasterizerState->handle : NULL;
}


static inline Shader *
CastShader(D3D10DDI_HSHADER hShader)
{
   return static_cast<Shader *>(hShader.pDrvPrivate);
}


static inline void *
CastPipeShader(D3D10DDI_HSHADER hShader)
{
   Shader *pShader = static_cast<Shader *>(hShader.pDrvPrivate);
   return pShader ? pShader->handle : NULL;
}


struct ElementLayout
{
   struct cso_velems_state state;
};


static inline ElementLayout *
CastElementLayout(D3D10DDI_HELEMENTLAYOUT hElementLayout)
{
   return static_cast<ElementLayout *>(hElementLayout.pDrvPrivate);
}

struct SamplerState
{
   void *handle;
};


static inline SamplerState *
CastSamplerState(D3D10DDI_HSAMPLER hSampler)
{
   return static_cast<SamplerState *>(hSampler.pDrvPrivate);
}


static inline void *
CastPipeSamplerState(D3D10DDI_HSAMPLER hSampler)
{
   SamplerState *pSamplerState = CastSamplerState(hSampler);
   return pSamplerState ? pSamplerState->handle : NULL;
}


struct ShaderResourceView
{
   struct pipe_sampler_view *handle;
   Resource *resource;
   bool buffer_raw;
   bool buffer_structured;
   UINT buffer_first_element;
   UINT buffer_num_elements;
   UINT buffer_stride;
};


static inline ShaderResourceView *
CastShaderResourceView(D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView)
{
   return static_cast<ShaderResourceView *>(hShaderResourceView.pDrvPrivate);
}


static inline struct pipe_sampler_view *
CastPipeShaderResourceView(D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView)
{
   ShaderResourceView *pSRView = CastShaderResourceView(hShaderResourceView);
   return pSRView ? pSRView->handle : NULL;
}


struct UnorderedAccessView
{
   struct pipe_resource *pipe_resource;
   /* The counter belongs to the view, not to the underlying structured buffer. */
   struct pipe_resource *counter_resource;
   struct pipe_image_view image;
   enum pipe_format clear_format;
   Resource *resource;
   bool buffer_raw;
   bool buffer_structured;
   bool buffer_counter;
   bool buffer_append;
   UINT buffer_first_element;
   UINT buffer_num_elements;
   UINT buffer_stride;
};


static inline UnorderedAccessView *
CastUnorderedAccessView(D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView)
{
   return static_cast<UnorderedAccessView *>(hUnorderedAccessView.pDrvPrivate);
}


struct Query
{
   D3D10DDI_QUERY Type;
   UINT Flags;

   unsigned pipe_type;
   struct pipe_query *handle;
   INT SeqNo;
   UINT GetDataCount;
   bool Emulated;

   D3D10_DDI_QUERY_DATA_PIPELINE_STATISTICS Statistics;
};


static inline Query *
CastQuery(D3D10DDI_HQUERY hQuery)
{
   return static_cast<Query *>(hQuery.pDrvPrivate);
}


static inline struct pipe_query *
CastPipeQuery(D3D10DDI_HQUERY hQuery)
{
   Query *pQuery = CastQuery(hQuery);
   return pQuery ? pQuery->handle : NULL;
}
