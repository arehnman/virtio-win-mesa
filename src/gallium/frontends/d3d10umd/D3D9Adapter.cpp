/*
 * SPDX-License-Identifier: MIT
 *
 * Copyright 2026 Ake Rehnman
 */
#include "D3D9Private.h"

#include <stddef.h>

extern "C" {
#include "gallium/winsys/yttrium/gdi/yttrium_venus.h"
}

static UINT
D3D9DriverVersion(const D3DDDIARG_OPENADAPTER *open_data)
{
   (void)open_data;
   return D3D_UMD_INTERFACE_VERSION_WIN7;
}

static float
D3D9FloatFromBits(uint32_t bits)
{
   float value;

   static_assert(sizeof(value) == sizeof(bits));
   memcpy(&value, &bits, sizeof(value));
   return value;
}

/* WARP is the oracle for the D3D9 runtime's adapter validation path.  Mirror
 * its DDI cap/format/query envelope closely, then layer in the compatibility
 * bits seen in older virtual WDDM drivers where they help legacy D3D9 apps.
 *
 */
static constexpr UINT d3d9_ms_types =
   (1u << (D3DDDIMULTISAMPLE_NONMASKABLE - 1)) |
   (1u << (D3DDDIMULTISAMPLE_2_SAMPLES - 1)) |
   (1u << (D3DDDIMULTISAMPLE_4_SAMPLES - 1)) |
   (1u << (D3DDDIMULTISAMPLE_8_SAMPLES - 1));

static const FORMATOP g_format_ops[] = {
   { D3DDDIFMT_X8R8G8B8, 0x009aec1f & ~FORMATOP_SRGBWRITE,
     d3d9_ms_types, 0, 0 },
   { D3DDDIFMT_R5G6B5, 0x008a6c1f, d3d9_ms_types, 0, 0 },
   { D3DDDIFMT_A8R8G8B8, 0x00dae01f,
     d3d9_ms_types, d3d9_ms_types, 0 },
   { D3DDDIFMT_A8B8G8R8, 0x00dae01f, d3d9_ms_types, 0, 0 },
   { D3DDDIFMT_A1R5G5B5, 0x00ca601f, d3d9_ms_types, 0, 0 },
   { D3DDDIFMT_X1R5G5B5, 0x00ca601f, d3d9_ms_types, 0, 0 },
   { D3DDDIFMT_A4R4G4B4, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_X4R4G4B4, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_A16B16G16R16, 0x00ca601f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_G16R16, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_A2B10G10R10, 0x00ca601f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_A16B16G16R16F, 0x00ca601f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_R16F, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_G16R16F, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_R32F, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_G32R32F, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_A32B32G32R32F, 0x00ca601f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_V8U8, 0x0083401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_Q8W8V8U8, 0x0083401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_V16U16, 0x0083401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_Q16W16V16U16, 0x0083401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_L8, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_L16, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3DDDIFMT_A8, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_A8L8, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3DDDIFMT_A2R10G10B10, 0x00ca601f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_DXT1, 0x00828007, 0, 0, 0 },
   { D3D9_FMT_DXT2, 0x00828007, 0, 0, 0 },
   { D3D9_FMT_DXT3, 0x00828007, 0, 0, 0 },
   { D3D9_FMT_DXT4, 0x00828007, 0, 0, 0 },
   { D3D9_FMT_DXT5, 0x00828007, 0, 0, 0 },
   { D3D9_FMT_ATI1, 0x0082d007, 0, 0, 8 },
   { D3D9_FMT_P8, 0x00824007, 0, 0, 0 },
   { D3D9_FMT_A8P8, 0x00820007, 0, 0, 0 },
   /* Auto depth-stencil creation can validate the blt MSAA mask even for a
    * fullscreen swapchain.  Both masks describe renderable sample counts;
    * neither advertises depth/stencil StretchRect support.
    */
   { D3DDDIFMT_D24S8, 0x000000c0, d3d9_ms_types, d3d9_ms_types, 0 },
   { D3DDDIFMT_D24X8, 0x000000c0, 0, 0, 0 },
   { D3DDDIFMT_D16, 0x000000c0, 0, 0, 0 },
   { D3D9_FMT_D16_LOCKABLE, 0x000000c0, 0, 0, 0 },
   { D3D9_FMT_D32F_LOCKABLE, 0x000000c0, 0, 0, 0 },
   { D3D9_FMT_DF16, 0x000410c1, 0, 0, 0 },
   { D3D9_FMT_DF24, 0x000410c1, 0, 0, 0 },
   { D3D9_FMT_INTZ, 0x000410c1, 0, 0, 0 },
   { D3D9_FMT_AYUV, 0x00005000, 0, 0, 32 },
   { D3D9_FMT_YUY2, 0x00005000, 0, 0, 16 },
   { D3D9_FMT_R8, 0x00c2401f, d3d9_ms_types, 0, 0 },
   { D3D9_FMT_G8R8, 0x00c2401f, d3d9_ms_types, 0, 0 },
};

static const D3DDDIQUERYTYPE g_query_types[] = {
   D3DDDIQUERYTYPE_EVENT,
   D3DDDIQUERYTYPE_OCCLUSION,
};

static void
D3D9FillCaps(const D3D9Adapter *adapter, D3DCAPS9 *caps)
{
   /* D3D9 removed the public name for the legacy D3DPTEXTURECAPS bit, but
    * the DDI caps field retains its numeric layout. */
   const DWORD texture_transparency_cap = 0x00000008u;

   memset(caps, 0, sizeof(*caps));

   caps->DeviceType = D3DDEVTYPE_HAL;
   caps->AdapterOrdinal = 0;
   caps->Caps = 0x00020000;
   caps->Caps2 = 0xe0020000;
   caps->Caps3 =
      0x000003a0 & ~D3DCAPS3_LINEAR_TO_SRGB_PRESENTATION;
   caps->PresentationIntervals = 0x8000000f;
   caps->CursorCaps = 0x00000001;
   caps->DevCaps = 0x061bbff0;
   /* POSITIONT depth is clamped, not clipped, so do not claim CLIPTLVERTS. */
   caps->PrimitiveMiscCaps =
      0x002eeff2 &
      ~(D3DPMISCCAPS_CLIPTLVERTS |
        D3DPMISCCAPS_CLIPPLANESCALEDPOINTS);
   caps->RasterCaps =
      0x0f732190 &
      ~D3DPRASTERCAPS_MULTISAMPLE_TOGGLE;
   caps->ZCmpCaps = 0x000000ff;
   caps->SrcBlendCaps =
      0x0000ffff &
      ~(D3DPBLENDCAPS_SRCCOLOR2 | D3DPBLENDCAPS_INVSRCCOLOR2);
   caps->DestBlendCaps =
      0x0000ffff &
      ~(D3DPBLENDCAPS_SRCCOLOR2 | D3DPBLENDCAPS_INVSRCCOLOR2 |
        D3DPBLENDCAPS_BOTHSRCALPHA | D3DPBLENDCAPS_BOTHINVSRCALPHA);
   caps->AlphaCmpCaps = 0x000000ff;
   caps->ShadeCaps = 0x00084208;
   caps->TextureCaps =
      0x0001ecdd & ~texture_transparency_cap;
   caps->TextureFilterCaps = 0x07030700;
   caps->CubeTextureFilterCaps = 0x07030700;
   caps->VolumeTextureFilterCaps = 0x07030700;
   /* The Windows 10 D3D9 runtime rejects CreateDevice before calling the UMD
    * when BORDER and MIRRORONCE are removed independently or together.  Keep
    * the runtime-required pair here.  SetTextureStageState accepts them for
    * compatibility and loudly records the transparent-black BORDER and
    * clamp-to-edge MIRRORONCE approximations.
    */
   caps->TextureAddressCaps = 0x0000003f;
   caps->VolumeTextureAddressCaps = 0x0000003f;
   caps->LineCaps = 0x0000003f & ~D3DLINECAPS_ANTIALIAS;
   caps->MaxTextureWidth = 0x00004000;
   caps->MaxTextureHeight = 0x00004000;
   caps->MaxVolumeExtent = 0x00004000;
   caps->MaxTextureRepeat = 0x00002000;
   caps->MaxTextureAspectRatio = 0x00002000;
   const UINT max_anisotropy = MIN2(
      MAX2((UINT)yttrium_venus_max_sampler_anisotropy(
              adapter ? adapter->msaa_venus : NULL), 1u),
      16u);
   caps->MaxAnisotropy = max_anisotropy;
   if (max_anisotropy <= 1) {
      const DWORD anisotropic_filter_caps =
         D3DPTFILTERCAPS_MINFANISOTROPIC |
         D3DPTFILTERCAPS_MAGFANISOTROPIC;
      caps->RasterCaps &= ~D3DPRASTERCAPS_ANISOTROPY;
      caps->TextureFilterCaps &= ~anisotropic_filter_caps;
      caps->CubeTextureFilterCaps &= ~anisotropic_filter_caps;
      caps->VolumeTextureFilterCaps &= ~anisotropic_filter_caps;
   }
   caps->MaxVertexW = D3D9FloatFromBits(0x501502f9);
   caps->GuardBandLeft = D3D9FloatFromBits(0xccbebc20);
   caps->GuardBandTop = D3D9FloatFromBits(0xccbebc20);
   caps->GuardBandRight = D3D9FloatFromBits(0x4cbebc20);
   caps->GuardBandBottom = D3D9FloatFromBits(0x4cbebc20);
   caps->ExtentsAdjust = 0.0f;
   caps->StencilCaps = 0x000001ff;
   caps->FVFCaps = 0x00100008;
   caps->TextureOpCaps = 0x03ffffff;
   caps->MaxTextureBlendStages = 0x00000008;
   caps->MaxSimultaneousTextures = 0x00000008;
   caps->VertexProcessingCaps = 0x0000017b;
   caps->MaxActiveLights = 0x00000008;
   caps->MaxUserClipPlanes = 0x00000008;
   caps->MaxVertexBlendMatrices = 0x00000004;
   caps->MaxVertexBlendMatrixIndex = 0x00000000;
   caps->MaxPointSize = D3D9FloatFromBits(0x46000000);
   caps->MaxPrimitiveCount = 0x007fffff;
   caps->MaxVertexIndex = 0x00ffffff;
   caps->MaxStreams = 0x00000010;
   caps->MaxStreamStride = 0x000000ff;
   caps->VertexShaderVersion = 0xfffe0300;
   caps->MaxVertexShaderConst = 0x00000100;
   caps->PixelShaderVersion = 0xffff0300;
   caps->PixelShader1xMaxValue = D3D9FloatFromBits(0x7f7fffff);
   caps->DevCaps2 =
      0x0000006f &
      ~(D3DDEVCAPS2_DMAPNPATCH |
        D3DDEVCAPS2_ADAPTIVETESSRTPATCH |
        D3DDEVCAPS2_ADAPTIVETESSNPATCH |
        D3DDEVCAPS2_PRESAMPLEDDMAPNPATCH);
   /* Windows validates this as at least 1.0 even when patch caps are absent. */
   caps->MaxNpatchTessellationLevel = 1.0f;
   caps->Reserved5 = 0;
   caps->MasterAdapterOrdinal = 0;
   caps->AdapterOrdinalInGroup = 0;
   caps->NumberOfAdaptersInGroup = 0;
   caps->DeclTypes =
      0x000003ff & ~(D3DDTCAPS_UDEC3 | D3DDTCAPS_DEC3N);
   caps->NumSimultaneousRTs = 0x00000004;
   caps->StretchRectFilterCaps = 0x03000300;
   caps->VS20Caps.Caps = 0x00000001;
   caps->VS20Caps.DynamicFlowControlDepth = 0x00000018;
   caps->VS20Caps.NumTemps = 0x00000020;
   caps->VS20Caps.StaticFlowControlDepth = 0x00000004;
   caps->PS20Caps.Caps = 0x0000001f;
   caps->PS20Caps.DynamicFlowControlDepth = 0x00000018;
   caps->PS20Caps.NumTemps = 0x00000020;
   caps->PS20Caps.StaticFlowControlDepth = 0x00000004;
   caps->PS20Caps.NumInstructionSlots = 0x00000200;
   caps->VertexTextureFilterCaps = 0x03030300;
   caps->MaxVShaderInstructionsExecuted = 0xffffffff;
   caps->MaxPShaderInstructionsExecuted = 0xffffffff;
   caps->MaxVertexShader30InstructionSlots = 0x00008000;
   caps->MaxPixelShader30InstructionSlots = 0x00008000;

}

/* D3DDDICAPS_GETD3D3CAPS uses the documented D3DHAL_GLOBALDRIVERDATA
 * layout.  Keep a private ABI definition here because the legacy d3dhal.h
 * header conflicts with the D3D9 headers in some supported toolchains.
 */
struct D3D9LegacyTransformCaps {
   DWORD dwSize;
   DWORD dwCaps;
};

struct D3D9LegacyLightingCaps {
   DWORD dwSize;
   DWORD dwCaps;
   DWORD dwLightingModel;
   DWORD dwNumLights;
};

struct D3D9LegacyPrimitiveCaps {
   DWORD dwSize;
   DWORD dwMiscCaps;
   DWORD dwRasterCaps;
   DWORD dwZCmpCaps;
   DWORD dwSrcBlendCaps;
   DWORD dwDestBlendCaps;
   DWORD dwAlphaCmpCaps;
   DWORD dwShadeCaps;
   DWORD dwTextureCaps;
   DWORD dwTextureFilterCaps;
   DWORD dwTextureBlendCaps;
   DWORD dwTextureAddressCaps;
   DWORD dwStippleWidth;
   DWORD dwStippleHeight;
};

struct D3D9LegacyDeviceDesc {
   DWORD dwSize;
   DWORD dwFlags;
   DWORD dcmColorModel;
   DWORD dwDevCaps;
   D3D9LegacyTransformCaps dtcTransformCaps;
   BOOL bClipping;
   D3D9LegacyLightingCaps dlcLightingCaps;
   D3D9LegacyPrimitiveCaps dpcLineCaps;
   D3D9LegacyPrimitiveCaps dpcTriCaps;
   DWORD dwDeviceRenderBitDepth;
   DWORD dwDeviceZBufferBitDepth;
   DWORD dwMaxBufferSize;
   DWORD dwMaxVertexCount;
};

struct D3D9LegacyGlobalDriverData {
   DWORD dwSize;
   D3D9LegacyDeviceDesc hwCaps;
   DWORD dwNumVertices;
   DWORD dwNumClipVertices;
   DWORD dwNumTextureFormats;
   void *lpTextureFormats;
};

static_assert(sizeof(D3D9LegacyTransformCaps) == 8);
static_assert(sizeof(D3D9LegacyLightingCaps) == 16);
static_assert(sizeof(D3D9LegacyPrimitiveCaps) == 56);
static_assert(sizeof(D3D9LegacyDeviceDesc) == 172);
static_assert(offsetof(D3D9LegacyGlobalDriverData, dwNumTextureFormats) == 184);
static_assert(sizeof(D3D9LegacyGlobalDriverData) ==
              (sizeof(void *) == 8 ? 200 : 192));

static constexpr DWORD d3d9_legacy_valid_color_model = 0x00000001u;
static constexpr DWORD d3d9_legacy_valid_dev_caps = 0x00000002u;
static constexpr DWORD d3d9_legacy_valid_render_bit_depth = 0x00000080u;
static constexpr DWORD d3d9_legacy_valid_z_bit_depth = 0x00000100u;
static constexpr DWORD d3d9_legacy_color_rgb = 2u;
static constexpr DWORD d3d9_legacy_depth_16 = 0x00000400u;
static constexpr DWORD d3d9_legacy_depth_24 = 0x00000200u;
static constexpr DWORD d3d9_legacy_depth_32 = 0x00000100u;

static constexpr DWORD d3d9_legacy_dev_float_tl_vertex = 0x00000001u;
static constexpr DWORD d3d9_legacy_dev_execute_system_memory = 0x00000010u;
static constexpr DWORD d3d9_legacy_dev_execute_video_memory = 0x00000020u;
static constexpr DWORD d3d9_legacy_dev_tl_vertex_system_memory = 0x00000040u;
static constexpr DWORD d3d9_legacy_dev_tl_vertex_video_memory = 0x00000080u;
static constexpr DWORD d3d9_legacy_dev_texture_system_memory = 0x00000100u;
static constexpr DWORD d3d9_legacy_dev_texture_video_memory = 0x00000200u;
static constexpr DWORD d3d9_legacy_dev_draw_prim_tl_vertex = 0x00000400u;
static constexpr DWORD d3d9_legacy_dev_render_after_flip = 0x00000800u;
static constexpr DWORD d3d9_legacy_dev_texture_nonlocal = 0x00001000u;
static constexpr DWORD d3d9_legacy_dev_draw_primitives2 = 0x00002000u;
static constexpr DWORD d3d9_legacy_dev_draw_primitives2_ex = 0x00008000u;
static constexpr DWORD d3d9_legacy_dev_hw_transform_light = 0x00010000u;
static constexpr DWORD d3d9_legacy_dev_blit_sys_to_nonlocal = 0x00020000u;
static constexpr DWORD d3d9_legacy_dev_hw_rasterization = 0x00080000u;

static void
D3D9FillLegacyCaps(const D3D9Adapter *adapter,
                   D3D9LegacyGlobalDriverData *caps)
{
   D3DCAPS9 d3d9_caps;
   D3D9FillCaps(adapter, &d3d9_caps);

   memset(caps, 0, sizeof(*caps));
   caps->dwSize = sizeof(*caps);

   D3D9LegacyDeviceDesc *hw = &caps->hwCaps;
   hw->dwSize = sizeof(*hw);
   hw->dwFlags = d3d9_legacy_valid_color_model |
                 d3d9_legacy_valid_dev_caps |
                 d3d9_legacy_valid_render_bit_depth |
                 d3d9_legacy_valid_z_bit_depth;
   hw->dcmColorModel = d3d9_legacy_color_rgb;

   const DWORD legacy_dev_caps =
      d3d9_legacy_dev_float_tl_vertex |
      d3d9_legacy_dev_execute_system_memory |
      d3d9_legacy_dev_tl_vertex_system_memory |
      d3d9_legacy_dev_texture_system_memory |
      d3d9_legacy_dev_texture_video_memory |
      d3d9_legacy_dev_draw_prim_tl_vertex |
      d3d9_legacy_dev_render_after_flip |
      d3d9_legacy_dev_texture_nonlocal |
      d3d9_legacy_dev_draw_primitives2 |
      d3d9_legacy_dev_draw_primitives2_ex |
      d3d9_legacy_dev_hw_transform_light |
      d3d9_legacy_dev_blit_sys_to_nonlocal |
      d3d9_legacy_dev_hw_rasterization;
   hw->dwDevCaps = d3d9_caps.DevCaps & legacy_dev_caps;

   /* These obsolete flags have mandatory values for legacy runtimes. */
   hw->dwDevCaps |= d3d9_legacy_dev_float_tl_vertex |
                    d3d9_legacy_dev_execute_system_memory |
                    d3d9_legacy_dev_tl_vertex_system_memory |
                    d3d9_legacy_dev_draw_prim_tl_vertex |
                    d3d9_legacy_dev_render_after_flip;
   hw->dwDevCaps &= ~(d3d9_legacy_dev_execute_video_memory |
                      d3d9_legacy_dev_tl_vertex_video_memory);

   hw->dtcTransformCaps.dwSize = sizeof(hw->dtcTransformCaps);
   hw->dlcLightingCaps.dwSize = sizeof(hw->dlcLightingCaps);
   hw->dpcLineCaps.dwSize = sizeof(hw->dpcLineCaps);
   hw->dpcTriCaps.dwSize = sizeof(hw->dpcTriCaps);
   hw->dwDeviceRenderBitDepth = d3d9_legacy_depth_16 |
                                d3d9_legacy_depth_32;
   hw->dwDeviceZBufferBitDepth = d3d9_legacy_depth_16 |
                                 d3d9_legacy_depth_24 |
                                 d3d9_legacy_depth_32;
}

static HRESULT
D3D9ValidateData(const D3DDDIARG_GETCAPS *data, UINT size)
{
   return data && data->pData && data->DataSize >= size ? S_OK : E_INVALIDARG;
}

static UINT
D3D9QueryMsaaTypes(D3D9Adapter *adapter)
{
   HDC hdc = GetDC(NULL);
   if (!hdc)
      return 0;

   struct gdikmt_device *device = gdikmt_create_from_hdc(hdc);
   ReleaseDC(NULL, hdc);
   if (!device)
      return 0;

   struct yttrium_venus *venus = yttrium_venus_create(device);
   if (!venus) {
      device->destroy(device);
      return 0;
   }

   const VkSampleCountFlags sample_counts =
      yttrium_venus_framebuffer_color_sample_counts(venus);
   UINT ms_types = 0;
   const D3DDDIMULTISAMPLE_TYPE sample_types[] = {
      D3DDDIMULTISAMPLE_2_SAMPLES,
      D3DDDIMULTISAMPLE_4_SAMPLES,
      D3DDDIMULTISAMPLE_8_SAMPLES,
   };
   const VkSampleCountFlagBits vk_sample_types[] = {
      VK_SAMPLE_COUNT_2_BIT,
      VK_SAMPLE_COUNT_4_BIT,
      VK_SAMPLE_COUNT_8_BIT,
   };
   for (UINT i = 0; i < ARRAYSIZE(sample_types); ++i) {
      if (sample_counts & vk_sample_types[i])
         ms_types |= 1u << (sample_types[i] - 1);
   }

   if (ms_types)
      ms_types |= 1u << (D3DDDIMULTISAMPLE_NONMASKABLE - 1);

   if (!sample_counts) {
      yttrium_venus_destroy(venus);
      device->destroy(device);
      return 0;
   }

   adapter->msaa_device = device;
   adapter->msaa_venus = venus;
   return ms_types & d3d9_ms_types;
}

static bool
D3D9FormatSupportsMultisample(const D3D9Adapter *adapter,
                              D3DDDIFORMAT format, BOOL flip,
                              D3DDDIMULTISAMPLE_TYPE ms_type)
{
   if (ms_type != D3DDDIMULTISAMPLE_NONE &&
       (ms_type < D3DDDIMULTISAMPLE_NONMASKABLE ||
        ms_type > D3DDDIMULTISAMPLE_16_SAMPLES))
      return false;

   for (UINT i = 0; i < ARRAYSIZE(g_format_ops); ++i) {
      if (g_format_ops[i].Format != format)
         continue;

      if (ms_type == D3DDDIMULTISAMPLE_NONE)
         return true;

      const UINT ms_mask = 1u << (ms_type - 1);
      const UINT supported =
         flip ? g_format_ops[i].FlipMsTypes : g_format_ops[i].BltMsTypes;
      return (supported & adapter->ms_types & ms_mask) != 0;
   }

   return false;
}

static HRESULT APIENTRY
D3D9CloseAdapter(HANDLE hAdapter)
{
   D3D9Adapter *adapter = (D3D9Adapter *)hAdapter;
   if (adapter) {
      yttrium_venus_destroy(adapter->msaa_venus);
      if (adapter->msaa_device)
         adapter->msaa_device->destroy(adapter->msaa_device);
      free(adapter);
   }
   return S_OK;
}

static HRESULT APIENTRY
D3D9CreateDevice(HANDLE hAdapter, D3DDDIARG_CREATEDEVICE *data)
{
   D3D9Tracef("CreateDevice hAdapter=%p data=%p pDeviceFuncs=%p hRTDevice=%p "
              "interface=0x%08x version=0x%08x flags=0x%08x "
              "cmd=%p/%u alloc=%p/%u patch=%p/%u\n",
              hAdapter, data, data ? data->pDeviceFuncs : NULL,
              data ? data->hDevice : NULL, data ? data->Interface : 0,
              data ? data->Version : 0, data ? data->Flags.Value : 0,
              data ? data->pCommandBuffer : NULL,
              data ? data->CommandBufferSize : 0,
              data ? data->pAllocationList : NULL,
              data ? data->AllocationListSize : 0,
              data ? data->pPatchLocationList : NULL,
              data ? data->PatchLocationListSize : 0);

   if (!hAdapter || !data || !data->pDeviceFuncs)
      return E_INVALIDARG;

   D3D9Adapter *adapter = (D3D9Adapter *)hAdapter;
   D3D9Device *device = (D3D9Device *)calloc(1, sizeof(*device));
   if (!device)
      return E_OUTOFMEMORY;

   device->hRTDevice = data->hDevice;
   if (data->pCallbacks)
      device->callbacks = *data->pCallbacks;

   device->gdi_device.hRTAdapter = adapter->hRTAdapter;
   device->gdi_device.hRTDevice = device->hRTDevice;
   device->gdi_device.KTCallbacks = device->callbacks;
   device->gdi_device.pAdapterCallbacks = &adapter->callbacks;
   device->gdi_device.use_legacy_signal_sync = true;
   gdikmt_d3dddi_fill_basefuncs(&device->gdi_device);

   HRESULT residency_hr = gdikmt_d3dddi_init_residency(
      &device->gdi_device, data->pCallbacks, data->Version);
   if (FAILED(residency_hr)) {
      free(device);
      return residency_hr;
   }

   device->screen = d3d10_create_screen(&device->gdi_device.base);
   if (!device->screen) {
      D3D9Tracef("CreateDevice failed to create Yttrium screen\n");
      device->gdi_device.base.destroy(&device->gdi_device.base);
      free(device);
      return E_FAIL;
   }

   device->pipe = device->screen->context_create(device->screen, NULL, 0);
   if (!device->pipe) {
      D3D9Tracef("CreateDevice failed to create Yttrium context\n");
      device->screen->destroy(device->screen);
      device->gdi_device.base.destroy(&device->gdi_device.base);
      free(device);
      return E_FAIL;
   }

   for (UINT i = 0; i < ARRAYSIZE(device->transforms); ++i)
      D3D9SetIdentityMatrix(&device->transforms[i]);
   union {
      float f;
      UINT u;
   } default_float = {1.0f};
   device->render_states[D3DDDIRS_LIGHTING] = TRUE;
   device->render_states[D3DDDIRS_LOCALVIEWER] = TRUE;
   device->render_states[D3DRS_FOGDENSITY] = default_float.u;
   device->render_states[D3DRS_POINTSIZE] = default_float.u;
   device->render_states[D3DRS_POINTSIZE_MIN] = default_float.u;
   default_float.f = 8192.0f;
   device->render_states[D3DRS_POINTSIZE_MAX] = default_float.u;
   default_float.f = 1.0f;
   device->render_states[D3DRS_POINTSCALE_A] = default_float.u;
   device->render_states[D3DRS_LASTPIXEL] = TRUE;
   device->render_states[D3DRS_MULTISAMPLEANTIALIAS] = TRUE;
   device->render_states[D3DRS_MULTISAMPLEMASK] = ~0u;
   device->render_states[D3DRS_BLENDFACTOR] = 0xffffffffu;
   device->render_states[D3DRS_STENCILMASK] = 0xffffffffu;
   device->render_states[D3DRS_STENCILWRITEMASK] = 0xffffffffu;
   device->render_states[D3DRS_FILLMODE] = D3DFILL_SOLID;
   const UINT default_color_write_mask =
      D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_GREEN |
      D3DCOLORWRITEENABLE_BLUE | D3DCOLORWRITEENABLE_ALPHA;
   device->render_states[D3DRS_COLORWRITEENABLE] = default_color_write_mask;
   device->render_states[D3DRS_COLORWRITEENABLE1] = default_color_write_mask;
   device->render_states[D3DRS_COLORWRITEENABLE2] = default_color_write_mask;
   device->render_states[D3DRS_COLORWRITEENABLE3] = default_color_write_mask;
   for (UINT stage = 0; stage < ARRAYSIZE(device->texture_stage_states);
        ++stage) {
      device->texture_stage_states[stage][D3DDDITSS_ADDRESSU] =
         D3DTADDRESS_WRAP;
      device->texture_stage_states[stage][D3DDDITSS_ADDRESSV] =
         D3DTADDRESS_WRAP;
      device->texture_stage_states[stage][D3DDDITSS_ADDRESSW] =
         D3DTADDRESS_WRAP;
      device->texture_stage_states[stage][D3DDDITSS_MINFILTER] =
         D3DTEXF_POINT;
      device->texture_stage_states[stage][D3DDDITSS_MAGFILTER] =
         D3DTEXF_POINT;
      device->texture_stage_states[stage][D3DDDITSS_MIPFILTER] =
         D3DTEXF_NONE;
      device->texture_stage_states[stage][D3DDDITSS_MAXANISOTROPY] = 1;
      device->texture_stage_states[stage]
                                  [D3DDDITSS_DISABLETEXTURECOLORKEY] = TRUE;
   }
   device->zrange.MinZ = 0.0f;
   device->zrange.MaxZ = 1.0f;

   D3D9FillDeviceFuncs(data->pDeviceFuncs);
   data->hDevice = device;
   return S_OK;
}

static HRESULT APIENTRY
D3D9GetCaps(HANDLE hAdapter, const D3DDDIARG_GETCAPS *data)
{
   D3D9Tracef("GetCaps enter hAdapter=%p data=%p type=%u pInfo=%p pData=%p size=%u\n",
              hAdapter, data, data ? data->Type : 0, data ? data->pInfo : NULL,
              data ? data->pData : NULL, data ? data->DataSize : 0);

   if (!hAdapter || !data)
      return E_INVALIDARG;

   const D3D9Adapter *adapter = (const D3D9Adapter *)hAdapter;

   switch (data->Type) {
   case D3DDDICAPS_DDRAW:
      if (FAILED(D3D9ValidateData(data, sizeof(DDRAW_CAPS))))
         return E_INVALIDARG;
      memset(data->pData, 0, data->DataSize);
      ((DDRAW_CAPS *)data->pData)->Caps = DDRAW_CAPS_ZBLTS |
                                          DDRAW_CAPS_BLTDEPTHFILL;
      ((DDRAW_CAPS *)data->pData)->Caps2 = DDRAW_CAPS2_DYNAMICTEXTURES;
      ((DDRAW_CAPS *)data->pData)->CKeyCaps = 0;
      ((DDRAW_CAPS *)data->pData)->FxCaps = 0;
      D3D9Tracef("GetCaps DDRAW caps=0x%08x caps2=0x%08x\n",
                 ((DDRAW_CAPS *)data->pData)->Caps,
                 ((DDRAW_CAPS *)data->pData)->Caps2);
      return S_OK;
   case D3DDDICAPS_DDRAW_MODE_SPECIFIC:
      if (FAILED(D3D9ValidateData(data, sizeof(DDRAW_MODE_SPECIFIC_CAPS))))
         return E_INVALIDARG;
      memset(data->pData, 0, data->DataSize);
      D3D9Tracef("GetCaps DDRAW_MODE_SPECIFIC head=%u\n",
                 ((DDRAW_MODE_SPECIFIC_CAPS *)data->pData)->Head);
      return S_OK;
   case D3DDDICAPS_GETFORMATCOUNT:
      if (FAILED(D3D9ValidateData(data, sizeof(UINT))))
         return E_INVALIDARG;
      *(UINT *)data->pData = ARRAYSIZE(g_format_ops);
      D3D9Tracef("GetCaps FORMATCOUNT count=%u\n", *(UINT *)data->pData);
      return S_OK;
   case D3DDDICAPS_GETFORMATDATA:
      if (FAILED(D3D9ValidateData(data, sizeof(g_format_ops))))
         return E_INVALIDARG;
      memcpy(data->pData, g_format_ops, sizeof(g_format_ops));
      for (UINT i = 0; i < ARRAYSIZE(g_format_ops); ++i) {
         FORMATOP *format_op = &((FORMATOP *)data->pData)[i];
         format_op->BltMsTypes &= adapter->ms_types;
         format_op->FlipMsTypes &= adapter->ms_types;
      }
      D3D9Tracef("GetCaps FORMATDATA bytes=%u count=%u x8r8g8b8=0x%08x "
                 "a8r8g8b8=0x%08x\n",
                 data->DataSize, (unsigned)ARRAYSIZE(g_format_ops),
                 g_format_ops[0].Operations, g_format_ops[2].Operations);
      return S_OK;
   case D3DDDICAPS_GETMULTISAMPLEQUALITYLEVELS:
      if (FAILED(D3D9ValidateData(data, sizeof(DDIMULTISAMPLEQUALITYLEVELSDATA))))
         return E_INVALIDARG;
      {
         DDIMULTISAMPLEQUALITYLEVELSDATA *ms =
            (DDIMULTISAMPLEQUALITYLEVELSDATA *)data->pData;
         ms->QualityLevels =
            D3D9FormatSupportsMultisample(adapter, ms->Format, ms->Flip,
                                          ms->MsType) ? 1 : 0;
         D3D9Tracef("GetCaps MSAA format=%u flip=%u type=%u quality=%u\n",
                    ms->Format, ms->Flip, ms->MsType, ms->QualityLevels);
      }
      return S_OK;
   case D3DDDICAPS_GETD3DQUERYCOUNT:
      if (FAILED(D3D9ValidateData(data, sizeof(UINT))))
         return E_INVALIDARG;
      *(UINT *)data->pData = ARRAYSIZE(g_query_types);
      D3D9Tracef("GetCaps QUERYCOUNT count=%u\n", *(UINT *)data->pData);
      return S_OK;
   case D3DDDICAPS_GETD3DQUERYDATA:
      if (FAILED(D3D9ValidateData(data, sizeof(g_query_types))))
         return E_INVALIDARG;
      memcpy(data->pData, g_query_types, sizeof(g_query_types));
      D3D9Tracef("GetCaps QUERYDATA bytes=%u count=%u\n",
                 data->DataSize, (unsigned)ARRAYSIZE(g_query_types));
      return S_OK;
   case D3DDDICAPS_GETD3D3CAPS:
      if (FAILED(D3D9ValidateData(data,
                                 sizeof(D3D9LegacyGlobalDriverData))))
         return E_INVALIDARG;
      D3D9FillLegacyCaps(adapter,
                         (D3D9LegacyGlobalDriverData *)data->pData);
      D3D9Tracef("GetCaps D3D3 size=%lu dev=0x%08lx render=0x%08lx "
                 "z=0x%08lx\n",
                 ((D3D9LegacyGlobalDriverData *)data->pData)->dwSize,
                 ((D3D9LegacyGlobalDriverData *)data->pData)
                    ->hwCaps.dwDevCaps,
                 ((D3D9LegacyGlobalDriverData *)data->pData)
                    ->hwCaps.dwDeviceRenderBitDepth,
                 ((D3D9LegacyGlobalDriverData *)data->pData)
                    ->hwCaps.dwDeviceZBufferBitDepth);
      return S_OK;
   case D3DDDICAPS_GETD3D9CAPS:
      if (FAILED(D3D9ValidateData(data, sizeof(D3DCAPS9))))
         return E_INVALIDARG;
      D3D9FillCaps(adapter, (D3DCAPS9 *)data->pData);
      D3D9Tracef("GetCaps D3D9 caps=0x%08lx caps2=0x%08lx "
                 "dev=0x%08lx dev2=0x%08lx vs=0x%08lx ps=0x%08lx\n",
                 ((D3DCAPS9 *)data->pData)->Caps,
                 ((D3DCAPS9 *)data->pData)->Caps2,
                 ((D3DCAPS9 *)data->pData)->DevCaps,
                 ((D3DCAPS9 *)data->pData)->DevCaps2,
                 ((D3DCAPS9 *)data->pData)->VertexShaderVersion,
                 ((D3DCAPS9 *)data->pData)->PixelShaderVersion);
      return S_OK;
   case D3DDDICAPS_GETGAMMARAMPCAPS:
      if (FAILED(D3D9ValidateData(data, sizeof(DDIGAMMACAPS))))
         return E_INVALIDARG;
      memset(data->pData, 0, data->DataSize);
      ((DDIGAMMACAPS *)data->pData)->GammaCaps = GAMMA_CAP_RGB256x3x16;
      D3D9Tracef("GetCaps GAMMA caps=0x%08x\n",
                 ((DDIGAMMACAPS *)data->pData)->GammaCaps);
      return S_OK;
   default:
      D3D9Tracef("GetCaps unsupported type=%u size=%u\n",
                 data->Type, data->DataSize);
      return E_NOTIMPL;
   }
}

EXTERN_C HRESULT APIENTRY
OpenAdapter(D3DDDIARG_OPENADAPTER *open_data)
{
   const UINT driver_version = D3D9DriverVersion(open_data);

   D3D9Tracef("OpenAdapter data=%p hAdapter=%p funcs=%p callbacks=%p version=0x%08x runtimeInterface=0x%08x runtimeVersion=0x%08x\n",
              open_data, open_data ? open_data->hAdapter : NULL,
              open_data ? open_data->pAdapterFuncs : NULL,
              open_data ? open_data->pAdapterCallbacks : NULL,
              driver_version, open_data ? open_data->Interface : 0,
              open_data ? open_data->Version : 0);

   if (!open_data || !open_data->pAdapterFuncs)
      return E_INVALIDARG;

   D3D9Adapter *adapter = (D3D9Adapter *)calloc(1, sizeof(*adapter));
   if (!adapter)
      return E_OUTOFMEMORY;

   adapter->hRTAdapter = open_data->hAdapter;
   if (open_data->pAdapterCallbacks)
      adapter->callbacks = *open_data->pAdapterCallbacks;
   adapter->ms_types = D3D9QueryMsaaTypes(adapter);

   open_data->hAdapter = adapter;
   open_data->pAdapterFuncs->pfnGetCaps = D3D9GetCaps;
   open_data->pAdapterFuncs->pfnCreateDevice = D3D9CreateDevice;
   open_data->pAdapterFuncs->pfnCloseAdapter = D3D9CloseAdapter;
   open_data->DriverVersion = driver_version;
   return S_OK;
}
