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
 * Device.cpp --
 *    Functions that provide the 3D device functionality.
 */


#include "Draw.h"
#include "DxgiFns.h"
#include "InputAssembly.h"
#include "OutputMerger.h"
#include "Query.h"
#include "Rasterizer.h"
#include "Resource.h"
#include "Shader.h"
#include "State.h"
#include "Format.h"

#include "Debug.h"

#include "util/format/u_format.h"
#include "util/u_sampler.h"
#include "util/u_framebuffer.h"

#include "gdikmt/gdikmt.h"
#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

EXTERN_C struct pipe_screen *
d3d10_create_screen(gdikmt_device *device);

static void APIENTRY DestroyDevice(D3D10DDI_HDEVICE hDevice);
static void APIENTRY RelocateDeviceFuncs(D3D10DDI_HDEVICE hDevice,
                                __in struct D3D10DDI_DEVICEFUNCS *pDeviceFunctions);
static void APIENTRY RelocateDeviceFuncs1(D3D10DDI_HDEVICE hDevice,
                                __in struct D3D10_1DDI_DEVICEFUNCS *pDeviceFunctions);
static void APIENTRY RelocateDeviceFuncs11(D3D10DDI_HDEVICE hDevice,
                                __in struct D3D11DDI_DEVICEFUNCS *pDeviceFunctions);
static void APIENTRY Flush(D3D10DDI_HDEVICE hDevice);
static void APIENTRY CheckFormatSupport(D3D10DDI_HDEVICE hDevice, DXGI_FORMAT Format,
                               __out UINT *pFormatCaps);
static void APIENTRY CheckMultisampleQualityLevels(D3D10DDI_HDEVICE hDevice,
                                          DXGI_FORMAT Format,
                                          UINT SampleCount,
                                          __out UINT *pNumQualityLevels);
static void APIENTRY SetTextFilterSize(D3D10DDI_HDEVICE hDevice, UINT Width, UINT Height);
static bool IsYttriumMsaaDepthStencilLoadOnlyFormat(DXGI_FORMAT format);
static bool IsYttriumMsaaFormatSupported(struct pipe_screen *screen,
                                         DXGI_FORMAT dxgi_format,
                                         unsigned sample_count);

static void
DeviceResetCallback(void *data, enum pipe_reset_status status)
{
   Device *pDevice = static_cast<Device *>(data);
   if (!pDevice || pDevice->device.base.runtime_destroying ||
       !pDevice->UMCallbacks.pfnSetErrorCb)
      return;

   yttrium_gdi_user_logf(
      "d3d10umd: reporting device removed to runtime reset_status=%u device=%p\n",
      (unsigned)status, pDevice);
   pDevice->UMCallbacks.pfnSetErrorCb(pDevice->hRTCoreLayer,
                                      D3DDDIERR_DEVICEREMOVED);
}

static unsigned
FeatureLevelFromCreateDevice(const D3D10DDIARG_CREATEDEVICE *pCreateData)
{
#if SUPPORT_D3D11
   switch (pCreateData->Interface) {
   case D3D11_0_DDI_INTERFACE_VERSION:
   case D3D11_0_7_DDI_INTERFACE_VERSION:
#if SUPPORT_D3D11_1
   case D3D11_1_DDI_INTERFACE_VERSION:
#endif
#if SUPPORT_D3D_WDDM1_3
   case D3DWDDM1_3_DDI_INTERFACE_VERSION:
#endif
      switch (D3D11DDI_EXTRACT_3DPIPELINELEVEL_FROM_FLAGS(pCreateData->Flags)) {
#if SUPPORT_D3D11_1
      case D3D11_1DDI_3DPIPELINELEVEL_9_1:
         return 0x9100;
      case D3D11_1DDI_3DPIPELINELEVEL_9_2:
         return 0x9200;
      case D3D11_1DDI_3DPIPELINELEVEL_9_3:
         return 0x9300;
#endif
      case D3D11DDI_3DPIPELINELEVEL_10_1:
         return 0xa100;
      case D3D11DDI_3DPIPELINELEVEL_11_0:
         return 0xb000;
#if SUPPORT_D3D11_1
      case D3D11_1DDI_3DPIPELINELEVEL_11_1:
         return 0xb100;
#endif
      case D3D11DDI_3DPIPELINELEVEL_10_0:
      default:
         return 0xa000;
      }
   default:
      break;
   }
#endif

#if SUPPORT_D3D10_1
   if (pCreateData->Interface == D3D10_1_DDI_INTERFACE_VERSION ||
       pCreateData->Interface == D3D10_1_x_DDI_INTERFACE_VERSION ||
       pCreateData->Interface == D3D10_1_7_DDI_INTERFACE_VERSION) {
      return 0xa100;
   }
#endif
   return 0xa000;
}

#if SUPPORT_D3D11_1
static BOOL APIENTRY
Flush11_1(D3D10DDI_HDEVICE hDevice, UINT FlushFlags)
{
   Flush(hDevice);
   return TRUE;
}

static void APIENTRY
RelocateDeviceFuncs11_1(D3D10DDI_HDEVICE hDevice,
                        __in struct D3D11_1DDI_DEVICEFUNCS *pDeviceFunctions)
{
   LOG_ENTRYPOINT();
}

#if SUPPORT_D3D_WDDM1_3
static void APIENTRY
RelocateDeviceFuncsWDDM1_3(D3D10DDI_HDEVICE hDevice,
                           __in struct D3DWDDM1_3DDI_DEVICEFUNCS *pDeviceFunctions)
{
   LOG_ENTRYPOINT();
}

static void APIENTRY
CheckMultisampleQualityLevelsWDDM1_3(D3D10DDI_HDEVICE hDevice,
                                     DXGI_FORMAT Format,
                                     UINT SampleCount,
                                     UINT Flags,
                                     __out UINT *pNumQualityLevels)
{
   if (Flags) {
      *pNumQualityLevels = 0;
      if ((Flags == D3D10_1_DDIARG_STANDARD_MULTISAMPLE_PATTERN ||
           Flags == D3D10_1_DDIARG_CENTER_MULTISAMPLE_PATTERN)) {
         Device *device = CastDevice(hDevice);
         if (device &&
             device->pipe &&
             !IsYttriumMsaaDepthStencilLoadOnlyFormat(Format) &&
             IsYttriumMsaaFormatSupported(device->pipe->screen, Format, SampleCount))
            *pNumQualityLevels = 1;
      }

      return;
   }

   CheckMultisampleQualityLevels(hDevice, Format, SampleCount, pNumQualityLevels);
}

static void APIENTRY
UpdateTileMappingsWDDM1_3(D3D10DDI_HDEVICE hDevice,
                          D3D10DDI_HRESOURCE hTiledResource,
                          UINT NumTiledResourceRegions,
                          const D3DWDDM1_3DDI_TILED_RESOURCE_COORDINATE *pTiledResourceRegionStartCoords,
                          const D3DWDDM1_3DDI_TILE_REGION_SIZE *pTiledResourceRegionSizes,
                          D3D10DDI_HRESOURCE hTilePool,
                          UINT NumRanges,
                          const UINT *pRangeFlags,
                          const UINT *pTilePoolStartOffsets,
                          const UINT *pRangeTileCounts,
                          UINT Flags)
{
}

static void APIENTRY
CopyTileMappingsWDDM1_3(D3D10DDI_HDEVICE hDevice,
                        D3D10DDI_HRESOURCE hDestTiledResource,
                        const D3DWDDM1_3DDI_TILED_RESOURCE_COORDINATE *pDestRegionStartCoord,
                        D3D10DDI_HRESOURCE hSourceTiledResource,
                        const D3DWDDM1_3DDI_TILED_RESOURCE_COORDINATE *pSourceRegionStartCoord,
                        const D3DWDDM1_3DDI_TILE_REGION_SIZE *pTileRegionSize,
                        UINT Flags)
{
}

static void APIENTRY
CopyTilesWDDM1_3(D3D10DDI_HDEVICE hDevice,
                 D3D10DDI_HRESOURCE hTiledResource,
                 const D3DWDDM1_3DDI_TILED_RESOURCE_COORDINATE *pTileRegionStartCoord,
                 const D3DWDDM1_3DDI_TILE_REGION_SIZE *pTileRegionSize,
                 D3D10DDI_HRESOURCE hBuffer,
                 UINT64 BufferStartOffsetInBytes,
                 UINT Flags)
{
}

static void APIENTRY
UpdateTilesWDDM1_3(D3D10DDI_HDEVICE hDevice,
                   D3D10DDI_HRESOURCE hDestTiledResource,
                   const D3DWDDM1_3DDI_TILED_RESOURCE_COORDINATE *pDestTileRegionStartCoord,
                   const D3DWDDM1_3DDI_TILE_REGION_SIZE *pDestTileRegionSize,
                   const void *pSourceTileData,
                   UINT Flags)
{
}

static void APIENTRY
TiledResourceBarrierWDDM1_3(D3D10DDI_HDEVICE hDevice,
                            D3D11DDI_HANDLETYPE TiledResourceAccessBeforeBarrierHandleType,
                            void *hTiledResourceAccessBeforeBarrier,
                            D3D11DDI_HANDLETYPE TiledResourceAccessAfterBarrierHandleType,
                            void *hTiledResourceAccessAfterBarrier)
{
}

static void APIENTRY
GetMipPackingWDDM1_3(D3D10DDI_HDEVICE hDevice,
                     D3D10DDI_HRESOURCE hTiledResource,
                     UINT *pNumPackedMips,
                     UINT *pNumTilesForPackedMips)
{
   *pNumPackedMips = 0;
   *pNumTilesForPackedMips = 0;
}

static void APIENTRY
ResizeTilePoolWDDM1_3(D3D10DDI_HDEVICE hDevice,
                      D3D10DDI_HRESOURCE hTilePool,
                      UINT64 NewSizeInBytes)
{
}

static void APIENTRY
SetMarkerWDDM1_3(D3D10DDI_HDEVICE hDevice)
{
}

static void APIENTRY
SetMarkerModeWDDM1_3(D3D10DDI_HDEVICE hDevice,
                     D3DWDDM1_3DDI_MARKER_TYPE Type,
                     UINT Flags)
{
}
#endif

static void APIENTRY
ResourceCopyRegion11_1(D3D10DDI_HDEVICE hDevice, D3D10DDI_HRESOURCE hResource,
                       UINT DstSubResource, UINT DstX, UINT DstY, UINT DstZ,
                       D3D10DDI_HRESOURCE hSrcResource, UINT SrcSubResource,
                       __in_opt const D3D10_DDI_BOX *pSrcBox, UINT CopyFlags)
{
   ResourceCopyRegion(hDevice, hResource, DstSubResource, DstX, DstY, DstZ,
                      hSrcResource, SrcSubResource, pSrcBox);
}

static void APIENTRY
ResourceUpdateSubResourceUP11_1(D3D10DDI_HDEVICE hDevice,
                                D3D10DDI_HRESOURCE hResource,
                                UINT DstSubResource,
                                __in_opt const D3D10_DDI_BOX *pDstBox,
                                __in const void *pSysMemUP,
                                UINT RowPitch,
                                UINT DepthPitch,
                                UINT CopyFlags)
{
   ResourceUpdateSubResourceUP(hDevice, hResource, DstSubResource, pDstBox,
                               pSysMemUP, RowPitch, DepthPitch);
}

static void APIENTRY
VsSetConstantBuffers11_1(D3D10DDI_HDEVICE hDevice, UINT StartSlot,
                         UINT NumBuffers,
                         __in_ecount(NumBuffers) const D3D10DDI_HRESOURCE *phBuffers,
                         const UINT *pFirstConstant,
                         const UINT *pNumConstants)
{
   SetConstantBuffersRange(MESA_SHADER_VERTEX, hDevice, StartSlot, NumBuffers,
                           phBuffers, pFirstConstant, pNumConstants);
}

static void APIENTRY
PsSetConstantBuffers11_1(D3D10DDI_HDEVICE hDevice, UINT StartSlot,
                         UINT NumBuffers,
                         __in_ecount(NumBuffers) const D3D10DDI_HRESOURCE *phBuffers,
                         const UINT *pFirstConstant,
                         const UINT *pNumConstants)
{
   SetConstantBuffersRange(MESA_SHADER_FRAGMENT, hDevice, StartSlot, NumBuffers,
                           phBuffers, pFirstConstant, pNumConstants);
}

static void APIENTRY
GsSetConstantBuffers11_1(D3D10DDI_HDEVICE hDevice, UINT StartSlot,
                         UINT NumBuffers,
                         __in_ecount(NumBuffers) const D3D10DDI_HRESOURCE *phBuffers,
                         const UINT *pFirstConstant,
                         const UINT *pNumConstants)
{
   SetConstantBuffersRange(MESA_SHADER_GEOMETRY, hDevice, StartSlot, NumBuffers,
                           phBuffers, pFirstConstant, pNumConstants);
}

static void APIENTRY
HsSetConstantBuffers11_1(D3D10DDI_HDEVICE hDevice, UINT StartSlot,
                         UINT NumBuffers,
                         __in_ecount(NumBuffers) const D3D10DDI_HRESOURCE *phBuffers,
                         const UINT *pFirstConstant,
                         const UINT *pNumConstants)
{
   SetConstantBuffersRange(MESA_SHADER_TESS_CTRL, hDevice, StartSlot,
                           NumBuffers, phBuffers, pFirstConstant,
                           pNumConstants);
}

static void APIENTRY
DsSetConstantBuffers11_1(D3D10DDI_HDEVICE hDevice, UINT StartSlot,
                         UINT NumBuffers,
                         __in_ecount(NumBuffers) const D3D10DDI_HRESOURCE *phBuffers,
                         const UINT *pFirstConstant,
                         const UINT *pNumConstants)
{
   SetConstantBuffersRange(MESA_SHADER_TESS_EVAL, hDevice, StartSlot,
                           NumBuffers, phBuffers, pFirstConstant,
                           pNumConstants);
}

static void APIENTRY
CsSetConstantBuffers11_1(D3D10DDI_HDEVICE hDevice, UINT StartSlot,
                         UINT NumBuffers,
                         __in_ecount(NumBuffers) const D3D10DDI_HRESOURCE *phBuffers,
                         const UINT *pFirstConstant,
                         const UINT *pNumConstants)
{
   SetConstantBuffersRange(MESA_SHADER_COMPUTE, hDevice, StartSlot, NumBuffers,
                           phBuffers, pFirstConstant, pNumConstants);
}

static SIZE_T APIENTRY
CalcPrivateBlendStateSize11_1(D3D10DDI_HDEVICE hDevice,
                              __in const D3D11_1_DDI_BLEND_DESC *pBlendDesc)
{
   return sizeof(BlendState);
}

static void APIENTRY
CreateBlendState11_1(D3D10DDI_HDEVICE hDevice,
                     __in const D3D11_1_DDI_BLEND_DESC *pBlendDesc,
                     D3D10DDI_HBLENDSTATE hBlendState,
                     D3D10DDI_HRTBLENDSTATE hRTBlendState)
{
   CreateBlendState11_1Impl(hDevice, pBlendDesc, hBlendState, hRTBlendState);
}

static SIZE_T APIENTRY
CalcPrivateRasterizerStateSize11_1(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11_1_DDI_RASTERIZER_DESC *pRasterizerDesc)
{
   return CalcPrivateRasterizerStateSize(hDevice,
      reinterpret_cast<const D3D10_DDI_RASTERIZER_DESC *>(pRasterizerDesc));
}

static void APIENTRY
CreateRasterizerState11_1(D3D10DDI_HDEVICE hDevice,
                          __in const D3D11_1_DDI_RASTERIZER_DESC *pRasterizerDesc,
                          D3D10DDI_HRASTERIZERSTATE hRasterizerState,
                          D3D10DDI_HRTRASTERIZERSTATE hRTRasterizerState)
{
   CreateRasterizerState(hDevice,
      reinterpret_cast<const D3D10_DDI_RASTERIZER_DESC *>(pRasterizerDesc),
      hRasterizerState, hRTRasterizerState);

   RasterizerState *state = CastRasterizerState(hRasterizerState);
   if (state)
      state->forced_sample_count = pRasterizerDesc->ForcedSampleCount;
}

static SIZE_T APIENTRY
CalcPrivateShaderSize11_1(D3D10DDI_HDEVICE hDevice,
                          __in_ecount(pShaderCode[1]) const UINT *pShaderCode,
                          __in const D3D11_1DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   return CalcPrivateShaderSize(hDevice, pShaderCode, NULL);
}

struct StageSignatures11_1
{
   D3D10DDIARG_SIGNATURE_ENTRY input[128];
   D3D10DDIARG_SIGNATURE_ENTRY output[128];
   D3D10DDIARG_STAGE_IO_SIGNATURES signatures;
};

static bool
ConvertStageSignatures11_1(
   D3D10DDI_HDEVICE hDevice,
   const D3D11_1DDIARG_STAGE_IO_SIGNATURES *source,
   StageSignatures11_1 *destination,
   const D3D10DDIARG_STAGE_IO_SIGNATURES **signatures)
{
   if (!source) {
      *signatures = NULL;
      return true;
   }

   if (source->NumInputSignatureEntries > ARRAY_SIZE(destination->input) ||
       source->NumOutputSignatureEntries > ARRAY_SIZE(destination->output) ||
       (source->NumInputSignatureEntries && !source->pInputSignature) ||
       (source->NumOutputSignatureEntries && !source->pOutputSignature)) {
      SetError(hDevice, E_INVALIDARG);
      return false;
   }

   for (unsigned i = 0; i < source->NumInputSignatureEntries; ++i) {
      const D3D11_1DDIARG_SIGNATURE_ENTRY2 &entry =
         source->pInputSignature[i];
      destination->input[i] = {entry.SystemValue, entry.Register, entry.Mask};
   }
   for (unsigned i = 0; i < source->NumOutputSignatureEntries; ++i) {
      const D3D11_1DDIARG_SIGNATURE_ENTRY2 &entry =
         source->pOutputSignature[i];
      destination->output[i] = {entry.SystemValue, entry.Register, entry.Mask};
   }

   destination->signatures = {
      destination->input,
      source->NumInputSignatureEntries,
      destination->output,
      source->NumOutputSignatureEntries,
   };
   *signatures = &destination->signatures;
   return true;
}

static void APIENTRY
CreateVertexShader11_1(D3D10DDI_HDEVICE hDevice,
                       __in_ecount(pShaderCode[1]) const UINT *pShaderCode,
                       D3D10DDI_HSHADER hShader,
                       D3D10DDI_HRTSHADER hRTShader,
                       __in const D3D11_1DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   StageSignatures11_1 converted = {};
   const D3D10DDIARG_STAGE_IO_SIGNATURES *signatures;

   if (!ConvertStageSignatures11_1(hDevice, pSignatures, &converted,
                                   &signatures))
      return;
   CreateVertexShader(hDevice, pShaderCode, hShader, hRTShader, signatures);
}

static void APIENTRY
CreateGeometryShader11_1(D3D10DDI_HDEVICE hDevice,
                         __in_ecount(pShaderCode[1]) const UINT *pShaderCode,
                         D3D10DDI_HSHADER hShader,
                         D3D10DDI_HRTSHADER hRTShader,
                         __in const D3D11_1DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   StageSignatures11_1 converted = {};
   const D3D10DDIARG_STAGE_IO_SIGNATURES *signatures;

   if (!ConvertStageSignatures11_1(hDevice, pSignatures, &converted,
                                   &signatures))
      return;
   CreateGeometryShader(hDevice, pShaderCode, hShader, hRTShader, signatures);
}

static void APIENTRY
CreatePixelShader11_1(D3D10DDI_HDEVICE hDevice,
                      __in_ecount(pShaderCode[1]) const UINT *pShaderCode,
                      D3D10DDI_HSHADER hShader,
                      D3D10DDI_HRTSHADER hRTShader,
                      __in const D3D11_1DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   StageSignatures11_1 converted = {};
   const D3D10DDIARG_STAGE_IO_SIGNATURES *signatures;

   if (!ConvertStageSignatures11_1(hDevice, pSignatures, &converted,
                                   &signatures))
      return;
   CreatePixelShader(hDevice, pShaderCode, hShader, hRTShader, signatures);
}

static SIZE_T APIENTRY
CalcPrivateGeometryShaderWithStreamOutput11_1(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pCreateGeometryShaderWithStreamOutput,
   __in const D3D11_1DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   return CalcPrivateGeometryShaderWithStreamOutput11(hDevice,
      pCreateGeometryShaderWithStreamOutput, NULL);
}

static void APIENTRY
CreateGeometryShaderWithStreamOutput11_1(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pCreateGeometryShaderWithStreamOutput,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D11_1DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   StageSignatures11_1 converted = {};
   const D3D10DDIARG_STAGE_IO_SIGNATURES *signatures;

   if (!ConvertStageSignatures11_1(hDevice, pSignatures, &converted,
                                   &signatures))
      return;
   CreateGeometryShaderWithStreamOutput11(hDevice,
      pCreateGeometryShaderWithStreamOutput, hShader, hRTShader,
      signatures);
}

static SIZE_T APIENTRY
CalcPrivateTessellationShaderSize11_1(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount(pShaderCode[1]) const UINT *pShaderCode,
   __in const D3D11_1DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   return CalcPrivateTessellationShaderSize(hDevice, pShaderCode, NULL);
}

struct TessellationSignatures11_1
{
   D3D10DDIARG_SIGNATURE_ENTRY input[32];
   D3D10DDIARG_SIGNATURE_ENTRY output[32];
   D3D10DDIARG_SIGNATURE_ENTRY patch_constant[32];
   D3D11DDIARG_TESSELLATION_IO_SIGNATURES signatures;
};

static bool
ConvertTessellationSignatures11_1(
   D3D10DDI_HDEVICE hDevice,
   const D3D11_1DDIARG_TESSELLATION_IO_SIGNATURES *source,
   TessellationSignatures11_1 *destination,
   const D3D11DDIARG_TESSELLATION_IO_SIGNATURES **signatures)
{
   if (!source) {
      *signatures = NULL;
      return true;
   }

   if (source->NumInputSignatureEntries > ARRAY_SIZE(destination->input) ||
       source->NumOutputSignatureEntries > ARRAY_SIZE(destination->output) ||
       source->NumPatchConstantSignatureEntries >
          ARRAY_SIZE(destination->patch_constant)) {
      SetError(hDevice, E_INVALIDARG);
      return false;
   }

   for (unsigned i = 0; i < source->NumInputSignatureEntries; ++i) {
      const D3D11_1DDIARG_SIGNATURE_ENTRY2 &entry =
         source->pInputSignature[i];
      destination->input[i] = {entry.SystemValue, entry.Register, entry.Mask};
   }
   for (unsigned i = 0; i < source->NumOutputSignatureEntries; ++i) {
      const D3D11_1DDIARG_SIGNATURE_ENTRY2 &entry =
         source->pOutputSignature[i];
      destination->output[i] = {entry.SystemValue, entry.Register, entry.Mask};
   }
   for (unsigned i = 0; i < source->NumPatchConstantSignatureEntries; ++i) {
      const D3D11_1DDIARG_SIGNATURE_ENTRY2 &entry =
         source->pPatchConstantSignature[i];
      destination->patch_constant[i] =
         {entry.SystemValue, entry.Register, entry.Mask};
   }

   destination->signatures = {
      destination->input,
      source->NumInputSignatureEntries,
      destination->output,
      source->NumOutputSignatureEntries,
      destination->patch_constant,
      source->NumPatchConstantSignatureEntries,
   };
   *signatures = &destination->signatures;
   return true;
}

static void APIENTRY
CreateHullShader11_1(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount(pShaderCode[1]) const UINT *pShaderCode,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D11_1DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   TessellationSignatures11_1 converted;
   const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *signatures;

   if (!ConvertTessellationSignatures11_1(hDevice, pSignatures, &converted,
                                          &signatures))
      return;
   CreateHullShader(hDevice, pShaderCode, hShader, hRTShader, signatures);
}

static void APIENTRY
CreateDomainShader11_1(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount(pShaderCode[1]) const UINT *pShaderCode,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D11_1DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   TessellationSignatures11_1 converted;
   const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *signatures;

   if (!ConvertTessellationSignatures11_1(hDevice, pSignatures, &converted,
                                          &signatures))
      return;
   CreateDomainShader(hDevice, pShaderCode, hShader, hRTShader, signatures);
}

static void APIENTRY
Discard11_1(D3D10DDI_HDEVICE hDevice, D3D11DDI_HANDLETYPE HandleType,
            void *hResourceOrView, const D3D10_DDI_RECT *pRects,
            UINT NumRects)
{
}

static void APIENTRY
AssignDebugBinary11_1(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader,
                      UINT BinarySize, const void *pBinary)
{
}

static void APIENTRY
CheckDirectFlipSupport11_1(D3D10DDI_HDEVICE hDevice,
                           D3D10DDI_HRESOURCE hResource1,
                           D3D10DDI_HRESOURCE hResource2,
                           UINT CheckDirectFlipFlags,
                           __out BOOL *pSupported)
{
   if (pSupported)
      *pSupported = FALSE;
}

static void APIENTRY
ClearView11_1(D3D10DDI_HDEVICE hDevice, D3D11DDI_HANDLETYPE ViewType,
              void *hView, const FLOAT Color[4],
              const D3D10_DDI_RECT *pRects, UINT NumRects)
{
   ClearView(hDevice, ViewType, hView, Color, pRects, NumRects);
}

static void
FillDeviceFuncs11_1(D3D11_1DDI_DEVICEFUNCS *funcs)
{
   funcs->pfnDefaultConstantBufferUpdateSubresourceUP =
      ResourceUpdateSubResourceUP11_1;
   funcs->pfnVsSetConstantBuffers = VsSetConstantBuffers11_1;
   funcs->pfnPsSetShaderResources = PsSetShaderResources;
   funcs->pfnPsSetShader = PsSetShader;
   funcs->pfnPsSetSamplers = PsSetSamplers;
   funcs->pfnVsSetShader = VsSetShader;
   funcs->pfnDrawIndexed = DrawIndexed;
   funcs->pfnDraw = Draw;
   funcs->pfnDynamicIABufferMapNoOverwrite = ResourceMap;
   funcs->pfnDynamicIABufferUnmap = ResourceUnmap;
   funcs->pfnDynamicConstantBufferMapDiscard = ResourceMap;
   funcs->pfnDynamicIABufferMapDiscard = ResourceMap;
   funcs->pfnDynamicConstantBufferUnmap = ResourceUnmap;
   funcs->pfnPsSetConstantBuffers = PsSetConstantBuffers11_1;
   funcs->pfnIaSetInputLayout = IaSetInputLayout;
   funcs->pfnIaSetVertexBuffers = IaSetVertexBuffers;
   funcs->pfnIaSetIndexBuffer = IaSetIndexBuffer;
   funcs->pfnDrawIndexedInstanced = DrawIndexedInstanced;
   funcs->pfnDrawInstanced = DrawInstanced;
   funcs->pfnDynamicResourceMapDiscard = ResourceMap;
   funcs->pfnDynamicResourceUnmap = ResourceUnmap;
   funcs->pfnGsSetConstantBuffers = GsSetConstantBuffers11_1;
   funcs->pfnGsSetShader = GsSetShader;
   funcs->pfnIaSetTopology = IaSetTopology;
   funcs->pfnStagingResourceMap = ResourceMap;
   funcs->pfnStagingResourceUnmap = ResourceUnmap;
   funcs->pfnVsSetShaderResources = VsSetShaderResources;
   funcs->pfnVsSetSamplers = VsSetSamplers;
   funcs->pfnGsSetShaderResources = GsSetShaderResources;
   funcs->pfnGsSetSamplers = GsSetSamplers;
   funcs->pfnSetRenderTargets = SetRenderTargets11;
   funcs->pfnShaderResourceViewReadAfterWriteHazard =
      ShaderResourceViewReadAfterWriteHazard;
   funcs->pfnResourceReadAfterWriteHazard = ResourceReadAfterWriteHazard;
   funcs->pfnSetBlendState = SetBlendState;
   funcs->pfnSetDepthStencilState = SetDepthStencilState;
   funcs->pfnSetRasterizerState = SetRasterizerState;
   funcs->pfnQueryEnd = QueryEnd;
   funcs->pfnQueryBegin = QueryBegin;
   funcs->pfnResourceCopyRegion = ResourceCopyRegion11_1;
   funcs->pfnResourceUpdateSubresourceUP = ResourceUpdateSubResourceUP11_1;
   funcs->pfnSoSetTargets = SoSetTargets;
   funcs->pfnDrawAuto = DrawAuto;
   funcs->pfnSetViewports = SetViewports;
   funcs->pfnSetScissorRects = SetScissorRects;
   funcs->pfnClearRenderTargetView = ClearRenderTargetView;
   funcs->pfnClearDepthStencilView = ClearDepthStencilView;
   funcs->pfnSetPredication = SetPredication;
   funcs->pfnQueryGetData = QueryGetData;
   funcs->pfnFlush = Flush11_1;
   funcs->pfnGenMips = GenMips;
   funcs->pfnResourceCopy = ResourceCopy;
   funcs->pfnResourceResolveSubresource = ResourceResolveSubResource;
   funcs->pfnResourceMap = ResourceMap;
   funcs->pfnResourceUnmap = ResourceUnmap;
   funcs->pfnResourceIsStagingBusy = ResourceIsStagingBusy;
   funcs->pfnRelocateDeviceFuncs = RelocateDeviceFuncs11_1;
   funcs->pfnCalcPrivateResourceSize = CalcPrivateResourceSize11;
   funcs->pfnCalcPrivateOpenedResourceSize = CalcPrivateOpenedResourceSize;
   funcs->pfnCreateResource = CreateResource11;
   funcs->pfnOpenResource = OpenResource;
   funcs->pfnDestroyResource = DestroyResource;
   funcs->pfnCalcPrivateShaderResourceViewSize =
      CalcPrivateShaderResourceViewSize11;
   funcs->pfnCreateShaderResourceView = CreateShaderResourceView11;
   funcs->pfnDestroyShaderResourceView = DestroyShaderResourceView;
   funcs->pfnCalcPrivateRenderTargetViewSize =
      CalcPrivateRenderTargetViewSize;
   funcs->pfnCreateRenderTargetView = CreateRenderTargetView;
   funcs->pfnDestroyRenderTargetView = DestroyRenderTargetView;
   funcs->pfnCalcPrivateDepthStencilViewSize =
      CalcPrivateDepthStencilViewSize11;
   funcs->pfnCreateDepthStencilView = CreateDepthStencilView11;
   funcs->pfnDestroyDepthStencilView = DestroyDepthStencilView;
   funcs->pfnCalcPrivateElementLayoutSize = CalcPrivateElementLayoutSize;
   funcs->pfnCreateElementLayout = CreateElementLayout;
   funcs->pfnDestroyElementLayout = DestroyElementLayout;
   funcs->pfnCalcPrivateBlendStateSize = CalcPrivateBlendStateSize11_1;
   funcs->pfnCreateBlendState = CreateBlendState11_1;
   funcs->pfnDestroyBlendState = DestroyBlendState;
   funcs->pfnCalcPrivateDepthStencilStateSize =
      CalcPrivateDepthStencilStateSize;
   funcs->pfnCreateDepthStencilState = CreateDepthStencilState;
   funcs->pfnDestroyDepthStencilState = DestroyDepthStencilState;
   funcs->pfnCalcPrivateRasterizerStateSize =
      CalcPrivateRasterizerStateSize11_1;
   funcs->pfnCreateRasterizerState = CreateRasterizerState11_1;
   funcs->pfnDestroyRasterizerState = DestroyRasterizerState;
   funcs->pfnCalcPrivateShaderSize = CalcPrivateShaderSize11_1;
   funcs->pfnCreateVertexShader = CreateVertexShader11_1;
   funcs->pfnCreateGeometryShader = CreateGeometryShader11_1;
   funcs->pfnCreatePixelShader = CreatePixelShader11_1;
   funcs->pfnCalcPrivateGeometryShaderWithStreamOutput =
      CalcPrivateGeometryShaderWithStreamOutput11_1;
   funcs->pfnCreateGeometryShaderWithStreamOutput =
      CreateGeometryShaderWithStreamOutput11_1;
   funcs->pfnDestroyShader = DestroyShader;
   funcs->pfnCalcPrivateSamplerSize = CalcPrivateSamplerSize;
   funcs->pfnCreateSampler = CreateSampler;
   funcs->pfnDestroySampler = DestroySampler;
   funcs->pfnCalcPrivateQuerySize = CalcPrivateQuerySize;
   funcs->pfnCreateQuery = CreateQuery;
   funcs->pfnDestroyQuery = DestroyQuery;
   funcs->pfnCheckFormatSupport = CheckFormatSupport;
   funcs->pfnCheckMultisampleQualityLevels =
      CheckMultisampleQualityLevels;
   funcs->pfnCheckCounterInfo = CheckCounterInfo;
   funcs->pfnCheckCounter = CheckCounter;
   funcs->pfnDestroyDevice = DestroyDevice;
   funcs->pfnSetTextFilterSize = SetTextFilterSize;
   funcs->pfnResourceConvert = ResourceCopy;
   funcs->pfnResourceConvertRegion = ResourceCopyRegion11_1;
   funcs->pfnDrawIndexedInstancedIndirect = DrawIndexedInstancedIndirect;
   funcs->pfnDrawInstancedIndirect = DrawInstancedIndirect;
   funcs->pfnHsSetShaderResources = HsSetShaderResources;
   funcs->pfnHsSetShader = HsSetShader;
   funcs->pfnHsSetSamplers = HsSetSamplers;
   funcs->pfnHsSetConstantBuffers = HsSetConstantBuffers11_1;
   funcs->pfnDsSetShaderResources = DsSetShaderResources;
   funcs->pfnDsSetShader = DsSetShader;
   funcs->pfnDsSetSamplers = DsSetSamplers;
   funcs->pfnDsSetConstantBuffers = DsSetConstantBuffers11_1;
   funcs->pfnCreateHullShader = CreateHullShader11_1;
   funcs->pfnCreateDomainShader = CreateDomainShader11_1;
   funcs->pfnCalcPrivateTessellationShaderSize =
      CalcPrivateTessellationShaderSize11_1;
   funcs->pfnPsSetShaderWithIfaces = SetShaderWithIfaces;
   funcs->pfnVsSetShaderWithIfaces = SetShaderWithIfaces;
   funcs->pfnGsSetShaderWithIfaces = SetShaderWithIfaces;
   funcs->pfnHsSetShaderWithIfaces = SetShaderWithIfaces;
   funcs->pfnDsSetShaderWithIfaces = SetShaderWithIfaces;
   funcs->pfnCsSetShaderWithIfaces = SetShaderWithIfaces;
   funcs->pfnCreateComputeShader = CreateComputeShader;
   funcs->pfnCsSetShader = CsSetShader;
   funcs->pfnCsSetShaderResources = CsSetShaderResources;
   funcs->pfnCsSetSamplers = CsSetSamplers;
   funcs->pfnCsSetConstantBuffers = CsSetConstantBuffers11_1;
   funcs->pfnCalcPrivateUnorderedAccessViewSize =
      CalcPrivateUnorderedAccessViewSize;
   funcs->pfnCreateUnorderedAccessView = CreateUnorderedAccessView;
   funcs->pfnDestroyUnorderedAccessView = DestroyUnorderedAccessView;
   funcs->pfnClearUnorderedAccessViewUint =
      ClearUnorderedAccessViewUint;
   funcs->pfnClearUnorderedAccessViewFloat =
      ClearUnorderedAccessViewFloat;
   funcs->pfnCsSetUnorderedAccessViews = CsSetUnorderedAccessViews;
   funcs->pfnDispatch = Dispatch;
   funcs->pfnDispatchIndirect = DispatchIndirect;
   funcs->pfnSetResourceMinLOD = SetResourceMinLOD;
   funcs->pfnCopyStructureCount = CopyStructureCount;
   funcs->pfnDiscard = Discard11_1;
   funcs->pfnAssignDebugBinary = AssignDebugBinary11_1;
   funcs->pfnDynamicConstantBufferMapNoOverwrite = ResourceMap;
   funcs->pfnCheckDirectFlipSupport = CheckDirectFlipSupport11_1;
   funcs->pfnClearView = ClearView11_1;
}

#if SUPPORT_D3D_WDDM1_3
static void
FillDeviceFuncsWDDM1_3(D3DWDDM1_3DDI_DEVICEFUNCS *funcs)
{
   FillDeviceFuncs11_1(reinterpret_cast<D3D11_1DDI_DEVICEFUNCS *>(funcs));

   funcs->pfnRelocateDeviceFuncs = RelocateDeviceFuncsWDDM1_3;
   funcs->pfnCheckMultisampleQualityLevels =
      CheckMultisampleQualityLevelsWDDM1_3;

   funcs->pfnUpdateTileMappings = UpdateTileMappingsWDDM1_3;
   funcs->pfnCopyTileMappings = CopyTileMappingsWDDM1_3;
   funcs->pfnCopyTiles = CopyTilesWDDM1_3;
   funcs->pfnUpdateTiles = UpdateTilesWDDM1_3;
   funcs->pfnTiledResourceBarrier = TiledResourceBarrierWDDM1_3;
   funcs->pfnGetMipPacking = GetMipPackingWDDM1_3;
   funcs->pfnResizeTilePool = ResizeTilePoolWDDM1_3;
   funcs->pfnSetMarker = SetMarkerWDDM1_3;
   funcs->pfnSetMarkerMode = SetMarkerModeWDDM1_3;
}
#endif
#endif

EXTERN_C bool use_old_tex_ops = false;

/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateDeviceSize --
 *
 *    The CalcPrivateDeviceSize function determines the size of a memory
 *    region that the user-mode display driver requires from the Microsoft
 *    Direct3D runtime to store frequently-accessed data.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateDeviceSize(D3D10DDI_HADAPTER hAdapter,                          // IN
                      __in const D3D10DDIARG_CALCPRIVATEDEVICESIZE *pData) // IN
{
   return sizeof(Device);
}

/*
 * ----------------------------------------------------------------------
 *
 * CreateDevice --
 *
 *    The CreateDevice function creates a graphics context that is
 *    referenced in subsequent calls.
 *
 * ----------------------------------------------------------------------
 */

HRESULT APIENTRY
CreateDevice(D3D10DDI_HADAPTER hAdapter,                 // IN
             __in D3D10DDIARG_CREATEDEVICE *pCreateData) // IN
{
   LOG_ENTRYPOINT();

   if (0) {
      DebugPrintf("hAdapter = %p\n", hAdapter);
      DebugPrintf("pKTCallbacks = %p\n", pCreateData->pKTCallbacks);
      DebugPrintf("p10_1DeviceFuncs = %p\n", pCreateData->p10_1DeviceFuncs);
      DebugPrintf("hDrvDevice = %p\n", pCreateData->hDrvDevice);
      DebugPrintf("DXGIBaseDDI = %p\n", pCreateData->DXGIBaseDDI);
      DebugPrintf("hRTCoreLayer = %p\n", pCreateData->hRTCoreLayer);
      DebugPrintf("pUMCallbacks = %p\n", pCreateData->pUMCallbacks);
   }

   switch (pCreateData->Interface) {
   case D3D10_0_DDI_INTERFACE_VERSION:
   case D3D10_0_x_DDI_INTERFACE_VERSION:
   case D3D10_0_7_DDI_INTERFACE_VERSION:
#if SUPPORT_D3D10_1
   case D3D10_1_DDI_INTERFACE_VERSION:
   case D3D10_1_x_DDI_INTERFACE_VERSION:
   case D3D10_1_7_DDI_INTERFACE_VERSION:
#endif
#if SUPPORT_D3D11
   case D3D11_0_DDI_INTERFACE_VERSION:
   case D3D11_0_7_DDI_INTERFACE_VERSION:
#if SUPPORT_D3D11_1
   case D3D11_1_DDI_INTERFACE_VERSION:
#endif
#if SUPPORT_D3D_WDDM1_3
   case D3DWDDM1_3_DDI_INTERFACE_VERSION:
#endif
#endif
      break;
   default:
      DebugPrintf("%s: unsupported interface version 0x%08x\n",
                  __func__, pCreateData->Interface);
      return E_FAIL;
   }

   Adapter *pAdapter = CastAdapter(hAdapter);

   Device *pDevice = CastDevice(pCreateData->hDrvDevice);
   memset(pDevice, 0, sizeof *pDevice);
   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; i++)
      pDevice->vertex_buffers[i].is_user_buffer = true;
   pDevice->feature_level = FeatureLevelFromCreateDevice(pCreateData);

   pDevice->device.hRTAdapter = (HANDLE)pAdapter->hRTAdapter;
   pDevice->device.hRTDevice = (HANDLE)pCreateData->hRTDevice.handle;

   pDevice->device.KTCallbacks = *pCreateData->pKTCallbacks;
   pDevice->device.pAdapterCallbacks = &pAdapter->AdapterCallbacks;
   pDevice->device.pDXGIBaseCallbacks = pCreateData->DXGIBaseDDI.pDXGIBaseCallbacks;

   gdikmt_d3dddi_fill_basefuncs(&pDevice->device);

   HRESULT residency_hr = gdikmt_d3dddi_init_residency(
      &pDevice->device, pCreateData->pKTCallbacks, D3D_UMD_INTERFACE_VERSION);
   if (FAILED(residency_hr))
      return residency_hr;

   pDevice->hRTCoreLayer = pCreateData->hRTCoreLayer;
   pDevice->UMCallbacks = *pCreateData->pUMCallbacks;

   mtx_init(&pDevice->CreateResourceMtx, mtx_plain);
   list_inithead(&pDevice->resources);

   struct pipe_screen *screen = d3d10_create_screen(&pDevice->device.base);
   if (!screen) {
      DebugPrintf("%s: failed to create pipe screen\n", __func__);
      yttrium_gdi_user_logf("d3d10umd: CreateDevice failed to create pipe screen hAdapter=%p hDevice=%p\n",
                            hAdapter.pDrvPrivate,
                            pCreateData->hDrvDevice.pDrvPrivate);
      pDevice->device.base.destroy(&pDevice->device.base);
      mtx_destroy(&pDevice->CreateResourceMtx);
      return E_FAIL;
   }

   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   if (!pipe) {
      DebugPrintf("%s: failed to create pipe context\n", __func__);
      yttrium_gdi_user_logf("d3d10umd: CreateDevice failed to create pipe context hAdapter=%p hDevice=%p screen=%p\n",
                            hAdapter.pDrvPrivate,
                            pCreateData->hDrvDevice.pDrvPrivate,
                            screen);
      screen->destroy(screen);
      pDevice->device.base.destroy(&pDevice->device.base);
      mtx_destroy(&pDevice->CreateResourceMtx);
      return E_FAIL;
   }

   struct cso_context *cso = cso_create_context(pipe, CSO_NO_VBUF);
   if (!cso) {
      DebugPrintf("%s: failed to create CSO context\n", __func__);
      yttrium_gdi_user_logf("d3d10umd: CreateDevice failed to create CSO context hAdapter=%p hDevice=%p screen=%p pipe=%p\n",
                            hAdapter.pDrvPrivate,
                            pCreateData->hDrvDevice.pDrvPrivate,
                            screen,
                            pipe);
      pipe->destroy(pipe);
      screen->destroy(screen);
      pDevice->device.base.destroy(&pDevice->device.base);
      mtx_destroy(&pDevice->CreateResourceMtx);
      return E_FAIL;
   }

   pDevice->screen = screen;
   pDevice->pipe = pipe;
   pDevice->cso = cso;

   if (pipe->set_device_reset_callback) {
      struct pipe_device_reset_callback callback = {
         DeviceResetCallback,
         pDevice,
      };
      pipe->set_device_reset_callback(pipe, &callback);
   }

   pDevice->empty_vs = CreateEmptyShader(pDevice, MESA_SHADER_VERTEX);
   pDevice->empty_fs = CreateEmptyShader(pDevice, MESA_SHADER_FRAGMENT);
   struct pipe_depth_stencil_alpha_state default_dsa;
   memset(&default_dsa, 0, sizeof default_dsa);
   default_dsa.depth_enabled = 1;
   default_dsa.depth_writemask = 1;
   default_dsa.depth_func = PIPE_FUNC_LESS;
   pDevice->default_depth_stencil_state =
      pipe->create_depth_stencil_alpha_state(pipe, &default_dsa);
   if (!pDevice->default_depth_stencil_state) {
      DebugPrintf("%s: failed to create default depth/stencil state\n", __func__);
      DeleteEmptyShader(pDevice, MESA_SHADER_FRAGMENT, pDevice->empty_fs);
      DeleteEmptyShader(pDevice, MESA_SHADER_VERTEX, pDevice->empty_vs);
      cso_destroy_context(cso);
      pipe->destroy(pipe);
      screen->destroy(screen);
      pDevice->device.base.destroy(&pDevice->device.base);
      mtx_destroy(&pDevice->CreateResourceMtx);
      return E_FAIL;
   }

   pipe->bind_vs_state(pipe, pDevice->empty_vs);
   pipe->bind_fs_state(pipe, pDevice->empty_fs);

   pDevice->max_dual_source_render_targets =
         screen->caps.max_dual_source_render_targets;

   pDevice->draw_so_target = NULL;

   if (0) {
      DebugPrintf("pDevice = %p\n", pDevice);
   }

   st_debug_parse();
   
   // HACK: Use old texture operations if on virgl or Yttrium.
   const char *screen_name = screen->get_name(screen);
   const bool is_yttrium =
      screen_name &&
      (strncmp(screen_name, "Yttrium", 7) == 0 ||
       strncmp(screen_name, "yttrium", 7) == 0);
   pDevice->yttrium_sampler_binding_map_enabled = is_yttrium;
   pDevice->constant_publication_enabled =
      pipe->const_uploader && screen_name &&
      strcmp(screen_name, "yttrium") == 0 &&
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_CONSTANT_BUFFER_PUBLICATION", true) &&
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_WORKER", true);
   if ((screen_name && strncmp(screen_name, "virgl", 5) == 0) ||
       is_yttrium) {
      use_old_tex_ops = true;
   }

   /*
    * Fill in the D3D10 DDI functions
    */
#if SUPPORT_D3D11
   if (pCreateData->Interface == D3D11_0_DDI_INTERFACE_VERSION ||
       pCreateData->Interface == D3D11_0_7_DDI_INTERFACE_VERSION) {
      memset(pCreateData->p11DeviceFuncs, 0, sizeof(*pCreateData->p11DeviceFuncs));
#if SUPPORT_D3D11_1
   } else if (pCreateData->Interface == D3D11_1_DDI_INTERFACE_VERSION) {
      memset(pCreateData->p11_1DeviceFuncs, 0, sizeof(*pCreateData->p11_1DeviceFuncs));
#endif
#if SUPPORT_D3D_WDDM1_3
   } else if (pCreateData->Interface == D3DWDDM1_3_DDI_INTERFACE_VERSION) {
      memset(pCreateData->pWDDM1_3DeviceFuncs, 0, sizeof(*pCreateData->pWDDM1_3DeviceFuncs));
#endif
   }
#endif
   D3D10DDI_DEVICEFUNCS *pDeviceFuncs = pCreateData->pDeviceFuncs;
   pDeviceFuncs->pfnDefaultConstantBufferUpdateSubresourceUP = ResourceUpdateSubResourceUP;
   pDeviceFuncs->pfnVsSetConstantBuffers = VsSetConstantBuffers;
   pDeviceFuncs->pfnPsSetShaderResources = PsSetShaderResources;
   pDeviceFuncs->pfnPsSetShader = PsSetShader;
   pDeviceFuncs->pfnPsSetSamplers = PsSetSamplers;
   pDeviceFuncs->pfnVsSetShader = VsSetShader;
   pDeviceFuncs->pfnDrawIndexed = DrawIndexed;
   pDeviceFuncs->pfnDraw = Draw;
   pDeviceFuncs->pfnDynamicIABufferMapNoOverwrite = ResourceMap;
   pDeviceFuncs->pfnDynamicIABufferUnmap = ResourceUnmap;
   pDeviceFuncs->pfnDynamicConstantBufferMapDiscard = ResourceMap;
   pDeviceFuncs->pfnDynamicIABufferMapDiscard = ResourceMap;
   pDeviceFuncs->pfnDynamicConstantBufferUnmap = ResourceUnmap;
   pDeviceFuncs->pfnPsSetConstantBuffers = PsSetConstantBuffers;
   pDeviceFuncs->pfnIaSetInputLayout = IaSetInputLayout;
   pDeviceFuncs->pfnIaSetVertexBuffers = IaSetVertexBuffers;
   pDeviceFuncs->pfnIaSetIndexBuffer = IaSetIndexBuffer;
   pDeviceFuncs->pfnDrawIndexedInstanced = DrawIndexedInstanced;
   pDeviceFuncs->pfnDrawInstanced = DrawInstanced;
   pDeviceFuncs->pfnDynamicResourceMapDiscard = ResourceMap;
   pDeviceFuncs->pfnDynamicResourceUnmap = ResourceUnmap;
   pDeviceFuncs->pfnGsSetConstantBuffers = GsSetConstantBuffers;
   pDeviceFuncs->pfnGsSetShader = GsSetShader;
   pDeviceFuncs->pfnIaSetTopology = IaSetTopology;
   pDeviceFuncs->pfnStagingResourceMap = ResourceMap;
   pDeviceFuncs->pfnStagingResourceUnmap = ResourceUnmap;
   pDeviceFuncs->pfnVsSetShaderResources = VsSetShaderResources;
   pDeviceFuncs->pfnVsSetSamplers = VsSetSamplers;
   pDeviceFuncs->pfnGsSetShaderResources = GsSetShaderResources;
   pDeviceFuncs->pfnGsSetSamplers = GsSetSamplers;
   pDeviceFuncs->pfnSetRenderTargets = SetRenderTargets;
   pDeviceFuncs->pfnShaderResourceViewReadAfterWriteHazard = ShaderResourceViewReadAfterWriteHazard;
   pDeviceFuncs->pfnResourceReadAfterWriteHazard = ResourceReadAfterWriteHazard;
   pDeviceFuncs->pfnSetBlendState = SetBlendState;
   pDeviceFuncs->pfnSetDepthStencilState = SetDepthStencilState;
   pDeviceFuncs->pfnSetRasterizerState = SetRasterizerState;
   pDeviceFuncs->pfnQueryEnd = QueryEnd;
   pDeviceFuncs->pfnQueryBegin = QueryBegin;
   pDeviceFuncs->pfnResourceCopyRegion = ResourceCopyRegion;
   pDeviceFuncs->pfnResourceUpdateSubresourceUP = ResourceUpdateSubResourceUP;
   pDeviceFuncs->pfnSoSetTargets = SoSetTargets;
   pDeviceFuncs->pfnDrawAuto = DrawAuto;
   pDeviceFuncs->pfnSetViewports = SetViewports;
   pDeviceFuncs->pfnSetScissorRects = SetScissorRects;
   pDeviceFuncs->pfnClearRenderTargetView = ClearRenderTargetView;
   pDeviceFuncs->pfnClearDepthStencilView = ClearDepthStencilView;
   pDeviceFuncs->pfnSetPredication = SetPredication;
   pDeviceFuncs->pfnQueryGetData = QueryGetData;
   pDeviceFuncs->pfnFlush = Flush;
   pDeviceFuncs->pfnGenMips = GenMips;
   pDeviceFuncs->pfnResourceCopy = ResourceCopy;
   pDeviceFuncs->pfnResourceResolveSubresource = ResourceResolveSubResource;
   pDeviceFuncs->pfnResourceMap = ResourceMap;
   pDeviceFuncs->pfnResourceUnmap = ResourceUnmap;
   pDeviceFuncs->pfnResourceIsStagingBusy = ResourceIsStagingBusy;
   pDeviceFuncs->pfnRelocateDeviceFuncs = RelocateDeviceFuncs;
   pDeviceFuncs->pfnCalcPrivateResourceSize = CalcPrivateResourceSize;
   pDeviceFuncs->pfnCalcPrivateOpenedResourceSize = CalcPrivateOpenedResourceSize;
   pDeviceFuncs->pfnCreateResource = CreateResource;
   pDeviceFuncs->pfnOpenResource = OpenResource;
   pDeviceFuncs->pfnDestroyResource = DestroyResource;
   pDeviceFuncs->pfnCalcPrivateShaderResourceViewSize = CalcPrivateShaderResourceViewSize;
   pDeviceFuncs->pfnCreateShaderResourceView = CreateShaderResourceView;
   pDeviceFuncs->pfnDestroyShaderResourceView = DestroyShaderResourceView;
   pDeviceFuncs->pfnCalcPrivateRenderTargetViewSize = CalcPrivateRenderTargetViewSize;
   pDeviceFuncs->pfnCreateRenderTargetView = CreateRenderTargetView;
   pDeviceFuncs->pfnDestroyRenderTargetView = DestroyRenderTargetView;
   pDeviceFuncs->pfnCalcPrivateDepthStencilViewSize = CalcPrivateDepthStencilViewSize;
   pDeviceFuncs->pfnCreateDepthStencilView = CreateDepthStencilView;
   pDeviceFuncs->pfnDestroyDepthStencilView = DestroyDepthStencilView;
   pDeviceFuncs->pfnCalcPrivateElementLayoutSize = CalcPrivateElementLayoutSize;
   pDeviceFuncs->pfnCreateElementLayout = CreateElementLayout;
   pDeviceFuncs->pfnDestroyElementLayout = DestroyElementLayout;
   pDeviceFuncs->pfnCalcPrivateBlendStateSize = CalcPrivateBlendStateSize;
   pDeviceFuncs->pfnCreateBlendState = CreateBlendState;
   pDeviceFuncs->pfnDestroyBlendState = DestroyBlendState;
   pDeviceFuncs->pfnCalcPrivateDepthStencilStateSize = CalcPrivateDepthStencilStateSize;
   pDeviceFuncs->pfnCreateDepthStencilState = CreateDepthStencilState;
   pDeviceFuncs->pfnDestroyDepthStencilState = DestroyDepthStencilState;
   pDeviceFuncs->pfnCalcPrivateRasterizerStateSize = CalcPrivateRasterizerStateSize;
   pDeviceFuncs->pfnCreateRasterizerState = CreateRasterizerState;
   pDeviceFuncs->pfnDestroyRasterizerState = DestroyRasterizerState;
   pDeviceFuncs->pfnCalcPrivateShaderSize = CalcPrivateShaderSize;
   pDeviceFuncs->pfnCreateVertexShader = CreateVertexShader;
   pDeviceFuncs->pfnCreateGeometryShader = CreateGeometryShader;
   pDeviceFuncs->pfnCreatePixelShader = CreatePixelShader;
   pDeviceFuncs->pfnCalcPrivateGeometryShaderWithStreamOutput = CalcPrivateGeometryShaderWithStreamOutput;
   pDeviceFuncs->pfnCreateGeometryShaderWithStreamOutput = CreateGeometryShaderWithStreamOutput;
   pDeviceFuncs->pfnDestroyShader = DestroyShader;
   pDeviceFuncs->pfnCalcPrivateSamplerSize = CalcPrivateSamplerSize;
   pDeviceFuncs->pfnCreateSampler = CreateSampler;
   pDeviceFuncs->pfnDestroySampler = DestroySampler;
   pDeviceFuncs->pfnCalcPrivateQuerySize = CalcPrivateQuerySize;
   pDeviceFuncs->pfnCreateQuery = CreateQuery;
   pDeviceFuncs->pfnDestroyQuery = DestroyQuery;
   pDeviceFuncs->pfnCheckFormatSupport = CheckFormatSupport;
   pDeviceFuncs->pfnCheckMultisampleQualityLevels = CheckMultisampleQualityLevels;
   pDeviceFuncs->pfnCheckCounterInfo = CheckCounterInfo;
   pDeviceFuncs->pfnCheckCounter = CheckCounter;
   pDeviceFuncs->pfnDestroyDevice = DestroyDevice;
   pDeviceFuncs->pfnSetTextFilterSize = SetTextFilterSize;
#if SUPPORT_D3D11
   if (pCreateData->Interface == D3D11_0_DDI_INTERFACE_VERSION ||
       pCreateData->Interface == D3D11_0_7_DDI_INTERFACE_VERSION) {
      memcpy(pCreateData->p11DeviceFuncs, pDeviceFuncs,
             sizeof(*pDeviceFuncs));
   }
#endif
   if (pCreateData->Interface == D3D10_1_DDI_INTERFACE_VERSION ||
       pCreateData->Interface == D3D10_1_x_DDI_INTERFACE_VERSION ||
       pCreateData->Interface == D3D10_1_7_DDI_INTERFACE_VERSION) {
      D3D10_1DDI_DEVICEFUNCS *p10_1DeviceFuncs = pCreateData->p10_1DeviceFuncs;
      p10_1DeviceFuncs->pfnRelocateDeviceFuncs = RelocateDeviceFuncs1;
      p10_1DeviceFuncs->pfnCalcPrivateShaderResourceViewSize = CalcPrivateShaderResourceViewSize1;
      p10_1DeviceFuncs->pfnCreateShaderResourceView = CreateShaderResourceView1;
      p10_1DeviceFuncs->pfnCalcPrivateBlendStateSize = CalcPrivateBlendStateSize1;
      p10_1DeviceFuncs->pfnCreateBlendState = CreateBlendState1;
      p10_1DeviceFuncs->pfnResourceConvert = ResourceCopy;
      p10_1DeviceFuncs->pfnResourceConvertRegion = ResourceCopyRegion;
   }
#if SUPPORT_D3D11
   if (pCreateData->Interface == D3D11_0_DDI_INTERFACE_VERSION ||
       pCreateData->Interface == D3D11_0_7_DDI_INTERFACE_VERSION) {
      D3D11DDI_DEVICEFUNCS *p11DeviceFuncs = pCreateData->p11DeviceFuncs;
      p11DeviceFuncs->pfnSetRenderTargets = SetRenderTargets11;
      p11DeviceFuncs->pfnRelocateDeviceFuncs = RelocateDeviceFuncs11;
      p11DeviceFuncs->pfnCalcPrivateResourceSize = CalcPrivateResourceSize11;
      p11DeviceFuncs->pfnCreateResource = CreateResource11;
      p11DeviceFuncs->pfnCalcPrivateShaderResourceViewSize =
         CalcPrivateShaderResourceViewSize11;
      p11DeviceFuncs->pfnCreateShaderResourceView = CreateShaderResourceView11;
      p11DeviceFuncs->pfnCalcPrivateDepthStencilViewSize =
         CalcPrivateDepthStencilViewSize11;
      p11DeviceFuncs->pfnCreateDepthStencilView = CreateDepthStencilView11;
      p11DeviceFuncs->pfnCalcPrivateBlendStateSize =
         CalcPrivateBlendStateSize1;
      p11DeviceFuncs->pfnCreateBlendState = CreateBlendState1;
      p11DeviceFuncs->pfnCalcPrivateGeometryShaderWithStreamOutput =
         CalcPrivateGeometryShaderWithStreamOutput11;
      p11DeviceFuncs->pfnCreateGeometryShaderWithStreamOutput =
         CreateGeometryShaderWithStreamOutput11;
      p11DeviceFuncs->pfnDrawIndexedInstancedIndirect =
         DrawIndexedInstancedIndirect;
      p11DeviceFuncs->pfnDrawInstancedIndirect = DrawInstancedIndirect;
      p11DeviceFuncs->pfnHsSetShaderResources = HsSetShaderResources;
      p11DeviceFuncs->pfnHsSetShader = HsSetShader;
      p11DeviceFuncs->pfnHsSetSamplers = HsSetSamplers;
      p11DeviceFuncs->pfnHsSetConstantBuffers = HsSetConstantBuffers;
      p11DeviceFuncs->pfnDsSetShaderResources = DsSetShaderResources;
      p11DeviceFuncs->pfnDsSetShader = DsSetShader;
      p11DeviceFuncs->pfnDsSetSamplers = DsSetSamplers;
      p11DeviceFuncs->pfnDsSetConstantBuffers = DsSetConstantBuffers;
      p11DeviceFuncs->pfnCreateHullShader = CreateHullShader;
      p11DeviceFuncs->pfnCreateDomainShader = CreateDomainShader;
      p11DeviceFuncs->pfnCalcPrivateTessellationShaderSize =
         CalcPrivateTessellationShaderSize;
      p11DeviceFuncs->pfnPsSetShaderWithIfaces = SetShaderWithIfaces;
      p11DeviceFuncs->pfnVsSetShaderWithIfaces = SetShaderWithIfaces;
      p11DeviceFuncs->pfnGsSetShaderWithIfaces = SetShaderWithIfaces;
      p11DeviceFuncs->pfnHsSetShaderWithIfaces = SetShaderWithIfaces;
      p11DeviceFuncs->pfnDsSetShaderWithIfaces = SetShaderWithIfaces;
      p11DeviceFuncs->pfnCsSetShaderWithIfaces = SetShaderWithIfaces;
      p11DeviceFuncs->pfnCreateComputeShader = CreateComputeShader;
      p11DeviceFuncs->pfnCsSetShader = CsSetShader;
      p11DeviceFuncs->pfnCsSetShaderResources = CsSetShaderResources;
      p11DeviceFuncs->pfnCsSetSamplers = CsSetSamplers;
      p11DeviceFuncs->pfnCsSetConstantBuffers = CsSetConstantBuffers;
      p11DeviceFuncs->pfnCalcPrivateUnorderedAccessViewSize =
         CalcPrivateUnorderedAccessViewSize;
      p11DeviceFuncs->pfnCreateUnorderedAccessView =
         CreateUnorderedAccessView;
      p11DeviceFuncs->pfnDestroyUnorderedAccessView =
         DestroyUnorderedAccessView;
      p11DeviceFuncs->pfnClearUnorderedAccessViewUint =
         ClearUnorderedAccessViewUint;
      p11DeviceFuncs->pfnClearUnorderedAccessViewFloat =
         ClearUnorderedAccessViewFloat;
      p11DeviceFuncs->pfnCsSetUnorderedAccessViews =
         CsSetUnorderedAccessViews;
      p11DeviceFuncs->pfnDispatch = Dispatch;
      p11DeviceFuncs->pfnDispatchIndirect = DispatchIndirect;
      p11DeviceFuncs->pfnCopyStructureCount = CopyStructureCount;
      p11DeviceFuncs->pfnResourceConvert = ResourceCopy;
      p11DeviceFuncs->pfnResourceConvertRegion = ResourceCopyRegion;
      p11DeviceFuncs->pfnSetResourceMinLOD = SetResourceMinLOD;
   }
#if SUPPORT_D3D11_1
   else if (pCreateData->Interface == D3D11_1_DDI_INTERFACE_VERSION) {
      D3D11_1DDI_DEVICEFUNCS *p11_1DeviceFuncs = pCreateData->p11_1DeviceFuncs;

      FillDeviceFuncs11_1(p11_1DeviceFuncs);
   }
#endif
#if SUPPORT_D3D_WDDM1_3
   else if (pCreateData->Interface == D3DWDDM1_3_DDI_INTERFACE_VERSION) {
      D3DWDDM1_3DDI_DEVICEFUNCS *pWDDM1_3DeviceFuncs =
         pCreateData->pWDDM1_3DeviceFuncs;

      FillDeviceFuncsWDDM1_3(pWDDM1_3DeviceFuncs);
   }
#endif
#endif

   /*
    * Fill in DXGI DDI functions
    */
#if SUPPORT_D3D11_1
#if SUPPORT_D3D_WDDM1_3
   if (pCreateData->Interface == D3DWDDM1_3_DDI_INTERFACE_VERSION) {
      DXGI1_3_DDI_BASE_FUNCTIONS *dxgiFuncs =
         pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions4;

      dxgiFuncs->pfnPresent = _Present;
      dxgiFuncs->pfnGetGammaCaps = _GetGammaCaps;
      dxgiFuncs->pfnSetDisplayMode = _SetDisplayMode;
      dxgiFuncs->pfnSetResourcePriority = _SetResourcePriority;
      dxgiFuncs->pfnQueryResourceResidency = _QueryResourceResidency;
      dxgiFuncs->pfnRotateResourceIdentities = _RotateResourceIdentities;
      dxgiFuncs->pfnBlt = _Blt;
      dxgiFuncs->pfnResolveSharedResource = _ResolveSharedResource;
      dxgiFuncs->pfnBlt1 = _Blt1;
      dxgiFuncs->pfnOfferResources = _OfferResources;
      dxgiFuncs->pfnReclaimResources = _ReclaimResources;
      dxgiFuncs->pfnGetMultiplaneOverlayCaps = _GetMultiplaneOverlayCaps;
      dxgiFuncs->pfnGetMultiplaneOverlayGroupCaps =
         _GetMultiplaneOverlayGroupCaps;
      dxgiFuncs->pfnReserved1 = _DxgiReserved;
      dxgiFuncs->pfnPresentMultiplaneOverlay = _PresentMultiplaneOverlay;
      dxgiFuncs->pfnReserved2 = _DxgiReserved;
      dxgiFuncs->pfnPresent1 = _Present1;
      dxgiFuncs->pfnCheckPresentDurationSupport =
         _CheckPresentDurationSupport;
   } else
#endif
   if (pCreateData->Interface == D3D11_1_DDI_INTERFACE_VERSION) {
      DXGI1_2_DDI_BASE_FUNCTIONS *dxgiFuncs =
         (DXGI1_2_DDI_BASE_FUNCTIONS *)pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions;

      dxgiFuncs->pfnPresent = _Present;
      dxgiFuncs->pfnGetGammaCaps = _GetGammaCaps;
      dxgiFuncs->pfnSetDisplayMode = _SetDisplayMode;
      dxgiFuncs->pfnSetResourcePriority = _SetResourcePriority;
      dxgiFuncs->pfnQueryResourceResidency = _QueryResourceResidency;
      dxgiFuncs->pfnRotateResourceIdentities = _RotateResourceIdentities;
      dxgiFuncs->pfnBlt = _Blt;
      dxgiFuncs->pfnResolveSharedResource = _ResolveSharedResource;
      dxgiFuncs->pfnBlt1 = _Blt1;
      dxgiFuncs->pfnOfferResources = _OfferResources;
      dxgiFuncs->pfnReclaimResources = _ReclaimResources;
      dxgiFuncs->pfnGetMultiplaneOverlayCaps = NULL;
      dxgiFuncs->pfnGetMultiplaneOverlayFilterRange = NULL;
      dxgiFuncs->pfnCheckMultiplaneOverlaySupport = NULL;
      dxgiFuncs->pfnPresentMultiplaneOverlay = NULL;
   } else
#endif
   {
      pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions->pfnPresent =
         _Present;
      pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions->pfnGetGammaCaps =
         _GetGammaCaps;
      pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions->pfnSetDisplayMode =
         _SetDisplayMode;
      pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions->pfnSetResourcePriority =
         _SetResourcePriority;
      pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions->pfnQueryResourceResidency =
         _QueryResourceResidency;
      pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions->pfnRotateResourceIdentities =
         _RotateResourceIdentities;
      pCreateData->DXGIBaseDDI.pDXGIDDIBaseFunctions->pfnBlt =
         _Blt;
   }

   return S_OK;
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyDevice --
 *
 *    The DestroyDevice function destroys a graphics context.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroyDevice(D3D10DDI_HDEVICE hDevice)   // IN
{
   unsigned i;

   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;

   /* Device-associated scratch allocations which survive until this callback
    * are owned by the runtime's device teardown.  Mark the phase before any
    * Gallium cleanup can release one so the callback backend does not make an
    * invalid late pfnDeallocateCb call. */
   pDevice->device.base.runtime_destroying = true;

   ResourceEvent(RESOURCE_EVENT_DEVICE_DESTROY_BEGIN,
                 (uint64_t)(uintptr_t)hDevice.pDrvPrivate,
                 pDevice,
                 pDevice ? pDevice->pipe : NULL,
                 0, 0, 0,
                 (uint64_t)(uintptr_t)(pDevice ? pDevice->screen : NULL));

   if (gdikmt_device_get_reset_status(&pDevice->device.base) == PIPE_NO_RESET)
      yttrium_gdi_flush_labeled(pipe, NULL, 0,
                                "D3D device destroy flush");

   for (i = 0; i < PIPE_MAX_SO_BUFFERS; ++i) {
      pipe_so_target_reference(&pDevice->so_targets[i], NULL);
   }
   if (pDevice->draw_so_target) {
      pipe_so_target_reference(&pDevice->draw_so_target, NULL);
   }

   pipe->bind_fs_state(pipe, NULL);
   pipe->bind_vs_state(pipe, NULL);
   pipe->bind_depth_stencil_alpha_state(pipe, NULL);
   if (pDevice->default_depth_stencil_state) {
      pipe->delete_depth_stencil_alpha_state(
         pipe, pDevice->default_depth_stencil_state);
   }
   if (pDevice->default_rasterizer_discard_state) {
      pipe->delete_rasterizer_state(pipe,
                                    pDevice->default_rasterizer_discard_state);
   }
   cso_destroy_context(pDevice->cso);

   DeleteEmptyShader(pDevice, MESA_SHADER_FRAGMENT, pDevice->empty_fs);
   DeleteEmptyShader(pDevice, MESA_SHADER_VERTEX, pDevice->empty_vs);

   util_unreference_framebuffer_state(&pDevice->fb);

   for (i = 0; i < PIPE_MAX_ATTRIBS; ++i) {
      if (!pDevice->vertex_buffers[i].is_user_buffer) {
         pipe_resource_reference(&pDevice->vertex_buffers[i].buffer.resource, NULL);
      }
   }

   pipe_resource_reference(&pDevice->index_buffer, NULL);

   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; ++stage) {
      for (unsigned slot = 0; slot < PIPE_MAX_CONSTANT_BUFFERS; ++slot)
         pipe_resource_reference(&pDevice->constant_buffers[stage][slot], NULL);
   }

   static struct pipe_sampler_view * sampler_views[PIPE_MAX_SHADER_SAMPLER_VIEWS];
   memset(sampler_views, 0, sizeof sampler_views);
   pipe->set_sampler_views(pipe, MESA_SHADER_FRAGMENT, 0,
                           PIPE_MAX_SHADER_SAMPLER_VIEWS, 0, sampler_views);
   pipe->set_sampler_views(pipe, MESA_SHADER_VERTEX, 0,
                           PIPE_MAX_SHADER_SAMPLER_VIEWS, 0, sampler_views);
   pipe->set_sampler_views(pipe, MESA_SHADER_GEOMETRY, 0,
                           PIPE_MAX_SHADER_SAMPLER_VIEWS, 0, sampler_views);

   DestroyDeviceResourceDiagnostics(hDevice);

   pipe->destroy(pipe);
   ResourceEvent(RESOURCE_EVENT_DEVICE_DESTROY_PIPE,
                 (uint64_t)(uintptr_t)hDevice.pDrvPrivate,
                 pDevice, NULL, 0, 0, 0,
                 (uint64_t)(uintptr_t)pDevice->screen);
   pDevice->screen->destroy(pDevice->screen);
   pDevice->device.base.destroy(&pDevice->device.base);
   ResourceEvent(RESOURCE_EVENT_DEVICE_DESTROY_END,
                 (uint64_t)(uintptr_t)hDevice.pDrvPrivate,
                 pDevice, NULL, 0, 0, 0, 0);

   mtx_destroy(&pDevice->CreateResourceMtx);
}


/*
 * ----------------------------------------------------------------------
 *
 * RelocateDeviceFuncs --
 *
 *    The RelocateDeviceFuncs function notifies the user-mode
 *    display driver about the new location of the driver function table.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
RelocateDeviceFuncs(D3D10DDI_HDEVICE hDevice,                           // IN
                    __in struct D3D10DDI_DEVICEFUNCS *pDeviceFunctions) // IN
{
   LOG_ENTRYPOINT();

   /*
    * Nothing to do as we don't store a pointer to this entity.
    */
}


/*
 * ----------------------------------------------------------------------
 *
 * RelocateDeviceFuncs1 --
 *
 *    The RelocateDeviceFuncs1 function notifies the user-mode
 *    display driver about the new location of the driver function table.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
RelocateDeviceFuncs1(D3D10DDI_HDEVICE hDevice,                           // IN
                    __in struct D3D10_1DDI_DEVICEFUNCS *pDeviceFunctions) // IN
{
   LOG_ENTRYPOINT();

   /*
    * Nothing to do as we don't store a pointer to this entity.
    */
}

void APIENTRY
RelocateDeviceFuncs11(D3D10DDI_HDEVICE hDevice,
                      __in struct D3D11DDI_DEVICEFUNCS *pDeviceFunctions)
{
   LOG_ENTRYPOINT();

   /*
    * Nothing to do as we don't store a pointer to this entity.
    */
}


/*
 * ----------------------------------------------------------------------
 *
 * Flush --
 *
 *    The Flush function submits outstanding hardware commands that
 *    are in the hardware command buffer to the display miniport driver.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
Flush(D3D10DDI_HDEVICE hDevice)  // IN
{
   LOG_ENTRYPOINT();

   Device *device = CastDevice(hDevice);
   struct pipe_context *pipe = device->pipe;
   const char *screen_name =
      pipe->screen && pipe->screen->get_name ?
      pipe->screen->get_name(pipe->screen) : NULL;
   const unsigned flush_flags =
      screen_name && strcmp(screen_name, "yttrium") == 0 ?
      PIPE_FLUSH_ASYNC : 0;

   yttrium_gdi_flush_labeled(pipe, NULL, flush_flags,
                             "D3D10 DDI Flush");
}


/*
 * ----------------------------------------------------------------------
 *
 * CheckFormatSupport --
 *
 *    The CheckFormatSupport function retrieves the capabilites that
 *    the device has with the specified format.
 *
 * ----------------------------------------------------------------------
 */

static bool
is_format_supported_with_fallback(struct pipe_screen *screen,
                                  enum pipe_format format,
                                  enum pipe_texture_target target,
                                  unsigned sample_count,
                                  unsigned storage_sample_count,
                                  unsigned bind)
{
   if (screen->is_format_supported(screen, format, target, sample_count,
                                   storage_sample_count, bind))
      return true;

   enum pipe_format fallback = FormatFallback(format);
   return fallback != PIPE_FORMAT_NONE &&
          screen->is_format_supported(screen, fallback, target, sample_count,
                                      storage_sample_count, bind);
}

static bool
IsYttriumMsaaSampleCount(unsigned sample_count)
{
   return sample_count == 2 || sample_count == 4 || sample_count == 8;
}

/*
 * D3D11 runtime validation expects MSAA quality consistency across typeless
 * depth resources and their typed depth/scalar siblings. Plane-only SRV
 * formats can expose multisample loads, but must not report quality levels.
 */
static bool
IsYttriumMsaaDepthStencilStorageFormat(DXGI_FORMAT format)
{
   switch (format) {
   case DXGI_FORMAT_R32G8X24_TYPELESS:
   case DXGI_FORMAT_R32_TYPELESS:
   case DXGI_FORMAT_R24G8_TYPELESS:
   case DXGI_FORMAT_R16_TYPELESS:
   case DXGI_FORMAT_D16_UNORM:
   case DXGI_FORMAT_D32_FLOAT:
   case DXGI_FORMAT_D24_UNORM_S8_UINT:
   case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      return true;
   default:
      return false;
   }
}

static bool
IsYttriumMsaaDepthStencilViewFormat(DXGI_FORMAT format)
{
   switch (format) {
   case DXGI_FORMAT_R32_FLOAT:
   case DXGI_FORMAT_R16_UNORM:
      return true;
   default:
      return false;
   }
}

static bool
IsYttriumMsaaDepthStencilLoadOnlyFormat(DXGI_FORMAT format)
{
   switch (format) {
   case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
   case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
   case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
   case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
      return true;
   default:
      return false;
   }
}

static bool
IsYttriumMsaaDepthStencilFamilyFormat(DXGI_FORMAT format)
{
   return IsYttriumMsaaDepthStencilStorageFormat(format) ||
          IsYttriumMsaaDepthStencilViewFormat(format) ||
          IsYttriumMsaaDepthStencilLoadOnlyFormat(format);
}

static bool
IsYttriumTypelessFormat(DXGI_FORMAT format)
{
   switch (format) {
   case DXGI_FORMAT_R32G32B32A32_TYPELESS:
   case DXGI_FORMAT_R32G32B32_TYPELESS:
   case DXGI_FORMAT_R16G16B16A16_TYPELESS:
   case DXGI_FORMAT_R32G32_TYPELESS:
   case DXGI_FORMAT_R32G8X24_TYPELESS:
   case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
   case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
   case DXGI_FORMAT_R10G10B10A2_TYPELESS:
   case DXGI_FORMAT_R8G8B8A8_TYPELESS:
   case DXGI_FORMAT_R16G16_TYPELESS:
   case DXGI_FORMAT_R32_TYPELESS:
   case DXGI_FORMAT_R24G8_TYPELESS:
   case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
   case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
   case DXGI_FORMAT_R8G8_TYPELESS:
   case DXGI_FORMAT_R16_TYPELESS:
   case DXGI_FORMAT_R8_TYPELESS:
   case DXGI_FORMAT_BC1_TYPELESS:
   case DXGI_FORMAT_BC2_TYPELESS:
   case DXGI_FORMAT_BC3_TYPELESS:
   case DXGI_FORMAT_BC4_TYPELESS:
   case DXGI_FORMAT_BC5_TYPELESS:
   case DXGI_FORMAT_B8G8R8A8_TYPELESS:
   case DXGI_FORMAT_B8G8R8X8_TYPELESS:
   case DXGI_FORMAT_BC6H_TYPELESS:
   case DXGI_FORMAT_BC7_TYPELESS:
      return true;
   default:
      return false;
   }
}

static bool
IsYttriumMsaaRenderTargetFormat(DXGI_FORMAT dxgi_format)
{
   if (dxgi_format == DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM)
      return false;

   if (IsYttriumTypelessFormat(dxgi_format))
      return false;

   enum pipe_format format = FormatTranslate(dxgi_format, false);
   return format != PIPE_FORMAT_NONE &&
          !util_format_is_depth_or_stencil(format) &&
          !util_format_is_compressed(format);
}

static bool
IsYttriumMsaaTypelessRenderTargetFamilyFormat(DXGI_FORMAT format)
{
   switch (format) {
   case DXGI_FORMAT_R32G32B32A32_TYPELESS:
   case DXGI_FORMAT_R32G32B32_TYPELESS:
   case DXGI_FORMAT_R16G16B16A16_TYPELESS:
   case DXGI_FORMAT_R32G32_TYPELESS:
   case DXGI_FORMAT_R10G10B10A2_TYPELESS:
   case DXGI_FORMAT_R8G8B8A8_TYPELESS:
   case DXGI_FORMAT_R16G16_TYPELESS:
   case DXGI_FORMAT_R8G8_TYPELESS:
   case DXGI_FORMAT_R8_TYPELESS:
   case DXGI_FORMAT_B8G8R8A8_TYPELESS:
   case DXGI_FORMAT_B8G8R8X8_TYPELESS:
      return true;
   default:
      return false;
   }
}

static bool
IsYttriumMsaaFormat(DXGI_FORMAT format)
{
   return IsYttriumMsaaRenderTargetFormat(format) ||
          IsYttriumMsaaTypelessRenderTargetFamilyFormat(format) ||
          IsYttriumMsaaDepthStencilFamilyFormat(format);
}

static bool
IsYttriumMsaaFormatSupported(struct pipe_screen *screen,
                             DXGI_FORMAT dxgi_format,
                             unsigned sample_count)
{
   if (!IsYttriumMsaaSampleCount(sample_count) ||
       !IsYttriumMsaaFormat(dxgi_format))
      return false;

   const bool depth_stencil =
      IsYttriumMsaaDepthStencilStorageFormat(dxgi_format);
   enum pipe_format format = FormatTranslate(dxgi_format, depth_stencil);
   if (format == PIPE_FORMAT_NONE)
      return false;

   const unsigned bind = IsYttriumMsaaDepthStencilLoadOnlyFormat(dxgi_format) ?
      PIPE_BIND_SAMPLER_VIEW :
      (depth_stencil ? PIPE_BIND_DEPTH_STENCIL : PIPE_BIND_RENDER_TARGET);
   return is_format_supported_with_fallback(screen, format, PIPE_TEXTURE_2D,
                                            sample_count, sample_count, bind);
}

static bool
IsYttriumMsaaFormatSupportedAtAnySampleCount(struct pipe_screen *screen,
                                             DXGI_FORMAT format)
{
   return IsYttriumMsaaFormatSupported(screen, format, 2) ||
          IsYttriumMsaaFormatSupported(screen, format, 4) ||
          IsYttriumMsaaFormatSupported(screen, format, 8);
}

static bool
blend_format_feature_levels(DXGI_FORMAT format,
                            unsigned *required,
                            unsigned *optional)
{
   *optional = 0;

   switch (format) {
   case DXGI_FORMAT_R16G16B16A16_UNORM:
   case DXGI_FORMAT_R16G16_UNORM:
   case DXGI_FORMAT_R16_UNORM:
      *required = 0xa100;
      *optional = 0xa000;
      return true;
   case DXGI_FORMAT_R16G16B16A16_SNORM:
   case DXGI_FORMAT_R16G16_SNORM:
   case DXGI_FORMAT_R16_SNORM:
   case DXGI_FORMAT_R8G8B8A8_SNORM:
   case DXGI_FORMAT_R8G8_SNORM:
   case DXGI_FORMAT_R8_SNORM:
      *required = 0xa100;
      return true;
   case DXGI_FORMAT_R32G32B32A32_FLOAT:
   case DXGI_FORMAT_R32G32_FLOAT:
   case DXGI_FORMAT_R32_FLOAT:
   case DXGI_FORMAT_R10G10B10A2_UNORM:
   case DXGI_FORMAT_R11G11B10_FLOAT:
   case DXGI_FORMAT_R16_FLOAT:
      *required = 0xa000;
      return true;
   case DXGI_FORMAT_R16G16B16A16_FLOAT:
      *required = 0x9300;
      return true;
   case DXGI_FORMAT_R8G8B8A8_UNORM:
   case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
   case DXGI_FORMAT_R8G8_UNORM:
   case DXGI_FORMAT_R8_UNORM:
   case DXGI_FORMAT_A8_UNORM:
   case DXGI_FORMAT_B5G6R5_UNORM:
   case DXGI_FORMAT_B8G8R8A8_UNORM:
   case DXGI_FORMAT_B8G8R8X8_UNORM:
   case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
   case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
      *required = 0x9100;
      return true;
   default:
      return false;
   }
}

static bool
is_blend_format_supported_for_feature_level(DXGI_FORMAT format,
                                            unsigned feature_level)
{
   unsigned required;
   unsigned optional;

   if (!blend_format_feature_levels(format, &required, &optional))
      return false;

   return feature_level >= required || (optional && feature_level >= optional);
}

#if SUPPORT_D3D11_1
static bool
is_logic_op_format_supported(DXGI_FORMAT format)
{
   switch (format) {
   case DXGI_FORMAT_R32G32B32A32_UINT:
   case DXGI_FORMAT_R32G32B32_UINT:
   case DXGI_FORMAT_R16G16B16A16_UINT:
   case DXGI_FORMAT_R32G32_UINT:
   case DXGI_FORMAT_R10G10B10A2_UINT:
   case DXGI_FORMAT_R8G8B8A8_UINT:
   case DXGI_FORMAT_R16G16_UINT:
   case DXGI_FORMAT_R32_UINT:
   case DXGI_FORMAT_R8G8_UINT:
   case DXGI_FORMAT_R16_UINT:
   case DXGI_FORMAT_R8_UINT:
      return true;
   default:
      return false;
   }
}
#endif

void APIENTRY
CheckFormatSupport(D3D10DDI_HDEVICE hDevice, // IN
                   DXGI_FORMAT Format,       // IN
                   __out UINT *pFormatCaps)  // OUT
{
   //LOG_ENTRYPOINT();

   Device *device = CastDevice(hDevice);
   struct pipe_context *pipe = device->pipe;
   struct pipe_screen *screen = pipe->screen;

   *pFormatCaps = 0;

   enum pipe_format format = FormatTranslate(Format, false);
   if (format == PIPE_FORMAT_NONE) {
      *pFormatCaps = D3D10_DDI_FORMAT_SUPPORT_NOT_SUPPORTED;
      return;
   }

   if (Format == DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM) {
      /*
       * We only need to support creation.
       * http://msdn.microsoft.com/en-us/library/windows/hardware/ff552818.aspx
       */
      return;
   }

   const bool typeless_format = IsYttriumTypelessFormat(Format);
   const bool depth_stencil_format = util_format_is_depth_or_stencil(format);

   if (!typeless_format &&
       !depth_stencil_format &&
       is_format_supported_with_fallback(screen, format, PIPE_TEXTURE_2D, 0, 0,
                                         PIPE_BIND_RENDER_TARGET)) {
      *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_RENDERTARGET;
      *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_BLENDABLE;
#if SUPPORT_D3D11_1
      if (is_logic_op_format_supported(Format) &&
          yttrium_gdi_screen_supports_logic_op(screen))
         *pFormatCaps |= D3D11_1DDI_FORMAT_SUPPORT_OUTPUT_MERGER_LOGIC_OP;
#endif

#if SUPPORT_MSAA
      if (is_format_supported_with_fallback(screen, format, PIPE_TEXTURE_2D,
                                            4, 4, PIPE_BIND_RENDER_TARGET)) {
         *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET;
      }
#endif
      if (IsYttriumMsaaRenderTargetFormat(Format) &&
          IsYttriumMsaaFormatSupportedAtAnySampleCount(screen, Format)) {
         *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET;
      }
   }

   if (!typeless_format &&
       !depth_stencil_format &&
       is_blend_format_supported_for_feature_level(Format,
                                                   device->feature_level)) {
      *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_RENDERTARGET;
      *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_BLENDABLE;
   }

   if (is_format_supported_with_fallback(screen, format, PIPE_TEXTURE_2D, 0, 0,
                                         PIPE_BIND_SAMPLER_VIEW)) {
      *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_SHADER_SAMPLE;

#if SUPPORT_MSAA
      if (is_format_supported_with_fallback(screen, format, PIPE_TEXTURE_2D,
                                            4, 4, PIPE_BIND_RENDER_TARGET)) {
         *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_MULTISAMPLE_LOAD;
      }
#endif
      if (IsYttriumMsaaRenderTargetFormat(Format) &&
          IsYttriumMsaaFormatSupportedAtAnySampleCount(screen, Format)) {
         *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_MULTISAMPLE_LOAD;
      }
   }

   if (IsYttriumMsaaDepthStencilFamilyFormat(Format) &&
       IsYttriumMsaaFormatSupportedAtAnySampleCount(screen, Format)) {
      if (IsYttriumMsaaDepthStencilStorageFormat(Format) ||
          IsYttriumMsaaDepthStencilViewFormat(Format)) {
         *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET;
      }
      *pFormatCaps |= D3D10_DDI_FORMAT_SUPPORT_MULTISAMPLE_LOAD;
   }

#if SUPPORT_D3D11
   if (!typeless_format &&
       !depth_stencil_format &&
       is_format_supported_with_fallback(screen, format, PIPE_TEXTURE_2D, 0, 0,
                                         PIPE_BIND_SHADER_IMAGE)) {
      *pFormatCaps |= D3D11_1DDI_FORMAT_SUPPORT_UAV_WRITES;
   }
#endif

   if ((*pFormatCaps & D3D10_DDI_FORMAT_SUPPORT_BLENDABLE) &&
       !is_blend_format_supported_for_feature_level(Format,
                                                    device->feature_level)) {
      *pFormatCaps &= ~D3D10_DDI_FORMAT_SUPPORT_BLENDABLE;
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * CheckMultisampleQualityLevels --
 *
 *    The CheckMultisampleQualityLevels function retrieves the number
 *    of quality levels that the device supports for the specified
 *    number of samples.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CheckMultisampleQualityLevels(D3D10DDI_HDEVICE hDevice,        // IN
                              DXGI_FORMAT Format,              // IN
                              UINT SampleCount,                // IN
                              __out UINT *pNumQualityLevels)   // OUT
{
   *pNumQualityLevels = 0;

   if (!IsYttriumMsaaFormat(Format) ||
       IsYttriumMsaaDepthStencilLoadOnlyFormat(Format) ||
       !IsYttriumMsaaSampleCount(SampleCount))
      return;

   struct pipe_context *pipe = CastPipeContext(hDevice);
   if (!pipe || !pipe->screen)
      return;

   if (IsYttriumMsaaFormatSupported(pipe->screen, Format, SampleCount))
      *pNumQualityLevels = 1;
}


/*
 * ----------------------------------------------------------------------
 *
 * SetTextFilterSize --
 *
 *    The SetTextFilterSize function sets the width and height
 *    of the monochrome convolution filter.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
SetTextFilterSize(D3D10DDI_HDEVICE hDevice,  // IN
                  UINT Width,                // IN
                  UINT Height)               // IN
{
   LOG_ENTRYPOINT();

   LOG_UNSUPPORTED(Width != 1 || Height != 1);
}
