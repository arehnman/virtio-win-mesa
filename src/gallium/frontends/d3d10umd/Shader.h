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
 * Shader.h --
 *    Functions that manipulate shader resources.
 */

#ifndef SHADER_H
#define SHADER_H

#include "DriverIncludes.h"

struct Device;
struct Shader;
struct UnorderedAccessView;

void BindShaderImages(Device *device, mesa_shader_stage stage,
                      unsigned start_slot = 0, unsigned num_views = 0);
void ResetUAVCounter(Device *device, UnorderedAccessView *view, UINT value);

void *
CreateEmptyShader(Device *pDevice,
                  mesa_shader_stage processor);

void
DeleteEmptyShader(Device *pDevice,
                  mesa_shader_stage processor, void *handle);

unsigned
ShaderFindOutputMapping(Shader *shader, unsigned registerIndex);

SIZE_T APIENTRY
CalcPrivateShaderSize(D3D10DDI_HDEVICE hDevice,
                      __in_ecount (pShaderCode[1]) const UINT *pShaderCode,
                      __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);

void APIENTRY
DestroyShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader);

SIZE_T APIENTRY
CalcPrivateSamplerSize(D3D10DDI_HDEVICE hDevice,
                       __in const D3D10_DDI_SAMPLER_DESC *pSamplerDesc);

void APIENTRY CreateSampler(D3D10DDI_HDEVICE hDevice,
                   __in const D3D10_DDI_SAMPLER_DESC *pSamplerDesc,
                   D3D10DDI_HSAMPLER hSampler, D3D10DDI_HRTSAMPLER hRTSampler);

void APIENTRY DestroySampler(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSAMPLER hSampler);

void APIENTRY CreateVertexShader(D3D10DDI_HDEVICE hDevice,
                        __in_ecount (pShaderCode[1]) const UINT *pCode,
                        D3D10DDI_HSHADER hShader, D3D10DDI_HRTSHADER hRTShader,
                        __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);

void APIENTRY VsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader);

void APIENTRY VsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews);
void APIENTRY VsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                          __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers);

void APIENTRY VsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
                   __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers);

void APIENTRY CreateGeometryShader(D3D10DDI_HDEVICE hDevice,
                          __in_ecount (pShaderCode[1]) const UINT *pCode,
                          D3D10DDI_HSHADER hShader, D3D10DDI_HRTSHADER hRTShader,
                          __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);

void APIENTRY GsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader);

void APIENTRY GsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews);

void APIENTRY GsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                          __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers);

void APIENTRY GsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
                   __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers);

SIZE_T APIENTRY CalcPrivateGeometryShaderWithStreamOutput(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D10DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pCreateGeometryShaderWithStreamOutput,
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);

void APIENTRY CreateGeometryShaderWithStreamOutput(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D10DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pCreateGeometryShaderWithStreamOutput,
   D3D10DDI_HSHADER hShader, D3D10DDI_HRTSHADER hRTShader,
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);

SIZE_T APIENTRY CalcPrivateGeometryShaderWithStreamOutput11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pCreateGeometryShaderWithStreamOutput,
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);

void APIENTRY CreateGeometryShaderWithStreamOutput11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pCreateGeometryShaderWithStreamOutput,
   D3D10DDI_HSHADER hShader, D3D10DDI_HRTSHADER hRTShader,
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);

void APIENTRY SoSetTargets(D3D10DDI_HDEVICE hDevice, UINT SOTargets, UINT ClearTargets,
                  __in_ecount (SOTargets) const D3D10DDI_HRESOURCE *phResource,
                  __in_ecount (SOTargets) const UINT *pOffsets);

void APIENTRY CreatePixelShader(D3D10DDI_HDEVICE hDevice,
                       __in_ecount (pShaderCode[1]) const UINT *pCode,
                       D3D10DDI_HSHADER hShader, D3D10DDI_HRTSHADER hRTShader,
                       __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures);
void APIENTRY PsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader);
bool RunPixelShaderEmulation(Device *pDevice);
bool RunVertexShaderEmulation(Device *pDevice, unsigned vertex_count);

void APIENTRY PsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews);

void APIENTRY PsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                          __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers);

void APIENTRY PsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
                   __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers);

SIZE_T APIENTRY CalcPrivateTessellationShaderSize(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (pShaderCode[1]) const UINT *pCode,
   __in const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures);

void APIENTRY CreateHullShader(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (pShaderCode[1]) const UINT *pCode,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures);

void APIENTRY CreateDomainShader(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (pShaderCode[1]) const UINT *pCode,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures);

void APIENTRY HsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader);
void APIENTRY DsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader);

void APIENTRY HsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews);

void APIENTRY DsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews);

void APIENTRY HsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                          __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers);

void APIENTRY DsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                          __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers);

void APIENTRY HsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
                   __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers);

void APIENTRY DsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
                   __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers);

void APIENTRY SetShaderWithIfaces(
   D3D10DDI_HDEVICE hDevice,
   D3D10DDI_HSHADER hShader,
   UINT NumClassInstances,
   __in_ecount (NumClassInstances) const UINT *pIfaces,
   __in_ecount (NumClassInstances) const D3D11DDIARG_POINTERDATA *pPointerData);

void APIENTRY CreateComputeShader(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (pShaderCode[1]) const UINT *pCode,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader);

void APIENTRY CsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader);

void APIENTRY CsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews);

void APIENTRY CsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                          __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers);

void
SetConstantBuffersRange(mesa_shader_stage shader_type,
                        D3D10DDI_HDEVICE hDevice,
                        UINT StartBuffer,
                        UINT NumBuffers,
                        const D3D10DDI_HRESOURCE *phBuffers,
                        const UINT *pFirstConstant,
                        const UINT *pNumConstants);

void
UpdateBufferInfoConstants(Device *pDevice, mesa_shader_stage shader_type);

void
UpdateBufferInfoUavConstants(Device *pDevice,
                             mesa_shader_stage shader_type,
                             unsigned first_slot,
                             unsigned num_slots);

void
UpdateBufferInfoSampleConstants(Device *pDevice,
                                mesa_shader_stage shader_type);

void APIENTRY CsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
                   __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers);

void APIENTRY Dispatch(D3D10DDI_HDEVICE hDevice,
                       UINT ThreadGroupCountX,
                       UINT ThreadGroupCountY,
                       UINT ThreadGroupCountZ);

void APIENTRY DispatchIndirect(D3D10DDI_HDEVICE hDevice,
                               D3D10DDI_HRESOURCE hBufferForArgs,
                               UINT AlignedByteOffsetForArgs);

void APIENTRY CopyStructureCount(D3D10DDI_HDEVICE hDevice,
                                 D3D10DDI_HRESOURCE hDstBuffer,
                                 UINT DstAlignedByteOffset,
                                 D3D11DDI_HUNORDEREDACCESSVIEW hSrcView);

void APIENTRY ShaderResourceViewReadAfterWriteHazard(
   D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,
   D3D10DDI_HRESOURCE hResource);

void RefreshBoundShaderResourceViews(Device *pDevice);

SIZE_T APIENTRY CalcPrivateShaderResourceViewSize(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D10DDIARG_CREATESHADERRESOURCEVIEW *pCreateShaderResourceView);

void APIENTRY CreateShaderResourceView(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D10DDIARG_CREATESHADERRESOURCEVIEW *pCreateShaderResourceView,
   D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,
   D3D10DDI_HRTSHADERRESOURCEVIEW hRTShaderResourceView);

SIZE_T APIENTRY CalcPrivateShaderResourceViewSize1(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D10_1DDIARG_CREATESHADERRESOURCEVIEW *pCreateShaderResourceView);

void APIENTRY CreateShaderResourceView1(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D10_1DDIARG_CREATESHADERRESOURCEVIEW *pCreateShaderResourceView,
   D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,
   D3D10DDI_HRTSHADERRESOURCEVIEW hRTShaderResourceView);

SIZE_T APIENTRY CalcPrivateShaderResourceViewSize11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATESHADERRESOURCEVIEW *pCreateShaderResourceView);

void APIENTRY CreateShaderResourceView11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATESHADERRESOURCEVIEW *pCreateShaderResourceView,
   D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,
   D3D10DDI_HRTSHADERRESOURCEVIEW hRTShaderResourceView);

void APIENTRY DestroyShaderResourceView(D3D10DDI_HDEVICE hDevice,
                               D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView);

SIZE_T APIENTRY CalcPrivateUnorderedAccessViewSize(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEUNORDEREDACCESSVIEW *pCreateUAView);

void APIENTRY CreateUnorderedAccessView(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEUNORDEREDACCESSVIEW *pCreateUAView,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView,
   D3D11DDI_HRTUNORDEREDACCESSVIEW hRTUnorderedAccessView);

void APIENTRY DestroyUnorderedAccessView(
   D3D10DDI_HDEVICE hDevice,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView);

void APIENTRY ClearUnorderedAccessViewUint(
   D3D10DDI_HDEVICE hDevice,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView,
   const UINT Values[4]);

void APIENTRY ClearUnorderedAccessViewFloat(
   D3D10DDI_HDEVICE hDevice,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView,
   const FLOAT Values[4]);

void APIENTRY CsSetUnorderedAccessViews(
   D3D10DDI_HDEVICE hDevice,
   UINT StartSlot,
   UINT NumViews,
   __in_ecount (NumViews) const D3D11DDI_HUNORDEREDACCESSVIEW *phUnorderedAccessView,
   __in_ecount (NumViews) const UINT *pUAVInitialCounts);

void APIENTRY GenMips(D3D10DDI_HDEVICE hDevice,
             D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView);

#endif   /* SHADER_H */
