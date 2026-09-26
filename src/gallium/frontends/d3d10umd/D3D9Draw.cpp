/*
 * SPDX-License-Identifier: MIT
 *
 * Copyright 2026 Ake Rehnman
 */
#include "D3D9Private.h"
#include "D3D9NineShader.h"
#include "D3D9ShaderTranslate.h"

#include "pipe/p_state.h"
#include "tgsi/tgsi_ureg.h"
#include "util/u_draw.h"
#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_gen_mipmap.h"
#include "util/u_sampler.h"
#include "util/u_upload_mgr.h"

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

#include <float.h>
#include <stdlib.h>

static float D3D9RenderStateFloat(UINT value);
static float D3D9PointSize(const D3D9Device *device);

static constexpr UINT D3D9_DEFAULT_VERTEX_INPUT_BUFFER = 16;
static constexpr UINT D3D9_SAMPLER_FETCH4 =
   (UINT)D3D9FourCC('G', 'E', 'T', '4');

static bool
D3D9OrderedContextWorkerEnabled()
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_WORKER", true) ? 1 : 0;
   }
   return enabled != 0;
}

static bool
D3D9ConsumeBackendDrawFailure(D3D9Device *device)
{
   if (!device ||
       !yttrium_gdi_take_draw_failure(device->pipe))
      return false;

   D3D9Warnf("draw failed owner=d3d9umd "
             "reason=backend_native_draw_failed action=return_E_FAIL\n");
   return true;
}

struct D3D9AutoUploadResources
{
   struct pipe_context *pipe = NULL;
   struct pipe_resource *resources[PIPE_MAX_ATTRIBS] = {};

   ~D3D9AutoUploadResources()
   {
      if (!pipe)
         return;

      for (UINT i = 0; i < ARRAYSIZE(resources); ++i)
         pipe_resource_release(pipe, resources[i]);
   }
};

static bool
D3D9ProjectionUsesZFog(const D3D9Device *device)
{
   const D3DMATRIX &projection = device->transforms[D3DTS_PROJECTION];

   return projection._34 == 0.0f && projection._44 == 1.0f;
}

static UINT
D3D9TranslatedPsFogKey(bool fog_enable, UINT fog_mode, bool zfog)
{
   return (fog_enable ? 1u : 0u) |
          ((fog_enable ? fog_mode : D3DFOG_NONE) << 1) |
          ((fog_enable && zfog ? 1u : 0u) << 3);
}

static UINT
D3D9TranslatedPsShadeKey(const D3D9Device *device)
{
   return device->render_states[D3DRS_SHADEMODE] == D3DSHADE_FLAT ?
      1u << 12 : 0u;
}

static bool
D3D9PixelShaderUsesPs1xSamplerTypes(const D3D9Object *shader)
{
   if (!shader || shader->kind != D3D9_OBJECT_SHADER ||
       shader->size < sizeof(UINT))
      return false;

   const UINT version = ((const UINT *)shader->data)[0];
   return (version >> 16) == 0xffff && ((version >> 8) & 0xff) < 2;
}

static UINT
D3D9Ps1xSamplerType(const D3D9Device *device, UINT stage)
{
   D3D9Resource *texture = D3D9CastResource(device->textures[stage]);
   if (!texture)
      return 0;

   struct pipe_resource *resource = texture->managed_source_pipe_resource ?
      texture->managed_source_pipe_resource :
      (texture->managed_default_pipe_resource ?
       texture->managed_default_pipe_resource : texture->pipe_resource);
   if (!resource)
      return 0;

   switch (resource->target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      return 1;
   case PIPE_TEXTURE_3D:
      return 3;
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return 2;
   default:
      return 0;
   }
}

static uint32_t
D3D9TranslatedPs1xSamplerTypes(const D3D9Device *device)
{
   uint32_t types = 0;
   const UINT count = MIN2((UINT)ARRAYSIZE(device->textures),
                           (UINT)NINE_MAX_SAMPLERS);

   for (UINT stage = 0; stage < count; ++stage)
      types |= D3D9Ps1xSamplerType(device, stage) << (stage * 2);

   return types;
}

static uint64_t
D3D9SmSamplerTypeFromTexture(const D3D9Resource *texture)
{
   if (!texture)
      return 0;

   struct pipe_resource *resource = texture->managed_source_pipe_resource ?
      texture->managed_source_pipe_resource :
      (texture->managed_default_pipe_resource ?
       texture->managed_default_pipe_resource : texture->pipe_resource);
   if (!resource)
      return 0;

   switch (resource->target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      return NINED3DSTT_1D;
   case PIPE_TEXTURE_3D:
      return NINED3DSTT_VOLUME;
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return NINED3DSTT_CUBE;
   default:
      return NINED3DSTT_2D;
   }
}

static void
D3D9TranslatedPsSamplerTypeOverrides(const D3D9Device *device,
                                     uint16_t sampler_mask,
                                     uint64_t *types,
                                     uint16_t *mask)
{
   *types = 0;
   *mask = 0;

   const UINT count = MIN2((UINT)ARRAYSIZE(device->textures),
                           (UINT)NINE_MAX_SAMPLERS_PS);
   for (UINT stage = 0; stage < count; ++stage) {
      if (!(sampler_mask & (1u << stage)))
         continue;

      D3D9Resource *texture = D3D9CastResource(device->textures[stage]);
      if (!texture)
         continue;

      *types |= D3D9SmSamplerTypeFromTexture(texture) << (stage * 3);
      *mask |= 1u << stage;
   }
}

static void
D3D9TranslatedPsFetch4(const D3D9Device *device, uint16_t sampler_mask,
                       uint16_t *fetch4, uint16_t *fetch4_ati1,
                       uint16_t *fetch4_projected_fallback)
{
   *fetch4 = 0;
   *fetch4_ati1 = 0;
   *fetch4_projected_fallback = 0;
   const UINT count = MIN2((UINT)ARRAYSIZE(device->texture_stage_states),
                           (UINT)NINE_MAX_SAMPLERS_PS);

   for (UINT stage = 0; stage < count; ++stage) {
      if ((sampler_mask & (1u << stage)) &&
          device->texture_stage_states[stage][D3DDDITSS_MIPMAPLODBIAS] ==
          D3D9_SAMPLER_FETCH4) {
         *fetch4 |= 1u << stage;

         D3D9Resource *texture = D3D9CastResource(device->textures[stage]);
         if (!texture)
            continue;

         if (texture->format == D3D9_FMT_ATI1)
            *fetch4_ati1 |= 1u << stage;
         else if (texture->format == D3DDDIFMT_A8R8G8B8)
            *fetch4_projected_fallback |= 1u << stage;
      }
   }
}

static UINT
D3D9DeclTypeComponentCount(BYTE type)
{
   switch (type) {
   case D3DDECLTYPE_FLOAT1:
      return 1;
   case D3DDECLTYPE_FLOAT2:
   case D3DDECLTYPE_SHORT2:
   case D3DDECLTYPE_SHORT2N:
   case D3DDECLTYPE_USHORT2N:
   case D3DDECLTYPE_FLOAT16_2:
      return 2;
   case D3DDECLTYPE_FLOAT3:
   case D3DDECLTYPE_DEC3N:
      return 3;
   case D3DDECLTYPE_FLOAT4:
   case D3DDECLTYPE_D3DCOLOR:
   case D3DDECLTYPE_UBYTE4:
   case D3DDECLTYPE_SHORT4:
   case D3DDECLTYPE_UBYTE4N:
   case D3DDECLTYPE_SHORT4N:
   case D3DDECLTYPE_USHORT4N:
   case D3DDECLTYPE_UDEC3:
   case D3DDECLTYPE_FLOAT16_4:
      return 4;
   default:
      return 0;
   }
}

static UINT
D3D9TexcoordDeclComponentCount(const D3D9Device *device, UINT index)
{
   const D3D9Object *decl = device ?
      (const D3D9Object *)device->vertex_shader_decl : NULL;
   if (!decl || decl->kind != D3D9_OBJECT_VERTEX_DECL)
      return 0;

   const D3DDDIVERTEXELEMENT *elements =
      (const D3DDDIVERTEXELEMENT *)decl->data;
   const UINT element_count = (UINT)(decl->size / sizeof(*elements));

   for (UINT i = 0; i < element_count; ++i) {
      const D3DDDIVERTEXELEMENT *element = &elements[i];
      if (element->Stream == 0xff || element->Type == D3DDECLTYPE_UNUSED)
         break;
      if (element->Usage == D3DDECLUSAGE_TEXCOORD &&
          element->UsageIndex == index)
         return D3D9DeclTypeComponentCount(element->Type);
   }

   return 0;
}

static uint8_t
D3D9TranslatedPsProjected(const D3D9Device *device)
{
   uint8_t projected = 0;
   const bool app_vs = device->vertex_shader_func != NULL;

   for (UINT stage = 0; stage < 4; ++stage) {
      const UINT flags =
         device->texture_stage_states[stage][D3DDDITSS_TEXTURETRANSFORMFLAGS];
      if (!(flags & D3DTTFF_PROJECTED))
         continue;

      /*
       * With an app vertex shader, D3D9 projects ps_1.x texture fetches from
       * the shader output coordinate.  The fixed-function COUNT bits only
       * describe fixed-function texture transform output size.
       */
      UINT count = app_vs ? D3DTTFF_COUNT4 : (flags & 0xff);
      if (count < D3DTTFF_COUNT1 || count > D3DTTFF_COUNT4)
         count = D3D9TexcoordDeclComponentCount(device, stage);
      if (count < D3DTTFF_COUNT1 || count > D3DTTFF_COUNT4)
         count = D3DTTFF_COUNT4;

      projected |= (uint8_t)((count - 1) << (2 * stage));
   }

   return projected;
}
static void
D3D9StoreFloat(uint32_t *dst, float value)
{
   memcpy(dst, &value, sizeof(value));
}

static void
D3D9OverlayNineLocalFloatConstants(uint32_t (*constants)[4],
                                   unsigned max_consts,
                                   const struct nine_range *ranges,
                                   const float *data)
{
   unsigned src = 0;

   if (!constants || !data)
      return;

   for (const struct nine_range *range = ranges; range; range = range->next) {
      for (int slot = range->bgn; slot < range->end; ++slot, ++src) {
         if (slot >= 0 && (unsigned)slot < max_consts)
            memcpy(constants[slot], &data[src * 4], sizeof(constants[slot]));
      }
   }
}

static void
D3D9StoreNineIntConstant(uint32_t dst[4], const INT src[4],
                         bool native_integers)
{
   if (native_integers) {
      memcpy(dst, src, sizeof(uint32_t[4]));
      return;
   }

   for (unsigned c = 0; c < 4; ++c) {
      const float value = (float)src[c];
      memcpy(&dst[c], &value, sizeof(value));
   }
}

static void
D3D9StoreNineBoolConstant(uint32_t dst[4], const BOOL *src,
                          bool native_integers)
{
   for (unsigned c = 0; c < 4; ++c) {
      if (native_integers) {
         dst[c] = src[c] ? 0xffffffffu : 0u;
      } else {
         const float value = src[c] ? 1.0f : 0.0f;
         memcpy(&dst[c], &value, sizeof(value));
      }
   }
}

static void
D3D9StoreNineVsConstantSlot(D3D9Device *device, uint32_t dst[4],
                            unsigned slot, bool native_integers)
{
   if (slot < NINE_MAX_CONST_F) {
      memcpy(dst, device->vs_float_constants[slot], sizeof(uint32_t[4]));
   } else if (slot < NINE_MAX_CONST_VS_SPE_OFFSET) {
      const unsigned rel = slot - NINE_MAX_CONST_F;

      if (rel < NINE_MAX_CONST_I) {
         D3D9StoreNineIntConstant(dst, device->vs_int_constants[rel],
                                  native_integers);
      } else {
         const unsigned bool_reg = rel - NINE_MAX_CONST_I;

         if (bool_reg < NINE_MAX_CONST_B / 4)
            D3D9StoreNineBoolConstant(dst,
                                      &device->vs_bool_constants[bool_reg * 4],
                                      native_integers);
      }
   } else {
      const unsigned special = slot - NINE_MAX_CONST_VS_SPE_OFFSET;

      if (special < ARRAYSIZE(device->clip_planes)) {
         D3D9StoreFloat(&dst[0], device->clip_planes[special][0]);
         D3D9StoreFloat(&dst[1], device->clip_planes[special][1]);
         D3D9StoreFloat(&dst[2], device->clip_planes[special][2]);
         D3D9StoreFloat(&dst[3], device->clip_planes[special][3]);
      } else if (special == 8) {
         const float point_size = D3D9PointSize(device);
         memcpy(&dst[0], &point_size, sizeof(point_size));
      } else if (special == 9) {
         const float width = device->viewport.Width ?
            (float)device->viewport.Width : 1.0f;
         const float height = device->viewport.Height ?
            (float)device->viewport.Height : 1.0f;
         const float min_z = device->zrange.MinZ;
         const float max_z = device->zrange.MaxZ ? device->zrange.MaxZ : 1.0f;
         const float scale_x = width / 2.0f;
         const float scale_y = -height / 2.0f;
         const float scale_z = max_z - min_z;
         const float inv_scale_x = scale_x != 0.0f ? 1.0f / scale_x : 1.0f;
         const float inv_scale_y = scale_y != 0.0f ? 1.0f / scale_y : 1.0f;
         const float inv_scale_z = scale_z != 0.0f ? 1.0f / scale_z : 1.0f;

         D3D9StoreFloat(&dst[0], inv_scale_x);
         D3D9StoreFloat(&dst[1], inv_scale_y);
         D3D9StoreFloat(&dst[2], inv_scale_z);
         D3D9StoreFloat(&dst[3], 1.0f);
      } else if (special == 10) {
         const float width = device->viewport.Width ?
            (float)device->viewport.Width : 1.0f;
         const float height = device->viewport.Height ?
            (float)device->viewport.Height : 1.0f;
         const float min_z = device->zrange.MinZ;
         const float half_width = width / 2.0f;
         const float half_height = height / 2.0f;
         const float translate_x = half_width + (float)device->viewport.X;
         const float translate_y = half_height + (float)device->viewport.Y;

         D3D9StoreFloat(&dst[0], translate_x);
         D3D9StoreFloat(&dst[1], translate_y);
         D3D9StoreFloat(&dst[2], min_z);
         D3D9StoreFloat(&dst[3], 0.0f);
      }
   }
}

static void
D3D9StoreNinePsConstantSlot(D3D9Device *device, uint32_t dst[4],
                            unsigned slot, bool native_integers)
{
   if (slot < NINE_MAX_CONST_F_PS3) {
      memcpy(dst, device->ps_float_constants[slot], sizeof(uint32_t[4]));
   } else if (slot < NINE_MAX_CONST_PS_SPE_OFFSET) {
      const unsigned rel = slot - NINE_MAX_CONST_F_PS3;

      if (rel < NINE_MAX_CONST_I) {
         D3D9StoreNineIntConstant(dst, device->ps_int_constants[rel],
                                  native_integers);
      } else {
         const unsigned bool_reg = rel - NINE_MAX_CONST_I;

         if (bool_reg < NINE_MAX_CONST_B / 4)
            D3D9StoreNineBoolConstant(dst,
                                      &device->ps_bool_constants[bool_reg * 4],
                                      native_integers);
      }
   } else {
      const unsigned special = slot - NINE_MAX_CONST_PS_SPE_OFFSET;

      if (special < 8) {
         dst[0] = device->texture_stage_states[special][D3DTSS_BUMPENVMAT00];
         dst[1] = device->texture_stage_states[special][D3DTSS_BUMPENVMAT01];
         dst[2] = device->texture_stage_states[special][D3DTSS_BUMPENVMAT10];
         dst[3] = device->texture_stage_states[special][D3DTSS_BUMPENVMAT11];
      } else if (special < 12) {
         const unsigned stage = (special - 8) * 2;

         dst[0] = device->texture_stage_states[stage][D3DTSS_BUMPENVLSCALE];
         dst[1] = device->texture_stage_states[stage][D3DTSS_BUMPENVLOFFSET];
         dst[2] =
            device->texture_stage_states[stage + 1][D3DTSS_BUMPENVLSCALE];
         dst[3] =
            device->texture_stage_states[stage + 1][D3DTSS_BUMPENVLOFFSET];
      } else if (special == 12) {
         const UINT color = device->render_states[D3DRS_FOGCOLOR];
         const float r = (float)((color >> 16) & 0xff) / 255.0f;
         const float g = (float)((color >> 8) & 0xff) / 255.0f;
         const float b = (float)(color & 0xff) / 255.0f;
         const float a = (float)((color >> 24) & 0xff) / 255.0f;

         D3D9StoreFloat(&dst[0], r);
         D3D9StoreFloat(&dst[1], g);
         D3D9StoreFloat(&dst[2], b);
         D3D9StoreFloat(&dst[3], a);
      } else if (special == 13) {
         const float fog_start =
            D3D9RenderStateFloat(device->render_states[D3DRS_FOGSTART]);
         const float fog_end =
            D3D9RenderStateFloat(device->render_states[D3DRS_FOGEND]);
         const float fog_density =
            D3D9RenderStateFloat(device->render_states[D3DRS_FOGDENSITY]);
         const UINT fog_mode =
            device->render_states[D3DRS_FOGTABLEMODE] & 0x3;
         const float denom = fog_end - fog_start;
         const float fog_coeff = denom != 0.0f ? 1.0f / denom : FLT_MAX;

         if (fog_mode == D3DFOG_LINEAR) {
            D3D9StoreFloat(&dst[0], fog_end);
            D3D9StoreFloat(&dst[1], fog_coeff);
         } else {
            D3D9StoreFloat(&dst[0], fog_density);
         }
         D3D9StoreFloat(&dst[2], fog_density);
      } else if (special == 14) {
         const float alpha_ref =
            (float)(device->render_states[D3DRS_ALPHAREF] & 0xff) / 255.0f;

         D3D9StoreFloat(&dst[0], alpha_ref);
      }
   }
}

static void *
D3D9TranslatedVsState(const D3D9Device *device)
{
   const D3D9Object *shader =
      (const D3D9Object *)device->vertex_shader_func;

   if (!shader || shader->kind != D3D9_OBJECT_SHADER)
      return NULL;

   return shader->translated_vs_cso;
}

static void *
D3D9TranslatedPsState(const D3D9Device *device)
{
   const D3D9Object *shader =
      (const D3D9Object *)device->pixel_shader;

   if (!shader || shader->kind != D3D9_OBJECT_SHADER)
      return NULL;

   return shader->translated_ps_cso;
}

static bool
D3D9ActiveVertexDeclHasPositionT(const D3D9Device *device);

static bool
D3D9CanUsePositionTDiffuseFixedFunction(const D3D9Device *device);

static bool
D3D9InitPositionTTexture0RuntimePsFixedFunctionVsInfo(
   const D3D9Device *device, const D3D9Object *ps, D3D9Object *vs_shader);

static bool
D3D9CanUsePositionTPsInputFixedFunction(const D3D9Device *device);

static bool
D3D9NoBoundTextures(const D3D9Device *device)
{
   if (!device)
      return false;

   for (UINT i = 0; i < ARRAYSIZE(device->textures); ++i) {
      if (device->textures[i])
         return false;
   }

   return true;
}

static bool
D3D9DepthTestActive(const D3D9Device *device)
{
   return device->depth_stencil &&
          device->render_states[D3DDDIRS_ZENABLE] != D3DZB_FALSE;
}

static HRESULT
D3D9UpdateTranslatedShaderState(D3D9Device *device)
{
   const bool fog_enable = device->render_states[D3DRS_FOGENABLE] != 0;
   const bool position_t = D3D9ActiveVertexDeclHasPositionT(device);
   const bool zfog = D3D9ProjectionUsesZFog(device);
   const UINT fog_mode = device->render_states[D3DRS_FOGTABLEMODE] & 0x3;
   const uint8_t projected = D3D9TranslatedPsProjected(device);
   const uint8_t clip_plane_mask =
      (uint8_t)(device->render_states[D3DRS_CLIPPLANEENABLE] & 0xff);

   D3D9Object *vs = (D3D9Object *)device->vertex_shader_func;
   if (vs && vs->kind == D3D9_OBJECT_SHADER) {
      const UINT key = (fog_enable ? 1u : 0u) |
                       (position_t ? 1u << 1 : 0u) |
                       ((UINT)clip_plane_mask << 2);
      if (vs->translated_vs_key != key) {
         HRESULT hr =
            D3D9TranslateVertexShaderForState(device, vs, fog_enable,
                                              position_t, clip_plane_mask);
         if (FAILED(hr))
            return hr;
      }
   }

   D3D9Object *ps = (D3D9Object *)device->pixel_shader;
   if (ps && ps->kind == D3D9_OBJECT_SHADER) {
      const uint32_t sampler_ps1xtypes =
         D3D9PixelShaderUsesPs1xSamplerTypes(ps) ?
         D3D9TranslatedPs1xSamplerTypes(device) : 0;
      uint64_t sampler_type_overrides;
      uint16_t sampler_type_override_mask;
      D3D9TranslatedPsSamplerTypeOverrides(device,
                                           ps->translated_ps_sampler_mask,
                                           &sampler_type_overrides,
                                           &sampler_type_override_mask);
      uint16_t fetch4;
      uint16_t fetch4_ati1;
      uint16_t fetch4_projected_fallback;
      D3D9TranslatedPsFetch4(device, ps->translated_ps_sampler_mask, &fetch4,
                             &fetch4_ati1, &fetch4_projected_fallback);
      const UINT key = D3D9TranslatedPsFogKey(fog_enable, fog_mode, zfog) |
         ((UINT)projected << 4) | D3D9TranslatedPsShadeKey(device) |
         (sampler_ps1xtypes << 13);
      if (ps->translated_ps_key != key ||
          ps->translated_ps_sampler_type_overrides !=
          sampler_type_overrides ||
          ps->translated_ps_sampler_type_override_mask !=
          sampler_type_override_mask ||
          ps->translated_ps_fetch4 != fetch4 ||
          ps->translated_ps_fetch4_ati1 != fetch4_ati1 ||
          ps->translated_ps_fetch4_projected_fallback !=
          fetch4_projected_fallback) {
         HRESULT hr = D3D9TranslatePixelShaderForState(device, ps, fog_enable,
                                                       fog_mode, zfog,
                                                       projected,
                                                       sampler_ps1xtypes,
                                                       sampler_type_overrides,
                                                       sampler_type_override_mask,
                                                       fetch4, fetch4_ati1,
                                                       fetch4_projected_fallback);
         if (FAILED(hr))
            return hr;
      }
   }

   return S_OK;
}

static bool
D3D9HasTranslatedShaderPair(const D3D9Device *device)
{
   return D3D9TranslatedVsState(device) && D3D9TranslatedPsState(device);
}

static bool
D3D9TranslatedShaderPairIsRuntimeFixedFunction(const D3D9Device *device)
{
   const D3D9Object *vs =
      (const D3D9Object *)device->vertex_shader_func;
   const D3D9Object *ps =
      (const D3D9Object *)device->pixel_shader;

   return D3D9HasTranslatedShaderPair(device) &&
          D3D9ShaderHasRuntimeFixedFunctionComment(vs) &&
          D3D9ShaderHasRuntimeFixedFunctionComment(ps);
}

static HRESULT
D3D9FailNoTranslatedShaderPair()
{
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9 draw without a translated "
                 "VS/PS pair; fixed-function and app shaders must use "
                 "Nine shader translation\n");
   return E_NOTIMPL;
}

static bool
D3D9ActiveVertexDeclHasPositionT(const D3D9Device *device)
{
   const D3D9Object *decl = device ?
      (const D3D9Object *)device->vertex_shader_decl : NULL;
   if (!decl || decl->kind != D3D9_OBJECT_VERTEX_DECL)
      return false;

   const D3DDDIVERTEXELEMENT *elements =
      (const D3DDDIVERTEXELEMENT *)decl->data;
   const UINT element_count = (UINT)(decl->size / sizeof(*elements));

   for (UINT i = 0; i < element_count; ++i) {
      const D3DDDIVERTEXELEMENT *element = &elements[i];
      if (element->Stream == 0xff || element->Type == D3DDECLTYPE_UNUSED)
         break;
      if (element->Usage == D3DDECLUSAGE_POSITIONT &&
          element->UsageIndex == 0)
         return true;
   }

   return false;
}

static bool
D3D9TranslatedVsOutputsPointSize(const D3D9Device *device)
{
   const D3D9Object *shader =
      (const D3D9Object *)device->vertex_shader_func;

   return shader && shader->kind == D3D9_OBJECT_SHADER &&
          shader->translated_vs_cso &&
          shader->translated_vs_outputs_point_size;
}

static enum mesa_prim
D3D9PrimitiveToPipe(D3DPRIMITIVETYPE primitive)
{
   switch (primitive) {
   case D3DPT_POINTLIST:
      return MESA_PRIM_POINTS;
   case D3DPT_LINELIST:
      return MESA_PRIM_LINES;
   case D3DPT_LINESTRIP:
      return MESA_PRIM_LINE_STRIP;
   case D3DPT_TRIANGLELIST:
      return MESA_PRIM_TRIANGLES;
   case D3DPT_TRIANGLESTRIP:
      return MESA_PRIM_TRIANGLE_STRIP;
   case D3DPT_TRIANGLEFAN:
      return MESA_PRIM_TRIANGLE_FAN;
   default:
      return MESA_PRIM_COUNT;
   }
}

static UINT
D3D9PrimitiveVertexCount(D3DPRIMITIVETYPE primitive, UINT primitive_count)
{
   switch (primitive) {
   case D3DPT_POINTLIST:
      return primitive_count;
   case D3DPT_LINELIST:
      return primitive_count * 2;
   case D3DPT_LINESTRIP:
      return primitive_count + 1;
   case D3DPT_TRIANGLELIST:
      return primitive_count * 3;
   case D3DPT_TRIANGLESTRIP:
   case D3DPT_TRIANGLEFAN:
      return primitive_count + 2;
   default:
      return 0;
   }
}

static uint
D3D9CompareToPipe(UINT func)
{
   switch (func) {
   case D3DCMP_NEVER:
      return PIPE_FUNC_NEVER;
   case D3DCMP_LESS:
      return PIPE_FUNC_LESS;
   case D3DCMP_EQUAL:
      return PIPE_FUNC_EQUAL;
   case D3DCMP_LESSEQUAL:
      return PIPE_FUNC_LEQUAL;
   case D3DCMP_GREATER:
      return PIPE_FUNC_GREATER;
   case D3DCMP_NOTEQUAL:
      return PIPE_FUNC_NOTEQUAL;
   case D3DCMP_GREATEREQUAL:
      return PIPE_FUNC_GEQUAL;
   case D3DCMP_ALWAYS:
   default:
      return PIPE_FUNC_ALWAYS;
   }
}

static uint
D3D9StencilOpToPipe(UINT op)
{
   switch (op) {
   case D3DSTENCILOP_KEEP:
      return PIPE_STENCIL_OP_KEEP;
   case D3DSTENCILOP_ZERO:
      return PIPE_STENCIL_OP_ZERO;
   case D3DSTENCILOP_REPLACE:
      return PIPE_STENCIL_OP_REPLACE;
   case D3DSTENCILOP_INCRSAT:
      return PIPE_STENCIL_OP_INCR;
   case D3DSTENCILOP_DECRSAT:
      return PIPE_STENCIL_OP_DECR;
   case D3DSTENCILOP_INVERT:
      return PIPE_STENCIL_OP_INVERT;
   case D3DSTENCILOP_INCR:
      return PIPE_STENCIL_OP_INCR_WRAP;
   case D3DSTENCILOP_DECR:
      return PIPE_STENCIL_OP_DECR_WRAP;
   default:
      return PIPE_STENCIL_OP_KEEP;
   }
}

static enum pipe_format
D3D9DeclTypeToPipeFormat(UINT type)
{
   switch (type) {
   case D3DDECLTYPE_FLOAT1:
      return PIPE_FORMAT_R32_FLOAT;
   case D3DDECLTYPE_FLOAT2:
      return PIPE_FORMAT_R32G32_FLOAT;
   case D3DDECLTYPE_FLOAT3:
      return PIPE_FORMAT_R32G32B32_FLOAT;
   case D3DDECLTYPE_FLOAT4:
      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   case D3DDECLTYPE_D3DCOLOR:
      return PIPE_FORMAT_B8G8R8A8_UNORM;
   case D3DDECLTYPE_UBYTE4:
      return PIPE_FORMAT_R8G8B8A8_USCALED;
   case D3DDECLTYPE_SHORT2:
      return PIPE_FORMAT_R16G16_SSCALED;
   case D3DDECLTYPE_SHORT4:
      return PIPE_FORMAT_R16G16B16A16_SSCALED;
   case D3DDECLTYPE_UBYTE4N:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case D3DDECLTYPE_SHORT2N:
      return PIPE_FORMAT_R16G16_SNORM;
   case D3DDECLTYPE_SHORT4N:
      return PIPE_FORMAT_R16G16B16A16_SNORM;
   case D3DDECLTYPE_USHORT2N:
      return PIPE_FORMAT_R16G16_UNORM;
   case D3DDECLTYPE_USHORT4N:
      return PIPE_FORMAT_R16G16B16A16_UNORM;
   case D3DDECLTYPE_FLOAT16_2:
      return PIPE_FORMAT_R16G16_FLOAT;
   case D3DDECLTYPE_FLOAT16_4:
      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   default:
      return PIPE_FORMAT_NONE;
   }
}

static uint16_t
D3D9DeclElementUsage(const D3DDDIVERTEXELEMENT *element)
{
   return nine_d3d9_to_nine_declusage(element->Usage, element->UsageIndex);
}

static bool
D3D9PositionTDiffuseDeclOnly(const D3D9Device *device)
{
   if (!device)
      return false;

   const D3D9Object *decl = (const D3D9Object *)device->vertex_shader_decl;
   if (!decl || decl->kind != D3D9_OBJECT_VERTEX_DECL)
      return false;

   const D3DDDIVERTEXELEMENT *elements =
      (const D3DDDIVERTEXELEMENT *)decl->data;
   const UINT element_count = (UINT)(decl->size / sizeof(*elements));
   bool position_t = false;
   bool color0 = false;

   for (UINT i = 0; i < element_count; ++i) {
      const D3DDDIVERTEXELEMENT *element = &elements[i];
      if (element->Stream == 0xff || element->Type == D3DDECLTYPE_UNUSED)
         break;

      const uint16_t usage = D3D9DeclElementUsage(element);
      if (usage == NINE_DECLUSAGE_i(POSITIONT, 0) &&
          element->Type == D3DDECLTYPE_FLOAT4) {
         position_t = true;
      } else if (usage == NINE_DECLUSAGE_i(COLOR, 0) &&
                 element->Type == D3DDECLTYPE_D3DCOLOR) {
         color0 = true;
      } else {
         return false;
      }
   }

   return position_t && color0;
}

static bool
D3D9CanUsePositionTDiffuseFixedFunction(const D3D9Device *device)
{
   return device && !device->vertex_shader_func && D3D9NoBoundTextures(device) &&
          D3D9PositionTDiffuseDeclOnly(device);
}

static bool
D3D9InitPositionTTexture0RuntimePsFixedFunctionVsInfo(
   const D3D9Device *device, const D3D9Object *ps, D3D9Object *vs_shader)
{
   if (!device || !ps || !vs_shader || !device->pixel_shader ||
       !device->textures[0])
      return false;
   if (ps->kind != D3D9_OBJECT_SHADER || !ps->translated_ps_cso ||
       !D3D9ShaderHasRuntimeFixedFunctionComment(ps))
      return false;

   for (UINT i = 1; i < ARRAYSIZE(device->textures); ++i) {
      if (device->textures[i])
         return false;
   }

   if ((ps->translated_ps_sampler_mask & 1) == 0 ||
       ps->translated_ps_sampler_targets[0] != TGSI_TEXTURE_2D)
      return false;

   const D3D9Object *decl = (const D3D9Object *)device->vertex_shader_decl;
   if (!decl || decl->kind != D3D9_OBJECT_VERTEX_DECL)
      return false;

   const D3DDDIVERTEXELEMENT *elements =
      (const D3DDDIVERTEXELEMENT *)decl->data;
   const UINT element_count = (UINT)(decl->size / sizeof(*elements));
   uint16_t position_usage = NINE_DECLUSAGE_NONE;
   bool texcoord0 = false;

   for (UINT i = 0; i < element_count; ++i) {
      const D3DDDIVERTEXELEMENT *element = &elements[i];
      if (element->Stream == 0xff || element->Type == D3DDECLTYPE_UNUSED)
         break;

      const uint16_t usage = D3D9DeclElementUsage(element);
      if (usage == NINE_DECLUSAGE_i(POSITIONT, 0) &&
          element->Type == D3DDECLTYPE_FLOAT4) {
         position_usage = usage;
      } else if (usage == NINE_DECLUSAGE_i(TEXCOORD, 0) &&
                 element->Type == D3DDECLTYPE_FLOAT2) {
         texcoord0 = true;
      } else {
         return false;
      }
   }

   if (position_usage == NINE_DECLUSAGE_NONE || !texcoord0)
      return false;

   memset(vs_shader, 0, sizeof(*vs_shader));
   vs_shader->kind = D3D9_OBJECT_SHADER;
   vs_shader->translated_vs_num_inputs = 2;
   vs_shader->translated_vs_input_map[0] = position_usage;
   vs_shader->translated_vs_input_map[1] = NINE_DECLUSAGE_i(TEXCOORD, 0);
   return true;
}

static bool
D3D9NineDeclUsageToTgsi(const D3D9Device *device, uint16_t usage,
                        unsigned *semantic, unsigned *index)
{
   const unsigned base = usage % NINE_DECLUSAGE_COUNT;
   const unsigned usage_index = usage / NINE_DECLUSAGE_COUNT;
   const bool want_texcoord = device->screen->caps.tgsi_texcoord;

   switch (base) {
   case NINE_DECLUSAGE_POSITION:
   case NINE_DECLUSAGE_POSITIONT:
      if (usage_index == 0) {
         *semantic = TGSI_SEMANTIC_POSITION;
         *index = 0;
      } else {
         *semantic = TGSI_SEMANTIC_GENERIC;
         *index = 10 * usage_index + 7;
      }
      return true;
   case NINE_DECLUSAGE_DEPTH:
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 10 * usage_index + 25;
      return true;
   case NINE_DECLUSAGE_COLOR:
      if (usage_index == 0) {
         *semantic = TGSI_SEMANTIC_COLOR;
         *index = 0;
      } else {
         *semantic = TGSI_SEMANTIC_GENERIC;
         *index = 10 * (usage_index - 1) + 8;
      }
      return true;
   case NINE_DECLUSAGE_FOG:
      if (usage_index)
         return false;
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 16;
      return true;
   case NINE_DECLUSAGE_PSIZE:
      if (usage_index)
         return false;
      *semantic = TGSI_SEMANTIC_PSIZE;
      *index = 0;
      return true;
   case NINE_DECLUSAGE_TEXCOORD:
      if (usage_index >= 16)
         return false;
      *semantic = want_texcoord && usage_index < 8 ?
         TGSI_SEMANTIC_TEXCOORD : TGSI_SEMANTIC_GENERIC;
      *index = usage_index;
      return true;
   case NINE_DECLUSAGE_BLENDWEIGHT:
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 10 * usage_index + 19;
      return true;
   case NINE_DECLUSAGE_BLENDINDICES:
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 10 * usage_index + 20;
      return true;
   case NINE_DECLUSAGE_NORMAL:
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 10 * usage_index + 21;
      return true;
   case NINE_DECLUSAGE_TANGENT:
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 10 * usage_index + 22;
      return true;
   case NINE_DECLUSAGE_BINORMAL:
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 10 * usage_index + 23;
      return true;
   case NINE_DECLUSAGE_TESSFACTOR:
      *semantic = TGSI_SEMANTIC_GENERIC;
      *index = 10 * usage_index + 24;
      return true;
   default:
      return false;
   }
}

static bool
D3D9AppendUniqueVsInput(D3D9Object *shader, uint16_t usage)
{
   if (usage == NINE_DECLUSAGE_NONE)
      return true;

   for (UINT i = 0; i < shader->translated_vs_num_inputs; ++i) {
      if (shader->translated_vs_input_map[i] == usage)
         return true;
   }

   if (shader->translated_vs_num_inputs >=
       ARRAYSIZE(shader->translated_vs_input_map))
      return false;

   shader->translated_vs_input_map[shader->translated_vs_num_inputs++] = usage;
   return true;
}

static bool
D3D9CanUsePositionTPsInputFixedFunction(const D3D9Device *device)
{
   const D3D9Object *ps = device ?
      (const D3D9Object *)device->pixel_shader : NULL;

   return device && !device->vertex_shader_func && ps &&
          ps->kind == D3D9_OBJECT_SHADER && ps->translated_ps_cso &&
          ps->translated_ps_num_inputs &&
          !D3D9ShaderHasRuntimeFixedFunctionComment(ps) &&
          D3D9ActiveVertexDeclHasPositionT(device);
}

static bool
D3D9InitPositionTPsInputFixedFunctionVsInfo(D3D9Object *shader,
                                            const D3D9Object *ps)
{
   memset(shader, 0, sizeof(*shader));
   shader->kind = D3D9_OBJECT_SHADER;

   if (!D3D9AppendUniqueVsInput(shader, NINE_DECLUSAGE_i(POSITIONT, 0)))
      return false;

   for (UINT input = 0; input < ps->translated_ps_num_inputs; ++input) {
      const uint16_t usage = ps->translated_ps_input_map[input];
      const unsigned base = usage % NINE_DECLUSAGE_COUNT;

      if (usage == NINE_DECLUSAGE_NONE)
         continue;
      if (base == NINE_DECLUSAGE_POSITION ||
          base == NINE_DECLUSAGE_POSITIONT)
         continue;
      if (!D3D9AppendUniqueVsInput(shader, usage))
         return false;
   }

   return true;
}

static void
D3D9InitPositionTDiffuseFixedFunctionVsInfo(D3D9Object *shader)
{
   memset(shader, 0, sizeof(*shader));
   shader->kind = D3D9_OBJECT_SHADER;
   shader->translated_vs_num_inputs = 2;
   shader->translated_vs_input_map[0] = NINE_DECLUSAGE_i(POSITIONT, 0);
   shader->translated_vs_input_map[1] = NINE_DECLUSAGE_i(COLOR, 0);
}

static void *
D3D9CreatePositionTPsInputFixedFunctionVs(D3D9Device *device,
                                          const D3D9Object *shader)
{
   struct ureg_program *ureg = ureg_create(MESA_SHADER_VERTEX);
   if (!ureg)
      return NULL;

   const float width = device->viewport.Width ? (float)device->viewport.Width :
      1.0f;
   const float height = device->viewport.Height ?
      (float)device->viewport.Height : 1.0f;
   const float min_z = device->zrange.MinZ;
   const float max_z = device->zrange.MaxZ;
   const float z_scale = max_z != min_z ? 1.0f / (max_z - min_z) : 0.0f;

   struct ureg_src inputs[PIPE_MAX_ATTRIBS];
   memset(inputs, 0, sizeof(inputs));
   for (UINT input = 0; input < shader->translated_vs_num_inputs; ++input)
      inputs[input] = ureg_DECL_vs_input(ureg, input);

   struct ureg_dst out_pos =
      ureg_DECL_output(ureg, TGSI_SEMANTIC_POSITION, 0);
   struct ureg_dst tmp = ureg_DECL_temporary(ureg);

   ureg_MOV(ureg, tmp, inputs[0]);
   ureg_ADD(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XYZ),
            ureg_src(tmp),
            ureg_imm4f(ureg, -(float)device->viewport.X,
                       -(float)device->viewport.Y, -min_z, 0.0f));
   ureg_MUL(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XYZ),
            ureg_src(tmp),
            ureg_imm4f(ureg, 2.0f / width, 2.0f / height, z_scale, 0.0f));
   ureg_ADD(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XY),
            ureg_src(tmp), ureg_imm4f(ureg, -1.0f, -1.0f, 0.0f, 0.0f));
   ureg_MOV(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_Y),
            ureg_negate(ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_Y)));
   ureg_CMP(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_W),
            ureg_negate(ureg_abs(ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W))),
            ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W),
            ureg_imm1f(ureg, 1.0f));
   ureg_RCP(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_W),
            ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W));
   ureg_MUL(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XYZ),
            ureg_src(tmp),
            ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W));
   ureg_MOV(ureg, out_pos, ureg_src(tmp));

   for (UINT input = 1; input < shader->translated_vs_num_inputs; ++input) {
      unsigned semantic, index;

      if (!D3D9NineDeclUsageToTgsi(device, shader->translated_vs_input_map[input],
                                   &semantic, &index)) {
         ureg_destroy(ureg);
         return NULL;
      }

      struct ureg_dst output =
         ureg_DECL_output(ureg, (enum tgsi_semantic)semantic, index);
      ureg_MOV(ureg, output, inputs[input]);
   }

   ureg_release_temporary(ureg, tmp);
   ureg_END(ureg);

   return ureg_create_shader_and_destroy(ureg, device->pipe);
}

static UINT
D3D9StreamSourceFreqValue(UINT divider)
{
   return divider & ~(D3DSTREAMSOURCE_INDEXEDDATA |
                      D3DSTREAMSOURCE_INSTANCEDATA);
}

static UINT
D3D9StreamInstanceDivisor(const D3D9Device *device, UINT stream)
{
   const UINT divider = device->streams[stream].divider;

   if (!(divider & D3DSTREAMSOURCE_INSTANCEDATA))
      return 0;

   const UINT divisor = D3D9StreamSourceFreqValue(divider);
   return divisor ? divisor : 1;
}

static UINT
D3D9DrawInstanceCount(const D3D9Device *device)
{
   const UINT divider = device->streams[0].divider;

   if (!(divider & D3DSTREAMSOURCE_INDEXEDDATA))
      return 1;

   return D3D9StreamSourceFreqValue(divider);
}

static bool
D3D9DeclElementMatchesInputUsage(uint16_t element_usage, uint16_t input_usage,
                                 bool position_t)
{
   if (element_usage == input_usage)
      return true;

   return position_t &&
          element_usage == NINE_DECLUSAGE_i(POSITIONT, 0) &&
          input_usage == NINE_DECLUSAGE_i(POSITION, 0);
}

static UINT
D3D9FillTranslatedVsVertexElements(const D3D9Device *device,
                                   const D3D9Object *shader,
                                   struct pipe_vertex_element *elements,
                                   UINT max_elements,
                                   uint32_t *stream_mask,
                                   bool *uses_default_input)
{
   const D3D9Object *decl = (const D3D9Object *)device->vertex_shader_decl;
   if (!decl || decl->kind != D3D9_OBJECT_VERTEX_DECL)
      return 0;

   const D3DDDIVERTEXELEMENT *decl_elements =
      (const D3DDDIVERTEXELEMENT *)decl->data;
   const UINT decl_element_count = (UINT)(decl->size / sizeof(*decl_elements));
   UINT count = 0;
   const bool position_t = D3D9ActiveVertexDeclHasPositionT(device);
   *stream_mask = 0;
   *uses_default_input = false;

   if (!shader || shader->translated_vs_num_inputs > max_elements ||
       D3D9_DEFAULT_VERTEX_INPUT_BUFFER >= PIPE_MAX_ATTRIBS)
      return 0;

   for (UINT input = 0; input < shader->translated_vs_num_inputs; ++input) {
      const uint16_t input_usage = shader->translated_vs_input_map[input];
      const D3DDDIVERTEXELEMENT *element = NULL;

      if (input_usage == NINE_DECLUSAGE_NONE)
         return 0;

      for (UINT i = 0; i < decl_element_count; ++i) {
         const D3DDDIVERTEXELEMENT *candidate = &decl_elements[i];

         if (candidate->Stream == 0xff ||
             candidate->Type == D3DDECLTYPE_UNUSED)
            break;
         if (D3D9DeclElementMatchesInputUsage(D3D9DeclElementUsage(candidate),
                                              input_usage, position_t)) {
            element = candidate;
            break;
         }
      }

      if (!element) {
         memset(&elements[input], 0, sizeof(elements[input]));
         elements[input].vertex_buffer_index = D3D9_DEFAULT_VERTEX_INPUT_BUFFER;
         elements[input].src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         *uses_default_input = true;
         count = input + 1;
         continue;
      }

      if (element->Stream >= ARRAYSIZE(device->streams) ||
          element->Method != D3DDECLMETHOD_DEFAULT)
         return 0;

      const enum pipe_format format = D3D9DeclTypeToPipeFormat(element->Type);
      if (format == PIPE_FORMAT_NONE)
         return 0;

      memset(&elements[input], 0, sizeof(elements[input]));
      elements[input].src_offset = element->Offset;
      elements[input].vertex_buffer_index = element->Stream;
      elements[input].src_format = format;
      elements[input].src_stride = device->streams[element->Stream].stride;
      elements[input].instance_divisor =
         D3D9StreamInstanceDivisor(device, element->Stream);
      *stream_mask |= 1u << element->Stream;
      count = input + 1;
   }

   return count;
}

static void *
D3D9CreatePositionTDiffuseFixedFunctionVs(D3D9Device *device)
{
   struct ureg_program *ureg = ureg_create(MESA_SHADER_VERTEX);
   if (!ureg)
      return NULL;

   const float width = device->viewport.Width ? (float)device->viewport.Width :
      1.0f;
   const float height = device->viewport.Height ?
      (float)device->viewport.Height : 1.0f;
   const float min_z = device->zrange.MinZ;
   const float max_z = device->zrange.MaxZ;
   const float z_scale = max_z != min_z ? 1.0f / (max_z - min_z) : 0.0f;

   struct ureg_src pos = ureg_DECL_vs_input(ureg, 0);
   struct ureg_src color = ureg_DECL_vs_input(ureg, 1);
   struct ureg_dst out_pos =
      ureg_DECL_output(ureg, TGSI_SEMANTIC_POSITION, 0);
   struct ureg_dst out_color =
      ureg_DECL_output(ureg, TGSI_SEMANTIC_COLOR, 0);
   struct ureg_dst tmp = ureg_DECL_temporary(ureg);

   ureg_MOV(ureg, tmp, pos);
   ureg_ADD(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XYZ),
            ureg_src(tmp),
            ureg_imm4f(ureg, -(float)device->viewport.X,
                       -(float)device->viewport.Y, -min_z, 0.0f));
   ureg_MUL(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XYZ),
            ureg_src(tmp),
            ureg_imm4f(ureg, 2.0f / width, 2.0f / height, z_scale, 0.0f));
   ureg_ADD(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XY),
            ureg_src(tmp), ureg_imm4f(ureg, -1.0f, -1.0f, 0.0f, 0.0f));
   ureg_MOV(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_Y),
            ureg_negate(ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_Y)));
   ureg_CMP(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_W),
            ureg_negate(ureg_abs(ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W))),
            ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W),
            ureg_imm1f(ureg, 1.0f));
   ureg_RCP(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_W),
            ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W));
   ureg_MUL(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_XYZ),
            ureg_src(tmp),
            ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W));
   ureg_MOV(ureg, out_pos, ureg_src(tmp));
   ureg_MOV(ureg, out_color, color);
   ureg_release_temporary(ureg, tmp);
   ureg_END(ureg);

   return ureg_create_shader_and_destroy(ureg, device->pipe);
}

struct D3D9AutoShaderState
{
   struct pipe_context *pipe = NULL;
   void *vs = NULL;

   ~D3D9AutoShaderState()
   {
      if (!pipe)
         return;
      if (vs) {
         pipe->bind_vs_state(pipe, NULL);
         pipe->delete_vs_state(pipe, vs);
      }
   }
};

static void *
D3D9CreatePointPositionGs(struct pipe_context *pipe, float x_scale,
                          float y_scale, bool use_input_point_size,
                          bool use_generic_point_size,
                          float point_size_min, float point_size_max)
{
   struct ureg_program *ureg = ureg_create(MESA_SHADER_GEOMETRY);
   struct ureg_src pos =
      ureg_src_dimension(ureg_DECL_input(ureg, TGSI_SEMANTIC_POSITION, 0,
                                         0, 1), 0);
   struct ureg_src point_size =
      use_input_point_size ?
      ureg_src_dimension(
         ureg_DECL_input(ureg,
                         use_generic_point_size ? TGSI_SEMANTIC_GENERIC :
                         TGSI_SEMANTIC_PSIZE,
                         use_generic_point_size ?
                         D3D9_NINE_POINT_SIZE_GENERIC : 0, 0, 1), 0) :
      pos;
   struct ureg_dst delta = ureg_DECL_temporary(ureg);
   struct ureg_dst clamped_point_size = ureg_dst_undef();
   struct ureg_dst out_pos =
      ureg_DECL_output(ureg, TGSI_SEMANTIC_POSITION, 0);
   struct ureg_dst out_texcoord0 =
      ureg_DECL_output(ureg, TGSI_SEMANTIC_GENERIC, 0);
   struct ureg_dst out_texcoord1 =
      ureg_DECL_output(ureg, TGSI_SEMANTIC_GENERIC, 1);

   ureg_property(ureg, TGSI_PROPERTY_GS_INPUT_PRIM, MESA_PRIM_POINTS);
   ureg_property(ureg, TGSI_PROPERTY_GS_OUTPUT_PRIM,
                 MESA_PRIM_TRIANGLE_STRIP);
   ureg_property(ureg, TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES, 6);
   ureg_property(ureg, TGSI_PROPERTY_GS_INVOCATIONS, 1);

   if (use_input_point_size) {
      clamped_point_size = ureg_DECL_temporary(ureg);
      ureg_MAX(ureg, ureg_writemask(clamped_point_size, TGSI_WRITEMASK_X),
               ureg_scalar(point_size, TGSI_SWIZZLE_X),
               ureg_imm1f(ureg, point_size_min));
      ureg_MIN(ureg, ureg_writemask(clamped_point_size, TGSI_WRITEMASK_X),
               ureg_scalar(ureg_src(clamped_point_size), TGSI_SWIZZLE_X),
               ureg_imm1f(ureg, point_size_max));
      point_size = ureg_src(clamped_point_size);
   }

   const UINT vertex_indices[6] = {0, 1, 2, 2, 1, 3};
   for (UINT vertex = 0; vertex < 6; ++vertex) {
      const UINT i = vertex_indices[vertex];
      const float x_sign = (i & 1) ? 1.0f : -1.0f;
      const float y_sign = (i & 2) ? -1.0f : 1.0f;
      const float s = (i & 1) ? 1.0f : 0.0f;
      const float t = (i & 2) ? 1.0f : 0.0f;

      ureg_MOV(ureg, out_pos, pos);
      ureg_MUL(ureg, ureg_writemask(delta, TGSI_WRITEMASK_X),
               ureg_scalar(point_size, use_input_point_size ?
                            TGSI_SWIZZLE_X : TGSI_SWIZZLE_W),
               ureg_imm1f(ureg, x_scale));
      ureg_MAD(ureg, ureg_writemask(out_pos, TGSI_WRITEMASK_X),
               ureg_scalar(ureg_src(delta), TGSI_SWIZZLE_X),
               ureg_imm1f(ureg, x_sign),
               ureg_scalar(pos, TGSI_SWIZZLE_X));
      ureg_MUL(ureg, ureg_writemask(delta, TGSI_WRITEMASK_Y),
               ureg_scalar(point_size, use_input_point_size ?
                            TGSI_SWIZZLE_X : TGSI_SWIZZLE_W),
               ureg_imm1f(ureg, y_scale));
      ureg_MAD(ureg, ureg_writemask(out_pos, TGSI_WRITEMASK_Y),
               ureg_scalar(ureg_src(delta), TGSI_SWIZZLE_Y),
               ureg_imm1f(ureg, y_sign),
               ureg_scalar(pos, TGSI_SWIZZLE_Y));
      ureg_MOV(ureg, out_texcoord0,
               ureg_imm4f(ureg, s, t, 0.0f, 0.0f));
      ureg_MOV(ureg, out_texcoord1,
               ureg_imm4f(ureg, s, t, 0.0f, 0.0f));
      ureg_EMIT(ureg, ureg_imm1u(ureg, 0));
      if (vertex == 2)
         ureg_ENDPRIM(ureg, ureg_imm1u(ureg, 0));
   }
   ureg_ENDPRIM(ureg, ureg_imm1u(ureg, 0));
   if (!ureg_dst_is_undef(clamped_point_size))
      ureg_release_temporary(ureg, clamped_point_size);
   ureg_release_temporary(ureg, delta);
   ureg_END(ureg);
   return ureg_create_shader_and_destroy(ureg, pipe);
}

static unsigned
D3D9BlendFactorToPipe(UINT blend)
{
   switch (blend) {
   case D3DBLEND_ZERO:
      return PIPE_BLENDFACTOR_ZERO;
   case D3DBLEND_ONE:
      return PIPE_BLENDFACTOR_ONE;
   case D3DBLEND_SRCCOLOR:
      return PIPE_BLENDFACTOR_SRC_COLOR;
   case D3DBLEND_INVSRCCOLOR:
      return PIPE_BLENDFACTOR_INV_SRC_COLOR;
   case D3DBLEND_SRCALPHA:
      return PIPE_BLENDFACTOR_SRC_ALPHA;
   case D3DBLEND_INVSRCALPHA:
      return PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   case D3DBLEND_DESTALPHA:
      return PIPE_BLENDFACTOR_DST_ALPHA;
   case D3DBLEND_INVDESTALPHA:
      return PIPE_BLENDFACTOR_INV_DST_ALPHA;
   case D3DBLEND_DESTCOLOR:
      return PIPE_BLENDFACTOR_DST_COLOR;
   case D3DBLEND_INVDESTCOLOR:
      return PIPE_BLENDFACTOR_INV_DST_COLOR;
   case D3DBLEND_SRCALPHASAT:
      return PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE;
   case D3DBLEND_BLENDFACTOR:
      return PIPE_BLENDFACTOR_CONST_COLOR;
   case D3DBLEND_INVBLENDFACTOR:
      return PIPE_BLENDFACTOR_INV_CONST_COLOR;
   default:
      return PIPE_BLENDFACTOR_ONE;
   }
}

static unsigned
D3D9BlendOpToPipe(UINT op)
{
   switch (op) {
   case D3DBLENDOP_SUBTRACT:
      return PIPE_BLEND_SUBTRACT;
   case D3DBLENDOP_REVSUBTRACT:
      return PIPE_BLEND_REVERSE_SUBTRACT;
   case D3DBLENDOP_MIN:
      return PIPE_BLEND_MIN;
   case D3DBLENDOP_MAX:
      return PIPE_BLEND_MAX;
   case D3DBLENDOP_ADD:
   default:
      return PIPE_BLEND_ADD;
   }
}

static unsigned
D3D9ColorWriteMaskToPipe(UINT mask)
{
   unsigned pipe_mask = 0;
   if (mask & D3DCOLORWRITEENABLE_RED)
      pipe_mask |= PIPE_MASK_R;
   if (mask & D3DCOLORWRITEENABLE_GREEN)
      pipe_mask |= PIPE_MASK_G;
   if (mask & D3DCOLORWRITEENABLE_BLUE)
      pipe_mask |= PIPE_MASK_B;
   if (mask & D3DCOLORWRITEENABLE_ALPHA)
      pipe_mask |= PIPE_MASK_A;
   return pipe_mask;
}

static void
D3D9SetBlendColor(D3D9Device *device)
{
   const UINT color = device->render_states[D3DRS_BLENDFACTOR];
   struct pipe_blend_color state;

   state.color[0] = (float)((color >> 16) & 0xff) / 255.0f;
   state.color[1] = (float)((color >> 8) & 0xff) / 255.0f;
   state.color[2] = (float)(color & 0xff) / 255.0f;
   state.color[3] = (float)((color >> 24) & 0xff) / 255.0f;
   device->pipe->set_blend_color(device->pipe, &state);
}

static void *
D3D9CreateBlendState(D3D9Device *device)
{
   struct pipe_blend_state state;
   memset(&state, 0, sizeof(state));

   if (device->render_states[D3DRS_ALPHABLENDENABLE]) {
      UINT src_blend = device->render_states[D3DRS_SRCBLEND] ?
         device->render_states[D3DRS_SRCBLEND] : D3DBLEND_ONE;
      UINT dst_blend = device->render_states[D3DRS_DESTBLEND] ?
         device->render_states[D3DRS_DESTBLEND] : D3DBLEND_ZERO;
      const UINT blend_op = device->render_states[D3DRS_BLENDOP] ?
         device->render_states[D3DRS_BLENDOP] : D3DBLENDOP_ADD;
      const bool separate_alpha =
         device->render_states[D3DRS_SEPARATEALPHABLENDENABLE] != 0;

      if (src_blend == D3DBLEND_BOTHSRCALPHA) {
         src_blend = D3DBLEND_SRCALPHA;
         dst_blend = D3DBLEND_INVSRCALPHA;
      } else if (src_blend == D3DBLEND_BOTHINVSRCALPHA) {
         src_blend = D3DBLEND_INVSRCALPHA;
         dst_blend = D3DBLEND_SRCALPHA;
      }

      const UINT src_blend_alpha =
         separate_alpha && device->render_states[D3DRS_SRCBLENDALPHA] ?
         device->render_states[D3DRS_SRCBLENDALPHA] : src_blend;
      const UINT dst_blend_alpha =
         separate_alpha && device->render_states[D3DRS_DESTBLENDALPHA] ?
         device->render_states[D3DRS_DESTBLENDALPHA] : dst_blend;
      const UINT blend_op_alpha =
         separate_alpha && device->render_states[D3DRS_BLENDOPALPHA] ?
         device->render_states[D3DRS_BLENDOPALPHA] : blend_op;

      state.rt[0].blend_enable = 1;
      state.rt[0].rgb_src_factor = D3D9BlendFactorToPipe(src_blend);
      state.rt[0].rgb_dst_factor = D3D9BlendFactorToPipe(dst_blend);
      state.rt[0].rgb_func = D3D9BlendOpToPipe(blend_op);
      state.rt[0].alpha_src_factor =
         D3D9BlendFactorToPipe(src_blend_alpha);
      state.rt[0].alpha_dst_factor =
         D3D9BlendFactorToPipe(dst_blend_alpha);
      state.rt[0].alpha_func = D3D9BlendOpToPipe(blend_op_alpha);
   } else {
      state.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
      state.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_ZERO;
      state.rt[0].rgb_func = PIPE_BLEND_ADD;
      state.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
      state.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_ZERO;
      state.rt[0].alpha_func = PIPE_BLEND_ADD;
   }

   static const UINT color_write_states[4] = {
      D3DRS_COLORWRITEENABLE,
      D3DRS_COLORWRITEENABLE1,
      D3DRS_COLORWRITEENABLE2,
      D3DRS_COLORWRITEENABLE3,
   };
   const unsigned first_mask = D3D9ColorWriteMaskToPipe(
      device->render_states[color_write_states[0]]);
   state.rt[0].colormask = first_mask;
   for (UINT i = 1; i < ARRAYSIZE(color_write_states); ++i) {
      state.rt[i] = state.rt[0];
      state.rt[i].colormask = D3D9ColorWriteMaskToPipe(
         device->render_states[color_write_states[i]]);
      if (state.rt[i].colormask != first_mask) {
         state.independent_blend_enable = 1;
         state.max_rt = ARRAYSIZE(color_write_states) - 1;
      }
   }

   return device->pipe->create_blend_state(device->pipe, &state);
}

static void *
D3D9CreateRasterizerState(D3D9Device *device, bool depth_clamp,
                          bool point_position_gs)
{
   struct pipe_context *pipe = device->pipe;
   struct pipe_rasterizer_state state;
   memset(&state, 0, sizeof(state));
   state.half_pixel_center = 1;
   state.bottom_edge_rule = 0;
   state.clip_halfz = 1;
   state.flatshade =
      device->render_states[D3DRS_SHADEMODE] == D3DSHADE_FLAT;
   state.flatshade_first = 1;
   state.depth_clip_near = depth_clamp ? 0 : 1;
   state.depth_clip_far = depth_clamp ? 0 : 1;
   state.depth_clamp = depth_clamp ? 1 : 0;
   state.scissor =
      device->render_states[D3DDDIRS_SCISSORTESTENABLE] ? 1 : 0;
   state.line_last_pixel =
      device->render_states[D3DRS_LASTPIXEL] ? 1 : 0;
   state.multisample = 1;
   state.cull_face = PIPE_FACE_NONE;
   switch (device->render_states[D3DRS_CULLMODE]) {
   case D3DCULL_CW:
      state.cull_face = PIPE_FACE_FRONT;
      break;
   case D3DCULL_CCW:
      state.cull_face = PIPE_FACE_BACK;
      break;
   default:
      break;
   }
   state.fill_front = PIPE_POLYGON_MODE_FILL;
   state.fill_back = PIPE_POLYGON_MODE_FILL;
   state.line_width = 1.0f;
   state.point_size = D3D9PointSize(device);
   state.point_quad_rasterization = !point_position_gs &&
      (state.point_size > 1.0f ||
       device->render_states[D3DRS_POINTSPRITEENABLE]);
   state.sprite_coord_mode = PIPE_SPRITE_COORD_UPPER_LEFT;
   if (!point_position_gs && device->render_states[D3DRS_POINTSPRITEENABLE])
      state.sprite_coord_enable = 0xff;
   return pipe->create_rasterizer_state(pipe, &state);
}

static float
D3D9PointSize(const D3D9Device *device)
{
   float point_size =
      D3D9RenderStateFloat(device->render_states[D3DRS_POINTSIZE]);
   if (point_size <= 0.0f)
      point_size = 1.0f;

   const float point_min =
      D3D9RenderStateFloat(device->render_states[D3DRS_POINTSIZE_MIN]);
   const float point_max =
      D3D9RenderStateFloat(device->render_states[D3DRS_POINTSIZE_MAX]);
   if (point_min > 0.0f && point_size < point_min)
      point_size = point_min;
   if (point_max > 0.0f && point_size > point_max)
      point_size = point_max;
   return point_size;
}

static void *
D3D9CreateDepthStencilState(D3D9Device *device)
{
   struct pipe_depth_stencil_alpha_state state;
   memset(&state, 0, sizeof(state));

   const bool depth_active = D3D9DepthTestActive(device);
   state.depth_enabled = depth_active ? 1 : 0;
   state.depth_writemask = depth_active &&
      device->render_states[D3DDDIRS_ZWRITEENABLE] ? 1 : 0;
   state.depth_func =
      D3D9CompareToPipe(device->render_states[D3DDDIRS_ZFUNC] ?
                        device->render_states[D3DDDIRS_ZFUNC] :
                        D3DCMP_LESSEQUAL);
   state.alpha_enabled =
      device->render_states[D3DRS_ALPHATESTENABLE] ? 1 : 0;
   state.alpha_func =
      D3D9CompareToPipe(device->render_states[D3DRS_ALPHAFUNC] ?
                        device->render_states[D3DRS_ALPHAFUNC] :
                        D3DCMP_ALWAYS);
   state.alpha_ref_value =
      (float)(device->render_states[D3DRS_ALPHAREF] & 0xff) / 255.0f;

   if (device->depth_stencil && device->render_states[D3DRS_STENCILENABLE]) {
      struct pipe_stencil_state *front = &state.stencil[0];
      struct pipe_stencil_state *back = &state.stencil[1];
      const UINT stencil_mask =
         device->render_states[D3DRS_STENCILMASK];
      const UINT stencil_write_mask =
         device->render_states[D3DRS_STENCILWRITEMASK];

      front->enabled = 1;
      front->func =
         D3D9CompareToPipe(device->render_states[D3DRS_STENCILFUNC] ?
                           device->render_states[D3DRS_STENCILFUNC] :
                           D3DCMP_ALWAYS);
      front->fail_op =
         D3D9StencilOpToPipe(device->render_states[D3DRS_STENCILFAIL] ?
                             device->render_states[D3DRS_STENCILFAIL] :
                             D3DSTENCILOP_KEEP);
      front->zfail_op =
         D3D9StencilOpToPipe(device->render_states[D3DRS_STENCILZFAIL] ?
                             device->render_states[D3DRS_STENCILZFAIL] :
                             D3DSTENCILOP_KEEP);
      front->zpass_op =
         D3D9StencilOpToPipe(device->render_states[D3DRS_STENCILPASS] ?
                             device->render_states[D3DRS_STENCILPASS] :
                             D3DSTENCILOP_KEEP);
      front->valuemask = stencil_mask;
      front->writemask = stencil_write_mask;

      if (device->render_states[D3DRS_TWOSIDEDSTENCILMODE]) {
         *back = *front;
         back->enabled = 1;
         back->func =
            D3D9CompareToPipe(device->render_states[D3DRS_CCW_STENCILFUNC] ?
                              device->render_states[D3DRS_CCW_STENCILFUNC] :
                              D3DCMP_ALWAYS);
         back->fail_op =
            D3D9StencilOpToPipe(
               device->render_states[D3DRS_CCW_STENCILFAIL] ?
               device->render_states[D3DRS_CCW_STENCILFAIL] :
               D3DSTENCILOP_KEEP);
         back->zfail_op =
            D3D9StencilOpToPipe(
               device->render_states[D3DRS_CCW_STENCILZFAIL] ?
               device->render_states[D3DRS_CCW_STENCILZFAIL] :
               D3DSTENCILOP_KEEP);
         back->zpass_op =
            D3D9StencilOpToPipe(
               device->render_states[D3DRS_CCW_STENCILPASS] ?
               device->render_states[D3DRS_CCW_STENCILPASS] :
               D3DSTENCILOP_KEEP);
      }
   }

   return device->pipe->create_depth_stencil_alpha_state(device->pipe, &state);
}

static void
D3D9BindViewport(D3D9Device *device)
{
   struct pipe_viewport_state state;
   memset(&state, 0, sizeof(state));

   const float width = device->viewport.Width ? (float)device->viewport.Width :
      1.0f;
   const float height = device->viewport.Height ? (float)device->viewport.Height :
      1.0f;
   const float min_z = device->zrange.MinZ;
   const float max_z = device->zrange.MaxZ ? device->zrange.MaxZ : 1.0f;
   const float half_width = width / 2.0f;
   const float half_height = height / 2.0f;

   state.scale[0] = half_width;
   state.scale[1] = -half_height;
   state.scale[2] = max_z - min_z;
   /* D3D9 defines pixel centers at integer window coordinates. */
   state.translate[0] = half_width + (float)device->viewport.X + 0.5f;
   state.translate[1] = half_height + (float)device->viewport.Y + 0.5f;
   state.translate[2] = min_z;
   device->pipe->set_viewport_states(device->pipe, 0, 1, &state);
}

static bool
D3D9BindFramebuffer(D3D9Device *device, struct pipe_framebuffer_state *fb)
{
   memset(fb, 0, sizeof(*fb));

   D3D9Resource *rt = D3D9CastResource(device->render_targets[0]);
   if (!rt || !rt->pipe_resource)
      return false;

   D3D9SubResource *rt_sub =
      D3D9GetSubResource(device->render_targets[0],
                         device->render_target_subresources[0]);
   if (!rt_sub)
      return false;

   fb->width = rt_sub->width;
   fb->height = rt_sub->height;

   const UINT max_render_targets =
      MIN2((UINT)ARRAYSIZE(device->render_targets),
           (UINT)ARRAYSIZE(fb->cbufs));
   for (UINT i = 0; i < max_render_targets; ++i) {
      D3D9Resource *cbuf_resource =
         D3D9CastResource(device->render_targets[i]);
      if (!cbuf_resource || !cbuf_resource->pipe_resource)
         continue;

      D3D9SubResource *cbuf_sub =
         D3D9GetSubResource(device->render_targets[i],
                            device->render_target_subresources[i]);
      if (!cbuf_sub || cbuf_sub->width != fb->width ||
          cbuf_sub->height != fb->height)
         continue;

      fb->cbufs[i].format = cbuf_resource->pipe_resource->format;
      if (device->render_states[D3DDDIRS_SRGBWRITEENABLE]) {
         const enum pipe_format srgb_format =
            util_format_srgb(fb->cbufs[i].format);
         if (srgb_format != PIPE_FORMAT_NONE)
            fb->cbufs[i].format = srgb_format;
      }
      fb->cbufs[i].level =
         D3D9PipeMipLevel(cbuf_resource,
                          device->render_target_subresources[i]);
      fb->cbufs[i].first_layer =
         D3D9PipeLayer(cbuf_resource,
                       device->render_target_subresources[i]);
      fb->cbufs[i].last_layer = fb->cbufs[i].first_layer;
      fb->cbufs[i].texture = cbuf_resource->pipe_resource;
      fb->nr_cbufs = i + 1;
   }

   D3D9Resource *zs = D3D9CastResource(device->depth_stencil);
   if (zs && zs->pipe_resource) {
      D3D9SubResource *zs_sub = D3D9GetSubResource(device->depth_stencil, 0);
      if (zs_sub && zs_sub->width >= fb->width && zs_sub->height >= fb->height) {
         fb->zsbuf.format = zs->pipe_resource->format;
         fb->zsbuf.level = 0;
         fb->zsbuf.first_layer = 0;
         fb->zsbuf.last_layer = 0;
         fb->zsbuf.texture = zs->pipe_resource;
      }
   }

   device->pipe->set_framebuffer_state(device->pipe, fb);
   return true;
}

static float
D3D9RenderStateFloat(UINT value)
{
   union {
      UINT u;
      float f;
   } data;
   data.u = value;
   return data.f;
}

static void *
D3D9CreateNineVsConstants(D3D9Device *device, unsigned *out_size)
{
   const D3D9Object *shader =
      (const D3D9Object *)device->vertex_shader_func;
   const unsigned *ranges = shader ? shader->translated_vs_const_ranges : NULL;
   const unsigned size = shader ? shader->translated_vs_const_used_size : 0;
   const unsigned max_consts = size / sizeof(uint32_t[4]);
   uint32_t (*constants)[4];
   unsigned dst = 0;
   const bool native_integers =
      device->screen->shader_caps[MESA_SHADER_VERTEX].integers;

   if (out_size)
      *out_size = 0;

   if (!size || !max_consts)
      return NULL;

   constants = (uint32_t (*)[4])calloc(1, size);
   if (!constants)
      return NULL;

   if (ranges) {
      for (unsigned range = 0; ranges[range * 2 + 1] && dst < max_consts;
           ++range) {
         const unsigned start = ranges[range * 2];
         const unsigned count = ranges[range * 2 + 1];

         for (unsigned i = 0; i < count && dst < max_consts; ++i, ++dst) {
            const unsigned slot = start + i;

            D3D9StoreNineVsConstantSlot(device, constants[dst], slot,
                                        native_integers);
         }
      }
   } else {
      for (unsigned i = 0; i < max_consts; ++i)
         D3D9StoreNineVsConstantSlot(device, constants[i], i,
                                     native_integers);
   }

   D3D9OverlayNineLocalFloatConstants(constants, max_consts,
                                      shader ?
                                      shader->translated_vs_lconstf_ranges :
                                      NULL,
                                      shader ?
                                      shader->translated_vs_lconstf_data :
                                      NULL);

   if (out_size)
      *out_size = size;
   return constants;
}

static void *
D3D9CreateNinePsConstants(D3D9Device *device, const D3D9Object *shader,
                          unsigned *out_size)
{
   const unsigned *ranges = shader ? shader->translated_ps_const_ranges : NULL;
   const unsigned size = shader ? shader->translated_ps_const_used_size : 0;
   const unsigned max_consts = size / sizeof(uint32_t[4]);
   uint32_t (*constants)[4];
   unsigned dst = 0;
   const bool native_integers =
      device->screen->shader_caps[MESA_SHADER_FRAGMENT].integers;

   if (out_size)
      *out_size = 0;

   if (!size || !max_consts)
      return NULL;

   constants = (uint32_t (*)[4])calloc(1, size);
   if (!constants)
      return NULL;

   if (ranges) {
      for (unsigned range = 0; ranges[range * 2 + 1] && dst < max_consts;
           ++range) {
         const unsigned start = ranges[range * 2];
         const unsigned count = ranges[range * 2 + 1];

         for (unsigned i = 0; i < count && dst < max_consts; ++i, ++dst) {
            const unsigned slot = start + i;

            D3D9StoreNinePsConstantSlot(device, constants[dst], slot,
                                        native_integers);
         }
      }
   } else {
      for (unsigned i = 0; i < max_consts; ++i)
         D3D9StoreNinePsConstantSlot(device, constants[i], i,
                                     native_integers);
   }

   D3D9OverlayNineLocalFloatConstants(constants, max_consts,
                                      shader ?
                                      shader->translated_ps_lconstf_ranges :
                                      NULL,
                                      shader ?
                                      shader->translated_ps_lconstf_data :
                                      NULL);

   if (out_size)
      *out_size = size;
   return constants;
}

static unsigned
D3D9TextureFilterToPipe(UINT filter)
{
   return (filter == D3DTEXF_LINEAR || filter == D3DTEXF_ANISOTROPIC) ?
      PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST;
}

static unsigned
D3D9TextureMipFilterToPipe(UINT filter)
{
   switch (filter) {
   case D3DTEXF_NONE:
      return PIPE_TEX_MIPFILTER_NONE;
   case D3DTEXF_LINEAR:
      return PIPE_TEX_MIPFILTER_LINEAR;
   default:
      return PIPE_TEX_MIPFILTER_NEAREST;
   }
}

static void
D3D9InitSamplerLod(struct pipe_sampler_state *sampler)
{
   sampler->min_lod = 0.0f;
   sampler->max_lod = PIPE_MAX_TEXTURE_LEVELS - 1;
}

static unsigned
D3D9TextureAddressToPipe(UINT address)
{
   switch (address) {
   case D3DTADDRESS_WRAP:
      return PIPE_TEX_WRAP_REPEAT;
   case D3DTADDRESS_MIRROR:
      return PIPE_TEX_WRAP_MIRROR_REPEAT;
   case D3DTADDRESS_CLAMP:
      return PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   case D3DTADDRESS_BORDER:
      return PIPE_TEX_WRAP_CLAMP_TO_BORDER;
   case D3DTADDRESS_MIRRORONCE:
      /* Venus2 does not enable VK_KHR_sampler_mirror_clamp_to_edge yet.
       * Keep runtime-compatible acceptance, but issue valid Vulkan using the
       * loud clamp approximation selected by D3D9SetTextureStageState. */
      return PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   default:
      return PIPE_TEX_WRAP_REPEAT;
   }
}

static bool
D3D9UploadTexture(D3D9Device *device, D3D9Resource *texture)
{
   if (!device || !texture)
      return false;

   for (UINT i = 0; i < texture->surf_count; ++i) {
      if (!D3D9UploadSubResource(device, texture, i))
         return false;
   }

   return true;
}

static UINT
D3D9TextureLayerCount(const struct pipe_resource *resource)
{
   if (!resource)
      return 0;

   switch (resource->target) {
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return resource->array_size;
   case PIPE_TEXTURE_3D:
      return resource->depth0;
   default:
      return 1;
   }
}

static bool
D3D9GenerateAutogenMipmaps(D3D9Device *device, D3D9Resource *texture)
{
   if (!device || !device->pipe || !texture || !texture->pipe_resource ||
       !texture->autogen_mipmap_dirty)
      return true;

   struct pipe_resource *resource = texture->pipe_resource;
   if (resource->target == PIPE_BUFFER || !resource->last_level) {
      texture->autogen_mipmap_dirty = false;
      return true;
   }

   const UINT layer_count = D3D9TextureLayerCount(resource);
   if (!layer_count)
      return false;

   if (!util_gen_mipmap(device->pipe, resource, resource->format, 0,
                        resource->last_level, 0, layer_count - 1,
                        PIPE_TEX_FILTER_LINEAR)) {
      static volatile LONG logged;
      D3D9WarnOncef(&logged, "unsupported D3D9 autogen mipmap "
                    "resource=%p target=%u format=%u levels=%u\n",
                    texture, resource->target, resource->format,
                    resource->last_level + 1);
      return false;
   }

   texture->autogen_mipmap_dirty = false;
   return true;
}

static UINT
D3D9TextureBaseLevel(D3D9Device *device, UINT stage, D3D9Resource *texture)
{
   const UINT mip_filter =
      device->texture_stage_states[stage][D3DDDITSS_MIPFILTER];
   const UINT texture_levels = texture->pipe_resource->last_level + 1;
   UINT level = mip_filter == D3DTEXF_NONE ? 0 :
      device->texture_stage_states[stage][D3DDDITSS_MAXMIPLEVEL];
   if (level >= texture_levels)
      level = texture_levels - 1;
   return level;
}

static UINT
D3D9TextureLastLevel(D3D9Device *device, UINT stage, D3D9Resource *texture,
                     UINT base_level)
{
   const UINT mip_filter =
      device->texture_stage_states[stage][D3DDDITSS_MIPFILTER];
   const UINT max_mip_level =
      device->texture_stage_states[stage][D3DDDITSS_MAXMIPLEVEL];
   return (mip_filter == D3DTEXF_NONE || max_mip_level) ? base_level :
      texture->pipe_resource->last_level;
}

static bool
D3D9FillSamplerState(D3D9Device *device, UINT stage,
                     struct pipe_sampler_state *sampler)
{
   if (!device || !sampler || stage >= ARRAYSIZE(device->texture_stage_states))
      return false;

   memset(sampler, 0, sizeof(*sampler));
   D3D9InitSamplerLod(sampler);
   sampler->wrap_s = D3D9TextureAddressToPipe(
      device->texture_stage_states[stage][D3DDDITSS_ADDRESSU]);
   sampler->wrap_t = D3D9TextureAddressToPipe(
      device->texture_stage_states[stage][D3DDDITSS_ADDRESSV]);
   sampler->wrap_r = D3D9TextureAddressToPipe(
      device->texture_stage_states[stage][D3DDDITSS_ADDRESSW]);
   sampler->min_img_filter =
      D3D9TextureFilterToPipe(device->texture_stage_states[stage][D3DDDITSS_MINFILTER]);
   sampler->mag_img_filter =
      D3D9TextureFilterToPipe(device->texture_stage_states[stage][D3DDDITSS_MAGFILTER]);
   sampler->min_mip_filter =
      D3D9TextureMipFilterToPipe(device->texture_stage_states[stage][D3DDDITSS_MIPFILTER]);
   const UINT lod_bias =
      device->texture_stage_states[stage][D3DDDITSS_MIPMAPLODBIAS];
   sampler->lod_bias = lod_bias == D3D9_SAMPLER_FETCH4 ? 0.0f :
      D3D9RenderStateFloat(lod_bias);
   if (device->texture_stage_states[stage][D3DDDITSS_MINFILTER] ==
          D3DTEXF_ANISOTROPIC ||
       device->texture_stage_states[stage][D3DDDITSS_MAGFILTER] ==
          D3DTEXF_ANISOTROPIC) {
      const UINT requested =
         device->texture_stage_states[stage][D3DDDITSS_MAXANISOTROPY];
      sampler->max_anisotropy = MIN2(MAX2(requested, 1u), 16u);
   }
   const UINT border =
      device->texture_stage_states[stage][D3DDDITSS_BORDERCOLOR];
   sampler->border_color.f[0] = ((border >> 16) & 0xff) / 255.0f;
   sampler->border_color.f[1] = ((border >> 8) & 0xff) / 255.0f;
   sampler->border_color.f[2] = (border & 0xff) / 255.0f;
   sampler->border_color.f[3] = ((border >> 24) & 0xff) / 255.0f;
   return true;
}

static UINT
D3D9NullTextureIndex(unsigned tgsi_target)
{
   switch (tgsi_target) {
   case TGSI_TEXTURE_1D:
   case TGSI_TEXTURE_SHADOW1D:
      return 0;
   case TGSI_TEXTURE_2D:
   case TGSI_TEXTURE_SHADOW2D:
      return 1;
   case TGSI_TEXTURE_3D:
      return 2;
   case TGSI_TEXTURE_CUBE:
      return 3;
   default:
      return UINT_MAX;
   }
}

static enum pipe_texture_target
D3D9NullPipeTarget(unsigned tgsi_target)
{
   switch (tgsi_target) {
   case TGSI_TEXTURE_1D:
   case TGSI_TEXTURE_SHADOW1D:
      return PIPE_TEXTURE_1D;
   case TGSI_TEXTURE_2D:
   case TGSI_TEXTURE_SHADOW2D:
      return PIPE_TEXTURE_2D;
   case TGSI_TEXTURE_3D:
      return PIPE_TEXTURE_3D;
   case TGSI_TEXTURE_CUBE:
      return PIPE_TEXTURE_CUBE;
   default:
      return PIPE_MAX_TEXTURE_TYPES;
   }
}

static struct pipe_resource *
D3D9GetNullTexture(D3D9Device *device, unsigned tgsi_target)
{
   const UINT index = D3D9NullTextureIndex(tgsi_target);
   if (!device || !device->screen || index >= ARRAYSIZE(device->null_textures))
      return NULL;

   if (device->null_textures[index])
      return device->null_textures[index];

   const enum pipe_texture_target target = D3D9NullPipeTarget(tgsi_target);
   if (target == PIPE_MAX_TEXTURE_TYPES)
      return NULL;

   struct pipe_resource templ;
   memset(&templ, 0, sizeof(templ));
   templ.target = target;
   templ.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   templ.width0 = 1;
   templ.height0 = 1;
   templ.depth0 = 1;
   templ.array_size = target == PIPE_TEXTURE_CUBE ? 6 : 1;
   templ.last_level = 0;
   templ.nr_samples = 1;
   templ.nr_storage_samples = 1;
   templ.bind = PIPE_BIND_SAMPLER_VIEW;
   templ.usage = PIPE_USAGE_IMMUTABLE;

   device->null_textures[index] =
      device->screen->resource_create(device->screen, &templ);
   return device->null_textures[index];
}

static bool
D3D9CreateNullTextureBinding(D3D9Device *device, UINT stage,
                             unsigned tgsi_target, void **sampler_state,
                             struct pipe_sampler_view **sampler_view)
{
   struct pipe_resource *resource = D3D9GetNullTexture(device, tgsi_target);
   if (!resource)
      return false;

   struct pipe_sampler_state sampler;
   if (!D3D9FillSamplerState(device, stage, &sampler))
      return false;

   struct pipe_sampler_view view;
   u_sampler_view_default_template(&view, resource, resource->format);
   view.target = resource->target;
   view.u.tex.first_level = 0;
   view.u.tex.last_level = 0;
   view.u.tex.first_layer = 0;
   view.u.tex.last_layer = resource->target == PIPE_TEXTURE_CUBE ? 5 : 0;
   view.swizzle_a = PIPE_SWIZZLE_1;
   if (resource->target == PIPE_TEXTURE_CUBE)
      sampler.seamless_cube_map = true;

   *sampler_state = device->pipe->create_sampler_state(device->pipe,
                                                       &sampler);
   *sampler_view = device->pipe->create_sampler_view(device->pipe, resource,
                                                     &view);
   if (!*sampler_state || !*sampler_view) {
      if (*sampler_state) {
         device->pipe->delete_sampler_state(device->pipe, *sampler_state);
         *sampler_state = NULL;
      }
      if (*sampler_view) {
         pipe_sampler_view_release(*sampler_view);
         *sampler_view = NULL;
      }
      return false;
   }

   return true;
}

static bool
D3D9CreateTextureBinding(D3D9Device *device, UINT stage,
                         void **sampler_state,
                         struct pipe_sampler_view **sampler_view)
{
   D3D9Resource *texture = D3D9CastResource(device->textures[stage]);
   if (!texture || !texture->pipe_resource ||
       texture->pipe_resource->target == PIPE_BUFFER ||
       !D3D9UploadTexture(device, texture) ||
       !D3D9GenerateAutogenMipmaps(device, texture))
      return false;

   struct pipe_sampler_state sampler;
   if (!D3D9FillSamplerState(device, stage, &sampler))
      return false;

   struct pipe_resource *view_resource = texture->managed_source_pipe_resource ?
      texture->managed_source_pipe_resource :
      (texture->managed_default_pipe_resource ?
       texture->managed_default_pipe_resource : texture->pipe_resource);
   const UINT view_base_level = texture->managed_source_pipe_resource ?
      texture->managed_source_base_level : 0;
   enum pipe_format view_format = view_resource->format;
   if (device->texture_stage_states[stage][D3DDDITSS_SRGBTEXTURE]) {
      const enum pipe_format srgb_format = util_format_srgb(view_format);
      if (srgb_format != PIPE_FORMAT_NONE)
         view_format = srgb_format;
   }

   const UINT level = D3D9TextureBaseLevel(device, stage, texture);
   const UINT last_level = D3D9TextureLastLevel(device, stage, texture, level);
   const UINT first_view_level =
      MIN2(view_base_level + level, view_resource->last_level);
   const UINT last_view_level =
      MIN2(view_base_level + last_level, view_resource->last_level);
   struct pipe_sampler_view view;
   u_sampler_view_default_template(&view, view_resource, view_format);
   const UINT layer_count = D3D9TextureLayerCount(view_resource);
   view.target = view_resource->target;
   view.u.tex.first_level = first_view_level;
   view.u.tex.last_level = last_view_level;
   view.u.tex.first_layer = 0;
   view.u.tex.last_layer = layer_count ? layer_count - 1 : 0;
   if (view.target == PIPE_TEXTURE_CUBE ||
       view.target == PIPE_TEXTURE_CUBE_ARRAY)
      sampler.seamless_cube_map = true;
   if (texture->format == D3D9_FMT_L8 ||
       texture->format == D3D9_FMT_L16 ||
       texture->format == D3D9_FMT_ATI1) {
      view.swizzle_g = PIPE_SWIZZLE_X;
      view.swizzle_b = PIPE_SWIZZLE_X;
      view.swizzle_a = PIPE_SWIZZLE_1;
   }
   if (texture->format == D3D9_FMT_R16F ||
       texture->format == D3D9_FMT_R32F) {
      view.swizzle_g = PIPE_SWIZZLE_1;
      view.swizzle_b = PIPE_SWIZZLE_1;
      view.swizzle_a = PIPE_SWIZZLE_1;
   }
   if (texture->format == D3D9_FMT_G16R16 ||
       texture->format == D3D9_FMT_G16R16F ||
       texture->format == D3D9_FMT_G32R32F) {
      view.swizzle_b = PIPE_SWIZZLE_1;
      view.swizzle_a = PIPE_SWIZZLE_1;
   }
   if (texture->format == D3D9_FMT_V8U8 ||
       texture->format == D3D9_FMT_V16U16) {
      view.swizzle_b = PIPE_SWIZZLE_1;
      view.swizzle_a = PIPE_SWIZZLE_1;
   }
   if (texture->format == D3D9_FMT_DF16 ||
       texture->format == D3D9_FMT_DF24) {
      view.swizzle_g = PIPE_SWIZZLE_X;
      view.swizzle_b = PIPE_SWIZZLE_X;
      view.swizzle_a = PIPE_SWIZZLE_1;
   }
   if (texture->format == D3D9_FMT_INTZ) {
      view.swizzle_g = PIPE_SWIZZLE_X;
      view.swizzle_b = PIPE_SWIZZLE_X;
      view.swizzle_a = PIPE_SWIZZLE_X;
   }

   *sampler_state = device->pipe->create_sampler_state(device->pipe,
                                                       &sampler);
   *sampler_view = device->pipe->create_sampler_view(device->pipe,
                                                     view_resource,
                                                     &view);
   if (!*sampler_state || !*sampler_view) {
      if (*sampler_state) {
         device->pipe->delete_sampler_state(device->pipe, *sampler_state);
         *sampler_state = NULL;
      }
      if (*sampler_view) {
         pipe_sampler_view_release(*sampler_view);
         *sampler_view = NULL;
      }
      return false;
   }

   return true;
}

static bool
D3D9BindTranslatedTextures(D3D9Device *device, void **sampler_states,
                           struct pipe_sampler_view **sampler_views,
                           UINT *ps_sampler_count,
                           UINT *vs_sampler_count)
{
   UINT ps_count = 0;
   UINT vs_count = 0;
   const D3D9Object *ps = device ?
      (const D3D9Object *)device->pixel_shader : NULL;
   const D3D9Object *vs = device ?
      (const D3D9Object *)device->vertex_shader_func : NULL;
   const uint16_t ps_sampler_mask =
      (ps && ps->kind == D3D9_OBJECT_SHADER) ?
      ps->translated_ps_sampler_mask : 0;
   const uint16_t vs_sampler_mask =
      (vs && vs->kind == D3D9_OBJECT_SHADER) ?
      vs->translated_vs_sampler_mask : 0;

   for (UINT i = 0; i < ARRAYSIZE(device->textures); ++i) {
      sampler_states[i] = NULL;
      sampler_views[i] = NULL;
   }

   for (UINT i = 0; i < NINE_MAX_SAMPLERS_PS; ++i) {
      if (!device->textures[i] && !(ps_sampler_mask & (1u << i)))
         continue;

      bool ok = false;
      if (device->textures[i]) {
         ok = D3D9CreateTextureBinding(device, i, &sampler_states[i],
                                       &sampler_views[i]);
      } else {
         ok = D3D9CreateNullTextureBinding(device, i,
                                           ps->translated_ps_sampler_targets[i],
                                           &sampler_states[i],
                                           &sampler_views[i]);
      }
      if (!ok) {
         for (UINT j = 0; j < ARRAYSIZE(device->textures); ++j) {
            if (sampler_states[j])
               device->pipe->delete_sampler_state(device->pipe,
                                                  sampler_states[j]);
            if (sampler_views[j]) {
               pipe_sampler_view_release(sampler_views[j]);
               sampler_views[j] = NULL;
            }
         }
         return false;
      }
      ps_count = i + 1;
   }

   if (ps_count) {
      device->pipe->bind_sampler_states(device->pipe, MESA_SHADER_FRAGMENT,
                                        0, ps_count, sampler_states);
      device->pipe->set_sampler_views(device->pipe, MESA_SHADER_FRAGMENT,
                                      0, ps_count, 0, sampler_views);
   }

   for (UINT i = 0; i < NINE_MAX_SAMPLERS_VS; ++i) {
      const UINT texture_stage = D3DVERTEXTEXTURESAMPLER0 + i;
      if (!device->textures[texture_stage] && !(vs_sampler_mask & (1u << i)))
         continue;

      bool ok = false;
      if (device->textures[texture_stage]) {
         ok = D3D9CreateTextureBinding(device, texture_stage,
                                       &sampler_states[texture_stage],
                                       &sampler_views[texture_stage]);
      } else {
         ok = D3D9CreateNullTextureBinding(device, texture_stage,
                                           vs->translated_vs_sampler_targets[i],
                                           &sampler_states[texture_stage],
                                           &sampler_views[texture_stage]);
      }
      if (!ok) {
         for (UINT j = 0; j < ARRAYSIZE(device->textures); ++j) {
            if (sampler_states[j])
               device->pipe->delete_sampler_state(device->pipe,
                                                  sampler_states[j]);
            if (sampler_views[j]) {
               pipe_sampler_view_release(sampler_views[j]);
               sampler_views[j] = NULL;
            }
         }
         return false;
      }
      vs_count = i + 1;
   }

   if (vs_count) {
      device->pipe->bind_sampler_states(device->pipe, MESA_SHADER_VERTEX,
                                        0, vs_count,
                                        &sampler_states[D3DVERTEXTEXTURESAMPLER0]);
      device->pipe->set_sampler_views(device->pipe, MESA_SHADER_VERTEX,
                                      0, vs_count, 0,
                                      &sampler_views[D3DVERTEXTEXTURESAMPLER0]);
   }

   *ps_sampler_count = ps_count;
   *vs_sampler_count = vs_count;
   return true;
}

static void
D3D9UnbindTranslatedTextures(D3D9Device *device, void **sampler_states,
                             struct pipe_sampler_view **sampler_views,
                             UINT ps_sampler_count, UINT vs_sampler_count)
{
   void *null_samplers[ARRAYSIZE(device->textures)];
   struct pipe_sampler_view *null_views[ARRAYSIZE(device->textures)];

   if (!ps_sampler_count && !vs_sampler_count)
      return;

   memset(null_samplers, 0, sizeof(null_samplers));
   memset(null_views, 0, sizeof(null_views));
   if (ps_sampler_count) {
      device->pipe->bind_sampler_states(device->pipe, MESA_SHADER_FRAGMENT, 0,
                                        ps_sampler_count, null_samplers);
      device->pipe->set_sampler_views(device->pipe, MESA_SHADER_FRAGMENT, 0,
                                      ps_sampler_count, 0, null_views);
   }
   if (vs_sampler_count) {
      device->pipe->bind_sampler_states(device->pipe, MESA_SHADER_VERTEX, 0,
                                        vs_sampler_count, null_samplers);
      device->pipe->set_sampler_views(device->pipe, MESA_SHADER_VERTEX, 0,
                                      vs_sampler_count, 0, null_views);
   }
   for (UINT i = 0; i < ps_sampler_count; ++i) {
      if (sampler_states[i])
         device->pipe->delete_sampler_state(device->pipe, sampler_states[i]);
      if (sampler_views[i])
         pipe_sampler_view_release(sampler_views[i]);
      sampler_states[i] = NULL;
      sampler_views[i] = NULL;
   }
   for (UINT i = 0; i < vs_sampler_count; ++i) {
      const UINT texture_stage = D3DVERTEXTEXTURESAMPLER0 + i;
      if (sampler_states[texture_stage])
         device->pipe->delete_sampler_state(device->pipe,
                                            sampler_states[texture_stage]);
      if (sampler_views[texture_stage])
         pipe_sampler_view_release(sampler_views[texture_stage]);
      sampler_states[texture_stage] = NULL;
      sampler_views[texture_stage] = NULL;
   }
}

static bool
D3D9BufferSubResourceHasLiveCpuData(const D3D9Resource *resource,
                                    const D3D9SubResource *sub)
{
   if (!resource || !sub || !sub->data)
      return false;

   if (sub->notify_only_locked)
      return true;

   return D3D9ResourceUsesCpuBufferStorage(resource) &&
      (sub->cpu_locked || sub->cpu_dirty);
}

static D3D9SubResource *
D3D9GetLiveCpuBufferSubResource(D3D9Resource *resource)
{
   if (!resource || !resource->surf_count)
      return NULL;

   D3D9SubResource *sub = &resource->surfaces[0];
   if (D3D9BufferSubResourceHasLiveCpuData(resource, sub))
      return sub;

   D3D9Resource *source = resource->managed_source_resource;
   if (!source || !source->surf_count)
      return NULL;

   D3D9SubResource *source_sub = &source->surfaces[0];
   return D3D9BufferSubResourceHasLiveCpuData(source, source_sub) ?
      source_sub : NULL;
}

static bool
D3D9FillStreamVertexBuffer(D3D9Device *device, UINT stream,
                           INT first_vertex_offset,
                           struct pipe_vertex_buffer *vbuffer)
{
   if (stream >= ARRAYSIZE(device->streams) ||
       (!device->streams[stream].sysmem &&
        !device->streams[stream].vertex_buffer) ||
       !device->streams[stream].stride)
      return false;

   const bool instance_data =
      device->streams[stream].divider & D3DSTREAMSOURCE_INSTANCEDATA;
   const INT stream_offset = instance_data ? 0 : first_vertex_offset;
   if (device->streams[stream].sysmem) {
      if (stream_offset < 0)
         return false;
      vbuffer->is_user_buffer = true;
      vbuffer->buffer.user = (const uint8_t *)device->streams[stream].sysmem +
                             stream_offset;
   } else {
      D3D9Resource *resource =
         D3D9CastResource(device->streams[stream].vertex_buffer);
      if (!resource || !resource->pipe_resource)
         return false;
      const int64_t buffer_offset =
         (int64_t)device->streams[stream].offset + stream_offset;
      if (buffer_offset < 0)
         return false;
      D3D9SubResource *resource_sub = resource->surf_count ?
         &resource->surfaces[0] : NULL;
      D3D9SubResource *sub = D3D9GetLiveCpuBufferSubResource(resource);
      if (sub) {
         if ((uint64_t)buffer_offset > sub->size)
            return false;
         vbuffer->is_user_buffer = true;
         vbuffer->buffer.user = sub->data + buffer_offset;
         vbuffer->buffer_offset = 0;
         return true;
      }
      if (resource_sub && resource_sub->cpu_dirty &&
          !D3D9UploadSubResource(device, resource, 0))
         return false;
      vbuffer->is_user_buffer = false;
      vbuffer->buffer.resource = resource->pipe_resource;
      vbuffer->buffer_offset = (unsigned)buffer_offset;
   }

   return true;
}

static bool
D3D9UploadWorkerUserVertexBuffers(
   D3D9Device *device,
   const struct pipe_vertex_element *elements,
   UINT element_count,
   bool indexed,
   INT index_bias,
   UINT min_index,
   UINT num_vertices,
   UINT start_vertex,
   UINT vertex_count,
   UINT instance_count,
   struct pipe_vertex_buffer *vbuffers,
   UINT vbuffer_count,
   D3D9AutoUploadResources *uploads)
{
   struct pipe_context *pipe = device ? device->pipe : NULL;
   if (!pipe || !pipe->stream_uploader || !elements || !element_count ||
       !vbuffers || !uploads)
      return false;

   uploads->pipe = pipe;

   const int64_t indexed_first = (int64_t)min_index + index_bias;
   if (indexed && indexed_first < 0)
      return false;

   const uint64_t vertex_first = indexed ?
      (uint64_t)indexed_first : start_vertex;
   const uint64_t vertices = indexed ?
      (num_vertices ? num_vertices : 1) : vertex_count;

   for (UINT stream = 0; stream < vbuffer_count; ++stream) {
      struct pipe_vertex_buffer *vbuffer = &vbuffers[stream];

      uint64_t span = 0;
      UINT stride = 0;
      UINT divisor = 0;
      bool used = false;
      for (UINT i = 0; i < element_count; ++i) {
         const struct pipe_vertex_element *element = &elements[i];
         if (element->vertex_buffer_index != stream)
            continue;

         const unsigned size = util_format_get_blocksize(
            (enum pipe_format)element->src_format);
         if (!size || (used && (stride != element->src_stride ||
                               divisor != element->instance_divisor)))
            return false;

         used = true;
         stride = element->src_stride;
         divisor = element->instance_divisor;
         span = MAX2(span, (uint64_t)element->src_offset + size);
      }

      /* vbuffer_count is the highest active stream plus one, so sparse
       * declarations leave empty entries in the array.  There is nothing to
       * snapshot for those entries. */
      if (!used)
         continue;
      if (!span || !vbuffer->buffer.user)
         return false;

      const uint64_t first = divisor ? 0 : vertex_first;
      const uint64_t count = divisor ?
         DIV_ROUND_UP((uint64_t)MAX2(instance_count, 1u), divisor) :
         vertices;
      if (!count || span > UINT_MAX ||
          (stride &&
           (first > UINT_MAX / stride ||
            count - 1 > (UINT_MAX - span) / stride)))
         return false;

      const uint64_t source_offset = first * stride;
      const uint64_t size = stride ? (count - 1) * stride + span : span;
      if (source_offset > UINT_MAX - size)
         return false;

      if (!vbuffer->is_user_buffer) {
         if (stream >= ARRAYSIZE(device->streams) ||
             !device->streams[stream].vertex_buffer)
            continue;

         D3D9Resource *resource = D3D9CastResource(
            device->streams[stream].vertex_buffer);
         D3D9SubResource *sub = resource && resource->surf_count ?
            &resource->surfaces[0] : NULL;
         if (!sub || !sub->worker_upload_buffer)
            continue;

         const uint64_t logical_source =
            (uint64_t)vbuffer->buffer_offset + source_offset;
         const uint64_t snapshot_start = sub->worker_upload_source_offset;
         const uint64_t snapshot_end =
            snapshot_start + sub->worker_upload_size;
         if (logical_source < snapshot_start ||
             logical_source + size < logical_source ||
             logical_source + size > snapshot_end)
            continue;

         const uint64_t new_buffer_offset =
            (uint64_t)sub->worker_upload_offset + vbuffer->buffer_offset;
         if (new_buffer_offset < snapshot_start ||
             new_buffer_offset - snapshot_start > UINT_MAX)
            return false;

         vbuffer->buffer.resource = sub->worker_upload_buffer;
         vbuffer->buffer_offset =
            (unsigned)(new_buffer_offset - snapshot_start);
         continue;
      }

      unsigned upload_offset = 0;
      struct pipe_resource *upload_buffer = NULL;
      struct pipe_resource *release_buffer = NULL;
      u_upload_data(pipe->stream_uploader,
                    (unsigned)source_offset,
                    (unsigned)size,
                    16,
                    (const uint8_t *)vbuffer->buffer.user + source_offset,
                    &upload_offset,
                    &upload_buffer,
                    &release_buffer);
      if (!upload_buffer || upload_offset < source_offset) {
         pipe_resource_release(pipe, release_buffer);
         return false;
      }

      uploads->resources[stream] = release_buffer;
      vbuffer->is_user_buffer = false;
      vbuffer->buffer.resource = upload_buffer;
      vbuffer->buffer_offset = upload_offset - (unsigned)source_offset;
   }

   return true;
}

static HRESULT
D3D9DrawDiffuse(D3D9Device *device, D3DPRIMITIVETYPE primitive_type,
                UINT primitive_count, UINT start_vertex,
                INT first_vertex_offset, bool indexed, INT index_bias,
                UINT start_index, UINT min_index, UINT num_vertices,
                UINT index_size, const void *user_indices,
                bool runtime_transformed_vertices)
{
   if (D3D9ConsumeBackendDrawFailure(device))
      return E_FAIL;

   const enum mesa_prim mode = D3D9PrimitiveToPipe(primitive_type);
   const UINT vertex_count =
      D3D9PrimitiveVertexCount(primitive_type, primitive_count);
   if (mode == MESA_PRIM_COUNT || !vertex_count)
      return E_INVALIDARG;
   if (indexed && index_size != 2 && index_size != 4)
      return E_INVALIDARG;
   if (indexed && (int64_t)min_index + index_bias < 0)
      return E_INVALIDARG;
   if (indexed && num_vertices && num_vertices - 1 > UINT_MAX - min_index)
      return E_INVALIDARG;

   D3D9Object fixed_function_vs_shader;
   D3D9AutoShaderState fixed_function_shaders;
   fixed_function_shaders.pipe = device->pipe;
   if (device->vertex_shader_func || device->pixel_shader) {
      HRESULT shader_state_hr = D3D9UpdateTranslatedShaderState(device);
      if (FAILED(shader_state_hr))
         return shader_state_hr;
   }

   void *translated_ps = D3D9TranslatedPsState(device);
   void *translated_vs = D3D9TranslatedVsState(device);
   const bool translated_shader_pair = translated_vs && translated_ps;
   const bool runtime_fixed_function_pair =
      translated_shader_pair &&
      D3D9TranslatedShaderPairIsRuntimeFixedFunction(device);
   const bool generate_position_t_ps_input_vs =
      !translated_vs && D3D9CanUsePositionTPsInputFixedFunction(device);
   const bool generate_position_t_diffuse_vs =
      !generate_position_t_ps_input_vs && !translated_vs &&
      D3D9CanUsePositionTDiffuseFixedFunction(device);
   const D3D9Object *current_ps =
      translated_ps ? (const D3D9Object *)device->pixel_shader : NULL;
   const bool generate_position_t_texture0_runtime_ps_vs =
      !generate_position_t_ps_input_vs && !generate_position_t_diffuse_vs &&
      !device->vertex_shader_func && !translated_vs && translated_ps &&
      D3D9InitPositionTTexture0RuntimePsFixedFunctionVsInfo(
         device, current_ps, &fixed_function_vs_shader);
   /* DrawPrimitive2 can carry runtime-processed SWVP vertices. */
   const bool generate_position_t_runtime_processed_diffuse_vs =
      runtime_transformed_vertices && D3D9NoBoundTextures(device) &&
      D3D9PositionTDiffuseDeclOnly(device);
   if (generate_position_t_runtime_processed_diffuse_vs)
      D3D9InitPositionTDiffuseFixedFunctionVsInfo(&fixed_function_vs_shader);
   const bool generate_position_t_runtime_processed_vs =
      runtime_transformed_vertices && D3D9ActiveVertexDeclHasPositionT(device) &&
      !generate_position_t_runtime_processed_diffuse_vs && translated_ps &&
      D3D9InitPositionTPsInputFixedFunctionVsInfo(&fixed_function_vs_shader,
                                                  current_ps);
   /*
    * D3D9 POSITIONT fixed-function draws consume app-provided texture
    * coordinates directly.  The runtime's generated FF VS may still contain
    * texture-transform code, so keep the translated FF PS but generate the
    * POSITIONT VS from the active declaration.
    */
   const bool replace_position_t_runtime_ff_vs =
      runtime_fixed_function_pair && D3D9ActiveVertexDeclHasPositionT(device) &&
      D3D9InitPositionTTexture0RuntimePsFixedFunctionVsInfo(
         device, current_ps, &fixed_function_vs_shader);

   if (((!translated_vs || replace_position_t_runtime_ff_vs) &&
        !generate_position_t_ps_input_vs &&
        !generate_position_t_diffuse_vs &&
        !generate_position_t_texture0_runtime_ps_vs &&
        !generate_position_t_runtime_processed_diffuse_vs &&
        !generate_position_t_runtime_processed_vs &&
        !replace_position_t_runtime_ff_vs) || !translated_ps)
      return D3D9FailNoTranslatedShaderPair();

   if (generate_position_t_ps_input_vs) {
      const D3D9Object *ps = (const D3D9Object *)device->pixel_shader;
      if (!D3D9InitPositionTPsInputFixedFunctionVsInfo(
             &fixed_function_vs_shader, ps))
         return E_FAIL;
   } else if (generate_position_t_diffuse_vs ||
              generate_position_t_runtime_processed_diffuse_vs) {
      D3D9InitPositionTDiffuseFixedFunctionVsInfo(&fixed_function_vs_shader);
   }

   const D3D9Object *translated_ps_shader =
      translated_ps ? (const D3D9Object *)device->pixel_shader : NULL;
   const D3D9Object *translated_vs_shader =
      (generate_position_t_ps_input_vs || generate_position_t_diffuse_vs ||
       generate_position_t_texture0_runtime_ps_vs ||
       generate_position_t_runtime_processed_diffuse_vs ||
       generate_position_t_runtime_processed_vs ||
       replace_position_t_runtime_ff_vs) ?
      &fixed_function_vs_shader :
      (translated_vs ? (const D3D9Object *)device->vertex_shader_func : NULL);

   const bool translated_vs_needs_constants =
      translated_vs_shader && translated_vs_shader->translated_vs_const_used_size;
   const bool translated_vs_point_size =
      translated_vs && !generate_position_t_runtime_processed_diffuse_vs &&
      !generate_position_t_runtime_processed_vs &&
      D3D9TranslatedVsOutputsPointSize(device);

   struct pipe_vertex_element elements[PIPE_MAX_ATTRIBS];
   uint32_t stream_mask = 0;
   bool uses_default_input = false;
   const UINT element_count =
      D3D9FillTranslatedVsVertexElements(device, translated_vs_shader, elements,
                                         ARRAYSIZE(elements), &stream_mask,
                                         &uses_default_input);
   if (!element_count || (!stream_mask && !uses_default_input))
      return E_FAIL;

   struct pipe_context *pipe = device->pipe;
   struct pipe_vertex_buffer vbuffers[PIPE_MAX_ATTRIBS];
   memset(vbuffers, 0, sizeof(vbuffers));
   UINT vbuffer_count = 0;
   for (UINT stream = 0; stream < ARRAYSIZE(device->streams); ++stream) {
      if (!(stream_mask & (1u << stream)))
         continue;
      if (!D3D9FillStreamVertexBuffer(device, stream, first_vertex_offset,
                                      &vbuffers[stream]))
         return E_INVALIDARG;
      vbuffer_count = stream + 1;
   }
   static const float default_input[4] = {};
   if (uses_default_input) {
      vbuffers[D3D9_DEFAULT_VERTEX_INPUT_BUFFER].is_user_buffer = true;
      vbuffers[D3D9_DEFAULT_VERTEX_INPUT_BUFFER].buffer.user = default_input;
      vbuffer_count = MAX2(vbuffer_count, D3D9_DEFAULT_VERTEX_INPUT_BUFFER + 1);
   }

   D3D9Resource *index_resource = NULL;
   const void *cpu_indices = NULL;
   if (indexed && !user_indices) {
      index_resource = D3D9CastResource(device->index_buffer);
      if (!index_resource || !index_resource->pipe_resource)
         return E_INVALIDARG;
      D3D9SubResource *index_sub =
         D3D9GetLiveCpuBufferSubResource(index_resource);
      if (index_sub) {
         const uint64_t index_offset = (uint64_t)start_index * index_size;
         if (index_offset > index_sub->size)
            return E_INVALIDARG;
         cpu_indices = index_sub->data + index_offset;
      } else if (index_resource->surf_count &&
                 index_resource->surfaces[0].cpu_dirty &&
                 !D3D9UploadSubResource(device, index_resource, 0)) {
         return E_FAIL;
      }
   }

   const UINT instance_count = D3D9DrawInstanceCount(device);
   D3D9AutoUploadResources worker_vertex_uploads;
   if (D3D9OrderedContextWorkerEnabled() &&
       !D3D9UploadWorkerUserVertexBuffers(
          device, elements, element_count, indexed, index_bias,
          min_index, num_vertices, start_vertex, vertex_count,
          instance_count, vbuffers, vbuffer_count,
          &worker_vertex_uploads))
      return E_FAIL;

   const bool point_position_gs =
      mode == MESA_PRIM_POINTS &&
      (translated_vs_point_size || D3D9PointSize(device) > 1.0f);
   if (generate_position_t_ps_input_vs || generate_position_t_diffuse_vs ||
       generate_position_t_texture0_runtime_ps_vs ||
       generate_position_t_runtime_processed_diffuse_vs ||
       generate_position_t_runtime_processed_vs ||
       replace_position_t_runtime_ff_vs) {
      fixed_function_shaders.vs =
         (generate_position_t_ps_input_vs ||
          replace_position_t_runtime_ff_vs ||
          generate_position_t_runtime_processed_vs ||
          generate_position_t_texture0_runtime_ps_vs) ?
         D3D9CreatePositionTPsInputFixedFunctionVs(device,
                                                   &fixed_function_vs_shader) :
         D3D9CreatePositionTDiffuseFixedFunctionVs(device);
      if (!fixed_function_shaders.vs)
         return E_FAIL;
      translated_vs = fixed_function_shaders.vs;
      fixed_function_vs_shader.translated_vs_cso = translated_vs;
   }
   void *blend = D3D9CreateBlendState(device);
   const bool position_t_depth_clamp =
      D3D9ActiveVertexDeclHasPositionT(device);
   void *rasterizer = D3D9CreateRasterizerState(device,
                                                position_t_depth_clamp,
                                                point_position_gs);
   void *dsa = D3D9CreateDepthStencilState(device);
   void *point_gs = NULL;
   if (point_position_gs) {
      const float point_size = D3D9PointSize(device);
      const float width = device->viewport.Width ? (float)device->viewport.Width :
         1.0f;
      const float height = device->viewport.Height ?
         (float)device->viewport.Height : 1.0f;

      float point_size_min =
         D3D9RenderStateFloat(device->render_states[D3DRS_POINTSIZE_MIN]);
      float point_size_max =
         D3D9RenderStateFloat(device->render_states[D3DRS_POINTSIZE_MAX]);
      if (point_size_min < 0.0f)
         point_size_min = 0.0f;
      if (point_size_max <= 0.0f)
         point_size_max = 8192.0f;

      point_gs = translated_vs_point_size ?
         D3D9CreatePointPositionGs(pipe, 1.0f / width, 1.0f / height, true,
                                   translated_vs_point_size,
                                   point_size_min, point_size_max) :
         D3D9CreatePointPositionGs(pipe, point_size / width,
                                   point_size / height, false, false,
                                   point_size_min, point_size_max);
   }

   unsigned constants_size = 0;
   void *constants = D3D9CreateNineVsConstants(device, &constants_size);
   if (!blend || !rasterizer || !dsa ||
       (point_position_gs && !point_gs) ||
       (!constants && translated_vs_needs_constants)) {
      if (blend)
         pipe->delete_blend_state(pipe, blend);
      if (rasterizer)
         pipe->delete_rasterizer_state(pipe, rasterizer);
      if (dsa)
         pipe->delete_depth_stencil_alpha_state(pipe, dsa);
      if (point_gs)
         pipe->delete_gs_state(pipe, point_gs);
      free(constants);
      return E_FAIL;
   }

   struct pipe_constant_buffer cb;
   memset(&cb, 0, sizeof(cb));
   cb.user_buffer = constants;
   cb.buffer_size = constants_size;

   void *translated_texture_samplers[ARRAYSIZE(device->textures)] = {};
   struct pipe_sampler_view *translated_texture_views[ARRAYSIZE(device->textures)] = {};
   UINT translated_ps_texture_count = 0;
   UINT translated_vs_texture_count = 0;
   void *fs_constants = NULL;
   unsigned fs_constants_size = 0;
   if (translated_ps_shader && translated_ps_shader->translated_ps_const_used_size) {
      if (!fs_constants) {
         fs_constants = D3D9CreateNinePsConstants(device,
                                                  translated_ps_shader,
                                                  &fs_constants_size);
      }
      if (!fs_constants) {
         pipe->delete_depth_stencil_alpha_state(pipe, dsa);
         pipe->delete_rasterizer_state(pipe, rasterizer);
         pipe->delete_blend_state(pipe, blend);
         if (point_gs)
            pipe->delete_gs_state(pipe, point_gs);
         free(constants);
         return E_FAIL;
      }
   }

   struct pipe_constant_buffer fs_cb;
   memset(&fs_cb, 0, sizeof(fs_cb));
   fs_cb.user_buffer = fs_constants;
   fs_cb.buffer_size = fs_constants_size;

   void *dynamic_vertex_elements =
      pipe->create_vertex_elements_state(pipe, element_count, elements);
   if (!dynamic_vertex_elements) {
      pipe->delete_depth_stencil_alpha_state(pipe, dsa);
      pipe->delete_rasterizer_state(pipe, rasterizer);
      pipe->delete_blend_state(pipe, blend);
      free(fs_constants);
      if (point_gs)
         pipe->delete_gs_state(pipe, point_gs);
      free(constants);
      return E_FAIL;
   }

   struct pipe_framebuffer_state fb;
   if (!D3D9BindFramebuffer(device, &fb)) {
      pipe->delete_depth_stencil_alpha_state(pipe, dsa);
      pipe->delete_rasterizer_state(pipe, rasterizer);
      pipe->delete_blend_state(pipe, blend);
      pipe->delete_vertex_elements_state(pipe, dynamic_vertex_elements);
      free(fs_constants);
      if (point_gs)
         pipe->delete_gs_state(pipe, point_gs);
      free(constants);
      return E_INVALIDARG;
   }
   D3D9BindViewport(device);
   pipe->bind_blend_state(pipe, blend);
   D3D9SetBlendColor(device);
   pipe->bind_rasterizer_state(pipe, rasterizer);
   pipe->bind_depth_stencil_alpha_state(pipe, dsa);
   if (device->render_states[D3DRS_STENCILENABLE]) {
      struct pipe_stencil_ref stencil_ref;
      stencil_ref.ref_value[0] =
         device->render_states[D3DRS_STENCILREF] & 0xff;
      stencil_ref.ref_value[1] = stencil_ref.ref_value[0];
      pipe->set_stencil_ref(pipe, stencil_ref);
   }
   if (pipe->set_sample_mask)
      pipe->set_sample_mask(pipe,
                            device->render_states[D3DRS_MULTISAMPLEMASK]);
   pipe->bind_vs_state(pipe, translated_vs);
   pipe->bind_fs_state(pipe, translated_ps);
   if (point_position_gs)
      pipe->bind_gs_state(pipe, point_gs);
   else
      pipe->bind_gs_state(pipe, NULL);
   pipe->set_constant_buffer(pipe, MESA_SHADER_VERTEX, 0, &cb);
   pipe->set_constant_buffer(pipe, MESA_SHADER_FRAGMENT, 0, &fs_cb);
   if (!D3D9BindTranslatedTextures(device, translated_texture_samplers,
                                   translated_texture_views,
                                   &translated_ps_texture_count,
                                   &translated_vs_texture_count)) {
      memset(&fb, 0, sizeof(fb));
      if (point_position_gs)
         pipe->bind_gs_state(pipe, NULL);
      pipe->set_framebuffer_state(pipe, &fb);
      pipe->delete_depth_stencil_alpha_state(pipe, dsa);
      pipe->delete_rasterizer_state(pipe, rasterizer);
      pipe->delete_blend_state(pipe, blend);
      pipe->delete_vertex_elements_state(pipe, dynamic_vertex_elements);
      D3D9UnbindTranslatedTextures(device, translated_texture_samplers,
                                   translated_texture_views,
                                   translated_ps_texture_count,
                                   translated_vs_texture_count);
      pipe->set_constant_buffer(pipe, MESA_SHADER_FRAGMENT, 0, NULL);
      pipe->set_constant_buffer(pipe, MESA_SHADER_VERTEX, 0, NULL);
      free(fs_constants);
      if (point_gs)
         pipe->delete_gs_state(pipe, point_gs);
      free(constants);
      return E_FAIL;
   }
   pipe->bind_vertex_elements_state(pipe, dynamic_vertex_elements);
   pipe->set_vertex_buffers(pipe, vbuffer_count, vbuffers);

   if (indexed) {
      struct pipe_draw_info info;
      memset(&info, 0, sizeof(info));
      info.mode = mode;
      info.index_size = index_size;
      info.instance_count = instance_count;
      info.index_bounds_valid = true;
      info.min_index = min_index;
      info.max_index = num_vertices ? min_index + num_vertices - 1 : min_index;
      if (user_indices) {
         info.has_user_indices = true;
         info.index.user = (const uint8_t *)user_indices + start_index * index_size;
      } else if (cpu_indices) {
         info.has_user_indices = true;
         info.index.user = cpu_indices;
      } else {
         info.index.resource = index_resource->pipe_resource;
      }

      struct pipe_draw_start_count_bias draw;
      memset(&draw, 0, sizeof(draw));
      draw.start = (user_indices || cpu_indices) ? 0 : start_index;
      draw.count = vertex_count;
      draw.index_bias = index_bias;
      pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   } else {
      util_draw_arrays(pipe, mode, start_vertex, vertex_count);
   }

   pipe->set_vertex_buffers(pipe, 0, NULL);
   pipe->bind_vertex_elements_state(pipe, NULL);
   pipe->delete_vertex_elements_state(pipe, dynamic_vertex_elements);
   D3D9UnbindTranslatedTextures(device, translated_texture_samplers,
                                translated_texture_views,
                                translated_ps_texture_count,
                                translated_vs_texture_count);
   pipe->set_constant_buffer(pipe, MESA_SHADER_FRAGMENT, 0, NULL);
   pipe->set_constant_buffer(pipe, MESA_SHADER_VERTEX, 0, NULL);
   if (point_position_gs)
      pipe->bind_gs_state(pipe, NULL);
   pipe->bind_vs_state(pipe, NULL);
   pipe->bind_fs_state(pipe, NULL);
   pipe->bind_depth_stencil_alpha_state(pipe, NULL);
   pipe->bind_rasterizer_state(pipe, NULL);
   pipe->bind_blend_state(pipe, NULL);
   memset(&fb, 0, sizeof(fb));
   pipe->set_framebuffer_state(pipe, &fb);
   pipe->delete_depth_stencil_alpha_state(pipe, dsa);
   pipe->delete_rasterizer_state(pipe, rasterizer);
   pipe->delete_blend_state(pipe, blend);
   free(fs_constants);
   if (point_gs)
      pipe->delete_gs_state(pipe, point_gs);
   free(constants);

   D3D9Tracef("Draw primitive=%u count=%u start=%u offset=%d vertices=%u "
              "indexed=%u index_bias=%d index_start=%u index_size=%u rhw=%u\n",
              primitive_type, primitive_count, start_vertex,
              first_vertex_offset, vertex_count, indexed ? 1 : 0,
              index_bias, start_index, index_size, 0);
   return D3D9ConsumeBackendDrawFailure(device) ? E_FAIL : S_OK;
}

HRESULT APIENTRY
D3D9DrawPrimitive(HANDLE hDevice, const D3DDDIARG_DRAWPRIMITIVE *data,
                  const UINT *indices)
{
   (void)indices;
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("DrawPrimitive data=%p primitive=%u start=%u count=%u\n",
              data, data ? data->PrimitiveType : 0, data ? data->VStart : 0,
              data ? data->PrimitiveCount : 0);
   if (!device || !data)
      return E_INVALIDARG;
   if (!data->PrimitiveCount)
      return S_OK;

   return D3D9DrawDiffuse(device, data->PrimitiveType,
                          data->PrimitiveCount, data->VStart, 0,
                          false, 0, 0, 0, 0, 0, NULL, false);
}

HRESULT APIENTRY
D3D9DrawIndexedPrimitive(HANDLE hDevice,
                         const D3DDDIARG_DRAWINDEXEDPRIMITIVE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("DrawIndexedPrimitive data=%p primitive=%u base=%d min=%u "
              "vertices=%u start=%u count=%u index=%p stride=%u\n",
              data, data ? data->PrimitiveType : 0,
              data ? data->BaseVertexIndex : 0, data ? data->MinIndex : 0,
              data ? data->NumVertices : 0, data ? data->StartIndex : 0,
              data ? data->PrimitiveCount : 0,
              device ? device->index_buffer : NULL,
              device ? device->index_stride : 0);
   if (!device || !data)
      return E_INVALIDARG;
   if (!data->PrimitiveCount)
      return S_OK;

   return D3D9DrawDiffuse(device, data->PrimitiveType,
                          data->PrimitiveCount, 0, 0, true,
                          data->BaseVertexIndex, data->StartIndex,
                          data->MinIndex, data->NumVertices,
                          device->index_stride, device->index_sysmem, false);
}

HRESULT APIENTRY
D3D9DrawPrimitive2(HANDLE hDevice, const D3DDDIARG_DRAWPRIMITIVE2 *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("DrawPrimitive2 data=%p primitive=%u first_offset=%u count=%u\n",
              data, data ? data->PrimitiveType : 0,
              data ? data->FirstVertexOffset : 0,
              data ? data->PrimitiveCount : 0);
   if (!device || !data)
      return E_INVALIDARG;
   if (!data->PrimitiveCount)
      return S_OK;

   return D3D9DrawDiffuse(device, data->PrimitiveType,
                          data->PrimitiveCount, 0,
                          data->FirstVertexOffset, false, 0, 0, 0, 0, 0,
                          NULL, true);
}

HRESULT APIENTRY
D3D9DrawIndexedPrimitive2(HANDLE hDevice,
                          const D3DDDIARG_DRAWINDEXEDPRIMITIVE2 *data,
                          UINT index_size,
                          const void *indices,
                          const UINT *index_remap)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("DrawIndexedPrimitive2 data=%p primitive=%u base_offset=%d "
              "min=%u vertices=%u start_offset=%u count=%u index_size=%u "
              "indices=%p\n",
              data, data ? data->PrimitiveType : 0,
              data ? data->BaseVertexOffset : 0, data ? data->MinIndex : 0,
              data ? data->NumVertices : 0, data ? data->StartIndexOffset : 0,
              data ? data->PrimitiveCount : 0, index_size, indices);
   if (!device || !data)
      return E_INVALIDARG;
   if (!data->PrimitiveCount)
      return S_OK;
   if (!indices || (index_size != 2 && index_size != 4))
      return E_INVALIDARG;
   if (data->StartIndexOffset % index_size)
      return E_INVALIDARG;
   if (index_remap) {
      static volatile LONG logged;
      D3D9WarnOncef(&logged, "unsupported D3D9DrawIndexedPrimitive2 "
                    "index remap table\n");
      return E_NOTIMPL;
   }

   return D3D9DrawDiffuse(device, data->PrimitiveType,
                          data->PrimitiveCount, 0, data->BaseVertexOffset,
                          true, 0, data->StartIndexOffset / index_size,
                          data->MinIndex, data->NumVertices, index_size,
                          indices, true);
}

void
D3D9DestroyDrawState(D3D9Device *device)
{
   (void)device;
}
