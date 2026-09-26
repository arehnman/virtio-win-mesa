/*
 * SPDX-License-Identifier: MIT
 *
 * Copyright 2026 Ake Rehnman
 */
#include "D3D9Private.h"
#include "D3D9ShaderTranslate.h"

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

#include "util/u_gen_mipmap.h"

static HRESULT APIENTRY
D3D9SetTexture(HANDLE hDevice, UINT stage, HANDLE hTexture)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || stage >= ARRAYSIZE(device->textures))
      return E_INVALIDARG;
   device->textures[stage] = hTexture;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetPixelShader(HANDLE hDevice, HANDLE shader)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetPixelShader shader=%p\n", shader);
   if (!device)
      return E_INVALIDARG;
   device->pixel_shader = shader;
   return S_OK;
}

static void
D3D9TraceVertexElements(UINT count, const D3DDDIVERTEXELEMENT *elements)
{
   if (!elements)
      return;

   for (UINT i = 0; i < count; ++i) {
      D3D9Tracef("  elem[%u] stream=%u offset=%u type=%u method=%u "
                 "usage=%u usage_index=%u\n",
                 i, elements[i].Stream, elements[i].Offset,
                 elements[i].Type, elements[i].Method, elements[i].Usage,
                 elements[i].UsageIndex);
   }
}

static HRESULT APIENTRY
D3D9SetVertexShaderFunc(HANDLE hDevice, HANDLE shader)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetVertexShaderFunc shader=%p\n", shader);
   if (!device)
      return E_INVALIDARG;
   device->vertex_shader_func = shader;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetVertexShaderDecl(HANDLE hDevice, HANDLE decl)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetVertexShaderDecl decl=%p\n", decl);
   if (!device)
      return E_INVALIDARG;
   device->vertex_shader_decl = decl;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetIndicesUm(HANDLE hDevice, UINT stride, const void *indices)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device)
      return E_INVALIDARG;
   D3D9Tracef("SetIndicesUm stride=%u ptr=%p\n", stride, indices);
   device->index_buffer = NULL;
   device->index_sysmem = indices;
   device->index_stride = stride;
   return S_OK;
}

static HRESULT APIENTRY
D3D9ValidateDevice(HANDLE hDevice, D3DDDIARG_VALIDATETEXTURESTAGESTATE *data)
{
   (void)hDevice;
   if (data) {
      static volatile LONG logged;
      D3D9WarnOncef(&logged,
                    "WARNING: D3D9 ValidateDevice is approximated; "
                    "reporting one pass without full state validation\n");
      data->NumPasses = 1;
   }
   return S_OK;
}

#define D3D9_OK_STUB_1(name, type) \
static HRESULT APIENTRY \
name(HANDLE hDevice, type *arg) \
{ \
   (void)hDevice; \
   (void)arg; \
   D3D9Warnf("emulation no-op " #name "\n"); \
   return S_OK; \
}

#define D3D9_UNSUPPORTED_STUB_1(name, type) \
static HRESULT APIENTRY \
name(HANDLE hDevice, type *arg) \
{ \
   (void)hDevice; \
   (void)arg; \
   static volatile LONG logged; \
   D3D9WarnOncef(&logged, "unsupported " #name "\n"); \
   return E_NOTIMPL; \
}

static HRESULT APIENTRY
D3D9SetRenderState(HANDLE hDevice, const D3DDDIARG_RENDERSTATE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->State >= ARRAYSIZE(device->render_states))
      return E_INVALIDARG;

   /* Legacy stipple rows have no enable state in the D3D9 DDI.  The runtime
    * may still restore them as inert state-block data. */
   if (data->State >= D3DDDIRS_STIPPLEPATTERN00 &&
       data->State <= D3DDDIRS_STIPPLEPATTERN31) {
      device->render_states[data->State] = data->Value;
      return S_OK;
   }

   /* The Windows D3D9 runtime restores the unassigned DDI slots between
    * DEPTHBIAS (195) and WRAP8 (198) while creating its initial state block.
    * They have no documented rasterization meaning, but rejecting them
    * prevents device creation.  Handle them outside the enum switch because
    * they intentionally have no D3DDDIRENDERSTATETYPE enumerators. */
   if ((UINT)data->State == 196u || (UINT)data->State == 197u) {
      static volatile LONG logged;
      D3D9WarnOncef(
         &logged,
         "D3D9 runtime reserved render state owner=d3d9-state "
         "reason=unassigned-ddi-slot fallback=ignore-runtime-metadata "
         "state=%u value=0x%08x\n",
         data->State, data->Value);
      device->render_states[data->State] = data->Value;
      return S_OK;
   }

   switch (data->State) {
   case D3DDDIRS_ZENABLE:
      if (data->Value != D3DZB_FALSE && data->Value != D3DZB_TRUE &&
          data->Value != D3DZB_USEW)
         return E_INVALIDARG;
      if (data->Value == D3DZB_USEW) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 W-buffer owner=d3d9-state "
            "reason=w-depth-not-implemented action=reject\n");
         return E_NOTIMPL;
      }
      break;
   case D3DDDIRS_SHADEMODE:
      if (data->Value != D3DSHADE_FLAT &&
          data->Value != D3DSHADE_GOURAUD &&
          data->Value != D3DSHADE_PHONG)
         return E_INVALIDARG;
      /* D3D9 defines PHONG as the GOURAUD fallback when it is not
       * advertised.  The rasterizer already implements that mapping. */
      break;
   case D3DDDIRS_DITHERENABLE:
      if (data->Value > TRUE)
         return E_INVALIDARG;
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 dithering owner=d3d9-state "
            "reason=dithering-not-advertised-or-implemented "
            "fallback=draw-without-dithering\n");
      }
      break;
   case D3DDDIRS_LASTPIXEL:
      if (data->Value > TRUE)
         return E_INVALIDARG;
      break;
   case D3DDDIRS_FILLMODE:
      if (data->Value != D3DFILL_POINT &&
          data->Value != D3DFILL_WIREFRAME &&
          data->Value != D3DFILL_SOLID)
         return E_INVALIDARG;
      if (data->Value != D3DFILL_SOLID) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 fill mode "
            "owner=d3d9-state reason=polygon-mode-not-implemented "
            "action=reject mode=%u\n",
            data->Value);
         return E_NOTIMPL;
      }
      break;
   case D3DDDIRS_DEPTHBIAS:
   case D3DDDIRS_SLOPESCALEDEPTHBIAS:
      if (data->Value & 0x7fffffffu) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 depth bias owner=d3d9-state "
            "reason=d3d9-depth-bias-scaling-not-implemented "
            "fallback=ignore-state-for-runtime-cap-compatibility "
            "state=%u value=0x%08x\n",
            data->State, data->Value);
      }
      break;
   case D3DDDIRS_MULTISAMPLEANTIALIAS:
      if (data->Value > TRUE)
         return E_INVALIDARG;
      if (!data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 multisample toggle owner=d3d9-state "
            "reason=per-draw-msaa-disable-not-implemented "
            "fallback=keep-msaa-enabled\n");
      }
      break;
   case D3DDDIRS_ANTIALIASEDLINEENABLE:
      if (data->Value > TRUE)
         return E_INVALIDARG;
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 antialiased lines owner=d3d9-state "
            "reason=line-antialiasing-not-implemented "
            "fallback=draw-unantialiased\n");
      }
      break;
   case D3DDDIRS_SRCBLEND:
   case D3DDDIRS_DESTBLEND:
   case D3DDDIRS_SRCBLENDALPHA:
   case D3DDDIRS_DESTBLENDALPHA:
      if (data->Value == D3DBLEND_SRCCOLOR2 ||
          data->Value == D3DBLEND_INVSRCCOLOR2 ||
          (data->State != D3DDDIRS_SRCBLEND &&
           (data->Value == D3DBLEND_BOTHSRCALPHA ||
            data->Value == D3DBLEND_BOTHINVSRCALPHA))) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 blend factor owner=d3d9-state "
            "reason=factor-not-valid-for-state-or-dual-source-unvalidated "
            "action=reject state=%u factor=%u\n",
            data->State, data->Value);
         return E_NOTIMPL;
      }
      break;
   case D3DDDIRS_WRAP0:
   case D3DDDIRS_WRAP1:
   case D3DDDIRS_WRAP2:
   case D3DDDIRS_WRAP3:
   case D3DDDIRS_WRAP4:
   case D3DDDIRS_WRAP5:
   case D3DDDIRS_WRAP6:
   case D3DDDIRS_WRAP7:
   case D3DDDIRS_WRAP8:
   case D3DDDIRS_WRAP9:
   case D3DDDIRS_WRAP10:
   case D3DDDIRS_WRAP11:
   case D3DDDIRS_WRAP12:
   case D3DDDIRS_WRAP13:
   case D3DDDIRS_WRAP14:
   case D3DDDIRS_WRAP15:
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 coordinate wrapping owner=d3d9-state "
            "reason=cylindrical-wrap-lowering-not-implemented "
            "action=reject state=%u value=0x%08x\n",
            data->State, data->Value);
         return E_NOTIMPL;
      }
      break;
   case D3DDDIRS_PATCHEDGESTYLE:
      if (data->Value == D3DPATCHEDGE_DISCRETE)
         break;
      goto unsupported_tessellation_state;
   case D3DDDIRS_POSITIONDEGREE:
      if (data->Value == D3DDEGREE_CUBIC)
         break;
      goto unsupported_tessellation_state;
   case D3DDDIRS_NORMALDEGREE:
      if (data->Value == D3DDEGREE_LINEAR)
         break;
      goto unsupported_tessellation_state;
   case D3DDDIRS_MINTESSELLATIONLEVEL:
   case D3DDDIRS_MAXTESSELLATIONLEVEL:
   case D3DDDIRS_ADAPTIVETESS_Z:
   case D3DDDIRS_PATCHSEGMENTS:
      if (data->Value == 0x3f800000u)
         break;
      goto unsupported_tessellation_state;
   case D3DDDIRS_ADAPTIVETESS_X:
   case D3DDDIRS_ADAPTIVETESS_Y:
   case D3DDDIRS_ADAPTIVETESS_W:
   case D3DDDIRS_ENABLEADAPTIVETESSELLATION:
   case D3DDDIRS_DELETERTPATCH:
      if (!data->Value)
         break;
unsupported_tessellation_state:
      {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 tessellation state owner=d3d9-state "
            "reason=patch-tessellation-not-implemented "
            "fallback=ignore-unadvertised-state state=%u value=0x%08x\n",
            data->State, data->Value);
         break;
      }
   case D3DDDIRS_COLORWRITEENABLE:
   case D3DDDIRS_COLORWRITEENABLE1:
   case D3DDDIRS_COLORWRITEENABLE2:
   case D3DDDIRS_COLORWRITEENABLE3:
      if (data->Value & ~(D3DCOLORWRITEENABLE_RED |
                          D3DCOLORWRITEENABLE_GREEN |
                          D3DCOLORWRITEENABLE_BLUE |
                          D3DCOLORWRITEENABLE_ALPHA))
         return E_INVALIDARG;
      break;
   /* Consumed directly by the D3D9 draw, shader, or framebuffer path. */
   case D3DDDIRS_ZWRITEENABLE:
   case D3DDDIRS_ALPHATESTENABLE:
   case D3DDDIRS_CULLMODE:
   case D3DDDIRS_ZFUNC:
   case D3DDDIRS_ALPHAREF:
   case D3DDDIRS_ALPHAFUNC:
   case D3DDDIRS_ALPHABLENDENABLE:
   case D3DDDIRS_FOGENABLE:
   case D3DDDIRS_FOGCOLOR:
   case D3DDDIRS_FOGTABLEMODE:
   case D3DDDIRS_FOGSTART:
   case D3DDDIRS_FOGEND:
   case D3DDDIRS_FOGDENSITY:
   case D3DDDIRS_STENCILENABLE:
   case D3DDDIRS_STENCILFAIL:
   case D3DDDIRS_STENCILZFAIL:
   case D3DDDIRS_STENCILPASS:
   case D3DDDIRS_STENCILFUNC:
   case D3DDDIRS_STENCILREF:
   case D3DDDIRS_STENCILMASK:
   case D3DDDIRS_STENCILWRITEMASK:
   case D3DDDIRS_CLIPPLANEENABLE:
   case D3DDDIRS_POINTSIZE:
   case D3DDDIRS_POINTSIZE_MIN:
   case D3DDDIRS_POINTSPRITEENABLE:
   case D3DDDIRS_POINTSIZE_MAX:
   case D3DDDIRS_MULTISAMPLEMASK:
   case D3DDDIRS_BLENDOP:
   case D3DDDIRS_SCISSORTESTENABLE:
   case D3DDDIRS_TWOSIDEDSTENCILMODE:
   case D3DDDIRS_CCW_STENCILFAIL:
   case D3DDDIRS_CCW_STENCILZFAIL:
   case D3DDDIRS_CCW_STENCILPASS:
   case D3DDDIRS_CCW_STENCILFUNC:
   case D3DDDIRS_BLENDFACTOR:
   case D3DDDIRS_SRGBWRITEENABLE:
   case D3DDDIRS_SEPARATEALPHABLENDENABLE:
   case D3DDDIRS_BLENDOPALPHA:
      break;
   /* Windows' tagged runtime fixed-function shaders own these states. */
   case D3DDDIRS_SPECULARENABLE:
   case D3DDDIRS_TEXTUREFACTOR:
   case D3DDDIRS_CLIPPING:
   case D3DDDIRS_LIGHTING:
   case D3DDDIRS_AMBIENT:
   case D3DDDIRS_FOGVERTEXMODE:
   case D3DDDIRS_RANGEFOGENABLE:
   case D3DDDIRS_COLORVERTEX:
   case D3DDDIRS_LOCALVIEWER:
   case D3DDDIRS_NORMALIZENORMALS:
   case D3DDDIRS_DIFFUSEMATERIALSOURCE:
   case D3DDDIRS_SPECULARMATERIALSOURCE:
   case D3DDDIRS_AMBIENTMATERIALSOURCE:
   case D3DDDIRS_EMISSIVEMATERIALSOURCE:
   case D3DDDIRS_VERTEXBLEND:
   case D3DDDIRS_INDEXEDVERTEXBLENDENABLE:
   case D3DDDIRS_TWEENFACTOR:
   case D3DDDIRS_POINTSCALEENABLE:
   case D3DDDIRS_POINTSCALE_A:
   case D3DDDIRS_POINTSCALE_B:
   case D3DDDIRS_POINTSCALE_C:
      break;
   /* Unsupported legacy raster effects must remain inert. */
   case D3DDDIRS_LINEPATTERN:
   case D3DDDIRS_ZVISIBLE:
   case D3DDDIRS_EDGEANTIALIAS:
   case D3DDDIRS_COLORKEYENABLE:
   case D3DDDIRS_OLDALPHABLENDENABLE:
   case D3DDDIRS_ZBIAS:
   case D3DDDIRS_TRANSLUCENTSORTINDEPENDENT:
   case D3DDDIRS_COLORKEYBLENDENABLE:
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 legacy raster state owner=d3d9-state "
            "reason=legacy-effect-not-implemented "
            "fallback=ignore-unadvertised-state state=%u value=0x%08x\n",
            data->State, data->Value);
      }
      break;
   /* Runtime/debug metadata with no direct rasterization effect here. */
   case D3DDDIRS_SCENECAPTURE:
   case D3DDDIRS_SOFTWAREVERTEXPROCESSING:
   case D3DDDIRS_DEBUGMONITORTOKEN:
      break;
   default:
      {
         static volatile LONG logged[256];
         D3D9WarnOncef(
            &logged[data->State],
            "unsupported D3D9 render state owner=d3d9-state "
            "reason=state-not-classified-or-implemented "
            "action=reject state=%u value=0x%08x\n",
            data->State, data->Value);
         return E_NOTIMPL;
      }
   }

   device->render_states[data->State] = data->Value;
   return S_OK;
}

static HRESULT APIENTRY
D3D9UpdateWInfo(HANDLE hDevice, const D3DDDIARG_WINFO *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data)
      return E_INVALIDARG;
   device->winfo = *data;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetScissorRect(HANDLE hDevice, const RECT *rect)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !rect || rect->right < rect->left ||
       rect->bottom < rect->top)
      return E_INVALIDARG;

   device->scissor_rect = *rect;

   if (!device->pipe || !device->pipe->set_scissor_states) {
      D3D9Warnf("unsupported D3D9SetScissorRect without pipe scissor hook\n");
      return E_NOTIMPL;
   }

   struct pipe_scissor_state state;
   state.minx = rect->left < 0 ? 0 : rect->left;
   state.miny = rect->top < 0 ? 0 : rect->top;
   state.maxx = rect->right < 0 ? 0 : rect->right;
   state.maxy = rect->bottom < 0 ? 0 : rect->bottom;
   device->pipe->set_scissor_states(device->pipe, 0, 1, &state);
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetTextureStageState(HANDLE hDevice,
                         const D3DDDIARG_TEXTURESTAGESTATE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->Stage >= ARRAYSIZE(device->texture_stage_states) ||
       data->State >= ARRAYSIZE(device->texture_stage_states[0]))
      return E_INVALIDARG;
   switch (data->State) {
   /* The Windows runtime owns fixed-function texture combining.  It passes
    * these values alongside the generated fixed-function shaders. */
   case D3DDDITSS_COLOROP:
   case D3DDDITSS_COLORARG1:
   case D3DDDITSS_COLORARG2:
   case D3DDDITSS_ALPHAOP:
   case D3DDDITSS_ALPHAARG1:
   case D3DDDITSS_ALPHAARG2:
   case D3DDDITSS_TEXCOORDINDEX:
   case D3DDDITSS_COLORARG0:
   case D3DDDITSS_ALPHAARG0:
   case D3DDDITSS_RESULTARG:
   case D3DDDITSS_CONSTANT:
      break;
   case D3DDDITSS_TEXTUREMAP:
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 texture-map state owner=d3d9-state "
            "reason=nonzero-texture-map-state-not-implemented "
            "action=reject stage=%u value=0x%08x\n",
            data->Stage, data->Value);
         return E_NOTIMPL;
      }
      break;
   /* These values are consumed directly by sampler setup, projected-texture
    * handling, or D3D9CreateNinePsConstants(). */
   case D3DDDITSS_BUMPENVMAT00:
   case D3DDDITSS_BUMPENVMAT01:
   case D3DDDITSS_BUMPENVMAT10:
   case D3DDDITSS_BUMPENVMAT11:
   case D3DDDITSS_MIPMAPLODBIAS:
   case D3DDDITSS_MAXMIPLEVEL:
   case D3DDDITSS_BUMPENVLSCALE:
   case D3DDDITSS_BUMPENVLOFFSET:
      break;
   case D3DDDITSS_BORDERCOLOR:
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "inactive D3D9 sampler border color owner=d3d9-state "
            "reason=custom-border-color-not-implemented "
            "compatibility=retain-metadata value=0x%08x\n",
            data->Value);
      }
      break;
   case D3DDDITSS_ADDRESSU:
   case D3DDDITSS_ADDRESSV:
   case D3DDDITSS_ADDRESSW:
      if (data->Value < D3DTADDRESS_WRAP ||
          data->Value > D3DTADDRESS_MIRRORONCE)
         return E_INVALIDARG;
      if (data->Value == D3DTADDRESS_BORDER ||
          data->Value == D3DTADDRESS_MIRRORONCE) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "approximated D3D9 texture address mode owner=d3d9-state "
            "reason=custom-border-or-mirror-clamp-not-enabled "
            "fallback=transparent-black-border-or-clamp-to-edge "
            "stage=%u state=%u mode=%u\n",
            data->Stage, data->State, data->Value);
      }
      break;
   case D3DDDITSS_MINFILTER:
   case D3DDDITSS_MAGFILTER:
      if (data->Value != D3DTEXF_POINT &&
          data->Value != D3DTEXF_LINEAR &&
          data->Value != D3DTEXF_ANISOTROPIC) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 texture filter owner=d3d9-state "
            "reason=filter-not-advertised-or-implemented "
            "action=reject stage=%u state=%u filter=%u\n",
            data->Stage, data->State, data->Value);
         return E_NOTIMPL;
      }
      break;
   case D3DDDITSS_MIPFILTER:
      if (data->Value != D3DTEXF_NONE &&
          data->Value != D3DTEXF_POINT &&
          data->Value != D3DTEXF_LINEAR)
         return E_INVALIDARG;
      break;
   case D3DDDITSS_MAXANISOTROPY:
      if (!data->Value || data->Value > 16)
         return E_INVALIDARG;
      break;
   case D3DDDITSS_SRGBTEXTURE:
      /* BOOL-valued D3D9 sampler states retain the caller's exact DWORD.
       * Native drivers accept arbitrary nonzero values and interpret them as
       * TRUE when creating sampler state. */
      break;
   case D3DDDITSS_TEXTURETRANSFORMFLAGS:
      /* The runtime's fixed-function converter owns these flags and native
       * drivers retain even non-canonical values.  The low count bits below
       * are only a hint for Yttrium's generated constant layout. */
      break;
   case D3DDDITSS_ELEMENTINDEX:
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "inactive D3D9 texture element state owner=d3d9-state "
            "reason=multi-element-texture-not-advertised-or-implemented "
            "compatibility=retain-metadata stage=%u state=%u value=0x%08x\n",
            data->Stage, data->State, data->Value);
      }
      break;
   case D3DDDITSS_DMAPOFFSET:
      if (data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "inactive D3D9 displacement-map state owner=d3d9-state "
            "reason=presampled-displacement-map-not-advertised-or-implemented "
            "compatibility=retain-metadata stage=%u state=%u value=0x%08x\n",
            data->Stage, data->State, data->Value);
      }
      break;
   case D3DDDITSS_DISABLETEXTURECOLORKEY:
      if (data->Value > TRUE)
         return E_INVALIDARG;
      if (!data->Value) {
         static volatile LONG logged;
         D3D9WarnOncef(
            &logged,
            "unsupported D3D9 texture color key owner=d3d9-state "
            "reason=texture-color-keying-not-implemented "
            "action=reject-enable stage=%u\n",
            data->Stage);
         return E_NOTIMPL;
      }
      break;
   case D3DDDITSS_TEXTURECOLORKEYVAL:
      /* Metadata only.  DISABLETEXTURECOLORKEY is initialized to TRUE and
       * attempts to enable the unsupported operation are rejected above. */
      break;
   default:
      return E_INVALIDARG;
   }
   if (data->Stage == 0 && data->State == D3DDDITSS_TEXTURETRANSFORMFLAGS) {
      const UINT count = data->Value & 0xff;
      if (count >= D3DTTFF_COUNT1 && count <= D3DTTFF_COUNT4) {
         device->texture_transform_count = count;
         device->texture_transform_count_pending = TRUE;
         device->texture_transform_count_consumed = FALSE;
      } else if (!data->Value) {
         device->texture_transform_count = 0;
         device->texture_transform_count_pending = FALSE;
         device->texture_transform_count_consumed = FALSE;
      }
   }
   device->texture_stage_states[data->Stage][data->State] = data->Value;
   return S_OK;
}

static HRESULT APIENTRY
D3D9MultiplyTransform(HANDLE hDevice, const D3DDDIARG_MULTIPLYTRANSFORM *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->TransformType >= ARRAYSIZE(device->transforms))
      return E_INVALIDARG;
   D3D9MatrixMultiply(&device->transforms[data->TransformType],
                      &device->transforms[data->TransformType],
                      &data->Matrix);
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetTransform(HANDLE hDevice, const D3DDDIARG_SETTRANSFORM *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->TransformType >= ARRAYSIZE(device->transforms))
      return E_INVALIDARG;
   device->transforms[data->TransformType] = data->Matrix;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetViewport(HANDLE hDevice, const D3DDDIARG_VIEWPORTINFO *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetViewport x=%u y=%u w=%u h=%u\n",
              data ? data->X : 0, data ? data->Y : 0,
              data ? data->Width : 0, data ? data->Height : 0);
   if (!device || !data)
      return E_INVALIDARG;
   device->viewport = *data;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetZRange(HANDLE hDevice, const D3DDDIARG_ZRANGE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetZRange min=%f max=%f\n",
              data ? data->MinZ : 0.0f, data ? data->MaxZ : 0.0f);
   if (!device || !data)
      return E_INVALIDARG;
   if (data->MinZ > data->MaxZ)
      return E_INVALIDARG;
   device->zrange = *data;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetMaterial(HANDLE hDevice, const D3DDDIARG_SETMATERIAL *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data)
      return E_INVALIDARG;
   device->material = *data;
   return S_OK;
}

static HRESULT APIENTRY
D3D9CreateLight(HANDLE hDevice, const D3DDDIARG_CREATELIGHT *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->Index >= ARRAYSIZE(device->lights))
      return E_INVALIDARG;
   memset(&device->lights[data->Index], 0, sizeof(device->lights[data->Index]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9DestroyLight(HANDLE hDevice, const D3DDDIARG_DESTROYLIGHT *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->Index >= ARRAYSIZE(device->lights))
      return E_INVALIDARG;
   memset(&device->lights[data->Index], 0, sizeof(device->lights[data->Index]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetClipPlane(HANDLE hDevice, const D3DDDIARG_SETCLIPPLANE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->Index >= ARRAYSIZE(device->clip_planes))
      return E_INVALIDARG;
   memcpy(device->clip_planes[data->Index], data->Plane,
          sizeof(device->clip_planes[data->Index]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetPalette(HANDLE hDevice, const D3DDDIARG_SETPALETTE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->PaletteHandle >= ARRAYSIZE(device->palettes))
      return E_INVALIDARG;
   device->palettes[data->PaletteHandle].flags = data->PaletteFlags;
   device->palettes[data->PaletteHandle].resource = data->hResource;
   return S_OK;
}

static UINT
D3D9MipLayerCount(const struct pipe_resource *resource)
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

static unsigned
D3D9GenerateMipFilter(UINT filter)
{
   return filter == D3DDDITEXF_POINT ? PIPE_TEX_FILTER_NEAREST :
      PIPE_TEX_FILTER_LINEAR;
}

static HRESULT APIENTRY
D3D9GenerateMipSubLevels(HANDLE hDevice,
                         const D3DDDIARG_GENERATEMIPSUBLEVELS *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Resource *resource = data ? D3D9CastResource(data->hResource) : NULL;
   if (!device || !device->pipe || !data || !resource ||
       !resource->pipe_resource)
      return E_INVALIDARG;

   struct pipe_resource *pipe_resource = resource->pipe_resource;
   if (pipe_resource->target == PIPE_BUFFER || !pipe_resource->last_level)
      return S_OK;
   if (!D3D9UploadSubResource(device, resource, 0))
      return E_FAIL;

   const UINT layer_count = D3D9MipLayerCount(pipe_resource);
   if (!layer_count)
      return E_INVALIDARG;

   if (!util_gen_mipmap(device->pipe, pipe_resource, pipe_resource->format,
                        0, pipe_resource->last_level, 0, layer_count - 1,
                        D3D9GenerateMipFilter(data->Filter))) {
      static volatile LONG logged;
      D3D9WarnOncef(&logged, "unsupported D3D9GenerateMipSubLevels "
                    "resource=%p target=%u format=%u levels=%u\n",
                    resource, pipe_resource->target, pipe_resource->format,
                    pipe_resource->last_level + 1);
      return E_FAIL;
   }

   resource->autogen_mipmap_dirty = false;
   return S_OK;
}

D3D9_UNSUPPORTED_STUB_1(D3D9StateSet, D3DDDIARG_STATESET)
D3D9_UNSUPPORTED_STUB_1(D3D9SetConvolutionKernelMono, const D3DDDIARG_SETCONVOLUTIONKERNELMONO)
D3D9_UNSUPPORTED_STUB_1(D3D9ComposeRects, const D3DDDIARG_COMPOSERECTS)

static HRESULT APIENTRY
D3D9CreateVideoProcessor(HANDLE hDevice,
                         D3DDDIARG_DXVAHD_CREATEVIDEOPROCESSOR *data)
{
   (void)hDevice;
   (void)data;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9CreateVideoProcessor\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9SetVideoProcessBltState(
   HANDLE hDevice, const D3DDDIARG_DXVAHD_SETVIDEOPROCESSBLTSTATE *data)
{
   (void)hDevice;
   (void)data;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9SetVideoProcessBltState\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9GetVideoProcessBltStatePrivate(
   HANDLE hDevice, D3DDDIARG_DXVAHD_GETVIDEOPROCESSBLTSTATEPRIVATE *data)
{
   (void)hDevice;
   (void)data;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9GetVideoProcessBltStatePrivate\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9SetVideoProcessStreamState(
   HANDLE hDevice, const D3DDDIARG_DXVAHD_SETVIDEOPROCESSSTREAMSTATE *data)
{
   (void)hDevice;
   (void)data;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9SetVideoProcessStreamState\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9GetVideoProcessStreamStatePrivate(
   HANDLE hDevice, D3DDDIARG_DXVAHD_GETVIDEOPROCESSSTREAMSTATEPRIVATE *data)
{
   (void)hDevice;
   (void)data;
   static volatile LONG logged;
   D3D9WarnOncef(&logged,
                 "unsupported D3D9GetVideoProcessStreamStatePrivate\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9VideoProcessBltHD(HANDLE hDevice,
                      const D3DDDIARG_DXVAHD_VIDEOPROCESSBLTHD *data)
{
   (void)hDevice;
   (void)data;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9VideoProcessBltHD\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9DestroyVideoProcessor(HANDLE hDevice, HANDLE videoProcessor)
{
   (void)hDevice;
   (void)videoProcessor;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9DestroyVideoProcessor\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9SetDisplayMode(HANDLE hDevice, const D3DDDIARG_SETDISPLAYMODE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetDisplayMode resource=%p sub=%u\n",
              data ? data->hResource : NULL,
              data ? data->SubResourceIndex : 0);

   if (!device || !data || !data->hResource)
      return E_INVALIDARG;

   if (!device->callbacks.pfnSetDisplayModeCb) {
      D3D9Warnf("unsupported D3D9SetDisplayMode without runtime callback\n");
      return E_NOTIMPL;
   }

   D3D9SubResource *sub =
      D3D9GetSubResource(data->hResource, data->SubResourceIndex);
   if (!sub || !sub->allocation) {
      D3D9Warnf("unsupported D3D9SetDisplayMode without primary allocation "
                "resource=%p sub=%u\n",
                data->hResource, data->SubResourceIndex);
      return E_INVALIDARG;
   }

   HRESULT hr = device->gdi_device.base.setDisplayMode(&device->gdi_device.base,
                                                       sub->allocation);
   D3D9Tracef("SetDisplayMode allocation=0x%lx hr=0x%08lx\n",
              (unsigned long)sub->allocation, hr);
   return hr;
}

static const char *
D3D9QueryTypeName(D3DDDIQUERYTYPE type)
{
   switch (type) {
   case D3DDDIQUERYTYPE_EVENT:
      return "EVENT";
   case D3DDDIQUERYTYPE_OCCLUSION:
      return "OCCLUSION";
   default:
      return "UNKNOWN";
   }
}

static HRESULT APIENTRY
D3D9CreateQuery(HANDLE hDevice, D3DDDIARG_CREATEQUERY *data)
{
   (void)hDevice;
   if (!data)
      return E_INVALIDARG;
   if (data->QueryType != D3DDDIQUERYTYPE_EVENT &&
       data->QueryType != D3DDDIQUERYTYPE_OCCLUSION)
      return E_NOTIMPL;

   D3D9Warnf("WARNING: D3D9 %s query is emulated type=%u\n",
             D3D9QueryTypeName(data->QueryType), data->QueryType);
   D3D9Object *query = D3D9CreateObject(D3D9_OBJECT_QUERY, 0, NULL);
   if (!query)
      return E_OUTOFMEMORY;

   query->query_type = data->QueryType;
   data->hQuery = query;
   return S_OK;
}

static HRESULT APIENTRY
D3D9IssueQuery(HANDLE hDevice, const D3DDDIARG_ISSUEQUERY *data)
{
   (void)hDevice;
   if (!data || !data->hQuery)
      return E_INVALIDARG;

   D3D9Object *query = (D3D9Object *)data->hQuery;
   if (query->kind != D3D9_OBJECT_QUERY)
      return E_INVALIDARG;
   D3D9Warnf("WARNING: D3D9 %s query issue is emulated type=%u "
             "flags=0x%08x\n", D3D9QueryTypeName(query->query_type),
             query->query_type, data->Flags.Value);
   query->query_flags = data->Flags;
   return S_OK;
}

static HRESULT APIENTRY
D3D9GetQueryData(HANDLE hDevice, const D3DDDIARG_GETQUERYDATA *data)
{
   if (!data || !data->hQuery)
      return E_INVALIDARG;

   D3D9Object *query = (D3D9Object *)data->hQuery;
   if (query->kind != D3D9_OBJECT_QUERY)
      return E_INVALIDARG;

   D3D9Warnf("WARNING: D3D9 %s query data is emulated type=%u "
             "flags=0x%08x\n", D3D9QueryTypeName(query->query_type),
             query->query_type, query->query_flags.Value);
   if (data->pData) {
      if (query->query_type == D3DDDIQUERYTYPE_EVENT)
         *(BOOL *)data->pData = TRUE;
      else if (query->query_type == D3DDDIQUERYTYPE_OCCLUSION)
         *(DWORD *)data->pData = 0;
   }

   return S_OK;
}

static HRESULT APIENTRY
D3D9SetIndices(HANDLE hDevice, const D3DDDIARG_SETINDICES *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetIndices index=%p stride=%u\n",
              data ? data->hIndexBuffer : NULL, data ? data->Stride : 0);
   if (!device || !data)
      return E_INVALIDARG;

   device->index_buffer = data->hIndexBuffer;
   device->index_sysmem = NULL;
   device->index_stride = data->Stride;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetStreamSource(HANDLE hDevice, const D3DDDIARG_SETSTREAMSOURCE *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("SetStreamSource stream=%u vb=%p offset=%u stride=%u\n",
              data ? data->Stream : 0, data ? data->hVertexBuffer : NULL,
              data ? data->Offset : 0, data ? data->Stride : 0);
   if (!device || !data || data->Stream >= ARRAYSIZE(device->streams))
      return E_INVALIDARG;

   device->streams[data->Stream].vertex_buffer = data->hVertexBuffer;
   device->streams[data->Stream].sysmem = NULL;
   device->streams[data->Stream].offset = data->Offset;
   device->streams[data->Stream].stride = data->Stride;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetStreamSourceFreq(HANDLE hDevice,
                        const D3DDDIARG_SETSTREAMSOURCEFREQ *data)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->Stream >= ARRAYSIZE(device->streams))
      return E_INVALIDARG;

   device->streams[data->Stream].divider = data->Divider;
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetVertexShaderConst(HANDLE hDevice,
                         const D3DDDIARG_SETVERTEXSHADERCONST *data,
                         const void *values)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data ||
       !D3D9CheckConstantRange(data->Register, data->Count,
                               ARRAYSIZE(device->vs_float_constants)) ||
       (data->Count && !values))
      return E_INVALIDARG;
   if (data->Count)
      memcpy(&device->vs_float_constants[data->Register][0], values,
             (size_t)data->Count * sizeof(device->vs_float_constants[0]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetStreamSourceUm(HANDLE hDevice,
                      const D3DDDIARG_SETSTREAMSOURCEUM *data,
                      const void *sysmem)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->Stream >= ARRAYSIZE(device->streams))
      return E_INVALIDARG;

   D3D9Tracef("SetStreamSourceUm stream=%u stride=%u ptr=%p\n",
              data->Stream, data->Stride, sysmem);
   device->streams[data->Stream].vertex_buffer = NULL;
   device->streams[data->Stream].sysmem = sysmem;
   device->streams[data->Stream].offset = 0;
   device->streams[data->Stream].stride = data->Stride;
   return S_OK;
}

static HRESULT APIENTRY
D3D9CreateVertexShaderDecl(HANDLE hDevice,
                           D3DDDIARG_CREATEVERTEXSHADERDECL *data,
                           const D3DDDIVERTEXELEMENT *elements)
{
   (void)hDevice;
   D3D9Tracef("CreateVertexShaderDecl data=%p elements=%p count=%u\n",
              data, elements, data ? data->NumVertexElements : 0);
   if (!data || (data->NumVertexElements && !elements))
      return E_INVALIDARG;
   D3D9TraceVertexElements(data->NumVertexElements, elements);

   const D3DDDIVERTEXELEMENT decl_end = {
      0xff, 0, D3DDECLTYPE_UNUSED, 0, 0, 0
   };
   const size_t element_count = (size_t)data->NumVertexElements + 1;
   D3DDDIVERTEXELEMENT *terminated_elements = NULL;
   const D3DDDIVERTEXELEMENT *stored_elements = elements;

   terminated_elements =
      (D3DDDIVERTEXELEMENT *)calloc(data->NumVertexElements + 1,
                                    sizeof(*terminated_elements));
   if (!terminated_elements)
      return E_OUTOFMEMORY;
   if (data->NumVertexElements)
      memcpy(terminated_elements, elements,
             (size_t)data->NumVertexElements * sizeof(*elements));
   terminated_elements[data->NumVertexElements] = decl_end;
   stored_elements = terminated_elements;

   D3D9Object *decl =
      D3D9CreateObject(D3D9_OBJECT_VERTEX_DECL,
                       element_count * sizeof(*stored_elements),
                       stored_elements);
   free(terminated_elements);
   if (!decl)
      return E_OUTOFMEMORY;

   data->ShaderHandle = decl;
   D3D9Tracef("CreateVertexShaderDecl handle=%p\n", decl);
   return S_OK;
}

static HRESULT APIENTRY
D3D9CreateVertexShaderFunc(HANDLE hDevice,
                           D3DDDIARG_CREATEVERTEXSHADERFUNC *data,
                           const UINT *tokens)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("CreateVertexShaderFunc data=%p tokens=%p size=%u\n",
              data, tokens, data ? data->Size : 0);
   if (!device || !data || (data->Size && !tokens))
      return E_INVALIDARG;

   D3D9Object *shader = D3D9CreateObject(D3D9_OBJECT_SHADER, data->Size,
                                         tokens);
   if (!shader)
      return E_OUTOFMEMORY;

   HRESULT translate_hr = D3D9TranslateVertexShader(device, shader);
   if (FAILED(translate_hr)) {
      D3D9Warnf("failing D3D9 vertex shader creation because Nine "
                "translation failed shader=%p hr=0x%08lx\n",
                shader, translate_hr);
      free(shader);
      return translate_hr;
   }

   data->ShaderHandle = shader;
   D3D9Tracef("CreateVertexShaderFunc handle=%p\n", shader);
   return S_OK;
}

static HRESULT APIENTRY
D3D9CreatePixelShader(HANDLE hDevice, D3DDDIARG_CREATEPIXELSHADER *data,
                      const UINT *tokens)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("CreatePixelShader data=%p tokens=%p size=%u\n",
              data, tokens, data ? data->CodeSize : 0);
   if (!device || !data || (data->CodeSize && !tokens))
      return E_INVALIDARG;

   D3D9Object *shader = D3D9CreateObject(D3D9_OBJECT_SHADER, data->CodeSize,
                                         tokens);
   if (!shader)
      return E_OUTOFMEMORY;

   HRESULT translate_hr = D3D9TranslatePixelShader(device, shader);
   if (FAILED(translate_hr)) {
      D3D9Warnf("failing D3D9 pixel shader creation because Nine "
                "translation failed shader=%p hr=0x%08lx\n",
                shader, translate_hr);
      free(shader);
      return translate_hr;
   }

   data->ShaderHandle = shader;
   D3D9Tracef("CreatePixelShader handle=%p\n", shader);
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetPixelShaderConst(HANDLE hDevice,
                        const D3DDDIARG_SETPIXELSHADERCONST *data,
                        const FLOAT *values)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data ||
       !D3D9CheckConstantRange(data->Register, data->Count,
                               ARRAYSIZE(device->ps_float_constants)) ||
       (data->Count && !values))
      return E_INVALIDARG;
   if (data->Count)
      memcpy(&device->ps_float_constants[data->Register][0], values,
             (size_t)data->Count * sizeof(device->ps_float_constants[0]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetVertexShaderConstI(HANDLE hDevice,
                          const D3DDDIARG_SETVERTEXSHADERCONSTI *data,
                          const INT *values)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data ||
       !D3D9CheckConstantRange(data->Register, data->Count,
                               ARRAYSIZE(device->vs_int_constants)) ||
       (data->Count && !values))
      return E_INVALIDARG;
   if (data->Count)
      memcpy(&device->vs_int_constants[data->Register][0], values,
             (size_t)data->Count * sizeof(device->vs_int_constants[0]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetVertexShaderConstB(HANDLE hDevice,
                          const D3DDDIARG_SETVERTEXSHADERCONSTB *data,
                          const BOOL *values)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data ||
       !D3D9CheckConstantRange(data->Register, data->Count,
                               ARRAYSIZE(device->vs_bool_constants)) ||
       (data->Count && !values))
      return E_INVALIDARG;
   if (data->Count)
      memcpy(&device->vs_bool_constants[data->Register], values,
             (size_t)data->Count * sizeof(device->vs_bool_constants[0]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetPixelShaderConstI(HANDLE hDevice,
                         const D3DDDIARG_SETPIXELSHADERCONSTI *data,
                         const INT *values)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data ||
       !D3D9CheckConstantRange(data->Register, data->Count,
                               ARRAYSIZE(device->ps_int_constants)) ||
       (data->Count && !values))
      return E_INVALIDARG;
   if (data->Count)
      memcpy(&device->ps_int_constants[data->Register][0], values,
             (size_t)data->Count * sizeof(device->ps_int_constants[0]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetPixelShaderConstB(HANDLE hDevice,
                         const D3DDDIARG_SETPIXELSHADERCONSTB *data,
                         const BOOL *values)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data ||
       !D3D9CheckConstantRange(data->Register, data->Count,
                               ARRAYSIZE(device->ps_bool_constants)) ||
       (data->Count && !values))
      return E_INVALIDARG;
   if (data->Count)
      memcpy(&device->ps_bool_constants[data->Register], values,
             (size_t)data->Count * sizeof(device->ps_bool_constants[0]));
   return S_OK;
}

static HRESULT APIENTRY
D3D9SetLight(HANDLE hDevice, const D3DDDIARG_SETLIGHT *data,
             const D3DDDI_LIGHT *light)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || data->Index >= ARRAYSIZE(device->lights))
      return E_INVALIDARG;

   if (data->DataType == D3DDDI_SETLIGHT_DATA) {
      if (!light)
         return E_INVALIDARG;
      device->lights[data->Index].data = *light;
      device->lights[data->Index].enabled = TRUE;
      memset(&device->vs_float_constants[10][0], 0,
             4 * sizeof(device->vs_float_constants[0]));
   } else if (data->DataType == D3DDDI_SETLIGHT_ENABLE) {
      device->lights[data->Index].enabled = TRUE;
   } else if (data->DataType == D3DDDI_SETLIGHT_DISABLE) {
      device->lights[data->Index].enabled = FALSE;
   } else {
      return E_INVALIDARG;
   }
   return S_OK;
}

static HRESULT APIENTRY
D3D9GetInfo(HANDLE hDevice, UINT type, void *data, UINT data_size)
{
   if (!hDevice)
      return E_INVALIDARG;

   switch (type) {
   case D3DDDIDEVINFOID_VCACHE:
      if (!data || data_size != sizeof(D3DDDIDEVINFO_VCACHE))
         return E_INVALIDARG;
      ((D3DDDIDEVINFO_VCACHE *)data)->Pattern =
         (UINT)D3D9FourCC('C', 'A', 'C', 'H');
      ((D3DDDIDEVINFO_VCACHE *)data)->OptMethod = 0;
      ((D3DDDIDEVINFO_VCACHE *)data)->CacheSize = 0;
      ((D3DDDIDEVINFO_VCACHE *)data)->MagicNumber = 0;
      return S_OK;
   default:
      D3D9Warnf("unsupported D3D9GetInfo type=%u size=%u\n",
                type, data_size);
      return E_NOTIMPL;
   }
}

static HRESULT APIENTRY
D3D9DrawRectPatch(HANDLE hDevice, const D3DDDIARG_DRAWRECTPATCH *data,
                  const D3DDDIRECTPATCH_INFO *info, const FLOAT *segments)
{
   (void)hDevice;
   (void)data;
   (void)info;
   (void)segments;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9DrawRectPatch\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9DrawTriPatch(HANDLE hDevice, const D3DDDIARG_DRAWTRIPATCH *data,
                 const D3DDDITRIPATCH_INFO *info, const FLOAT *segments)
{
   (void)hDevice;
   (void)data;
   (void)info;
   (void)segments;
   static volatile LONG logged;
   D3D9WarnOncef(&logged, "unsupported D3D9DrawTriPatch\n");
   return E_NOTIMPL;
}

static HRESULT APIENTRY
D3D9UpdatePalette(HANDLE hDevice, const D3DDDIARG_UPDATEPALETTE *data,
                  const PALETTEENTRY *entries)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device || !data || (data->NumEntries && !entries) ||
       data->PaletteHandle >= ARRAYSIZE(device->palettes) ||
       data->StartIndex >= ARRAYSIZE(device->palettes[0].entries) ||
       data->NumEntries > ARRAYSIZE(device->palettes[0].entries) -
          data->StartIndex)
      return E_INVALIDARG;

   if (data->NumEntries)
      memcpy(&device->palettes[data->PaletteHandle].entries[data->StartIndex],
             entries,
             (size_t)data->NumEntries *
                sizeof(device->palettes[data->PaletteHandle].entries[0]));
   return S_OK;
}


static HRESULT APIENTRY
D3D9DestroyQuery(HANDLE hDevice, const HANDLE query)
{
   (void)hDevice;
   free(query);
   return S_OK;
}

static HRESULT APIENTRY
D3D9DeleteShader(HANDLE hDevice, HANDLE shader)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Object *object = (D3D9Object *)shader;
   if (device && device->pipe && object &&
       object->kind == D3D9_OBJECT_SHADER && object->translated_vs_cso)
      device->pipe->delete_vs_state(device->pipe, object->translated_vs_cso);
   if (device && device->pipe && object &&
       object->kind == D3D9_OBJECT_SHADER && object->translated_ps_cso)
      device->pipe->delete_fs_state(device->pipe, object->translated_ps_cso);
   if (object && object->kind == D3D9_OBJECT_SHADER) {
      free(object->translated_vs_const_ranges);
      free(object->translated_ps_const_ranges);
      free(object->translated_vs_lconstf_ranges);
      free(object->translated_vs_lconstf_data);
      free(object->translated_ps_lconstf_ranges);
      free(object->translated_ps_lconstf_data);
   }
   free(shader);
   return S_OK;
}

static HRESULT APIENTRY
D3D9Flush(HANDLE hDevice)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   D3D9Tracef("Flush hDevice=%p pipe=%p\n", hDevice,
              device ? device->pipe : NULL);
   if (!device)
      return E_INVALIDARG;

   if (device->pipe)
      yttrium_gdi_flush_labeled(device->pipe, NULL, PIPE_FLUSH_ASYNC,
                                "D3D9 DDI Flush");
   return S_OK;
}

static HRESULT APIENTRY
D3D9DestroyDevice(HANDLE hDevice)
{
   D3D9Device *device = (D3D9Device *)hDevice;
   if (!device)
      return S_OK;

   /* Match the D3D10/11 teardown rule for device-associated scratch
    * allocations released by Gallium below. */
   device->gdi_device.base.runtime_destroying = true;

   D3D9DestroyDrawState(device);
   for (UINT i = 0; i < ARRAYSIZE(device->null_textures); ++i)
      pipe_resource_reference(&device->null_textures[i], NULL);
   if (device->pipe)
      device->pipe->destroy(device->pipe);
   if (device->screen)
      device->screen->destroy(device->screen);
   device->gdi_device.base.destroy(&device->gdi_device.base);
   free(device);
   return S_OK;
}

void
D3D9FillDeviceFuncs(D3DDDI_DEVICEFUNCS *funcs)
{
   memset(funcs, 0, sizeof(*funcs));

   funcs->pfnSetRenderState = D3D9SetRenderState;
   funcs->pfnUpdateWInfo = D3D9UpdateWInfo;
   funcs->pfnValidateDevice = D3D9ValidateDevice;
   funcs->pfnSetTextureStageState = D3D9SetTextureStageState;
   funcs->pfnSetTexture = D3D9SetTexture;
   funcs->pfnSetPixelShader = D3D9SetPixelShader;
   funcs->pfnSetPixelShaderConst = D3D9SetPixelShaderConst;
   funcs->pfnSetStreamSourceUm = D3D9SetStreamSourceUm;
   funcs->pfnSetIndices = D3D9SetIndices;
   funcs->pfnSetIndicesUm = D3D9SetIndicesUm;
   funcs->pfnDrawPrimitive = D3D9DrawPrimitive;
   funcs->pfnDrawIndexedPrimitive = D3D9DrawIndexedPrimitive;
   funcs->pfnDrawRectPatch = D3D9DrawRectPatch;
   funcs->pfnDrawTriPatch = D3D9DrawTriPatch;
   funcs->pfnDrawPrimitive2 = D3D9DrawPrimitive2;
   funcs->pfnDrawIndexedPrimitive2 = D3D9DrawIndexedPrimitive2;
   funcs->pfnVolBlt = D3D9VolBlt;
   funcs->pfnBufBlt = D3D9BufBlt;
   funcs->pfnTexBlt = D3D9TexBlt;
   funcs->pfnStateSet = D3D9StateSet;
   funcs->pfnSetPriority = D3D9SetPriority;
   funcs->pfnClear = D3D9Clear;
   funcs->pfnUpdatePalette = D3D9UpdatePalette;
   funcs->pfnSetPalette = D3D9SetPalette;
   funcs->pfnSetVertexShaderConst = D3D9SetVertexShaderConst;
   funcs->pfnMultiplyTransform = D3D9MultiplyTransform;
   funcs->pfnSetTransform = D3D9SetTransform;
   funcs->pfnSetViewport = D3D9SetViewport;
   funcs->pfnSetZRange = D3D9SetZRange;
   funcs->pfnSetMaterial = D3D9SetMaterial;
   funcs->pfnSetLight = D3D9SetLight;
   funcs->pfnCreateLight = D3D9CreateLight;
   funcs->pfnDestroyLight = D3D9DestroyLight;
   funcs->pfnSetClipPlane = D3D9SetClipPlane;
   funcs->pfnGetInfo = D3D9GetInfo;
   funcs->pfnLock = D3D9Lock;
   funcs->pfnUnlock = D3D9Unlock;
   funcs->pfnCreateResource = D3D9CreateResource;
   funcs->pfnDestroyResource = D3D9DestroyResource;
   funcs->pfnSetDisplayMode = D3D9SetDisplayMode;
   funcs->pfnPresent = D3D9Present;
   funcs->pfnFlush = D3D9Flush;
   funcs->pfnCreateVertexShaderFunc = D3D9CreateVertexShaderFunc;
   funcs->pfnDeleteVertexShaderFunc = D3D9DeleteShader;
   funcs->pfnSetVertexShaderFunc = D3D9SetVertexShaderFunc;
   funcs->pfnCreateVertexShaderDecl = D3D9CreateVertexShaderDecl;
   funcs->pfnDeleteVertexShaderDecl = D3D9DeleteShader;
   funcs->pfnSetVertexShaderDecl = D3D9SetVertexShaderDecl;
   funcs->pfnSetVertexShaderConstI = D3D9SetVertexShaderConstI;
   funcs->pfnSetVertexShaderConstB = D3D9SetVertexShaderConstB;
   funcs->pfnSetScissorRect = D3D9SetScissorRect;
   funcs->pfnSetStreamSource = D3D9SetStreamSource;
   funcs->pfnSetStreamSourceFreq = D3D9SetStreamSourceFreq;
   funcs->pfnSetConvolutionKernelMono = D3D9SetConvolutionKernelMono;
   funcs->pfnComposeRects = D3D9ComposeRects;
   funcs->pfnBlt = D3D9Blt;
   funcs->pfnColorFill = D3D9ColorFill;
   funcs->pfnDepthFill = D3D9DepthFill;
   funcs->pfnCreateQuery = D3D9CreateQuery;
   funcs->pfnDestroyQuery = D3D9DestroyQuery;
   funcs->pfnIssueQuery = D3D9IssueQuery;
   funcs->pfnGetQueryData = D3D9GetQueryData;
   funcs->pfnSetRenderTarget = D3D9SetRenderTarget;
   funcs->pfnSetDepthStencil = D3D9SetDepthStencil;
   funcs->pfnGenerateMipSubLevels = D3D9GenerateMipSubLevels;
   funcs->pfnSetPixelShaderConstI = D3D9SetPixelShaderConstI;
   funcs->pfnSetPixelShaderConstB = D3D9SetPixelShaderConstB;
   funcs->pfnCreatePixelShader = D3D9CreatePixelShader;
   funcs->pfnDeletePixelShader = D3D9DeleteShader;
   funcs->pfnDestroyDevice = D3D9DestroyDevice;
   funcs->pfnQueryResourceResidency = D3D9QueryResourceResidency;
   funcs->pfnOpenResource = D3D9OpenResource;
   funcs->pfnGetCaptureAllocationHandle = D3D9GetCaptureAllocationHandle;
   funcs->pfnCaptureToSysMem = D3D9CaptureToSysMem;
   funcs->pfnCreateVideoProcessor = D3D9CreateVideoProcessor;
   funcs->pfnSetVideoProcessBltState = D3D9SetVideoProcessBltState;
   funcs->pfnGetVideoProcessBltStatePrivate =
      D3D9GetVideoProcessBltStatePrivate;
   funcs->pfnSetVideoProcessStreamState = D3D9SetVideoProcessStreamState;
   funcs->pfnGetVideoProcessStreamStatePrivate =
      D3D9GetVideoProcessStreamStatePrivate;
   funcs->pfnVideoProcessBltHD = D3D9VideoProcessBltHD;
   funcs->pfnDestroyVideoProcessor = D3D9DestroyVideoProcessor;
   funcs->pfnResolveSharedResource = D3D9ResolveSharedResource;
}
