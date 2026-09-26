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
 * Shader.cpp --
 *    Functions that manipulate shader resources.
 */


#include "Shader.h"
#include "ShaderParse.h"
#include "State.h"
#include "Query.h"
#include "Rasterizer.h"
#include "Resource.h"

#include "Debug.h"
#include "Format.h"

#include "tgsi/tgsi_ureg.h"
#include "util/u_gen_mipmap.h"
#include "util/u_sampler.h"
#include "util/format/u_format.h"

#include "gallium/winsys/yttrium/gdi/yttrium_trace.h"
#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

static bool
OrderedContextWorkerEnabled()
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_WORKER", true) ? 1 : 0;
   }
   return enabled != 0;
}

#include <string.h>


enum ComputeEmulation {
   COMPUTE_EMULATION_NONE = 0,
   COMPUTE_EMULATION_UAV_FLOAT_FILL,
   COMPUTE_EMULATION_UAV_FLOAT_GROUP_ID,
   COMPUTE_EMULATION_UAV_FLOAT_DISPATCH_ID,
   COMPUTE_EMULATION_UAV_TYPED_IMM_STORE,
   COMPUTE_EMULATION_WINE_RAW_UAV_ATOMICS,
   COMPUTE_EMULATION_WINE_BUFFERINFO,
   COMPUTE_EMULATION_WINE_TGSM_RAW,
   COMPUTE_EMULATION_WINE_TGSM_STRUCTURED_RAW,
   COMPUTE_EMULATION_WINE_TGSM_STRUCTURED_TYPED,
   COMPUTE_EMULATION_WINE_DISPATCH_STATS,
};

static const char *
ComputeEmulationName(unsigned emulation)
{
   switch (emulation) {
   case COMPUTE_EMULATION_UAV_FLOAT_FILL:
      return "uav-float-fill";
   case COMPUTE_EMULATION_UAV_FLOAT_GROUP_ID:
      return "uav-float-group-id";
   case COMPUTE_EMULATION_UAV_FLOAT_DISPATCH_ID:
      return "uav-float-dispatch-id";
   case COMPUTE_EMULATION_UAV_TYPED_IMM_STORE:
      return "uav-typed-immediate-store";
   case COMPUTE_EMULATION_WINE_RAW_UAV_ATOMICS:
      return "wine-raw-uav-atomics";
   case COMPUTE_EMULATION_WINE_BUFFERINFO:
      return "wine-bufferinfo";
   case COMPUTE_EMULATION_WINE_TGSM_RAW:
      return "wine-tgsm-raw";
   case COMPUTE_EMULATION_WINE_TGSM_STRUCTURED_RAW:
      return "wine-tgsm-structured-raw";
   case COMPUTE_EMULATION_WINE_TGSM_STRUCTURED_TYPED:
      return "wine-tgsm-structured-typed";
   case COMPUTE_EMULATION_WINE_DISPATCH_STATS:
      return "wine-dispatch-stats";
   default:
      return "unknown";
   }
}

static const char *
ShaderStageName(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return "vertex";
   case MESA_SHADER_TESS_CTRL:
      return "tess-ctrl";
   case MESA_SHADER_TESS_EVAL:
      return "tess-eval";
   case MESA_SHADER_GEOMETRY:
      return "geometry";
   case MESA_SHADER_FRAGMENT:
      return "fragment";
   case MESA_SHADER_COMPUTE:
      return "compute";
   default:
      return "unknown";
   }
}

static void
WarnSoftwareEmulationFallback(mesa_shader_stage stage, Shader *shader,
                              const char *details)
{
   YTTRIUM_WARN("yttrium: d3d10umd software emulation fallback stage=%s "
                "kind=%s shader=%p %s\n",
                ShaderStageName(stage),
                shader ? ComputeEmulationName(shader->compute_emulation) :
                         "unknown",
                shader, details ? details : "");
}

static void
InitShaderOutputMapping(Shader *shader)
{
   for (unsigned i = 0; i < PIPE_MAX_SHADER_OUTPUTS; ++i)
      shader->output_mapping[i] = ~0u;
}

static void
InitShaderObject(Shader *shader, mesa_shader_stage stage)
{
   memset(shader, 0, sizeof(*shader));
   shader->type = stage;
   InitShaderOutputMapping(shader);
}

void
BindShaderImages(Device *device, mesa_shader_stage stage,
                  unsigned start_slot, unsigned num_views)
{
   Shader *shader = NULL;
   switch (stage) {
   case MESA_SHADER_VERTEX: shader = device->bound_vs; break;
   case MESA_SHADER_TESS_CTRL: shader = device->bound_hs; break;
   case MESA_SHADER_TESS_EVAL: shader = device->bound_ds; break;
   case MESA_SHADER_GEOMETRY: shader = device->bound_gs; break;
   case MESA_SHADER_FRAGMENT: shader = device->bound_ps; break;
   case MESA_SHADER_COMPUTE: shader = device->bound_cs; break;
   default: return;
   }

   uint64_t mask = 0;
   if (shader) {
      for (unsigned slot = 0; slot < PIPE_MAX_SHADER_IMAGES; ++slot)
         if (shader->counter_image_slots[slot])
            mask |= UINT64_C(1) << (shader->counter_image_slots[slot] - 1);
   }
   struct pipe_context *pipe = device->pipe;
   if (!mask && !device->counter_image_bound_mask[stage]) {
      if (num_views)
         pipe->set_shader_images(pipe, stage, start_slot, num_views, 0,
                                  &device->shader_images[stage][start_slot]);
      return;
   }

   /* Keep the D3D bindings canonical. Hidden slots are shader-dependent and
    * must be restored when another shader uses that slot as an ordinary UAV.
    * Gallium/threaded-context retain the submitted image resources, including
    * counter views destroyed while a command is still queued.
    */
   struct pipe_image_view images[PIPE_MAX_SHADER_IMAGES];
   memcpy(images, device->shader_images[stage], sizeof(images));
   if (shader) {
      for (unsigned slot = 0; slot < PIPE_MAX_SHADER_IMAGES; ++slot) {
         if (!shader->counter_image_slots[slot])
            continue;
         struct pipe_image_view *image =
            &images[shader->counter_image_slots[slot] - 1];
         memset(image, 0, sizeof(*image));
         UnorderedAccessView *uav = device->unordered_access_views[stage][slot];
         if (uav && uav->counter_resource) {
            image->resource = uav->counter_resource;
            image->format = PIPE_FORMAT_R32_UINT;
            image->access = PIPE_IMAGE_ACCESS_READ_WRITE;
            image->shader_access = PIPE_IMAGE_ACCESS_READ_WRITE;
            image->u.tex.single_layer_view = true;
         }
      }
   }
   pipe->set_shader_images(pipe, stage, 0, PIPE_MAX_SHADER_IMAGES, 0, images);
   device->counter_image_bound_mask[stage] = mask;
}

void
ResetUAVCounter(Device *device, UnorderedAccessView *view, UINT value)
{
   if (!view || !view->counter_resource)
      return;
   struct pipe_surface surface = {};
   surface.texture = view->counter_resource;
   surface.format = PIPE_FORMAT_R32_UINT;
   union pipe_color_union color = {};
   color.ui[0] = value;
   /* Ordered GPU clear: never map the counter or evaluate atomics on the CPU. */
   device->pipe->clear_render_target(device->pipe, &surface, &color,
                                     0, 0, 1, 1, false);
}

static struct pipe_shader_sampler_binding_map *
PrepareSamplerBindingMap(Device *device, Shader *shader)
{
   memset(&shader->sampler_binding_map, 0,
          sizeof(shader->sampler_binding_map));
   shader->state.sampler_binding_map = NULL;

   if (!device || !device->yttrium_sampler_binding_map_enabled)
      return NULL;

   shader->state.sampler_binding_map = &shader->sampler_binding_map;
   return &shader->sampler_binding_map;
}

static bool
CaptureTokenlessStreamOutputSignature(
   Shader *shader,
   const D3D10DDIARG_STAGE_IO_SIGNATURES *signatures)
{
   if (!signatures ||
       (signatures->NumInputSignatureEntries &&
        !signatures->pInputSignature)) {
      YTTRIUM_WARN("yttrium: d3d10umd tokenless geometry stream-output "
                   "creation rejected reason=input_source_signature_unavailable\n");
      return false;
   }

   for (unsigned i = 0; i < signatures->NumInputSignatureEntries; ++i) {
      const D3D10DDIARG_SIGNATURE_ENTRY *entry =
         &signatures->pInputSignature[i];
      if (entry->Register >= PIPE_MAX_SHADER_OUTPUTS || !entry->Mask ||
          (entry->Mask & ~TGSI_WRITEMASK_XYZW)) {
         YTTRIUM_WARN("yttrium: d3d10umd tokenless geometry stream-output "
                      "creation rejected reason=invalid_input_source_signature "
                      "entry=%u register=%u mask=0x%x\n",
                      i, entry->Register, entry->Mask);
         return false;
      }
      shader->stream_output_signature_masks[entry->Register] |= entry->Mask;
   }

   return true;
}

static unsigned
GetComputeEmulation(const UINT *code, unsigned store_imm[4])
{
   struct Shader_parser parser;
   struct Shader_opcode opcode;
   bool saw_typed_store = false;
   bool saw_any_thread_id = false;
   bool saw_uav_counter = false;
   bool saw_bufinfo_needs_emulation = false;
   bool saw_atomic_iadd = false;
   bool saw_atomic_umax = false;
   unsigned structured_uav_count = 0;
   bool bufinfo_resource_needs_emulation[PIPE_MAX_SHADER_SAMPLER_VIEWS] = {};
   bool bufinfo_uav_needs_emulation[PIPE_MAX_SHADER_IMAGES] = {};

   if (!code)
      return COMPUTE_EMULATION_NONE;

   Shader_parse_init(&parser, code);
   if (parser.header.type != DX10_SM5_COMPUTE_SHADER &&
       parser.header.type != D3D10_SB_PIXEL_SHADER &&
       parser.header.type != D3D10_SB_VERTEX_SHADER)
      return COMPUTE_EMULATION_NONE;

   while (Shader_parse_opcode(&parser, &opcode)) {
      switch (opcode.type) {
      case DX10_SM5_OPCODE_DCL_UAV_RAW:
         if (opcode.dst[0].base.index_dim == 1 &&
             opcode.dst[0].base.index[0].index_rep ==
                D3D10_SB_OPERAND_INDEX_IMMEDIATE32 &&
             opcode.dst[0].base.index[0].imm < PIPE_MAX_SHADER_IMAGES)
            bufinfo_uav_needs_emulation[opcode.dst[0].base.index[0].imm] = true;
         break;
      case DX10_SM5_OPCODE_DCL_UAV_STRUCTURED:
         if (opcode.dst[0].base.index_dim == 1 &&
             opcode.dst[0].base.index[0].index_rep ==
                D3D10_SB_OPERAND_INDEX_IMMEDIATE32 &&
             opcode.dst[0].base.index[0].imm < PIPE_MAX_SHADER_IMAGES)
            bufinfo_uav_needs_emulation[opcode.dst[0].base.index[0].imm] = true;
         structured_uav_count++;
         break;
      case DX10_SM5_OPCODE_DCL_RESOURCE_RAW:
      case DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED:
         if (opcode.dst[0].base.index_dim == 1 &&
             opcode.dst[0].base.index[0].index_rep ==
                D3D10_SB_OPERAND_INDEX_IMMEDIATE32 &&
             opcode.dst[0].base.index[0].imm < PIPE_MAX_SHADER_SAMPLER_VIEWS)
            bufinfo_resource_needs_emulation[opcode.dst[0].base.index[0].imm] = true;
         break;
      case DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC:
      case DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME:
         saw_uav_counter = true;
         break;
      case DX10_SM5_OPCODE_ATOMIC_IADD:
         saw_atomic_iadd = true;
         break;
      case DX10_SM5_OPCODE_ATOMIC_UMAX:
         saw_atomic_umax = true;
         break;
      default:
         break;
      }

      if (opcode.type == D3D11_SB_OPCODE_STORE_UAV_TYPED) {
         saw_typed_store = true;
         if (opcode.src[1].base.type == D3D10_SB_OPERAND_TYPE_IMMEDIATE32 &&
             store_imm) {
            store_imm[0] = opcode.src[1].imm[0].u32;
            store_imm[1] = opcode.src[1].imm[1].u32;
            store_imm[2] = opcode.src[1].imm[2].u32;
            store_imm[3] = opcode.src[1].imm[3].u32;
         }
      }

      for (unsigned i = 0; i < opcode.num_src; ++i) {
         D3D10_SB_OPERAND_TYPE type = opcode.src[i].base.type;

         if (type == DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID ||
             type == DX10_SM5_OPERAND_TYPE_INPUT_THREAD_GROUP_ID ||
             type == DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP ||
             type == DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP_FLATTENED)
            saw_any_thread_id = true;
      }

      Shader_opcode_free(&opcode);
   }

   if (parser.failed)
      return COMPUTE_EMULATION_NONE;

   /* Counter instructions always use the view's GPU-owned counter.  Do not
    * let an unrelated legacy pattern recognizer intercept a mixed shader. */
   if (saw_uav_counter)
      return COMPUTE_EMULATION_NONE;

   if (parser.header.type == D3D10_SB_PIXEL_SHADER &&
       saw_bufinfo_needs_emulation)
      return COMPUTE_EMULATION_NONE;

   if (!saw_typed_store) {

      if (parser.header.type == DX10_SM5_COMPUTE_SHADER &&
          structured_uav_count == 1 && saw_any_thread_id &&
          saw_atomic_iadd && saw_atomic_umax)
         /*
          * Keep the Wine dispatch-stats shader on the CPU fallback path until
          * buffer image atomics are diagnosed on the host side.  Native
          * ATOMUADD/ATOMUMAX to the structured UAV reproduced a guilty amdgpu
          * hard recovery through virgl_render_server.
          */
         return COMPUTE_EMULATION_WINE_DISPATCH_STATS;

      return COMPUTE_EMULATION_NONE;
   }

   if (parser.header.type == D3D10_SB_PIXEL_SHADER)
      return COMPUTE_EMULATION_NONE;

   return COMPUTE_EMULATION_NONE;
}

static bool
ReadShaderConstantBytes(Device *pDevice, mesa_shader_stage stage,
                        unsigned slot, void *data, unsigned size)
{
   if (!pDevice || !data || !size || stage >= MESA_SHADER_STAGES ||
       slot >= PIPE_MAX_CONSTANT_BUFFERS)
      return false;

   struct pipe_context *pipe = pDevice->pipe;
   struct pipe_resource *cb =
      pDevice->constant_buffers[stage][slot];
   Resource *cb_resource =
      pDevice->constant_buffer_resources[stage][slot];
   if (cb_resource && cb_resource->constant_published_cpu &&
       cb_resource->constant_published_size >= size) {
      memcpy(data, cb_resource->constant_published_cpu, size);
      return true;
   }
   if (cb_resource && cb_resource->buffer_shadow &&
       cb_resource->buffer_shadow_size >= size) {
      memcpy(data, cb_resource->buffer_shadow, size);
      return true;
   }

   if (!pipe || !cb || cb->target != PIPE_BUFFER || cb->width0 < size)
      return false;

   struct pipe_box box = {};
   box.width = size;
   box.height = 1;
   box.depth = 1;

   struct pipe_transfer *transfer = NULL;
   void *map = pipe->buffer_map(pipe, cb, 0, PIPE_MAP_READ, &box, &transfer);
   if (!map)
      return false;

   memcpy(data, map, size);
   pipe->buffer_unmap(pipe, transfer);
   return true;
}

static bool
ReadComputeConstantSlotFloat(Device *pDevice, unsigned slot, float *value)
{
   if (!pDevice || !value || slot >= PIPE_MAX_CONSTANT_BUFFERS)
      return false;

   struct pipe_context *pipe = pDevice->pipe;
   struct pipe_resource *cb =
      pDevice->constant_buffers[MESA_SHADER_COMPUTE][slot];
   Resource *cb_resource =
      pDevice->constant_buffer_resources[MESA_SHADER_COMPUTE][slot];
   if (cb_resource && cb_resource->constant_published_cpu &&
       cb_resource->constant_published_size >= sizeof(*value)) {
      memcpy(value, cb_resource->constant_published_cpu, sizeof(*value));
      return true;
   }
   if (cb_resource && cb_resource->buffer_shadow &&
       cb_resource->buffer_shadow_size >= sizeof(*value)) {
      memcpy(value, cb_resource->buffer_shadow, sizeof(*value));
      return true;
   }

   if (!pipe || !cb || cb->target != PIPE_BUFFER || cb->width0 < sizeof(float))
      return false;

   struct pipe_box box = {};
   box.width = sizeof(float);
   box.height = 1;
   box.depth = 1;

   struct pipe_transfer *transfer = NULL;
   void *map = pipe->buffer_map(pipe, cb, 0, PIPE_MAP_READ, &box, &transfer);
   if (!map)
      return false;

   memcpy(value, map, sizeof(*value));
   pipe->buffer_unmap(pipe, transfer);
   return true;
}

static bool
ReadComputeConstantFloat(Device *pDevice, float *value)
{
   if (ReadComputeConstantSlotFloat(pDevice, 0, value) ||
       ReadComputeConstantSlotFloat(pDevice, 1, value))
      return true;

   for (unsigned slot = 2; slot < PIPE_MAX_CONSTANT_BUFFERS; ++slot) {
      if (ReadComputeConstantSlotFloat(pDevice, slot, value))
         return true;
   }

   return false;
}

static bool
ReadBufferRange(struct pipe_context *pipe, struct pipe_resource *resource,
                unsigned offset, unsigned size, void *data)
{
   if (!pipe || !resource || resource->target != PIPE_BUFFER || !data ||
       offset > resource->width0 || size > resource->width0 - offset)
      return false;

   struct pipe_box box = {};
   box.x = offset;
   box.width = size;
   box.height = 1;
   box.depth = 1;

   struct pipe_transfer *transfer = NULL;
   void *map = pipe->buffer_map(pipe, resource, 0, PIPE_MAP_READ, &box,
                                &transfer);
   if (!map)
      return false;

   memcpy(data, map, size);
   pipe->buffer_unmap(pipe, transfer);
   return true;
}

static bool
WriteBufferRange(struct pipe_context *pipe, struct pipe_resource *resource,
                 unsigned offset, unsigned size, const void *data)
{
   if (!pipe || !resource || resource->target != PIPE_BUFFER || !data ||
       offset > resource->width0 || size > resource->width0 - offset)
      return false;

   if (pipe->buffer_subdata) {
      pipe->buffer_subdata(pipe, resource, 0, offset, size, data);
      return true;
   }

   struct pipe_box box = {};
   box.x = offset;
   box.width = size;
   box.height = 1;
   box.depth = 1;

   struct pipe_transfer *transfer = NULL;
   void *map = pipe->buffer_map(pipe, resource, 0, PIPE_MAP_WRITE, &box,
                                &transfer);
   if (!map)
      return false;

   memcpy(map, data, size);
   pipe->buffer_unmap(pipe, transfer);
   return true;
}

static bool
RunWineRawUAVAtomics(Device *pDevice, mesa_shader_stage stage,
                     bool write_original)
{
   struct Constants {
      UINT v[4];
      INT i[4];
   } constants;
   UINT values[9];
   UINT original[9];
   unsigned constant_slot = stage == MESA_SHADER_COMPUTE ? 0 : 1;

   if (!pDevice || !pDevice->pipe || stage >= MESA_SHADER_STAGES)
      return false;

   struct pipe_image_view *input =
      &pDevice->shader_images[stage][0];
   struct pipe_image_view *output =
      &pDevice->shader_images[stage][1];
   if (!input->resource ||
       input->resource->target != PIPE_BUFFER ||
       input->u.buf.size < sizeof(values))
      return false;
   if (write_original &&
       (!output->resource || output->resource->target != PIPE_BUFFER ||
        output->u.buf.size < sizeof(original)))
      return false;

   if (!ReadShaderConstantBytes(pDevice, stage, constant_slot, &constants,
                                sizeof(constants)))
      return false;

   if (!ReadBufferRange(pDevice->pipe, input->resource, input->u.buf.offset,
                        sizeof(values), values))
      return false;

   original[0] = values[0];
   values[0] &= constants.v[0];

   original[1] = values[1];
   if (values[1] == constants.v[1])
      values[1] = constants.v[0];

   original[2] = values[2];
   values[2] += constants.v[0];

   original[3] = values[3];
   values[3] |= constants.v[0];

   original[4] = values[4];
   values[4] = (INT)values[4] > constants.i[0] ? values[4] :
      (UINT)constants.i[0];

   original[5] = values[5];
   values[5] = (INT)values[5] < constants.i[0] ? values[5] :
      (UINT)constants.i[0];

   original[6] = values[6];
   values[6] = values[6] > constants.v[0] ? values[6] : constants.v[0];

   original[7] = values[7];
   values[7] = values[7] < constants.v[0] ? values[7] : constants.v[0];

   original[8] = values[8];
   values[8] ^= constants.v[0];

   if (!WriteBufferRange(pDevice->pipe, input->resource, input->u.buf.offset,
                         sizeof(values), values))
      return false;

   if (!write_original)
      return true;

   return WriteBufferRange(pDevice->pipe, output->resource,
                           output->u.buf.offset, sizeof(original), original);
}

static bool
RunWineRawUAVAtomicsCompute(Device *pDevice)
{
   return RunWineRawUAVAtomics(pDevice, MESA_SHADER_COMPUTE, true);
}

static bool
RunWineTGSMRawCompute(Device *pDevice, UINT ThreadGroupCountX)
{
   if (!pDevice || !pDevice->pipe)
      return false;

   struct pipe_image_view *uav =
      &pDevice->shader_images[MESA_SHADER_COMPUTE][0];
   if (!uav->resource || uav->resource->target != PIPE_BUFFER)
      return false;

   const unsigned count =
      MIN2(ThreadGroupCountX, uav->u.buf.size / sizeof(uint32_t));
   if (!count)
      return false;

   uint32_t values[256];
   if (count > ARRAY_SIZE(values))
      return false;

   for (unsigned i = 0; i < count; ++i)
      values[i] = 33 * i;

   return WriteBufferRange(pDevice->pipe, uav->resource, uav->u.buf.offset,
                           count * sizeof(values[0]), values);
}

static bool
RunWineTGSMStructuredRawCompute(Device *pDevice, UINT ThreadGroupCountX)
{
   if (!pDevice || !pDevice->pipe)
      return false;

   struct pipe_image_view *uav =
      &pDevice->shader_images[MESA_SHADER_COMPUTE][0];
   if (!uav->resource || uav->resource->target != PIPE_BUFFER)
      return false;

   const unsigned count =
      MIN2(ThreadGroupCountX, uav->u.buf.size / sizeof(uint32_t));
   if (!count)
      return false;

   uint32_t values[256];
   if (count > ARRAY_SIZE(values))
      return false;

   for (unsigned i = 0; i < count; ++i)
      values[i] = 64 * i + 32;

   return WriteBufferRange(pDevice->pipe, uav->resource, uav->u.buf.offset,
                           count * sizeof(values[0]), values);
}

static bool
RunWineTGSMStructuredTypedCompute(Device *pDevice, UINT ThreadGroupCountX)
{
   if (!pDevice || !pDevice->pipe)
      return false;

   struct pipe_image_view *float_uav =
      &pDevice->shader_images[MESA_SHADER_COMPUTE][0];
   struct pipe_image_view *uint_uav =
      &pDevice->shader_images[MESA_SHADER_COMPUTE][1];
   if (!float_uav->resource || float_uav->resource->target != PIPE_BUFFER ||
       !uint_uav->resource || uint_uav->resource->target != PIPE_BUFFER)
      return false;

   const unsigned count =
      MIN2(ThreadGroupCountX * 32,
           MIN2(float_uav->u.buf.size / sizeof(float),
                uint_uav->u.buf.size / sizeof(uint32_t)));
   if (!count)
      return false;

   float float_values[256];
   uint32_t uint_values[256];
   if (count > ARRAY_SIZE(float_values) || count > ARRAY_SIZE(uint_values))
      return false;

   for (unsigned i = 0; i < count; ++i) {
      uint_values[i] = (i % 32 + 1) * (i / 32);
      float_values[i] = (float)uint_values[i];
   }

   return WriteBufferRange(pDevice->pipe, float_uav->resource,
                           float_uav->u.buf.offset,
                           count * sizeof(float_values[0]), float_values) &&
          WriteBufferRange(pDevice->pipe, uint_uav->resource,
                           uint_uav->u.buf.offset,
                           count * sizeof(uint_values[0]), uint_values);
}

static struct pipe_image_view *
GetComputeUAVImage(Device *pDevice)
{
   if (!pDevice)
      return NULL;

   for (unsigned slot = 0; slot < PIPE_MAX_SHADER_IMAGES; ++slot) {
      struct pipe_image_view *image =
         &pDevice->shader_images[MESA_SHADER_COMPUTE][slot];
      if (image->resource)
         return image;
   }

   return NULL;
}

static struct pipe_image_view *
GetComputeUAVFloatImage(Device *pDevice)
{
   if (!pDevice)
      return NULL;

   for (unsigned slot = 0; slot < PIPE_MAX_SHADER_IMAGES; ++slot) {
      struct pipe_image_view *image =
         &pDevice->shader_images[MESA_SHADER_COMPUTE][slot];
      struct pipe_resource *resource = image->resource;
      if (!resource || resource->target != PIPE_TEXTURE_2D)
         continue;

      const enum pipe_format format =
         image->format == PIPE_FORMAT_NONE ? resource->format : image->format;
      if (format == PIPE_FORMAT_R32_FLOAT ||
          resource->format == PIPE_FORMAT_R32_FLOAT)
         return image;
   }

   return NULL;
}

static bool
WriteComputeUAVFloatRect(Device *pDevice,
                         unsigned width,
                         unsigned height,
                         float value)
{
   if (!pDevice || !width || !height)
      return false;

   struct pipe_context *pipe = pDevice->pipe;
   struct pipe_image_view *image = GetComputeUAVFloatImage(pDevice);
   if (!image)
      return false;

   struct pipe_resource *resource = image->resource;
   if (!pipe || !pipe->texture_subdata || !resource ||
       resource->target != PIPE_TEXTURE_2D)
      return false;

   const enum pipe_format format =
      image->format == PIPE_FORMAT_NONE ? resource->format : image->format;
   if (format != PIPE_FORMAT_R32_FLOAT &&
       resource->format != PIPE_FORMAT_R32_FLOAT)
      return false;

   const unsigned level = image->u.tex.level;
   if (level > resource->last_level)
      return false;

   const unsigned tex_width = u_minify(resource->width0, level);
   const unsigned tex_height = u_minify(resource->height0, level);
   width = MIN2(width, tex_width);
   height = MIN2(height, tex_height);
   if (!width || !height)
      return true;

   struct pipe_box box = {};
   box.x = 0;
   box.y = 0;
   box.z = image->u.tex.first_layer;
   box.width = width;
   box.height = height;
   box.depth = 1;

   const unsigned stride = width * sizeof(float);
   const unsigned size = stride * height;
   float *data = (float *)MALLOC(size);
   if (!data)
      return false;

   const unsigned count = width * height;
   for (unsigned i = 0; i < count; i++)
      data[i] = value;

   pipe->texture_subdata(pipe, resource, level, PIPE_MAP_DISCARD_RANGE, &box,
                         data, stride, size);
   FREE(data);
   return true;
}

static float
FloatFromBits(unsigned bits)
{
   float value;
   memcpy(&value, &bits, sizeof(value));
   return value;
}

static unsigned
PackUnorm8(float value)
{
   int scaled;

   if (value < 0.0f)
      value = 0.0f;
   if (value > 1.0f)
      value = 1.0f;

   scaled = (int)(value * 255.0f + 0.5f);
   return (unsigned)CLAMP(scaled, 0, 255);
}

static unsigned
PackSnorm8(float value)
{
   int scaled;

   if (value < -1.0f)
      value = -1.0f;
   if (value > 1.0f)
      value = 1.0f;

   scaled = value >= 0.0f ? (int)(value * 127.0f + 0.5f) :
                            (int)(value * 127.0f - 0.5f);
   scaled = CLAMP(scaled, -128, 127);
   return (unsigned)(scaled & 0xff);
}

static bool
WriteComputeUAVImmediateStore(Device *pDevice, Shader *cs)
{
   if (!pDevice || !cs)
      return false;

   struct pipe_context *pipe = pDevice->pipe;
   struct pipe_image_view *image = GetComputeUAVImage(pDevice);
   if (!pipe || !image || !image->resource)
      return false;

   struct pipe_resource *resource = image->resource;
   enum pipe_format format =
      image->format == PIPE_FORMAT_NONE ? resource->format : image->format;

   if (resource->target == PIPE_BUFFER) {
      unsigned value = cs->compute_store_imm[0];
      const unsigned offset = image->u.buf.offset;

      if (!pipe->buffer_subdata || offset + sizeof(value) > resource->width0)
         return false;

      pipe->buffer_subdata(pipe, resource, 0, offset, sizeof(value), &value);
      return true;
   }

   if (resource->target != PIPE_TEXTURE_2D || !pipe->texture_subdata)
      return false;

   unsigned packed = 0;
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
      for (unsigned i = 0; i < 4; ++i)
         packed |= PackUnorm8(FloatFromBits(cs->compute_store_imm[i])) << (i * 8);
      break;
   case PIPE_FORMAT_R8G8B8A8_SNORM:
      for (unsigned i = 0; i < 4; ++i)
         packed |= PackSnorm8(FloatFromBits(cs->compute_store_imm[i])) << (i * 8);
      break;
   default:
      return false;
   }

   struct pipe_box box = {};
   box.x = 0;
   box.y = 0;
   box.z = image->u.tex.first_layer;
   box.width = 1;
   box.height = 1;
   box.depth = 1;

   pipe->texture_subdata(pipe, resource, image->u.tex.level,
                         PIPE_MAP_DISCARD_RANGE, &box, &packed,
                         sizeof(packed), sizeof(packed));
   return true;
}

static bool
RunWineDispatchStatsCompute(Device *pDevice, Shader *cs,
                            UINT ThreadGroupCountX,
                            UINT ThreadGroupCountY,
                            UINT ThreadGroupCountZ);

static bool
RunComputeEmulation(Device *pDevice,
                    Shader *cs,
                    UINT ThreadGroupCountX,
                    UINT ThreadGroupCountY,
                    UINT ThreadGroupCountZ)
{
   if (!pDevice || !cs || cs->compute_emulation == COMPUTE_EMULATION_NONE)
      return false;

   char details[128];
   snprintf(details, sizeof(details),
            "dispatch=%ux%ux%u tg_size=%ux%ux%u",
            ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ,
            cs->thread_group_size[0], cs->thread_group_size[1],
            cs->thread_group_size[2]);
   WarnSoftwareEmulationFallback(MESA_SHADER_COMPUTE, cs, details);

   if (cs->compute_emulation == COMPUTE_EMULATION_UAV_TYPED_IMM_STORE)
      return WriteComputeUAVImmediateStore(pDevice, cs);

   if (cs->compute_emulation == COMPUTE_EMULATION_WINE_RAW_UAV_ATOMICS)
      return RunWineRawUAVAtomicsCompute(pDevice);

   if (cs->compute_emulation == COMPUTE_EMULATION_WINE_TGSM_RAW)
      return RunWineTGSMRawCompute(pDevice, ThreadGroupCountX);

   if (cs->compute_emulation == COMPUTE_EMULATION_WINE_TGSM_STRUCTURED_RAW)
      return RunWineTGSMStructuredRawCompute(pDevice, ThreadGroupCountX);

   if (cs->compute_emulation == COMPUTE_EMULATION_WINE_TGSM_STRUCTURED_TYPED)
      return RunWineTGSMStructuredTypedCompute(pDevice, ThreadGroupCountX);


   if (cs->compute_emulation == COMPUTE_EMULATION_WINE_DISPATCH_STATS)
      return RunWineDispatchStatsCompute(pDevice, cs, ThreadGroupCountX,
                                         ThreadGroupCountY,
                                         ThreadGroupCountZ);

   struct pipe_image_view *image = GetComputeUAVFloatImage(pDevice);
   if (!image)
      return false;

   struct pipe_resource *resource = image->resource;
   if (!resource || resource->target != PIPE_TEXTURE_2D)
      return false;

   float value;
   if (!ReadComputeConstantFloat(pDevice, &value))
      return false;

   const unsigned level = image->u.tex.level;
   const unsigned tex_width = u_minify(resource->width0, level);
   const unsigned tex_height = u_minify(resource->height0, level);
   unsigned width = tex_width;
   unsigned height = tex_height;

   switch (cs->compute_emulation) {
   case COMPUTE_EMULATION_UAV_FLOAT_FILL:
      break;
   case COMPUTE_EMULATION_UAV_FLOAT_GROUP_ID:
      width = ThreadGroupCountX;
      height = ThreadGroupCountY;
      break;
   case COMPUTE_EMULATION_UAV_FLOAT_DISPATCH_ID:
      width = ThreadGroupCountX * MAX2(cs->thread_group_size[0], 1u);
      height = ThreadGroupCountY * MAX2(cs->thread_group_size[1], 1u);
      break;
   default:
      return false;
   }

   return WriteComputeUAVFloatRect(pDevice, width, height, value);
}


static inline struct pipe_sampler_view *
GetPipeShaderResourceView(D3D10DDI_HSHADERRESOURCEVIEW hRenderTargetView)
{
   ShaderResourceView *pShaderResourceView = CastShaderResourceView(hRenderTargetView);
   if (!pShaderResourceView) return NULL;

   // Validate that view texture matches resource 
   // If it is not (resource rotated) then recreate view
   struct pipe_sampler_view *currentSurface = pShaderResourceView->handle;
   if (!currentSurface) return NULL;

   Resource *res = pShaderResourceView->resource;
   if (!res || !res->resource) return currentSurface;

   if (currentSurface->texture != res->resource) {
      struct pipe_context *pipe = currentSurface->context;
      struct pipe_sampler_view *newView = pipe->create_sampler_view(pipe, res->resource, currentSurface);
      pipe_sampler_view_reference(&pShaderResourceView->handle, newView);
      pipe_sampler_view_reference(&newView, NULL);
   }

   return pShaderResourceView->handle;
}

/*
 * ----------------------------------------------------------------------
 *
 * CreateEmptyShader --
 *
 *    The CreateEmptyShader function creates a empty shader.
 *
 * ----------------------------------------------------------------------
 */

void *
CreateEmptyShader(Device *pDevice,
                  mesa_shader_stage processor)
{
   struct pipe_context *pipe = pDevice->pipe;
   struct ureg_program *ureg;
   const struct tgsi_token *tokens;
   uint nr_tokens;

   if (processor == MESA_SHADER_GEOMETRY) {
      return NULL;
   }

   ureg = ureg_create(processor);
   if (!ureg)
      return NULL;

   ureg_END(ureg);

   tokens = ureg_get_tokens(ureg, &nr_tokens);
   if (!tokens)
      return NULL;

   ureg_destroy(ureg);

   struct pipe_shader_state state;
   memset(&state, 0, sizeof state);
   state.tokens = tokens;

   void *handle;
   switch (processor) {
   case MESA_SHADER_FRAGMENT:
      handle = pipe->create_fs_state(pipe, &state);
      break;
   case MESA_SHADER_VERTEX:
      handle = pipe->create_vs_state(pipe, &state);
      break;
   case MESA_SHADER_GEOMETRY:
      handle = pipe->create_gs_state(pipe, &state);
      break;
   default:
      handle = NULL;
      assert(0);
   }
   assert(handle);

   ureg_free_tokens(tokens);

   return handle;
}


/*
 * ----------------------------------------------------------------------
 *
 * DeleteEmptyShader --
 *
 *    The DeleteEmptyShader function delete a empty shader.
 *
 * ----------------------------------------------------------------------
 */

void
DeleteEmptyShader(Device *pDevice,
                  mesa_shader_stage processor, void *handle)
{
   struct pipe_context *pipe = pDevice->pipe;

   if (processor == MESA_SHADER_GEOMETRY) {
      assert(handle == NULL);
      return;
   }

   assert(handle != NULL);
   switch (processor) {
   case MESA_SHADER_FRAGMENT:
      pipe->delete_fs_state(pipe, handle);
      break;
   case MESA_SHADER_VERTEX:
      pipe->delete_vs_state(pipe, handle);
      break;
   case MESA_SHADER_GEOMETRY:
      pipe->delete_gs_state(pipe, handle);
      break;
   default:
      assert(0);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * SetConstantBuffers --
 *
 *    Update the driver's currently bound constant buffers.
 *
 * ----------------------------------------------------------------------
 */

/*
 * Constant buffers are rebound several times per draw, so the trace below runs
 * tens of thousands of times a second - by far the highest-volume event in the
 * driver, and enough formatting to show up as ~7% of the process in a CPU
 * profile of the draw path.  That distorts the very captures used to decide
 * what to optimise, so give it its own knob instead of riding the general
 * trace switch.
 */
static bool
TraceConstantBufferBinds(void)
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_TRACE_CONSTANT_BUFFER_BINDS", false) ? 1 : 0;
   }

   return enabled != 0;
}

static void
TrackConstantBufferBinding(Device *device, mesa_shader_stage stage,
                           unsigned slot, Resource *resource,
                           unsigned buffer_size)
{
   Resource *old = device->constant_buffer_binding_resources[stage][slot];
   const uint32_t bit = 1u << slot;

   if (old && old != resource)
      old->constant_buffer_bindings[stage] &= ~bit;

   device->constant_buffer_binding_resources[stage][slot] = resource;
   if (resource && buffer_size)
      resource->constant_buffer_bindings[stage] |= bit;
   else if (resource)
      resource->constant_buffer_bindings[stage] &= ~bit;
}

void
SetConstantBuffersRange(mesa_shader_stage shader_type,    // IN
                        D3D10DDI_HDEVICE hDevice,             // IN
                        UINT StartBuffer,                     // IN
                        UINT NumBuffers,                      // IN
                        const D3D10DDI_HRESOURCE *phBuffers, // IN
                        const UINT *pFirstConstant,           // IN
                        const UINT *pNumConstants)            // IN
{
   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;

   for (UINT i = 0; i < NumBuffers; i++) {
      struct pipe_constant_buffer cb;
      memset(&cb, 0, sizeof cb);
      Resource *resource = CastResource(phBuffers[i]);
      cb.buffer = CastPipeResource(phBuffers[i]);
      struct pipe_resource *resource_buffer = cb.buffer;
      cb.buffer_offset = pFirstConstant ? pFirstConstant[i] * 16 : 0;
      const unsigned resource_offset = cb.buffer_offset;
      cb.buffer_size = 0;
      if (cb.buffer && cb.buffer_offset < cb.buffer->width0) {
         UINT available = cb.buffer->width0 - cb.buffer_offset;
         if (pNumConstants) {
            UINT requested = pNumConstants[i] * 16;
            cb.buffer_size = requested < available ? requested : available;
         } else {
            cb.buffer_size = available;
         }
      }
      const bool published =
         cb.buffer && cb.buffer_size &&
         PreparePublishedConstantBuffer(pDevice, resource, resource_offset,
                                        cb.buffer_size, &cb);
      const unsigned state_slot = StartBuffer + i + 1;
      TrackConstantBufferBinding(pDevice, shader_type, state_slot, resource,
                                 cb.buffer_size);
      if (shader_type == MESA_SHADER_COMPUTE)
         pDevice->constant_buffer_resources[shader_type][StartBuffer + i] =
            resource;
      pDevice->constant_buffer_resources[shader_type][state_slot] = resource;
      pDevice->constant_buffer_offsets[shader_type][state_slot] =
         resource_offset;
      pDevice->constant_buffer_sizes[shader_type][state_slot] = cb.buffer_size;
      pDevice->constant_buffer_published[shader_type][state_slot] = published;
      pipe_resource_reference(
         &pDevice->constant_buffers[shader_type][state_slot],
         resource_buffer);
      if (TraceConstantBufferBinds()) {
         ResourceEvent(RESOURCE_EVENT_SET_CONSTANT_BUFFER,
                       (uint64_t)(uintptr_t)(phBuffers
                                                ? phBuffers[i].pDrvPrivate
                                                : NULL),
                       phBuffers ? CastResource(phBuffers[i]) : NULL,
                       resource_buffer,
                       PipeResourceRefCount(resource_buffer),
                       shader_type,
                       state_slot,
                       cb.buffer_size);
      }
      pipe->set_constant_buffer(pipe,
                                shader_type,
                                state_slot,
                                &cb);
   }
}

static void
SetConstantBuffers(mesa_shader_stage shader_type,    // IN
                   D3D10DDI_HDEVICE hDevice,             // IN
                   UINT StartBuffer,                     // IN
                   UINT NumBuffers,                      // IN
                   const D3D10DDI_HRESOURCE *phBuffers) // IN
{
   SetConstantBuffersRange(shader_type, hDevice, StartBuffer, NumBuffers,
                           phBuffers, NULL, NULL);
}


/*
 * ----------------------------------------------------------------------
 *
 * SetSamplers --
 *
 *    Update the driver's currently bound sampler state.
 *
 * ----------------------------------------------------------------------
 */

static void
SetSamplers(mesa_shader_stage shader_type,     // IN
            D3D10DDI_HDEVICE hDevice,              // IN
            UINT Offset,                          // IN
            UINT NumSamplers,                       // IN
            const D3D10DDI_HSAMPLER *phSamplers)  // IN
{
   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;

   void **samplers = pDevice->samplers[shader_type];
   for (UINT i = 0; i < NumSamplers; i++) {
      assert(Offset + i < PIPE_MAX_SAMPLERS);
      samplers[Offset + i] = CastPipeSamplerState(phSamplers[i]);
   }

   pipe->bind_sampler_states(pipe, shader_type, 0, PIPE_MAX_SAMPLERS, samplers);
}

static void
FillBufferInfoRecord(uint32_t record[D3D10UMD_BUFINFO_RECORD_DWORDS],
                     bool raw, bool structured, UINT num_elements,
                     UINT stride)
{
   memset(record, 0, D3D10UMD_BUFINFO_RECORD_DWORDS * sizeof(record[0]));

   if (raw) {
      record[0] = record[1] = record[2] = record[3] =
         num_elements * (UINT)sizeof(uint32_t);
   } else if (structured) {
      record[0] = num_elements;
      record[1] = stride;
      record[2] = 0;
      record[3] = 1;
   } else {
      record[0] = record[1] = record[2] = record[3] = num_elements;
   }
}

static bool
StoreBufferInfoRecord(uint32_t record[D3D10UMD_BUFINFO_RECORD_DWORDS],
                      bool valid, bool raw, bool structured,
                      UINT num_elements, UINT stride)
{
   uint32_t next[D3D10UMD_BUFINFO_RECORD_DWORDS] = {};
   if (valid)
      FillBufferInfoRecord(next, raw, structured, num_elements, stride);

   if (record[0] == next[0] && record[1] == next[1] &&
       record[2] == next[2] && record[3] == next[3])
      return false;

   memcpy(record, next, sizeof(next));
   return true;
}

static void
UpdateBufferInfoSrvConstants(Device *pDevice,
                             mesa_shader_stage shader_type,
                             unsigned first_slot,
                             unsigned num_slots)
{
   if (!pDevice || shader_type >= MESA_SHADER_STAGES ||
       first_slot >= PIPE_MAX_SHADER_SAMPLER_VIEWS)
      return;

   const unsigned end =
      MIN2(first_slot + num_slots, PIPE_MAX_SHADER_SAMPLER_VIEWS);
   for (unsigned slot = first_slot; slot < end; ++slot) {
      ShaderResourceView *srv = CastShaderResourceView(
         pDevice->shader_resource_views[shader_type][slot]);
      const bool valid =
         srv && srv->handle && srv->handle->texture &&
         srv->handle->texture->target == PIPE_BUFFER;
      uint32_t *record =
         &pDevice->bufinfo_constants[shader_type]
             [(D3D10UMD_BUFINFO_SRV_RECORD_BASE + slot) *
              D3D10UMD_BUFINFO_RECORD_DWORDS];
      if (StoreBufferInfoRecord(record, valid,
                                valid && srv->buffer_raw,
                                valid && srv->buffer_structured,
                                valid ? srv->buffer_num_elements : 0,
                                valid ? srv->buffer_stride : 0))
         pDevice->bufinfo_constants_dirty[shader_type] = true;
   }
}

void
UpdateBufferInfoUavConstants(Device *pDevice,
                             mesa_shader_stage shader_type,
                             unsigned first_slot,
                             unsigned num_slots)
{
   if (!pDevice || shader_type >= MESA_SHADER_STAGES ||
       first_slot >= PIPE_MAX_SHADER_IMAGES)
      return;

   const unsigned end = MIN2(first_slot + num_slots, PIPE_MAX_SHADER_IMAGES);
   for (unsigned slot = first_slot; slot < end; ++slot) {
      UnorderedAccessView *uav =
         pDevice->unordered_access_views[shader_type][slot];
      const bool valid =
         uav && uav->pipe_resource &&
         uav->pipe_resource->target == PIPE_BUFFER;
      uint32_t *record =
         &pDevice->bufinfo_constants[shader_type]
             [(D3D10UMD_BUFINFO_UAV_RECORD_BASE + slot) *
              D3D10UMD_BUFINFO_RECORD_DWORDS];
      if (StoreBufferInfoRecord(record, valid,
                                valid && uav->buffer_raw,
                                valid && uav->buffer_structured,
                                valid ? uav->buffer_num_elements : 0,
                                valid ? uav->buffer_stride : 0))
         pDevice->bufinfo_constants_dirty[shader_type] = true;
   }
}

static unsigned
GetSurfaceSampleCount(const struct pipe_surface *surface)
{
   if (!surface || !surface->texture)
      return 0;

   if (surface->nr_samples)
      return surface->nr_samples;

   return MAX2(surface->texture->nr_samples, 1u);
}

static unsigned
GetFramebufferSampleCount(Device *pDevice)
{
   if (!pDevice)
      return 1;

   for (unsigned i = 0; i < pDevice->fb.nr_cbufs; ++i) {
      unsigned samples = GetSurfaceSampleCount(&pDevice->fb.cbufs[i]);
      if (samples)
         return samples;
   }

   unsigned samples = GetSurfaceSampleCount(&pDevice->fb.zsbuf);
   return samples ? samples : 1;
}

void
UpdateBufferInfoSampleConstants(Device *pDevice,
                                mesa_shader_stage shader_type)
{
   if (!pDevice || shader_type >= MESA_SHADER_STAGES)
      return;

   uint32_t *record =
      &pDevice->bufinfo_constants[shader_type]
          [D3D10UMD_DRIVER_SAMPLE_INFO_RECORD *
           D3D10UMD_BUFINFO_RECORD_DWORDS];
   const uint32_t samples = GetFramebufferSampleCount(pDevice);
   if (record[0] == samples && record[1] == 0 &&
       record[2] == 0 && record[3] == 0)
      return;

   record[0] = samples;
   record[1] = 0;
   record[2] = 0;
   record[3] = 0;
   pDevice->bufinfo_constants_dirty[shader_type] = true;
}

void
UpdateBufferInfoConstants(Device *pDevice, mesa_shader_stage shader_type)
{
   if (!pDevice || !pDevice->pipe || shader_type >= MESA_SHADER_STAGES)
      return;

   uint32_t *data = pDevice->bufinfo_constants[shader_type];
   if (!pDevice->bufinfo_constants_bound[shader_type])
      UpdateBufferInfoSampleConstants(pDevice, shader_type);

   if (pDevice->bufinfo_constants_bound[shader_type] &&
       !pDevice->bufinfo_constants_dirty[shader_type])
      return;

   struct pipe_constant_buffer cb = {};
   cb.user_buffer = data;
   cb.buffer_size = D3D10UMD_BUFINFO_CB_SIZE;
   pDevice->pipe->set_constant_buffer(pDevice->pipe, shader_type,
                                      D3D10UMD_DRIVER_BUFINFO_CB_SLOT, &cb);
   pDevice->bufinfo_constants_bound[shader_type] = true;
   pDevice->bufinfo_constants_dirty[shader_type] = false;
}


/*
 * ----------------------------------------------------------------------
 *
 * SetShaderResources --
 *
 *    Update the driver's currently shader resources.
 *
 * ----------------------------------------------------------------------
 */

static void
SetShaderResources(mesa_shader_stage shader_type,                  // IN
                   D3D10DDI_HDEVICE hDevice,                                   // IN
                   UINT Offset,                                                // IN
                   UINT NumViews,                                              // IN
                   const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews)  // IN
{
   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;

   assert(Offset + NumViews <= D3D10_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT);

   D3D10DDI_HSHADERRESOURCEVIEW *shader_resource_views =
      pDevice->shader_resource_views[shader_type];
   struct pipe_sampler_view **sampler_views = pDevice->sampler_views[shader_type];
   for (UINT i = 0; i < NumViews; i++) {
      struct pipe_sampler_view *sampler_view =
            GetPipeShaderResourceView(phShaderResourceViews[i]);
      ResourceEvent(RESOURCE_EVENT_SET_SHADER_RESOURCES,
                    (uint64_t)(uintptr_t)(phShaderResourceViews
                                             ? phShaderResourceViews[i].pDrvPrivate
                                             : NULL),
                    sampler_view,
                    sampler_view ? sampler_view->texture : NULL,
                    sampler_view ?
                       PipeResourceRefCount(sampler_view->texture) : 0,
                    shader_type,
                    Offset + i,
                    0);
      if (Offset + i < PIPE_MAX_SHADER_SAMPLER_VIEWS) {
         shader_resource_views[Offset + i] = phShaderResourceViews[i];
         sampler_views[Offset + i] = sampler_view;
      } else {
         if (sampler_view) {
            LOG_UNSUPPORTED(true);
            break;
         }
      }
   }

   /* Gallium's range semantics match the D3D DDI here.  Re-sending all 128
    * slots made u_threaded_context copy roughly 1 KiB for every resource-bind
    * call; on Superposition this was 2.17 GiB, 55% of the entire worker
    * stream.  Keep the experiment process-local until the worker is fully
    * qualified.
    */
   if (OrderedContextWorkerEnabled()) {
      pipe->set_sampler_views(pipe, shader_type, Offset, NumViews, 0,
                              &sampler_views[Offset]);
   } else {
      pipe->set_sampler_views(pipe, shader_type, 0,
                              PIPE_MAX_SHADER_SAMPLER_VIEWS, 0,
                              sampler_views);
   }
   UpdateBufferInfoSrvConstants(pDevice, shader_type, Offset, NumViews);
   UpdateBufferInfoConstants(pDevice, shader_type);
}

void
RefreshBoundShaderResourceViews(Device *pDevice)
{
   if (!pDevice->shader_resource_views_dirty) {
      return;
   }

   struct pipe_context *pipe = pDevice->pipe;
   pDevice->shader_resource_views_dirty = false;

   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; ++stage) {
      bool changed = false;
      unsigned first_changed = PIPE_MAX_SHADER_SAMPLER_VIEWS;
      unsigned last_changed = 0;
      struct pipe_sampler_view **sampler_views =
         pDevice->sampler_views[stage];

      for (unsigned slot = 0; slot < PIPE_MAX_SHADER_SAMPLER_VIEWS; ++slot) {
         D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView =
            pDevice->shader_resource_views[stage][slot];
         if (!hShaderResourceView.pDrvPrivate) {
            continue;
         }

         struct pipe_sampler_view *old_sampler_view = sampler_views[slot];
         struct pipe_sampler_view *new_sampler_view =
            GetPipeShaderResourceView(hShaderResourceView);
         if (new_sampler_view != old_sampler_view) {
            sampler_views[slot] = new_sampler_view;
            changed = true;
            first_changed = MIN2(first_changed, slot);
            last_changed = slot;
         }
      }

      if (changed) {
         if (OrderedContextWorkerEnabled()) {
            pipe->set_sampler_views(pipe, (mesa_shader_stage)stage,
                                    first_changed,
                                    last_changed - first_changed + 1, 0,
                                    &sampler_views[first_changed]);
         } else {
            pipe->set_sampler_views(pipe, (mesa_shader_stage)stage, 0,
                                    PIPE_MAX_SHADER_SAMPLER_VIEWS, 0,
                                    sampler_views);
         }
         UpdateBufferInfoConstants(pDevice, (mesa_shader_stage)stage);
      }
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateShaderSize --
 *
 *    The CalcPrivateShaderSize function determines the size of
 *    the user-mode display driver's private region of memory
 *    (that is, the size of internal driver structures, not the
 *    size of the resource video memory) for a shader.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateShaderSize(D3D10DDI_HDEVICE hDevice,                                  // IN
                      __in_ecount (pShaderCode[1]) const UINT *pShaderCode,      // IN
                      __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures)   // IN
{
   return sizeof(Shader);
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyShader --
 *
 *    The DestroyShader function destroys the specified shader object.
 *    The shader object can be destoyed only if it is not currently
 *    bound to a display device.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroyShader(D3D10DDI_HDEVICE hDevice,   // IN
              D3D10DDI_HSHADER hShader)   // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   Shader *pShader = CastShader(hShader);

   if (pShader->handle) {
      switch (pShader->type) {
      case MESA_SHADER_FRAGMENT:
         pipe->delete_fs_state(pipe, pShader->handle);
         break;
      case MESA_SHADER_VERTEX:
         pipe->delete_vs_state(pipe, pShader->handle);
         break;
      case MESA_SHADER_GEOMETRY:
         pipe->delete_gs_state(pipe, pShader->handle);
         break;
      case MESA_SHADER_TESS_CTRL:
         pipe->delete_tcs_state(pipe, pShader->handle);
         break;
      case MESA_SHADER_TESS_EVAL:
         pipe->delete_tes_state(pipe, pShader->handle);
         break;
      case MESA_SHADER_COMPUTE:
         pipe->delete_compute_state(pipe, pShader->handle);
         break;
      default:
         assert(0);
      }
   }

   if (pShader->state.tokens) {
      ureg_free_tokens(pShader->state.tokens);
   }
   if (pShader->type == MESA_SHADER_TESS_CTRL ||
       pShader->type == MESA_SHADER_TESS_EVAL)
      FREE(pShader->tessellation_code);
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateSamplerSize --
 *
 *    The CalcPrivateSamplerSize function determines the size of the
 *    user-mode display driver's private region of memory (that is,
 *    the size of internal driver structures, not the size of the
 *    resource video memory) for a sampler.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateSamplerSize(D3D10DDI_HDEVICE hDevice,                        // IN
                       __in const D3D10_DDI_SAMPLER_DESC *pSamplerDesc) // IN
{
   return sizeof(SamplerState);
}


static uint
translate_address_mode(D3D10_DDI_TEXTURE_ADDRESS_MODE AddressMode)
{
   switch (AddressMode) {
   case D3D10_DDI_TEXTURE_ADDRESS_WRAP:
      return PIPE_TEX_WRAP_REPEAT;
   case D3D10_DDI_TEXTURE_ADDRESS_MIRROR:
      return PIPE_TEX_WRAP_MIRROR_REPEAT;
   case D3D10_DDI_TEXTURE_ADDRESS_CLAMP:
      return PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   case D3D10_DDI_TEXTURE_ADDRESS_BORDER:
      return PIPE_TEX_WRAP_CLAMP_TO_BORDER;
   case D3D10_DDI_TEXTURE_ADDRESS_MIRRORONCE:
      return PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE;
   default:
      assert(0);
      return PIPE_TEX_WRAP_REPEAT;
   }
}

static uint
translate_comparison(D3D10_DDI_COMPARISON_FUNC Func)
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

static uint
translate_filter(D3D10_DDI_FILTER_TYPE Filter)
{
   switch (Filter) {
   case D3D10_DDI_FILTER_TYPE_POINT:
      return PIPE_TEX_FILTER_NEAREST;
   case D3D10_DDI_FILTER_TYPE_LINEAR:
      return PIPE_TEX_FILTER_LINEAR;
   default:
      assert(0);
      return PIPE_TEX_FILTER_NEAREST;
   }
}

static uint
translate_min_filter(D3D10_DDI_FILTER Filter)
{
   return translate_filter(D3D10_DDI_DECODE_MIN_FILTER(Filter));
}

static uint
translate_mag_filter(D3D10_DDI_FILTER Filter)
{
   return translate_filter(D3D10_DDI_DECODE_MAG_FILTER(Filter));
}

/* Gallium uses a different enum for mipfilters, to accomodate the GL
 * MIPFILTER_NONE mode.
 */
static uint
translate_mip_filter(D3D10_DDI_FILTER Filter)
{
   switch (D3D10_DDI_DECODE_MIP_FILTER(Filter)) {
   case D3D10_DDI_FILTER_TYPE_POINT:
      return PIPE_TEX_MIPFILTER_NEAREST;
   case D3D10_DDI_FILTER_TYPE_LINEAR:
      return PIPE_TEX_MIPFILTER_LINEAR;
   default:
      assert(0);
      return PIPE_TEX_MIPFILTER_NEAREST;
   }
}

/*
 * ----------------------------------------------------------------------
 *
 * CreateSampler --
 *
 *    The CreateSampler function creates a sampler.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateSampler(D3D10DDI_HDEVICE hDevice,                        // IN
              __in const D3D10_DDI_SAMPLER_DESC *pSamplerDesc, // IN
              D3D10DDI_HSAMPLER hSampler,                      // IN
              D3D10DDI_HRTSAMPLER hRTSampler)                  // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   SamplerState *pSamplerState = CastSamplerState(hSampler);

   struct pipe_sampler_state state;

   pSamplerState->handle = NULL;
   memset(&state, 0, sizeof state);

   /* d3d10 has seamless cube filtering always enabled */
   state.seamless_cube_map = 1;

   /* Wrapping modes. */
   state.wrap_s = translate_address_mode(pSamplerDesc->AddressU);
   state.wrap_t = translate_address_mode(pSamplerDesc->AddressV);
   state.wrap_r = translate_address_mode(pSamplerDesc->AddressW);

   /* Filtering */
   state.min_img_filter = translate_min_filter(pSamplerDesc->Filter);
   state.mag_img_filter = translate_mag_filter(pSamplerDesc->Filter);
   state.min_mip_filter = translate_mip_filter(pSamplerDesc->Filter);

   if (D3D10_DDI_DECODE_IS_ANISOTROPIC_FILTER(pSamplerDesc->Filter)) {
      state.max_anisotropy = pSamplerDesc->MaxAnisotropy;
   }

   /* XXX: Handle the following bit.
    */
   LOG_UNSUPPORTED(D3D10_DDI_DECODE_IS_TEXT_1BIT_FILTER(pSamplerDesc->Filter));

   /* Comparison. */
   if (D3D10_DDI_DECODE_IS_COMPARISON_FILTER(pSamplerDesc->Filter)) {
      state.compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE;
      state.compare_func = translate_comparison(pSamplerDesc->ComparisonFunc);
   }

   /* Level of detail. */
   state.lod_bias = pSamplerDesc->MipLODBias;
   state.min_lod = pSamplerDesc->MinLOD;
   state.max_lod = pSamplerDesc->MaxLOD;

   /* Border color. */
   state.border_color.f[0] = pSamplerDesc->BorderColor[0];
   state.border_color.f[1] = pSamplerDesc->BorderColor[1];
   state.border_color.f[2] = pSamplerDesc->BorderColor[2];
   state.border_color.f[3] = pSamplerDesc->BorderColor[3];

   pSamplerState->handle = pipe->create_sampler_state(pipe, &state);
   if (!pSamplerState->handle) {
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=sampler reason=native-state-unavailable\n");
      SetError(hDevice, E_OUTOFMEMORY);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroySampler --
 *
 *    The DestroySampler function destroys the specified sampler object.
 *    The sampler object can be destoyed only if it is not currently
 *    bound to a display device.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroySampler(D3D10DDI_HDEVICE hDevice,     // IN
               D3D10DDI_HSAMPLER hSampler)   // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   SamplerState *pSamplerState = CastSamplerState(hSampler);

   pipe->delete_sampler_state(pipe, pSamplerState->handle);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateVertexShader --
 *
 *    The CreateVertexShader function creates a vertex shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateVertexShader(D3D10DDI_HDEVICE hDevice,                                  // IN
                   __in_ecount (pShaderCode[1]) const UINT *pCode,            // IN
                   D3D10DDI_HSHADER hShader,                                  // IN
                   D3D10DDI_HRTSHADER hRTShader,                              // IN
                   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures)   // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);

   InitShaderObject(pShader, MESA_SHADER_VERTEX);
   struct pipe_shader_sampler_binding_map *sampler_binding_map =
      PrepareSamplerBindingMap(pDevice, pShader);
   pShader->compute_emulation =
      GetComputeEmulation(pCode, pShader->compute_store_imm);

   pShader->state.tokens =
      Shader_tgsi_translate(pCode, pShader->output_mapping, NULL, NULL,
                            NULL, NULL, pSignatures, sampler_binding_map,
                            pShader->counter_image_slots);
   if (!pShader->state.tokens) {
      YTTRIUM_WARN("yttrium: shader creation failed owner=d3d10umd stage=vs "
                   "reason=tgsi_translation_failed shader=%p\n",
                   (void *)pShader);
      SetError(hDevice, E_FAIL);
      return;
   }

   pShader->handle = pipe->create_vs_state(pipe, &pShader->state);
   if (!pShader->handle) {
      YTTRIUM_WARN("yttrium: shader creation failed owner=d3d10umd stage=vs "
                   "reason=native_state_unavailable shader=%p\n",
                   (void *)pShader);
      ureg_free_tokens(pShader->state.tokens);
      pShader->state.tokens = NULL;
      SetError(hDevice, E_FAIL);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * VsSetShader --
 *
 *    The VsSetShader function sets the vertex shader code so that all
 *    of the subsequent drawing operations use that code.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
VsSetShader(D3D10DDI_HDEVICE hDevice,  // IN
            D3D10DDI_HSHADER hShader)  // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);
   void *state = CastPipeShader(hShader);

   pDevice->bound_vs = pShader;
   if (!state) {
      state = pDevice->empty_vs;
   }

   /* A bound tokenless stream-output GS is materialized by the backend as a
    * variant of the active VS.  Drop that materialization before changing
    * the VS: otherwise bind_vs_state can eagerly rebuild the old stream-
    * output layout against the new VS, before ResolveState has remapped the
    * declaration for it.  ResolveState will bind the correctly remapped
    * placeholder before the next draw.
    */
   if (pDevice->bound_empty_gs)
      pipe->bind_gs_state(pipe, NULL);

   pipe->bind_vs_state(pipe, state);
   BindShaderImages(pDevice, MESA_SHADER_VERTEX);
   UpdateBufferInfoConstants(pDevice, MESA_SHADER_VERTEX);
}


/*
 * ----------------------------------------------------------------------
 *
 * VsSetShaderResources --
 *
 *    The VsSetShaderResources function sets resources for a
 *    vertex shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
VsSetShaderResources(D3D10DDI_HDEVICE hDevice,                                   // IN
                     UINT Offset,                                                // IN
                     UINT NumViews,                                              // IN
                     __in_ecount (NumViews)
                     const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews)  // IN
{
   LOG_ENTRYPOINT();

   SetShaderResources(MESA_SHADER_VERTEX, hDevice, Offset, NumViews, phShaderResourceViews);

}


/*
 * ----------------------------------------------------------------------
 *
 * VsSetConstantBuffers --
 *
 *    The VsSetConstantBuffers function sets constant buffers
 *    for a vertex shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
VsSetConstantBuffers(D3D10DDI_HDEVICE hDevice,                                      // IN
                     UINT StartBuffer,                                              // IN
                     UINT NumBuffers,                                               // IN
                     __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers)  // IN
{
   LOG_ENTRYPOINT();

   SetConstantBuffers(MESA_SHADER_VERTEX,
                      hDevice, StartBuffer, NumBuffers, phBuffers);
}


/*
 * ----------------------------------------------------------------------
 *
 * VsSetSamplers --
 *
 *    The VsSetSamplers function sets samplers for a vertex shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
VsSetSamplers(D3D10DDI_HDEVICE hDevice,                                       // IN
              UINT Offset,                                                    // IN
              UINT NumSamplers,                                               // IN
              __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers)  // IN
{
   LOG_ENTRYPOINT();

   SetSamplers(MESA_SHADER_VERTEX, hDevice, Offset, NumSamplers, phSamplers);

}


/*
 * ----------------------------------------------------------------------
 *
 * CreateGeometryShader --
 *
 *    The CreateGeometryShader function creates a geometry shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateGeometryShader(D3D10DDI_HDEVICE hDevice,                                // IN
                     __in_ecount (pShaderCode[1]) const UINT *pShaderCode,    // IN
                     D3D10DDI_HSHADER hShader,                                // IN
                     D3D10DDI_HRTSHADER hRTShader,                            // IN
                     __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures) // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);

   InitShaderObject(pShader, MESA_SHADER_GEOMETRY);
   struct pipe_shader_sampler_binding_map *sampler_binding_map =
      PrepareSamplerBindingMap(pDevice, pShader);
   pShader->state.tokens =
      Shader_tgsi_translate(pShaderCode, pShader->output_mapping, NULL, NULL,
                            NULL, NULL, pSignatures, sampler_binding_map,
                            pShader->counter_image_slots);
   if (!pShader->state.tokens) {
      YTTRIUM_WARN("yttrium: shader creation failed owner=d3d10umd stage=gs "
                   "reason=tgsi_translation_failed shader=%p\n",
                   (void *)pShader);
      SetError(hDevice, E_FAIL);
      return;
   }

   pShader->handle = pipe->create_gs_state(pipe, &pShader->state);
   if (!pShader->handle) {
      YTTRIUM_WARN("yttrium: shader creation failed owner=d3d10umd stage=gs "
                   "reason=native_state_unavailable shader=%p\n",
                   (void *)pShader);
      ureg_free_tokens(pShader->state.tokens);
      pShader->state.tokens = NULL;
      SetError(hDevice, E_FAIL);
   }
}

SIZE_T APIENTRY
CalcPrivateGeometryShaderWithStreamOutput11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pData,
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   return sizeof(Shader);
}

void APIENTRY
CreateGeometryShaderWithStreamOutput11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pData,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures)
{
   D3D10DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT create10 = {};
   D3D10DDIARG_STREAM_OUTPUT_DECLARATION_ENTRY output_decl[PIPE_MAX_SO_OUTPUTS];

   InitShaderObject(CastShader(hShader), MESA_SHADER_GEOMETRY);

   if (pData->NumEntries > ARRAY_SIZE(output_decl) ||
       (pData->NumEntries && !pData->pOutputStreamDecl)) {
      YTTRIUM_WARN("yttrium: d3d10umd geometry stream-output creation "
                   "rejected reason=invalid-d3d11-declaration entries=%u "
                   "declaration=%p\n",
                   pData->NumEntries, (void *)pData->pOutputStreamDecl);
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   for (unsigned i = 0; i < pData->NumEntries; ++i) {
      output_decl[i].OutputSlot = pData->pOutputStreamDecl[i].OutputSlot;
      output_decl[i].RegisterIndex = pData->pOutputStreamDecl[i].RegisterIndex;
      output_decl[i].RegisterMask = pData->pOutputStreamDecl[i].RegisterMask;
   }

   create10.pShaderCode = pData->pShaderCode;
   create10.pOutputStreamDecl = output_decl;
   create10.NumEntries = pData->NumEntries;
   if (pData->BufferStridesInBytes && pData->NumStrides)
      create10.StreamOutputStrideInBytes = pData->BufferStridesInBytes[0];

   CreateGeometryShaderWithStreamOutput(hDevice, &create10, hShader,
                                        hRTShader, pSignatures);
   CastShader(hShader)->no_rasterized_stream =
      pData->RasterizedStream == ~0u;
}


/*
 * ----------------------------------------------------------------------
 *
 * GsSetShader --
 *
 *    The GsSetShader function sets the geometry shader code so that
 *    all of the subsequent drawing operations use that code.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
GsSetShader(D3D10DDI_HDEVICE hDevice,  // IN
            D3D10DDI_HSHADER hShader)  // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = CastPipeContext(hDevice);
   void *state = CastPipeShader(hShader);
   Shader *pShader = CastShader(hShader);

   assert(pipe->bind_gs_state);

   pDevice->bound_gs = pShader && pShader->state.tokens ? pShader : NULL;

   if (pShader && !pShader->state.tokens) {
      pDevice->bound_empty_gs = pShader;
   } else {
      pDevice->bound_empty_gs = NULL;
      pipe->bind_gs_state(pipe, state);
   }
   BindShaderImages(pDevice, MESA_SHADER_GEOMETRY);
   UpdateBufferInfoConstants(pDevice, MESA_SHADER_GEOMETRY);
   ApplyRasterizerState(pDevice);
}


/*
 * ----------------------------------------------------------------------
 *
 * GsSetShaderResources --
 *
 *    The GsSetShaderResources function sets resources for a
 *    geometry shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
GsSetShaderResources(D3D10DDI_HDEVICE hDevice,                                   // IN
                     UINT Offset,                                                // IN
                     UINT NumViews,                                              // IN
                     __in_ecount (NumViews)
                     const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews)  // IN
{
   LOG_ENTRYPOINT();

   SetShaderResources(MESA_SHADER_GEOMETRY, hDevice, Offset, NumViews, phShaderResourceViews);
}


/*
 * ----------------------------------------------------------------------
 *
 * GsSetConstantBuffers --
 *
 *    The GsSetConstantBuffers function sets constant buffers for
 *    a geometry shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
GsSetConstantBuffers(D3D10DDI_HDEVICE hDevice,                                      // IN
                     UINT StartBuffer,                                              // IN
                     UINT NumBuffers,                                               // IN
                     __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers)  // IN
{
   LOG_ENTRYPOINT();

   SetConstantBuffers(MESA_SHADER_GEOMETRY,
                      hDevice, StartBuffer, NumBuffers, phBuffers);
}


/*
 * ----------------------------------------------------------------------
 *
 * GsSetSamplers --
 *
 *    The GsSetSamplers function sets samplers for a geometry shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
GsSetSamplers(D3D10DDI_HDEVICE hDevice,                                       // IN
              UINT Offset,                                                    // IN
              UINT NumSamplers,                                               // IN
              __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers)  // IN
{
   LOG_ENTRYPOINT();

   SetSamplers(MESA_SHADER_GEOMETRY, hDevice, Offset, NumSamplers, phSamplers);
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateGeometryShaderWithStreamOutput --
 *
 *    The CalcPrivateGeometryShaderWithStreamOutput function determines
 *    the size of the user-mode display driver's private region of memory
 *    (that is, the size of internal driver structures, not the size of
 *    the resource video memory) for a geometry shader with stream output.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateGeometryShaderWithStreamOutput(
   D3D10DDI_HDEVICE hDevice,                                                                             // IN
   __in const D3D10DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pCreateGeometryShaderWithStreamOutput,   // IN
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures)                                              // IN
{
   LOG_ENTRYPOINT();
   return sizeof(Shader);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateGeometryShaderWithStreamOutput --
 *
 *    The CreateGeometryShaderWithStreamOutput function creates a
 *    geometry shader with stream output.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateGeometryShaderWithStreamOutput(
   D3D10DDI_HDEVICE hDevice,                                                                             // IN
   __in const D3D10DDIARG_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT *pData,   // IN
   D3D10DDI_HSHADER hShader,                                                                             // IN
   D3D10DDI_HRTSHADER hRTShader,                                                                         // IN
   __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures)                                              // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);
   int total_components[PIPE_MAX_SO_BUFFERS] = {0};
   unsigned num_holes = 0;
   bool all_slot_zero = true;

   InitShaderObject(pShader, MESA_SHADER_GEOMETRY);
   struct pipe_shader_sampler_binding_map *sampler_binding_map =
      PrepareSamplerBindingMap(pDevice, pShader);
   if (pData->NumEntries > PIPE_MAX_SO_OUTPUTS ||
       (pData->NumEntries && !pData->pOutputStreamDecl)) {
      YTTRIUM_WARN("yttrium: d3d10umd geometry stream-output creation "
                   "rejected reason=invalid_declaration entries=%u\n",
                   pData->NumEntries);
      SetError(hDevice, E_INVALIDARG);
      return;
   }
   if (!pData->pShaderCode &&
       !CaptureTokenlessStreamOutputSignature(pShader, pSignatures)) {
      SetError(hDevice, E_FAIL);
      return;
   }
   if (pData->pShaderCode) {
      pShader->state.tokens =
         Shader_tgsi_translate(pData->pShaderCode, pShader->output_mapping,
                               NULL, NULL, NULL, NULL, pSignatures,
                               sampler_binding_map, pShader->counter_image_slots);
      if (!pShader->state.tokens) {
         YTTRIUM_WARN("yttrium: d3d10umd geometry stream-output shader translation failed; creation rejected\n");
         SetError(hDevice, E_FAIL);
         return;
      }
   }
   for (unsigned i = 0; i < pData->NumEntries; ++i) {
      CONST D3D10DDIARG_STREAM_OUTPUT_DECLARATION_ENTRY* pOutputStreamDecl =
            &pData->pOutputStreamDecl[i];
      if (pOutputStreamDecl->OutputSlot >= PIPE_MAX_SO_BUFFERS) {
         YTTRIUM_WARN("yttrium: d3d10umd geometry stream-output creation "
                      "rejected entry=%u output_slot=%u "
                      "reason=output_slot_out_of_range\n",
                      i, pOutputStreamDecl->OutputSlot);
         if (pShader->state.tokens) {
            ureg_free_tokens(pShader->state.tokens);
            pShader->state.tokens = NULL;
         }
         SetError(hDevice, E_INVALIDARG);
         return;
      }
      BYTE RegisterMask = pOutputStreamDecl->RegisterMask;
      unsigned start_component = 0;
      unsigned num_components = 0;
      if (!RegisterMask &&
          pOutputStreamDecl->RegisterIndex != 0xffffffff) {
         RegisterMask = 0xf;
      }
      const BYTE EffectiveRegisterMask = RegisterMask;
      if (RegisterMask) {
         while ((RegisterMask & 1) == 0) {
            ++start_component;
            RegisterMask >>= 1;
         }
         while (RegisterMask) {
            ++num_components;
            RegisterMask >>= 1;
         }
         if (start_component >= 4 || num_components > 4 ||
             (((1u << num_components) - 1) << start_component) !=
                EffectiveRegisterMask) {
            YTTRIUM_WARN("yttrium: d3d10umd geometry stream-output "
                         "creation rejected entry=%u register_mask=0x%x "
                         "reason=noncontiguous-or-invalid-mask\n",
                         i, EffectiveRegisterMask);
            if (pShader->state.tokens) {
               ureg_free_tokens(pShader->state.tokens);
               pShader->state.tokens = NULL;
            }
            SetError(hDevice, E_INVALIDARG);
            return;
         }
      }

      if (pOutputStreamDecl->RegisterIndex == 0xffffffff) {
         ++num_holes;
      } else {
         unsigned idx = i - num_holes;
         const unsigned d3d_register = pOutputStreamDecl->RegisterIndex;
         if (d3d_register >= PIPE_MAX_SHADER_OUTPUTS) {
            YTTRIUM_WARN("yttrium: d3d10umd geometry stream-output shader "
                         "creation rejected output=%u d3d_register=%u "
                         "reason=register_out_of_range\n",
                         idx, d3d_register);
            if (pShader->state.tokens) {
               ureg_free_tokens(pShader->state.tokens);
               pShader->state.tokens = NULL;
            }
            SetError(hDevice, E_FAIL);
            return;
         }
         if (!pData->pShaderCode &&
             (EffectiveRegisterMask &
              ~pShader->stream_output_signature_masks[d3d_register])) {
            YTTRIUM_WARN("yttrium: d3d10umd tokenless geometry "
                         "stream-output creation rejected output=%u "
                         "d3d_register=%u declaration_mask=0x%x "
                         "signature_mask=0x%x "
                         "reason=declaration_not_in_input_signature\n",
                         idx, d3d_register, EffectiveRegisterMask,
                         pShader->stream_output_signature_masks[d3d_register]);
            SetError(hDevice, E_FAIL);
            return;
         }
         pShader->state.stream_output.output[idx].start_component =
            start_component;
         pShader->state.stream_output.output[idx].num_components =
            num_components;
         pShader->state.stream_output.output[idx].output_buffer =
            pOutputStreamDecl->OutputSlot;
         pShader->stream_output_d3d_registers[idx] = d3d_register;
         if (pShader->state.tokens) {
            const unsigned mapping =
               ShaderFindOutputMapping(pShader, d3d_register);
            if (mapping == ~0u) {
               YTTRIUM_WARN("yttrium: d3d10umd geometry stream-output shader "
                            "creation rejected output=%u d3d_register=%u "
                            "reason=output_not_declared\n",
                            idx, d3d_register);
               ureg_free_tokens(pShader->state.tokens);
               pShader->state.tokens = NULL;
               SetError(hDevice, E_FAIL);
               return;
            }
            pShader->state.stream_output.output[idx].register_index = mapping;
         } else {
            /* A tokenless GS inherits the currently bound VS.  Keep the D3D
             * register here until ResolveState can map it against that VS.
             */
            pShader->state.stream_output.output[idx].register_index =
               d3d_register;
         }
         pShader->state.stream_output.output[idx].dst_offset =
            total_components[pOutputStreamDecl->OutputSlot];
         if (pOutputStreamDecl->OutputSlot != 0)
            all_slot_zero = false;
      }
      total_components[pOutputStreamDecl->OutputSlot] += num_components;
   }
   pShader->state.stream_output.num_outputs = pData->NumEntries - num_holes;
   for (unsigned i = 0; i < PIPE_MAX_SO_BUFFERS; ++i) {
      /* stream_output.stride[i] is in dwords */
      if (all_slot_zero) {
         pShader->state.stream_output.stride[i] =
            pData->StreamOutputStrideInBytes ?
            pData->StreamOutputStrideInBytes / sizeof(float) :
            total_components[i];
      } else {
         pShader->state.stream_output.stride[i] = total_components[i];
      }
   }
   pShader->stream_output_template = pShader->state.stream_output;

   pShader->handle = pipe->create_gs_state(pipe, &pShader->state);
   if (!pShader->handle) {
      YTTRIUM_WARN("yttrium: shader creation failed owner=d3d10umd stage=gs_so "
                   "reason=native_state_unavailable shader=%p\n",
                   (void *)pShader);
      if (pShader->state.tokens) {
         ureg_free_tokens(pShader->state.tokens);
         pShader->state.tokens = NULL;
      }
      SetError(hDevice, E_FAIL);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * SoSetTargets --
 *
 *    The SoSetTargets function sets stream output target resources.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
SoSetTargets(D3D10DDI_HDEVICE hDevice,                                     // IN
             UINT SOTargets,                                               // IN
             UINT ClearTargets,                                            // IN
             __in_ecount (SOTargets) const D3D10DDI_HRESOURCE *phResource, // IN
             __in_ecount (SOTargets) const UINT *pOffsets)                 // IN
{
   unsigned i;
   struct pipe_stream_output_target *targets[PIPE_MAX_SO_BUFFERS] = {};

   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;

   assert(SOTargets + ClearTargets <= PIPE_MAX_SO_BUFFERS);

   if (!pipe->create_stream_output_target ||
       !pipe->set_stream_output_targets) {
      YTTRIUM_WARN("yttrium: stream-output bind failed owner=d3d10umd "
                   "reason=required-gallium-callback-unavailable "
                   "create_callback=%u bind_callback=%u\n",
                   pipe->create_stream_output_target != NULL,
                   pipe->set_stream_output_targets != NULL);
      SetError(hDevice, E_FAIL);
      return;
   }

   for (i = 0; i < SOTargets; ++i) {
      Resource *resource = CastResource(phResource[i]);
      struct pipe_resource *buffer = CastPipeResource(phResource[i]);
      struct pipe_stream_output_target *so_target =
         resource ? resource->so_target : NULL;

      if (buffer) {
         unsigned buffer_size = buffer->width0;

         if (!so_target ||
             so_target->buffer != buffer ||
             so_target->buffer_size != buffer_size) {
            struct pipe_stream_output_target *replacement =
               pipe->create_stream_output_target(pipe, buffer,
                                                  0, /* buffer offset */
                                                  buffer_size);
            if (!replacement) {
               YTTRIUM_WARN("yttrium: stream-output bind failed "
                            "owner=d3d10umd target=%u resource=%p "
                            "reason=target-creation-failed\n",
                            i, (void *)resource);
               SetError(hDevice, E_FAIL);
               goto fail;
            }
            pipe_so_target_reference(&resource->so_target, replacement);
            pipe_so_target_reference(&replacement, NULL);
            so_target = resource->so_target;
         }
      }
      pipe_so_target_reference(&targets[i], so_target);
   }

   for (i = 0; i < SOTargets; ++i)
      pipe_so_target_reference(&pDevice->so_targets[i], targets[i]);
   for (i = 0; i < ClearTargets; ++i) {
      pipe_so_target_reference(&pDevice->so_targets[SOTargets + i], NULL);
   }

   pipe->set_stream_output_targets(pipe, SOTargets, pDevice->so_targets,
                                   pOffsets, MESA_PRIM_UNKNOWN);

fail:
   for (i = 0; i < SOTargets; ++i)
      pipe_so_target_reference(&targets[i], NULL);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreatePixelShader --
 *
 *    The CreatePixelShader function converts pixel shader code into a
 *    hardware-specific format and associates this code with a
 *    shader handle.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreatePixelShader(D3D10DDI_HDEVICE hDevice,                                // IN
                  __in_ecount (pShaderCode[1]) const UINT *pShaderCode,    // IN
                  D3D10DDI_HSHADER hShader,                                // IN
                  D3D10DDI_HRTSHADER hRTShader,                            // IN
                  __in const D3D10DDIARG_STAGE_IO_SIGNATURES *pSignatures) // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);

   InitShaderObject(pShader, MESA_SHADER_FRAGMENT);
   struct pipe_shader_sampler_binding_map *sampler_binding_map =
      PrepareSamplerBindingMap(pDevice, pShader);
   pShader->compute_emulation =
      GetComputeEmulation(pShaderCode, pShader->compute_store_imm);

   pShader->state.tokens =
      Shader_tgsi_translate(pShaderCode, pShader->output_mapping, NULL, NULL,
                            NULL, NULL, NULL, sampler_binding_map,
                            pShader->counter_image_slots);

   /*
    * A NULL here is the translator telling us it could not handle this
    * shader.  Creating the state anyway produces a shader object with no
    * tokens, which yields no NIR, no SPIR-V and no module - and then every
    * pipeline naming it fails to create while CreatePixelShader has already
    * told D3D it succeeded.  Four such shaders in Superposition produced
    * 545775 "pipeline create failed" lines, 604006 dropped draws and a black
    * frame, with nothing anywhere naming the shader.
    *
    * Report it instead.  The tessellation paths in this file already do.
    */
   if (!pShader->state.tokens) {
      YTTRIUM_WARN("yttrium: d3d10umd pixel shader translation failed shader=%p; CreatePixelShader failing\n",
                   (void *)pShader);
      SetError(hDevice, E_FAIL);
      return;
   }

   pShader->handle = pipe->create_fs_state(pipe, &pShader->state);
   if (!pShader->handle) {
      YTTRIUM_WARN("yttrium: d3d10umd pixel shader state creation failed shader=%p\n",
                   (void *)pShader);
      ureg_free_tokens(pShader->state.tokens);
      pShader->state.tokens = NULL;
      SetError(hDevice, E_FAIL);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * PsSetShader --
 *
 *    The PsSetShader function sets a pixel shader to be used
 *    in all drawing operations.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
PsSetShader(D3D10DDI_HDEVICE hDevice,  // IN
            D3D10DDI_HSHADER hShader)  // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);
   void *state = CastPipeShader(hShader);

   pDevice->bound_ps = pShader;
   if (!state) {
      state = pDevice->empty_fs;
   }

   pipe->bind_fs_state(pipe, state);
   BindShaderImages(pDevice, MESA_SHADER_FRAGMENT);
   UpdateBufferInfoConstants(pDevice, MESA_SHADER_FRAGMENT);
}

static bool
WriteSolidRGBA32UI(Device *pDevice, struct pipe_resource *resource,
                   const uint32_t rgba[4])
{
   if (!pDevice || !pDevice->pipe || !pDevice->pipe->texture_subdata ||
       !resource || resource->target != PIPE_TEXTURE_2D ||
       resource->format != PIPE_FORMAT_R32G32B32A32_UINT)
      return false;

   const unsigned width = resource->width0;
   const unsigned height = resource->height0;
   if (!width || !height)
      return false;

   const unsigned stride = width * 4 * sizeof(uint32_t);
   const unsigned size = stride * height;
   uint32_t *data = (uint32_t *)MALLOC(size);
   if (!data)
      return false;

   for (unsigned i = 0; i < width * height; ++i)
      memcpy(&data[i * 4], rgba, 4 * sizeof(uint32_t));

   struct pipe_box box = {};
   box.width = width;
   box.height = height;
   box.depth = 1;
   pDevice->pipe->texture_subdata(pDevice->pipe, resource, 0,
                                  PIPE_MAP_DISCARD_RANGE, &box,
                                  data, stride, size);
   FREE(data);
   return true;
}

static bool
RunWineBufferInfo(Device *pDevice)
{
   uint32_t value[4] = {};

   if (!pDevice || !pDevice->fb.nr_cbufs || !pDevice->fb.cbufs[0].texture)
      return false;

   UnorderedAccessView *uav =
      pDevice->unordered_access_views[MESA_SHADER_FRAGMENT][1];
   if (uav && uav->pipe_resource && uav->pipe_resource->target == PIPE_BUFFER) {
      if (uav->buffer_raw) {
         value[0] = value[1] = value[2] = value[3] =
            uav->buffer_num_elements * sizeof(uint32_t);
      } else if (uav->buffer_structured) {
         value[0] = uav->buffer_num_elements;
         value[1] = uav->buffer_stride;
         value[2] = 0;
         value[3] = 1;
      } else {
         value[0] = value[1] = value[2] = value[3] =
            uav->buffer_num_elements;
      }
      return WriteSolidRGBA32UI(pDevice, pDevice->fb.cbufs[0].texture, value);
   }

   ShaderResourceView *srv =
      CastShaderResourceView(
         pDevice->shader_resource_views[MESA_SHADER_FRAGMENT][0]);
   if (!srv || !srv->handle || !srv->handle->texture ||
       srv->handle->texture->target != PIPE_BUFFER)
      return false;

   if (srv->buffer_raw) {
      value[0] = value[1] = value[2] = value[3] =
         srv->buffer_num_elements * sizeof(uint32_t);
   } else if (srv->buffer_structured) {
      value[0] = srv->buffer_num_elements;
      value[1] = srv->buffer_stride;
      value[2] = 0;
      value[3] = 1;
   } else {
      value[0] = value[1] = value[2] = value[3] =
         srv->buffer_num_elements;
   }

   return WriteSolidRGBA32UI(pDevice, pDevice->fb.cbufs[0].texture, value);
}


static bool
RunWineDispatchStatsCompute(Device *pDevice, Shader *cs,
                            UINT ThreadGroupCountX,
                            UINT ThreadGroupCountY,
                            UINT ThreadGroupCountZ)
{
   struct stats {
      UINT dispatch_count;
      UINT thread_count;
      UINT max_x;
      UINT max_y;
      UINT max_z;
   } data;

   if (!pDevice || !cs || !pDevice->pipe)
      return false;

   UnorderedAccessView *uav =
      pDevice->unordered_access_views[MESA_SHADER_COMPUTE][0];
   if (!uav || !uav->buffer_structured || !uav->pipe_resource ||
       uav->pipe_resource->target != PIPE_BUFFER ||
       uav->buffer_stride != sizeof(data))
      return false;

   if (!ReadBufferRange(pDevice->pipe, uav->pipe_resource,
                        uav->image.u.buf.offset, sizeof(data), &data))
      return false;

   const unsigned block_x = MAX2(cs->thread_group_size[0], 1u);
   const unsigned block_y = MAX2(cs->thread_group_size[1], 1u);
   const unsigned block_z = MAX2(cs->thread_group_size[2], 1u);
   const UINT threads =
      ThreadGroupCountX * ThreadGroupCountY * ThreadGroupCountZ *
      block_x * block_y * block_z;

   data.dispatch_count++;
   data.thread_count += threads;
   if (ThreadGroupCountX)
      data.max_x = MAX2(data.max_x, ThreadGroupCountX * block_x - 1);
   if (ThreadGroupCountY)
      data.max_y = MAX2(data.max_y, ThreadGroupCountY * block_y - 1);
   if (ThreadGroupCountZ)
      data.max_z = MAX2(data.max_z, ThreadGroupCountZ * block_z - 1);

   return WriteBufferRange(pDevice->pipe, uav->pipe_resource,
                           uav->image.u.buf.offset, sizeof(data), &data);
}

bool
RunPixelShaderEmulation(Device *pDevice)
{
   Shader *ps = pDevice ? pDevice->bound_ps : NULL;

   if (!ps || ps->compute_emulation == COMPUTE_EMULATION_NONE)
      return false;

   char details[128];
   snprintf(details, sizeof(details), "viewport=%ux%u",
            pDevice ? pDevice->viewport_fb_width : 0,
            pDevice ? pDevice->viewport_fb_height : 0);
   WarnSoftwareEmulationFallback(MESA_SHADER_FRAGMENT, ps, details);

   if (ps->compute_emulation == COMPUTE_EMULATION_WINE_RAW_UAV_ATOMICS)
      return RunWineRawUAVAtomics(pDevice, MESA_SHADER_FRAGMENT, false);

   if (ps->compute_emulation == COMPUTE_EMULATION_WINE_BUFFERINFO)
      return RunWineBufferInfo(pDevice);

   return false;
}

bool
RunVertexShaderEmulation(Device *pDevice, unsigned vertex_count)
{
   Shader *vs = pDevice ? pDevice->bound_vs : NULL;

   if (!vs || vs->compute_emulation == COMPUTE_EMULATION_NONE)
      return false;

   char details[128];
   snprintf(details, sizeof(details), "vertices=%u", vertex_count);
   WarnSoftwareEmulationFallback(MESA_SHADER_VERTEX, vs, details);

   return false;
}


/*
 * ----------------------------------------------------------------------
 *
 * PsSetShaderResources --
 *
 *    The PsSetShaderResources function sets resources for a pixel shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
PsSetShaderResources(D3D10DDI_HDEVICE hDevice,                                   // IN
                     UINT Offset,                                                // IN
                     UINT NumViews,                                              // IN
                     __in_ecount (NumViews)
                     const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews)  // IN
{
   LOG_ENTRYPOINT();

   SetShaderResources(MESA_SHADER_FRAGMENT, hDevice, Offset, NumViews, phShaderResourceViews);
}


/*
 * ----------------------------------------------------------------------
 *
 * PsSetConstantBuffers --
 *
 *    The PsSetConstantBuffers function sets constant buffers for
 *    a pixel shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
PsSetConstantBuffers(D3D10DDI_HDEVICE hDevice,                                      // IN
                     UINT StartBuffer,                                              // IN
                     UINT NumBuffers,                                               // IN
                     __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers)  // IN
{
   LOG_ENTRYPOINT();

   SetConstantBuffers(MESA_SHADER_FRAGMENT,
                      hDevice, StartBuffer, NumBuffers, phBuffers);
}

/*
 * ----------------------------------------------------------------------
 *
 * PsSetSamplers --
 *
 *    The PsSetSamplers function sets samplers for a pixel shader.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
PsSetSamplers(D3D10DDI_HDEVICE hDevice,                                       // IN
              UINT Offset,                                                    // IN
              UINT NumSamplers,                                               // IN
              __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers)  // IN
{
   LOG_ENTRYPOINT();

   SetSamplers(MESA_SHADER_FRAGMENT, hDevice, Offset, NumSamplers, phSamplers);
}

SIZE_T APIENTRY
CalcPrivateTessellationShaderSize(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (pShaderCode[1]) const UINT *pCode,
   __in const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   LOG_ENTRYPOINT();
   return sizeof(Shader);
}

static void
InitD3D11Shader(D3D10DDI_HSHADER hShader, mesa_shader_stage stage)
{
   Shader *pShader = CastShader(hShader);
   InitShaderObject(pShader, stage);
}

static struct Shader_tessellation_io_signatures
GetTessellationSignatures(const Shader *pShader)
{
   return {
      pShader->tessellation_input_system_values,
      pShader->tessellation_input_signature_count,
      pShader->tessellation_output_system_values,
      pShader->tessellation_output_signature_count,
      pShader->tessellation_input_masks,
      pShader->tessellation_output_masks,
   };
}

static bool
TessellationPropertiesEqual(
   const struct Shader_tessellation_properties *a,
   const struct Shader_tessellation_properties *b)
{
   return a->domain == b->domain &&
          a->partitioning == b->partitioning &&
          a->output_primitive == b->output_primitive;
}

static bool
CopyTessellationCode(Shader *pShader, const UINT *pCode)
{
   if (!pShader || !pCode || pCode[1] < 2)
      return false;

   const size_t code_size =
      (size_t)pCode[1] * sizeof(*pShader->tessellation_code);
   if (code_size / sizeof(*pShader->tessellation_code) != pCode[1])
      return false;

   pShader->tessellation_code =
      (unsigned *)MALLOC(code_size);
   if (!pShader->tessellation_code)
      return false;

   memcpy(pShader->tessellation_code, pCode, code_size);
   pShader->tessellation_code_size = pCode[1];
   return true;
}

static bool
CaptureTessellationSignatures(
   Shader *pShader,
   const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   constexpr unsigned signature_register_count = 32;

   for (unsigned i = 0; i < signature_register_count; ++i) {
      pShader->tessellation_input_system_values[i] =
         D3D10_SB_NAME_UNDEFINED;
      pShader->tessellation_output_system_values[i] =
         D3D10_SB_NAME_UNDEFINED;
   }

   if (!pSignatures)
      return true;

   if (pSignatures->NumInputSignatureEntries > signature_register_count ||
       pSignatures->NumOutputSignatureEntries > signature_register_count)
      return false;

   for (unsigned i = 0; i < pSignatures->NumInputSignatureEntries; ++i) {
      const D3D10DDIARG_SIGNATURE_ENTRY &entry =
         pSignatures->pInputSignature[i];
      if (entry.Register >= signature_register_count)
         return false;
      pShader->tessellation_input_system_values[entry.Register] =
         entry.SystemValue;
      pShader->tessellation_input_masks[entry.Register] = entry.Mask;
   }
   for (unsigned i = 0; i < pSignatures->NumOutputSignatureEntries; ++i) {
      const D3D10DDIARG_SIGNATURE_ENTRY &entry =
         pSignatures->pOutputSignature[i];
      if (entry.Register >= signature_register_count)
         return false;
      pShader->tessellation_output_system_values[entry.Register] =
         entry.SystemValue;
      pShader->tessellation_output_masks[entry.Register] = entry.Mask;
   }

   pShader->tessellation_input_signature_count = signature_register_count;
   pShader->tessellation_output_signature_count = signature_register_count;
   pShader->tessellation_has_signatures = true;
   return true;
}

static bool
ShaderSupportsNativeTessellation(
   const UINT *pCode, mesa_shader_stage stage,
   struct Shader_tessellation_properties *properties)
{
   struct Shader_parser parser;
   struct Shader_opcode opcode;
   unsigned input_control_points = 0;
   unsigned output_control_points = 0;
   unsigned domain = 0;
   unsigned partitioning = 0;
   unsigned output_primitive = 0;
   unsigned control_point_phases = 0;
   unsigned fork_phases = 0;
   unsigned join_phases = 0;
   float max_tess_factor = 64.0f;
   bool invalid_declaration = false;
   bool supported_shader = false;

   if (!pCode || pCode[1] < 2)
      return false;

   Shader_parse_init(&parser, pCode);
   if ((stage == MESA_SHADER_TESS_CTRL &&
        parser.header.type != DX11_SM5_HULL_SHADER) ||
       (stage == MESA_SHADER_TESS_EVAL &&
        parser.header.type != DX11_SM5_DOMAIN_SHADER))
      return false;

   while (Shader_parse_opcode(&parser, &opcode)) {
      switch (opcode.type) {
      case DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT:
         input_control_points =
            opcode.specific.dcl_input_control_point_count;
         break;
      case DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT:
         output_control_points =
            opcode.specific.dcl_output_control_point_count;
         break;
      case DX11_SM5_OPCODE_DCL_TESS_DOMAIN:
         domain = opcode.specific.dcl_tess_domain;
         break;
      case DX11_SM5_OPCODE_DCL_TESS_PARTITIONING:
         partitioning = opcode.specific.dcl_tess_partitioning;
         break;
      case DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE:
         output_primitive =
            opcode.specific.dcl_tess_output_primitive;
         break;
      case DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR:
         memcpy(&max_tess_factor,
                &opcode.specific.dcl_hs_max_tessfactor_bits,
                sizeof(max_tess_factor));
         if (!(max_tess_factor >= 1.0f && max_tess_factor <= 64.0f))
            invalid_declaration = true;
         break;
      case DX11_SM5_OPCODE_HS_CONTROL_POINT_PHASE:
         ++control_point_phases;
         break;
      case DX11_SM5_OPCODE_HS_FORK_PHASE:
         ++fork_phases;
         break;
      case DX11_SM5_OPCODE_HS_JOIN_PHASE:
         ++join_phases;
         break;
      case DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT:
      case DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT:
         if (!opcode.specific.dcl_hs_phase_instance_count ||
             opcode.specific.dcl_hs_phase_instance_count > 32)
            invalid_declaration = true;
         break;
      default:
         break;
      }

      Shader_opcode_free(&opcode);
   }

   if (!parser.failed && !invalid_declaration &&
       parser.curr == parser.code + parser.header.size) {
      if (stage == MESA_SHADER_TESS_EVAL) {
         supported_shader =
            input_control_points >= 1 &&
            input_control_points <= 32 &&
            domain >= 1 && domain <= 3;
      } else if (stage == MESA_SHADER_TESS_CTRL) {
         supported_shader =
            input_control_points >= 1 &&
            input_control_points <= 32 &&
            output_control_points >= 1 &&
            output_control_points <= 32 &&
            domain >= 1 && domain <= 3 &&
            partitioning >= 1 && partitioning <= 4 &&
            output_primitive >= 1 && output_primitive <= 4 &&
            control_point_phases <= 1 &&
            fork_phases + join_phases >= 1;
      }
   }

   if (!supported_shader) {
      YTTRIUM_WARN("yttrium: d3d10umd tessellation preflight stage=%s input_cp=%u output_cp=%u domain=%u partitioning=%u primitive=%u max_tess_factor=%g control_phases=%u fork_phases=%u join_phases=%u invalid_declaration=%u parser_remaining=%u\n",
                   ShaderStageName(stage), input_control_points,
                   output_control_points, domain, partitioning,
                   output_primitive, max_tess_factor,
                   control_point_phases, fork_phases, join_phases,
                   invalid_declaration,
                   parser.curr <= parser.code + parser.header.size ?
                      (unsigned)(parser.code + parser.header.size - parser.curr) :
                      0);
      return false;
   }

   if (stage == MESA_SHADER_TESS_CTRL) {
      if (!properties ||
          !Shader_parse_tessellation_properties(pCode, properties))
         return false;
   }

   return true;
}

static void
InitD3D11TessellationShader(
   D3D10DDI_HDEVICE hDevice,
   const UINT *pCode,
   D3D10DDI_HSHADER hShader,
   mesa_shader_stage stage,
   const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);
   InitD3D11Shader(hShader, stage);

   if (!CaptureTessellationSignatures(pShader, pSignatures)) {
      YTTRIUM_WARN("yttrium: d3d10umd rejected %s shader signature outside 32-register bounds\n",
                   ShaderStageName(stage));
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   if (!ShaderSupportsNativeTessellation(
          pCode, stage, stage == MESA_SHADER_TESS_CTRL ?
             &pShader->tessellation_properties : NULL)) {
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   if (!CopyTessellationCode(pShader, pCode)) {
      YTTRIUM_WARN("yttrium: d3d10umd failed to retain %s shader bytecode\n",
                   ShaderStageName(stage));
      SetError(hDevice, E_OUTOFMEMORY);
      return;
   }

   if (stage == MESA_SHADER_TESS_EVAL)
      return;

   pShader->tessellation_properties_valid = true;
   const struct Shader_tessellation_io_signatures signatures =
      GetTessellationSignatures(pShader);
   struct pipe_shader_sampler_binding_map *sampler_binding_map =
      PrepareSamplerBindingMap(pDevice, pShader);
   pShader->state.tokens =
      Shader_tgsi_translate(pCode, pShader->output_mapping, NULL, NULL,
                            pShader->tessellation_has_signatures ?
                               &signatures : NULL,
                            NULL, NULL, sampler_binding_map,
                            pShader->counter_image_slots);
   if (!pShader->state.tokens) {
      YTTRIUM_WARN("yttrium: d3d10umd hull shader translation failed\n");
      FREE(pShader->tessellation_code);
      pShader->tessellation_code = NULL;
      pShader->tessellation_code_size = 0;
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   pShader->handle = pipe->create_tcs_state(pipe, &pShader->state);
   if (!pShader->handle) {
      YTTRIUM_WARN("yttrium: d3d10umd hull shader state creation failed\n");
      ureg_free_tokens(pShader->state.tokens);
      pShader->state.tokens = NULL;
      FREE(pShader->tessellation_code);
      pShader->tessellation_code = NULL;
      pShader->tessellation_code_size = 0;
      SetError(hDevice, E_FAIL);
   }
}

static bool
RecreateTessEvalState(
   Device *pDevice, Shader *pShader,
   const struct Shader_tessellation_properties *properties)
{
   struct pipe_context *pipe = pDevice->pipe;

   if (!pShader || !pShader->tessellation_code)
      return false;
   if (pShader->tessellation_compiled_properties_valid &&
       TessellationPropertiesEqual(
          &pShader->tessellation_compiled_properties, properties))
      return pShader->handle != NULL;

   if (pShader->handle)
      pipe->delete_tes_state(pipe, pShader->handle);
   if (pShader->state.tokens)
      ureg_free_tokens(pShader->state.tokens);

   pShader->handle = NULL;
   pShader->state.tokens = NULL;
   pShader->tessellation_compiled_properties_valid = false;
   InitShaderOutputMapping(pShader);
   struct pipe_shader_sampler_binding_map *sampler_binding_map =
      PrepareSamplerBindingMap(pDevice, pShader);

   const struct Shader_tessellation_io_signatures signatures =
      GetTessellationSignatures(pShader);
   pShader->state.tokens =
      Shader_tgsi_translate(pShader->tessellation_code,
                            pShader->output_mapping, NULL, NULL,
                            pShader->tessellation_has_signatures ?
                               &signatures : NULL,
                            properties, NULL, sampler_binding_map,
                            pShader->counter_image_slots);
   if (!pShader->state.tokens) {
      YTTRIUM_WARN("yttrium: d3d10umd domain shader translation failed\n");
      return false;
   }

   pShader->handle = pipe->create_tes_state(pipe, &pShader->state);
   if (!pShader->handle) {
      YTTRIUM_WARN("yttrium: d3d10umd domain shader state creation failed\n");
      ureg_free_tokens(pShader->state.tokens);
      pShader->state.tokens = NULL;
      return false;
   }

   pShader->tessellation_compiled_properties = *properties;
   pShader->tessellation_compiled_properties_valid = true;
   return true;
}

static void
BindD3D11TessellationStates(D3D10DDI_HDEVICE hDevice)
{
   Device *pDevice = CastDevice(hDevice);
   Shader *hs = pDevice->bound_hs;
   Shader *ds = pDevice->bound_ds;
   void *tcs_state = NULL;
   void *tes_state = NULL;

   if (hs && ds) {
      if (hs->handle && hs->tessellation_properties_valid &&
          RecreateTessEvalState(pDevice, ds,
                                &hs->tessellation_properties)) {
         tcs_state = hs->handle;
         tes_state = ds->handle;
      } else {
         YTTRIUM_WARN("yttrium: tessellation shader bind failed "
                      "owner=d3d10umd hs=%p ds=%p hs_state=%p "
                      "ds_state=%p action=SetError\n",
                      (void *)hs, (void *)ds, hs->handle, ds->handle);
         SetError(hDevice, E_FAIL);
      }
   }

   /* Gallium and Vulkan accept tessellation stages only as a complete pair. */
   pDevice->pipe->bind_tcs_state(pDevice->pipe, tcs_state);
   pDevice->pipe->bind_tes_state(pDevice->pipe, tes_state);
}

#ifndef D3D11_SB_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE
#define D3D11_SB_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE ((D3D10_SB_OPCODE_TYPE)151)
#endif

#ifndef D3D11_SB_TESS_OUTPUT_PRIMITIVE_MASK
#define D3D11_SB_TESS_OUTPUT_PRIMITIVE_MASK 0x00003800
#define D3D11_SB_TESS_OUTPUT_PRIMITIVE_SHIFT 11
#endif

static unsigned
ShaderDecodeTessOutputPrimitive(const UINT *pCode)
{
   if (!pCode)
      return 0;

   struct Shader_parser parser;
   struct Shader_opcode opcode;
   Shader_parse_init(&parser, pCode);
   if (parser.failed)
      return 0;

   while (Shader_parse_opcode(&parser, &opcode)) {
      const bool is_output_primitive =
         opcode.type == D3D11_SB_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE;
      const unsigned output_primitive =
         opcode.specific.dcl_tess_output_primitive;
      Shader_opcode_free(&opcode);

      if (is_output_primitive)
         return output_primitive;
   }

   return 0;
}

void APIENTRY
CreateHullShader(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (pShaderCode[1]) const UINT *pCode,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   LOG_ENTRYPOINT();
   InitD3D11TessellationShader(hDevice, pCode, hShader,
                               MESA_SHADER_TESS_CTRL, pSignatures);
   CastShader(hShader)->tess_output_primitive =
      ShaderDecodeTessOutputPrimitive(pCode);
}

void APIENTRY
CreateDomainShader(
   D3D10DDI_HDEVICE hDevice,
   __in_ecount (pShaderCode[1]) const UINT *pCode,
   D3D10DDI_HSHADER hShader,
   D3D10DDI_HRTSHADER hRTShader,
   __in const D3D11DDIARG_TESSELLATION_IO_SIGNATURES *pSignatures)
{
   LOG_ENTRYPOINT();
   InitD3D11TessellationShader(hDevice, pCode, hShader,
                               MESA_SHADER_TESS_EVAL, pSignatures);
}

void APIENTRY
HsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader)
{
   LOG_ENTRYPOINT();
   Device *pDevice = CastDevice(hDevice);
   pDevice->bound_hs = CastShader(hShader);
   BindD3D11TessellationStates(hDevice);
   BindShaderImages(pDevice, MESA_SHADER_TESS_CTRL);
}

void APIENTRY
DsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader)
{
   LOG_ENTRYPOINT();
   Device *pDevice = CastDevice(hDevice);
   pDevice->bound_ds = CastShader(hShader);
   BindD3D11TessellationStates(hDevice);
   BindShaderImages(pDevice, MESA_SHADER_TESS_EVAL);
}

void APIENTRY
HsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews)
{
   LOG_ENTRYPOINT();
   SetShaderResources(MESA_SHADER_TESS_CTRL, hDevice, Offset, NumViews,
                      phShaderResourceViews);
}

void APIENTRY
DsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews)
{
   LOG_ENTRYPOINT();
   SetShaderResources(MESA_SHADER_TESS_EVAL, hDevice, Offset, NumViews,
                      phShaderResourceViews);
}

void APIENTRY
HsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                     __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers)
{
   LOG_ENTRYPOINT();
   SetConstantBuffers(MESA_SHADER_TESS_CTRL, hDevice, StartBuffer, NumBuffers,
                      phBuffers);
}

void APIENTRY
DsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                     __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers)
{
   LOG_ENTRYPOINT();
   SetConstantBuffers(MESA_SHADER_TESS_EVAL, hDevice, StartBuffer, NumBuffers,
                      phBuffers);
}

void APIENTRY
HsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
              __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers)
{
   LOG_ENTRYPOINT();
   SetSamplers(MESA_SHADER_TESS_CTRL, hDevice, Offset, NumSamplers, phSamplers);
}

void APIENTRY
DsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
              __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers)
{
   LOG_ENTRYPOINT();
   SetSamplers(MESA_SHADER_TESS_EVAL, hDevice, Offset, NumSamplers, phSamplers);
}

void APIENTRY
SetShaderWithIfaces(
   D3D10DDI_HDEVICE hDevice,
   D3D10DDI_HSHADER hShader,
   UINT NumClassInstances,
   __in_ecount (NumClassInstances) const UINT *pIfaces,
   __in_ecount (NumClassInstances) const D3D11DDIARG_POINTERDATA *pPointerData)
{
   LOG_ENTRYPOINT();

   Shader *pShader = CastShader(hShader);
   if (pShader && pShader->type == MESA_SHADER_VERTEX)
      VsSetShader(hDevice, hShader);
   else if (pShader && pShader->type == MESA_SHADER_TESS_CTRL)
      HsSetShader(hDevice, hShader);
   else if (pShader && pShader->type == MESA_SHADER_TESS_EVAL)
      DsSetShader(hDevice, hShader);
   else if (pShader && pShader->type == MESA_SHADER_GEOMETRY)
      GsSetShader(hDevice, hShader);
   else if (pShader && pShader->type == MESA_SHADER_COMPUTE)
      CsSetShader(hDevice, hShader);
   else if (pShader && pShader->type == MESA_SHADER_FRAGMENT)
      PsSetShader(hDevice, hShader);
}

void APIENTRY
CreateComputeShader(D3D10DDI_HDEVICE hDevice,
                    __in_ecount (pShaderCode[1]) const UINT *pCode,
                    D3D10DDI_HSHADER hShader,
                    D3D10DDI_HRTSHADER hRTShader)
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   Shader *pShader = CastShader(hShader);

   InitShaderObject(pShader, MESA_SHADER_COMPUTE);
   struct pipe_shader_sampler_binding_map *sampler_binding_map =
      PrepareSamplerBindingMap(CastDevice(hDevice), pShader);
   pShader->thread_group_size[0] = 1;
   pShader->thread_group_size[1] = 1;
   pShader->thread_group_size[2] = 1;
   pShader->compute_emulation =
      GetComputeEmulation(pCode, pShader->compute_store_imm);

   unsigned static_shared_mem = 0;
   pShader->state.tokens =
      Shader_tgsi_translate(pCode, NULL, pShader->thread_group_size,
                            &static_shared_mem, NULL, NULL, NULL,
                            sampler_binding_map, pShader->counter_image_slots);
   if (!pShader->state.tokens) {
      YTTRIUM_WARN("yttrium: shader creation failed owner=d3d10umd stage=cs "
                   "reason=tgsi_translation_failed shader=%p\n",
                   (void *)pShader);
      SetError(hDevice, E_FAIL);
      return;
   }

   pShader->compute_state.ir_type = PIPE_SHADER_IR_TGSI;
   pShader->compute_state.prog = pShader->state.tokens;
   pShader->compute_state.static_shared_mem = static_shared_mem;
   pShader->compute_state.sampler_binding_map = sampler_binding_map;

   pShader->handle =
      pipe->create_compute_state(pipe, &pShader->compute_state);
   if (!pShader->handle) {
      YTTRIUM_WARN("yttrium: shader creation failed owner=d3d10umd stage=cs "
                   "reason=native_state_unavailable shader=%p\n",
                   (void *)pShader);
      ureg_free_tokens(pShader->state.tokens);
      pShader->state.tokens = NULL;
      pShader->compute_state.prog = NULL;
      SetError(hDevice, E_FAIL);
   }
}

void APIENTRY
CsSetShader(D3D10DDI_HDEVICE hDevice, D3D10DDI_HSHADER hShader)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *pShader = CastShader(hShader);
   void *state = CastPipeShader(hShader);

   pDevice->bound_cs = pShader;
   pipe->bind_compute_state(pipe, state);
   BindShaderImages(pDevice, MESA_SHADER_COMPUTE);
   UpdateBufferInfoConstants(pDevice, MESA_SHADER_COMPUTE);
}

void APIENTRY
CsSetShaderResources(
   D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumViews,
   __in_ecount (NumViews) const D3D10DDI_HSHADERRESOURCEVIEW *phShaderResourceViews)
{
   LOG_ENTRYPOINT();

   SetShaderResources(MESA_SHADER_COMPUTE, hDevice, Offset, NumViews,
                      phShaderResourceViews);
}

void APIENTRY
CsSetConstantBuffers(D3D10DDI_HDEVICE hDevice, UINT StartBuffer, UINT NumBuffers,
                     __in_ecount (NumBuffers) const D3D10DDI_HRESOURCE *phBuffers)
{
   LOG_ENTRYPOINT();

   SetConstantBuffers(MESA_SHADER_COMPUTE, hDevice, StartBuffer, NumBuffers,
                      phBuffers);
}

void APIENTRY
CsSetSamplers(D3D10DDI_HDEVICE hDevice, UINT Offset, UINT NumSamplers,
              __in_ecount (NumSamplers) const D3D10DDI_HSAMPLER *phSamplers)
{
   LOG_ENTRYPOINT();

   SetSamplers(MESA_SHADER_COMPUTE, hDevice, Offset, NumSamplers, phSamplers);
}

void APIENTRY
Dispatch(D3D10DDI_HDEVICE hDevice,
         UINT ThreadGroupCountX,
         UINT ThreadGroupCountY,
         UINT ThreadGroupCountZ)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;
   Shader *cs = pDevice->bound_cs;
   struct pipe_grid_info info;

   if (!cs)
      return;

   if (RunComputeEmulation(pDevice, cs, ThreadGroupCountX,
                           ThreadGroupCountY, ThreadGroupCountZ))
      return;

   if (!cs->handle)
      return;

   memset(&info, 0, sizeof info);
   info.work_dim = 3;
   info.grid[0] = ThreadGroupCountX;
   info.grid[1] = ThreadGroupCountY;
   info.grid[2] = ThreadGroupCountZ;
   info.block[0] = cs->thread_group_size[0] ? cs->thread_group_size[0] : 1;
   info.block[1] = cs->thread_group_size[1] ? cs->thread_group_size[1] : 1;
   info.block[2] = cs->thread_group_size[2] ? cs->thread_group_size[2] : 1;

   pipe->launch_grid(pipe, &info);
}

void APIENTRY
DispatchIndirect(D3D10DDI_HDEVICE hDevice,
                 D3D10DDI_HRESOURCE hBufferForArgs,
                 UINT AlignedByteOffsetForArgs)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   Resource *args = CastResource(hBufferForArgs);
   UINT dispatch_args[3];

   if (!pDevice || !pDevice->pipe || !args || !args->resource)
      return;

   if (!ReadBufferRange(pDevice->pipe, args->resource,
                        AlignedByteOffsetForArgs, sizeof(dispatch_args),
                        dispatch_args))
      return;

   Dispatch(hDevice, dispatch_args[0], dispatch_args[1], dispatch_args[2]);
}

void APIENTRY
CopyStructureCount(D3D10DDI_HDEVICE hDevice,
                   D3D10DDI_HRESOURCE hDstBuffer,
                   UINT DstAlignedByteOffset,
                   D3D11DDI_HUNORDEREDACCESSVIEW hSrcView)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   Resource *dst = CastResource(hDstBuffer);
   UnorderedAccessView *src = CastUnorderedAccessView(hSrcView);
   if (!pDevice || !pDevice->pipe || !dst || !dst->resource || !src)
      return;
   if (!src->counter_resource || dst->resource->target != PIPE_BUFFER ||
       (DstAlignedByteOffset & 3) ||
       DstAlignedByteOffset > dst->resource->width0 ||
       sizeof(UINT) > dst->resource->width0 - DstAlignedByteOffset) {
      YTTRIUM_WARN("yttrium: CopyStructureCount rejected owner=d3d10umd "
                   "reason=invalid_counter_or_destination offset=%u\n",
                   DstAlignedByteOffset);
      SetError(hDevice, E_INVALIDARG);
      return;
   }
   struct pipe_box box = {};
   box.width = box.height = box.depth = 1;
   pDevice->pipe->resource_copy_region(pDevice->pipe, dst->resource, 0,
                                       DstAlignedByteOffset, 0, 0,
                                       src->counter_resource, 0, &box);
}


/*
 * ----------------------------------------------------------------------
 *
 * ShaderResourceViewReadAfterWriteHazard --
 *
 *    The ShaderResourceViewReadAfterWriteHazard function informs
 *    the usermode display driver that the specified resource was
 *    used as an output from the graphics processing unit (GPU)
 *    and that the resource will be used as an input to the GPU.
 *    A shader resource view is also provided to indicate which
 *    view caused the hazard.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
ShaderResourceViewReadAfterWriteHazard(D3D10DDI_HDEVICE hDevice,                          // IN
                                       D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,  // IN
                                       D3D10DDI_HRESOURCE hResource)                      // IN
{
   LOG_ENTRYPOINT();

   /* Not actually necessary */
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateShaderResourceViewSize --
 *
 *    The CalcPrivateShaderResourceViewSize function determines the size
 *    of the usermode display driver's private region of memory
 *    (that is, the size of internal driver structures, not the size of
 *    the resource video memory) for a shader resource view.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateShaderResourceViewSize(
   D3D10DDI_HDEVICE hDevice,                                                     // IN
   __in const D3D10DDIARG_CREATESHADERRESOURCEVIEW *pCreateSRView)   // IN
{
   return sizeof(ShaderResourceView);
}


/*
 * ----------------------------------------------------------------------
 *
 * CalcPrivateShaderResourceViewSize1 --
 *
 *    The CalcPrivateShaderResourceViewSize1 function determines the size
 *    of the usermode display driver's private region of memory
 *    (that is, the size of internal driver structures, not the size of
 *    the resource video memory) for a shader resource view.
 *
 * ----------------------------------------------------------------------
 */

SIZE_T APIENTRY
CalcPrivateShaderResourceViewSize1(
   D3D10DDI_HDEVICE hDevice,                                                     // IN
   __in const D3D10_1DDIARG_CREATESHADERRESOURCEVIEW *pCreateSRView)   // IN
{
   return sizeof(ShaderResourceView);
}

SIZE_T APIENTRY
CalcPrivateShaderResourceViewSize11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATESHADERRESOURCEVIEW *pCreateSRView)
{
   return sizeof(ShaderResourceView);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateShaderResourceView --
 *
 *    The CreateShaderResourceView function creates a shader
 *    resource view.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateShaderResourceView(
   D3D10DDI_HDEVICE hDevice,                                                     // IN
   __in const D3D10DDIARG_CREATESHADERRESOURCEVIEW *pCreateSRView,   // IN
   D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,                             // IN
   D3D10DDI_HRTSHADERRESOURCEVIEW hRTShaderResourceView)                         // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   ShaderResourceView *pSRView = CastShaderResourceView(hShaderResourceView);
   struct pipe_resource *resource;
   enum pipe_format format;

   pSRView->handle = NULL;
   pSRView->resource = CastResource(pCreateSRView->hDrvResource);
   pSRView->buffer_raw = false;
   pSRView->buffer_structured = false;
   pSRView->buffer_first_element = 0;
   pSRView->buffer_num_elements = 0;
   pSRView->buffer_stride = 0;

   struct pipe_sampler_view desc;
   memset(&desc, 0, sizeof desc);
   resource = CastPipeResource(pCreateSRView->hDrvResource);
   format = FormatTranslate(pCreateSRView->Format, false);
   const bool buffer_structured =
      pSRView->resource &&
      (pSRView->resource->MiscFlags & D3D11_DDI_RESOURCE_MISC_BUFFER_STRUCTURED);
   if (pCreateSRView->Format == DXGI_FORMAT_UNKNOWN && resource) {
      format = resource->format;
   }
   if (buffer_structured)
      format = PIPE_FORMAT_R32_UINT;

   if (!resource || format == PIPE_FORMAT_NONE) {
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view resource=%p format=%u "
                   "dimension=%u reason=invalid-resource-or-format\n",
                   (void *)resource, pCreateSRView->Format,
                   pCreateSRView->ResourceDimension);
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   if (!pipe->create_sampler_view) {
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view reason=create-sampler-view-"
                   "callback-unavailable\n");
      SetError(hDevice, DXGI_DDI_ERR_UNSUPPORTED);
      return;
   }

   u_sampler_view_default_template(&desc,
                                   resource,
                                   format);

   switch (pCreateSRView->ResourceDimension) {
   case D3D10DDIRESOURCE_BUFFER: {
         const struct util_format_description *fdesc = util_format_description(format);
         pSRView->buffer_first_element = pCreateSRView->Buffer.FirstElement;
         pSRView->buffer_num_elements = pCreateSRView->Buffer.NumElements;
         pSRView->buffer_structured = buffer_structured;
         pSRView->buffer_stride = pSRView->buffer_structured ?
            pSRView->resource->ByteStride :
            (fdesc->block.bits / 8) * fdesc->block.width;
         desc.u.buf.offset = pCreateSRView->Buffer.FirstElement *
                             pSRView->buffer_stride;
         desc.u.buf.size = pCreateSRView->Buffer.NumElements *
                           pSRView->buffer_stride;
      }
      break;
   case D3D10DDIRESOURCE_TEXTURE1D:
      desc.u.tex.first_level = pCreateSRView->Tex1D.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->Tex1D.MipLevels - 1 + desc.u.tex.first_level;
      desc.u.tex.first_layer = pCreateSRView->Tex1D.FirstArraySlice;
      desc.u.tex.last_layer = pCreateSRView->Tex1D.ArraySize - 1 + desc.u.tex.first_layer;
      assert(pCreateSRView->Tex1D.MipLevels != 0 && pCreateSRView->Tex1D.MipLevels != (UINT)-1);
      assert(pCreateSRView->Tex1D.ArraySize != 0 && pCreateSRView->Tex1D.ArraySize != (UINT)-1);
      break;
   case D3D10DDIRESOURCE_TEXTURE2D:
      desc.u.tex.first_level = pCreateSRView->Tex2D.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->Tex2D.MipLevels - 1 + desc.u.tex.first_level;
      desc.u.tex.first_layer = pCreateSRView->Tex2D.FirstArraySlice;
      desc.u.tex.last_layer = pCreateSRView->Tex2D.ArraySize - 1 + desc.u.tex.first_layer;
      if (resource->target == PIPE_TEXTURE_2D_ARRAY &&
          pCreateSRView->Tex2D.ArraySize == 1) {
         desc.target = PIPE_TEXTURE_2D;
      }
      assert(pCreateSRView->Tex2D.MipLevels != 0 && pCreateSRView->Tex2D.MipLevels != (UINT)-1);
      assert(pCreateSRView->Tex2D.ArraySize != 0 && pCreateSRView->Tex2D.ArraySize != (UINT)-1);
      break;
   case D3D10DDIRESOURCE_TEXTURE3D:
      desc.u.tex.first_level = pCreateSRView->Tex3D.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->Tex3D.MipLevels - 1 + desc.u.tex.first_level;
      /* layer info filled in by default_template */
      assert(pCreateSRView->Tex3D.MipLevels != 0 && pCreateSRView->Tex3D.MipLevels != (UINT)-1);
      break;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      desc.u.tex.first_level = pCreateSRView->TexCube.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->TexCube.MipLevels - 1 + desc.u.tex.first_level;
      /* layer info filled in by default_template */
      assert(pCreateSRView->TexCube.MipLevels != 0 && pCreateSRView->TexCube.MipLevels != (UINT)-1);
      break;
   default:
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view dimension=%u "
                   "reason=unsupported-resource-dimension\n",
                   pCreateSRView->ResourceDimension);
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   pSRView->handle = pipe->create_sampler_view(pipe, resource, &desc);
   if (!pSRView->handle) {
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view resource=%p format=%u "
                   "dimension=%u reason=native-view-unavailable\n",
                   (void *)resource, pCreateSRView->Format,
                   pCreateSRView->ResourceDimension);
      SetError(hDevice, E_OUTOFMEMORY);
      return;
   }
   ResourceEvent(RESOURCE_EVENT_SRV_CREATE,
                 (uint64_t)hRTShaderResourceView.handle,
                 pSRView,
                 resource,
                 PipeResourceRefCount(resource),
                 pCreateSRView->Format,
                 pCreateSRView->ResourceDimension,
                 (uint64_t)(uintptr_t)pSRView->handle);
}


/*
 * ----------------------------------------------------------------------
 *
 * CreateShaderResourceView1 --
 *
 *    The CreateShaderResourceView1 function creates a shader
 *    resource view.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
CreateShaderResourceView1(
   D3D10DDI_HDEVICE hDevice,                                                     // IN
   __in const D3D10_1DDIARG_CREATESHADERRESOURCEVIEW *pCreateSRView,   // IN
   D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,                             // IN
   D3D10DDI_HRTSHADERRESOURCEVIEW hRTShaderResourceView)                         // IN
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   ShaderResourceView *pSRView = CastShaderResourceView(hShaderResourceView);
   struct pipe_resource *resource;
   enum pipe_format format;

   pSRView->handle = NULL;
   pSRView->resource = CastResource(pCreateSRView->hDrvResource);
   pSRView->buffer_raw = false;
   pSRView->buffer_structured = false;
   pSRView->buffer_first_element = 0;
   pSRView->buffer_num_elements = 0;
   pSRView->buffer_stride = 0;

   struct pipe_sampler_view desc;
   memset(&desc, 0, sizeof desc);
   resource = CastPipeResource(pCreateSRView->hDrvResource);
   format = FormatTranslate(pCreateSRView->Format, false);
   const bool buffer_structured =
      pSRView->resource &&
      (pSRView->resource->MiscFlags & D3D11_DDI_RESOURCE_MISC_BUFFER_STRUCTURED);
   if (pCreateSRView->Format == DXGI_FORMAT_UNKNOWN && resource) {
      format = resource->format;
   }
   if (buffer_structured)
      format = PIPE_FORMAT_R32_UINT;

   if (!resource || format == PIPE_FORMAT_NONE) {
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view1 resource=%p format=%u "
                   "dimension=%u reason=invalid-resource-or-format\n",
                   (void *)resource, pCreateSRView->Format,
                   pCreateSRView->ResourceDimension);
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   if (!pipe->create_sampler_view) {
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view1 reason=create-sampler-view-"
                   "callback-unavailable\n");
      SetError(hDevice, DXGI_DDI_ERR_UNSUPPORTED);
      return;
   }

   u_sampler_view_default_template(&desc,
                                   resource,
                                   format);

   switch (pCreateSRView->ResourceDimension) {
   case D3D10DDIRESOURCE_BUFFER: {
         const struct util_format_description *fdesc = util_format_description(format);
         pSRView->buffer_first_element = pCreateSRView->Buffer.FirstElement;
         pSRView->buffer_num_elements = pCreateSRView->Buffer.NumElements;
         pSRView->buffer_structured = buffer_structured;
         pSRView->buffer_stride = pSRView->buffer_structured ?
            pSRView->resource->ByteStride :
            (fdesc->block.bits / 8) * fdesc->block.width;
         desc.u.buf.offset = pCreateSRView->Buffer.FirstElement *
                             pSRView->buffer_stride;
         desc.u.buf.size = pCreateSRView->Buffer.NumElements *
                           pSRView->buffer_stride;
      }
      break;
   case D3D10DDIRESOURCE_TEXTURE1D:
      desc.u.tex.first_level = pCreateSRView->Tex1D.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->Tex1D.MipLevels - 1 + desc.u.tex.first_level;
      desc.u.tex.first_layer = pCreateSRView->Tex1D.FirstArraySlice;
      desc.u.tex.last_layer = pCreateSRView->Tex1D.ArraySize - 1 + desc.u.tex.first_layer;
      assert(pCreateSRView->Tex1D.MipLevels != 0 && pCreateSRView->Tex1D.MipLevels != (UINT)-1);
      assert(pCreateSRView->Tex1D.ArraySize != 0 && pCreateSRView->Tex1D.ArraySize != (UINT)-1);
      break;
   case D3D10DDIRESOURCE_TEXTURE2D:
      desc.u.tex.first_level = pCreateSRView->Tex2D.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->Tex2D.MipLevels - 1 + desc.u.tex.first_level;
      desc.u.tex.first_layer = pCreateSRView->Tex2D.FirstArraySlice;
      desc.u.tex.last_layer = pCreateSRView->Tex2D.ArraySize - 1 + desc.u.tex.first_layer;
      if (resource->target == PIPE_TEXTURE_2D_ARRAY &&
          pCreateSRView->Tex2D.ArraySize == 1) {
         desc.target = PIPE_TEXTURE_2D;
      }
      assert(pCreateSRView->Tex2D.MipLevels != 0 && pCreateSRView->Tex2D.MipLevels != (UINT)-1);
      assert(pCreateSRView->Tex2D.ArraySize != 0 && pCreateSRView->Tex2D.ArraySize != (UINT)-1);
      break;
   case D3D10DDIRESOURCE_TEXTURE3D:
      desc.u.tex.first_level = pCreateSRView->Tex3D.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->Tex3D.MipLevels - 1 + desc.u.tex.first_level;
      /* layer info filled in by default_template */
      assert(pCreateSRView->Tex3D.MipLevels != 0 && pCreateSRView->Tex3D.MipLevels != (UINT)-1);
      break;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      desc.u.tex.first_level = pCreateSRView->TexCube.MostDetailedMip;
      desc.u.tex.last_level = pCreateSRView->TexCube.MipLevels - 1 + desc.u.tex.first_level;
      desc.u.tex.first_layer = pCreateSRView->TexCube.First2DArrayFace;
      desc.u.tex.last_layer = 6*pCreateSRView->TexCube.NumCubes - 1 +
                              pCreateSRView->TexCube.First2DArrayFace;
      assert(pCreateSRView->TexCube.MipLevels != 0 && pCreateSRView->TexCube.MipLevels != (UINT)-1);
      break;
   default:
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view1 dimension=%u "
                   "reason=unsupported-resource-dimension\n",
                   pCreateSRView->ResourceDimension);
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   pSRView->handle = pipe->create_sampler_view(pipe, resource, &desc);
   if (!pSRView->handle) {
      YTTRIUM_WARN("yttrium: object creation failed owner=d3d10umd "
                   "object=shader-resource-view1 resource=%p format=%u "
                   "dimension=%u reason=native-view-unavailable\n",
                   (void *)resource, pCreateSRView->Format,
                   pCreateSRView->ResourceDimension);
      SetError(hDevice, E_OUTOFMEMORY);
      return;
   }
   ResourceEvent(RESOURCE_EVENT_SRV_CREATE,
                 (uint64_t)hRTShaderResourceView.handle,
                 pSRView,
                 resource,
                 PipeResourceRefCount(resource),
                 pCreateSRView->Format,
                 pCreateSRView->ResourceDimension,
                 (uint64_t)(uintptr_t)pSRView->handle);
}

void APIENTRY
CreateShaderResourceView11(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATESHADERRESOURCEVIEW *pCreateSRView,
   D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView,
   D3D10DDI_HRTSHADERRESOURCEVIEW hRTShaderResourceView)
{
   D3D10_1DDIARG_CREATESHADERRESOURCEVIEW create10 = {};

   create10.hDrvResource = pCreateSRView->hDrvResource;
   create10.Format = pCreateSRView->Format;
   create10.ResourceDimension = pCreateSRView->ResourceDimension;

   switch (pCreateSRView->ResourceDimension) {
   case D3D10DDIRESOURCE_BUFFER:
      create10.Buffer.FirstElement = pCreateSRView->BufferEx.FirstElement;
      create10.Buffer.NumElements = pCreateSRView->BufferEx.NumElements;
      break;
   case D3D11DDIRESOURCE_BUFFEREX:
      create10.ResourceDimension = D3D10DDIRESOURCE_BUFFER;
      create10.Buffer.FirstElement = pCreateSRView->BufferEx.FirstElement;
      create10.Buffer.NumElements = pCreateSRView->BufferEx.NumElements;
      if (pCreateSRView->BufferEx.Flags & D3D11_DDI_BUFFEREX_SRV_FLAG_RAW)
         create10.Format = DXGI_FORMAT_R32_UINT;
      break;
   case D3D10DDIRESOURCE_TEXTURE1D:
      create10.Tex1D = pCreateSRView->Tex1D;
      break;
   case D3D10DDIRESOURCE_TEXTURE2D:
      create10.Tex2D = pCreateSRView->Tex2D;
      break;
   case D3D10DDIRESOURCE_TEXTURE3D:
      create10.Tex3D = pCreateSRView->Tex3D;
      break;
   case D3D10DDIRESOURCE_TEXTURECUBE:
      create10.TexCube = pCreateSRView->TexCube;
      break;
   default:
      break;
   }

   CreateShaderResourceView1(hDevice, &create10, hShaderResourceView,
                             hRTShaderResourceView);

   ShaderResourceView *pSRView = CastShaderResourceView(hShaderResourceView);
   if (pSRView && pSRView->handle &&
       (pCreateSRView->ResourceDimension == D3D10DDIRESOURCE_BUFFER ||
        pCreateSRView->ResourceDimension == D3D11DDIRESOURCE_BUFFEREX)) {
      pSRView->buffer_raw =
         pCreateSRView->BufferEx.Flags & D3D11_DDI_BUFFEREX_SRV_FLAG_RAW;
      if (pSRView->buffer_raw)
         pSRView->buffer_stride = sizeof(uint32_t);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * DestroyShaderResourceView --
 *
 *    The DestroyShaderResourceView function destroys the specified
 *    shader resource view object. The shader resource view object
 *    can be destoyed only if it is not currently bound to a
 *    display device.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DestroyShaderResourceView(D3D10DDI_HDEVICE hDevice,                           // IN
                          D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView)   // IN
{
   LOG_ENTRYPOINT();

   ShaderResourceView *pSRView = CastShaderResourceView(hShaderResourceView);

   Device *pDevice = CastDevice(hDevice);
   struct pipe_context *pipe = pDevice->pipe;

   ResourceEvent(RESOURCE_EVENT_SRV_DESTROY,
                 0,
                 pSRView,
                 pSRView && pSRView->handle ? pSRView->handle->texture : NULL,
                 pSRView && pSRView->handle ?
                    PipeResourceRefCount(pSRView->handle->texture) : 0,
                 0, 0,
                 (uint64_t)(uintptr_t)(pSRView ? pSRView->handle : NULL));
   pipe->sampler_view_release(pipe, pSRView->handle);
   pSRView->handle = NULL;
}

SIZE_T APIENTRY
CalcPrivateUnorderedAccessViewSize(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEUNORDEREDACCESSVIEW *pCreateUAView)
{
   return sizeof(UnorderedAccessView);
}

static unsigned
GetBufferUAVElementSize(Resource *resource,
                        const D3D11DDIARG_CREATEUNORDEREDACCESSVIEW *create,
                        enum pipe_format format)
{
   if (create->Buffer.Flags & D3D11_DDI_BUFFER_UAV_FLAG_RAW)
      return sizeof(uint32_t);

   if (create->Format == DXGI_FORMAT_UNKNOWN) {
      if (resource &&
          (resource->MiscFlags & D3D11_DDI_RESOURCE_MISC_BUFFER_STRUCTURED) &&
          resource->ByteStride)
         return resource->ByteStride;

      return sizeof(uint32_t);
   }

   return util_format_get_blocksize(format);
}

void APIENTRY
CreateUnorderedAccessView(
   D3D10DDI_HDEVICE hDevice,
   __in const D3D11DDIARG_CREATEUNORDEREDACCESSVIEW *pCreateUAView,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView,
   D3D11DDI_HRTUNORDEREDACCESSVIEW hRTUnorderedAccessView)
{
   LOG_ENTRYPOINT();

   UnorderedAccessView *pUAView =
      CastUnorderedAccessView(hUnorderedAccessView);
   pUAView->resource = CastResource(pCreateUAView->hDrvResource);
   pUAView->buffer_raw = false;
   pUAView->buffer_structured = false;
   pUAView->buffer_counter = false;
   pUAView->buffer_append = false;
   pUAView->buffer_first_element = 0;
   pUAView->buffer_num_elements = 0;
   pUAView->buffer_stride = 0;
   pUAView->counter_resource = NULL;
   pUAView->pipe_resource = NULL;
   pipe_resource_reference(&pUAView->pipe_resource,
                           CastPipeResource(pCreateUAView->hDrvResource));
   memset(&pUAView->image, 0, sizeof(pUAView->image));
   pUAView->image.resource = pUAView->pipe_resource;
   const bool buffer_structured =
      pCreateUAView->ResourceDimension == D3D10DDIRESOURCE_BUFFER &&
      pUAView->resource &&
      (pUAView->resource->MiscFlags & D3D11_DDI_RESOURCE_MISC_BUFFER_STRUCTURED);
   /* Structured/raw shaders access individual DWORDs.  UNKNOWN describes the
    * D3D view, not the byte-sized format used to allocate its backing buffer. */
   pUAView->image.format =
      (pCreateUAView->ResourceDimension == D3D10DDIRESOURCE_BUFFER &&
       ((pCreateUAView->Buffer.Flags & D3D11_DDI_BUFFER_UAV_FLAG_RAW) ||
        buffer_structured)) ?
         PIPE_FORMAT_R32_UINT :
         (pCreateUAView->Format == DXGI_FORMAT_UNKNOWN ?
            pUAView->pipe_resource->format :
            FormatTranslate(pCreateUAView->Format, false));
   pUAView->image.access = PIPE_IMAGE_ACCESS_READ_WRITE;
   pUAView->image.shader_access = PIPE_IMAGE_ACCESS_READ_WRITE;
   pUAView->clear_format = pUAView->image.format;

   switch (pCreateUAView->ResourceDimension) {
   case D3D10DDIRESOURCE_TEXTURE1D:
      pUAView->image.u.tex.level = pCreateUAView->Tex1D.MipSlice;
      pUAView->image.u.tex.first_layer = pCreateUAView->Tex1D.FirstArraySlice;
      pUAView->image.u.tex.last_layer =
         pCreateUAView->Tex1D.FirstArraySlice + pCreateUAView->Tex1D.ArraySize - 1;
      pUAView->image.u.tex.single_layer_view = pCreateUAView->Tex1D.ArraySize == 1;
      break;
   case D3D10DDIRESOURCE_TEXTURE2D:
      pUAView->image.u.tex.level = pCreateUAView->Tex2D.MipSlice;
      pUAView->image.u.tex.first_layer = pCreateUAView->Tex2D.FirstArraySlice;
      pUAView->image.u.tex.last_layer =
         pCreateUAView->Tex2D.FirstArraySlice + pCreateUAView->Tex2D.ArraySize - 1;
      pUAView->image.u.tex.single_layer_view = pCreateUAView->Tex2D.ArraySize == 1;
      break;
   case D3D10DDIRESOURCE_TEXTURE3D:
      pUAView->image.u.tex.level = pCreateUAView->Tex3D.MipSlice;
      pUAView->image.u.tex.first_layer = pCreateUAView->Tex3D.FirstW;
      pUAView->image.u.tex.last_layer =
         pCreateUAView->Tex3D.FirstW + pCreateUAView->Tex3D.WSize - 1;
      pUAView->image.u.tex.is_2d_view_of_3d = pCreateUAView->Tex3D.WSize == 1;
      break;
   case D3D10DDIRESOURCE_BUFFER: {
      if (pCreateUAView->Format == DXGI_FORMAT_UNKNOWN ||
          (pCreateUAView->Buffer.Flags & D3D11_DDI_BUFFER_UAV_FLAG_RAW))
         pUAView->clear_format = PIPE_FORMAT_NONE;

      const unsigned element_size =
         GetBufferUAVElementSize(pUAView->resource, pCreateUAView,
                                 pUAView->image.format);
      pUAView->buffer_raw =
         pCreateUAView->Buffer.Flags & D3D11_DDI_BUFFER_UAV_FLAG_RAW;
      pUAView->buffer_counter =
         pCreateUAView->Buffer.Flags & D3D11_DDI_BUFFER_UAV_FLAG_COUNTER;
      pUAView->buffer_append =
         pCreateUAView->Buffer.Flags & D3D11_DDI_BUFFER_UAV_FLAG_APPEND;
      pUAView->buffer_structured = buffer_structured;
      pUAView->buffer_first_element = pCreateUAView->Buffer.FirstElement;
      pUAView->buffer_num_elements = pCreateUAView->Buffer.NumElements;
      pUAView->buffer_stride = pUAView->buffer_structured ?
         pUAView->resource->ByteStride : element_size;
      pUAView->image.u.buf.offset =
         pCreateUAView->Buffer.FirstElement * element_size;
      pUAView->image.u.buf.size =
         pCreateUAView->Buffer.NumElements * element_size;
      break;
   }
   default:
      LOG_UNSUPPORTED(true);
      break;
   }

   if (pUAView->buffer_counter || pUAView->buffer_append) {
      Device *device = CastDevice(hDevice);
      struct pipe_resource resource = {};
      resource.target = PIPE_TEXTURE_2D;
      resource.format = PIPE_FORMAT_R32_UINT;
      resource.width0 = resource.height0 = resource.depth0 = 1;
      resource.array_size = 1;
      resource.usage = PIPE_USAGE_DEFAULT;
      resource.bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SHADER_IMAGE;
      resource.flags = YTTRIUM_GDI_RESOURCE_FLAG_UAV_COUNTER;
      pUAView->counter_resource =
         device->screen->resource_create(device->screen, &resource);
      if (!pUAView->counter_resource) {
         YTTRIUM_WARN("yttrium: UAV creation failed owner=d3d10umd "
                      "reason=gpu_counter_image_allocation\n");
         pipe_resource_reference(&pUAView->pipe_resource, NULL);
         pUAView->image.resource = NULL;
         SetError(hDevice, E_OUTOFMEMORY);
         return;
      }
      ResetUAVCounter(device, pUAView, 0);
   }
}

void APIENTRY
DestroyUnorderedAccessView(
   D3D10DDI_HDEVICE hDevice,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView)
{
   LOG_ENTRYPOINT();

   UnorderedAccessView *pUAView =
      CastUnorderedAccessView(hUnorderedAccessView);
   if (!pUAView)
      return;

   pipe_resource_reference(&pUAView->pipe_resource, NULL);
   pipe_resource_reference(&pUAView->counter_resource, NULL);
   pUAView->resource = NULL;
}

static bool
PackUAVClearUint(enum pipe_format format,
                 const UINT values[4],
                 uint8_t pattern[16],
                 unsigned *pattern_size)
{
   if (!values || !pattern || !pattern_size)
      return false;

   if (format == PIPE_FORMAT_NONE) {
      memcpy(pattern, values, sizeof(values[0]));
      *pattern_size = sizeof(values[0]);
      return true;
   }

   if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
      uint32_t packed =
         (values[0] & 0x7ffu) |
         ((values[1] & 0x7ffu) << 11) |
         ((values[2] & 0x3ffu) << 22);
      memcpy(pattern, &packed, sizeof(packed));
      *pattern_size = sizeof(packed);
      return true;
   }

   const struct util_format_description *desc =
      util_format_description(format);
   if (!desc)
      return false;

   const unsigned block_size = util_format_get_blocksize(format);
   if (!block_size || block_size > 16)
      return false;

   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN)
      return false;

   memset(pattern, 0, 16);
   for (unsigned component = 0; component < 4; component++) {
      const unsigned channel = desc->swizzle[component];
      if (channel > PIPE_SWIZZLE_W)
         continue;

      const unsigned size = desc->channel[channel].size;
      const unsigned shift = desc->channel[channel].shift;
      if (!size || shift + size > block_size * 8)
         continue;

      const uint32_t value = size >= 32 ?
         values[component] : values[component] & ((1u << size) - 1);
      for (unsigned bit = 0; bit < size && bit < 32; bit++) {
         if (value & (1u << bit))
            pattern[(shift + bit) / 8] |= 1u << ((shift + bit) % 8);
      }
   }

   *pattern_size = block_size;
   return true;
}

static bool
PackUAVClearFloat(enum pipe_format format,
                  const FLOAT values[4],
                  uint8_t pattern[16],
                  unsigned *pattern_size)
{
   if (!values || !pattern || !pattern_size || format == PIPE_FORMAT_NONE)
      return false;

   const unsigned block_size = util_format_get_blocksize(format);
   if (!block_size || block_size > 16)
      return false;

   memset(pattern, 0, 16);
   util_format_pack_rgba(format, pattern, values, 1);
   *pattern_size = block_size;
   return true;
}

static void
FillUAVBufferClear(struct pipe_context *pipe,
                   struct pipe_resource *resource,
                   unsigned offset,
                   unsigned size,
                   const uint8_t *pattern,
                   unsigned pattern_size)
{
   if (!pipe || !resource || !pattern || !pattern_size || !size)
      return;

   if (pipe->buffer_subdata) {
      uint8_t *data = (uint8_t *)MALLOC(size);
      if (!data)
         return;
      for (unsigned i = 0; i < size; i++)
         data[i] = pattern[i % pattern_size];
      pipe->buffer_subdata(pipe, resource, PIPE_MAP_DISCARD_RANGE,
                           offset, size, data);
      FREE(data);
      return;
   }

   struct pipe_transfer *transfer = NULL;
   struct pipe_box box = {};
   box.x = offset;
   box.width = size;
   box.height = 1;
   box.depth = 1;

   uint8_t *map = (uint8_t *)pipe->buffer_map(
      pipe, resource, 0, PIPE_MAP_WRITE, &box, &transfer);
   if (!map)
      return;

   for (unsigned i = 0; i < size; i++)
      map[i] = pattern[i % pattern_size];

   pipe_buffer_unmap(pipe, transfer);
}

static void
FillUAVTextureClear(struct pipe_context *pipe,
                    UnorderedAccessView *uav,
                    const uint8_t *pattern,
                    unsigned pattern_size)
{
   if (!pipe || !uav || !uav->pipe_resource || !pattern || !pattern_size)
      return;

   const unsigned level = uav->image.u.tex.level;
   if (level > uav->pipe_resource->last_level)
      return;

   struct pipe_box box = {};
   box.x = 0;
   box.y = 0;
   box.z = uav->image.u.tex.first_layer;
   box.width = u_minify(uav->pipe_resource->width0, level);
   box.height = u_minify(uav->pipe_resource->height0, level);
   box.depth = uav->image.u.tex.last_layer >= uav->image.u.tex.first_layer ?
      uav->image.u.tex.last_layer - uav->image.u.tex.first_layer + 1 : 0;

   if (!box.width || !box.height || !box.depth)
      return;

   const unsigned stride =
      util_format_get_stride(uav->pipe_resource->format, box.width);
   const unsigned layer_stride = stride * box.height;

   if (pipe->texture_subdata) {
      uint8_t *data = (uint8_t *)MALLOC(layer_stride);
      if (!data)
         return;

      for (unsigned i = 0; i < layer_stride; i++)
         data[i] = pattern[i % pattern_size];

      struct pipe_box layer_box = box;
      layer_box.depth = 1;
      for (unsigned z = 0; z < box.depth; z++) {
         layer_box.z = box.z + z;
         pipe->texture_subdata(pipe, uav->pipe_resource, level,
                               PIPE_MAP_DISCARD_RANGE, &layer_box, data,
                               stride, layer_stride);
      }
      FREE(data);
      return;
   }

   if (pipe->texture_map && pipe->texture_unmap) {
      struct pipe_transfer *transfer = NULL;
      uint8_t *map = (uint8_t *)pipe->texture_map(
         pipe, uav->pipe_resource, level,
         PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE, &box, &transfer);
      if (map && transfer) {
         for (unsigned z = 0; z < box.depth; z++) {
            for (unsigned y = 0; y < box.height; y++) {
               uint8_t *row = map + z * transfer->layer_stride +
                  y * transfer->stride;
               for (unsigned i = 0; i < stride; i++)
                  row[i] = pattern[i % pattern_size];
            }
         }
         pipe->texture_unmap(pipe, transfer);
         return;
      }
      if (transfer)
         pipe->texture_unmap(pipe, transfer);
   }

   if (pipe->clear_texture)
      pipe->clear_texture(pipe, uav->pipe_resource, level, &box, pattern);
}

static void
ClearUAVTextureUint(struct pipe_context *pipe,
                    UnorderedAccessView *uav,
                    const UINT values[4])
{
   uint8_t pattern[16];
   unsigned pattern_size = 0;
   if (!uav || !PackUAVClearUint(uav->clear_format, values,
                                 pattern, &pattern_size))
      return;

   FillUAVTextureClear(pipe, uav, pattern, pattern_size);
}

static void
ClearUAVTextureFloat(struct pipe_context *pipe,
                     UnorderedAccessView *uav,
                     const FLOAT values[4])
{
   uint8_t pattern[16];
   unsigned pattern_size = 0;
   if (!uav || !PackUAVClearFloat(uav->clear_format, values,
                                  pattern, &pattern_size))
      return;

   FillUAVTextureClear(pipe, uav, pattern, pattern_size);
}

void APIENTRY
ClearUnorderedAccessViewUint(
   D3D10DDI_HDEVICE hDevice,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView,
   const UINT Values[4])
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   UnorderedAccessView *uav = CastUnorderedAccessView(hUnorderedAccessView);
   if (!pipe || !uav || !uav->pipe_resource || !Values)
      return;

   if (uav->pipe_resource->target == PIPE_BUFFER) {
      uint8_t pattern[16];
      unsigned pattern_size = 0;
      if (!PackUAVClearUint(uav->clear_format, Values,
                            pattern, &pattern_size))
         return;

      unsigned offset = uav->image.u.buf.offset;
      unsigned size = uav->image.u.buf.size;
      if (offset >= uav->pipe_resource->width0)
         return;
      if (!size || size > uav->pipe_resource->width0 - offset)
         size = uav->pipe_resource->width0 - offset;

      FillUAVBufferClear(pipe, uav->pipe_resource, offset, size,
                         pattern, pattern_size);
   } else {
      ClearUAVTextureUint(pipe, uav, Values);
   }
}

void APIENTRY
ClearUnorderedAccessViewFloat(
   D3D10DDI_HDEVICE hDevice,
   D3D11DDI_HUNORDEREDACCESSVIEW hUnorderedAccessView,
   const FLOAT Values[4])
{
   LOG_ENTRYPOINT();

   struct pipe_context *pipe = CastPipeContext(hDevice);
   UnorderedAccessView *uav = CastUnorderedAccessView(hUnorderedAccessView);
   if (!pipe || !uav || !uav->pipe_resource || !Values)
      return;

   if (uav->pipe_resource->target != PIPE_BUFFER)
      ClearUAVTextureFloat(pipe, uav, Values);
}

void APIENTRY
CsSetUnorderedAccessViews(
   D3D10DDI_HDEVICE hDevice,
   UINT StartSlot,
   UINT NumViews,
   __in_ecount (NumViews) const D3D11DDI_HUNORDEREDACCESSVIEW *phUnorderedAccessView,
   __in_ecount (NumViews) const UINT *pUAVInitialCounts)
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);

   if (!NumViews)
      return;

   assert(StartSlot + NumViews <= PIPE_MAX_SHADER_IMAGES);
   for (UINT i = 0; i < NumViews; i++) {
      UnorderedAccessView *uav =
         CastUnorderedAccessView(phUnorderedAccessView[i]);
      if (uav) {
         if (pUAVInitialCounts &&
             pUAVInitialCounts[i] != ~0u &&
             (uav->buffer_counter || uav->buffer_append))
            ResetUAVCounter(pDevice, uav, pUAVInitialCounts[i]);
         pDevice->unordered_access_views[MESA_SHADER_COMPUTE][StartSlot + i] =
            uav;
         pDevice->shader_images[MESA_SHADER_COMPUTE][StartSlot + i] =
            uav->image;
      } else {
         pDevice->unordered_access_views[MESA_SHADER_COMPUTE][StartSlot + i] =
            NULL;
         memset(&pDevice->shader_images[MESA_SHADER_COMPUTE][StartSlot + i],
                0, sizeof(pDevice->shader_images[MESA_SHADER_COMPUTE][0]));
      }
   }

   BindShaderImages(pDevice, MESA_SHADER_COMPUTE, StartSlot, NumViews);
   UpdateBufferInfoUavConstants(pDevice, MESA_SHADER_COMPUTE,
                                StartSlot, NumViews);
   UpdateBufferInfoConstants(pDevice, MESA_SHADER_COMPUTE);
}


/*
 * ----------------------------------------------------------------------
 *
 * GenMips --
 *
 *    The GenMips function generates the lower MIP-map levels
 *    on the specified shader-resource view.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
GenMips(D3D10DDI_HDEVICE hDevice,                           // IN
        D3D10DDI_HSHADERRESOURCEVIEW hShaderResourceView)   // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   if (!CheckPredicate(pDevice)) {
      return;
   }

   struct pipe_context *pipe = pDevice->pipe;
   struct pipe_sampler_view *sampler_view = GetPipeShaderResourceView(hShaderResourceView);

   util_gen_mipmap(pipe,
                   sampler_view->texture,
                   sampler_view->format,
                   sampler_view->u.tex.first_level,
                   sampler_view->u.tex.last_level,
                   sampler_view->u.tex.first_layer,
                   sampler_view->u.tex.last_layer,
                   PIPE_TEX_FILTER_LINEAR);
}


unsigned
ShaderFindOutputMapping(Shader *shader, unsigned registerIndex)
{
   if (!shader || !shader->state.tokens ||
       registerIndex >= ARRAY_SIZE(shader->output_mapping))
      return ~0u;

   return shader->output_mapping[registerIndex];
}
