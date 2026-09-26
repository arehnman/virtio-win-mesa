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
 * Draw.h --
 *    Functions that render 3D primitives.
 */


#include "Draw.h"
#include "State.h"
#include "Shader.h"

#include "Debug.h"

#include "util/u_draw.h"
#include "util/u_memory.h"

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"
#include "gallium/winsys/yttrium/gdi/yttrium_trace.h"

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

static bool
ConsumeBackendDrawFailure(D3D10DDI_HDEVICE hDevice)
{
   if (!yttrium_gdi_take_draw_failure(CastPipeContext(hDevice)))
      return false;

   YTTRIUM_WARN("yttrium: draw failed owner=d3d10umd "
                "reason=backend_native_draw_failed action=SetError\n");
   SetError(hDevice, E_FAIL);
   return true;
}

static unsigned
ClampedUAdd(unsigned a,
            unsigned b)
{
   unsigned c = a + b;
   if (c < a) {
      return 0xffffffff;
   }
   return c;
}


/* stride is required in order to set the element data */
static void
update_velems(Device *pDevice)
{
   if (!pDevice->velems_changed)
      return;

   if(pDevice->element_layout) {
      struct cso_velems_state *state = &pDevice->element_layout->state;
      for (unsigned i = 0; i < state->count; i++)
         state->velems[i].src_stride = pDevice->vertex_strides[state->velems[i].vertex_buffer_index];
      cso_set_vertex_elements(pDevice->cso, state);
   } else {
      /* IASetInputLayout(NULL) must retire the previous layout.  System-value
       * only shaders need no vertex buffers, including on indirect draws.
       */
      struct cso_velems_state empty = {};
      cso_set_vertex_elements(pDevice->cso, &empty);
   }

   pDevice->velems_changed = false;
}

/* A tokenless geometry shader inherits the active vertex shader.  Its stream
 * output declaration names original D3D output registers, while translation
 * packs the active shader's outputs into TGSI register indices.  Always map
 * from the preserved D3D registers so switching vertex shaders cannot remap
 * an already-remapped index.
 */
static bool
ResolveState(D3D10DDI_HDEVICE hDevice)
{
   Device *pDevice = CastDevice(hDevice);

   RefreshBoundShaderResourceViews(pDevice);

   if (pDevice->bound_empty_gs) {
      Shader *gs = pDevice->bound_empty_gs;
      Shader *vs = pDevice->bound_vs;
      struct pipe_context *pipe = pDevice->pipe;
      if (!vs || !vs->state.tokens) {
         YTTRIUM_WARN("yttrium: d3d10umd tokenless geometry stream-output "
                      "remap failed; draw rejected vs=%p "
                      "reason=vertex_shader_unavailable\n",
                      vs);
         SetError(hDevice, E_FAIL);
         return false;
      }

      const struct pipe_stream_output_info *stream_output_template =
         &gs->stream_output_template;
      struct pipe_stream_output_info resolved = {};
      memcpy(resolved.stride, stream_output_template->stride,
             sizeof(resolved.stride));

      for (unsigned i = 0; i < stream_output_template->num_outputs; ++i) {
         const unsigned d3d_register =
            gs->stream_output_d3d_registers[i];
         const unsigned mapping = ShaderFindOutputMapping(vs, d3d_register);
         if (mapping == ~0u) {
            const struct pipe_stream_output *unwritten =
               &stream_output_template->output[i];
            const unsigned declaration_mask =
               ((1u << unwritten->num_components) - 1u) <<
               unwritten->start_component;
            if (d3d_register >= PIPE_MAX_SHADER_OUTPUTS ||
                (declaration_mask &
                 ~gs->stream_output_signature_masks[d3d_register])) {
               YTTRIUM_WARN("yttrium: d3d10umd tokenless geometry "
                            "stream-output remap failed; draw rejected "
                            "output=%u d3d_register=%u mask=0x%x "
                            "reason=unmapped_output_not_in_signature\n",
                            i, d3d_register, declaration_mask);
               SetError(hDevice, E_FAIL);
               return false;
            }

            /* The complete tokenless-SO signature may contain an output that
             * this particular VS does not declare.  Its captured value is
             * undefined.  Omit that write while preserving the declaration's
             * destination offset and the complete record stride.
             */
            continue;
         }

         struct pipe_stream_output *output =
            &resolved.output[resolved.num_outputs++];
         *output = stream_output_template->output[i];
         output->register_index = mapping;
      }

      if (stream_output_template->num_outputs && !resolved.num_outputs) {
         YTTRIUM_WARN("yttrium: d3d10umd tokenless geometry stream-output "
                      "remap failed; draw rejected vs=%p "
                      "reason=no_declared_outputs_for_primitive_accounting\n",
                      vs);
         SetError(hDevice, E_FAIL);
         return false;
      }

      const bool remapped =
         memcmp(&resolved, &gs->state.stream_output, sizeof(resolved)) != 0;
      if (remapped) {
         struct pipe_shader_state resolved_state = gs->state;
         resolved_state.stream_output = resolved;
         void *resolved_handle = pipe->create_gs_state(pipe, &resolved_state);
         if (!resolved_handle) {
            YTTRIUM_WARN("yttrium: d3d10umd tokenless geometry stream-output "
                         "state creation failed; draw rejected vs=%p "
                         "outputs=%u\n",
                         vs, resolved.num_outputs);
            SetError(hDevice, E_FAIL);
            return false;
         }

         pipe->delete_gs_state(pipe, gs->handle);
         gs->handle = resolved_handle;
         gs->state.stream_output = resolved;
      }
      pipe->bind_gs_state(pipe, gs->handle);
   }
   update_velems(pDevice);

   if (pDevice->vbuffers_changed) {
      unsigned count = PIPE_MAX_ATTRIBS;

      if (OrderedContextWorkerEnabled()) {
         count = 0;
         for (unsigned i = PIPE_MAX_ATTRIBS; i > 0; i--) {
            const struct pipe_vertex_buffer *vb =
               &pDevice->vertex_buffers[i - 1];
            if (vb->is_user_buffer ? vb->buffer.user != NULL :
                                     vb->buffer.resource != NULL) {
               count = i;
               break;
            }
         }
      }

      cso_set_vertex_buffers(pDevice->cso, count,
                             pDevice->vertex_buffers);
      pDevice->vbuffers_changed = false;
   }

   return true;
}


static struct pipe_resource *
create_null_index_buffer(struct pipe_context *ctx, uint num_indices,
                         unsigned *restart_index, unsigned *index_size,
                         unsigned *ib_offset)
{
   unsigned buf_size = num_indices * sizeof(unsigned);
   unsigned *buf = (unsigned*)MALLOC(buf_size);
   struct pipe_resource *ibuf;

   memset(buf, 0, buf_size);

   ibuf = pipe_buffer_create_with_data(ctx,
                                       PIPE_BIND_INDEX_BUFFER,
                                       PIPE_USAGE_IMMUTABLE,
                                       buf_size, buf);
   *index_size = 4;
   *restart_index = 0xffffffff;
   *ib_offset = 0;

   FREE(buf);

   return ibuf;
}

/*
 * ----------------------------------------------------------------------
 *
 * Draw --
 *
 *    The Draw function draws nonindexed primitives.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
Draw(D3D10DDI_HDEVICE hDevice,   // IN
     UINT VertexCount,           // IN
     UINT StartVertexLocation)   // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);

   if (ConsumeBackendDrawFailure(hDevice))
      return;
   if (!ResolveState(hDevice))
      return;
   if (RunVertexShaderEmulation(pDevice, VertexCount))
      return;
   if (RunPixelShaderEmulation(pDevice))
      return;

   assert(pDevice->primitive < MESA_PRIM_COUNT);
   util_draw_arrays(pDevice->pipe,
                    pDevice->primitive,
                    StartVertexLocation,
                    VertexCount);
   (void)ConsumeBackendDrawFailure(hDevice);
}


/*
 * ----------------------------------------------------------------------
 *
 * DrawIndexed --
 *
 *    The DrawIndexed function draws indexed primitives.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DrawIndexed(D3D10DDI_HDEVICE hDevice,  // IN
            UINT IndexCount,           // IN
            UINT StartIndexLocation,   // IN
            INT BaseVertexLocation)    // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_draw_info info;
   struct pipe_draw_start_count_bias draw;
   struct pipe_resource *null_ib = NULL;
   unsigned restart_index = pDevice->restart_index;
   unsigned index_size = pDevice->index_size;
   unsigned ib_offset = pDevice->ib_offset;

   if (ConsumeBackendDrawFailure(hDevice))
      return;
   assert(pDevice->primitive < MESA_PRIM_COUNT);

   /* XXX I don't think draw still needs this? */
   if (!pDevice->index_buffer) {
      null_ib =
         create_null_index_buffer(pDevice->pipe,
                                  StartIndexLocation + IndexCount,
                                  &restart_index, &index_size, &ib_offset);
   }

   if (!ResolveState(hDevice)) {
      if (null_ib)
         pipe_resource_reference(&null_ib, NULL);
      return;
   }
   if (RunPixelShaderEmulation(pDevice)) {
      if (null_ib)
         pipe_resource_reference(&null_ib, NULL);
      return;
   }

   util_draw_init_info(&info);
   info.index_size = index_size;
   info.mode = pDevice->primitive;
   draw.start = ClampedUAdd(StartIndexLocation, ib_offset / index_size);
   draw.count = IndexCount;
   info.index.resource = null_ib ? null_ib : pDevice->index_buffer;
   draw.index_bias = BaseVertexLocation;
   info.primitive_restart = true;
   info.restart_index = restart_index;

   pDevice->pipe->draw_vbo(pDevice->pipe, &info, 0, NULL, &draw, 1);
   (void)ConsumeBackendDrawFailure(hDevice);

   if (null_ib) {
      pipe_resource_reference(&null_ib, NULL);
   }
}


/*
 * ----------------------------------------------------------------------
 *
 * DrawInstanced --
 *
 *    The DrawInstanced function draws particular instances
 *    of nonindexed primitives.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DrawInstanced(D3D10DDI_HDEVICE hDevice,      // IN
              UINT VertexCountPerInstance,   // IN
              UINT InstanceCount,            // IN
              UINT StartVertexLocation,      // IN
              UINT StartInstanceLocation)    // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);

   if (ConsumeBackendDrawFailure(hDevice))
      return;
   if (!InstanceCount) {
      return;
   }

   if (!ResolveState(hDevice))
      return;
   if (RunPixelShaderEmulation(pDevice))
      return;

   assert(pDevice->primitive < MESA_PRIM_COUNT);
   util_draw_arrays_instanced(pDevice->pipe,
                              pDevice->primitive,
                              StartVertexLocation,
                              VertexCountPerInstance,
                              StartInstanceLocation,
                              InstanceCount);
   (void)ConsumeBackendDrawFailure(hDevice);
}


/*
 * ----------------------------------------------------------------------
 *
 * DrawIndexedInstanced --
 *
 *    The DrawIndexedInstanced function draws particular
 *    instances of indexed primitives.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DrawIndexedInstanced(D3D10DDI_HDEVICE hDevice,   // IN
                     UINT IndexCountPerInstance, // IN
                     UINT InstanceCount,         // IN
                     UINT StartIndexLocation,    // IN
                     INT BaseVertexLocation,     // IN
                     UINT StartInstanceLocation) // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_draw_info info;
   struct pipe_draw_start_count_bias draw;
   struct pipe_resource *null_ib = NULL;
   unsigned restart_index = pDevice->restart_index;
   unsigned index_size = pDevice->index_size;
   unsigned ib_offset = pDevice->ib_offset;

   if (ConsumeBackendDrawFailure(hDevice))
      return;
   assert(pDevice->primitive < MESA_PRIM_COUNT);

   if (!InstanceCount) {
      return;
   }

   /* XXX I don't think draw still needs this? */
   if (!pDevice->index_buffer) {
      null_ib =
         create_null_index_buffer(pDevice->pipe,
                                  StartIndexLocation + IndexCountPerInstance,
                                  &restart_index, &index_size, &ib_offset);
   }

   if (!ResolveState(hDevice)) {
      if (null_ib)
         pipe_resource_reference(&null_ib, NULL);
      return;
   }
   if (RunPixelShaderEmulation(pDevice)) {
      if (null_ib)
         pipe_resource_reference(&null_ib, NULL);
      return;
   }

   util_draw_init_info(&info);
   info.index_size = index_size;
   info.mode = pDevice->primitive;
   draw.start = ClampedUAdd(StartIndexLocation, ib_offset / index_size);
   draw.count = IndexCountPerInstance;
   info.index.resource = null_ib ? null_ib : pDevice->index_buffer;
   draw.index_bias = BaseVertexLocation;
   info.start_instance = StartInstanceLocation;
   info.instance_count = InstanceCount;
   info.primitive_restart = true;
   info.restart_index = restart_index;

   pDevice->pipe->draw_vbo(pDevice->pipe, &info, 0, NULL, &draw, 1);
   (void)ConsumeBackendDrawFailure(hDevice);

   if (null_ib) {
      pipe_resource_reference(&null_ib, NULL);
   }
}

static void
DrawIndirect(D3D10DDI_HDEVICE hDevice,
             D3D10DDI_HRESOURCE hBufferForArgs,
             UINT AlignedByteOffsetForArgs, bool indexed)
{
   Device *pDevice = CastDevice(hDevice);
   Resource *pArgs = CastResource(hBufferForArgs);
   const UINT args_size = (indexed ? 5 : 4) * sizeof(UINT);
   if (ConsumeBackendDrawFailure(hDevice))
      return;
   if (!pArgs || !pArgs->resource ||
       pArgs->resource->target != PIPE_BUFFER ||
       !(pArgs->resource->bind & PIPE_BIND_COMMAND_ARGS_BUFFER) ||
       (AlignedByteOffsetForArgs & 3) ||
       AlignedByteOffsetForArgs > pArgs->resource->width0 ||
       args_size > pArgs->resource->width0 - AlignedByteOffsetForArgs ||
       (indexed && (!pDevice->index_buffer ||
                    (pDevice->index_size != 2 && pDevice->index_size != 4) ||
                    pDevice->ib_offset % pDevice->index_size))) {
      YTTRIUM_WARN("yttrium: draw failed owner=d3d10umd "
                   "component=DrawIndirect reason=invalid_buffer_or_offset "
                   "indexed=%u offset=%u action=SetError\n",
                   indexed, AlignedByteOffsetForArgs);
      SetError(hDevice, E_INVALIDARG);
      return;
   }

   if (!ResolveState(hDevice))
      return;

   struct pipe_draw_info info;
   util_draw_init_info(&info);
   info.mode = pDevice->primitive;
   if (indexed) {
      info.index_size = pDevice->index_size;
      info.index.resource = pDevice->index_buffer;
      info.primitive_restart = true;
      info.restart_index = pDevice->restart_index;
   }
   struct pipe_draw_indirect_info indirect = {};
   indirect.buffer = pArgs->resource;
   indirect.offset = AlignedByteOffsetForArgs;
   indirect.stride = args_size;
   indirect.draw_count = 1;
   /* Gallium adds draw.start to indirect firstIndex.  Preserve the IA byte
    * offset here; the backend binds the index buffer at that byte offset. */
   struct pipe_draw_start_count_bias draw = {};
   draw.start = indexed ? pDevice->ib_offset / pDevice->index_size : 0;
   pDevice->pipe->draw_vbo(pDevice->pipe, &info, 0, &indirect, &draw, 1);
   (void)ConsumeBackendDrawFailure(hDevice);
}

void APIENTRY
DrawIndexedInstancedIndirect(D3D10DDI_HDEVICE hDevice,
                             D3D10DDI_HRESOURCE hBufferForArgs,
                             UINT AlignedByteOffsetForArgs)
{
   LOG_ENTRYPOINT();
   DrawIndirect(hDevice, hBufferForArgs, AlignedByteOffsetForArgs, true);
}

void APIENTRY
DrawInstancedIndirect(D3D10DDI_HDEVICE hDevice,
                      D3D10DDI_HRESOURCE hBufferForArgs,
                      UINT AlignedByteOffsetForArgs)
{
   LOG_ENTRYPOINT();

   DrawIndirect(hDevice, hBufferForArgs, AlignedByteOffsetForArgs, false);
}


/*
 * ----------------------------------------------------------------------
 *
 * DrawAuto --
 *
 *    The DrawAuto function works similarly to the Draw function,
 *    except DrawAuto is used for the special case where vertex
 *    data is written through the stream-output unit and then
 *    recycled as a vertex buffer. The driver determines the number
 *    of primitives, in part, by how much data was written to the
 *    buffer through stream output.
 *
 * ----------------------------------------------------------------------
 */

void APIENTRY
DrawAuto(D3D10DDI_HDEVICE hDevice)  // IN
{
   LOG_ENTRYPOINT();

   Device *pDevice = CastDevice(hDevice);
   struct pipe_draw_info info;
   struct pipe_draw_indirect_info indirect;

   if (ConsumeBackendDrawFailure(hDevice))
      return;
   if (!pDevice->draw_so_target) {
      YTTRIUM_WARN("yttrium: draw failed owner=d3d10umd "
                   "component=DrawAuto reason=no_stream_output_source "
                   "action=SetError\n");
      SetError(hDevice, E_FAIL);
      return;
   }

   assert(pDevice->primitive < MESA_PRIM_COUNT);

   if (!ResolveState(hDevice))
      return;
   if (RunPixelShaderEmulation(pDevice))
      return;

   util_draw_init_info(&info);
   info.mode = pDevice->primitive;
   memset(&indirect, 0, sizeof indirect);
   indirect.count_from_stream_output = pDevice->draw_so_target;

   pDevice->pipe->draw_vbo(pDevice->pipe, &info, 0, &indirect, NULL, 1);
   (void)ConsumeBackendDrawFailure(hDevice);
}
